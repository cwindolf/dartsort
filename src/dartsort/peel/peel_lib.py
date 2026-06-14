import warnings
from collections import namedtuple
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from spikeinterface.core import BaseRecording
from torch import Tensor

from ..detect import convexity_filter, detect_and_deduplicate
from ..util.internal_config import (
    ComputationConfig,
    FitSamplingConfig,
    PeakSign,
    ThresholdingConfig,
    WaveformConfig,
)
from ..util.job_util import ensure_computation_config
from ..util.spiketorch import grab_spikes, ptp, subtract_spikes_
from ..util.waveform_util import get_relative_subset, make_channel_index
from .peel_base import PeelingBatchResult

if TYPE_CHECKING:
    from ..transform.pipeline import WaveformPipeline
    from ..util.internal_config import PeakSign


def denoiser_time_shifts(
    waveforms: Tensor,
    channels: Tensor,
    voltages: Tensor,
    subtract_rel_inds: Tensor | None,
    trough_offset_samples: int,
    spike_length_samples: int,
    peak_sign: "PeakSign",
    denoiser_realignment_shift: int,
    denoiser_realignment_channel: Literal["detection", "denoised"],
) -> Tensor:
    # extract main channel traces
    if denoiser_realignment_channel == "detection":
        assert subtract_rel_inds is not None
        main_channel_rel_inds = subtract_rel_inds[channels]
    elif denoiser_realignment_channel == "denoised":
        main_channel_rel_inds = ptp(waveforms).nan_to_num_(nan=-torch.inf).argmax(dim=1)
    else:
        assert False
    denoised_main_channel_traces = waveforms.take_along_dim(
        dim=2, indices=main_channel_rel_inds[:, None, None]
    )
    nwf = len(waveforms)
    assert denoised_main_channel_traces.shape == (nwf, spike_length_samples, 1)
    denoised_main_channel_traces = denoised_main_channel_traces[:, :, 0]

    # extract window around trough
    start = trough_offset_samples - denoiser_realignment_shift
    end = trough_offset_samples + denoiser_realignment_shift + 1
    snips = denoised_main_channel_traces[:, start:end]
    assert snips.shape == (nwf, 2 * denoiser_realignment_shift + 1)

    # handle the sign of the events so that we can align to maxima
    if peak_sign == "both":
        snips.mul_(torch.sign(voltages)[:, None])
    elif peak_sign == "neg":
        snips.neg_()
    else:
        assert peak_sign == "pos"

    # find shifts just by argmax
    peaks = snips.argmax(dim=1)
    dt = peaks.sub_(denoiser_realignment_shift).to(dtype=torch.int16)
    return dt


def check_residual_decrease(
    orig_wfs: Tensor | None,
    dn_wfs: Tensor,
    decrease_objective: Literal["deconv", "norm", "normsq"] = "deconv",
    threshold=10.0,
    save_residnorm_decrease=False,
    overwrite_orig_waveforms: bool = False,
    local_whiteners: Tensor | None = None,
    whitening_kernel: Tensor | None = None,
    channels: Tensor | None = None,
) -> tuple[Tensor | None, dict[str, Tensor]]:
    if not threshold:
        return None, {}
    assert orig_wfs is not None

    if local_whiteners is not None:
        assert channels is not None
        W = local_whiteners[channels]

        # remove nans
        if overwrite_orig_waveforms:
            orig_wfs = orig_wfs.nan_to_num_()
        else:
            orig_wfs = orig_wfs.nan_to_num()
        dn_wfs = dn_wfs.nan_to_num()

        # spatial mul -- putting temporal dim last here
        orig_wfs = W.bmm(orig_wfs.mT)
        dn_wfs = W.bmm(dn_wfs.mT)

        # temporal conv if needed
        if whitening_kernel is not None:
            *shp, t = orig_wfs.shape
            k = whitening_kernel[None, None]
            orig_wfs = F.conv1d(orig_wfs.view(-1, 1, t), k, padding="same")
            dn_wfs = F.conv1d(dn_wfs.view(-1, 1, t), k, padding="same")
            orig_wfs = orig_wfs.view(*shp, t)
            dn_wfs = dn_wfs.view(*shp, t)

    if decrease_objective == "deconv":
        if overwrite_orig_waveforms:
            buf = orig_wfs.mul_(dn_wfs).nan_to_num_()
            conv = buf.sum(dim=(1, 2))
            torch.square(dn_wfs, out=buf)
            norm = buf.nan_to_num_().sum(dim=(1, 2))
        else:
            dn_wfs = dn_wfs.nan_to_num()
            conv = (orig_wfs * dn_wfs).nan_to_num_().sum(dim=(1, 2))
            norm = dn_wfs.square_().sum(dim=(1, 2))
        reduction = conv.mul_(2.0).sub_(norm)
        threshold = threshold**2
    elif decrease_objective in ("norm", "normsq"):
        orig_decobj = orig_wfs.square().sum(dim=(1, 2))
        orig_wfs = orig_wfs.sub_(dn_wfs).nan_to_num_()
        new_decobj = orig_wfs.square_().sum(dim=(1, 2))
        if decrease_objective == "norm":
            orig_decobj = orig_decobj.sqrt_()
            new_decobj = new_decobj.sqrt_()
        else:
            threshold = threshold**2
        reduction = orig_decobj - new_decobj
    else:
        assert False

    keep = cast(torch.Tensor, threshold < reduction)
    (keep,) = keep.nonzero(as_tuple=True)
    if save_residnorm_decrease:
        features = dict(residnorm_decreases=reduction)
    else:
        features = {}
    return keep, features


ChunkSubtractionResult = namedtuple(
    "ChunkSubtractionResult",
    [
        "n_spikes",
        "times_samples",
        "channels",
        "collisioncleaned_waveforms",
        "residual",
        "features",
    ],
)


def subtract_chunk(
    traces: Tensor,
    channel_index: Tensor,
    denoising_pipeline: "WaveformPipeline",
    extract_index: Tensor | None = None,
    extract_mask: Tensor | None = None,
    trough_offset_samples=42,
    spike_length_samples=121,
    left_margin=0,
    right_margin=0,
    detection_threshold=4.0,
    peak_sign: PeakSign = "both",
    realign_to_denoiser=False,
    denoiser_realignment_shift=5,
    denoiser_realignment_channel="detection",
    convexity_threshold=None,
    convexity_radius=3,
    peak_channel_index: Tensor | None = None,
    dedup_channel_index: Tensor | None = None,
    subtract_rel_inds: Tensor | None = None,
    dedup_rel_inds: Tensor | None = None,
    residnorm_decrease_threshold=16.0,
    decrease_objective: Literal["norm", "normsq", "deconv"] = "deconv",
    local_whiteners: Tensor | None = None,
    whitening_kernel: Tensor | None = None,
    relative_peak_radius=5,
    dedup_temporal_radius=7,
    remove_exact_duplicates=True,
    pos_dedup_temporal_radius=None,
    dedup_batch_size=512,
    no_subtraction=False,
    max_iter=100,
    trough_priority: float | None = None,
    growth_tolerance: float | None = None,
    cumulant_order=None,
    save_iteration=False,
    save_residnorm_decrease=False,
    compute_collidedness=False,
) -> ChunkSubtractionResult:
    """Core peeling routine for subtraction"""
    if no_subtraction:
        threshold_res = threshold_chunk(
            traces,
            channel_index,
            detection_threshold=detection_threshold,
            peak_sign=peak_sign,
            peak_channel_index=peak_channel_index,
            dedup_channel_index=dedup_channel_index,
            dedup_rel_inds=dedup_rel_inds,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            left_margin=left_margin,
            right_margin=right_margin,
            relative_peak_radius=relative_peak_radius,
            temporal_dedup_radius_samples=dedup_temporal_radius,
            dedup_batch_size=dedup_batch_size,
            remove_exact_duplicates=remove_exact_duplicates,
            cumulant_order=cumulant_order,
            convexity_threshold=convexity_threshold,
            convexity_radius=convexity_radius,
            max_spikes_per_chunk=None,
            quiet=False,
        )
        waveforms, features = denoising_pipeline(
            threshold_res["waveforms"], channels=threshold_res["channels"]
        )
        return ChunkSubtractionResult(
            n_spikes=threshold_res["n_spikes"],
            times_samples=threshold_res["times_rel"],
            channels=threshold_res["channels"],
            collisioncleaned_waveforms=waveforms,
            residual=None,
            features=features,
        )

    # validate arguments to avoid confusing error messages later
    re_extract = extract_index is not None
    if extract_index is None:
        extract_index = channel_index
    else:
        assert extract_mask is not None
    assert 0 <= left_margin < traces.shape[0]
    assert 0 <= right_margin < traces.shape[0]
    assert traces.shape[1] == channel_index.shape[0]
    if dedup_channel_index is not None:
        assert traces.shape[1] == dedup_channel_index.shape[0]

    # can only subtract spikes with trough time >=trough_offset and <max_trough
    post_trough_samples = spike_length_samples - trough_offset_samples
    max_trough_time = traces.shape[0] - post_trough_samples

    if growth_tolerance is not None:
        gtol = traces.abs().add_(growth_tolerance)
    else:
        gtol = None

    # initialize residual, it needs to be padded to support
    # our channel indexing convention. this copies the input.
    residual = F.pad(traces, (0, 1), value=torch.nan)

    subtracted_waveforms = []
    spike_times = []
    spike_channels = []
    spike_features = []
    detection_mask = torch.ones_like(residual)
    dedup_temporal_ix = torch.arange(
        -dedup_temporal_radius, dedup_temporal_radius + 1, device=residual.device
    )
    pos_dedup_temporal_ix = None
    if pos_dedup_temporal_radius:
        pos_dedup_temporal_ix = torch.arange(
            -pos_dedup_temporal_radius,
            pos_dedup_temporal_radius + 1,
            device=residual.device,
        )

    for it in range(max_iter):
        residual_det = residual[:, :-1]
        if it and gtol is not None:
            residual_det = residual_det.clamp(-gtol, gtol)

        times_samples, channels = detect_and_deduplicate(
            residual_det,
            detection_threshold,
            peak_channel_index=peak_channel_index,
            dedup_channel_index=channel_index,
            peak_sign=peak_sign,
            relative_peak_radius=relative_peak_radius,
            dedup_temporal_radius=spike_length_samples,
            spatial_dedup_batch_size=dedup_batch_size,
            remove_exact_duplicates=remove_exact_duplicates,
            dedup_index_inds=subtract_rel_inds,
            detection_mask=detection_mask[:, :-1] if it else None,
            trough_priority=trough_priority,
            cumulant_order=cumulant_order,
        )
        if not times_samples.numel():
            break

        if it:
            keep = detection_mask[times_samples, channels]
            (keep,) = keep.nonzero(as_tuple=True)
            times_samples = times_samples[keep]
            channels = channels[keep]

        if not times_samples.numel():
            break

        keep = convexity_filter(
            residual,
            times_samples,
            channels,
            threshold=convexity_threshold,
            radius=convexity_radius,
        )
        times_samples = times_samples[keep]
        channels = channels[keep]
        if not times_samples.numel():
            break

        voltages = residual[times_samples, channels]

        # never look at these again.
        time_ix = times_samples.unsqueeze(1) + dedup_temporal_ix
        time_ix = time_ix.clamp_(0, traces.shape[0] - 1)
        if dedup_channel_index is not None:
            chan_ix = dedup_channel_index[channels]
        else:
            chan_ix = channels.unsqueeze(1)
        detection_mask[time_ix[:, :, None], chan_ix[:, None, :]] = 0.0

        # take extra care to exclude positive peaks appearing near stronger troughs
        if pos_dedup_temporal_radius:
            assert pos_dedup_temporal_ix is not None
            (neg,) = (voltages < 0).nonzero(as_tuple=True)
            time_ix = times_samples[neg].unsqueeze(1) + pos_dedup_temporal_ix
            time_ix = time_ix.clamp_(0, traces.shape[0] - 1)
            if dedup_channel_index is not None:
                chan_ix = dedup_channel_index[channels[neg]]
            else:
                chan_ix = channels[neg].unsqueeze(1)

            pd_mask = torch.ones_like(detection_mask)
            pd_mask[time_ix[:, :, None], chan_ix[:, None, :]] = 0
            pd_mask = torch.logical_or(pd_mask, residual < 0)
            detection_mask = torch.logical_and(detection_mask, pd_mask)

        # throw away spikes which cannot be subtracted
        keep = times_samples == times_samples.clamp(
            trough_offset_samples, max_trough_time
        )
        (keep,) = keep.nonzero(as_tuple=True)

        if not keep.numel():
            break
        times_samples = times_samples[keep]
        channels = channels[keep]
        voltages = voltages[keep]

        # -- read waveforms, denoise, and test residnorm decrease
        waveforms = grab_spikes(
            residual,
            times_samples,
            channels,
            channel_index,
            trough_offset=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            buffer=0,
            already_padded=True,
        )

        if residnorm_decrease_threshold:
            original_waveforms = waveforms.nan_to_num()
        else:
            original_waveforms = None

        waveforms, features = denoising_pipeline(waveforms, channels=channels)

        resid_keep, new_feats = check_residual_decrease(
            original_waveforms,
            waveforms,
            decrease_objective=decrease_objective,
            threshold=residnorm_decrease_threshold,
            save_residnorm_decrease=save_residnorm_decrease,
            local_whiteners=local_whiteners,
            whitening_kernel=whitening_kernel,
            channels=channels,
        )
        features.update(new_feats)
        if resid_keep is not None:
            if not resid_keep.numel():
                break
            assert original_waveforms is not None
            if resid_keep.numel() < len(original_waveforms):
                waveforms = waveforms[resid_keep]
                times_samples = times_samples[resid_keep]
                channels = channels[resid_keep]
                voltages = voltages[resid_keep]
                for k, ft in features.items():
                    features[k] = ft[resid_keep]

        # -- subtract in place
        residual = subtract_spikes_(
            residual,
            times_samples,
            channels,
            channel_index,
            waveforms,
            trough_offset=trough_offset_samples,
            buffer=0,
            already_padded=True,
            in_place=True,
        )

        # -- follow the nn's realignment advice, if requested
        if realign_to_denoiser:
            features["time_shifts"] = denoiser_time_shifts(
                waveforms,
                channels,
                voltages,
                subtract_rel_inds,
                trough_offset_samples,
                spike_length_samples,
                peak_sign,
                denoiser_realignment_shift,
                denoiser_realignment_channel,
            )

        # -- store this iter's outputs
        spike_times.append(times_samples)
        spike_channels.append(channels)
        spike_features.append(features)
        if save_iteration:
            spike_features[-1]["iteration"] = torch.full_like(times_samples, it)
        subtracted_waveforms.append(waveforms)

    # check if we got no spikes
    if not spike_times:
        return empty_chunk_subtraction_result(
            spike_length_samples,
            channel_index,
            residual[left_margin : traces.shape[0] - right_margin, :-1],
        )

    # concatenate all of the thresholds together into single tensors
    spike_times = torch.concatenate(spike_times)
    spike_channels = torch.concatenate(spike_channels)
    subtracted_waveforms = torch.concatenate(subtracted_waveforms)
    spike_features = {
        k: torch.concatenate([ff[k] for ff in spike_features])
        for k in spike_features[0].keys()
    }

    # discard spikes in the margins and sort times_samples for caller
    max_valid_t = traces.shape[0] - right_margin - 1
    keep = spike_times == spike_times.clamp(left_margin, max_valid_t)
    (keep,) = keep.nonzero(as_tuple=True)
    if not keep.numel():
        return empty_chunk_subtraction_result(
            spike_length_samples,
            channel_index,
            residual[left_margin : traces.shape[0] - right_margin, :-1],
        )

    keep = keep[torch.argsort(spike_times[keep])]
    subtracted_waveforms = subtracted_waveforms[keep]
    spike_times = spike_times[keep]
    spike_channels = spike_channels[keep]
    for k in spike_features:
        spike_features[k] = spike_features[k][keep]

    # if extract_index != subtract_index, re-do the channels for the subtracted wfs
    if re_extract:
        subtracted_waveforms = get_relative_subset(
            subtracted_waveforms, spike_channels, extract_mask
        )

    # construct collision-cleaned waveforms
    collisioncleaned_waveforms = grab_spikes(
        residual,
        spike_times,
        spike_channels,
        extract_index,
        trough_offset=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        buffer=0,
        already_padded=True,
    )
    if compute_collidedness:
        coll = collisioncleaned_waveforms.square().nanmean(dim=(1, 2)).sqrt_()
        spike_features["collidedness"] = coll
    collisioncleaned_waveforms += subtracted_waveforms

    # offset spike times_samples according to margin
    spike_times -= left_margin

    # apply time shifts
    if "time_shifts" in spike_features:
        spike_times += spike_features["time_shifts"]

    # strip margin and padding channel off the residual
    residual = residual[left_margin : traces.shape[0] - right_margin, :-1].cpu()

    return ChunkSubtractionResult(
        n_spikes=spike_times.numel(),
        times_samples=spike_times,
        channels=spike_channels,
        collisioncleaned_waveforms=collisioncleaned_waveforms,
        residual=residual,
        features=spike_features,
    )


def empty_chunk_subtraction_result(spike_length_samples, channel_index, residual):
    empty_waveforms = torch.empty(
        (0, spike_length_samples, channel_index.shape[1]),
        dtype=residual.dtype,
    )
    empty_times_or_chans = torch.empty((0,), dtype=torch.long)
    return ChunkSubtractionResult(
        n_spikes=0,
        times_samples=empty_times_or_chans,
        channels=empty_times_or_chans,
        collisioncleaned_waveforms=empty_waveforms,
        residual=residual,
        features={},
    )


def threshold_chunk(
    traces,
    channel_index,
    detection_threshold=4.0,
    peak_sign: PeakSign = "both",
    peak_channel_index=None,
    dedup_channel_index=None,
    dedup_rel_inds=None,
    dedup_batch_size=512,
    trough_offset_samples=42,
    spike_length_samples=121,
    left_margin=0,
    right_margin=0,
    relative_peak_radius=5,
    temporal_dedup_radius_samples=7,
    remove_exact_duplicates=True,
    max_spikes_per_chunk=None,
    thinning=0.0,
    time_jitter=0,
    trough_priority=None,
    spatial_jitter_channel_index=None,
    cumulant_order=None,
    convexity_threshold=None,
    convexity_radius=3,
    return_waveforms=True,
    rg=None,
    quiet=False,
) -> PeelingBatchResult:
    n_index = channel_index.shape[1]
    times_rel, channels, energies = detect_and_deduplicate(
        traces,
        detection_threshold,
        peak_channel_index=peak_channel_index,
        dedup_channel_index=dedup_channel_index,
        dedup_index_inds=dedup_rel_inds,
        spatial_dedup_batch_size=dedup_batch_size,
        peak_sign=peak_sign,
        dedup_temporal_radius=temporal_dedup_radius_samples,
        remove_exact_duplicates=remove_exact_duplicates,
        relative_peak_radius=relative_peak_radius,
        return_energies=True,
        trough_priority=trough_priority,
        cumulant_order=cumulant_order,
    )
    if not times_rel.numel():
        return PeelingBatchResult(
            n_spikes=0,
            orig_times_rel=times_rel,
            times_rel=times_rel,
            orig_channels=channels,
            channels=channels,
            voltages=energies,
            waveforms=energies.view(-1, spike_length_samples, n_index),
        )
    if convexity_threshold:
        keep = convexity_filter(
            traces,
            times_rel,
            channels,
            threshold=convexity_threshold,
            radius=convexity_radius,
        )
        times_rel = times_rel[keep]
        channels = channels[keep]
        energies = energies[keep]
        del keep

    orig_times_rel = times_rel
    orig_channels = channels
    if thinning is not None or time_jitter or spatial_jitter_channel_index is not None:
        keep, times_rel, channels = perturb_detections(
            times_rel,
            channels,
            thinning=thinning,
            time_jitter=time_jitter,
            spatial_jitter_channel_index=spatial_jitter_channel_index,
            rg=rg,
        )
        orig_times_rel = orig_times_rel[keep]
        orig_channels = orig_channels[keep]
        energies = energies[keep]
        del keep

    # want only peaks in the chunk
    min_time = left_margin + trough_offset_samples
    tail_samples = spike_length_samples - trough_offset_samples
    max_time = traces.shape[0] - right_margin - tail_samples - 1
    valid = times_rel == times_rel.clamp(min_time, max_time)
    (valid,) = valid.nonzero(as_tuple=True)
    orig_times_rel = orig_times_rel[valid]
    times_rel = times_rel[valid]
    channels = channels[valid]
    orig_channels = orig_channels[valid]
    voltages = traces[orig_times_rel, orig_channels]
    n_detect = times_rel.numel()
    if not n_detect:
        return PeelingBatchResult(
            n_spikes=0,
            times_rel=times_rel,
            channels=channels,
            voltages=energies,
            orig_times_rel=orig_times_rel,
            orig_channels=orig_channels,
        )

    if max_spikes_per_chunk is not None:
        if n_detect > max_spikes_per_chunk and not quiet:
            warnings.warn(
                f"{n_detect} spikes in chunk was larger than "
                f"{max_spikes_per_chunk=}. Keeping the top ones."
            )
            energies = energies[valid]
            best = torch.argsort(energies)[-max_spikes_per_chunk:]
            best = best.sort().values
            del energies

            times_rel = times_rel[best]
            channels = channels[best]
            voltages = voltages[best]
            orig_channels = orig_channels[best]
            orig_times_rel = orig_times_rel[best]

    # load up the waveforms for this chunk
    if return_waveforms:
        waveforms = grab_spikes(
            traces,
            times_rel,
            channels,
            channel_index,
            trough_offset=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            already_padded=False,
            pad_value=torch.nan,
        )
    else:
        waveforms = None

    # offset times for caller
    orig_times_rel -= left_margin
    times_rel -= left_margin

    res = PeelingBatchResult(
        n_spikes=times_rel.numel(),
        orig_times_rel=orig_times_rel,
        orig_channels=orig_channels,
        times_rel=times_rel,
        channels=channels,
        voltages=voltages,
    )
    if waveforms is not None:
        res["waveforms"] = waveforms
    return res


def perturb_detections(
    times_rel,
    channels,
    thinning: float = 0,
    time_jitter=0,
    spatial_jitter_channel_index=None,
    rg: np.random.Generator | None = None,
):
    keep = slice(None)
    if not (thinning or time_jitter or spatial_jitter_channel_index is not None):
        return keep, times_rel, channels

    n = len(times_rel)
    if not n:
        return keep, times_rel, channels

    if thinning:
        assert 0 <= thinning <= 1
        assert rg is not None
        keep = rg.binomial(n=1, p=1.0 - thinning, size=n)
        keep = torch.from_numpy(np.flatnonzero(keep))
        keep = keep.to(times_rel)

        times_rel = times_rel[keep]
        channels = channels[keep]

    n = len(times_rel)
    if time_jitter:
        assert rg is not None
        jitter = rg.integers(low=-time_jitter, high=time_jitter + 1)
        times_rel = times_rel + torch.asarray(
            jitter, dtype=times_rel.dtype, device=times_rel.device
        )

    if spatial_jitter_channel_index is not None:
        assert rg is not None
        n_channels = len(spatial_jitter_channel_index)
        n_valid = (spatial_jitter_channel_index < n_channels).sum(1)
        n_valid = n_valid[channels].cpu()
        rel_ix = rg.integers(0, high=n_valid)
        rel_ix = torch.from_numpy(rel_ix).to(channels)
        channels = spatial_jitter_channel_index[channels, rel_ix]

    return keep, times_rel, channels


def shave_chunk(
    *,
    traces,
    channel_index,
    denoising_pipeline,
    residnorm_decrease_threshold,
    detection_threshold=4.0,
    peak_sign: PeakSign = "both",
    peak_channel_index=None,
    dedup_channel_index=None,
    dedup_rel_inds=None,
    dedup_batch_size=512,
    trough_offset_samples=42,
    spike_length_samples=121,
    left_margin=0,
    right_margin=0,
    relative_peak_radius=5,
    temporal_dedup_radius_samples=7,
    remove_exact_duplicates=True,
    trough_priority=None,
) -> tuple[torch.Tensor, PeelingBatchResult]:
    times_rel, channels = detect_and_deduplicate(
        traces,
        detection_threshold,
        peak_channel_index=peak_channel_index,
        dedup_channel_index=dedup_channel_index,
        dedup_index_inds=dedup_rel_inds,
        spatial_dedup_batch_size=dedup_batch_size,
        peak_sign=peak_sign,
        dedup_temporal_radius=temporal_dedup_radius_samples,
        remove_exact_duplicates=remove_exact_duplicates,
        relative_peak_radius=relative_peak_radius,
        return_energies=False,
        trough_priority=trough_priority,
    )
    # throw away spikes which cannot be extracted
    post_trough_samples = spike_length_samples - trough_offset_samples
    max_trough_time = traces.shape[0] - post_trough_samples
    keep = times_rel == times_rel.clamp(trough_offset_samples, max_trough_time)
    (keep,) = keep.nonzero(as_tuple=True)
    times_rel = times_rel[keep]
    channels = channels[keep]
    features = dict(voltages=traces[times_rel, channels])
    if not times_rel.numel():
        return traces, PeelingBatchResult(
            n_spikes=0, times_rel=times_rel, channels=channels, **features
        )

    # grab, denoise, subtract, add back
    traces = F.pad(traces, (0, 1), value=torch.nan)
    waveforms = grab_spikes(
        traces,
        times_rel,
        channels,
        channel_index,
        trough_offset=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        buffer=0,
        already_padded=True,
    )
    original_waveforms = waveforms
    waveforms, features = denoising_pipeline(waveforms, channels=channels, **features)
    if (
        waveforms.untyped_storage().data_ptr()
        == original_waveforms.untyped_storage().data_ptr()
    ):
        original_waveforms = original_waveforms.clone()
    resid_keep, new_feats = check_residual_decrease(
        original_waveforms,
        waveforms,
        threshold=residnorm_decrease_threshold,
        overwrite_orig_waveforms=True,
    )
    features.update(new_feats)
    if resid_keep is not None:
        if not resid_keep.numel():
            return traces[:, :-1], PeelingBatchResult(
                n_spikes=0, times_rel=times_rel, channels=channels
            )
        if resid_keep.numel() < len(original_waveforms):
            waveforms = waveforms[resid_keep]

            times_rel = times_rel[resid_keep]
            channels = channels[resid_keep]
            for k, ft in features.items():
                features[k] = ft[resid_keep]

    residual = subtract_spikes_(
        traces,
        times_rel,
        channels,
        channel_index,
        waveforms,
        trough_offset=trough_offset_samples,
        buffer=0,
        already_padded=True,
        in_place=True,
    )

    # peaks in chunk are...
    min_time = left_margin + trough_offset_samples
    tail_samples = spike_length_samples - trough_offset_samples
    max_time = traces.shape[0] - right_margin - tail_samples - 1
    valid = times_rel == times_rel.clamp(min_time, max_time)
    (valid,) = valid.nonzero(as_tuple=True)
    if valid.numel() < times_rel.numel():
        waveforms = waveforms[valid]
        times_rel = times_rel[valid]
        channels = channels[valid]
        for k, ft in features.items():
            features[k] = ft[valid]
    if not valid.numel():
        return residual[:, :-1], PeelingBatchResult(
            n_spikes=0, times_rel=times_rel, channels=channels
        )

    # collision-cleaned waveforms
    waveforms += grab_spikes(
        residual,
        times_rel,
        channels,
        channel_index,
        trough_offset=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        buffer=0,
        already_padded=True,
    )
    features["waveforms"] = waveforms

    # offset times for caller
    times_rel -= left_margin
    res = PeelingBatchResult(
        n_spikes=times_rel.numel(), times_rel=times_rel, **features
    )
    return residual[:, :-1], res


def threshold_to_fit(
    pipeline: "WaveformPipeline",
    recording: BaseRecording,
    waveform_cfg: WaveformConfig,
    channel_index: Tensor,
    spatial_dedup_radius: float | None,
    threshold_cfg: ThresholdingConfig,
    sampling_cfg: FitSamplingConfig,
    max_waveforms_fit: int | None = None,
    n_residual_snips: int | None = None,
    computation_cfg: ComputationConfig | None = None,
    tmp_dir=None,
):
    """Run a Thresholding peeling to fit a FeaturizationPipeline.

    Used by subtraction to fit initial NN denoisers.
    """
    from ..transform import Waveform, WaveformPipeline
    from ..util.data_util import subsample_waveforms
    from .threshold import Threshold

    computation_cfg = ensure_computation_config(computation_cfg)

    geom = recording.get_channel_locations()
    waveform_node = Waveform(
        channel_index=channel_index,
        waveform_cfg=waveform_cfg,
        sampling_frequency=recording.sampling_frequency,
    )
    waveform_pipeline = WaveformPipeline([waveform_node])

    if spatial_dedup_radius:
        dn_dedup_ci = make_channel_index(geom, spatial_dedup_radius, to_torch=True)
        dn_dedup_ci = dn_dedup_ci.to(channel_index)
    else:
        dn_dedup_ci = channel_index
    trainer = Threshold(
        recording=recording,
        channel_index=channel_index,
        featurization_pipeline=waveform_pipeline,
        p=threshold_cfg,
        waveform_cfg=waveform_cfg,
        dedup_channel_index=dn_dedup_ci,
        fit_sampling_cfg=sampling_cfg,
    )

    if max_waveforms_fit is None:
        max_waveforms_fit = sampling_cfg.max_waveforms_fit

    if pipeline.needs_residual():
        n_resid_snips = n_residual_snips or sampling_cfg.n_residual_snips
    else:
        n_resid_snips = None

    if tmp_dir is None:
        tmp_dir = computation_cfg.tmpdir_parent
    with TemporaryDirectory(dir=tmp_dir) as temp_dir:
        temp_hdf5_filename = Path(temp_dir) / "subtraction_denoiser0_fit.h5"
        try:
            trainer.run_subsampled_peeling(
                temp_hdf5_filename,
                stop_after_n_waveforms=max_waveforms_fit,
                task_name="Load initial denoiser fit data",
                total_residual_snips=n_resid_snips,
                computation_cfg=computation_cfg,
            )

            # get fit weights
            device = computation_cfg.actual_device()
            waveforms, fixed_properties = subsample_waveforms(
                temp_hdf5_filename,
                fit_sampling=sampling_cfg.fit_sampling,
                random_state=sampling_cfg.seed,
                n_waveforms_fit=max_waveforms_fit,
                fit_max_reweighting=sampling_cfg.fit_max_reweighting,
                voltages_dataset_name="voltages",
                waveforms_dataset_name="waveforms",
                subsample_by_weighting=True,
            )

            # fit the thing
            pipeline = pipeline.to(device)
            pipeline.fit(
                recording=recording,
                waveforms=waveforms,
                computation_cfg=computation_cfg,
                hdf5_filename=temp_hdf5_filename,
                **fixed_properties,  # type: ignore
            )
            pipeline.to("cpu")
        finally:
            if temp_hdf5_filename.exists():
                temp_hdf5_filename.unlink()

    return pipeline
