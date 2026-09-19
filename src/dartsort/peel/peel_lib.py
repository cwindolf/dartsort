import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
import torch.nn.functional as F
from spikeinterface.core import BaseRecording
from torch import Tensor

from ..detect import detect_and_deduplicate
from ..util.internal_config import (
    ComputationConfig,
    FitSamplingConfig,
    PeakSign,
    ThresholdingConfig,
    WaveformConfig,
)
from ..util.job_util import ensure_computation_config
from ..util.spiketorch import grab_spikes, subtract_spikes_
from ..util.torch_util import torch_compile
from ..util.waveform_util import make_channel_index
from .peel_base import PeelingBatchResult

if TYPE_CHECKING:
    from ..transform.pipeline import WaveformPipeline


def check_residual_decrease(
    orig_wfs: Tensor | None,
    dn_wfs: Tensor,
    threshold=10.0,
    save_residnorm_decrease=False,
    overwrite_orig_waveforms: bool = False,
    local_whiteners: Tensor | None = None,
    whitening_kernel: Tensor | None = None,
    channels: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    if not threshold:
        mask = dn_wfs.new_ones(len(dn_wfs), dtype=torch.bool)
        return mask, {}
    assert orig_wfs is not None

    orig_wfs = flatten_denan_and_whiten_batched(
        orig_wfs,
        channels,
        local_whiteners,
        whitening_kernel,
        overwrite=overwrite_orig_waveforms,
    )
    dn_wfs = flatten_denan_and_whiten_batched(
        dn_wfs, channels, local_whiteners, whitening_kernel
    )

    if overwrite_orig_waveforms:
        buf = orig_wfs.mul_(dn_wfs)
        conv = buf.sum(dim=1)
        torch.square(dn_wfs, out=buf)
        norm = buf.sum(dim=1)
    else:
        conv = (orig_wfs * dn_wfs).sum(dim=1)
        norm = dn_wfs.square_().sum(dim=1)
    reduction = conv.mul_(2.0).sub_(norm)
    threshold = threshold**2

    mask = cast(torch.Tensor, threshold < reduction)
    if save_residnorm_decrease:
        features = dict(residnorm_decreases=reduction)
    else:
        features = {}
    return mask, features


@torch_compile
def flatten_denan_and_whiten_batched(
    wfs: torch.Tensor,
    channels: torch.Tensor | None,
    local_whiteners: Tensor | None,
    whitening_kernel: Tensor | None,
    overwrite: bool = False,
    batch_size: int = 4096,
):
    N = wfs.shape[0]
    T = wfs.shape[1]
    C = wfs.shape[2]
    if overwrite:
        wfs = wfs.nan_to_num_()
    else:
        wfs = wfs.nan_to_num()

    if local_whiteners is None:
        return wfs.view(N, -1)

    assert channels is not None

    W = local_whiteners[channels]
    wfs = W.bmm(wfs.mT)
    if whitening_kernel is None:
        return wfs.view(N, -1)

    k = whitening_kernel[None, None]
    NC = N * C
    wfs = wfs.view(NC, 1, T)

    for i0 in range(0, NC, batch_size):
        i1 = min(NC, i0 + batch_size)
        nb = i1 - i0
        y = wfs[i0:i1]
        if nb < batch_size:
            y = F.pad(y, (0, 0, 0, 0, 0, batch_size - nb))
        z = F.conv1d(y, k, padding="same")
        if nb < batch_size:
            z = z[:nb]
        wfs[i0:i1] = z

    return wfs.view(N, -1)


def threshold_chunk(
    traces,
    channel_index,
    detection_threshold=4.0,
    peak_sign: PeakSign = "both",
    peak_channel_index=None,
    dedup_channel_index=None,
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
    return_waveforms=True,
    rg=None,
    quiet=False,
) -> PeelingBatchResult:
    n_index = channel_index.shape[1]
    times_rel, channels, energies = detect_and_deduplicate(
        traces,
        threshold=detection_threshold,
        peak_channel_index=peak_channel_index,
        dedup_neighborhoods=dedup_channel_index,
        peak_sign=peak_sign,
        dedup_temporal_radius=temporal_dedup_radius_samples,
        remove_exact_duplicates=remove_exact_duplicates,
        relative_peak_radius=relative_peak_radius,
        return_energies=True,
        trough_priority=trough_priority,
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
                f"{max_spikes_per_chunk=}. Keeping the top ones.",
                stacklevel=2,
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
        dedup_neighborhoods=dedup_channel_index,
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
        tmp_dir = computation_cfg.maybe_tmpdir_parent()
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
            waveform_dict, fixed_properties = subsample_waveforms(
                temp_hdf5_filename,
                fit_sampling=sampling_cfg.fit_sampling,
                random_state=sampling_cfg.seed,
                n_waveforms_fit=max_waveforms_fit,
                fit_max_reweighting=sampling_cfg.fit_max_reweighting,
                voltages_dataset_name="voltages",
                waveforms_dataset_name="waveforms",
                subsample_by_weighting=True,
            )
            waveforms = waveform_dict["waveforms"]
            if not len(waveforms):
                raise ValueError(
                    "Found no spikes when thresholding to get model fitting data. "
                    "This usually indicates a preprocessing issue, since it means "
                    f"that no spikes could be found at threshold {threshold_cfg.detection_threshold}."
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
