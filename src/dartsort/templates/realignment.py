import gc
from dataclasses import replace
import math

import numpy as np
import torch
import torch.nn.functional as F
from spikeinterface.core import BaseRecording
from tqdm.auto import trange

from ..util.data_util import DARTsortSorting
from ..util.internal_config import (
    ComputationConfig,
    RealignStrategy,
    TemplateRealignmentConfig,
    WaveformConfig,
    default_waveform_cfg,
)
from ..util.job_util import ensure_computation_config
from ..util.logging_util import get_logger
from ..util.spiketorch import ptp
from .templates import TemplateData

logger = get_logger(__name__)


def realign(
    *,
    recording: BaseRecording,
    sorting: DARTsortSorting,
    realign_cfg: TemplateRealignmentConfig | None = None,
    waveform_cfg: WaveformConfig = default_waveform_cfg,
    computation_cfg: ComputationConfig | None = None,
    motion_est=None,
) -> tuple[DARTsortSorting, TemplateData | None]:
    if realign_cfg is None:
        return sorting, None
    if not realign_cfg.realign_peaks:
        return sorting, None
    if not realign_cfg.realign_shift_ms:
        return sorting, None

    realignment_waveform_cfg = WaveformConfig(
        ms_before=waveform_cfg.ms_before + realign_cfg.realign_shift_ms,
        ms_after=waveform_cfg.ms_after + realign_cfg.realign_shift_ms,
    )
    templates = TemplateData.from_config(
        recording=recording,
        sorting=sorting,
        template_cfg=realign_cfg.template_cfg,
        waveform_cfg=realignment_waveform_cfg,
        computation_cfg=computation_cfg,
        motion_est=motion_est,
    )
    trough_offset_samples = waveform_cfg.trough_offset_samples(
        sampling_frequency=recording.sampling_frequency
    )
    spike_length_samples = waveform_cfg.spike_length_samples(
        sampling_frequency=recording.sampling_frequency
    )

    template_shifts, aligned_templates = realign_templates(
        templates=templates.templates,
        snrs_by_channel=templates.snrs_by_channel(),
        unit_ids=templates.unit_ids,
        padded_trough_offset_samples=realignment_waveform_cfg.trough_offset_samples(
            sampling_frequency=recording.sampling_frequency
        ),
        trough_offset_samples=trough_offset_samples,
        realign_strategy=realign_cfg.realign_strategy,
        trough_factor=realign_cfg.trough_factor,
        computation_cfg=computation_cfg,
    )
    templates = replace(
        templates,
        templates=aligned_templates,
        trough_offset_samples=trough_offset_samples,
    )
    aligned_sorting = apply_time_shifts(
        sorting,
        template_shifts=template_shifts,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        recording_length_samples=recording.get_num_samples(),
    )
    return aligned_sorting, templates


@torch.jit.script
def singlechan_alignments(
    traces: torch.Tensor, trough_factor: float = 3.0, dim: int = 1
) -> torch.Tensor:
    aligner_traces = torch.where(traces < 0, trough_factor * traces, -traces)
    offsets = aligner_traces.argmin(dim=dim)
    return offsets


def get_main_channels_and_alignments(
    template_data=None, trough_factor=3.0, templates=None, main_channels=None
):
    if templates is None:
        assert template_data is not None
        templates = template_data.templates
    if main_channels is None:
        main_channels = np.ptp(templates, axis=1).argmax(1)
    mc_traces = np.take_along_axis(templates, main_channels[:, None, None], axis=2)[
        :, :, 0
    ]
    aligner_traces = np.where(mc_traces < 0, trough_factor * mc_traces, -mc_traces)
    aligner_traces = np.abs(aligner_traces, out=aligner_traces)
    # break ties late
    tmax = aligner_traces.shape[1] - 1
    offsets = tmax - aligner_traces[:, ::-1].argmax(1)
    return main_channels, mc_traces, offsets


def estimate_offset(
    *,
    templates,
    computation_cfg: ComputationConfig,
    snrs_by_channel=None,
    strategy="mainchan_trough_factor",
    trough_factor=3.0,
    min_weight=0.75,
    main_channels=None,
    padded_trough_offset_samples=42,
    trough_offset_samples=42,
) -> torch.Tensor:
    templates = torch.asarray(templates)

    if strategy == "mainchan_trough_factor":
        _, _, offsets = get_main_channels_and_alignments(
            None,
            trough_factor=trough_factor,
            templates=templates.numpy(force=True),
            main_channels=main_channels,
        )
        return offsets - padded_trough_offset_samples

    if strategy == "dredge":
        offsets = dredge_realign(
            templates=templates,
            main_channels=main_channels,
            snrs_by_channel=snrs_by_channel,
            trough_factor=trough_factor,
            padded_trough_offset_samples=padded_trough_offset_samples,
            trough_offset_samples=trough_offset_samples,
            computation_cfg=computation_cfg,
        )
        gc.collect()
        torch.cuda.empty_cache()
        return offsets

    if strategy == "snr_weighted_trough_factor":
        assert snrs_by_channel is not None
        weights = torch.asarray(snrs_by_channel).clone()
        weights /= weights.amax(dim=1, keepdim=True)
        weights[weights < min_weight] = 0.0
        weights /= weights.sum(dim=1, keepdim=True)
        tmp = templates.clone()
        tmp[tmp < 0] *= trough_factor
        tmp.abs_()
        offsets = tmp.argmax(dim=1).double()
        offsets = torch.sum(offsets * weights, dim=1)
        offsets = torch.round(offsets).long()
        return offsets - padded_trough_offset_samples

    if strategy == "normsq_weighted_trough_factor":
        tmp = templates.square()
        weights = tmp.sum(dim=1)
        weights /= weights.amax(dim=1, keepdim=True)
        weights[weights < min_weight] = 0.0
        weights /= weights.sum(dim=1, keepdim=True)
        tmp[:] = templates
        tmp[tmp < 0] *= trough_factor
        tmp.abs_()
        offsets = tmp.argmax(dim=1).double()
        offsets = torch.sum(offsets * weights, dim=1)
        offsets = torch.round(offsets).long()
        return offsets - padded_trough_offset_samples

    if strategy == "ampsq_weighted_trough_factor":
        weights = ptp(templates).square()
        weights /= weights.amax(dim=1, keepdim=True)
        weights[weights < min_weight] = 0.0
        weights /= weights.sum(dim=1, keepdim=True)
        tmp = templates.clone()
        tmp[tmp < 0] *= trough_factor
        tmp.abs_()
        offsets = tmp.argmax(dim=1).double()
        offsets = torch.sum(offsets * weights, dim=1)
        offsets = torch.round(offsets).long()
        return offsets - padded_trough_offset_samples

    assert False


def realign_templates(
    *,
    templates,
    snrs_by_channel=None,
    unit_ids=None,
    main_channels=None,
    padded_trough_offset_samples=42,
    trough_offset_samples=42,
    realign_strategy: RealignStrategy = "mainchan_trough_factor",
    trough_factor=3.0,
    computation_cfg: ComputationConfig | None = None,
):
    computation_cfg = ensure_computation_config(computation_cfg)
    if main_channels is None:
        if realign_strategy in ("mainchan_trough_factor", "dredge"):
            if snrs_by_channel is not None:
                main_channels = snrs_by_channel.argmax(1)
            else:
                main_channels = np.ptp(templates, axis=1).argmax(1)
        elif realign_strategy.startswith("snr_weighted"):
            assert snrs_by_channel is not None
            main_channels = snrs_by_channel.argmax(1)
        elif realign_strategy.startswith("normsq_weighted"):
            main_channels = np.square(templates).sum(1).argmax(1)
        elif realign_strategy.startswith("ampsq_weighted"):
            main_channels = np.ptp(templates).max(1).argmax(1)
        else:
            assert False
    assert main_channels is not None

    # find template peak time
    max_shift = padded_trough_offset_samples - trough_offset_samples
    template_shifts__ = estimate_offset(
        computation_cfg=computation_cfg,
        templates=templates,
        snrs_by_channel=snrs_by_channel,
        strategy=realign_strategy,
        trough_factor=trough_factor,
        main_channels=main_channels,
        padded_trough_offset_samples=padded_trough_offset_samples,
        trough_offset_samples=trough_offset_samples,
    )
    if torch.is_tensor(template_shifts__):
        template_shifts_ = template_shifts__.numpy(force=True)
    else:
        template_shifts_ = template_shifts__

    # clip if needed
    template_shifts_[np.abs(template_shifts_) > max_shift] = 0

    # find unit sample time shifts
    if unit_ids is None:
        template_shifts = template_shifts_
    else:
        template_shifts = np.zeros(unit_ids.max() + 1, dtype=np.int64)
        template_shifts[unit_ids] = template_shifts_

    aligned_templates = trim_templates_to_shift(
        templates=templates,
        max_shift=max_shift,
        template_shifts=template_shifts,
        unit_ids=unit_ids,
    )

    return template_shifts, aligned_templates


def trim_templates_to_shift(
    templates, max_shift=0, template_shifts=None, unit_ids=None
):
    if not max_shift or template_shifts is None:
        return templates

    n, t, c = templates.shape

    # trim templates
    aligned_spike_len = t - 2 * max_shift
    aligned_templates = np.empty((n, aligned_spike_len, c), dtype=templates.dtype)
    if unit_ids is None:
        assert len(templates) == len(template_shifts) == n
    else:
        template_shifts = template_shifts[unit_ids]
    for i, dt in enumerate(template_shifts):
        aligned_templates[i] = templates[
            i, max_shift + dt : max_shift + dt + aligned_spike_len
        ]

    return aligned_templates


def apply_time_shifts(
    sorting,
    template_shifts=None,
    template_data=None,
    trough_offset_samples=None,
    spike_length_samples=None,
    recording_length_samples=None,
) -> DARTsortSorting:
    if template_shifts is not None:
        unit_ids = None
    elif template_data is not None and template_data.properties:
        if "template_time_shifts" in template_data.properties:
            template_shifts = template_data.properties["template_time_shifts"]
        elif "time_shifts" in template_data.properties:
            template_shifts = template_data.properties["time_shifts"]
        unit_ids = template_data.unit_ids
    else:
        unit_ids = None

    if template_shifts is None:
        return sorting

    ixs = sorting.labels
    valid = np.flatnonzero(ixs >= 0)
    ixsv = ixs[valid]
    if unit_ids is not None and not np.array_equal(unit_ids, np.arange(len(unit_ids))):
        ixsv = np.searchsorted(unit_ids, ixsv)
        assert np.array_equal(unit_ids[ixsv], sorting.labels[valid])

    new_times = sorting.times_samples.copy()
    new_times[valid] += template_shifts[ixsv]

    if recording_length_samples is not None:
        assert spike_length_samples is not None
        assert trough_offset_samples is not None
        labels = sorting.labels.copy()
        tail_samples = spike_length_samples - trough_offset_samples
        highlim = recording_length_samples - tail_samples
        labels[new_times < trough_offset_samples] = -1
        labels[new_times >= highlim] = -1
    else:
        labels = sorting.labels

    return sorting.ephemeral_replace(labels=labels, times_samples=new_times)


def dredge_realign(
    *,
    templates: torch.Tensor,
    main_channels: torch.Tensor | None,
    snrs_by_channel: torch.Tensor | None,
    min_spatial_cosine: float = 0.75,
    min_corr: float = 0.5,
    trough_factor: float = 3.0,
    padded_trough_offset_samples: int,
    trough_offset_samples: int,
    computation_cfg: ComputationConfig,
    dredge_window_fraction: float = 0.5,
    eps=1e-3,
) -> torch.Tensor:
    from dredge.dredgelib import newton_solve_rigid

    max_shift = padded_trough_offset_samples - trough_offset_samples

    if main_channels is None and snrs_by_channel is not None:
        main_channels = snrs_by_channel.argmax(1)

    # start with trough alignment
    main_channels, main_channel_traces, offsets = get_main_channels_and_alignments(
        None,
        trough_factor=trough_factor,
        templates=templates.numpy(force=True),
        main_channels=main_channels,
    )
    offsets = offsets - padded_trough_offset_samples
    offsets[np.abs(offsets) > max_shift] = 0

    # trim to alignment
    templates_trim0 = trim_templates_to_shift(
        templates=templates.numpy(force=True),
        max_shift=max_shift,
        template_shifts=offsets,
    )

    # which pairs to convolve?
    sp = snrs_by_channel if snrs_by_channel is not None else ptp(templates_trim0)
    sp = torch.asarray(sp)
    sp = sp / torch.linalg.norm(sp, dim=1)[:, None]
    spcos = sp @ sp.T
    pair_mask = spcos >= min_spatial_cosine
    ii, jj = pair_mask.nonzero(as_tuple=True)
    triu = ii < jj
    ii = ii[triu]
    jj = jj[triu]
    logger.dartsortdebug(
        f"DREDge align: convolving {ii.numel()} pairs (mincos={min_spatial_cosine:.2f})."
    )

    # get best lags and correlations
    templates_trim0 = torch.asarray(
        templates_trim0, dtype=torch.float, device=computation_cfg.actual_device()
    )
    dredge_max_shift = math.floor(max_shift * dredge_window_fraction)
    lags, corrs = _pairwise_correlate_templates(
        templates=templates_trim0, ii=ii, jj=jj, max_shift=dredge_max_shift
    )
    lags = lags.numpy(force=True)
    corrs = corrs.numpy(force=True)
    logger.dartsortdebug(
        f"DREDge align: {100 * (corrs >= min_corr).mean():.1f}% "
        "of pairs were correlated enough."
    )

    # densify
    k = len(templates)
    D = np.zeros((k, k))
    C = D.copy()
    D[ii, jj] = lags
    D[jj, ii] = -lags
    C[ii, jj] = corrs
    C[jj, ii] = corrs
    np.fill_diagonal(C, 1.0)

    # corrs->weights
    C[C < min_corr] = 0.0

    # call out to dredge
    Sigma0inv = np.eye(C.shape[0]) * eps
    p, *_ = newton_solve_rigid(D, C, Sigma0inv)
    p = np.rint(p).astype(offsets.dtype)

    # adjust offsets
    offsets = torch.asarray(offsets) - torch.asarray(p)

    return offsets


def _pairwise_correlate_templates(templates, ii, jj, max_shift: int, batch_size=256):
    lags = torch.arange(-max_shift, max_shift + 1).to(templates.device)
    k = len(templates)
    npair = len(ii)

    # compute norms...
    normsa = templates.new_zeros((k, lags.shape[0]))
    normsb = templates.new_zeros((k, lags.shape[0]))
    for i0 in range(0, k, batch_size):
        i1 = min(k, i0 + batch_size)

        filt = torch.square(templates[i0:i1])[:, None]
        ones = torch.ones_like(templates[i0:i1])[None, :]
        conv = F.conv2d(ones, filt, padding=(max_shift, 0), groups=i1 - i0)
        assert conv.shape == (1, i1 - i0, lags.shape[0], 1)
        conv = conv[0, :, :, 0].sqrt_()
        normsa[i0:i1] = conv
        normsb[i0:i1] = torch.flip(conv, dims=(1,))

    best_lag = templates.new_zeros(ii.shape)
    best_corr = templates.new_zeros(ii.shape)

    for i0 in trange(0, npair, batch_size, desc="DREDge template alignment"):
        i1 = min(npair, i0 + batch_size)

        iib = ii[i0:i1]
        jjb = jj[i0:i1]

        filt = templates[iib][:, None]
        inpt = templates[jjb][None, :]
        conv = F.conv2d(inpt, filt, padding=(max_shift, 0), groups=i1 - i0)
        assert conv.shape == (1, i1 - i0, lags.shape[0], 1)
        conv = conv[0, :, :, 0].div_(normsa[iib] * normsb[jjb])
        best_corr[i0:i1], lag_ix = conv.max(dim=1)
        best_lag[i0:i1] = lags[lag_ix]

    return best_lag, best_corr
