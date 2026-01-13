from dataclasses import replace

import torch
import numpy as np
from spikeinterface.core import BaseRecording

from ..util.internal_config import (
    RealignStrategy,
    TemplateConfig,
    WaveformConfig,
    TemplateRealignmentConfig,
    ComputationConfig,
    default_waveform_cfg,
)
from ..util.data_util import DARTsortSorting
from ..util.spiketorch import ptp
from .templates import TemplateData


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
        spike_length_samples=spike_length_samples,
        realign_strategy=realign_cfg.realign_strategy,
        trough_factor=realign_cfg.trough_factor,
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
    offsets = np.abs(aligner_traces).argmax(1)
    return main_channels, mc_traces, offsets


def estimate_offset(
    templates,
    denoising_tsvd=None,
    tsvd_slice=slice(None),
    snrs_by_channel=None,
    strategy="mainchan_trough_factor",
    trough_factor=3.0,
    min_weight=0.75,
    main_channels=None,
):
    templates = torch.asarray(templates)
    if strategy == "mainchan_trough_factor":
        _, _, offsets = get_main_channels_and_alignments(
            None,
            trough_factor=trough_factor,
            templates=templates.numpy(force=True),
            main_channels=main_channels,
        )
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
        return offsets

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
        return offsets

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
        return offsets

    assert False


def realign_templates(
    templates,
    denoising_tsvd=None,
    snrs_by_channel=None,
    unit_ids=None,
    main_channels=None,
    padded_trough_offset_samples=42,
    trough_offset_samples=42,
    spike_length_samples=121,
    max_shift=20,
    realign_strategy: RealignStrategy = "mainchan_trough_factor",
    trough_factor=3.0,
):
    if main_channels is None:
        if realign_strategy.startswith("mainchan"):
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
    pad = padded_trough_offset_samples - trough_offset_samples
    template_peak_times = estimate_offset(
        templates,
        denoising_tsvd=denoising_tsvd,
        snrs_by_channel=snrs_by_channel,
        strategy=realign_strategy,
        trough_factor=trough_factor,
        main_channels=main_channels,
        tsvd_slice=slice(pad, pad + spike_length_samples),
    )

    # find unit sample time shifts
    if torch.is_tensor(template_peak_times):
        template_peak_times = template_peak_times.numpy(force=True)
    assert template_peak_times is not None
    template_shifts_ = template_peak_times - padded_trough_offset_samples
    template_shifts_[np.abs(template_shifts_) > max_shift] = 0
    if unit_ids is None:
        template_shifts = template_shifts_
    else:
        template_shifts = np.zeros(unit_ids.max() + 1, dtype=np.int64)
        template_shifts[unit_ids] = template_shifts_

    aligned_templates = trim_templates_to_shift(
        templates,
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
    if template_data is not None and template_data.properties:
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
