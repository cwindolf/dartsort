"""A naive version of the subtraction peeler

The Shaver is in between thresholding and full subtraction. It detects the
spikes that thresholding would, discards those that the denoiser doesn't like,
and does a naive collision cleaning in one pass.

In other words, the Shaver leaves razor bumps.
"""
# TODO fit denoising pipeline
# TODO shave() in main

import numpy as np
import torch
from spikeinterface.core import BaseRecording

from ..transform import WaveformPipeline
from ..util.data_util import SpikeDataset
from ..util.internal_config import (
    FeaturizationConfig,
    FitSamplingConfig,
    ThresholdingConfig,
    WaveformConfig,
    default_peeling_fit_sampling_cfg,
    default_thresholding_cfg,
    default_waveform_cfg,
)
from ..util.waveform_util import get_channel_index_rel_inds, make_channel_index
from .peel_base import BasePeeler, PeelingBatchResult, peeling_empty_result
from .peel_lib import shave_chunk


class Shaver(BasePeeler):
    def __init__(
        self,
        recording: BaseRecording,
        channel_index: np.ndarray | torch.Tensor,
        denoising_pipeline: WaveformPipeline,
        featurization_pipeline: WaveformPipeline | None = None,
        p: ThresholdingConfig = default_thresholding_cfg,
        waveform_cfg: WaveformConfig = default_waveform_cfg,
        peak_channel_index=None,
        dedup_channel_index=None,
        fit_sampling_cfg: FitSamplingConfig = default_peeling_fit_sampling_cfg,
        dtype=torch.float,
    ):
        fixed_prop_keys = ("channels", "times_seconds")
        spike_length_samples = waveform_cfg.spike_length_samples(
            recording.sampling_frequency
        )
        super().__init__(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            chunk_length_samples=p.chunk_length_samples,
            waveform_cfg=waveform_cfg,
            chunk_margin_samples=self.next_margin(spike_length_samples, factor=5),
            fit_sampling_cfg=fit_sampling_cfg,
            dtype=dtype,
            fixed_property_keys=fixed_prop_keys,
        )
        self.p = p
        self.peel_kind = f"Shave {p.detection_threshold}"
        self.denoising_pipeline = denoising_pipeline
        self.dedup_batch_size = self.nearest_batch_length()

        geom = recording.get_channel_locations()
        if peak_channel_index is None and p.relative_peak_radius_um:
            peak_channel_index = make_channel_index(
                geom, p.relative_peak_radius_um, to_torch=True
            )
        if dedup_channel_index is None and p.spatial_dedup_radius_um:
            dedup_channel_index = make_channel_index(
                geom, p.spatial_dedup_radius_um, to_torch=True
            )
        self.register_buffer_or_none("dedup_channel_index", dedup_channel_index)
        if dedup_channel_index is not None:
            dedup_rel_inds = get_channel_index_rel_inds(dedup_channel_index)
        else:
            dedup_rel_inds = None
        self.register_buffer_or_none("dedup_rel_inds", dedup_rel_inds)
        self.register_buffer_or_none("peak_channel_index", peak_channel_index)

    @classmethod
    def from_config(
        cls,
        *,
        recording: BaseRecording,
        waveform_cfg: WaveformConfig,
        thresholding_cfg: ThresholdingConfig,
        featurization_cfg: FeaturizationConfig,
        sampling_cfg: FitSamplingConfig,
        extract_channel_index: torch.Tensor | None = None,
        denoising_pipeline: WaveformPipeline,
        featurization_pipeline: WaveformPipeline | None = None,
    ):
        geom = torch.tensor(recording.get_channel_locations())
        if extract_channel_index is None:
            channel_index = make_channel_index(
                geom, featurization_cfg.extract_radius, to_torch=True
            )
        else:
            channel_index = extract_channel_index

        if featurization_pipeline is None:
            featurization_pipeline = WaveformPipeline.from_config(
                geom=geom,
                channel_index=channel_index,
                featurization_cfg=featurization_cfg,
                waveform_cfg=waveform_cfg,
                sampling_frequency=recording.sampling_frequency,
            )

        return cls(
            recording,
            channel_index,
            denoising_pipeline=denoising_pipeline,
            featurization_pipeline=featurization_pipeline,
            p=thresholding_cfg,
            waveform_cfg=waveform_cfg,
            fit_sampling_cfg=sampling_cfg,
        )

    def out_datasets(self):
        datasets = super().out_datasets()
        datasets.append(SpikeDataset(name="voltages", shape_per_spike=(), dtype=float))
        return datasets

    def peel_chunk(
        self,
        traces,
        *,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
        return_waveforms=True,
    ) -> PeelingBatchResult:
        residual, res = shave_chunk(
            traces=traces,
            channel_index=self.b.channel_index,
            denoising_pipeline=self.denoising_pipeline,
            residnorm_decrease_threshold=self.p.shave_score,
            detection_threshold=self.p.detection_threshold,
            peak_sign=self.p.peak_sign,
            dedup_channel_index=self.b.dedup_channel_index,
            dedup_rel_inds=self.b.dedup_rel_inds,
            dedup_batch_size=self.dedup_batch_size,
            trough_offset_samples=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
            left_margin=left_margin,
            right_margin=right_margin,
            peak_channel_index=self.b.peak_channel_index,
            temporal_dedup_radius_samples=self.p.temporal_dedup_radius_samples,
            remove_exact_duplicates=self.p.remove_exact_duplicates,
            trough_priority=self.p.trough_priority,
        )
        if not res["n_spikes"]:
            return peeling_empty_result

        # get absolute times
        times_rel = res.pop("times_rel")
        assert times_rel is not None
        res["times_samples"] = times_rel + chunk_start_samples
        assert "waveforms" in res
        waveforms = res.pop("waveforms")
        if return_waveforms:
            res["collisioncleaned_waveforms"] = waveforms
        if return_residual:
            res["residual"] = residual[left_margin : traces.shape[0] - right_margin]
        return res
