from typing import Literal

import numpy as np
import torch
from spikeinterface import BaseRecording

from ..transform import WaveformPipeline
from ..util import spiketorch
from ..util.data_util import DARTsortSorting
from ..util.internal_config import (
    FeaturizationConfig,
    FitSamplingConfig,
    WaveformConfig,
)
from ..util.waveform_util import make_channel_index
from .peel_base import (
    BasePeeler,
    PeelingBatchResult,
    SpikeDataset,
    peeling_empty_result,
)


class GrabAndFeaturize(BasePeeler):
    """Extract and featurize raw waveform snippets at known times."""

    peel_kind = "Grab and featurize"

    def __init__(
        self,
        recording,
        channel_index,
        featurization_pipeline,
        times_samples,
        channels,
        labels=None,
        trough_offset_samples=42,
        spike_length_samples=121,
        chunk_length_samples=30_000,
        n_waveforms_fit=20_000,
        max_waveforms_fit=50_000,
        fit_sampling: Literal["random", "amp_reweighted"] = "random",
        n_seconds_fit=40,
        fit_subsampling_random_state: int | np.random.Generator = 0,
        dtype=torch.float,
    ):
        super().__init__(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            chunk_length_samples=chunk_length_samples,
            chunk_margin_samples=max(
                trough_offset_samples, spike_length_samples - trough_offset_samples
            ),
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            n_seconds_fit=n_seconds_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
            n_waveforms_fit=n_waveforms_fit,
            max_waveforms_fit=max_waveforms_fit,
            fit_sampling=fit_sampling,
            dtype=dtype,
        )
        self.register_buffer("times_samples", torch.asarray(times_samples))
        self.register_buffer("channels", torch.asarray(channels))
        if labels is not None:
            self.register_buffer("labels", torch.asarray(labels))
        else:
            self.labels = None
        assert self.times_samples.ndim == 1
        assert self.times_samples.shape == self.channels.shape

    def out_datasets(self):
        datasets = super().out_datasets()
        datasets.append(
            SpikeDataset(name="indices", shape_per_spike=(), dtype=np.int64)
        )
        if self.labels is not None:
            datasets.append(
                SpikeDataset(name="labels", shape_per_spike=(), dtype=np.int64)
            )
        return datasets

    def process_chunk(
        self,
        chunk_start_samples: int,
        *,
        chunk_end_samples: int | None = None,
        return_residual: bool = False,
        skip_features: bool = False,
        n_resid_snips: int | None = None,
        to_cpu: bool = True,
    ) -> "PeelingBatchResult":
        """Override process_chunk to skip empties."""
        if chunk_end_samples is None:
            chunk_end_samples = min(
                self.recording.get_num_samples(),
                chunk_start_samples + self.chunk_length_samples,
            )
        t_clip = self.b.times_samples.clip(chunk_start_samples, chunk_end_samples - 1)
        in_chunk = self.b.times_samples == t_clip
        if not in_chunk.any():
            return peeling_empty_result

        res = super().process_chunk(
            chunk_start_samples,
            n_resid_snips=n_resid_snips,
            chunk_end_samples=chunk_end_samples,
            return_residual=return_residual,
            skip_features=skip_features,
            to_cpu=to_cpu,
        )
        return res

    @classmethod
    def from_config(
        cls,
        recording: BaseRecording,
        *,
        sorting: DARTsortSorting,
        waveform_cfg: WaveformConfig,
        featurization_cfg: FeaturizationConfig,
        sampling_cfg: FitSamplingConfig,
    ):
        geom = torch.tensor(recording.get_channel_locations())
        channel_index = make_channel_index(
            geom, featurization_cfg.extract_radius, to_torch=True
        )
        featurization_pipeline = WaveformPipeline.from_config(
            geom=geom,
            channel_index=channel_index,
            featurization_cfg=featurization_cfg,
            waveform_cfg=waveform_cfg,
            sampling_frequency=recording.sampling_frequency,
        )
        trough_offset_samples = waveform_cfg.trough_offset_samples(
            recording.sampling_frequency
        )
        spike_length_samples = waveform_cfg.spike_length_samples(
            recording.sampling_frequency
        )
        return cls(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            times_samples=sorting.times_samples,
            channels=sorting.channels,
            labels=sorting.labels,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            chunk_length_samples=30_000,
            n_seconds_fit=100,
            dtype=torch.float,
            n_waveforms_fit=sampling_cfg.n_waveforms_fit,
            max_waveforms_fit=sampling_cfg.max_waveforms_fit,
            fit_subsampling_random_state=sampling_cfg.fit_subsampling_random_state,
            fit_sampling=sampling_cfg.fit_sampling,
        )

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
        assert not return_residual

        max_t = chunk_start_samples + self.chunk_length_samples - 1
        in_chunk = self.b.times_samples == self.b.times_samples.clip(
            chunk_start_samples, max_t
        )
        (in_chunk,) = in_chunk.nonzero(as_tuple=True)

        if not in_chunk.numel():
            return peeling_empty_result

        chunk_left = chunk_start_samples - left_margin
        times_rel = self.b.times_samples[in_chunk] - chunk_left
        channels = self.b.channels[in_chunk]

        assert times_rel.ndim == 1
        assert channels.ndim == 1

        n_spikes = in_chunk.numel()
        res = PeelingBatchResult(
            n_spikes=n_spikes,
            indices=in_chunk,
            times_samples=self.b.times_samples[in_chunk],
            channels=channels,
        )
        if return_waveforms:
            res["collisioncleaned_waveforms"] = spiketorch.grab_spikes(
                traces,
                times_rel,
                channels,
                self.channel_index,
                trough_offset=self.trough_offset_samples,
                spike_length_samples=self.spike_length_samples,
                already_padded=False,
                pad_value=torch.nan,
            )
        if self.labels is not None:
            res["labels"] = self.labels[in_chunk]
        return res
