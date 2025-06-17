import torch
import numpy as np

from .peel_base import BasePeeler, SpikeDataset
from ..transform import WaveformPipeline
from ..util.waveform_util import make_channel_index
from ..util import spiketorch


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
        n_chunks_fit=40,
        fit_subsampling_random_state=0,
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
            n_chunks_fit=n_chunks_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
            dtype=dtype,
        )
        self.register_buffer("times_samples", torch.asarray(times_samples))
        self.register_buffer("channels", torch.asarray(channels))
        self.labels = labels
        assert self.times_samples.ndim == 1
        assert self.times_samples.shape == self.channels.shape

    def out_datasets(self):
        datasets = super().out_datasets()
        datasets.append(SpikeDataset(name="indices", shape_per_spike=(), dtype=np.int64))
        if self.labels is not None:
            datasets.append(SpikeDataset(name="labels", shape_per_spike=(), dtype=np.int64))
        return datasets

    def process_chunk(
        self,
        chunk_start_samples,
        n_resid_snips=None,
        chunk_end_samples=None,
        return_residual=False,
        skip_features=False,
    ):
        """Override process_chunk to skip empties."""
        if chunk_end_samples is None:
            chunk_end_samples = min(
                self.recording.get_num_samples(),
                chunk_start_samples + self.chunk_length_samples,
            )
        in_chunk = self.times_samples == self.times_samples.clip(
            chunk_start_samples, chunk_end_samples - 1
        )
        if not in_chunk.any():
            return dict(n_spikes=0)

        return super().process_chunk(
            chunk_start_samples,
            n_resid_snips=n_resid_snips,
            chunk_end_samples=chunk_end_samples,
            return_residual=return_residual,
            skip_features=skip_features,
        )

    @classmethod
    def from_config(cls, sorting, recording, waveform_cfg, featurization_cfg):
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
            recording,
            channel_index,
            featurization_pipeline,
            sorting.times_samples,
            sorting.channels,
            labels=sorting.labels,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            chunk_length_samples=30_000,
            n_chunks_fit=100,
            dtype=torch.float,
        )

    def peel_chunk(
        self,
        traces,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
        return_waveforms=True,
    ):
        assert not return_residual

        max_t = chunk_start_samples + self.chunk_length_samples - 1
        in_chunk = self.times_samples == self.times_samples.clip(chunk_start_samples, max_t)
        (in_chunk,) = in_chunk.nonzero(as_tuple=True)

        if not in_chunk.numel():
            return dict(n_spikes=0)

        chunk_left = chunk_start_samples - left_margin
        times_rel = self.times_samples[in_chunk] - chunk_left
        channels = self.channels[in_chunk]

        assert times_rel.ndim == 1
        assert channels.ndim == 1

        waveforms = None
        if return_waveforms:
            waveforms = spiketorch.grab_spikes(
                traces,
                times_rel,
                channels,
                self.channel_index,
                trough_offset=self.trough_offset_samples,
                spike_length_samples=self.spike_length_samples,
                already_padded=False,
                pad_value=torch.nan,
            )

        res = dict(
            n_spikes=in_chunk.numel(),
            indices=in_chunk,
            times_samples=self.times_samples[in_chunk],
            channels=channels,
            collisioncleaned_waveforms=waveforms,
        )
        if self.labels is not None:
            res['labels'] = self.labels[in_chunk]
        return res
