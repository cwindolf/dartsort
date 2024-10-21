import torch
from dartsort.util import spiketorch

from .peel_base import BasePeeler, SpikeDataset


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
            chunk_margin_samples=max(trough_offset_samples, spike_length_samples - trough_offset_samples),
            n_chunks_fit=n_chunks_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
            dtype=dtype,
        )
        self.trough_offset_samples = trough_offset_samples
        self.spike_length_samples = spike_length_samples
        self.register_buffer("times_samples", times_samples)
        self.register_buffer("channels", channels)

    def out_datasets(self):
        datasets = super().out_datasets()
        datasets.append(
            SpikeDataset(name="indices", shape_per_spike=(), dtype=int)
        )
        return datasets

    def process_chunk(self, chunk_start_samples, chunk_end_samples=None, return_residual=False):
        """Override process_chunk to skip empties."""
        if chunk_end_samples is None:
            chunk_end_samples = min(
                self.recording.get_num_samples(),
                chunk_start_samples + self.chunk_length_samples,
            )
        in_chunk = self.times_samples == self.times_samples.clip(chunk_start_samples, chunk_end_samples - 1)
        if not in_chunk.any():
            return dict(n_spikes=0)

        return super().process_chunk(chunk_start_samples, return_residual=return_residual)

    def peel_chunk(
        self,
        traces,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
    ):
        assert not return_residual

        in_chunk = torch.nonzero(
            (self.times_samples >= chunk_start_samples)
            & (
                self.times_samples
                < chunk_start_samples + self.chunk_length_samples
            )
        ).squeeze()

        if not in_chunk.numel():
            return dict(n_spikes=0)

        chunk_left = chunk_start_samples - left_margin
        times_rel = self.times_samples[in_chunk] - chunk_left
        channels = self.channels[in_chunk]

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

        return dict(
            n_spikes=in_chunk.numel(),
            indices=in_chunk,
            times_samples=self.times_samples[in_chunk],
            channels=channels,
            collisioncleaned_waveforms=waveforms,
        )
