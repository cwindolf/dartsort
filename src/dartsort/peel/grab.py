import torch
from dartsort.util import spiketorch

from .peel_base import BasePeeler


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
    ):
        super().__init__(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            chunk_length_samples=chunk_length_samples,
            chunk_margin_samples=spike_length_samples,
            n_chunks_fit=n_chunks_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
        )
        self.trough_offset_samples = trough_offset_samples
        self.spike_length_samples = spike_length_samples
        self.register_buffer("times_samples", times_samples)
        self.register_buffer("channels", channels)

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
        times_samples = self.times_samples[in_chunk] - chunk_start_samples
        channels = self.channels[in_chunk]

        waveforms = spiketorch.grab_spikes(
            traces,
            times_samples,
            channels,
            self.channel_index,
            trough_offset=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
            already_padded=False,
            pad_value=torch.nan,
        )

        return dict(
            n_spikes=in_chunk.numel(),
            times_samples=times_samples,
            channels=channels,
            collisioncleaned_waveforms=waveforms,
        )
