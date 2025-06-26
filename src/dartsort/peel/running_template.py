"""Fast template extractor

Algorithm: Each unit / pitch shift combo has a raw and low-rank template
computed. These are shift-averaged and then weighted combined.

The raw and low-rank templates are computed using Welford.
"""
import torch

from .grab import GrabAndFeaturize


class RunningTemplates(GrabAndFeaturize):

    def __init__(
        self,
        recording,
        channel_index,
        featurization_pipeline,
        times_samples,
        channels,
        labels,
        motion_est=None,
        trough_offset_samples=42,
        spike_length_samples=121,
        chunk_length_samples=30_000,
        n_seconds_fit=40,
        fit_subsampling_random_state=0,
    ):
        n_channels = recording.get_num_channels()
        full_channel_index = torch.tile(torch.arange(n_channels), (n_channels, 1))
        super().__init__(
            recording,
            featurization_pipeline,
            times_samples,
            channels,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            chunk_length_samples=chunk_length_samples,
            n_seconds_fit=n_seconds_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
        )
