import torch
from dartsort.detect import detect_and_deduplicate
from dartsort.util import spiketorch

from .peel_base import BasePeeler


class ThresholdAndFeaturize(BasePeeler):
    def __init__(
        self,
        recording,
        channel_index,
        featurization_pipeline=None,
        trough_offset_samples=42,
        spike_length_samples=121,
        detection_threshold=5.0,
        chunk_length_samples=30_000,
        peak_sign="both",
        spatial_dedup_channel_index=None,
        n_chunks_fit=40,
        max_waveforms_fit=50_000,
        fit_subsampling_random_state=0,
    ):
        super().__init__(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            chunk_length_samples=chunk_length_samples,
            chunk_margin_samples=spike_length_samples,
            n_chunks_fit=n_chunks_fit,
            max_waveforms_fit=max_waveforms_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
        )

        self.trough_offset_samples = trough_offset_samples
        self.spike_length_samples = spike_length_samples
        self.peak_sign = peak_sign
        if spatial_dedup_channel_index is not None:
            self.register_buffer(
                "spatial_dedup_channel_index",
                spatial_dedup_channel_index,
            )
        else:
            self.spatial_dedup_channel_index = None
        self.detection_threshold = detection_threshold
        self.peel_kind = f"Threshold {detection_threshold}"

    def peel_chunk(
        self,
        traces,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
    ):
        times_rel, channels = detect_and_deduplicate(
            traces,
            self.detection_threshold,
            dedup_channel_index=self.spatial_dedup_channel_index,
            peak_sign=self.peak_sign,
        )
        if not times_rel.numel():
            return dict(n_spikes=0)

        # want only peaks in the chunk
        min_time = max(left_margin, self.spike_length_samples)
        max_time = traces.shape[0] - max(
            right_margin, self.spike_length_samples - self.trough_offset_samples
        )
        valid = (times_rel >= min_time) & (times_rel < max_time)
        times_rel = times_rel[valid]
        if not times_rel.numel():
            return dict(n_spikes=0)
        channels = channels[valid]

        # load up the waveforms for this chunk
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

        # get absolute times
        times_samples = times_rel + chunk_start_samples - left_margin

        peel_result = dict(
            n_spikes=times_rel.numel(),
            times_samples=times_samples,
            channels=channels,
            collisioncleaned_waveforms=waveforms,
        )
        return peel_result
