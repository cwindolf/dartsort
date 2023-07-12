from collections import namedtuple

import torch
import torch.nn.functional as F
from dartsort.detect import detect_and_deduplicate
from dartsort.transform import WaveformPipeline
from dartsort.util import spiketorch
from dartsort.util.waveform_util import make_channel_index

from .base import BasePeeler


class SubtractionPeeler(BasePeeler):
    peeling_needs_fit = True

    def __init__(
        self,
        recording,
        channel_index,
        subtraction_denoising_pipeline,
        featurization_pipeline,
        trough_offset_samples=42,
        spike_length_samples=121,
        detection_thresholds=[12, 10, 8, 6, 5, 4],
        chunk_length_samples=30_000,
        peak_sign="neg",
        spatial_dedup_channel_index=None,
        n_seconds_fit=40,
        fit_subsampling_random_state=0,
        device=None,
    ):
        super().___init__(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            chunk_length_samples=chunk_length_samples,
            chunk_margin_samples=max(
                trough_offset_samples,
                spike_length_samples - trough_offset_samples,
            ),
            n_seconds_fit=n_seconds_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
            device=device,
        )

        self.trough_offset_samples = trough_offset_samples
        self.spike_length_samples = spike_length_samples
        self.peak_sign = peak_sign
        if spatial_dedup_channel_index is None:
            self.register_buffer(
                "spatial_dedup_channel_index",
                spatial_dedup_channel_index,
            )
        else:
            self.spatial_dedup_channel_index = None
        self.detection_thresholds = detection_thresholds

        self.add_module(
            "subtraction_denoising_pipeline", subtraction_denoising_pipeline
        )
        # we may be featurizing during subtraction, register the features
        for transformer in self.subtraction_denoising_pipeline.transformers:
            self.out_datasets.append(transformer.spike_dataset)

    @classmethod
    def from_config(
        cls, recording, subtraction_config, featurization_config, device=None
    ):
        # construct denoising and featurization pipelines
        subtraction_denoising_pipeline = WaveformPipeline.from_config(
            subtraction_config.subtraction_denoising_config
        )
        featurization_pipeline = WaveformPipeline.from_config(
            featurization_config
        )

        # waveform extraction channel neighborhoods
        channel_index = make_channel_index(
            recording.get_channel_locations(),
            subtraction_config.extract_radius,
        )
        # per-threshold spike event deduplication channel neighborhoods
        spatial_dedup_channel_index = make_channel_index(
            recording.get_channel_locations(),
            subtraction_config.spatial_dedup_radius,
        )

        return cls(
            recording,
            channel_index,
            subtraction_denoising_pipeline,
            featurization_pipeline,
            trough_offset_samples=subtraction_config.trough_offset_samples,
            spike_length_samples=subtraction_config.spike_length_samples,
            detection_thresholds=subtraction_config.detection_thresholds,
            chunk_length_samples=subtraction_config.chunk_length_samples,
            peak_sign=subtraction_config.peak_sign,
            spatial_dedup_channel_index=spatial_dedup_channel_index,
            n_seconds_fit=subtraction_config.n_seconds_fit,
            fit_subsampling_random_state=subtraction_config.fit_subsampling_random_state,
            device=device,
        )

    def peel_chunk(
        self,
        traces,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
    ):
        subtraction_result = subtract_chunk(
            traces,
            self.channel_index,
            self.subtraction_denoising_pipeline,
            trough_offset_samples=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
            left_margin=0,
            right_margin=0,
            detection_thresholds=[12, 10, 8, 6, 5, 4],
            peak_sign="neg",
            spatial_dedup_channel_index=None,
            in_place=True,
        )

        peel_result = dict(
            n_spikes=subtraction_result.n_spikes,
            times=subtraction_result.times,
            channels=subtraction_result.channels,
            collisioncleaned_waveforms=subtraction_result.collisioncleaned_waveforms,
        )
        peel_result.update(subtraction_result.features)
        if return_residual:
            peel_result["residual"] = subtraction_result.residual

        return peel_result


ChunkSubtractionResult = namedtuple(
    "ChunkSubtractionResult",
    [
        "n_spikes",
        "times",
        "channels",
        "collisioncleaned_waveforms",
        "residual",
        "features",
    ],
)


def subtract_chunk(
    traces,
    channel_index,
    denoising_pipeline,
    trough_offset_samples=42,
    spike_length_samples=121,
    left_margin=0,
    right_margin=0,
    detection_thresholds=[12, 10, 8, 6, 5, 4],
    peak_sign="neg",
    spatial_dedup_channel_index=None,
):
    """Core peeling routine for subtraction"""
    assert 0 <= left_margin < traces.shape[0]
    assert 0 <= right_margin < traces.shape[0]
    assert traces.shape[1] == channel_index.shape[0]
    if spatial_dedup_channel_index is not None:
        assert traces.shape[1] == spatial_dedup_channel_index.shape[0]
    assert detection_thresholds == sorted(detection_thresholds)
    # initialize residual, it needs to be padded to support
    # our channel indexing convention. this copies the input.
    residual = F.pad(traces, (0, 1), value=torch.nan)

    subtracted_waveforms = []
    spike_times = []
    spike_channels = []
    spike_features = []

    for threshold in detection_thresholds:
        # -- detect and extract waveforms
        # detection has more args which we don't expose right now
        times, channels = detect_and_deduplicate(
            residual,
            threshold,
            dedup_channel_index=spatial_dedup_channel_index,
            peak_sign=peak_sign,
        )
        if not times.numel():
            continue

        spike_times.append(times)
        spike_channels.append(channels)
        waveforms = spiketorch.grab_spikes(
            residual,
            times,
            channels,
            channel_index,
            trough_offset=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            buffer=0,
            already_padded=True,
        )

        # -- denoise
        waveforms, features = denoising_pipeline(waveforms)
        subtracted_waveforms.append(waveforms)
        spike_features.append(features)

        # -- subtract in place
        spiketorch.subtract_spikes_(
            residual,
            times,
            channels,
            channel_index,
            waveforms,
            trough_offset=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            buffer=0,
            already_padded=True,
            in_place=True,
        )

    subtracted_waveforms = torch.concatenate(subtracted_waveforms)
    spike_times = torch.concatenate(spike_times)
    spike_channels = torch.concatenate(spike_channels)
    spike_features = {
        k: torch.concatenate([f[k] for f in spike_features])
        for k in spike_features[0].keys()
    }

    # discard spikes in the margins
    keep = torch.nonzero(
        (spike_times >= left_margin)
        & (spike_times < traces.shape[0] - right_margin)
    )
    subtracted_waveforms = subtracted_waveforms[keep]
    spike_times = spike_times[keep]
    spike_channels = spike_channels[keep]
    spike_features = {k: v[keep] for k, v in spike_features.items()}

    # construct collision-cleaned waveforms
    collisioncleaned_waveforms = spiketorch.grab_spikes(
        residual,
        spike_times,
        spike_channels,
        channel_index,
        trough_offset=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        buffer=0,
        already_padded=True,
    )
    collisioncleaned_waveforms += subtracted_waveforms

    # offset spike times according to margin
    spike_times -= left_margin

    # strip margin and padding channel off the residual
    residual = residual[left_margin : traces.shape[0] - right_margin, :-1]

    return ChunkSubtractionResult(
        n_spikes=spike_times.numel(),
        times=spike_times,
        channels=spike_channels,
        collisioncleaned_waveforms=collisioncleaned_waveforms,
        residual=residual,
        features=spike_features,
    )
