import warnings

import torch
from dartsort.detect import detect_and_deduplicate
from dartsort.util import spiketorch
from dartsort.util.data_util import SpikeDataset

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
        max_spikes_per_chunk=None,
        peak_sign="both",
        fit_sampling="random",
        fit_max_reweighting=20.0,
        spatial_dedup_channel_index=None,
        relative_peak_radius_samples=5,
        dedup_temporal_radius_samples=7,
        n_chunks_fit=40,
        n_waveforms_fit=20_000,
        max_waveforms_fit=50_000,
        fit_subsampling_random_state=0,
        dtype=torch.float,
    ):
        super().__init__(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            chunk_length_samples=chunk_length_samples,
            chunk_margin_samples=spike_length_samples,
            n_chunks_fit=n_chunks_fit,
            n_waveforms_fit=n_waveforms_fit,
            max_waveforms_fit=max_waveforms_fit,
            fit_sampling=fit_sampling,
            fit_subsampling_random_state=fit_subsampling_random_state,
            fit_max_reweighting=fit_max_reweighting,
            dtype=dtype,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
        )

        self.relative_peak_radius_samples = relative_peak_radius_samples
        self.dedup_temporal_radius_samples = dedup_temporal_radius_samples
        self.peak_sign = peak_sign
        if spatial_dedup_channel_index is not None:
            self.register_buffer(
                "spatial_dedup_channel_index",
                spatial_dedup_channel_index,
            )
        else:
            self.spatial_dedup_channel_index = None
        self.detection_threshold = detection_threshold
        self.max_spikes_per_chunk = max_spikes_per_chunk
        self.peel_kind = f"Threshold {detection_threshold}"

    def out_datasets(self):
        datasets = super().out_datasets()
        datasets.append(
            SpikeDataset(name="voltages", shape_per_spike=(), dtype=float)
        )
        return datasets

    def peel_chunk(
        self,
        traces,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
    ):
        threshold_res = threshold_chunk(
            traces,
            self.channel_index,
            detection_threshold=self.detection_threshold,
            peak_sign="both",
            spatial_dedup_channel_index=None,
            trough_offset_samples=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
            left_margin=left_margin,
            right_margin=right_margin,
            relative_peak_radius=5,
            dedup_temporal_radius=7,
            max_spikes_per_chunk=None,
            quiet=False,
        )

        # get absolute times
        times_samples = threshold_res['times_rel'] + chunk_start_samples

        peel_result = dict(
            n_spikes=threshold_res['n_spikes'],
            times_samples=times_samples,
            channels=threshold_res['channels'],
            voltages=threshold_res['voltages'],
            collisioncleaned_waveforms=threshold_res['waveforms'],
        )
        return peel_result


def threshold_chunk(
    traces,
    channel_index,
    detection_threshold=4,
    peak_sign="both",
    spatial_dedup_channel_index=None,
    trough_offset_samples=42,
    spike_length_samples=121,
    left_margin=0,
    right_margin=0,
    relative_peak_radius=5,
    dedup_temporal_radius=7,
    max_spikes_per_chunk=None,
    quiet=False,
):
    n_index = channel_index.shape[1]
    times_rel, channels, energies = detect_and_deduplicate(
        traces,
        detection_threshold,
        dedup_channel_index=spatial_dedup_channel_index,
        peak_sign=peak_sign,
        dedup_temporal_radius=dedup_temporal_radius,
        relative_peak_radius=relative_peak_radius,
        return_energies=True,
    )
    if not times_rel.numel():
        return dict(
            n_spikes=0,
            times_rel=times_rel,
            channels=channels,
            voltages=energies,
            waveforms=energies.view(-1, spike_length_samples, n_index),
        )

    # want only peaks in the chunk
    min_time = max(left_margin, spike_length_samples)
    max_time = traces.shape[0] - max(
        right_margin, spike_length_samples - trough_offset_samples
    )
    valid = (times_rel >= min_time) & (times_rel < max_time)
    times_rel = times_rel[valid]
    n_detect = times_rel.numel()
    if not n_detect:
        return dict(
            n_spikes=0,
            times_rel=times_rel,
            channels=channels,
            voltages=energies,
            waveforms=energies.view(-1, spike_length_samples, n_index),
        )
    channels = channels[valid]
    voltages = traces[times_rel, channels]

    if max_spikes_per_chunk is not None:
        if n_detect > max_spikes_per_chunk and not quiet:
            warnings.warn(
                f"{n_detect} spikes in chunk was larger than "
                f"{max_spikes_per_chunk=}. Keeping the top ones."
            )
            energies = energies[valid]
            best = torch.argsort(energies)[-max_spikes_per_chunk:]
            best = best.sort().values
            del energies

            times_rel = times_rel[best]
            channels = channels[best]
            voltages = voltages[best]

    # load up the waveforms for this chunk
    waveforms = spiketorch.grab_spikes(
        traces,
        times_rel,
        channels,
        channel_index,
        trough_offset=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        already_padded=False,
        pad_value=torch.nan,
    )

    # offset times for caller
    times_rel -= left_margin

    return dict(
        n_spikes=times_rel.numel(),
        times_rel=times_rel,
        channels=channels,
        voltages=voltages,
        waveforms=waveforms,
    )
