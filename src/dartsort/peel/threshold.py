import warnings

import numpy as np
import torch


from ..detect import detect_and_deduplicate
from ..transform import WaveformPipeline
from ..util import spiketorch
from ..util.data_util import SpikeDataset
from ..util.waveform_util import make_channel_index

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
        fit_max_reweighting=4.0,
        spatial_dedup_channel_index=None,
        relative_peak_radius_samples=5,
        dedup_temporal_radius_samples=7,
        n_chunks_fit=40,
        n_waveforms_fit=20_000,
        max_waveforms_fit=50_000,
        fit_subsampling_random_state=0,
        thinning=0.0,
        time_jitter=0,
        trough_priority=None,
        spatial_jitter_channel_index=None,
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
        self.trough_priority = trough_priority
        if spatial_dedup_channel_index is not None:
            self.register_buffer(
                "spatial_dedup_channel_index", spatial_dedup_channel_index
            )
        else:
            self.spatial_dedup_channel_index = None
        self.detection_threshold = detection_threshold
        self.max_spikes_per_chunk = max_spikes_per_chunk
        self.peel_kind = f"Threshold {detection_threshold}"

        # parameters for random perturbation of detections
        if thinning:
            assert thinning < min(
                trough_offset_samples, spike_length_samples - trough_offset_samples
            )
        self.thinning = thinning
        self.time_jitter = time_jitter
        if spatial_jitter_channel_index is not None:
            self.register_buffer(
                "spatial_jitter_channel_index", spatial_jitter_channel_index
            )
        else:
            self.spatial_jitter_channel_index = None
        self.is_random = (
            thinning or time_jitter or spatial_jitter_channel_index is not None
        )

    @classmethod
    def from_config(
        cls, recording, waveform_config, thresholding_config, featurization_config
    ):
        geom = torch.tensor(recording.get_channel_locations())
        channel_index = make_channel_index(
            geom, featurization_config.extract_radius, to_torch=True
        )

        featurization_pipeline = WaveformPipeline.from_config(
            geom,
            channel_index,
            featurization_config,
            waveform_config,
            sampling_frequency=recording.sampling_frequency,
        )

        spatial_dedup_channel_index = None
        if thresholding_config.spatial_dedup_radius:
            spatial_dedup_channel_index = make_channel_index(
                geom, thresholding_config.spatial_dedup_radius, to_torch=True
            )

        spatial_jitter_channel_index = None
        if thresholding_config.spatial_jitter_radius:
            spatial_jitter_channel_index = make_channel_index(
                geom, thresholding_config.spatial_jitter_radius, to_torch=True
            )

        # waveform logic
        trough_offset_samples = waveform_config.trough_offset_samples(
            recording.sampling_frequency
        )
        spike_length_samples = waveform_config.spike_length_samples(
            recording.sampling_frequency
        )

        return cls(
            recording,
            channel_index,
            featurization_pipeline=featurization_pipeline,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            detection_threshold=thresholding_config.detection_threshold,
            chunk_length_samples=thresholding_config.chunk_length_samples,
            max_spikes_per_chunk=thresholding_config.max_spikes_per_chunk,
            peak_sign=thresholding_config.peak_sign,
            fit_sampling=thresholding_config.fit_sampling,
            fit_max_reweighting=thresholding_config.fit_max_reweighting,
            spatial_dedup_channel_index=spatial_dedup_channel_index,
            relative_peak_radius_samples=thresholding_config.relative_peak_radius_samples,
            dedup_temporal_radius_samples=thresholding_config.dedup_temporal_radius_samples,
            n_chunks_fit=thresholding_config.n_chunks_fit,
            n_waveforms_fit=thresholding_config.n_waveforms_fit,
            max_waveforms_fit=thresholding_config.max_waveforms_fit,
            fit_subsampling_random_state=thresholding_config.fit_subsampling_random_state,
            thinning=thresholding_config.thinning,
            time_jitter=thresholding_config.time_jitter,
            spatial_jitter_channel_index=spatial_jitter_channel_index,
            trough_priority=thresholding_config.trough_priority,
        )

    def out_datasets(self):
        datasets = super().out_datasets()
        if self.is_random:
            datasets.append(
                SpikeDataset(name="orig_times_samples", shape_per_spike=(), dtype=float)
            )
            datasets.append(
                SpikeDataset(name="orig_channels", shape_per_spike=(), dtype=float)
            )
        datasets.append(SpikeDataset(name="voltages", shape_per_spike=(), dtype=float))
        return datasets

    def peel_chunk(
        self,
        traces,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
        return_waveforms=True,
    ):
        threshold_res = threshold_chunk(
            traces,
            self.channel_index,
            detection_threshold=self.detection_threshold,
            peak_sign=self.peak_sign,
            spatial_dedup_channel_index=self.spatial_dedup_channel_index,
            trough_offset_samples=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
            left_margin=left_margin,
            right_margin=right_margin,
            dedup_temporal_radius=self.dedup_temporal_radius_samples,
            thinning=self.thinning,
            time_jitter=self.time_jitter,
            spatial_jitter_channel_index=self.spatial_jitter_channel_index,
            rg=self.rg if self.is_random else None,
            max_spikes_per_chunk=None,
            return_waveforms=return_waveforms,
            trough_priority=self.trough_priority,
            quiet=False,
        )

        # get absolute times
        times_samples = threshold_res["times_rel"] + chunk_start_samples

        peel_result = dict(
            n_spikes=threshold_res["n_spikes"],
            times_samples=times_samples,
            channels=threshold_res["channels"],
            voltages=threshold_res["voltages"],
            collisioncleaned_waveforms=threshold_res["waveforms"],
        )
        if self.is_random:
            peel_result["orig_times_samples"] = (
                threshold_res["orig_times_rel"] + chunk_start_samples
            )
            peel_result["orig_channels"] = threshold_res["orig_channels"]
        if return_residual:
            # note, this is same as the input.
            peel_result["residual"] = traces[
                left_margin : traces.shape[0] - right_margin
            ]
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
    thinning=0,
    time_jitter=0,
    trough_priority=None,
    spatial_jitter_channel_index=None,
    return_waveforms=True,
    rg=None,
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
        trough_priority=trough_priority,
    )
    if not times_rel.numel():
        return dict(
            n_spikes=0,
            orig_times_rel=times_rel,
            times_rel=times_rel,
            orig_channels=channels,
            channels=channels,
            voltages=energies,
            waveforms=energies.view(-1, spike_length_samples, n_index),
        )

    orig_times_rel = times_rel
    orig_channels = channels
    keep, times_rel, channels = perturb_detections(
        times_rel,
        channels,
        thinning=thinning,
        time_jitter=time_jitter,
        spatial_jitter_channel_index=spatial_jitter_channel_index,
        rg=rg,
    )
    orig_times_rel = orig_times_rel[keep]
    orig_channels = orig_channels[keep]
    energies = energies[keep]

    # want only peaks in the chunk
    min_time = left_margin + trough_offset_samples
    tail_samples = spike_length_samples - trough_offset_samples
    max_time = traces.shape[0] - right_margin - tail_samples - 1
    valid = times_rel == times_rel.clamp(min_time, max_time)
    (valid,) = valid.nonzero(as_tuple=True)
    orig_times_rel = orig_times_rel[valid]
    times_rel = times_rel[valid]
    channels = channels[valid]
    orig_channels = orig_channels[valid]
    voltages = traces[orig_times_rel, orig_channels]
    n_detect = times_rel.numel()
    if not n_detect:
        return dict(
            n_spikes=0,
            times_rel=times_rel,
            channels=channels,
            voltages=energies,
            orig_times_rel=orig_times_rel,
            orig_channels=orig_channels,
            waveforms=None,
        )

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
            orig_channels = orig_channels[best]
            orig_times_rel = orig_times_rel[best]

    # load up the waveforms for this chunk
    waveforms = None
    if return_waveforms:
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
    orig_times_rel -= left_margin
    times_rel -= left_margin

    return dict(
        n_spikes=times_rel.numel(),
        orig_times_rel=orig_times_rel,
        orig_channels=orig_channels,
        times_rel=times_rel,
        channels=channels,
        voltages=voltages,
        waveforms=waveforms,
    )


def perturb_detections(
    times_rel,
    channels,
    thinning=0,
    time_jitter=0,
    spatial_jitter_channel_index=None,
    rg=None,
):
    keep = slice(None)
    if not (thinning or time_jitter or spatial_jitter_channel_index is not None):
        return keep, times_rel, channels

    n = len(times_rel)
    if not n:
        return keep, times_rel, channels

    if thinning:
        assert 0 <= thinning <= 1
        keep = rg.binomial(n=1, p=1.0 - thinning, size=n)
        keep = torch.from_numpy(np.flatnonzero(keep))
        keep = keep.to(times_rel)

        times_rel = times_rel[keep]
        channels = channels[keep]

    n = len(times_rel)
    if time_jitter:
        jitter = rg.integers(low=-time_jitter, high=time_jitter + 1)
        times_rel = times_rel + torch.asarray(
            jitter, dtype=times_rel.dtype, device=times_rel.device
        )

    if spatial_jitter_channel_index is not None:
        n_channels = len(spatial_jitter_channel_index)
        n_valid = (spatial_jitter_channel_index < n_channels).sum(1)
        n_valid = n_valid[channels].cpu()
        rel_ix = rg.integers(0, high=n_valid)
        rel_ix = torch.from_numpy(rel_ix).to(channels)
        channels = spatial_jitter_channel_index[channels, rel_ix]

    return keep, times_rel, channels
