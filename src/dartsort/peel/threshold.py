import warnings

import numpy as np
import torch


from ..detect import detect_and_deduplicate, convexity_filter
from ..transform import WaveformPipeline
from ..util import spiketorch
from ..util.data_util import SpikeDataset
from ..util.waveform_util import make_channel_index, get_channel_index_rel_inds

from .peel_base import BasePeeler, PeelingBatchResult, peeling_empty_result


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
        relative_peak_channel_index=None,
        spatial_dedup_channel_index=None,
        relative_peak_radius_samples=5,
        temporal_dedup_radius_samples=7,
        cumulant_order=None,
        convexity_threshold=None,
        convexity_radius=3,
        remove_exact_duplicates=True,
        n_seconds_fit=100,
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
            chunk_margin_samples=self.next_margin(spike_length_samples, factor=5),
            n_seconds_fit=n_seconds_fit,
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
        self.temporal_dedup_radius_samples = temporal_dedup_radius_samples
        self.remove_exact_duplicates = remove_exact_duplicates
        self.peak_sign = peak_sign
        self.trough_priority = trough_priority
        self.convexity_threshold = convexity_threshold
        self.convexity_radius = convexity_radius
        self.cumulant_order = cumulant_order
        self.dedup_batch_size = self.nearest_batch_length()
        if spatial_dedup_channel_index is not None:
            self.register_buffer(
                "spatial_dedup_channel_index", spatial_dedup_channel_index
            )
            self.register_buffer(
                "spatial_dedup_rel_inds",
                get_channel_index_rel_inds(spatial_dedup_channel_index),
            )
        else:
            self.spatial_dedup_channel_index = None
        if relative_peak_channel_index is not None:
            self.register_buffer(
                "relative_peak_channel_index", relative_peak_channel_index
            )
        else:
            self.relative_peak_channel_index = None
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
    def from_config(cls, recording, waveform_cfg, thresholding_cfg, featurization_cfg):
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

        relative_peak_channel_index = None
        if thresholding_cfg.relative_peak_radius_um:
            relative_peak_channel_index = make_channel_index(
                geom, thresholding_cfg.relative_peak_radius_um, to_torch=True
            )

        spatial_dedup_channel_index = None
        if thresholding_cfg.spatial_dedup_radius:
            spatial_dedup_channel_index = make_channel_index(
                geom, thresholding_cfg.spatial_dedup_radius, to_torch=True
            )

        spatial_jitter_channel_index = None
        if thresholding_cfg.spatial_jitter_radius:
            spatial_jitter_channel_index = make_channel_index(
                geom, thresholding_cfg.spatial_jitter_radius, to_torch=True
            )

        # waveform logic
        trough_offset_samples = waveform_cfg.trough_offset_samples(
            recording.sampling_frequency
        )
        spike_length_samples = waveform_cfg.spike_length_samples(
            recording.sampling_frequency
        )

        return cls(
            recording,
            channel_index,
            featurization_pipeline=featurization_pipeline,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            detection_threshold=thresholding_cfg.detection_threshold,
            chunk_length_samples=thresholding_cfg.chunk_length_samples,
            max_spikes_per_chunk=thresholding_cfg.max_spikes_per_chunk,
            peak_sign=thresholding_cfg.peak_sign,
            fit_sampling=thresholding_cfg.fit_sampling,
            fit_max_reweighting=thresholding_cfg.fit_max_reweighting,
            relative_peak_channel_index=relative_peak_channel_index,
            spatial_dedup_channel_index=spatial_dedup_channel_index,
            relative_peak_radius_samples=thresholding_cfg.relative_peak_radius_samples,
            temporal_dedup_radius_samples=thresholding_cfg.temporal_dedup_radius_samples,
            remove_exact_duplicates=thresholding_cfg.remove_exact_duplicates,
            cumulant_order=thresholding_cfg.cumulant_order,
            convexity_threshold=thresholding_cfg.convexity_threshold,
            convexity_radius=thresholding_cfg.convexity_radius,
            n_seconds_fit=thresholding_cfg.n_seconds_fit,
            n_waveforms_fit=thresholding_cfg.n_waveforms_fit,
            max_waveforms_fit=thresholding_cfg.max_waveforms_fit,
            fit_subsampling_random_state=thresholding_cfg.fit_subsampling_random_state,
            thinning=thresholding_cfg.thinning,
            time_jitter=thresholding_cfg.time_jitter,
            spatial_jitter_channel_index=spatial_jitter_channel_index,
            trough_priority=thresholding_cfg.trough_priority,
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
        *,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
        return_waveforms=True,
    ) -> PeelingBatchResult:
        threshold_res = threshold_chunk(
            traces,
            self.channel_index,
            detection_threshold=self.detection_threshold,
            peak_sign=self.peak_sign,
            spatial_dedup_channel_index=self.spatial_dedup_channel_index,
            spatial_dedup_rel_inds=self.spatial_dedup_rel_inds,
            dedup_batch_size=self.dedup_batch_size,
            trough_offset_samples=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
            left_margin=left_margin,
            right_margin=right_margin,
            relative_peak_channel_index=self.relative_peak_channel_index,
            temporal_dedup_radius_samples=self.temporal_dedup_radius_samples,
            remove_exact_duplicates=self.remove_exact_duplicates,
            cumulant_order=self.cumulant_order,
            convexity_threshold=self.convexity_threshold,
            convexity_radius=self.convexity_radius,
            thinning=self.thinning,
            time_jitter=self.time_jitter,
            spatial_jitter_channel_index=self.spatial_jitter_channel_index,
            rg=self.rg if self.is_random else None,
            max_spikes_per_chunk=None,
            return_waveforms=return_waveforms,
            trough_priority=self.trough_priority,
            quiet=False,
        )
        if not threshold_res["n_spikes"]:
            return peeling_empty_result

        # get absolute times
        times_rel = threshold_res["times_rel"]
        assert times_rel is not None
        times_samples = times_rel + chunk_start_samples

        peel_result = PeelingBatchResult(
            n_spikes=threshold_res["n_spikes"],
            times_samples=times_samples,
            channels=threshold_res["channels"],
            voltages=threshold_res["voltages"],
        )
        if return_waveforms:
            assert "waveforms" in threshold_res
            peel_result["collisioncleaned_waveforms"] = threshold_res["waveforms"]
        if self.is_random:
            orig_times_rel = threshold_res["orig_times_rel"]
            assert orig_times_rel is not None
            peel_result["orig_times_samples"] = orig_times_rel + chunk_start_samples
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
    detection_threshold=4.0,
    peak_sign="both",
    relative_peak_channel_index=None,
    spatial_dedup_channel_index=None,
    spatial_dedup_rel_inds=None,
    dedup_batch_size=512,
    trough_offset_samples=42,
    spike_length_samples=121,
    left_margin=0,
    right_margin=0,
    relative_peak_radius=5,
    temporal_dedup_radius_samples=7,
    remove_exact_duplicates=True,
    max_spikes_per_chunk=None,
    thinning=0.0,
    time_jitter=0,
    trough_priority=None,
    spatial_jitter_channel_index=None,
    cumulant_order=None,
    convexity_threshold=None,
    convexity_radius=3,
    return_waveforms=True,
    rg=None,
    quiet=False,
) -> PeelingBatchResult:
    n_index = channel_index.shape[1]
    times_rel, channels, energies = detect_and_deduplicate(  # type: ignore
        traces,
        detection_threshold,
        relative_peak_channel_index=relative_peak_channel_index,
        dedup_channel_index=spatial_dedup_channel_index,
        dedup_index_inds=spatial_dedup_rel_inds,
        spatial_dedup_batch_size=dedup_batch_size,
        peak_sign=peak_sign,
        dedup_temporal_radius=temporal_dedup_radius_samples,
        remove_exact_duplicates=remove_exact_duplicates,
        relative_peak_radius=relative_peak_radius,
        return_energies=True,
        trough_priority=trough_priority,
        cumulant_order=cumulant_order,
    )
    if not times_rel.numel():
        return PeelingBatchResult(
            n_spikes=0,
            orig_times_rel=times_rel,
            times_rel=times_rel,
            orig_channels=channels,
            channels=channels,
            voltages=energies,
            waveforms=energies.view(-1, spike_length_samples, n_index),
        )
    keep = convexity_filter(
        traces,
        times_rel,
        channels,
        threshold=convexity_threshold,
        radius=convexity_radius,
    )
    times_rel = times_rel[keep]
    channels = channels[keep]
    energies = energies[keep]

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
        return PeelingBatchResult(
            n_spikes=0,
            times_rel=times_rel,
            channels=channels,
            voltages=energies,
            orig_times_rel=orig_times_rel,
            orig_channels=orig_channels,
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
    else:
        waveforms = None

    # offset times for caller
    orig_times_rel -= left_margin
    times_rel -= left_margin

    res = PeelingBatchResult(
        n_spikes=times_rel.numel(),
        orig_times_rel=orig_times_rel,
        orig_channels=orig_channels,
        times_rel=times_rel,
        channels=channels,
        voltages=voltages,
    )
    if waveforms is not None:
        res["waveforms"] = waveforms
    return res


def perturb_detections(
    times_rel,
    channels,
    thinning: float=0,
    time_jitter=0,
    spatial_jitter_channel_index=None,
    rg: np.random.Generator | None = None,
):
    keep = slice(None)
    if not (thinning or time_jitter or spatial_jitter_channel_index is not None):
        return keep, times_rel, channels

    n = len(times_rel)
    if not n:
        return keep, times_rel, channels

    if thinning:
        assert 0 <= thinning <= 1
        assert rg is not None
        keep = rg.binomial(n=1, p=1.0 - thinning, size=n)
        keep = torch.from_numpy(np.flatnonzero(keep))
        keep = keep.to(times_rel)

        times_rel = times_rel[keep]
        channels = channels[keep]

    n = len(times_rel)
    if time_jitter:
        assert rg is not None
        jitter = rg.integers(low=-time_jitter, high=time_jitter + 1)
        times_rel = times_rel + torch.asarray(
            jitter, dtype=times_rel.dtype, device=times_rel.device
        )

    if spatial_jitter_channel_index is not None:
        assert rg is not None
        n_channels = len(spatial_jitter_channel_index)
        n_valid = (spatial_jitter_channel_index < n_channels).sum(1)
        n_valid = n_valid[channels].cpu()
        rel_ix = rg.integers(0, high=n_valid)
        rel_ix = torch.from_numpy(rel_ix).to(channels)
        channels = spatial_jitter_channel_index[channels, rel_ix]

    return keep, times_rel, channels
