"""A simple residual updating template matcher

Interesting code details to revisit:
 - Peak criterion is peak>(conv width) neighbors. What if
   we used smaller number of neighbors and deduplicated?
 - Why did the old code have this refractory thing inside
   high_res_peak? I think that was how they handled dedup?
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from dartsort.detect import detect_and_deduplicate
from dartsort.transform import WaveformPipeline
from dartsort.util import spiketorch
from dartsort.util.data_util import SpikeDataset
from dartsort.util.waveform_util import make_channel_index

from . import template_util
from .peel_base import BasePeeler


class ResidualUpdateTemplateMatchingPeeler(BasePeeler):
    peel_kind = "TemplateMatching"

    def __init__(
        self,
        recording,
        template_data,
        channel_index,
        featurization_pipeline,
        motion_est=None,
        svd_compression_rank=10,
        temporal_upsampling_factor=8,
        upsampling_peak_window_radius=8,
        min_channel_amplitude=1.0,
        refractory_radius_frames=10,
        amplitude_scaling_variance=0.0,
        amplitude_scaling_boundary=0.5,
        trough_offset_samples=42,
        threshold=30.0,
        chunk_length_samples=30_000,
        n_chunks_fit=40,
        fit_subsampling_random_state=0,
        max_iter=1000,
    ):
        n_templates, spike_length_samples = template_data.templates.shape[:2]
        super().__init__(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            chunk_length_samples=chunk_length_samples,
            chunk_margin_samples=2 * spike_length_samples,
            n_chunks_fit=n_chunks_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
        )

        # process templates
        (
            temporal_components,
            singular_values,
            spatial_components,
        ) = template_util.svd_compress_templates(
            template_data.templates,
            min_channel_amplitude=min_channel_amplitude,
            rank=svd_compression_rank,
        )
        self.handle_upsampling(
            temporal_components,
            temporal_upsampling_factor=temporal_upsampling_factor,
            upsampling_peak_window_radius=upsampling_peak_window_radius,
        )

        # main properties
        self.threshold = threshold
        self.refractory_radius_frames = refractory_radius_frames
        self.max_iter = max_iter
        self.n_templates = n_templates
        self.spike_length_samples = spike_length_samples
        self.geom = recording.get_channel_locations()
        self.svd_compression_rank = svd_compression_rank
        self.n_channels = len(self.geom)
        self.obj_pad_len = max(refractory_radius_frames, upsampling_peak_window_radius)
        self.n_registered_channels = (
            len(template_data.registered_geom)
            if template_data.registered_geom is not None
            else self.n_channels
        )

        # waveform extraction
        self.channel_index = channel_index
        self.registered_template_ampvecs = template_data.templates.ptp(1)

        # torch buffers
        self.register_buffer("temporal_components", torch.tensor(temporal_components))
        self.register_buffer("singular_values", torch.tensor(singular_values))
        self.register_buffer("spatial_components", torch.tensor(spatial_components))
        self.register_buffer(
            "_refrac_ix",
            torch.arange(-refractory_radius_frames, refractory_radius_frames + 1),
        )

        # amplitude scaling properties
        self.is_scaling = bool(amplitude_scaling_variance)
        self.amplitude_scaling_variance = amplitude_scaling_variance
        self.amp_scale_min = 1 / (1 + amplitude_scaling_boundary)
        self.amp_scale_max = 1 + amplitude_scaling_boundary

        # drift-related properties
        self.is_drifting = motion_est is not None
        self.registered_geom = template_data.registered_geom
        self.registered_template_depths_um = template_data.registered_template_depths_um

        self.handle_template_groups(template_data.unit_ids)
        self.check_shapes()

        self.fixed_output_data += [
            ("temporal_components", temporal_components),
            ("singular_values", singular_values),
            ("spatial_components", spatial_components),
            (
                "upsampled_temporal_components",
                self.upsampled_temporal_components.numpy(force=True).copy(),
            ),
        ]
        if self.is_drifting:
            self.fixed_output_data.append(
                ("registered_geom", template_data.registered_geom)
            )

    def out_datasets(self):
        datasets = super().out_datasets()
        return datasets + [
            SpikeDataset(name="template_indices", shape_per_spike=(), dtype=int),
            SpikeDataset(name="labels", shape_per_spike=(), dtype=int),
            SpikeDataset(name="upsampling_indices", shape_per_spike=(), dtype=int),
            SpikeDataset(name="scalings", shape_per_spike=(), dtype=float),
            SpikeDataset(name="scores", shape_per_spike=(), dtype=float),
        ]

    def check_shapes(self):
        # this is more like documentation than a necessary check
        assert self.temporal_components.shape == (
            self.n_templates,
            self.spike_length_samples,
            self.svd_compression_rank,
        )
        assert self.singular_values.shape == (
            self.n_templates,
            self.svd_compression_rank,
        )
        assert self.spatial_components.shape == (
            self.n_templates,
            self.svd_compression_rank,
            self.n_registered_channels,
        )
        assert self.upsampled_temporal_components.shape == (
            self.n_templates,
            self.spike_length_samples,
            self.temporal_upsampling_factor,
            self.svd_compression_rank,
        )
        assert self.unit_ids.shape == (self.n_templates,)

    def handle_template_groups(self, unit_ids):
        self.unit_ids = unit_ids
        self.grouped_temps = True
        unique_units = np.unique(unit_ids)
        if unique_units.size == unit_ids.size:
            self.grouped_temps = False

        if not self.grouped_temps:
            return

        assert unit_ids.shape == (self.n_templates,)
        group_index = [np.flatnonzero(unit_ids == u) for u in unit_ids]
        max_group_size = max(map(len, group_index))

        # like a channel index, sort of
        # this is a n_templates x group_size array that maps each
        # template index to the set of other template indices that
        # are part of its group. so that the array is not ragged,
        # we pad rows with -1s when their group is smaller than the
        # largest group.
        group_index = np.full((self.n_unit, max_group_size), -1)
        for j, row in enumerate(group_index):
            group_index[j, : len(row)] = row
        self.group_index = torch.tensor(group_index)

    def handle_upsampling(
        self,
        temporal_components,
        temporal_upsampling_factor=8,
        upsampling_peak_window_radius=8,
    ):
        self.temporal_upsampling_factor = temporal_upsampling_factor
        upsampled_temporal_components = temporal_components
        if temporal_upsampling_factor > 1:
            upsampled_temporal_components = template_util.temporally_upsampled_indices(
                temporal_components,
                temporal_upsampling_factor=temporal_upsampling_factor,
            )
            self.register_buffer(
                "upsampled_temporal_components",
                torch.tensor(upsampled_temporal_components),
            )
            self.register_buffer(
                "upsampling_window",
                torch.arange(
                    -upsampling_peak_window_radius, upsampling_peak_window_radius + 1
                ),
            )
            self.upsampling_window_len = 2 * upsampling_peak_window_radius
            center = upsampling_peak_window_radius * temporal_upsampling_factor
            radius = temporal_upsampling_factor // 2 + temporal_upsampling_factor % 2
            self.register_buffer(
                "upsampled_peak_search_window",
                torch.arange(center - radius, center + radius + 1),
            )
            self.register_buffer(
                "peak_to_upsampling_index",
                torch.concatenate(
                    [
                        torch.arange(radius, -1, -1),
                        (temporal_upsampling_factor - 1) - torch.arange(radius),
                    ]
                ),
            )
            self.register_buffer(
                "peak_to_time_shift", torch.tensor([0] * (radius + 1) + [1] * radius)
            )

    @classmethod
    def from_config(
        cls,
        recording,
        matching_config,
        featurization_config,
        template_data,
        motion_est=None,
    ):
        geom = torch.tensor(recording.get_channel_locations())
        channel_index = make_channel_index(
            geom, matching_config.extract_radius, to_torch=True
        )
        featurization_pipeline = WaveformPipeline.from_config(
            geom, channel_index, featurization_config
        )
        return cls(
            recording,
            template_data,
            channel_index,
            featurization_pipeline,
            motion_est=motion_est,
            svd_compression_rank=matching_config.template_svd_compression_rank,
            temporal_upsampling_factor=matching_config.template_temporal_upsampling_factor,
            min_channel_amplitude=matching_config.template_min_channel_amplitude,
            refractory_radius_frames=matching_config.refractory_radius_frames,
            amplitude_scaling_variance=matching_config.amplitude_scaling_variance,
            amplitude_scaling_boundary=matching_config.amplitude_scaling_boundary,
            trough_offset_samples=matching_config.trough_offset_samples,
            threshold=matching_config.threshold,
            chunk_length_samples=matching_config.chunk_length_samples,
            n_chunks_fit=matching_config.n_chunks_fit,
            fit_subsampling_random_state=matching_config.fit_subsampling_random_state,
            max_iter=matching_config.max_iter,
        )

    def peel_chunk(
        self,
        traces,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
    ):
        # get current template set
        chunk_center_samples = chunk_start_samples + self.chunk_length_samples // 2
        chunk_center_seconds = self.recording.sample_index_to_time(chunk_center_samples)
        compressed_template_data = self.templates_at_time(chunk_center_seconds)

        # deconvolve
        match_results = self.match_chunk(
            traces,
            compressed_template_data,
            trough_offset_samples=42,
            left_margin=0,
            right_margin=0,
            threshold=30,
        )

        # process spike times and create return result
        match_results["times_samples"] += chunk_start_samples - left_margin

        return match_results

    def templates_at_time(self, t_s):
        """Extract the right spatial components for each unit."""
        if self.is_drifting:
            cur_spatial = template_util.templates_at_time(
                t_s,
                self.spatial_components,
                self.geom,
                registered_template_depths_um=self.registered_template_depths_um,
                registered_geom=self.registered_geom,
                motion_est=self.motion_est,
            )
            cur_ampvecs = template_util.templates_at_time(
                t_s,
                self.registered_template_ampvecs[:, None, :],
                self.geom,
                registered_template_depths_um=self.registered_template_depths_um,
                registered_geom=self.registered_geom,
                motion_est=self.motion_est,
            )
            max_channels = cur_ampvecs[:, 0, :].argmax(1)
        else:
            cur_spatial = self.spatial_components
            max_channels = self.registered_template_ampvecs.argmax(1)

        return CompressedTemplateData(
            cur_spatial,
            self.singular_values,
            self.temporal_components,
            self.upsampled_temporal_components,
            max_channels,
        )

    def match_chunk(
        self,
        traces,
        compressed_template_data,
        trough_offset_samples=42,
        left_margin=0,
        right_margin=0,
        threshold=30,
    ):
        """Core peeling routine for subtraction"""
        # initialize residual, it needs to be padded to support our channel
        # indexing convention (used later to extract small channel
        # neighborhoods). this copies the input.
        residual_padded = F.pad(traces, (0, 1), value=torch.nan)
        residual = residual_padded[:, :-1]

        # name objective variables so that we can update them in-place later
        conv = None
        conv_len = traces.shape[0] - self.spike_length_samples + 1
        padded_obj_len = conv_len + 2 * self.obj_pad_len
        padded_objective = torch.zeros(
            self.n_templates + 1,
            padded_obj_len,
            dtype=traces.dtype,
            device=traces.device,
        )
        # padded objective has an extra unit (for group_index) and refractory
        # padding (for easier implementation of enforce_refractory)
        objective = padded_objective[
            :-1, self.refractory_radius_frames : -self.refractory_radius_frames
        ]
        neg_temp_normsq = -compressed_template_data.template_norms_squared[:, None]

        # manages buffers for spike train data (peak times, labels, etc)
        peaks = MatchingPeaks()

        # main loop
        for _ in range(self.max_iter):
            # update objective
            conv = compressed_template_data.convolve(residual, out=conv)
            # unscaled objective for coarse peaks, scaled when finding high res peak
            torch.add(neg_temp_normsq, conv, alpha=2.0, out=objective)

            # find high-res peaks
            new_peaks = self.find_peaks(conv, padded_objective, peaks, neg_temp_normsq)
            if new_peaks is None:
                break

            # subtract them
            # offset times: conv result peaks with valid padding are offset
            # by spike len - 1 samples from the corresponding trace peaks
            sample_times = new_peaks.times + self.spike_length_samples - 1
            compressed_template_data.subtract(
                residual,
                sample_times,
                new_peaks.template_indices,
                new_peaks.upsampling_indices,
                new_peaks.scalings,
            )

            # update spike train
            peaks.extend(new_peaks)
        peaks.sort()

        # extract collision-cleaned waveforms on small neighborhoods
        channels, waveforms = self.get_collisioncleaned_waveforms()

        return dict(
            times_samples=peaks.times + self.trough_offset_samples,
            channels=channels,
            labels=self.unit_ids[peaks.template_indices],
            template_indices=peaks.template_indices,
            upsampling_indices=peaks.upsampling_indices,
            scalings=peaks.scalings,
            scores=peaks.scores,
            collisioncleaned_waveforms=waveforms,
        )

    def find_peaks(self, conv, padded_objective, peaks, neg_temp_normsq):
        # zeroth step: enforce refractoriness.
        self.enforce_refractory(
            padded_objective,
            peaks.times + self.obj_pad_len,
            peaks.template_indices,
        )

        # first step: coarse peaks. not temporally upsampled or amplitude-scaled.
        objective = padded_objective[:-1, self.obj_pad_len : -self.obj_pad_len]
        times, template_indices = detect_and_deduplicate(
            objective.T,
            self.threshold,
            dedup_channel_index=None,
            peak_sign="pos",
            # add 1 here to account for possible time_shifts later
            relative_peak_radius=self.spike_length_samples + 1,
            dedup_temporal_radius=0,
            # dedup_temporal_radius=self.spike_length_samples + 1,
        )
        if not times.numel():
            return None

        # second step: high-res peaks (upsampled and/or amp-scaled)
        time_shifts, upsampling_indices, scalings, scores = self.find_fancy_peaks(
            conv, objective, times, template_indices, neg_temp_normsq
        )
        if time_shifts is not None:
            times += time_shifts

        return MatchingPeaks(
            n_spikes=times.size,
            times=times,
            template_indices=template_indices,
            upsampling_indices=upsampling_indices,
            scalings=scalings,
            scores=scores,
        )

    def enforce_refractory(self, objective, times, template_indices):
        # overwrite objective with -inf to enforce refractoriness
        time_ix = times[None, :] + self._refrac_ix[:, None]
        if self.grouped_temps:
            unit_ix = self.group_index[template_indices]
        else:
            unit_ix = template_indices[:, None]
        objective[unit_ix, time_ix] = -torch.inf

    def find_fancy_peaks(
        self, conv, objective, times, template_indices, neg_temp_normsq
    ):
        """Given coarse peaks, find temporally upsampled and scaled ones."""
        # tricky bit. we search for upsampled peaks to the left and right
        # of the original peak. when the up-peak comes to the right, we
        # use one of the upsampled templates, no problem. when the peak
        # comes to the left, it's different: it came from one of the upsampled
        # templates shifted one sample (spike time += 1).
        if self.up_factor == 1 and not self.is_scaling:
            return None, None, None, objective[template_indices, times]

        if self.is_scaling and self.up_factor == 1:
            inv_lambda = 1 / self.amplitude_scaling_variance
            b = conv[times, template_indices] + inv_lambda
            a = neg_temp_normsq[template_indices] + inv_lambda
            scalings = torch.clip(b / a, self.amp_scale_min, self.amp_scale_max)
            scores = 2.0 * scalings * b - torch.square(scalings) * a - inv_lambda
            return None, None, scalings, scores

        # below, we are upsampling.
        # get clips of objective function around the peaks
        # we'll use the scaled objective here.
        time_ix = times[:, None] + self.upsampling_window[None, :]
        clip_ix = (template_indices[:, None], time_ix)
        upsampled_clip_len = (
            self.upsampling_window_len * self.temporal_upsampling_factor
        )
        if self.is_scaling:
            high_res_conv = spiketorch.real_resample(
                conv[clip_ix], upsampled_clip_len, dim=1
            )
            inv_lambda = 1.0 / self.amplitude_scaling_variance
            b = high_res_conv + inv_lambda
            a = neg_temp_normsq[template_indices] + inv_lambda
            scalings = torch.clip(b / a, self.amp_scale_min, self.amp_scale_max)
            high_res_obj = (
                2.0 * scalings * b - torch.square(scalings) * a[:, None] - inv_lambda
            )
        else:
            scalings = None
            obj_clips = objective[clip_ix]
            high_res_obj = spiketorch.real_resample(
                obj_clips, upsampled_clip_len, dim=1
            )

        # zoom into a small upsampled area and determine the
        # upsampled template and time shifts
        scores, zoom_peak = torch.max(
            high_res_obj[:, self.upsampled_peak_search_window], dim=1
        )
        upsampling_indices = self.peak_to_upsampling_index[zoom_peak]
        time_shifts = self.peak_to_time_shifts[zoom_peak]

        return time_shifts, upsampling_indices, scalings, scores

    def get_collisioncleaned_waveforms(
        self, residual_padded, peaks, compressed_template_data
    ):
        channels = compressed_template_data.max_channels[peaks.template_indices]
        waveforms = spiketorch.grab_spikes(
            residual_padded,
            peaks.times,
            channels,
            self.channel_index,
            trough_offset=0,
            spike_length_samples=self.spike_length_samples,
            buffer=0,
            already_padded=True,
        )
        spatial = compressed_template_data.spatial_singular[
            peaks.template_indices[:, None, None],
            :,
            self.channel_index[channels][:, None, :],
        ]
        temporal = compressed_template_data.upsampled_temporal_components[
            peaks.template_indices,
            peaks.upsampling_indices,
        ]
        torch.baddbmm(waveforms, temporal, spatial, out=waveforms)
        return channels, waveforms


@dataclass
class CompressedTemplateData:
    """Objects of this class are returned by ResidualUpdateTemplateMatchingPeeler.templates_at_time()"""

    spatial_components: torch.Tensor
    singular_values: torch.Tensor
    temporal_components: torch.Tensor
    upsampled_temporal_components: torch.Tensor
    max_channels: torch.LongTensor

    def __post_init__(self):
        (
            self.n_templates,
            self.spike_length_samples,
            self.rank,
        ) = self.temporal_components.shape
        # squared l2 norms are the sums of squared singular values
        self.template_norms_squared = torch.square(self.singular_values).sum(1)
        self.spatial_singular = (
            self.spatial_components * self.singular_values[:, :, None]
        )
        self.chan_ix = torch.arange(
            self.spatial_components.shape[2], device=self.spatial_components.device
        )
        self.time_ix = torch.arange(
            self.spike_length_samples, device=self.spatial_components.device
        )

    def convolve(self, traces, out=None):
        """This is not the fastest strategy on GPU, but it's low-memory and fast on CPU."""
        if out is None:
            out = torch.zeros(
                1,
                self.n_templates,
                traces.shape[0] - self.spike_length_samples + 1,
                dtype=traces.dtype,
                device=traces.device,
            )
        else:
            assert out.shape == (
                self.n_templates,
                traces.shape[0] - self.spike_length_samples + 1,
            )
            out = out[None]

        for q in range(self.rank):
            # units x time
            rec_spatial = self.spatial_singular[:, q, :] @ traces.T
            # convolve with temporal components -- units x time
            temporal = self.temporal_components[:, :, q]
            # conv1d with groups! only convolve each unit with its own temporal filter.
            out += F.conv1d(
                rec_spatial[None], temporal[:, None, :], groups=self.n_templates
            )

        # back to units x time (remove extra dim used for conv1d)
        return out[0]

    def subtract(
        self,
        traces,
        times,
        template_indices,
        upsampling_indices,
        scalings,
    ):
        batch_templates = torch.einsum(
            "n,nrc,ntr",
            scalings,
            self.spatial_singular[template_indices],
            self.upsampled_temporal_components[template_indices, upsampling_indices],
        )
        time_ix = times[:, None, None] + self.time_ix[None, :, None]
        spiketorch.add_at_(
            traces, (time_ix, self.chan_ix[None, None, :]), batch_templates, sign=-1
        )


class MatchingPeaks:
    BUFFER_INIT: int = 1000
    BUFFER_GROWTH: float = 1.5

    def __init__(
        self,
        n_spikes: int = 0,
        times: Optional[torch.LongTensor] = None,
        template_indices: Optional[torch.LongTensor] = None,
        upsampling_indices: Optional[torch.LongTensor] = None,
        scalings: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
    ):
        self.n_spikes = n_spikes
        if times is None:
            cur_buf_size = self.BUFFER_INIT
            self._times = torch.zeros(cur_buf_size, dtype=int)
        else:
            cur_buf_size = times.size
            assert cur_buf_size == n_spikes
        if template_indices is None:
            self._template_indices = torch.zeros(cur_buf_size, dtype=int)
        if scalings is None:
            self._scalings = torch.zeros(cur_buf_size)
        if upsampling_indices is None:
            self._upsampling_indices = torch.zeros(cur_buf_size, dtype=int)
        if scores is None:
            self._scores = torch.zeros(cur_buf_size)

    @property
    def times(self):
        return self._times[: self.n_spikes]

    @property
    def template_indices(self):
        return self._template_indices[: self.n_spikes]

    @property
    def upsampling_indices(self):
        return self._upsampling_indices[: self.n_spikes]

    @property
    def scalings(self):
        return self._scalings[: self.n_spikes]

    @property
    def scores(self):
        return self._scores[: self.n_spikes]

    def grow_buffers(self, min_size=0):
        new_buf_size = max(min_size, int(self.cur_buf_size * self.BUFFER_GROWTH))
        new_times = torch.zeros(new_buf_size, dtype=self._times.dtype)
        new_template_indices = torch.zeros(
            new_buf_size, dtype=self._template_indices.dtype
        )
        new_upsampling_indices = torch.zeros(
            new_buf_size, dtype=self._upsampling_indices.dtype
        )
        new_scalings = torch.zeros(new_buf_size, dtype=self._scalings.dtype)
        new_scores = torch.zeros(new_buf_size, dtype=self._scores.dtype)

        new_times[: self.n_spikes] = self.times
        new_template_indices[: self.n_spikes] = self.template_indices
        new_upsampling_indices[: self.n_spikes] = self.upsampling_indices
        new_scalings[: self.n_spikes] = self.scalings
        new_scores[: self.n_spikes] = self.scores

        self.cur_buf_size = new_buf_size
        self._times = new_times
        self._template_indices = new_template_indices
        self._upsampling_indices = new_upsampling_indices
        self._scalings = new_scalings
        self._scores = new_scores

    def sort(self):
        order = torch.argsort(self.times[: self.n_spikes])
        self._times[: self.n_spikes] = self.times[order]
        self._template_indices[: self.n_spikes] = self.template_indices[order]
        self._upsampling_indices[: self.n_spikes] = self.upsampling_indices[order]
        self._scalings[: self.n_spikes] = self.scalings[order]
        self._scores[: self.n_spikes] = self.scores[order]

    def extend(self, other):
        new_n_spikes = other.n_spikes + self.n_spikes
        if new_n_spikes > self.cur_buf_size:
            self.grow_buffers(min_size=new_n_spikes)
        self._times[self.n_spikes : new_n_spikes] = other.times
        self._template_indices[self.n_spikes : new_n_spikes] = other.template_indices
        self._upsampling_indices[
            self.n_spikes : new_n_spikes
        ] = other.upsampling_indices
        self._scalings[self.n_spikes : new_n_spikes] = other.scalings
        self._scores[self.n_spikes : new_n_spikes] = other.scores
        self.n_spikes = new_n_spikes
