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
from dartsort.templates import template_util
from dartsort.templates.pairwise import CompressedPairwiseConv
from dartsort.transform import WaveformPipeline
from dartsort.util import drift_util, spiketorch
from dartsort.util.data_util import SpikeDataset
from dartsort.util.waveform_util import make_channel_index
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

from .peel_base import BasePeeler


class ObjectiveUpdateTemplateMatchingPeeler(BasePeeler):
    peel_kind = "TemplateMatching"

    def __init__(
        self,
        recording,
        template_data,
        channel_index,
        featurization_pipeline,
        motion_est=None,
        svd_compression_rank=10,
        coarse_objective=True,
        temporal_upsampling_factor=8,
        upsampling_peak_window_radius=8,
        min_channel_amplitude=1.0,
        refractory_radius_frames=10,
        amplitude_scaling_variance=0.0,
        amplitude_scaling_boundary=0.5,
        conv_ignore_threshold=5.0,
        coarse_approx_error_threshold=5.0,
        trough_offset_samples=42,
        threshold=50.0,
        chunk_length_samples=30_000,
        n_chunks_fit=40,
        max_waveforms_fit=50_000,
        n_waveforms_fit=20_000,
        fit_subsampling_random_state=0,
        fit_sampling="random",
        max_iter=1000,
        dtype=torch.float,
    ):
        n_templates, spike_length_samples = template_data.templates.shape[:2]
        super().__init__(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            chunk_length_samples=chunk_length_samples,
            chunk_margin_samples=2 * template_data.templates.shape[1],
            n_chunks_fit=n_chunks_fit,
            max_waveforms_fit=max_waveforms_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
            fit_sampling=fit_sampling,
            n_waveforms_fit=n_waveforms_fit,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            dtype=dtype,
        )

        # main properties
        self.template_data = template_data
        self.coarse_objective = coarse_objective
        self.temporal_upsampling_factor = temporal_upsampling_factor
        self.upsampling_peak_window_radius = upsampling_peak_window_radius
        self.svd_compression_rank = svd_compression_rank
        self.min_channel_amplitude = min_channel_amplitude
        self.threshold = threshold
        self.conv_ignore_threshold = conv_ignore_threshold
        self.coarse_approx_error_threshold = coarse_approx_error_threshold
        self.refractory_radius_frames = refractory_radius_frames
        self.max_iter = max_iter
        self.n_templates = n_templates
        self.geom = recording.get_channel_locations()
        self.n_channels = len(self.geom)
        self.obj_pad_len = max(
            refractory_radius_frames,
            upsampling_peak_window_radius,
            self.spike_length_samples - 1,
        )
        self.n_registered_channels = (
            len(template_data.registered_geom)
            if template_data.registered_geom is not None
            else self.n_channels
        )

        # waveform extraction
        self.channel_index = channel_index
        self.registered_template_ampvecs = np.ptp(template_data.templates, 1)

        # amplitude scaling properties
        self.is_scaling = bool(amplitude_scaling_variance)
        self.amplitude_scaling_variance = amplitude_scaling_variance
        self.amp_scale_min = 1 / (1 + amplitude_scaling_boundary)
        self.amp_scale_max = 1 + amplitude_scaling_boundary

        # drift-related properties
        self.is_drifting = motion_est is not None
        self.motion_est = motion_est
        self.registered_geom = template_data.registered_geom
        self.registered_template_depths_um = template_data.registered_template_depths_um
        if self.is_drifting:
            self.fixed_output_data.append(
                ("registered_geom", template_data.registered_geom)
            )
            self.registered_geom_kdtree = KDTree(self.registered_geom)
            self.geom_kdtree = KDTree(self.geom)
            self.match_distance = pdist(self.geom).min() / 2.0

        # some parts of this constructor are deferred to precompute_peeling_data
        self._needs_precompute = True

    def peeling_needs_fit(self):
        return self._needs_precompute

    def precompute_peeling_data(
        self, save_folder, overwrite=False, n_jobs=0, device=None
    ):
        self.build_template_data(
            save_folder,
            self.template_data,
            temporal_upsampling_factor=self.temporal_upsampling_factor,
            upsampling_peak_window_radius=self.upsampling_peak_window_radius,
            svd_compression_rank=self.svd_compression_rank,
            min_channel_amplitude=self.min_channel_amplitude,
            overwrite=overwrite,
            n_jobs=n_jobs,
            device=device,
        )
        # couple more torch buffers
        self.register_buffer(
            "_refrac_ix",
            torch.arange(
                -self.refractory_radius_frames, self.refractory_radius_frames + 1
            ),
        )
        self.register_buffer("_rank_ix", torch.arange(self.svd_compression_rank))
        self.check_shapes()
        self._needs_precompute = False

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
        assert self.unit_ids.shape == (self.n_templates,)

    def handle_template_groups(self, obj_unit_ids, unit_ids):
        """Grouped templates in objective

        If not coarse_objective, then several rows of the objective may
        belong to the same unit. They must be handled together when imposing
        refractory conditions.
        """
        self.register_buffer("unit_ids", torch.from_numpy(unit_ids))
        self.register_buffer("obj_unit_ids", torch.from_numpy(obj_unit_ids))
        units, fine_to_coarse, counts = np.unique(
            unit_ids, return_counts=True, return_inverse=True
        )
        self.register_buffer("fine_to_coarse", torch.from_numpy(fine_to_coarse))
        self.grouped_temps = True
        unique_units = np.unique(unit_ids)
        if unique_units.size == unit_ids.size:
            self.grouped_temps = False

        if not self.grouped_temps:
            self.register_buffer("superres_index", torch.arange(len(unit_ids))[:, None])
            return
        assert unit_ids.shape == (self.n_templates,)

        superres_index = np.full((len(obj_unit_ids), counts.max()), self.n_templates)
        for j, u in enumerate(obj_unit_ids):
            my_sup = np.flatnonzero(unit_ids == u)
            superres_index[j, : len(my_sup)] = my_sup
        self.register_buffer("superres_index", torch.from_numpy(superres_index))

        if self.coarse_objective:
            return

        # like a channel index, sort of
        # this is a n_templates x group_size array that maps each
        # template index to the set of other template indices that
        # are part of its group. so that the array is not ragged,
        # we pad rows with -1s when their group is smaller than the
        # largest group.
        group_index = np.full((self.n_templates, counts.max()), -1)
        for j, u in enumerate(unit_ids):
            row = np.flatnonzero(unit_ids == u)
            group_index[j, : len(row)] = row
        self.register_buffer("group_index", torch.from_numpy(group_index))

    def build_template_data(
        self,
        save_folder,
        template_data,
        temporal_upsampling_factor=8,
        upsampling_peak_window_radius=8,
        svd_compression_rank=10,
        min_channel_amplitude=1.0,
        overwrite=False,
        n_jobs=0,
        device=None,
    ):
        dtype = template_data.templates.dtype
        low_rank_templates = template_util.svd_compress_templates(
            template_data,
            min_channel_amplitude=min_channel_amplitude,
            rank=svd_compression_rank,
        )
        temporal_components = low_rank_templates.temporal_components.astype(dtype)
        singular_values = low_rank_templates.singular_values.astype(dtype)
        spatial_components = low_rank_templates.spatial_components.astype(dtype)
        self.register_buffer("temporal_components", torch.tensor(temporal_components))
        self.register_buffer("singular_values", torch.tensor(singular_values))
        self.register_buffer("spatial_components", torch.tensor(spatial_components))
        compressed_upsampled_temporal = self.handle_upsampling(
            temporal_components,
            ptps=np.ptp(template_data.templates, 1).max(1),
            temporal_upsampling_factor=temporal_upsampling_factor,
            upsampling_peak_window_radius=upsampling_peak_window_radius,
        )

        # handle the case where objective is not superres
        if self.coarse_objective:
            coarse_template_data = template_data.coarsen()
            coarse_low_rank_templates = template_util.svd_compress_templates(
                coarse_template_data,
                min_channel_amplitude=min_channel_amplitude,
                rank=svd_compression_rank,
            )
            temporal_components = coarse_low_rank_templates.temporal_components.astype(
                dtype
            )
            singular_values = coarse_low_rank_templates.singular_values.astype(dtype)
            spatial_components = coarse_low_rank_templates.spatial_components.astype(
                dtype
            )
            self.objective_template_depths_um = (
                coarse_template_data.registered_template_depths_um
            )
            self.register_buffer(
                "objective_temporal_components", torch.tensor(temporal_components)
            )
            self.register_buffer(
                "objective_singular_values", torch.tensor(singular_values)
            )
            self.register_buffer(
                "objective_spatial_components", torch.tensor(spatial_components)
            )
            self.obj_n_templates = spatial_components.shape[0]
        else:
            coarse_template_data = template_data
            coarse_low_rank_templates = low_rank_templates
            self.objective_template_depths_um = self.registered_template_depths_um
            self.register_buffer(
                "objective_temporal_components", self.temporal_components
            )
            self.register_buffer("objective_singular_values", self.singular_values)
            self.register_buffer(
                "objective_spatial_components", self.spatial_components
            )
            self.obj_n_templates = self.n_templates
        self.handle_template_groups(
            coarse_template_data.unit_ids, self.template_data.unit_ids
        )
        convlen = self.chunk_length_samples + self.chunk_margin_samples
        block_size, *_ = spiketorch._calc_oa_lens(convlen, self.spike_length_samples)
        self.register_buffer(
            "objective_temporalf",
            torch.fft.rfft(self.objective_temporal_components, dim=1, n=block_size),
        )

        chunk_starts = np.arange(
            0, self.recording.get_num_samples(), self.chunk_length_samples
        )
        chunk_ends = np.minimum(
            chunk_starts + self.chunk_length_samples, self.recording.get_num_samples()
        )
        chunk_centers_samples = (chunk_starts + chunk_ends) / 2
        chunk_centers_s = self.recording._recording_segments[0].sample_index_to_time(
            chunk_centers_samples
        )
        self.pairwise_conv_db = CompressedPairwiseConv.from_template_data(
            save_folder / "pconv.h5",
            template_data=coarse_template_data,
            low_rank_templates=coarse_low_rank_templates,
            template_data_b=template_data,
            low_rank_templates_b=low_rank_templates,
            compressed_upsampled_temporal=compressed_upsampled_temporal,
            chunk_time_centers_s=chunk_centers_s,
            motion_est=self.motion_est,
            geom=self.geom,
            overwrite=overwrite,
            conv_ignore_threshold=self.conv_ignore_threshold,
            coarse_approx_error_threshold=self.coarse_approx_error_threshold,
            device=device,
            n_jobs=n_jobs,
        )

        self.fixed_output_data += [
            ("temporal_components", temporal_components),
            ("singular_values", singular_values),
            ("spatial_components", spatial_components),
        ]

    def handle_upsampling(
        self,
        temporal_components,
        ptps,
        temporal_upsampling_factor=8,
        upsampling_peak_window_radius=8,
    ):
        compressed_upsampled_temporal = template_util.compressed_upsampled_templates(
            temporal_components,
            ptps=ptps,
            max_upsample=temporal_upsampling_factor,
        )
        self.register_buffer(
            "compressed_upsampling_map",
            torch.tensor(compressed_upsampled_temporal.compressed_upsampling_map),
        )
        self.register_buffer(
            "compressed_upsampling_index",
            torch.tensor(compressed_upsampled_temporal.compressed_upsampling_index),
        )
        self.register_buffer(
            "compressed_index_to_upsampling_index",
            torch.tensor(
                compressed_upsampled_temporal.compressed_index_to_upsampling_index
            ),
        )
        self.register_buffer(
            "compressed_upsampled_temporal",
            torch.tensor(compressed_upsampled_temporal.compressed_upsampled_templates),
        )
        return compressed_upsampled_temporal

    @classmethod
    def from_config(
        cls,
        recording,
        waveform_config,
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
        trough_offset_samples = waveform_config.trough_offset_samples(
            recording.sampling_frequency
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
            conv_ignore_threshold=matching_config.conv_ignore_threshold,
            coarse_approx_error_threshold=matching_config.coarse_approx_error_threshold,
            trough_offset_samples=trough_offset_samples,
            threshold=matching_config.threshold,
            chunk_length_samples=matching_config.chunk_length_samples,
            n_chunks_fit=matching_config.n_chunks_fit,
            max_waveforms_fit=matching_config.max_waveforms_fit,
            fit_subsampling_random_state=matching_config.fit_subsampling_random_state,
            n_waveforms_fit=matching_config.n_waveforms_fit,
            fit_sampling=matching_config.fit_sampling,
            max_iter=matching_config.max_iter,
        )

    def peel_chunk(
        self,
        traces,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
        return_conv=False,
    ):
        # get current template set
        chunk_center_samples = chunk_start_samples + self.chunk_length_samples // 2

        segment = self.recording._recording_segments[0]
        chunk_center_seconds = segment.sample_index_to_time(chunk_center_samples)
        compressed_template_data = self.templates_at_time(chunk_center_seconds)

        # deconvolve
        match_results = self.match_chunk(
            traces,
            compressed_template_data,
            trough_offset_samples=self.trough_offset_samples,
            left_margin=left_margin,
            right_margin=right_margin,
            threshold=self.threshold,
            return_residual=return_residual,
            return_conv=return_conv,
        )

        # process spike times and create return result
        match_results["times_samples"] += chunk_start_samples - left_margin

        return match_results

    def templates_at_time(self, t_s, spatial_mask=None):
        """Handle drift -- grab the right spatial neighborhoods."""
        pconvdb = self.pairwise_conv_db
        pitch_shifts_a = pitch_shifts_b = None
        if (
            self.objective_spatial_components.device.type == "cuda"
            and not pconvdb.device.type == "cuda"
        ):
            pconvdb.to(self.objective_spatial_components.device)
        if self.is_drifting:
            assert spatial_mask is None
            pitch_shifts_b, cur_spatial = template_util.templates_at_time(
                t_s,
                self.spatial_components,
                self.geom,
                registered_template_depths_um=self.registered_template_depths_um,
                registered_geom=self.registered_geom,
                motion_est=self.motion_est,
                return_pitch_shifts=True,
                geom_kdtree=self.geom_kdtree,
                match_distance=self.match_distance,
            )
            if self.coarse_objective:
                pitch_shifts_a, cur_obj_spatial = template_util.templates_at_time(
                    t_s,
                    self.objective_spatial_components,
                    self.geom,
                    registered_template_depths_um=self.objective_template_depths_um,
                    registered_geom=self.registered_geom,
                    motion_est=self.motion_est,
                    return_pitch_shifts=True,
                    geom_kdtree=self.geom_kdtree,
                    match_distance=self.match_distance,
                )
            else:
                cur_obj_spatial = cur_spatial
                pitch_shifts_a = pitch_shifts_b
            cur_ampvecs = drift_util.get_waveforms_on_static_channels(
                self.registered_template_ampvecs[:, None, :],
                self.registered_geom,
                n_pitches_shift=pitch_shifts_b,
                registered_geom=self.geom,
                target_kdtree=self.geom_kdtree,
                match_distance=self.match_distance,
                fill_value=0.0,
            )
            max_channels = cur_ampvecs[:, 0, :].argmax(1)
            # pitch_shifts_a = torch.as_tensor(pitch_shifts_a)
            # pitch_shifts_b = torch.as_tensor(pitch_shifts_b)
            pitch_shifts_a = torch.as_tensor(
                pitch_shifts_a, device=cur_obj_spatial.device
            )
            pitch_shifts_b = torch.as_tensor(
                pitch_shifts_b, device=cur_obj_spatial.device
            )
            # pconvdb = pconvdb.at_shifts(pitch_shifts_a, pitch_shifts_b)
        else:
            cur_spatial = self.spatial_components
            cur_obj_spatial = self.objective_spatial_components
            if spatial_mask is not None:
                cur_spatial = cur_spatial[:, :, spatial_mask]
                cur_obj_spatial = cur_obj_spatial[:, :, spatial_mask]
            max_channels = self.registered_template_ampvecs.argmax(1)

        # if not pconvdb._is_torch:
        # pconvdb.to("cpu")
        # if cur_obj_spatial.device.type == "cuda" and not pconvdb.device.type == "cuda":
        #     pconvdb.to(cur_obj_spatial.device, pin=True)

        return MatchingTemplateData(
            objective_spatial_components=cur_obj_spatial,
            objective_singular_values=self.objective_singular_values,
            objective_temporal_components=self.objective_temporal_components,
            objective_temporalf=self.objective_temporalf,
            fine_to_coarse=self.fine_to_coarse,
            coarse_objective=self.coarse_objective,
            spatial_components=cur_spatial,
            singular_values=self.singular_values,
            temporal_components=self.temporal_components,
            compressed_upsampling_map=self.compressed_upsampling_map,
            compressed_upsampling_index=self.compressed_upsampling_index,
            compressed_index_to_upsampling_index=self.compressed_index_to_upsampling_index,
            compressed_upsampled_temporal=self.compressed_upsampled_temporal,
            max_channels=torch.as_tensor(max_channels, device=cur_obj_spatial.device),
            pairwise_conv_db=pconvdb,
            # shifts_a=None,
            # shifts_b=None,
            shifts_a=pitch_shifts_a,
            shifts_b=pitch_shifts_b,
        )

    def match_chunk(
        self,
        traces,
        compressed_template_data,
        trough_offset_samples=42,
        left_margin=0,
        right_margin=0,
        threshold=30,
        return_collisioncleaned_waveforms=True,
        return_residual=False,
        return_conv=False,
        unit_mask=None,
    ):
        """Core peeling routine for subtraction"""
        # initialize residual, it needs to be padded to support our channel
        # indexing convention (used later to extract small channel
        # neighborhoods). this copies the input.
        residual_padded = F.pad(traces, (0, 1), value=torch.nan)
        residual = residual_padded[:, :-1]

        # name objective variables so that we can update them in-place later
        conv_len = traces.shape[0] - self.spike_length_samples + 1
        padded_obj_len = conv_len + 2 * self.obj_pad_len
        padded_conv = torch.zeros(
            self.obj_n_templates,
            padded_obj_len,
            dtype=traces.dtype,
            device=traces.device,
        )
        padded_objective = torch.zeros(
            self.obj_n_templates + 1,
            padded_obj_len,
            dtype=traces.dtype,
            device=traces.device,
        )
        refrac_mask = torch.zeros_like(padded_objective)
        # padded objective has an extra unit (for group_index) and refractory
        # padding (for easier implementation of enforce_refractory)

        # manages buffers for spike train data (peak times, labels, etc)
        peaks = MatchingPeaks(device=traces.device)

        # initialize convolution
        compressed_template_data.convolve(
            residual.T, padding=self.obj_pad_len, out=padded_conv
        )

        # main loop
        for it in range(self.max_iter):
            # find high-res peaks
            new_peaks = self.find_peaks(
                residual,
                padded_conv,
                padded_objective,
                refrac_mask,
                compressed_template_data,
                unit_mask=unit_mask,
            )
            if new_peaks is None:
                break

            # enforce refractoriness
            self.enforce_refractory(
                refrac_mask,
                new_peaks.times + self.obj_pad_len,
                new_peaks.objective_template_indices,
                new_peaks.template_indices,
            )

            # subtract them
            # old_norm = torch.linalg.norm(residual) ** 2
            compressed_template_data.subtract(
                residual_padded,
                new_peaks.times,
                new_peaks.template_indices,
                new_peaks.upsampling_indices,
                new_peaks.scalings,
            )
            compressed_template_data.subtract_conv(
                padded_conv,
                new_peaks.times,
                new_peaks.template_indices,
                new_peaks.upsampling_indices,
                new_peaks.scalings,
                conv_pad_len=self.obj_pad_len,
            )

            # update spike train
            peaks.extend(new_peaks)

        # subset to peaks inside the margin and sort for the caller
        max_time = traces.shape[0] - right_margin - 1
        valid = peaks.times == peaks.times.clamp(left_margin, max_time)
        peaks.subset(*torch.nonzero(valid, as_tuple=True), sort=True)

        # extract collision-cleaned waveforms on small neighborhoods
        channels = waveforms = None
        if return_collisioncleaned_waveforms:
            channels, waveforms = compressed_template_data.get_collisioncleaned_waveforms(
                residual_padded,
                peaks,
                self.channel_index,
                spike_length_samples=self.spike_length_samples,
            )

        res = dict(
            n_spikes=peaks.n_spikes,
            times_samples=peaks.times + self.trough_offset_samples,
            channels=channels,
            labels=self.unit_ids[peaks.template_indices],
            template_indices=peaks.template_indices,
            upsampling_indices=peaks.upsampling_indices,
            scalings=peaks.scalings,
            scores=peaks.scores,
            collisioncleaned_waveforms=waveforms,
        )
        if return_residual:
            res["residual"] = residual[left_margin : traces.shape[0] - right_margin]
        if return_conv:
            res["conv"] = padded_conv
        return res

    def find_peaks(
        self,
        residual,
        padded_conv,
        padded_objective,
        refrac_mask,
        compressed_template_data,
        unit_mask=None,
    ):
        # update the coarse objective
        torch.add(
            compressed_template_data.objective_template_norms_squared.neg()[:, None],
            padded_conv,
            alpha=2.0,
            out=padded_objective[:-1],
        )

        # first step: coarse peaks. not temporally upsampled or amplitude-scaled.
        objective = (padded_objective + refrac_mask)[
            :-1, self.obj_pad_len : -self.obj_pad_len
        ]
        if unit_mask is not None:
            objective[torch.logical_not(unit_mask)] = -torch.inf
        # formerly used detect_and_deduplicate, but that was slow.
        objective_max, max_obj_template = objective.max(dim=0)
        times = spiketorch.argrelmax(objective_max, self.spike_length_samples, self.threshold)
        obj_template_indices = max_obj_template[times]
        # remove peaks inside the padding
        if not times.numel():
            return None

        residual_snips = None
        if self.coarse_objective or self.temporal_upsampling_factor > 1:
            residual_snips = spiketorch.grab_spikes_full(
                residual,
                times,
                trough_offset=0,
                spike_length_samples=self.spike_length_samples + 1,
            )

        # second step: high-res peaks (upsampled and/or amp-scaled)
        (
            time_shifts,
            upsampling_indices,
            scalings,
            template_indices,
            scores,
        ) = compressed_template_data.fine_match(
            padded_conv[obj_template_indices, times + self.obj_pad_len],
            objective_max[times],
            residual_snips,
            obj_template_indices,
            # times,
            amp_scale_variance=self.amplitude_scaling_variance,
            amp_scale_min=self.amp_scale_min,
            amp_scale_max=self.amp_scale_max,
            superres_index=self.superres_index,
        )
        if time_shifts is not None:
            times += time_shifts

        return MatchingPeaks(
            n_spikes=times.numel(),
            times=times,
            objective_template_indices=obj_template_indices,
            template_indices=template_indices,
            upsampling_indices=upsampling_indices,
            scalings=scalings,
            scores=scores,
        )

    def enforce_refractory(
        self, objective, times, objective_template_indices, template_indices
    ):
        if not times.numel():
            return
        # overwrite objective with -inf to enforce refractoriness
        time_ix = times[:, None] + self._refrac_ix[None, :]
        if not self.grouped_temps:
            row_ix = template_indices[:, None]
        elif self.coarse_objective:
            row_ix = objective_template_indices[:, None]
        elif self.grouped_temps:
            row_ix = self.group_index[template_indices]
        else:
            assert False
        objective[row_ix[:, :, None], time_ix[:, None, :]] = -torch.inf


@dataclass
class MatchingTemplateData:
    """All the data and math needed for computing convs etc in a single static chunk of data

    This is the 'model' for template matching in a MVC analogy. The class above is the controller.
    Objects of this class are returned by ObjectiveUpdateTemplateMatchingPeeler.templates_at_time(),
    which handles the drift logic and lets this class be simple.
    """

    objective_spatial_components: torch.Tensor
    objective_singular_values: torch.Tensor
    objective_temporal_components: torch.Tensor
    objective_temporalf: torch.Tensor
    fine_to_coarse: torch.LongTensor
    coarse_objective: bool
    spatial_components: torch.Tensor
    singular_values: torch.Tensor
    temporal_components: torch.Tensor
    compressed_upsampling_map: torch.LongTensor
    compressed_upsampling_index: torch.LongTensor
    compressed_index_to_upsampling_index: torch.LongTensor
    compressed_upsampled_temporal: torch.Tensor
    max_channels: torch.LongTensor
    pairwise_conv_db: CompressedPairwiseConv
    shifts_a: Optional[torch.Tensor]
    shifts_b: Optional[torch.Tensor]

    def __post_init__(self):
        (
            self.n_templates,
            self.spike_length_samples,
            self.rank,
        ) = self.temporal_components.shape
        assert self.spatial_components.shape[:2] == (self.n_templates, self.rank)
        assert self.compressed_upsampled_temporal.shape[1:] == (
            self.spike_length_samples,
            self.rank,
        )
        assert self.singular_values.shape == (self.n_templates, self.rank)
        device = self.spatial_components.device
        self.temporal_upsampling_factor = self.compressed_upsampling_index.shape[1]
        self.n_compressed_upsampled_templates = self.compressed_upsampling_map.max() + 1

        # squared l2 norms are usually the sums of squared singular values:
        # self.template_norms_squared = torch.square(self.singular_values).sum(1)
        # in this case, we have subset the spatial components, so use a diff formula
        self.objective_n_templates = self.objective_spatial_components.shape[0]
        self.objective_spatial_singular = (
            self.objective_spatial_components
            * self.objective_singular_values[:, :, None]
        )
        self.spatial_singular = (
            self.spatial_components * self.singular_values[:, :, None]
        )
        self.objective_template_norms_squared = torch.square(
            self.objective_spatial_singular
        ).sum((1, 2))
        self.template_norms_squared = torch.square(self.spatial_singular).sum((1, 2))
        self.chan_ix = torch.arange(self.spatial_components.shape[2], device=device)
        self.rank_ix = torch.arange(self.rank, device=device)
        self.time_ix = torch.arange(self.spike_length_samples, device=device)
        self.conv_lags = torch.arange(
            -self.spike_length_samples + 1, self.spike_length_samples, device=device
        )

    def convolve(self, traces, padding=0, out=None):
        """Convolve the objective templates with traces."""
        return spiketorch.convolve_lowrank(
            traces,
            self.objective_spatial_singular,
            self.objective_temporal_components,
            padding=padding,
            out=out,
        )

    def subtract_conv(
        self,
        conv,
        times,
        template_indices,
        upsampling_indices,
        scalings,
        conv_pad_len=0,
        batch_size=256,
    ):
        n_spikes = times.shape[0]
        for batch_start in range(0, n_spikes, batch_size):
            batch_end = min(batch_start + batch_size, n_spikes)
            (
                template_indices_a,
                template_indices_b,
                times_sub,
                pconvs,
            ) = self.pairwise_conv_db.query(
                template_indices_a=None,
                template_indices_b=template_indices[batch_start:batch_end],
                upsampling_indices_b=upsampling_indices[batch_start:batch_end],
                scalings_b=scalings[batch_start:batch_end],
                times_b=times[batch_start:batch_end],
                grid=True,
                device=conv.device,
                shifts_a=self.shifts_a,
                shifts_b=(
                    self.shifts_b[template_indices[batch_start:batch_end]]
                    if self.shifts_b is not None
                    else None
                ),
            )
            ix_template = template_indices_a[:, None]
            ix_time = times_sub[:, None] + (conv_pad_len + self.conv_lags)[None, :]
            spiketorch.add_at_(
                conv,
                (ix_template, ix_time),
                pconvs,
                sign=-1,
            )

    def fine_match(
        self,
        convs,
        objs,
        residual_snips,
        objective_template_indices,
        # times,
        amp_scale_variance=0.0,
        amp_scale_min=None,
        amp_scale_max=None,
        superres_index=None,
    ):
        """Determine superres ids, temporal upsampling, and scaling

        Given coarse matches (unit ids at times) and the current residual,
        pick the best superres template, the best temporal offset, and the
        best amplitude scaling.

        We used to upsample the objective to figure out the temporal upsampling,
        but with superres in the picture we are now not computing the objective
        using the same templates that we temporally upsample. So, instead
        we use a greedy strategy: first pick the best (non-temporally upsampled)
        superres template, then pick the upsampling and scaling at the same time.
        These are all done by dotting everything and computing the objective,
        which is probably more expensive than what we had before.

        Returns
        -------
        time_shifts : Optional[array]
        upsampling_indices : Optional[array]
        scalings : Optional[array]
        template_indices : array
        objs : array
        """
        if (
            not self.coarse_objective
            and self.temporal_upsampling_factor == 1
            and not amp_scale_variance
        ):
            return None, None, None, objective_template_indices, objs

        if self.coarse_objective or self.temporal_upsampling_factor > 1:
            # snips is a window padded by one sample, so that we have the
            # traces snippets at the current times and one step back
            n_spikes, window_length_samples, n_chans = residual_snips.shape
            # spike_length_samples = window_length_samples - 1
            # grab the current traces
            snips = residual_snips[:, 1:]
            # snips_dt = F.unfold(
            #     residual_snips[:, None, :, :], (spike_length_samples, snips.shape[2])
            # )
            # snips_dt = snips_dt.reshape(
            #     len(snips), spike_length_samples, snips.shape[2], 2
            # )

        if self.coarse_objective:
            # TODO best I came up with, but it still syncs
            superres_ix = superres_index[objective_template_indices]
            dup_ix, column_ix = (superres_ix < self.n_templates).nonzero(as_tuple=True)
            template_indices = superres_ix[dup_ix, column_ix]
            convs = torch.baddbmm(
                self.temporal_components[template_indices],
                snips[dup_ix],
                self.spatial_singular[template_indices].mT,
            ).sum((1, 2))
            # convs = torch.einsum(
            #     "jtc,jrc,jtr->j",
            #     snips[dup_ix],
            #     self.spatial_singular[template_indices],
            #     self.temporal_components[template_indices],
            # )
            norms = self.template_norms_squared[template_indices]
            objs = torch.full(superres_ix.shape, -torch.inf, device=convs.device)
            objs[dup_ix, column_ix] = 2 * convs - norms
            objs, best_column_ix = objs.max(dim=1)
            row_ix = torch.arange(best_column_ix.numel(), device=best_column_ix.device)
            template_indices = superres_ix[row_ix, best_column_ix]
        else:
            template_indices = objective_template_indices
            norms = self.template_norms_squared[template_indices]
            objs = objs

        if self.temporal_upsampling_factor == 1 and not amp_scale_variance:
            return None, None, None, template_indices, objs

        if self.temporal_upsampling_factor == 1:
            # just scaling
            inv_lambda = 1 / amp_scale_variance
            b = convs + inv_lambda
            a = norms + inv_lambda
            scalings = torch.clip(b / a, amp_scale_min, amp_scale_max)
            objs = 2 * scalings * b - torch.square(scalings) * a - inv_lambda
            return None, None, scalings, template_indices, objs

        # unpack the current traces and the traces one step back
        snips_prev = residual_snips[:, :-1]
        # snips_dt = torch.stack((snips_prev, snips), dim=3)

        # now, upsampling
        # repeat the superres logic, the comp up index acts the same
        comp_up_ix = self.compressed_upsampling_index[template_indices]
        dup_ix, column_ix = (
            comp_up_ix < self.n_compressed_upsampled_templates
        ).nonzero(as_tuple=True)
        comp_up_indices = comp_up_ix[dup_ix, column_ix]
        # convs = torch.einsum(
        #     "jtcd,jrc,jtr->jd",
        #     snips_dt[dup_ix],
        #     self.spatial_singular[template_indices[dup_ix]],
        #     self.compressed_upsampled_temporal[comp_up_indices],
        # )
        temps = torch.bmm(
            self.compressed_upsampled_temporal[comp_up_indices],
            self.spatial_singular[template_indices[dup_ix]],
        ).view(len(comp_up_indices), -1)
        convs = torch.linalg.vecdot(snips[dup_ix].view(len(temps), -1), temps)
        convs_prev = torch.linalg.vecdot(snips_prev[dup_ix].view(len(temps), -1), temps)

        # convs_r = torch.round(convs).to(int).numpy()
        # convs_prev_r = torch.round(convs_prev).to(int).numpy()
        # convs = torch.einsum(
        #     "jtc,jrc,jtr->j",
        #     snips[dup_ix],
        #     self.spatial_singular[template_indices[dup_ix]],
        #     self.compressed_upsampled_temporal[comp_up_indices],
        # )
        # convs_prev = torch.einsum(
        #     "jtc,jrc,jtr->j",
        #     snips_prev[dup_ix],
        #     self.spatial_singular[template_indices[dup_ix]],
        #     self.compressed_upsampled_temporal[comp_up_indices],
        # )
        better = convs >= convs_prev
        convs = torch.maximum(convs, convs_prev)

        norms = norms[dup_ix]
        objs = torch.full(comp_up_ix.shape, -torch.inf, device=convs.device)
        if amp_scale_variance:
            inv_lambda = 1 / amp_scale_variance
            b = convs + inv_lambda
            a = norms + inv_lambda
            scalings = torch.clip(b / a, amp_scale_min, amp_scale_max)
            objs[dup_ix, column_ix] = (
                2 * scalings * b - torch.square(scalings) * a - inv_lambda
            )
        else:
            objs[dup_ix, column_ix] = 2 * convs - norms
            scalings = None
        objs, best_column_ix = objs.max(dim=1)

        row_ix = torch.arange(len(objs), device=best_column_ix.device)
        comp_up_indices = comp_up_ix[row_ix, best_column_ix]
        upsampling_indices = self.compressed_index_to_upsampling_index[comp_up_indices]

        # prev convs were one step earlier
        # time_shifts = torch.full(comp_up_ix.shape, -1, device=convs.device)
        # time_shifts[dup_ix, column_ix] += better
        time_shifts = torch.full(comp_up_ix.shape, 0, device=convs.device)
        time_shifts[dup_ix, column_ix] += better.to(int)
        time_shifts = time_shifts[row_ix, best_column_ix]

        return time_shifts, upsampling_indices, scalings, template_indices, objs

    def subtract(
        self,
        traces,
        times,
        template_indices,
        upsampling_indices,
        scalings,
        batch_templates=...,
    ):
        """Subtract templates from traces."""
        compressed_up_inds = self.compressed_upsampling_map[
            template_indices, upsampling_indices
        ]
        batch_templates = torch.einsum(
            "n,nrc,ntr->ntc",
            scalings,
            self.spatial_singular[template_indices],
            self.compressed_upsampled_temporal[compressed_up_inds],
        )
        time_ix = times[:, None, None] + self.time_ix[None, :, None]
        spiketorch.add_at_(
            traces, (time_ix, self.chan_ix[None, None, :]), batch_templates, sign=-1
        )

    def get_collisioncleaned_waveforms(
        self, residual_padded, peaks, channel_index, spike_length_samples=121
    ):
        channels = self.max_channels[peaks.template_indices]
        waveforms = spiketorch.grab_spikes(
            residual_padded,
            peaks.times,
            channels,
            channel_index,
            trough_offset=0,
            spike_length_samples=spike_length_samples,
            buffer=0,
            already_padded=True,
        )
        padded_spatial = F.pad(self.spatial_singular, (0, 1))
        spatial = padded_spatial[
            peaks.template_indices[:, None, None],
            self.rank_ix[None, :, None],
            channel_index[channels][:, None, :],
        ]
        comp_up_ix = self.compressed_upsampling_map[
            peaks.template_indices, peaks.upsampling_indices
        ]
        temporal = self.compressed_upsampled_temporal[comp_up_ix]
        torch.baddbmm(waveforms, temporal, spatial, out=waveforms)
        return channels, waveforms


class MatchingPeaks:
    BUFFER_INIT: int = 1500
    BUFFER_GROWTH: float = 1.5

    def __init__(
        self,
        n_spikes: int = 0,
        times: Optional[torch.LongTensor] = None,
        objective_template_indices: Optional[torch.LongTensor] = None,
        template_indices: Optional[torch.LongTensor] = None,
        upsampling_indices: Optional[torch.LongTensor] = None,
        scalings: Optional[torch.Tensor] = None,
        scores: Optional[torch.Tensor] = None,
        device=None,
    ):
        self.n_spikes = n_spikes
        self._times = times
        self._template_indices = template_indices
        self._objective_template_indices = objective_template_indices
        self._upsampling_indices = upsampling_indices
        self._scalings = scalings
        self._scores = scores

        if device is None and times is not None:
            device = times.device
        if times is None:
            self.cur_buf_size = self.BUFFER_INIT
            self._times = torch.zeros(self.cur_buf_size, dtype=int, device=device)
        else:
            self.cur_buf_size = times.numel()
            assert self.cur_buf_size == n_spikes
        if template_indices is None:
            self._template_indices = torch.zeros(
                self.cur_buf_size, dtype=int, device=device
            )
        if objective_template_indices is None:
            self._objective_template_indices = torch.zeros(
                self.cur_buf_size, dtype=int, device=device
            )
        if scalings is None:
            self._scalings = torch.ones(self.cur_buf_size, device=device)
        if upsampling_indices is None:
            self._upsampling_indices = torch.zeros(
                self.cur_buf_size, dtype=int, device=device
            )
        if scores is None:
            self._scores = torch.zeros(self.cur_buf_size, device=device)

    @property
    def times(self):
        return self._times[: self.n_spikes]

    @property
    def template_indices(self):
        return self._template_indices[: self.n_spikes]

    @property
    def objective_template_indices(self):
        return self._objective_template_indices[: self.n_spikes]

    @property
    def upsampling_indices(self):
        return self._upsampling_indices[: self.n_spikes]

    @property
    def scalings(self):
        return self._scalings[: self.n_spikes]

    @property
    def scores(self):
        return self._scores[: self.n_spikes]

    def subset(self, which, sort=False):
        self._times = self.times[which]
        if sort:
            self._times, order = torch.sort(self._times, stable=True)
            which = which[order]
        self._template_indices = self.template_indices[which]
        self._objective_template_indices = self.objective_template_indices[which]
        self._upsampling_indices = self.upsampling_indices[which]
        self._scalings = self.scalings[which]
        self._scores = self.scores[which]
        self.n_spikes = self._times.numel()

    def grow_buffers(self, min_size=0):
        sz = max(min_size, int(self.cur_buf_size * self.BUFFER_GROWTH))
        k = self.n_spikes
        self._times = _grow_buffer(self._times, k, sz)
        self._template_indices = _grow_buffer(self._template_indices, k, sz)
        self._objective_template_indices = _grow_buffer(
            self._objective_template_indices, k, sz
        )
        self._upsampling_indices = _grow_buffer(self._upsampling_indices, k, sz)
        self._scalings = _grow_buffer(self._scalings, k, sz)
        self._scores = _grow_buffer(self._scores, k, sz)
        self.cur_buf_size = sz

    def sort(self):
        sl = slice(0, self.n_spikes)

        times = self._times[sl]
        order = torch.argsort(times, stable=True)

        self._times[sl] = times[order]
        self._template_indices[sl] = self.template_indices[order]
        self._objective_template_indices[sl] = self.objective_template_indices[order]
        self._upsampling_indices[sl] = self.upsampling_indices[order]
        self._scalings[sl] = self.scalings[order]
        self._scores[sl] = self.scores[order]

    def extend(self, other):
        new_n_spikes = other.n_spikes + self.n_spikes
        sl_new = slice(self.n_spikes, new_n_spikes)

        if new_n_spikes > self.cur_buf_size:
            self.grow_buffers(min_size=new_n_spikes)

        self._times[sl_new] = other.times
        self._template_indices[sl_new] = other.template_indices
        self._objective_template_indices[sl_new] = other.objective_template_indices
        self._upsampling_indices[sl_new] = other.upsampling_indices
        self._scalings[sl_new] = other.scalings
        self._scores[sl_new] = other.scores

        self.n_spikes = new_n_spikes


def _grow_buffer(x, old_length, new_size):
    new = torch.empty(new_size, dtype=x.dtype, device=x.device)
    new[:old_length] = x[:old_length]
    return new
