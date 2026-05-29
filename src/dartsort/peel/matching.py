"""A simple residual updating template matcher."""

from pathlib import Path
from typing import Self, cast

import numpy as np
import torch
import torch.nn.functional as F
from spikeinterface.core import BaseRecording
from torch import Tensor

from ..templates import LowRankTemplates, TemplateData
from ..transform import WaveformPipeline
from ..util.data_util import SpikeDataset
from ..util.internal_config import (
    ComputationConfig,
    FeaturizationConfig,
    FitSamplingConfig,
    MatchingConfig,
    WaveformConfig,
    default_matching_cfg,
    default_peeling_fit_sampling_cfg,
    default_waveform_cfg,
)
from ..util.logging_util import get_logger
from ..util.motion import MotionInfo
from ..util.waveform_util import make_channel_index
from .matching_util import (
    ChunkTemplateData,
    MatchingPeaks,
    MatchingTemplates,
    MatchingTemplatesBuilder,
)
from .peel_base import BasePeeler, PeelingBatchResult

logger = get_logger(__name__)


class ObjectiveUpdateTemplateMatchingPeeler(BasePeeler):
    peel_kind = "TemplateMatching"

    def __init__(
        self,
        recording,
        channel_index,
        featurization_pipeline,
        matching_templates: MatchingTemplates | None = None,
        matching_templates_builder: MatchingTemplatesBuilder | None = None,
        p: MatchingConfig = default_matching_cfg,
        waveform_cfg: WaveformConfig = default_waveform_cfg,
        fpctrl_spike_counts=None,
        fit_sampling_cfg: FitSamplingConfig = default_peeling_fit_sampling_cfg,
        save_collidedness=False,
        whiten_features=True,
        parent_sorting_hdf5_path: str | Path | None = None,
        dtype=torch.float,
    ):
        if matching_templates is not None:
            spike_length_samples = matching_templates.spike_length_samples
        elif matching_templates_builder is not None:
            spike_length_samples = matching_templates_builder.spike_length_samples
        else:
            raise ValueError("Need either a MatchingTemplates or a builder.")

        fixed_prop_keys = ("channels", "labels", "times_seconds")
        if save_collidedness:
            fixed_prop_keys = fixed_prop_keys + ("collidedness",)

        super().__init__(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            chunk_length_samples=p.chunk_length_samples,
            chunk_margin_samples=p.margin_factor * spike_length_samples + 1,
            waveform_cfg=waveform_cfg,
            fit_sampling_cfg=fit_sampling_cfg,
            fixed_property_keys=fixed_prop_keys,
            dtype=dtype,
        )
        self.p = p
        self.matching_templates = matching_templates
        self.matching_templates_builder = matching_templates_builder
        self.thresholdsq: float | None = None  # set in precompute
        self.save_collidedness = save_collidedness
        self.whiten_features = whiten_features

        # fp control threshold stuff (TODO: remove?)
        self.fpctrl_spike_counts = fpctrl_spike_counts
        self.parent_sorting_hdf5_path = parent_sorting_hdf5_path

        geom = recording.get_channel_locations()
        self.picking_channels = p.channel_selection != "template"
        if p.channel_selection == "amplitude":
            assert p.channel_selection_radius is not None
            channel_selection_index = make_channel_index(
                geom, p.channel_selection_radius, to_torch=True
            )
        elif p.channel_selection == "template":
            channel_selection_index = None
        else:
            assert False
        self.register_buffer_or_none("channel_selection_index", channel_selection_index)
        self.is_upsampling = p.up_factor > 1

        # amplitude scaling properties
        self.is_scaling = p.amplitude_scaling_variance > 0
        self.inv_lambda = (
            1.0 / p.amplitude_scaling_variance if self.is_scaling else float("inf")
        )
        self.amp_scale_max = 1.0 + p.amplitude_scaling_boundary
        self.amp_scale_min = 1.0 / self.amp_scale_max
        self.obj_pad_len = max(p.refractory_radius_frames, self.spike_length_samples)
        conv_len = (
            self.chunk_length_samples
            + 2 * self.chunk_margin_samples
            + 2 * self.obj_pad_len
        )
        self.register_buffer("obj_arange", torch.arange(conv_len))

    def peeling_needs_precompute(self):
        return self.matching_templates is None or self.thresholdsq is None

    def precompute_peeling_data(
        self,
        save_folder,
        overwrite=False,
        computation_cfg: ComputationConfig | None = None,
    ):
        if self.matching_templates is None:
            assert self.matching_templates_builder is not None
            self.matching_templates = self.matching_templates_builder.build(
                save_folder, computation_cfg=computation_cfg, overwrite=overwrite
            )
        self.pick_threshold()

    def out_datasets(self):
        datasets = super().out_datasets()
        datasets.extend(
            [
                SpikeDataset(name="template_inds", shape_per_spike=(), dtype=np.int32),
                SpikeDataset(name="labels", shape_per_spike=(), dtype=np.int32),
                SpikeDataset(name="scores", shape_per_spike=(), dtype=np.float32),
            ]
        )
        if self.save_collidedness:
            datasets.append(SpikeDataset("collidedness", (), "float32"))
        if self.is_scaling:
            datasets.append(
                SpikeDataset(name="scalings", shape_per_spike=(), dtype=np.float32),
            )
        if self.is_upsampling:
            datasets.append(
                SpikeDataset(name="up_inds", shape_per_spike=(), dtype=np.int8)
            )
            datasets.append(
                SpikeDataset(name="time_shifts", shape_per_spike=(), dtype=np.int8),
            )
        return datasets

    @classmethod
    def from_config(
        cls,
        recording: BaseRecording,
        *,
        waveform_cfg: WaveformConfig,
        matching_cfg: MatchingConfig,
        featurization_cfg: FeaturizationConfig,
        sampling_cfg: FitSamplingConfig,
        template_data: TemplateData | None,
        motion: MotionInfo | None = None,
        parent_sorting_hdf5_path=None,
    ) -> Self:
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
        featurization_pipeline.attach_motion(motion)

        trough_offset_samples = waveform_cfg.trough_offset_samples(
            recording.sampling_frequency
        )
        if template_data is None:
            assert matching_cfg.precomputed_templates_npz is not None
            template_data = TemplateData.from_npz(
                matching_cfg.precomputed_templates_npz
            )
        assert trough_offset_samples == template_data.trough_offset_samples

        if motion is None:
            motion = MotionInfo.from_motion_est(geom=geom.numpy())

        nofeat = featurization_cfg.skip or featurization_cfg.denoise_only
        do_tpca = featurization_cfg.save_input_tpca_projs and not nofeat
        if do_tpca and featurization_cfg.tpca_from_templates:
            from ..transform import TemporalPCAFeaturizer as TF

            (tpca,) = [f for f in featurization_pipeline if isinstance(f, TF)]
            tpca.initialize_from_templates(template_data)

        builder = MatchingTemplatesBuilder(
            recording=recording,
            template_data=template_data,
            matching_cfg=matching_cfg,
            motion=motion,
        )

        logger.info(
            "Constructing a matcher with template kind %s, drift %senabled, "
            "scaling variance %s, compression rank %s, upsampling factor %s, "
            "refrac radius %s.",
            matching_cfg.template_type,
            "" if motion.drifting else "not ",
            matching_cfg.amplitude_scaling_variance,
            matching_cfg.template_svd_compression_rank,
            matching_cfg.up_factor,
            matching_cfg.refractory_radius_frames,
        )
        save_collidedness = (
            featurization_cfg.save_collidedness and not featurization_cfg.skip
        )

        return cls(
            recording=recording,
            matching_templates_builder=builder,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            p=matching_cfg,
            waveform_cfg=waveform_cfg,
            fit_sampling_cfg=sampling_cfg,
            parent_sorting_hdf5_path=parent_sorting_hdf5_path,
            save_collidedness=save_collidedness,
            whiten_features=matching_cfg.whiten_features,
            fpctrl_spike_counts=template_data.spike_counts
            if matching_cfg.threshold == "fp_control"
            else None,
        )

    def peel_chunk(
        self,
        traces,
        *,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
        return_waveforms=True,
        return_conv=False,
    ) -> PeelingBatchResult:
        assert self.matching_templates is not None
        # get chunk center time and template info at that time
        chunk_center_samples = chunk_start_samples + self.chunk_length_samples // 2
        segment = self.recording._recording_segments[0]
        chunk_center_seconds = float(segment.sample_index_to_time(chunk_center_samples))
        chunk_template_data = self.matching_templates.data_at_time(
            t_s=chunk_center_seconds,
            scaling=self.is_scaling,
            inv_lambda=self.inv_lambda,
            scale_min=self.amp_scale_min,
            scale_max=self.amp_scale_max,
        )

        # deconvolve
        match_results = self.match_chunk(
            traces,
            chunk_template_data,
            left_margin=left_margin,
            right_margin=right_margin,
            return_residual=return_residual,
            return_conv=return_conv,
            return_collisioncleaned_waveforms=return_waveforms,
        )

        # process spike times and create return result
        if match_results["n_spikes"]:
            match_results["times_samples"] += chunk_start_samples - left_margin  # type: ignore
        if match_results["n_spikes"] > self.p.max_spikes_per_second:
            raise ValueError(
                f"Too many spikes {match_results['n_spikes']} > {self.p.max_spikes_per_second}."
            )

        return match_results

    def match_chunk(
        self,
        traces: Tensor,
        chunk_template_data: ChunkTemplateData,
        left_margin=0,
        right_margin=0,
        return_collisioncleaned_waveforms=True,
        return_residual=False,
        return_conv=False,
        unit_mask=None,
        max_iter: int | None = None,
    ) -> PeelingBatchResult:
        """Core peeling routine for subtraction"""
        if max_iter is None:
            max_iter = self.p.max_iter

        # note, this is chans major (transpose of traces)
        traces_wh = chunk_template_data.whiten_traces(traces)

        # initialize residual
        residual = traces_wh.T if self.whiten_features else traces
        residual_padded = F.pad(residual, (0, 1), value=torch.nan)
        residual = residual_padded[:, :-1]

        # name objective variables so that we can update them in-place later
        # padded objective has an extra unit (for group_index) and refractory
        # padding (for easier implementation of enforce_refractory)
        valid_len = traces.shape[0] - self.spike_length_samples + 1
        padded_obj_len = valid_len + 2 * self.obj_pad_len
        padded_conv = traces.new_zeros(
            chunk_template_data.obj_n_templates, padded_obj_len
        )
        padded_scalings = padded_conv.clone() if self.is_scaling else None
        padded_objective = traces.new_zeros(
            chunk_template_data.obj_n_templates + 1, padded_obj_len
        )
        if self.p.refractory_radius_frames:
            refrac_mask = torch.zeros_like(padded_objective)
        else:
            refrac_mask = None

        # initialize convolution
        chunk_template_data.convolve(
            traces_wh, padding=self.obj_pad_len, out=padded_conv
        )

        # main loop
        previous_peaks = current_peaks = prev_refrac_mask = None
        prev_update_residual = None
        for cd_it in range(self.p.cd_iter + 1):
            initializing_cd = not cd_it
            coarse_only = self.p.coarse_cd and cd_it < self.p.cd_iter

            # we always need to update the residual in the final iteration
            # in "cd iterations", we may not need to update the residual.
            update_residual = chunk_template_data.needs_residual and not coarse_only

            if not initializing_cd and refrac_mask is not None:
                refrac_mask = torch.zeros_like(refrac_mask)

            current_peaks = []
            for match_it in range(max_iter):
                if not initializing_cd:
                    assert previous_peaks is not None
                if (
                    not initializing_cd
                    and previous_peaks is not None
                    and len(previous_peaks)
                ):
                    assert prev_update_residual is not None
                    prev_peaks = previous_peaks.pop()
                    if prev_update_residual:
                        chunk_template_data.unsubtract(residual_padded, prev_peaks)
                    chunk_template_data.unsubtract_conv(
                        padded_conv, prev_peaks, padding=self.obj_pad_len
                    )
                    if self.p.refractory_radius_frames:
                        chunk_template_data.forget_refractory(
                            prev_refrac_mask, prev_peaks, offset=self.obj_pad_len
                        )
                    if refrac_mask is not None:
                        assert prev_refrac_mask is not None
                        apply_refrac_mask = refrac_mask + prev_refrac_mask
                    else:
                        apply_refrac_mask = None
                else:
                    apply_refrac_mask = refrac_mask

                # find spikes
                new_peaks = self.find_peaks(
                    residual=residual,
                    padded_conv=padded_conv,
                    padded_scalings=padded_scalings,
                    padded_objective=padded_objective,
                    refrac_mask=apply_refrac_mask,
                    chunk_template_data=chunk_template_data,
                    unit_mask=unit_mask,
                    coarse_only=coarse_only,
                )
                if new_peaks is None or not new_peaks.n_spikes:
                    break

                # enforce refractoriness
                if self.p.refractory_radius_frames:
                    chunk_template_data.enforce_refractory(
                        refrac_mask, new_peaks, offset=self.obj_pad_len
                    )

                # subtract them
                if update_residual:
                    chunk_template_data.subtract(residual_padded, new_peaks)
                chunk_template_data.subtract_conv(
                    padded_conv, new_peaks, padding=self.obj_pad_len
                )

                # update spike train
                current_peaks.append(new_peaks)

            # some of this round's peaks will be added back in before
            # each iteration in the next round
            previous_peaks = current_peaks
            previous_peaks.reverse()
            prev_refrac_mask = refrac_mask
            prev_update_residual = update_residual

        assert current_peaks is not None
        peaks = MatchingPeaks.concatenate(current_peaks)

        # compute the residual now if not done above
        if not chunk_template_data.needs_residual:
            chunk_template_data.subtract(residual_padded, peaks)

        if not peaks.n_spikes:
            res = PeelingBatchResult(n_spikes=0)
            if return_residual:
                residual = residual[left_margin : traces.shape[0] - right_margin]
                res["residual"] = residual
            if return_conv:
                res["conv"] = padded_conv
            return res

        # subset to peaks inside the margin and sort for the caller
        max_time = traces.shape[0] - right_margin - 1
        peaks = peaks.subset_by_time(
            left_margin, max_time, offset=self.trough_offset_samples
        )
        assert peaks.times is not None
        assert peaks.template_inds is not None

        # construct return value
        res = peaks_to_batch_result(
            peaks=peaks,
            trough_offset_samples=self.trough_offset_samples,
            unit_ids=chunk_template_data.unit_ids,
            trough_shifts=peaks.time_shifts,
        )
        # extract collision-cleaned waveforms on small neighborhoods
        if return_collisioncleaned_waveforms or self.picking_channels:
            cc = chunk_template_data.get_collisioncleaned_waveforms(
                residual_padded=residual_padded,
                peaks=peaks,
                channels=self.p.channel_selection,
                channel_index=self.b.channel_index,
                channel_selection_index=self.b.channel_selection_index,
                with_coll=self.save_collidedness,
            )
            channels, waveforms, collidedness = cc
            if self.save_collidedness:
                assert collidedness is not None
                res["collidedness"] = collidedness
        else:
            assert self.p.channel_selection == "template"
            channels = chunk_template_data.main_channels[peaks.template_inds]
            waveforms = None
        res["channels"] = channels
        if return_collisioncleaned_waveforms:
            assert waveforms is not None
            res["collisioncleaned_waveforms"] = waveforms
        if return_residual:
            residual = residual[left_margin : traces.shape[0] - right_margin]
            res["residual"] = residual
        if return_conv:
            res["conv"] = padded_conv
        return res

    def find_peaks(
        self,
        *,
        residual: Tensor,
        padded_conv: Tensor,
        padded_objective: Tensor,
        padded_scalings: Tensor | None,
        refrac_mask: Tensor | None,
        chunk_template_data: ChunkTemplateData,
        unit_mask=None,
        coarse_only=False,
    ):
        # update the coarse objective
        chunk_template_data.obj_from_conv(
            conv=padded_conv, out=padded_objective[:-1], scalings_out=padded_scalings
        )

        # enforce refractoriness
        objective = padded_objective[:-1]
        if refrac_mask is not None:
            objective = objective + refrac_mask[:-1]
        if unit_mask is not None:
            objective[torch.logical_not(unit_mask)] = -torch.inf

        # find peaks in the coarse objective
        assert self.thresholdsq is not None
        coarse_peaks = chunk_template_data.coarse_match(
            objective=objective,
            scalings=padded_scalings,
            thresholdsq=self.thresholdsq,
            obj_arange=self.b.obj_arange,
            padding=self.obj_pad_len,
        )
        if coarse_only or not coarse_peaks.n_spikes:
            return coarse_peaks

        # high-res peaks (upsampled or grouped)
        fine_peaks = chunk_template_data.fine_match(
            peaks=coarse_peaks,
            residual=residual,
            conv=padded_conv,
            padding=self.obj_pad_len,
        )

        return fine_peaks

    def pick_threshold(self):
        from scipy.stats import norm

        # TODO: remove?
        if self.is_scaling and self.p.amplitude_scaling_variance < torch.inf:
            # adjust threshold by the scaling prior's constant term
            # nb, everything is x2 so halves are gone.
            scstd = np.sqrt(self.p.amplitude_scaling_variance)
            norm_const = -(np.log(2.0) + np.log(np.pi) + 2.0 * np.log(scstd))

            # adjust by boundary
            nm = norm(loc=1.0, scale=scstd)
            p_up = nm.cdf(self.amp_scale_max)
            p_lo = nm.cdf(self.amp_scale_min)
            Z = p_up - p_lo

            # renormalize
            scale_const = norm_const - 2.0 * np.log(Z)

            if isinstance(self.p.threshold, float):
                tb = np.sqrt(max(0.0, self.p.threshold**2 - scale_const))
                _msg = f"In norm units, that's from {self.p.threshold:0.2f}->{tb:0.2f}"
            else:
                _msg = ""
            logger.dartsortdebug(
                f"matching: Amplitude scaling with std {scstd:0.4f} adjusts "
                f"theshold by {scale_const:0.4f} in squared units. {_msg}"
            )
        else:
            scale_const = 0.0

        if isinstance(self.p.threshold, float):
            self.thresholdsq = self.p.threshold**2 - scale_const
            return

        assert self.p.threshold == "fp_control"
        from ..util import noise_util

        # fit noise to residuals from the previous detection step
        assert self.parent_sorting_hdf5_path is not None
        assert self.matching_templates is not None
        lrt = cast(LowRankTemplates, self.matching_templates.obj_lrts)
        self.thresholdsq = noise_util.fp_control_threshold_from_h5(
            hdf5_path=self.parent_sorting_hdf5_path,
            low_rank_templates=lrt.shift_to_best_channels(
                self.recording.get_channel_locations(), self.matching_templates.rgeom
            ),
            rg=self.fit_subsampling_random_state,
            refractory_radius_frames=self.p.refractory_radius_frames,
            num_frames=self.recording.get_num_samples(),  # TODO: wrong in subsampled case.
            max_fp_per_input_spike=self.p.max_fp_per_input_spike,
            unit_ids=self.matching_templates.obj_unit_ids,
            spike_counts=self.fpctrl_spike_counts,
        )
        assert isinstance(self.thresholdsq, float)
        self.thresholdsq -= scale_const
        logger.info(
            f"Matcher picked threshold^2 {self.thresholdsq} for strategy fp_control."
        )


def peaks_to_batch_result(
    peaks: MatchingPeaks,
    trough_offset_samples: int,
    unit_ids: Tensor,
    trough_shifts: Tensor | None,
) -> PeelingBatchResult:
    if not peaks.n_spikes:
        return PeelingBatchResult(n_spikes=0)

    assert peaks.times is not None
    assert peaks.template_inds is not None
    times_samples = peaks.times + trough_offset_samples
    if trough_shifts is not None:
        times_samples += trough_shifts
    res = PeelingBatchResult(
        n_spikes=peaks.n_spikes,
        times_samples=times_samples,
        labels=unit_ids[peaks.template_inds],
        template_inds=peaks.template_inds,
    )
    if peaks.up_inds is not None:
        res["up_inds"] = peaks.up_inds
    if trough_shifts is not None:
        res["time_shifts"] = trough_shifts
    if peaks.scalings is not None:
        res["scalings"] = peaks.scalings
    if peaks.scores is not None:
        res["scores"] = peaks.scores
    return res
