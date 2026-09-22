"""A simple residual updating template matcher."""

from typing import Self

import numpy as np
import torch
import torch.nn.functional as F
from spikeinterface.core import BaseRecording
from torch import Tensor

from ..templates import TemplateData
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
from ..util.py_util import panic
from ..util.waveform_util import full_channel_index, make_channel_index
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
        fit_sampling_cfg: FitSamplingConfig = default_peeling_fit_sampling_cfg,
        *,
        save_collidedness=False,
        whiten_features=False,
        whiten_kernel_length=0,
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
            fixed_prop_keys = (*fixed_prop_keys, "collidedness")

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
        self.p: MatchingConfig = p
        self.matching_templates = matching_templates
        self.matching_templates_builder = matching_templates_builder
        self.thresholdsq: float = self.p.threshold * self.p.threshold
        self.save_collidedness = save_collidedness
        self.whiten_features = whiten_features

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
            panic(p.channel_selection)
        self.register_buffer_or_none("channel_selection_index", channel_selection_index)
        self.is_upsampling = p.up_factor > 1

        # amplitude scaling properties
        self.is_scaling = p.amplitude_scaling_variance > 0
        self.inv_lambda = (
            1.0 / p.amplitude_scaling_variance if self.is_scaling else float("inf")
        )
        self.is_free_scaling = (
            self.inv_lambda == 0 and p.amplitude_scaling_boundary > 1e6
        )
        self.amp_scale_max = 1.0 + p.amplitude_scaling_boundary
        self.amp_scale_min = 1.0 / self.amp_scale_max
        self.whiten_pad = max(0, whiten_kernel_length - 1)
        self.obj_pad_len = self.spike_length_samples + self.whiten_pad
        conv_len = (
            self.chunk_length_samples
            + 2 * self.chunk_margin_samples
            + 2 * self.obj_pad_len
        )
        self.register_buffer("obj_arange", torch.arange(conv_len))

    def peeling_needs_precompute(self):
        return self.matching_templates is None

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
    ) -> Self:
        if motion is not None:
            geom = torch.asarray(motion.geom)
        else:
            geom = torch.tensor(recording.get_channel_locations())
            motion = MotionInfo.from_motion_est(geom=geom.numpy())
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
        if template_data.temporal_kernel is None:
            whiten_kernel_length = 0
        else:
            whiten_kernel_length = template_data.temporal_kernel.shape[0]

        nofeat = featurization_cfg.skip or featurization_cfg.denoise_only
        do_tpca = featurization_cfg.save_input_tpca_projs and not nofeat
        if do_tpca and featurization_cfg.tpca_from_templates:
            from ..transform import TemporalPCAFeaturizer

            (tpca,) = [
                f
                for f in featurization_pipeline
                if isinstance(f, TemporalPCAFeaturizer)
            ]
            tpca.initialize_from_templates(template_data)

        builder = MatchingTemplatesBuilder(
            recording=recording,
            template_data=template_data,
            matching_cfg=matching_cfg,
            motion=motion,
        )

        logger.info(
            "Constructing a matcher with template kind %s, drift %senabled, "
            "scaling variance %s, compression rank %s, upsampling factor %s.",
            matching_cfg.template_type,
            "" if motion.drifting else "not ",
            matching_cfg.amplitude_scaling_variance,
            matching_cfg.template_svd_compression_rank,
            matching_cfg.up_factor,
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
            save_collidedness=save_collidedness,
            whiten_features=matching_cfg.whiten_features,
            whiten_kernel_length=whiten_kernel_length,
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
        return_clean_waveforms=False,
    ) -> PeelingBatchResult:
        assert self.matching_templates is not None
        chunk_center_seconds = self._chunk_center_seconds(chunk_start_samples)
        if self.whiten_features:
            resid_offset = self.whiten_pad
        else:
            resid_offset = 0
        chunk_template_data = self.matching_templates.data_at_time(
            t_s=chunk_center_seconds,
            scaling=self.is_scaling,
            free_scaling=self.is_free_scaling,
            inv_lambda=self.inv_lambda,
            scale_min=self.amp_scale_min,
            scale_max=self.amp_scale_max,
            resid_offset=resid_offset,
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
            return_clean_waveforms=return_clean_waveforms,
        )

        # process spike times and create return result
        if match_results["n_spikes"]:
            match_results["times_samples"] += chunk_start_samples - left_margin
        if match_results["n_spikes"] > self.p.max_spikes_per_second:
            raise ValueError(
                f"Too many spikes {match_results['n_spikes']} > {self.p.max_spikes_per_second}."
            )

        return match_results

    def match_chunk(
        self,
        traces: Tensor,
        chunk_template_data: ChunkTemplateData,
        *,
        left_margin=0,
        right_margin=0,
        return_collisioncleaned_waveforms=True,
        return_clean_waveforms=False,
        return_residual=False,
        return_conv=False,
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
        # padded objective has an extra unit (for group_index)
        valid_len = traces.shape[0] - self.spike_length_samples - self.whiten_pad + 1
        padded_obj_len = valid_len + 2 * self.obj_pad_len + self.whiten_pad
        padded_conv = traces.new_zeros(
            chunk_template_data.obj_n_templates, padded_obj_len
        )
        padded_objective = traces.new_zeros(
            chunk_template_data.obj_n_templates + 1, padded_obj_len
        )

        # initialize convolution
        chunk_template_data.convolve(
            traces_wh, padding=self.obj_pad_len, out=padded_conv
        )

        # main loop
        previous_peaks = current_peaks = None
        prev_update_residual = None
        for cd_it in range(self.p.cd_iter + 1):
            initializing_cd = not cd_it
            coarse_only = self.p.coarse_cd and cd_it < self.p.cd_iter

            # we always need to update the residual in the final iteration
            # in "cd iterations", we may not need to update the residual.
            update_residual = chunk_template_data.needs_residual and not coarse_only

            current_peaks = []
            for _ in range(max_iter):
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

                # find spikes
                new_peaks = self.find_peaks(
                    residual=residual,
                    padded_conv=padded_conv,
                    padded_objective=padded_objective,
                    chunk_template_data=chunk_template_data,
                    coarse_only=coarse_only,
                )
                if new_peaks is None or not new_peaks.n_spikes:
                    break

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
            prev_update_residual = update_residual

        assert current_peaks is not None
        peaks = MatchingPeaks.concatenate(current_peaks)

        # compute the residual now if not done above
        if not chunk_template_data.needs_residual:
            chunk_template_data.subtract(residual_padded, peaks)

        assert residual.shape[0] == traces.shape[0]
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
        if return_clean_waveforms:
            ci_full = full_channel_index(self.b.channel_index.shape[0], True)
            ci_full = ci_full.to(self.b.channel_index)
            res["clean_waveforms"] = chunk_template_data.get_clean_waveforms(
                peaks=peaks,
                channels=channels,
                channel_index=ci_full,
            )
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
        chunk_template_data: ChunkTemplateData,
        coarse_only=False,
    ):
        coarse_peaks = chunk_template_data.quick_match(
            padded_conv=padded_conv,
            padded_objective_buf=padded_objective,
            thresholdsq=self.thresholdsq,
            obj_arange=self.b.obj_arange,
            exclude_extra_padding=self.whiten_pad // 2,
            padding=self.obj_pad_len,
            return_scalings=coarse_only or not self.is_upsampling,
            peak_dt=self.p.peak_dt,
        )
        if coarse_only or not coarse_peaks.n_spikes:
            return coarse_peaks
        return chunk_template_data.fine_match(
            peaks=coarse_peaks,
            residual=residual,
            conv=padded_conv,
            padding=self.obj_pad_len,
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
