import gc
import tempfile
from pathlib import Path

import numpy as np
import torch
from spikeinterface.core import BaseRecording

from ..transform import Voltage, Waveform, WaveformPipeline
from ..util import job_util
from ..util.data_util import SpikeDataset, subsample_waveforms
from ..util.internal_config import (
    FeaturizationConfig,
    FitSamplingConfig,
    SubtractionConfig,
    ThresholdingConfig,
    WaveformConfig,
    default_peeling_fit_sampling_cfg,
    default_subtraction_cfg,
    default_waveform_cfg,
)
from ..util.waveform_util import (
    get_channel_index_rel_inds,
    make_channel_index,
    relative_channel_subset_index,
)
from .peel_base import BasePeeler, PeelingBatchResult
from .peel_lib import subtract_chunk, threshold_to_fit


class SubtractionPeeler(BasePeeler):
    peel_kind = "Subtraction"

    def __init__(
        self,
        recording: BaseRecording,
        channel_index: np.ndarray | torch.Tensor,
        subtraction_denoising_pipeline: WaveformPipeline,
        featurization_pipeline: WaveformPipeline | None,
        p: SubtractionConfig = default_subtraction_cfg,
        waveform_cfg: WaveformConfig = default_waveform_cfg,
        fit_sampling_cfg: FitSamplingConfig = default_peeling_fit_sampling_cfg,
        save_iteration=False,
        save_residnorm_decrease=False,
        save_collidedness=False,
        dtype=torch.float,
    ):
        fixed_property_keys = ("channels", "times_seconds")
        if p.realign_to_denoiser:
            fixed_property_keys = fixed_property_keys + ("time_shifts",)
        if save_collidedness:
            fixed_property_keys = fixed_property_keys + ("collidedness",)
        spike_length_samples = waveform_cfg.spike_length_samples(
            recording.sampling_frequency
        )
        super().__init__(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            chunk_length_samples=p.chunk_length_samples,
            chunk_margin_samples=self.next_margin(2 * spike_length_samples),
            fit_sampling_cfg=fit_sampling_cfg,
            waveform_cfg=waveform_cfg,
            fixed_property_keys=fixed_property_keys,
            dtype=dtype,
        )
        self.p = p
        self.save_iteration = save_iteration
        self.save_residnorm_decrease = save_residnorm_decrease
        self.save_collidedness = save_collidedness
        self.dedup_batch_size = self.nearest_batch_length()

        geom = recording.get_channel_locations()
        sub_channel_index = make_channel_index(
            geom, p.subtract_radius_um, to_torch=True
        )
        self.register_buffer("sub_channel_index", sub_channel_index)
        self.register_buffer(
            "subtract_index_rel_inds", get_channel_index_rel_inds(sub_channel_index)
        )
        self.fixed_output_data.append(
            (
                "sub_channel_index",
                self.b.sub_channel_index.numpy(force=True).copy(),
            ),
        )
        self.extract_subtract_same = torch.equal(
            self.b.sub_channel_index, self.b.channel_index
        )
        if not self.extract_subtract_same:
            esmask = relative_channel_subset_index(
                self.sub_channel_index, self.channel_index
            )
            self.register_buffer("extract_subtract_mask", torch.asarray(esmask))
        else:
            self.extract_subtract_mask = None
        if p.spatial_dedup_radius_um:
            dedup_channel_index = make_channel_index(
                geom, p.spatial_dedup_radius_um, to_torch=True
            )
        else:
            dedup_channel_index = None
        self.register_buffer_or_none("dedup_channel_index", dedup_channel_index)
        if p.relative_peak_radius_um:
            peak_channel_index = make_channel_index(
                geom, p.relative_peak_radius_um, to_torch=True
            )
        else:
            peak_channel_index = None
        self.register_buffer_or_none("peak_channel_index", peak_channel_index)
        if dedup_channel_index is not None:
            dedup_rel_inds = get_channel_index_rel_inds(dedup_channel_index)
        else:
            dedup_rel_inds = None
        self.register_buffer_or_none("dedup_rel_inds", dedup_rel_inds)
        self.add_module(
            "subtraction_denoising_pipeline", subtraction_denoising_pipeline
        )

        # first denoiser fitting parameters
        _p = 1.0 - p.first_denoiser_thinning
        can_thin = recording.get_total_duration() > fit_sampling_cfg.n_seconds_fit / _p
        self.first_denoiser_thinning = p.first_denoiser_thinning if can_thin else 0.0

    def out_datasets(self):
        datasets = super().out_datasets()

        if self.save_iteration:
            datasets.append(SpikeDataset("iteration", (), "int16"))
        if self.p.realign_to_denoiser:
            datasets.append(SpikeDataset("time_shifts", (), "int16"))
        if self.save_residnorm_decrease:
            datasets.append(SpikeDataset("residnorm_decreases", (), "float32"))
        if self.save_collidedness:
            datasets.append(SpikeDataset("collidedness", (), "float32"))

        # we may be featurizing during subtraction, register the features
        datasets.extend(self.subtraction_denoising_pipeline.spike_datasets())

        return datasets

    def peeling_needs_fit(self):
        return self.subtraction_denoising_pipeline.needs_fit()

    def peeling_needs_precompute(self):
        return self.subtraction_denoising_pipeline.needs_precompute()

    def save_models(self, save_folder):
        super().save_models(save_folder)
        sub_denoise_pt = Path(save_folder) / "subtraction_denoising_pipeline.pt"
        torch.save(self.subtraction_denoising_pipeline.state_dict(), sub_denoise_pt)

    def load_models(self, save_folder):
        super().load_models(save_folder)
        sub_denoise_pt = Path(save_folder) / "subtraction_denoising_pipeline.pt"
        if sub_denoise_pt.exists():
            state_dict = torch.load(sub_denoise_pt)
            self.subtraction_denoising_pipeline.load_state_dict(state_dict)

    @classmethod
    def from_config(
        cls,
        *,
        recording: BaseRecording,
        waveform_cfg: WaveformConfig,
        subtraction_cfg: SubtractionConfig,
        featurization_cfg: FeaturizationConfig,
        sampling_cfg: FitSamplingConfig,
    ):
        # waveform extraction channel neighborhoods
        geom = torch.tensor(recording.get_channel_locations())
        channel_index = make_channel_index(
            geom, featurization_cfg.extract_radius, to_torch=True
        )
        sub_channel_index = make_channel_index(
            geom, subtraction_cfg.subtract_radius_um, to_torch=True
        )

        # construct denoising and featurization pipelines
        subtraction_denoising_pipeline = WaveformPipeline.from_config(
            geom=geom,
            channel_index=sub_channel_index,
            featurization_cfg=subtraction_cfg.subtraction_denoising_cfg,
            waveform_cfg=waveform_cfg,
            sampling_frequency=recording.sampling_frequency,
        )
        featurization_pipeline = WaveformPipeline.from_config(
            geom=geom,
            channel_index=channel_index,
            featurization_cfg=featurization_cfg,
            waveform_cfg=waveform_cfg,
            sampling_frequency=recording.sampling_frequency,
        )
        save_collidedness = (
            featurization_cfg.save_collidedness and not featurization_cfg.skip
        )

        return cls(
            recording=recording,
            channel_index=channel_index,
            subtraction_denoising_pipeline=subtraction_denoising_pipeline,
            featurization_pipeline=featurization_pipeline,
            p=subtraction_cfg,
            waveform_cfg=waveform_cfg,
            fit_sampling_cfg=sampling_cfg,
            save_iteration=subtraction_cfg.save_iteration,
            save_residnorm_decrease=subtraction_cfg.save_residnorm_decrease,
            save_collidedness=save_collidedness,
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
    ):
        del return_waveforms  # always done here

        extract_index = None if self.extract_subtract_same else self.channel_index
        traces = traces.to(self.dtype)

        subtraction_result = subtract_chunk(
            traces,
            self.sub_channel_index,
            self.subtraction_denoising_pipeline,
            extract_index=extract_index,
            extract_mask=self.extract_subtract_mask,
            trough_offset_samples=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
            left_margin=left_margin,
            right_margin=right_margin,
            detection_threshold=self.p.detection_threshold,
            relative_peak_radius=self.p.relative_peak_radius_samples,
            peak_sign=self.p.peak_sign,
            peak_channel_index=self.b.peak_channel_index,
            dedup_channel_index=self.b.dedup_channel_index,
            dedup_batch_size=self.dedup_batch_size,
            dedup_temporal_radius=self.p.temporal_dedup_radius_samples,
            remove_exact_duplicates=self.p.remove_exact_duplicates,
            pos_dedup_temporal_radius=self.p.positive_temporal_dedup_radius_samples,
            residnorm_decrease_threshold=self.p.residnorm_decrease_threshold,
            decrease_objective=self.p.decrease_objective,
            trough_priority=self.p.trough_priority,
            growth_tolerance=self.p.growth_tolerance,
            cumulant_order=self.p.cumulant_order,
            convexity_threshold=self.p.convexity_threshold,
            convexity_radius=self.p.convexity_radius,
            save_iteration=self.save_iteration,
            save_residnorm_decrease=self.save_residnorm_decrease,
            max_iter=self.p.max_iter,
            subtract_rel_inds=self.subtract_index_rel_inds,
            dedup_rel_inds=self.dedup_rel_inds,
            realign_to_denoiser=self.p.realign_to_denoiser,
            denoiser_realignment_shift=self.p.denoiser_realignment_shift,
            denoiser_realignment_channel=self.p.denoiser_realignment_channel,
            compute_collidedness=self.save_collidedness,
        )

        # add in chunk_start_samples
        times_samples = subtraction_result.times_samples + chunk_start_samples

        peel_result = PeelingBatchResult(
            n_spikes=subtraction_result.n_spikes,
            times_samples=times_samples,
            channels=subtraction_result.channels,
            collisioncleaned_waveforms=subtraction_result.collisioncleaned_waveforms,
        )
        peel_result.update(subtraction_result.features)
        if return_residual:
            peel_result["residual"] = subtraction_result.residual

        return peel_result

    def precompute_peeler_models(
        self, save_folder, overwrite=False, computation_cfg=None
    ):
        self.subtraction_denoising_pipeline.precompute()

    def fit_peeler_models(self, save_folder, tmp_dir=None, computation_cfg=None):
        # when fitting peelers for subtraction, there are basically
        # two cases. fitting featurizers is easy -- they don't modify
        # the waveforms. fitting denoisers is hard -- they do. each
        # denoiser that needs fitting will affect any transformer
        # that comes after it in the pipeline.

        # but, we usually only have one denoiser that needs fitting
        # in the pipeline, which is temporal PCA (after pretrained NN)
        # so we will cheat for now:
        # just remove all the denoisers that need fitting, run peeling,
        # and fit everything

        while self._fit_subtraction_transformers(
            save_folder,
            tmp_dir=tmp_dir,
            computation_cfg=computation_cfg,
            which="denoisers",
        ):
            pass
        self._fit_subtraction_transformers(
            save_folder,
            tmp_dir=tmp_dir,
            computation_cfg=computation_cfg,
            which="featurizers",
        )

        gc.collect()
        torch.cuda.empty_cache()

    def _fit_subtraction_transformers(
        self, save_folder, tmp_dir=None, computation_cfg=None, which="denoisers"
    ):
        """Fit models which are run during the subtraction step

        These include denoisers and featurizers. Featurizers are easy, we just fit them
        to the extracted waveforms output from a mini-subtraction. Denoisers are a bit
        harder, since they influence the actual waveforms that are extracted. In that sense,
        you need to fit them serially with a new mini subtraction each time.
        """
        if which == "denoisers":
            needs_fit = any(
                t.is_denoiser and t.needs_fit()
                for t in self.subtraction_denoising_pipeline
            )
        elif which == "featurizers":
            assert not any(
                t.is_denoiser and t.needs_fit()
                for t in self.subtraction_denoising_pipeline
            )
            needs_fit = any(
                t.is_featurizer and t.needs_fit()
                for t in self.subtraction_denoising_pipeline
            )
        else:
            assert False

        if not needs_fit:
            return False

        if computation_cfg is None:
            computation_cfg = job_util.get_global_computation_config()
        device = computation_cfg.actual_device()

        orig_denoise = self.subtraction_denoising_pipeline
        init_voltage_feature = Voltage(
            channel_index=self.sub_channel_index,
            name="subtract_fit_voltages",
            waveform_cfg=self.waveform_cfg,
            sampling_frequency=self.recording.sampling_frequency,
        )
        init_waveform_feature = Waveform(
            channel_index=self.sub_channel_index,
            name="subtract_fit_waveforms",
            waveform_cfg=self.waveform_cfg,
            sampling_frequency=self.recording.sampling_frequency,
        )
        ifeats = [init_voltage_feature, init_waveform_feature]
        if which == "denoisers":
            # add all the already fitted denoisers until we hit the next unfitted one
            already_fitted = []
            fit_feats = []
            for t in orig_denoise:
                if t.is_denoiser:
                    if t.needs_fit():
                        fit_feats = [t]
                        break
                    already_fitted.append(t)

            # this is the sequence of transforms to actually use in fitting
            fit_feats = already_fitted + fit_feats
        else:
            fit_feats = None
            already_fitted = [t for t in orig_denoise if t.is_denoiser]

        # if fitting the very first denoiser...
        if which == "denoisers" and not already_fitted:
            assert fit_feats is not None
            fit_pipeline = WaveformPipeline(
                fit_feats,
                waveform_cfg=self.waveform_cfg,
                sampling_frequency=self.recording.sampling_frequency,
            )
            self._threshold_to_fit(
                tmp_dir, fit_pipeline, computation_cfg=computation_cfg
            )
            return True

        self.subtraction_denoising_pipeline = WaveformPipeline(ifeats + already_fitted)

        # and we don't need any features for this
        orig_featurization_pipeline = self.featurization_pipeline
        self.featurization_pipeline = WaveformPipeline([])

        # run mini subtraction
        with tempfile.TemporaryDirectory(dir=tmp_dir) as temp_dir:
            temp_hdf5_filename = Path(temp_dir) / f"subtraction_{which[:-1]}_fit.h5"
            try:
                self.run_subsampled_peeling(
                    temp_hdf5_filename,
                    computation_cfg=computation_cfg,
                    task_name=f"Load examples for {which[:-1]} fitting",
                )

                # fit featurization pipeline and reassign
                # work in a try finally so we can delete the temp file
                # in case of an issue or a keyboard interrupt
                waveforms, fixed_properties = subsample_waveforms(
                    temp_hdf5_filename,
                    fit_sampling=self.fit_sampling_cfg.fit_sampling,
                    random_state=self.fit_subsampling_random_state,
                    n_waveforms_fit=self.fit_sampling_cfg.n_waveforms_fit,
                    fit_max_reweighting=self.fit_sampling_cfg.fit_max_reweighting,
                    voltages_dataset_name="subtract_fit_voltages",
                    waveforms_dataset_name="subtract_fit_waveforms",
                    device="cpu" if which == "denoisers" else device,
                )
                # these are on CPU for now.
                assert fit_feats is not None
                fit_denoise = WaveformPipeline(
                    fit_feats,
                    waveform_cfg=self.waveform_cfg,
                    sampling_frequency=self.recording.sampling_frequency,
                )
                fit_denoise = fit_denoise.to(device)
                fit_denoise.fit(
                    recording=self.recording,
                    waveforms=waveforms,
                    computation_cfg=computation_cfg,
                    hdf5_filename=temp_hdf5_filename,
                    waveforms_dataset_name="subtract_fit_waveforms",
                    **fixed_properties,
                )
                fit_denoise = fit_denoise.to("cpu")
                self.subtraction_denoising_pipeline = orig_denoise
                self.featurization_pipeline = orig_featurization_pipeline
            finally:
                self.to("cpu")
                if temp_hdf5_filename.exists():
                    temp_hdf5_filename.unlink()
        return True

    def _threshold_to_fit(self, tmp_dir, fit_pipeline, computation_cfg):
        threshold_cfg = ThresholdingConfig(
            detection_threshold=self.p.detection_threshold,
            peak_sign=self.p.peak_sign,
            relative_peak_radius_um=self.p.relative_peak_radius_um,
            relative_peak_radius_samples=self.p.relative_peak_radius_samples,
            temporal_dedup_radius_samples=self.spike_length_samples,
            time_jitter=self.p.first_denoiser_temporal_jitter,
            thinning=self.first_denoiser_thinning,
            cumulant_order=self.p.cumulant_order,
            remove_exact_duplicates=self.p.remove_exact_duplicates,
            convexity_threshold=self.p.convexity_threshold,
            trough_priority=self.p.trough_priority,
            convexity_radius=self.p.convexity_radius,
            spatial_jitter_radius=self.p.first_denoiser_spatial_jitter,
        )
        return threshold_to_fit(
            pipeline=fit_pipeline,
            recording=self.recording,
            waveform_cfg=self.waveform_cfg,
            channel_index=self.b.sub_channel_index,
            spatial_dedup_radius=self.p.first_denoiser_spatial_dedup_radius,
            threshold_cfg=threshold_cfg,
            sampling_cfg=self.fit_sampling_cfg,
            max_waveforms_fit=self.p.first_denoiser_max_waveforms_fit,
            computation_cfg=computation_cfg,
            tmp_dir=tmp_dir,
        )
