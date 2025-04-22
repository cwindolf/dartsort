import tempfile
import warnings
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn.functional as F

from ..detect import (
    detect_and_deduplicate,
    singlechan_template_detect_and_deduplicate,
)
from ..transform import (
    SingleChannelTemplates,
    Voltage,
    Waveform,
    WaveformPipeline,
)
from ..util import peel_util, spiketorch, job_util
from ..util.waveform_util import (
    get_relative_subset,
    make_channel_index,
    relative_channel_subset_index,
)

from .peel_base import BasePeeler
from .threshold import threshold_chunk, ThresholdAndFeaturize


class SubtractionPeeler(BasePeeler):
    peel_kind = "Subtraction"

    def __init__(
        self,
        recording,
        channel_index,
        subtraction_denoising_pipeline,
        featurization_pipeline,
        subtract_channel_index=None,
        trough_offset_samples=42,
        spike_length_samples=121,
        detection_threshold=4,
        chunk_length_samples=30_000,
        peak_sign="both",
        spatial_dedup_channel_index=None,
        n_chunks_fit=40,
        max_waveforms_fit=50_000,
        n_waveforms_fit=20_000,
        fit_subsampling_random_state=0,
        fit_sampling="random",
        fit_max_reweighting=4.0,
        residnorm_decrease_threshold=3.162,
        use_singlechan_templates=False,
        n_singlechan_templates=10,
        singlechan_threshold=40.0,
        singlechan_alignment_padding=20,
        first_denoiser_max_waveforms_fit=250_000,
        first_denoiser_thinning=0.5,
        first_denoiser_temporal_jitter=3,
        first_denoiser_spatial_jitter=35.0,
        dtype=torch.float,
    ):
        super().__init__(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            chunk_length_samples=chunk_length_samples,
            chunk_margin_samples=2 * spike_length_samples,
            n_chunks_fit=n_chunks_fit,
            max_waveforms_fit=max_waveforms_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
            n_waveforms_fit=n_waveforms_fit,
            fit_max_reweighting=fit_max_reweighting,
            fit_sampling=fit_sampling,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            dtype=dtype,
        )

        self.peak_sign = peak_sign
        if subtract_channel_index is None:
            subtract_channel_index = channel_index.clone().detach()
        self.register_buffer("subtract_channel_index", subtract_channel_index)
        self.fixed_output_data.append(
            (
                "subtract_channel_index",
                self.subtract_channel_index.numpy(force=True).copy(),
            ),
        )
        self.extract_subtract_same = torch.equal(
            self.subtract_channel_index, self.channel_index
        )
        if not self.extract_subtract_same:
            esmask = relative_channel_subset_index(
                self.subtract_channel_index, self.channel_index
            )
            self.register_buffer("extract_subtract_mask", esmask)
        else:
            self.extract_subtract_mask = None
        if spatial_dedup_channel_index is not None:
            self.register_buffer(
                "spatial_dedup_channel_index", spatial_dedup_channel_index
            )
        else:
            self.spatial_dedup_channel_index = None
        self.detection_threshold = detection_threshold
        self.residnorm_decrease_threshold = residnorm_decrease_threshold
        self.add_module(
            "subtraction_denoising_pipeline", subtraction_denoising_pipeline
        )

        # first denoiser fitting parameters
        self.first_denoiser_max_waveforms_fit = first_denoiser_max_waveforms_fit
        self.first_denoiser_thinning = first_denoiser_thinning
        self.first_denoiser_temporal_jitter = first_denoiser_temporal_jitter
        self.first_denoiser_spatial_jitter = first_denoiser_spatial_jitter

        # singlechan template based detection
        self.use_singlechan_templates = use_singlechan_templates
        self.have_singlechan_templates = False
        self.singlechan_threshold = singlechan_threshold
        if use_singlechan_templates:
            sc_feat = SingleChannelTemplates(
                self.channel_index,
                n_centroids=n_singlechan_templates,
                alignment_padding=singlechan_alignment_padding,
                trough_offset_samples=self.trough_offset_samples,
                spike_length_samples=self.spike_length_samples,
                max_waveforms=n_waveforms_fit,
            )
            if self.featurization_pipeline is None:
                self.featurization_pipeline = WaveformPipeline([sc_feat])
            else:
                self.featurization_pipeline.transformers.insert(0, sc_feat)

    def out_datasets(self):
        datasets = super().out_datasets()

        # we may be featurizing during subtraction, register the features
        for transformer in self.subtraction_denoising_pipeline.transformers:
            if transformer.is_featurizer:
                datasets.extend(transformer.spike_datasets)

        return datasets

    def peeling_needs_fit(self):
        return self.subtraction_denoising_pipeline.needs_fit()

    def peeling_needs_precompute(self):
        return self.subtraction_denoising_pipeline.needs_precompute()

    def save_models(self, save_folder):
        super().save_models(save_folder)

        sub_denoise_pt = Path(save_folder) / "subtraction_denoising_pipeline.pt"
        torch.save(self.subtraction_denoising_pipeline, sub_denoise_pt)

    def load_models(self, save_folder):
        super().load_models(save_folder)

        sub_denoise_pt = Path(save_folder) / "subtraction_denoising_pipeline.pt"
        if sub_denoise_pt.exists():
            self.subtraction_denoising_pipeline = torch.load(
                sub_denoise_pt, weights_only=True
            )

    @classmethod
    def from_config(
        cls,
        recording,
        waveform_config,
        subtraction_config,
        featurization_config,
    ):
        if subtraction_config.use_universal_templates:
            from . import UniversalTemplatesMatchingPeeler

            return UniversalTemplatesMatchingPeeler.from_config(
                recording,
                waveform_config=waveform_config,
                subtraction_config=subtraction_config,
                featurization_config=featurization_config,
            )

        # waveform extraction channel neighborhoods
        geom = torch.tensor(recording.get_channel_locations())
        channel_index = make_channel_index(
            geom, featurization_config.extract_radius, to_torch=True
        )
        subtract_channel_index = make_channel_index(
            geom, subtraction_config.subtract_radius, to_torch=True
        )
        # per-threshold spike event deduplication channel neighborhoods
        spatial_dedup_channel_index = make_channel_index(
            geom, subtraction_config.spatial_dedup_radius, to_torch=True
        )

        # construct denoising and featurization pipelines
        subtraction_denoising_pipeline = WaveformPipeline.from_config(
            geom,
            subtract_channel_index,
            subtraction_config.subtraction_denoising_config,
            waveform_config,
            sampling_frequency=recording.sampling_frequency,
        )
        featurization_pipeline = WaveformPipeline.from_config(
            geom,
            channel_index,
            featurization_config,
            waveform_config,
            sampling_frequency=recording.sampling_frequency,
        )

        # waveform logic
        trough_offset_samples = waveform_config.trough_offset_samples(
            recording.sampling_frequency
        )
        spike_length_samples = waveform_config.spike_length_samples(
            recording.sampling_frequency
        )

        if trough_offset_samples != 42 or spike_length_samples != 121:
            # temporary warning just so I can see if this happens
            warnings.warn(
                f"waveform_config {trough_offset_samples=} {spike_length_samples=} "
                f"since {recording.sampling_frequency=}"
            )
        singlechan_alignment_padding = int(
            subtraction_config.singlechan_alignment_padding_ms
            * (recording.sampling_frequency / 1000)
        )

        return cls(
            recording,
            channel_index,
            subtraction_denoising_pipeline,
            featurization_pipeline,
            subtract_channel_index=subtract_channel_index,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            detection_threshold=subtraction_config.detection_threshold,
            chunk_length_samples=subtraction_config.chunk_length_samples,
            peak_sign=subtraction_config.peak_sign,
            spatial_dedup_channel_index=spatial_dedup_channel_index,
            n_chunks_fit=subtraction_config.n_chunks_fit,
            max_waveforms_fit=subtraction_config.max_waveforms_fit,
            fit_sampling=subtraction_config.fit_sampling,
            fit_max_reweighting=subtraction_config.fit_max_reweighting,
            n_waveforms_fit=subtraction_config.n_waveforms_fit,
            fit_subsampling_random_state=subtraction_config.fit_subsampling_random_state,
            residnorm_decrease_threshold=subtraction_config.residnorm_decrease_threshold,
            use_singlechan_templates=subtraction_config.use_singlechan_templates,
            n_singlechan_templates=subtraction_config.n_singlechan_templates,
            singlechan_threshold=subtraction_config.singlechan_threshold,
            singlechan_alignment_padding=singlechan_alignment_padding,
        )

    def peel_chunk(
        self,
        traces,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
        return_waveforms=True,
    ):
        extract_index = None if self.extract_subtract_same else self.channel_index
        traces = traces.to(self.dtype)
        if self.have_singlechan_templates:
            sc_feat = self.featurization_pipeline.transformers[0]
            assert isinstance(sc_feat, SingleChannelTemplates)
            singlechan_kw = dict(
                singlechan_templates=sc_feat.templates,
                singlechan_threshold=self.singlechan_threshold,
                singlechan_trough_offset=sc_feat.template_trough,
            )
        else:
            singlechan_kw = {}

        subtraction_result = subtract_chunk(
            traces,
            self.subtract_channel_index,
            self.subtraction_denoising_pipeline,
            extract_index=extract_index,
            extract_mask=self.extract_subtract_mask,
            trough_offset_samples=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
            left_margin=left_margin,
            right_margin=right_margin,
            detection_threshold=self.detection_threshold,
            peak_sign=self.peak_sign,
            spatial_dedup_channel_index=self.spatial_dedup_channel_index,
            residnorm_decrease_threshold=self.residnorm_decrease_threshold,
            **singlechan_kw,
        )

        # add in chunk_start_samples
        times_samples = subtraction_result.times_samples + chunk_start_samples

        peel_result = dict(
            n_spikes=subtraction_result.n_spikes,
            times_samples=times_samples,
            channels=subtraction_result.channels,
            collisioncleaned_waveforms=subtraction_result.collisioncleaned_waveforms,
        )
        peel_result.update(subtraction_result.features)
        if return_residual:
            peel_result["residual"] = subtraction_result.residual

        return peel_result

    def precompute_peeler_models(self):
        self.subtraction_denoising_pipeline.precompute()

    def fit_featurization_pipeline(
        self, save_folder, tmp_dir=None, computation_config=None
    ):
        super().fit_featurization_pipeline(
            save_folder, tmp_dir=tmp_dir, computation_config=computation_config
        )
        if self.use_singlechan_templates:
            self.have_singlechan_templates = True

    def fit_peeler_models(self, save_folder, tmp_dir=None, computation_config=None):
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
            computation_config=computation_config,
            which="denoisers",
        ):
            pass
        self._fit_subtraction_transformers(
            save_folder,
            tmp_dir=tmp_dir,
            computation_config=computation_config,
            which="featurizers",
        )

    def _fit_subtraction_transformers(
        self, save_folder, tmp_dir=None, computation_config=None, which="denoisers"
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

        if computation_config is None:
            computation_config = job_util.get_global_computation_config()
        device = computation_config.actual_device()

        orig_denoise = self.subtraction_denoising_pipeline
        init_voltage_feature = Voltage(
            channel_index=self.subtract_channel_index,
            name="subtract_fit_voltages",
        )
        init_waveform_feature = Waveform(
            channel_index=self.subtract_channel_index,
            name="subtract_fit_waveforms",
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
            already_fitted = [t for t in orig_denoise if t.is_denoiser]

        # if fitting the very first denoiser...
        if which == "denoisers" and not already_fitted:
            fit_pipeline = WaveformPipeline(fit_feats)
            self._threshold_to_fit(
                tmp_dir, fit_pipeline, computation_config=computation_config
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
                    computation_config=computation_config,
                    task_name=f"Load examples for {which[:-1]} fitting",
                )

                # fit featurization pipeline and reassign
                # work in a try finally so we can delete the temp file
                # in case of an issue or a keyboard interrupt
                channels, waveforms, weights = peel_util.subsample_waveforms(
                    temp_hdf5_filename,
                    fit_sampling=self.fit_sampling,
                    random_state=self.fit_subsampling_random_state,
                    n_waveforms_fit=self.n_waveforms_fit,
                    fit_max_reweighting=self.fit_max_reweighting,
                    voltages_dataset_name="subtract_fit_voltages",
                    waveforms_dataset_name="subtract_fit_waveforms",
                )

                channels = torch.as_tensor(channels, device=device)
                waveforms = torch.as_tensor(waveforms, device=device)
                fit_denoise = WaveformPipeline(fit_feats)
                fit_denoise = fit_denoise.to(device)
                fit_denoise.fit(
                    waveforms,
                    max_channels=channels,
                    recording=self.recording,
                    weights=weights,
                )
                fit_denoise = fit_denoise.to("cpu")
                self.subtraction_denoising_pipeline = orig_denoise
                self.featurization_pipeline = orig_featurization_pipeline
            finally:
                self.to("cpu")
                if temp_hdf5_filename.exists():
                    temp_hdf5_filename.unlink()
        return True

    def _threshold_to_fit(self, tmp_dir, fit_pipeline, computation_config):
        geom = self.recording.get_channel_locations()
        spatial_jitter_index = None
        if self.first_denoiser_spatial_jitter:
            spatial_jitter_index = make_channel_index(
                geom, self.first_denoiser_spatial_jitter, to_torch=True
            )
        waveform_pipeline = WaveformPipeline([Waveform(self.subtract_channel_index)])
        full_dedup_channel_index = make_channel_index(geom, 1e10, to_torch=True)
        trainer = ThresholdAndFeaturize(
            self.recording,
            detection_threshold=self.detection_threshold,
            channel_index=self.subtract_channel_index,
            spatial_dedup_channel_index=full_dedup_channel_index,
            featurization_pipeline=waveform_pipeline,
            dedup_temporal_radius_samples=self.spike_length_samples,
            thinning=self.first_denoiser_thinning,
            time_jitter=self.first_denoiser_temporal_jitter,
            spatial_jitter_channel_index=spatial_jitter_index,
            peak_sign=self.peak_sign,
        )

        with tempfile.TemporaryDirectory(dir=tmp_dir) as temp_dir:
            temp_hdf5_filename = Path(temp_dir) / f"subtraction_denoiser0_fit.h5"
            try:
                trainer.peel(
                    temp_hdf5_filename,
                    shuffle=True,
                    stop_after_n_waveforms=self.first_denoiser_max_waveforms_fit,
                    task_name=f"Load examples for initial denoiser fitting",
                    computation_config=computation_config,
                )

                # get fit weights
                channels, waveforms, weights = peel_util.subsample_waveforms(
                    temp_hdf5_filename,
                    fit_sampling=self.fit_sampling,
                    random_state=self.fit_subsampling_random_state,
                    n_waveforms_fit=self.first_denoiser_max_waveforms_fit,
                    fit_max_reweighting=self.fit_max_reweighting,
                    voltages_dataset_name="voltages",
                    waveforms_dataset_name="waveforms",
                    subsample_by_weighting=True,
                )

                # fit the thing
                fit_pipeline.fit(
                    waveforms, channels, recording=self.recording, weights=weights
                )
            finally:
                if temp_hdf5_filename.exists():
                    temp_hdf5_filename.unlink()


ChunkSubtractionResult = namedtuple(
    "ChunkSubtractionResult",
    [
        "n_spikes",
        "times_samples",
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
    extract_index=None,
    extract_mask=None,
    trough_offset_samples=42,
    spike_length_samples=121,
    left_margin=0,
    right_margin=0,
    detection_threshold=4,
    peak_sign="both",
    spatial_dedup_channel_index=None,
    residnorm_decrease_threshold=3.162,  # sqrt(10)
    relative_peak_radius=5,
    dedup_temporal_radius=7,
    singlechan_templates=None,
    singlechan_threshold=None,
    singlechan_trough_offset=None,
    no_subtraction=False,
    max_iter=100,
):
    """Core peeling routine for subtraction"""
    if no_subtraction:
        threshold_res = threshold_chunk(
            traces,
            channel_index,
            detection_threshold=detection_threshold,
            peak_sign=peak_sign,
            spatial_dedup_channel_index=spatial_dedup_channel_index,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            left_margin=left_margin,
            right_margin=right_margin,
            relative_peak_radius=relative_peak_radius,
            dedup_temporal_radius=dedup_temporal_radius,
            max_spikes_per_chunk=None,
            quiet=False,
        )
        waveforms, features = denoising_pipeline(
            threshold_res["waveforms"], threshold_res["channels"]
        )
        return ChunkSubtractionResult(
            n_spikes=threshold_res["n_spikes"],
            times_samples=threshold_res["times_rel"],
            channels=threshold_res["channels"],
            collisioncleaned_waveforms=waveforms,
            residual=None,
            features=features,
        )

    # validate arguments to avoid confusing error messages later
    re_extract = extract_index is not None
    if extract_index is None:
        extract_index = channel_index
    else:
        assert extract_mask is not None
    assert 0 <= left_margin < traces.shape[0]
    assert 0 <= right_margin < traces.shape[0]
    assert traces.shape[1] == channel_index.shape[0]
    if spatial_dedup_channel_index is not None:
        assert traces.shape[1] == spatial_dedup_channel_index.shape[0]

    # can only subtract spikes with trough time >=trough_offset and <max_trough
    traces_length_samples = traces.shape[0]
    post_trough_samples = spike_length_samples - trough_offset_samples
    max_trough_time = traces_length_samples - post_trough_samples

    # resid norm decrease logic
    check_resid = bool(residnorm_decrease_threshold)
    residnormsq_thresh = residnorm_decrease_threshold**2

    # initialize residual, it needs to be padded to support
    # our channel indexing convention. this copies the input.
    residual = F.pad(traces, (0, 1), value=torch.nan)

    subtracted_waveforms = []
    spike_times = []
    spike_channels = []
    spike_features = []
    detection_mask = torch.ones_like(residual)
    dedup_temporal_ix = torch.arange(
        -dedup_temporal_radius, dedup_temporal_radius, device=residual.device
    )

    for _ in range(max_iter):
        if singlechan_templates is None:
            times_samples, channels = detect_and_deduplicate(
                residual[:, :-1],
                detection_threshold,
                dedup_channel_index=channel_index,
                peak_sign=peak_sign,
                detection_mask=detection_mask[:, :-1],
                relative_peak_radius=relative_peak_radius,
                dedup_temporal_radius=spike_length_samples,
            )
        else:
            times_samples, channels = singlechan_template_detect_and_deduplicate(
                residual[:, :-1],
                singlechan_templates,
                threshold=singlechan_threshold,
                trough_offset_samples=singlechan_trough_offset,
                dedup_channel_index=channel_index,
                relative_peak_radius=relative_peak_radius,
                dedup_temporal_radius=spike_length_samples,
                detection_mask=detection_mask[:, :-1],
            )
        if not times_samples.numel():
            break

        # throw away spikes which cannot be subtracted
        keep = torch.logical_and(
            times_samples >= trough_offset_samples, times_samples < max_trough_time
        )
        (keep,) = keep.nonzero(as_tuple=True)

        if not keep.numel():
            break
        times_samples = times_samples[keep]
        channels = channels[keep]

        # never look at these again.
        time_ix = times_samples.unsqueeze(1) + dedup_temporal_ix.unsqueeze(0)
        if spatial_dedup_channel_index is not None:
            chan_ix = spatial_dedup_channel_index[channels]
        else:
            chan_ix = channels.unsqueeze(1)
        detection_mask[time_ix[:, :, None], chan_ix[:, None, :]] = 0.0

        # read waveforms, denoise, and test residnorm decrease
        waveforms = spiketorch.grab_spikes(
            residual,
            times_samples,
            channels,
            channel_index,
            trough_offset=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            buffer=0,
            already_padded=True,
        )

        if check_resid:
            residuals = torch.nan_to_num(waveforms)
        waveforms, features = denoising_pipeline(waveforms, channels)

        # test residual norm decrease
        if check_resid:
            orig_normsq = residuals.square().sum(dim=(1, 2))
            residuals = residuals.sub_(waveforms).nan_to_num_()
            new_normsq = residuals.square_().sum(dim=(1, 2))
            keep = new_normsq < orig_normsq.sub_(residnormsq_thresh)
            (keep,) = keep.nonzero(as_tuple=True)
            if not keep.numel():
                break
            if keep.numel() < new_normsq.numel():
                waveforms = waveforms[keep]
                times_samples = times_samples[keep]
                channels = channels[keep]
                for k in features:
                    features[k] = features[k][keep]

        # store this iter's outputs
        spike_times.append(times_samples)
        spike_channels.append(channels)
        spike_features.append(features)
        subtracted_waveforms.append(waveforms)

        # -- subtract in place
        spiketorch.subtract_spikes_(
            residual,
            times_samples,
            channels,
            channel_index,
            waveforms,
            trough_offset=trough_offset_samples,
            buffer=0,
            already_padded=True,
            in_place=True,
        )

    # check if we got no spikes
    if not spike_times:
        return empty_chunk_subtraction_result(
            spike_length_samples,
            channel_index,
            residual[left_margin : traces.shape[0] - right_margin, :-1],
        )

    # concatenate all of the thresholds together into single tensors
    spike_times = [t.cpu() for t in spike_times]
    spike_channels = [t.cpu() for t in spike_channels]
    spike_times = torch.concatenate(spike_times)
    spike_channels = torch.concatenate(spike_channels)
    subtracted_waveforms = torch.concatenate(subtracted_waveforms)
    spike_features_list = spike_features
    spike_features = {}
    feature_keys = list(spike_features_list[0].keys())
    for k in feature_keys:
        this_feature_list = []
        for f in spike_features_list:
            this_feature_list.append(f[k].cpu())
            del f[k]
        spike_features[k] = torch.concatenate(this_feature_list)
        del this_feature_list
    del spike_features_list

    # discard spikes in the margins and sort times_samples for caller
    keep = torch.logical_and(
        spike_times >= left_margin, spike_times < traces.shape[0] - right_margin
    )
    (keep,) = keep.cpu().nonzero(as_tuple=True)
    if not keep.numel():
        return empty_chunk_subtraction_result(
            spike_length_samples,
            channel_index,
            residual[left_margin : traces.shape[0] - right_margin, :-1],
        )

    keep = keep[torch.argsort(spike_times[keep])]
    subtracted_waveforms = subtracted_waveforms[keep]
    spike_times = spike_times[keep]
    spike_channels = spike_channels[keep]
    for k in spike_features:
        spike_features[k] = spike_features[k][keep]

    # if extract_index != subtract_index, re-do the channels for the subtracted wfs
    if re_extract:
        subtracted_waveforms = get_relative_subset(
            subtracted_waveforms, spike_channels, extract_mask
        )

    # construct collision-cleaned waveforms
    collisioncleaned_waveforms = spiketorch.grab_spikes(
        residual,
        spike_times,
        spike_channels,
        extract_index,
        trough_offset=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        buffer=0,
        already_padded=True,
    )
    collisioncleaned_waveforms += subtracted_waveforms

    # offset spike times_samples according to margin
    spike_times -= left_margin

    # strip margin and padding channel off the residual
    residual = residual[left_margin : traces.shape[0] - right_margin, :-1].cpu()

    return ChunkSubtractionResult(
        n_spikes=spike_times.numel(),
        times_samples=spike_times,
        channels=spike_channels,
        collisioncleaned_waveforms=collisioncleaned_waveforms,
        residual=residual,
        features=spike_features,
    )


def subtract_chunk_old(
    traces,
    channel_index,
    denoising_pipeline,
    extract_index=None,
    extract_mask=None,
    trough_offset_samples=42,
    spike_length_samples=121,
    left_margin=0,
    right_margin=0,
    detection_thresholds=[12, 10, 8, 6, 5, 4],
    peak_sign="both",
    spatial_dedup_channel_index=None,
    residnorm_decrease_threshold=3.162,  # sqrt(10)
    persist_deduplication=True,
    relative_peak_radius=5,
    dedup_temporal_radius=7,
    no_subtraction=False,
):
    """Core peeling routine for subtraction"""
    if no_subtraction:
        threshold_res = threshold_chunk(
            traces,
            channel_index,
            detection_threshold=min(detection_thresholds),
            peak_sign=peak_sign,
            spatial_dedup_channel_index=spatial_dedup_channel_index,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            left_margin=left_margin,
            right_margin=right_margin,
            relative_peak_radius=relative_peak_radius,
            dedup_temporal_radius=dedup_temporal_radius,
            max_spikes_per_chunk=None,
            quiet=False,
        )
        waveforms, features = denoising_pipeline(
            threshold_res["waveforms"], threshold_res["channels"]
        )
        return ChunkSubtractionResult(
            n_spikes=threshold_res["n_spikes"],
            times_samples=threshold_res["times_rel"],
            channels=threshold_res["channels"],
            collisioncleaned_waveforms=waveforms,
            residual=None,
            features=features,
        )

    # validate arguments to avoid confusing error messages later
    re_extract = extract_index is not None
    if extract_index is None:
        extract_index = channel_index
    else:
        assert extract_mask is not None
    assert 0 <= left_margin < traces.shape[0]
    assert 0 <= right_margin < traces.shape[0]
    assert traces.shape[1] == channel_index.shape[0]
    if spatial_dedup_channel_index is not None:
        assert traces.shape[1] == spatial_dedup_channel_index.shape[0]
    assert all(
        detection_thresholds[i] > detection_thresholds[i + 1]
        for i in range(len(detection_thresholds) - 1)
    )

    # can only subtract spikes with trough time >=trough_offset and <max_trough
    traces_length_samples = traces.shape[0]
    post_trough_samples = spike_length_samples - trough_offset_samples
    max_trough_time = traces_length_samples - post_trough_samples

    # initialize residual, it needs to be padded to support
    # our channel indexing convention. this copies the input.
    residual = F.pad(traces, (0, 1), value=torch.nan)

    subtracted_waveforms = []
    spike_times = []
    spike_channels = []
    spike_features = []
    if persist_deduplication:
        detection_mask = torch.ones_like(residual)
        dedup_temporal_ix = torch.arange(
            -dedup_temporal_radius, dedup_temporal_radius, device=residual.device
        )

    for j, threshold in enumerate(detection_thresholds):
        # -- detect and extract waveforms
        # detection has more args which we don't expose right now
        step_mask = None
        if persist_deduplication and j > 0:
            step_mask = detection_mask[:, :-1]
        times_samples, channels = detect_and_deduplicate(
            residual[:, :-1],
            threshold,
            dedup_channel_index=spatial_dedup_channel_index,
            peak_sign=peak_sign,
            detection_mask=step_mask,
            relative_peak_radius=relative_peak_radius,
            dedup_temporal_radius=dedup_temporal_radius,
        )
        if not times_samples.numel():
            continue

        # throw away spikes which cannot be subtracted
        keep = torch.logical_and(
            times_samples >= trough_offset_samples, times_samples < max_trough_time
        )
        times_samples = times_samples[keep]
        if not times_samples.numel():
            continue
        channels = channels[keep]

        # read waveforms, denoise, and test residnorm decrease
        waveforms = spiketorch.grab_spikes(
            residual,
            times_samples,
            channels,
            channel_index,
            trough_offset=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            buffer=0,
            already_padded=True,
        )

        if residnorm_decrease_threshold:
            residuals = torch.nan_to_num(waveforms)
        waveforms, features = denoising_pipeline(waveforms, channels)

        # test residual norm decrease
        if residnorm_decrease_threshold:
            orig_norm = torch.linalg.norm(residuals, dim=(1, 2))
            residuals -= torch.nan_to_num(waveforms)
            sub_norm = torch.linalg.norm(residuals, dim=(1, 2))
            keep = sub_norm < orig_norm - residnorm_decrease_threshold
            waveforms = waveforms[keep]
            times_samples = times_samples[keep]
            channels = channels[keep]
            for k in features:
                features[k] = features[k][keep]
            if not times_samples.numel():
                continue

        # store this threshold's outputs
        spike_times.append(times_samples)
        spike_channels.append(channels)
        subtracted_waveforms.append(waveforms)
        spike_features.append(features)

        # -- subtract in place
        spiketorch.subtract_spikes_(
            residual,
            times_samples,
            channels,
            channel_index,
            waveforms,
            trough_offset=trough_offset_samples,
            buffer=0,
            already_padded=True,
            in_place=True,
        )
        if persist_deduplication:
            time_ix = times_samples.unsqueeze(1) + dedup_temporal_ix.unsqueeze(0)
            if spatial_dedup_channel_index is not None:
                chan_ix = spatial_dedup_channel_index[channels]
            else:
                chan_ix = channels.unsqueeze(1)
            detection_mask[time_ix[:, :, None], chan_ix[:, None, :]] = 0.0
        del times_samples, channels, waveforms, features

    # check if we got no spikes
    if not any(t.numel() for t in spike_times):
        return empty_chunk_subtraction_result(
            spike_length_samples,
            channel_index,
            residual[left_margin : traces.shape[0] - right_margin, :-1],
        )

    # concatenate all of the thresholds together into single tensors
    subtracted_waveforms = torch.concatenate(subtracted_waveforms)
    spike_times = torch.concatenate(spike_times)
    spike_channels = torch.concatenate(spike_channels)
    spike_features_list = spike_features
    spike_features = {}
    feature_keys = list(spike_features_list[0].keys())
    for k in feature_keys:
        this_feature_list = []
        for f in spike_features_list:
            this_feature_list.append(f[k])
            del f[k]
        spike_features[k] = torch.concatenate(this_feature_list)
        del this_feature_list
    del spike_features_list

    # discard spikes in the margins and sort times_samples for caller
    keep = torch.nonzero(
        (spike_times >= left_margin) & (spike_times < traces.shape[0] - right_margin)
    )[:, 0]
    if not keep.any():
        return empty_chunk_subtraction_result(
            spike_length_samples,
            channel_index,
            residual[left_margin : traces.shape[0] - right_margin, :-1],
        )
    keep = keep[torch.argsort(spike_times[keep])]
    subtracted_waveforms = subtracted_waveforms[keep]
    spike_times = spike_times[keep]
    spike_channels = spike_channels[keep]
    for k in spike_features:
        spike_features[k] = spike_features[k][keep]

    # if extract_index != subtract_index, re-do the channels for the subtracted wfs
    if re_extract:
        subtracted_waveforms = get_relative_subset(
            subtracted_waveforms, spike_channels, extract_mask
        )

    # construct collision-cleaned waveforms
    collisioncleaned_waveforms = spiketorch.grab_spikes(
        residual,
        spike_times,
        spike_channels,
        extract_index,
        trough_offset=trough_offset_samples,
        spike_length_samples=spike_length_samples,
        buffer=0,
        already_padded=True,
    )
    collisioncleaned_waveforms += subtracted_waveforms

    # offset spike times_samples according to margin
    spike_times -= left_margin

    # strip margin and padding channel off the residual
    residual = residual[left_margin : traces.shape[0] - right_margin, :-1]

    return ChunkSubtractionResult(
        n_spikes=spike_times.numel(),
        times_samples=spike_times,
        channels=spike_channels,
        collisioncleaned_waveforms=collisioncleaned_waveforms,
        residual=residual,
        features=spike_features,
    )


def empty_chunk_subtraction_result(spike_length_samples, channel_index, residual):
    empty_waveforms = torch.empty(
        (0, spike_length_samples, channel_index.shape[1]),
        dtype=residual.dtype,
    )
    empty_times_or_chans = torch.empty((0,), dtype=torch.long)
    return ChunkSubtractionResult(
        n_spikes=0,
        times_samples=empty_times_or_chans,
        channels=empty_times_or_chans,
        collisioncleaned_waveforms=empty_waveforms,
        residual=residual,
        features={},
    )
