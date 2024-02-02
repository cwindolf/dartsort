import warnings
from collections import namedtuple
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
from dartsort.detect import detect_and_deduplicate
from dartsort.transform import Waveform, WaveformPipeline
from dartsort.util import spiketorch
from dartsort.util.waveform_util import make_channel_index

from .peel_base import BasePeeler


class SubtractionPeeler(BasePeeler):
    peel_kind = "Subtraction"

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
        n_chunks_fit=40,
        fit_subsampling_random_state=0,
        residnorm_decrease_threshold=3.162,
    ):
        super().__init__(
            recording=recording,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            chunk_length_samples=chunk_length_samples,
            chunk_margin_samples=2 * spike_length_samples,
            n_chunks_fit=n_chunks_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
        )

        self.trough_offset_samples = trough_offset_samples
        self.spike_length_samples = spike_length_samples
        self.peak_sign = peak_sign
        if spatial_dedup_channel_index is not None:
            self.register_buffer(
                "spatial_dedup_channel_index",
                spatial_dedup_channel_index,
            )
        else:
            self.spatial_dedup_channel_index = None
        self.detection_thresholds = detection_thresholds
        self.residnorm_decrease_threshold = residnorm_decrease_threshold

        self.add_module(
            "subtraction_denoising_pipeline", subtraction_denoising_pipeline
        )

    def out_datasets(self):
        datasets = super().out_datasets()

        # we may be featurizing during subtraction, register the features
        for transformer in self.subtraction_denoising_pipeline.transformers:
            if transformer.is_featurizer:
                datasets.append(transformer.spike_dataset)

        return datasets

    def peeling_needs_fit(self):
        return self.subtraction_denoising_pipeline.needs_fit()

    def save_models(self, save_folder):
        super().save_models(save_folder)

        sub_denoise_pt = Path(save_folder) / "subtraction_denoising_pipeline.pt"
        torch.save(self.subtraction_denoising_pipeline, sub_denoise_pt)

    def load_models(self, save_folder):
        super().load_models(save_folder)

        sub_denoise_pt = Path(save_folder) / "subtraction_denoising_pipeline.pt"
        if sub_denoise_pt.exists():
            self.subtraction_denoising_pipeline = torch.load(sub_denoise_pt)

    @classmethod
    def from_config(
        cls, recording, waveform_config, subtraction_config, featurization_config
    ):
        # waveform extraction channel neighborhoods
        geom = torch.tensor(recording.get_channel_locations())
        channel_index = make_channel_index(
            geom, subtraction_config.extract_radius, to_torch=True
        )
        # per-threshold spike event deduplication channel neighborhoods
        spatial_dedup_channel_index = make_channel_index(
            geom, subtraction_config.spatial_dedup_radius, to_torch=True
        )

        # construct denoising and featurization pipelines
        subtraction_denoising_pipeline = WaveformPipeline.from_config(
            geom,
            channel_index,
            subtraction_config.subtraction_denoising_config,
        )
        featurization_pipeline = WaveformPipeline.from_config(
            geom, channel_index, featurization_config
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

        return cls(
            recording,
            channel_index,
            subtraction_denoising_pipeline,
            featurization_pipeline,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            detection_thresholds=subtraction_config.detection_thresholds,
            chunk_length_samples=subtraction_config.chunk_length_samples,
            peak_sign=subtraction_config.peak_sign,
            spatial_dedup_channel_index=spatial_dedup_channel_index,
            n_chunks_fit=subtraction_config.n_chunks_fit,
            fit_subsampling_random_state=subtraction_config.fit_subsampling_random_state,
            residnorm_decrease_threshold=subtraction_config.residnorm_decrease_threshold,
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
            left_margin=left_margin,
            right_margin=right_margin,
            detection_thresholds=self.detection_thresholds,
            peak_sign=self.peak_sign,
            spatial_dedup_channel_index=self.spatial_dedup_channel_index,
            residnorm_decrease_threshold=self.residnorm_decrease_threshold,
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

    def fit_peeler_models(self, save_folder, n_jobs=0, device=None):
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
        self._fit_subtraction_transformers(
            save_folder, n_jobs=n_jobs, device=device, which="denoisers"
        )
        self._fit_subtraction_transformers(
            save_folder, n_jobs=n_jobs, device=device, which="featurizers"
        )

    def _fit_subtraction_transformers(
        self, save_folder, n_jobs=0, device=None, which="denoisers"
    ):
        """Handle fitting either denoisers or featurizers since the logic is similar"""
        if not any(
            (
                (t.is_denoiser if which == "denoisers" else t.is_featurizer)
                and t.needs_fit()
            )
            for t in self.subtraction_denoising_pipeline
        ):
            return

        orig_denoise = self.subtraction_denoising_pipeline
        init_waveform_feature = Waveform(
            channel_index=self.channel_index,
            name="subtract_fit_waveforms",
        )
        if which == "denoisers":
            self.subtraction_denoising_pipeline = WaveformPipeline(
                [init_waveform_feature]
                + [t for t in orig_denoise if (t.is_denoiser and not t.needs_fit())]
            )
        else:
            self.subtraction_denoising_pipeline = WaveformPipeline(
                [init_waveform_feature] + [t for t in orig_denoise if t.is_denoiser]
            )

        # and we don't need any features for this
        orig_featurization_pipeline = self.featurization_pipeline
        self.featurization_pipeline = WaveformPipeline([])

        # run mini subtraction
        temp_hdf5_filename = Path(save_folder) / "subtraction_denoiser_fit.h5"
        try:
            self.run_subsampled_peeling(
                temp_hdf5_filename,
                n_jobs=n_jobs,
                device=device,
                task_name="Fit subtraction denoisers",
            )

            # fit featurization pipeline and reassign
            # work in a try finally so we can delete the temp file
            # in case of an issue or a keyboard interrupt
            with h5py.File(temp_hdf5_filename) as h5:
                waveforms = torch.from_numpy(h5["subtract_fit_waveforms"][:])
                channels = torch.from_numpy(h5["channels"][:])
            orig_denoise.fit(waveforms, max_channels=channels)
            self.subtraction_denoising_pipeline = orig_denoise
            self.featurization_pipeline = orig_featurization_pipeline
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
    trough_offset_samples=42,
    spike_length_samples=121,
    left_margin=0,
    right_margin=0,
    detection_thresholds=[12, 10, 8, 6, 5, 4],
    peak_sign="neg",
    spatial_dedup_channel_index=None,
    residnorm_decrease_threshold=3.162,  # sqrt(10)
):
    """Core peeling routine for subtraction"""
    # validate arguments to avoid confusing error messages later
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

    for threshold in detection_thresholds:
        # -- detect and extract waveforms
        # detection has more args which we don't expose right now
        times_samples, channels = detect_and_deduplicate(
            residual[:, :-1],
            threshold,
            dedup_channel_index=spatial_dedup_channel_index,
            peak_sign=peak_sign,
        )
        if not times_samples.numel():
            continue

        # throw away spikes which cannot be subtracted
        keep = (times_samples >= trough_offset_samples) & (
            times_samples < max_trough_time
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
