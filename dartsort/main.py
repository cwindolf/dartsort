from dartsort.config import FeaturizationConfig, SubtractionConfig
from dartsort.peel.subtract import SubtractionPeeler
from dartsort.transform import WaveformPipeline
from dartsort.util.data_util import SpikeTrain
from dartsort.util.waveform_util import make_channel_index

default_featurization_config = FeaturizationConfig()
default_subtraction_config = SubtractionConfig()


def dartsort(
    recording,
    output_folder,
    *more_args_tbd,
    featurization_config=default_featurization_config,
    subtraction_config=default_subtraction_config,
    dartsort_config=...,
    n_jobs=0,
    overwrite=False,
    show_progress=True,
    device=None,
):
    # coming soon
    pass


def subtract(
    recording,
    output_hdf5_filename,
    featurization_config=default_featurization_config,
    subtraction_config=default_subtraction_config,
    chunk_starts_samples=None,
    n_jobs=0,
    overwrite=False,
    residual_filename=None,
    show_progress=True,
    device=None,
):
    # construct denoising and featurization pipelines
    subtraction_denoising_pipeline = WaveformPipeline.from_config(
        subtraction_config.subtraction_denoising_config
    )
    featurization_pipeline = WaveformPipeline.from_config(featurization_config)

    # waveform extraction channel neighborhoods
    channel_index = make_channel_index(
        recording.get_channel_locations(), subtraction_config.extract_radius
    )
    # per-threshold spike event deduplication channel neighborhoods
    spatial_dedup_channel_index = make_channel_index(
        recording.get_channel_locations(),
        subtraction_config.spatial_dedup_radius,
    )

    subtraction_peeler = SubtractionPeeler(
        recording,
        channel_index,
        subtraction_denoising_pipeline,
        featurization_pipeline,
        trough_offset_samples=subtraction_config.trough_offset_samples,
        spike_length_samples=subtraction_config.spike_length_samples,
        detection_thresholds=subtraction_config.detection_thresholds,
        chunk_length_samples=subtraction_config.chunk_length_samples,
        peak_sign=subtraction_config.peak_sign,
        spatial_dedup_channel_index=spatial_dedup_channel_index,
        n_seconds_fit=subtraction_config.n_seconds_fit,
        fit_subsampling_random_state=0,
        device=device,
    )

    subtraction_peeler.peel(
        output_hdf5_filename,
        chunk_starts_samples=chunk_starts_samples,
        n_jobs=n_jobs,
        overwrite=overwrite,
        residual_filename=residual_filename,
        show_progress=show_progress,
    )

    return SpikeTrain
