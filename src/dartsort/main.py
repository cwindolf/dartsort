from dataclasses import asdict
from pathlib import Path

from dartsort.cluster.initial import ensemble_chunks
from dartsort.cluster.merge import merge_templates
from dartsort.cluster.split import split_clusters
from dartsort.config import (DARTsortConfig, default_clustering_config,
                             default_dartsort_config,
                             default_featurization_config,
                             default_matching_config,
                             default_motion_estimation_config,
                             default_split_merge_config,
                             default_subtraction_config,
                             default_template_config, default_waveform_config)
from dartsort.peel import (ObjectiveUpdateTemplateMatchingPeeler,
                           SubtractionPeeler)
from dartsort.templates import TemplateData
from dartsort.util.data_util import DARTsortSorting, check_recording
from dartsort.util.peel_util import run_peeler
from dartsort.util.registration_util import estimate_motion


def dartsort_from_config(
    recording,
    config_path,
):
    # stub for eventual function that reads a config file
    pass


def dartsort(
    recording,
    output_directory,
    cfg: DARTsortConfig = default_dartsort_config,
    motion_est=None,
    matching_iterations=1,
    n_jobs=0,
    overwrite=False,
    show_progress=True,
    device=None,
):
    output_directory = Path(output_directory)

    # initialization: subtraction, motion estimation and initial clustering
    sorting, sub_h5 = subtract(
        recording,
        output_directory,
        waveform_config=cfg.waveform_config,
        featurization_config=cfg.featurization_config,
        subtraction_config=cfg.subtraction_config,
        n_jobs=n_jobs,
        overwrite=overwrite,
        device=device,
    )
    if motion_est is None:
        motion_est = estimate_motion(
            recording,
            sorting,
            output_directory,
            overwrite=overwrite,
            device=device,
            **asdict(cfg.motion_estimation_config),
        )
    sorting = cluster(
        sub_h5,
        recording,
        output_directory=output_directory,
        overwrite=overwrite,
        motion_est=motion_est,
        clustering_config=cfg.clustering_config,
    )

    # E/M iterations
    for step in range(matching_iterations):
        # M step: refine clusters
        sorting = split_merge(
            sorting,
            recording,
            motion_est,
            output_directory=output_directory,
            overwrite=overwrite,
            split_merge_config=cfg.split_merge_config,
            n_jobs_split=n_jobs,
            n_jobs_merge=n_jobs,
            split_npz=f"split{step}.npz",
            merge_npz=f"merge{step}.npz",
        )

        # E step: deconvolution
        sorting, match_h5 = match(
            recording,
            sorting,
            output_directory,
            motion_est=motion_est,
            template_config=cfg.template_config,
            waveform_config=cfg.waveform_config,
            featurization_config=cfg.featurization_config,
            matching_config=cfg.matching_config,
            n_jobs_templates=n_jobs,
            n_jobs_match=n_jobs,
            overwrite=overwrite,
            device=device,
            hdf5_filename=f"matching{step}.h5",
            model_subdir=f"matching{step}_models",
        )

    # done~
    return sorting


def subtract(
    recording,
    output_directory,
    waveform_config=default_waveform_config,
    featurization_config=default_featurization_config,
    subtraction_config=default_subtraction_config,
    chunk_starts_samples=None,
    n_jobs=0,
    overwrite=False,
    residual_filename=None,
    show_progress=True,
    device=None,
    hdf5_filename="subtraction.h5",
    model_subdir="subtraction_models",
):
    check_recording(recording)
    subtraction_peeler = SubtractionPeeler.from_config(
        recording,
        waveform_config=waveform_config,
        subtraction_config=subtraction_config,
        featurization_config=featurization_config,
    )
    detections, output_hdf5_filename = run_peeler(
        subtraction_peeler,
        output_directory,
        hdf5_filename,
        model_subdir,
        featurization_config,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        n_jobs=n_jobs,
        residual_filename=residual_filename,
        show_progress=show_progress,
        device=device,
    )
    return detections, output_hdf5_filename


def cluster(
    hdf5_filename,
    recording,
    output_directory=None,
    overwrite=False,
    motion_est=None,
    clustering_config=default_clustering_config,
    output_npz="initial_clustering.npz",
):
    if output_directory is not None:
        output_npz = Path(output_directory) / output_npz
        if not overwrite and output_npz.exists():
            return DARTsortSorting.load(output_npz)

    # TODO: have this accept a sorting and expect it to contain basic feats.
    sorting = ensemble_chunks(
        hdf5_filename,
        recording,
        clustering_config=clustering_config,
        motion_est=motion_est,
    )

    if output_directory is not None and overwrite:
        DARTsortSorting.save(output_npz)

    return sorting


def split_merge(
    sorting,
    recording,
    motion_est,
    output_directory=None,
    overwrite=False,
    split_merge_config=default_split_merge_config,
    n_jobs_split=0,
    n_jobs_merge=0,
    split_npz="split.npz",
    merge_npz="merge.npz",
):
    split_exists = merge_exists = False
    if output_directory is not None:
        split_npz = Path(output_directory) / split_npz
        split_exists = split_npz.exists()
        merge_npz = Path(output_directory) / merge_npz
        merge_exists = merge_npz.exists()
        if not overwrite and merge_exists:
            return DARTsortSorting.load(merge_npz)

    if not overwrite and split_exists:
        split_sorting = DARTsortSorting.load(split_npz)
    else:
        split_sorting = split_clusters(
            sorting,
            split_strategy=split_merge_config.split_strategy,
            recursive=split_merge_config.recursive_split,
            n_jobs=n_jobs_split,
        )
        if output_directory is not None and overwrite:
            split_sorting.save(split_npz)

    if not overwrite and merge_exists:
        merge_sorting = DARTsortSorting.load(merge_npz)
    else:
        merge_sorting = merge_templates(
            split_sorting,
            recording,
            template_config=split_merge_config.merge_template_config,
            merge_distance_threshold=split_merge_config.merge_distance_threshold,
            n_jobs=n_jobs_merge,
            n_jobs_templates=n_jobs_merge,
        )
        if output_directory is not None and overwrite:
            merge_sorting.save(merge_npz)

    return merge_sorting


def match(
    recording,
    sorting=None,
    output_directory=None,
    motion_est=None,
    waveform_config=default_waveform_config,
    template_config=default_template_config,
    featurization_config=default_featurization_config,
    matching_config=default_matching_config,
    chunk_starts_samples=None,
    n_jobs_templates=0,
    n_jobs_match=0,
    overwrite=False,
    residual_filename=None,
    show_progress=True,
    device=None,
    hdf5_filename="matching0.h5",
    model_subdir="matching0_models",
    template_npz_filename="template_data.npz",
):
    assert output_directory is not None
    model_dir = Path(output_directory) / model_subdir

    # compute templates
    trough_offset_samples = waveform_config.trough_offset_samples(
        recording.sampling_frequency
    )
    spike_length_samples = waveform_config.spike_length_samples(
        recording.sampling_frequency
    )
    template_data = TemplateData.from_config(
        recording,
        sorting,
        template_config=template_config,
        motion_est=motion_est,
        n_jobs=n_jobs_templates,
        save_folder=model_dir,
        overwrite=overwrite,
        device=device,
        save_npz_name=template_npz_filename,
        trough_offset_samples=trough_offset_samples,
        spike_length_samples=spike_length_samples,
    )

    # instantiate peeler
    matching_peeler = ObjectiveUpdateTemplateMatchingPeeler.from_config(
        recording,
        waveform_config,
        matching_config,
        featurization_config,
        template_data,
        motion_est=motion_est,
    )
    sorting, output_hdf5_filename = run_peeler(
        matching_peeler,
        output_directory,
        hdf5_filename,
        model_subdir,
        featurization_config,
        chunk_starts_samples=chunk_starts_samples,
        overwrite=overwrite,
        n_jobs=n_jobs_match,
        residual_filename=residual_filename,
        show_progress=show_progress,
        device=device,
    )
    return sorting, output_hdf5_filename
