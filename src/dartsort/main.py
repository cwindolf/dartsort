from dataclasses import asdict
from pathlib import Path

from dartsort.cluster.initial import ensemble_chunks
from dartsort.cluster.merge import merge_templates
from dartsort.cluster.split import split_clusters
from dartsort.config import (default_clustering_config,
                             default_featurization_config,
                             default_matching_config,
                             default_motion_estimation_config,
                             default_split_merge_config,
                             default_subtraction_config,
                             default_template_config)
from dartsort.peel import (ObjectiveUpdateTemplateMatchingPeeler,
                           SubtractionPeeler)
from dartsort.templates import TemplateData
from dartsort.util.data_util import check_recording
from dartsort.util.peel_util import run_peeler
from dartsort.util.registration_util import estimate_motion


def dartsort_from_config(
    recording,
    config_path,
):
    pass


def dartsort(
    recording,
    output_directory,
    featurization_config=default_featurization_config,
    motion_estimation_config=default_motion_estimation_config,
    subtraction_config=default_subtraction_config,
    matching_config=default_subtraction_config,
    template_config=default_template_config,
    clustering_config=default_clustering_config,
    split_merge_config=default_split_merge_config,
    motion_est=None,
    matching_iterations=1,
    n_jobs=0,
    overwrite=False,
    show_progress=True,
    device=None,
):
    # initialization: subtraction, motion estimation and initial clustering
    sorting, sub_h5 = subtract(
        recording,
        output_directory,
        featurization_config=featurization_config,
        subtraction_config=subtraction_config,
        n_jobs=n_jobs,
        overwrite=overwrite,
        device=device,
    )
    if motion_est is None:
        motion_est = estimate_motion(
            recording, sorting, overwrite=overwrite, **asdict(motion_estimation_config)
        )
    sorting = cluster(
        sub_h5,
        recording,
        motion_est=motion_est,
        clustering_config=clustering_config,
    )

    # E/M iterations
    for step in range(matching_iterations):
        # M step: refine clusters
        sorting = split_merge(
            sorting,
            recording,
            motion_est,
            split_merge_config=split_merge_config,
            n_jobs_split=n_jobs,
            n_jobs_merge=n_jobs,
        )

        # E step: deconvolution
        sorting, match_h5 = match(
            recording,
            sorting,
            output_directory,
            motion_est=motion_est,
            template_config=template_config,
            featurization_config=featurization_config,
            matching_config=matching_config,
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
    motion_est=None,
    clustering_config=default_clustering_config,
):
    # TODO: have this accept a sorting and expect it to contain basic feats.
    sorting = ensemble_chunks(
        hdf5_filename,
        recording,
        clustering_config=clustering_config,
        motion_est=motion_est,
    )
    return sorting


def split_merge(
    sorting,
    recording,
    motion_est,
    split_merge_config=default_split_merge_config,
    n_jobs_split=0,
    n_jobs_merge=0,
):
    split_sorting = split_clusters(
        sorting,
        split_strategy=split_merge_config.split_strategy,
        recursive=split_merge_config.recursive_split,
        n_jobs=n_jobs_split,
    )
    merge_sorting = merge_templates(
        split_sorting,
        recording,
        template_config=split_merge_config.merge_template_config,
        merge_distance_threshold=split_merge_config.merge_distance_threshold,
        n_jobs=n_jobs_merge,
        n_jobs_templates=n_jobs_merge,
    )
    return merge_sorting


def match(
    recording,
    sorting=None,
    output_directory=None,
    motion_est=None,
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
    template_data = TemplateData.from_config(
        recording,
        sorting,
        template_config,
        motion_est=motion_est,
        n_jobs=n_jobs_templates,
        save_folder=model_dir,
        overwrite=overwrite,
        device=device,
        save_npz_name=template_npz_filename,
    )

    # instantiate peeler
    matching_peeler = ObjectiveUpdateTemplateMatchingPeeler.from_config(
        recording,
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
