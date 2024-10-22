from dataclasses import asdict
from pathlib import Path

import numpy as np
from dartsort.cluster.initial import ensemble_chunks
from dartsort.cluster.merge import merge_templates
from dartsort.cluster.split import split_clusters
from dartsort.config import (DARTsortConfig, default_clustering_config,
                             default_dartsort_config,
                             default_featurization_config,
                             default_matching_config,
                             default_split_merge_config,
                             default_subtraction_config,
                             default_template_config, default_waveform_config)
from dartsort.peel import (ObjectiveUpdateTemplateMatchingPeeler,
                           SubtractionPeeler)
from dartsort.templates import TemplateData
from dartsort.util.data_util import (DARTsortSorting, check_recording,
                                     keep_only_most_recent_spikes)
from dartsort.util.peel_util import run_peeler
from dartsort.util.registration_util import estimate_motion


def dartsort(
    recording,
    output_directory,
    cfg: DARTsortConfig = default_dartsort_config,
    motion_est=None,
    n_jobs_gpu=None,
    n_jobs_cpu=None,
    overwrite=False,
    show_progress=True,
    device=None,
):
    output_directory = Path(output_directory)

    n_jobs = n_jobs_gpu
    if n_jobs is None:
        n_jobs = cfg.computation_config.actual_n_jobs_gpu
    n_jobs_cluster = n_jobs_cpu
    if n_jobs_cluster is None:
        n_jobs_cluster = cfg.computation_config.n_jobs_cpu

    # first step: subtraction and motion estimation
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
    if cfg.subtract_only:
        return sorting

    # clustering E/M. start by initializing clusters.
    sorting = cluster(
        sub_h5,
        recording,
        output_directory=output_directory,
        overwrite=overwrite,
        motion_est=motion_est,
        clustering_config=cfg.clustering_config,
    )

    # E/M iterations
    for step in range(cfg.matching_iterations):
        # M step: refine clusters
        if step > 0 or cfg.do_initial_split_merge:
            sorting = split_merge(
                sorting,
                recording,
                motion_est,
                output_directory=output_directory,
                overwrite=overwrite,
                split_merge_config=cfg.split_merge_config,
                n_jobs_split=n_jobs_cluster,
                n_jobs_merge=n_jobs,
                split_npz=f"split{step}.npz",
                merge_npz=f"merge{step}.npz",
            )

        # E step: deconvolution
        is_final = step == cfg.matching_iterations - 1
        prop = 1.0 if is_final else cfg.intermediate_matching_subsampling
        sorting, match_h5 = match(
            recording,
            sorting,
            output_directory,
            motion_est=motion_est,
            template_config=cfg.template_config,
            waveform_config=cfg.waveform_config,
            featurization_config=cfg.featurization_config,
            matching_config=cfg.matching_config,
            subsampling_proportion=prop,
            n_jobs_templates=n_jobs,
            n_jobs_match=n_jobs,
            overwrite=overwrite,
            device=device,
            hdf5_filename=f"matching{step}.h5",
            model_subdir=f"matching{step}_models",
        )

    if cfg.do_final_split_merge:
        sorting = split_merge(
            sorting,
            recording,
            motion_est,
            output_directory=output_directory,
            overwrite=overwrite,
            split_merge_config=cfg.split_merge_config,
            n_jobs_split=n_jobs_cluster,
            n_jobs_merge=n_jobs,
            split_npz=f"split{step+1}.npz",
            merge_npz=f"merge{step+1}.npz",
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

    if output_directory is not None:
        sorting.save(output_npz)

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
            motion_est=motion_est,
        )
        if output_directory is not None:
            split_sorting.save(split_npz)

    if not overwrite and merge_exists:
        merge_sorting = DARTsortSorting.load(merge_npz)
    else:
        merge_sorting = merge_templates(
            split_sorting,
            recording,
            motion_est=motion_est,
            template_config=split_merge_config.merge_template_config,
            merge_distance_threshold=split_merge_config.merge_distance_threshold,
            min_spatial_cosine=split_merge_config.min_spatial_cosine,
            linkage=split_merge_config.linkage,
            n_jobs=n_jobs_merge,
            n_jobs_templates=n_jobs_merge,
        )
        if output_directory is not None:
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
    template_data=None,
    template_npz_filename="template_data.npz",
):
    assert output_directory is not None
    model_dir = Path(output_directory) / model_subdir

    if not template_config.time_tracking:
        # compute templates
        trough_offset_samples = waveform_config.trough_offset_samples(
            recording.sampling_frequency
        )
        spike_length_samples = waveform_config.spike_length_samples(
            recording.sampling_frequency
        )
        if template_data is None:
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
    else:
        # compute chunk time ranges
        chunk_samples = (
            recording.sampling_frequency * template_config.chunk_size_s
        )
        n_chunks = recording.get_num_samples() / chunk_samples
        # we'll count the remainder as a chunk if it's at least 2/3 of one
        n_chunks = np.floor(n_chunks) + (n_chunks - np.floor(n_chunks) > 0.66)
        n_chunks = int(max(1, n_chunks))

        # evenly divide the recording into chunks
        assert recording.get_num_segments() == 1
        start_time_s, end_time_s = recording._recording_segments[
            0
        ].sample_index_to_time(np.array([0, recording.get_num_samples() - 1]))
        chunk_times_s = np.linspace(start_time_s, end_time_s, num=n_chunks + 1)
        chunk_time_ranges_s = list(zip(chunk_times_s[:-1], chunk_times_s[1:]))

        sorting_list, output_hdf5_filename_list = [], []

        for j, chunk_time_range in enumerate(chunk_time_ranges_s):
            print(f"chunk_{j}")
            model_subdir_chunk = f"chunk_{j}_" + model_subdir
            model_dir_chunk = Path(output_directory) / model_subdir_chunk

            chunk_starts_samples = np.arange(
                chunk_time_range[0] * recording.sampling_frequency,
                chunk_time_range[1] * recording.sampling_frequency,
                matching_config.chunk_length_samples,
            ).astype("int")

            if sorting is not None:
                sorting_chunk = keep_only_most_recent_spikes(
                    sorting,
                    n_min_spikes=template_config.spikes_per_unit,
                    latest_time_sample=chunk_time_range[1]
                    * recording.sampling_frequency,
                )
            else:
                sorting_chunk = None

            # compute templates
            trough_offset_samples = waveform_config.trough_offset_samples(
                recording.sampling_frequency
            )
            spike_length_samples = waveform_config.spike_length_samples(
                recording.sampling_frequency
            )
            template_data = TemplateData.from_config(
                recording,
                sorting_chunk,
                template_config=template_config,
                motion_est=motion_est,
                n_jobs=n_jobs_templates,
                save_folder=model_dir_chunk,
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
            sorting_chunk, output_hdf5_filename = run_peeler(
                matching_peeler,
                output_directory,
                f"chunk_{j}_" + hdf5_filename,
                model_subdir_chunk,
                featurization_config,
                chunk_starts_samples=chunk_starts_samples,
                overwrite=overwrite,
                n_jobs=n_jobs_match,
                residual_filename=residual_filename,
                show_progress=show_progress,
                device=device,
            )

            sorting_list.append(sorting_chunk)
            output_hdf5_filename_list.append(output_hdf5_filename)

        return sorting_list, output_hdf5_filename_list
