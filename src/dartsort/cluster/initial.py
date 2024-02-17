"""Initial clustering

Before template matching and before split/merge, we need to initialize
unit labels. Here, we implement methods for initializing the unit labels
inside shorter chunks (`cluster_chunk`) and across groups of shorter
chunks (`cluster_across_chunks`).

These functions expect inputs which are the HDF5 files that come from
running a BasePeeler on one or more chunks. So, they are expected to be
combined with calls to `main.subtract()`, as implemented in the
`main.initial_clustering` function (TODO!).
"""
from dataclasses import replace

import h5py
import numpy as np
from dartsort.util import job_util
from dartsort.util.data_util import DARTsortSorting

from . import cluster_util, density, ensemble_utils, forward_backward


def cluster_chunk(
    peeling_hdf5_filename,
    clustering_config,
    chunk_time_range_s=None,
    motion_est=None,
    recording=None,
    amplitudes_dataset_name="denoised_ptp_amplitudes",
    depth_order=True,
):
    """Cluster spikes from a single segment

    Arguments
    ---------
    peeling_hdf5_filename : str or Path
    chunk_time_range_s : start and end time of chunk (seconds) in an iterable
    motion_est : optional dredge.motion_util.MotionEstimate
    clustering_config: ClusteringConfig

    Returns
    -------
    sorting : DARTsortSorting
    """
    assert clustering_config.cluster_strategy in (
        "closest_registered_channels",
        "grid_snap",
        "hdbscan",
        "density_peaks",
    )

    with h5py.File(peeling_hdf5_filename, "r") as h5:
        times_samples = h5["times_samples"][:]
        channels = h5["channels"][:]
        times_s = h5["times_seconds"][:]
        xyza = h5["point_source_localizations"][:]
        amps = h5[amplitudes_dataset_name][:]
        geom = h5["geom"][:]
    in_chunk = ensemble_utils.get_indices_in_chunk(times_s, chunk_time_range_s)
    labels = -1 * np.ones(len(times_samples))
    extra_features = {
        "point_source_localizations": xyza,
        amplitudes_dataset_name: amps,
        "times_seconds": times_s,
    }

    if clustering_config.cluster_strategy == "closest_registered_channels":
        labels[in_chunk] = cluster_util.closest_registered_channels(
            times_s[in_chunk], xyza[in_chunk, 0], xyza[in_chunk, 2], geom, motion_est
        )
    elif clustering_config.cluster_strategy == "grid_snap":
        labels[in_chunk] = cluster_util.grid_snap(
            times_s[in_chunk],
            xyza[in_chunk, 0],
            xyza[in_chunk, 2],
            geom,
            grid_dx=clustering_config.grid_dx,
            grid_dz=clustering_config.grid_dz,
            motion_est=motion_est,
        )
    elif clustering_config.cluster_strategy == "hdbscan":
        labels[in_chunk] = cluster_util.hdbscan_clustering(
            recording,
            times_s[in_chunk],
            times_samples[in_chunk],
            xyza[in_chunk, 0],
            xyza[in_chunk, 2],
            amps[in_chunk],
            geom,
            motion_est,
            min_cluster_size=clustering_config.min_cluster_size,
            min_samples=clustering_config.min_samples,
            log_c=clustering_config.log_c,
            cluster_selection_epsilon=clustering_config.cluster_selection_epsilon,
            scales=clustering_config.feature_scales,
            adaptive_feature_scales=clustering_config.adaptive_feature_scales,
            recursive=clustering_config.recursive,
            remove_duplicates=clustering_config.remove_duplicates,
            remove_big_units=clustering_config.remove_big_units,
            zstd_big_units=clustering_config.zstd_big_units,
        )
    elif clustering_config.cluster_strategy == "density_peaks":
        z = xyza[in_chunk, 2]
        if motion_est is not None:
            z = motion_est.correct_s(times_s[in_chunk], z)
        scales = clustering_config.feature_scales
        ampfeat = scales[2] * np.log(clustering_config.log_c + amps[in_chunk])
        res = density.density_peaks_clustering(
            np.c_[scales[0] * xyza[in_chunk, 0], scales[1] * z, ampfeat],
            sigma_local=clustering_config.sigma_local,
            sigma_local_low=clustering_config.sigma_local_low,
            sigma_regional=clustering_config.sigma_regional,
            sigma_regional_low=clustering_config.sigma_regional_low,
            n_neighbors_search=clustering_config.n_neighbors_search,
            radius_search=clustering_config.radius_search,
            remove_clusters_smaller_than=clustering_config.remove_clusters_smaller_than,
            noise_density=clustering_config.noise_density,
            triage_quantile_per_cluster=clustering_config.triage_quantile_per_cluster,
            revert=clustering_config.triage_quantile_per_cluster,
            workers=4,
            return_extra=clustering_config.attach_density_feature,
        )
        
        if clustering_config.attach_density_feature:
            labels[in_chunk] = res["labels"]
            extra_features["density_ratio"] = np.full(labels.size, np.nan)
            extra_features["density_ratio"][in_chunk] = res["density"]
        else:
            labels[in_chunk] = res
    else:
        assert False

    sorting = DARTsortSorting(
        times_samples=times_samples,
        channels=channels,
        labels=labels,
        parent_h5_path=peeling_hdf5_filename,
        extra_features=extra_features,
    )
    
    if depth_order:
        sorting = cluster_util.reorder_by_depth(sorting, motion_est=motion_est)

    return sorting


def cluster_chunks(
    peeling_hdf5_filename,
    recording,
    clustering_config,
    motion_est=None,
):
    """Divide the recording into chunks, and cluster each chunk

    Returns a list of sortings. Each sorting labels all of the spikes in the
    recording with -1s outside the chunk, to allow for overlaps.
    """
    chunk_samples = recording.sampling_frequency * clustering_config.chunk_size_s

    # determine number of chunks
    # if we're not ensembling, that's 1 chunk.
    if (
        not clustering_config.ensemble_strategy
        or clustering_config.ensemble_strategy.lower() == "none"
    ):
        n_chunks = 1
    else:
        n_chunks = recording.get_num_samples() / chunk_samples
        # we'll count the remainder as a chunk if it's at least 2/3 of one
        n_chunks = np.floor(n_chunks) + (n_chunks - np.floor(n_chunks) > 0.66)
        n_chunks = int(max(1, n_chunks))

    # evenly divide the recording into chunks
    assert recording.get_num_segments() == 1
    start_time_s, end_time_s = recording._recording_segments[0].sample_index_to_time(
        np.array([0, recording.get_num_samples() - 1])
    )
    chunk_times_s = np.linspace(start_time_s, end_time_s, num=n_chunks + 1)
    chunk_time_ranges_s = list(zip(chunk_times_s[:-1], chunk_times_s[1:]))

    # cluster each chunk. can be parallelized in the future.
    sortings = [
        cluster_chunk(
            peeling_hdf5_filename,
            clustering_config,
            chunk_time_range_s=chunk_range,
            motion_est=motion_est,
            recording=recording,
        )
        for chunk_range in chunk_time_ranges_s
    ]

    return chunk_time_ranges_s, sortings


def ensemble_chunks(
    peeling_hdf5_filename,
    recording,
    clustering_config,
    computation_config=None,
    motion_est=None,
):
    """Initial clustering combined across chunks of time

    Arguments
    ---------
    peeling_hdf5_filename : str or Path
    recording: RecordingExtractor
    clustering_config: ClusteringConfig
    motion_est : optional dredge.motion_util.MotionEstimate

    Returns
    -------
    sorting  : DARTsortSorting
    """
    assert clustering_config.ensemble_strategy in ("forward_backward", "split_merge")
    if computation_config is None:
        computation_config = job_util.get_global_computation_config()

    # get chunk sortings
    chunk_time_ranges_s, chunk_sortings = cluster_chunks(
        peeling_hdf5_filename,
        recording,
        clustering_config,
        motion_est=motion_est,
    )
    if len(chunk_sortings) == 1:
        return chunk_sortings[0]

    if clustering_config.ensemble_strategy == "forward_backward":
        labels = forward_backward.forward_backward(
            recording,
            chunk_time_ranges_s,
            chunk_sortings,
            log_c=clustering_config.log_c,
            feature_scales=clustering_config.feature_scales,
            adaptive_feature_scales=clustering_config.adaptive_feature_scales,
            motion_est=motion_est,
        )
        sorting = replace(chunk_sortings[0], labels=labels)
    elif clustering_config.ensemble_strategy == "split_merge":
        sorting = ensemble_utils.split_merge_ensemble(
            recording,
            chunk_sortings,
            motion_est=motion_est,
            split_merge_config=clustering_config.split_merge_ensemble_config,
            n_jobs_split=computation_config.n_jobs_cpu,
            n_jobs_merge=computation_config.actual_n_jobs_gpu,
            device=computation_config.actual_device,
            show_progress=True,
        )

    return sorting
