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
from dartsort.util.data_util import DARTsortSorting, chunk_time_ranges
from dartsort.config import ClusteringConfig

from . import cluster_util, density, ensemble_utils, forward_backward


def cluster_chunk(
    peeling_hdf5_filename,
    clustering_config,
    sorting=None,
    chunk_time_range_s=None,
    motion_est=None,
    recording=None,
    amplitudes_dataset_name="denoised_ptp_amplitudes",
    localizations_dataset_name="point_source_localizations",
    depth_order=True,
    ramp_num_spikes=[10, 60],
    ramp_ptp=[2, 6],
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

    if sorting is None:
        sorting = DARTsortSorting.from_peeling_hdf5(peeling_hdf5_filename)
    xyza = getattr(sorting, localizations_dataset_name)
    amps = getattr(sorting, amplitudes_dataset_name)

    if recording is None:
        with h5py.File(sorting.parent_h5_path, "r") as h5:
            geom = h5["geom"][:]
    else:
        geom = recording.get_channel_locations()

    to_cluster = ensemble_utils.get_indices_in_chunk(sorting.times_seconds, chunk_time_range_s)
    to_cluster = np.setdiff1d(to_cluster, np.flatnonzero(sorting.labels < -1))
    labels = np.full_like(sorting.labels, -1)
    extra_features = sorting.extra_features

    if clustering_config.cluster_strategy == "closest_registered_channels":
        labels[to_cluster] = cluster_util.closest_registered_channels(
            sorting.times_seconds[to_cluster],
            xyza[to_cluster, 0],
            xyza[to_cluster, 2],
            geom,
            motion_est,
        )
    elif clustering_config.cluster_strategy == "grid_snap":
        labels[to_cluster] = cluster_util.grid_snap(
            sorting.times_seconds[to_cluster],
            xyza[to_cluster, 0],
            xyza[to_cluster, 2],
            geom,
            grid_dx=clustering_config.grid_dx,
            grid_dz=clustering_config.grid_dz,
            motion_est=motion_est,
        )
    elif clustering_config.cluster_strategy == "hdbscan":
        labels[to_cluster] = cluster_util.hdbscan_clustering(
            recording,
            sorting.times_seconds[to_cluster],
            sorting.times_samples[to_cluster],
            xyza[to_cluster, 0],
            xyza[to_cluster, 2],
            amps[to_cluster],
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
        z = xyza[to_cluster, 2]
        if motion_est is not None:
            z = motion_est.correct_s(sorting.times_seconds[to_cluster], z)
        z_not_reg = xyza[to_cluster, 2]
        scales = clustering_config.feature_scales
        ampfeat = scales[2] * np.log(clustering_config.log_c + amps[to_cluster])
        res = density.density_peaks_clustering(
            np.c_[scales[0] * xyza[to_cluster, 0], scales[1] * z, ampfeat],
            geom=geom,
            y=xyza[to_cluster, 1],
            z_not_reg=z_not_reg,
            use_y_triaging=clustering_config.use_y_triaging,
            sigma_local=clustering_config.sigma_local,
            sigma_local_low=clustering_config.sigma_local_low,
            sigma_regional=clustering_config.sigma_regional,
            sigma_regional_low=clustering_config.sigma_regional_low,
            n_neighbors_search=clustering_config.n_neighbors_search,
            radius_search=clustering_config.radius_search,
            remove_clusters_smaller_than=clustering_config.remove_clusters_smaller_than,
            noise_density=clustering_config.noise_density,
            triage_quantile_per_cluster=clustering_config.triage_quantile_per_cluster,
            ramp_triage_per_cluster=clustering_config.ramp_triage_per_cluster,
            revert=clustering_config.revert,
            triage_quantile_before_clustering=clustering_config.triage_quantile_before_clustering,
            amp_no_triaging_before_clustering=clustering_config.amp_no_triaging_before_clustering,
            amp_no_triaging_after_clustering=clustering_config.amp_no_triaging_after_clustering,
            distance_dependent_noise_density=clustering_config.distance_dependent_noise_density,
            outlier_radius=clustering_config.outlier_radius,
            outlier_neighbor_count=clustering_config.outlier_neighbor_count,
            scales=scales,
            log_c=clustering_config.log_c,
            workers=clustering_config.workers,
            return_extra=clustering_config.attach_density_feature,
        )

        if clustering_config.remove_small_far_clusters:
            # TODO: move this out into a new function, if it is used?
            # the arguments ramp_ptp and ramp_num_spikes should be put into a config object.
            if clustering_config.attach_density_feature:
                labels_sort = res["labels"]
            else:
                labels_sort = res
            z = xyza[to_cluster, 2]
            if motion_est is not None:
                z = motion_est.correct_s(times_s[to_cluster], z)
            all_med_ptp = []
            all_med_z_spread = []
            all_med_x_spread = []
            num_spikes = []
            for k in np.unique(labels_sort)[np.unique(labels_sort)>-1]:
                all_med_ptp.append(np.median(amps[to_cluster[labels_sort == k]]))
                all_med_x_spread.append(xyza[to_cluster[labels_sort == k], 0].std())
                all_med_z_spread.append(z[labels_sort == k].std())
                num_spikes.append((labels_sort == k).sum())

            all_med_ptp = np.array(all_med_ptp)
            all_med_x_spread = np.array(all_med_x_spread)
            all_med_z_spread = np.array(all_med_z_spread)
            num_spikes = np.array(num_spikes)

            # ramp from ptp 2 to 6 with n spikes from 60 to 10 per minute!
            idx_low = np.flatnonzero(np.logical_and(
                np.isin(labels_sort, np.flatnonzero(num_spikes<=(chunk_time_range_s[1]-chunk_time_range_s[0])/60*(ramp_num_spikes[1] - (all_med_ptp - ramp_ptp[0])/(ramp_ptp[1]-ramp_ptp[0])*(ramp_num_spikes[1]-ramp_num_spikes[0])))),
                np.isin(labels_sort, np.flatnonzero(all_med_ptp<=ramp_ptp[1]))
            ))
            if clustering_config.attach_density_feature:
                res["labels"][idx_low] = -1
            else:
                res[idx_low] = -1

        if clustering_config.attach_density_feature:
            labels[to_cluster] = res["labels"]
            extra_features["density_ratio"] = np.full(labels.size, np.nan)
            extra_features["density_ratio"][to_cluster] = res["density"]
        else:
            labels[to_cluster] = res
    else:
        assert False

    sorting = DARTsortSorting(
        times_samples=sorting.times_samples,
        channels=sorting.channels,
        labels=labels,
        sampling_frequency=sorting.sampling_frequency,
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
    sorting=None,
    motion_est=None,
    amplitudes_dataset_name='denoised_ptp_amplitudes',
):
    """Divide the recording into chunks, and cluster each chunk

    Returns a list of sortings. Each sorting labels all of the spikes in the
    recording with -1s outside the chunk, to allow for overlaps.
    """
    # determine number of chunks
    # if we're not ensembling, that's 1 chunk.
    if (
        not clustering_config.ensemble_strategy
        or clustering_config.ensemble_strategy.lower() == "none"
    ):
        chunk_length_samples = None
        chunk_time_ranges_s = [None]
    else:
        chunk_length_samples = (
            recording.sampling_frequency * clustering_config.chunk_size_s
        )
        chunk_time_ranges_s = chunk_time_ranges(recording, chunk_length_samples)

    # cluster each chunk. can be parallelized in the future.
    sortings = [
        cluster_chunk(
            peeling_hdf5_filename,
            clustering_config,
            sorting=sorting,
            chunk_time_range_s=chunk_range,
            motion_est=motion_est,
            recording=recording,
            amplitudes_dataset_name=amplitudes_dataset_name,
        )
        for chunk_range in chunk_time_ranges_s
    ]

    return chunk_time_ranges_s, sortings


def ensemble_chunks(
    peeling_hdf5_filename,
    recording,
    clustering_config,
    sorting=None,
    computation_config=None,
    motion_est=None,
    **kwargs,
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
    assert clustering_config.ensemble_strategy in (
        "forward_backward",
        "split_merge",
        "none",
        None,
    )
    if computation_config is None:
        computation_config = job_util.get_global_computation_config()

    # get chunk sortings
    chunk_time_ranges_s, chunk_sortings = cluster_chunks(
        peeling_hdf5_filename,
        recording,
        clustering_config,
        sorting=sorting,
        motion_est=motion_est,
        **kwargs,
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


def initial_clustering(
    recording,
    sorting=None,
    peeling_hdf5_filename=None,
    clustering_config=None,
    computation_config=None,
    motion_est=None,
    **kwargs,
):
    if sorting is None:
        sorting = DARTsortSorting.from_peeling_hdf5(peeling_hdf5_filename)
    if peeling_hdf5_filename is None:
        peeling_hdf5_filename = sorting.parent_h5_path

    return ensemble_chunks(
        peeling_hdf5_filename=peeling_hdf5_filename,
        recording=recording,
        clustering_config=clustering_config,
        sorting=sorting,
        computation_config=computation_config,
        motion_est=motion_est,
        **kwargs,
    )

    