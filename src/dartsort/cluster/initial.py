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
import h5py
from dartsort.util.data_util import DARTsortSorting

from . import cluster_util, ensemble_utils
import numpy as np


def cluster_chunk(
    peeling_hdf5_filename,
    clustering_config,
    chunk_time_range_s=None,
    motion_est=None,
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
    strategy = clustering_config.cluster_strategy
    feature_scales = clustering_config.feature_scales
    assert strategy in ("closest_registered_channels","hdbscan","ensembling_hdbscan",)
    if strategy == "closest_registered_channels":
        with h5py.File(peeling_hdf5_filename, "r") as h5:
            times_samples = h5["times_samples"][:]
            channels = h5["channels"][:]
            times_s = h5["times_seconds"][:]
            xyza = h5["point_source_localizations"][:]
            amps = h5["denoised_amplitudes"][:]
            geom = h5["geom"][:]
            
        in_chunk = ensemble_utils.get_indices_in_chunk(times_s, chunk_time_range_s)
        labels = -1 * np.ones(len(times_samples))
        labels[in_chunk] = cluster_util.closest_registered_channels(
            times_s[in_chunk], xyza[in_chunk, 0], xyza[in_chunk, 2], geom, motion_est
        )
        #triaging here maybe
        sorting = DARTsortSorting(
            times_samples=times_samples,
            channels=channels,
            labels=labels,
            extra_features=dict(
                point_source_localizations=xyza,
                denoised_amplitudes=amps,
                times_seconds=times_s,
            ),
        )
    elif strategy == "hdbscan":
        #hdbscan specific parameters
        min_cluster_size = clustering_config.min_cluster_size
        min_samples = clustering_config.min_samples
        cluster_selection_epsilon = clustering_config.cluster_selection_epsilon
        with h5py.File(peeling_hdf5_filename, "r") as h5:
            times_samples = h5["times_samples"][:]
            channels = h5["channels"][:]
            times_s = h5["times_seconds"][:]
            xyza = h5["point_source_localizations"][:]
            amps = h5["denoised_amplitudes"][:]
            geom = h5["geom"][:]  
        in_chunk = ensemble_utils.get_indices_in_chunk(times_s, chunk_time_range_s)
        labels = -1 * np.ones(len(times_samples))
        labels[in_chunk]  = cluster_util.hdbscan_clustering(
            times_s[in_chunk], 
            xyza[in_chunk, 0], 
            xyza[in_chunk, 2], 
            geom, amps[in_chunk], 
            motion_est, 
            min_cluster_size=min_cluster_size, 
            min_samples=min_samples, 
            cluster_selection_epsilon=cluster_selection_epsilon,
            scales=feature_scales,
        )
        sorting = DARTsortSorting(
            times_samples=times_samples,
            channels=channels,
            labels=labels,
            extra_features=dict(
                point_source_localizations=xyza,
                denoised_amplitudes=amps,
                times_seconds=times_s,
            ),
        )
    else:
        raise ValueError
    return sorting

def ensemble_chunks(
    peeling_hdf5_filename,
    recording,
    clustering_config,
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
      : DARTsortSorting  
    """
    #get params from config
    ensemble_strategy = clustering_config.ensemble_strategy
    if ensemble_strategy is None:
        sorting = cluster_chunk(peeling_hdf5_filename, 
                                clustering_config,
                                chunk_time_range_s=None,
                                motion_est=motion_est,
                            )
    else:
        assert ensemble_strategy in ("forward_backward","meet")
        #for loop cluster chunks
        chunk_size_s = clustering_config.chunk_size_s
        if ensemble_strategy == "forward_backward":
            with h5py.File(peeling_hdf5_filename, "r") as h5:
                times_samples = h5["times_samples"][:]
                times_seconds = h5["times_seconds"][:]
                channels = h5["channels"][:]
                xyza = h5["point_source_localizations"][:]
                amps = h5["denoised_amplitudes"][:]
                geom = h5["geom"][:]
            labels = ensemble_utils.ensembling_hdbscan(
                recording, 
                times_seconds, 
                times_samples, 
                xyza[:, 0], 
                xyza[:, 2], 
                geom, 
                amps, 
                clustering_config, 
                motion_est,
            )
            sorting = DARTsortSorting(
                times_samples=times_samples,
                channels=channels,
                labels=labels,
                extra_features=dict(
                    point_source_localizations=xyza,
                    denoised_amplitudes=amps,
                    times_seconds=times_seconds,
                ),
            )
    return sorting

