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
    chunk_time_range_s=None,
    motion_est=None,
    strategy="closest_registered_channels",
):
    """Cluster spikes from a single segment

    Arguments
    ---------
    peeling_hdf5_filename : str or Path
    chunk_time_range_s : start and end time of chunk (seconds) in an iterable
    motion_est : optional dredge.motion_util.MotionEstimate
    strategy : one of "closest_registered_channels" or other choices tba

    Returns
    -------
    sorting : DARTsortSorting
    """
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
            times_s[in_chunk] , xyza[in_chunk, 0], xyza[in_chunk, 2], geom, amps[in_chunk], motion_est
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
    chunk_size_s=300,
    motion_est=None,
    ensemble_strategy="forward_backward",
    cluster_strategy="closest_registered_channels",
):
    """Cluster spikes from a single segment

    Arguments
    ---------
    peeling_hdf5_filename : str or Path
    chunk_size_s : time in seconds for each chunk to be clustered and ensembled
    motion_est : optional dredge.motion_util.MotionEstimate
    ensemble_strategy : one of "forward_backward" or other choices tba
    cluster_strategy : one of "closest_registered_channels" or other choices tba
    
    Returns
    -------
      : DARTsortSorting
    """
    if ensemble_strategy is None:
        raise ValueError
        # Or do we want to do regular clustering here?
        #cluster full chunk with no ensembling
        # sorting = cluster_chunk(peeling_hdf5_filename, 
        #                         chunk_time_range_s=None,
        #                         motion_est=motion_est,
        #                         strategy=cluster_strategy,
        #                     )
    else:
        assert ensemble_strategy in ("forward_backward","meet")
        #for loop cluster chunks
        if ensemble_strategy == "forward_backward":
            with h5py.File(peeling_hdf5_filename, "r") as h5:
                times_samples = h5["times_samples"][:]
                times_seconds = h5["times_seconds"][:]
                channels = h5["channels"][:]
                xyza = h5["point_source_localizations"][:]
                amps = h5["denoised_amplitudes"][:]
                geom = h5["geom"][:]
            labels = ensemble_utils.ensembling_hdbscan(
                recording, times_seconds, times_samples, xyza[:, 0], xyza[:, 2], geom, amps, motion_est, chunk_size_s,
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

