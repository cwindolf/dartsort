import numpy as np
import torch
from scipy.signal import argrelmin
import torch.nn.functional as F
from scipy.ndimage import maximum_filter1d


def detect_and_deduplicate(
    recording, threshold, channel_index, buffer_size, device
):
    spike_index, energy = voltage_threshold(recording, threshold)

    # deduplicate
    spike_index_dedup, energy_dedup = deduplicate_gpu(
    # spike_index_dedup, energy_dedup = deduplicate_sp(
        spike_index, energy, recording.shape, channel_index, #device=device
    )

    # update times wrt buffer size, remove spikes in buffer
    spike_index_dedup[:, 0] -= buffer_size
    tidx = (spike_index_dedup[:, 0] >= 0) & (
        spike_index_dedup[:, 0] < recording.shape[0] - 2 * buffer_size
    )
    spike_index_dedup = spike_index_dedup[tidx]
    energy_dedup = energy_dedup[tidx]

    return spike_index_dedup, energy_dedup


def voltage_threshold(recording, threshold, order=5):
    T, C = recording.shape
    ts, mcs = argrelmin(recording, axis=0, order=order)
    spike_index = np.c_[ts, mcs]
    energy = recording[ts, mcs]
    which = energy < -threshold
    return spike_index[which], np.abs(energy[which])


def deduplicate(spike_index, energy, channel_index, max_window=5.):
    # -- sparse spatiotemporal max pool
    # we need neighbors
    kdt = KDTree(spike_index)
    # edges = kdt.query_pairs(r=5., p=1., )
    
    # get spatiotemporal neighbors by constructing a sparse distance
    # matrix where l1 dist < max_window, and using the LIL rows data
    # structure which holds the list of nonzero inds for each row
    Ds = kdt.sparse_distance_matrix(kdt, 5., p=1)
    rows = Ds.tolil().rows
    max_neighbs = max(map(len, rows))

    # spike neighbor index
    neighb_index = np.full(
        (len(spike_index), max_neighbs), len(spike_index)
    )
    for i, row in enumerate(rows):
        neighb_index[i, :len(row)] = row

    # do the max pool
    max_energy = -1e-8 + np.max(
        np.r_[energy, [0]][neighb_index],
        axis=1,
    )
    dedup_inds = np.flatnonzero(energy >= max_energy)

    return spike_index[dedup_inds], energy[dedup_inds]


@torch.no_grad()
def deduplicate_gpu(
    spike_index,
    energy,
    recording_shape,
    channel_index,
    max_window=5,
):
    spike_index_torch = torch.tensor(spike_index)
    energy_torch = torch.tensor(energy)
    times = spike_index_torch[:, 0]
    chans = spike_index_torch[:, 1]
    
    # initialize energy train
    energy_train = torch.zeros(
        recording_shape, dtype=energy_torch.dtype,
    )
    energy_train[times, chans] = energy_torch

    # get temporal max
    max_energy = F.max_pool2d(
        energy_train[None, None], 
        kernel_size=[max_window * 2 + 1, 1],
        stride=1,
        padding=[max_window, 0],
    )[0, 0]

    # get spatial max
    max_energy = F.pad(max_energy, (0, 1))
    max_energy = torch.max(max_energy[:, channel_index], 2)[0] - 1e-8

    # deduplicated spikes: temporal and spatial local max
    which = torch.nonzero(energy_torch >= max_energy[times, chans]).cpu().numpy()[:, 0]
    energy_dedup = energy[which]
    spike_index_dedup = spike_index[which]

    return spike_index_dedup, energy_dedup


def deduplicate_sp(
    spike_index,
    energy,
    recording_shape,
    channel_index,
    max_window=5,
):
    times = spike_index[:, 0]
    chans = spike_index[:, 1]
    
    # initialize energy train
    energy_train = np.zeros(
        recording_shape, dtype=energy.dtype,
    )
    energy_train[times, chans] = energy

    # get temporal max
    max_energy = maximum_filter1d(energy_train, 2 * max_window + 1, mode="constant", axis=0)

    # get spatial max
    max_energy = np.pad(max_energy, [(0, 0), (0, 1)])
    max_energy = np.max(max_energy[:, channel_index], axis=2)

    # deduplicated spikes: temporal and spatial local max
    which = np.flatnonzero(energy >= max_energy[times, chans] - 1e-8)
    energy_dedup = energy[which]
    spike_index_dedup = spike_index[which]

    return spike_index_dedup, energy_dedup
