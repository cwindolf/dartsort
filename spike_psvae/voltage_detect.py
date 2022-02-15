import numpy as np
import torch
from scipy.signal import argrelmin
import torch.nn.functional as F
from scipy.spatial import KDTree


def detect_and_deduplicate(
    recording, threshold, channel_index, buffer_size, device
):
    spike_index, energy = voltage_threshold(recording, threshold)

    # move to gpu
    spike_index = torch.from_numpy(spike_index)
    energy = torch.from_numpy(energy)
    spike_index = spike_index.to(device)
    energy = energy.to(device)

    # deduplicate
    spike_index_dedup, energy_dedup = deduplicate_gpu(
    # spike_index_dedup, energy_dedup = deduplicate(
        spike_index, energy, recording.shape, channel_index
    )
    spike_index_dedup = spike_index_dedup.cpu().numpy()
    energy_dedup = energy_dedup.cpu().numpy()

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
    spike_index = []
    energy = []

    for c in range(C):
        single_chan_rec = recording[:, c]
        index = argrelmin(single_chan_rec, order=order)[0]
        which = np.flatnonzero(single_chan_rec[index] < -threshold)
        if which.size:
            index = index[which]
            spike_index.append(np.c_[index, np.full(len(index), c)])
            energy.append(np.abs(single_chan_rec[index]))
            
    spike_index = np.concatenate(spike_index, axis=0)
    energy = np.concatenate(energy, axis=0)


    return spike_index, energy


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
    spike_index_torch,
    energy_torch,
    recording_shape,
    channel_index,
    max_window=5,
):
    device = spike_index_torch.device

    # initialize energy train
    energy_train = torch.zeros(
        recording_shape, dtype=energy_torch.dtype, device=device
    )
    energy_train[
        spike_index_torch[:, 0], spike_index_torch[:, 1]
    ] = energy_torch

    # get temporal max
    maxpool = torch.nn.MaxPool2d(
        kernel_size=[max_window * 2 + 1, 1],
        stride=1,
        padding=[max_window, 0],
    )
    max_energy = maxpool(energy_train[None, None])[0, 0]
    
    # get spatial max
    max_energy = F.pad(max_energy, (0, 1))
    max_energy = torch.max(max_energy[:, channel_index], 2)[0] - 1e-8

    # deduplicated spikes: temporal and spatial local max
    spike_index_dedup = torch.nonzero(
        (energy_train >= max_energy) & (energy_train > 0)
    )
    energy_dedup = energy_train[spike_index_dedup]

    return spike_index_dedup, energy_dedup
