import numpy as np
import torch
from scipy.signal import argrelmin


def detect_and_deduplicate(
    recording, threshold, channel_index, buffer_size, device
):
    spike_index, energy = voltage_threshold(recording, threshold)

    # move to gpu
    spike_index = torch.from_numpy(spike_index)
    energy = torch.from_numpy(energy)
    spike_index.to(device)
    energy.to(device)

    # deduplicate
    spike_index_dedup, energy_dedup = deduplicate_gpu(
        spike_index, energy, recording, channel_index
    )
    spike_index_dedup = spike_index_dedup.cpu().numpy()
    energy_dedup = energy_dedup.cpu().numpy()

    # update times wrt buffer size, remove spikes in buffer
    spike_index_dedup[:, 0] -= buffer_size
    tidx = (spike_index_dedup[:, 0] >= 0) & (
        spike_index_dedup[:, 0] < recording.shape - 2 * buffer_size
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
            spike_index.append((index, np.full(len(index), c)))
            energy.append(np.abs(single_chan_rec[index]))

    spike_index = np.concatenate(spike_index, axis=0)
    energy = np.array(energy, dtype=np.float32)

    return spike_index, energy


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
    energy_train = torch.zeros(recording_shape).to(device)
    energy_train[
        spike_index_torch[:, 0], spike_index_torch[:, 1]
    ] = energy_torch

    # get temporal max
    maxpool = torch.nn.MaxPool2d(
        kernel_size=[max_window * 2 + 1, 1], stride=1, padding=[max_window, 0]
    )
    max_energy = maxpool(energy_train[None, None])[0, 0]
    # get spatial max
    max_energy = torch.cat(
        (max_energy, torch.zeros([max_energy.shape[0], 1]).to(device)), 1
    )
    max_energy = torch.max(max_energy[:, channel_index], 2)[0] - 1e-8

    # deduplicated spikes: temporal and spatial local max
    spike_index_dedup = torch.nonzero(
        (energy_train >= max_energy) & (energy_train > 0)
    )
    energy_dedup = energy_train[spike_index_dedup]

    return spike_index_dedup, energy_dedup
