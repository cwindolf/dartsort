import numpy as np
import torch
from scipy.signal import argrelmin
import torch.nn.functional as F


def detect_and_deduplicate(
    recording, threshold, channel_index, buffer_size, device
):

    if torch.device(device).type == "cuda":
        times, chans, energy = torch_voltage_detect_dedup(
            recording,
            threshold,
            channel_index=channel_index,
            order=5,
            device=device,
        )
        spike_index = np.c_[times.cpu().numpy(), chans.cpu().numpy()]
        energy = energy.cpu().numpy()
    else:
        spike_index, energy = voltage_threshold(recording, threshold)
        spike_index, energy = deduplicate_torch(
            spike_index,
            energy,
            recording.shape,
            channel_index,  # device=device
        )

    # update times wrt buffer size, remove spikes in buffer
    spike_index[:, 0] -= buffer_size
    tidx = (spike_index[:, 0] >= 0) & (
        spike_index[:, 0] < recording.shape[0] - 2 * buffer_size
    )
    spike_index = spike_index[tidx]
    energy = energy[tidx]

    return spike_index, energy


def voltage_threshold(recording, threshold, order=5):
    T, C = recording.shape
    ts, mcs = argrelmin(recording, axis=0, order=order)
    spike_index = np.c_[ts, mcs]
    energy = recording[ts, mcs]
    which = energy < -threshold
    return spike_index[which], np.abs(energy[which])


@torch.no_grad()
def deduplicate_torch(
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
        recording_shape,
        dtype=energy_torch.dtype,
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
    max_energy = torch.max(max_energy[:, channel_index], 2)[0]

    # deduplicated spikes: temporal and spatial local max
    which = (
        torch.nonzero(energy_torch >= max_energy[times, chans] - 1e-8)
        .cpu()
        .numpy()[:, 0]
    )
    energy_dedup = energy[which]
    spike_index_dedup = spike_index[which]

    return spike_index_dedup, energy_dedup


@torch.no_grad()
def torch_voltage_detect_dedup(
    recording, threshold, order=5, channel_index=None, device=None
):
    """Voltage thresholding detection and deduplication

    Arguments
    ---------
    recording : ndarray, T x C
    threshold : float
        Should be >0. Minimum trough depth of a spike, so it's a
        threshold for -voltage.
    order : int
        How many temporal neighbors to compare with during argrelmin
        / deduplication
    channel_index : ndarray
    device : string or torch device

    Returns
    -------
    times, chans, energies
        Such that spike_index can be created as
        np.c_[times.cpu().numpy(), chans.cpu().numpy()]
    """
    T, C = recording.shape
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- torch argrelmin
    neg_recording = torch.as_tensor(
        -recording, device=device, dtype=torch.float
    )
    max_energies, inds = F.max_pool2d_with_indices(
        neg_recording[None, None],
        kernel_size=[2 * 5 + 1, 1],
        stride=1,
        padding=[5, 0],
    )
    max_energies = max_energies[0, 0]
    inds = inds[0, 0]
    # torch `inds` gives loc of argmax at each position
    # find those which actually *were* the max
    unique_inds = inds.unique()
    window_max_inds = unique_inds[inds.view(-1)[unique_inds] == unique_inds]

    # voltage threshold
    max_energies_at_inds = max_energies.view(-1)[window_max_inds]
    which = torch.nonzero(max_energies_at_inds > threshold).squeeze()

    # -- unravel the spike index
    # (right now the indices are into flattened recording)
    times = torch.div(window_max_inds, C, rounding_mode="floor")
    times = times[which]

    # TODO
    # this is for compatibility with scipy argrelmin, which does not allow
    # minima at the boundary. unclear if it's necessary to keep this.
    # I think likely not since we throw away spikes detected in the
    # buffer anyway?
    compat_times = torch.nonzero(
        (0 < times) & (times < recording.shape[0] - 1)
    ).squeeze()
    times = times[compat_times]
    res_inds = which[compat_times]
    chans = window_max_inds[res_inds] % C
    energies = max_energies_at_inds[res_inds]

    # -- deduplication
    # We deduplicate if the channel index is provided.
    if channel_index is not None:
        # -- temporal max pool
        # still not sure why we can't just use `max_energies` instead of making
        # this sparsely populated array, but it leads to a different result.
        max_energies[:] = 0
        max_energies[times, chans] = energies
        max_energies = F.max_pool2d(
            max_energies[None, None],
            kernel_size=[2 * 5 + 1, 1],
            stride=1,
            padding=[5, 0],
        )[0, 0]
        # -- spatial max pool with channel index
        max_energies = F.pad(max_energies, (0, 1))
        max_energies = torch.max(max_energies[:, channel_index], 2)[0]
        # -- deduplication
        dedup = torch.nonzero(
            energies >= max_energies[times, chans] - 1e-8
        ).squeeze()
        times = times[dedup]
        chans = chans[dedup]
        energies = energies[dedup]

    return times, chans, energies
