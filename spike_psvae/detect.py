"""
GPU implementation notes:
A second of data uses
    30,000 * 400 * 4 ~= 48 MB
A typical GPU might have 8GB memory, say 6 are free.
That means we can keep like 100 batches of data at a time.
The channel index can have like ~30/40 neighbors per channel,
so the spatial max pooling operations below are heavy.
If we want to run 10 threads, each can keep 10 copies of the
data. So let's batch up the spatial max pool to limit the memory
consumption by MAXCOPYx. Thus we batch up into batches of length
    30000 / (channel_index.shape[1] / MAXCOPY)
"""
import numpy as np
import torch
from scipy.signal import argrelmin
from torch import nn
import torch.nn.functional as F


MAXCOPY = 8


def detect_and_deduplicate(
    recording,
    threshold,
    channel_index,
    buffer_size,
    nn_detector=None,
    nn_denoiser=None,
    spike_length_samples=121,
    device="cpu",
):
    """Wrapper for CPU/GPU and NN/voltage detection

    Handles device logic and extracts waveforms on the current device
    for the caller.

    Returns
    -------
    spike_index : (N spikes, 2)
        int numpy array
    recording
        Either the original recording, or a torch version if we're
        on GPU.
    """
    if nn_detector is None:
        spike_index, _ = voltage_detect_and_deduplicate(
            recording,
            threshold,
            channel_index,
            buffer_size,
            device=device,
        )
    else:
        assert nn_denoiser is not None
        spike_index, _ = nn_detect_and_deduplicate(
            recording,
            voltage_threshold,
            channel_index,
            buffer_size,
            nn_detector,
            nn_denoiser,
            spike_length_samples=121,
            device="cpu",
        )

    return spike_index


# -- nn detection


class Detect(nn.Module):
    def __init__(self, channel_index, n_filters=[16, 8, 8], spike_size=4):
        super(Detect, self).__init__()

        self.spike_size = spike_size
        self.channel_index = channel_index
        n_neigh = self.channel_index.shape[1]
        feat1, feat2, feat3 = n_filters

        self.temporal_filter1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=feat1,
                kernel_size=[spike_size, 1],
                stride=1,
                padding=[(self.spike_size - 1) // 2, 0],
            ),
            nn.ReLU(),
        )
        self.temporal_filter2 = nn.Sequential(
            nn.Conv2d(feat1, feat2, [1, 1], 1, 0),
            nn.ReLU(),
        )
        self.out = nn.Linear(feat2 * n_neigh, 1)

    def forward(self, x):
        x = x[:, None]
        x = self.temporal_filter1(x)
        x = self.temporal_filter2(x)[:, :, 0]
        x = x.reshape(x.shape[0], -1)
        x = self.out(x)
        return torch.sigmoid(x)

    def forward_recording(self, recording_tensor):
        x = recording_tensor[None, None]
        x = self.temporal_filter1(x)
        x = self.temporal_filter2(x)

        zero_buff = torch.zeros([1, x.shape[1], x.shape[2], 1]).to(x.device)
        x = torch.cat((x, zero_buff), 3)[0]
        x = x[:, :, self.channel_index].permute(1, 2, 0, 3)
        x = self.out(
            x.reshape(
                recording_tensor.shape[0] * recording_tensor.shape[1], -1
            )
        )
        x = x.reshape(recording_tensor.shape[0], recording_tensor.shape[1])

        return x

    def get_spike_times(
        self,
        recording_tensor,
        max_window=7,
        threshold=0.5,
        buffer=None,
    ):
        probs = self.forward_recording(recording_tensor)
        maxpool = torch.nn.MaxPool2d(
            kernel_size=[max_window, 1],
            stride=1,
            padding=[(max_window - 1) // 2, 0],
        )
        temporal_max = maxpool(probs[None])[0] - 1e-8

        spike_index_torch = torch.nonzero(
            (probs >= temporal_max)
            & (probs > np.log(threshold / (1 - threshold)))
        )

        # remove edge spikes
        if buffer is None:
            buffer = self.spike_size // 2

        spike_index_torch = spike_index_torch[
            (spike_index_torch[:, 0] > buffer)
            & (spike_index_torch[:, 0] < recording_tensor.shape[0] - buffer)
        ]

        return spike_index_torch

    def load(self, fname_model):
        checkpoint = torch.load(fname_model, map_location="cpu")
        self.load_state_dict(checkpoint)
        return self


def nn_detect_and_deduplicate(
    recording,
    energy_threshold,
    channel_index,
    buffer_size,
    nn_detector,
    nn_denoiser,
    spike_length_samples=121,
    trough_offset=42,
    device="cpu",
):
    # detect in batches
    T = recording.shape[0]
    max_neighbs = channel_index.shape[1]
    batch_size = int(np.ceil(T / (max_neighbs / MAXCOPY)))
    spike_inds = []
    recording_torch = torch.as_tensor(recording, device=device)
    for bs in range(0, T, batch_size):
        be = min(T, bs + batch_size)
        spike_index_batch = nn_detector.get_spike_times(
            recording_torch[bs:be],
            voltage_threshold=voltage_threshold,
            buffer=buffer_size,
        )
        spike_inds.append(spike_index_batch)
    spike_index_torch = torch.cat(spike_inds)

    # get energies as PTP of max channel traces
    trange = torch.arange(-trough_offset, spike_length_samples - trough_offset)
    tix = spike_index_torch[:, 0, None] + trange[None, :]
    maxchantraces = recording_torch[tix, spike_index_torch[:, 1]]
    maxchantraces = nn_denoiser(maxchantraces)
    energy = maxchantraces.max(1) - maxchantraces.min(1)
    del maxchantraces

    # threshold
    spike_index_torch = spike_index_torch[energy > energy_threshold]

    # deduplicate
    spike_index_dedup, energy_dedup = deduplicate_torch(
        spike_index_torch,
        energy,
        recording.shape,
        channel_index,
        max_window=7,
    )

    # de-buffer for caller
    spike_index_dedup[:, 0] -= buffer_size

    return spike_index_dedup, energy_dedup


# -- voltage detection


def voltage_detect_and_deduplicate(
    recording,
    threshold,
    channel_index,
    buffer_size,
    device="cpu",
):
    if torch.device(device).type == "cuda":
        times, chans, energy, rec = torch_voltage_detect_dedup(
            recording,
            threshold,
            channel_index=channel_index,
            order=5,
            device=device,
        )
        if len(times):
            spike_index = np.c_[times.cpu().numpy(), chans.cpu().numpy()]
            energy = energy.cpu().numpy()
        else:
            return [], []
    else:
        spike_index, energy = voltage_threshold(recording, threshold)
        if not len(spike_index):
            return [], []
        spike_index, energy = deduplicate_torch(
            spike_index,
            energy,
            recording.shape,
            channel_index,
        )

    # update times wrt buffer size
    spike_index[:, 0] -= buffer_size

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
    max_window=7,
):
    spike_index_torch = torch.as_tensor(spike_index)
    energy_torch = torch.as_tensor(energy)
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
    T = recording_shape[0]
    max_neighbs = channel_index.shape[1]
    batch_size = int(np.ceil(T / (max_neighbs / MAXCOPY)))
    for bs in range(0, T, batch_size):
        be = min(T, bs + batch_size)
        max_energy[bs:be] = torch.max(
            F.pad(max_energy[bs:be], (0, 1))[:, channel_index], 2
        )[0]

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
    recording,
    threshold,
    order=5,
    max_window=7,
    channel_index=None,
    device=None,
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
    recording = torch.as_tensor(recording, device=device, dtype=torch.float)
    max_energies, inds = F.max_pool2d_with_indices(
        -recording[None, None],
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
    if not which.size():
        return [], [], []

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
    if not len(compat_times):
        return [], [], []
    times = times[compat_times]
    res_inds = which[compat_times]
    chans = window_max_inds[res_inds] % C
    energies = max_energies_at_inds[res_inds]

    # -- deduplication
    # We deduplicate if the channel index is provided.
    if channel_index is not None:
        channel_index = torch.tensor(
            channel_index, device=device, dtype=torch.long
        )

        # -- temporal max pool
        # still not sure why we can't just use `max_energies` instead of making
        # this sparsely populated array, but it leads to a different result.
        max_energies[:] = 0
        max_energies[times, chans] = energies
        max_energies = F.max_pool2d(
            max_energies[None, None],
            kernel_size=[2 * max_window + 1, 1],
            stride=1,
            padding=[max_window, 0],
        )[0, 0]

        # -- spatial max pool with channel index
        # batch size heuristic, see __doc__
        max_neighbs = channel_index.shape[1]
        batch_size = int(np.ceil(T / (max_neighbs / MAXCOPY)))
        for bs in range(0, T, batch_size):
            be = min(T, bs + batch_size)
            max_energies[bs:be] = torch.max(
                F.pad(max_energies[bs:be], (0, 1))[:, channel_index], 2
            )[0]

        # -- deduplication
        dedup = torch.nonzero(
            energies >= max_energies[times, chans] - 1e-8
        ).squeeze()
        if not len(dedup):
            return [], [], []
        times = times[dedup]
        chans = chans[dedup]
        energies = energies[dedup]

    return times, chans, energies, recording
