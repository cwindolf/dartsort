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
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import argrelmin
from torch import nn

MAXCOPY = 8
DEFAULT_DEDUP_T = 7


def detect_and_deduplicate(
    recording,
    threshold,
    channel_index,
    buffer_size,
    peak_sign="neg",
    nn_detector=None,
    nn_denoiser=None,
    denoiser_detector=None,
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
    if nn_detector is None and denoiser_detector is None:
        spike_index, _ = voltage_detect_and_deduplicate(
            recording,
            threshold,
            channel_index,
            buffer_size,
            peak_sign=peak_sign,
            device=device,
        )
    elif nn_detector is not None:
        spike_index, _ = nn_detect_and_deduplicate(
            recording,
            threshold,
            channel_index,
            buffer_size,
            nn_detector,
            nn_denoiser,
            spike_length_samples=121,
            device=device,
        )
    elif denoiser_detector is not None:
        times, chans, _ = denoiser_detect_dedup(
            recording,
            threshold,
            denoiser_detector,
            channel_index=channel_index,
            device=device,
        )
        if times.numel():
            spike_index = torch.stack((times, chans), dim=1)
            spike_index[:, 0] -= buffer_size
        else:
            return np.array([])

    if torch.is_tensor(spike_index):
        spike_index = spike_index.cpu().numpy()

    return spike_index


# -- nn detection


class Detect(nn.Module):
    def __init__(self, channel_index, n_filters=[16, 8, 8], spike_size=121):
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
    trough_search=4,
    device="cpu",
):
    # detect in batches
    T = recording.shape[0]
    max_neighbs = channel_index.shape[1]
    batch_size = int(np.ceil(T / (32 * max_neighbs / MAXCOPY)))
    spike_inds = []
    recording_torch = torch.as_tensor(recording, device=device)
    for bs in range(0, T, batch_size):
        be = min(T, bs + batch_size)
        spike_index_batch = nn_detector.get_spike_times(
            recording_torch[bs:be],
        )
        spike_index_batch[:, 0] += bs
        spike_inds.append(spike_index_batch)
    spike_index_torch = torch.cat(spike_inds)

    # correct for detector/denoiser offset
    search_tix = spike_index_torch[:, 0, None] + torch.arange(
        0, 2 * trough_search + 1, device=spike_index_torch.device
    )
    search_traces = F.pad(
        recording_torch, (0, 0, trough_search, trough_search)
    )[search_tix, spike_index_torch[:, 1, None]]
    shifts = search_traces.argmin(1) - (trough_search + 1)
    spike_index_torch[:, 0] += shifts
    del search_traces, shifts, search_tix
    # this could in theory lead to duplicates
    spike_index_torch = torch.unique(spike_index_torch, dim=0)

    # get energies just by voltage at detection
    energy = torch.abs(
        recording_torch[spike_index_torch[:, 0], spike_index_torch[:, 1]]
    )

    # threshold
    which = energy > energy_threshold
    if not which.any():
        return torch.tensor([]), torch.tensor([])
    spike_index_torch = spike_index_torch[which]
    energy = energy[which]
    # print("after", energy.shape, energy.min(), energy.mean(), energy.max())

    # deduplicate
    spike_index_dedup, energy_dedup = deduplicate_torch(
        spike_index_torch,
        energy,
        recording.shape,
        channel_index,
        max_window=7,
        device=device,
    )
    # print("dedup", len(spike_index_dedup))

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
    peak_sign="neg",
    max_window=DEFAULT_DEDUP_T,
):
    if torch.is_tensor(recording):
        times, chans, energy = torch_voltage_detect_dedup(
            recording,
            threshold,
            channel_index=channel_index,
            order=5,
            device=device,
            peak_sign=peak_sign,
            max_window=max_window,
        )
        if times.numel():
            spike_index = torch.stack(
                (torch.atleast_1d(times), torch.atleast_1d(chans)), dim=1
            )
        else:
            return np.array([]), np.array([])
    else:
        spike_index, energy = voltage_threshold(
            recording, threshold, peak_sign=peak_sign
        )
        if not len(spike_index):
            return np.array([]), np.array([])
        spike_index, energy = deduplicate_torch(
            spike_index,
            energy,
            recording.shape,
            channel_index,
            device=device,
            max_window=max_window,
        )

    # update times wrt buffer size
    spike_index[:, 0] -= buffer_size

    return spike_index, energy


def voltage_threshold(recording, threshold, peak_sign="neg", order=5):
    T, C = recording.shape
    if peak_sign == "both":
        recording = -np.abs(recording)
    else:
        assert peak_sign == "neg"
    ts, mcs = argrelmin(recording, axis=0, order=order)
    spike_index = np.c_[ts, mcs]
    energy = recording[ts, mcs]
    which = energy < -threshold
    return spike_index[which], np.abs(energy[which])


def deduplicate_torch(
    spike_index,
    energy,
    recording_shape,
    channel_index,
    max_window=DEFAULT_DEDUP_T,
    device="cpu",
):
    spike_index_torch = torch.as_tensor(spike_index, device=device)
    energy_torch = torch.as_tensor(energy, device=device)
    times = spike_index_torch[:, 0]
    chans = spike_index_torch[:, 1]

    # initialize energy train
    energy_train = torch.zeros(
        recording_shape,
        dtype=energy_torch.dtype,
        device=device,
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


def torch_voltage_detect_dedup(
    recording,
    threshold,
    peak_sign="neg",
    order=5,
    max_window=DEFAULT_DEDUP_T,
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
    max_window : int
        How many temporal neighbors to compare with during dedup
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

    # -- torch argrelMAX not argrelmin, on the negative
    if peak_sign == "neg":
        neg_recording = torch.as_tensor(
            -recording, device=device, dtype=torch.float
        )
    elif peak_sign == "both":
        neg_recording = torch.abs(
            torch.as_tensor(recording, device=device, dtype=torch.float)
        )
    else:
        assert False
    max_energies, inds = F.max_pool2d_with_indices(
        neg_recording[None, None],
        kernel_size=[2 * order + 1, 1],
        stride=1,
        padding=[order, 0],
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
    if not which.numel():
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

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
    if not compat_times.numel():
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
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
        if not dedup.numel():
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        times = times[dedup]
        chans = chans[dedup]
        energies = energies[dedup]

    times = torch.atleast_1d(times)
    chans = torch.atleast_1d(chans)
    energies = torch.atleast_1d(energies)

    return times, chans, energies


# -- denoiser-based nn detection


class PeakToPeak(nn.Module):
    __constants__ = ["dim"]
    dim: int

    def __init__(self, dim=-1):
        super(PeakToPeak, self).__init__()
        self.dim = dim

    def forward(self, input):
        return (
            torch.max(input, dim=self.dim)[0]
            - torch.min(input, dim=self.dim)[0]
        )


# a torch debugging classic
# class Shape(nn.Module):
#     def __init__(self, name):
#         super(Shape, self).__init__()
#         self.name = name
#     def forward(self, input):
#         print(self.name, input.shape)
#         return input


class DenoiserDetect(nn.Module):
    def __init__(self, denoiser, output_t_range=(25, 60)):
        super(DenoiserDetect, self).__init__()
        denoiser = deepcopy(denoiser)
        # -- turn the denoiser into a convolutional net
        # convert linear head into a convolutional layer
        # we just grab a portion of the denoiser output since not all
        # of it is relevant for the PTP
        t0, t1 = output_t_range
        kernel_size = (
            denoiser.out.out_features
            - denoiser.conv1[0].kernel_size[0]
            - denoiser.conv2[0].kernel_size[0]
            + 2
        )
        self.conv_linear = nn.Conv1d(
            in_channels=denoiser.conv2[0].out_channels,
            out_channels=t1 - t0,
            kernel_size=kernel_size,
        )
        # T x in_features = (in_chans * kernel_size)
        W = denoiser.out.weight
        W = W.reshape(
            denoiser.out.out_features,
            denoiser.conv2[0].out_channels,
            kernel_size,
        )
        # outc x inc x kernel size
        with torch.no_grad():
            self.conv_linear.weight[:] = W[t0:t1].detach()
            self.conv_linear.bias[:] = denoiser.out.bias[t0:t1].detach()

        # feedforward convolutional arch with PTP head
        self.ff = nn.Sequential(
            # Shape("in"),
            denoiser.conv1,
            # Shape("after conv1"),
            denoiser.conv2,
            # self.denoiser.conv3,  # denoiser conv3 is unused...
            # Shape("after conv2"),
            self.conv_linear,
            # Shape("after fake conv"),
            PeakToPeak(1),
            # Shape("after ptp"),
        )

    def forward(self, input):
        return self.ff(input)

    def forward_recording(self, recording_tensor):
        # input is DxT
        # just adds C dim. so we have Tx1xD, depth on the batch dim
        return self.ff(recording_tensor.T[:, None, :]).T


def denoiser_detect_dedup(
    recording,
    ptp_threshold,
    denoiser_detector,
    order=5,
    max_window=DEFAULT_DEDUP_T,
    channel_index=None,
    device=None,
):
    """Denoiser-based thresholding detection and deduplication

    Arguments
    ---------
    recording : ndarray, T x C
    ptp_threshold : float
        Should be >0
    order : int
        How many temporal neighbors to compare with during argrelmin
    max_window : int
        How many temporal neighbors to compare with during dedup
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
    ptps = denoiser_detector.forward_recording(
        torch.as_tensor(recording, device=device, dtype=torch.float)
    )
    max_energies, inds = F.max_pool2d_with_indices(
        ptps[None, None],
        kernel_size=[2 * order + 1, 1],
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
    which = torch.nonzero(max_energies_at_inds > ptp_threshold).squeeze()
    if not which.numel():
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

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
    if not compat_times.numel():
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
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
        if not dedup.numel():
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        times = times[dedup]
        chans = chans[dedup]
        energies = energies[dedup]

    return times, chans, energies
