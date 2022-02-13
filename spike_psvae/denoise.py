import numpy as np
import numpy.linalg as la
from sklearn.decomposition import PCA
import torch
from torch import nn


class SingleChanDenoiser(nn.Module):
    """Cleaned up a little. Why is conv3 here and commented out in forward?"""

    def __init__(
        self, n_filters=[16, 8, 4], filter_sizes=[5, 11, 21], spike_size=121
    ):
        super(SingleChanDenoiser, self).__init__()
        feat1, feat2, feat3 = n_filters
        size1, size2, size3 = filter_sizes
        self.conv1 = nn.Sequential(nn.Conv1d(1, feat1, size1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(feat1, feat2, size2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(feat2, feat3, size3), nn.ReLU())
        n_input_feat = feat2 * (spike_size - size1 - size2 + 2)
        self.out = nn.Linear(n_input_feat, spike_size)

    def forward(self, x):
        x = x[:, None]
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        return self.out(x)

    def load(self, fname_model=f"../pretrained/single_chan_denoiser.pt"):
        checkpoint = torch.load(fname_model, map_location="cpu")
        self.load_state_dict(checkpoint)


def temporal_align(waveforms, maxchans=None, offset=42):
    N, T, C = waveforms.shape
    if maxchans is None:
        maxchans = waveforms.ptp(1).argmax(1)
    offsets = waveforms[np.arange(N), :, maxchans].argmin(1)
    rolls = offset - offsets
    out = np.empty_like(waveforms)
    pads = [(0, 0), (0, 0)]
    for i, roll in enumerate(rolls):
        if roll > 0:
            pads[0] = (roll, 0)
            start, end = 0, T
        elif roll < 0:
            pads[0] = (0, -roll)
            start, end = -roll, T - roll
        else:
            out[i] = waveforms[i]
            continue

        pwf = np.pad(waveforms[i], pads, mode="linear_ramp")
        out[i] = pwf[start:end, :]

    return out, rolls


def enforce_decrease(waveform, in_place=False):
    n_chan = waveform.shape[1]
    wf = waveform if in_place else waveform.copy()
    max_chan = wf.ptp(0).argmax()

    max_chan_even = max_chan - max_chan % 2
    for i in range(4, max_chan_even, 2):
        if wf[:, max_chan_even - i - 2].ptp() > wf[:, max_chan_even - i].ptp():
            wf[:, max_chan_even - i - 2] = (
                wf[:, max_chan_even - i - 2]
                * wf[:, max_chan_even - i].ptp()
                / wf[:, max_chan_even - i - 2].ptp()
            )
    for i in range(4, n_chan - max_chan_even - 2, 2):
        if wf[:, max_chan_even + i + 2].ptp() > wf[:, max_chan_even + i].ptp():
            wf[:, max_chan_even + i + 2] = (
                wf[:, max_chan_even + i + 2]
                * wf[:, max_chan_even + i].ptp()
                / wf[:, max_chan_even + i + 2].ptp()
            )

    max_chan_odd = max_chan - max_chan % 2 + 1
    for i in range(4, max_chan_odd, 2):
        if wf[:, max_chan_odd - i - 2].ptp() > wf[:, max_chan_odd - i].ptp():
            wf[:, max_chan_odd - i - 2] = (
                wf[:, max_chan_odd - i - 2]
                * wf[:, max_chan_odd - i].ptp()
                / wf[:, max_chan_odd - i - 2].ptp()
            )
    for i in range(4, n_chan - max_chan_odd - 1, 2):
        if wf[:, max_chan_odd + i + 2].ptp() > wf[:, max_chan_odd + i].ptp():
            wf[:, max_chan_odd + i + 2] = (
                wf[:, max_chan_odd + i + 2]
                * wf[:, max_chan_odd + i].ptp()
                / wf[:, max_chan_odd + i + 2].ptp()
            )

    return wf
