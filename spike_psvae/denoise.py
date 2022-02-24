import numpy as np

# import numpy.linalg as la
import torch

from pathlib import Path

from sklearn.decomposition import PCA
from torch import nn
from tqdm.auto import trange


pretrained_path = (
    Path(__file__).parent.parent / "pretrained/single_chan_denoiser.pt"
)


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

    def load(self, fname_model=pretrained_path):
        checkpoint = torch.load(fname_model, map_location="cpu")
        self.load_state_dict(checkpoint)
        return self


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


def invert_temporal_align(aligned, rolls):
    T = aligned.shape[1]
    out = np.empty_like(aligned)
    pads = [(0, 0), (0, 0)]
    for i, roll in enumerate(-rolls):
        if roll > 0:
            pads[0] = (roll, 0)
            start, end = 0, T
        elif roll < 0:
            pads[0] = (0, -roll)
            start, end = -roll, T - roll
        else:
            out[i] = aligned[i]
            continue

        pwf = np.pad(aligned[i], pads, mode="linear_ramp")
        out[i] = pwf[start:end, :]

    return out


def enforce_decrease_new(waveform, in_place=False):
    n_chan = waveform.shape[1]
    wf = waveform if in_place else waveform.copy()
    ptp = wf.ptp(0)
    max_chan = ptp.argmax()

    max_chan_even = max_chan - max_chan % 2
    max_chan_odd = max_chan_even + 1    
    
    len_reg = (max_chan_even-2)//2
    regularizer = np.zeros(len_reg)
    max_ptp = ptp[max_chan_even-2]
    for i in range(len_reg):
        max_ptp = min(max_ptp, ptp[max_chan_even-4-2*i])
        regularizer[len_reg-i-1] = ptp[max_chan_even-4-2*i]/max_ptp
    wf[:, np.arange(0,max_chan_even-2, 2)] /= regularizer
    
    len_reg = (n_chan -1- max_chan_even-2)//2
    regularizer = np.zeros(len_reg)
    max_ptp = ptp[max_chan_even+2]
    for i in range(len_reg):
        max_ptp = min(max_ptp, ptp[max_chan_even+4+2*i])
        regularizer[i] = ptp[max_chan_even+4+2*i]/max_ptp
    wf[:, np.arange(max_chan_even+4,n_chan, 2)] /= regularizer

    len_reg = (max_chan_odd-2)//2
    regularizer = np.zeros(len_reg)
    max_ptp = ptp[max_chan_odd-2]
    for i in range(len_reg):
        max_ptp = min(max_ptp, ptp[max_chan_odd-4-2*i])
        regularizer[len_reg-i-1] = ptp[max_chan_odd-4-2*i]/max_ptp
    wf[:, np.arange(1,max_chan_odd-2, 2)] /= regularizer
    
    len_reg = (n_chan -1- max_chan_odd-2)//2
    regularizer = np.zeros(len_reg)
    max_ptp = ptp[max_chan_odd+2]
    for i in range(len_reg):
        max_ptp = min(max_ptp, ptp[max_chan_odd+4+2*i])
        regularizer[i] = ptp[max_chan_odd+4+2*i]/max_ptp
    wf[:, np.arange(max_chan_odd+4,n_chan, 2)] /= regularizer
    
    return wf


@torch.inference_mode()
def cleaned_waveforms(
    waveforms, spike_index, firstchans, residual, s_start=0, tpca_rank=7, pbar=True
):
    N, T, C = waveforms.shape
    denoiser = SingleChanDenoiser().load()
    cleaned = np.empty((N, C, T), dtype=waveforms.dtype)
    ixs = trange(len(spike_index), desc="Cleaning and denoising") if pbar else range(len(spike_index))
    for ix in ixs:
        t, mc = spike_index[ix]
        fc = firstchans[ix]
        t = t - s_start

        if t + 79 > residual.shape[0]:
            raise ValueError("Spike time outside range")

        cleaned[ix] = denoiser(
            torch.as_tensor(
                (residual[t - 42 : t + 79, fc : fc + C] + waveforms[ix]).T,
                dtype=torch.float,
            )
        ).numpy()

    tpca = PCA(tpca_rank)
    cleaned = cleaned.reshape(N * C, T)
    cleaned = tpca.inverse_transform(tpca.fit_transform(cleaned))
    cleaned = cleaned.reshape(N, C, T).transpose(0, 2, 1)

    for i in range(N):
        enforce_decrease(cleaned[i], in_place=True)

    return cleaned
