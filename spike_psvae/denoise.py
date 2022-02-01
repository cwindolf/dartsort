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


def fit_temporal_pca(
    waveforms, n_train=10_000, n_temporal_wfs=3, pca_rank=3, seed=0
):
    print("baby")
    # extract training set
    rg = np.random.default_rng(seed)
    train_ix = rg.choice(len(waveforms), replace=False, size=n_train)
    train_ix.sort()

    # get temporal components
    u, s, vh = la.svd(waveforms[train_ix], full_matrices=False)

    # fit PCAs
    pca_temporal = [
        PCA(pca_rank).fit(u[:, :, k]) for k in range(n_temporal_wfs)
    ]

    return pca_temporal


def apply_temporal_pca(pca_temporal, waveforms):
    n, t, c = waveforms.shape
    n_temporal_wfs = len(pca_temporal)
    u, s, vh = la.svd(waveforms, full_matrices=False)
    u = np.stack(
        [
            pca.inverse_transform(pca.transform(u[:, :, k]))
            for k, pca in enumerate(pca_temporal)
        ],
        axis=-1,
    )
    return u @ (s[:, :n_temporal_wfs, None] * vh[:, :n_temporal_wfs])


def enforce_decrease(waveform):
    n_chan = waveform.shape[1]
    wf = waveform.copy()

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
