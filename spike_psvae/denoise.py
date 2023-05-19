# %%
import numpy as np
from scipy.spatial.distance import cdist

# import numpy.linalg as la
import torch

from pathlib import Path

from sklearn.decomposition import PCA
from torch import nn
from tqdm.auto import trange

from .denoise_temporal_decrease import (
    _enforce_temporal_decrease_right, _enforce_temporal_decrease_left
)


pretrained_path = (
    Path(__file__).parent.parent / "pretrained/single_chan_denoiser.pt"
)


class SingleChanDenoiser(nn.Module):
    """Cleaned up a little. Why is conv3 here and commented out in forward?"""

    def __init__(
        # self, n_filters=[16, 8, 4], filter_sizes=[5, 11, 21], spike_size=121
        self, n_filters=[16, 8], filter_sizes=[5, 11], spike_size=121
    ):
        super(SingleChanDenoiser, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1, n_filters[0], filter_sizes[0]), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(n_filters[0], n_filters[1], filter_sizes[1]), nn.ReLU())
        if len(n_filters) > 2:
            self.conv3 = nn.Sequential(nn.Conv1d(n_filters[1], n_filters[2], filter_sizes[2]), nn.ReLU())
        n_input_feat = n_filters[1] * (spike_size - filter_sizes[0] - filter_sizes[1] + 2)
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
    

def denoise_with_phase_shift(chan_wfs, phase_shift, dn, chan_ci_idx, spk_sign, offset=42, small_threshold = 2, corr_th = 0.8):
    wfs_to_denoise = torch.roll(chan_wfs, - phase_shift)
    wfs = wfs_to_denoise[None, None, :]
    wfs_denoised = dn(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
    wfs_denoised = wfs_denoised.detach().numpy()

    single_chan_denoised = np.roll(np.squeeze(wfs_denoised), phase_shift)
    
    which = slice(offset-10, offset+10)
    d_s_corr = np.dot(np.squeeze(wfs_denoised)[which], wfs_to_denoise[which])/np.sqrt(np.dot(np.squeeze(wfs_denoised)[which], np.squeeze(wfs_denoised)[which]) * np.dot(wfs_to_denoise[which], wfs_to_denoise[which]))
    #CHECK THE CORRELATION BETWEEN THE DENOISED WAVEFORM AND THE RAW WAVEFORM, HALLUCINATION WILL HAVE A SMALL VALUE
    if (np.ptp(np.squeeze(wfs_denoised))<small_threshold) & (d_s_corr<corr_th):
        phase_shifted = 0
        halu_idx = 1
    else:
        phase_shifted = np.argmax(wfs_denoised * spk_sign) - offset + phase_shift
        halu_idx = 0

    return single_chan_denoised, phase_shifted, halu_idx


def multichan_phase_shift_denoise(waveforms, geom, extract_channel_index, SinglChanDenoiser, maxchans=None, CH_N = 384, offset=42):
    N, T, C = waveforms.shape
    DenoisedWF = np.zeros(waveforms.shape)
    if maxchans is None:
        maxchans = waveforms.ptp(1).argmax(1)
        
    x_pitch = np.diff(np.unique(geom[:,0]))[0]
    y_pitch = np.diff(np.unique(geom[:,1]))[0]
    
    
    for i in range(len(maxchans)):
        mcs = int(maxchans[i])
        ci = extract_channel_index[mcs]
        non_nan_idx = np.where(ci<CH_N)[0]
        
        full_wfs = waveforms[i, :, non_nan_idx]#.T
        
        # print(np.shape(full_wfs))

        ci = ci[non_nan_idx]
        l = len(ci)

        # BFS to shift the phase
        ci_graph = dict()
        ci_geom = geom[ci]
        mc_idx = np.where(ci == mcs)[0]
        for ch in range(l):
            group = np.where(((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == y_pitch))|
                               ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 0) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 2 * y_pitch)) |
                               ((np.abs(ci_geom[:,0] - ci_geom[ch,0]) == 2 * x_pitch) & (np.abs(ci_geom[:,1] - ci_geom[ch,1]) == 0)))[0]
            ci_graph[ch] = group
            
        # print(mcs)
        # print(ci)


        mc_neighbor_idx = np.concatenate((ci_graph[mc_idx[0]], mc_idx))
        
        
        wfs_mc_neighbors = torch.swapaxes(waveforms[i,:, mc_neighbor_idx], 0, 1)[None, :, :]
        # print(np.shape(wfs_mc_neighbors))
        wfs_denoised_mc_neighbors = SinglChanDenoiser(torch.FloatTensor(wfs_mc_neighbors).reshape(-1, 121)).reshape(wfs_mc_neighbors.shape)
        wfs_denoised_mc_neighbors = wfs_denoised_mc_neighbors.detach().numpy()
        # print(np.shape(wfs_denoised_mc_neighbors))
        try:
            real_maxCH = mc_neighbor_idx[np.argmax(np.ptp(wfs_denoised_mc_neighbors, 2))]
        except:
            print(np.shape(ci_graph[mc_idx[0]]))
            print(np.shape(wfs_mc_neighbors))
            print(np.shape(wfs_denoised_mc_neighbors))
            real_maxCH = mc_idx[0]
        # print(np.shape(real_maxCH))    
        mcs_idx = np.squeeze(real_maxCH)

        for ch in range(l):
            group = ci_graph[ch]
            if (len(np.where(group > mcs_idx)[0])!=0) & (len(np.where(group < mcs_idx)[0])!=0) & (ch!= mcs_idx):
                if ch>mcs_idx:
                    ci_graph[ch] = np.append(group[group>mcs_idx], mcs_idx)
                else:
                    ci_graph[ch] = np.append(group[group<mcs_idx], mcs_idx)

        previous_ch_idx = mcs_idx


        spk_denoised_wfs = np.zeros((T, l))

        full_wfs = waveforms[i, :, non_nan_idx]

        mcs_wfs = full_wfs[:, mcs_idx]

        wfs = mcs_wfs[None, None, :]
        wfs_denoised = SinglChanDenoiser(torch.FloatTensor(wfs).reshape(-1, 121)).reshape(wfs.shape)
        wfs_denoised = np.squeeze(wfs_denoised.detach().numpy())
        # print(np.shape(wfs_denoised))
        spk_denoised_wfs[:,mcs_idx] = wfs_denoised

        mcs_phase_shift = np.argmax(np.abs(wfs_denoised)) - offset


        spk_sign = np.sign(wfs_denoised[offset + mcs_phase_shift])
        
        CH_checked = np.zeros(l)
        CH_phase_shift = np.zeros(l)
        parent = np.zeros(l) * np.nan

        parent_peak_phase = np.zeros(l)
        CH_phase_shift[mcs_idx] = mcs_phase_shift

        wfs_ptp = np.zeros(l)
        halluci_idx = np.zeros(l)
        wfs_ptp[mcs_idx] = np.ptp(wfs_denoised)
        CH_checked[mcs_idx] = 1
        q = []
        q.append(int(mcs_idx))

        while len(q)>0:
            u = q.pop()
            v = ci_graph[u]

            for k in v:
                if CH_checked[k] == 0:
                    neighbors = ci_graph[k]
                    checked_neighbors = neighbors[CH_checked[neighbors] == 1]
                    
                    phase_shift_ref = np.argmax(wfs_ptp[checked_neighbors])

                    threshold = max(0.3* wfs_ptp[mcs_idx], 3)

                    rest_phase_shift = np.delete(parent_peak_phase, checked_neighbors[phase_shift_ref])
                    if (wfs_ptp[checked_neighbors[phase_shift_ref]] > threshold) & (np.min(np.abs(rest_phase_shift - CH_phase_shift[checked_neighbors[phase_shift_ref]]))<=5):
                        parent_peak_phase[k] = CH_phase_shift[checked_neighbors[phase_shift_ref]]
                    else:
                        parent_peak_phase[k] = 0

                    spk_denoised_wfs[:,k], CH_phase_shift[k], halluci_idx[k] = denoise_with_phase_shift(full_wfs[:,k], int(parent_peak_phase[k]), SinglChanDenoiser, k, spk_sign)
                    parent[k] = checked_neighbors[phase_shift_ref]
                    wfs_ptp[k] = np.ptp(spk_denoised_wfs[:,k])
                    q.insert(0, k)
                    CH_checked[k] = 1
                    
            if np.sum(halluci_idx[v])>=3:
                q_partial = v.tolist()
                while len(q_partial)>0:
                    x = q_partial.pop()
                    y = ci_graph[x]
                    for z in y:
                        if CH_checked[z] == 0:
                            CH_checked[z] = 1
                            q_partial.insert(0,z)
                            halluci_idx[z] = 1
                            
        DenoisedWF[i,:,non_nan_idx] = spk_denoised_wfs.T
    return DenoisedWF

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


def enforce_decrease(waveform, max_chan=None, in_place=False):
    n_chan = waveform.shape[1]
    wf = waveform if in_place else waveform.copy()
    ptp = wf.ptp(0)
    if max_chan is None:
        max_chan = ptp.argmax()

    max_chan_even = max_chan - max_chan % 2
    max_chan_odd = max_chan_even + 1

    len_reg = (max_chan_even - 2) // 2
    if len_reg > 0:
        regularizer = np.zeros(len_reg)
        max_ptp = ptp[max_chan_even - 2]
        for i in range(len_reg):
            max_ptp = min(max_ptp, ptp[max_chan_even - 4 - 2 * i])
            regularizer[len_reg - i - 1] = (
                ptp[max_chan_even - 4 - 2 * i] / max_ptp
            )
        wf[:, np.arange(0, max_chan_even - 2, 2)] /= regularizer

    len_reg = (n_chan - 1 - max_chan_even - 2) // 2
    if len_reg > 0:
        regularizer = np.zeros(len_reg)
        max_ptp = ptp[max_chan_even + 2]
        for i in range(len_reg):
            max_ptp = min(max_ptp, ptp[max_chan_even + 4 + 2 * i])
            regularizer[i] = ptp[max_chan_even + 4 + 2 * i] / max_ptp
        wf[:, np.arange(max_chan_even + 4, n_chan, 2)] /= regularizer

    len_reg = (max_chan_odd - 2) // 2
    if len_reg > 0:
        regularizer = np.zeros(len_reg)
        max_ptp = ptp[max_chan_odd - 2]
        for i in range(len_reg):
            max_ptp = min(max_ptp, ptp[max_chan_odd - 4 - 2 * i])
            regularizer[len_reg - i - 1] = (
                ptp[max_chan_odd - 4 - 2 * i] / max_ptp
            )
        wf[:, np.arange(1, max_chan_odd - 2, 2)] /= regularizer

    len_reg = (n_chan - 1 - max_chan_odd - 2) // 2
    if len_reg > 0:
        regularizer = np.zeros(len_reg)
        max_ptp = ptp[max_chan_odd + 2]
        for i in range(len_reg):
            max_ptp = min(max_ptp, ptp[max_chan_odd + 4 + 2 * i])
            regularizer[i] = ptp[max_chan_odd + 4 + 2 * i] / max_ptp
        wf[:, np.arange(max_chan_odd + 4, n_chan, 2)] /= regularizer

    return wf


def enforce_decrease_np1(waveform, max_chan=None, in_place=False):
    n_chan = waveform.shape[1]
    wf = waveform if in_place else waveform.copy()
    ptp = wf.ptp(0)
    if max_chan is None:
        max_chan = ptp[16:28].argmax() + 16

    max_chan_a = max_chan - max_chan % 4
    for i in range(4, max_chan_a, 4):
        if wf[:, max_chan_a - i - 4].ptp() > wf[:, max_chan_a - i].ptp():
            wf[:, max_chan_a - i - 4] = (
                wf[:, max_chan_a - i - 4]
                * wf[:, max_chan_a - i].ptp()
                / wf[:, max_chan_a - i - 4].ptp()
            )
    for i in range(4, n_chan - max_chan_a - 4, 4):
        if wf[:, max_chan_a + i + 4].ptp() > wf[:, max_chan_a + i].ptp():
            wf[:, max_chan_a + i + 4] = (
                wf[:, max_chan_a + i + 4]
                * wf[:, max_chan_a + i].ptp()
                / wf[:, max_chan_a + i + 4].ptp()
            )

    max_chan_b = max_chan - max_chan % 4 + 1
    for i in range(4, max_chan_b, 4):
        if wf[:, max_chan_b - i - 4].ptp() > wf[:, max_chan_b - i].ptp():
            wf[:, max_chan_b - i - 4] = (
                wf[:, max_chan_b - i - 4]
                * wf[:, max_chan_b - i].ptp()
                / wf[:, max_chan_b - i - 4].ptp()
            )
    for i in range(4, n_chan - max_chan_b - 3, 4):
        if wf[:, max_chan_b + i + 4].ptp() > wf[:, max_chan_b + i].ptp():
            wf[:, max_chan_b + i + 4] = (
                wf[:, max_chan_b + i + 4]
                * wf[:, max_chan_b + i].ptp()
                / wf[:, max_chan_b + i + 4].ptp()
            )

    max_chan_c = max_chan - max_chan % 4 + 2
    for i in range(4, max_chan_c, 4):
        if wf[:, max_chan_c - i - 4].ptp() > wf[:, max_chan_c - i].ptp():
            wf[:, max_chan_c - i - 4] = (
                wf[:, max_chan_c - i - 4]
                * wf[:, max_chan_c - i].ptp()
                / wf[:, max_chan_c - i - 4].ptp()
            )
    for i in range(4, n_chan - max_chan_c - 2, 4):
        if wf[:, max_chan_c + i + 4].ptp() > wf[:, max_chan_c + i].ptp():
            wf[:, max_chan_c + i + 4] = (
                wf[:, max_chan_c + i + 4]
                * wf[:, max_chan_c + i].ptp()
                / wf[:, max_chan_c + i + 4].ptp()
            )

    max_chan_d = max_chan - max_chan % 4 + 3
    for i in range(4, max_chan_d, 4):
        if wf[:, max_chan_d - i - 4].ptp() > wf[:, max_chan_d - i].ptp():
            wf[:, max_chan_d - i - 4] = (
                wf[:, max_chan_d - i - 4]
                * wf[:, max_chan_d - i].ptp()
                / wf[:, max_chan_d - i - 4].ptp()
            )
    for i in range(4, n_chan - max_chan_d - 3, 4):
        if wf[:, max_chan_d + i + 4].ptp() > wf[:, max_chan_d + i].ptp():
            wf[:, max_chan_d + i + 4] = (
                wf[:, max_chan_d + i + 4]
                * wf[:, max_chan_d + i].ptp()
                / wf[:, max_chan_d + i + 4].ptp()
            )

    return wf


def make_shell(channel, geom, n_jumps=1):
    """See make_shells"""
    pt = geom[channel]
    dists = cdist([pt], geom).ravel()
    radius = np.unique(dists)[1 : n_jumps + 1][-1]
    return np.setdiff1d(np.flatnonzero(dists <= radius + 1e-8), [channel])


def make_shells(geom, n_jumps=1):
    """Get the neighbors of a channel within a radius

    That radius is found by figuring out the distance to the closest channel,
    then the channel which is the next closest (but farther than the closest),
    etc... for n_jumps.

    So, if n_jumps is 1, it will return the indices of channels which are
    as close as the closest channel. If n_jumps is 2, it will include those
    and also the indices of the next-closest channels. And so on...

    Returns
    -------
    shell_neighbors : list
        List of length geom.shape[0] (aka, the number of channels)
        The ith entry in the list is an array with the indices of the neighbors
        of the ith channel.
        i is not included in these arrays (a channel is not in its own shell).
    """
    return [make_shell(c, geom, n_jumps=n_jumps) for c in range(geom.shape[0])]


def make_radial_order_parents(
    geom, channel_index, n_jumps_per_growth=1, n_jumps_parent=3
):
    """Pre-computes a helper data structure for enforce_decrease_shells"""
    n_channels = len(channel_index)

    # which channels should we consider as possible parents for each channel?
    shells = make_shells(geom, n_jumps=n_jumps_parent)

    radial_parents = []
    for channel, neighbors in enumerate(channel_index):
        channel_parents = []

        # the closest shell will do nothing
        already_seen = [channel]
        shell0 = make_shell(channel, geom, n_jumps=n_jumps_per_growth)
        already_seen += sorted(c for c in shell0 if c not in already_seen)

        # so we start at the second jump
        jumps = 2
        while len(already_seen) < (neighbors < n_channels).sum():
            # grow our search -- what are the next-closest channels?
            new_shell = make_shell(
                channel, geom, n_jumps=jumps * n_jumps_per_growth
            )
            new_shell = list(
                sorted(
                    c
                    for c in new_shell
                    if (c not in already_seen) and (c in neighbors)
                )
            )

            # for each new channel, find the intersection of the channels
            # from previous shells and that channel's shell in `shells`
            for new_chan in new_shell:
                parents = np.intersect1d(shells[new_chan], already_seen)
                parents_rel = np.flatnonzero(np.isin(neighbors, parents))
                if not len(parents_rel):
                    # this can happen for some strange geometries
                    # in that case, let's just bail.
                    continue
                channel_parents.append(
                    (np.flatnonzero(neighbors == new_chan).item(), parents_rel)
                )

            # add this shell to what we have seen
            already_seen += new_shell
            jumps += 1

        radial_parents.append(channel_parents)

    return radial_parents


def enforce_decrease_shells(
    waveforms, maxchans, radial_parents, in_place=False
):
    """Radial enforce decrease"""
    N, T, C = waveforms.shape
    assert maxchans.shape == (N,)

    # compute original ptps and allocate storage for decreasing ones
    is_torch = False
    if torch.is_tensor(waveforms):
        orig_ptps = (waveforms.max(dim=1).values - waveforms.min(dim=1).values).cpu().numpy()
        is_torch = True
    else:
        orig_ptps = waveforms.ptp(axis=1)
    decreasing_ptps = orig_ptps.copy()

    # loop to enforce ptp decrease
    for i in range(N):
        decr_ptp = decreasing_ptps[i]
        for c, parents_rel in radial_parents[maxchans[i]]:
            if decr_ptp[c] > decr_ptp[parents_rel].max():
                decr_ptp[c] *= decr_ptp[parents_rel].max() / decr_ptp[c]

    # apply decreasing ptps to the original waveforms
    rescale = (decreasing_ptps / orig_ptps)[:, None, :]
    if is_torch:
        rescale = torch.as_tensor(rescale, device=waveforms.device)
    if in_place:
        waveforms *= rescale
    else:
        waveforms = waveforms * rescale

    return waveforms


def enforce_temporal_decrease(
    waveforms,
    left=20,
    right=100,
    trough_offset=42,
    in_place=False,
):
    """Enforce monotonicity of abs values at the edges

    Finds the peaks to the left and right of the trough, and
    makes sure we decrease to either side of those.
    """
    N, T, C = waveforms.shape
    waveforms = waveforms.transpose(0, 2, 1).reshape(N * C, T)

    if not in_place:
        waveforms = waveforms.copy()

    if left > 0:
        # not good to do this in a data-driven way
        # because of collisions
        # left_peaks = waveforms[:, :trough_offset].argmax(1)
        left_peaks = np.full(N * C, left)
        _enforce_temporal_decrease_left(
            waveforms, left_peaks
        )

    if right is not None and (right < T):
        # right_peaks = waveforms[:, trough_offset:].argmax(1)
        right_peaks = np.full(N * C, right)
        _enforce_temporal_decrease_right(
            waveforms, right_peaks
        )

    waveforms = waveforms.reshape(N, C, T).transpose(0, 2, 1)

    return waveforms


@torch.no_grad()
def cleaned_waveforms(
    waveforms,
    spike_index,
    firstchans,
    residual,
    s_start=0,
    tpca_rank=7,
    pbar=True,
):
    N, T, C = waveforms.shape
    denoiser = SingleChanDenoiser().load()
    cleaned = np.empty((N, C, T), dtype=waveforms.dtype)
    ixs = (
        trange(len(spike_index), desc="Cleaning and denoising")
        if pbar
        else range(len(spike_index))
    )
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


# %%
def denoise_wf_nn_tmp_single_channel(wf, denoiser, device):
    denoiser = denoiser.to(device)
    n_data, n_times, n_chans = wf.shape
    if wf.shape[0] > 0:
        wf_reshaped = wf.transpose(0, 2, 1).reshape(-1, n_times)
        wf_torch = torch.FloatTensor(wf_reshaped).to(device)
        denoised_wf = denoiser(wf_torch).data
        denoised_wf = denoised_wf.reshape(n_data, n_chans, n_times)
        denoised_wf = denoised_wf.cpu().data.numpy().transpose(0, 2, 1)

        del wf_torch
    else:
        denoised_wf = np.zeros(
            (wf.shape[0], wf.shape[1] * wf.shape[2]), "float32"
        )

    return denoised_wf
