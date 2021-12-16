import numpy as np
import h5py
import torch
from torch import nn
from tqdm.auto import tqdm, trange

from . import waveform_utils


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

    def load(self, fname_model):
        checkpoint = torch.load(fname_model, map_location="cpu")
        self.load_state_dict(checkpoint)


def spike_train_to_index(spike_train, templates):
    """Convert a kilosort spike train to a spike index

    KS spike train contains (sample, id) pairs. Spike index
    contains (sample, max channel) pairs.

    Output times are min PTP times, output max chans are KS
    template max chans.
    """
    n_templates = templates.shape[0]
    template_ptps = templates.ptp(1)
    template_maxchans = template_ptps.argmax(1)

    cluster_ids = spike_train[:, 1]
    template_offsets = np.argmin(
        templates[np.arange(n_templates), :, template_maxchans],
        axis=1,
    )
    spike_offsets = template_offsets[cluster_ids] - 42
    start_times = spike_train[:, 0] + spike_offsets

    spike_index = np.c_[start_times, template_maxchans[cluster_ids]]
    return spike_index


@torch.no_grad()
def get_denoised_waveforms(
    standardized_bin,
    spike_index,
    geom,
    channel_radius=10,
    denoiser_weights_path="../pretrained/single_chan_denoiser.pt",
    T=121,
    threshold=0,
    dtype=np.float32,
    geomkind="updown",
    batch_size=128,
    device=None,
):
    num_channels = geom.shape[0]
    standardized = np.memmap(standardized_bin, dtype=dtype, mode="r")
    standardized = standardized.reshape(-1, num_channels)

    # load denoiser
    denoiser = SingleChanDenoiser()
    denoiser.load(denoiser_weights_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    denoiser.to(device)

    # spike times are min PTP times, which need to be centered in our output
    # also, make sure we don't read past the edge of the file
    read_times = spike_index[:, 0] - T // 2
    good = np.flatnonzero(
        (read_times >= 0) & (read_times + T < standardized.shape[0])
    )
    read_times = read_times[good]
    spike_index = spike_index[good]

    # helper function for data loading
    def get_batch(start, end):
        nonlocal standardized
        times = read_times[start:end]
        maxchans = spike_index[start:end, 1]
        inds = good[start:end]
        waveforms = np.stack(
            [standardized[t : t + T] + 0 for t in times],
            axis=0,
        )
        waveforms_trimmed, firstchans = waveform_utils.get_local_waveforms(
            waveforms,
            channel_radius,
            geom,
            maxchans=maxchans,
            geomkind=geomkind,
            compute_firstchans=True,
        )
        return waveforms_trimmed, inds, firstchans

    # -- initialize variables for main loop
    # we probably won't find this many spikes that cross the threshold,
    # but we can use it to allocate storage
    max_n_spikes = len(spike_index)
    C = 2 * channel_radius + 2 * (geomkind == "standard")
    raw_waveforms = np.empty(
        (max_n_spikes, T, C),
        dtype=dtype,
    )
    denoised_waveforms = np.empty(
        (max_n_spikes, T, C),
        dtype=dtype,
    )
    count = 0  # how many spikes have exceeded the threshold?
    indices = np.empty(max_n_spikes, dtype=int)
    firstchans = np.empty(max_n_spikes, dtype=int)

    # main loop
    for i in trange(max_n_spikes // batch_size + 1):
        start = i * batch_size
        end = min(max_n_spikes, (i + 1) * batch_size)
        batch_wfs, batch_inds, batch_firstchans = get_batch(start, end)
        batch_wfs_ = torch.as_tensor(
            batch_wfs.transpose(0, 2, 1), device=device
        )

        n_batch = batch_wfs.shape[0]
        if n_batch:
            denoised_batch = denoiser(batch_wfs_.reshape(-1, T)).cpu().numpy()
            denoised_batch = denoised_batch.reshape(n_batch, C, T)
            denoised_batch = denoised_batch.transpose(0, 2, 1)
            
            big = range(n_batch)
            n_big = n_batch
            if threshold > 0:
                big = denoised_batch.ptp(1).max(1) > threshold
                n_big = big.sum()

            if n_big:
                raw_waveforms[count : count + n_big] = batch_wfs[big]
                denoised_waveforms[count : count + n_big] = denoised_batch[big]
                indices[count : count + n_big] = batch_inds[big]
                firstchans[count : count + n_big] = batch_firstchans[big]
                count += n_big

    # trim the places we did not fill
    raw_waveforms = raw_waveforms[:count]
    denoised_waveforms = denoised_waveforms[:count]
    indices = indices[:count]
    firstchans = firstchans[:count]

    return raw_waveforms, denoised_waveforms, indices, firstchans
