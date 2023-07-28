import numpy as np
import h5py
import torch
from tqdm.auto import trange

from . import waveform_utils
from .denoise import SingleChanDenoiser


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
    T=121,
    threshold=0,
    dtype=np.float32,
    geomkind="updown",
    pad_for_denoiser=0,
    batch_size=128,
    device=None,
    inmem=True,
):
    assert "firstchan" not in geomkind

    num_channels = geom.shape[0]
    standardized = np.memmap(standardized_bin, dtype=dtype, mode="r")
    standardized = standardized.reshape(-1, num_channels)

    # load denoiser
    denoiser = SingleChanDenoiser()
    denoiser.load()
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
        times = read_times[start:end]
        maxchans = spike_index[start:end, 1]
        inds = good[start:end]
        waveforms = np.stack(
            [standardized[t : t + T] + 0 for t in times],
            axis=0,
        )
        waveforms_trimmed, firstchans = waveform_utils.get_local_waveforms(
            waveforms,
            channel_radius + pad_for_denoiser,
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
    if inmem:
        raw_waveforms = np.empty(
            (max_n_spikes, T, C),
            dtype=dtype,
        )
        denoised_waveforms = np.empty(
            (max_n_spikes, T, C),
            dtype=dtype,
        )
    else:
        print("Working out of core. Clear the temp files when you're done.")
        temp = h5py.File("___tmp.h5", "w")
        raw_waveforms = temp.create_dataset(
            name="raw",
            shape=(1, T, C),
            maxshape=(max_n_spikes, T, C),
            dtype=dtype,
        )
        denoised_waveforms = temp.create_dataset(
            name="denoised",
            shape=(1, T, C),
            maxshape=(max_n_spikes, T, C),
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
        if not n_batch:
            continue

        denoised_batch = denoiser(batch_wfs_.reshape(-1, T)).cpu().numpy()
        denoised_batch = denoised_batch.reshape(
            n_batch, C + 2 * pad_for_denoiser, T
        )
        denoised_batch = denoised_batch.transpose(0, 2, 1)

        big = range(n_batch)
        n_big = n_batch
        if threshold > 0:
            big = denoised_batch.ptp(1).max(1) > threshold
            n_big = big.sum()

        if not n_big:
            continue

        batch_wfs = batch_wfs[big]
        denoised_batch = denoised_batch[big]
        batch_inds = batch_inds[big]
        batch_firstchans = batch_firstchans[big]

        if pad_for_denoiser:
            denoised_maxchans = denoised_batch.ptp(1).argmax(1)
            denoised_maxchans -= denoised_maxchans % 2
            low = np.maximum(0, denoised_maxchans - channel_radius)
            low = np.minimum(2 * pad_for_denoiser, low)
            batch_wfs = np.stack(
                [batch_wfs[i, :, low[i] : low[i] + C] for i in range(n_batch)],
                axis=0,
            )
            denoised_batch = np.stack(
                [
                    denoised_batch[i, :, low[i] : low[i] + C]
                    for i in range(n_batch)
                ],  # noqa
                axis=0,
            )
            batch_firstchans += low

        if not inmem:
            raw_waveforms.resize(count + n_big, axis=0)
            denoised_waveforms.resize(count + n_big, axis=0)

        raw_waveforms[count : count + n_big] = batch_wfs
        denoised_waveforms[count : count + n_big] = denoised_batch
        indices[count : count + n_big] = batch_inds
        firstchans[count : count + n_big] = batch_firstchans
        count += n_big

    # trim the places we did not fill
    if inmem:
        raw_waveforms = raw_waveforms[:count]
        denoised_waveforms = denoised_waveforms[:count]
    indices = indices[:count]
    firstchans = firstchans[:count]

    return raw_waveforms, denoised_waveforms, indices, firstchans
