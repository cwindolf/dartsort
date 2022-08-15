import numpy as np
from . import spikeio


def make_labels_contiguous(labels, in_place=False, return_unique=False):
    untriaged = np.flatnonzero(labels >= 0)
    unique, contiguous = np.unique(labels[untriaged], return_inverse=True)
    out = labels if in_place else labels.copy()
    out[untriaged] = contiguous
    if return_unique:
        return out, unique
    return out


def clean_align_and_get_templates(
    spike_train,
    geom,
    bin_file,
    max_memory_mb=2048,
    reducer=np.median,
    max_shift=3,
    min_n_spikes=0,
    n_samples=250,
    spike_length_samples=121,
    trough_offset=42,
    pbar=False,
    seed=0,
):
    """
    A helper function for cleaning and aligning spike trains
    that returns aligned templates.

    Removes small clusters, computes padded templates to determine
    alignment, aligns, and crops templates.

    Returns
    -------
    aligned_spike_train : np.array, same shape as spike_train
    templates : of shape (aligned_spike_train[:, 1].max(), spike_len_samples, n_channels)
    """
    rg = np.random.default_rng(seed)

    # remove small units, make labels contiguous
    units, counts = np.unique(spike_train[:, 1], return_counts=True)
    if min_n_spikes > 0:
        too_small = np.isin(spike_train[:, 1], units[counts < min_n_spikes])
        spike_train[too_small, 1] = -1
    spike_train[:, 1] = make_labels_contiguous(spike_train[:, 1])
    times, labels = spike_train.T
    n_units = labels.max() + 1

    # pad for shift detection
    spike_length_load = spike_length_samples + 2 * max_shift
    trough_offset_load = trough_offset + max_shift

    # how many units can we process at once?
    spike_bytes = np.dtype(np.float32).itemsize * geom.shape[0] * spike_length_load
    n_spikes_at_once = (max_memory_mb * 1_000_000) // spike_bytes
    units_at_once = n_spikes_at_once // n_samples
    units_at_once = max(1, units_at_once)
    buffer = np.empty((units_at_once * n_samples, spike_length_load, geom.shape[0]))
    print(units_at_once, "units at once")

    # we will iterate through chunks of labels
    starts = range(0, n_units, units_at_once)
    if pbar:
        starts = tqdm(starts, desc="Aligned templates")

    for start_label in starts:
        end_label = min(n_units, start_label + units_at_once)
        labels_batch = np.arange(start_label, end_label)

        # which spikes to load?
        # need to be careful to keep track of which spike belongs to which
        # unit before and after sorting
        load_labels = []
        load_times = []
        for label in labels_batch:
            which = np.flatnonzero(spike_train[:, 1] == label)
            choice = rg.choice(which, size=min(which.size, n_samples), replace=False)
            load_labels.append(np.full_like(choice, label))
            load_times.append(times[choice])
        load_labels = np.concatenate(load_labels)
        load_times = np.concatenate(load_times)

        # now we know which sorted load time is in which unit
        sort = np.argsort(load_times)
        load_times = load_times[sort]
        load_labels = load_labels[sort]

        # load padded waveforms
        spikeio.read_waveforms(
            load_times,
            bin_file,
            geom.shape[0],
            spike_length_samples=spike_length_load,
            trough_offset=trough_offset_load,
            buffer=buffer,
        )
        waveforms = buffer[:len(load_times)]