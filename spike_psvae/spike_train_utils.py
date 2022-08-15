import numpy as np
from tqdm.auto import tqdm
from . import spikeio


def make_labels_contiguous(
    labels, in_place=False, return_orig_unit_labels=False
):
    """Remove empty units."""
    assert labels.ndim == 1
    out = labels if in_place else labels.copy()
    del labels  # I cannot modify now it unless in place

    untriaged = np.flatnonzero(out >= 0)
    unique, contiguous = np.unique(out[untriaged], return_inverse=True)
    out[untriaged] = contiguous

    if return_orig_unit_labels:
        return out, unique
    return out


def clean_align_and_get_templates(
    spike_train,
    n_channels,
    bin_file,
    min_n_spikes=0,
    reducer=np.median,
    max_shift=3,
    n_samples=250,
    spike_length_samples=121,
    trough_offset=42,
    pbar=False,
    seed=0,
    dtype=np.float32,
):
    """
    A helper function for cleaning and aligning spike trains
    that returns aligned templates.

    Removes small clusters, computes padded templates to determine
    alignment, aligns, and crops templates.

    This will change the label space if there are empty units, or
    units with fewer than min_n_spikes spikes.

    This will change the order of the spikes if realignment occurs

    Returns
    -------
    aligned_spike_train : np.array, same shape as spike_train
    order : the argsort after aligning
    templates : of shape (aligned_spike_train[:, 1].max(), spike_len_samples, n_channels)
    """
    aligned_spike_train = spike_train.copy()
    del spike_train  # we cannot modify original now

    # randomness used when sampling spikes for templates
    rg = np.random.default_rng(seed)

    # clean spike train: remove small units, make labels contiguous
    units, counts = np.unique(aligned_spike_train[:, 1], return_counts=True)
    if min_n_spikes > 0:
        too_small_units = units[counts < min_n_spikes]
        # don't want to touch the triaged spikes
        too_small_units = too_small_units[too_small_units >= 0]
        too_small = np.isin(aligned_spike_train[:, 1], too_small_units)
        print(
            f"Spike train cleaning will remove {too_small_units.size} "
            f"units with < {min_n_spikes} spikes"
        )
        # mark these spikes as triaged
        # (rather than deleting, to keep the same shape for the spike train)
        aligned_spike_train[too_small, 1] = -1
    make_labels_contiguous(aligned_spike_train[:, 1], in_place=True)
    times, labels = aligned_spike_train.T
    n_units = labels.max() + 1

    # pad for shift detection
    spike_length_load = spike_length_samples + 2 * max_shift
    trough_offset_load = trough_offset + max_shift

    # we'll store the final templates, not the padded ones
    templates = np.zeros(
        (n_units, spike_length_samples, n_channels),
        dtype=dtype,
    )

    # a padded waveform storage buffer
    buffer = np.empty(
        (n_samples, spike_length_load, n_channels), dtype=dtype
    )

    # we will iterate through chunks of labels
    units = range(n_units)
    if pbar:
        units = tqdm(units, desc="Align and get templates")
    for unit in units:
        in_unit = np.flatnonzero(aligned_spike_train[:, 1] == unit)
        if not in_unit.size:
            continue

        # pick waveforms
        to_load = rg.choice(
            in_unit, size=min(n_samples, in_unit.size), replace=False
        )

        # load padded waveforms
        waveforms, skipped = spikeio.read_waveforms(
            aligned_spike_train[to_load, 0],
            bin_file,
            n_channels,
            spike_length_samples=spike_length_load,
            trough_offset=trough_offset_load,
            buffer=buffer,
            dtype=dtype,
        )

        # find trough misalignment
        template = reducer(waveforms, axis=0)
        template_mc = template.ptp(0).argmax()
        trough = template[:, template_mc].argmin()
        # shift is actual trough - desired trough
        # so, if shift > 0, we need to subtract it
        shift = trough - trough_offset_load
        if abs(shift) > max_shift:
            shift = 0
        if shift != 0:
            aligned_spike_train[in_unit, 0] -= shift

        # crop aligned template and store it
        # we use a + here not a -!
        # subtracting means moving the origin to the right
        templates[unit] = template[
            max_shift + shift : max_shift + shift + spike_length_samples
        ]

    # sort so that times are increasing, but keep track of the order
    # so that the caller can handle bookkeeping
    order = np.argsort(aligned_spike_train[:, 0])
    aligned_spike_train = aligned_spike_train[order]

    return aligned_spike_train, order, templates
