import numpy as np
from tqdm.auto import tqdm
from . import spikeio, localize_index


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
    sort_by_time=True,
    max_shift=0,
    n_samples=250,
    spike_length_samples=121,
    trough_offset=42,
    pbar=True,
    seed=0,
    dtype=np.float32,
    remove_empty_units=True,
    remove_double_counted=False,
    order_units_by_z=False,
    geom=None,
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

    # clean spike train: remove small units, make labels contiguous
    units, counts = np.unique(aligned_spike_train[:, 1], return_counts=True)
    print(
        (aligned_spike_train[:, 1].max() + 1) - (units.size - 1),
        "inactive units",
    )
    if min_n_spikes > 0:
        too_small_units = units[counts < min_n_spikes]
        # don't want to touch the triaged spikes
        too_small_units = too_small_units[too_small_units >= 0]
        too_small = np.isin(aligned_spike_train[:, 1], too_small_units)
        print(
            f"Spike train cleaning will remove {too_small_units.size} "
            f"active units with < {min_n_spikes} spikes"
        )
        # mark these spikes as triaged
        # (rather than deleting, to keep the same shape for the spike train)
        aligned_spike_train[too_small, 1] = -1

    if remove_empty_units:
        make_labels_contiguous(aligned_spike_train[:, 1], in_place=True)

    if remove_double_counted:
        # if a unit has the same spike twice, triage it away
        # False by default because this should be used with caution!
        # only a good idea when we are confident that we
        # have well-isolated units.
        units = np.unique(aligned_spike_train[:, 1])
        for unit in units[units >= 0]:
            in_unit = np.flatnonzero(aligned_spike_train[:, 1] == unit)
            ust = aligned_spike_train[in_unit]
            unique_times, times_index = np.unique(ust, return_index=True)
            to_remove = ~np.isin(np.arange(len(ust)), times_index)
            aligned_spike_train[in_unit[to_remove], 1] = -1

    aligned_spike_train, templates, template_shifts = align_by_templates(
        n_channels,
        bin_file,
        aligned_spike_train,
        max_shift=max_shift,
        n_samples=n_samples,
        spike_length_samples=spike_length_samples,
        trough_offset=trough_offset,
        reducer=reducer,
        seed=seed,
        dtype=dtype,
        pbar=pbar,
        in_place=True,
    )

    # sort so that times are increasing, but keep track of the order
    # so that the caller can handle bookkeeping
    if sort_by_time:
        order = np.argsort(aligned_spike_train[:, 0], kind="stable")
        aligned_spike_train = aligned_spike_train[order]
    else:
        order = np.arange(len(aligned_spike_train))

    if order_units_by_z:
        assert geom is not None
        _, _, _, tz, _ = localize_index.localize_ptps_index(
            templates.ptp(1),
            geom,
            templates.ptp(1).argmax(1),
            np.array([np.arange(geom.shape[0])] * geom.shape[0]),
            radius=100,
        )
        zord = np.argsort(tz)
        templates = templates[zord]
        template_shifts = template_shifts[zord]
        zord_inv = np.argsort(zord)
        zord_inv = np.concatenate([[-1], zord_inv], axis=0)
        aligned_spike_train = np.c_[
            aligned_spike_train[:, 0],
            zord_inv[1 + aligned_spike_train[:, 1]],
        ]

    return aligned_spike_train, order, templates, template_shifts


def align_by_templates(
    n_channels,
    bin_file,
    spike_train,
    max_shift=0,
    n_samples=250,
    spike_length_samples=121,
    trough_offset=42,
    reducer=np.median,
    seed=0,
    dtype=np.float32,
    pbar=True,
    in_place=False,
):
    spike_train = spike_train if in_place else spike_train.copy()

    n_units = spike_train[:, 1].max() + 1

    # randomness used when sampling spikes for templates
    rg = np.random.default_rng(seed)

    # pad for shift detection
    spike_length_load = spike_length_samples + 2 * max_shift
    trough_offset_load = trough_offset + max_shift

    # we'll store the final templates, not the padded ones
    templates = np.zeros(
        (n_units, spike_length_samples, n_channels),
        dtype=dtype,
    )

    # a padded waveform storage buffer
    buffer = np.empty((n_samples, spike_length_load, n_channels), dtype=dtype)

    # we will iterate through chunks of labels
    units = range(n_units)
    if pbar:
        units = tqdm(units, desc="Align and get templates")
    template_shifts = np.zeros(n_units, dtype=int)
    for unit in units:
        in_unit = np.flatnonzero(spike_train[:, 1] == unit)
        if not in_unit.size:
            continue

        # pick waveforms
        to_load = rg.choice(
            in_unit, size=min(n_samples, in_unit.size), replace=False
        )

        # load padded waveforms
        waveforms, skipped = spikeio.read_waveforms(
            spike_train[to_load, 0],
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
        trough = np.abs(template[:, template_mc]).argmax()
        # shift is actual trough - desired trough
        # so, if shift > 0, we need to subtract it
        shift = trough - trough_offset_load
        if abs(shift) > max_shift:
            shift = 0
        if shift != 0:
            spike_train[in_unit, 0] += shift
        template_shifts[unit] = shift

        # crop aligned template and store it
        # we use a + here not a -!
        # subtracting means moving the origin to the right
        templates[unit] = template[
            max_shift + shift : max_shift + shift + spike_length_samples
        ]

    return spike_train, templates, template_shifts
