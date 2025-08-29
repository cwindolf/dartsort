from dataclasses import replace
from logging import getLogger

import numpy as np
from dartsort.util import drift_util


logger = getLogger(__name__)


def superres_sorting(
    sorting,
    geom,
    motion_est=None,
    strategy="motion_estimate",
    localizations_dataset_name="point_source_localizations",
    superres_bin_size_um=10.0,
    min_spikes_per_bin=5,
    probe_margin_um=200.0,
):
    """Construct the spatially superresolved spike train

    The superres spike train assigns new labels to each unit
    to capture shape variation due to vertical drift. Each
    unit is broken into several sub-units, and spikes are
    assigned into sub-units (superres units) depending on
    their position and estimated displacement.

    This function just needs to compute superres bin labels;
    the pitch shifting is taken care of during template
    computation.

    There is a strategy flag here to allow experiments
    with how this behaves.

    Strategies:
     - motion_estimate : bin spikes purely based on the drift
       estimate. if the motion estimate is entirely correct,
       then that means we should bin by the displacement estimate
       mod the pitch.
     - drift_pitch_loc_bin : spikes are coarsely motion corrected
       by integer multiples of the pitch using the motion estimate,
       and then these coarsely corrected positions are used to bin
       for superres


    Returns
    -------
    dict with keys : values
        group_ids : np.array
            Int array such that group_ids[superres_id] is the original id of
            superres unit superres_id
        sorting : DARTsortSorting
        properties : dict of np.array
            Each array in here is of shape (n_superres_units,) and contains
            bookkeeping information about the superres process.
            For instance, for motion_estimate binning, we store the centers
            of the bins (which are modulo pitch).
    """
    pitch = drift_util.get_pitch(geom)
    full_labels = sorting.labels.copy()

    # load spike depths
    # TODO: relying on this index feels wrong
    spike_times_s = sorting.times_seconds
    spike_depths_um = sorting.extra_features[localizations_dataset_name][:, 2]

    # remove spikes far away from the probe
    if probe_margin_um is not None:
        valid = spike_depths_um == np.clip(
            spike_depths_um,
            geom[:, 1].min() - probe_margin_um,
            geom[:, 1].max() + probe_margin_um,
        )
        full_labels[~valid] = -1

    # handle triaging
    kept = np.flatnonzero(full_labels >= 0)
    labels = full_labels[kept]
    spike_times_s = spike_times_s[kept]
    spike_depths_um = spike_depths_um[kept]

    # make superres spike train
    properties = {}
    if strategy in ("none", None):
        superres_to_original = np.arange(labels.max() + 1)
        superres_sorting = sorting
    elif strategy == "motion_estimate":
        superres_labels, superres_to_original, bin_centers = motion_estimate_strategy(
            labels,
            spike_times_s,
            spike_depths_um,
            pitch,
            motion_est,
            superres_bin_size_um=superres_bin_size_um,
        )
        properties["motion_estimate_bin_centers"] = bin_centers
    elif strategy == "drift_pitch_loc_bin":
        superres_labels, superres_to_original = drift_pitch_loc_bin_strategy(
            labels,
            spike_times_s,
            spike_depths_um,
            pitch,
            motion_est,
            superres_bin_size_um=superres_bin_size_um,
        )
    else:
        raise ValueError(f"Unknown superres {strategy=}")

    # handle too-small units
    superres_labels, superres_to_original = remove_small_superres_units(
        superres_labels, superres_to_original, min_spikes_per_bin=min_spikes_per_bin
    )

    # back to un-triaged label space
    full_labels[kept] = superres_labels
    superres_sorting = replace(sorting, labels=full_labels)
    return dict(
        group_ids=superres_to_original,
        sorting=superres_sorting,
        properties=properties,
    )


def motion_estimate_strategy(
    original_labels,
    spike_times_s,
    spike_depths_um,
    pitch,
    motion_est,
    superres_bin_size_um=10.0,
    bin_round_atol=0.1,
):
    """ """
    # reg_pos = pos - disp, pos = reg_pos + disp
    # so, disp is the motion of spikes relative to fixed probe
    if motion_est is None:
        displacements = np.zeros_like(spike_depths_um)
    else:
        displacements = motion_est.disp_at_s(spike_times_s, spike_depths_um)

    n_bins = pitch // superres_bin_size_um
    remainder = pitch - n_bins * superres_bin_size_um
    if not np.isclose(remainder, 0.0):
        n_bins = n_bins + (remainder > bin_round_atol)
        logger.info(
            f"Superres bin size didn't divide the pitch. Rounding it down from "
            f"{superres_bin_size_um} to {pitch / n_bins}, for {n_bins} bins."
        )
        superres_bin_size_um = pitch / n_bins

    mod_positions = displacements % pitch
    bin_centers = np.arange(n_bins) * superres_bin_size_um + superres_bin_size_um / 2
    bin_ids = mod_positions // superres_bin_size_um
    bin_ids = bin_ids.astype(original_labels.dtype)
    orig_label_and_bin, superres_labels = np.unique(
        np.c_[original_labels, bin_ids], axis=0, return_inverse=True
    )
    superres_to_original = orig_label_and_bin[:, 0]
    bin_centers = bin_centers[orig_label_and_bin[:, 1]]
    return superres_labels, superres_to_original, bin_centers


def drift_pitch_loc_bin_strategy(
    original_labels,
    spike_times_s,
    spike_depths_um,
    pitch,
    motion_est,
    superres_bin_size_um=10.0,
):
    n_pitches_shift = drift_util.get_spike_pitch_shifts(
        spike_depths_um, pitch=pitch, times_s=spike_times_s, motion_est=motion_est
    )
    coarse_reg_depths = spike_depths_um + n_pitches_shift * pitch

    bin_ids = coarse_reg_depths // superres_bin_size_um
    bin_ids = bin_ids.astype(int)
    orig_label_and_bin, superres_labels = np.unique(
        np.c_[original_labels, bin_ids], axis=0, return_inverse=True
    )
    superres_to_original = orig_label_and_bin[:, 0]
    return superres_labels, superres_to_original


def remove_small_superres_units(
    superres_labels, superres_to_original, min_spikes_per_bin
):
    if not min_spikes_per_bin:
        return superres_labels, superres_to_original

    slabels, scounts = np.unique(superres_labels, return_counts=True)

    # new labels
    kept = scounts >= min_spikes_per_bin
    kept_labels = slabels[kept]
    relabeling = np.full_like(slabels, -1)
    relabeling[kept] = np.arange(kept_labels.size)

    # relabel
    superres_labels = relabeling[superres_labels]
    superres_to_original = superres_to_original[kept_labels]

    return superres_labels, superres_to_original
