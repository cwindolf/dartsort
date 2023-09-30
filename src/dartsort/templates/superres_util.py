from dataclasses import replace

import numpy as np
from dartsort.util import drift_util


def superres_sorting(
    sorting,
    spike_times_s,
    spike_depths_um,
    geom,
    motion_est=None,
    strategy="drift_pitch_loc_bin",
    superres_bin_size_um=10.0,
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
    superres_to_original : np.array
        Int array such that superres_to_original[superres_id] is the original id of superres unit
        superres_id
    superres_sorting : DARTsortSorting
    """
    pitch = drift_util.get_pitch(geom)
    labels = sorting.labels

    if strategy in ("none", None):
        superres_to_original = np.arange(labels.max() + 1)
        superres_sorting = sorting
    elif strategy == "motion_estimate":
        superres_labels, superres_to_original = motion_estimate_strategy(
            labels,
            spike_times_s,
            spike_depths_um,
            pitch,
            motion_est,
            superres_bin_size_um=superres_bin_size_um,
        )
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

    superres_sorting = replace(sorting, labels=superres_labels)
    return superres_to_original, superres_sorting


def motion_estimate_strategy(
    original_labels,
    spike_times_s,
    spike_depths_um,
    pitch,
    motion_est,
    superres_bin_size_um=10.0,
):
    """ """
    # reg_pos = pos - disp, pos = reg_pos + disp
    # so, disp is the motion of spikes relative to fixed probe
    displacements = motion_est.disp_at_s(spike_times_s, spike_depths_um)
    mod_positions = displacements % pitch
    bin_ids = mod_positions // superres_bin_size_um
    orig_label_and_bin, superres_labels = np.unique(
        np.c_[original_labels, bin_ids], axis=0, return_inverse=True
    )
    superres_to_original = orig_label_and_bin[:, 0]
    return superres_labels, superres_to_original


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
    orig_label_and_bin, superres_labels = np.unique(
        np.c_[original_labels, bin_ids], axis=0, return_inverse=True
    )
    superres_to_original = orig_label_and_bin[:, 0]
    return superres_labels, superres_to_original
