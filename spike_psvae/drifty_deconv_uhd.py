import h5py
import numpy as np
import tempfile
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing
from . import deconvolve, snr_templates, spike_train_utils, reassignment
from .waveform_utils import get_pitch, pitch_shift_templates
from .extract_deconv import extract_deconv


def superres_spike_train(
    spike_train, z_abs, bin_size_um, t_end=100,
    units_spread=None, min_spikes_bin=None, n_spikes_max_recent = 1000, fs=30000,
):
    """
    remove min_spikes_bin by default - it's worse to end up with a template that is mean of all spikes!!!
    units_spread is the spread of each registered clusters np.std(z_reg[spt[:, 1]==unit])*1.65 
    """
    assert spike_train.shape == (*z_abs.shape, 2)
    assert bin_size_um > 0


    # bin the spikes to create a binned "superres spike train"
    # we'll use this spike train in an expanded label space to compute templates
    superres_labels = np.full_like(spike_train[:, 1], -1)
    # this will keep track of which superres template corresponds to which bin,
    # information which we will need later to determine how to shift the templates
    superres_label_to_bin_id = []
    superres_label_to_orig_label = []
    unit_labels = np.unique(spike_train[spike_train[:, 1] >= 0, 1])
    medians_at_computation = np.zeros(unit_labels.max()+1)
    cur_superres_label = 0
    for u in unit_labels:
        in_u = np.flatnonzero(spike_train[:, 1] == u)

        # Get most recent spikes
        count_unit = np.logical_and(spike_train[:, 0]<t_end*fs, spike_train[:, 1]==u).sum()
        if count_unit>n_spikes_max_recent:
            in_u = np.flatnonzero(np.logical_and(spike_train[:, 0]<t_end*fs, 
                                                           spike_train[:, 1]==u))[-n_spikes_max_recent:]
        else:
            in_u = np.flatnonzero(spike_train[:, 1]==u)[:n_spikes_max_recent]

        # center the z positions in this unit using the median
        centered_z = z_abs[in_u].copy()
        medians_at_computation[u] = np.median(centered_z)
        centered_z -= medians_at_computation[u]

        # convert them to bin identities by adding half the bin size and
        # floor dividing by the bin size
        # this corresponds to bins like:
        #      ... | bin -1 | bin 0 | bin 1 | ...
        #   ... -3bin/2 , -bin/2, bin/2, 3bin/2, ...
        bin_ids = (centered_z + bin_size_um / 2) // bin_size_um
        occupied_bins, bin_counts = np.unique(bin_ids, return_counts=True)
        if units_spread is not None:
            # np.abs(bin_ids) <= (np.abs(centered_z)+ bin_size_um / 2)//bin_size_um <= (max_z_dist + bin_size_um / 2)//bin_size_um
            bin_counts = bin_counts[
                np.abs(occupied_bins)
                <= (units_spread[u] + bin_size_um / 2) // bin_size_um
            ]
            occupied_bins = occupied_bins[
                np.abs(occupied_bins)
                <= (units_spread[u] + bin_size_um / 2) // bin_size_um
            ]
        # IS THAT NEEDED - min_spikes_bin=6
        if min_spikes_bin is None:
            for bin_id in occupied_bins:
                superres_labels[in_u[bin_ids == bin_id]] = cur_superres_label
                superres_label_to_bin_id.append(bin_id)
                superres_label_to_orig_label.append(u)
                cur_superres_label += 1
        else:
            if bin_counts.max() >= min_spikes_bin:
                for bin_id in occupied_bins[bin_counts >= min_spikes_bin]:
                    superres_labels[in_u[bin_ids == bin_id]] = cur_superres_label
                    superres_label_to_bin_id.append(bin_id)
                    superres_label_to_orig_label.append(u)
                    cur_superres_label += 1
            # what if no template was computed for u
            else:
                superres_labels[in_u] = cur_superres_label
                superres_label_to_bin_id.append(0)
                superres_label_to_orig_label.append(u)
                cur_superres_label += 1

    superres_label_to_bin_id = np.array(superres_label_to_bin_id)
    superres_label_to_orig_label = np.array(superres_label_to_orig_label)
    return (
        superres_labels,
        superres_label_to_bin_id,
        superres_label_to_orig_label,
        medians_at_computation
    )


def superres_denoised_templates(
    spike_train,
    z_abs,
    bin_size_um,
    geom,
    raw_binary_file,
    t_end=100,
    min_spikes_bin=None,
    units_spread=None,
    max_spikes_per_unit=200, #per superres unit 
    n_spikes_max_recent = 1000,
    denoise_templates=True,
    do_temporal_decrease=True,
    zero_radius_um=70, #reduce this value in uhd compared to NP1/NP2
    reducer=np.median,
    snr_threshold=5.0 * np.sqrt(100),
    spike_length_samples=121,
    trough_offset=42,
    do_tpca=True,
    tpca=None,
    tpca_rank=5,
    tpca_radius=75,
    tpca_n_wfs=50_000,
    fs=30000,
    pbar=True,
    seed=0,
    n_jobs=-1,
):

    (
        superres_labels,
        superres_label_to_bin_id,
        superres_label_to_orig_label,
        medians_at_computation
    ) = superres_spike_train(
        spike_train,
        z_abs,
        bin_size_um,
        t_end,
        units_spread,
        min_spikes_bin,
        n_spikes_max_recent,
        fs,
    )

    templates, extra = snr_templates.get_templates(
        np.c_[spike_train[:, 0], superres_labels],
        geom,
        raw_binary_file,
        max_spikes_per_unit=max_spikes_per_unit,
        do_temporal_decrease=do_temporal_decrease,
        zero_radius_um=zero_radius_um,
        reducer=reducer,
        snr_threshold=snr_threshold,
        spike_length_samples=spike_length_samples,
        trough_offset=trough_offset,
        do_tpca=do_tpca,
        tpca=tpca,
        tpca_rank=tpca_rank,
        tpca_radius=tpca_radius,
        tpca_n_wfs=tpca_n_wfs,
        pbar=pbar,
        seed=seed,
        n_jobs=n_jobs,
        raw_only=not denoise_templates,
    )
    return (
        templates,
        superres_label_to_bin_id,
        superres_label_to_orig_label,
        medians_at_computation
    )




def shift_superres_templates(
    time_bin,
    superres_templates,
    superres_label_to_bin_id,
    superres_label_to_orig_label,
    bin_size_um,
    geom,
    positions_over_time_clusters,
    medians_at_computation,
    fill_value=0.0,
):

    """
    This version shifts by every (possible - if enough templates) mod 
    """
    pitch = get_pitch(geom)
    bins_per_pitch = pitch / bin_size_um
    # if bins_per_pitch != int(bins_per_pitch):
    #     raise ValueError(
    #         f"The pitch of this probe is {pitch}, but the bin size "
    #         f"{bin_size_um} does not evenly divide it."
    #     )
    shifted_templates = superres_templates.copy()


    #shift every unit separately
    for unit in np.unique(superres_label_to_orig_label):
        # shift in bins, rounded towards 0
        bins_shift = np.round((positions_over_time_clusters[unit, time_bin] - medians_at_computation[unit])/bin_size_um)
        if bins_shift!=0:
            # How to do the shifting?
            # We break the shift into two pieces: the number of full pitches,
            # and the remaining bins after shifting by full pitches.
            n_pitches_shift = int(
                bins_shift / bins_per_pitch
            )  # want to round towards 0, not //

            bins_shift_rem = bins_shift - bins_per_pitch * n_pitches_shift

            # Now, first we do the pitch shifts
            shifted_templates_unit = pitch_shift_templates(
                n_pitches_shift, geom, superres_templates[superres_label_to_orig_label==unit], fill_value=fill_value
            )
            # Now, do the mod shift bins_shift_rem
            # IDEA: take the bottom bin and shift it above - 
            # If more than bins_per_pitch templates - OK, can shift 
            # Only special case np.abs(bins_shift_rem)<=pitch/2 and n_temp <=pitch/2 -> better not to shift (no information gain)

            n_temp = (superres_label_to_orig_label==unit).sum()
            if bins_shift_rem<0:
                if bins_shift_rem<-pitch/2 or n_temp>pitch/2:
                    idx_mod_shift = np.flatnonzero(np.isin(superres_label_to_bin_id[superres_label_to_orig_label==unit], superres_label_to_bin_id[superres_label_to_orig_label==unit].min()-np.arange(-bins_shift_rem)+pitch))
                    n_temp_shift = len(idx_mod_shift)
                    shifted_templates_unit[-n_temp_shift:] = pitch_shift_templates(
                        -1, geom, shifted_templates_unit[idx_mod_shift], fill_value=fill_value
                    ) 

                    # The rest of the shift is handled by updating bin ids
                    # This part doesn't matter for the recovered spike train, since
                    # the template doesn't change, but it could matter for z tracking
                    superres_label_to_bin_id[superres_label_to_orig_label==unit] = np.roll(superres_label_to_bin_id[superres_label_to_orig_label==unit], -len(idx_mod_shift))
            elif bins_shift_rem>0:
                if bins_shift_rem>pitch/2 or n_temp>pitch/2:
                    idx_mod_shift = np.flatnonzero(np.isin(superres_label_to_bin_id[superres_label_to_orig_label==unit], superres_label_to_bin_id[superres_label_to_orig_label==unit].max()+np.arange(bins_shift_rem)-pitch))
                    n_temp_shift = len(idx_mod_shift)
                    shifted_templates_unit[:n_temp_shift] = pitch_shift_templates(
                        1, geom, shifted_templates_unit[idx_mod_shift], fill_value=fill_value
                    ) #shift by 1 pitch as we taked templates that are at max()+shift-pitch
                    # update bottom templates - we remove <= "space" at the bottom than we add on top

                    # The rest of the shift is handled by updating bin ids
                    # This part doesn't matter for the recovered spike train, since
                    # the template doesn't change, but it could matter for z tracking

                    # !!! That's an approximation - maybe we'll need to change if we do z tracking, shoul;d be fine for now - is ok if we have bins that are "continuous" per unit
                    superres_label_to_bin_id[superres_label_to_orig_label==unit] = np.roll(superres_label_to_bin_id[superres_label_to_orig_label==unit], n_temp_shift)
            shifted_templates[superres_label_to_orig_label==unit]=shifted_templates_unit

    return shifted_templates, superres_label_to_bin_id
