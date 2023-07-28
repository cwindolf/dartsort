r"""Free templates

See `free_templates.md` for lots of info.
"""
import numpy as np
from .waveform_utils import get_pitch


# -- geometry utilities
# making the padded virtual probe
# extracting individual shifted templates from a padded template
# etc


def is_lexsorted(geom):
    return np.array_equal(geom, np.lexsort(geom))


def virtual_probe(geom):
    """Make the virtual probe

    This is the probe padded on both sides by P-1 pitches. There
    can be missing channels, and we want to fill in all of the gaps,
    """
    assert geom.ndim == 2 and geom.shape[1] == 2
    assert is_lexsorted(geom)
    pitch = get_pitch(geom)
    n_pitches = np.ceil(np.ptp(geom[:, 1]) / pitch).astype(int)

    all_shifted_geoms = np.array(
        [geom + [[0, k * pitch]] for k in np.arange(-n_pitches + 1, n_pitches)]
    )
    virtual_geom = np.unique(all_shifted_geoms.reshape(-1, 2), axis=0)

    return virtual_geom


def virtual_channels_at_pitch_shift(virtual_geom, geom, n_pitches_shift):
    """What channels on the virtual probe should be loaded at a pitch shift?
    """
    pitch = get_pitch(geom)
    shifted_geom = geom + [[0, n_pitches_shift * pitch]]

    # what channels does this shifted original geom match in the virtual probe?
    geom_matching = (virtual_geom[:, None, :] == shifted_geom[None, :, :]).all(axis=2)
    # each virtual channel matches at most one shifted real channel
    # and that each shifted real channel has a match
    assert geom_matching.sum() == geom.shape[0]
    assert geom_matching.max(axis=1) == 1

    virtual_chans, matched_orig_chans = np.nonzero(geom_matching)
    # since everything is lexsorted, the matching should give the
    # original channels in their original order
    assert np.array_equal(matched_orig_chans, np.arange(geom.shape[0]))

    return virtual_chans


def padded_virtual_waveform(waveform, virtual_geom, geom, n_pitches_shift, fill_value=np.nan):
    """Pad a waveform to the virtual probe, offset by n_pitches_shift pitches
    """
    assert waveform.shape[1] == geom.shape[0]
    virtual_chans = virtual_channels_at_pitch_shift(virtual_geom, geom, n_pitches_shift)

    padded_waveform = np.full(
        (waveform.shape[0], virtual_geom.shape[0]),
        fill_value,
        dtype=waveform.dtype,
    )
    padded_waveform[:, virtual_chans] = waveform

    return padded_waveform


def get_templates_on_shifted_channels(templates, virtual_geom, geom, n_pitches_shift):
    """Get the templates on the real geometry at the current pitch shift
    """
    single = templates.ndim == 2
    if single:
        templates = templates[None]
    assert templates.shape[2] == virtual_geom.shape[0]

    virtual_chans = virtual_channels_at_pitch_shift(virtual_geom, geom, n_pitches_shift)
    shifted_templates = templates[:, :, virtual_chans]

    if single:
        shifted_templates = shifted_templates[0]

    return shifted_templates


# -- computing free templates


def free_spike_train(labels, z_abs, z_reg, bin_size_um, geom, min_spikes_bin=10, max_z_dist=None, mode="hybrid"):
    """Compute k_i and b_i from the writeup, get superres unit ids, etc

    This gets all of the info needed to compute templates, and returns
    all associated metadata about which final superres unit goes to which
    original unit, what pitch shifts and superres bins are associated with
    the templates, etc.
    """
    assert labels.shape == z_abs.shape == z_reg.shape == (labels.shape[0],)

    pitch = get_pitch(geom)

    # p_i in the writeup
    disps = z_abs - z_reg

    # original label set
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]

    # pitch shift for each spike -- k_i in the writeup
    shift_ids = np.zeros_like(labels)
    # superres bin for each spike -- b_i in the writeup
    superres_bin_ids = np.zeros_like(labels)
    # free labels: spikes that are in the same unit came from the
    # same original unit and have the same superres bin id
    free_labels = np.full_like(labels, -1)
    # maps free labels to their superres bin id
    free_label_to_superres_bin = []
    # maps free labels to their original unit id
    free_label_to_orig_label = []
    # unit "home position" \bar{r} which tells us how to shift it in deconv
    reg_means = []

    # now, for each unit, we will figure out the pitch shift,
    # superres bin id, and free label for all spikes
    for u in unique_labels:
        in_unit = np.flatnonzero(labels == u)

        # \bar{r} in the writeup
        reg_mean = np.median(z_reg[in_unit])
        reg_means.append(reg_mean)
        disp_mean = np.median(disps[in_unit])

        if mode in ("z", "p"):
            # o_i in the writeup
            if mode == "z":
                offsets = z_abs[in_unit] - reg_mean
            else:
                offsets = disps[in_unit] - disp_mean

            # k_i and s_i
            pitch_shifts_in_unit = (offsets + pitch / 2) // pitch
            sub_pitch_drift = offsets - pitch_shifts_in_unit * pitch
            assert (sub_pitch_drift >= 0).all()
            # this corresponds to bins like:
            #      ... | bin -1 | bin 0 | bin 1 | ...
            #   ... -3bin/2 , -bin/2, bin/2, 3bin/2, ...
            superres_bin_ids_in_unit = (sub_pitch_drift + bin_size_um / 2) // bin_size_um
        elif mode == "hybrid":
            # k_i and s_i
            pitch_shifts_in_unit = (disps[in_unit] - disp_mean + pitch / 2) // pitch
            sub_pitch_drift = z_abs[in_unit] - reg_mean - pitch_shifts_in_unit * pitch
            # this corresponds to bins like:
            #      ... | bin -1 | bin 0 | bin 1 | ...
            #   ... -3bin/2 , -bin/2, bin/2, 3bin/2, ...
            superres_bin_ids_in_unit = (sub_pitch_drift + bin_size_um / 2) // bin_size_um
        else:
            raise ValueError(f"Unknown mode {mode}.")

        superres_bin_ids[in_unit] = superres_bin_ids_in_unit

        # make sure bins have enough spikes and remove far away bins
        occupied_bins, bin_counts = np.unique(superres_bin_ids_in_unit, return_counts=True)
        within_z_dist = occupied_bins * bin_size_um <= max_z_dist + bin_size_um / 2

        for ix in np.flatnonzero((bin_counts >= min_spikes_bin) & within_z_dist):
            



