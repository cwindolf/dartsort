"""Utility functions for dealing with drifting channels

The main concept here is the "registered geometry" made by the
function `registered_geometry`. The idea is to extend the
probe geometry to cover the range of drift experienced in the
recording. The probe's pitch (unit at which its geometry repeats
vertically) is the integer unit at which we shift channels when
extending the geometry, so that the registered probe contains the
original probe as a subset, as well as copies of the probe shifted
by integer numbers of pitches. As many shifted copies are created
as needed to capture all the drift.
"""
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

from .waveform_util import get_pitch

# -- registered geometry and templates helpers


def registered_geometry(
    geom,
    motion_est=None,
    upward_drift=None,
    downward_drift=None,
):
    """Extend the probe's channel positions according to the range of motion

    The probe will be extended upwards according to the amount of upward
    drift and downward according to the amount of downward drift, where
    here we mean drift of the probe relative to the spikes.

    But, the convention for motion estimation is the opposite: motion of
    the spikes relative to the probe. In other words, the displacement
    in motion_est is such that
        registered z = original z - displacement
    So, if the probe moved up from t1->t2 (so that z(t2) < z(t1), since z
    is the spike's absolute position on the probe), this would register
    as a negative displacement. So, upward motion of the probe means a
    negative displacement value estimated from spikes!
    """
    assert geom.ndim == 2
    pitch = get_pitch(geom)

    # figure out how much upward and downward motion there is
    if motion_est is not None:
        # these look flipped! why? see __doc__
        downward_drift = max(0, motion_est.displacement.max())
        upward_drift = max(0, -motion_est.displacement.min())
    else:
        assert upward_drift is not None
        assert downward_drift is not None
    assert upward_drift >= 0
    assert downward_drift >= 0

    # pad with an integral number of pitches for simplicity
    pitches_pad_up = int(np.ceil(upward_drift / pitch))
    pitches_pad_down = int(np.ceil(downward_drift / pitch))

    # we have to be careful about floating point error here
    # two sites may be different due to floating point error
    # we know they are the same if their distance is smaller than:
    min_distance = pdist(geom, metric="sqeuclidean").min() / 2

    # find all registered site positions
    # TODO make this not quadratic
    unique_shifted_positions = list(geom)
    for shift in range(-pitches_pad_down, pitches_pad_up + 1):
        shifted_geom = geom + [0, pitch * shift]
        for site in shifted_geom:
            if not any(
                np.square(p - site).sum() < min_distance
                for p in unique_shifted_positions
            ):
                unique_shifted_positions.append(site)
    unique_shifted_positions = np.array(unique_shifted_positions)

    # order by depth first, then horizontal position (unique goes the other way)
    registered_geom = unique_shifted_positions[
        np.lexsort(unique_shifted_positions.T)
    ]
    assert np.isclose(get_pitch(registered_geom), pitch)

    return registered_geom


def registered_channels(channels, geom, n_pitches_shift, registered_geom):
    """What registered channels do `channels` land on after shifting by `n_pitches_shift`?"""
    pitch = get_pitch(geom)
    shifted_positions = geom.copy()[channels]
    shifted_positions[:, 1] += n_pitches_shift * pitch

    registered_kdtree = KDTree(registered_geom)
    min_distance = pdist(registered_geom, metric="sqeuclidean").min() / 2
    distances, registered_channels = registered_kdtree.query(
        shifted_positions, distance_upper_bound=min_distance
    )
    # make sure there were no unmatched points
    assert np.all(registered_channels < len(registered_geom))

    return registered_channels


def registered_channel_index(channel_index, geom, registered_geom):
    """ """


def get_spike_pitch_shifts(times_s, depths_um, geom, motion_est):
    """"""
    pass


def registered_average():
    pass


# -- waveform channel neighborhood shifting helpers


def shifted_channel_neighborhood(
    n_pitches_shift,
    target_channels,
    geom,
    registered_geom=None,
):
    """Determine a drifting channel neighborhood

    If the channels in target_channels shifted by n_pitches_shift
    pitches, what channels would they land on?

    These are the channels at the positions of the target_channels,
    displaced by n_pitches_shift * pitch. These might not exist,
    so the shifted channel neighborhood could be smaller than the
    target channel neighborhood. In that case, channels which were
    missed are replaced with n_channels_tot.

    Arguments
    ---------
    n_pitches_shift : int
    target_channels : 1d integer array
    geom : (n_channels_tot, 2) array

    Returns
    -------
    shifted_channels : 1d integer array of size == target_channels.size
    """
    pitch = get_pitch(geom)
    if registered_geom is None:
        registered_geom = geom
    offset = [0, n_pitches_shift * pitch]
    shifted_geom = geom + offset
    target_positions = registered_geom[target_channels]
    shifted_channels = []
    for target in target_positions:
        match = np.flatnonzero(np.isclose(shifted_geom, target).all(axis=1))
        if not match.size:
            shifted_channels.append(len(geom))
        elif match.size == 1:
            shifted_channels.append(match[0])
        else:
            assert False

    shifted_channels = np.array(shifted_channels)
    return shifted_channels


def get_waveforms_on_shifted_channel_subset(
    waveforms,
    main_channels,
    channel_index,
    target_channels,
    n_pitches_shift,
    geom,
    registered_geom=None,
    fill_value=np.nan,
):
    """Load a set of waveforms on a static subset of channels, even under drift

    waveforms[i] lives on the channels in channel_index[main_channels[i]], and we
    want to load it on a target set of channels `target_channels`. That's easy to
    do without drift: just restrict to the intersection of target_channels and
    channel_index[main_channels[i]] (say, using waveform_util.get_channel_subset).
    But with drift, we no longer want to load a static subset of channels. Rather,
    each waveform needs to be loaded on a subset of channels determined by
    n_pitches_shift[i], obtained by the function `shifted_channel_neighborhood`

    This function restricts each waveform to the intersection of its channels,
    channel_index[main_channels[i]], and the channels at the positions of
    target_channels shifted vertically by n_pitches_shift[i] * pitch

    Arguments
    ---------
    waveforms : (n_spikes, t (optional), c) array
    main_channels : int (n_spikes,) array
    channel_index : int (n_channels_tot, c) array
    target_channels : int (n_channels_target,) array
    n_pitches_shift : int (n_spikes,) array
    geom : (n_channels_tot, probe_dim) array
    fill_value : float
        The value to impute when a target channel does not land on a channel
        when shifted according to n_pitches_shift

    Returns
    -------
    out_waveforms : (n_spikes, t (optional), n_channels_target) array
    """
    # validate inputs to avoid confusing errors
    assert waveforms.ndim in (2, 3)
    # this also supports amplitude vectors (i.e., 2d arrays)
    two_d = waveforms.ndim == 2
    if two_d:
        waveforms = waveforms[:, None, :]
    n_spikes, t, c = waveforms.shape
    n_channels_tot, c_ = channel_index.shape
    assert c_ == c
    assert main_channels.shape == (n_spikes,)
    assert n_pitches_shift.shape == (n_spikes,)
    assert geom.ndim == 2 and geom.shape[0] == n_channels_tot
    assert target_channels.ndim == 1
    if registered_geom is None:
        registered_geom = geom

    out_waveforms = np.full(
        (n_spikes, t, target_channels.size),
        fill_value,
        dtype=waveforms.dtype,
    )

    # figure out what shifts are being performed
    pitches_shift_uniq = np.unique(n_pitches_shift)

    # restrict each set of waveforms to the correct shifted channels
    for i, pitch_shift in enumerate(pitches_shift_uniq):
        in_batch = np.flatnonzero(n_pitches_shift == pitch_shift)
        shifted_target_channels = shifted_channel_neighborhood(
            pitch_shift,
            target_channels,
            geom,
            registered_geom=registered_geom,
        )
        out_waveforms[in_batch] = get_waveforms_on_fixed_channel_subset(
            waveforms[in_batch],
            main_channels=main_channels[in_batch],
            channel_index=channel_index,
            target_channels=shifted_target_channels,
            fill_value=fill_value,
        )

    if two_d:
        out_waveforms = out_waveforms[:, 0, :]

    return out_waveforms


def get_waveforms_on_fixed_channel_subset(
    waveforms,
    main_channels,
    channel_index,
    target_channels,
    fill_value=np.nan,
):
    """Restrict waveforms on varying channels to a fixed channel neighborhood

    waveforms[i] lived on channel_index[main_channels[i]]?
    Well, now it lives on target_channels with its friends j \\neq i.

    Arguments
    ---------
    waveforms : (n_spikes, t (optional), c) array
    main_channels : int (n_spikes,) array
    channel_index : int (n_channels_tot, c) array
    target_channels : int (n_channels_target,) array
    fill_value : float
        The value to impute where a target channel is not present in
        channel_index[main_channels[i]]

    Returns
    -------
    out_waveforms : (n_spikes, t (optional), n_channels_target) array
    """
    # validate inputs to avoid confusing errors
    assert waveforms.ndim in (2, 3)
    # this also supports amplitude vectors (i.e., 2d arrays)
    two_d = waveforms.ndim == 2
    if two_d:
        waveforms = waveforms[:, None, :]
    n_spikes, t, c = waveforms.shape
    n_channels_tot, c_ = channel_index.shape
    assert c_ == c
    assert main_channels.shape == (n_spikes,)
    assert main_channels.dtype.kind == "i"
    assert target_channels.ndim == 1

    out_waveforms = np.full(
        (n_spikes, t, target_channels.size),
        fill_value,
        dtype=waveforms.dtype,
    )

    for i in range(n_spikes):
        my_target_chans, targets_found = np.nonzero(
            channel_index[main_channels[i]].reshape(-1, 1)
            == target_channels.reshape(1, -1)
        )
        out_waveforms[i, :, targets_found] = waveforms[i, :, my_target_chans]

    if two_d:
        out_waveforms = out_waveforms[:, 0, :]

    return out_waveforms
