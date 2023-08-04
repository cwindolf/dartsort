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
import warnings

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
    min_distance = pdist(registered_geom).min() / 2
    distances, registered_channels = registered_kdtree.query(
        shifted_positions, distance_upper_bound=min_distance
    )
    # make sure there were no unmatched points
    assert np.all(registered_channels < len(registered_geom))

    return registered_channels


def registered_average(
    waveforms,
    n_pitches_shift,
    geom,
    registered_geom,
    main_channels=None,
    channel_index=None,
    reducer=np.nanmedian,
    pad_value=0.0,
):
    static_waveforms = get_waveforms_on_static_channels(
        waveforms,
        geom,
        main_channels=main_channels,
        channel_index=channel_index,
        n_pitches_shift=n_pitches_shift,
        registered_geom=registered_geom,
        fill_value=np.nan,
    )

    # take the mean and return
    # suppress all-NaN slice warning
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        average = reducer(static_waveforms, axis=0)
    if not np.isnan(pad_value):
        average = np.nan_to_num(average, copy=False, nan=pad_value)

    return average


def get_spike_pitch_shifts(
    depths_um, geom, registered_depths_um=None, times_s=None, motion_est=None
):
    """Figure out coarse pitch shifts based on spike positions

    Determine the number of pitches the probe would need to shift in
    order to coarsely align a waveform to its registered position.
    """
    pitch = get_pitch(geom)
    if registered_depths_um is None:
        displacement = -motion_est.disp_at_s(times_s, depths_um)
    else:
        displacement = registered_depths_um - depths_um

    # if displacement > 0, then the registered position is below the original
    # and, to be conservative, round towards 0 rather than using //
    n_pitches_shift = (displacement / pitch).astype(int)

    return n_pitches_shift


# -- waveform channel neighborhood shifting


def get_waveforms_on_static_channels(
    waveforms,
    geom,
    main_channels=None,
    channel_index=None,
    target_channels=None,
    n_pitches_shift=None,
    registered_geom=None,
    fill_value=np.nan,
):
    """Load a set of drifting waveforms on a static set of channels

    Arguments
    ---------
    waveforms : (n_spikes, t (optional), c) array
    geom : (n_channels_tot, probe_dim) array
        Original channel positions
    main_channels : int (n_spikes,) array
        `waveforms[i]` lives on channels `channel_index[main_channels[i]]`
        in the original geometry array `geom`
    channel_index : int (n_channels_tot, c) array
    target_channels : int (n_channels_target,) array
        Optional subset of channels to restrict to
        `waveforms[i]` will be restricted to the intersection of its channels
        and this channel subset, after correcting from drift
    n_pitches_shift : int (n_spikes,) array
        The number of probe pitches (see `get_pitch`) by which each spike's
        channel neighborhood will be shifted before matching with target channels
    registered_geom : (n_registered_channels, probe_dim) array
        If supplied, the target channel positions are loaded from here
    fill_value : float
        The value to impute when a target channel does not land on a channel
        when shifted according to n_pitches_shift

    Returns
    -------
    static_waveforms : (n_spikes, t (optional), n_channels_target) array
    """
    # this supports amplitude vectors (i.e., 2d arrays)
    two_d = waveforms.ndim == 2

    # validate inputs to avoid confusing errors
    assert waveforms.ndim in (2, 3)
    if two_d:
        waveforms = waveforms[:, None, :]
    n_spikes, t, c = waveforms.shape
    assert geom.ndim == 2
    n_channels_tot = geom.shape[0]
    if main_channels is not None:
        assert channel_index is not None
        n_channels_tot_, c_ = channel_index.shape
        assert c_ == c and n_channels_tot_ == n_channels_tot
        assert main_channels.shape == (n_spikes,)
    else:
        # if per-wf channels are not supplied, then we assume
        # that waveforms live on all channels
        assert c == n_channels_tot
    if n_pitches_shift is not None:
        assert n_pitches_shift.shape == (n_spikes,)

    # grab the positions of the channels that we are targeting
    target_geom = geom
    if registered_geom is not None:
        target_geom = registered_geom
    if target_channels is not None:
        target_geom = target_geom[target_channels]

    # figure out the positions of the channels that the waveforms live on
    pitch = get_pitch(geom)
    if n_pitches_shift is None:
        n_pitches_shift = np.zeros(n_spikes)
    shifts = np.c_[np.zeros(n_spikes), n_pitches_shift * pitch]
    if main_channels is None:
        # the case where all waveforms live on all channels, but
        # these channels may be shifting
        # shape is n_spikes, len(geom), spatial dim
        moving_positions = geom[None, :, :] + shifts[:, None, :]
    else:
        # the case where each waveform lives on its own channels
        # nans will never be matched in k-d query below
        padded_geom = np.pad(
            geom.astype(float), [(0, 1), (0, 0)], constant_values=np.nan
        )
        # shape is n_spikes, c, spatial dim
        moving_positions = (
            padded_geom[channel_index[main_channels]] + shifts[:, None, :]
        )

    # find where each moving position lands using a k-d tree
    target_kdtree = KDTree(target_geom)
    match_distance = pdist(geom).min() / 2
    moving_positions = moving_positions.reshape(n_spikes * c, geom.shape[1])
    valid_chan = ~np.isnan(moving_positions).any(axis=1)
    shifted_channels = np.full(n_spikes * c, target_kdtree.n)
    _, shifted_channels[valid_chan] = target_kdtree.query(
        moving_positions[valid_chan],
        distance_upper_bound=match_distance,
    )
    shifted_channels = shifted_channels.reshape(n_spikes, c)

    # scatter the waveforms into their static channel neighborhoods
    n_static_channels = len(target_geom)
    static_waveforms = np.full(
        (n_spikes, t, n_static_channels + 1),
        fill_value=fill_value,
        dtype=waveforms.dtype,
    )
    spike_ix = np.arange(n_spikes)[:, None, None]
    time_ix = np.arange(t)[None, :, None]
    chan_ix = shifted_channels[:, None, :]
    static_waveforms[spike_ix, time_ix, chan_ix] = waveforms
    static_waveforms = static_waveforms[:, :, :n_static_channels]

    if two_d:
        static_waveforms = static_waveforms[:, 0, :]

    return static_waveforms