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

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

from .spiketorch import fast_nanmedian
from .waveform_util import get_pitch

# -- registered geometry and templates helpers


def registered_geometry(
    geom,
    motion_est=None,
    upward_drift=None,
    downward_drift=None,
):
    """Pad probe to motion extent

    This adds extra channels to a probe layout, matching the probe's geometry.
    For instance, say we had a tiny NP1-like probe that looks like this:

     o o     -
      o o    | 20um row spacing
     o o     | 60um total height
      o o    -

    Now, say that the probe moves from depth 0 at time 0 up to depth 20um
    relative to the brain tissue at time 10:

                time 0          time 10

                                o o
    depth 40                     o o
                o o             o o
    depth 20     o o             o o
                o o
    depth 0      o o

    In this case, the motion estimate at time 0 would be 0, and the motion estimate
    at time 10 would be 20.

    This function would add new fake "target channels" (marked with xs):

                o o
    depth 20     o o
                o o
    depth 0      o o
                x x
    depth -20    x x

    Later, in interpolation, the motion will be added to these target channels. The
    "o" channels will then always line up with their true positions, and the x channels
    can be handled by extrapolation or nan padding, etc.

    This function works by guessing the vertical gap between channels (called "pitch" here).
    In the NP1-like example above, that's 40um, not 20um, due to the staggered layout. Then,
    the extent of the motion is determined and rounded away from 0 to the nearest pitch.
    Copies of the probe shifted by each pitch in the range of this extent are then created,
    and the unique set of site positions is grabbed to finish.

    Another way of looking at it:
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
    registered_geom = unique_shifted_positions[np.lexsort(unique_shifted_positions.T)]
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
    registered_kdtree=None,
    match_distance=None,
    main_channels=None,
    channel_index=None,
    reducer=fast_nanmedian,
    work_buffer=None,
    pad_value=0.0,
    return_n_samples=False,
):
    static_waveforms = get_waveforms_on_static_channels(
        waveforms,
        geom,
        main_channels=main_channels,
        channel_index=channel_index,
        n_pitches_shift=n_pitches_shift,
        registered_geom=registered_geom,
        target_kdtree=registered_kdtree,
        match_distance=match_distance,
        out=work_buffer,
        fill_value=np.nan,
    )

    if return_n_samples:
        # remove time dim if any
        c = static_waveforms
        if static_waveforms.ndim == 3:
            c = static_waveforms[:, 0, :]
        n_samples = np.isfinite(c).sum(axis=0)

    # take the mean and return
    average = reducer(static_waveforms, axis=0)
    if not np.isnan(pad_value):
        average = np.nan_to_num(average, copy=False, nan=pad_value)

    if return_n_samples:
        return average, n_samples

    return average


def registered_template(
    waveforms,
    n_pitches_shift,
    geom,
    registered_geom,
    registered_kdtree=None,
    match_distance=None,
    min_fraction_at_shift=0.5,
    min_count_at_shift=25,
    pad_value=0.0,
    reducer=fast_nanmedian,
):
    """registered_average of waveforms would be more accurate if reducer=median, but slow."""
    two_d = waveforms.ndim == 2
    if two_d:
        waveforms = waveforms[:, None, :]
    uniq_shifts, counts = np.unique(n_pitches_shift, return_counts=True)
    is_tensor = torch.is_tensor(waveforms)
    if is_tensor:
        drifty_templates = torch.zeros(
            (len(uniq_shifts), *waveforms.shape[1:]),
            dtype=waveforms.dtype,
            device=waveforms.device,
        )
    else:
        drifty_templates = np.zeros(
            (len(uniq_shifts), *waveforms.shape[1:]),
            dtype=waveforms.dtype,
        )
    if is_tensor:
        n_pitches_shift = torch.as_tensor(n_pitches_shift, device=waveforms.device)
    for i, u in enumerate(uniq_shifts):
        drifty_templates[i] = reducer(waveforms[n_pitches_shift == u], axis=0)
    if is_tensor:
        drifty_templates = drifty_templates.numpy(force=True)

    static_templates = get_waveforms_on_static_channels(
        drifty_templates,
        geom,
        n_pitches_shift=uniq_shifts,
        registered_geom=registered_geom,
        target_kdtree=registered_kdtree,
        match_distance=match_distance,
        fill_value=np.nan,
    )

    # weighted mean is easier than weighted median, and we want this to be weighted
    valid = ~np.isnan(static_templates[:, 0, :])
    counts_ = counts[:, None]
    fractions = counts_ / counts.sum()
    valid &= (fractions >= min_fraction_at_shift) | (counts_ >= min_count_at_shift)
    weights = valid[:, None, :] * counts_[:, None, :]
    weights = weights / np.maximum(weights.sum(0), 1)
    template = (np.nan_to_num(static_templates) * weights).sum(0)
    dtype = str(waveforms.dtype).split(".")[1] if is_tensor else waveforms.dtype
    template = template.astype(dtype)
    template[:, ~valid.any(0)] = np.nan
    if not np.isnan(pad_value):
        template = np.nan_to_num(template, copy=False, nan=pad_value)

    assert template.ndim == 2
    if two_d:
        template = template[0, :]

    return template


def get_spike_pitch_shifts(
    depths_um,
    geom=None,
    registered_depths_um=None,
    times_s=None,
    motion_est=None,
    pitch=None,
):
    """Figure out coarse pitch shifts based on spike positions

    Determine the number of pitches the probe would need to shift in
    order to coarsely align a waveform to its registered position.
    """
    if pitch is None:
        pitch = get_pitch(geom)
    if registered_depths_um is None and motion_est is None:
        probe_displacement = np.zeros_like(depths_um)
    elif registered_depths_um is None:
        probe_displacement = -motion_est.disp_at_s(times_s, depths_um)
    else:
        probe_displacement = registered_depths_um - depths_um

    # if probe_displacement > 0, then the registered position is below the original
    # and, to be conservative, round towards 0 rather than using //
    # sometimes nans can sneak in here... let's just give them 0 disps.
    probe_displacement = np.nan_to_num(probe_displacement)
    n_pitches_shift = (probe_displacement / pitch).astype(int)

    return n_pitches_shift


def invert_motion_estimate(motion_est, t_s, registered_depths_um):
    """ """
    assert np.isscalar(t_s)

    if (
        hasattr(motion_est, "spatial_bin_centers_um")
        and motion_est.spatial_bin_centers_um is not None
    ):
        # nonrigid motion
        bin_centers = motion_est.spatial_bin_centers_um
        t_s = np.full(bin_centers.shape, t_s)
        bin_center_disps = motion_est.disp_at_s(t_s, depth_um=bin_centers)
        # registered_bin_centers = motion_est.correct_s(t_s, depths_um=bin_centers)
        registered_bin_centers = bin_centers - bin_center_disps
        assert np.all(np.diff(registered_bin_centers) > 0), "Invertibility issue."
        disps = np.interp(
            registered_depths_um.clip(
                registered_bin_centers.min(),
                registered_bin_centers.max(),
            ),
            registered_bin_centers,
            bin_center_disps,
        )
    else:
        # rigid motion
        disps = motion_est.disp_at_s(t_s)

    return registered_depths_um + disps


# -- waveform channel neighborhood shifting


def get_waveforms_on_static_channels(
    waveforms,
    geom,
    main_channels=None,
    channel_index=None,
    target_channels=None,
    n_pitches_shift=None,
    registered_geom=None,
    target_kdtree=None,
    match_distance=None,
    out=None,
    fill_value=np.nan,
    workers=4,
):
    """Load a set of drifting waveforms on a static set of channels

    Waveforms are by default detected waveforms on the full probe with channel
    locations stored in geom. If main_channels and channel_index are supplied,
    then waveforms[i] appears on the channels indexed by channel_index[main_channels[i]].

    Now, the user wants to extract a subset of channels from each waveform such that
    the same channel in each subsetted waveform is always at the same physical position,
    accounting for drift. Here, the drift is inputted by n_pitches_shift.

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
    if match_distance is None:
        match_distance = pdist(geom).min() / 2

    # grab the positions of the channels that we are targeting
    target_geom = geom
    if registered_geom is not None:
        target_geom = registered_geom
    if target_channels is not None:
        pad_channel = target_geom.max(0) + 10 * match_distance
        target_geom = np.concatenate((target_geom, pad_channel[None]))
        target_geom = target_geom[target_channels]

    # make kdtree
    if target_kdtree is None:
        target_kdtree = KDTree(target_geom)
    else:
        assert target_kdtree.n == len(target_geom)
    n_static_channels = len(target_geom)

    # figure out the positions of the channels that the waveforms live on
    pitch = get_pitch(geom)
    if n_pitches_shift is None:
        n_pitches_shift = np.zeros(n_spikes, dtype=int)
    # shifts = np.c_[np.zeros(n_spikes), n_pitches_shift * pitch]
    if main_channels is None:
        # the case where all waveforms live on all channels, but
        # these channels may be shifting
        # shape is n_spikes, len(geom), spatial dim
        static_waveforms = _full_probe_shifting_fast(
            waveforms,
            geom,
            pitch,
            target_kdtree,
            n_pitches_shift,
            match_distance,
            fill_value,
            out=out,
        )
        if two_d:
            static_waveforms = static_waveforms[:, 0, :]
        return static_waveforms
        # moving_positions = geom[None, :, :] + shifts[:, None, :]
        # valid_chan = slice(None)
    # else:

    # the case where each waveform lives on its own channels
    # nans will never be matched in k-d query below
    padded_geom = np.pad(geom.astype(float), [(0, 1), (0, 0)], constant_values=np.nan)

    # ok, the kdtree query can get expensive when we have lots of these shifting
    # positions. it turns out to be worth it to go through the effort of figuring
    # out what the unique valid moving positions are and just targeting those
    channels_and_shifts = np.c_[main_channels, n_pitches_shift]
    uniq_channels_and_shifts, uniq_inv = np.unique(
        channels_and_shifts, axis=0, return_inverse=True
    )
    uniq_channels, uniq_n_pitches_shift = uniq_channels_and_shifts.T
    uniq_shifts = np.c_[np.zeros(uniq_channels.shape[0]), uniq_n_pitches_shift * pitch]
    uniq_moving_pos = (
        padded_geom[channel_index[uniq_channels]] + uniq_shifts[:, None, :]
    )
    uniq_valid = channel_index[uniq_channels] < n_channels_tot

    _, uniq_shifted_channels = target_kdtree.query(
        uniq_moving_pos[uniq_valid],
        distance_upper_bound=match_distance,
        workers=workers,
    )
    if uniq_shifted_channels.size < uniq_channels.shape[0] * c:
        uniq_shifted_channels_ = uniq_shifted_channels
        uniq_shifted_channels = np.full((uniq_channels.shape[0], c), target_kdtree.n)
        uniq_shifted_channels[uniq_valid] = uniq_shifted_channels_
    uniq_shifted_channels = uniq_shifted_channels.reshape(uniq_channels.shape[0], c)

    # ok, now we can return to non-unique world
    shifted_channels = uniq_shifted_channels[uniq_inv]

    #     # shape is n_spikes, c, spatial dim
    #     moving_positions = padded_geom[channel_index[main_channels]] + shifts[:, None, :]
    #     # valid_chan = ~np.isnan(moving_positions).any(axis=1)
    #     valid_chan = channel_index[main_channels] < n_channels_tot

    #     _, uniq_shifted_channels = target_kdtree.query(
    #         moving_positions[valid_chan[:, :]],
    #         distance_upper_bound=match_distance,
    #         workers=workers,
    #     )
    #     shifted_channels = uniq_shifted_channels[uniq_inv]

    # now, if not all chans were valid, fix that up...
    # if shifted_channels.size < n_spikes * c:
    #     shifted_channels_ = shifted_channels
    #     shifted_channels = np.full((n_spikes, c), target_kdtree.n)
    #     shifted_channels[valid_chan] = shifted_channels_
    # shifted_channels = shifted_channels.reshape(n_spikes, c)

    # scatter the waveforms into their static channel neighborhoods
    if out is None:
        if torch.is_tensor(waveforms):
            static_waveforms = torch.full(
                (n_spikes, t, n_static_channels + 1),
                fill_value=fill_value,
                dtype=waveforms.dtype,
                device=waveforms.device,
            )
        else:
            static_waveforms = np.full(
                (n_spikes, t, n_static_channels + 1),
                fill_value=fill_value,
                dtype=waveforms.dtype,
            )
    else:
        assert out.shape == (n_spikes, t, n_static_channels + 1)
        out.fill(fill_value)
        static_waveforms = out
    spike_ix = np.arange(n_spikes)[:, None, None]
    time_ix = np.arange(t)[None, :, None]
    chan_ix = shifted_channels[:, None, :]
    static_waveforms[spike_ix, time_ix, chan_ix] = waveforms
    static_waveforms = static_waveforms[:, :, :n_static_channels]
    if two_d:
        static_waveforms = static_waveforms[:, 0, :]

    return static_waveforms


def _full_probe_shifting_fast(
    waveforms,
    geom,
    pitch,
    target_kdtree,
    n_pitches_shift,
    match_distance,
    fill_value,
    out=None,
):
    is_tensor = torch.is_tensor(waveforms)

    if out is None:
        if is_tensor:
            static_waveforms = torch.full(
                (*waveforms.shape[:2], target_kdtree.n + 1),
                fill_value=fill_value,
                dtype=waveforms.dtype,
                device=waveforms.device,
            )
        else:
            static_waveforms = np.full(
                (*waveforms.shape[:2], target_kdtree.n + 1),
                fill_value=fill_value,
                dtype=waveforms.dtype,
            )
    else:
        assert out.shape == (*waveforms.shape[:2], target_kdtree.n + 1)
        out.fill(fill_value)
        static_waveforms = out

    no_shift = n_pitches_shift is None
    if no_shift:
        n_pitches_shift = [0]
    unps, shift_inverse = np.unique(n_pitches_shift, return_inverse=True)
    shifted_channels = np.full((len(unps), waveforms.shape[2]), target_kdtree.n)
    for i, nps in enumerate(unps):
        moving_geom = geom + [0, pitch * nps]
        _, shifted_channels[i] = target_kdtree.query(
            moving_geom, distance_upper_bound=match_distance
        )
        # static_waveforms[which[:, None, None], tix[None, :, None], shifted_channels[None, None, :]] = waveforms[which]
    nix = np.arange(waveforms.shape[0])
    tix = np.arange(waveforms.shape[1])
    static_waveforms[
        nix[:, None, None],
        tix[None, :, None],
        shifted_channels[shift_inverse][:, None, :],
    ] = waveforms
    return static_waveforms[:, :, : target_kdtree.n]


def static_channel_neighborhoods(
    geom,
    main_channels,
    channel_index,
    target_channels=None,
    pitch=None,
    n_pitches_shift=None,
    registered_geom=None,
    target_kdtree=None,
    match_distance=None,
    workers=4,
):
    # validate inputs to avoid confusing errors
    n_spikes = main_channels.shape[0]
    assert geom.ndim == 2
    n_channels_tot = geom.shape[0]
    assert channel_index is not None
    n_channels_tot_, c = channel_index.shape
    assert n_channels_tot_ == n_channels_tot
    assert main_channels.shape == (n_spikes,)
    if n_pitches_shift is not None:
        assert n_pitches_shift.shape == (n_spikes,)
    if match_distance is None:
        match_distance = pdist(geom).min() / 2

    # grab the positions of the channels that we are targeting
    target_geom = geom
    if registered_geom is not None:
        target_geom = registered_geom
    if target_channels is not None:
        pad_channel = target_geom.max(0) + 10 * match_distance
        target_geom = np.concatenate((target_geom, pad_channel[None]))
        target_geom = target_geom[target_channels]

    # make kdtree
    if target_kdtree is None:
        target_kdtree = KDTree(target_geom)
    else:
        assert target_kdtree.n == len(target_geom)

    if pitch is None:
        pitch = get_pitch(geom)
    if n_pitches_shift is None:
        n_pitches_shift = np.zeros(n_spikes, dtype=int)

    # the case where each waveform lives on its own channels
    # nans will never be matched in k-d query below
    padded_geom = np.pad(geom.astype(float), [(0, 1), (0, 0)], constant_values=np.nan)

    # ok, the kdtree query can get expensive when we have lots of these shifting
    # positions. it turns out to be worth it to go through the effort of figuring
    # out what the unique valid moving positions are and just targeting those
    channels_and_shifts = np.c_[main_channels, n_pitches_shift]
    uniq_channels_and_shifts, uniq_inv = np.unique(
        channels_and_shifts, axis=0, return_inverse=True
    )
    uniq_channels, uniq_n_pitches_shift = uniq_channels_and_shifts.T
    uniq_shifts = np.c_[np.zeros(uniq_channels.shape[0]), uniq_n_pitches_shift * pitch]
    uniq_moving_pos = (
        padded_geom[channel_index[uniq_channels]] + uniq_shifts[:, None, :]
    )
    uniq_valid = channel_index[uniq_channels] < n_channels_tot

    _, uniq_shifted_channels = target_kdtree.query(
        uniq_moving_pos[uniq_valid],
        distance_upper_bound=match_distance,
        workers=workers,
    )
    if uniq_shifted_channels.size < uniq_channels.shape[0] * c:
        uniq_shifted_channels_ = uniq_shifted_channels
        uniq_shifted_channels = np.full((uniq_channels.shape[0], c), target_kdtree.n)
        uniq_shifted_channels[uniq_valid] = uniq_shifted_channels_
    uniq_shifted_channels = uniq_shifted_channels.reshape(uniq_channels.shape[0], c)

    # ok, now we can return to non-unique world
    shifted_channels = uniq_shifted_channels[uniq_inv]

    return shifted_channels


def grab_static(waveforms, shifted_channels, n_static_channels, fill_value=np.nan):
    two_d = waveforms.ndim == 2
    if two_d:
        waveforms = waveforms[:, None, :]
    n_spikes, t, c = waveforms.shape
    if torch.is_tensor(waveforms):
        static_waveforms = torch.full(
            (n_spikes, t, n_static_channels + 1),
            fill_value=fill_value,
            dtype=waveforms.dtype,
            device=waveforms.device,
        )
    else:
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

# -- which templates appear at which shifts in a recording?
#    and, which pairs of shifted templates appear together?


@dataclass
class TemplateShiftIndex:
    """Return value for get_shift_and_unit_pairs"""

    n_shifted_templates: int
    # shift index -> shift
    all_pitch_shifts: np.ndarray
    # (template ix, shift index) -> shifted template index
    template_shift_index: np.ndarray
    # (shifted temp ix, shifted temp ix) -> did these appear at the same time
    shifted_temp_ix_to_temp_ix: np.ndarray
    shifted_temp_ix_to_shift: np.ndarray

    @classmethod
    def from_shift_matrix(cls, shifts):
        """shift: n_times x n_templates"""
        all_shifts = np.unique(shifts)
        n_templates = shifts.shape[1]
        pairs = np.stack(
            np.broadcast_arrays(np.arange(n_templates)[None, :], shifts), axis=2
        )
        pairs = np.unique(pairs.reshape(shifts.size, 2), axis=0)
        n_shifted_templates = len(pairs)
        shift_ix = np.searchsorted(all_shifts, pairs[:, 1])
        template_shift_index = np.full(
            (n_templates, len(all_shifts)), n_shifted_templates
        )
        template_shift_index[pairs[:, 0], shift_ix] = np.arange(n_shifted_templates)
        return cls(
            n_shifted_templates,
            all_shifts,
            template_shift_index,
            *pairs.T,
        )

    def shifts_to_shifted_ids(self, template_ids, shifts):
        shift_ixs = np.searchsorted(self.all_pitch_shifts, shifts)
        return self.template_shift_index[template_ids, shift_ixs]


def static_template_shift_index(n_templates):
    temp_ixs = np.arange(n_templates)
    return TemplateShiftIndex(
        n_templates,
        np.zeros(1),
        temp_ixs[:, None],
        temp_ixs,
        np.zeros_like(temp_ixs),
    )


def get_shift_and_unit_pairs(
    chunk_time_centers_s,
    geom,
    template_data_a,
    template_data_b=None,
    motion_est=None,
):
    if template_data_b is None:
        template_data_b = template_data_a

    na = template_data_a.templates.shape[0]
    nb = template_data_b.templates.shape[0]

    if motion_est is None:
        shift_index_a = static_template_shift_index(na)
        shift_index_b = static_template_shift_index(nb)
        cooccurrence = np.ones((na, nb), dtype=bool)
        return shift_index_a, shift_index_b, cooccurrence

    reg_depths_um_a = template_data_a.registered_template_depths_um
    reg_depths_um_b = template_data_b.registered_template_depths_um
    same = np.array_equal(reg_depths_um_a, reg_depths_um_b)
    if same:
        reg_depths_um = reg_depths_um_a
    else:
        reg_depths_um = np.concatenate((reg_depths_um_a, reg_depths_um_b))

    # figure out all shifts for all units at all times
    unreg_depths_um = np.stack(
        [
            invert_motion_estimate(motion_est, t_s, reg_depths_um)
            for t_s in chunk_time_centers_s
        ],
        axis=0,
    )
    assert unreg_depths_um.shape == (len(chunk_time_centers_s), len(reg_depths_um))
    diff = reg_depths_um - unreg_depths_um
    pitch_shifts = get_spike_pitch_shifts(
        depths_um=reg_depths_um,
        pitch=get_pitch(geom),
        registered_depths_um=unreg_depths_um,
    )
    if same:
        shifts_a = shifts_b = pitch_shifts
    else:
        shifts_a = pitch_shifts[:, :na]
        shifts_b = pitch_shifts[:, na:]

    # assign ids to pitch/shift pairs
    template_shift_index_a = TemplateShiftIndex.from_shift_matrix(shifts_a)
    if same:
        template_shift_index_b = template_shift_index_a
    else:
        template_shift_index_b = TemplateShiftIndex.from_shift_matrix(shifts_b)

    # co-occurrence matrix: do these shifted templates appear together?
    cooccurrence = np.zeros(
        (
            template_shift_index_a.n_shifted_templates,
            template_shift_index_b.n_shifted_templates,
        ),
        dtype=bool,
    )
    temps_a = np.arange(na)
    temps_b = np.arange(nb)
    for j in range(len(chunk_time_centers_s)):
        shifted_ids_a = template_shift_index_a.shifts_to_shifted_ids(
            temps_a, shifts_a[j]
        )
        if same:
            shifted_ids_b = shifted_ids_a
        else:
            shifted_ids_b = template_shift_index_b.shifts_to_shifted_ids(
                temps_b, shifts_b[j]
            )
        cooccurrence[shifted_ids_a[:, None], shifted_ids_b[None, :]] = 1

    return template_shift_index_a, template_shift_index_b, cooccurrence
