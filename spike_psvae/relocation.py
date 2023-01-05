"""Helper functions for point source relocation
"""

import numpy as np
# from numba import njit
from .waveform_utils import get_pitch


def point_source_ptp(xyza, wf_channels, geom, fill_value=np.nan):
    wf_geoms = np.pad(geom, [(0, 1), (0, 0)], constant_values=fill_value)
    wf_geoms = wf_geoms[wf_channels]
    return xyza[:, 3][:, None] / (
        np.square(xyza[:, 1])[:, None]
        + np.sum(
            np.square(
                wf_geoms
                - xyza[:, (0, 2)][:, None, :]
            ),
            axis=2,
        )
    )


def relocate_simple(
    waveforms,
    xyza_from,
    xyza_to,
    geom,
    max_channels=None,
    channel_index=None,
    wf_channels=None,
):
    if wf_channels is None:
        wf_channels = channel_index[max_channels]
    ptp_from = point_source_ptp(xyza_from, wf_channels, geom)
    ptp_to = point_source_ptp(xyza_to, wf_channels, geom)
    return waveforms * (ptp_to / ptp_from)[:, None, :]


def shifted_chans(
    n_pitches_shift,
    target_channels,
    geom,
):
    pitch = get_pitch(geom)
    orig_locs = geom[target_channels]
    orig_locs[:, 1] -= n_pitches_shift * pitch
    matching = (orig_locs[:, None, :] == geom[None, :, :]).all(axis=2)
    assert max(matching.sum(1).max(), matching.sum(0).max()) <= 1
    matched, matchix = np.nonzero(matching)
    orig_chans = np.full_like(target_channels, geom.shape[0])
    orig_chans[matched] = matchix
    return orig_chans


def get_relocated_waveforms_on_channel_subset(
    max_channels,
    waveforms,
    xyza_from,
    z_to,
    channel_index,
    geom,
    target_channels,
    fill_value=np.nan,
):
    """Relocated waveforms on a specific group of channels `target_channels`
    """
    # we will handle the "integer part" of the drift by just grabbing
    # different channels in the waveforms, and the remainder by point
    # source relocation
    z_drift = z_to - xyza_from[:, 2]
    pitch = get_pitch(geom)
    # want to round towards 0, not //
    n_pitches_shift = (z_drift / pitch).astype(int)
    z_drift_rem = z_drift - pitch * n_pitches_shift

    # -- first, handle the integer part of the shift
    # we want to grab the original channels which would land on the target channels
    # after shifting by n_pitches_shift. this is per-waveform, then.
    # start by finding the channel neighborhoods for each unique value of n_pitches_shift
    pitches_shift_uniq, pitch_index = np.unique(
        n_pitches_shift, return_inverse=True
    )
    orig_chans_uniq = np.array(
        [
            shifted_chans(pitches_shift, target_channels, geom)
            for pitches_shift in pitches_shift_uniq
        ]
    )
    orig_chans = orig_chans_uniq[pitch_index]

    # now, grab the waveforms on those channels.
    shifted_waveforms = restrict_wfs_to_chans(
        waveforms,
        max_channels,
        channel_index,
        orig_chans,
        fill_value=fill_value,
    )

    # -- now, the remaining shift is done with point source
    xyza_cur = xyza_from.copy()
    xyza_cur[:, 2] += pitch * n_pitches_shift
    xyza_to = xyza_cur.copy()
    xyza_to[:, 2] += z_drift_rem
    shifted_waveforms = relocate_simple(
        shifted_waveforms,
        xyza_cur,
        xyza_to,
        geom,
        wf_channels=target_channels,
    )

    return shifted_waveforms


# @njit(cache=False)
def restrict_wfs_to_chans(
    waveforms, maxchans, channel_index, dest_channels, fill_value=np.nan
):
    N, T, C = waveforms.shape
    assert N == maxchans.size
    assert C == channel_index.shape[1]
    N_, c = dest_channels.shape
    assert N == N_

    out_waveforms = np.full((N, T, c), fill_value, dtype=waveforms.dtype)
    for n in range(N):
        chans_in_target, target_found = np.nonzero(
            channel_index[maxchans[n]].reshape(-1, 1) == dest_channels[n].reshape(1, -1)
        )
        out_waveforms[n, :, target_found] = waveforms[n, :, chans_in_target]
        # for t, c in zip(target_found, chans_in_target):
        #     out_waveforms[n, :, t] = waveforms[n, :, c]

    return out_waveforms
