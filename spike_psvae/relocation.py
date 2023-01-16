"""Helper functions for point source relocation
"""
import numpy as np
from .waveform_utils import get_pitch, restrict_wfs_to_chans
from .spikeio import read_waveforms


def point_source_ptp(xyza, wf_channels, geom, fill_value=np.nan):
    wf_geoms = np.pad(geom, [(0, 1), (0, 0)], constant_values=fill_value)
    wf_geoms = wf_geoms[wf_channels]
    return xyza[:, 3][:, None] / (
        np.square(xyza[:, 1])[:, None]
        + np.sum(
            np.square(wf_geoms - xyza[:, (0, 2)][:, None, :]),
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
    """Point source relocation."""
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

    Arguments
    ---------
    max_channels : array of shape (n_spikes,)
    waveforms : array of shape (n_spikes, spike_length_samples, neighbor_chans)
    xyza_from : array of shape (n_spikes, 4)
        The columns here should be x, y, z_abs, alpha
    z_to : array of shape (n_spikes,)
        z coordinates to shift towards -- probably z_reg.
    channel_index : array of shape (total_chans, neighbor_chans)
        For each max channel `c`, we extract waveforms with that max channel on
        the channel neighborhood `channel_index[c]`
        So, the nth waveform waveforms[n] was extracted on
        `channel_index[max_channels[n]]`.
    geom : array of shape (total_chans, 2)
    target_channels : array of shape (n_target_channels,)
        The set of channel
    """
    assert xyza_from.shape[0] == z_to.shape[0] == max_channels.shape[0] == waveforms.shape[0]

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
        max_channels=max_channels,
        channel_index=channel_index,
        dest_channels=orig_chans,
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


def load_relocated_waveforms_on_channel_subset(
    spike_index,
    raw_bin,
    xyza_from,
    z_to,
    geom,
    target_channels,
    fill_value=np.nan,
    trough_offset=42,
    spike_length_samples=121,
):
    """Relocated waveforms on a specific group of channels `target_channels`"""
    assert spike_index.shape[0] == xyza_from.shape[0] == z_to.shape[0]
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
    shifted_waveforms, skipped = read_waveforms(
        spike_index[:, 0],
        raw_bin,
        geom.shape[0],
        channels=orig_chans,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
        fill_value=fill_value,
    )
    kept = np.setdiff1d(np.arange(len(spike_index)), skipped)

    # -- now, the remaining shift is done with point source
    xyza_cur = xyza_from[kept].copy()
    xyza_cur[:, 2] += pitch * n_pitches_shift
    xyza_to = xyza_cur.copy()
    xyza_to[:, 2] += z_drift_rem[kept]
    shifted_waveforms = relocate_simple(
        shifted_waveforms,
        xyza_cur,
        xyza_to,
        geom,
        wf_channels=target_channels,
    )

    return shifted_waveforms, skipped
