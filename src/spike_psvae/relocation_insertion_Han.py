"""Helper functions for point source relocation
"""
import numpy as np
from .waveform_utils import get_pitch, restrict_wfs_to_chans
from .spikeio import read_waveforms


def point_source_ptp(xyza, wf_channels, geom, fill_value=np.nan):
    print('point_source_ptp_shape:')
    print(np.shape(geom))
    
    wf_geoms = np.pad(geom, [(0, 1), (0, 0)], constant_values=fill_value)
    wf_geoms = wf_geoms[wf_channels]
    
    print('xyza shape:')
    print(np.shape(xyza))
    
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
    """Relocated waveforms on a specific group of channels `target_channels`"""
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
    print(f"{orig_chans.shape=} {max_channels.shape=} {waveforms.shape=}")


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

# kriging interpolation kilosort

def relocate_kriging(shifted_waveforms,
        xyza_cur,
        xyza_to,
        geom,
        wf_channels,
    ):
    
    N, T, C = np.shape(shifted_waveforms)
    shifted_waveforms_orig = np.zeros((N, T, C))
    
    for i in range(N):
        shifted_wf = shifted_waveforms[i, :, :]

        if wf_channels is None:
            wf_channels = channel_index[max_channels]

        xp = geom[wf_channels][0:24,:]

        Kxx = kernel2D(xp, xp, 15)
        
        yp = xp.copy()
        yp[:,1] = yp[:,1] + xyza_to[i, 2] - xyza_cur[i,2]
        # yp = xp + xyza_to[i, [0, 2]] - xyza_cur[i,[0, 2]]

        Kyx = kernel2D(yp, xp, 15)


        M = np.matmul(Kyx, np.linalg.inv(Kxx + .01 * np. eye(np.shape(Kxx)[0])))
        
       # M = np.divide(Kyx, Kxx + .01 * np. eye(np.shape(Kxx)[0]))
    
        shifted_wfs = np.matmul(M, shifted_wf.T)
        
        shifted_waveforms_orig[i,:,:] = shifted_wfs.T
    
    
    return shifted_waveforms_orig
    

def get_relocated_waveforms_kriging(
    max_channels,
    waveforms,
    xyza_from,
    z_to,
    channel_index,
    geom,
    target_channels,
    fill_value=0,
):
    """Relocated waveforms on a specific group of channels `target_channels`"""
    assert xyza_from.shape[0] == z_to.shape[0] == max_channels.shape[0] == waveforms.shape[0]
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
    print(f"{orig_chans.shape=} {max_channels.shape=} {waveforms.shape=}")


    # now, grab the waveforms on those channels.
    shifted_waveforms = restrict_wfs_to_chans(
        waveforms,
        max_channels=max_channels,
        channel_index=channel_index,
        dest_channels=orig_chans,
        fill_value=fill_value,
    )
    
    # shifted_waveforms = waveforms

    # -- now, the remaining shift is done with point source
    xyza_cur = xyza_from.copy()
    xyza_cur[:, 2] += pitch * n_pitches_shift
    xyza_to = xyza_cur.copy()
    xyza_to[:, 2] += z_drift_rem
    shifted_waveforms = relocate_kriging(
        shifted_waveforms,
        xyza_cur,
        xyza_to,
        geom,
        wf_channels=target_channels,
    )

    return shifted_waveforms

def kernel2D(xp, yp, sig):
    distx = abs(xp[:, 0][None,:] - yp[:, 0][None,:].T)
    disty = abs(xp[:, 1][None,:] - yp[:, 1][None,:].T)
    
    sigx = sig
    sigy = 1.5 * sig
    
    p = 1
    K = np.exp(- (distx/sigx)**p - (disty/sigy)**p)
    
    return K
    