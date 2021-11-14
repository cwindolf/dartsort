import numpy as np


def relativize_z(z_abs, maxchans, geom):
    """Take absolute z coords -> relative to max channel z."""
    return z_abs - geom[maxchans.astype(int), 1]


def get_local_chans(geom, maxchan, channel_radius, ptp):
    """Gets indices of channels around the maxchan"""
    G, d = geom.shape
    assert d == 2
    assert ptp.ndim == 1
    C = ptp.shape[0]

    # Deal with edge cases
    low = maxchan - channel_radius
    high = maxchan + channel_radius
    if low < 0:
        low = 0
        high = 2 * channel_radius
        return low, high
    if high > geom.shape[0]:
        high = geom.shape[0]
        low = geom.shape[0] - 2 * channel_radius
        return low, high

    # -- See if we are going "up" or "down"
    # how to compute depends on ptp shape
    if C == G:
        # here we can use the original logic
        up = ptp[maxchan + 2] > ptp[maxchan - 2]
    elif C == 2 * channel_radius:
        # here we need to figure things out...
        local_maxchan = ptp.argmax()
        # local_maxchan should not push this out of bounds...
        up = ptp[local_maxchan + 2] > ptp[local_maxchan - 2]
    else:
        raise ValueError(
            f"Not sure how to get local geom when ptp has {C} channels"
        )

    odd = maxchan % 2
    low += 2 * up - odd
    high += 2 * up - odd

    return low, high


def get_local_geom(geom, maxchan, channel_radius, ptp, return_z_maxchan=False):
    """Gets geometry of `2 * channel_radius` chans near maxchan"""
    low, high = get_local_chans(geom, maxchan, channel_radius, ptp)
    local_geom = geom[low:high].copy()
    z_maxchan = geom[maxchan, 1]
    local_geom[:, 1] -= z_maxchan

    if return_z_maxchan:
        return local_geom, z_maxchan
    return local_geom


def get_local_waveforms(waveforms, channel_radius, maxchans=None):
    """NxTxCfull -> NxTx(2*channel radius). So, takes a batch."""
    N, T, Cfull = waveforms.shape

    compute_maxchans = maxchans is None
    if compute_maxchans:
        maxchans = waveforms.ptp(1).argmax(1)

    local_waveforms = np.empty(
        (N, T, 2 * channel_radius), dtype=waveforms.dtype
    )
    for n in range(N):
        low = maxchans[n] - channel_radius
        high = maxchans[n] + channel_radius
        if low < 0:
            low = 0
            high = 2 * channel_radius
        if high > Cfull:
            high = Cfull
            low = Cfull - 2 * channel_radius
        local_waveforms[n] = waveforms[n, :, low:high]

    if compute_maxchans:
        return local_waveforms, maxchans
    return local_waveforms
