import numpy as np


def relativize_z(z_abs, maxchans, geom):
    """Take absolute z coords -> relative to max channel z."""
    return z_abs - geom[maxchans.astype(int), 1]


def get_local_chans_updown(geom, maxchan, channel_radius, ptp):
    """Gets indices of channels around the maxchan"""
    G, d = geom.shape
    assert d == 2
    assert ptp.ndim == 1
    C = ptp.shape[0]
    maxchan = int(maxchan)

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


def get_local_chans_standard(geom, maxchan, channel_radius):
    """Gets indices of channels around the maxchan"""
    G, d = geom.shape
    assert d == 2
    maxchan = int(maxchan)

    # Deal with edge cases
    low = maxchan - channel_radius
    high = maxchan + channel_radius + 2
    if low < 0:
        low = 0
        high = 2 * channel_radius + 2
    if high > geom.shape[0]:
        high = geom.shape[0]
        low = geom.shape[0] - 2 * channel_radius - 2

    return low, high


def get_local_chans(
    geom, maxchan, channel_radius, ptp=None, geomkind="updown"
):
    if geomkind == "updown":
        assert ptp is not None
        return get_local_chans_updown(geom, maxchan, channel_radius, ptp)
    elif geomkind == "standard":
        return get_local_chans_standard(geom, maxchan, channel_radius)
    else:
        raise ValueError(f"Unknown geomkind={geomkind}")


def get_local_geom(
    geom,
    maxchan,
    channel_radius,
    ptp=None,
    return_z_maxchan=False,
    geomkind="updown",
):
    """Gets geometry of `2 * channel_radius` chans near maxchan"""
    low, high = get_local_chans(
        geom, maxchan, channel_radius, ptp=ptp, geomkind=geomkind
    )
    local_geom = geom[low:high].copy()
    z_maxchan = geom[int(maxchan), 1]
    local_geom[:, 1] -= z_maxchan

    if return_z_maxchan:
        return local_geom, z_maxchan
    return local_geom


def get_local_waveforms(
    waveforms, channel_radius, geom, maxchans=None, geomkind="updown"
):
    """NxTxCfull -> NxTx(2*channel radius). So, takes a batch."""
    N, T, Cfull = waveforms.shape

    compute_maxchans = maxchans is None
    ptps = waveforms.ptp(1)
    if compute_maxchans:
        maxchans = ptps.argmax(1)

    local_waveforms = np.empty(
        (N, T, 2 * channel_radius), dtype=waveforms.dtype
    )
    for n in range(N):
        low, high = get_local_chans(
            geom, maxchans[n], channel_radius, ptps[n], geomkind=geomkind
        )
        local_waveforms[n] = waveforms[n, :, low:high]

    if compute_maxchans:
        return local_waveforms, maxchans
    return local_waveforms
