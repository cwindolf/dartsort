import numpy as np


def relativize_z(z_abs, maxchans, geom):
    """Take absolute z coords -> relative to max channel z."""
    return z_abs - geom[maxchans.astype(int), 1]


def maxchan_from_firstchan(firstchan, wf):
    return firstchan + wf.ptp(0).argmax()


def updown_decision(geom, maxchan, channel_radius, ptp):
    """Gets indices of channels around the maxchan"""
    G, d = geom.shape
    assert d == 2
    assert ptp.ndim == 1
    C = ptp.shape[0]
    maxchan = int(maxchan)

    # Deal with edge cases
    low = maxchan - channel_radius
    high = maxchan + channel_radius
    if low <= 0:
        return True
    # low and high should be  - (maxchan % 2) above,
    # but adding that now would break compatibility.
    # so, the limit has been lowered here as a hack.
    if high >= geom.shape[0] - 1:
        return False

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
            f"Not sure how to get local geom when ptp has {C} channels "
            f"and channel_radius={channel_radius}"
        )

    return up


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
    if low <= 0:
        low = 0
        high = 2 * channel_radius
        return low, high
    # low and high should be  - (maxchan % 2) above,
    # but adding that now would break compatibility.
    # so, the limit has been lowered here as a hack.
    if high >= geom.shape[0] - 1:
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

    low += 2 * up - (maxchan % 2)
    high += 2 * up - (maxchan % 2)

    return low, high


def get_local_chans_standard(geom, maxchan, channel_radius):
    """Gets indices of channels around the maxchan"""
    G, d = geom.shape
    assert d == 2
    maxchan = int(maxchan)
    if maxchan % 2:
        maxchan = maxchan - 1

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


def get_local_chans_firstchan(geom, firstchan, channel_radius):
    """Gets indices of channels around the maxchan"""
    G, d = geom.shape
    assert d == 2

    # Deal with edge cases
    low = firstchan
    high = firstchan + 2 * channel_radius
    assert low >= 0
    assert high <= G

    return low, high


def get_local_chans_firstchanstandard(geom, firstchan, channel_radius):
    """Gets indices of channels around the maxchan"""
    G, d = geom.shape
    assert d == 2

    # Deal with edge cases
    low = firstchan
    high = firstchan + 2 * channel_radius + 1
    assert low >= 0
    assert high <= G

    return low, high


def get_local_chans(
    geom, maxchan, channel_radius, ptp=None, firstchan=None, geomkind="updown"
):
    if geomkind == "updown":
        assert ptp is not None
        return get_local_chans_updown(geom, maxchan, channel_radius, ptp)
    elif geomkind == "standard":
        return get_local_chans_standard(geom, maxchan, channel_radius)
    elif geomkind == "firstchan":
        assert firstchan is not None
        return get_local_chans_firstchan(geom, firstchan, channel_radius)
    elif geomkind == "firstchanstandard":
        assert firstchan is not None
        return get_local_chans_firstchanstandard(
            geom, firstchan, channel_radius
        )
    else:
        raise ValueError(f"Unknown geomkind={geomkind}")


def get_local_geom(
    geom,
    maxchan,
    channel_radius,
    ptp=None,
    firstchan=None,
    return_z_maxchan=False,
    geomkind="updown",
):
    """Gets the geometry of some neighborhood of chans near maxchan"""
    low, high = get_local_chans(
        geom,
        maxchan,
        channel_radius,
        ptp=ptp,
        firstchan=firstchan,
        geomkind=geomkind,
    )
    local_geom = geom[low:high].copy()
    z_maxchan = geom[int(maxchan), 1]
    local_geom[:, 1] -= z_maxchan

    if return_z_maxchan:
        return local_geom, z_maxchan
    return local_geom


def get_local_waveforms(
    waveforms,
    channel_radius,
    geom,
    maxchans=None,
    firstchans=None,
    geomkind="updown",
    compute_firstchans=False,
):
    """NxTxCfull -> NxTx(2*channel radius). So, takes a batch."""
    N, T, Cfull = waveforms.shape

    compute_maxchans = maxchans is None
    ptps = waveforms.ptp(1)
    if compute_maxchans:
        maxchans = ptps.argmax(1)

    local_waveforms = np.empty(
        (N, T, 2 * channel_radius + 2 * ("standard" in geomkind)),
        dtype=waveforms.dtype,
    )
    lows = []
    for n in range(N):
        low, high = get_local_chans(
            geom,
            maxchans[n],
            channel_radius,
            ptps[n],
            firstchan=None if firstchans is None else firstchans[n],
            geomkind=geomkind,
        )
        local_waveforms[n] = waveforms[n, :, low:high]
        lows.append(low)

    if compute_maxchans:
        return local_waveforms, maxchans
    if compute_firstchans:
        return local_waveforms, np.array(lows)
    return local_waveforms


def as_standard_local(waveforms, maxchans, geom, channel_radius=8):
    if waveforms.shape[2] == geom.shape[0]:
        local_waveforms = get_local_waveforms(
            waveforms, channel_radius, geom, maxchans, geomkind="standard"
        )
        return local_waveforms
    elif waveforms.shape[2] == 2 + 2 * channel_radius:
        return waveforms
    elif waveforms.shape[2] == 4 + 2 * channel_radius:
        print(
            f"Lossy conversion from 'updown' geom on {waveforms.shape[2]} "
            f"chans to 'standard' geom on {2 * channel_radius + 2} chans."
        )
        local_waveforms = np.empty(
            (*waveforms.shape[:2], 2 + 2 * channel_radius),
            dtype=waveforms.dtype,
        )

        for n, up in enumerate(
            updown_decision(geom, maxchan, channel_radius + 2, ptp)
            for maxchan, ptp in zip(maxchans, waveforms.ptp(1))
        ):
            # what goes up must come down
            if up:
                local_waveforms[n] = waveforms[n, :, :-2]
            else:
                local_waveforms[n] = waveforms[n, :, 2:]
        return local_waveforms
    else:
        raise ValueError("Not sure how to convert to standard local.")
