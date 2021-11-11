import numpy as np


def relativize_z(z_abs, maxchans, geom):
    """Take absolute z coords -> relative to max channel z."""
    # z_rels = np.empty_like(z_abs)
    # for j, (z, mc) in enumerate(zip(z_abs, maxchans)):
    #     z_rels[j] = z - geom[mc, 1]
    return z_abs - geom[maxchans.astype(int), 1]


def get_local_geom(geom, maxchan, channel_radius, return_z_maxchan=False):
    """
    Gets `2 * channel_radius` chans near maxchan. Deals with the boundary.
    """
    # Deal with the boundary
    low = maxchan - channel_radius
    high = maxchan + channel_radius
    if low < 0:
        low = 0
        high = 2 * channel_radius
    if high > geom.shape[0]:
        high = geom.shape[0]
        low = geom.shape[0] - 2 * channel_radius

    # Extract geometry and relativize z around the max channel
    local_geom = geom[low:high].copy()
    z_maxchan = geom[maxchan, 1]
    local_geom[:, 1] -= z_maxchan

    if return_z_maxchan:
        return local_geom, z_maxchan
    return local_geom


def get_local_waveforms(waveforms, channel_radius, maxchans=None):
    """NxTxCfull -> NxTx(2*channel radius). So, takes a batch.
    """
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
