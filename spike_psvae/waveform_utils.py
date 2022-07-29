import numpy as np
from scipy.spatial.distance import cdist


def channel_index_subset(geom, channel_index, n_channels=None, radius=None):
    """Restrict channel index to fewer channels

    Creates a boolean mask of the same shape as the channel index
    that you can use to restrict waveforms extracted using that
    channel index to fewer channels.

    Operates in two modes:
        n_channels is not None:
            This is the old style where we will grab the n_channels
            channels whose indices are nearest to the max chan index.
        radius is not None:
            Grab channels within a spatial radius.
    """
    subset = np.empty(shape=channel_index.shape, dtype=bool)
    pgeom = np.pad(geom, [(0, 1), (0, 0)], constant_values=-2 * geom.max())
    for c in range(len(geom)):
        if n_channels is not None:
            low = max(0, c - n_channels // 2)
            low = min(len(geom) - n_channels, low)
            high = min(len(geom), low + n_channels)
            subset[c] = (low <= channel_index[c]) & (channel_index[c] < high)
        elif radius is not None:
            dists = cdist([geom[c]], pgeom[channel_index[c]]).ravel()
            subset[c] = dists <= radius
        else:
            subset[c] = True
    return subset


def get_channel_subset(waveforms, max_channels, channel_index_subset, fill_value=np.nan):
    """You have waveforms on C channels, and you want them on fewer.

    You can use a channel_index_subset obtained from the function `channel_index_subset`
    above together with this function to do it.

    E.g. you have waveforms on 40 channels extracted using `channel_index`, and you
    want 10 channels in the end. You get:

    ```
    subset = channel_index_subset(geom, channel_index, n_channels=10)
    ```

    (The number of channels extracted at each max channel will be:
    ```
    n_chans_in_subset = subset.sum(axis=1)[max_channels]
    ```
    if you need these numbers.)

    Then you can subset your waveforms with
    ```
    wfs_sub = get_channel_subset(wfs, maxchans, subset)
    ```
    """
    N, T, C = waveforms.shape
    n_channels, C_ = channel_index_subset.shape
    assert C == C_

    # convert to relative channel offsets
    rel_sub_channel_index = np.tile(np.arange(C), (n_channels, 1))
    rel_sub_channel_index[~channel_index_subset] = C

    waveforms = np.pad(
        waveforms, [(0, 0), (0, 0), (0, 1)], constant_values=fill_value
    )

    return waveforms[
        np.arange(N)[:, None, None],
        np.arange(T)[None, :, None],
        rel_sub_channel_index[max_channels][:, None, :],
    ]



def get_maxchan_traces(waveforms, channel_index, maxchans):
    index_of_mc = np.argwhere(channel_index == np.arange(len(channel_index))[:, None])
    assert (index_of_mc[:, 0] == np.arange(len(channel_index))).all()
    index_of_mc = index_of_mc[:, 1]
    rel_maxchans = index_of_mc[maxchans]
    maxchan_traces = waveforms[np.arange(len(waveforms)), :, rel_maxchans]
    return maxchan_traces


def relativize_z(z_abs, maxchans, geom):
    """Take absolute z coords -> relative to max channel z."""
    return z_abs - geom[maxchans.astype(int), 1]


def maxchan_from_firstchan(firstchan, wf):
    return firstchan + wf.ptp(0).argmax()


def temporal_align(waveforms, offset=42):
    N, T, C = waveforms.shape
    maxchans = waveforms.ptp(1).argmax(1)
    offsets = waveforms[np.arange(N), :, maxchans].argmin(1)
    rolls = offset - offsets
    out = np.empty_like(waveforms)
    pads = [(0, 0), (0, 0)]
    for i, roll in enumerate(rolls):
        if roll > 0:
            pads[0] = (roll, 0)
            start, end = 0, T
        elif roll < 0:
            pads[0] = (0, -roll)
            start, end = -roll, T - roll
        else:
            out[i] = waveforms[i]
            continue

        pwf = np.pad(waveforms[i], pads, mode="linear_ramp")
        out[i] = pwf[start:end, :]

    return out


def get_local_chans(geom, firstchan, n_channels):
    """Gets indices of channels around the maxchan"""
    G, d = geom.shape
    assert d == 2
    assert not n_channels % 2

    # Deal with edge cases
    low = firstchan
    high = firstchan + n_channels
    assert low >= 0
    assert high <= G

    return low, high


def get_local_geom(
    geom,
    firstchan,
    maxchan,
    n_channels,
    return_z_maxchan=False,
):
    """Gets the geometry of some neighborhood of chans near maxchan"""
    low, high = get_local_chans(geom, firstchan, n_channels)
    local_geom = geom[low:high].copy()
    z_maxchan = geom[int(maxchan), 1]
    local_geom[:, 1] -= z_maxchan

    if return_z_maxchan:
        return local_geom, z_maxchan
    return local_geom


def relativize_waveforms(
    wfs, firstchans_orig, z, geom, maxchans_orig=None, feat_chans=18
):
    """
    Extract fewer channels.
    """
    chans_down = feat_chans // 2
    chans_down -= chans_down % 2

    stdwfs = np.zeros(
        (wfs.shape[0], wfs.shape[1], feat_chans), dtype=wfs.dtype
    )

    firstchans_std = firstchans_orig.copy().astype(int)
    maxchans_std = np.zeros(firstchans_orig.shape, dtype=int)
    if z is not None:
        z_rel = np.zeros_like(z)

    for i in range(wfs.shape[0]):
        wf = wfs[i]
        if maxchans_orig is None:
            mcrel = wf.ptp(0).argmax()
        else:
            mcrel = maxchans_orig[i] - firstchans_orig[i]
        mcrix = mcrel - mcrel % 2
        if z is not None:
            z_rel[i] = z[i] - geom[firstchans_orig[i] + mcrel, 1]

        low, high = mcrix - chans_down, mcrix + feat_chans - chans_down
        if low < 0:
            low, high = 0, feat_chans
        if high > wfs.shape[2]:
            low, high = wfs.shape[2] - feat_chans, wfs.shape[2]

        firstchans_std[i] += low
        stdwfs[i] = wf[:, low:high]
        maxchans_std[i] = firstchans_std[i] + stdwfs[i].ptp(0).argmax()

    if z is not None:
        return stdwfs, firstchans_std, maxchans_std, z_rel, chans_down
    else:
        return stdwfs, firstchans_std, maxchans_std, chans_down

def relativize_waveforms_np1(
    wfs, firstchans_orig, geom, maxchans_orig, feat_chans=20
):
    """
    Extract fewer channels.
    """
    chans_down = feat_chans // 2
    chans_down -= chans_down % 4

    stdwfs = np.zeros(
        (wfs.shape[0], wfs.shape[1], feat_chans), dtype=wfs.dtype
    )

    firstchans_std = firstchans_orig.copy().astype(int)
    maxchans_std = np.zeros(firstchans_orig.shape, dtype=int)

    for i in range(wfs.shape[0]):
        wf = wfs[i]
        if maxchans_orig is None:
            mcrel = wf.ptp(0).argmax()
        else:
            mcrel = maxchans_orig[i] - firstchans_orig[i]
        mcrix = mcrel - mcrel % 4

        low, high = mcrix - chans_down, mcrix + feat_chans - chans_down
        if low < 0:
            low, high = 0, feat_chans
        if high > wfs.shape[2]:
            low, high = wfs.shape[2] - feat_chans, wfs.shape[2]

        firstchans_std[i] += low
        stdwfs[i] = wf[:, low:high]

    return stdwfs, firstchans_std, chans_down
    