import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.decomposition import PCA

from . import spikeio


def fit_tpca_bin(
    spike_index,
    geom,
    binary_file,
    tpca_rank=5,
    tpca_n_wfs=50_000,
    trough_offset=42,
    spike_length_samples=121,
    spatial_radius=75,
    seed=0,
):
    rg = np.random.default_rng(seed)
    tpca_channel_index = make_channel_index(
        geom, spatial_radius, steps=1, distance_order=False, p=1
    )
    choices = rg.choice(
        len(spike_index), size=min(len(spike_index), tpca_n_wfs), replace=False
    )
    choices.sort()
    tpca_waveforms, skipped_idx = spikeio.read_waveforms(
        spike_index[choices, 0],
        binary_file,
        geom.shape[0],
        channel_index=tpca_channel_index,
        trough_offset=trough_offset,
        spike_length_samples=spike_length_samples,
        max_channels=spike_index[choices, 1],
    )
    # NTC -> NCT
    tpca_waveforms = tpca_waveforms.transpose(0, 2, 1).reshape(
        -1, spike_length_samples
    )
    which = np.isfinite(tpca_waveforms[:, 0])
    tpca_waveforms = tpca_waveforms[which]
    tpca = PCA(tpca_rank).fit(tpca_waveforms)

    return tpca


def apply_tpca(waveforms, tpca):
    if tpca is None:
        return waveforms
    single = waveforms.ndim == 2
    if single:
        waveforms = waveforms[None]
    n, t, c = waveforms.shape
    waveforms = waveforms.transpose(0, 2, 1).reshape(n * c, t)
    valid = ~np.isnan(waveforms).any(axis=1)
    waveforms[valid] = tpca.inverse_transform(tpca.transform(waveforms[valid]))
    waveforms = waveforms.reshape(n, c, t).transpose(0, 2, 1)
    if single:
        waveforms = waveforms[0]
    return waveforms


# -- channels / geometry helpers
"""
For example let’s say we use a Neuropixels probe
and create a channel index for channels which are
<=200um apart from each other. This will be a
(384, n_neighbors)  shaped integer array, where
n_neighbors should be ~40 for 200um.
Then channel_index[i] is an array with 40 integers
containing all of the indices of the neighboring
channels. Since different channels may have different
numbers of neighbors, we need to fill in some gaps in
this array, and those will be filled with the value 384
(or whatever the number of channels is
"""


def n_steps_neigh_channels(neighbors_matrix, steps):
    """Compute a neighbors matrix by considering neighbors of neighbors

    Parameters
    ----------
    neighbors_matrix: numpy.ndarray
        Neighbors matrix
    steps: int
        Number of steps to still consider channels as neighbors

    Returns
    -------
    numpy.ndarray (n_channels, n_channels)
        Symmetric boolean matrix with the i, j as True if the ith and jth
        channels are considered neighbors
    """
    # Compute neighbors of neighbors via matrix powers
    output = np.eye(neighbors_matrix.shape[0]) + neighbors_matrix
    return np.linalg.matrix_power(output, steps) > 0


def order_channels_by_distance(reference, channels, geom):
    """Order channels by distance using certain channel as reference
    Parameters
    ----------
    reference: int
        Reference channel
    channels: np.ndarray
        Channels to order
    geom
        Geometry matrix
    Returns
    -------
    numpy.ndarray
        1D array with the channels ordered by distance using the reference
        channels
    numpy.ndarray
        1D array with the indexes for the ordered channels
    """
    coord_main = geom[reference]
    coord_others = geom[channels]
    idx = np.argsort(np.sum(np.square(coord_others - coord_main), axis=1))
    return channels[idx], idx


def make_contiguous_channel_index(n_channels, n_neighbors=40):
    channel_index = []
    for c in range(n_channels):
        low = max(0, c - n_neighbors // 2)
        low = min(n_channels - n_neighbors, low)
        channel_index.append(np.arange(low, low + n_neighbors))
    channel_index = np.array(channel_index)

    return channel_index


def make_channel_index(geom, radius, steps=1, distance_order=False, p=2):
    """
    Compute an array whose whose ith row contains the ordered
    (by distance) neighbors for the ith channel
    """
    C = geom.shape[0]

    # get neighbors matrix
    neighbors = squareform(pdist(geom, metric="minkowski", p=p)) <= radius
    neighbors = n_steps_neigh_channels(neighbors, steps=steps)

    # max number of neighbors for all channels
    n_neighbors = np.max(np.sum(neighbors, 0))

    # initialize channel index
    # entries for channels which don't have as many neighbors as
    # others will be filled with the total number of channels
    # (an invalid index into the recording, but this behavior
    # is useful e.g. in the spatial max pooling for deduplication)
    channel_index = np.full((C, n_neighbors), C, dtype=int)

    # fill every row in the matrix (one per channel)
    for current in range(C):
        # indexes of current channel neighbors
        ch_idx = np.flatnonzero(neighbors[current])

        # sort them by distance
        if distance_order:
            ch_idx, _ = order_channels_by_distance(current, ch_idx, geom)

        # fill entries with the sorted neighbor indexes
        channel_index[current, : ch_idx.shape[0]] = ch_idx

    return channel_index


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


def channel_index_is_subset(channel_index_a, channel_index_b):
    if not np.all(
        np.array(channel_index_a.shape) <= np.array(channel_index_b.shape)
    ):
        return False

    n_channels = channel_index_a.shape[0]

    for row_a, row_b in zip(channel_index_a, channel_index_b):
        if not np.isin(np.setdiff1d(row_a, [n_channels]), row_b).all():
            return False

    return True


def get_channel_subset(
    waveforms, max_channels, channel_index_subset, fill_value=np.nan
):
    """You have waveforms on C channels, and you want them on fewer.

    You can use a channel_index_subset obtained from the function
    `channel_index_subset` above together with this function to do it.

    E.g. you have waveforms on 40 channels extracted using
    `channel_index`, and you want 10 channels in the end. You get:

    ```
    subset = channel_index_subset(geom, channel_index, n_channels=10)
    ```
    This will be a boolean array of the same shape as the original
    channel index. You can make your own too, of course.

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

    max_sub_chans = channel_index_subset.sum(axis=1).max()

    # convert to relative channel offsets
    rel_sub_channel_index = []
    for mask in channel_index_subset:
        s = np.flatnonzero(mask)
        s = list(s) + [C] * (max_sub_chans - len(s))
        rel_sub_channel_index.append(s)
    rel_sub_channel_index = np.array(rel_sub_channel_index)

    waveforms = np.pad(
        waveforms, [(0, 0), (0, 0), (0, 1)], constant_values=fill_value
    )

    return waveforms[
        np.arange(N)[:, None, None],
        np.arange(T)[None, :, None],
        rel_sub_channel_index[max_channels][:, None, :],
    ]


def channel_subset_by_index(
    waveforms,
    max_channels,
    channel_index_full,
    channel_index_new,
    fill_value=np.nan,
):
    """Restrict waveforms to channels in new channel index."""
    # boolean mask of same shape as channel_index_full
    n_channels = channel_index_full.shape[0]
    channel_index_mask = np.array(
        [
            # by removing n_channels, chans outside array are treated
            # like excluded channels and will be loaded with fill_value
            # by get_channel_subset
            # this could be surprising if fill_value here is different
            # from the one used when loading the waveforms originally
            np.isin(row_full, np.setdiff1d(row_new, [n_channels]))
            for row_full, row_new in zip(channel_index_full, channel_index_new)
        ]
    )

    return get_channel_subset(waveforms, max_channels, channel_index_mask)


def get_maxchan_traces(waveforms, channel_index, maxchans):
    index_of_mc = np.argwhere(
        channel_index == np.arange(len(channel_index))[:, None]
    )
    assert (index_of_mc[:, 0] == np.arange(len(channel_index))).all()
    index_of_mc = index_of_mc[:, 1]
    rel_maxchans = index_of_mc[maxchans]
    maxchan_traces = waveforms[np.arange(len(waveforms)), :, rel_maxchans]
    return maxchan_traces


# thinking about numbaing this... maybe an inner for loop manually checking the ==?
# @njit(cache=False)
def restrict_wfs_to_chans(
    waveforms,
    max_channels=None,
    channel_index=None,
    source_channels=None,
    dest_channels=None,
    fill_value=np.nan,
):
    """You have `waveforms` on channels from `channel_index` according to `max_channels`.
    You want them on just `dest_channels`, which can be a 1d array of channels
    or a 2d array giving specific channels for each waveform. And you'll get them.
    """
    N, T, C = waveforms.shape

    # handle source channels
    if (
        (max_channels is None and channel_index is None)
        == (source_channels is None)
    ):
        raise ValueError(
            "Please supply either max_channels and channel_index or source_channels"
        )
    if source_channels is None:
        assert (N,) == max_channels.shape
        assert C == channel_index.shape[1]
        source_channels = channel_index[max_channels]
    source_channels = np.atleast_1d(source_channels)
    # make at least 2d with empty dim to start if necessary
    if source_channels.ndim == 1:
        source_channels = source_channels[None, :]
    n_source, c = source_channels.shape
    assert n_source in (1, N)

    # handle dest channels
    assert dest_channels is not None
    dest_channels = np.atleast_1d(dest_channels)
    # make at least 2d with empty dim to start if necessary
    if dest_channels.ndim == 1:
        dest_channels = dest_channels[None, :]
    n_dest, c = dest_channels.shape
    assert n_dest in (1, N)

    out_waveforms = np.full((N, T, c), fill_value, dtype=waveforms.dtype)
    for n in range(N):
        chans_in_target, target_found = np.nonzero(
            source_channels[n % n_source].reshape(-1, 1)
            == dest_channels[n % n_dest].reshape(1, -1)
        )
        out_waveforms[n, :, target_found] = waveforms[n, :, chans_in_target]

    return out_waveforms


# -- channel shifting stuff


def get_pitch(geom):
    """Guess the pitch, even for probes with gaps or channels missing at random

    This is the unit at which the probe repeats itself, computed as the
    vertical distance between electrodes in the same column. Of course, this
    makes little sense for probes with non-lattice layouts, or even worse,
    horizontal probes (i.e., x positions are unique) where it cannot be defined.

    So for NP1, it's not every row, but every 2 rows! And for a probe with a
    zig-zag arrangement, it would be also 2 vertical distances between channels.
    """
    x_uniq = np.unique(geom[:, 0])

    pitch = np.inf
    for x in x_uniq:
        y_uniq_at_x = np.unique(geom[geom[:, 0] == x, 1])
        if y_uniq_at_x.size > 1:
            pitch = min(pitch, np.diff(y_uniq_at_x).min())

    if np.isinf(pitch):
        raise ValueError("Horizontal probe.")

    return pitch


def pitch_shift_templates(n_pitches_shift, geom, templates, fill_value=0.0):
    if n_pitches_shift == 0:
        return templates

    pitch = get_pitch(geom)
    # + or -? if the drift was +x, then we want to load from channel at +x
    shifted_geom = geom - [[0, n_pitches_shift * pitch]]
    geom_matching = (shifted_geom[:, None, :] == geom[None, :, :]).all(axis=2)

    new_templates = np.full_like(templates, fill_value=fill_value)
    for shifted_ix, matched_orig in enumerate(
        np.flatnonzero(gm) for gm in geom_matching
    ):
        if matched_orig.size:
            assert matched_orig.size == 1
            new_templates[:, :, shifted_ix] = templates[:, :, matched_orig[0]]

    return new_templates


def temporal_align(waveforms, maxchans, offset=42):
    N, T, C = waveforms.shape
    offsets = np.abs(waveforms[np.arange(N), :, maxchans]).argmax(1)
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
