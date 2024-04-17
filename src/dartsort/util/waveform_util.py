"""
A collection of helper functions for dealing with which channels
waveforms are extracted on.

Right now, this file is a mix of numpy and torch, and it could be
a bit confusing which functions are torch/numpy-compatible.
Ideally they should all be both.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist, pdist, squareform

# -- geometry utils


def get_pitch(geom, direction=1):
    """Guess the pitch, even for probes with gaps or channels missing at random

    This is the unit at which the probe repeats itself, computed as the
    vertical distance between electrodes in the same column. Of course, this
    makes little sense for probes with non-lattice layouts, or even worse,
    horizontal probes (i.e., x positions are unique) where it cannot be defined.

    So for NP1, it's not every row, but every 2 rows! And for a probe with a
    zig-zag arrangement, it would be also 2 vertical distances between channels.

    Copied from https://github.com/cwindolf/dartsort
    """
    other_dims = [i for i in range(geom.shape[1]) if i != direction]
    other_dims_uniq = np.unique(geom[:, other_dims], axis=0)

    pitch = np.inf
    for pos in other_dims_uniq:
        at_x = np.all(geom[:, other_dims] == pos, axis=1)
        y_uniq_at_x = np.unique(geom[at_x, 1])
        if y_uniq_at_x.size > 1:
            pitch = min(pitch, np.diff(y_uniq_at_x).min())

    if np.isinf(pitch):
        raise ValueError("Horizontal probe.")

    return pitch


def fill_geom_holes(geom):
    pitch = get_pitch(geom)
    pitches_pad = int(np.ceil(geom[:, 1].ptp() / pitch))

    # we have to be careful about floating point error here
    # two sites may be different due to floating point error
    # we know they are the same if their distance is smaller than:
    min_distance = pdist(geom, metric="sqeuclidean").min() / 2

    # find all shifted site positions
    # TODO make this not quadratic
    unique_shifted_positions = list(geom)
    is_original = [True] * len(geom)
    for shift in range(-pitches_pad, pitches_pad + 1):
        shifted_geom = geom + [0, pitch * shift]
        sz = shifted_geom[:, 1]
        szinside = sz == sz.clip(
            geom[:, 1].min() - np.sqrt(min_distance),
            geom[:, 1].max() + np.sqrt(min_distance),
        )
        shifted_geom = shifted_geom[szinside]
        dists = cdist(shifted_geom, unique_shifted_positions, metric="sqeuclidean")
        for site, dists in zip(shifted_geom, dists):
            if np.all(dists > min_distance):
                unique_shifted_positions.append(site)
                is_original.append(False)
    unique_shifted_positions = np.array(unique_shifted_positions)
    is_original = np.array(is_original)

    # order by depth first, then horizontal position (unique goes the other way)
    order = np.lexsort(unique_shifted_positions.T)
    filled_geom = unique_shifted_positions[order]
    is_original = is_original[order]
    assert np.isclose(get_pitch(filled_geom), pitch)

    return filled_geom, is_original


# -- channel index creation


def make_channel_index(
    geom,
    radius,
    p=2,
    pad_val=None,
    to_torch=False,
    fill_holes=False,
):
    """
    Compute an array whose whose ith row contains the ordered
    (by distance) neighbors for the ith channel
    """
    C = geom.shape[0]
    if pad_val is None:
        pad_val = C

    if fill_holes:
        return make_filled_channel_index(
            geom, radius, p=p, pad_val=pad_val, to_torch=to_torch
        )

    # get neighbors matrix
    neighbors = squareform(pdist(geom, metric="minkowski", p=p)) <= radius

    # max number of neighbors for all channels
    n_neighbors = np.max(np.sum(neighbors, 0))

    # initialize channel index
    # entries for channels which don't have as many neighbors as
    # others will be filled with the total number of channels
    # (an invalid index into the recording, but this behavior
    # is useful e.g. in the spatial max pooling for deduplication)
    channel_index = np.full((C, n_neighbors), pad_val, dtype=int)

    # fill every row in the matrix (one per channel)
    for c in range(C):
        # indices of c's neighbors
        ch_idx = np.flatnonzero(neighbors[c])
        channel_index[c, : ch_idx.shape[0]] = ch_idx

    if to_torch:
        channel_index = torch.LongTensor(channel_index)

    return channel_index


def make_filled_channel_index(geom, radius, p=2, pad_val=None, to_torch=False):
    C = geom.shape[0]
    if pad_val is None:
        pad_val = C

    filled_geom, is_original = fill_geom_holes(geom)
    neighbors = cdist(geom, filled_geom, metric="minkowski", p=p) <= radius
    n_neighbors = np.max(np.sum(neighbors, 0))
    channel_index = np.full((C, n_neighbors), pad_val, dtype=int)

    # fill every row in the matrix (one per channel)
    for c in range(C):
        # indices of c's neighbors
        ch_idx = np.flatnonzero(neighbors[c])
        ch_valid = is_original[ch_idx]
        ch_idx[~ch_valid] = pad_val
        channel_index[c, : ch_idx.shape[0]] = ch_idx

    if to_torch:
        channel_index = torch.LongTensor(channel_index)

    return channel_index


def make_contiguous_channel_index(n_channels, n_neighbors=40):
    """Channel index with linear neighborhoods in channel id space"""
    channel_index = []
    for c in range(n_channels):
        low = max(0, c - n_neighbors // 2)
        low = min(n_channels - n_neighbors, low)
        channel_index.append(np.arange(low, low + n_neighbors))
    channel_index = np.array(channel_index)

    return channel_index


def make_pitch_channel_index(geom, n_neighbor_rows=1, pitch=None):
    """Channel neighborhoods which are whole pitches"""
    n_channels = geom.shape[0]
    if pitch is None:
        pitch = get_pitch(geom)
    neighbors = (
        np.abs(geom[:, 1][:, None] - geom[:, 1][None, :])
        <= n_neighbor_rows * pitch
    )
    channel_index = np.full((n_channels, neighbors.sum(1).max()), n_channels)
    for c in range(n_channels):
        my_neighbors = np.flatnonzero(neighbors[c])
        channel_index[c, : my_neighbors.size] = my_neighbors
    return channel_index


def make_pitch_channel_index_no_nans_for_plotting(
    geom, n_neighbor_rows=1, pitch=None
):
    """
    Channel neighborhoods which are whole pitches
    This function will select all the n_neighbor_rows inside the probe so that wfs are not nans
    """
    n_channels = geom.shape[0]
    if pitch is None:
        pitch = get_pitch(geom)
    neighbors = (
        np.abs(geom[:, 1][:, None] - geom[:, 1][None, :])
        <= n_neighbor_rows * pitch
    )
    channel_index = np.full((n_channels, neighbors.sum(1).max()), n_channels)
    for c in range(n_channels):
        my_neighbors = np.flatnonzero(neighbors[c])
        channel_index[c, : my_neighbors.size] = my_neighbors
        if channel_index[c].max() == n_channels:
            if c > n_channels // 2:
                channel_index[c] = np.arange(
                    n_channels - channel_index.shape[1], n_channels
                )
            else:
                channel_index[c] = np.arange(0, channel_index.shape[1])
    return channel_index


def full_channel_index(n_channels):
    """Everyone is everone's neighbor"""
    return (
        np.arange(n_channels)[None, :] * np.ones(n_channels, dtype=int)[:, None]
    )


# -- extracting single channels which were not part of padding


# This is used heavily in dartsort.transform, where for instance
# we might want to avoid the nans used as padding channels when
# fitting models.


def get_channels_in_probe(waveforms, max_channels, channel_index):
    n, t, c = waveforms.shape
    assert max_channels.shape == (n,)
    assert channel_index.ndim == 2 and channel_index.shape[1] == c
    waveforms = waveforms.permute(0, 2, 1)
    in_probe_index = channel_index < channel_index.shape[0]
    channels_in_probe = in_probe_index[max_channels]
    waveforms_in_probe = waveforms[channels_in_probe]
    return channels_in_probe, waveforms_in_probe


def set_channels_in_probe(
    waveforms_in_probe_src,
    waveforms_full_dest,
    channels_in_probe,
    in_place=False,
):
    waveforms_full_dest = waveforms_full_dest.permute(0, 2, 1)
    if not in_place:
        waveforms_full_dest = waveforms_full_dest.clone()
    waveforms_full_dest[channels_in_probe] = waveforms_in_probe_src
    return waveforms_full_dest.permute(0, 2, 1)


# -- channel subsetting


def channel_subset_by_radius(
    waveforms,
    max_channels,
    channel_index,
    geom,
    radius=None,
    n_channels_subset=None,
    fill_value=torch.nan,
    return_new_channel_index=True,
):
    """Restrict waveforms (or amplitude vectors) to channels inside a radius"""
    channel_index_mask = get_channel_index_mask(
        geom, channel_index, radius=radius, n_channels_subset=n_channels_subset
    )
    waveforms_subset = get_channel_subset(
        waveforms, max_channels, channel_index_mask
    )
    if return_new_channel_index:
        new_channel_index = mask_to_channel_index(
            channel_index, channel_index_mask
        )
        return waveforms_subset, new_channel_index
    return waveforms_subset


def channel_subset_by_index(
    waveforms,
    max_channels,
    channel_index_full,
    channel_index_new,
    fill_value=torch.nan,
):
    """Restrict waveforms to channels in new channel index"""
    # boolean mask of same shape as channel_index_full
    n_channels = channel_index_full.shape[0]
    # figure out which channels in channel_index_full are still
    # present in the form of a boolean mask
    channel_index_mask = torch.tensor(
        [
            np.isin(row_full, np.setdiff1d(row_new, [n_channels]))
            for row_full, row_new in zip(channel_index_full, channel_index_new)
        ]
    )
    return get_channel_subset(waveforms, max_channels, channel_index_mask)


def get_channel_index_mask(
    geom, channel_index, radius=None, n_channels_subset=None
):
    """Get a boolean mask showing if channels are inside a radial/linear subset

    Subsetting is controlled by a radius or by a number of channels. Radius
    takes priority.
    """
    assert geom.ndim == channel_index.ndim == 2
    assert geom.shape[0] == channel_index.shape[0]
    is_tensor = torch.is_tensor(channel_index)
    npx = torch if is_tensor else np

    if is_tensor:
        subset = torch.zeros(
            size=channel_index.shape, device=channel_index.device, dtype=bool
        )
        pgeom = F.pad(geom, (0, 0, 0, 1), value=torch.nan)
    else:
        subset = np.zeros(shape=channel_index.shape, dtype=bool)
        pgeom = np.pad(geom, [(0, 1), (0, 0)], constant_values=np.nan)

    for c in range(len(geom)):
        if radius is not None:
            dists = npx.square(geom[c][None] - pgeom[channel_index[c]]).sum(1)
            subset[c] = dists <= radius**2
        elif n_channels_subset is not None:
            low = max(0, c - n_channels_subset // 2)
            low = min(len(geom) - n_channels_subset, low)
            high = min(len(geom), low + n_channels_subset)
            subset[c] = (low <= channel_index[c]) & (channel_index[c] < high)
        else:
            subset[c] = True

    return subset


def mask_to_relative(channel_index_mask):
    assert channel_index_mask.ndim == 2
    max_sub_chans = channel_index_mask.sum(axis=1).max()
    original_max_neighbs = channel_index_mask.shape[1]
    n_channels_tot = channel_index_mask.shape[0]

    is_tensor = torch.is_tensor(channel_index_mask)
    if is_tensor:
        rel_sub_channel_index = torch.full(
            (n_channels_tot, max_sub_chans),
            original_max_neighbs,
            device=channel_index_mask.device,
        )
    else:
        rel_sub_channel_index = np.full(
            (n_channels_tot, max_sub_chans), original_max_neighbs
        )

    for i, mask in enumerate(channel_index_mask):
        if is_tensor:
            nz = mask.nonzero().squeeze()
            nnz = nz.numel()
        else:
            nz = np.flatnonzero(mask)
            nnz = nz.size
        if nnz:
            rel_sub_channel_index[i, :nnz] = nz

    return rel_sub_channel_index


def mask_to_channel_index(channel_index, channel_index_mask):
    assert channel_index.shape == channel_index_mask.shape
    max_sub_chans = channel_index_mask.sum(axis=1).max()
    n_channels = channel_index.shape[0]

    is_tensor = torch.is_tensor(channel_index_mask)
    if is_tensor:
        new_channel_index = torch.full(
            (n_channels, max_sub_chans),
            n_channels,
            device=channel_index_mask.device,
        )
    else:
        new_channel_index = np.full(
            (n_channels, max_sub_chans), fill_value=n_channels
        )

    for i, mask in enumerate(channel_index_mask):
        if is_tensor:
            which = mask.nonzero().squeeze()
            nnz = which.numel()
        else:
            which = np.flatnonzero(mask)
            nnz = which.size
        if nnz:
            new_channel_index[i, :nnz] = channel_index[i][which]
    return new_channel_index


def get_channel_subset(
    waveforms, max_channels, channel_index_mask, fill_value=np.nan
):
    """Given a binary mask indicating which channels to keep, grab those channels"""
    if waveforms.ndim == 3:
        N, T, C = waveforms.shape
        pads = [(0, 0), (0, 0)]
    elif waveforms.ndim == 2:
        # for instance, amplitudes
        N, C = waveforms.shape
        pads = [(0, 0)]
    else:
        assert False
    n_channels, C_ = channel_index_mask.shape
    assert C == C_

    # convert to relative channel offsets
    rel_sub_channel_index = mask_to_relative(channel_index_mask)
    if torch.is_tensor(waveforms):
        npx = torch
        waveforms = F.pad(waveforms, (0, 1), value=fill_value)
    else:
        npx = np
        waveforms = np.pad(
            waveforms, [*pads, (0, 1)], constant_values=fill_value
        )

    if waveforms.ndim == 3:
        return waveforms[
            npx.arange(N)[:, None, None],
            npx.arange(T)[None, :, None],
            rel_sub_channel_index[max_channels][:, None, :],
        ]

    # waveforms.ndim == 2:
    return waveforms[
        npx.arange(N)[:, None],
        rel_sub_channel_index[max_channels][:, :],
    ]
