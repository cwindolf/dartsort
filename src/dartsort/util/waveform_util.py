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
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist, pdist, squareform
from tqdm.auto import trange

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
        y_uniq_at_x = np.unique(geom[at_x, direction])
        if y_uniq_at_x.size > 1:
            pitch = min(pitch, np.diff(y_uniq_at_x).min())

    if np.isinf(pitch):
        raise ValueError("Horizontal probe.")

    return pitch


def fill_geom_holes(geom):
    pitch = get_pitch(geom)
    pitches_pad = int(np.ceil(np.ptp(geom[:, 1]) / pitch))

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
        # sz = shifted_geom[:, 1]
        # szinside = sz == sz.clip(
        #     geom[:, 1].min() - np.sqrt(min_distance),
        #     geom[:, 1].max() + np.sqrt(min_distance),
        # )
        # shifted_geom = shifted_geom[szinside]
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


def regularize_geom(geom, radius=0):
    """Re-order, fill holes, and optionally expand geometry to make it 'regular'

    Used in make_regular_channel_index. That docstring has some info about what's
    going on here.
    """
    eps = pdist(geom).min() / 2.0
    radius = np.atleast_1d(radius)
    radius = np.broadcast_to(radius, geom.shape[1:])

    if torch.is_tensor(geom):
        geom = geom.numpy()

    rgeom = geom.copy()
    for j in range(geom.shape[1]):
        # skip empty dims
        if np.ptp(geom[:, j]) < eps:
            continue
        rgeom = _regularize_1d(rgeom, radius=max(eps, radius[j]), eps=eps, dim=j)

    # order regularized geom by depth and then x
    order = np.lexsort(rgeom.T)
    rgeom = rgeom[order]

    return rgeom, eps


def _regularize_1d(geom, radius, eps, dim=1):
    total = np.ptp(geom[:, dim])
    dim_pitch = get_pitch(geom, direction=dim)
    steps = int(np.ceil(total / dim_pitch))

    min_pos = geom[:, dim].min() - radius - eps
    max_pos = geom[:, dim].max() + radius + eps

    all_positions = []
    offset = np.zeros(geom.shape[1])
    for step in range(-steps, steps + 1):
        offset[dim] = step * dim_pitch
        # add positions within the radius
        offset_geom = geom + offset
        keepers = offset_geom[:, dim].clip(min_pos, max_pos) == offset_geom[:, dim]
        all_positions.append(offset_geom[keepers])

    all_positions = np.concatenate(all_positions)
    all_positions = np.unique(all_positions, axis=0)

    # deal with fp tolerance
    dists = squareform(pdist(all_positions))
    A = dists < eps
    n_neighbs = A.sum(0)
    if n_neighbs.max() == 1:
        return all_positions
    else:
        assert n_neighbs.max() > 1

    from scipy.cluster.hierarchy import linkage, fcluster

    Z = linkage(A.astype(np.float32))
    labels = fcluster(Z, 1.1, criterion="distance")
    labels -= 1
    return all_positions[np.unique(labels)]


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
    Compute an array whose whose ith row contains the neighbors for the ith channel
    """
    C = geom.shape[0]
    if not radius:
        return single_channel_index(C, to_torch=to_torch)
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
    if not radius:
        return single_channel_index(C, to_torch=to_torch)
    if pad_val is None:
        pad_val = C

    filled_geom, is_original = fill_geom_holes(geom)
    filled_kdt = KDTree(filled_geom)
    _, original_inds = filled_kdt.query(geom)
    assert np.array_equal(filled_geom[original_inds], geom)
    neighbors = cdist(geom, filled_geom, metric="minkowski", p=p) <= radius
    n_neighbors = np.max(np.sum(neighbors, 0))
    channel_index = np.full((C, n_neighbors), pad_val, dtype=int)

    # fill every row in the matrix (one per channel)
    for c in range(C):
        # indices of c's neighbors
        filled_ch_idx = np.flatnonzero(neighbors[c])
        ch_idx = np.searchsorted(original_inds, filled_ch_idx)
        ch_idx[np.logical_not(is_original[filled_ch_idx])] = pad_val
        channel_index[c] = ch_idx

    if to_torch:
        channel_index = torch.LongTensor(channel_index)

    return channel_index


def make_regular_channel_index(geom, radius, p=2, to_torch=False):
    """Channel index for multi-channel models

    In this channel index, the layout of channels around the max channel is
    always consistent -- but this is achieved by dummy channels. This makes
    for a consistent layout to input into multi-channel feature learners, at
    least relative to the detection channel. However, those learners have to
    deal with masked out dummy channels.

    Example:

    Let's say the probe looks like:
     o o
      o o
     o
      o o
    And, let's say that our channel index's radius is twice the vertical
    spacing (and say this is the same as the horizontal spacing). Then
    the top left channel might have (eyeballing it) 4 neighbors (excluding
    itself), the second row left channel maybe 5, the second row right channel
    only 4 due to the hole.

    This is padded out to a dummy probe (masked chans as xs) with holes filled in:
     x x x x x x
      x x x x x x
     x x o o x x
      x x o o x x
     x x o x x x
      x x o o x x
     x x x x x x
      x x x x x x
    Now, all of the real channels have the same number of neighbors and in fact
    exist in the same spatial relationship with their channel neighborhood. It's
    just that some of those neighbors are fake. But, extracting a radiul channel
    index here will lead to a consistent layout -- as long as the channels are
    ordered right (!!).

    Note that for probes which don't have a regular layout, this just doesn't
    make sense at all. I'm thinking of ones that have weird layouts where one channel
    is way bigger than the others and serves as a reference, etc -- throw those
    away!
    """
    rgeom, eps = regularize_geom(geom=geom, radius=radius)
    if np.array(radius).size > 1:
        radius = np.sqrt(np.square(radius).sum())

    # determine original geom's position in the regularized one, and which
    # channels are fake chans (they are unmatched in the query)
    kdt = KDTree(geom)
    dists, reg2orig = kdt.query(rgeom, k=1, distance_upper_bound=eps)
    # the usual extra padding chan
    reg2orig = np.concatenate((reg2orig, [kdt.n]))

    # make regularized channel index...
    rci = make_channel_index(rgeom, radius, p=p)

    # subset it to non-fake chans and replace regularized channel indices with
    # the corresponding original channel index (or len(geom) if fake)
    real_reg = np.flatnonzero(reg2orig < kdt.n)
    real_reg_ix = reg2orig[real_reg]
    ordered_real_reg = real_reg[np.argsort(real_reg_ix)]
    channel_index = reg2orig[rci[ordered_real_reg]]

    if to_torch:
        channel_index = torch.from_numpy(channel_index)

    return channel_index


def regularize_channel_index(geom, channel_index, p=2, to_torch=False):
    """Convert a channel index to the "regular" format

    Need to know the p used in the first place.
    """
    nchans = len(geom)

    # current radius
    radius = 0
    for j, row in enumerate(channel_index):
        row = row[row < nchans]
        radius = max(radius, cdist(geom[row], geom[j][None]).max())

    regular_channel_index = make_regular_channel_index(geom, radius, p=p)

    for c in range(nchans):
        regrow = regular_channel_index[c]
        origrow = channel_index[c]
        invalid = np.logical_not(np.isin(regrow, origrow))
        regular_channel_index[c, invalid] = nchans

        # check that this is a re-encoding of the same channels
        newc = regular_channel_index[c]
        newc = newc[newc < nchans]
        oldc = origrow[origrow < nchans]
        assert np.array_equal(np.sort(newc), np.sort(oldc))

    if to_torch:
        regular_channel_index = torch.from_numpy(regular_channel_index)

    return regular_channel_index


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
        np.abs(geom[:, 1][:, None] - geom[:, 1][None, :]) <= n_neighbor_rows * pitch
    )
    channel_index = np.full((n_channels, neighbors.sum(1).max()), n_channels)
    for c in range(n_channels):
        my_neighbors = np.flatnonzero(neighbors[c])
        channel_index[c, : my_neighbors.size] = my_neighbors
    return channel_index


def make_pitch_channel_index_no_nans_for_plotting(geom, n_neighbor_rows=1, pitch=None):
    """
    Channel neighborhoods which are whole pitches
    This function will select all the n_neighbor_rows inside the probe so that wfs are not nans
    """
    n_channels = geom.shape[0]
    if pitch is None:
        pitch = get_pitch(geom)
    neighbors = (
        np.abs(geom[:, 1][:, None] - geom[:, 1][None, :]) <= n_neighbor_rows * pitch
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


def full_channel_index(n_channels, to_torch=False):
    """Everyone is everone's neighbor"""
    ci = np.arange(n_channels)[None, :] * np.ones(n_channels, dtype=int)[:, None]
    if to_torch:
        ci = torch.tensor(ci)
    return ci


def single_channel_index(n_channels, to_torch=False):
    """Lonely islands"""
    ci = np.arange(n_channels)[:, None]
    if to_torch:
        ci = torch.tensor(ci)
    return ci


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
    waveforms_subset = get_channel_subset(waveforms, max_channels, channel_index_mask)
    if return_new_channel_index:
        new_channel_index = mask_to_channel_index(channel_index, channel_index_mask)
        return waveforms_subset, new_channel_index
    return waveforms_subset


def channel_subset_mask(channel_index_full, channel_index_new, to_torch=True):
    n_channels = channel_index_full.shape[0]
    mask = np.stack(
        [
            np.isin(row_full, np.setdiff1d(row_new, [n_channels]))
            for row_full, row_new in zip(channel_index_full, channel_index_new)
        ],
        axis=0,
    )
    if to_torch:
        mask = torch.tensor(mask)
    return mask


def channel_subset_by_index(
    waveforms,
    max_channels,
    channel_index_full,
    channel_index_new,
    fill_value=torch.nan,
    chunk_length=None,
):
    """Restrict waveforms to channels in new channel index"""
    # boolean mask of same shape as channel_index_full
    # figure out which channels in channel_index_full are still
    # present in the form of a boolean mask
    channel_index_mask = channel_subset_mask(channel_index_full, channel_index_new)
    if torch.is_tensor(channel_index_full):
        channel_index_mask = torch.from_numpy(channel_index_mask)
    return get_channel_subset(
        waveforms, max_channels, channel_index_mask, chunk_length=chunk_length
    )


def get_channel_index_mask(geom, channel_index, radius=None, n_channels_subset=None):
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
        new_channel_index = np.full((n_channels, max_sub_chans), fill_value=n_channels)

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
    waveforms,
    max_channels,
    channel_index_mask,
    fill_value=np.nan,
    chunk_length=None,
    in_place=False,
    out=None,
    rel_sub_channel_index=None,
    show_progress=True,
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
    is_torch = torch.is_tensor(waveforms)
    npx = np
    if is_torch:
        npx = torch

    if rel_sub_channel_index is None:
        rel_sub_channel_index = mask_to_relative(channel_index_mask)
        if is_torch:
            rel_sub_channel_index = torch.as_tensor(rel_sub_channel_index)
    n_chan_sub = rel_sub_channel_index.shape[1]

    if in_place and out is None:
        out = waveforms[..., :n_chan_sub]

    if chunk_length is not None:
        if out is None:
            out = npx.zeros_like(waveforms[..., :n_chan_sub])
        xrange = trange if show_progress else range
        for bs in xrange(0, len(out), chunk_length):
            sl = slice(bs, min(len(out), bs + chunk_length))
            get_channel_subset(
                waveforms[sl],
                max_channels[sl],
                channel_index_mask,
                fill_value=fill_value,
                chunk_length=None,
                out=out[sl],
                rel_sub_channel_index=rel_sub_channel_index,
            )
        return out

    # convert to relative channel offsets
    if is_torch:
        waveforms = F.pad(waveforms, (0, 1), value=fill_value)
        take_along_dim = torch.take_along_dim
    else:
        waveforms = np.pad(waveforms, [*pads, (0, 1)], constant_values=fill_value)

        def take_along_dim(x, ix, dim, out=None):
            res = np.take_along_axis(x, ix, axis=dim)
            if out is not None:
                out[:] = res
                res = out
            return res

    inds = rel_sub_channel_index[max_channels]
    if waveforms.ndim == 3:
        inds = inds[:, None, :]

    return take_along_dim(
        waveforms,
        inds,
        dim=waveforms.ndim - 1,
        out=out,
    )


def relative_channel_subset_index(channel_index_full, channel_index_new, to_torch=True):
    mask = channel_subset_mask(channel_index_full, channel_index_new, to_torch=True)
    rel_sub_channel_index = mask_to_relative(mask)
    return rel_sub_channel_index


def get_relative_subset(
    waveforms, max_channels, rel_sub_channel_index, fill_value=torch.nan
):
    waveforms = F.pad(waveforms, (0, 1), value=fill_value)
    index = rel_sub_channel_index[max_channels]
    if waveforms.ndim == 3:
        index = index[:, None, :].broadcast_to(
            index.shape[0], waveforms.shape[1], index.shape[1]
        )
    return torch.gather(waveforms, waveforms.ndim - 1, index)


def grab_main_channels(waveforms, main_channels, channel_index, keepdim=False):
    nc = len(channel_index)
    _, relative_positions = np.nonzero((channel_index == np.arange(nc)[:, None]))
    assert relative_positions.shape == (nc,)
    inds = relative_positions[main_channels]
    res = np.take_along_axis(waveforms, inds[:, None, None], axis=2)
    if keepdim:
        return res
    return res[:, :, 0]
