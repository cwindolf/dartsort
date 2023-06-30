"""
A collection of helper functions for dealing with which channels
waveforms are extracted on.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist

# -- channel index creation


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
    radius,
    fill_value=torch.nan,
    return_new_channel_index=True,
):
    """Restrict waveforms (or amplitude vectors) to channels inside a radius"""
    channel_index_mask = get_channel_index_mask(
        geom, channel_index, radius=radius
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


def get_channel_index_mask(geom, channel_index, radius=None, n_channels=None):
    """Get a boolean mask showing if channels are inside a radial/linear subset

    Subsetting is controlled by a radius or by a number of channels. Radius
    takes priority.
    """
    assert geom.ndim == channel_index.ndim == 2
    assert geom.shape[0] == channel_index.shape[0]
    subset = np.zeros(shape=channel_index.shape, dtype=bool)
    pgeom = np.pad(geom, [(0, 1), (0, 0)], constant_values=-2 * geom.max())
    for c in range(len(geom)):
        if radius is not None:
            dists = cdist([geom[c]], pgeom[channel_index[c]]).ravel()
            subset[c] = dists <= radius
        elif n_channels is not None:
            low = max(0, c - n_channels // 2)
            low = min(len(geom) - n_channels, low)
            high = min(len(geom), low + n_channels)
            subset[c] = (low <= channel_index[c]) & (channel_index[c] < high)
        else:
            subset[c] = True
    return subset


def mask_to_relative(channel_index_mask):
    assert channel_index_mask.ndim == 2
    rel_sub_channel_index = []
    max_sub_chans = channel_index_mask.sum(axis=1).max()
    C = channel_index_mask.shape[1]
    for mask in channel_index_mask:
        s = np.flatnonzero(mask)
        s = list(s) + [C] * (max_sub_chans - len(s))
        rel_sub_channel_index.append(s)
    rel_sub_channel_index = np.array(rel_sub_channel_index)
    return rel_sub_channel_index


def mask_to_channel_index(channel_index, channel_index_mask):
    assert channel_index.shape == channel_index_mask.shape
    max_sub_chans = channel_index_mask.sum(axis=1).max()
    n_channels = channel_index.shape[0]
    new_channel_index = np.full(
        (n_channels, max_sub_chans), fill_value=n_channels
    )
    for i, mask in enumerate(channel_index_mask):
        which = np.flatnonzero(mask)
        new_channel_index[i, : which.size] = channel_index[i][which]
    return new_channel_index


def get_channel_subset(
    waveforms, max_channels, channel_index_mask, fill_value=np.nan
):
    """Given a binary mask indicating which channels to keep, grab those channels"""
    if waveforms.ndim == 3:
        N, T, C = waveforms.shape
    elif waveforms.ndim == 2:
        # for instance, amplitudes
        N, C = waveforms.shape
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
            waveforms, [(0, 0), (0, 0), (0, 1)], constant_values=fill_value
        )

    if waveforms.ndim == 3:
        return waveforms[
            npx.arange(N)[:, None, None],
            npx.arange(T)[None, :, None],
            rel_sub_channel_index[max_channels][:, None, :],
        ]

    # waveforms.ndim == 2:
    return waveforms[
        npx.arange(N)[:, None, None],
        rel_sub_channel_index[max_channels][:, None, :],
    ]
