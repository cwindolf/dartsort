from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from ..util.py_util import panic
from ..util.torch_util import torch_compile


@torch.inference_mode()
def detect_and_deduplicate(
    traces: Tensor,
    threshold: float,
    peak_sign: Literal["pos", "neg", "both"] = "neg",
    relative_peak_radius=5,
    peak_channel_index: Tensor | None = None,
    dedup_temporal_radius=11,
    dedup_neighborhoods: torch.Tensor | None = None,
    trough_priority: float | None = None,
    batch_size=1024,
    *,
    remove_exact_duplicates=True,
    detection_mask: Tensor | None = None,
    exclude_edges=True,
    return_energies=False,
):
    """Detect and deduplicate peaks

    torch-based peak detection and deduplication, relying
    on max pooling and scatter operations

    Parameters
    ----------
    traces : time by channels tensor
    threshold : float
    dedup_neighborhoods : channels by n_neighbors tensor
        Channel neighbors index. (See waveform_util for
        more on this format.) If supplied, peaks are kept
        only when they are the largest among their neighbors
        as described by this array
    peak_sign : one of "neg", "pos", "both"
    relative_peak_radius : int
        How many temporal neighbors must you be taller than
        to be considered a peak?
    dedup_temporal_radius : int
        Only the largest peak within this sliding radius
        will be kept
    remove_exact_duplicates : bool
        When peaks tie exactly, keep only the one which comes first
        in (time, channel) order rather than all of them

    Returns
    -------
    times, chans : tensors of shape (n_peaks,)
        peak times in samples relative to start of traces, along
        with corresponding channels
    """
    T = traces.shape[0]
    all_peaks = torch.zeros_like(traces, dtype=torch.bool)
    will_dedup = bool(dedup_temporal_radius) or dedup_neighborhoods is not None
    pad = relative_peak_radius + dedup_temporal_radius
    batch_size = batch_size - 2 * pad
    assert batch_size > 0

    for i0 in range(0, T, batch_size):
        i1 = min(T, i0 + batch_size)
        i00 = max(0, i0 - pad)
        istart = i0 - i00
        i11 = min(T, i1 + pad)
        iend = istart + (i1 - i0)

        # life is easier in channels-major here
        X = F.pad(traces[i00:i11].T, (0, 0, 0, 1))

        # data for threshold and deduplication criteria
        if peak_sign == "neg":
            Xdd = Xth = X.neg_()
        elif peak_sign == "pos":
            Xdd = Xth = X
        elif peak_sign == "both":
            if trough_priority:
                # deduplication takes trough priority into account.
                Xdd = F.leaky_relu(X, negative_slope=-trough_priority)
                Xth = X.abs_()
            else:
                Xdd = Xth = X.abs_()
        else:
            panic(peak_sign)

        # -- detect peaks
        detect = Xth[:-1] > threshold
        peak = _is_extreme(
            Xth,
            dt=relative_peak_radius,
            neighbors=peak_channel_index,
        )
        detect = detect.logical_and_(peak)
        tmp = peak
        del peak

        if detection_mask is not None:
            detect = detect.logical_and_(detection_mask[i00:i11].T)

        # check if deduping
        if not will_dedup:
            all_peaks[i0:i1] = detect[:, istart:iend].T
            continue

        # -- deduplicate peaks
        mask_out = torch.logical_not(detect, out=tmp)
        Xdd[:-1].masked_fill_(mask_out, 0.0)

        # no-threshold max pool for deduplication
        if remove_exact_duplicates:
            dedup = _is_unique_extreme(
                Xdd, dt=dedup_temporal_radius, neighbors=dedup_neighborhoods
            )
        else:
            dedup = _is_extreme(
                Xdd, dt=dedup_temporal_radius, neighbors=dedup_neighborhoods
            )
        all_peaks[i0:i1] = detect.logical_and_(dedup)[:, istart:iend].T

    if exclude_edges:
        all_peaks[0].zero_()
        all_peaks[-1].zero_()

    times, chans = all_peaks.nonzero(as_tuple=True)
    if return_energies:
        return times, chans, traces[times, chans].abs_()
    else:
        return times, chans


@torch_compile
def _is_extreme(
    X: Tensor,
    dt: int = 5,
    neighbors: Tensor | None = None,
):
    if neighbors is not None:
        # CT -> C, n_neighbors, T
        Xneighb = X[neighbors]
        Xmax = Xneighb.amax(dim=1)
    else:
        Xmax = X[:-1]
    Xmax = F.max_pool1d(
        Xmax[None],
        stride=(1,),
        kernel_size=(2 * dt + 1,),
        padding=(dt,),
    )
    Xmax = Xmax[0]

    # if max pool made you grow, or if thresholding made you grow,
    # then you were not a peak
    peak = torch.ge(X[:-1], Xmax)

    return peak


@torch_compile
def _is_unique_extreme(
    X: Tensor,
    dt: int = 5,
    neighbors: Tensor | None = None,
):
    if neighbors is not None:
        # CT -> C, n_neighbors, T
        Xneighb = X[neighbors]
        Xmax, neighb_pos = Xneighb.max(dim=1)
        argchan = neighbors.gather(1, neighb_pos)
    else:
        Xmax = X[:-1]
        argchan = None

    Xmax, argtime = F.max_pool1d_with_indices(
        Xmax[None],
        stride=(1,),
        kernel_size=(2 * dt + 1,),
        padding=(dt,),
    )
    argtime = argtime[0]

    tref = torch.arange(argtime.shape[1], device=X.device)
    peak = argtime == tref
    if argchan is not None:
        cref = torch.arange(argchan.shape[0], device=X.device).unsqueeze(1)
        peak = peak.logical_and_(argchan == cref)

    return peak


@torch_compile
def is_extreme_transpose_no_pad(
    X: Tensor,
    dt: int = 5,
    neighbors: Tensor | None = None,
    batch_size: int = 4096,
):
    if neighbors is not None:
        T, C = X.shape
        Xmax = X.new_empty((T, C - 1))
        for i0 in range(0, T, batch_size):
            i1 = min(T, i0 + batch_size)
            # TC -> T, C, n_neighbors
            Xneighb = X[i0:i1, neighbors]
            torch.amax(Xneighb, dim=2, out=Xmax[i0:i1])
    else:
        Xmax = X[:, :-1]
    Xmax = F.max_pool2d(
        Xmax[None, None, :, :],
        stride=(1, 1),
        kernel_size=(2 * dt + 1, 1),
        padding=(dt, 0),
    )
    Xmax = Xmax[0, 0]

    # if max pool made you grow, or if thresholding made you grow,
    # then you were not a peak
    peak = torch.ge(X[:, :-1], Xmax)

    return peak


_arange_cache = {}


def detect_and_globally_deduplicate(
    traces: Tensor,
    threshold: float,
    peak_sign: Literal["pos", "neg", "both"] = "neg",
    relative_peak_radius=5,
    peak_channel_index: Tensor | None = None,
    dedup_temporal_radius=11,
    trough_priority: float | None = None,
    *,
    remove_exact_duplicates=True,
    detection_mask: Tensor | None = None,
    exclude_edges=True,
):
    Xth = peak_sign_to_pos(traces, peak_sign)
    peak_map = is_extreme_transpose_no_pad(
        Xth, dt=relative_peak_radius, neighbors=peak_channel_index
    )
    return deduplicate_globally(
        traces,
        Xth,
        peak_map,
        threshold,
        peak_sign=peak_sign,
        dedup_temporal_radius=dedup_temporal_radius,
        trough_priority=trough_priority,
        remove_exact_duplicates=remove_exact_duplicates,
        detection_mask=detection_mask,
        exclude_edges=exclude_edges,
    )


def peak_sign_to_pos(
    traces: Tensor, peak_sign: Literal["pos", "neg", "both"] = "neg"
) -> Tensor:
    if peak_sign == "neg":
        Xth = traces.neg()
    elif peak_sign == "pos":
        Xth = traces.clone()
    elif peak_sign == "both":
        Xth = traces.abs()
    else:
        panic(peak_sign)
    Xth[:, -1].fill_(-torch.inf)
    return Xth


def update_peak_map(
    peak_map: Tensor,
    traces: Tensor,
    starts: Tensor,
    width: int,
    margin: int,
    peak_sign: Literal["pos", "neg", "both"] = "neg",
    relative_peak_radius: int = 5,
    peak_channel_index: Tensor | None = None,
) -> None:
    """Update peak_map in place where needed"""
    T = traces.shape[0]
    n = starts.numel()
    offsets = torch.arange(width, device=traces.device)
    time_ix = starts[:, None] + offsets

    Xth = peak_sign_to_pos(traces[time_ix].reshape(n * width, -1), peak_sign)
    slab_map = is_extreme_transpose_no_pad(
        Xth, dt=relative_peak_radius, neighbors=peak_channel_index
    )
    slab_map = slab_map.view(n, width, -1)

    keep = ((offsets >= margin) & (offsets < width - margin)).expand(n, width).clone()
    keep[starts == 0, :margin] = True
    keep[starts + width == T, width - margin :] = True
    peak_map[time_ix[keep]] = slab_map[keep]


def deduplicate_globally(
    traces: Tensor,
    Xth: Tensor,
    peak_map: Tensor,
    threshold: float,
    peak_sign: Literal["pos", "neg", "both"] = "neg",
    dedup_temporal_radius=11,
    trough_priority: float | None = None,
    *,
    remove_exact_duplicates=True,
    detection_mask: Tensor | None = None,
    exclude_edges=True,
):
    detect = Xth[:, :-1] > threshold
    detect = detect.logical_and_(peak_map)
    if detection_mask is not None:
        detect.logical_and_(detection_mask)
    tmp = torch.empty_like(detect)

    if peak_sign == "both" and trough_priority:
        # Xdd = F.leaky_relu(traces, negative_slope=-trough_priority)
        # equivalently, up to a constant...
        coef = (1 - trough_priority) / (1 + trough_priority)
        Xdd = Xth[:, :-1].add_(traces[:, :-1], alpha=coef)
    else:
        Xdd = Xth[:, :-1]
    mask_out = torch.logical_not(detect, out=tmp)
    Xdd.masked_fill_(mask_out, 0.0)

    maxdd, maxchan = Xdd.max(dim=1)
    tmaxdd, tinds = F.max_pool1d_with_indices(
        maxdd[None, None],
        kernel_size=(2 * dedup_temporal_radius + 1,),
        padding=(dedup_temporal_radius,),
        stride=(1,),
    )
    tmaxdd = tmaxdd[0, 0]
    tinds = tinds[0, 0]
    keep = maxdd >= tmaxdd

    if remove_exact_duplicates:
        ark = (traces.device.type, traces.device.index, traces.shape[0], exclude_edges)
        if ark in _arange_cache:
            tref = _arange_cache[ark]
        else:
            tref = torch.arange(traces.shape[0], device=traces.device)
            if exclude_edges:
                tref[[0, traces.shape[0] - 1]] = traces.shape[0] + 2
            _arange_cache[ark] = tref
        keep.logical_and_(tinds == tref)
    elif exclude_edges:
        keep[0].zero_()
        keep[-1].zero_()

    (which,) = keep.nonzero(as_tuple=True)
    return tinds[which], maxchan[which]
