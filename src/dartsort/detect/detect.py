from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from ..util.py_util import panic
from ..util.torch_util import torch_compile

# holds values used for deduplicating identical peaks
_salt = {}
_pepper = {}


@torch.inference_mode()
def detect_and_deduplicate(
    traces: Tensor,
    threshold: float,
    peak_sign: Literal["pos", "neg", "both"] = "neg",
    relative_peak_radius=5,
    peak_channel_index: Tensor | None = None,
    dedup_temporal_radius=11,
    dedup_channel_index: torch.Tensor | None = None,
    trough_priority: float | None = None,
    batch_size=1024,
    *,
    remove_exact_duplicates=True,
    detection_mask: Tensor | None = None,
    exclude_edges=True,
    return_energies=False,
    dedup_salt_eps=1e-4,
):
    """Detect and deduplicate peaks

    torch-based peak detection and deduplication, relying
    on max pooling and scatter operations

    Parameters
    ----------
    traces : time by channels tensor
    threshold : float
    dedup_channel_index : channels by n_neighbors tensor
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

    Returns
    -------
    times, chans : tensors of shape (n_peaks,)
        peak times in samples relative to start of traces, along
        with corresponding channels
    """
    T, C = traces.shape
    all_peaks = torch.zeros_like(traces, dtype=torch.bool)
    will_dedup = bool(dedup_temporal_radius) or dedup_channel_index is not None
    pad = relative_peak_radius + dedup_temporal_radius
    batch_size = batch_size - 2 * pad
    assert batch_size > 0

    if dedup_channel_index is not None and remove_exact_duplicates:
        key = (traces.device.type, traces.device.index, C)
        if key in _salt:
            dedup_chans_salt = _salt[key]
        else:
            # favor earlier channels when deduplicating identical values (at float16 prec)
            dedup_chans_salt = torch.linspace(
                dedup_salt_eps, 0.0, steps=C, device=traces.device
            )
            _salt[key] = dedup_chans_salt = dedup_chans_salt.unsqueeze(1)
    else:
        dedup_chans_salt = None

    if dedup_temporal_radius and remove_exact_duplicates:
        slen = 2 * dedup_temporal_radius
        nrep = (T // slen) + 1
        Tp = slen * nrep
        key = (traces.device.type, traces.device.index, Tp, dedup_temporal_radius)
        if key in _pepper:
            dedup_time_salt = _pepper[key]
        else:
            # favor earlier times when deduplicating identical values (at float16 prec)
            dedup_time_salt = torch.linspace(
                dedup_salt_eps, 0.0, steps=slen, device=traces.device
            )
            dedup_time_salt = dedup_time_salt.repeat(nrep)
            _pepper[key] = dedup_time_salt = dedup_time_salt
    else:
        dedup_time_salt = None

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
            Xdd,
            dt=relative_peak_radius,
            neighbors=peak_channel_index,
        )
        detect = detect.logical_and_(peak)
        tmp = peak
        del peak

        # check if deduping
        if not will_dedup:
            all_peaks[i0:i1] = detect[:, istart:iend].T
            continue

        # -- deduplicate peaks
        if dedup_chans_salt is not None:
            Xdd[:-1] += dedup_chans_salt
        if dedup_time_salt is not None:
            Xdd[:-1] += dedup_time_salt[i00:i11]
        mask_out = torch.logical_not(detect, out=tmp)
        if detection_mask is not None:
            mask_out.logical_and_(detection_mask[i00:i11].T)
        Xdd[:-1].masked_fill_(mask_out, 0.0)

        # no-threshold max pool for deduplication
        dedup = _is_extreme(
            Xdd, dt=dedup_temporal_radius, neighbors=dedup_channel_index
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
