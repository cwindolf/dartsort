import torch
import torch.nn.functional as F


def detect_and_deduplicate(
    traces,
    threshold,
    peak_sign="neg",
    relative_peak_radius=5,
    relative_peak_channel_index=None,
    dedup_temporal_radius=11,
    dedup_channel_index=None,
    spatial_dedup_batch_size=512,
    exclude_edges=True,
    return_energies=False,
    detection_mask=None,
    trough_priority=None,
    cumulant_order=None,
):
    """Detect and deduplicate peaks

    torch-based peak detection and deduplication, relying
    on max pooling and scatter operations

    TODO: reuse bufs and pre-pad.

    Arguments
    ---------
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
    relative_peak_radius = 5
    cumulant_win_size= 11
    if cumulant_order is not None:
        # TODO: combine.
        return detect_and_deduplicate_2d_filters(
            traces,
            cumulant_order=cumulant_order,
            threshold=threshold,
            cumulant_win_size=cumulant_win_size,
            dedup_channel_index=dedup_channel_index,
            peak_sign=peak_sign,
            relative_peak_radius=relative_peak_radius,
            spatial_dedup_batch_size=spatial_dedup_batch_size,
            exclude_edges=exclude_edges,
            return_energies=return_energies,
            detection_mask=detection_mask,
            trough_priority=trough_priority,
        )


    nsamples, nchans = traces.shape
    all_dedup = isinstance(dedup_channel_index, str) and dedup_channel_index == "all"
    if not all_dedup and dedup_channel_index is not None:
        assert dedup_channel_index.shape[0] == nchans

    # -- handle peak sign. we use max pool below, so make peaks positive
    if peak_sign == "neg":
        energies = traces.neg()
    elif peak_sign == "both":
        energies = traces.abs()
    else:
        assert peak_sign == "pos"
        # no need to copy since max pooling will
        energies = traces

    # we used to implement with max_pool -> unique, but we can use max_unpool
    # to speed up the second step temporal max pooling
    energies, indices = F.max_pool1d_with_indices(
        energies.T.unsqueeze(0),
        kernel_size=2 * relative_peak_radius + 1,
        stride=1,
        padding=relative_peak_radius,
    )
    # spatial peak criterion
    if relative_peak_channel_index is not None:
        # we are in 1CT right now
        max_energies = F.pad(energies[0], (0, 0, 0, 1))
        for batch_start in range(0, nsamples, spatial_dedup_batch_size):
            batch_end = batch_start + spatial_dedup_batch_size
            torch.amax(
                max_energies[relative_peak_channel_index, batch_start:batch_end],
                dim=1,
                out=max_energies[:nchans, batch_start:batch_end,],
            )
        energies.masked_fill_(max_energies[:nchans] > energies[0], 0.0)
    # unpool will set non-maxima to 0
    energies = F.max_unpool1d(
        energies,
        indices,
        kernel_size=2 * relative_peak_radius + 1,
        stride=1,
        padding=relative_peak_radius,
        output_size=energies.shape,
    )
    # remove peaks smaller than our threshold
    F.threshold_(energies, threshold, 0.0)
    if trough_priority and peak_sign == "both":
        tp = torch.where(traces.T < 0, trough_priority, 1.0)
        energies.mul_(tp)

    # -- temporal deduplication
    if detection_mask is not None:
        energies.mul_(detection_mask.T.to(energies))
    if dedup_temporal_radius:
        max_energies = F.max_pool1d(
            energies,
            kernel_size=2 * dedup_temporal_radius + 1,
            stride=1,
            padding=dedup_temporal_radius,
        )
    else:
        max_energies = energies
    # back to TC
    energies = energies[0].T
    max_energies = max_energies[0].T

    # -- spatial deduplication
    # this is max pooling within the channel index's neighborhood's
    if all_dedup:
        max_energies = max_energies.amax(dim=1, keepdim=True)
    elif dedup_channel_index is not None:
        # pad channel axis with extra chan of 0s
        max_energies = F.pad(max_energies, (0, 1))
        for batch_start in range(0, nsamples, spatial_dedup_batch_size):
            batch_end = batch_start + spatial_dedup_batch_size
            torch.amax(
                max_energies[batch_start:batch_end, dedup_channel_index],
                dim=2,
                out=max_energies[batch_start:batch_end, :nchans],
            )
        max_energies = max_energies[:, :nchans]

    # if temporal/spatial max made you grow, you were not a peak!
    if dedup_temporal_radius or (dedup_channel_index is not None):
        # max_energies[max_energies > energies] = 0.0
        max_energies.masked_fill_(max_energies > energies, 0.0)

    # sparsify and return
    if exclude_edges:
        max_energies[[0, -1], :] = 0.0
    times, chans = torch.nonzero(max_energies, as_tuple=True)

    if return_energies:
        return times, chans, energies[times, chans]

    return times, chans


def singlechan_template_detect_and_deduplicate(
    traces,
    singlechan_templates,
    threshold=40.0,
    trough_offset_samples=42,
    relative_peak_channel_index=None,
    dedup_channel_index=None,
    relative_peak_radius=5,
    dedup_temporal_radius=7,
    spatial_dedup_batch_size=512,
    exclude_edges=True,
    return_energies=False,
    detection_mask=None,
):
    """Detect spikes by per-channel matching with normalized templates

    See peel/universal_util.py to get some templates.
    """
    # convolve with templates
    conv_traces = traces.T.unsqueeze(1)
    conv_filt = singlechan_templates.unsqueeze(1)
    full = 2 * (singlechan_templates.shape[1] // 2)
    conv = F.conv1d(conv_traces, conv_filt, padding=full)

    # exactly align the convolution with the original traces
    offset = full - trough_offset_samples
    conv = conv[:, :, offset : offset + len(traces)]

    # convert to scaled deconvolution objective
    # when templates are normalized, the decrease in residual normsq
    # due to subtracting a scaled template is just the conv squared.
    obj = conv.square_().amax(dim=1).T

    # get peaks
    times, chans = detect_and_deduplicate(
        obj,
        threshold=threshold,
        relative_peak_channel_index=relative_peak_channel_index,
        dedup_channel_index=dedup_channel_index,
        peak_sign="pos",
        relative_peak_radius=relative_peak_radius,
        dedup_temporal_radius=dedup_temporal_radius,
        spatial_dedup_batch_size=spatial_dedup_batch_size,
        exclude_edges=exclude_edges,
        return_energies=False,
        detection_mask=detection_mask,
    )

    if return_energies:
        return times, chans, traces[times, chans]

    return times, chans

@torch.no_grad()  # remove if you need gradients
def compute_sliding_2d_cumulant(data: torch.Tensor, order: int, win_size: int):
    """
    Efficient sliding cumulant over 2D windows (mean or variance only) without unfold/view.
    Args:
        data: (C, H, W) tensor
        order: 1 for mean, 2 for variance
        win_size: odd kernel size
    Returns:
        (C, H, W) tensor
    """
    if order == 0:
        return data
    if win_size % 2 == 0:
        raise ValueError("win_size must be odd for symmetric padding")
    if order not in (1, 2):
        raise ValueError("This fast path supports only order=1 (mean) or order=2 (variance).")

    C, H, W = data.shape
    pad = win_size // 2

    # Pad spatial dims with reflect to match your original behavior
    x = F.pad(data, (pad, pad, pad, pad), mode='reflect')  # (C, H+2p, W+2p)

    # avg_pool2d expects (N, C, H, W); use a dummy batch dim
    x = x.unsqueeze(0)  # (1, C, H+2p, W+2p)

    # Local mean via average pooling (stride=1, valid after our manual padding)
    mean = F.avg_pool2d(x, kernel_size=win_size, stride=1)      # (1, C, H, W)

    if order == 1:
        return mean.squeeze(0)

    # order == 2: variance = E[x^2] - (E[x])^2
    ex2 = F.avg_pool2d(x * x, kernel_size=win_size, stride=1)   # (1, C, H, W)
    var = ex2 - mean * mean

    # Numerical guard: tiny negative values to zero due to FP roundoff
    var = torch.clamp(var, min=0.0)

    return var.squeeze(0)

# def compute_sliding_2d_cumulant(radiality, order, win_size, chunk_size=256):
#     """
#     Compute sliding cumulant statistics (mean, variance, skewness, kurtosis) over spatial 2D windows,
#     processing the W-axis in manageable chunks.
#
#     Args:
#         radiality: (C, H, W) tensor
#         order: cumulant order (1=mean, 2=variance, 3=skewness, 4=kurtosis)
#         win_size: size of spatial window (must be odd for symmetry)
#         chunk_size: number of W-axis columns to process per chunk (default: 30,000)
#
#     Returns:
#         Tensor of shape (C, H, W) with the cumulant statistic at each spatial location
#     """
#     C, H, W = radiality.shape
#
#     if win_size % 2 == 0:
#         raise ValueError("win_size must be odd for symmetric padding")
#
#     padding = win_size // 2
#
#     # Pad spatial dimensions
#     padded = F.pad(radiality.unsqueeze(1), (padding, padding, padding, padding), mode='reflect')  # (C, 1, H+2p, W+2p)
#
#     results = []
#     start = 0
#     while start < W:
#         end = min(start + chunk_size, W)
#
#         # Extract current chunk with extra padding
#         # Pad adds padding to both sides, so for columns start:end in the original,
#         # we need columns start:end + 2*padding in padded space
#         padded_start = start
#         padded_end = end + 2 * padding
#
#         chunk = padded[:, :, :, padded_start:padded_end]  # (C, 1, H+2p, chunk_width + 2p)
#
#         # Unfold to extract sliding windows
#         windows = chunk.unfold(2, win_size, 1).unfold(3, win_size, 1)  # (C, 1, H, chunk_width, win_size, win_size)
#         windows = windows.contiguous().view(C, H, end - start, -1)  # (C, H, chunk_width, win_size*win_size)
#
#         # Cumulant calculations
#         if order == 1:
#             result = windows.mean(dim=-1)
#         elif order == 2:
#             result = windows.var(dim=-1, unbiased=False)
#         elif order == 3:
#             mean = windows.mean(dim=-1, keepdim=True)
#             std = windows.std(dim=-1, unbiased=False, keepdim=True) + 1e-8
#             result = (((windows - mean) / std) ** 3).mean(dim=-1)
#         elif order == 4:
#             mean = windows.mean(dim=-1, keepdim=True)
#             std = windows.std(dim=-1, unbiased=False, keepdim=True) + 1e-8
#             result = (((windows - mean) / std) ** 4).mean(dim=-1) - 3
#         else:
#             raise ValueError(f"Unsupported order: {order}")
#
#         results.append(result)  # (C, H, chunk_width)
#         start = end
#
#     return torch.cat(results, dim=-1)  # (C, H, W)


def detect_and_deduplicate_2d_filters(
        traces,
        cum_traces=None,
        cumulant_order=2,
        cumulant_win_size=11,
        threshold=2.0,
        dedup_channel_index=None,          # kept for API; unused here
        peak_sign="neg",                   # "neg" or "both" supported; localization is on troughs
        relative_peak_radius=5,            # spatial+temporal NMS radius
        spatial_dedup_batch_size=512,      # kept for API; unused here
        exclude_edges=True,
        return_energies=False,
        detection_mask=None,
        trough_priority=None,              # kept for API; unused here
        # NEW knobs (sane defaults):
        future_peak_window=(10, 5),        # lookahead (time, space) for the “followed by a peak” criterion
        min_contrast_abs=2.0,             # require peak - trough >= this (if set)
):
    """
    Localize trough minima efficiently while still using a robust energy to gate candidates.
    - Gating: cum_traces (std/radiality) >= threshold
    - Localization: trough = local 2D minimum of 'traces'
    - Validation: a future positive peak exists within 'future_peak_window'
    - Dedup: NMS on contrast map within 'relative_peak_radius'
    """
    assert traces.dim() == 2, "traces must be (T, C)"
    T, C = traces.shape

    # Build (1,1,T,C) tensors
    tr4 = traces.unsqueeze(0).unsqueeze(0)               # (1,1,T,C)

    # cumulant computed if needed
    if cumulant_order and cum_traces is None:
        # expects (C,H,W), so adapt; we just need a fast local std-like map
        ct = compute_sliding_2d_cumulant(traces.unsqueeze(0), cumulant_order, cumulant_win_size)
        cum_traces = ct.unsqueeze(0)                      # (1,1,T,C)
    elif cum_traces is not None:
        cum_traces = cum_traces                           # assume already (1,1,T,C)
    else:
        # fall back to a light local absolute deviation proxy using pooling (optional)
        pad = cumulant_win_size // 2
        mean = F.avg_pool2d(tr4, kernel_size=cumulant_win_size, stride=1, padding=pad)
        cum_traces = F.avg_pool2d((tr4 - mean).abs(), kernel_size=cumulant_win_size, stride=1, padding=pad)

    # Threshold the cum_traces
    thresh_mask = cum_traces >= threshold                     # (1,1,T,C)
    if detection_mask is not None:
        dm = detection_mask.to(thresh_mask.dtype).unsqueeze(0).unsqueeze(0)
        thresh_mask = thresh_mask & (dm > 0)

    # Local trough candidates via 2D min-pooling (i.e., max-pooling on -traces)
    k = 2 * relative_peak_radius + 1
    neg_tr4 = -tr4
    local_neg_max = F.max_pool2d(neg_tr4, kernel_size=k, stride=1, padding=relative_peak_radius)
    trough_mask = (neg_tr4 == local_neg_max)              # equal to local minimum in original
    # Only keep troughs inside the gated regions
    trough_mask = trough_mask & thresh_mask

    # Positive pixel mask
    pos_mask = (tr4 > 0).float()  # (1,1,T,C)
    neigh_radius = 3

    # Neighborhood size
    ksize = 2 * neigh_radius + 1
    neigh_area = ksize * ksize

    # Count positives in each neighborhood (fast via max_pool2d over floats with stride=1)
    # For counting, we use avg_pool2d multiplied by area instead of max_pool
    pos_frac = F.avg_pool2d(pos_mask, kernel_size=ksize, stride=1, padding=neigh_radius)

    # Keep only candidates with <= frac_thresh positive fraction
    keep_mask = pos_frac <= 0.3

    trough_mask = trough_mask & keep_mask

    # 5) “Followed by a peak” + contrast check (fast, causal window)
    # future max of positive part within [t, t+W]
    pos_tr4 = tr4.clamp_min(0)
    fut_max = F.max_pool2d(pos_tr4, kernel_size=future_peak_window, stride=1, padding=(0, 0))
    # Align back to (T,C): pad bottom with window to keep same length
    fut_max = F.pad(fut_max, (0, future_peak_window[1]-1, 0, future_peak_window[0]-1))  # (1,1,T,C)

    # Peak–trough contrast at each (t,c): peak_future - trough_value
    trough_val = tr4                                        # (1,1,T,C), typically negative at troughs
    contrast = fut_max - trough_val                         # higher is better

    # Contrast thresholds
    contrast_mask = torch.ones_like(trough_mask, dtype=torch.bool)
    if min_contrast_abs is not None:
        contrast_mask = contrast_mask & (contrast >= min_contrast_abs)

    # Keep only troughs that pass contrast checks
    trough_mask = trough_mask & contrast_mask

    # 6) Non-maximum suppression on contrast to deduplicate troughs
    #    We only score positions that are troughs; everything else is zeroed.
    scored = contrast * trough_mask
    local_max = F.max_pool2d(scored, kernel_size=k, stride=1, padding=relative_peak_radius)
    keep = (scored == local_max) & (scored > 0)

    # 7) Edges
    if exclude_edges:
        keep[..., 0, :] = False
        keep[..., -1, :] = False

    # 8) Return indices (+ optional energies = contrast at kept minima)
    times, chans = torch.nonzero(keep[0,0], as_tuple=True)

    if return_energies:
        return times, chans, scored[0,0][times, chans]

    # import matplotlib.pyplot as plt
    # import matplotlib
    # import numpy as np
    # matplotlib.use("TkAgg")
    # fig, ax = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
    # ax[0, 0].set_title("Raw data")
    # ax[0, 0].imshow(traces.T, aspect='auto', interpolation='nearest', cmap='seismic', vmin=-5, vmax=5)
    # ax[0, 0].set_xlabel('Time (ms)')
    # ax[0, 0].set_ylabel('Amplitude (V)')
    # ax[0, 1].imshow(cum_traces[0,0].T, aspect='auto', interpolation='nearest', cmap='seismic', vmin=-15, vmax=15)
    # ax[0, 1].set_title("Cumulant of raw data")
    # ax[0, 1].set_xlabel('Time (ms)')
    # ax[0, 1].set_ylabel('Amplitude (V)')
    # ax[1, 0].set_title("Localizations over raw data")
    # ax[1, 0].imshow(traces.T, aspect='auto', interpolation='nearest', cmap='seismic', vmin=-5, vmax=5)
    # ax[1, 0].set_xlabel('Time (ms)')
    # ax[1, 0].set_ylabel('Amplitude (V)')
    # ax[1, 0].plot(times, chans, 'yo', markersize=6)
    # if(detection_mask is not None):
    #     ax[1, 1].set_title("Localizations over detection mask")
    #     ax[1, 1].imshow(detection_mask.T, aspect='auto', interpolation='nearest')
    #     ax[1, 1].set_xlabel('Time (ms)')
    #     ax[1, 1].set_ylabel('Amplitude (V)')
    #     ax[1, 1].plot(times, chans, 'yo', markersize=6)
    # plt.tight_layout()
    # plt.show()

    return times, chans
