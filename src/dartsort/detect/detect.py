import torch
import torch.nn.functional as F


def detect_and_deduplicate(
    traces,
    threshold,
    dedup_channel_index=None,
    peak_sign="neg",
    relative_peak_radius=5,
    dedup_temporal_radius=7,
    spatial_dedup_batch_size=512,
    exclude_edges=True,
    return_energies=False,
    detection_mask=None,
):
    """Detect and deduplicate peaks

    torch-based peak detection and deduplication, relying
    on max pooling and scatter operations

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
    detection_mask : tensor
        If supplied, this floating tensor of 1s or 0s will act
        as a gate to suppress detections in the 0s

    Returns
    -------
    times, chans : tensors of shape (n_peaks,)
        peak times in samples relative to start of traces, along
        with corresponding channels
    """
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

    # optionally remove censored peaks
    if detection_mask is not None:
        energies.mul_(detection_mask)

    # -- torch temporal relative maxima as pooling operation
    # we used to implement with max_pool2d -> unique, but
    # we can use max_unpool2d to speed up the second step
    # temporal max pooling
    energies, indices = F.max_pool1d_with_indices(
        energies.T.unsqueeze(0),
        kernel_size=2 * relative_peak_radius + 1,
        stride=1,
        padding=relative_peak_radius,
    )
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

    # -- temporal deduplication
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
        max_energies = max_energies.max(dim=1, keepdim=True).values
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
        return times, chans, max_energies[times, chans]

    return times, chans


def singlechan_template_detect_and_deduplicate(
    traces,
    singlechan_templates,
    threshold=40.0,
    trough_offset_samples=42,
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
