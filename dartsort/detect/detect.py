# TODO compare with spatial dedup before temporal dedup
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

    Returns
    -------
    times, chans : tensors of shape (n_peaks,)
        peak times in samples relative to start of traces, along
        with corresponding channels
    """
    nsamples, nchans = traces.shape
    if dedup_channel_index is not None:
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
    # for use with pooling functions, become 11TC
    energies = energies[None, None]

    # -- torch temporal relative maxima as pooling operation
    # we used to implement with max_pool2d -> unique, but
    # we can use max_unpool2d to speed up the second step
    # temporal max pooling
    energies, indices = F.max_pool2d_with_indices(
        energies,
        kernel_size=[2 * relative_peak_radius + 1, 1],
        stride=1,
        padding=[relative_peak_radius, 0],
    )
    # unpool will set non-maxima to 0
    energies = F.max_unpool2d(
        energies,
        indices,
        kernel_size=[2 * relative_peak_radius + 1, 1],
        stride=1,
        padding=[relative_peak_radius, 0],
        output_size=energies.shape,
    )
    # remove peaks smaller than our threshold
    energies[energies < threshold] = 0.0
    # could early exit if no peaks found
    # probably don't need to optimize for this case
    # if energies.max() == 0.0:
    #     return torch.tensor([]), torch.tensor([])

    # -- temporal deduplication
    max_energies, indices = F.max_pool2d_with_indices(
        energies,
        kernel_size=[2 * dedup_temporal_radius + 1, 1],
        stride=1,
        padding=[dedup_temporal_radius, 0],
    )
    # back to TC
    energies = energies[0, 0]
    max_energies = max_energies[0, 0]

    # -- spatial deduplication
    # we would like to max pool again on the other axis,
    # but that doesn't support any old radial neighborhood
    if dedup_channel_index is not None:
        # pad channel axis with extra chan of 0s
        max_energies = F.pad(max_energies, (0, 1))
        for batch_start in range(0, nsamples, spatial_dedup_batch_size):
            batch_end = batch_start + spatial_dedup_batch_size
            max_energies[batch_start:batch_end, :nchans] = torch.max(
                max_energies[batch_start:batch_end, dedup_channel_index], dim=2
            ).values
        max_energies = max_energies[:, :nchans]

    # if temporal/spatial max made you grow, you were not a peak!
    max_energies[max_energies > energies] = 0.0

    # sparsify and return
    times, chans = torch.nonzero(max_energies, as_tuple=True)
    # amplitudes = max_energies[times, chans]
    return times, chans
