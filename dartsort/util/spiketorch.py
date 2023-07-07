import torch
import torch.nn.functional as F


def ptp(waveforms, dim=1):
    return waveforms.max(dim=dim).values - waveforms.min(dim=dim).values


def ravel_multi_index(multi_index, dims):
    """torch implementation of np.ravel_multi_index

    Only implements order="C"

    Arguments
    ---------
    multi_index : tuple of LongTensor
    dims : tuple of ints
        Shape of tensor to be indexed

    Returns
    -------
    raveled_indices : LongTensor
        Indices into the flattened tensor of shape `dims`
    """
    assert len(multi_index) == len(dims)
    if any(
        torch.any((ix < 0) | (ix >= d)) for ix, d in zip(multi_index, dims)
    ):
        raise ValueError("Out of bounds indices in ravel_multi_index")

    # collect multi indices
    multi_index = torch.broadcast_tensors(*multi_index)
    multi_index = torch.stack(multi_index)
    # stride along each axis
    strides = (
        multi_index.new_tensor([1, *reversed(dims[1:])]).cumprod(0).flip(0)
    )
    # apply strides (along first axis) and reshape
    strides = strides.view(-1, *([1] * (multi_index.ndim - 1)))
    raveled_indices = (strides * multi_index).sum(0)
    return raveled_indices.view(-1)


def add_at_(dest, ix, src, sign=1):
    """Pytorch version of np.{add,subtract}.at

    Adds src into dest in place at indices (in dest) specified
    by tuple of index arrays ix. So, indices in ix should be
    locations in dest, but the arrays constituting ix should
    have shapes which broadcast to src.shape.

    Will add multiple times into the same indices. Check out
    docs for scatter_add_ and np.ufunc.at for more details.
    """
    if sign == -1:
        src = src.neg()
    elif sign != 1:
        src = sign * src
    dest.view(-1).scatter_add_(
        0,
        ravel_multi_index(ix, dest.shape),
        src.reshape(-1),
    )


def add_spikes_(
    traces,
    trough_times,
    max_channels,
    channel_index,
    waveforms,
    trough_offset=42,
    buffer=0,
    sign=1,
    in_place=True,
    already_padded=True,
    pad_value=torch.nan,
):
    """Add or subtract spikes into a tensor of traces

    Adds or subtracts (sign=-1) spike waveforms into an array of
    traces at times according to trough_times, trough_offset, and buffer,
    and at channels according to channel_index and max_channels.

    Uses add_at_ above to add overlapping regions multiple times.
    Regular old Tensor.add_() does not do this!
    """
    n_spikes, spike_length_samples, spike_n_chans = waveforms.shape
    T, C_ = traces.shape
    # traces may be padded with an extra chan, so C is the real n_chans
    C = C_ - int(already_padded)
    assert channel_index.shape == (C, spike_n_chans)
    assert max_channels.shape == trough_times.shape == (n_spikes,)

    if not already_padded:
        traces = F.pad(traces, (0, 1), value=pad_value)
    elif not in_place:
        traces = traces.clone()

    spike_sample_offsets = torch.arange(
        buffer - trough_offset,
        buffer - trough_offset + spike_length_samples,
        device=trough_times.device,
    )
    time_ix = trough_times[:, None, None] + spike_sample_offsets[None, :, None]
    chan_ix = channel_index[max_channels][:, None, :]
    add_at_(
        traces,
        (time_ix, chan_ix),
        waveforms,
        sign=sign,
    )
    return traces


def subtract_spikes_(
    traces,
    trough_times,
    max_channels,
    channel_index,
    waveforms,
    trough_offset=42,
    buffer=0,
    in_place=True,
    already_padded=True,
    pad_value=torch.nan,
):
    return add_spikes_(
        traces,
        trough_times,
        max_channels,
        channel_index,
        waveforms,
        trough_offset=trough_offset,
        buffer=buffer,
        in_place=in_place,
        already_padded=already_padded,
        pad_value=pad_value,
        sign=-1,
    )


def reduce_at_(dest, ix, src, reduce, include_self=True):
    """Pytorch version of np.ufunc.at for a couple of ones torch has

    Similar to add_at_ but reducing in place with things other than addition
    """
    dest.view(-1).scatter_reduce_(
        0,
        ravel_multi_index(ix, dest.shape),
        src.reshape(-1),
        reduce=reduce,
        include_self=include_self,
    )
