import math

import torch
import torch.nn.functional as F
from scipy.signal._signaltools import _calc_oa_lens
from torch.fft import irfft, rfft


def fast_nanmedian(x, axis=-1):
    is_tensor = torch.is_tensor(x)
    x = torch.nanmedian(torch.as_tensor(x), dim=axis).values
    if is_tensor:
        return x
    else:
        return x.numpy()


def ptp(waveforms, dim=1):
    is_tensor = torch.is_tensor(waveforms)
    if not is_tensor:
        return waveforms.ptp(axis=dim)
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

    # collect multi indices
    multi_index = torch.broadcast_tensors(*multi_index)
    multi_index = torch.stack(multi_index)
    # stride along each axis
    strides = multi_index.new_tensor([1, *reversed(dims[1:])]).cumprod(0).flip(0)
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
    flat_ix = ravel_multi_index(ix, dest.shape)
    dest.view(-1).scatter_add_(
        0,
        flat_ix,
        src.reshape(-1),
    )


def grab_spikes(
    traces,
    trough_times,
    max_channels,
    channel_index,
    trough_offset=42,
    spike_length_samples=121,
    buffer=0,
    already_padded=True,
    pad_value=torch.nan,
):
    """Grab spikes from a tensor of traces"""
    assert trough_times.ndim == 1
    assert max_channels.shape == trough_times.shape

    if not already_padded:
        traces = F.pad(traces, (0, 1), value=pad_value)

    spike_sample_offsets = torch.arange(
        buffer - trough_offset,
        buffer - trough_offset + spike_length_samples,
        device=trough_times.device,
    )
    time_ix = trough_times[:, None] + spike_sample_offsets[None, :]
    chan_ix = channel_index[max_channels]
    return traces[time_ix[:, :, None], chan_ix[:, None, :]]


def grab_spikes_full(
    traces,
    trough_times,
    trough_offset=42,
    spike_length_samples=121,
    buffer=0,
):
    """Grab spikes from a tensor of traces"""
    assert trough_times.ndim == 1
    spike_sample_offsets = torch.arange(
        buffer - trough_offset,
        buffer - trough_offset + spike_length_samples,
        device=trough_times.device,
    )
    time_ix = trough_times[:, None] + spike_sample_offsets[None, :]
    chan_ix = torch.arange(traces.shape[1], device=traces.device)
    return traces[time_ix[:, :, None], chan_ix[None, None, :]]


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
    assert trough_times.ndim == 1
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


_cdtypes = {torch.float32: torch.complex64, torch.float64: torch.complex128}


def real_resample(x, num, dim=0):
    """torch version of a special case of scipy.signal.resample

    Resamples x to have num elements on dim=dim. This is a direct
    copy of the scipy code in the case where there is no window
    and the data is not complex.
    """
    Nx = x.shape[dim]
    # f = rfft(x, dim=dim)
    cdtype = _cdtypes[x.dtype]

    # pad output spectrum
    newshape = list(x.shape)
    newshape[dim] = num // 2 + 1
    g = torch.zeros(newshape, dtype=cdtype, device=x.device)
    N = min(num, Nx)
    nyq = N // 2 + 1
    sl = [slice(None)] * x.ndim
    sl[dim] = slice(0, nyq)
    rfft(x, dim=dim, out=g[tuple(sl)])
    # g[tuple(sl)] = f[tuple(sl)]

    # split/join nyquist components if present
    if N % 2 == 0:
        sl[dim] = slice(N // 2, N // 2 + 1)
        if num < Nx:  # downsampling
            g[tuple(sl)] *= 2.0
        elif num > Nx:  # upsampling
            g[tuple(sl)] *= 0.5

    # inverse transform
    y = irfft(g, num, dim=dim)
    y *= float(num) / float(Nx)

    return y


def steps_and_pad(s1, in1_step, s2, in2_step, block_size, overlap):
    shape_final = s1 + s2 - 1
    # figure out n steps and padding
    if s1 > in1_step:
        nstep1 = math.ceil((s1 + 1) / in1_step)
        if (block_size - overlap) * nstep1 < shape_final:
            nstep1 += 1

        pad1 = nstep1 * in1_step - s1
    else:
        nstep1 = 1
        pad1 = 0

    if s2 > in2_step:
        nstep2 = math.ceil((s2 + 1) / in2_step)
        if (block_size - overlap) * nstep2 < shape_final:
            nstep2 += 1

        pad2 = nstep2 * in2_step - s2
    else:
        nstep2 = 1
        pad2 = 0
    return nstep1, pad1, nstep2, pad2


def depthwise_oaconv1d(input, weight, f2=None, padding=0):
    """Depthwise correlation (F.conv1d with groups=in_chans) with overlap-add"""
    # conv on last axis
    # assert input.ndim == weight.ndim == 2
    n1 = input.shape[0]
    n2 = weight.shape[0]
    assert n1 == n2
    s1 = input.shape[1]
    s2 = weight.shape[1]
    assert s1 >= s2

    shape_final = s1 + s2 - 1
    block_size, overlap, in1_step, in2_step = _calc_oa_lens(s1, s2)
    nstep1, pad1, nstep2, pad2 = steps_and_pad(
        s1, in1_step, s2, in2_step, block_size, overlap
    )

    if pad1 > 0:
        input = F.pad(input, (0, pad1))
    input = input.reshape(n1, nstep1, in1_step)

    # freq domain correlation
    f1 = torch.fft.rfft(input, n=block_size)
    if f2 is None:
        f2 = torch.fft.rfft(weight, n=block_size)
    # .conj() here to do cross-correlation instead of convolution (time reversal property of rfft)
    f1.mul_(f2.conj()[:, None, :])
    res = torch.fft.irfft(f1, n=block_size)

    # overlap add part with torch
    fold_input = res.reshape(n1, nstep1, block_size).permute(0, 2, 1)
    fold_out_len = nstep1 * in1_step + overlap
    fold_res = F.fold(
        fold_input,
        output_size=(1, fold_out_len),
        kernel_size=(1, block_size),
        stride=(1, in1_step),
    )
    assert fold_res.shape == (n1, 1, 1, fold_out_len)

    oa = fold_res.reshape(n1, fold_out_len)
    # this is the full convolution
    print(f"oaconv orig {oa.shape=}")
    # oa = oa[:, : shape_final]
    print(f"oaconv full {oa.shape=}")
    # extract correct padding
    valid_len = s1 - s2 + 1
    valid_start = s2 - 1
    assert valid_start >= padding
    oa = oa[:, valid_start - padding:valid_start + valid_len + padding]
    print(f"oaconv {oa.shape=} {valid_len=} {valid_start=} {padding=}")
    print(f"oaconv {(valid_start - padding)=} {(valid_start + valid_len + padding)=}")

    return oa
