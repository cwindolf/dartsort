import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fftpack import next_fast_len
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
        return np.ptp(waveforms, axis=dim)
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
    if len(dims) == 1:
        if isinstance(multi_index, tuple):
            assert len(multi_index) == 1
            multi_index = multi_index[0]
        assert multi_index.ndim == 1
        return multi_index

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


def argrelmax(x, radius, threshold, exclude_edge=True):
    x1 = F.max_pool1d(
        x[None, None],
        kernel_size=2 * radius + 1,
        padding=radius,
        stride=1,
    )[0, 0]
    x1[x < x1] = 0
    F.threshold_(x1, threshold, 0.0)
    ix = torch.nonzero(x1)[:, 0]
    if exclude_edge:
        return ix[(ix > 0) & (ix < x.numel() - 1)]
    return ix


_cdtypes = {torch.float32: torch.complex64, torch.float64: torch.complex128}


def convolve_lowrank(
    traces,
    spatial_singular,
    temporal_components,
    padding=0,
    out=None,
):
    """Depthwise convolution of traces with templates"""
    n_templates, spike_length_samples, rank = temporal_components.shape
    out_len = traces.shape[1] + 2 * padding - spike_length_samples + 1
    if out is None:
        out = torch.empty(
            (n_templates, out_len),
            dtype=traces.dtype,
            device=traces.device,
        )
    else:
        assert out.shape == (n_templates, out_len)

    for q in range(rank):
        # units x time
        rec_spatial = spatial_singular[:, q, :] @ traces

        # convolve with temporal components -- units x time
        temporal = temporal_components[:, :, q]

        # conv1d with groups! only convolve each unit with its own temporal filter
        conv = F.conv1d(
            rec_spatial[None],
            temporal[:, None, :],
            groups=n_templates,
            padding=padding,
        )[0]

        # o-a turns out not to be helpful, sadly
        # conv = depthwise_oaconv1d(
        #     rec_spatial, temporal, padding=padding, f2=temporalf[:, :, q]
        # )

        if q:
            out += conv
        else:
            out.copy_(conv)

    # back to units x time (remove extra dim used for conv1d)
    return out


def nancov(x, correction=1):
    xtx = x.T @ x
    mask = x.isfinite().to(x)
    nobs = mask.T @ mask
    return xtx / (nobs + correction)


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


def _calc_oa_lens(s1, s2, block_size=None):
    """Modified from scipy"""
    import math
    from scipy.special import lambertw

    fallback = (s1 + s2 - 1, None, s1, s2)
    if s1 == s2 or s1 == 1 or s2 == 1:
        return fallback
    if s2 > s1:
        s1, s2 = s2, s1
        swapped = True
    else:
        swapped = False

    if s2 >= s1 / 2 and block_size is None:
        return fallback
    overlap = s2 - 1
    opt_size = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
    if block_size is None:
        block_size = next_fast_len(math.ceil(opt_size))

    # Use conventional FFT convolve if there is only going to be one block.
    if block_size >= s1:
        return fallback

    if not swapped:
        in1_step = block_size - s2 + 1
        in2_step = s2
    else:
        in1_step = s2
        in2_step = block_size - s2 + 1

    return block_size, overlap, in1_step, in2_step


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

    # shape_full = s1 + s2 - 1
    block_size, overlap, in1_step, in2_step = _calc_oa_lens(s1, s2)

    # overlap=None is a signal that no useful blocks for OA can
    # be found, do a vanilla FFT correlation
    if overlap is None:
        f1 = torch.fft.rfft(input, n=s1)
        f2 = torch.fft.rfft(torch.flip(weight, (-1,)), n=s1)
        f1.mul_(
            f2[
                :,
                None:,
            ]
        )
        res = torch.fft.irfft(f1, n=s1)
        valid_len = s1 - s2 + 1
        valid_start = s2 - 1
        assert valid_start >= padding
        res = res[:, valid_start - padding : valid_start + valid_len + padding]
        return res

    nstep1, pad1, nstep2, pad2 = steps_and_pad(
        s1, in1_step, s2, in2_step, block_size, overlap
    )

    if pad1 > 0:
        input = F.pad(input, (0, pad1))
    input = input.reshape(n1, nstep1, in1_step)

    # freq domain correlation
    f1 = torch.fft.rfft(input, n=block_size)
    if f2 is None:
        # flip direction of templates to perform cross-correlation
        f2 = torch.fft.rfft(torch.flip(weight, (-1,)), n=block_size)

    f1.mul_(f2[:, None, :])
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
    # oa = oa[:, : shape_final]
    # extract correct padding
    valid_len = s1 - s2 + 1
    valid_start = s2 - 1
    assert valid_start >= padding
    oa = oa[:, valid_start - padding : valid_start + valid_len + padding]

    return oa


def single_inv_oaconv1d(input, f2, s2, block_size, padding=0, norm="backward"):
    """Depthwise correlation (F.conv1d with groups=in_chans) with overlap-add"""
    # conv on last axis
    # assert input.ndim == weight.ndim == 2
    n1, s1 = input.shape
    assert s1 >= s2
    valid_len = s1 - s2 + 1
    valid_start = s2 - 1

    # shape_full = s1 + s2 - 1
    block_size, overlap, in1_step, in2_step = _calc_oa_lens(
        s1, s2, block_size=block_size
    )
    assert overlap is not None
    # case is hard to support...

    nstep1, pad1, nstep2, pad2 = steps_and_pad(
        s1, in1_step, s2, in2_step, block_size, overlap
    )

    if pad1 > 0:
        input = F.pad(input, (0, pad1))
    input = input.reshape(n1, nstep1, in1_step)

    # freq domain correlation
    f1 = torch.fft.rfft(input, n=block_size, norm=norm)
    if f1.shape[2] > f2.shape[0]:
        f2 = F.pad(f2, (0, f1.shape[2] - f2.shape[0]))

    f1.mul_(f2)
    res = torch.fft.irfft(f1, n=block_size, norm=norm)

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
    # oa = oa[:, : shape_final]
    # extract correct padding
    assert valid_start >= padding
    oa = oa[:, valid_start - padding : valid_start + valid_len + padding]

    return oa


# -- channel reindexing


def get_relative_index(source_channel_index, target_channel_index):
    """Pre-compute a channel reindexing helper structure.

    Inputs have shapes:
        source_channel_index.shape == (n_chans, n_source_chans)
        target_channel_index.shape == (n_chans, n_target_chans)

    This returns an array (relative_index) of shape (n_chans, n_target_chans)
    which knows how to translate between the source and target indices:

        relative_index[c, j] = index of target_channel_index[c, j] in source_channel_index[c]
                               if present, else n_source_chans (i.e., an invalid index)
                               (or, n_source chans if target_channel_index[c, j] is n_chans)

    See below:
        reindex(max_channels, source_waveforms, relative_index)
    """
    n_chans, n_source_chans = source_channel_index.shape
    n_chans_, n_target_chans = target_channel_index.shape
    assert n_chans == n_chans_
    relative_index = torch.full_like(target_channel_index, n_source_chans)
    for c in range(n_chans):
        row = source_channel_index[c]
        for j in range(n_target_chans):
            targ = target_channel_index[c, j]
            if targ == n_chans:
                continue
            mask = row == targ
            if not mask.any():
                continue
            (ixs,) = mask.nonzero(as_tuple=True)
            assert ixs.numel() == 1
            relative_index[c, j] = ixs[0]
    return relative_index


def reindex(
    max_channels,
    source_waveforms,
    relative_index,
    already_padded=False,
    pad_value=torch.nan,
):
    """"""
    rel_ix = relative_index[max_channels].unsqueeze(1)
    if not already_padded:
        source_waveforms = F.pad(source_waveforms, (0, 1), value=pad_value)
    return torch.take_along_dim(source_waveforms, rel_ix, dim=2)
