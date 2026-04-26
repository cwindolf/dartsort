import math
import warnings
from logging import getLogger
from typing import overload

import linear_operator
import numpy as np
import torch
import torch.nn.functional as F
from linear_operator.utils.cholesky import psd_safe_cholesky
from scipy.fftpack import next_fast_len
from scipy.spatial.distance import squareform
from sklearn.utils.extmath import svd_flip
from torch import Tensor
from torch.fft import irfft, rfft
from tqdm.auto import trange

HAVE_CUPY = False
try:
    import cupy as cp

    HAVE_CUPY = True
except ImportError:
    cp = None  # type: ignore

logger = getLogger(__name__)
log2pi = torch.log(torch.tensor(2 * np.pi))
_1 = torch.tensor(1.0)
_0 = torch.tensor(0.0)


def spawn_torch_rg(
    seed: int | np.random.Generator | torch.Generator = 0,
    device: str | torch.device | None = "cpu",
):
    if device is None:
        device = "cpu"
    device = torch.device(device)
    if isinstance(seed, torch.Generator) and seed.device == device:
        return seed
    elif isinstance(seed, torch.Generator):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed.seed())
        return generator
    nprg = np.random.default_rng(seed)
    seeder = nprg.spawn(1)[0]
    seed = int.from_bytes(seeder.bytes(8))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def ll_via_inv_quad(cov, y, inv_quad_only=False):
    inv_quad, logdet = linear_operator.inv_quad_logdet(
        cov, y.T, logdet=True, reduce_inv_quad=False
    )
    assert inv_quad is not None
    if inv_quad_only:
        return inv_quad
    ll = inv_quad.add_(logdet + log2pi * y.shape[1]).mul_(-0.5)
    return ll


def fast_nanmedian(x, axis=-1):
    is_tensor = torch.is_tensor(x)
    x = torch.nanmedian(torch.as_tensor(x), dim=axis).values
    if is_tensor:
        return x
    else:
        return x.numpy()


def nanmean(x, axis=-1):
    is_tensor = torch.is_tensor(x)
    x = torch.nanmean(torch.as_tensor(x), dim=axis)
    if is_tensor:
        return x
    else:
        return x.numpy()


def sign(x):
    """torch.sign, but nonzero."""
    s = torch.sign(x)
    s.add_(0.1)
    torch.sign(s, out=s)
    return s


@overload
def ptp(waveforms: Tensor, dim: int = 1, keepdims: bool = False) -> Tensor: ...


@overload
def ptp(waveforms: np.ndarray, dim: int = 1, keepdims: bool = False) -> np.ndarray: ...


def ptp(waveforms, dim=1, keepdims=False):
    is_tensor = torch.is_tensor(waveforms)
    waveforms = torch.asarray(waveforms)
    if waveforms.shape[dim] > 1:
        amin, amax = torch.aminmax(waveforms, dim=dim)
        v = amax.sub_(amin)
    else:
        v = waveforms.abs().amax(dim=dim)
    if keepdims:
        v = v.unsqueeze(dim)
    if is_tensor:
        return v
    return v.numpy()


@torch.jit.script
def mean_elbo_dim1(Q: Tensor, log_liks: Tensor) -> Tensor:
    logQ = Q.log().nan_to_num_(nan=None, neginf=0.0)
    logP = log_liks.nan_to_num(nan=None, neginf=0.0)
    oelbo = logP.sub_(logQ).mul_(Q).sum(dim=1)
    oelbo = oelbo.mean()
    return oelbo


@torch.jit.script
def elbo(Q: Tensor, log_liks: Tensor, reduce_mean: bool = True, dim: int = 1) -> Tensor:
    logQ = Q.log().nan_to_num_(neginf=0.0)
    logP = log_liks.nan_to_num(neginf=0.0)
    oelbo = logP.sub_(logQ).mul_(Q).sum(dim=dim)
    if reduce_mean:
        oelbo = oelbo.mean()
    return oelbo


@torch.jit.script
def entropy(Q: Tensor, reduce_mean: bool = True, dim: int = 1) -> Tensor:
    logQ = Q.log().nan_to_num_(neginf=0.0)
    H = logQ.mul_(Q).sum(dim=dim)
    if reduce_mean:
        H = H.mean()
    return H.neg_()


@torch.jit.script
def ecl(
    resps: Tensor, log_liks: Tensor, cl_alpha: float = 1.0, reduce_mean: bool = True
) -> Tensor:
    h = entropy(resps, dim=1, reduce_mean=reduce_mean)
    log_lik = log_liks.logsumexp(dim=1)
    if reduce_mean:
        log_lik = log_lik.mean()
    crit = log_lik - cl_alpha * h
    return crit


def taper(waveforms, t_start=10, t_end=20, dim=1):
    nt = waveforms.shape[dim]
    t0 = torch.linspace(-torch.pi, 0.0, steps=t_start)
    t1 = torch.zeros(nt - t_start - t_end)
    t2 = torch.linspace(0, -torch.pi, steps=t_end)
    domain = torch.concatenate((t0, t1, t2))
    window = torch.cos(domain).add(1.0).div(2.0)
    for j in range(dim):
        window = window.unsqueeze(0)
    for j in range(dim + 1, waveforms.ndim):
        window = window.unsqueeze(-1)
    return waveforms * window.to(waveforms)


def minmax(x: np.ndarray) -> np.ndarray:
    x = x - np.min(x)
    return x / np.max(x)


def svd_lowrank_helper(
    x: Tensor,
    rank: int,
    *,
    n_oversamples: int = 10,
    fit_dtype=torch.double,
    niter=21,
    M=None,
    with_loadings: bool = False,
    device=None,
) -> tuple[Tensor | None, Tensor, Tensor, Tensor]:
    assert x.ndim == 2
    q = min(rank + n_oversamples, *x.shape)
    orig_dtype = x.dtype
    x = x.to(dtype=fit_dtype, device=device)
    if M is not None:
        M = M.to(dtype=fit_dtype, device=device)
    U, S, V = torch.svd_lowrank(x, q=q, M=M, niter=niter)
    U = U[..., :rank]
    S = S[..., :rank]
    V = V[..., :rank]
    Vt = V.T

    # fix sign ambiguity for better reproducibility in unit tests
    U, Vt = svd_flip(U.numpy(force=True), Vt.numpy(force=True))
    U = torch.asarray(U, dtype=orig_dtype, device=S.device).contiguous()
    Vt = torch.asarray(Vt, dtype=orig_dtype, device=S.device).contiguous()

    if with_loadings:
        loadings = U * S[..., None, :]
    else:
        loadings = None
    components = Vt
    explained_variance = S.square() / (x.shape[0] - 1)
    whitener = torch.sqrt(explained_variance)
    return loadings, components, explained_variance, whitener


def shared_temporal_pconv(temporal_comps: Tensor, up_temporal_comps: Tensor) -> Tensor:
    rank, t = temporal_comps.shape
    assert t >= rank
    rank_, up, t_ = up_temporal_comps.shape
    assert t == t_
    assert rank == rank_

    # NIL = rank, 1, t
    inp = temporal_comps[:, None, :]
    # OIL = rank * up, 1, t. rank major, not up major.
    fil = up_temporal_comps.reshape(rank * up, 1, t)
    # NOL = rank, rank * up, 2 * t - 1
    pconv = F.conv1d(input=inp, weight=fil, padding=t - 1)
    assert pconv.shape == (rank, rank * up, 2 * t - 1)
    pconv = pconv.view(rank, rank, up, 2 * t - 1)
    pconv = torch.flip(pconv, dims=(3,))

    return pconv


@torch.jit.script
def full_shared_pconv(
    tconv: Tensor, spatial_sing: Tensor, batch_size: int = 64
) -> Tensor:
    rank, rank_, up, conv_len = tconv.shape
    n_units, rank__, chans = spatial_sing.shape
    assert rank == rank_ == rank__
    out = spatial_sing.new_empty((n_units, n_units, up, conv_len))
    spatial_sing_flat = spatial_sing.view(n_units * rank, chans)
    tconv_flat = tconv.view(rank * rank, up * conv_len)

    for i0 in range(0, n_units, batch_size):
        i1 = min(n_units, i0 + batch_size)
        chunksz = (i1 - i0) * n_units
        spatial_left = spatial_sing[i0:i1]
        spatial_outer = spatial_left.view((i1 - i0) * rank, chans) @ spatial_sing_flat.T
        spatial_outer = spatial_outer.view(i1 - i0, rank, n_units, rank)
        spatial_outer = spatial_outer.permute(0, 2, 1, 3).reshape(chunksz, rank * rank)
        torch.mm(spatial_outer, tconv_flat, out=out[i0:i1].view(chunksz, up * conv_len))

    return out


@torch.jit.script
def best_shared_pconv(
    tconv: Tensor, spatial_sing: Tensor, batch_size: int = 1024
) -> tuple[Tensor, Tensor]:
    rank, rank_, conv_len = tconv.shape
    n_units, rank__, chans = spatial_sing.shape
    assert rank == rank_ == rank__
    lag_offset = conv_len // 2

    conv_out = spatial_sing.new_empty((n_units, n_units))
    lag_out = torch.empty_like(conv_out, dtype=torch.int32)

    itriu = torch.triu_indices(n_units, n_units, offset=1)
    ii = itriu[0]
    jj = itriu[1]
    ntriu = ii.shape[0]

    conv_flat = conv_out.new_empty((ntriu,))
    lag_flat = lag_out.new_empty((ntriu,))

    tconv_flat = tconv.view(rank * rank, conv_len)

    for i0 in range(0, ntriu, batch_size):
        i1 = min(ntriu, i0 + batch_size)
        bn = i1 - i0
        bii = ii[i0:i1]
        bjj = jj[i0:i1]

        sii = spatial_sing[bii]
        sjj = spatial_sing[bjj]

        # contract channels dim
        spatial_outer = torch.bmm(sii, sjj.mT)

        # contract rank with tconv
        bconv = spatial_outer.view(bn, rank * rank) @ tconv_flat

        # reduce. would use out but it's weird in script.
        conv_flat[i0:i1], lag_flat[i0:i1] = bconv.max(dim=1)

    # squareform
    conv_out[ii, jj] = conv_flat
    conv_out[jj, ii] = conv_flat
    lag_flat -= lag_offset
    lag_out[ii, jj] = lag_flat
    lag_out[jj, ii] = -lag_flat

    lag_out.diagonal().zero_()
    normsq = torch.linalg.norm(spatial_sing.view(n_units, -1), dim=1).square_()
    conv_out.diagonal().copy_(normsq)

    return conv_out, lag_out


@torch.jit.script
def weighted_best_lagged_scaled_normeuc_dist(
    tconv: Tensor,
    spatial_sing: Tensor,
    weights: Tensor,
    batch_size: int = 1024,
    scale_var: float = 0.01**2,
    scale_boundary: float = 1.0 / 3.0,
) -> tuple[Tensor, Tensor, Tensor]:
    rank, rank_, conv_len = tconv.shape
    n_units, rank__, chans = spatial_sing.shape
    assert rank == rank_ == rank__
    lag_offset = conv_len // 2

    d_out = spatial_sing.new_empty((n_units, n_units))
    lag_out = torch.empty_like(d_out, dtype=torch.int32)
    iou_out = torch.empty_like(d_out)

    itriu = torch.triu_indices(n_units, n_units, offset=1)
    ii = itriu[0]
    jj = itriu[1]
    ntriu = ii.shape[0]

    d_flat = d_out.new_empty((ntriu,))
    lag_flat = lag_out.new_empty((ntriu,))
    iou_flat = iou_out.new_empty((ntriu,))

    tconv_flat = tconv.view(rank * rank, conv_len)

    inv_lambda = torch.tensor(1.0 / scale_var).to(d_out)
    scale_max = 1.0 + scale_boundary
    _0 = torch.tensor(0.0).to(d_out)
    scmin = torch.tensor(1.0 / scale_max).to(d_out)
    scmax = torch.tensor(scale_max).to(d_out)

    for i0 in range(0, ntriu, batch_size):
        i1 = min(ntriu, i0 + batch_size)
        bn = i1 - i0
        bii = ii[i0:i1]
        bjj = jj[i0:i1]
        sii = spatial_sing[bii]
        sjj = spatial_sing[bjj]
        wii = weights[bii]
        wjj = weights[bjj]

        # soft intersection over union while computing "intersection weights"
        wmin = torch.minimum(wii, wjj)
        wminsum = wmin.sum(dim=1)
        wmin /= wminsum[:, None]
        wmaxsum = torch.maximum(wii, wjj).sum(dim=1)
        torch.divide(wminsum, wmaxsum, out=iou_flat[i0:i1])

        # weighted contraction of channels dim
        wmin = wmin[:, None, :]
        sii *= wmin
        sjj *= wmin
        spatial_outer = torch.bmm(sii, sjj.mT)

        # contract rank with tconv
        bconv = spatial_outer.view(bn, rank * rank) @ tconv_flat

        # take the best convolution and lag
        dots, lag_flat[i0:i1] = bconv.max(dim=1)

        # weighted norms
        normsqii = sii.square_().sum(dim=(1, 2))
        normsqjj = sjj.square_().sum(dim=(1, 2))

        # now, the scaling part of the distance, as in deconvolution
        # you can think of either ii or jj as the target/template;
        # here we try both and take the minimum.
        # scii is the scaling considering ii as template; similar for jj.
        b = dots + inv_lambda
        scii = (b / (normsqii + inv_lambda)).clamp_(scmin, scmax)
        scjj = (b / (normsqjj + inv_lambda)).clamp_(scmin, scmax)

        # ||target - sc*template||^2 = ||target||^2 + sc^2normsq - 2 sc dot
        scnormsqii = scii.square().mul_(normsqii)
        scnormsqjj = scjj.square().mul_(normsqjj)
        dii = (scnormsqii + normsqjj).sub_(scii * dots, alpha=2.0)
        djj = (scnormsqjj + normsqii).sub_(scjj * dots, alpha=2.0)

        # for normalizing scaled distance, divide by sqrt(targnormsq * sc^2*tempnormsq)
        dii /= normsqjj.mul_(scnormsqii).sqrt_()
        djj /= normsqii.mul_(scnormsqjj).sqrt_()

        # take minimums, enforce >= 0, square root
        dd = dii.clamp_(_0, djj)
        torch.sqrt(dd, out=d_flat[i0:i1])

    # squareform
    d_out[ii, jj] = d_flat
    d_out[jj, ii] = d_flat
    iou_out[ii, jj] = iou_flat
    iou_out[jj, ii] = iou_flat
    lag_flat -= lag_offset
    lag_out[ii, jj] = lag_flat
    lag_out[jj, ii] = -lag_flat

    lag_out.diagonal().zero_()
    d_out.diagonal().zero_()

    return d_out, lag_out, iou_out


def weighted_normeuc_distance(means, weights, batch_size=512, min_iou=0.75):
    assert means.ndim == 3
    assert weights.ndim == 2
    k = means.shape[0]
    assert weights.shape == (k, means.shape[2])

    ii, jj = torch.triu_indices(k, k, offset=1)
    npair = ii.shape[0]

    pdist = means.new_full((npair,), torch.inf)

    weights = weights / weights.amax(dim=1, keepdims=True)

    for i0 in trange(0, npair, batch_size, desc="WeightedNormEuc"):
        i1 = min(npair, i0 + batch_size)

        iii = ii[i0:i1]
        jjj = jj[i0:i1]

        wi = weights[iii]
        wj = weights[jjj]
        wmin = torch.minimum(wi, wj)
        wminsum = wmin.sum(1)
        piou = wminsum / torch.maximum(wi, wj).sum(1)
        (valid,) = (piou >= min_iou).nonzero(as_tuple=True)

        xi = means[iii[valid]]
        xj = means[jjj[valid]]

        w = wmin[valid, None, :] / wminsum[valid, None, None]
        assert w.shape == (xi.shape[0], 1, xi.shape[2])
        dist = (xi - xj).square_().mul_(w).mean(dim=(1, 2))
        nmi = (xi.square_().mul_(w)).mean(dim=(1, 2)).sqrt_()
        nmj = (xj.square_().mul_(w)).mean(dim=(1, 2)).sqrt_()

        dist = dist.div_(nmi).div_(nmj)
        pdist[i0 + valid] = dist
    pdist = pdist.numpy(force=True)
    return squareform(pdist)


def scaled_normeuc_from_dots(
    dots: Tensor,
    scale_var: float = 0.01**2,
    scale_boundary: float = 1.0 / 3.0,
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> Tensor:
    inv_lambda = 1.0 / scale_var
    if scale_min is None:
        assert scale_max is None
        scale_max = 1.0 + scale_boundary
        scale_min = 1.0 / scale_max
    else:
        assert scale_max is not None

    normsq = dots.diagonal().contiguous()

    # optimal penalized scaling
    # columns are targets ("recording"), rows are the "templates"
    if scale_var > 0:
        b = dots + inv_lambda
        a = normsq[:, None] + inv_lambda
        sc = b.div_(a).clamp_(scale_min, scale_max)
    else:
        sc = torch.ones_like(dots)

    # euclidean distance identity
    col_norm = normsq.sqrt()
    row_norm = sc * col_norm[:, None]
    dist = row_norm.square() + normsq[None, :]
    dist -= sc.mul_(dots).mul_(2.0)
    del sc
    dist.relu_().sqrt_()

    # divide by geom mean of norms
    dist /= row_norm.sqrt_()
    dist /= col_norm[None, :].sqrt_()

    # handle blanks
    dist.masked_fill_(dots == 0.0, torch.inf)
    dist.diagonal().zero_()

    # symmetrize
    dist = torch.minimum(dist, dist.T)

    return dist


@torch.jit.script
def scaled_normeuc_distance(
    means: Tensor,
    scale_std: float = 0.01,
    scale_min: float = 2.0 / 3.0,
    scale_max: float = 4.0 / 3.0,
    batch_size: int = 8192,
):
    means = means.reshape(means.shape[0], -1)
    K = means.shape[0]

    # euc dist foil identity helpful for scaled dist
    norm = torch.linalg.norm(means, dim=1)
    normsq = norm.square()

    # dot upper tri
    dots = means.new_full((K, K), torch.nan)
    ix = torch.triu_indices(K, K)
    ii = ix[0]
    jj = ix[1]
    nix = ii.shape[0]
    for i0 in range(0, nix, batch_size):
        i1 = min(nix, i0 + batch_size)
        bii = ii[i0:i1]
        bjj = jj[i0:i1]
        dots[bii, bjj] = means[bii, None, :].bmm(means[bjj, :, None])[:, 0, 0]
    del means

    # fill in dot lower tri, diagonal
    dots[jj, ii] = dots[ii, jj]
    dots.diagonal().copy_(normsq)

    return scaled_normeuc_from_dots(
        dots, scale_var=scale_std**2, scale_min=scale_min, scale_max=scale_max
    )


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

    multi_index = torch.broadcast_tensors(*multi_index)
    strides = torch.tensor([1, *reversed(dims[1:])]).cumprod(0).flip(0)
    out = torch.zeros_like(multi_index[0])
    for j, s in enumerate(strides):
        out += s * multi_index[j]
    return out.view(-1)

    # collect multi indices
    # multi_index = torch.broadcast_tensors(*multi_index)
    # multi_index = torch.stack(multi_index)
    # # stride along each axis
    # strides = multi_index.new_tensor([1, *reversed(dims[1:])]).cumprod(0).flip(0)
    # # apply strides (along first axis) and reshape
    # strides = strides.view(-1, *([1] * (multi_index.ndim - 1)))
    # raveled_indices = (strides * multi_index).sum(0)
    # return raveled_indices.view(-1)


def torch_add_at_(dest, ix, src, sign=1):
    """Pytorch version of np.{add,subtract}.at

    Adds src into dest in place at indices (in dest) specified
    by tuple of index arrays ix. So, indices in ix should be
    locations in dest, but the arrays constituting ix should
    have shapes which broadcast to src.shape.

    Will add multiple times into the same indices. Check out
    docs for scatter_add_ and np.ufunc.at for more details.
    """
    flat_ix = ravel_multi_index(ix, dest.shape)
    if isinstance(src, (float, int)):
        src = torch.tensor(src * sign, dtype=dest.dtype, device=dest.device)
        src = src.broadcast_to(flat_ix.numel())
    else:
        src = src.reshape(-1)
        if sign == -1:
            src = src._neg_view()
        elif sign != 1:
            src = sign * src
    dest.view(-1).scatter_add_(0, flat_ix.to(dest.device), src)


def cupy_add_at_(dest, ix, src, sign=1):
    assert cp is not None
    if torch.is_tensor(dest):
        assert dest.device.type == "cuda"
    dest = cp.asarray(dest)
    if isinstance(ix, tuple):
        ix = tuple(cp.asarray(ii) for ii in ix)
    else:
        ix = cp.asarray(ix)
    if not isinstance(src, (float, int)):
        src = cp.asarray(src)
    if sign == 1:
        cp.add.at(dest, ix, src)
    elif sign == -1:
        cp.subtract.at(dest, ix, src)
    else:
        raise NotImplementedError(f"Need to implement {sign=} in cupy_add_at_.")


add_at_ = torch_add_at_


def try_cupy_add_at_(dest, ix, src, sign=1):
    if not HAVE_CUPY or dest.device.type != "cuda":
        if dest.device.type == "cuda":
            warnings.warn("No cupy.")
        return torch_add_at_(dest, ix, src, sign)
    else:
        return cupy_add_at_(dest, ix, src, sign)


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


@torch.jit.script
def argrelmax(
    *,
    x: Tensor,
    radius: int,
    threshold: float,
    arange: Tensor,
):
    x1, inds = F.max_pool1d_with_indices(
        x[None, None], kernel_size=(2 * radius + 1,), padding=(radius,), stride=(1,)
    )
    x1 = x1[0, 0]
    inds = inds[0, 0]
    # exclude non-maxima and exact duplicates
    mask = torch.logical_or(x < x1, inds != arange)
    x1.masked_fill_(mask, 0.0)
    F.threshold(x1, threshold, 0.0, inplace=True)
    x1.masked_fill_(mask, 0.0)
    # exclude edge
    x1[0].zero_()
    x1[-1].zero_()
    ix = torch.nonzero(x1)[:, 0]
    return ix


@torch.jit.script
def argrelmax_dedup(
    peak_radius: int = 1,
    *,
    x: Tensor,
    dedup_radius: int,
    threshold: float,
    arange: Tensor,
    padding: int,
):
    """Modification of scipy's argrelmax for template matching

    This finds peaks>threshold separated by radius, subject to some extra checks.
     - The peaks will not be in the boundary (`padding`), which in the
       matching context would mean they could not be subtracted (off the edge)
     - Exact duplicates will be removed (this is the inds and arange stuff)

    Also, as in scipy, peaks on the very edge (of x itself, or of the non-boundary
    region) are ignored, since they may not be local maxima. On the edge of the
    boundary region, they are included when they are local maxima, since we have
    padding there to check.

    There's also an issue which arises in the usual argrelmax: peaks in the boundary
    get compared to peaks in the valid region by the max pooling, so that valid peaks
    get ignored. And this cascades into the interior, so that you eventually see a
    U-shaped frontier of peaks > threshold which are falsely not included. The impl
    below handles this case by detecting local maxs without the boundary, excluding
    the boundary, then deduplicating.

    NB: exclude region is padding + 1, because later matching steps may adjust the
    peak time by +/- 1 (temporal upsampling stuff). Make your chunk margin 1 sample
    bigger if you care about that.
    """
    nt = x.shape[0]
    xv = x.clone()
    x = x[None, None]
    xv[: padding + 1].zero_()
    xv[nt - padding - 1 :].zero_()
    xv = xv[None, None]
    x1, inds1 = F.max_pool1d_with_indices(
        x, kernel_size=(2 * peak_radius + 1,), padding=(peak_radius,), stride=(1,)
    )
    remove1 = torch.logical_or(xv < x1, inds1 != arange)
    x1.masked_fill_(remove1, 0.0)
    F.threshold(x1, threshold, 0.0, inplace=True)
    x2, inds2 = F.max_pool1d_with_indices(
        x1, kernel_size=(2 * dedup_radius + 1,), padding=(dedup_radius,), stride=(1,)
    )
    remove2 = torch.logical_or(x1 < x2, inds2 != arange)
    x2.masked_fill_(remove2, 0.0)
    x2 = x2[0, 0]
    return x2.nonzero()[:, 0]


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
            (n_templates, out_len), dtype=traces.dtype, device=traces.device
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
            rec_spatial[None], temporal[:, None, :], groups=n_templates, padding=padding
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


def nancov(
    x,
    weights=None,
    correction=1,
    nan_free=False,
    return_nobs=False,
    force_posdef=False,
    eps=0.0,
):
    """Pairwise covariance estimate

    Covariances are estimated from masked observations in each feature.
    The result may not be positive definite, but we'll eigh it for you if
    you want.
    """
    if not nan_free:
        mask = x.isfinite().to(x)
        x = x.nan_to_num()

    if weights is not None:
        xtx = (x.T * weights) @ x
        if nan_free:
            nobs = weights.sum(0)
        else:
            nobs = (mask.T * weights) @ mask
    else:
        xtx = x.T @ x
        if nan_free:
            nobs = np.array(len(x), dtype=x.dtype)
        else:
            nobs = mask.T @ mask
    denom = nobs - correction
    denom[denom <= 0] = 1
    cov = xtx / denom

    if force_posdef:
        try:
            if eps:
                np.fill_diagonal(cov, np.diagonal(cov) + eps)
            vals, vecs = torch.linalg.eigh(cov)
            good = vals > 0
            cov = (vecs[:, good] * vals[good]) @ vecs[:, good].T
        except Exception as e:
            if not cov.isfinite().all():
                raise e
            else:
                warnings.warn(
                    f"Error in nancov's eigh, shown below, was ignored because the covariance remained finite. {e}"
                )

    if return_nobs:
        return cov, nobs
    return cov


def cosine_distance(means, means_b=None, true_distance=True):
    means = means.reshape(means.shape[0], -1)
    sym = means_b is None
    if sym:
        means_b = means
    else:
        assert means_b is not None
        means_b = means_b.reshape(means_b.shape[0], -1)
    dot = means @ means_b.T
    norm = means.square().sum(1).sqrt_()
    blank = norm == 0
    norm[blank] = 1
    if sym:
        blank_b = blank
        norm_b = norm
    else:
        norm_b = means_b.square().sum(1).sqrt_()
        blank_b = norm_b == 0
        norm_b[blank_b] = 1
    dot /= norm[:, None]
    dot /= norm_b[None, :]
    dist = torch.subtract(_1, dot, out=dot)
    if true_distance:
        dist.mul_(2.0).sqrt_()
    dist[blank] = torch.inf
    dist[:, blank_b] = torch.inf
    if sym:
        dist.diagonal().zero_()
    return dist


def weighted_normsup_distance(means, weights, batch_size=512, min_iou=0.75):
    assert means.ndim == 3
    assert weights.ndim == 2
    k, p = means.shape[:2]
    assert weights.shape == (k, means.shape[2])

    ii, jj = torch.triu_indices(k, k, offset=1)
    npair = ii.shape[0]

    pdist = means.new_full((npair,), torch.inf)
    piou = means.new_zeros(npair)

    weights = weights / weights.amax(dim=1, keepdims=True)

    for i0 in trange(0, npair, batch_size, desc="WeightedNormSup"):
        i1 = min(npair, i0 + batch_size)

        iii = ii[i0:i1]
        jjj = jj[i0:i1]

        wi = weights[iii]
        wj = weights[jjj]
        wmin = torch.minimum(wi, wj)
        wminsum = wmin.sum(1)
        piou[i0:i1] = wmin.sum(1) / torch.maximum(wi, wj).sum(1)
        (valid,) = (piou[i0:i1] >= min_iou).nonzero(as_tuple=True)

        xi = means[iii[valid]]
        xj = means[jjj[valid]]

        # w = wi[valid] * wj[valid]
        w = wmin[valid] / (p * wminsum[valid, None])
        dist = (xi - xj).abs_().mul_(w[:, None]).amax(dim=(1, 2))

        nmi = (xi.abs_().mul_(w[:, None])).amax(dim=(1, 2))
        nmj = (xj.abs_().mul_(w[:, None])).amax(dim=(1, 2))

        dist = dist.div_(torch.minimum(nmi, nmj))
        pdist[i0 + valid] = dist
    pdist = pdist.numpy(force=True)
    return squareform(pdist)


def maxz_distance(means, stderrs, weights, batch_size=512, min_iou=0.75):
    assert means.ndim == 3
    assert weights.ndim == 2
    k, p = means.shape[:2]
    assert weights.shape == (k, means.shape[2])

    ii, jj = torch.triu_indices(k, k, offset=1)
    npair = ii.shape[0]

    pdist = means.new_full((npair,), torch.inf)
    piou = means.new_zeros(npair)

    weights = weights / weights.amax(dim=1, keepdims=True)

    for i0 in trange(0, npair, batch_size, desc="MaxZ"):
        i1 = min(npair, i0 + batch_size)

        iii = ii[i0:i1]
        jjj = jj[i0:i1]

        wi = weights[iii]
        wj = weights[jjj]
        wmin = torch.minimum(wi, wj)
        piou[i0:i1] = wmin.sum(1) / torch.maximum(wi, wj).sum(1)
        (valid,) = (piou[i0:i1] >= min_iou).nonzero(as_tuple=True)

        xi = means[iii[valid]]
        xj = means[jjj[valid]]
        sei = stderrs[iii[valid]]
        sej = stderrs[jjj[valid]]

        # se = sei + sej  # deliberately multiplying by 2 to get a chi thing.
        se = 2.0 * torch.minimum(sei, sej)
        z_chan = (xi - xj).div_(se).square_().mean(dim=1).sqrt_()
        max_z = z_chan.amax(dim=1)
        pdist[i0 + valid] = max_z
    pdist = pdist.numpy(force=True)
    return squareform(pdist)


@torch.jit.script
def normeuc_distance(means: Tensor):
    """|a-b|/sqrt(|a||b|)"""
    means = means.reshape(means.shape[0], -1)
    norms_sqrt = means.square().sum(dim=1).sqrt_().sqrt_()
    blank = norms_sqrt == 0
    dist = torch.cdist(means, means)
    dist.div_(norms_sqrt[:, None])
    dist.div_(norms_sqrt[None, :])
    dist[blank] = torch.inf
    dist[:, blank] = torch.inf
    dist.diagonal().zero_()
    return dist


def woodbury_kl_divergence(C, mu, W=None, mus=None, Ws=None, out=None, batch_size=8):
    """KL divergence with the lemmas, up to affine constant with respect to mu and W

    Here's the logic between the lines below. Variable names follow this notation.

    If W is None, then
        2 DKL_0(other || self) = (mu_self - mu_other)' C^-1 (mu_self - mu_other)
    where d is the dimension.

    To do n triangular matmuls rather than n^2 full ones, compute this via:
        mu' = C^{-1/2} mu
        (mu_self - mu_other)' C^-1 (mu_self - mu_other) = |mu'_self - mu'_other|^2

    Otherwise,
        2 DKL(other || self) = tr(S_self^-1 S_other)
                               + log(|S_self| / |S_other|)
                               + (mu_self - mu_other)' S_self^-1 (mu_self - mu_other)
                               - d

    For the log dets,
        log|S| = log|C + WW'| = log|C| + log|I_M + W'C^-1W|
    Forget about log|C|, even though it's precomputed. For the other part, to save
    some matmuls, let
        U = C^{-1/2}W
        cap = I_M + W'C^{-1}W = I_m + U'U
    Note that log|C| will cancel.

    For the trace part, we have
        S_self^-1 S_other = (C + Ws Ws')^-1 (C + Wo Wo')
            = [C^-1 - C^-1 Ws caps^-1 Ws' C^-1] (C + Wo Wo')
            = C^-1 C
              + C^-1 Wo Wo'
              - C^-1 Ws caps^-1 Ws'
              - C^-1 Ws caps^-1 Ws' C^-1 Wo Wo'

    tr(C^-1 C) = d, so that cancels the constant.

    To save matmuls, introduce
        Vs = Us caps^{-1/2}

    Then by cyclic property, save more operations by rewriting
        (A+) tr(C^-1 Wo Wo') = tr(Uo' Uo)
    and
        (B-) tr(C^-1 Ws caps^-1 Ws') = tr(Us caps^-1 Us')
                                     = tr(Vs Vs')
    and
        (C-) tr(C^-1 Ws caps^-1 Ws' C^-1 Wo Wo')
                = tr(Wo' C^{-1/2} C^{-1/2} Ws caps^{-1/2} caps^{-1/2} Ws' C^{-1/2} C^{-1/2} Wo)
                = tr(Uo' Vs Vs' Uo)

    The Mahalanobis term remains. Woodbury says
        (mus - muo)' (C + Ws Ws')^-1 (mus - muo)
            = (mus - muo)' [C^-1 - C^-1 Ws cap^-1 Ws' C^-1] (mus - muo)
            = |mu's - mu'o|^2
              - (mu's - mu'o)' C^{-1/2} Ws cap^-1 Ws' C^{-1/2} (mu's - mu'o)
            = |mu's - mu'o|^2 - |caps^{-1/2} Ws' C^{-1/2} (mu's - mu'o)|^2
            = |mu's - mu'o|^2 - |Vs' (mu's - mu'o)|^2

    Returns
    -------
    out : Tensor
        out[i, j] = const(wrt mu,W) * (D(component i || component j) + const(wrt mu,W))
        i = other, j = self
    """
    n, d = mu.shape
    assert C.shape == (d, d)
    if out is None:
        out = mu.new_empty((n, n))

    Cchol = C.cholesky()  # .to_dense()
    # Some weird issues with solve_triangular using the batched input...
    # it seems to allocate something big?
    # Ccholinv = torch.linalg.solve_triangular(
    #     Cchol, torch.eye(len(Cchol), out=torch.empty_like(Cchol)), upper=False
    # )
    Ccholinv = Cchol.inverse().to_dense()
    # mu_ = torch.linalg.solve_triangular(Cchol, mu.unsqueeze(2), upper=False)
    mu_ = mu @ Ccholinv.T

    if W is None:
        # else, better to do this later
        for bs in range(0, n, batch_size):
            osl = slice(bs, min(bs + batch_size, n))
            out[osl] = (mu_[None] - mu_[osl, None]).square_().sum(dim=2)
        out *= 0.5
        return out

    M = W.shape[2]
    assert W.shape == (n, d, M)
    # U = torch.linalg.solve_triangular(Cchol, W, upper=False)
    U = torch.einsum("de,ndm->nem", Ccholinv, W)

    # first part of trace
    UTU = U.mT.bmm(U)
    # term A: it's an other-vec
    trA = UTU.diagonal(dim1=-2, dim2=-1).sum(dim=1)
    # assert torch.isclose(trA, U.square().sum(dim=(1, 2))).all()

    # make capacitance matrix and get its Cholesky
    cap = UTU
    del UTU
    cap.diagonal(dim1=-2, dim2=-1).add_(1.0)
    capchol = psd_safe_cholesky(cap)
    # V is (n, d, M)
    V = torch.linalg.solve_triangular(capchol.mT, U, upper=True, left=False)

    # log dets via Cholesky
    logdet = capchol.diagonal(dim1=-2, dim2=-1).log().sum(dim=1).mul_(2.0)
    # assert torch.isclose(logdet, cap.logdet()).all()

    # trace term C: it's a matrix of Frob inner products
    tmp = None
    for bs in range(0, n, batch_size):
        osl = slice(bs, min(bs + batch_size, n))
        if tmp is not None:
            tmp = tmp[: osl.stop - bs]
        tmp = torch.matmul(V.mT[None], U[osl, None], out=tmp)
        out[osl] = -tmp.square_().sum(dim=(2, 3))

    # mahalanobis term
    dmu = None
    for bs in range(0, n, batch_size):
        osl = slice(bs, min(bs + batch_size, n))
        if dmu is not None:
            dmu = dmu[: osl.stop - bs]
        dmu = torch.sub(mu_[None], mu_[osl, None], out=dmu)
        corr = torch.einsum("osd,sdm->osm", dmu, V)
        corr = corr.square_().sum(dim=2)
        out[osl] -= corr
        mah = dmu.square_().sum(dim=2)
        out[osl] += mah

    # trace term B: it's a self-vec
    trB = V.square_().sum(dim=(1, 2))

    # put back together
    # remember: self dim is dim 1, other dim is dim 0
    out += logdet[None, :]
    out -= logdet[:, None]
    out += trA[:, None]
    out -= trB[None, :]

    out *= 0.5

    return out


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


def isin_sorted(x, y):
    """Like torch.isin(x, y), but faster by assuming both sorted."""
    if not y.numel():
        return torch.zeros(x.shape, dtype=torch.bool, device=x.device)
    ix = torch.searchsorted(y, x, side="right") - 1
    return x == y[ix]


def average_by_label(x, labels, channels, n_channels, weights=None):
    """weights should sum to 1 already in each group."""
    n = x.shape[0]
    assert x.ndim == 3
    assert labels.shape == (n,)
    assert channels.shape[0] == n
    if weights is None:
        weights = x.new_ones(n)
    unique_labels = labels.unique()
    k = unique_labels.amax() + 1
    assert labels.shape == weights.shape

    out = x.new_zeros((k + 1, x.shape[1], n_channels + 1))
    counts = x.new_zeros((k + 1, n_channels + 1))
    f_arange = torch.arange(x.shape[1]).to(x.device)
    for u in unique_labels:
        (in_u,) = (labels == u).nonzero(as_tuple=True)
        wu = weights[in_u]
        cu = channels[in_u]
        wu = wu[:, None].broadcast_to(cu.shape).contiguous()
        counts[u].scatter_add_(dim=0, index=cu.view(-1), src=wu.view(-1))
        wu.div_(counts[u][cu])

        wxu = x[in_u].mul_(wu[:, None])
        torch_add_at_(
            out[u, None],
            (
                torch.zeros_like(in_u)[:, None, None],
                f_arange[None, :, None],
                cu[:, None],
            ),
            wxu,
        )

    return out[:k, :, :n_channels], counts[:k, :n_channels]


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
