"""Library for flavors of kernel interpolation and data interp utilities"""

from typing import Protocol, cast, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from ..cluster.gmm.stable_features import SpikeNeighborhoods
from .data_util import yield_masked_chunks
from .internal_config import (
    InterpKernel,
    InterpMethod,
    InterpolationParams,
    default_interpolation_params,
)
from .torch_util import BModule
from .waveform_util import make_channel_index

if TYPE_CHECKING:
    from .noise_util import EmbeddedNoise
else:
    EmbeddedNoise = None  # type: ignore


def interpolate_by_chunk(
    mask,
    dataset,
    geom,
    channel_index,
    channels,
    shifts,
    registered_geom,
    target_channels,
    params: InterpolationParams = default_interpolation_params,
    device=None,
    store_on_device=False,
    show_progress=True,
    shift_dim=1,
):
    """Interpolate data living in an HDF5 file

    If dataset is a h5py.Dataset and mask is a boolean array indicating
    positions of data to load, this iterates over the HDF5 chunks to
    quickly scan through the data, applying interpolation to all the
    features.

    Arguments
    ---------
    mask : boolean np.ndarray
        Load and interpolate these entries. Shape should be
        (n_spikes_full,), and let's say it has n_spikes nonzero entries.
    dataset : h5py.Dataset
        Chunked dataset, shape (n_spikes_full, feature_dim, n_source_channels)
        Can only be chunked on the first axis
    geom : array or tensor
    channel_index : int array or tensor
    channels : int array or tensor
        Shape (n_spikes,)
    shifts : array or tensor
        Shape (n_spikes,) or (n_spikes, n_source_channels)
    registered_geom : array or tensor
    target_channels : int array or tensor
        (n_spikes, n_target_channels)
    sigma : float
        Kernel bandwidth
    interpolation_method : str
    device : torch device
    store_on_device : bool
        Allocate the output tensor on gpu?
    show_progress : bool

    Returns
    -------
    out : torch.Tensor
        (n_spikes, feature_dim, n_target_chans)
    """
    # check shapes
    assert geom.shape[1] == 2, "Haven't implemented 3d."
    if torch.is_tensor(mask):
        mask = mask.numpy(force=True)
    (n_spikes_full,) = mask.shape
    assert mask.dtype.kind == "b"
    n_spikes = mask.sum()
    assert channels.shape == (n_spikes,)
    n_source_chans = channel_index.shape[1]
    assert n_source_chans == dataset.shape[2]
    assert n_spikes_full == dataset.shape[0]
    n_target_chans = target_channels.shape[1]
    assert target_channels.shape == (n_spikes, n_target_chans)
    assert shifts.shape == (n_spikes,)
    assert len(geom) == len(channel_index)
    feature_dim = dataset.shape[1]

    params = params.normalize()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dtype = torch.from_numpy(np.empty((), dtype=dataset.dtype)).dtype

    # allocate output
    storage_device = device if store_on_device else "cpu"
    out_shape = n_spikes, feature_dim, n_target_chans
    out = torch.empty(out_shape, dtype=dtype, device=storage_device)

    # build data needed for interpolation
    source_geom = pad_geom(geom, dtype=dtype, device=device)
    target_geom = pad_geom(registered_geom, dtype=dtype, device=device)
    # here, shifts = reg_depths - depths = -usual displacement
    # (reg_depths = depths - displacement)
    shifts = torch.as_tensor(shifts, dtype=dtype)
    target_channels = torch.as_tensor(target_channels)
    channel_index = torch.as_tensor(channel_index, device=device)
    channels = torch.as_tensor(channels)

    # make interpolator
    erp = StableFeaturesInterpolator(
        source_geom=source_geom,
        target_geom=target_geom,
        channel_index=channel_index,
        params=params,
    )

    for ixs, chunk_features in yield_masked_chunks(
        mask, dataset, show_progress=show_progress, desc_prefix="Interpolating"
    ):
        # interpolate, store
        chunk_features = torch.from_numpy(chunk_features).to(device)
        out[ixs] = erp.interp(
            features=chunk_features,
            source_main_channels=channels[ixs].to(device),
            target_channels=target_channels[ixs],
            source_shifts=shifts[ixs].to(device),
            allow_destroy=True,
        ).to(out)

    return out


def interp_precompute(
    source_geom=None,
    channel_index=None,
    source_pos=None,
    params: InterpolationParams = default_interpolation_params,
    source_geom_is_padded=True,
):
    params = params.normalize()
    if params.method in ("nearest", "kernel", "normalized", "zero"):
        return None
    assert params.method in ("kriging", "krigingnormalized")

    if source_pos is None:
        assert source_geom is not None
        source_geom = torch.asarray(source_geom)
        n_source_chans = len(source_geom) - source_geom_is_padded
        if channel_index is None:
            channel_index = torch.arange(n_source_chans)[None]
            channel_index = channel_index.to(source_geom.device)
        source_pos = source_geom[channel_index]
        valid = channel_index < n_source_chans
    else:
        assert source_geom is None
        assert channel_index is None
        source_pos = torch.asarray(source_pos)
        valid = source_pos[..., 0].isfinite()
    assert source_pos.ndim == 3
    valid = valid.to(source_pos.device)

    ns = len(source_pos)
    neighb_size = source_pos.shape[1]
    dim = source_pos.shape[2]
    if params.kriging_poly_degree < 0:
        design_vars = 0
    elif params.kriging_poly_degree == 0:
        design_vars = 1
    elif params.kriging_poly_degree == 1:
        design_vars = 1 + dim
    elif params.kriging_poly_degree == 2:
        design_vars = 1 + dim + (dim * (dim + 1)) // 2
    else:
        assert False

    solvers = source_pos.new_zeros(
        (ns, neighb_size + design_vars, neighb_size + design_vars)
    )
    design_inds = torch.arange(neighb_size, neighb_size + design_vars)
    design_inds = design_inds.to(source_pos.device)
    design_zeros = source_pos.new_zeros((design_vars, design_vars))

    source_kernels = get_kernel(
        source_pos,
        kernel_name=params.kernel.removesuffix("normalized"),
        sigma=params.sigma,
        rq_alpha=params.rq_alpha,
        normalized=params.method.endswith("normalized"),
        smoothing_lambda=params.smoothing_lambda,
    )
    for j in range(ns):
        (present,) = valid[j].nonzero(as_tuple=True)
        kernel = source_kernels[j][present][:, present]

        if design_vars:
            pos = source_pos[j][present] / params.sigma
            const = pos.new_ones((pos.shape[0], 1))
            ix = torch.concatenate((present, design_inds), dim=0)
            if params.kriging_poly_degree == 0:
                design = const
            elif params.kriging_poly_degree == 1:
                design = torch.cat([pos, const], dim=1)
            elif params.kriging_poly_degree == 2:
                if dim == 1:
                    design = torch.cat([pos, pos.square(), const], dim=1)
                elif dim == 2:
                    p01 = (pos[:, 0] * pos[:, 1]).unsqueeze(1)
                    design = torch.cat([pos, pos.square(), p01, const], dim=1)
                elif dim == 3:
                    p01 = (pos[:, 0] * pos[:, 1]).unsqueeze(1)
                    p02 = (pos[:, 0] * pos[:, 2]).unsqueeze(1)
                    p12 = (pos[:, 1] * pos[:, 2]).unsqueeze(1)
                    design = torch.cat([pos, pos.square(), p01, p02, p12, const], dim=1)
                else:
                    assert False
            else:
                assert False
            assert design.shape[1] == design_vars

            top = torch.cat([kernel, design], dim=1)
            bot = torch.cat([design.T, design_zeros], dim=1)
            to_solve = torch.cat([top, bot], dim=0)
        else:
            to_solve = kernel
            ix = present

        # double+pinv is helpful here. numerics can get weird with large domains.
        solver = torch.linalg.pinv(to_solve.double(), hermitian=True)
        solvers[j, ix[:, None], ix[None, :]] = solver.to(solvers.dtype)

    return solvers


def full_probe_precompute(
    source_geom: torch.Tensor, channel_index: torch.Tensor, params: InterpolationParams
):
    """When kriging on the full probe, numerical problems arise due to solving huge systems

    This breaks up the problem into neighborhoods, so that each output depends only on
    a neighborhood of inputs. Same trick that numpy doees in RBFInterpolator if you ask
    it nicely.

    Usually, precomputed data shape is (nneighbs, nc_neighb + design, nc_neighb + design).
    This one makes a shape (1, nc, nc + design, nc + design), and it hits a special case
    in kriging_solve triggered by the shape. Sorry!

    TODO can this be simplified?
    """
    precomputed_data = interp_precompute(
        source_geom=pad_geom(source_geom),
        channel_index=channel_index,
        params=params,
        source_geom_is_padded=True,
    )
    if precomputed_data is None:
        return None

    nc, nc_pc, nc_pc_ = precomputed_data.shape
    assert nc_pc == nc_pc_
    assert nc_pc >= channel_index.shape[1]
    extra_dim = nc_pc - channel_index.shape[1]
    assert extra_dim >= 0
    # embed into full probe...
    pc_full = precomputed_data.new_zeros((nc, nc + extra_dim, nc + extra_dim))
    for j in range(nc):
        chans = channel_index[j]
        (valid,) = (chans < nc).nonzero(as_tuple=True)
        cvalid = chans[valid]
        pc_full[j, cvalid[:, None], cvalid[None, :]] = precomputed_data[
            j, valid[:, None], valid[None, :]
        ]
        if extra_dim > 0:
            pc_full[j, -extra_dim:, -extra_dim:] = precomputed_data[
                j, -extra_dim:, -extra_dim:
            ]
            pc_full[j, -extra_dim:, cvalid[None, :]] = precomputed_data[
                j, -extra_dim:, valid[None, :]
            ]
            pc_full[j, cvalid[:, None], -extra_dim:] = precomputed_data[
                j, valid[:, None], -extra_dim:
            ]
    precomputed_data = pc_full[None]
    return precomputed_data


def kernel_interpolate(
    features,
    source_pos,
    target_pos,
    params: InterpolationParams = default_interpolation_params,
    precomputed_data=None,
    allow_destroy=False,
    out=None,
):
    """Kernel interpolation of multi-channel features or waveforms

    Arguments
    ---------
    features : torch.Tensor
        n_spikes, feature_dim, n_source_channels
        These can be masked, indicated by nans here and in the same
        places of source_pos
    source_pos : torch.Tensor
        n_spikes, n_source_channels, spatial_dim
    target_pos : torch.Tensor
        n_spikes, n_target_channels, spatial_dim
        These can also be masked, indicate with nans and you will
        get nans in those positions
    sigma : float
        Spatial bandwidth of RBF kernels
    allow_destroy : bool
        We need to overwrite nans in the features with 0s. If you
        allow me, I'll do that in-place.
    out : torch.Tensor
        Storage for target

    Returns
    -------
    features : torch.Tensor
        n_spikes, feature_dim, n_target_channels
    """
    features = torch.asarray(features)
    source_pos = torch.asarray(source_pos)
    target_pos = torch.asarray(target_pos)

    params = params.normalize()
    extrap_diff = params.extrap_diff()

    if extrap_diff:
        if params.actual_extrap_method == "kriging":
            # haven't supported different precomputed kriging data
            assert params.kernel == params.actual_extrap_kernel
        features_out = _kernel_interpolate(
            features=features,
            source_pos=source_pos,
            target_pos=target_pos,
            method=params.actual_extrap_method,
            kernel_name=params.actual_extrap_kernel,
            sigma=params.sigma,
            rq_alpha=params.rq_alpha,
            kriging_poly_degree=params.kriging_poly_degree,
            smoothing_lambda=params.smoothing_lambda,
            precomputed_data=precomputed_data,
        )
    else:
        features_out = None

    if precomputed_data is None:
        # if at all possible, don't hit this branch. it's just here for vis.
        precomputed_data = interp_precompute(source_pos=source_pos, params=params)

    features = _kernel_interpolate(
        features=features,
        source_pos=source_pos,
        target_pos=target_pos,
        method=params.method,
        kernel_name=params.kernel,
        sigma=params.sigma,
        rq_alpha=params.rq_alpha,
        kriging_poly_degree=params.kriging_poly_degree,
        smoothing_lambda=params.smoothing_lambda,
        precomputed_data=precomputed_data,
        allow_destroy=allow_destroy,
        out=out,
    )

    if extrap_diff:
        # control over extrapolation with another method...
        assert features_out is not None
        targ_extrap = extrap_mask(source_pos, target_pos)[:, None]
        features = torch.where(targ_extrap, features_out, features, out=features)

    return features


def _kernel_interpolate(
    *,
    features: torch.Tensor,
    source_pos: torch.Tensor,
    target_pos: torch.Tensor,
    method: InterpMethod,
    kernel_name: InterpKernel,
    sigma: float,
    rq_alpha: float,
    kriging_poly_degree: int,
    smoothing_lambda: float,
    precomputed_data,
    allow_destroy=False,
    out=None,
):
    kernel = get_kernel(
        source_pos=source_pos,
        target_pos=target_pos,
        kernel_name=kernel_name.removesuffix("normalized"),
        sigma=sigma,
        rq_alpha=rq_alpha,
        normalized=method.endswith("normalized"),
        smoothing_lambda=smoothing_lambda,
    )

    features = torch.nan_to_num(features, out=features if allow_destroy else None)
    if method == "kriging":
        assert precomputed_data is not None
        precomputed_data = precomputed_data.to(features)
        features = kriging_solve(
            target_pos,
            kernel,
            features,
            solvers=precomputed_data,
            sigma=sigma,
            poly_degree=kriging_poly_degree,
        )
    else:
        features = torch.bmm(features, kernel, out=out)

    # nan-ify nonexistent chans
    needs_nan = torch.isnan(target_pos).all(2).unsqueeze(1)
    needs_nan = needs_nan.broadcast_to(features.shape)
    features.masked_fill_(needs_nan, torch.nan)

    return features


def get_kernel(
    source_pos,
    target_pos=None,
    kernel_name="rbf",
    sigma=20.0,
    rq_alpha=1.0,
    normalized=False,
    smoothing_lambda=0.0,
):
    assert source_pos.ndim == 3
    assert source_pos.shape[2] in (1, 2, 3)

    if kernel_name == "zero":
        tc = source_pos.shape[1] if target_pos is None else target_pos.shape[1]
        return source_pos.new_zeros(*source_pos.shape[:2], tc)
    elif kernel_name == "nearest":
        kernel = nearest_kernel(source_pos, target_pos)
    elif kernel_name == "idw":
        kernel = idw_kernel(source_pos, target_pos)
    elif kernel_name == "rbf":
        kernel = log_rbf(source_pos=source_pos, target_pos=target_pos, sigma=sigma)
        if normalized:
            kernel = F.softmax(kernel, dim=1)
        else:
            kernel = kernel.exp_()
    elif kernel_name == "multiquadric":
        kernel = multiquadric_kernel(
            source_pos=source_pos, target_pos=target_pos, sigma=sigma
        )
    elif kernel_name == "rq":
        kernel = rq_kernel(
            source_pos=source_pos, target_pos=target_pos, sigma=sigma, alpha=rq_alpha
        )
    elif kernel_name == "thinplate":
        kernel = thin_plate_greens(
            source_pos=source_pos, target_pos=target_pos, sigma=sigma
        )
    else:
        assert False

    kernel = kernel.nan_to_num_()
    if normalized and kernel_name not in ("rbf", "nearest"):
        kernel = kernel.div_(kernel.sum(dim=1, keepdim=True))

    if smoothing_lambda:
        kernel.diagonal(dim1=-2, dim2=-1).add_(smoothing_lambda)

    return kernel


def kriging_solve(target_pos, kernels, features, solvers, sigma=1.0, poly_degree=-1):
    n, rank = features.shape[:2]
    n_, n_targ, dim = target_pos.shape
    assert n == n_

    if poly_degree == -1:
        y = features
        pass
    elif poly_degree == 0:
        zero = features.new_zeros((n, rank, 1))
        y = torch.concatenate([features, zero], dim=2)
        const = features.new_ones((n, 1, n_targ))
        kernels = torch.concatenate([kernels, const], dim=1)
    elif poly_degree == 1:
        zero = features.new_zeros((n, rank, 1 + dim))
        y = torch.concatenate([features, zero], dim=2)
        const = features.new_ones((n, 1, n_targ))
        xy = (target_pos / sigma).nan_to_num_().mT
        kernels = torch.concatenate([kernels, xy, const], dim=1)
    elif poly_degree == 2:
        ddim = 1 + dim + (dim * (dim + 1)) // 2
        zero = features.new_zeros((n, rank, ddim))
        y = torch.concatenate([features, zero], dim=2)
        const = features.new_ones((n, 1, n_targ))
        xy = (target_pos / sigma).nan_to_num_().mT
        if dim == 1:
            xysq = (xy.square(),)
        elif dim == 2:
            xysq = (xy.square(), xy[:, 0:1] * xy[:, 1:2])
        elif dim == 3:
            xysq = (
                xy.square(),
                xy[:, 0:1] * xy[:, 1:2],
                xy[:, 0:1] * xy[:, 2:3],
                xy[:, 1:2] * xy[:, 2:3],
            )
        else:
            assert False
        kernels = torch.concatenate([kernels, xy, *xysq, const], dim=1)
    else:
        assert False

    if solvers.ndim == 3:
        return y.bmm(solvers).bmm(kernels)

    assert solvers.ndim == 4
    assert solvers.shape[0] == 1
    # per-channel case (each output chan has its own local neighb)
    # this is meant to mimic the "neighbors" option to scipy's RBFInterpolator,
    # but the implementation here is obscure. the rest of the logic is only shown
    # once, and that's in full_probe_precompute.
    # assumes that output channels and input channels are the same, and that all
    # inputs share the same neighborhood-solver per channel.
    solvers = solvers[0]
    c = kernels.shape[2]
    assert c == solvers.shape[0]
    c_ = kernels.shape[1]
    assert c_ >= c
    assert c_ == solvers.shape[1] == solvers.shape[2]
    out = y.new_zeros((*y.shape[:2], c))
    for cc in range(c):
        # just reducing memory use here relative to the einsum below
        out[:, :, cc] = torch.einsum("ntp,pq,nq->nt", y, solvers[cc], kernels[:, :, cc])
    # return torch.einsum("ntp,cpq,nqc->ntc", y, solvers, kernels)
    return out


def bake_interpolation_1d(
    xx: torch.Tensor, xx_: torch.Tensor, params: InterpolationParams
) -> tuple[torch.Tensor, int]:
    params = params.normalize()

    k = get_kernel(
        source_pos=xx[None, :, None],
        target_pos=xx_[None, :, None],
        kernel_name=params.kernel,
        normalized=params.method.endswith("normalized"),
        sigma=params.sigma,
        rq_alpha=params.rq_alpha,
        smoothing_lambda=params.smoothing_lambda,
    )
    nsrc = xx.shape[0]
    assert k.shape == (1, nsrc, xx_.shape[0])
    k = k[0]
    # add design matrix terms on source dim, ie first dim...
    need_design = params.method.startswith("kriging")
    if need_design and params.kriging_poly_degree == 0:
        k = torch.concatenate([k, torch.ones_like(xx_[None])], dim=0)
    elif need_design and params.kriging_poly_degree == 1:
        k = torch.concatenate([k, xx_[None], torch.ones_like(xx_[None])], dim=0)
    elif need_design:
        assert False

    pdata = interp_precompute(source_pos=xx[None, :, None], params=params)
    if pdata is not None:
        assert pdata.shape == (1, k.shape[0], k.shape[0])
        k = pdata[0] @ k
    zpad = int(need_design) * (1 + params.kriging_poly_degree)
    return k, zpad


def pad_geom(geom, dtype=torch.float, device=None):
    geom = torch.as_tensor(geom, dtype=dtype, device=device)
    geom = F.pad(geom, (0, 0, 0, 1), value=torch.nan)
    return geom


def idw_kernel(source_pos, target_pos=None):
    d = get_rsq(source_pos, target_pos, nan=None)
    kernel = d.sqrt_().reciprocal_()
    kernel = kernel.nan_to_num_()
    kernel /= kernel.sum(dim=0)
    return kernel


def nearest_kernel(source_pos, target_pos=None):
    d = get_rsq(source_pos, target_pos, nan=torch.inf)
    kernel = torch.zeros_like(d)
    kernel.scatter_(1, d.argmin(dim=1, keepdim=True), 1)
    return kernel


def log_rbf(source_pos, target_pos=None, sigma=None):
    """Log of RBF kernel

    This handles missing values in source_pos or target_pos, indicated by
    nans, by replacing them with -inf so that they exp to 0.

    Arguments
    ---------
    source_pos : torch.tensor
        n source locations
    target_pos : torch.tensor
        m target locations
    sigma : float

    Returns
    -------
    kernel : torch.tensor
        n by m
    """
    kernel = get_rsq(source_pos, target_pos, sigma, nan=None).div_(-2.0)
    kernel = kernel.nan_to_num_(nan=-torch.inf)
    return kernel


def rq_kernel(source_pos, target_pos=None, sigma=None, alpha=1.0):
    kernel = get_rsq(source_pos, target_pos, sigma)
    kernel = kernel.add_(1.0).pow_(-alpha)
    return kernel


def multiquadric_kernel(source_pos, target_pos=None, sigma=None):
    kernel = get_rsq(source_pos, target_pos, sigma)
    kernel = kernel.add_(1.0).pow_(-0.5).neg_()
    return kernel


def thin_plate_greens(source_pos, target_pos=None, sigma=1.0):
    rsq = get_rsq(source_pos, target_pos, sigma)
    r = rsq.sqrt()
    is_small = r < 1
    small = r.mul(r.pow(r).log_())
    big = rsq.mul_(r.log_())
    kernel = torch.where(is_small, small, big)
    return kernel


def get_rsq(
    source_pos,
    target_pos=None,
    sigma: float | None = 1.0,
    batch_size=16,
    nan: float | None = 0.0,
):
    have_sigma = sigma is not None
    if have_sigma and isinstance(sigma, (float, int)):
        have_sigma = sigma != 1.0
    if have_sigma:
        source_pos = source_pos / sigma
    if target_pos is None:
        target_pos = source_pos
    elif have_sigma:
        target_pos = target_pos / sigma
    assert source_pos.ndim == target_pos.ndim == 3
    assert source_pos.shape[0] == target_pos.shape[0]
    assert source_pos.shape[2] == target_pos.shape[2]

    n, nsrc = source_pos.shape[:2]
    rsq = source_pos.new_zeros((n, nsrc, target_pos.shape[1]))
    for bs in range(0, nsrc, batch_size):
        sl = slice(bs, min(bs + batch_size, nsrc))
        tmp = source_pos[:, sl, None] - target_pos[:, None, :]
        tmp = tmp.square_().sum(dim=3)
        if nan is not None:
            tmp = tmp.nan_to_num_(nan=nan)
        rsq[:, sl] = tmp

    return rsq


def extrap_mask(source_pos, target_pos, eps=1e-3):
    """Only works for vertical shift."""
    source_x_uniq = source_pos[..., 0].unique()
    source_x_uniq = source_x_uniq[source_x_uniq.isfinite()]

    targ_x, targ_y = target_pos[..., 0], target_pos[..., 1]

    # algorithm: start with all finite. on each iter, remove some from the set.
    targ_extrap = targ_y.isfinite()

    for x in source_x_uniq:
        source_in_col = torch.isclose(source_pos[..., 0], x)
        targ_not_in_col = torch.isclose(targ_x, x).logical_not_()

        source_col_y = torch.where(source_in_col, source_pos[..., 1], torch.nan)
        source_low = source_col_y.nan_to_num(nan=torch.inf).amin(dim=1).sub_(eps)
        source_high = source_col_y.nan_to_num(nan=-torch.inf).amax(dim=1).add_(eps)

        targ_col_y = targ_y.masked_fill(targ_not_in_col, torch.nan)
        # this is true for nans, but thats okay.
        targ_outside = targ_col_y != targ_col_y.clamp(
            min=source_low[:, None], max=source_high[:, None]
        )
        targ_extrap.logical_and_(targ_outside)

    return targ_extrap


class StableFeaturesInterpolator(BModule):
    def __init__(
        self,
        *,
        source_geom: torch.Tensor,
        target_geom: torch.Tensor,
        channel_index: torch.Tensor,
        params: InterpolationParams,
        shift_dim: int = 1,
        dtype=torch.float,
    ):
        super().__init__()
        self.params = params.normalize()
        self.shift_dim = shift_dim
        assert source_geom.shape[0] == 1 + channel_index.shape[0]
        assert source_geom.ndim == target_geom.ndim == 2
        assert source_geom.shape[1] == target_geom.shape[1] == 2
        self.register_buffer("source_geom", source_geom.to(dtype=dtype))
        self.register_buffer("target_geom", target_geom.to(dtype=dtype))
        self.register_buffer("channel_index", channel_index)
        neighb_data = interp_precompute(
            source_geom=self.b.source_geom,
            channel_index=channel_index,
            params=self.params,
        )
        self.has_neighb_data = neighb_data is not None
        self.register_buffer_or_none("neighb_data", neighb_data)

    def interp(
        self,
        features: torch.Tensor,
        source_main_channels: torch.Tensor,
        target_channels: torch.Tensor,
        source_shifts: torch.Tensor,
        allow_destroy: bool = False,
    ) -> torch.Tensor:
        assert target_channels.ndim == 2
        assert target_channels.shape[0] == source_main_channels.shape[0]

        # allows per-channel shifts with 2d input
        if source_shifts.ndim == 1:
            source_shifts = source_shifts.unsqueeze(1)
        if self.shift_dim == 0:
            source_shifts = torch.stack(
                [source_shifts, torch.zeros_like(source_shifts)], dim=-1
            )
        elif self.shift_dim == 1:
            source_shifts = torch.stack(
                [torch.zeros_like(source_shifts), source_shifts], dim=-1
            )
        else:
            assert False

        # used to shift the source, but for kriging it's better to shift targets
        # so that we can cache source kernel choleskys (i.e., my neighb_data)
        source_channels = self.b.channel_index[source_main_channels]
        source_pos = self.b.source_geom[source_channels]  # + source_shifts

        # where are they going?
        target_pos = self.b.target_geom[target_channels] - source_shifts

        if self.has_neighb_data:
            pcomp = self.b.neighb_data[source_main_channels]
        else:
            pcomp = None

        return kernel_interpolate(
            features=features,
            source_pos=source_pos,
            target_pos=target_pos,
            params=self.params,
            precomputed_data=pcomp,
            allow_destroy=allow_destroy,
        )


class NeighborhoodFiller(Protocol):
    def interp_to_chans(
        self,
        waveforms: torch.Tensor,
        neighborhood_ids: torch.Tensor,
        target_channels: torch.Tensor | slice,
    ) -> torch.Tensor: ...


class NeighborhoodInterpolator(BModule):
    def __init__(
        self,
        prgeom: torch.Tensor,
        neighborhoods: SpikeNeighborhoods,
        params: InterpolationParams = default_interpolation_params,
    ):
        super().__init__()
        assert len(prgeom) == neighborhoods.n_channels + 1
        self.params: InterpolationParams = params.normalize()
        self.register_buffer("prgeom", prgeom.clone())
        self.b.prgeom[-1].fill_(torch.nan)
        neighb_data = interp_precompute(
            source_geom=self.prgeom,
            channel_index=neighborhoods.neighborhoods,
            source_geom_is_padded=True,
            params=self.params,
        )
        self.register_buffer_or_none("neighb_data", neighb_data)
        self.register_buffer("neighb_pos", self.b.prgeom[neighborhoods.b.neighborhoods])

    def interp_to_chans(
        self,
        waveforms: torch.Tensor,
        neighborhood_ids: torch.Tensor,
        target_channels: torch.Tensor | slice,
    ):
        if target_channels is None:
            targ_pos = self.b.prgeom
        else:
            targ_pos = self.b.prgeom[target_channels]
        targ_pos = targ_pos[None].broadcast_to((len(waveforms), *targ_pos.shape))
        source_pos = self.b.neighb_pos[neighborhood_ids]
        neighb_data = self.b.neighb_data
        if neighb_data is not None:
            neighb_data = neighb_data[neighborhood_ids]
        return kernel_interpolate(
            features=waveforms,
            source_pos=source_pos,
            target_pos=targ_pos,
            precomputed_data=neighb_data,
            params=self.params,
        )


class NeighborhoodImputer(BModule):
    def __init__(self, noise: EmbeddedNoise, neighborhoods: SpikeNeighborhoods):
        super().__init__()
        prec = all_neighb_precisions(noise, neighborhoods)
        self.register_buffer("prec", prec)
        chans_arange = torch.arange(noise.n_channels, device=prec.device)
        self.register_buffer("chans_arange", chans_arange)
        self.neighborhoods = neighborhoods
        self.noise = noise

    def interp_to_chans(
        self,
        waveforms: torch.Tensor,
        neighborhood_ids: torch.Tensor,
        target_channels: torch.Tensor | slice,
    ) -> torch.Tensor:
        if not torch.is_tensor(target_channels):
            target_channels = self.b.chans_arange[target_channels]
        source_channels = self.neighborhoods.b.neighborhoods[neighborhood_ids]
        kernel = self.noise.cov_batch(source_channels, target_channels).to_dense()
        waveforms = waveforms.view(waveforms.shape[0], 1, -1)
        waveforms = waveforms.bmm(self.b.prec[neighborhood_ids])
        # waveforms = waveforms.view(waveforms.shape[0], -1)
        waveforms = torch.bmm(waveforms, kernel)
        return waveforms.view(waveforms.shape[0], self.noise.rank, -1)


class FullProbeInterpolator(BModule):
    """Interpolate from the registered geom to appropriate drifting geom channels."""

    def __init__(
        self,
        *,
        geom: torch.Tensor,
        rgeom: torch.Tensor,
        neighborhood_radius: float,
        motion_est,
        params: InterpolationParams,
    ):
        super().__init__()

        params = params.normalize()
        rchannel_index = make_channel_index(
            rgeom, radius=neighborhood_radius, to_torch=True
        )
        self.motion_est = motion_est
        rchannel_index = rchannel_index.to(device=rgeom.device)
        self.register_buffer_or_none(
            "data", full_probe_precompute(rgeom, rchannel_index, params)
        )
        self.g_depths = geom[:, 1].numpy(force=True)
        self.register_buffer("geom", geom)
        self.register_buffer("rgeom", rgeom)
        self.c_src: int = rgeom.shape[0]
        self.c_targ: int = geom.shape[0]
        self.dim: int = rgeom.shape[1]

    def interp_at_time(self, t_s: float, waveforms: torch.Tensor) -> torch.Tensor:
        assert waveforms.shape[2] == self.c_src
        # move geom to its position at time t_s
        shift = torch.zeros_like(self.b.geom)
        if self.motion_est is not None:
            disp = self.motion_est.disp_at_s(
                t_s=np.array([t_s]), depth_um=self.g_depths, grid=True
            )
            assert disp.shape[1] == 1
            shift[:, 1].copy_(torch.from_numpy(disp[:, 0]), non_blocking=True)

        # interpolate from static geom to shifted geom
        n = waveforms.shape[0]
        return kernel_interpolate(
            features=waveforms,
            source_pos=self.b.rgeom[None].broadcast_to(n, self.c_src, self.dim),
            target_pos=(self.b.geom + shift).broadcast_to(n, self.c_targ, self.dim),
            precomputed_data=self.b.data,
        )


def all_neighb_precisions(noise: EmbeddedNoise, neighborhoods: SpikeNeighborhoods):
    channel_index = neighborhoods.b.neighborhoods
    nneighb = channel_index.shape[0]
    nc_obs = channel_index.shape[1]
    rank = noise.rank
    nc = noise.n_channels
    dev = channel_index.device

    Cooinv = torch.zeros((nneighb, rank, nc_obs, rank, nc_obs), device=dev)

    for j, joix in enumerate(channel_index):
        (joixvix,) = (joix < nc).nonzero(as_tuple=True)
        joixv = joix[joixvix]
        ncoi = joixv.numel()

        jCoo = noise.marginal_covariance(
            channels=joixv, cache_prefix=neighborhoods.name, cache_key=j
        )

        jL = jCoo.cholesky(upper=False)  # C = LL'
        jLinv = jL.inverse().to_dense()
        jCooinv = jLinv.T @ jLinv  # Cinv = Linv' Linv

        # fancy inds to front! love that.
        jCooinv = jCooinv.view(rank, ncoi, rank, ncoi)
        Cooinv[j, :, joixvix[:, None], :, joixvix[None]] = jCooinv.permute(1, 3, 0, 2)

    obsdim = rank * nc_obs
    Cooinv = Cooinv.view(nneighb, obsdim, obsdim)
    return Cooinv
