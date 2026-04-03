"""Library for flavors of kernel interpolation and data interp utilities"""

from typing import TYPE_CHECKING, Self

import numpy as np
import torch
import torch.nn.functional as F

from .data_util import yield_masked_chunks
from .internal_config import (
    InterpKernel,
    InterpMethod,
    InterpolationParams,
    default_interpolation_params,
)
from .motion import MotionInfo
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
    trim_to_rank: int | None = None,
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
    if trim_to_rank:
        out_feature_dim = min(trim_to_rank, feature_dim)
    else:
        out_feature_dim = feature_dim
    out_shape = n_spikes, out_feature_dim, n_target_chans
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
        shift_dim=shift_dim,
    )
    for sli, chunk_features in yield_masked_chunks(
        mask, dataset, show_progress=show_progress, desc_prefix="Interpolating"
    ):
        # interpolate, store
        if trim_to_rank is not None:
            chunk_features = chunk_features[:, :trim_to_rank]
        chunk_features = torch.asarray(chunk_features, device=device, dtype=dtype)
        out[sli] = erp.interp(
            features=chunk_features,
            source_main_channels=channels[sli].to(device),
            target_channels=target_channels[sli],
            source_shifts=shifts[sli].to(device),
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
    if params.method in ("nearest", "kernel", "normalized", "zero", "nan"):
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
        polyharmonic_order=params.polyharmonic_order,
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
    return interp_precompute(
        source_geom=pad_geom(source_geom),
        channel_index=channel_index,
        params=params,
        source_geom_is_padded=True,
    )


def kernel_interpolate(
    features,
    source_pos,
    target_pos,
    params: InterpolationParams = default_interpolation_params,
    precomputed_data=None,
    neighborhoods=None,
    solver_map=None,
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
            polyharmonic_order=params.polyharmonic_order,
            precomputed_data=precomputed_data,
            neighborhoods=neighborhoods,
            solver_map=solver_map,
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
        polyharmonic_order=params.polyharmonic_order,
        precomputed_data=precomputed_data,
        solver_map=solver_map,
        neighborhoods=neighborhoods,
        allow_destroy=allow_destroy,
        out=out,
    )

    if extrap_diff:
        # control over extrapolation with another method...
        assert features_out is not None
        targ_extrap = extrap_mask(source_pos, target_pos)[:, None].broadcast_to(
            features.shape
        )
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
    polyharmonic_order: int | float,
    precomputed_data,
    neighborhoods,
    solver_map=None,
    allow_destroy=False,
    out=None,
):
    shared_kernel = source_pos.ndim == target_pos.ndim == 2
    if shared_kernel:
        source_pos = source_pos[None]
        target_pos = target_pos[None]
    else:
        assert source_pos.ndim == target_pos.ndim == 3
    out_shape = (*features.shape[:2], target_pos.shape[1])
    is_nearest = method == "nearest" or kernel_name == "nearest"
    is_clampna = method == "clampna" or kernel_name == "clampna"
    do_nearest = is_nearest or is_clampna
    d = None
    if out is not None:
        assert out.shape == out_shape
    if method == "nan" or kernel_name == "nan":
        if out is None:
            return features.new_full(out_shape, torch.nan)
        else:
            return out.fill_(torch.nan)
    elif method == "zero" or kernel_name == "zero":
        if out is None:
            return features.new_zeros(out_shape)
        else:
            return out.zero_()
    elif do_nearest:
        d = get_rsq(source_pos, target_pos, nan=torch.inf)
        dmin, nixs = d.min(dim=1, keepdim=True)
        nixs = nixs.broadcast_to(out_shape)
        out = torch.take_along_dim(input=features, dim=2, indices=nixs, out=out)
        if is_nearest:
            return out
        elif is_clampna:
            out.masked_fill_(dmin > (2 * sigma) ** 2, torch.nan)
            return out
        else:
            assert False

    kernel = get_kernel(
        source_pos=source_pos,
        target_pos=target_pos,
        kernel_name=kernel_name.removesuffix("normalized"),
        sigma=sigma,
        rq_alpha=rq_alpha,
        normalized=method.endswith("normalized"),
        smoothing_lambda=smoothing_lambda,
        polyharmonic_order=polyharmonic_order,
    )

    features = torch.nan_to_num(features, out=features if allow_destroy else None)
    if method == "kriging":
        assert precomputed_data is not None
        precomputed_data = precomputed_data.to(features)
        features = kriging_solve(
            target_pos=target_pos,
            kernels=kernel,
            features=features,
            solvers=precomputed_data,
            solver_map=solver_map,
            neighborhoods=neighborhoods,
            sigma=sigma,
            poly_degree=kriging_poly_degree,
        )
    else:
        features = torch.bmm(features, kernel, out=out)

    # nan-ify nonexistent chans
    needs_nan = torch.isnan(target_pos).all(2).unsqueeze(1)
    needs_nan = needs_nan.broadcast_to(features.shape)
    features.masked_fill_(needs_nan, torch.nan)
    assert features.shape == out_shape

    return features


def get_kernel(
    source_pos,
    target_pos=None,
    kernel_name="rbf",
    sigma=20.0,
    rq_alpha=1.0,
    normalized=False,
    smoothing_lambda=0.0,
    polyharmonic_order=2,
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
    elif kernel_name == "polyharmonic":
        kernel = polyharmonic_rbf(
            source_pos=source_pos,
            target_pos=target_pos,
            sigma=sigma,
            order=polyharmonic_order,
        )
    else:
        assert False

    kernel = kernel.nan_to_num_()
    if normalized and kernel_name not in ("rbf", "nearest"):
        kernel = kernel.div_(kernel.sum(dim=1, keepdim=True))

    if smoothing_lambda:
        kernel.diagonal(dim1=-2, dim2=-1).add_(smoothing_lambda)

    return kernel


def kriging_solve(
    target_pos: torch.Tensor,
    kernels: torch.Tensor,
    features: torch.Tensor,
    solvers: torch.Tensor,
    neighborhoods: torch.Tensor | None,
    solver_map: torch.Tensor | None = None,
    sigma=1.0,
    poly_degree=-1,
) -> torch.Tensor:
    if neighborhoods is None:
        assert solvers.ndim == 3
        kernels, y, _ = kriging_poly_expand(
            target_pos, features, kernels, poly_degree, sigma
        )
        assert y is not None
        return y.bmm(solvers).bmm(kernels)
    else:
        return kriging_neighborhood_solve(
            solvers=solvers,
            features=features,
            kernels=kernels,
            target_pos=target_pos,
            solver_map=solver_map,
            neighborhoods=neighborhoods,
            poly_degree=poly_degree,
            sigma=sigma,
        )


def kriging_neighborhood_solve(
    solvers: torch.Tensor,
    features: torch.Tensor,
    kernels: torch.Tensor,
    target_pos: torch.Tensor,
    solver_map: torch.Tensor | None,
    neighborhoods: torch.Tensor,
    poly_degree: int,
    sigma: float,
) -> torch.Tensor:
    """This is like the "neighborhoods" option to numpy's RBFInterpolator

    Each output channel is handled independently using a per-neighborhood solver.
    This is for numerical reasons. It's used in the full-probe interpolation where
    we'd otherwise be trying to invert large ill-conditioned kriging coefficient
    matrices.
    """
    assert kernels.ndim == 3, 1
    assert kernels.shape[0] == 1, 2  # shared kernel in this context
    kernel = kernels[0]
    assert kernel.shape[0] == features.shape[2], 3
    del kernels
    assert target_pos.shape[0] == 1
    target_pos = target_pos[0]

    input_channels, output_channels = kernel.shape
    arange_out = torch.arange(output_channels)
    # add extra row of 0s so that these can be indexed by the neighborhoods array,
    # which will contain n_channels entries
    kernel = F.pad(kernel, (0, 0, 0, 1))
    features = F.pad(features, (0, 1))

    if solver_map is not None:
        solvers = solvers[solver_map]
        neighborhoods = neighborhoods[solver_map]
    n_neighborhoods, nc_neighb = neighborhoods.shape

    # construct output kernels for each neighborhood
    # each output channel's kernel is k[:, neighb[outchan], outchan], plus the
    # expansion that would normally be done by kriging_poly_expand below
    neighb_kernels = kernel[neighborhoods, arange_out[:, None]]
    assert neighb_kernels.shape == (n_neighborhoods, nc_neighb), 4
    neighb_kernels, _, extra_dim = kriging_poly_expand(
        target_pos=target_pos[:, None],
        features=None,
        kernels=neighb_kernels[:, :, None],
        poly_degree=poly_degree,
        sigma=sigma,
    )
    assert neighb_kernels.shape == (n_neighborhoods, nc_neighb + extra_dim, 1), 5
    neighb_solved = solvers.bmm(neighb_kernels)
    assert neighb_solved.shape == (n_neighborhoods, nc_neighb + extra_dim, 1), 6
    neighb_solved = neighb_solved[:, :, 0]

    # now, pad out neighborhoods with extra_dim `input_channels`s so that the
    # zero-padding is carried out correctly in the loop
    neighborhoods_padded = F.pad(neighborhoods, (0, extra_dim), value=input_channels)
    features_flat = features.view(-1, features.shape[2])
    assert features_flat.shape[1] == input_channels + 1
    assert neighborhoods_padded.ndim == 2
    assert neighborhoods_padded.shape == neighb_solved.shape
    out_flat = _kneighb_loop(features_flat, neighborhoods_padded, neighb_solved)
    assert out_flat.shape[1] == output_channels, 7
    out = out_flat.view(*features.shape[:2], output_channels)
    return out


@torch.jit.script
def _kneighb_loop(
    features_padded_flat: torch.Tensor,
    neighborhoods_padded: torch.Tensor,
    neighb_solved: torch.Tensor,
):
    out = features_padded_flat.new_empty(
        (features_padded_flat.shape[0], neighborhoods_padded.shape[0])
    )
    for j in range(neighborhoods_padded.shape[0]):
        neighb = neighborhoods_padded[j]
        fj = features_padded_flat[:, neighb]
        torch.mv(fj, neighb_solved[j], out=out[:, j])
    return out


def kriging_poly_expand(
    target_pos: torch.Tensor,
    features: torch.Tensor | None,
    kernels: torch.Tensor,
    poly_degree: int,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Add polynomial basis terms to features and kernels

    Features get extra rows of zeros, and kernels get the basis stuff.

    The zeros get tacked on along the last dimension of features, which is the
    input channels dimension to the kernel.

    Similarly, the kernels get the basis terms stacked onto their input dimension,
    which is the first post-batch dimension, dim 1.
    """
    n, n_targ, dim = target_pos.shape
    assert kernels.shape[0] == n
    assert kernels.shape[2] == n_targ
    if features is not None:
        assert n == features.shape[0]
    if poly_degree == -1:
        extra_dim = 0
    elif poly_degree == 0:
        extra_dim = 1
        const = kernels.new_ones((n, 1, n_targ))
        kernels = torch.concatenate([kernels, const], dim=1)
    elif poly_degree == 1:
        extra_dim = 1 + dim
        const = kernels.new_ones((n, 1, n_targ))
        xy = (target_pos / sigma).nan_to_num_().mT
        kernels = torch.concatenate([kernels, xy, const], dim=1)
    elif poly_degree == 2:
        extra_dim = 1 + dim + (dim * (dim + 1)) // 2
        const = kernels.new_ones((n, 1, n_targ))
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

    if extra_dim and features is not None:
        rank = features.shape[1]
        zero = features.new_zeros((n, rank, extra_dim))
        y = torch.concatenate([features, zero], dim=2)
    else:
        y = features

    return kernels, y, extra_dim


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
        polyharmonic_order=params.polyharmonic_order,
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
    big = rsq.mul_(r.log())
    kernel = torch.where(is_small, small, big)
    return kernel


def polyharmonic_rbf(source_pos, target_pos=None, sigma=1.0, order: int | float=2.0):
    rsq = get_rsq(source_pos, target_pos, sigma)
    r = rsq.sqrt_()
    if isinstance(order, int) and (order % 2):
        return r.pow_(order)
    is_small = r < 1
    small = r.pow(order - 1).mul(r.pow(r).log_())
    big = r.pow(order).mul_(r.log())
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
    if source_pos.ndim == target_pos.ndim == 2:
        source_pos = source_pos[None]
        target_pos = target_pos[None]
    assert source_pos.ndim == target_pos.ndim == 3
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


class SpikeNeighborhoods(BModule):
    def __init__(
        self,
        n_channels: int,
        neighborhood_ids,
        neighborhoods,
        features=None,
        neighborhood_members=None,
        device=None,
        name=None,
    ):
        """SpikeNeighborhoods

        Sparsely keep track of which channels each spike lives on. Used to query
        which core sets are overlapped completely by unit channel neighborhoods.

        Arguments
        ---------
        neighborhood_ids : torch.Tensor
            Size (n_spikes,), the neighborhood id for each spike
        neighborhoods : list[torch.Tensor]
            The channels in each neighborhood
        neighborhood_members : list[torch.Tensor]
            The indices of spikes in each neighborhood
        """
        super().__init__()
        self.name = name
        self.n_channels = n_channels
        self.register_buffer("neighborhood_ids", neighborhood_ids.long())
        self.register_buffer("chans_arange", torch.arange(n_channels, dtype=torch.long))
        self.register_buffer("neighborhoods", neighborhoods.long())
        self.n_neighborhoods = len(neighborhoods)

        # store neighborhoods as an indicator matrix
        # also store nonzero-d masks
        indicators = torch.zeros((n_channels, len(neighborhoods)), device=device)
        masks = []
        mask_slices = []
        offset = 0
        for j, nhood in enumerate(neighborhoods):
            jvalid = nhood < n_channels
            indicators[nhood[jvalid], j] = 1.0
            (jvalid,) = jvalid.nonzero(as_tuple=True)
            jvalid = jvalid.long()
            njvalid = jvalid.numel()
            assert njvalid
            masks.append(jvalid)
            mask_slices.append(slice(offset, offset + njvalid))
            offset += njvalid
        self.register_buffer("indicators", indicators)
        self.register_buffer("channel_counts", indicators.sum(0))
        self.register_buffer("_masks", torch.concatenate(masks, dim=0))
        self._mask_slices = mask_slices

        if neighborhood_members is None:
            # cache lookups
            neighborhood_members = []
            for j in range(len(neighborhoods)):
                (in_nhood,) = torch.nonzero(neighborhood_ids == j, as_tuple=True)
                in_nhood = in_nhood.long()
                neighborhood_members.append(in_nhood.cpu())
        assert len(neighborhood_members) == self.n_neighborhoods

        # it's a pain to store dicts with register_buffer, so store offsets
        _neighborhood_members = torch.empty(
            sum(v.numel() for v in neighborhood_members), dtype=torch.long
        )
        self.neighborhood_members_slices = []
        neighborhood_member_offset = 0
        neighborhood_popcounts = []
        for j in range(len(neighborhoods)):
            nhoodmemsz = neighborhood_members[j].numel()
            nhoodmemsl = slice(
                neighborhood_member_offset, neighborhood_member_offset + nhoodmemsz
            )
            _neighborhood_members[nhoodmemsl] = neighborhood_members[j]
            self.neighborhood_members_slices.append(nhoodmemsl)
            neighborhood_member_offset += nhoodmemsz
            neighborhood_popcounts.append(nhoodmemsz)
        # self.register_buffer("_neighborhood_members", _neighborhood_members)
        # seems that indices want to live on cpu.
        self._neighborhood_members = _neighborhood_members.cpu()
        self.register_buffer("popcounts", torch.tensor(neighborhood_popcounts))

        if features is not None:
            _features_valid = []
            for j in range(len(neighborhoods)):
                f = features[self.neighborhood_members(j)]
                f = f[..., self.valid_mask(j).to(f.device)]
                if device is not None and device.type == "cuda":
                    f = f.pin_memory()
                _features_valid.append(f)
            self._features_valid = _features_valid
        self.to(device=device)

    @classmethod
    def from_channels(
        cls,
        channels,
        n_channels,
        neighborhood_ids=None,
        neighborhoods=None,
        device=None,
        deduplicate=False,
        features=None,
        name=None,
    ):
        if neighborhood_ids is not None:
            assert neighborhoods is not None
            return cls.from_known_ids(
                n_channels=n_channels,
                neighborhood_ids=neighborhood_ids,
                neighborhoods=neighborhoods,
                device=device,
                deduplicate=deduplicate,
                features=features,
                name=name,
            )
        if device is not None:
            channels = channels.to(device)
        neighborhoods, neighborhood_ids = torch.unique(
            channels, dim=0, return_inverse=True
        )
        neighborhoods = neighborhoods.long()
        neighborhood_ids = neighborhood_ids.long()
        return cls(
            n_channels=n_channels,
            neighborhoods=neighborhoods,
            neighborhood_ids=neighborhood_ids,
            features=features,
            device=channels.device,
            name=name,
        )

    @classmethod
    def from_known_ids(
        cls,
        *,
        n_channels: int,
        neighborhood_ids,
        neighborhoods,
        device=None,
        deduplicate=False,
        features=None,
        name=None,
    ):
        neighborhoods = torch.asarray(neighborhoods, dtype=torch.long)
        neighborhood_ids = torch.asarray(neighborhood_ids, dtype=torch.long)
        if device is not None:
            neighborhoods = neighborhoods.to(device)
            neighborhood_ids = neighborhood_ids.to(device)
        if deduplicate:
            neighborhoods, old2new = torch.unique(
                neighborhoods, dim=0, return_inverse=True
            )
            neighborhood_ids = old2new[neighborhood_ids]
            kept_ids, neighborhood_ids = torch.unique(
                neighborhood_ids, return_inverse=True
            )
            neighborhoods = neighborhoods[kept_ids]
        return cls(
            n_channels=n_channels,
            neighborhoods=neighborhoods,
            neighborhood_ids=neighborhood_ids,
            features=features,
            device=device,
            name=name,
        )

    def slice(self, indices: torch.Tensor | slice) -> Self:
        return self.__class__(
            n_channels=self.n_channels,
            neighborhood_ids=self.b.neighborhood_ids[indices],
            neighborhoods=self.b.neighborhoods,
            device=self.b.neighborhoods.device,
            name=self.name,
        )

    def has_feature_cache(self):
        return hasattr(self, "_features_valid")

    def valid_mask(self, id):
        return self._masks[self._mask_slices[id]]  # type: ignore

    def neighborhood_channels(self, id):
        nhc = self.b.neighborhoods[id]
        return nhc[nhc < self.n_channels]

    def missing_channels(self, id):
        return self.b.chans_arange[self.b.indicators[:, id] == 0]

    def neighborhood_members(self, id):
        return self._neighborhood_members[self.neighborhood_members_slices[id]]

    def neighborhood_features(
        self, id, batch_start=None, batch_size=None, batch_buffer=None
    ):
        f = self._features_valid[id]
        if batch_start is not None:
            f = f[batch_start : batch_start + batch_size]
        if batch_buffer is not None:
            batch_buffer[: len(f)] = f
            return batch_buffer[: len(f)]
        else:
            return f

    def subset_neighborhoods(self, channels, min_coverage=1.0, batch_size=None):
        """Return info on neighborhoods which cover the channel set well enough

        Define coverage for a neighborhood and a channel group as the intersection
        size divided by the neighborhood's size.

        Returns
        -------
        neighborhood_info : list of tuples
            Each entry is, in order,
             - neighborhood id
             - neighborhood channels array
             - neighborhood member indices
             - optional batch start
            representing a batch of spikes living on that neighborhood.
        n_spikes : int
            The total number of spikes in the neighborhood.
        """
        inds = self.b.indicators[channels]
        coverage = inds.sum(0) / self.b.channel_counts
        (covered_ids,) = torch.nonzero(coverage >= min_coverage, as_tuple=True)
        n_spikes = self.b.popcounts[covered_ids].sum()

        neighborhood_info = []
        for j in covered_ids:
            jneighb = self.b.neighborhoods[j]
            jmems = self.neighborhood_members(j)
            if batch_size is None or len(jmems) < batch_size:
                neighborhood_info.append((j, jneighb, jmems, None))
            else:
                for bs in range(0, len(jmems), batch_size):
                    mem_batch = jmems[bs : bs + batch_size]
                    neighborhood_info.append((j, jneighb, mem_batch, bs))

        return covered_ids, neighborhood_info, n_spikes

    def spike_neighborhoods(
        self, channels, neighborhood_ids=None, spike_indices=None, min_coverage=1.0
    ):
        """Like subset_neighborhoods, but for an already chosen collection of spikes

        This is used when subsetting log likelihood calculations.
        In this case, the returned neighborhood_member_indices keys are relative:
        spike_indices[neighborhood_member_indices] are the actual indices.
        """
        if neighborhood_ids is None:
            assert spike_indices is not None
            neighborhood_ids = self.b.neighborhood_ids[spike_indices]
        assert neighborhood_ids is not None

        covered_ids = torch.unique(neighborhood_ids)
        if min_coverage:
            covered_ids = covered_ids.to(self.indicators.device)
            inds = self.b.indicators[channels][:, covered_ids]
            coverage = inds.sum(0) / self.b.channel_counts[covered_ids]
            covered = coverage >= min_coverage
            covered_ids = covered_ids[covered].cpu()
            neighborhood_ids = neighborhood_ids.cpu()

        neighborhood_info = [
            (
                j,
                self.b.neighborhoods[j],
                *(neighborhood_ids == j).nonzero(as_tuple=True),
                None,
            )
            for j in covered_ids
        ]
        n_spikes = self.b.popcounts[covered_ids].sum()
        return neighborhood_info, n_spikes

    def adjacency(self, overlap=0.5):
        overlaps = self.b.indicators.T @ self.b.indicators
        assert overlaps.shape == (self.n_neighborhoods, self.n_neighborhoods)
        counts = self.b.indicators.sum(0)
        overlaps /= torch.minimum(counts[:, None], counts)
        return (overlaps >= overlap - 1e-5).to(torch.float)

    def partial_order(self):
        """ret[i, j] == 1 iff neighb j subset neighb i"""
        inds = self.b.indicators.T  # nneighb x nc
        po = (inds[:, None, :] >= inds[None, :, :]).all(2)
        assert po.shape == (self.n_neighborhoods, self.n_neighborhoods)
        return po


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
        self.erp_params = params.normalize()
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
            params=self.erp_params,
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
            params=self.erp_params,
            precomputed_data=pcomp,
            allow_destroy=allow_destroy,
        )


class NeighborhoodFiller(BModule):
    batch_size: int

    def interp_to_chans(
        self,
        waveforms: torch.Tensor,
        neighborhood_ids: torch.Tensor,
        target_channels: torch.Tensor | slice,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if waveforms.shape[0] <= self.batch_size and out is None:
            return self._interp_to_chans(waveforms, neighborhood_ids, target_channels)

        if torch.is_tensor(target_channels):
            assert target_channels.ndim == 1
            ntarg = target_channels.shape[0]
        else:
            ntarg = len(range(*target_channels.indices(len(self.b.prgeom))))
        out_shp = (*waveforms.shape[:2], ntarg)
        if out is None:
            out = waveforms.new_zeros(out_shp)
        else:
            assert out.shape == out_shp
        n = out.shape[0]
        for i0 in range(0, n, self.batch_size):
            i1 = min(n, i0 + self.batch_size)
            out[i0:i1] = self._interp_to_chans(
                waveforms=waveforms[i0:i1],
                neighborhood_ids=neighborhood_ids[i0:i1],
                target_channels=target_channels,
            )
        return out

    def _interp_to_chans(
        self,
        waveforms: torch.Tensor,
        neighborhood_ids: torch.Tensor,
        target_channels: torch.Tensor | slice,
    ) -> torch.Tensor:
        raise NotImplementedError


class NeighborhoodInterpolator(NeighborhoodFiller):
    def __init__(
        self,
        prgeom: torch.Tensor,
        neighborhoods: SpikeNeighborhoods,
        params: InterpolationParams = default_interpolation_params,
        batch_size: int = 1024,
    ):
        super().__init__()
        assert len(prgeom) == neighborhoods.n_channels + 1
        self.erp_params: InterpolationParams = params.normalize()
        self.batch_size = batch_size
        self.register_buffer("prgeom", prgeom.clone())
        self.b.prgeom[-1].fill_(torch.nan)
        neighb_data = interp_precompute(
            source_geom=self.prgeom,
            channel_index=neighborhoods.neighborhoods,
            source_geom_is_padded=True,
            params=self.erp_params,
        )
        self.register_buffer_or_none("neighb_data", neighb_data)
        self.register_buffer("neighb_pos", self.b.prgeom[neighborhoods.b.neighborhoods])

    def _interp_to_chans(
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
            params=self.erp_params,
        )


class NeighborhoodImputer(NeighborhoodFiller):
    def __init__(
        self,
        noise: EmbeddedNoise,
        neighborhoods: SpikeNeighborhoods,
        batch_size: int = 1024,
    ):
        super().__init__()
        prec = all_neighb_precisions(noise, neighborhoods)
        self.register_buffer("prec", prec)
        chans_arange = torch.arange(noise.n_channels, device=prec.device)
        self.register_buffer("chans_arange", chans_arange)
        self.neighborhoods = neighborhoods
        self.noise = noise
        self.batch_size = batch_size

    def _interp_to_chans(
        self,
        waveforms: torch.Tensor,
        neighborhood_ids: torch.Tensor,
        target_channels: torch.Tensor | slice,
    ) -> torch.Tensor:
        if not torch.is_tensor(target_channels):
            target_channels = self.b.chans_arange[target_channels]
        source_channels = self.neighborhoods.b.neighborhoods[neighborhood_ids]

        waveforms = waveforms.view(waveforms.shape[0], 1, -1)
        waveforms = waveforms.bmm(self.b.prec[neighborhood_ids])
        waveforms = waveforms.view(waveforms.shape[0], self.noise.rank, -1)

        return self.noise.cov_batch_mul(waveforms, source_channels, target_channels)


class ToFullProbeInterpolator(BModule):
    """Interpolate from the drifting geom to the registered probe.

    Would suggest avoiding extrapolation in your @params.

    Algorithm:
     - Find nearest registered channel for drifting geom at each time
        - Shift each geom channel at each time and query rgeom kdtree
     - Extract those channels from the registered geom
     - Interpolate from the geom, treated as static, to the rgeom shifted
       inversely. (This is so we can cache precomputed kriging solvers.)
    """

    def __init__(
        self, *, motion: MotionInfo, params: InterpolationParams, device: torch.device
    ):
        super().__init__()
        self.erp_params = params.normalize()
        geom = torch.asarray(motion.geom, dtype=torch.float, device=device)
        rgeom = torch.asarray(motion.rgeom, dtype=torch.float, device=device)
        channel_index = make_channel_index(
            geom, radius=self.erp_params.neighborhood_radius, to_torch=True
        )
        self.motion = motion
        channel_index = channel_index.to(device=geom.device)
        self.register_buffer_or_none(
            "data", full_probe_precompute(geom, channel_index, self.erp_params)
        )
        if self.b.data is None:
            channel_index = None
        self.register_buffer_or_none("channel_index", channel_index)
        self.rg_depths = rgeom[:, 1].numpy(force=True)
        self.register_buffer("geom", geom)
        self.register_buffer("rgeom", rgeom)
        self.dim = geom.shape[1]

    def interp_at_time(self, t_s: float, waveforms: torch.Tensor) -> torch.Tensor:
        assert waveforms.shape[2] == self.b.geom.shape[0]

        # get the target geom, which is the rgeom shifted to match the drift
        if self.motion.drifting:
            disp = self.motion.disp_at_s(
                times_s=np.array([t_s]), depths_um=self.rg_depths, grid=True
            )
            assert disp.shape[1] == 1
            tgeom = self.b.rgeom.clone()
            depth_shift = torch.tensor(
                disp[:, 0], device=tgeom.device, dtype=tgeom.dtype
            )
            tgeom[:, 1] += depth_shift
        else:
            tgeom = self.b.rgeom

        # which rgeom channel does each shifted geom channel land on?
        # solvers are selected as geom chans, though.
        sgeom = self.b.geom
        dist = torch.cdist(tgeom, sgeom)
        solver_map = dist.argmin(1)

        # interpolate from static geom to shifted rgeom
        return kernel_interpolate(
            features=waveforms,
            source_pos=sgeom,
            target_pos=tgeom,
            precomputed_data=self.b.data,
            neighborhoods=self.b.channel_index,
            solver_map=solver_map,
            params=self.erp_params,
        )


class FromFullProbeInterpolator(BModule):
    """Interpolate from the registered geom to appropriate drifting geom channels."""

    def __init__(
        self, *, motion: MotionInfo, params: InterpolationParams, device: torch.device
    ):
        super().__init__()
        geom = torch.asarray(motion.geom, dtype=torch.float, device=device)
        rgeom = torch.asarray(motion.rgeom, dtype=torch.float, device=device)

        self.erp_params = params.normalize()
        rchannel_index = make_channel_index(
            rgeom, radius=self.erp_params.neighborhood_radius, to_torch=True
        )
        self.motion = motion
        rchannel_index = rchannel_index.to(device=rgeom.device)
        self.register_buffer_or_none(
            "data", full_probe_precompute(rgeom, rchannel_index, self.erp_params)
        )
        if self.b.data is None:
            rchannel_index = None
        self.register_buffer_or_none("rchannel_index", rchannel_index)
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
        if self.motion.drifting:
            disp = self.motion.disp_at_s(
                times_s=np.array([t_s]), depths_um=self.g_depths, grid=True
            )
            assert disp.shape[1] == 1
            shift[:, 1].copy_(torch.tensor(disp[:, 0]))

        # which rgeom channel does each shifted geom channel land on?
        sgeom = self.b.geom - shift
        dist = torch.cdist(sgeom, self.b.rgeom)
        # closest rgeom channel to each shifted channel determines which
        # precomputed solver is selected
        solver_map = dist.argmin(1)

        # interpolate from static geom to shifted geom
        n = waveforms.shape[0]
        return kernel_interpolate(
            features=waveforms,
            source_pos=self.b.rgeom,
            target_pos=self.b.geom - shift,
            precomputed_data=self.b.data,
            neighborhoods=self.b.rchannel_index,
            solver_map=solver_map,
            params=self.erp_params,
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
