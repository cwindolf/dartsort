"""Library for flavors of kernel interpolation and data interp utilities"""

import numpy as np
import torch
import torch.nn.functional as F

from .data_util import yield_masked_chunks
from .drift_util import get_spike_pitch_shifts, static_channel_neighborhoods


def interpolate_by_chunk(
    mask,
    dataset,
    geom,
    channel_index,
    channels,
    shifts,
    registered_geom,
    target_channels,
    method="normalized",
    extrap_method=None,
    kernel_name="rbf",
    kriging_poly_degree=-1,
    sigma=10.0,
    rq_alpha=1.0,
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
    assert mask.dtype == np.bool_
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

    # if needed, precompute things
    precomputed_data = interp_precompute(
        source_geom=source_geom,
        channel_index=channel_index,
        method=method,
        kernel_name=kernel_name,
        sigma=sigma,
        rq_alpha=rq_alpha,
        kriging_poly_degree=kriging_poly_degree,
    )

    for ixs, chunk_features in yield_masked_chunks(
        mask, dataset, show_progress=show_progress, desc_prefix="Interpolating"
    ):
        # where are the spikes?
        source_channels = channel_index[channels[ixs]].to(device)
        source_shifts = shifts[ixs].to(device)
        if source_shifts.ndim == 1:
            # allows per-channel shifts
            source_shifts = source_shifts.unsqueeze(1)
        if shift_dim == 0:
            source_shifts = torch.stack(
                [source_shifts, torch.zeros_like(source_shifts)], dim=-1
            )
        elif shift_dim == 1:
            source_shifts = torch.stack(
                [torch.zeros_like(source_shifts), source_shifts], dim=-1
            )
        else:
            assert False

        # used to shift the source, but for kriging it's better to shift targets
        # so that we can cache source kernel choleskys
        source_pos = source_geom[source_channels]  # + source_shifts

        # where are they going?
        target_pos = target_geom[target_channels[ixs]] - source_shifts

        pcomp_batch = None
        if precomputed_data is not None:
            pcomp_batch = precomputed_data[channels[ixs]]

        # interpolate, store
        chunk_features = torch.from_numpy(chunk_features).to(device)
        chunk_res = kernel_interpolate(
            chunk_features,
            source_pos,
            target_pos,
            method=method,
            extrap_method=extrap_method,
            kernel_name=kernel_name,
            sigma=sigma,
            rq_alpha=rq_alpha,
            kriging_poly_degree=kriging_poly_degree,
            precomputed_data=pcomp_batch,
            allow_destroy=True,
        )
        out[ixs] = chunk_res.to(out)

    return out


def interp_precompute(
    source_geom=None,
    channel_index=None,
    source_pos=None,
    method="normalized",
    kernel_name="rbf",
    sigma=20.0,
    rq_alpha=1.0,
    kriging_poly_degree=-1,
    source_geom_is_padded=True,
):
    if method in ("nearest", "kernel", "normalized", "zero"):
        return None
    assert method in ("kriging", "krigingnormalized")

    if source_pos is None:
        assert source_geom is not None
        n_source_chans = len(source_geom) - source_geom_is_padded
        if channel_index is None:
            channel_index = torch.arange(n_source_chans)[None]
            channel_index = channel_index.to(source_geom.device)
        else:
            assert len(channel_index) == n_source_chans
        source_pos = source_geom[channel_index]
        valid = channel_index < n_source_chans
    else:
        assert source_geom is None
        assert channel_index is None
        valid = source_pos[..., 0].isfinite()
    assert source_pos.ndim == 3
    valid = valid.to(source_pos.device)

    ns = len(source_pos)
    neighb_size = source_pos.shape[1]
    if kriging_poly_degree < 0:
        design_vars = 0
    elif kriging_poly_degree == 0:
        design_vars = 1
    elif kriging_poly_degree == 1:
        design_vars = 1 + source_pos.shape[2]
    else:
        assert False

    solvers = source_pos.new_zeros(
        (ns, neighb_size + design_vars, neighb_size + design_vars)
    )
    design_inds = torch.arange(neighb_size, neighb_size + design_vars)
    design_inds = design_inds.to(source_pos.device)
    design_zeros = source_pos.new_zeros((design_vars, design_vars))

    source_kernels = get_kernel(
        source_pos, kernel_name=kernel_name, sigma=sigma, rq_alpha=rq_alpha
    )
    for j in range(ns):
        (present,) = valid[j].nonzero(as_tuple=True)
        kernel = source_kernels[j][present][:, present]

        if design_vars:
            pos = source_pos[j][present] / sigma
            const = pos.new_ones((pos.shape[0], 1))
            ix = torch.concatenate((present, design_inds), dim=0)
            if kriging_poly_degree == 0:
                design = const
            else:
                design = torch.cat([pos, const], dim=1)

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


def get_kernel(
    source_pos,
    target_pos=None,
    kernel_name="rbf",
    sigma=20.0,
    rq_alpha=1.0,
    normalized=False,
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
        kernel = multiquadric_kernel(source_pos=source_pos, target_pos=target_pos, sigma=sigma)
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
    else:
        assert False

    if solvers.ndim == 3:
        return y.bmm(solvers).bmm(kernels)

    assert solvers.ndim == 4
    assert solvers.shape[0] == 1
    # per-channel case (each output chan has its own local neighb)
    # this is meant to mimic the "neighbors" option to scipy's RBFInterpolator,
    # but the implementation here is obscure. the rest of the logic is only shown
    # once, and that's in the residual interpolation in noise_util.
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


def pad_geom(geom, dtype=torch.float, device=None):
    geom = torch.as_tensor(geom, dtype=dtype, device=device)
    geom = F.pad(geom, (0, 0, 0, 1), value=torch.nan)
    return geom


def kernel_interpolate(
    features,
    source_pos,
    target_pos,
    method="normalized",
    extrap_method=None,
    kernel_name="rbf",
    sigma=20.0,
    rq_alpha=1.0,
    kriging_poly_degree=-1,
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
    if method == "nearest":
        method = "kernel"
        kernel_name = "nearest"
    elif method == "zero":
        method = "kernel"
        kernel_name = "zero"
    extrap_kernel_name = kernel_name
    if extrap_method == "nearest":
        extrap_method = "kernel"
        extrap_kernel_name = "nearest"
    elif method == "zero":
        method = "kernel"
        kernel_name = "zero"
    extrap_diff = extrap_method != method or extrap_kernel_name != kernel_name

    features_out = None
    if extrap_method is not None and extrap_diff:
        features_out = kernel_interpolate(
            features,
            source_pos,
            target_pos,
            method=extrap_method,
            kernel_name=extrap_kernel_name,
            sigma=sigma,
            rq_alpha=rq_alpha,
            kriging_poly_degree=kriging_poly_degree,
            precomputed_data=precomputed_data,
        )

    if precomputed_data is None:
        # if at all possible, don't hit this branch. it's just here for vis.
        precomputed_data = interp_precompute(
            source_pos=source_pos,
            method=method,
            kernel_name=kernel_name,
            sigma=sigma,
            rq_alpha=rq_alpha,
            kriging_poly_degree=kriging_poly_degree,
        )

    kernel = get_kernel(
        source_pos=source_pos,
        target_pos=target_pos,
        kernel_name=kernel_name,
        sigma=sigma,
        rq_alpha=rq_alpha,
        normalized=method.endswith("normalized"),
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
    features[needs_nan] = torch.nan

    if extrap_method is not None and extrap_diff:
        # control over extrapolation with another method...
        assert features_out is not None
        targ_extrap = extrap_mask(source_pos, target_pos)
        features = torch.where(
            targ_extrap[:, None], features_out, features, out=features
        )

    return features


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


def get_rsq(source_pos, target_pos=None, sigma=1.0, batch_size=16, nan=0.0):
    if not isinstance(sigma, (float, int)) or sigma != 1.0:
        source_pos = source_pos / sigma
    if target_pos is None:
        target_pos = source_pos
    elif not isinstance(sigma, (float, int)) or sigma != 1.0:
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

    # targ_extrap = target_pos[..., 0].clamp(x_low, x_high) != target_pos[..., 0]
    targ_extrap = targ_y.isfinite()

    for x in source_x_uniq:
        source_in_col = torch.isclose(source_pos[..., 0], x)
        targ_in_col = torch.isclose(targ_x, x)

        source_col_y = torch.where(source_in_col, source_pos[..., 1], torch.nan)
        targ_col_y = torch.where(targ_in_col, targ_y, torch.nan)

        source_low = source_col_y.nan_to_num(nan=torch.inf).amin(dim=1) - 1e-3
        source_high = source_col_y.nan_to_num(nan=-torch.inf).amax(dim=1) + 1e-3
        assert torch.isfinite(source_low).all()
        assert torch.isfinite(source_high).all()

        col_mask = (targ_col_y > source_high[:, None]).logical_or_(
            targ_col_y < source_low[:, None]
        )
        targ_extrap[targ_in_col] = targ_extrap[targ_in_col].logical_and_(
            col_mask[targ_in_col]
        )

    targ_extrap = torch.logical_and(targ_extrap, targ_y.isfinite())
    return targ_extrap
