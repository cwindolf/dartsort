"""Library for flavors of kernel interpolation and data interp utilities"""

from linear_operator import to_linear_operator
from linear_operator.operators import CholLinearOperator
import numpy as np
import torch
import torch.nn.functional as F

from .data_util import yield_masked_chunks
from .drift_util import get_spike_pitch_shifts, static_channel_neighborhoods

interp_kinds = (
    "nearest",
    "rbf",
    "normalized",
    "kriging",
    "kriging_normalized",
    "idw",
    "thinplate",
    "rq",
    "rq_0.5",
    "rq_1",
    "rq_2",
    "rq_normalized",
    "rq_normalized_0.5",
    "rq_normalized_1",
    "rq_normalized_2",
)


def interpolate_by_chunk(
    mask,
    dataset,
    geom,
    channel_index,
    channels,
    shifts,
    registered_geom,
    target_channels,
    sigma=10.0,
    interpolation_method="normalized",
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
    # devices, dtypes, shapes
    assert interpolation_method in interp_kinds
    assert geom.shape[1] == 2, "Haven't implemented 3d."

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dtype = torch.from_numpy(np.empty((), dtype=dataset.dtype)).dtype
    n_spikes = mask.sum()
    assert channels.shape == (n_spikes,)
    n_source_chans = channel_index.shape[1]
    assert n_source_chans == dataset.shape[2]
    n_target_chans = target_channels.shape[1]
    assert target_channels.shape == (n_spikes, n_target_chans)
    feature_dim = dataset.shape[1]

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
    skis = tpd = None
    source_kernel_invs = thin_plate_data = None
    do_skis = interpolation_method.startswith("kriging")
    if do_skis:
        source_kernel_invs = get_source_kernel_pinvs(
            source_geom, channel_index, sigma=sigma
        )
    if interpolation_method == "thinplate":
        thin_plate_data = thin_plate_precompute(source_geom, channel_index, sigma)

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

        if do_skis:
            assert source_kernel_invs is not None
            skis = source_kernel_invs[channels[ixs]]
        if interpolation_method == "thinplate":
            assert thin_plate_data is not None
            tpd = thin_plate_data[channels[ixs]]

        # interpolate, store
        chunk_features = torch.from_numpy(chunk_features).to(device)
        chunk_res = kernel_interpolate(
            chunk_features,
            source_pos,
            target_pos,
            source_kernel_invs=skis,
            thin_plate_data=tpd,
            sigma=sigma,
            allow_destroy=True,
            interpolation_method=interpolation_method,
        )
        out[ixs] = chunk_res.to(out)

    return out


def get_source_kernel_pinvs(source_geom, channel_index=None, sigma=20.0, eps=1e-5):
    """channel_index None means we'll make one inv of the full probe."""
    if channel_index is None:
        source_pos = source_geom[None]
    else:
        source_pos = source_geom[channel_index]
    source_kernels = log_rbf(source_pos, sigma=sigma)
    invs = torch.zeros_like(source_kernels)
    for j, sk in enumerate(source_kernels):
        (m,) = sk[0].isfinite().nonzero(as_tuple=True)
        sk = sk[m[:, None], m[None, :]]
        assert sk.isfinite().all()
        sk = sk.exp_()
        invs[j, m[:, None], m[None, :]] = torch.linalg.inv(sk)
    return invs


def thin_plate_precompute(source_geom, channel_index, sigma, source_geom_is_padded=True):
    """Precompute data for 2D thin plate spline interpolation

    This is copying skimage, more or less.
    https://www.geometrictools.com/Documentation/ThinPlateSplines.pdf
    is also a helpful reference.
    """
    assert source_geom.shape[1] == 2  # 3d also possible...
    n_source_chans = len(source_geom) - source_geom_is_padded
    if channel_index is None:
        channel_index = torch.arange(n_source_chans)[None]
        channel_index = channel_index.to(source_geom.device)
    else:
        assert len(channel_index) == n_source_chans

    n_source_neighbs = len(channel_index)
    neighb_size = channel_index.shape[1]
    Linv = source_geom.new_zeros((n_source_neighbs, neighb_size + 3, neighb_size + 3))
    design_inds = torch.arange(neighb_size, neighb_size + 3).to(source_geom.device)
    zeros3 = source_geom.new_zeros((3, 3))

    for j in range(n_source_neighbs):
        chans = channel_index[j]
        (present,) = (chans < n_source_chans).nonzero(as_tuple=True)
        npres = present.numel()
        source_pos = source_geom[chans[present]] / sigma
        assert source_pos.isfinite().all()
        ix = torch.concatenate((present, design_inds), dim=0)

        M = thin_plate_greens(source_pos)

        # design matrix
        N = torch.cat([source_pos, source_pos.new_ones((npres, 1))], dim=1)

        Ltop = torch.cat([M, N], dim=1)
        Lbot = torch.cat([N.T, zeros3], dim=1)
        L = torch.cat([Ltop, Lbot], dim=0)

        Linv[j, ix[:, None], ix[None]] = torch.linalg.inv(L)

    return Linv


def thin_plate_solve(source_pos, target_pos, Linv, features, sigma):
    G = thin_plate_greens(source_pos, target_pos, sigma).nan_to_num_()

    n, rank, neighb_size = features.shape
    Y = torch.cat([features, features.new_zeros((n, rank, 3))], dim=2)
    AB = torch.einsum("nij,npj->npi", Linv, Y)
    A = AB[..., :-3]
    N = torch.cat([target_pos / sigma, torch.ones_like(target_pos[..., :1])], dim=2)
    offset = torch.einsum("npd,nid->npi", AB[..., -3:], N)

    pred = torch.baddbmm(offset, A, G)
    return pred


def pad_geom(geom, dtype=torch.float, device=None):
    geom = torch.as_tensor(geom, dtype=dtype, device=device)
    geom = F.pad(geom, (0, 0, 0, 1), value=torch.nan)
    return geom


def kernel_interpolate(
    features,
    source_pos,
    target_pos,
    source_kernel_invs=None,
    thin_plate_data=None,
    sigma=10.0,
    allow_destroy=False,
    interpolation_method="normalized",
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
    source_kernel_invs : optional torch.Tensor
        Precomputed inverses of source-to-source kernel matrices,
        if you have them, for use in kriging
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
    assert interpolation_method in interp_kinds

    thinplate = interpolation_method == "thinplate"
    features_in = targ_extrap = None
    if thinplate:
        assert thin_plate_data is not None
        Linv = thin_plate_data

        features = torch.nan_to_num(features, out=features if allow_destroy else None)
        features_in = thin_plate_solve(source_pos, target_pos, Linv, features, sigma)

        needs_nan = torch.isnan(target_pos).all(2).unsqueeze(1)
        needs_nan = needs_nan.broadcast_to(features_in.shape)
        features_in[needs_nan] = torch.nan

        # are we extrapolating anywhere? if so, thinplate is a bad choice.
        targ_extrap = extrap_mask(source_pos, target_pos)
        # we'll use normalized to extrapolate below
        interpolation_method = "normalized"

    # -- build kernel
    if interpolation_method == "nearest":
        d = torch.cdist(source_pos, target_pos).nan_to_num(nan=torch.inf)
        n, ns, nt = d.shape
        kernel = torch.zeros_like(d)
        kernel.scatter_(1, d.argmin(dim=1, keepdim=True), 1)
    elif interpolation_method == "idw":
        kernel = idw_kernel(source_pos, target_pos)
    elif interpolation_method.startswith("rq"):
        try:
            alpha = interpolation_method.removeprefix("rq_")
            alpha = alpha.removeprefix("normalized_")
            alpha = float(alpha)
        except ValueError:
            alpha = 1.0
        kernel = rq_kernel(source_pos, target_pos, sigma, alpha)
        if "normalized" in interpolation_method:
            kernel /= kernel.sum(1, keepdim=True)
    else:
        kernel = log_rbf(source_pos, target_pos, sigma)
        if interpolation_method == "normalized":
            kernel = F.softmax(kernel, dim=1)
            kernel.nan_to_num_()
        elif interpolation_method.startswith("kriging"):
            kernel = kernel.exp_()
            if source_kernel_invs is None:
                sk = log_rbf(source_pos, sigma=sigma).exp_()
                kernel = torch.linalg.lstsq(sk.cpu(), kernel.cpu(), driver="gelsd")
                kernel = kernel.solution.to(features)
            else:
                kernel = source_kernel_invs @ kernel
                kernel = kernel.nan_to_num_()
            if interpolation_method == "kriging_normalized":
                kernel = kernel / kernel.sum(1, keepdim=True)
        elif interpolation_method == "rbf":
            kernel = kernel.exp_()
        else:
            assert False

    # -- apply kernel
    features = torch.nan_to_num(features, out=features if allow_destroy else None)
    features = torch.bmm(features, kernel, out=out)

    # nan-ify nonexistent chans
    needs_nan = torch.isnan(target_pos).all(2).unsqueeze(1)
    needs_nan = needs_nan.broadcast_to(features.shape)
    features[needs_nan] = torch.nan

    if thinplate:
        # finish up extrapolation business
        assert targ_extrap is not None
        assert features_in is not None
        features = torch.where(targ_extrap[:, None], features, features_in)

    return features


def idw_kernel(source_pos, target_pos=None):
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
    source_pos = source_pos
    if target_pos is None:
        target_pos = source_pos
    kernel = source_pos[:, :, None] - target_pos[:, None, :]
    kernel = kernel.square_().sum(dim=3).sqrt_().reciprocal_()
    kernel = kernel.nan_to_num_()
    kernel /= kernel.sum(dim=0)
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
    source_pos = source_pos / sigma
    if target_pos is None:
        target_pos = source_pos
    else:
        target_pos = target_pos / sigma
    kernel = source_pos[:, :, None] - target_pos[:, None, :]
    kernel = kernel.square_().sum(dim=3).div_(-2.0)
    kernel = kernel.nan_to_num_(nan=-torch.inf)
    return kernel


def rq_kernel(source_pos, target_pos=None, sigma=None, alpha=1.0):
    source_pos = source_pos / sigma
    if target_pos is None:
        target_pos = source_pos
    else:
        target_pos = target_pos / sigma
    kernel = source_pos[:, :, None] - target_pos[:, None, :]
    kernel = kernel.square_().sum(dim=3).nan_to_num_()
    kernel = kernel.add_(1.0).pow_(-alpha)
    return kernel


def thin_plate_greens(source_pos, target_pos=None, sigma=1.0):
    alpha = 1.0 / (8 * torch.pi)
    single = source_pos.ndim == 2
    if sigma != 1.0:
        source_pos = source_pos / sigma
    if target_pos is None:
        target_pos = source_pos
    elif sigma != 1.0:
        target_pos = target_pos / sigma
    if single:
        source_pos = source_pos[None]
        target_pos = target_pos[None]

    rsq = source_pos[:, :, None] - target_pos[:, None, :]
    rsq = rsq.square_().sum(dim=3)
    r = rsq.sqrt()

    kernel = alpha * torch.where(r < 1, r * torch.log(r ** r), rsq * r.log())

    if single:
        kernel = kernel[0]
    return kernel


def extrap_mask(source_pos, target_pos, eps=1e-3):
    """Only works for vertical shift."""
    source_x_uniq = source_pos[..., 0].unique()
    source_x_uniq = source_x_uniq[source_x_uniq.isfinite()]
    x_low, x_high = source_x_uniq.amin() - eps, source_x_uniq.amax() + eps

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

        col_mask = (targ_col_y > source_high[:, None]).logical_or_(targ_col_y < source_low[:, None])
        targ_extrap[targ_in_col] = targ_extrap[targ_in_col].logical_and_(col_mask[targ_in_col])

    targ_extrap = torch.logical_and(targ_extrap, targ_y.isfinite())
    return targ_extrap
