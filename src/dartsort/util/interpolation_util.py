"""Library for flavors of kernel interpolation and data interp utilities"""

import numpy as np
import torch
import torch.nn.functional as F
from dartsort.util.data_util import yield_masked_chunks
from dartsort.util.drift_util import (get_spike_pitch_shifts,
                                      static_channel_neighborhoods)

interp_kinds = (
    "nearest",
    "rbf",
    "normalized",
    "kriging",
    "kriging_normalized",
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
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dtype = torch.from_numpy(np.empty((), dtype=dataset.dtype)).dtype
    n_spikes = mask.sum()
    assert channels.shape == (n_spikes,)
    n_source_chans = channel_index.shape[1]
    n_target_chans = target_channels.shape[1]
    assert target_channels.shape == (n_spikes, n_target_chans)
    feature_dim = dataset.shape[1]
    assert channel_index.shape[1] == dataset.shape[2]

    # allocate output
    storage_device = device if store_on_device else "cpu"
    out_shape = n_spikes, feature_dim, n_target_chans
    out = torch.empty(out_shape, dtype=dtype, device=storage_device)

    # build data needed for interpolation
    source_geom = pad_geom(geom, dtype=dtype, device=device)
    target_geom = pad_geom(registered_geom, dtype=dtype, device=device)
    # here, shifts = reg_depths - depths
    shifts = torch.as_tensor(shifts, dtype=dtype).to(device)
    target_channels = torch.as_tensor(target_channels, device=device)
    channel_index = torch.as_tensor(channel_index, device=device)
    channels = torch.as_tensor(channels, device=device)

    # if needed, precompute kriging pinvs
    skis = None
    do_skis = interpolation_method.startswith("kriging")
    if do_skis:
        source_kernel_invs = get_source_kernel_pinvs(source_geom, channel_index, sigma=sigma)

    for ixs, chunk_features in yield_masked_chunks(
        mask, dataset, show_progress=show_progress, desc_prefix="Interpolating"
    ):
        # where are the spikes?
        source_channels = channel_index[channels[ixs]]
        source_shifts = shifts[ixs]
        if source_shifts.ndim == 1:
            # allows per-channel shifts
            source_shifts = source_shifts.unsqueeze(1)
        source_shifts = source_shifts.unsqueeze(-1)
        # used to shift the source, but for kriging it's better to shift targets
        # so that we can cache source kernel choleskys
        source_pos = source_geom[source_channels] # + source_shifts

        # where are they going?
        target_pos = target_geom[target_channels[ixs]] - source_shifts

        if do_skis:
            skis = source_kernel_invs[channels[ixs]]

        # interpolate, store
        chunk_features = torch.from_numpy(chunk_features).to(device)
        chunk_res = kernel_interpolate(
            chunk_features,
            source_pos,
            target_pos,
            source_kernel_invs=skis,
            sigma=sigma,
            allow_destroy=True,
            interpolation_method=interpolation_method,
        )
        out[ixs] = chunk_res.to(out)

    return out


def get_source_kernel_pinvs(source_geom, channel_index=None, sigma=20.0, atol=1e-5, rtol=1e-5):
    """channel_index None means we'll make one inv of the full probe."""
    if channel_index is None:
        source_pos = source_geom[None]
    else:
        source_pos = source_geom[channel_index]
    source_kernels = log_rbf(source_pos, sigma=sigma).exp_().nan_to_num_()
    pinvs = torch.linalg.pinv(source_kernels, atol=atol, rtol=rtol, hermitian=True)
    return pinvs


def pad_geom(geom, dtype=torch.float, device=None):
    geom = torch.as_tensor(geom, dtype=dtype, device=device)
    geom = F.pad(geom, (0, 0, 0, 1), value=torch.nan)
    return geom


def kernel_interpolate(
    features,
    source_pos,
    target_pos,
    source_kernel_invs=None,
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

    # -- build kernel
    if interpolation_method == "nearest":
        d = torch.cdist(source_pos, target_pos).nan_to_num(nan=torch.inf)
        n, ns, nt = d.shape
        kernel = torch.zeros_like(d)
        kernel.scatter_(1, d.argmin(dim=1, keepdim=True), 1)
    else:
        tic = time.perf_counter()
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

    return features


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
    if target_pos is None:
        target_pos = source_pos
    kernel = torch.cdist(source_pos, target_pos)
    kernel = kernel.square_().mul_(-1.0 / (2 * sigma**2))
    torch.nan_to_num(kernel, nan=-torch.inf, out=kernel)
    return kernel
