import torch
import torch.nn.functional as F
from dartsort.util.torch_optimization_util import batched_levenberg_marquardt
from dartsort.util.waveform_util import (channel_subset_by_radius,
                                         full_channel_index)
from torch import vmap
from torch.func import grad_and_value, hessian


def localize_amplitude_vectors(
    amplitude_vectors,
    geom,
    main_channels,
    channel_index=None,
    radius=None,
    n_channels_subset=None,
    logbarrier=True,
    model="pointsource",
    dtype=torch.double,
    y0=1.0,
    levenberg_marquardt_kwargs=None,
    th_dipole_proj_dist=250.0,
):
    """Localize a bunch of amplitude vectors with torch

    Here, amplitude_vectors is a tensor with shape (n_spikes, c), where
    n_spikes is the number of spikes and c is the number of channels each
    spike was extracted on, according to the channel_index structure
    (which has shape (n_channels_tot, c), matching geom's
    (n_channels_tot, 2) shape). main_channels is the detection
    channel for each spike, and spike i lives on the channels in
    channel_index[main_channels[i]]. Some spikes don't occupy c channels
    when the channel neighborhoods in channel_index are for instance
    determined by a distance threshold, in which case the values
    in channel_index[main_channels[i]] will be n_channels_tot for
    such cases and the values in amplitude_vectors may be nan or
    another filler value.

    Localization is run on a subset of channels determined by `radius`
    (or `n_channels_subset`), so that it is possible to compute
    amplitude vectors on a larger channel neighborhood than you would
    want to trust your localization model over.

    Arguments
    ---------
    amplitude_vectors : tensor, shape (n_spikes, c)
    geom : tensor, shape (n_channels_tot, 2)
    main_channels : LongTensor, shape (n_spikes,)
    channel_index : LongTensor, shape (n_channels_tot, c)
    radius : float
        Only consider channels closer than this number (in microns)
        to geom[main_channels[i]] for amplitude_vectors[i]

    Returns
    -------
    dictionary with keys: x, y, z_rel, z_abs, alpha
        All (n_spikes,) tensors
    """
    # only monopole implemented in torch for now
    # maybe this will become a wrapper function if we want more models.
    # and, this is why we return a dict, different models will have different
    # parameters
    assert model in ("com", "pointsource", "dipole")
    n_spikes, c = amplitude_vectors.shape
    n_channels_tot = len(geom)
    if channel_index is None:
        assert c == n_channels_tot
        channel_index = full_channel_index(n_channels_tot)
    assert channel_index.shape == (n_channels_tot, c)
    assert main_channels.shape == (n_spikes,)
    # we'll return numpy if user sent numpy
    is_numpy = not torch.is_tensor(amplitude_vectors)

    # handle channel subsetting
    if radius is not None or n_channels_subset is not None:
        amplitude_vectors, channel_index = channel_subset_by_radius(
            amplitude_vectors,
            main_channels,
            channel_index,
            geom,
            radius=radius,
            n_channels_subset=n_channels_subset,
        )

    # torch everyone
    amplitude_vectors = torch.as_tensor(
        amplitude_vectors,
        dtype=dtype,
    )
    device = amplitude_vectors.device
    geom = torch.as_tensor(geom, dtype=dtype, device=device)
    channel_index = torch.as_tensor(channel_index, device=device)

    # nan to num to avoid some masking
    amplitude_vectors = torch.nan_to_num(amplitude_vectors)

    # figure out which chans are outside the probe
    # so that we can avoid counting them in optimization
    in_probe_channel_index = (channel_index < n_channels_tot).to(dtype)
    in_probe_mask = in_probe_channel_index[main_channels]

    # local geometries of each amplitude vector
    geom_pad = F.pad(geom, (0, 0, 0, 1))
    local_geoms = geom_pad[channel_index[main_channels]]
    local_geoms[:, :, 1] -= geom[main_channels, 1][:, None]

    # center of mass initialization
    com = torch.divide(
        (amplitude_vectors[:, :, None] * local_geoms).sum(1),
        amplitude_vectors.sum(1)[:, None],
    )
    xcom, zcom = com.T

    if model == "com":
        z_abs_com = zcom + geom[main_channels, 1]
        return dict(x=xcom, z_rel=zcom, z_abs=z_abs_com)

    # normalized PTP vectors
    # this helps to keep the objective in a similar range, so we can use
    # fixed constants in regularizers like the log barrier
    max_amplitudes = torch.max(amplitude_vectors, dim=1).values
    normalized_amp_vecs = amplitude_vectors / max_amplitudes[:, None]

    # -- torch optimize
    if levenberg_marquardt_kwargs is None:
        levenberg_marquardt_kwargs = {}

    # initialize with center of mass
    locs = torch.column_stack((xcom, torch.full_like(xcom, y0), zcom))

    if model == "pointsource":
        locs, i = batched_levenberg_marquardt(
            locs,
            vmap_point_source_grad_and_mse,
            vmap_point_source_hessian,
            extra_args=(normalized_amp_vecs, in_probe_mask, local_geoms),
            **levenberg_marquardt_kwargs,
        )

        # finish: get alpha closed form
        x, y0, z_rel = locs.T
        y = F.softplus(y0)
        alpha = vmap_point_source_find_alpha(
            amplitude_vectors, in_probe_mask, x, y, z_rel, local_geoms
        )
        z_abs = z_rel + geom[main_channels, 1]

        results = dict(x=x, y=y, z_rel=z_rel, z_abs=z_abs, alpha=alpha)
        if is_numpy:
            results = {k: v.numpy(force=True) for k, v in results.items()}
        return results

    elif model == "dipole":
        locs, i = batched_levenberg_marquardt(
            locs,
            vmap_dipole_grad_and_mse,
            vmap_dipole_hessian,
            extra_args=(normalized_amp_vecs, local_geoms),
            **levenberg_marquardt_kwargs,
        )

        x, y0, z_rel = locs.T
        y = F.softplus(y0)
        projected_dist = vmap_dipole_find_projection_distance(
            normalized_amp_vecs, x, y, z_rel, local_geoms
        )     

        z_abs = z_rel + geom[main_channels, 1]

        if is_numpy:
            x = x.numpy(force=True)
            y = y.numpy(force=True)
            z_rel = z_rel.numpy(force=True)
            z_abs = z_abs.numpy(force=True)
            projected_dist = projected_dist.numpy(force=True)

        return dict(x=x, y=y, z_rel=z_rel, z_abs=z_abs, alpha=projected_dist)

    else:
        assert False


# -- point source / dipole model library functions
def point_source_amplitude_at(x, y, z, alpha, local_geom):
    """Point source model predicted amplitude at local_geom given location"""
    dxs = torch.square(x - local_geom[:, 0])
    dzs = torch.square(z - local_geom[:, 1])
    dys = torch.square(y)
    return alpha / torch.sqrt(dxs + dzs + dys)


def point_source_find_alpha(amp_vec, channel_mask, x, y, z, local_geoms):
    """We can solve for the brightness (alpha) of the source in closed form given x,y,z"""
    amp1_vec = point_source_amplitude_at(x, y, z, 1.0, local_geoms)
    alpha = torch.divide(
        (amp1_vec * amp_vec * channel_mask).sum(),
        torch.square(amp1_vec * channel_mask).sum(),
    )
    return alpha

def dipole_find_projection_distance(normalized_amp_vec, x, y, z, local_geom):
    """COmpute a value dist/dipole in x,z that tells us if dipole or monopole is better"""
    
    dxs = x - local_geom[:, 0]
    dzs = z - local_geom[:, 1]
    dys = y.expand(dzs.size())
    duv = torch.stack([dxs, dys, dzs], dim=1)
    X = duv / torch.pow(torch.sum(torch.square(duv), dim=1), 3/2)[:, None]
    beta = torch.matmul(torch.linalg.pinv(torch.matmul(X.T, X)), torch.matmul(X.T, normalized_amp_vec))
    beta /= torch.sqrt(torch.square(beta).sum())
    dipole_planar_direction = torch.sqrt(torch.square(beta[[0, 2]]).sum())
    closest_chan = torch.argmin(torch.sum(torch.square(duv), dim=1))
    min_duv = duv[closest_chan[None]][0] #workaround around vmap doesn't work for one dim tensor .item()
    val_th = torch.sqrt(torch.square(min_duv).sum())/dipole_planar_direction
    return val_th

def point_source_mse(loc, amplitude_vector, channel_mask, local_geom, logbarrier=True):
    """Objective in point source model

    Arguments
    ---------
    loc : tensor of shape (3,)
        Here, this is the x, y0, z positions, where y = softplus(y0)
    amplitude_vector : tensor of shape (n_chans,)
    channel_mask : tensor of shape (n_chans,)
        A binary mask. Channels with 0s here are excluded from the
        objective, useful for amplitudes from waveforms extracted
        on a sparse/incomplete set of channels.
    local_geom : tensor of shape (n_chans, 2)
    logbarrier : bool

    Returns
    -------
    obj : scalar
        The objective, to be minimized
    """
    x, y0, z = loc
    y = F.softplus(y0)

    alpha = point_source_find_alpha(amplitude_vector, channel_mask, x, y, z, local_geom)
    obj = torch.square(
        amplitude_vector - point_source_amplitude_at(x, y, z, alpha, local_geom)
    ).mean()
    if logbarrier:
        obj -= torch.log(10.0 * y) / 10000.0
        # idea for logbarrier on points which run away
        # obj -= torch.log(1000.0 - torch.sqrt(torch.square(x) + torch.square(z))).sum() / 10000.0
    return obj


def dipole_mse(loc, amplitude_vector, local_geom, logbarrier=True):
    """Dipole model predicted amplitude at local_geom given location"""

    x, y0, z = loc
    y = F.softplus(y0)

    dxs = x - local_geom[:, 0]
    dzs = z - local_geom[:, 1]
    dys =  y.expand(dzs.size())
    
    duv = torch.stack([dxs, dys, dzs], dim=1)
    X = duv / (torch.pow(torch.sum(torch.square(duv), dim=1), 3/2)[:, None]) #Add 1e-4?
    # beta = torch.linalg.lstsq(X, amplitude_vector[:, None])[0]
    # beta = torch.linalg.solve(torch.matmul(X.T, X), torch.matmul(X.T, amplitude_vector))
    beta = torch.matmul(torch.linalg.pinv(torch.matmul(X.T, X)), torch.matmul(X.T, amplitude_vector))
    qtq = torch.matmul(X, beta)
    
    obj = torch.square(amplitude_vector - qtq).mean()
    if logbarrier:
        obj -= torch.log(10.0 * y) / 10000.0

    return obj


# vmapped functions for use in the optimizer, and might be handy for users too
vmap_point_source_grad_and_mse = vmap(grad_and_value(point_source_mse))
vmap_point_source_hessian = vmap(hessian(point_source_mse))
vmap_point_source_find_alpha = vmap(point_source_find_alpha)

vmap_dipole_grad_and_mse = vmap(grad_and_value(dipole_mse))
vmap_dipole_hessian = vmap(hessian(dipole_mse))
vmap_dipole_find_projection_distance = vmap(dipole_find_projection_distance)
