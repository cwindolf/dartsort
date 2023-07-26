import torch
import torch.nn.functional as F

from .optutils import batched_levenberg_marquardt, batched_newton
from .waveform_utils import binary_subset_to_relative, channel_index_subset

have_vmap = False
try:
    from torch import vmap
    from torch.func import hessian, grad_and_value

    have_vmap = True
except ImportError:
    pass


def bptp_at(x, y, z, alpha, local_geoms):
    dxs = torch.square(x[:, None] - local_geoms[:, :, 0])
    dzs = torch.square(z[:, None] - local_geoms[:, :, 1])
    dys = torch.square(y[:, None])
    return alpha[:, None] / torch.sqrt(dxs + dzs + dys)


def bfind_alpha(nptps, nan_mask, x, y, z, local_geoms):
    qtqs = bptp_at(x, y, z, torch.ones_like(x), local_geoms)
    alpha = (qtqs * nptps).sum(1) / torch.square(qtqs * nan_mask).sum(1)
    return alpha


def bmse(locs, nptps, nan_mask, local_geoms, logbarrier=True):
    x, y0, z = locs.T
    y = F.softplus(y0)

    alpha = find_alpha(nptps, nan_mask, x, y, z, local_geoms)
    ret = (
        torch.square(nptps - ptp_at(x, y, z, alpha, local_geoms)).mean(1).sum()
    )
    if logbarrier:
        ret -= torch.log1p(10.0 * y).sum() / 10000.0
        # ret -= torch.log(1000.0 - torch.sqrt(torch.square(x) + torch.square(y) + torch.square(z))).sum()
    return ret


def ptp_at(x, y, z, alpha, local_geom):
    dxs = torch.square(x - local_geom[:, 0])
    dzs = torch.square(z - local_geom[:, 1])
    dys = torch.square(y)
    return alpha / torch.sqrt(dxs + dzs + dys)


def find_alpha(nptps, nan_mask, x, y, z, local_geoms):
    qtqs = ptp_at(x, y, z, 1.0, local_geoms)
    alpha = (qtqs * nptps).sum() / torch.square(qtqs * nan_mask).sum()
    return alpha


def mse(loc, nptp, nan_mask, local_geom, logbarrier=True):
    x, y0, z = loc
    y = F.softplus(y0)

    alpha = find_alpha(nptp, nan_mask, x, y, z, local_geom)
    ret = torch.square(nptp - ptp_at(x, y, z, alpha, local_geom)).mean()
    if logbarrier:
        ret -= torch.log(10.0 * y) / 10000.0
        # idea for logbarrier on points which run away
        # ret -= torch.log(1000.0 - torch.sqrt(torch.square(x) + torch.square(z))).sum() / 10000.0
    return ret


def localize_ptps_index_newton(
    ptps,
    geom,
    maxchans,
    channel_index,
    n_channels=None,
    radius=None,
    logbarrier=True,
    model="pointsource",
    dtype=torch.double,
    optimizer="lbfgs",
    y0=10.0,
    max_steps=100,
    convergence_x=1e-6,
    convergence_g=1e-6,
    wolfe_c1=1e-4,
    wolfe_c2=0.1,
    max_ls=25,
    tikhonov=0.01,
    method="cholesky",
):
    """Localize a bunch of PTPs with torch

    Returns
    -------
    xs, ys, z_rels, z_abss, alphas
    """
    N, C = ptps.shape

    # handle channel subsetting
    nc = len(channel_index)
    subset = channel_index_subset(
        geom, channel_index, n_channels=n_channels, radius=radius
    )
    subset = binary_subset_to_relative(subset)
    channel_index_pad = F.pad(
        torch.as_tensor(channel_index), (0, 1, 0, 0), value=nc
    )
    channel_index = channel_index_pad[torch.arange(nc)[:, None], subset]
    # pad with 0s rather than nans, we will mask below.
    ptps = F.pad(ptps, (0, 1, 0, 0))[
        torch.arange(N)[:, None], subset[maxchans]
    ]

    # torch everyone
    device = ptps.device
    ptps = torch.as_tensor(ptps, dtype=dtype, device=device)
    geom = torch.as_tensor(geom, dtype=dtype, device=device)
    channel_index = torch.as_tensor(channel_index, device=device)

    # nan to num to avoid some masking
    ptps = torch.nan_to_num(ptps)

    # figure out which chans are outside the probe
    in_probe_channel_index = (channel_index < nc).to(dtype)
    nan_mask = in_probe_channel_index[maxchans]

    # local geometries in each ptp
    geom_pad = F.pad(geom, (0, 0, 0, 1))
    local_geoms = geom_pad[channel_index[maxchans]]
    local_geoms[:, :, 1] -= geom[maxchans, 1][:, None]

    # center of mass initialization
    com = (ptps[:, :, None] * local_geoms).sum(1) / ptps.sum(1)[:, None]
    xcom, zcom = com.T

    if model == "com":
        z_abs_com = zcom + geom[maxchans, 1]
        nancom = torch.full_like(xcom, torch.nan)
        return xcom, nancom, zcom, z_abs_com, nancom
    else:
        assert model == "pointsource"

    # normalized PTP vectors
    maxptps, _ = torch.max(ptps, dim=1)
    nptps = ptps / maxptps[:, None]

    # -- torch optimize
    # initialize with center of mass
    xcom = xcom.cpu()
    zcom = zcom.cpu()
    ptps = ptps.cpu()
    geom = geom.cpu()
    nptps, nan_mask, local_geoms = (
        nptps.cpu(),
        nan_mask.cpu(),
        local_geoms.cpu(),
    )
    locs = torch.column_stack((xcom, torch.full_like(xcom, y0), zcom))
    locs, nevals, i = batched_newton(
        locs,
        vgrad_and_func,
        vhess,
        extra_args=(nptps, nan_mask, local_geoms),
        max_steps=max_steps,
        convergence_x=convergence_x,
        convergence_g=convergence_g,
        tikhonov=tikhonov,
        method=method,
        max_ls=max_ls,
        wolfe_c1=wolfe_c1,
        wolfe_c2=wolfe_c2,
    )

    # finish: get alpha closed form
    x, y0, z_rel = locs.T
    y = F.softplus(y0)
    alpha = bfind_alpha(ptps, nan_mask, x, y, z_rel, local_geoms)
    z_abs = z_rel + geom[maxchans, 1]

    return x, y, z_rel, z_abs, alpha


def localize_ptps_index_lm(
    ptps,
    geom,
    maxchans,
    channel_index,
    n_channels=None,
    radius=None,
    logbarrier=True,
    model="pointsource",
    dtype=torch.double,
    y0=1.0,
    max_steps=250,
    convergence_err=1e-10,
    convergence_g=1e-10,
    scale_problem="hessian",
    lambd=100.0,
    nu=10.0,
    min_scale=1e-2,
):
    """Localize a bunch of PTPs with torch

    Returns
    -------
    xs, ys, z_rels, z_abss, alphas
    """
    N, C = ptps.shape

    # handle channel subsetting
    nc = len(channel_index)
    subset = channel_index_subset(
        geom, channel_index, n_channels=n_channels, radius=radius
    )
    subset = torch.as_tensor(binary_subset_to_relative(subset), device=ptps.device)
    channel_index_pad = F.pad(
        torch.as_tensor(channel_index, device=ptps.device), (0, 1, 0, 0), value=nc
    )
    channel_index = channel_index_pad[torch.arange(nc)[:, None], subset]
    # pad with 0s rather than nans, we will mask below.
    ptps = F.pad(ptps, (0, 1, 0, 0))[
        torch.arange(N)[:, None], subset[maxchans]
    ]

    # torch everyone
    device = ptps.device
    ptps = torch.as_tensor(ptps, dtype=dtype, device=device)
    geom = torch.as_tensor(geom, dtype=dtype, device=device)
    channel_index = torch.as_tensor(channel_index, device=device)

    # nan to num to avoid some masking
    ptps = torch.nan_to_num(ptps)

    # figure out which chans are outside the probe
    in_probe_channel_index = (channel_index < nc).to(dtype)
    nan_mask = in_probe_channel_index[maxchans]

    # local geometries in each ptp
    geom_pad = F.pad(geom, (0, 0, 0, 1))
    local_geoms = geom_pad[channel_index[maxchans]]
    local_geoms[:, :, 1] -= geom[maxchans, 1][:, None]

    # center of mass initialization
    com = (ptps[:, :, None] * local_geoms).sum(1) / ptps.sum(1)[:, None]
    xcom, zcom = com.T

    if model == "com":
        z_abs_com = zcom + geom[maxchans, 1]
        nancom = torch.full_like(xcom, torch.nan)
        return xcom, nancom, zcom, z_abs_com, nancom
    else:
        assert model == "pointsource"

    # normalized PTP vectors
    maxptps, _ = torch.max(ptps, dim=1)
    nptps = ptps / maxptps[:, None]

    # -- torch optimize
    # initialize with center of mass
    locs = torch.column_stack((xcom, torch.full_like(xcom, y0), zcom))
    locs, i = batched_levenberg_marquardt(
        locs,
        vgrad_and_func,
        vhess,
        extra_args=(nptps, nan_mask, local_geoms),
        max_steps=max_steps,
        convergence_err=convergence_err,
        convergence_g=convergence_g,
        scale_problem=scale_problem,
        nu=nu,
        lambd=lambd,
        min_scale=min_scale,
    )

    # finish: get alpha closed form
    x, y0, z_rel = locs.T
    y = F.softplus(y0)
    alpha = bfind_alpha(ptps, nan_mask, x, y, z_rel, local_geoms)
    z_abs = z_rel + geom[maxchans, 1]

    return x, y, z_rel, z_abs, alpha


# -- a pytorch impl of batched newton method

if have_vmap:
    vgrad_and_func = vmap(grad_and_value(mse))
    vhess = vmap(hessian(mse))
