"""Localization with channel subsetting based on channel index
"""
import torch
import torch.nn.functional as F
from torch.optim import LBFGS, Adam

from .waveform_utils import channel_index_subset, binary_subset_to_relative

# how to initialize y?
Y0 = 20.0


def ptp_at(x, y, z, alpha, local_geoms):
    dxs = torch.square(x[:, None] - local_geoms[:, :, 0])
    dzs = torch.square(z[:, None] - local_geoms[:, :, 1])
    dys = torch.square(y[:, None])
    return alpha[:, None] / torch.sqrt(dxs + dzs + dys)


def find_alpha(nptps, nan_mask, x, y, z, local_geoms):
    qtqs = ptp_at(x, y, z, torch.ones_like(x), local_geoms)
    alpha = (qtqs * nptps).sum(1) / torch.square(qtqs * nan_mask).sum(1)
    return alpha


def mse(nptps, nan_mask, locs, local_geoms, logbarrier=True):
    x, y0, z = locs.T
    y = F.softplus(y0)
    alpha = find_alpha(nptps, nan_mask, x, y, z, local_geoms)
    ret = (
        torch.square(nptps - ptp_at(x, y, z, alpha, local_geoms)).mean(1).sum()
    )
    if logbarrier:
        ret += torch.log1p(10.0 * y).sum() / 10000.0
    return ret


def localize_ptps_index(
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
    channel_index_pad = F.pad(torch.as_tensor(channel_index), (0, 1, 0, 0), value=nc)
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

    # figure out which chans are outside the probe
    in_probe_channel_index = (channel_index < nc).to(torch.double)
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
    maxptps, _ = torch.max(torch.nan_to_num(ptps, -1.0), dim=1)
    nptps = ptps / maxptps[:, None]

    # -- torch optimize
    # initialize with center of mass
    locs = torch.column_stack((xcom, torch.full_like(xcom, Y0), zcom))
    with torch.enable_grad():
        locs.requires_grad_()

        # get optimizer
        if optimizer == "lbfgs":
            opt = LBFGS((locs,), max_iter=1000, line_search_fn="strong_wolfe")
        else:
            opt = Adam((locs,), 0.1)

        # build error closure
        def closure():
            opt.zero_grad()
            loss = mse(nptps, nan_mask, locs, local_geoms, logbarrier=logbarrier)
            loss.backward()
            return loss

        # run the thing
        if optimizer == "lbfgs":
            opt.step(closure)
        else:
            for _ in range(1000):
                opt.step(closure)

    # finish: get alpha closed form
    x, y0, z_rel = locs.T
    y = F.softplus(y0)
    alpha = find_alpha(ptps, nan_mask, x, y, z_rel, local_geoms)
    z_abs = z_rel + geom[maxchans, 1]

    return x, y, z_rel, z_abs, alpha
