import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors, cm, ticker
import scipy.linalg as la
from scipy import signal
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import colorcet
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from tqdm.auto import trange, tqdm
from itertools import repeat

from . import statistics, waveform_utils, localization, point_source_centering
from .point_source_centering import relocate_simple


sns.set_style("ticks")


darkpurple = plt.cm.Purples(0.99)
purple = plt.cm.Purples(0.75)
lightpurple = plt.cm.Purples(0.5)
darkgreen = plt.cm.Greens(0.99)
green = plt.cm.Greens(0.75)
lightgreen = plt.cm.Greens(0.5)


class MidpointNormalize(colors.Normalize):
    # class from the mpl docs:
    # https://matplotlib.org/users/colormapnorms.html

    def __init__(self, vmin=None, vmax=None, midpoint=0.0, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        out = np.ma.masked_array(np.interp(value, x, y))
        return out


def normbatch(batch):
    batch = batch - batch.min(axis=(1, 2), keepdims=True)
    return batch / batch.max(axis=(1, 2), keepdims=True)


@torch.no_grad()
def mosaic(xs, pad=0, padval=255, vmin=np.inf, vmax=-np.inf):
    assert all(x.shape == xs[0].shape for x in xs)
    nrows = len(xs)
    B, H, W = xs[0].shape
    grid = torch.stack(list(map(torch.as_tensor, xs)), dim=0)  # nrowsBHW
    vmin = min(grid.min(), vmin)
    grid -= vmin
    grid *= 255.0 / max(grid.max(), vmax - vmin)
    grid = grid.to(torch.uint8)
    if pad > 0:
        grid = F.pad(grid, (pad, pad, pad, pad), value=padval)
    grid = grid.permute(0, 2, 1, 3).reshape(
        (H + 2 * pad) * nrows, B * (W + 2 * pad), 1
    )
    grid = grid[pad:-pad, pad:-pad, :]
    return grid


def tukey_scatter(x, y, iqrs=1.5, ax=None, **kwargs):
    ax = ax or plt.gca()
    x_25, x_75 = np.percentile(x, [25, 75])
    x_iqr = x_75 - x_25
    y_25, y_75 = np.percentile(x, [25, 75])
    y_iqr = y_75 - y_25
    inliers = np.flatnonzero(
        (x_25 - iqrs * x_iqr < x)
        & (x < x_75 + iqrs * x_iqr)
        & (y_25 - iqrs * y_iqr < y)
        & (y < y_75 + iqrs * y_iqr)
    )
    return ax.scatter(x[inliers], y[inliers], **kwargs)


def labeledmosaic(
    xs,
    rowlabels=None,
    pad=1,
    padval=255,
    ax=None,
    cbar=True,
    separate_norm=False,
    collabels="abcdefghijklmnopqrstuvwxyz",
    vmin=None,
    vmax=None,
):
    if rowlabels is None:
        rowlabels = range(1, len(xs) + 1)

    if separate_norm:
        assert not cbar
        xs = [normbatch(x) for x in xs]

    if vmin is None:
        gvmin = np.inf
        vmin = min(x.min() for x in xs)
    else:
        gvmin = vmin
        assert all((x >= vmin).all() for x in xs)

    if vmax is None:
        gvmax = -np.inf
        vmax = max(x.max() for x in xs)
    else:
        gvmax = vmax
        assert all((x <= vmax).all() for x in xs)

    B, H, W = xs[0].shape
    grid = mosaic(xs, pad=pad, padval=padval, vmin=gvmin, vmax=gvmax).numpy()
    grid = np.pad(grid, [(14, 0), (24, 0), (0, 0)], constant_values=padval)
    ax = ax or plt.gca()
    ax.imshow(
        np.broadcast_to(grid, (*grid.shape[:2], 3)),
        interpolation="nearest",
    )
    ax.axis("off")

    for i, label in enumerate(rowlabels):
        ax.text(
            8,
            i * (H + 2 * pad) + H / 2 + 14,
            label,
            rotation="vertical",
            ha="center",
            va="center",
        )

    for b in range(B):
        ax.text(
            24 + b * (W + 2 * pad) + W / 2,
            4,
            collabels[b],
            ha="center",
            va="center",
            fontsize=8,
        )

    if cbar:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05)
        cbar = plt.colorbar(cm.ScalarMappable(norm, "gray"), cax)
        ticks = [x for x in range(-50, 50, 5) if vmin <= x <= vmax]
        if len(ticks) > 1:
            cbar.set_ticks(ticks)
            cbar.ax.set_yticklabels(ticks, fontsize=8)


def plot_single_ptp_np2(ptp, ax, label, color, code):
    ptp_left = ptp[::2]
    ptp_right = ptp[1::2]
    (handle,) = ax.plot(ptp_left, c=color, label=label)
    (dhandle,) = ax.plot(ptp_right, "--", c=color)
    if code:
        ax.text(
            0.1,
            0.9,
            code,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
    ax.set_xticks([0, ptp.shape[0] // 2 - 1])
    return handle, dhandle


def plot_single_ptp_np1(ptp, ax, label, color, code):
    ptp_a = ptp[::4]
    ptp_b = ptp[1::4]
    ptp_c = ptp[2::4]
    ptp_d = ptp[3::4]
    (handle,) = ax.plot(ptp_a, c=color, label=label)
    (dhandle,) = ax.plot(ptp_b, c=color)
    (handle,) = ax.plot(ptp_c, c=color, label=label)
    (dhandle,) = ax.plot(ptp_d, c=color)
    if code:
        ax.text(
            0.1,
            0.9,
            code,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
    ax.set_xticks([0, ptp.shape[0] // 4 - 1])
    return handle, dhandle


def regline(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    b = ((x - x.mean()) * (y - y.mean())).sum() / np.square(x - x.mean()).sum()
    a = y.mean() - b * x.mean()
    x0, x1 = ax.get_xlim()
    r = np.corrcoef(x, y)[0, 1]
    ax.plot([x0, x1], [a + b * x0, a + b * x1], lw=1, color="red")
    ax.text(
        kwargs.get("rloc", 0.9),
        0.8,
        f"$\\rho={r:.2f}$",
        horizontalalignment="right",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize=6,
        color="red",
        backgroundcolor=[1, 1, 1, 0.75],
    )


def corr_scatter(
    xs,
    ys,
    xlabels,
    ylabels,
    colors,
    alphas,
    suptitle=None,
    grid=True,
    rloc=0.9,
    axes=None,
    do_aspect=True,
):
    nxs = xs.shape[1]
    nys = ys.shape[1] if grid else 1
    if axes is None:
        fig, axes = plt.subplots(
            nys,
            nxs,
            sharey="row" if grid else False,
            sharex="col",
            figsize=(6, 6) if grid else (7, 3),
        )

    if grid:
        for i in range(nxs):
            for j in range(nys):
                axes[j, i].scatter(
                    xs[:, i],
                    ys[:, j],
                    c=colors,
                    alpha=alphas,
                    s=0.1,
                    cmap=plt.cm.viridis,
                )
                if i == 0:
                    axes[j, i].set_ylabel(ylabels[j], fontsize=8)
                if j == nys - 1:
                    axes[j, i].set_xlabel(xlabels[i], fontsize=8)
                if do_aspect:
                    axes[j, i].set_box_aspect(1)
                regline(xs[:, i], ys[:, j], ax=axes[j, i], rloc=rloc)
        if suptitle:
            fig.suptitle(suptitle)
    else:
        for i in range(nxs):
            axes[i].scatter(
                xs[:, i],
                ys[:, i],
                c=colors,
                alpha=alphas,
                s=0.1,
                cmap=plt.cm.viridis,
            )
            axes[i].set_ylabel(ylabels[i], fontsize=8)
            axes[i].set_xlabel(xlabels[i], fontsize=8)
            axes[i].set_box_aspect(1)
            regline(xs[:, i], ys[:, i], ax=axes[i], rloc=rloc)
        if suptitle:
            fig.suptitle(suptitle, y=0.95)
    plt.tight_layout()


def plotlocs(
    x,
    y,
    z,
    alpha,
    maxptps,
    geom,
    feats=None,
    xlim=None,
    ylim=None,
    alim=None,
    zlim=None,
    which=slice(None),
    clip=True,
    suptitle=None,
    figsize=(8, 8),
    gs=1,
    cm=plt.cm.viridis,
):
    """Localization scatter plot

    Plots localizations (x, z, log y, log alpha) against the probe geometry,
    using max PTP to color the points.

    Arguments
    ---------
    x, y, z, alpha, maxptps : 1D np arrays of the same shape
    geom : n_channels x 2
    feats : optional, additional features to scatter
    *lim: optional axes lims if the default looks weird
    which : anything which can index x,y,z,alpha
        A subset of spikes to plot
    clip : bool
        If true (default), clip maxptps to the range 3-13 when coloring
        spikes
    """
    maxptps = maxptps[which]
    nmaxptps = 0.1
    cmaxptps = maxptps
    if clip:
        nmaxptps = 0.25 + 0.74 * (maxptps - maxptps.min()) / (
            maxptps.max() - maxptps.min()
        )
        cmaxptps = np.clip(maxptps, 3, 13)

    x = x[which]
    y = y[which]
    alpha = alpha[which]
    z = z[which]
    nfeats = 0
    if feats is not None:
        nfeats = feats.shape[1]

    fig, axes = plt.subplots(1, 3 + nfeats, sharey=True, figsize=figsize)
    aa, ab, ac = axes[:3]
    aa.scatter(x, z, s=0.1, alpha=nmaxptps, c=cmaxptps, cmap=cm)
    aa.scatter(geom[:, 0], geom[:, 1], color="orange", marker="s", s=gs)
    logy = np.log(y)
    ab.scatter(logy, z, s=0.1, alpha=nmaxptps, c=cmaxptps, cmap=cm)
    loga = np.log(alpha)
    ac.scatter(loga, z, s=0.1, alpha=nmaxptps, c=cmaxptps, cmap=cm)
    aa.set_ylabel("z")
    aa.set_xlabel("x")
    ab.set_xlabel("$\\log y$")
    ac.set_xlabel("$\\log \\alpha$")
    if xlim is None:
        aa.set_xlim(np.percentile(x, [0, 100]) + [-10, 10])
    else:
        aa.set_xlim(xlim)
    if ylim is None:
        ab.set_xlim(np.percentile(logy, [0, 100]))
    else:
        ab.set_xlim(ylim)
    # ab.set_xlim([-0.5, 6])
    if alim is None:
        ac.set_xlim(np.percentile(loga, [0, 100]))
    else:
        ac.set_xlim(alim)

    if suptitle:
        fig.suptitle(suptitle, y=0.95)

    if feats is not None:
        for ax, f in zip(axes[3:], feats.T):
            ax.scatter(f[which], z, s=0.1, alpha=nmaxptps, c=cmaxptps, cmap=cm)
            ax.set_xlim(np.percentile(f, [0, 100]))

    if zlim is None:
        aa.set_ylim([z.min() - 10, z.max() + 10])
    else:
        aa.set_ylim(zlim)
    
    return fig


def gplot(
    waveform,
    maxchan,
    channel_index,
    geom,
    yscale=None,
    xscale=0.9,
    trough=42,
    ax=None,
    color="k",
    lw=1,
    ls="-",
    pad=0.1,
    label=None,
    vline=False,
    subar=False,
):
    """Scale is in units of inter-channel Z dists
    """
    if ax is None:
        ax = plt.gca()
    
    if yscale is None:
        yscale = 1 / waveform.ptp(0).max()
        
    gscalex = np.abs(geom[1, 0] - geom[0, 0])
    gscaley = geom[2, 1] - geom[0, 1]
    waveform = yscale * gscaley * waveform
    
    domain = (xscale * gscalex / waveform.shape[0]) * np.arange(
        -trough,
        -trough + waveform.shape[0],
    )
    
    lines = []
    ext = (np.inf, -np.inf)
    lox = np.inf
    loy = np.inf
    for trace, chan in zip(waveform.T, channel_index[maxchan]):
        lines.append(domain + geom[chan, 0])
        lox = min(lox, lines[-1].min())
        trace = trace + geom[chan, 1]
        loy = min(loy, geom[chan, 1])
        ext = (min(ext[0], trace.min()), max(ext[1], trace.max()))
        lines.append(trace)
        if vline:
            ax.axvline(domain[trough] + geom[chan, 0], c="silver", lw=1)
    
    p = ax.plot(*lines, color=color, lw=lw, label=label, ls=ls)
    ax.set_ylim([ext[0] - pad * gscaley, ext[1] + pad * gscaley])
    
    if subar:
        sp = ax.plot([lox + gscalex / 4, lox + gscalex / 4], [loy, loy + 1 * yscale * gscaley], color="k", lw=2, solid_capstyle="butt")
        return p, sp
    
    return p


def plot_ptp(ptp, axes, label, color, codes):
    for j, ax in enumerate(axes.flat):
        handle, dhandle = plot_single_ptp_np2(
            ptp[j], ax, label, color, codes[j]
        )
    return handle, dhandle


def vis_ptps(
    ptps,
    labels=repeat(""),
    colors=repeat("black"),
    subplots_kwargs=dict(sharex=True, sharey=True, figsize=(5, 5)),
    codes="abcdefghijklmnopqrstuvwxyz",
    legloc="upper center",
    legend=True,
):
    ptps = np.array([np.array(ptp) for ptp in ptps])
    K, N, C = ptps.shape
    # assert len(labels) == K == len(colors)
    n = int(np.sqrt(N))

    fig, axes = plt.subplots(n, n, **subplots_kwargs)
    handles = {}
    dhandles = {}
    for k, (ptp, label, color) in enumerate(zip(ptps, labels, colors)):
        handles[k], dhandles[k] = plot_ptp(ptp, axes, label, color, codes)

    if legend:
        if K == 1:
            plt.figlegend(
                handles=list(handles.values()) + list(dhandles.values()),
                labels=[label + ", left channels", label + ", right channels"],
                loc=legloc,
                frameon=False,
                fancybox=False,
                borderpad=0,
                borderaxespad=0,
                ncol=2,
            )
        else:
            plt.figlegend(
                handles=list(handles.values()),
                labels=labels,
                loc=legloc,
                frameon=False,
                fancybox=False,
                borderpad=0,
                borderaxespad=0,
                ncol=len(handles),
            )
    plt.tight_layout(pad=0.5)
    for ax in axes.flat:
        ax.set_box_aspect(1.0)
    return fig, axes


def locrelocplots(
    h5, wf_key="denoised_waveforms", name="", seed=0, threshold=6.0
):
    rg = np.random.default_rng(seed)
    big = np.flatnonzero(h5["maxptp"][:] > threshold)
    inds = rg.choice(big, size=8)
    inds.sort()
    wfs = h5[wf_key][inds]
    orig_ptp = wfs.ptp(1)
    _, _, C = wfs.shape
    if (C // 2) % 2:
        geomkind = (
            "firstchanstandard" if "first_channels" in h5 else "standard"
        )
    else:
        geomkind = "firstchan" if "first_channels" in h5 else "updown"

    wfs_reloc_yza, stereo_ptp_yza, pred_ptp = relocate_simple(
        wfs,
        h5["geom"][:],
        h5["max_channels"][inds],
        h5["x"][inds],
        h5["y"][inds],
        h5["z_rel"][inds],
        h5["alpha"][inds],
        channel_radius=(C // 2) - (C // 2) % 2,
        firstchans=h5["first_channels"][inds]
        if "first_channels" in h5
        else None,
        geomkind=geomkind,
        relocate_dims="yza",
    )
    wfs_reloc_yza = wfs_reloc_yza.numpy()
    stereo_ptp_yza = stereo_ptp_yza.numpy()
    pred_ptp = pred_ptp.numpy()

    wfs_reloc_xyza, stereo_ptp_xyza, pred_ptp_ = relocate_simple(
        wfs,
        h5["geom"][:],
        h5["max_channels"][inds],
        h5["x"][inds],
        h5["y"][inds],
        h5["z_rel"][inds],
        h5["alpha"][inds],
        channel_radius=(C // 2) - (C // 2) % 2,
        firstchans=h5["first_channels"][inds]
        if "first_channels" in h5
        else None,
        geomkind=geomkind,
        relocate_dims="xyza",
    )
    wfs_reloc_xyza = wfs_reloc_xyza.numpy()
    stereo_ptp_xyza = stereo_ptp_xyza.numpy()
    pred_ptp_ = pred_ptp_.numpy()

    assert np.all(pred_ptp_ == pred_ptp)

    yza_ptp = wfs_reloc_yza.ptp(1)
    xyza_ptp = wfs_reloc_xyza.ptp(1)

    codes = "abcdefghij"
    fig, axes = plt.subplots(
        4, 3 * 2, figsize=(6, 4), sharex=True, sharey=True
    )
    la = "observed ptp"
    laa = "predicted ptp"
    lb = "$yz\\alpha$ relocated ptp"
    lbb = "$yz\\alpha$ standard ptp"
    lc = "$xyz\\alpha$ relocated ptp"
    lcc = "$xyz\\alpha$ standard ptp"
    ha, _ = plot_ptp(orig_ptp, axes[:, :2], la, "black", codes)
    haa, _ = plot_ptp(pred_ptp, axes[:, :2], laa, "silver", codes)
    hb, _ = plot_ptp(yza_ptp, axes[:, 2:4], lb, darkgreen, codes)
    hbb, _ = plot_ptp(stereo_ptp_yza, axes[:, 2:4], lbb, lightgreen, codes)
    hc, _ = plot_ptp(xyza_ptp, axes[:, 4:], lc, darkpurple, codes)
    hcc, _ = plot_ptp(stereo_ptp_xyza, axes[:, 4:], lcc, lightpurple, codes)

    fig.legend(
        # handles=[ha, hb, hc, haa, hbb, hcc],
        handles=[ha, haa, hb, hbb, hc, hcc],
        # labels=[la, lb, lc, laa, lbb, lcc],
        labels=[la, laa, lb, lbb, lc, lcc],
        loc="lower center",
        frameon=False,
        fancybox=False,
        borderpad=0,
        borderaxespad=0,
        ncol=3,
    )
    fig.suptitle(
        f"{name} PTPs before/after relocation ($yz\\alpha$, $xyz\\alpha$)",
        y=0.95,
    )
    # plt.tight_layout(pad=0.1)
    for ax in axes.flat:
        ax.set_box_aspect(1.0)

    return fig, axes


def pca_resid_plot(wfs, ax=None, c="b", name=None, pad=1, K=25):
    wfs = wfs.reshape(wfs.shape[0], -1)
    wfs = wfs - wfs.mean(axis=0, keepdims=True)
    v = np.square(la.svdvals(wfs)[: K - pad]) / np.prod(wfs.shape)
    ax = ax or plt.gca()
    totvar = np.square(wfs).mean()
    residvar = totvar - np.cumsum(v)
    ax.plot(([totvar] * pad + [*residvar]), marker=".", ms=4, c=c, label=name)


def pca_invert_plot(
    wfs_orig, wfs, q=None, p=None, ax=None, c="b", name=None, pad=0, K=25
):
    # apply PCA to relocated wfs
    wfshape = wfs.shape
    wfs = wfs.reshape(wfs.shape[0], -1)
    print("wfs", wfs.shape)
    means = wfs.mean(axis=0, keepdims=True)
    print(means.shape)
    cwfs = wfs - means
    print(np.square(cwfs).mean())
    U, s, Vh = la.svd(cwfs, full_matrices=False)
    # if pad == 0:
    #     tv = np.square(cwfs).mean()
    #     v = np.square(s[:K]) / np.prod(wfs.shape)
    #     plt.plot(np.r_[[tv], tv - np.cumsum(v)], color="silver")
    # print(wfs.shape, q.shape, p.shape)

    # for each k, get error in unrelocated space
    if q is not None:
        invert = (p / q)[:, None, :]
    recon = np.empty_like(wfs, dtype=np.float64)
    print("recon", recon.shape)
    recon[:] = means
    v = []
    for k in trange(K - pad):
        # pca reconstruction
        if k > 0:
            # recon += U[:, k, None] * (s[k] * Vh[None, k, :])
            recon = means + U[:, :k] @ (np.diag(s[:k]) @ Vh[:k])

        # invert relocation
        if q is not None:
            recon_ = recon.reshape(wfshape) * invert
        else:
            recon_ = recon.reshape(wfshape)

        # compute error
        err = np.square(wfs_orig - recon_).mean()
        v.append(err)

    # plot
    ax = ax or plt.gca()
    totvar = v[0]
    if pad:
        ax.plot(([totvar] * pad + v), marker=".", ms=4, c=c, label=name)
    else:
        ax.plot(v[:K], marker=".", ms=4, c=c, label=name)
    return v


def reloc_pcaresidplot(
    h5,
    wf_key="denoised_waveforms",
    name="",
    B=50_000,
    seed=0,
    threshold=6.0,
    ax=None,
    nolabel=False,
    kind="resid",
):
    rg = np.random.default_rng(seed)
    big = np.flatnonzero(h5["maxptp"][:] > threshold)
    inds = rg.choice(big, size=B, replace=False)
    inds.sort()

    wfs = h5[wf_key][inds]
    wfs_yza, q_hat_yza, p_hat = relocate_simple(
        wfs,
        h5["geom"][:],
        h5["max_channels"][:][inds],
        h5["x"][:][inds],
        h5["y"][:][inds],
        h5["z_rel"][:][inds],
        h5["alpha"][:][inds],
        relocate_dims="yza",
        geomkind="firstchanstandard",
        firstchans=h5["first_channels"][:][inds],
        channel_radius=8,
    )
    wfs_xyza, q_hat_xyza, p_hat_ = relocate_simple(
        wfs,
        h5["geom"][:],
        h5["max_channels"][:][inds],
        h5["x"][:][inds],
        h5["y"][:][inds],
        h5["z_rel"][:][inds],
        h5["alpha"][:][inds],
        relocate_dims="xyza",
        geomkind="firstchanstandard",
        firstchans=h5["first_channels"][:][inds],
        channel_radius=8,
    )
    wfs_yza, q_hat_yza, p_hat = map(
        lambda x: x.cpu().numpy(), (wfs_yza, q_hat_yza, p_hat)
    )
    wfs_xyza, q_hat_xyza, p_hat_ = map(
        lambda x: x.cpu().numpy(), (wfs_xyza, q_hat_xyza, p_hat_)
    )

    ax = ax or plt.gca()
    if kind == "resid":
        pca_resid_plot(wfs, ax=ax, name="Unrelocated", c="k")
        pca_resid_plot(
            wfs_yza, ax=ax, name="$yz\\alpha$ relocated", c=green, pad=3
        )
        pca_resid_plot(
            wfs_xyza, ax=ax, name="$xyz\\alpha$ relocated", c=purple, pad=4
        )
    elif kind == "invert":
        pca_invert_plot(
            wfs,
            wfs,
            np.ones((B, wfs.shape[-1])),
            np.ones((B, wfs.shape[-1])),
            ax=ax,
            name="Unrelocated",
            c="k",
        )
        pca_invert_plot(
            wfs,
            wfs_yza,
            q_hat_yza,
            p_hat,
            ax=ax,
            name="$yz\\alpha$ relocated",
            c=green,
            pad=3,
        )
        pca_invert_plot(
            wfs,
            wfs_xyza,
            q_hat_xyza,
            p_hat,
            ax=ax,
            name="$xyz\\alpha$ relocated",
            c=purple,
            pad=4,
        )
    else:
        raise ValueError
    ax.semilogy()
    yt = [0.5, 0.1]
    ax.set_yticks(yt, list(map(str, yt)))
    ax.legend(fancybox=False, frameon=False)
    if not nolabel:
        ax.set_ylabel("PCA remaining variance (s.u.)")
        ax.set_xlabel("number of factors")
    if name:
        ax.set_title(name)


def traceplot(waveform, axes=None, label="", c="k", alpha=1, strip=True, lw=1):
    if axes is None:
        fig, axes = plt.subplots(
            1,
            waveform.shape[1],
            sharex=True,
            sharey="row",
            figsize=(2 * waveform.shape[1], 2),
        )
    assert (waveform.shape[1],) == axes.shape
    for ax, wf in zip(axes, waveform.T):
        (line,) = ax.plot(wf, color=c, label=label, alpha=alpha, lw=lw)
        if strip:
            sns.despine(ax=ax, bottom=True, left=True)
        ax.set_xticks([])
        ax.grid(color="gray")
        ax.set_axisbelow(True)
    return line


def pcarecontrace(
    h5, wf_key="denoised_waveforms", B=3, K=10, seed=0, threshold=6.0
):
    rg = np.random.default_rng(seed)
    big = np.flatnonzero(h5["maxptp"][:] > threshold)
    inds = rg.choice(big, size=B, replace=False)
    inds.sort()

    fig, axes = plt.subplots(3 * B, 16, sharex=True, sharey="row")

    wfs = h5[wf_key][inds]
    wfs_yza, q_hat_yza, p_hat = relocate_simple(
        wfs,
        h5["geom"][:],
        h5["max_channels"][:][inds],
        h5["x"][:][inds],
        h5["y"][:][inds],
        h5["z_rel"][:][inds],
        h5["alpha"][:][inds],
        relocate_dims="yza",
        geomkind="firstchanstandard",
        firstchans=h5["first_channels"][:][inds],
        channel_radius=8,
    )
    wfs_xyza, q_hat_xyza, p_hat_ = relocate_simple(
        wfs,
        h5["geom"][:],
        h5["max_channels"][:][inds],
        h5["x"][:][inds],
        h5["y"][:][inds],
        h5["z_rel"][:][inds],
        h5["alpha"][:][inds],
        relocate_dims="xyza",
        geomkind="firstchanstandard",
        firstchans=h5["first_channels"][:][inds],
        channel_radius=8,
    )

    ls = h5["loadings_orig"][inds, :K]
    ls_yza = h5["loadings_yza"][inds, :K]
    ls_xyza = h5["loadings_xyza"][inds, :K]

    recons = np.einsum("ijk,li->ljk", h5["pcs_orig"][:K], ls)
    recons_yza = np.einsum("ijk,li->ljk", h5["pcs_orig"][:K], ls)
    recons_xyza = np.einsum("ijk,li->ljk", h5["pcs_orig"][:K], ls)

    recons = (
        np.einsum("ijk,li->ljk", h5["pcs_orig"][:K], ls)
        + h5["mean_orig"][:][None, :, :]
    )  # noqa
    recons_yza = (
        np.einsum("ijk,li->ljk", h5["pcs_orig"][:K], ls_yza)
        + h5["mean_yza"][:][None, :, :]
    )  # noqa
    recons_xyza = (
        np.einsum("ijk,li->ljk", h5["pcs_orig"][:K], ls_xyza)
        + h5["mean_xyza"][:][None, :, :]
    )  # noqa

    labs = "abcdefghijklmnopqrstuvwxyz"
    for i in range(B):
        traceplot(
            wfs[i, :, 1:-1],
            axes[3 * i],
            label=f"{labs[i]} orig.",
            c="black",
            strip=False,
        )  # noqa
        traceplot(
            recons[i, :, 1:-1],
            axes[3 * i],
            label=f"{labs[i]} recon.",
            c="silver",
            strip=False,
        )  # noqa
        traceplot(
            wfs_yza[i, :, 1:-1],
            axes[3 * i + 1],
            label=f"{labs[i]} $yz\\alpha$",
            c=darkgreen,
            strip=False,
        )  # noqa
        traceplot(
            recons_yza[i, :, 1:-1],
            axes[3 * i + 1],
            label=f"{labs[i]} $yz\\alpha$ recon.",
            c=lightgreen,
            strip=False,
        )  # noqa
        traceplot(
            wfs_xyza[i, :, 1:-1],
            axes[3 * i + 2],
            label=f"{labs[i]} $xyz\\alpha$",
            c=darkpurple,
            strip=False,
        )  # noqa
        traceplot(
            recons_xyza[i, :, 1:-1],
            axes[3 * i + 2],
            label=f"{labs[i]} $xyz\\alpha$ recon.",
            c=lightpurple,
            strip=False,
        )  # noqa
    fig.tight_layout(pad=0)


def pca_tracevis(pcs, wfs, title=None, cut=4, strip=False):
    pal = sns.color_palette(n_colors=len(pcs))

    if cut > 0:
        pcs = pcs[:, :, cut:-cut]
        wfs = wfs[:, :, cut:-cut]

    fig, axes = plt.subplots(2, pcs.shape[2], sharey="row", sharex=True)

    handles = []
    labels = []
    for i in range(len(pcs)):
        l = traceplot(pcs[i], axes[0], c=pal[i], strip=strip)
        handles.append(l)
        labels.append(f"pc {i + 1}")

    for wf in wfs:
        l = traceplot(wf, axes[1], alpha=0.5, strip=strip)

    if title:
        fig.suptitle(title)

    fig.legend(
        handles + [l], labels + ["random wfs"], fancybox=False, facecolor="w"
    )
    fig.tight_layout(pad=0.25)

    return fig, axes


def lcorrs(disp, y, alpha, pcs, maxptp, plotmask=None, kind="scatter"):
    df = pd.DataFrame(
        dict(
            disp=disp,
            y=y,
            alpha=alpha,
            pc1=pcs[:, 0],
            pc2=pcs[:, 1],
            pc3=pcs[:, 2],
            maxptp=maxptp,
        )
    )
    if plotmask:
        df = df[plotmask]

    grid = sns.pairplot(
        data=df.sample(frac=0.1),
        x_vars=["pc1", "pc2", "pc3"],
        y_vars=["disp", "y", "alpha"],
        hue="maxptp",
        kind=kind,
        plot_kws=dict(alpha=0.5),
    )

    for i, xv in enumerate(["pc1", "pc2", "pc3"]):
        for j, yv in enumerate(["disp", "y", "alpha"]):
            sp = statistics.spearmanr(df[yv].values, df[xv].values).correlation
            gcs = statistics.gcs(df[yv].values, df[xv].values)
            grid.axes[j, i].set_title(f"Spear: {sp:0.2f}, GCS: {gcs:0.2f}")

    return grid


def gcsboxes(disp, pcs, labels, ax=None, color=None):
    good = np.flatnonzero(labels >= 0)
    disp = disp[good]
    pcs = pcs[good]
    labels = labels[good]
    gcss = [[]] * pcs.shape[1]
    df = pd.DataFrame(columns=["pc", "gcs"])
    for k in np.unique(labels):
        clust = np.flatnonzero(labels == k)
        for j, pc in enumerate(pcs.T):
            gcs = statistics.gcsorig(disp[clust], pc[clust])
            gcss[j].append(gcs)
            df = df.append({"pc": j, "gcs": gcs}, ignore_index=True)
    ax = ax or plt.gca()
    # ax.violinplot(
    #     np.array(gcss).T,
    #     showextrema=False,
    #     showmedians=True,
    #     quantiles=[[0.05, 0.95]] * pcs.shape[1],
    #     bw_method="silverman",
    # )
    g = sns.stripplot(
        x="pc",
        y="gcs",
        data=df,
        ax=ax,
        size=2,
        alpha=0.5,
        color=color,
    )
    g.set(xlabel=None, ylabel=None)


def spearmanrboxes(disp, pcs, labels, ax=None, color=None):
    good = np.flatnonzero(labels >= 0)
    disp = disp[good]
    pcs = pcs[good]
    labels = labels[good]
    spearmanrs = [[]] * pcs.shape[1]
    df = pd.DataFrame(columns=["pc", "r"])
    for k in np.unique(labels):
        clust = np.flatnonzero(labels == k)
        for j, pc in enumerate(pcs.T):
            r = statistics.spearmanr(disp[clust], pc[clust]).correlation
            spearmanrs[j].append(r)
            df = df.append({"pc": j, "r": r}, ignore_index=True)
    ax = ax or plt.gca()
    # ax.violinplot(
    #     np.array(spearmanrs).T,
    #     showextrema=False,
    #     showmedians=True,
    #     quantiles=[[0.05, 0.95]] * pcs.shape[1],
    #     bw_method="silverman",
    # )
    g = sns.stripplot(
        x="pc",
        y="r",
        data=df,
        ax=ax,
        size=2,
        alpha=0.5,
        color=color,
    )
    g.set(xlabel=None, ylabel=None)


def trendline(times, values, mf=21):
    T0 = int(np.floor(times.min()))
    T = int(np.ceil(times.max()))
    out = []
    tdomain = []
    for t in range(T0, T):
        mask = np.flatnonzero((t <= times) & (times < t + 1))
        if len(mask):
            out.append(values[mask].mean())
            tdomain.append(t)
    out = signal.medfilt(np.array(out), mf)
    return np.array(tdomain), out


def get_unit_ix(h5, which, unit, by):
    times = h5["spike_index"][:, 0] / 30000
    ids = h5["spike_train"][:, 1]

    if by == "ptp":
        template_maxptps = h5["templates"][:].ptp(1).ptp(1)
        top = np.argsort(template_maxptps)[::-1][unit]
    elif by == "count":
        ids = h5["spike_train"][:, 1]
        _, counts = np.unique(ids, return_counts=True)
        top = np.argsort(counts)[::-1][unit]
        print(counts[top])
    elif by == "counttrend":
        template_maxptps = h5["templates"][:].ptp(1).ptp(1)
        psfits = template_psfit(h5)
        uncollided = psfits < np.median(psfits)
        thebig = np.flatnonzero(uncollided)
        T = int(np.ceil(times.max()))
        counts = np.empty(h5["templates"].shape[0])
        counts[:] = np.inf
        temp = np.empty(h5["templates"].shape[0])
        temp[:] = np.inf
        for t in range(100, T - 100, 2):
            ix = np.flatnonzero((t <= times) & (times < (t + 5)))
            vals, tcounts = np.unique(ids[ix], return_counts=True)
            temp[vals] = tcounts
            counts = np.minimum(counts, temp)
            temp[:] = np.inf
        top = np.argsort(counts)[::-1]
        top = top[np.flatnonzero(counts[top] < np.inf)[0] :]
        top = top[np.isin(top, thebig)]
        top = top[unit]
    else:
        raise ValueError(f"not sure how to rank units by {by}")

    return top, ids == top


def scatter_loadings(
    h5,
    which="orig",
    unit=0,
    style="-",
    cm=plt.cm.Greys,
    pc=0,
    ax=None,
    by="counttrend",
):
    times = h5["spike_index"][:, 0] / 30000
    gtimes = (100 <= times) & (times < 900)
    loadings = h5[f"loadings_{which}"][:, pc]
    loadings /= loadings.std()

    top, where = get_unit_ix(h5, which, unit, by)

    ax = ax or plt.gca()

    where = np.flatnonzero(where & gtimes)
    # color = cm((top_K + k) / (2 * top_K) - 0.01)
    ts = times[where]
    ls = loadings[where]
    ax.scatter(
        ts,
        ls,
        color=cm(0.5),
        alpha=0.1,
        s=1,
    )
    tt, trend = trendline(ts, ls)
    (l,) = ax.plot(
        tt, trend, lw=1, color=cm(0.9), linestyle=style, label=f"pc {pc}"
    )
    ax.set_xlim([tt[0], tt[-1]])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(2, integer=True))
    # ax.set_yticks([])

    return (
        trend.min() - 0.5 * trend.std(),
        trend.max() + 0.5 * trend.std(),
        l,
        top,
    )


def loadings_vs_disp(h5, unit, p, name="", by="count"):
    fig, axes = plt.subplot_mosaic(
        "a\na\nb\nb\nc\nc\nd", sharex=True, figsize=(2.75, 4)
    )
    aa = axes["a"]
    ab = axes["b"]
    ac = axes["c"]
    ad = axes["d"]

    styles = ["-", "--", "-.", ":"]
    lines = []

    for ax, which, cmap in zip(
        [aa, ab, ac],
        ["orig", "yza", "xyza"],
        [plt.cm.Greys, plt.cm.Greens, plt.cm.Purples],
    ):
        mn, mx = np.inf, -np.inf
        for pc in range(4):
            lo, hi, line, top = scatter_loadings(
                h5,
                unit=unit,
                pc=pc,
                cm=cmap,
                which=which,
                ax=ax,
                style=styles[pc],
                by=by,
            )
            mn = min(lo, mn)
            mx = max(hi, mx)
            if which == "orig":
                lines.append(line)
        ax.set_ylim([mn, mx])

    fig.legend(
        lines,
        [f"PC{j + 1}" for j in range(4)],
        loc=[0.19, 0.895],
        frameon=False,
        ncol=4,
        columnspacing=2,
        fontsize=6,
    )

    aa.set_ylabel("orig.\\ loadings")
    ab.set_ylabel("$yz\\alpha$ loadings")
    ac.set_ylabel("$xyz\\alpha$ loadings")

    ad.plot(np.arange(100, 900), p[100:-100])
    ad.set_ylabel("disp.")
    ad.set_xlabel("time", labelpad=-10)
    ad.set_box_aspect()
    ad.set_xticks([100, 900])
    ad.set_yticks([])
    fig.suptitle(f"{name} unit {top} PC loadings", y=0.95)

    # fig.tight_layout(pad=0.05)


def pairplot_loadings(h5, unit, which, name="", by="counttrend"):
    top, ix = get_unit_ix(h5, which, unit, by)
    ix = np.flatnonzero(ix)
    meandisp = h5["z_abs"][:].mean() - h5["z_reg"][:].mean()
    z_disp = np.abs(h5["z_abs"][:][ix] - h5["z_reg"][:][ix] - meandisp)
    ys = h5["y"][:][ix]
    alphas = h5["alpha"][:][ix]
    loadings = h5[f"loadings_{which}"][:][ix]
    maxptp = h5["maxptp"][:][ix]

    grid = lcorrs(z_disp, ys, alphas, loadings, maxptp, kind="scatter")

    return top, grid


def template_psfit(h5):
    templates = h5["templates"][:]
    nztemplates = templates.ptp(1).ptp(1) > 0

    twfs, maxchans = waveform_utils.get_local_waveforms(
        templates[nztemplates],
        8,
        h5["geom"][:],
        maxchans=None,
        geomkind="standard",
    )
    x, y, z_rel, z_abs, alpha = localization.localize_waveforms(
        twfs,
        h5["geom"][:],
        maxchans=maxchans,
        channel_radius=8,
        geomkind="standard",
    )
    ptp, ptp_hat = point_source_centering.ptp_fit(
        twfs,
        h5["geom"][:],
        maxchans,
        x,
        y,
        z_rel,
        alpha,
        channel_radius=8,
        geomkind="standard",
    )

    res = np.empty(templates.shape[0])
    res[:] = np.inf
    res[nztemplates] = np.square(ptp - ptp_hat).mean(axis=1)

    return res


def plot_template_psfit(h5):
    psfits = template_psfit(h5)
    plt.hist(psfits[psfits < np.inf], bins=128)


def uncolboxes(h5, which="orig", kind="gcs", ax=None):
    psfits = template_psfit(h5)
    good = np.flatnonzero(psfits < np.median(psfits[psfits < np.inf]))
    ids = h5["spike_train"][:, 1]
    whichinds = np.isin(ids, good)
    uncolids = ids[whichinds]
    uniqs, counts = np.unique(uncolids, return_counts=True)
    newgood = uniqs[counts > 1000]
    whichinds = np.isin(ids, newgood)
    ids = ids[whichinds]
    pcs = h5[f"loadings_{which}"][:][whichinds, :5]
    disp = np.abs(h5["z_reg"][:][whichinds] - h5["z_abs"][:][whichinds])
    color = {"orig": "k", "yza": darkgreen, "xyza": darkpurple}[which]

    if kind == "gcs":
        gcsboxes(disp, pcs, ids, ax=ax, color=color)
    elif kind == "spear":
        spearmanrboxes(disp, pcs, ids, ax=ax, color=color)


def cluster_scatter(
    xs,
    ys,
    ids,
    c=None,
    do_ellipse=True,
    ax=None,
    n_std=1.0,
    zlim=None,
    alpha=0.05,
):
    cm = np.array(colorcet.glasbey_hv)
    c_ids = ids
    if c is not None:
        cm = np.array(colorcet.bmy)
        c_ids = c - c.min()
        c_ids /= c_ids.max() / 255
        c_ids = c_ids.astype(int)
        print(c_ids)

    ax = ax or plt.gca()
    # scatter and collect gaussian info
    means = {}
    covs = {}
    for k in np.unique(ids):
        where = np.flatnonzero(ids == k)
        xk = xs[where]
        yk = ys[where]
        means[k] = xk.mean(), yk.mean()
        covs[k] = np.cov(xk, yk)
        color = cm[c_ids[where] % 256]
        ax.scatter(xk, yk, s=1, c=color, alpha=alpha)

    xlow = np.inf
    xhigh = -np.inf
    ylow = np.inf
    yhigh = -np.inf
    for k in means.keys():
        mean_x, mean_y = means[k]
        cov = covs[k]
        vx, vy = cov[0, 0], cov[1, 1]
        rho = cov[0, 1] / np.sqrt(vx * vy)

        if do_ellipse and c is None:
            color = cm[c_ids[np.flatnonzero(ids == k)[0]] % 256]
            ell = Ellipse(
                (0, 0),
                width=2 * np.sqrt(1 + rho),
                height=2 * np.sqrt(max(0, 1 - rho)),
                facecolor=(0, 0, 0, 0),
                edgecolor=color,
                linewidth=1,
            )
            transform = (
                transforms.Affine2D()
                .rotate_deg(45)
                .scale(n_std * np.sqrt(vx), n_std * np.sqrt(vy))
                .translate(mean_x, mean_y)
            )
            ell.set_transform(transform + ax.transData)
            ax.add_patch(ell)

        if zlim is None or zlim[0] < mean_y < zlim[1]:
            xlow = min(xlow, mean_x - (n_std + 0.1) * np.sqrt(vx))
            xhigh = max(xhigh, mean_x + (n_std + 0.1) * np.sqrt(vx))
            ylow = min(ylow, mean_y - (n_std + 0.2) * np.sqrt(vy))
            yhigh = max(yhigh, mean_y + (n_std + 0.2) * np.sqrt(vy))

    return xlow, xhigh, ylow, yhigh


def sortedclust_pcvis(h5, name, zlims=None, top_K=50):
    ids = h5["spike_train"][:, 1].astype(int)
    uniqs, counts = np.unique(ids, return_counts=True)
    threshold = max(1000, np.sort(counts)[::-1][:top_K][-1])
    good = uniqs[np.flatnonzero(counts > threshold)]

    whichspikes = np.isin(ids, good)
    ids = ids[whichspikes]
    z = h5["z_reg"][:][whichspikes]

    fig, axes = plt.subplots(4, 3, sharey=True, figsize=(8.5, 11))

    k2txt = {"x": "$x$", "y": "$y$", "alpha": "$\\alpha$"}
    which2txt = {
        "orig": "orig.\\",
        "yza": "$yz\\alpha$",
        "xyza": "$xyz\\alpha$",
    }

    for ax, k in zip(axes[0, :], ["x", "y", "alpha"]):
        horz = h5[k][:][whichspikes]
        xlow, xhigh, ylow, yhigh = cluster_scatter(
            horz, z, ids, ax=ax, zlim=zlims
        )
        ax.set_xlim([xlow, xhigh])
        if zlims is not None:
            ax.set_ylim(zlims)
        else:
            ax.set_ylim([ylow, yhigh])
        ax.set_xlabel(k2txt[k])

    for which, axs in zip(["orig", "yza", "xyza"], axes[1:]):
        loadings = h5[f"loadings_{which}"][:, :3]
        loadings /= np.std(loadings, axis=0, keepdims=True)
        loadings = loadings[whichspikes]
        for i in range(3):
            xlow, xhigh, ylow, yhigh = cluster_scatter(
                loadings[:, i], z, ids, ax=axs[i], zlim=zlims
            )
            axs[i].set_xlim([xlow, xhigh])
            if zlims is not None:
                axs[i].set_ylim(zlims)
            else:
                axs[i].set_ylim([ylow, yhigh])
            axs[i].set_xlabel(which2txt[which] + " pc " + str(i + 1))

    for ax in axes[:, 0]:
        ax.set_ylabel("$z$")

    fig.suptitle(
        f"{name}: Spike sorter clusters visualized in the PCA spaces", y=1
    )
    plt.tight_layout(pad=0.5)


def relocclusts(
    name,
    x,
    y,
    z,
    alpha,
    pcs_orig,
    pcs_yza,
    pcs_xyza,
    cs,
    aris,
    zlims=None,
):
    fig, axes = plt.subplots(6, 3, sharey=True, figsize=(8.5, 11))
    ario = aris["orig"]
    ariy = aris["yza"]
    arix = aris["xyza"]
    co = cs["orig"]
    cy = cs["yza"]
    cx = cs["xyza"]

    k2txt = {"x": "$x$", "y": "$y$", "alpha": "$\\alpha$"}
    which2txt = {
        "orig": "Unrelocated",
        "yza": "$yz\\alpha$",
        "xyza": "$xyz\\alpha$",
    }

    for j, (which, pcs, c, ari) in enumerate(
        zip(
            ["orig", "yza", "xyza"],
            [pcs_orig, pcs_yza, pcs_xyza],
            [co, cy, cx],
            [ario, ariy, arix],
        )
    ):
        axes[2 * j, 1].set_title(
            which2txt[which]
            + f" {len(np.unique(c))} clusters, "
            + f" ARI to sorter: {ari:0.2f}",
            fontsize=10,
        )
        for ax, k, v in zip(axes[2 * j], ["x", "y", "alpha"], [x, y, alpha]):
            xlow, xhigh, ylow, yhigh = cluster_scatter(
                v, z, c, ax=ax, zlim=zlims
            )
            ax.set_xlabel(k2txt[k])
            ax.set_xlim([xlow, xhigh])
            if zlims is not None:
                ax.set_ylim(zlims)
            else:
                ax.set_ylim([ylow, yhigh])

        for ax, k, v in zip(axes[2 * j + 1], range(1, 4), pcs.T):
            xlow, xhigh, ylow, yhigh = cluster_scatter(
                v, z, c, ax=ax, zlim=zlims
            )
            ax.set_xlabel(f"pc{k}")
            ax.set_xlim([xlow, xhigh])
            if zlims is not None:
                ax.set_ylim(zlims)
            else:
                ax.set_ylim([ylow, yhigh])

    for ax in axes[:, 0]:
        ax.set_ylabel("$z$")

    fig.suptitle(f"{name}: ISO-SPLIT clusters by feature", y=1)
    plt.tight_layout(pad=0.1)
