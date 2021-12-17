import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors, cm
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from . import statistics
from .point_source_centering import relocate_simple


sns.set_style("ticks")


darkpurple = plt.cm.Purples(0.99)
lightpurple = plt.cm.Purples(0.5)
darkgreen = plt.cm.Greens(0.99)
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
def mosaic(xs, pad=0, padval=255):
    assert all(x.shape == xs[0].shape for x in xs)
    nrows = len(xs)
    B, H, W = xs[0].shape
    grid = torch.stack(list(map(torch.as_tensor, xs)), dim=0)  # nrowsBHW
    grid -= grid.min()
    grid *= 255.0 / grid.max()
    grid = grid.to(torch.uint8)
    if pad > 0:
        grid = F.pad(grid, (pad, pad, pad, pad), value=padval)
    grid = grid.permute(0, 2, 1, 3).reshape(
        (H + 2 * pad) * nrows, B * (W + 2 * pad), 1
    )
    grid = grid[pad:-pad, pad:-pad, :]
    return grid


def labeledmosaic(
    xs,
    rowlabels,
    pad=0,
    padval=255,
    ax=None,
    cbar=True,
    separate_norm=False,
    collabels="abcdefghijklmnopqrstuvwxyz",
):
    if separate_norm:
        assert not cbar
        xs = [normbatch(x) for x in xs]

    vmin = min(x.min() for x in xs)
    vmax = max(x.max() for x in xs)
    B, H, W = xs[0].shape
    grid = mosaic(xs, pad=pad, padval=padval).numpy()
    grid = np.pad(grid, [(14, 0), (20, 0), (0, 0)], constant_values=padval)
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
            20 + b * (W + 2 * pad) + W / 2,
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


def plot_ptp(ptp, axes, label, color, codes):
    for j, ax in enumerate(axes.flat):
        ptp_left = ptp[j, ::2]
        ptp_right = ptp[j, 1::2]
        handle = ax.plot(ptp_left, c=color, label=label)
        dhandle = ax.plot(ptp_right, "--", c=color)
        ax.text(
            0.1,
            0.9,
            codes[j],
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        return handle, dhandle


def vis_ptps(
    ptps,
    labels,
    colors,
    subplots_kwargs=dict(sharex=True, sharey=True, figsize=(5, 5)),
    codes="abcdefghijklmnopqrstuvwxyz",
):
    ptps = np.array([np.array(ptp) for ptp in ptps])
    K, N, C = ptps.shape
    assert len(labels) == K == len(colors)
    n = int(np.sqrt(N))

    fig, axes = plt.subplots(n, n, **subplots_kwargs)
    handles = {}
    dhandles = {}
    for k, (ptp, label, color) in enumerate(zip(ptps, labels, colors)):
        handles[k], dhandles[k] = plot_ptp(ptp, axes, label, color, codes)

    if K == 1:
        plt.figlegend(
            handles=list(handles.values()) + list(dhandles.values()),
            labels=[label + ", left channels", label + ", right channels"],
            loc="upper center",
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
            loc="upper center",
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


def locrelocplots(h5, wf_key="denoised_waveforms", seed=0):
    rg = np.random.default_rng(seed)
    N = len(h5[wf_key])
    inds = rg.choice(N, size=8)
    inds.sort()
    wfs = h5[wf_key][inds]
    orig_ptp = wfs.ptp(1)

    wfs_reloc_yza, stereo_ptp_yza, pred_ptp = relocate_simple(
        wfs,
        h5["geom"][:],
        h5["max_channels"][inds],
        h5["x"][inds],
        h5["y"][inds],
        h5["z_rel"][inds],
        h5["alpha"][inds],
        channel_radius=10,
        firstchans=h5["first_channels"][inds]
        if "first_channels" in h5
        else None,
        geomkind="firstchan" if "first_channels" in h5 else "updown",
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
        channel_radius=10,
        firstchans=h5["first_channels"][inds]
        if "first_channels" in h5
        else None,
        geomkind="firstchan" if "first_channels" in h5 else "updown",
        relocate_dims="xyza",
    )
    wfs_reloc_xyza = wfs_reloc_xyza.numpy()
    stereo_ptp_xyza = stereo_ptp_xyza.numpy()
    pred_ptp_ = pred_ptp_.numpy()

    assert np.all(pred_ptp_ == pred_ptp)

    yza_ptp = wfs_reloc_yza.ptp(1)
    xyza_ptp = wfs_reloc_xyza.ptp(1)

    codes = "abcdefgh"
    fig, axes = plt.subplots(
        5, 3 * 2, figsize=(6, 4), sharex=True, sharey=True
    )
    la = "observed ptp"
    laa = "predicted ptp"
    lb = "yza relocated ptp"
    lbb = "yza standard ptp"
    lc = "xyza relocated ptp"
    lcc = "xyza standard ptp"
    ha, _ = plot_ptp(orig_ptp, axes[:, :2], la, "black", codes)
    haa, _ = plot_ptp(pred_ptp, axes[:, :2], laa, "silver", codes)
    hb, _ = plot_ptp(yza_ptp, axes[:, 2:4], lb, darkgreen, codes)
    hbb, _ = plot_ptp(stereo_ptp_yza, axes[:, 2:4], lbb, lightgreen, codes)
    hc, _ = plot_ptp(xyza_ptp, axes[:, 4:], lc, darkpurple, codes)
    hcc, _ = plot_ptp(stereo_ptp_xyza, axes[:, 4:], lcc, lightpurple, codes)

    fig.figlegend(
        handles=[ha, hb, hc, haa, hbb, hcc],
        labels=[la, lb, lc, laa, lbb, lcc],
        loc="upper center",
        frameon=False,
        fancybox=False,
        borderpad=0,
        borderaxespad=0,
        ncol=3,
    )


def traceplot(waveform, axes, label="", c="k", alpha=1, strip=True, lw=1):
    assert (waveform.shape[1],) == axes.shape
    for ax, wf in zip(axes, waveform.T):
        (line,) = ax.plot(wf, color=c, label=label, alpha=alpha, lw=lw)
        if strip:
            sns.despine(ax=ax, bottom=True, left=True)
        ax.set_xticks([])
        ax.grid(color="gray")
        ax.set_axisbelow(True)
    return line


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


def lcorrs(disp, y, alpha, pcs, maxptp, plotmask):
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

    grid = sns.pairplot(
        data=df[plotmask].sample(frac=0.1),
        x_vars=["pc1", "pc2", "pc3"],
        y_vars=["disp", "y", "alpha"],
        hue="maxptp",
    )

    for i, xv in enumerate(["pc1", "pc2", "pc3"]):
        for j, yv in enumerate(["disp", "y", "alpha"]):
            sp = statistics.spearmanr(df[yv].values, df[xv].values).correlation
            gcs = statistics.gcs(df[yv].values, df[xv].values)
            grid.axes[j, i].set_title(f"Spear: {sp:0.2f}, GCS: {gcs:0.2f}")

    return grid


def gcsboxes(disp, pcs, labels):
    good = np.flatnonzero(labels >= 0)
    disp = disp[good]
    pcs = pcs[good]
    labels = labels[good]
    Kmax = labels.max() + 1
    gcss = [[]] * pcs.shape[1]
    for k in range(Kmax):
        clust = np.flatnonzero(labels == k)
        for j, pc in enumerate(pcs.T):
            gcss[j].append(statistics.gcs(disp[clust], pc[clust]))
    plt.boxplot(np.array(gcss).T)


def spearmanrboxes(disp, pcs, labels):
    good = np.flatnonzero(labels >= 0)
    disp = disp[good]
    pcs = pcs[good]
    labels = labels[good]
    Kmax = labels.max() + 1
    spearmanrs = [[]] * pcs.shape[1]
    for k in range(Kmax):
        clust = np.flatnonzero(labels == k)
        for j, pc in enumerate(pcs.T):
            spearmanrs[j].append(
                statistics.spearmanr(disp[clust], pc[clust]).correlation
            )
    plt.boxplot(np.array(spearmanrs).T)
