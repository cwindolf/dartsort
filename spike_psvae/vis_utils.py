import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors, cm
import numpy as np
import torch
import torch.nn.functional as F


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


def labeledmosaic(xs, rowlabels, pad=0, padval=255, ax=None, cbar=True, separate_norm=False, collabels="abcdefghijklmnopqrstuvwxyz"):
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
    for k, (ptp, label, color) in enumerate(zip(ptps, labels, colors)):
        for j in range(n * n):
            ax = axes.flat[j]
            ptp_left = ptp[j, ::2]
            ptp_right = ptp[j, 1::2]
            handles[k], = ax.plot(ptp_left, c=color, label=label)
            ax.plot(ptp_right, "--", c=color)
            ax.text(
                0.1,
                0.9,
                codes[j],
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )

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
        ax.set_box_aspect(1.)
    return fig, axes
