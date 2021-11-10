import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors, cm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid


@torch.no_grad()
def mosaic(xs, pad=0, padval=255):
    assert all(x.shape == xs[0].shape for x in xs)
    nrows = len(xs)
    B, H, W = xs[0].shape
    grid = torch.stack(xs, dim=0)  # nrowsBHW
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


def labeledmosaic(xs, rowlabels, pad=0, padval=255, ax=None, cbar=True):
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
            "abcdefghijklmnopqrstuvwxyz"[b],
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
        cbar.set_ticks(ticks)
        cbar.ax.set_yticklabels(ticks, fontsize=8)
