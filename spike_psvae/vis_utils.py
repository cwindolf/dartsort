import matplotlib.pyplot as plt
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
    return grid


def labeledmosaic(xs, rowlabels, pad=0, padval=255, ax=None):
    B, H, W = xs[0].shape
    grid = mosaic(xs, pad=pad, padval=padval).numpy()
    grid = np.pad(grid, [(0, 0), (12, 0), (0, 0)], constant_values=padval)
    ax = ax or plt.gca()
    ax.imshow(
        np.broadcast_to(grid, (*grid.shape[:2], 3)),
        interpolation="nearest",
    )
    ax.axis("off")

    for i, label in enumerate(rowlabels):
        ax.text(
            6,
            i * (H + 2 * pad) + H / 2 + pad,
            label,
            rotation="vertical",
            ha="center",
            va="center",
        )
