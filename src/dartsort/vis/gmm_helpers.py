import numpy as np
import torch

from .colors import glasbey1024


def get_neighbors(gmm, unit_id, n_neighbors=5):
    means, covs, logdets = gmm.stack_units()
    dists = gmm.units[unit_id].divergence(means, covs, logdets, kind=gmm.distance_metric)
    dists = dists.view(-1)
    order = torch.argsort(dists)
    assert order[0] == unit_id
    return order[:n_neighbors + 1]


def amp_double_scatter(gmm, indices, panel, unit_id=None, labels=None, viol_ms=None):
    ax_time, ax_dist = panel.subplots(ncols=2, width_ratios=[5, 1], sharey=True)
    ax_time.set_ylabel("max tpca norm")
    ax_time.set_xlabel("time (s)")
    ax_dist.set_xlabel("count")
    if not indices.numel():
        return
    amps = gmm.data.amps[indices]
    t = gmm.data.times_seconds[indices]
    dt_ms = np.diff(t) * 1000
    if viol_ms is not None:
        small = dt_ms <= viol_ms

    if labels is not None:
        c = glasbey1024[labels]
    elif unit_id is not None:
        c = glasbey1024[unit_id]
    else:
        assert False

    ax_time.scatter(t, amps, s=3, lw=0, color=c)
    if viol_ms is not None and small.any():
        small = np.logical_or(
            np.pad(small, (1, 0), constant_values=False),
            np.pad(small, (0, 1), constant_values=False),
        )
        ax_time.scatter(t[small], amps[small], s=3, lw=0, color="k")

    histk = dict(histtype="step", bins=64, orientation="horizontal")
    if labels is None:
        ax_dist.hist(amps, color=c, label="unit", **histk)
    else:
        for j in np.unique(labels):
            ax_dist.hist(amps[labels == j], color=glasbey1024[j], label="unit", **histk)
