import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy
from matplotlib.colors import to_hex
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage

from .colors import glasbey1024


def scatter_max_channel_waveforms(
    axis,
    template_data,
    waveform_height=0.05,
    waveform_width=0.95,
    show_geom=True,
    geom_scatter_kwargs={"marker": "s", "lw": 0, "s": 3},
    lw=1,
    colors=glasbey1024,
    **plot_kwargs,
):
    dx = np.ptp(waveform_width * template_data.registered_geom[:, 0])
    dz = np.ptp(template_data.registered_geom[:, 1])
    max_abs_amp = np.abs(template_data.templates).max()
    zscale = dz * waveform_height / max_abs_amp

    xrel = np.linspace(-dx / 2, dx / 2, num=template_data.templates.shape[1])
    locs = template_data.template_locations()
    locsx = locs["x"]
    locsz = locs["z_abs"]

    if show_geom:
        axis.scatter(*template_data.registered_geom.T, **geom_scatter_kwargs)

    for j, (u, temp) in enumerate(zip(template_data.unit_ids, template_data.templates)):
        ptpvec = np.ptp(temp, 0)
        if ptpvec.max() == 0:
            continue
        mc = ptpvec.argmax()
        mctrace = temp[:, mc]

        xc = locsx[j]
        zc = locsz[j]
        c = colors[u % len(colors)]
        axis.plot(xc + xrel, zc + zscale * mctrace, lw=lw, color=c, **plot_kwargs)


def distance_matrix_dendro(
    panel,
    distances,
    unit_ids=None,
    dendrogram_linkage=None,
    dendrogram_threshold=0.25,
    show_unit_labels=False,
    vmax=1.0,
    image_cmap=plt.cm.RdGy,
    show_values=False,
    label=None,
):
    show_dendrogram = dendrogram_linkage is not None
    dendro_width = (
        (
            0.7,
        )
        if show_dendrogram
        else ()
    )

    gs = panel.add_gridspec(
        nrows=3,
        ncols=2 + show_dendrogram,
        height_ratios=[0.5, 1, 0.5],
        width_ratios=[2, 0.15, *dendro_width],
    )
    ax_im = panel.add_subplot(gs[:, 0])
    ax_cbar = panel.add_subplot(gs[1, 1])
    if show_dendrogram:
        scipy.cluster.hierarchy.set_link_color_palette(list(map(to_hex, glasbey1024)))
        ax_dendro = panel.add_subplot(gs[:, 2], sharey=ax_im)
        ax_dendro.axis("off")

        Z, labels = get_linkage(
            distances, method=dendrogram_linkage, threshold=dendrogram_threshold
        )
        dendro = dendrogram(
            Z,
            ax=ax_dendro,
            color_threshold=dendrogram_threshold,
            distance_sort=True,
            orientation="right",
            above_threshold_color="k",
        )
        order = np.array(dendro["leaves"])
    else:
        order = np.arange(distances.shape[0])

    im = ax_im.imshow(
        distances[order][:, order],
        vmin=0,
        vmax=vmax,
        cmap=image_cmap,
        extent=[0, len(distances) * 10, 0, len(distances) * 10] if show_dendrogram else None,
        origin="lower",
    )
    if show_values:
        sc = 10 if show_dendrogram else 1
        so = 5 if show_dendrogram else 0
        for (j, i), val in np.ndenumerate(distances[order][:, order]):
            lc = invert(image_cmap(val / vmax))
            ax_im.text(so + sc * i, so + sc * j, f"{val:.2f}", ha="center", va="center", clip_on=True, color=lc)
    if show_unit_labels:
        if unit_ids is None:
            unit_ids = np.arange(distances.shape[0])
        sc = 10 if show_dendrogram else 1
        so = 5 if show_dendrogram else 0
        ax_im.set_xticks(so + sc * np.arange(len(order)), unit_ids[order])
        ax_im.set_yticks(so + sc * np.arange(len(order)), unit_ids[order])
        for i, (tx, ty) in enumerate(
            zip(ax_im.xaxis.get_ticklabels(), ax_im.yaxis.get_ticklabels())
        ):
            tx.set_color(glasbey1024[unit_ids[i]])
            ty.set_color(glasbey1024[unit_ids[i]])
    else:
        ax_im.set_xticks([])
        ax_im.set_yticks([])

    plt.colorbar(im, cax=ax_cbar, label=label)
    ax_cbar.set_yticks([0, vmax])
    if label:
        ax_cbar.set_ylabel("template distance", labelpad=-5)
    return ax_im


_k = np.array([0., 0., 0., 1.])
_w = np.array([1., 1., 1., 1.])


def invert(color):
    color = np.array(color)
    if color[:3].mean() > 0.5:
        return _k
    return _w


def get_linkage(dists, method="complete", threshold=0.25):
    pdist = dists[np.triu_indices(dists.shape[0], k=1)].copy()
    pdist[~np.isfinite(pdist)] = 1_000_000 + pdist[np.isfinite(pdist)].max()
    # complete linkage: max dist between all pairs across clusters.
    Z = linkage(pdist, method=method)
    # extract flat clustering using our max dist threshold
    labels = fcluster(Z, threshold, criterion="distance")
    return Z, labels


def density_peaks_study(X, density_result, dims=[0, 1], fig=None, axes=None, idx=None, inv=None, **scatter_kw):
    if inv is None:
        idx = np.arange(len(X))
        inv = np.arange(len(X))
    if fig is None and axes is None:
        fig, axes = plt.subplots(ncols=3, layout="constrained", figsize=(9, 3), sharey=True)
    elif axes is None:
        axes = fig.subplots(ncols=3, sharey=True)

    scatter_kw = dict(lw=0, s=5) | scatter_kw

    density = density_result["density"][inv]
    labels = density_result["labels"][inv]
    good = density_result["nhdn"][inv] < len(density_result["density"])
    nhdns = np.full_like(inv, -1)
    nhdns[good] = idx[density_result["nhdn"][inv][good]]

    axes[0].scatter(*X[:, dims].T, c=density, **scatter_kw)
    missed = labels < 0
    if missed.any():
        axes[1].scatter(*X[missed][:, dims].T, c="gray", **scatter_kw)
    if ~missed.any():
        axes[1].scatter(
            *X[~missed][:, dims].T, c=density[~missed], **scatter_kw
        )
    for i in range(len(X)):
        nhdn = nhdns[i]
        if nhdn < 0:
            continue
        x = X[i, dims]
        dx = X[nhdn, dims] - x
        axes[1].arrow(
            *x, *dx, length_includes_head=True, width=0, color="k"
        )
    colors = np.concatenate([[[0.5, 0.5, 0.5]], glasbey1024])
    axes[2].scatter(*X[:, dims].T, c=colors[labels + 1], **scatter_kw)
    axes[2].scatter(
        *X[missed][:, dims].T, c="gray", **scatter_kw
    )
    return fig, axes


def isi_hist(times_s, axis, max_ms=5, bin_ms=0.1, color="k", label=None, histtype="bar", alpha=1.0):
    dt_ms = np.diff(times_s) * 1000
    bin_edges = np.arange(
        0,
        max_ms + bin_ms,
        bin_ms,
    )
    # counts, _ = np.histogram(dt_ms, bin_edges)
    # bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    # axis.bar(bin_centers, counts)
    axis.hist(dt_ms, bin_edges, color=color, label=label, histtype=histtype, alpha=alpha)
    axis.set_xlabel("isi (ms)")
    axis.set_ylabel(f"count (out of {dt_ms.size} total isis)")


def correlogram(times_a, times_b=None, max_lag=50):
    lags = np.arange(-max_lag, max_lag + 1)
    ccg = np.zeros(len(lags), dtype=int)

    times_a = np.sort(times_a)
    auto = times_b is None
    if auto:
        times_b = times_a
    else:
        times_b = np.sort(times_b)

    for i, lag in enumerate(lags):
        lagged_b = times_b + lag
        insertion_inds = np.searchsorted(times_a, lagged_b)
        found = insertion_inds < len(times_a)
        ccg[i] = np.sum(times_a[insertion_inds[found]] == lagged_b[found])

    if auto:
        ccg[lags == 0] = 0

    return lags, ccg


def bar(ax, x, y, **kwargs):
    dx = np.diff(x).min()
    x0 = np.concatenate((x - dx, x[-1:] + dx))
    return ax.stairs(y, x0, **kwargs)


def plot_correlogram(axis, times_a, times_b=None, max_lag=50, color="k", fill=True, **stairs_kwargs):
    lags, ccg = correlogram(times_a, times_b=times_b, max_lag=max_lag)
    axis.set_xlabel("lag (samples)")
    return bar(axis, lags, ccg, fill=fill, color=color, **stairs_kwargs)
