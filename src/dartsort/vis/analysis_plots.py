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
    waveform_width=0.75,
    show_geom=True,
    geom_scatter_kwargs={"marker": "s", "lw": 0, "s": 3},
    lw=1,
    colors=glasbey1024,
    **plot_kwargs,
):
    dx = template_data.registered_geom[:, 0].ptp()
    dz = template_data.registered_geom[:, 1].ptp()
    max_abs_amp = np.abs(template_data.templates).max()
    zscale = dz * waveform_height / max_abs_amp

    xrel = np.linspace(-dx / 2, dx / 2, num=template_data.templates.shape[1])
    locs = template_data.template_locations()
    locsx = locs["x"]
    locsz = locs["z_abs"]

    if show_geom:
        axis.scatter(*template_data.registered_geom.T, **geom_scatter_kwargs)

    for j, (u, temp) in enumerate(
        zip(template_data.unit_ids, template_data.templates)
    ):
        ptpvec = temp.ptp(0)
        if ptpvec.max() == 0:
            continue
        mc = ptpvec.argmax()
        mctrace = temp[:, mc]

        xc = locsx[j]
        zc = locsz[j]
        c = colors[u % len(colors)]
        axis.plot(
            xc + xrel, zc + zscale * mctrace, lw=lw, color=c, **plot_kwargs
        )


def distance_matrix_dendro(
    panel,
    distances,
    unit_ids=None,
    dendrogram_linkage=None,
    dendrogram_threshold=0.25,
    show_unit_labels=False,
    vmax=1.0,
    image_cmap=plt.cm.RdGy,
):
    show_dendrogram = dendrogram_linkage is not None
    dendro_width = (0.7, 0.15,) if show_dendrogram else ()

    gs = panel.add_gridspec(
        nrows=3,
        ncols=3 + 2 * show_dendrogram,
        wspace=0.0,
        hspace=0,
        height_ratios=[0.5, 1, 0.5],
        width_ratios=[2, 0.15, *dendro_width, 0.1],
    )
    ax_im = panel.add_subplot(gs[:, 0])
    ax_cbar = panel.add_subplot(gs[1, -1])
    if show_dendrogram:
        scipy.cluster.hierarchy.set_link_color_palette(
            list(map(to_hex, glasbey1024))
        )
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
        extent=[0, 100, 0, 100],
        vmin=0,
        vmax=vmax,
        cmap=image_cmap,
        origin="lower",
    )
    if show_unit_labels:
        if unit_ids is None:
            unit_ids = np.arange(distances.shape[0])
        ax_im.set_xticks(10 * np.arange(len(order)) + 5, unit_ids[order])
        ax_im.set_yticks(10 * np.arange(len(order)) + 5, unit_ids[order])
    else:
        ax_im.set_xticks([])
        ax_im.set_yticks([])

    plt.colorbar(im, cax=ax_cbar, label="template distance")
    ax_cbar.set_yticks([0, vmax])
    ax_cbar.set_ylabel("template distance", labelpad=-5)


def get_linkage(dists, method="complete", threshold=0.25):
    pdist = dists[np.triu_indices(dists.shape[0], k=1)].copy()
    pdist[~np.isfinite(pdist)] = 1_000_000 + pdist[np.isfinite(pdist)].max()
    # complete linkage: max dist between all pairs across clusters.
    Z = linkage(pdist, method=method)
    # extract flat clustering using our max dist threshold
    labels = fcluster(Z, threshold, criterion="distance")
    return Z, labels
