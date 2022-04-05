import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from scipy.spatial import cdist
from spikeinterface.numpyextractors import NumpySorting
from spikeinterface.toolkit import compute_correlograms

import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse


ccolors = cc.glasbey


def get_ccolor(k):
    if k == -1:
        return "#808080"
    else:
        return ccolors[k % len(ccolors)]


def cluster_scatter(
    xs, ys, ids, ax=None, n_std=2.0, excluded_ids={-1}, s=1, alpha=0.5
):
    ax = ax or plt.gca()
    # scatter and collect gaussian info
    means = {}
    covs = {}
    for k in np.unique(ids):
        where = np.flatnonzero(ids == k)
        xk = xs[where]
        yk = ys[where]
        color = get_ccolor(k)
        ax.scatter(xk, yk, s=s, color=color, alpha=alpha)
        if k not in excluded_ids:
            x_mean, y_mean = xk.mean(), yk.mean()
            xycov = np.cov(xk, yk)
            means[k] = x_mean, y_mean
            covs[k] = xycov
            ax.annotate(str(k), (x_mean, y_mean))

    for k in means.keys():
        mean_x, mean_y = means[k]
        cov = covs[k]

        with np.errstate(invalid="ignore"):
            vx, vy = cov[0, 0], cov[1, 1]
            rho = cov[0, 1] / np.sqrt(vx * vy)
        if not np.isfinite([vx, vy, rho]).all():
            continue

        color = get_ccolor(k)
        ell = Ellipse(
            (0, 0),
            width=2 * np.sqrt(1 + rho),
            height=2 * np.sqrt(1 - rho),
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


def array_scatter(
    labels,
    geom,
    x,
    z,
    maxptp,
    zlim=(-50, 3900),
    axes=None,
):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(5, 8))

    cluster_scatter(
        x,
        z,
        labels,
        ax=axes[0],
        s=20,
        alpha=0.05,
    )
    axes[0].scatter(*geom.T, c="orange", marker="s")
    axes[0].set_ylabel("z")
    axes[0].set_xlabel("x")

    cluster_scatter(
        maxptp,
        z,
        labels,
        ax=axes[1],
        s=20,
        alpha=0.05,
    )
    axes[1].set_xlabel("scaled ptp")

    axes[2].scatter(x, z, c=plt.cm.viridis(np.clip(maxptp, 3, 15)), alpha=0.1)
    axes[2].scatter(*geom.T, c="orange", marker="s")
    axes[2].set_title("ptps")
    axes[0].set_ylim(zlim)
    return fig


def plot_waveforms_geom(
    main_cluster_id,
    clusters_to_plot,
    labels,
    geom,
    channel_index,
    spike_index,
    maxptps,
    waveforms=None,
    raw_bin=None,
    residual_bin=None,
    spikes_plot=100,
    t_range=(22, 72),
    num_rows=3,
    alpha=0.1,
    h_shift=0,
    do_mean=False,
    ax=None,
):
    ax = ax or plt.gca()
    rg = np.random.default_rng(0)

    assert not h_shift
    assert not do_mean

    # what channels will we plot?
    vals, counts = np.unique(
        spike_index[np.flatnonzero(labels == main_cluster_id), 1]
    )
    z_uniq, z_ids = np.unique(geom[:, 1], return_inverse=True)
    mcid = z_ids[vals[counts.argmax()]]
    channels_plot = np.flatnonzero((z_ids >= mcid - 3) & (z_ids <= mcid + 3))

    # scatter the channels
    ax.scatter(*geom[channels_plot].T, c="orange", marker="s")
    for c in channels_plot:
        ax.annotate(c, (geom[c, 0], geom[c, 1]))

    # indexing / plotting helpers
    t_range = np.arange(42 - t_range[0], 42 + t_range[1])
    times_plot = t_range - 42
    x_uniq = np.unique(geom[:, 0])
    times_plot *= 0.5 * (x_uniq[1] - x_uniq[0]) / np.abs(times_plot).max()
    which_chans_loc = np.isin(channel_index, channels_plot)

    # how to scale things?
    all_max_ptp = maxptps[
        np.isin(labels, (*clusters_to_plot, main_cluster_id))
    ].max()
    scale = 0.5 * (z_uniq[1] - z_uniq[0]) / max(7, all_max_ptp)

    # loop over clusters and plot waveforms
    for cid in (*clusters_to_plot, main_cluster_id):
        # masks and indices for these wfs
        in_cluster = labels == cid
        which = np.flatnonzero(in_cluster)
        choices = rg.choice(which, size=spikes_plot, replace=False)
        choices.sort()

        # indexing tools
        time_ix = spike_index[choices, 0, None] + t_range[None, :]
        chan_ix = channels_plot[None, :]

        # get waveforms
        if raw_bin is not None:
            raw_data = np.memmap(raw_bin, dtype=np.float32)
            raw_data = raw_data.reshape(-1, 384)
            waveforms = raw_data[time_ix, chan_ix]
        else:
            assert waveforms is not None
            waveforms = waveforms[
                choices,
                t_range[0] : t_range[1],
                which_chans_loc[spike_index[choices, 1]],
            ]

        # add residual?
        if residual_bin is not None:
            res_data = np.memmap(residual_bin, dtype=np.float32)
            res_data = res_data.reshape(-1, 384)
            waveforms = waveforms + res_data[time_ix, chan_ix]

        # scale
        waveforms *= scale

        # add geom locations
        waveforms += geom[channels_plot, 1][None, None, :]
        times = times_plot[:, None] + geom[channels_plot, 0][None, :]
        times = np.broadcast_to(times[None, ...], waveforms.shape)

        # plot
        ax.plot(
            *(
                l
                for x in zip(
                    waveforms.transpose(0, 2, 1), times.transpose(0, 2, 1)
                )
                for l in x
            ),
            alpha=alpha,
            color=get_ccolor(cid),
        )


single_unit_mosaic = """\
abcpqr
abcpqr
abcxy.
abcijk
"""


def single_unit_summary(
    cluster_id,
    clusterer,
    labels,
    geom,
    kept_inds,
    x,
    z,
    maxptps,
    channel_index,
    spike_index,
    wfs_localized,
    wfs_subtracted,
    raw_bin,
    residual_bin,
    spikes_plot=100,
    num_rows=3,
):
    # 2 neighbor clusters
    cluster_centers = [
        clusterer.weighted_cluster_centroid(l)
        for l in np.setdiff1d(np.unique(labels), [-1])
    ]
    closest_clusters = np.argsort(
        cdist([cluster_centers[cluster_id]], cluster_centers)
    )[1 : 3]

    fig, axes = plt.subplot_mosaic(single_unit_mosaic)

    # -- waveform plots
    for ax, w, raw, res in zip(
        "abc",
        [None, wfs_subtracted, wfs_localized],
        [raw_bin, None, None],
        [None, residual_bin, None],
    ):
        plot_waveforms_geom(
            cluster_id,
            closest_clusters,
            labels,
            geom,
            channel_index,
            spike_index,
            maxptps,
            waveforms=w,
            raw_bin=raw,
            residual_bin=res,
            spikes_plot=spikes_plot,
            num_rows=num_rows,
            alpha=0.1,
            ax=axes[ax],
        )

    # -- scatter plots
    in_shown_clusters = np.flatnonzero(
        np.isin(labels, (*closest_clusters, cluster_id))
    )
    zlim = (z[in_shown_clusters].min(), z[in_shown_clusters].max())
    array_scatter(
        labels,
        geom,
        x,
        z,
        maxptps,
        zlim=zlim,
        axes=[axes["p"], axes["q"], axes["r"]],
    )

    # -- this unit stats
    in_main_cluster = np.flatnonzero(labels == cluster_id)
    s_cluster = spike_index[in_main_cluster, 0]
    t_cluster = s_cluster / 30000
    maxptp_cluster = maxptps[in_main_cluster]
    z_cluster = z[in_main_cluster]

    # ptp vs t plot
    axes["x"].plot(t_cluster, maxptp_cluster)
    axes["x"].set_xlabel("t (s)")
    axes["x"].set_ylabel("ptp")

    # ptp vs z plot
    axes["y"].scatter(z_cluster, maxptp_cluster)
    axes["x"].set_xlabel("z")
    axes["x"].set_ylabel("ptp")

    # ISI plot
    isi_ms = 1000 * np.diff(t_cluster)
    axes["k"].hist(isi_ms, bins=np.arange(11))

    # cross correlograms
    for ax, unit in zip("ij", closest_clusters):
        in_other = np.flatnonzero(labels == unit)
        sorting = NumpySorting.from_times_labels(
            times_list=np.r_[s_cluster, spike_index[in_other, 0]],
            labels_list=np.c_[
                np.zeros(len(s_cluster), dtype=int),
                np.ones(len(in_other), dtype=int),
            ],
            sampling_frequency=30000,
        )
        correlograms, bins = compute_correlograms(
            sorting, symmetrize=True, window_ms=10.0, bin_ms=1.0
        )
        axes[ax].bar(bins[1:], correlograms[0][1], width=1.0, align="center")
        axes[ax].set_xticks(bins[1:])
        axes[ax].set_xlabel("lag (ms)")
        axes[ax].set_title(f"cross corellogram {cluster_id} <-> {unit}")
