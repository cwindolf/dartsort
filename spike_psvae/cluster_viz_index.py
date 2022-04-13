import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from scipy.spatial.distance import cdist
from spikeinterface import NumpySorting
from spikeinterface.toolkit import compute_correlograms
from spikeinterface.comparison import compare_two_sorters
from matplotlib_venn import venn2

import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse


def get_ccolor(k):
    if k == -1:
        return "#808080"
    else:
        return cc.glasbey[k % len(cc.glasbey)]


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
        ax.scatter(xk, yk, s=s, color=color, alpha=alpha, marker=".")
        if k not in excluded_ids:
            x_mean, y_mean = xk.mean(), yk.mean()
            xycov = np.cov(xk, yk)
            means[k] = x_mean, y_mean
            covs[k] = xycov
            ax.annotate(str(k), (x_mean, y_mean), size=s)

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
        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(15, 15))

    cluster_scatter(
        x,
        z,
        labels,
        ax=axes[0],
        s=10,
        alpha=0.05,
    )
    axes[0].scatter(*geom.T, c="orange", marker="s", s=10)
    axes[0].set_ylabel("z")
    axes[0].set_xlabel("x")

    cluster_scatter(
        maxptp,
        z,
        labels,
        ax=axes[1],
        s=10,
        alpha=0.05,
    )
    axes[1].set_xlabel("maxptp")
    axes[2].scatter(
        x,
        z,
        c=np.clip(maxptp, 3, 15),
        alpha=0.1,
        marker=".",
        cmap=plt.cm.viridis,
    )
    axes[2].scatter(*geom.T, c="orange", marker="s", s=10)
    axes[2].set_title("ptps")
    axes[0].set_ylim(zlim)

    if fig is not None:
        plt.tight_layout()

    return fig


def plot_waveforms_geom(
    main_cluster_id,
    neighbor_clusters,
    labels,
    geom,
    channel_index,
    spike_index,
    maxptps,
    all_waveforms=None,
    raw_bin=None,
    residual_bin=None,
    spikes_plot=100,
    t_range=(22, 72),
    num_rows=3,
    alpha=0.1,
    h_shift=0,
    do_mean=False,
    ax=None,
    colors=None,
):
    ax = ax or plt.gca()

    # what channels will we plot?
    vals, counts = np.unique(
        spike_index[np.flatnonzero(labels == main_cluster_id), 1],
        return_counts=True,
    )
    z_uniq, z_ids = np.unique(geom[:, 1], return_inverse=True)
    mcid = z_ids[vals[counts.argmax()]]
    channels_plot = np.flatnonzero((z_ids >= mcid - 3) & (z_ids <= mcid + 3))

    # how to scale things?
    all_max_ptp = maxptps[
        np.isin(labels, (*neighbor_clusters, main_cluster_id))
    ].max()
    scale = (z_uniq[1] - z_uniq[0]) / max(7, all_max_ptp)

    times_plot = np.arange(t_range[0] - 42, t_range[1] - 42).astype(float)
    x_uniq = np.unique(geom[:, 0])
    times_plot *= (x_uniq[1] - x_uniq[0]) / np.abs(times_plot).max()

    # scatter the channels
    ax.scatter(*geom[channels_plot].T, c="orange", marker="s")
    for c in channels_plot:
        ax.annotate(c, (geom[c, 0], geom[c, 1]))

    if raw_bin:
        raw_data = np.memmap(raw_bin, dtype=np.float32)
        raw_data = raw_data.reshape(-1, 384)

    if residual_bin:
        res_data = np.memmap(residual_bin, dtype=np.float32)
        res_data = res_data.reshape(-1, 384)

    for j, cluster_id in enumerate((main_cluster_id, *neighbor_clusters)):
        if colors is None:
            color = get_ccolor(cluster_id)
        else:
            color = colors[j]
        in_cluster = np.flatnonzero(labels == cluster_id)
        num_plot_cluster = min(len(in_cluster), spikes_plot)
        some_in_cluster = np.random.default_rng(0).choice(
            in_cluster, replace=False, size=num_plot_cluster
        )
        some_in_cluster.sort()

        if raw_bin:
            spike_times = spike_index[some_in_cluster][:, 0]
            waveforms = []
            for t in spike_times:
                waveforms.append(
                    raw_data[t - 42 : t + 79, channels_plot].copy()
                )
            waveforms = np.asarray(waveforms)
        else:
            waveforms = all_waveforms[some_in_cluster]

        if residual_bin:
            spike_times = spike_index[some_in_cluster][:, 0]
            residuals = []
            for t in spike_times:
                residuals.append(
                    res_data[t - 42 : t + 79, channels_plot].copy()
                )
            residuals = np.asarray(residuals)
        if do_mean:
            waveforms = np.expand_dims(np.mean(waveforms, axis=0), 0)

        vertical_lines = set()
        draw_lines = []
        for i in range(num_plot_cluster):
            if raw_bin:
                wf_chans = channels_plot
            else:
                wf_chans = channel_index[spike_index[some_in_cluster[i], 1]]

            for k, channel in enumerate(channels_plot):
                if channel in wf_chans:
                    trace = waveforms[
                        i,
                        t_range[0] : t_range[1],
                        np.flatnonzero(np.isin(wf_chans, channel))[0],
                    ]
                else:
                    continue
                if residual_bin:
                    trace += residuals[i, t_range[0] : t_range[1], k]

                waveform = trace * scale
                draw_lines.append(geom[channel, 0] + times_plot)
                draw_lines.append(waveform + geom[channel, 1])
                max_vert_line = geom[channel, 0]
                if max_vert_line not in vertical_lines:
                    vertical_lines.add(max_vert_line)
                    ax.axvline(max_vert_line, linestyle="--")
        ax.plot(
            *draw_lines,
            alpha=alpha,
            c=color,
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
        cdist([cluster_centers[cluster_id]], cluster_centers)[0]
    )[1:3]

    fig, axes = plt.subplot_mosaic(single_unit_mosaic, figsize=(15, 10))
    axes["a"].get_shared_y_axes().join(axes["a"], axes["b"])
    axes["a"].get_shared_y_axes().join(axes["a"], axes["c"])
    axes["p"].get_shared_y_axes().join(axes["p"], axes["q"])
    axes["p"].get_shared_y_axes().join(axes["p"], axes["r"])

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
            all_waveforms=w,
            raw_bin=raw,
            residual_bin=res,
            spikes_plot=spikes_plot,
            num_rows=num_rows,
            alpha=0.1,
            ax=axes[ax],
        )
    axes["b"].set_yticks([])
    axes["c"].set_yticks([])

    # -- scatter plots
    in_shown_clusters = np.flatnonzero(
        np.isin(labels, (*closest_clusters, cluster_id))
    )
    zlim = (z[in_shown_clusters].min(), z[in_shown_clusters].max())
    array_scatter(
        labels[in_shown_clusters],
        geom,
        x[in_shown_clusters],
        z[in_shown_clusters],
        maxptps[in_shown_clusters],
        zlim=zlim,
        axes=[axes["p"], axes["q"], axes["r"]],
    )
    axes["q"].set_yticks([])
    axes["r"].set_yticks([])

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
            labels_list=np.r_[
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

    return fig


def plot_agreement_venn(
    cluster_id1,
    geom,
    channel_index,
    spike_index1,
    spike_index2,
    name1,
    name2,
    raw_bin,
    maxptps1,
    maxptps2,
    match_score=0.5,
    spikes_plot=100,
    delta_frames=12,
):
    # make spikeinterface objects
    sorting1 = NumpySorting(
        times_list=spike_index1[:, 0],
        labels_list=spike_index1[:, 1],
        sampling_frequency=30000,
    )
    sorting2 = NumpySorting(
        times_list=spike_index2[:, 0],
        labels_list=spike_index2[:, 1],
        sampling_frequency=30000,
    )
    comp = compare_two_sorters(
        sorting1,
        sorting2,
        sorting1_name=name1,
        sorting2_name=name2,
        match_score=0.5,
    )

    # get best match
    match2 = comp.get_best_unit_match1(cluster_id1)
    if not (match2 and match2 > -1):
        print("match2", match2)
        return
    match_frac = comp.get_agreement_fraction(cluster_id1, match2)

    (
        ind_st1,
        ind_st2,
        not_match_ind_st1,
        not_match_ind_st2,
    ) = compute_spiketrain_agreement(cluster_id1, match2, delta_frames)
    fig = plt.figure(figsize=(24, 12))
    grid = (1, 3)
    ax_venn = plt.subplot2grid(grid, (0, 0))
    ax_sorting1 = plt.subplot2grid(grid, (0, 1))
    ax_sorting2 = plt.subplot2grid(grid, (0, 2))

    subsets = [len(not_match_ind_st1), len(not_match_ind_st2), len(ind_st1)]
    v = venn2(
        subsets=subsets,
        set_labels=["unit{}".format(cluster_id1), "unit{}".format(match2)],
        ax=ax_venn,
    )
    v.get_patch_by_id("10").set_color("red")
    v.get_patch_by_id("01").set_color("blue")
    v.get_patch_by_id("11").set_color("goldenrod")
    ax_venn.set_title(
        f"{name1}{cluster_id1} + {name2}{match2}, "
        f"{match_frac.round(2)*100}% agreement"
    )

    # fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,12))
    n_match_1 = len(ind_st1)
    n_unmatch_1 = len(not_match_ind_st1)
    match_unmatch_labels = np.r_[
        np.zeros(n_match_1, dtype=int),
        np.ones(n_unmatch_1, dtype=int),
    ]
    match_unmatch_spike_index = np.r_[
        spike_index1[ind_st1],
        spike_index1[not_match_ind_st1],
    ]
    match_unmatch_maxptps = np.r_[
        maxptps1[ind_st1],
        maxptps1[not_match_ind_st1],
    ]

    plot_waveforms_geom(
        0,
        [1],
        match_unmatch_labels,
        geom,
        channel_index,
        match_unmatch_spike_index,
        match_unmatch_maxptps,
        raw_bin=raw_bin,
        spikes_plot=spikes_plot,
        num_rows=3,
        ax=ax_sorting1,
        colors=["goldenrod", "red"],
    )

    n_match_2 = len(ind_st2)
    n_unmatch_2 = len(not_match_ind_st2)
    match_unmatch_labels = np.r_[
        np.zeros(n_match_2, dtype=int),
        np.ones(n_unmatch_2, dtype=int),
    ]
    match_unmatch_spike_index = np.r_[
        spike_index2[ind_st2],
        spike_index2[not_match_ind_st2],
    ]
    match_unmatch_maxptps = np.r_[
        maxptps2[ind_st2],
        maxptps2[not_match_ind_st2],
    ]

    plot_waveforms_geom(
        0,
        [1],
        match_unmatch_labels,
        geom,
        channel_index,
        match_unmatch_spike_index,
        match_unmatch_maxptps,
        raw_bin=raw_bin,
        spikes_plot=spikes_plot,
        num_rows=3,
        ax=ax_sorting1,
        colors=["goldenrod", "blue"],
    )

    return fig


def compute_spiketrain_agreement(st_1, st_2, delta_frames=12):
    # create figure for each match
    times_concat = np.concatenate((st_1, st_2))
    membership = np.concatenate(
        (np.ones(st_1.shape) * 1, np.ones(st_2.shape) * 2)
    )
    indices = times_concat.argsort()
    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    diffs = times_concat_sorted[1:] - times_concat_sorted[:-1]
    inds = np.where(
        (diffs <= delta_frames)
        & (membership_sorted[:-1] != membership_sorted[1:])
    )[0]
    if len(inds) > 0:
        inds2 = inds[np.where(inds[:-1] + 1 != inds[1:])[0]] + 1
        inds2 = np.concatenate((inds2, [inds[-1]]))
        times_matched = times_concat_sorted[inds2]
        # # find and label closest spikes
        ind_st1 = np.array(
            [np.abs(st_1 - tm).argmin() for tm in times_matched]
        )
        ind_st2 = np.array(
            [np.abs(st_2 - tm).argmin() for tm in times_matched]
        )
        not_match_ind_st1 = np.ones(st_1.shape[0], bool)
        not_match_ind_st1[ind_st1] = False
        not_match_ind_st1 = np.where(not_match_ind_st1)[0]
        not_match_ind_st2 = np.ones(st_2.shape[0], bool)
        not_match_ind_st2[ind_st2] = False
        not_match_ind_st2 = np.where(not_match_ind_st2)[0]
    else:
        ind_st1 = np.asarray([]).astype("int")
        ind_st2 = np.asarray([]).astype("int")
        not_match_ind_st1 = np.asarray([]).astype("int")
        not_match_ind_st2 = np.asarray([]).astype("int")

    return ind_st1, ind_st2, not_match_ind_st1, not_match_ind_st2
