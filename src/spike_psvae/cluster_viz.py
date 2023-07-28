# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import colorcet
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# %%
# matplotlib.use('Agg')
from matplotlib_venn import venn2
from spikeinterface.extractors import NumpySorting
from spikeinterface.postprocessing import compute_correlograms
from spikeinterface.comparison import compare_two_sorters
from spikeinterface.widgets import plot_agreement_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

# %%
from spike_psvae.spikeio import read_waveforms
from spike_psvae.cluster_utils import (
    compute_spiketrain_agreement,
    get_unit_similarities,
    get_closest_clusters_kilosort,
    get_closest_clusters_hdbscan,
)
from spike_psvae.denoise import denoise_wf_nn_tmp_single_channel
import colorcet as cc

# %%
plt.rcParams["axes.xmargin"] = 0
matplotlib.rcParams.update({"font.size": 10})
plt.rcParams["axes.ymargin"] = 0
ccolors = cc.glasbey[:31]


# %%
def get_ccolor(k):
    if k == -1:
        return "#808080"
    else:
        return ccolors[k % len(ccolors)]


# %%
def cluster_scatter(
    xs,
    ys,
    ids,
    ax=None,
    n_std=2.0,
    excluded_ids={-1},
    s=1,
    alpha=0.5,
    do_ellipse=True,
    fontsize=None,
    linewidth_ellipse=1.,
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
            if do_ellipse:
                x_mean, y_mean = xk.mean(), yk.mean()
                if fontsize is None:
                    ax.annotate(str(k), (x_mean, y_mean))
                else:
                    ax.annotate(str(k), (x_mean, y_mean), fontsize=fontsize)
                xycov = np.cov(xk, yk)
                means[k] = x_mean, y_mean
                covs[k] = xycov

    if not do_ellipse:
        return

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
            linewidth=linewidth_ellipse,
        )
        transform = (
            transforms.Affine2D()
            .rotate_deg(45)
            .scale(n_std * np.sqrt(vx), n_std * np.sqrt(vy))
            .translate(mean_x, mean_y)
        )
        ell.set_transform(transform + ax.transData)
        ax.add_patch(ell)


# %%
def array_scatter(
    labels,
    geom,
    x,
    z,
    maxptp,
    zlim=(-50, 3900),
    xlim=None,
    ptplim=None,
    axes=None,
    do_ellipse=True,
    figsize = (15, 15),
):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3, sharey=True, figsize=figsize)

    excluded_ids = {-1}
    if not do_ellipse:
        excluded_ids = np.unique(labels)

    cluster_scatter(
        x,
        z,
        labels,
        ax=axes[0],
        s=10,
        alpha=0.05,
        excluded_ids=excluded_ids,
        do_ellipse=do_ellipse,
    )
    axes[0].scatter(*geom.T, c="orange", marker="s", s=10)
    axes[0].scatter(geom[0, 0], geom[0, 1], c="orange", marker="s", s=10, label='Channel Locations')
    axes[0].set_ylabel("Registered Depth (um)", fontsize=14)
    axes[0].set_xlabel("x (um)", fontsize=14)
    axes[0].legend(fontsize=14, loc='upper left')
    axes[0].tick_params(axis='x', labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)

    cluster_scatter(
        maxptp,
        z,
        labels,
        ax=axes[1],
        s=10,
        alpha=0.05,
        excluded_ids=excluded_ids,
        do_ellipse=do_ellipse,
    )
    axes[1].set_xlabel("Amplitude (s.u.)", fontsize=16)
    axes[1].tick_params(axis='x', labelsize=16)
    
    axes[2].scatter(
        x,
        z,
        c=np.clip(maxptp, 3, 15),
        alpha=0.1,
        marker=".",
        cmap=plt.cm.jet,
    )
    axes[2].scatter(*geom.T, c="orange", marker="s", s=10)
    axes[2].set_title("ptps")
    axes[0].set_ylim(zlim)
    axes[1].set_ylim(zlim)
    axes[2].set_ylim(zlim)
    if xlim is not None:
        axes[0].set_xlim(xlim)
    if ptplim is not None:
        axes[1].set_xlim(ptplim)

    if fig is not None:
        plt.tight_layout()

    return fig, axes


# %%
def array_scatter_with_deconv_score_fading(
    labels,
    geom,
    x,
    z,
    maxptp,
    log_dist_metric,
    disp, 
    labels_temp_comp,
    x_temp_comp,
    z_temp_comp,
    labels_temp_comp_fading,
    x_temp_comp_fading,
    z_temp_comp_fading,
    labels_fading,
    x_fading,
    z_fading,
    maxptp_fading,
    log_dist_metric_fading,
    time_start,
    time_end,
    zlim=(-50, 350),
    xlim=(-30, 70),
    ptps_lim=(0, 50),
    axes=None,
    do_ellipse=True,
    alpha=1,
    alpha_fading=0.05,
    color_map=plt.cm.jet,
):
    
    geom_shifted = geom - [0, disp[time_start:time_end].mean()]
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 5, sharey=True, figsize=(25, 15))

    excluded_ids = {-1}
    if not do_ellipse:
        excluded_ids = np.unique(labels)

    axes[0].text(xlim[0]+5, zlim[1]-10, "Time {} s".format(time_start), fontsize='large')
    cluster_scatter(
        x,
        z,
        labels,
        ax=axes[0],
        s=10,
        alpha=alpha,
        excluded_ids=excluded_ids,
        do_ellipse=False,
    )
    cluster_scatter(
        x_fading,
        z_fading,
        labels_fading,
        ax=axes[0],
        s=10,
        alpha=alpha_fading,
        excluded_ids=excluded_ids,
        do_ellipse=do_ellipse,
    )
    axes[0].scatter(*geom_shifted.T, c="orange", marker="s", s=10)
    axes[0].set_ylabel("z")
    axes[0].set_xlabel("x")
    
    axes[1].scatter(
        x,
        z,
        c=log_dist_metric,
        alpha=alpha,
        marker=".",
        cmap=color_map,
    )
    
    axes[1].scatter(
        x_fading,
        z_fading,
        c=log_dist_metric_fading,
        alpha=alpha_fading,
        marker=".",
        cmap=color_map
    )
    axes[1].scatter(*geom_shifted.T, c="orange", marker="s", s=10)
    axes[1].set_title("Deconv Score")
    
    cluster_scatter(
        x_temp_comp,
        z_temp_comp,
        labels_temp_comp,
        ax=axes[2],
        s=10,
        alpha=alpha,
        excluded_ids=excluded_ids,
        do_ellipse=False,
    )
    cluster_scatter(
        x_temp_comp_fading,
        z_temp_comp_fading,
        labels_temp_comp_fading,
        ax=axes[2],
        s=10,
        alpha=alpha_fading,
        excluded_ids=excluded_ids,
        do_ellipse=do_ellipse,
    )
    axes[2].scatter(*geom_shifted.T, c="orange", marker="s", s=10)
    axes[2].set_xlabel("x")
    axes[2].set_title("Spikes Temp Computation")

    cluster_scatter(
        maxptp,
        z,
        labels,
        ax=axes[3],
        s=10,
        alpha=alpha,
        excluded_ids=excluded_ids,
        do_ellipse=False,
    )
    
    cluster_scatter(
        maxptp_fading,
        z_fading,
        labels_fading,
        ax=axes[3],
        s=10,
        alpha=alpha_fading,
        excluded_ids=excluded_ids,
        do_ellipse=do_ellipse,
    )
    axes[3].set_xlabel("maxptp")
    
    axes[4].scatter(
        x,
        z,
        c=np.clip(maxptp, 3, 15),
        alpha=alpha,
        marker=".",
        cmap=color_map
    )
    axes[4].scatter(
        x_fading,
        z_fading,
        c=np.clip(maxptp_fading, 3, 15),
        alpha=alpha_fading,
        marker=".",
        cmap=color_map
    )
    
    axes[4].scatter(*geom_shifted.T, c="orange", marker="s", s=10)
    axes[4].set_title("ptps")

    axes[0].set_ylim(zlim)
    axes[1].set_ylim(zlim)
    axes[2].set_ylim(zlim)
    axes[3].set_ylim(zlim)
    axes[4].set_ylim(zlim)

    axes[0].set_xlim(xlim)
    axes[1].set_xlim(xlim)
    axes[2].set_xlim(xlim)
    axes[4].set_xlim(xlim)
    axes[3].set_xlim(ptps_lim)

    if fig is not None:
        plt.tight_layout()

    return fig, axes


# %%
def array_scatter_with_deconv_score(
    labels,
    geom,
    x,
    z,
    maxptp,
    log_dist_metric,
    disp, 
    time_start,
    time_end,
    zlim=(-50, 350),
    xlim=(-30, 70),
    axes=None,
    do_ellipse=True,
    alpha=0.1,
):
    
    geom_shifted = geom - [0, disp[time_start:time_end].mean()]
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 4, sharey=True, figsize=(20, 15))

    excluded_ids = {-1}
    if not do_ellipse:
        excluded_ids = np.unique(labels)

    cluster_scatter(
        x,
        z,
        labels,
        ax=axes[0],
        s=10,
        alpha=alpha,
        excluded_ids=excluded_ids,
        do_ellipse=do_ellipse,
    )
    axes[0].scatter(*geom_shifted.T, c="orange", marker="s", s=10)
    axes[0].set_ylabel("z")
    axes[0].set_xlabel("x")
    
    axes[1].scatter(
        x,
        z,
        c=log_dist_metric,
        alpha=alpha,
        marker=".",
        cmap=plt.cm.viridis,
    )
    axes[1].scatter(*geom_shifted.T, c="orange", marker="s", s=10)
    axes[1].set_title("Deconv Score")

    cluster_scatter(
        maxptp,
        z,
        labels,
        ax=axes[2],
        s=10,
        alpha=alpha,
        excluded_ids=excluded_ids,
        do_ellipse=do_ellipse,
    )
    axes[2].set_xlabel("maxptp")
    
    axes[3].scatter(
        x,
        z,
        c=np.clip(maxptp, 3, 15),
        alpha=alpha,
        marker=".",
        cmap=plt.cm.viridis,
    )
    axes[3].scatter(*geom_shifted.T, c="orange", marker="s", s=10)
    axes[3].set_title("ptps")

    axes[0].set_ylim(zlim)
    axes[1].set_ylim(zlim)
    axes[2].set_ylim(zlim)
    axes[3].set_ylim(zlim)

    axes[0].set_xlim(xlim)
    axes[1].set_xlim(xlim)
    axes[3].set_xlim(xlim)

    if fig is not None:
        plt.tight_layout()

    return fig, axes


# %%
def array_scatter_with_pcs(
    labels,
    geom,
    x,
    z,
    maxptp,
    pcs1,
    pcs2,
    zlim=(-50, 3900),
    axes=None,
    do_ellipse=True,
):
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 5, sharey=True, figsize=(25, 15))

    excluded_ids = {-1}
    if not do_ellipse:
        excluded_ids = np.unique(labels)

    cluster_scatter(
        x,
        z,
        labels,
        ax=axes[0],
        s=10,
        alpha=0.05,
        excluded_ids=excluded_ids,
        do_ellipse=do_ellipse,
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
        excluded_ids=excluded_ids,
        do_ellipse=do_ellipse,
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

    cluster_scatter(
        pcs1,
        z,
        labels,
        ax=axes[3],
        s=10,
        alpha=0.05,
        excluded_ids=excluded_ids,
        do_ellipse=do_ellipse,
    )
    axes[3].set_xlabel("PC1")
    
    cluster_scatter(
        pcs2,
        z,
        labels,
        ax=axes[4],
        s=10,
        alpha=0.05,
        excluded_ids=excluded_ids,
        do_ellipse=do_ellipse,
    )
    axes[4].set_xlabel("PC2")
    
    axes[0].set_ylim(zlim)
    axes[1].set_ylim(zlim)
    axes[2].set_ylim(zlim)
    axes[3].set_ylim(zlim)
    axes[4].set_ylim(zlim)

    if fig is not None:
        plt.tight_layout()

    return fig, axes


# %%
def plot_waveforms_geom_unit(
    geom,
    first_chans_cluster,
    mcs_abs_cluster,
    spike_times,
    max_ptps_cluster=None,
    raw_bin=None,
    residual_bin=None,
    waveforms_cluster=None,
    denoiser=None,
    device=None,
    num_spikes_plot=100,
    t_range=(30, 90),
    num_channels=40,
    num_rows=3,
    alpha=0.1,
    h_shift=0,
    subset_indices=None,
    scale=None,
    do_mean=False,
    annotate=False,
    ax=None,
    color="blue",
):
    ax = ax or plt.gca()
    some_in_cluster = np.random.default_rng(0).choice(
        list(range(len(spike_times))),
        replace=False,
        size=min(len(spike_times), num_spikes_plot),
    )
    first_chans_cluster = first_chans_cluster[some_in_cluster]
    mcs_abs_cluster = mcs_abs_cluster[some_in_cluster]
    spike_times = spike_times[some_in_cluster]

    # what channels will we plot?
    vals, counts = np.unique(
        mcs_abs_cluster,
        return_counts=True,
    )
    z_uniq, z_ids = np.unique(geom[:, 1], return_inverse=True)
    mcid = z_ids[vals[counts.argmax()]]
    channels_plot = np.flatnonzero(
        (z_ids >= mcid - num_rows) & (z_ids <= mcid + num_rows)
    )

    # how to scale things?
    if scale is None:
        if max_ptps_cluster is not None:
            max_ptps_cluster = max_ptps_cluster[some_in_cluster]
            all_max_ptp = max_ptps_cluster.max()
            scale = (z_uniq[1] - z_uniq[0]) / max(7, all_max_ptp)
        else:
            scale = 7

    times_plot = np.arange(t_range[0] - 42, t_range[1] - 42).astype(float)
    x_uniq = np.unique(geom[:, 0])
    times_plot *= (x_uniq[1] - x_uniq[0]) / np.abs(times_plot).max()

    # scatter the channels
    ax.scatter(*geom[channels_plot].T, c="orange", marker="s")
    if annotate:
        for c in channels_plot:
            ax.annotate(c, (geom[c, 0], geom[c, 1]))

    if raw_bin is not None:
        # raw data and spike times passed in
        waveforms_read = read_waveforms(
            spike_times, raw_bin, geom.shape[0], spike_length_samples=121
        )[0]
        waveforms = []
        for i, waveform in enumerate(waveforms_read):
            waveforms.append(
                waveform[
                    :,
                    int(first_chans_cluster[i]) : int(first_chans_cluster[i])
                    + num_channels,
                ].copy()
            )
        waveforms = np.asarray(waveforms)
    elif waveforms_cluster is None:
        # no raw data and no waveforms passed - bad!
        raise ValueError("need to input raw_bin or waveforms")
    else:
        # waveforms passed in
        waveforms = waveforms_cluster[some_in_cluster]
        if residual_bin is not None:
            # add residuals
            residuals_read = read_waveforms(
                spike_times, residual_bin, geom.shape[0], spike_length_samples=121
            )[0]
            residuals = []
            for i, residual in enumerate(residuals_read):
                residuals.append(
                    residual[
                        :,
                        int(first_chans_cluster[i]) : int(
                            first_chans_cluster[i]
                        )
                        + num_channels,
                    ]
                )
            residuals = np.asarray(residuals)
            waveforms = waveforms + residuals
    if denoiser is not None and device is not None:
        # denoise waveforms
        waveforms = denoise_wf_nn_tmp_single_channel(
            waveforms, denoiser, device
        )
    if do_mean:
        # plot the mean rather than the invididual spikes
        waveforms = np.expand_dims(np.mean(waveforms, axis=0), 0)
    draw_lines = []
    for i in range(min(len(waveforms), num_spikes_plot)):
        for k, channel in enumerate(
            range(
                int(first_chans_cluster[i]),
                int(first_chans_cluster[i]) + waveforms.shape[2],
            )
        ):
            if channel in channels_plot:
                trace = waveforms[
                    i,
                    t_range[0] : t_range[1],
                    k,
                ]
            else:
                continue
            waveform = trace * scale
            draw_lines.append(geom[channel, 0] + times_plot + h_shift)
            draw_lines.append(waveform + geom[channel, 1])
    ax.plot(
        *draw_lines,
        alpha=alpha,
        c=color,
    )


# %%
def plot_waveforms_geom(
    main_cluster_id,
    neighbor_clusters,
    labels,
    spike_index,
    firstchans,
    maxptps,
    geom,
    raw_bin=None,
    residual_bin=None,
    waveforms=None,
    denoiser=None,
    device=None,
    num_spikes_plot=100,
    t_range=(30, 90),
    num_channels=40,
    num_rows=3,
    alpha=0.1,
    annotate=False,
    colors=None,
    do_mean=False,
    scale=None,
    ax=None,
):
    ax = ax or plt.gca()

    # what channels will we plot?
    vals, counts = np.unique(
        spike_index[np.flatnonzero(labels == main_cluster_id), 1],
        return_counts=True,
    )
    z_uniq, z_ids = np.unique(geom[:, 1], return_inverse=True)
    mcid = z_ids[vals[counts.argmax()]]
    channels_plot = np.flatnonzero(
        (z_ids >= mcid - num_rows) & (z_ids <= mcid + num_rows)
    )

    # how to scale things?
    if scale is None:
        all_max_ptp = maxptps[
            np.isin(labels, (*neighbor_clusters, main_cluster_id))
        ].max()
        scale = (z_uniq[1] - z_uniq[0]) / max(7, all_max_ptp)

    times_plot = np.arange(t_range[0] - 42, t_range[1] - 42).astype(float)
    x_uniq = np.unique(geom[:, 0])
    times_plot *= (x_uniq[1] - x_uniq[0]) / np.abs(times_plot).max()

    # scatter the channels
    ax.scatter(*geom[channels_plot].T, c="orange", marker="s")
    if annotate:
        for c in channels_plot:
            ax.annotate(c, (geom[c, 0], geom[c, 1]))

    # plot each cluster
    for j, cluster_id in reversed(
        list(enumerate((main_cluster_id, *neighbor_clusters)))
    ):
        if colors is None:
            color = get_ccolor(cluster_id)
        else:
            color = colors[j]
        in_cluster = np.flatnonzero(labels == cluster_id)
        some_in_cluster = np.random.default_rng(0).choice(
            in_cluster,
            replace=False,
            size=min(len(in_cluster), num_spikes_plot),
        )
        some_in_cluster.sort()
        first_chans_cluster = firstchans[some_in_cluster]
        spike_times = spike_index[:, 0][some_in_cluster]
        if raw_bin is not None:
            # raw data and spike times passed in
            waveforms_read = read_waveforms(
                spike_times, raw_bin, geom.shape[0], spike_length_samples=121
            )[0]
            waveforms_in_cluster = []
            for i, waveform in enumerate(waveforms_read):
                waveforms_in_cluster.append(
                    waveform[
                        :,
                        int(first_chans_cluster[i]) : int(
                            first_chans_cluster[i]
                        )
                        + num_channels,
                    ].copy()
                )
            waveforms_in_cluster = np.asarray(waveforms_in_cluster)
        elif waveforms is None:
            # no raw data and no waveforms passed - bad!
            raise ValueError("need to input raw_bin or waveforms")
        else:
            # waveforms passed in
            waveforms_in_cluster = waveforms[some_in_cluster]
            if residual_bin is not None:
                # add residuals
                residuals_read = read_waveforms(
                    spike_times, residual_bin, geom.shape[0], spike_length_samples=121
                )[0]
                residuals = []
                for i, residual in enumerate(residuals_read):
                    residuals.append(
                        residual[
                            :,
                            int(first_chans_cluster[i]) : int(
                                first_chans_cluster[i]
                            )
                            + num_channels,
                        ]
                    )
                residuals = np.asarray(residuals)
                waveforms_in_cluster = waveforms_in_cluster + residuals
        if denoiser is not None and device is not None:
            # denoise waveforms
            waveforms_in_cluster = denoise_wf_nn_tmp_single_channel(
                waveforms_in_cluster, denoiser, device
            )
        if do_mean:
            # plot the mean rather than the invididual spikes
            waveforms_in_cluster = np.expand_dims(
                np.mean(waveforms_in_cluster, axis=0), 0
            )
        vertical_lines = set()
        draw_lines = []
        for i in range(min(len(waveforms_in_cluster), num_spikes_plot)):
            for k, channel in enumerate(
                range(
                    int(first_chans_cluster[i]),
                    int(first_chans_cluster[i])
                    + waveforms_in_cluster.shape[2],
                )
            ):
                if channel in channels_plot:
                    trace = waveforms_in_cluster[
                        i,
                        t_range[0] : t_range[1],
                        k,
                    ]
                else:
                    continue
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


# %%
def plot_venn_agreement(
    cluster_id_1,
    cluster_id_2,
    match_ind,
    not_match_ind_st1,
    not_match_ind_st2,
    ax=None,
):
    if ax is None:
        plt.figure(figsize=(12, 12))
        ax = plt.gca()
    lab_st1 = cluster_id_1
    lab_st2 = cluster_id_2
    subsets = [len(not_match_ind_st1), len(not_match_ind_st2), len(match_ind)]
    v = venn2(
        subsets=subsets,
        set_labels=["unit{}".format(lab_st1), "unit{}".format(lab_st2)],
        ax=ax,
    )
    v.get_patch_by_id("10").set_color("red")
    v.get_patch_by_id("01").set_color("blue")
    v.get_patch_by_id("11").set_color("goldenrod")
    sets = ["10", "11", "01"]
    return ax


# %%
def plot_self_agreement(labels, spike_times, fig=None):
    # matplotlib.rcParams.update({'font.size': 22})
    indices_list = []
    labels_list = []
    for cluster_id in np.unique(labels):
        label_ids = np.where(labels == cluster_id)
        indices = spike_times[label_ids]
        num_labels = label_ids[0].shape[0]
        indices_list.append(indices)
        labels_list.append((np.zeros(num_labels) + cluster_id).astype("int"))
    sorting = NumpySorting.from_times_labels(
        times_list=np.concatenate(indices_list),
        labels_list=np.concatenate(labels_list),
        sampling_frequency=30000,
    )
    sorting_comparison = compare_two_sorters(sorting, sorting)
    if fig is None:
        fig = plt.figure(figsize=(36, 36))
    plot_agreement_matrix(sorting_comparison, figure=fig, ordered=False)
    return fig


# %%
def plot_isi_distribution(spike_train, ax=None, cdf=True, bins=None):
    if ax is None:
        fig = plt.figure(figsize=(6, 3))
        ax = fig.gca()
    ax.set_xlabel("isi (ms)")
    ax.set_xticks(np.arange(0, 11, 2.5))
    ax.set_xlim([-0.01, 10.1])
    ax.set_ylabel("mass")

    spike_train_diff = np.diff(spike_train) / 30000
    spike_train_diff = spike_train_diff[np.where(spike_train_diff < 0.01)]
    spike_train_diff = spike_train_diff * 1000  # convert 1/10 to ms
    if np.all(spike_train_diff > 10):
        return ax

    if bins is None:
        bins = np.arange(0, 10.1, 0.5)
    elif isinstance(bins, float):
        bins = np.arange(0, 10.1, bins)
    y, x, _ = ax.hist(
        spike_train_diff, bins=bins, density=False
    )
    if cdf:
        try:
            sns.ecdfplot(spike_train_diff, ax=ax)
        except KeyError as e:
            print("Ignoring seaborn error", e)
        #     sns.distplot(spike_train_diff)

    return ax


# %%
def plot_single_unit_summary(
    cluster_id,
    labels,
    spike_index,
    cluster_centers,
    geom,
    xs,
    zs,
    maxptps,
    firstchans,
    wfs_full_denoise,
    wfs_subtracted,
    raw_bin,
    residual_bin,
    num_spikes_plot=100,
    num_rows_plot=3,
    t_range=(30, 90),
    plot_all_points=False,
    num_channels=40,
):
    matplotlib.rcParams.update({"font.size": 30})
    label_indices = np.where(labels == cluster_id)

    closest_clusters = get_closest_clusters_hdbscan(
        cluster_id, cluster_centers, num_close_clusters=2
    )
    # scales = (1,10,1,15,30) #predefined scales for each feature
    features = np.concatenate(
        (
            np.expand_dims(xs, 1),
            np.expand_dims(zs, 1),
            np.expand_dims(maxptps, 1),
        ),
        axis=1,
    )

    close_labels_indices = np.where(
        (labels == cluster_id)
        | (labels == closest_clusters[0])
        | (labels == closest_clusters[1])
    )
    all_cluster_features_close = features[close_labels_indices]
    all_labels_close = labels[close_labels_indices]
    # print(all_labels_close)

    # buffers for range of scatter plots
    z_buffer = 5
    x_buffer = 5
    ptp_cutoff = 0.5

    z_cutoff = (
        np.min(all_cluster_features_close[:, 1] - z_buffer),
        np.max(all_cluster_features_close[:, 1] + z_buffer),
    )
    x_cutoff = (
        np.min(all_cluster_features_close[:, 0] - x_buffer),
        np.max(all_cluster_features_close[:, 0] + x_buffer),
    )
    ptps_cutoff = (
        np.min(all_cluster_features_close[:, 2] - ptp_cutoff),
        np.max(all_cluster_features_close[:, 2] + ptp_cutoff),
    )

    if plot_all_points:
        plot_features = features
        plot_labels = labels
        cm_plot = np.clip(maxptps, 3, 15)
    else:
        plot_features = all_cluster_features_close
        plot_labels = all_labels_close
        cm_plot = np.clip(maxptps, 3, 15)[close_labels_indices]

    fig = plt.figure(figsize=(24 + 18 * 3, 36))
    grid = (6, 6)
    ax_raw = plt.subplot2grid(grid, (0, 0), rowspan=6)
    ax_cleaned = plt.subplot2grid(grid, (0, 1), rowspan=6)
    ax_denoised = plt.subplot2grid(grid, (0, 2), rowspan=6)
    ax_ptp = plt.subplot2grid(grid, (3, 3))
    ax_ptp_z = plt.subplot2grid(grid, (3, 4))
    ax_xcorr1 = plt.subplot2grid(grid, (5, 3))
    ax_xcorr2 = plt.subplot2grid(grid, (5, 4))
    ax_isi = plt.subplot2grid(grid, (5, 5))
    ax_scatter_xz = plt.subplot2grid(grid, (0, 3), rowspan=3)
    ax_scatter_sptpz = plt.subplot2grid(grid, (0, 4), rowspan=3)
    ax_scatter_xzptp = plt.subplot2grid(grid, (0, 5), rowspan=3)

    ax = ax_scatter_xz
    xs, zs, ids = plot_features[:, 0], plot_features[:, 1], plot_labels
    cluster_scatter(
        xs, zs, ids, ax=ax, excluded_ids=set([-1]), s=200, alpha=0.3
    )
    ax.scatter(geom[:, 0], geom[:, 1], s=100, c="orange", marker="s")
    ax.set_title(f"x vs. z")
    ax.set_ylabel("z")
    ax.set_xlabel("x")
    ax.set_ylim(z_cutoff)
    ax.set_xlim(x_cutoff)

    ax = ax_scatter_sptpz
    ys, zs, ids = plot_features[:, 2], plot_features[:, 1], plot_labels
    cluster_scatter(
        ys, zs, ids, ax=ax, excluded_ids=set([-1]), s=200, alpha=0.3
    )
    ax.set_title(f"ptp vs. z")
    ax.set_xlabel("ptp")
    ax.set_yticks([])
    ax.set_ylim(z_cutoff)
    ax.set_xlim(ptps_cutoff)

    ax = ax_scatter_xzptp
    ax.scatter(
        xs, zs, s=200, c=cm_plot, marker=".", cmap=plt.cm.viridis, alpha=0.3
    )
    ax.scatter(geom[:, 0], geom[:, 1], s=100, c="orange", marker="s")
    ax.set_title("ptps")
    ax.set_yticks([])
    ax.set_ylim(z_cutoff)
    ax.set_xlim(x_cutoff)

    ax = ax_ptp
    ptps_cluster = features[:, 2][label_indices]
    spike_train_s = spike_index[:, 0][label_indices] / 30000
    ax.plot(spike_train_s, ptps_cluster)
    ax.set_ylabel("ptp")
    ax.set_xlabel("seconds")

    ax = ax_ptp_z
    zs_cluster = features[:, 1][label_indices]
    ax.scatter(zs_cluster, ptps_cluster)
    ax.set_xlabel("zs")
    ax.set_ylabel("ptps")

    ax = ax_isi
    spike_train = spike_index[:, 0][label_indices]
    ax.set_xlabel("ms")
    spike_train_diff = np.diff(spike_train) / 30000
    spike_train_diff = spike_train_diff[np.where(spike_train_diff < 0.01)]
    spike_train_diff = spike_train_diff * 1000
    ax.hist(spike_train_diff, bins=np.arange(11))
    ax.set_xticks(range(11))
    ax.set_title("isis")
    ax.set_xlim([-1, 10])

    axes = [ax_xcorr1, ax_xcorr2]
    for i, cluster_isi_id in enumerate(closest_clusters):
        spike_train_2 = spike_index[:, 0][
            np.flatnonzero(labels == cluster_isi_id)
        ]
        sorting = NumpySorting.from_times_labels(
            times_list=np.concatenate((spike_train, spike_train_2)),
            labels_list=np.concatenate(
                (
                    np.zeros(spike_train.shape[0]).astype("int"),
                    np.zeros(spike_train_2.shape[0]).astype("int") + 1,
                )
            ),
            sampling_frequency=30000,
        )
        bin_ms = 1.0
        correlograms, bins = compute_correlograms(
            sorting, symmetrize=True, window_ms=10.0, bin_ms=bin_ms
        )
        axes[i].bar(bins[1:], correlograms[0][1], width=bin_ms, align="center")
        axes[i].set_xticks(bins[1:])
        axes[i].set_xlabel("lag (ms)")
        axes[i].set_title(
            f"{cluster_id}_{cluster_isi_id}_xcorrelogram.png", pad=20
        )

    z_uniq, z_ids = np.unique(geom[:, 1], return_inverse=True)
    all_max_ptp = maxptps[
        np.isin(labels, (*closest_clusters, cluster_id))
    ].max()
    scale = (z_uniq[1] - z_uniq[0]) / max(7, all_max_ptp)

    ax = ax_denoised
    # figure params
    num_rows = num_rows_plot
    plot_waveforms_geom(
        cluster_id,
        closest_clusters,
        labels,
        spike_index,
        firstchans,
        maxptps,
        geom,
        num_spikes_plot=num_spikes_plot,
        t_range=t_range,
        num_channels=num_channels,
        num_rows=num_rows,
        alpha=0.1,
        ax=ax,
        scale=scale,
        waveforms=wfs_full_denoise,
    )
    ax.set_title("denoised waveforms")
    ax.set_ylabel("z")
    ax.set_xlabel("x")
    ax.set_yticks([])

    ax = ax_raw
    plot_waveforms_geom(
        cluster_id,
        closest_clusters,
        labels,
        spike_index,
        firstchans,
        maxptps,
        geom,
        num_spikes_plot=num_spikes_plot,
        t_range=t_range,
        num_channels=num_channels,
        num_rows=num_rows,
        alpha=0.1,
        ax=ax,
        scale=scale,
        raw_bin=raw_bin,
    )
    ax.set_title("raw waveforms")
    ax.set_xlim(ax_denoised.get_xlim())
    ax.set_ylim(ax_denoised.get_ylim())
    ax.set_xlabel("x")
    ax.set_ylabel("z")

    ax = ax_cleaned
    plot_waveforms_geom(
        cluster_id,
        closest_clusters,
        labels,
        spike_index,
        firstchans,
        maxptps,
        geom,
        num_spikes_plot=num_spikes_plot,
        t_range=t_range,
        num_channels=num_channels,
        num_rows=num_rows,
        alpha=0.1,
        ax=ax,
        scale=scale,
        waveforms=wfs_subtracted,
        residual_bin=residual_bin,
    )
    ax.set_title("collision-cleaned waveforms")
    ax.set_yticks([])
    ax.set_xlim(ax_denoised.get_xlim())
    ax.set_ylim(ax_denoised.get_ylim())
    ax.set_xlabel("x")
    matplotlib.rcParams.update({"font.size": 10})

    return fig


# %%
def plot_agreement_venn(
    cluster_id_1,
    cluster_id_2,
    st_1,
    st_2,
    firstchans_cluster_sorting1,
    mcs_abs_cluster_sorting1,
    firstchans_cluster_sorting2,
    mcs_abs_cluster_sorting2,
    geom,
    raw_bin,
    scale=7,
    sorting1_name="1",
    sorting2_name="2",
    num_channels=40,
    num_spikes_plot=100,
    t_range=(30, 90),
    num_rows=3,
    alpha=0.1,
    delta_frames=12,
):
    lab_st1 = cluster_id_1
    lab_st2 = cluster_id_2
    (
        ind_st1,
        ind_st2,
        not_match_ind_st1,
        not_match_ind_st2,
    ) = compute_spiketrain_agreement(st_1, st_2, delta_frames)
    agreement = len(ind_st1) / (len(st_1) + len(st_2) - len(ind_st1))
    fig = plt.figure(figsize=(12, 6))
    grid = (1, 3)
    ax_venn = plt.subplot2grid(grid, (0, 0))
    ax_sorting1 = plt.subplot2grid(grid, (0, 1))
    ax_sorting2 = plt.subplot2grid(grid, (0, 2))

    subsets = [len(not_match_ind_st1), len(not_match_ind_st2), len(ind_st1)]
    v = venn2(
        subsets=subsets,
        set_labels=["unit{}".format(lab_st1), "unit{}".format(lab_st2)],
        ax=ax_venn,
    )
    v.get_patch_by_id("10").set_color("red")
    v.get_patch_by_id("01").set_color("blue")
    v.get_patch_by_id("11").set_color("goldenrod")
    ax_venn.set_title(
        f"{sorting1_name}{lab_st1} + {sorting2_name}{lab_st2}, {np.round(agreement, 2)*100}% agreement"
    )

    # fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,12))
    h_shifts = [-10, 10]
    colors = ["goldenrod", "red"]
    indices = [ind_st1, not_match_ind_st1]
    for indices_match, color, h_shift in zip(indices, colors, h_shifts):
        if len(indices_match) > 0:
            firstchans_cluster_sorting = firstchans_cluster_sorting1[
                indices_match
            ]
            mcs_abs_cluster_sorting = mcs_abs_cluster_sorting1[indices_match]
            spike_times = st_1[indices_match]
            # geom, first_chans_cluster, mcs_abs_cluster, max_ptps_cluster, spike_times,
            plot_waveforms_geom_unit(
                geom,
                firstchans_cluster_sorting,
                mcs_abs_cluster_sorting,
                spike_times,
                raw_bin=raw_bin,
                num_spikes_plot=num_spikes_plot,
                t_range=t_range,
                num_channels=num_channels,
                num_rows=num_rows,
                do_mean=False,
                scale=scale,
                h_shift=h_shift,
                alpha=alpha,
                ax=ax_sorting1,
                color=color,
            )
    colors = ["goldenrod", "blue"]
    indices = [ind_st2, not_match_ind_st2]
    for indices_match, color, h_shift in zip(indices, colors, h_shifts):
        if len(indices_match) > 0:
            mcs_abs_cluster_sorting = mcs_abs_cluster_sorting2[indices_match]
            firstchans_cluster_sorting = firstchans_cluster_sorting2[
                indices_match
            ]
            spike_times = st_2[indices_match]
            plot_waveforms_geom_unit(
                geom,
                firstchans_cluster_sorting,
                mcs_abs_cluster_sorting,
                spike_times,
                raw_bin=raw_bin,
                num_spikes_plot=num_spikes_plot,
                t_range=t_range,
                num_channels=num_channels,
                num_rows=num_rows,
                do_mean=False,
                scale=scale,
                h_shift=h_shift,
                alpha=alpha,
                ax=ax_sorting2,
                color=color,
            )
    return fig


# %%
def plot_agreement_venn_better(
    cluster_id_1,
    cluster_id_2,
    st_1,
    st_2,
    firstchans_cluster_sorting1,
    mcs_abs_cluster_sorting1,
    firstchans_cluster_sorting2,
    mcs_abs_cluster_sorting2,
    geom,
    raw_bin,
    hdb_cluster_depth_means,
    kilo_cluster_depth_means,
    spike_index_yass,
    spike_index_ks,
    labels_yass,
    labels_ks,
    scale=7,
    sorting1_name="1",
    sorting2_name="2",
    num_channels=40,
    num_spikes_plot=100,
    t_range=(30, 90),
    num_rows=3,
    alpha=0.1,
    delta_frames=12,
    num_close_clusters=5,
):
    
    
    vals, counts = np.unique(
        mcs_abs_cluster_sorting1,
        return_counts=True,
    )
    z_uniq, z_ids = np.unique(geom[:, 1], return_inverse=True)
    mcid = z_ids[vals[counts.argmax()]]

    lab_st1 = cluster_id_1
    lab_st2 = cluster_id_2
    (
        ind_st1,
        ind_st2,
        not_match_ind_st1,
        not_match_ind_st2,
    ) = compute_spiketrain_agreement(st_1, st_2, delta_frames)
    agreement = len(ind_st1) / (len(st_1) + len(st_2) - len(ind_st1))
    fig = plt.figure(figsize=(20, 20))

    gs = fig.add_gridspec(
        6,
        3,
        width_ratios=(1, 1, 1),
        height_ratios=(6, 2.5, 2.5, 2.5, 2.5, 2.5),
    )

    ax_venn = fig.add_subplot(gs[0, 0])
    ax_sorting1 = fig.add_subplot(gs[0, 1])
    ax_sorting2 = fig.add_subplot(gs[0, 2])

    ax_templates = fig.add_subplot(gs[1, :])
    ax_wfs_shared_yass = fig.add_subplot(gs[2, :])
    ax_wfs_shared_ks = fig.add_subplot(gs[3, :])
    ax_yass_temps = fig.add_subplot(gs[4, :])
    ax_ks_temps = fig.add_subplot(gs[5, :])

    subsets = [len(not_match_ind_st1), len(not_match_ind_st2), len(ind_st1)]
    v = venn2(
        subsets=subsets,
        set_labels=["unit{}".format(lab_st1), "unit{}".format(lab_st2)],
        ax=ax_venn,
    )
    if len(not_match_ind_st1) > 0:
        v.get_patch_by_id("10").set_color("red")
    if len(not_match_ind_st2) > 0:
        v.get_patch_by_id("01").set_color("blue")
    if len(ind_st1) > 0:
        v.get_patch_by_id("11").set_color("goldenrod")
    ax_venn.set_title(
        f"{sorting1_name}{lab_st1} + {sorting2_name}{lab_st2}, {np.round(agreement, 2)*100}% agreement"
    )

    closest_clusters_hdb = get_closest_clusters_kilosort(
        cluster_id_1,
        hdb_cluster_depth_means,
        num_close_clusters=num_close_clusters,
    )
    closest_clusters_kilo = get_closest_clusters_kilosort(
        cluster_id_2,
        kilo_cluster_depth_means,
        num_close_clusters=num_close_clusters,
    )

    first_chan_yass_ks = np.median(firstchans_cluster_sorting1) + 5

    some_in_cluster = np.random.choice(
        (labels_yass == cluster_id_1).sum(),
        replace=False,
        size=min((labels_yass == cluster_id_1).sum(), num_spikes_plot),
    )
    waveforms_read = read_waveforms(
        spike_index_yass[labels_yass == cluster_id_1, 0][some_in_cluster],
        raw_bin,
        geom.shape[0],
        spike_length_samples=121,
        channels=np.arange(first_chan_yass_ks, first_chan_yass_ks + 20).astype(
            "int"
        ),
    )[0]
    temp1 = waveforms_read.mean(0)
    ax_yass_temps.plot(temp1.T.flatten(), color="red")

    for j in range(num_close_clusters):
        some_in_cluster = np.random.choice(
            (labels_yass == closest_clusters_hdb[j]).sum(),
            replace=False,
            size=min(
                (labels_yass == closest_clusters_hdb[j]).sum(), num_spikes_plot
            ),
        )
        waveforms_read = read_waveforms(
            spike_index_yass[labels_yass == closest_clusters_hdb[j], 0][
                some_in_cluster
            ],
            raw_bin,
            geom.shape[0],
            spike_length_samples=121,
            channels=np.arange(
                int(first_chan_yass_ks), int(first_chan_yass_ks) + 20
            ),
        )[0]
        ax_yass_temps.plot(waveforms_read.mean(0).T.flatten())
    for i in range(20):
        ax_yass_temps.axvline(121 + 121 * i, c="black")
    ax_yass_temps.set_title(f"{sorting1_name} close templates")

    some_in_cluster = np.random.choice(
        (labels_ks == cluster_id_2).sum(),
        replace=False,
        size=min((labels_ks == cluster_id_2).sum(), num_spikes_plot),
    )
    waveforms_read = read_waveforms(
        spike_index_ks[labels_ks == cluster_id_2, 0][some_in_cluster],
        raw_bin,
        geom.shape[0],
        spike_length_samples=121,
        channels=np.arange(first_chan_yass_ks, first_chan_yass_ks + 20).astype(
            "int"
        ),
    )[0]
    temp2 = waveforms_read.mean(0)
    ax_ks_temps.plot(temp2.T.flatten(), color="blue")

    for j in range(num_close_clusters):
        some_in_cluster = np.random.choice(
            list(range((labels_ks == closest_clusters_kilo[j]).sum())),
            replace=False,
            size=min(
                (labels_ks == closest_clusters_kilo[j]).sum(), num_spikes_plot
            ),
        )
        if len(some_in_cluster) > 0:
            waveforms_read = read_waveforms(
                spike_index_ks[labels_ks == closest_clusters_kilo[j], 0][
                    some_in_cluster
                ],
                raw_bin,
                geom.shape[0],
                spike_length_samples=121,
                channels=np.arange(
                    first_chan_yass_ks, first_chan_yass_ks + 20
                ).astype("int"),
            )[0]
            ax_ks_temps.plot(waveforms_read.mean(0).T.flatten())
    for i in range(20):
        ax_ks_temps.axvline(121 + 121 * i, c="black")
    ax_ks_temps.set_title(f"{sorting2_name} close templates")

    # fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,12))
    h_shifts = [-10, 10]
    colors = ["goldenrod", "red"]
    indices = [ind_st1, not_match_ind_st1]
    cmp = 0
    shared_mc1 = -1
    for indices_match, color, h_shift in zip(indices, colors, h_shifts):
        if len(indices_match) > 0:
            firstchans_cluster_sorting = firstchans_cluster_sorting1[
                indices_match
            ]
            mcs_abs_cluster_sorting = mcs_abs_cluster_sorting1[indices_match]
            spike_times = st_1[indices_match]
            # geom, first_chans_cluster, mcs_abs_cluster, max_ptps_cluster, spike_times,
            (
                waveforms,
                first_chan_cluster,
            ) = plot_waveforms_geom_unit_with_return(
                geom,
                firstchans_cluster_sorting,
                mcs_abs_cluster_sorting,
                spike_times,
                z_ids, 
                mcid,
                raw_bin=raw_bin,
                num_spikes_plot=num_spikes_plot,
                t_range=t_range,
                num_channels=num_channels,
                num_rows=num_rows,
                do_mean=False,
                scale=scale,
                h_shift=h_shift,
                alpha=alpha,
                ax=ax_sorting1,
                color=color,
            )
            if shared_mc1 < 0:
                shared_mc1 = waveforms.mean(0).ptp(0).argmax()
            for i in range(min(len(waveforms), num_spikes_plot)):
                ax_wfs_shared_yass.plot(
                    waveforms[
                        i, :, shared_mc1 - 5 : shared_mc1 + 6
                    ].T.flatten(),
                    alpha=alpha,
                    color=color,
                )
            for i in range(10):
                ax_wfs_shared_yass.axvline(121 + 121 * i, c="black")
            ax_templates.plot(
                waveforms[:, :, shared_mc1 - 5 : shared_mc1 + 6]
                .mean(0)
                .T.flatten(),
                color=color,
            )
            cmp += 1
    colors = ["goldenrod", "blue"]
    indices = [ind_st2, not_match_ind_st2]
    shared_mc2 = -1
    for indices_match, color, h_shift in zip(indices, colors, h_shifts):
        if len(indices_match) > 0:
            mcs_abs_cluster_sorting = mcs_abs_cluster_sorting2[indices_match]
            firstchans_cluster_sorting = firstchans_cluster_sorting2[
                indices_match
            ]
            spike_times = st_2[indices_match]

            (
                waveforms,
                first_chan_cluster,
            ) = plot_waveforms_geom_unit_with_return(
                geom,
                firstchans_cluster_sorting,
                mcs_abs_cluster_sorting,
                spike_times,
                z_ids,
                mcid,
                raw_bin=raw_bin,
                num_spikes_plot=num_spikes_plot,
                t_range=t_range,
                num_channels=num_channels,
                num_rows=num_rows,
                do_mean=False,
                scale=scale,
                h_shift=h_shift,
                alpha=alpha,
                ax=ax_sorting2,
                color=color,
            )
            # mc = waveforms.mean(0).ptp(0).argmax()
            if shared_mc2 < 0:
                shared_mc2 = waveforms.mean(0).ptp(0).argmax()

            for i in range(min(len(waveforms), num_spikes_plot)):
                ax_wfs_shared_ks.plot(
                    waveforms[
                        i, :, shared_mc2 - 5 : shared_mc2 + 6
                    ].T.flatten(),
                    alpha=alpha,
                    color=color,
                )
            for i in range(10):
                ax_wfs_shared_ks.axvline(121 + 121 * i, c="black")
            ax_templates.plot(
                waveforms[:, :, shared_mc2 - 5 : shared_mc2 + 6]
                .mean(0)
                .T.flatten(),
                color=color,
            )
    for i in range(10):
        ax_templates.axvline(121 + 121 * i, c="black")

    return fig


# %%
def plot_waveforms_geom_unit_with_return(
    geom,
    first_chans_cluster,
    mcs_abs_cluster,
    spike_times,
    z_ids, 
    mcid,
    max_ptps_cluster=None,
    raw_bin=None,
    residual_bin=None,
    waveforms_cluster=None,
    denoiser=None,
    device=None,
    num_spikes_plot=100,
    t_range=(30, 90),
    num_channels=40,
    num_rows=3,
    alpha=0.1,
    h_shift=0,
    subset_indices=None,
    scale=None,
    do_mean=False,
    annotate=False,
    ax=None,
    color="blue",
):
    ax = ax or plt.gca()
    some_in_cluster = np.random.default_rng(0).choice(
        list(range(len(spike_times))),
        replace=False,
        size=min(len(spike_times), num_spikes_plot),
    )
    first_chans_cluster = first_chans_cluster[some_in_cluster]
    mcs_abs_cluster = mcs_abs_cluster[some_in_cluster]
    spike_times = spike_times[some_in_cluster]

    # what channels will we plot?
    vals, counts = np.unique(
        mcs_abs_cluster,
        return_counts=True,
    )
#     z_uniq, z_ids = np.unique(geom[:, 1], return_inverse=True)
#     mcid = z_ids[vals[counts.argmax()]]
    channels_plot = np.flatnonzero(
        (z_ids >= mcid - num_rows) & (z_ids <= mcid + num_rows)
    )

    # how to scale things?
    if scale is None:
        if max_ptps_cluster is not None:
            max_ptps_cluster = max_ptps_cluster[some_in_cluster]
            all_max_ptp = max_ptps_cluster.max()
            scale = (z_uniq[1] - z_uniq[0]) / max(7, all_max_ptp)
        else:
            scale = 7

    times_plot = np.arange(t_range[0] - 42, t_range[1] - 42).astype(float)
    x_uniq = np.unique(geom[:, 0])
    times_plot *= (x_uniq[1] - x_uniq[0]) / np.abs(times_plot).max()

    # scatter the channels
    ax.scatter(*geom[channels_plot].T, c="orange", marker="s")
    if annotate:
        for c in channels_plot:
            ax.annotate(c, (geom[c, 0], geom[c, 1]))

    if raw_bin is not None:
        # raw data and spike times passed in
        if len(np.unique(first_chans_cluster))>1:
            waveforms_read = read_waveforms(
                spike_times, raw_bin, geom.shape[0], spike_length_samples=121
            )[0]
            waveforms = []
            for i, waveform in enumerate(waveforms_read):
                waveforms.append(
                    waveform[
                        :,
                        int(first_chans_cluster[i]) : int(first_chans_cluster[i])
                        + num_channels,
                    ].copy()
                )
            waveforms = np.asarray(waveforms)
        else:
            waveforms = read_waveforms(
                spike_times,
                raw_bin,
                geom.shape[0],
                channels=np.arange(first_chans_cluster[0], first_chans_cluster[0]+num_channels),
                spike_length_samples=121,
            )[0]

    elif waveforms_cluster is None:
        # no raw data and no waveforms passed - bad!
        raise ValueError("need to input raw_bin or waveforms")
    else:
        # waveforms passed in
        waveforms = waveforms_cluster[some_in_cluster]
        if residual_bin is not None:
            # add residuals
            residuals_read = read_waveforms(
                spike_times, residual_bin, geom.shape[0], spike_length_samples=121
            )[0]
            residuals = []
            for i, residual in enumerate(residuals_read):
                residuals.append(
                    residual[
                        :,
                        int(first_chans_cluster[i]) : int(
                            first_chans_cluster[i]
                        )
                        + num_channels,
                    ]
                )
            residuals = np.asarray(residuals)
            waveforms = waveforms + residuals
    if denoiser is not None and device is not None:
        # denoise waveforms
        waveforms = denoise_wf_nn_tmp_single_channel(
            waveforms, denoiser, device
        )
    if do_mean:
        # plot the mean rather than the invididual spikes
        waveforms = np.expand_dims(np.mean(waveforms, axis=0), 0)
    draw_lines = []
    for i in range(min(len(waveforms), num_spikes_plot)):
        for k, channel in enumerate(
            range(
                int(first_chans_cluster[i]),
                int(first_chans_cluster[i]) + waveforms.shape[2],
            )
        ):
            if channel in channels_plot:
                trace = waveforms[
                    i,
                    t_range[0] : t_range[1],
                    k,
                ]
            else:
                continue
            waveform = trace * scale
            draw_lines.append(geom[channel, 0] + times_plot + h_shift)
            draw_lines.append(waveform + geom[channel, 1])
    ax.plot(
        *draw_lines,
        alpha=alpha,
        c=color,
    )
    ax.set_xticks([])
    # ax.yaxis.tick_right()
    return waveforms, first_chans_cluster


# %%
def plot_unit_similarity_heatmaps(
    cluster_id,
    st_1,
    closest_clusters,
    sorting,
    geom,
    raw_data_bin,
    num_channels_similarity=20,
    num_close_clusters_plot=10,
    num_close_clusters=30,
    shifts_align=[0],
    order_by="similarity",
    normalize_agreement_by="both",
    ax_similarity=None,
    ax_agreement=None,
):
    if ax_similarity is None:
        plt.figure(figsize=(9, 3))
        ax_similarity = plt.gca()

    if ax_agreement is None:
        plt.figure(figsize=(9, 3))
        ax_agreement = plt.gca()

    (
        original_template,
        closest_clusters,
        similarities,
        agreements,
        templates,
        shifts,
    ) = get_unit_similarities(
        cluster_id,
        st_1,
        closest_clusters,
        sorting,
        geom,
        raw_data_bin,
        num_channels_similarity,
        num_close_clusters,
        shifts_align,
        order_by,
        normalize_agreement_by,
    )

    agreements = agreements[:num_close_clusters_plot]
    similarities = similarities[:num_close_clusters_plot]
    closest_clusters = closest_clusters[:num_close_clusters_plot]
    templates = templates[:num_close_clusters_plot]
    shifts = shifts[:num_close_clusters_plot]

    y_axis_labels = [cluster_id]
    x_axis_labels = closest_clusters
    g = sns.heatmap(
        np.expand_dims(similarities, 0),
        vmin=0,
        vmax=max(similarities),
        cmap="RdYlGn_r",
        annot=np.expand_dims(similarities, 0),
        xticklabels=x_axis_labels,
        yticklabels=y_axis_labels,
        ax=ax_similarity,
        cbar=False,
    )
    ax_similarity.set_title("Max Abs Norm Similarity")
    g = sns.heatmap(
        np.expand_dims(agreements, 0),
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        annot=np.expand_dims(agreements, 0),
        xticklabels=x_axis_labels,
        yticklabels=y_axis_labels,
        ax=ax_agreement,
        cbar=False,
    )
    ax_agreement.set_title(
        f"Agreement (normalized by {normalize_agreement_by})"
    )

    return (
        ax_similarity,
        ax_agreement,
        original_template,
        closest_clusters,
        similarities,
        agreements,
        templates,
        shifts,
    )


# %%
def plot_unit_similarities(
    cluster_id,
    closest_clusters,
    sorting1,
    sorting2,
    geom,
    raw_data_bin,
    recoring_duration,
    num_channels=40,
    num_spikes_plot=100,
    num_channels_similarity=20,
    num_close_clusters_plot=10,
    num_close_clusters=30,
    shifts_align=np.arange(-3, 4),
    order_by="similarity",
    normalize_agreement_by="both",
    denoised_waveforms=None,
    cluster_labels=None,
    non_triaged_idxs=None,
    triaged_mcs_abs=None,
    triaged_firstchans=None,
):
    do_denoised_waveform = (
        denoised_waveforms is not None
        and cluster_labels is not None
        and non_triaged_idxs is not None
        and triaged_mcs_abs is not None
        and triaged_firstchans is not None
    )
    fig = plt.figure(figsize=(24, 12))
    if do_denoised_waveform:
        gs = gridspec.GridSpec(4, 4)
    else:
        gs = gridspec.GridSpec(4, 3)
    gs.update(hspace=0.75)
    ax_sim = plt.subplot(gs[0, :2])
    ax_agree = plt.subplot(gs[1, :2])
    ax_isi = plt.subplot(gs[2, :2])
    ax_raw_wf = plt.subplot(gs[:4, 2])
    ax_raw_wf_flat = plt.subplot(gs[3, :2])
    if do_denoised_waveform:
        ax_denoised_wf = plt.subplot(gs[:4, 3])

    st_1 = sorting1.get_unit_spike_train(cluster_id)
    firing_rate = len(st_1) / recoring_duration  # in seconds

    # compute similarity to closest kilosort clusters
    (
        _,
        _,
        original_template,
        closest_clusters,
        similarities,
        agreements,
        templates,
        shifts,
    ) = plot_unit_similarity_heatmaps(
        cluster_id,
        st_1,
        closest_clusters,
        sorting2,
        geom,
        raw_data_bin,
        num_channels_similarity=num_channels_similarity,
        num_close_clusters_plot=num_close_clusters_plot,
        num_close_clusters=num_close_clusters,
        ax_similarity=ax_sim,
        ax_agreement=ax_agree,
        shifts_align=shifts_align,
        order_by=order_by,
        normalize_agreement_by=normalize_agreement_by,
    )

    max_ptp_channel = np.argmax(original_template.ptp(0))
    max_ptp = np.max(original_template.ptp(0))
    z_uniq, z_ids = np.unique(geom[:, 1], return_inverse=True)
    scale = (z_uniq[1] - z_uniq[0]) / max(7, max_ptp)

    plot_isi_distribution(st_1, ax=ax_isi)

    most_similar_cluster = closest_clusters[0]
    most_similar_shift = shifts[0]
    h_shifts = [-5, 5]
    t_shifts = [0, most_similar_shift]
    colors = [("blue", "darkblue"), ("red", "darkred")]
    cluster_ids_plot = [cluster_id, most_similar_cluster]
    sortings_plot = [sorting1, sorting2]
    for cluster_id_plot, color, sorting_plot, h_shift, t_shift in zip(
        cluster_ids_plot, colors, sortings_plot, h_shifts, t_shifts
    ):
        spike_times = (
            sorting_plot.get_unit_spike_train(cluster_id_plot) + t_shift
        )
        mcs_abs_cluster = (
            np.zeros(len(spike_times)).astype("int") + max_ptp_channel
        )
        first_chans_cluster = (mcs_abs_cluster - 20).clip(min=0)

        # waveform_scale = 1/25

        plot_waveforms_geom_unit(
            geom,
            first_chans_cluster,
            mcs_abs_cluster,
            spike_times,
            max_ptps_cluster=None,
            raw_bin=raw_data_bin,
            num_spikes_plot=num_spikes_plot,
            num_channels=num_channels,
            alpha=0.1,
            h_shift=h_shift,
            scale=scale,
            ax=ax_raw_wf,
            color=color[0],
        )
        plot_waveforms_geom_unit(
            geom,
            first_chans_cluster,
            mcs_abs_cluster,
            spike_times,
            max_ptps_cluster=None,
            raw_bin=raw_data_bin,
            num_spikes_plot=num_spikes_plot,
            num_channels=num_channels,
            alpha=0.1,
            h_shift=h_shift,
            scale=scale,
            do_mean=True,
            ax=ax_raw_wf,
            color=color[1],
        )

        # if do_denoised_waveform:
        #     mcs_abs_cluster = triaged_mcs_abs[cluster_labels==cluster_id_plot]
        #     first_chans_cluster = triaged_firstchans[cluster_labels==cluster_id_plot]
        #     waveforms = denoised_waveforms[non_triaged_idxs[cluster_labels==cluster_id_plot]]
        #     plot_waveforms_unit_geom(geom, num_channels, first_chans_cluster, mcs_abs_cluster, waveforms, x_geom_scale = 1/15,
        #                              y_geom_scale = 1/10, waveform_scale = waveform_scale, spikes_plot = num_spikes_plot, waveform_shape=(30,70), num_rows=3,
        #                              alpha=.2, h_shift=h_shift, do_mean=False, ax=ax_denoised_wf, color=color[0])
        #     plot_waveforms_unit_geom(geom, num_channels, first_chans_cluster, mcs_abs_cluster, waveforms, x_geom_scale = 1/15,
        #                              y_geom_scale = 1/10, waveform_scale = waveform_scale, spikes_plot = num_spikes_plot, waveform_shape=(30,70), num_rows=3,
        #                              alpha=1, h_shift=h_shift, do_mean=True, ax=ax_denoised_wf, color=color[1])

    ax_raw_wf.set_title(
        f"cluster {cluster_id}/cluster {most_similar_cluster} raw, shift {most_similar_shift}"
    )
    if do_denoised_waveform:
        ax_denoised_wf.set_title(
            f"cluster {cluster_id}/cluster {most_similar_cluster} denoised, shift {most_similar_shift}"
        )
    channel_range = (
        max(max_ptp_channel - num_channels_similarity // 2, 0),
        max_ptp_channel + num_channels_similarity // 2,
    )
    template1 = original_template[:, channel_range[0] : channel_range[1]]
    most_similar_template = templates[0]
    if most_similar_shift == 0:
        most_similar_template_flattened = most_similar_template.T.flatten()
    elif most_similar_shift < 0:
        most_similar_template_flattened = np.pad(
            most_similar_template.T.flatten(),
            ((-most_similar_shift, 0)),
            mode="constant",
        )[:most_similar_shift]
    else:
        most_similar_template_flattened = np.pad(
            most_similar_template.T.flatten(),
            ((0, most_similar_shift)),
            mode="constant",
        )[most_similar_shift:]
    ax_raw_wf_flat.plot(template1.T.flatten(), color="blue")
    ax_raw_wf_flat.plot(most_similar_template_flattened, color="red")
    ax_raw_wf_flat.set_title(
        f"cluster {cluster_id}/cluster {most_similar_cluster} templates flat, shift {most_similar_shift}"
    )
    fig.suptitle(
        f"cluster {cluster_id}, firing rate: {'%.1f' % round(firing_rate,2)} Hz, max ptp: {'%.1f' % round(max_ptp,2)}"
    )
    return fig


# %%
def diagnostic_plots(
    cluster_id_1,
    cluster_id_2,
    st_1,
    st_2,
    templates_yass,
    templates_ks,
    mcs_abs_cluster_sorting1,
    mcs_abs_cluster_sorting2,
    geom,
    raw_bin,
    hdb_cluster_depth_means,
    kilo_cluster_depth_means,
    spike_index_yass,
    spike_index_ks,
    labels_yass,
    labels_ks,
    closest_clusters_hdb,
    closest_clusters_kilo,
    scale=7,
    sorting1_name="1",
    sorting2_name="2",
    num_channels=40,
    num_spikes_plot=100,
    t_range=(30, 90),
    num_rows=3,
    alpha=0.1,
    delta_frames=12,
    num_close_clusters=5,
    tpca_rank=5,
):
    print(f"{st_1.shape=}, {st_2.shape=}")

    if not st_1.size and not st_2.size:
        raise ValueError("nothing in either spike train")
    lab_st1 = cluster_id_1
    lab_st2 = cluster_id_2
    (
        ind_st1,
        ind_st2,
        not_match_ind_st1,
        not_match_ind_st2,
    ) = compute_spiketrain_agreement(st_1, st_2, delta_frames)
    print(f"{ind_st1.shape=}, {ind_st2.shape=}, {not_match_ind_st1.shape=}, {not_match_ind_st2.shape=}")
    agreement = len(ind_st1) / (len(st_1) + len(st_2) - len(ind_st1))
    fig = plt.figure(figsize=(12, 20))

    gs = fig.add_gridspec(8, 12, height_ratios=(5, 1, 2, 2, 2, 1, 2, 1))

    ax_sorting1 = fig.add_subplot(gs[0, :6])
    ax_sorting2 = fig.add_subplot(gs[0, 6:])

    ax_venn = fig.add_subplot(gs[1, :3])
    ax_lda_blue_yellow = fig.add_subplot(gs[1, 3:6])
    ax_lda_red_yellow = fig.add_subplot(gs[1, 6:9])
    ax_lda_blue_red = fig.add_subplot(gs[1, 9:])

    ax_wfs_shared_yass = fig.add_subplot(gs[2, :])
    ax_wfs_shared_ks = fig.add_subplot(gs[3, :])

    ax_templates_yass = fig.add_subplot(gs[4, :])
    ax_isi_yass = fig.add_subplot(gs[5, :3])
    ax_LDA1_yass = fig.add_subplot(gs[5, 3:6])
    ax_LDA2_yass = fig.add_subplot(gs[5, 6:9])
    ax_PCs_yass = fig.add_subplot(gs[5, 9:])

    ax_templates_ks = fig.add_subplot(gs[6, :])
    ax_isi_ks = fig.add_subplot(gs[7, :3])
    ax_LDA1_ks = fig.add_subplot(gs[7, 3:6])
    ax_LDA2_ks = fig.add_subplot(gs[7, 6:9])
    ax_PCs_ks = fig.add_subplot(gs[7, 9:])

    list_ax_wfs_shared_ks = [0]
    for cm_x in np.arange(0, 675, 10):
        if cm_x > 0:
            if cm_x % 60 == 0:
                list_ax_wfs_shared_ks.append(cm_x + 1)
            else:
                list_ax_wfs_shared_ks.append(cm_x)

    ax_wfs_shared_yass.set_xticks(list_ax_wfs_shared_ks, minor=True)
    #     ax_test_split_ptp = fig.add_subplot(gs[8, 3:6])
    #     ax_test_split_temp_proj = fig.add_subplot(gs[8, 6:9])

    plot_isi_distribution(st_1, ax=ax_isi_yass)
    plot_isi_distribution(st_2, ax=ax_isi_ks)

    def apply_tpca(wfs_a, wfs_b):
        wfs = np.r_[wfs_a.transpose(0, 2, 1), wfs_b.transpose(0, 2, 1)]
        N, C, T = wfs.shape
        tpca = PCA(tpca_rank)
        wfs = tpca.fit_transform(wfs.reshape(-1, T))
        wfs = wfs.reshape(N, C, tpca_rank).reshape(N, -1)
        return wfs

    subsets = [len(not_match_ind_st1), len(not_match_ind_st2), len(ind_st1)]
    v = venn2(
        subsets=subsets,
        set_labels=[
            f"{sorting1_name} {lab_st1}",
            f"{sorting2_name} {lab_st2}",
        ],
        ax=ax_venn,
    )
    if len(not_match_ind_st1) > 0:
        v.get_patch_by_id("10").set_color("red")
    if len(not_match_ind_st2) > 0:
        v.get_patch_by_id("01").set_color("blue")
    if len(ind_st1) > 0:
        v.get_patch_by_id("11").set_color("goldenrod")
    for text in v.subset_labels:
        if text is not None:
            text.set_fontsize(6)
    for text in v.set_labels:
        if text is not None:
            text.set_fontsize(8)

    ax_venn.set_title(f"{np.round(agreement, 2)*100}% agreement")

    ax_lda_blue_red.set_title(f"Distinct {sorting2_name}/{sorting1_name} LDA")
    ax_lda_blue_yellow.set_title(f"Shared/{sorting2_name} LDA")
    ax_lda_red_yellow.set_title(f"Shared/{sorting1_name} LDA")

    # fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,12))
    h_shifts = [-10, 10]
    colors = ["goldenrod", "red"]
    indices = [ind_st1, not_match_ind_st1]
    # FIX CHANNEL INDEX!!
    
    vals, counts = np.unique(
        mcs_abs_cluster_sorting1,
        return_counts=True,
    )
    z_uniq, z_ids = np.unique(geom[:, 1], return_inverse=True)
    mcid = z_ids[vals[counts.argmax()]]
    
    cmp = 0
    wfs_shared = np.array([])
    wfs_lda_red = np.array([])
    if ind_st1.size:
        u, c = np.unique(mcs_abs_cluster_sorting1[ind_st1], return_counts=True)
        shared_mc = u[c.argmax()]
    elif not_match_ind_st1.size:
        u, c = np.unique(mcs_abs_cluster_sorting1[not_match_ind_st1], return_counts=True)
        shared_mc = u[c.argmax()]
    elif not_match_ind_st2.size:
        u, c = np.unique(mcs_abs_cluster_sorting2[not_match_ind_st2], return_counts=True)
        shared_mc = u[c.argmax()]
    else:
        assert False
    template_red = None
    template_black = None
    for indices_match, color, h_shift in zip(indices, colors, h_shifts):
        if len(indices_match) > 0:
            # firstchans_cluster_sorting = firstchans_cluster_sorting1[
            #     indices_match
            # ]
            # mcs_abs_cluster_sorting = mcs_abs_cluster_sorting1[indices_match]
            mcs_abs_cluster_sorting = np.full(
                indices_match.shape,
                shared_mc,
            )
#             print(mcs_abs_cluster_sorting)
            firstchans_cluster_sorting = np.minimum(
                geom.shape[0] - num_channels,
                np.maximum(
                    mcs_abs_cluster_sorting - num_channels // 2, 0
                )
            )
            spike_times = st_1[indices_match]
            # geom, first_chans_cluster, mcs_abs_cluster, max_ptps_cluster, spike_times,
            (
                waveforms,
                first_chan_cluster,
            ) = plot_waveforms_geom_unit_with_return(
                geom,
                firstchans_cluster_sorting,
                mcs_abs_cluster_sorting,
                spike_times,
                z_ids, 
                mcid,
                raw_bin=raw_bin,
                num_spikes_plot=num_spikes_plot,
                t_range=t_range,
                num_channels=num_channels,
                num_rows=num_rows,
                do_mean=False,
                scale=scale,
                h_shift=h_shift,
                alpha=alpha,
                ax=ax_sorting1,
                color=color,
            )
            mc = (
                shared_mc - firstchans_cluster_sorting
            )  # waveforms.mean(0).ptp(0).argmax()
            start = np.maximum(mc - 5, 0)
            end = start + 11
            for i in range(min(len(waveforms), num_spikes_plot)):
                ax_wfs_shared_yass.plot(
                    waveforms[i, 30:-30, start[i] : end[i]].T.flatten(),
                    alpha=alpha,
                    color=color,
                )
            for i in range(10):
                ax_wfs_shared_yass.axvline(61 + 61 * i, c="black")
            #             ax_templates.plot(waveforms[:, :, mc-5:mc+6].mean(0).T.flatten(), color=color)
            cmp += 1
            # print("waveforms", waveforms.shape)
            if color == "goldenrod":
                wfs_shared = np.stack(
                    [
                        waveforms[i, :, start[i] : end[i]]
                        for i in range(len(waveforms))
                    ],
                    axis=0,
                )
                template_black = wfs_shared.mean(0)
                # print("shared", wfs_shared.shape, mc, flush=True)
            if color == "red":
                wfs_lda_red = np.stack(
                    [
                        waveforms[i, :, start[i] : end[i]]
                        for i in range(len(waveforms))
                    ],
                    axis=0,
                )
                template_red = wfs_lda_red.mean(0)
                # print("red", wfs_shared.shape, mc, flush=True)

    ## PTP/ template split
    #     waveforms_yass_to_split = np.concatenate((wfs_shared, wfs_lda_red))
    #     mc = templates_yass[cluster_id_1].ptp(0).argmax()

    #     ptps = waveforms_yass_to_split[:, :, 5].ptp(1)
    #     value_dpt, cut_calue = isocut(ptps)

    #     ax_test_split_ptp.hist(ptps, bins = 50)
    #     ax_test_split_ptp.set_title(str(value_dpt) + '\n' + str(cut_calue))

    #     temp_unit = templates_yass[cluster_id_1, :, mc]
    #     norm_wfs = np.sqrt(np.square(waveforms_yass_to_split[:, :, 5]).sum(1))
    #     temp_proj = np.einsum('ij,j->i', waveforms_yass_to_split[:, :, 5], templates_yass[cluster_id_1, :, mc])/norm_wfs
    #     value_dpt, cut_calue = isocut(temp_proj)

    #     ax_test_split_temp_proj.hist(temp_proj, bins = 50)
    #     ax_test_split_temp_proj.set_title(str(value_dpt) + '\n' + str(cut_calue))
    #     ax_wfs_shared_yass.set_xticks([])

    #     wfs_mc = waveforms_yass_to_split[:, :, 5]
    #     wfs_mc = wfs_mc[wfs_mc.ptp(1).argsort()]
    #     lower = int(waveforms_yass_to_split.shape[0]*0.05)
    #     upper = int(waveforms_yass_to_split.shape[0]*0.95)
    #     max_diff = 0
    #     max_diff_N = 0
    #     for n in tqdm(np.arange(lower, upper)):
    #         temp_1 = np.mean(wfs_mc[:n], axis = 0)
    #         temp_2 = np.mean(wfs_mc[n:], axis = 0)
    #         if np.abs(temp_1-temp_2).max() > max_diff:
    #             max_diff = np.abs(temp_1-temp_2).max()
    #             max_diff_N = n
    #     print(max_diff)
    #     print(max_diff_N)

    colors = ["goldenrod", "blue"]
    indices = [ind_st2, not_match_ind_st2]
    wfs_lda_blue = np.array([])
    template_blue = None

    for indices_match, color, h_shift in zip(indices, colors, h_shifts):
        if len(indices_match) > 0:
            mcs_abs_cluster_sorting = np.full(indices_match.shape, shared_mc)
            firstchans_cluster_sorting = np.minimum(
                geom.shape[0] - num_channels,
                np.maximum(
                    mcs_abs_cluster_sorting - num_channels // 2, 0
                )
            )
            # mcs_abs_cluster_sorting = mcs_abs_cluster_sorting2[indices_match]
            # firstchans_cluster_sorting = firstchans_cluster_sorting2[
            #     indices_match
            # ]
            spike_times = st_2[indices_match]

            (
                waveforms,
                first_chan_cluster,
            ) = plot_waveforms_geom_unit_with_return(
                geom,
                firstchans_cluster_sorting,
                mcs_abs_cluster_sorting,
                spike_times,
                z_ids, 
                mcid,
                raw_bin=raw_bin,
                num_spikes_plot=num_spikes_plot,
                t_range=t_range,
                num_channels=num_channels,
                num_rows=num_rows,
                do_mean=False,
                scale=scale,
                h_shift=h_shift,
                alpha=alpha,
                ax=ax_sorting2,
                color=color,
            )
            mc = (
                shared_mc - firstchans_cluster_sorting
            )  # waveforms.mean(0).ptp(0).argmax()
            start = np.maximum(mc - 5, 0)
            end = start + 11
            for i in range(min(len(waveforms), num_spikes_plot)):
                ax_wfs_shared_ks.plot(
                    waveforms[i, 30:-30, start[i] : end[i]].T.flatten(),
                    alpha=alpha,
                    color=color,
                )
            for i in range(10):
                ax_wfs_shared_ks.axvline(61 + 61 * i, c="black")
            if color == "blue":
                wfs_lda_blue = np.stack(
                    [
                        waveforms[i, :, start[i] : end[i]]
                        for i in range(len(waveforms))
                    ],
                    axis=0,
                )
                template_blue = wfs_lda_blue.mean(0)

    lda_labels = np.zeros(wfs_shared.shape[0] + wfs_lda_red.shape[0])
    lda_labels[: wfs_shared.shape[0]] = 1
    if wfs_lda_red.size and wfs_shared.size:
        lda_comps = LDA(n_components=1).fit_transform(
            apply_tpca(wfs_shared, wfs_lda_red),
            lda_labels,
        )
        ax_lda_red_yellow.hist(
            lda_comps[: wfs_shared.shape[0], 0],
            bins=25,
            color="goldenrod",
            alpha=0.5,
        )
        ax_lda_red_yellow.hist(
            lda_comps[wfs_shared.shape[0] :, 0],
            bins=25,
            color="red",
            alpha=0.5,
        )

    if wfs_lda_blue.size and wfs_shared.size:
        lda_labels = np.zeros(wfs_shared.shape[0] + wfs_lda_blue.shape[0])
        lda_labels[: wfs_shared.shape[0]] = 1
        lda_comps = LDA(n_components=1).fit_transform(
            apply_tpca(wfs_shared, wfs_lda_blue),
            lda_labels,
        )
        ax_lda_blue_yellow.hist(
            lda_comps[: wfs_shared.shape[0], 0],
            bins=25,
            color="goldenrod",
            alpha=0.5,
        )
        ax_lda_blue_yellow.hist(
            lda_comps[wfs_shared.shape[0] :, 0],
            bins=25,
            color="blue",
            alpha=0.5,
        )

    if (
        len(not_match_ind_st1) > 0
        and len(not_match_ind_st2) > 0
        and wfs_lda_red.size
        and wfs_lda_blue.size
        and wfs_lda_blue.shape[0] > 2
        and wfs_lda_red.shape[0] > 2
    ):
        lda_labels = np.zeros(wfs_lda_blue.shape[0] + wfs_lda_red.shape[0])
        lda_labels[: wfs_lda_blue.shape[0]] = 1
        lda_comps = LDA(n_components=1).fit_transform(
            apply_tpca(wfs_lda_blue, wfs_lda_red),
            lda_labels,
        )
        ax_lda_blue_red.hist(
            lda_comps[: wfs_lda_blue.shape[0], 0],
            bins=25,
            color="red",
            alpha=0.5,
        )
        ax_lda_blue_red.hist(
            lda_comps[wfs_lda_blue.shape[0] :, 0],
            bins=25,
            color="blue",
            alpha=0.5,
        )

    #     ax_wfs_shared_ks.set_xticks([])
    ax_wfs_shared_ks.set_xticks(list_ax_wfs_shared_ks, minor=True)

    mc = templates_yass[cluster_id_1].ptp(0).argmax()

    ax_wfs_shared_yass.set_ylim(
        (
            templates_yass[cluster_id_1, :, mc].min() - 2,
            templates_yass[cluster_id_1, :, mc].max() + 2,
        )
    )

    ax_wfs_shared_yass.set_yticks(
        np.arange(
            templates_yass[cluster_id_1, :, mc].min() - 2,
            templates_yass[cluster_id_1, :, mc].max() + 2,
            1,
        ),
        minor=True,
    )

    ax_wfs_shared_yass.grid(which="both")    
    color_array_yass_close = ["red", "cyan", "lime", "fuchsia"]
    pc_scatter = PCA(2)

    # CHANGE TO ADD RED / BLACK
    if template_black is not None:
        ax_templates_yass.plot(
            template_black[30:-30].T.flatten(),
            c="black"
            #         templates_yass[cluster_id_1, 30:-30, mc_plot - 5 : mc_plot + 5].T.flatten(),
            #         c=color_array_yass_close[0],
        )
    else:
        print("No black template")
    if template_red is not None:
        ax_templates_yass.plot(template_red[30:-30].T.flatten(), c="red")
    else:
        print("No red template")
    some_in_cluster = np.random.choice(
        list(range((labels_yass == cluster_id_1).sum())),
        replace=False,
        size=min((labels_yass == cluster_id_1).sum(), num_spikes_plot),
    )
    waveforms_unit = read_waveforms(
        spike_index_yass[labels_yass == cluster_id_1, 0][some_in_cluster],
        raw_bin,
        geom.shape[0],
        spike_length_samples=121,
        channels=np.arange(max(mc - 10, 0), min(mc + 10, geom.shape[0])).astype("int"),
    )[0]
    pcs_unit = pc_scatter.fit_transform(
        waveforms_unit.reshape(waveforms_unit.shape[0], -1)
    )

    ax_PCs_yass.scatter(
        pcs_unit[:, 0], pcs_unit[:, 1], s=2, c=color_array_yass_close[0]
    )

    for j in range(2):
        ax_templates_yass.plot(
            templates_yass[
                closest_clusters_hdb[j], 11:61+11, shared_mc - 5 : shared_mc + 6
            ].T.flatten(),
            c=color_array_yass_close[j + 1],
        )

        #         print("abs distance :")
        #         print(np.abs(templates_yass[closest_clusters_hdb[j], :, mc - 5 : mc + 5]-templates_yass[cluster_id_1, :, mc - 5 : mc + 5]).max())
        #         print("cosine distance :")
        #         print(scipy.spatial.distance.cosine(templates_yass[closest_clusters_hdb[j], :, mc - 5 : mc + 5].flatten(), templates_yass[cluster_id_1, :, mc - 5 : mc + 5].flatten()))

        some_in_cluster = np.random.choice(
            list(range((labels_yass == closest_clusters_hdb[j]).sum())),
            replace=False,
            size=min(
                (labels_yass == closest_clusters_hdb[j]).sum(), num_spikes_plot
            ),
        )
        waveforms_unit_bis = read_waveforms(
            spike_index_yass[labels_yass == closest_clusters_hdb[j], 0][
                some_in_cluster
            ],
            raw_bin,
            geom.shape[0],
            spike_length_samples=121,
            channels=np.arange(max(mc - 10, 0), min(mc + 10, geom.shape[0])).astype("int"),
        )[0]
        lda_labels = np.zeros(
            waveforms_unit_bis.shape[0] + waveforms_unit.shape[0]
        )
        lda_labels[: waveforms_unit.shape[0]] = 1
        lda_comps = LDA(n_components=1).fit_transform(
            apply_tpca(waveforms_unit, waveforms_unit_bis),
            lda_labels,
        )
        if j == 0:
            ax_LDA_yass = ax_LDA1_yass
        else:
            ax_LDA_yass = ax_LDA2_yass

        ax_LDA_yass.hist(
            lda_comps[: waveforms_unit.shape[0], 0],
            bins=25,
            color=color_array_yass_close[0],
            alpha=0.5,
        )
        ax_LDA_yass.hist(
            lda_comps[waveforms_unit.shape[0] :, 0],
            bins=25,
            color=color_array_yass_close[j + 1],
            alpha=0.5,
        )

    #         pcs_unit = pc_scatter.fit_transform(waveforms_unit_bis.reshape(waveforms_unit_bis.shape[0], -1))
    #         ax_PCs_yass.scatter(pcs_unit[:, 0], pcs_unit[:, 1], s=2, c=color_array_yass_close[j+1])
    ax_PCs_yass.yaxis.tick_right()

    for i in range(10):
        ax_templates_yass.axvline(61 + 61 * i, c="black")
    ax_templates_yass.set_xticks([])
    ax_templates_yass.set_title(
        f"{sorting1_name} close Units: {closest_clusters_hdb[0]}, {closest_clusters_hdb[1]}, {closest_clusters_hdb[2]}"
    )

    ax_LDA1_yass.set_xticks([])
    ax_LDA2_yass.set_xticks([])

    t0 = templates_yass[cluster_id_1, :, mc - 5 : mc + 5].T.flatten()
    t1 = templates_yass[
        closest_clusters_hdb[0], :, mc - 5 : mc + 5
    ].T.flatten()
    t2 = templates_yass[
        closest_clusters_hdb[1], :, mc - 5 : mc + 5
    ].T.flatten()
    try:
        ax_LDA1_yass.set_title(
            f"LDA: {closest_clusters_hdb[0]}, temp. dist {np.abs(t0 - t1).max():0.2f}"
        )
        ax_LDA2_yass.set_title(
            f"LDA: {closest_clusters_hdb[1]}, temp. dist {np.abs(t0 - t2).max():0.2f}"
        )
    except:
        print("problem neighbors")
    ax_PCs_yass.set_title("2 PCs")

    mc = templates_ks[cluster_id_2].ptp(0).argmax()

    ax_wfs_shared_ks.set_ylim(
        (
            templates_ks[cluster_id_2, :, mc].min() - 2,
            templates_ks[cluster_id_2, :, mc].max() + 2,
        )
    )
    ax_wfs_shared_ks.set_yticks(
        np.arange(
            templates_ks[cluster_id_2, :, mc].min() - 2,
            templates_ks[cluster_id_2, :, mc].max() + 2,
            1,
        ),
        minor=True,
    )
    ax_wfs_shared_ks.grid(which="both")

    color_array_ks_close = ["blue", "green", "magenta", "orange"]
    pc_scatter = PCA(2)

    # CHANGE TO ADD RED / BLACK
    if template_black is not None:
        ax_templates_ks.plot(
            template_black[30:-30].T.flatten(),
            c="black"
            #         templates_ks[cluster_id_2, 30:-30, mc_plot - 5 : mc_plot + 5].T.flatten(),
            #         c=color_array_ks_close[0],
        )
    else:
        print("again, no black template")
    if template_blue is not None:
        ax_templates_ks.plot(template_blue[30:-30].T.flatten(), c="blue")
    else:
        print("No blue template")
    some_in_cluster = np.random.choice(
        list(range((labels_ks == cluster_id_2).sum())),
        replace=False,
        size=min((labels_ks == cluster_id_2).sum(), num_spikes_plot),
    )
    load_times = spike_index_ks[labels_ks == cluster_id_2, 0][some_in_cluster].astype('int')
    waveforms_unit = read_waveforms(
        load_times,
        raw_bin,
        geom.shape[0],
        spike_length_samples=121,
        channels=np.arange(max(mc - 10, 0), min(mc + 10, geom.shape[0])).astype("int"),
    )[0]
    pcs_unit = pc_scatter.fit_transform(
        waveforms_unit.reshape(waveforms_unit.shape[0], -1)
    )

    ax_PCs_ks.scatter(
        pcs_unit[:, 0], pcs_unit[:, 1], s=2, c=color_array_ks_close[0]
    )

    for j in range(2):

        ax_templates_ks.plot(
            templates_ks[
                closest_clusters_kilo[j], 11:61+11, shared_mc - 5 : shared_mc + 6
            ].T.flatten(),
            c=color_array_ks_close[j + 1],
        )

        some_in_cluster = np.random.choice(
            list(range((labels_ks == closest_clusters_kilo[j]).sum())),
            replace=False,
            size=min(
                (labels_ks == closest_clusters_kilo[j]).sum(), num_spikes_plot
            ),
        )

        if (labels_ks == closest_clusters_kilo[j]).sum() > 0:
            waveforms_unit_bis = read_waveforms(
                spike_index_ks[labels_ks == closest_clusters_kilo[j], 0][
                    some_in_cluster
                ],
                raw_bin,
                geom.shape[0],
                spike_length_samples=121,
                channels=np.arange(max(mc - 10, 0), min(mc + 10, geom.shape[0])).astype("int"),
            )[0]
            # DO TPCA
            lda_labels = np.zeros(
                waveforms_unit_bis.shape[0] + waveforms_unit.shape[0]
            )
            lda_labels[: waveforms_unit.shape[0]] = 1
            lda_comps = LDA(n_components=1).fit_transform(
                apply_tpca(waveforms_unit, waveforms_unit_bis),
                lda_labels,
            )
            if j == 0:
                ax_LDA1_ks.hist(
                    lda_comps[: waveforms_unit.shape[0], 0],
                    bins=25,
                    color=color_array_ks_close[0],
                    alpha=0.5,
                )
                ax_LDA1_ks.hist(
                    lda_comps[waveforms_unit.shape[0] :, 0],
                    bins=25,
                    color=color_array_ks_close[j + 1],
                    alpha=0.5,
                )
            else:
                ax_LDA2_ks.hist(
                    lda_comps[: waveforms_unit.shape[0], 0],
                    bins=25,
                    color=color_array_ks_close[0],
                    alpha=0.5,
                )
                ax_LDA2_ks.hist(
                    lda_comps[waveforms_unit.shape[0] :, 0],
                    bins=25,
                    color=color_array_ks_close[j + 1],
                    alpha=0.5,
                )
    #         if len(waveforms_unit_bis) > 2:
    #             pcs_unit = pc_scatter.fit_transform(waveforms_unit_bis.reshape(waveforms_unit_bis.shape[0], -1))
    #             ax_PCs_ks.scatter(pcs_unit[:, 0], pcs_unit[:, 1], s=2, c=color_array_ks_close[j+1])
    ax_LDA1_ks.set_xticks([])
    ax_LDA2_ks.set_xticks([])
    ax_PCs_ks.yaxis.tick_right()

    t0 = templates_ks[cluster_id_2, :, mc - 5 : mc + 5]
    t1 = templates_ks[closest_clusters_kilo[0], :, mc - 5 : mc + 5]
    t2 = templates_ks[closest_clusters_kilo[1], :, mc - 5 : mc + 5]
    try:
        ax_LDA1_ks.set_title(
            f"LDA: {closest_clusters_kilo[0]}, temp. dist {np.abs(t0 - t1).max():0.2f}"
        )
        ax_LDA2_ks.set_title(
            f"LDA: {closest_clusters_kilo[1]}, temp. dist {np.abs(t0 - t2).max():0.2f}"
        )
    except Exception as e:
        print("problem KS neighbors", e)
    ax_PCs_ks.set_title("2 PCs")

    for i in range(10):
        ax_templates_ks.axvline(61 + 61 * i, c="black")
    ax_templates_ks.set_xticks([])
    ax_templates_ks.set_title(
        f"{sorting2_name} close Units: {closest_clusters_kilo[0]}, {closest_clusters_kilo[1]}"
    )

    return fig, np.round(agreement, 2) * 100


# %% [markdown]
# def plot_unit_similarities_summary(cluster_id, closest_clusters, sorting1, sorting2, geom, raw_data_bin, recoring_duration, num_channels=40, num_spikes_plot=100, num_channels_similarity=20,
#                                    num_close_clusters_plot=10, num_close_clusters=30, shifts_align = np.arange(-3,4), order_by ='similarity', normalize_agreement_by="both", denoised_waveforms=None,
#                                    cluster_labels=None, non_triaged_idxs=None, triaged_mcs_abs=None, triaged_firstchans=None):

# %% [markdown]
#     # ###Kilosort
#     cluster_ids_all = sorting1.get_unit_ids()
#     cluster_ids_list = [cluster_ids_all[i * n:(i + 1) * n] for i in range((len(cluster_ids_all) + n - 1) // n )]
#     for subset_id, cluster_ids in enumerate(cluster_ids_list):
#     np.random.seed(0)
#     cluster_ids = np.random.choice(sorting_kilo.get_unit_ids(), 50)
#     num_clusters = len(cluster_ids)
#     fig = plt.figure(figsize=(32, 3*num_clusters))
#     gs = gridspec.GridSpec(num_clusters,9)
#     gs.update(hspace=0.5)
#     num_close_clusters_plot = 10
#     for i, cluster_id in enumerate(tqdm(cluster_ids)):
#         matplotlib.rcParams.update({'font.size': 10})
#         ax_cid = plt.subplot(gs[i,0])
#         ax_fr = plt.subplot(gs[i,1])
#         ax_maxptp = plt.subplot(gs[i,2])
#         ax_cos = plt.subplot(gs[i,3:5])
#         ax_agree  = plt.subplot(gs[i,5:7])
#         ax_isi = plt.subplot(gs[i,7:9])

# %% [markdown]
#         st_1 = sorting_kilo.get_unit_spike_train(cluster_id)
#         firing_rate = len(st_1) / recording_duration #in seconds
#         waveforms1 = read_waveforms(st_1, raw_data_bin, geom, n_times=121)[0]
#         template1 = np.mean(waveforms1, axis=0)
#         max_ptp_channel = np.argmax(template1.ptp(0))
#         max_ptp = np.max(template1.ptp(0))
#         channel_range = (max(max_ptp_channel-num_channels_cosine//2,0),max_ptp_channel+num_channels_cosine//2)
#         template1 = template1[:,channel_range[0]:channel_range[1]]

# %% [markdown]
#         #compute K closest clsuters
#         curr_cluster_depth = kilo_cluster_depth_means[cluster_id]
#         dist_to_other_cluster_dict = {cluster_id:abs(mean_depth-curr_cluster_depth) for (cluster_id,mean_depth) in kilo_cluster_depth_means.items()}
#         closest_clusters = [y[0] for y in sorted(dist_to_other_cluster_dict.items(), key = lambda x: x[1])[1:1+num_close_clusters]]

# %% [markdown]
#         similarities = []
#         agreements = []
#         for closest_cluster in closest_clusters:
#             st_2 = sorting_kilo.get_unit_spike_train(closest_cluster)
#             waveforms2 = read_waveforms(st_2, raw_data_bin, geom, n_times=121)[0]
#             template2 = np.mean(waveforms2, axis=0)[:,channel_range[0]:channel_range[1]]
#             # similarity = sklearn.metrics.pairwise.cosine_similarity(np.expand_dims(template1.flatten(),0), np.expand_dims(template2.flatten(),0))
#             # similarities.append(similarity[0][0])
#             similarity = np.max(np.abs(template1 - template2))
#             similarities.append(similarity)
#             ind_st1, ind_st2, not_match_ind_st1, not_match_ind_st2 = compute_spiketrain_agreement(st_1, st_2, delta_frames=12)
#             agreement = len(ind_st1) / (len(st_1) + len(st_2) - len(ind_st1))
#             agreements.append(agreement)

# %% [markdown]
#         agreements = np.asarray(agreements).round(2)
#         similarities = np.asarray(similarities).round(2)
#         closest_clusters = np.asarray(closest_clusters)
#         # most_similar_idxs = np.flip(np.argsort(similarities))
#         most_similar_idxs = np.argsort(similarities)
#         agreements = agreements[most_similar_idxs]
#         similarities = similarities[most_similar_idxs]
#         closest_clusters = closest_clusters[most_similar_idxs]

# %% [markdown]
#         y_axis_labels = [f"Unit {cluster_id}"]
#         x_axis_labels = closest_clusters
#         g = sns.heatmap(np.expand_dims(similarities,0), vmin=0, vmax=max(similarities), cmap='RdYlGn_r', annot=np.expand_dims(similarities,0),xticklabels=x_axis_labels, yticklabels=y_axis_labels, ax=ax_cos,cbar=False)
#         ax_cos.set_title("Cosine Similarity");
#         g = sns.heatmap(np.expand_dims(agreements,0), vmin=0, vmax=1, cmap='RdYlGn', annot=np.expand_dims(agreements,0),xticklabels=x_axis_labels, yticklabels=y_axis_labels, ax=ax_agree,cbar=False)
#         ax_agree.set_title("Agreement");

# %% [markdown]
#         plot_isi_distribution(st_1, ax=ax_isi);
#         matplotlib.rcParams.update({'font.size': 22})

# %% [markdown]
#         ax_cid.text(0.15, 0.45, f"Unit id: {cluster_id}")
#         ax_cid.set_xticks([])
#         ax_cid.set_yticks([])

# %% [markdown]
#         ax_fr.text(0.15, 0.45, f"FR: {'%.1f' % round(firing_rate,2)} Hz")
#         ax_fr.set_xticks([])
#         ax_fr.set_yticks([])

# %% [markdown]
#         ax_maxptp.text(0.1, 0.45, f"max ptp: {'%.1f' % round(max_ptp,2)}")
#         ax_maxptp.set_xticks([])
#         ax_maxptp.set_yticks([])
#     plt.close(fig)
#     fig.savefig(f"kilosort_cluster_summaries_norm.png")


# %%
def get_outliers_wfs(    
    spike_times,
    raw_bin,
    shared_mc,
    geom,
    num_spikes_plot=100,

# %%
):
    some_in_cluster = np.random.default_rng(0).choice(
        list(range(len(spike_times))),
        replace=False,
        size=min(len(spike_times), num_spikes_plot),
    )

    spike_times = spike_times[some_in_cluster]
    first_chans_cluster = first_chans_cluster[some_in_cluster]


    if raw_bin is not None:
        start = np.maximum(shared_mc - 5, 0)
        end = start + 11                
        waveforms = read_waveforms(
            spike_times, raw_bin, geom.shape[0], channels = np.arange(start, end), spike_length_samples=121
        )[0]
        return waveforms
    else:
        return None


# %%

# %%
def diagnostic_plots_with_outliers(
    cluster_id_1,
    cluster_id_2,
    st_1,
    spike_times_outliers_1,
    st_2,
    templates_yass,
    templates_ks,
    mcs_abs_cluster_sorting1,
    mcs_abs_cluster_sorting2,
    geom,
    raw_bin,
    hdb_cluster_depth_means,
    kilo_cluster_depth_means,
    spike_index_yass,
    spike_index_ks,
    labels_yass,
    labels_ks,
    closest_clusters_hdb,
    closest_clusters_kilo,
    scale=7,
    sorting1_name="1",
    sorting2_name="2",
    num_channels=40,
    num_spikes_plot=100,
    t_range=(30, 90),
    num_rows=3,
    alpha=0.1,
    delta_frames=12,
    num_close_clusters=5,
    tpca_rank=5,
):
    print(f"{st_1.shape=}, {st_2.shape=}")

    if not st_1.size and not st_2.size:
        raise ValueError("nothing in either spike train")
    lab_st1 = cluster_id_1
    lab_st2 = cluster_id_2
    (
        ind_st1,
        ind_st2,
        not_match_ind_st1,
        not_match_ind_st2,
    ) = compute_spiketrain_agreement(st_1, st_2, delta_frames)
    print(f"{ind_st1.shape=}, {ind_st2.shape=}, {not_match_ind_st1.shape=}, {not_match_ind_st2.shape=}")
    agreement = len(ind_st1) / (len(st_1) + len(st_2) - len(ind_st1))
    fig = plt.figure(figsize=(12, 20))

    gs = fig.add_gridspec(9, 12, height_ratios=(5, 1, 2, 2, 2, 2, 1, 2, 1))

    ax_sorting1 = fig.add_subplot(gs[0, :6])
    ax_sorting2 = fig.add_subplot(gs[0, 6:])

    ax_venn = fig.add_subplot(gs[1, :3])
    ax_lda_blue_yellow = fig.add_subplot(gs[1, 3:6])
    ax_lda_red_yellow = fig.add_subplot(gs[1, 6:9])
    ax_lda_blue_red = fig.add_subplot(gs[1, 9:])

    ax_wfs_shared_yass = fig.add_subplot(gs[2, :])
    ax_wfs_shared_ks = fig.add_subplot(gs[3, :])
    ax_wfs_outliers = fig.add_subplot(gs[4, :])

    ax_templates_yass = fig.add_subplot(gs[5, :])
    ax_isi_yass = fig.add_subplot(gs[6, :3])
    ax_LDA1_yass = fig.add_subplot(gs[6, 3:6])
    ax_LDA2_yass = fig.add_subplot(gs[6, 6:9])
    ax_PCs_yass = fig.add_subplot(gs[6, 9:])

    ax_templates_ks = fig.add_subplot(gs[7, :])
    ax_isi_ks = fig.add_subplot(gs[8, :3])
    ax_LDA1_ks = fig.add_subplot(gs[8, 3:6])
    ax_LDA2_ks = fig.add_subplot(gs[8, 6:9])
    ax_PCs_ks = fig.add_subplot(gs[8, 9:])

    list_ax_wfs_shared_ks = [0]
    for cm_x in np.arange(0, 675, 10):
        if cm_x > 0:
            if cm_x % 60 == 0:
                list_ax_wfs_shared_ks.append(cm_x + 1)
            else:
                list_ax_wfs_shared_ks.append(cm_x)

    ax_wfs_shared_yass.set_xticks(list_ax_wfs_shared_ks, minor=True)
    #     ax_test_split_ptp = fig.add_subplot(gs[8, 3:6])
    #     ax_test_split_temp_proj = fig.add_subplot(gs[8, 6:9])

    plot_isi_distribution(st_1, ax=ax_isi_yass)
    plot_isi_distribution(st_2, ax=ax_isi_ks)

    def apply_tpca(wfs_a, wfs_b):
        wfs = np.r_[wfs_a.transpose(0, 2, 1), wfs_b.transpose(0, 2, 1)]
        N, C, T = wfs.shape
        tpca = PCA(tpca_rank)
        wfs = tpca.fit_transform(wfs.reshape(-1, T))
        wfs = wfs.reshape(N, C, tpca_rank).reshape(N, -1)
        return wfs

    subsets = [len(not_match_ind_st1), len(not_match_ind_st2), len(ind_st1)]
    v = venn2(
        subsets=subsets,
        set_labels=[
            f"{sorting1_name} {lab_st1}",
            f"{sorting2_name} {lab_st2}",
        ],
        ax=ax_venn,
    )
    if len(not_match_ind_st1) > 0:
        v.get_patch_by_id("10").set_color("red")
    if len(not_match_ind_st2) > 0:
        v.get_patch_by_id("01").set_color("blue")
    if len(ind_st1) > 0:
        v.get_patch_by_id("11").set_color("goldenrod")
    for text in v.subset_labels:
        if text is not None:
            text.set_fontsize(6)
    for text in v.set_labels:
        if text is not None:
            text.set_fontsize(8)

    ax_venn.set_title(f"{np.round(agreement, 2)*100}% agreement")

    ax_lda_blue_red.set_title(f"Distinct {sorting2_name}/{sorting1_name} LDA")
    ax_lda_blue_yellow.set_title(f"Shared/{sorting2_name} LDA")
    ax_lda_red_yellow.set_title(f"Shared/{sorting1_name} LDA")

    # fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,12))
    h_shifts = [-10, 10]
    colors = ["goldenrod", "red"]
    indices = [ind_st1, not_match_ind_st1]
    # FIX CHANNEL INDEX!!
    
    vals, counts = np.unique(
        mcs_abs_cluster_sorting1,
        return_counts=True,
    )
    z_uniq, z_ids = np.unique(geom[:, 1], return_inverse=True)
    mcid = z_ids[vals[counts.argmax()]]
    
    cmp = 0
    wfs_shared = np.array([])
    wfs_lda_red = np.array([])
    if ind_st1.size:
        u, c = np.unique(mcs_abs_cluster_sorting1[ind_st1], return_counts=True)
        shared_mc = u[c.argmax()]
    elif not_match_ind_st1.size:
        u, c = np.unique(mcs_abs_cluster_sorting1[not_match_ind_st1], return_counts=True)
        shared_mc = u[c.argmax()]
    elif not_match_ind_st2.size:
        u, c = np.unique(mcs_abs_cluster_sorting2[not_match_ind_st2], return_counts=True)
        shared_mc = u[c.argmax()]
    else:
        assert False
    template_red = None
    template_black = None
    for indices_match, color, h_shift in zip(indices, colors, h_shifts):
        if len(indices_match) > 0:
            # firstchans_cluster_sorting = firstchans_cluster_sorting1[
            #     indices_match
            # ]
            # mcs_abs_cluster_sorting = mcs_abs_cluster_sorting1[indices_match]
            mcs_abs_cluster_sorting = np.full(indices_match.shape, shared_mc)
#             print(mcs_abs_cluster_sorting)
            firstchans_cluster_sorting = np.maximum(
                mcs_abs_cluster_sorting - 20, 0
            )
            spike_times = st_1[indices_match]
            # geom, first_chans_cluster, mcs_abs_cluster, max_ptps_cluster, spike_times,
            (
                waveforms,
                first_chan_cluster,
            ) = plot_waveforms_geom_unit_with_return(
                geom,
                firstchans_cluster_sorting,
                mcs_abs_cluster_sorting,
                spike_times,
                z_ids, 
                mcid,
                raw_bin=raw_bin,
                num_spikes_plot=num_spikes_plot,
                t_range=t_range,
                num_channels=num_channels,
                num_rows=num_rows,
                do_mean=False,
                scale=scale,
                h_shift=h_shift,
                alpha=alpha,
                ax=ax_sorting1,
                color=color,
            )
            mc = (
                shared_mc - firstchans_cluster_sorting
            )  # waveforms.mean(0).ptp(0).argmax()
            start = np.maximum(mc - 5, 0)
            end = start + 11
            for i in range(min(len(waveforms), num_spikes_plot)):
                ax_wfs_shared_yass.plot(
                    waveforms[i, 30:-30, start[i] : end[i]].T.flatten(),
                    alpha=alpha,
                    color=color,
                )
            for i in range(10):
                ax_wfs_shared_yass.axvline(61 + 61 * i, c="black")
            #             ax_templates.plot(waveforms[:, :, mc-5:mc+6].mean(0).T.flatten(), color=color)
            cmp += 1
            # print("waveforms", waveforms.shape)
            if color == "goldenrod":
                wfs_shared = np.stack(
                    [
                        waveforms[i, :, start[i] : end[i]]
                        for i in range(len(waveforms))
                    ],
                    axis=0,
                )
                template_black = wfs_shared.mean(0)
                # print("shared", wfs_shared.shape, mc, flush=True)
            if color == "red":
                wfs_lda_red = np.stack(
                    [
                        waveforms[i, :, start[i] : end[i]]
                        for i in range(len(waveforms))
                    ],
                    axis=0,
                )
                template_red = wfs_lda_red.mean(0)
                # print("red", wfs_shared.shape, mc, flush=True)

    ## PTP/ template split
    #     waveforms_yass_to_split = np.concatenate((wfs_shared, wfs_lda_red))
    #     mc = templates_yass[cluster_id_1].ptp(0).argmax()

    #     ptps = waveforms_yass_to_split[:, :, 5].ptp(1)
    #     value_dpt, cut_calue = isocut(ptps)

    #     ax_test_split_ptp.hist(ptps, bins = 50)
    #     ax_test_split_ptp.set_title(str(value_dpt) + '\n' + str(cut_calue))

    #     temp_unit = templates_yass[cluster_id_1, :, mc]
    #     norm_wfs = np.sqrt(np.square(waveforms_yass_to_split[:, :, 5]).sum(1))
    #     temp_proj = np.einsum('ij,j->i', waveforms_yass_to_split[:, :, 5], templates_yass[cluster_id_1, :, mc])/norm_wfs
    #     value_dpt, cut_calue = isocut(temp_proj)

    #     ax_test_split_temp_proj.hist(temp_proj, bins = 50)
    #     ax_test_split_temp_proj.set_title(str(value_dpt) + '\n' + str(cut_calue))
    #     ax_wfs_shared_yass.set_xticks([])

    #     wfs_mc = waveforms_yass_to_split[:, :, 5]
    #     wfs_mc = wfs_mc[wfs_mc.ptp(1).argsort()]
    #     lower = int(waveforms_yass_to_split.shape[0]*0.05)
    #     upper = int(waveforms_yass_to_split.shape[0]*0.95)
    #     max_diff = 0
    #     max_diff_N = 0
    #     for n in tqdm(np.arange(lower, upper)):
    #         temp_1 = np.mean(wfs_mc[:n], axis = 0)
    #         temp_2 = np.mean(wfs_mc[n:], axis = 0)
    #         if np.abs(temp_1-temp_2).max() > max_diff:
    #             max_diff = np.abs(temp_1-temp_2).max()
    #             max_diff_N = n
    #     print(max_diff)
    #     print(max_diff_N)

    colors = ["goldenrod", "blue"]
    indices = [ind_st2, not_match_ind_st2]
    wfs_lda_blue = np.array([])
    template_blue = None

    for indices_match, color, h_shift in zip(indices, colors, h_shifts):
        if len(indices_match) > 0:
            mcs_abs_cluster_sorting = np.full(indices_match.shape, shared_mc)
            firstchans_cluster_sorting = np.maximum(
                mcs_abs_cluster_sorting - 20, 0
            )
            # mcs_abs_cluster_sorting = mcs_abs_cluster_sorting2[indices_match]
            # firstchans_cluster_sorting = firstchans_cluster_sorting2[
            #     indices_match
            # ]
            spike_times = st_2[indices_match]

            (
                waveforms,
                first_chan_cluster,
            ) = plot_waveforms_geom_unit_with_return(
                geom,
                firstchans_cluster_sorting,
                mcs_abs_cluster_sorting,
                spike_times,
                z_ids, 
                mcid,
                raw_bin=raw_bin,
                num_spikes_plot=num_spikes_plot,
                t_range=t_range,
                num_channels=num_channels,
                num_rows=num_rows,
                do_mean=False,
                scale=scale,
                h_shift=h_shift,
                alpha=alpha,
                ax=ax_sorting2,
                color=color,
            )
            mc = (
                shared_mc - firstchans_cluster_sorting
            )  # waveforms.mean(0).ptp(0).argmax()
            start = np.maximum(mc - 5, 0)
            end = start + 11
            for i in range(min(len(waveforms), num_spikes_plot)):
                ax_wfs_shared_ks.plot(
                    waveforms[i, 30:-30, start[i] : end[i]].T.flatten(),
                    alpha=alpha,
                    color=color,
                )
            for i in range(10):
                ax_wfs_shared_ks.axvline(61 + 61 * i, c="black")
            if color == "blue":
                wfs_lda_blue = np.stack(
                    [
                        waveforms[i, :, start[i] : end[i]]
                        for i in range(len(waveforms))
                    ],
                    axis=0,
                )
                template_blue = wfs_lda_blue.mean(0)

    waveforms_outliers = get_outliers_wfs(    
        spike_times_outliers_1,
        raw_bin,
        shared_mc,
        geom
    )

    if waveforms_outliers is not None:
        for i in range(len(waveforms_outliers)):
            ax_wfs_outliers.plot(
                waveforms_outliers[i, 15:-15].T.flatten(),
                alpha=0.05,
                color='k',
            )
        for i in range(10):
            ax_wfs_shared_ks.axvline(91 + 91 * i, c="black")

        ax_wfs_shared_yass.grid(which="both")    
        ax_wfs_shared_yass.set_title("Outliers")



    lda_labels = np.zeros(wfs_shared.shape[0] + wfs_lda_red.shape[0])
    lda_labels[: wfs_shared.shape[0]] = 1
    if wfs_lda_red.size and wfs_shared.size:
        lda_comps = LDA(n_components=1).fit_transform(
            apply_tpca(wfs_shared, wfs_lda_red),
            lda_labels,
        )
        ax_lda_red_yellow.hist(
            lda_comps[: wfs_shared.shape[0], 0],
            bins=25,
            color="goldenrod",
            alpha=0.5,
        )
        ax_lda_red_yellow.hist(
            lda_comps[wfs_shared.shape[0] :, 0],
            bins=25,
            color="red",
            alpha=0.5,
        )

    if wfs_lda_blue.size and wfs_shared.size:
        lda_labels = np.zeros(wfs_shared.shape[0] + wfs_lda_blue.shape[0])
        lda_labels[: wfs_shared.shape[0]] = 1
        lda_comps = LDA(n_components=1).fit_transform(
            apply_tpca(wfs_shared, wfs_lda_blue),
            lda_labels,
        )
        ax_lda_blue_yellow.hist(
            lda_comps[: wfs_shared.shape[0], 0],
            bins=25,
            color="goldenrod",
            alpha=0.5,
        )
        ax_lda_blue_yellow.hist(
            lda_comps[wfs_shared.shape[0] :, 0],
            bins=25,
            color="blue",
            alpha=0.5,
        )

    if (
        len(not_match_ind_st1) > 0
        and len(not_match_ind_st2) > 0
        and wfs_lda_red.size
        and wfs_lda_blue.size
    ):
        lda_labels = np.zeros(wfs_lda_blue.shape[0] + wfs_lda_red.shape[0])
        lda_labels[: wfs_lda_blue.shape[0]] = 1
        lda_comps = LDA(n_components=1).fit_transform(
            apply_tpca(wfs_lda_blue, wfs_lda_red),
            lda_labels,
        )
        ax_lda_blue_red.hist(
            lda_comps[: wfs_lda_blue.shape[0], 0],
            bins=25,
            color="red",
            alpha=0.5,
        )
        ax_lda_blue_red.hist(
            lda_comps[wfs_lda_blue.shape[0] :, 0],
            bins=25,
            color="blue",
            alpha=0.5,
        )

    #     ax_wfs_shared_ks.set_xticks([])
    ax_wfs_shared_ks.set_xticks(list_ax_wfs_shared_ks, minor=True)

    mc = templates_yass[cluster_id_1].ptp(0).argmax()

    ax_wfs_shared_yass.set_ylim(
        (
            templates_yass[cluster_id_1, :, mc].min() - 2,
            templates_yass[cluster_id_1, :, mc].max() + 2,
        )
    )

    ax_wfs_shared_yass.set_yticks(
        np.arange(
            templates_yass[cluster_id_1, :, mc].min() - 2,
            templates_yass[cluster_id_1, :, mc].max() + 2,
            1,
        ),
        minor=True,
    )

    ax_wfs_shared_yass.grid(which="both")    
    color_array_yass_close = ["red", "cyan", "lime", "fuchsia"]
    pc_scatter = PCA(2)

    # CHANGE TO ADD RED / BLACK
    if template_black is not None:
        ax_templates_yass.plot(
            template_black[30:-30].T.flatten(),
            c="black"
            #         templates_yass[cluster_id_1, 30:-30, mc_plot - 5 : mc_plot + 5].T.flatten(),
            #         c=color_array_yass_close[0],
        )
    else:
        print("No black template")
    if template_red is not None:
        ax_templates_yass.plot(template_red[30:-30].T.flatten(), c="red")
    else:
        print("No red template")
    some_in_cluster = np.random.choice(
        list(range((labels_yass == cluster_id_1).sum())),
        replace=False,
        size=min((labels_yass == cluster_id_1).sum(), num_spikes_plot),
    )
    waveforms_unit = read_waveforms(
        spike_index_yass[labels_yass == cluster_id_1, 0][some_in_cluster],
        raw_bin,
        geom.shape[0],
        spike_length_samples=121,
        channels=np.arange(mc - 10, mc + 10).astype("int"),
    )[0]
    pcs_unit = pc_scatter.fit_transform(
        waveforms_unit.reshape(waveforms_unit.shape[0], -1)
    )

    ax_PCs_yass.scatter(
        pcs_unit[:, 0], pcs_unit[:, 1], s=2, c=color_array_yass_close[0]
    )

    for j in range(2):
        ax_templates_yass.plot(
            templates_yass[
                closest_clusters_hdb[j], 11:61+11, shared_mc - 5 : shared_mc + 6
            ].T.flatten(),
            c=color_array_yass_close[j + 1],
        )

        #         print("abs distance :")
        #         print(np.abs(templates_yass[closest_clusters_hdb[j], :, mc - 5 : mc + 5]-templates_yass[cluster_id_1, :, mc - 5 : mc + 5]).max())
        #         print("cosine distance :")
        #         print(scipy.spatial.distance.cosine(templates_yass[closest_clusters_hdb[j], :, mc - 5 : mc + 5].flatten(), templates_yass[cluster_id_1, :, mc - 5 : mc + 5].flatten()))

        some_in_cluster = np.random.choice(
            list(range((labels_yass == closest_clusters_hdb[j]).sum())),
            replace=False,
            size=min(
                (labels_yass == closest_clusters_hdb[j]).sum(), num_spikes_plot
            ),
        )
        waveforms_unit_bis = read_waveforms(
            spike_index_yass[labels_yass == closest_clusters_hdb[j], 0][
                some_in_cluster
            ],
            raw_bin,
            geom.shape[0],
            spike_length_samples=121,
            channels=np.arange(mc - 10, mc + 10).astype("int"),
        )[0]
        lda_labels = np.zeros(
            waveforms_unit_bis.shape[0] + waveforms_unit.shape[0]
        )
        lda_labels[: waveforms_unit.shape[0]] = 1
        lda_comps = LDA(n_components=1).fit_transform(
            apply_tpca(waveforms_unit, waveforms_unit_bis),
            lda_labels,
        )
        if j == 0:
            ax_LDA_yass = ax_LDA1_yass
        else:
            ax_LDA_yass = ax_LDA2_yass

        ax_LDA_yass.hist(
            lda_comps[: waveforms_unit.shape[0], 0],
            bins=25,
            color=color_array_yass_close[0],
            alpha=0.5,
        )
        ax_LDA_yass.hist(
            lda_comps[waveforms_unit.shape[0] :, 0],
            bins=25,
            color=color_array_yass_close[j + 1],
            alpha=0.5,
        )

    #         pcs_unit = pc_scatter.fit_transform(waveforms_unit_bis.reshape(waveforms_unit_bis.shape[0], -1))
    #         ax_PCs_yass.scatter(pcs_unit[:, 0], pcs_unit[:, 1], s=2, c=color_array_yass_close[j+1])
    ax_PCs_yass.yaxis.tick_right()

    for i in range(10):
        ax_templates_yass.axvline(61 + 61 * i, c="black")
    ax_templates_yass.set_xticks([])
    ax_templates_yass.set_title(
        f"{sorting1_name} close Units: {closest_clusters_hdb[0]}, {closest_clusters_hdb[1]}, {closest_clusters_hdb[2]}"
    )

    ax_LDA1_yass.set_xticks([])
    ax_LDA2_yass.set_xticks([])

    t0 = templates_yass[cluster_id_1, :, mc - 5 : mc + 5].T.flatten()
    t1 = templates_yass[
        closest_clusters_hdb[0], :, mc - 5 : mc + 5
    ].T.flatten()
    t2 = templates_yass[
        closest_clusters_hdb[1], :, mc - 5 : mc + 5
    ].T.flatten()
    try:
        ax_LDA1_yass.set_title(
            f"LDA: {closest_clusters_hdb[0]}, temp. dist {np.abs(t0 - t1).max():0.2f}"
        )
        ax_LDA2_yass.set_title(
            f"LDA: {closest_clusters_hdb[1]}, temp. dist {np.abs(t0 - t2).max():0.2f}"
        )
    except:
        print("problem neighbors")
    ax_PCs_yass.set_title("2 PCs")

    mc = templates_ks[cluster_id_2].ptp(0).argmax()

    ax_wfs_shared_ks.set_ylim(
        (
            templates_ks[cluster_id_2, :, mc].min() - 2,
            templates_ks[cluster_id_2, :, mc].max() + 2,
        )
    )
    ax_wfs_shared_ks.set_yticks(
        np.arange(
            templates_ks[cluster_id_2, :, mc].min() - 2,
            templates_ks[cluster_id_2, :, mc].max() + 2,
            1,
        ),
        minor=True,
    )
    ax_wfs_shared_ks.grid(which="both")

    color_array_ks_close = ["blue", "green", "magenta", "orange"]
    pc_scatter = PCA(2)

    # CHANGE TO ADD RED / BLACK
    if template_black is not None:
        ax_templates_ks.plot(
            template_black[30:-30].T.flatten(),
            c="black"
            #         templates_ks[cluster_id_2, 30:-30, mc_plot - 5 : mc_plot + 5].T.flatten(),
            #         c=color_array_ks_close[0],
        )
    else:
        print("again, no black template")
    if template_blue is not None:
        ax_templates_ks.plot(template_blue[30:-30].T.flatten(), c="blue")
    else:
        print("No blue template")
    some_in_cluster = np.random.choice(
        list(range((labels_ks == cluster_id_2).sum())),
        replace=False,
        size=min((labels_ks == cluster_id_2).sum(), num_spikes_plot),
    )
    load_times = spike_index_ks[labels_ks == cluster_id_2, 0][some_in_cluster].astype('int')
    waveforms_unit = read_waveforms(
        load_times,
        raw_bin,
        geom.shape[0],
        spike_length_samples=121,
        channels=np.arange(mc - 10, mc + 10).astype('int'),
    )[0]
    pcs_unit = pc_scatter.fit_transform(
        waveforms_unit.reshape(waveforms_unit.shape[0], -1)
    )

    ax_PCs_ks.scatter(
        pcs_unit[:, 0], pcs_unit[:, 1], s=2, c=color_array_ks_close[0]
    )

    for j in range(2):

        ax_templates_ks.plot(
            templates_ks[
                closest_clusters_kilo[j], 11:61+11, shared_mc - 5 : shared_mc + 6
            ].T.flatten(),
            c=color_array_ks_close[j + 1],
        )

        some_in_cluster = np.random.choice(
            list(range((labels_ks == closest_clusters_kilo[j]).sum())),
            replace=False,
            size=min(
                (labels_ks == closest_clusters_kilo[j]).sum(), num_spikes_plot
            ),
        )

        if (labels_ks == closest_clusters_kilo[j]).sum() > 0:
            waveforms_unit_bis = read_waveforms(
                spike_index_ks[labels_ks == closest_clusters_kilo[j], 0][
                    some_in_cluster
                ],
                raw_bin,
                geom.shape[0],
                spike_length_samples=121,
                channels=np.arange(mc - 10, mc + 10).astype("int"),
            )[0]
            # DO TPCA
            lda_labels = np.zeros(
                waveforms_unit_bis.shape[0] + waveforms_unit.shape[0]
            )
            lda_labels[: waveforms_unit.shape[0]] = 1
            lda_comps = LDA(n_components=1).fit_transform(
                apply_tpca(waveforms_unit, waveforms_unit_bis),
                lda_labels,
            )
            if j == 0:
                ax_LDA1_ks.hist(
                    lda_comps[: waveforms_unit.shape[0], 0],
                    bins=25,
                    color=color_array_ks_close[0],
                    alpha=0.5,
                )
                ax_LDA1_ks.hist(
                    lda_comps[waveforms_unit.shape[0] :, 0],
                    bins=25,
                    color=color_array_ks_close[j + 1],
                    alpha=0.5,
                )
            else:
                ax_LDA2_ks.hist(
                    lda_comps[: waveforms_unit.shape[0], 0],
                    bins=25,
                    color=color_array_ks_close[0],
                    alpha=0.5,
                )
                ax_LDA2_ks.hist(
                    lda_comps[waveforms_unit.shape[0] :, 0],
                    bins=25,
                    color=color_array_ks_close[j + 1],
                    alpha=0.5,
                )
    #         if len(waveforms_unit_bis) > 2:
    #             pcs_unit = pc_scatter.fit_transform(waveforms_unit_bis.reshape(waveforms_unit_bis.shape[0], -1))
    #             ax_PCs_ks.scatter(pcs_unit[:, 0], pcs_unit[:, 1], s=2, c=color_array_ks_close[j+1])
    ax_LDA1_ks.set_xticks([])
    ax_LDA2_ks.set_xticks([])
    ax_PCs_ks.yaxis.tick_right()

    t0 = templates_ks[cluster_id_2, :, mc - 5 : mc + 5]
    t1 = templates_ks[closest_clusters_kilo[0], :, mc - 5 : mc + 5]
    t2 = templates_ks[closest_clusters_kilo[1], :, mc - 5 : mc + 5]
    try:
        ax_LDA1_ks.set_title(
            f"LDA: {closest_clusters_kilo[0]}, temp. dist {np.abs(t0 - t1).max():0.2f}"
        )
        ax_LDA2_ks.set_title(
            f"LDA: {closest_clusters_kilo[1]}, temp. dist {np.abs(t0 - t2).max():0.2f}"
        )
    except Exception as e:
        print("problem KS neighbors", e)
    ax_PCs_ks.set_title("2 PCs")

    for i in range(10):
        ax_templates_ks.axvline(61 + 61 * i, c="black")
    ax_templates_ks.set_xticks([])
    ax_templates_ks.set_title(
        f"{sorting2_name} close Units: {closest_clusters_kilo[0]}, {closest_clusters_kilo[1]}"
    )

    return fig, np.round(agreement, 2) * 100
