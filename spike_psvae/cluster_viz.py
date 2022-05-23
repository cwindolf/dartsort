import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import colorcet
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib import cm
# matplotlib.use('Agg')
from matplotlib_venn import venn3, venn3_circles, venn2
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
import spikeinterface 
from spikeinterface.toolkit import compute_correlograms
from spikeinterface.comparison import compare_two_sorters
from spikeinterface.widgets import plot_agreement_matrix
from tqdm import tqdm
from matplotlib_venn import venn3, venn3_circles, venn2
import seaborn as sns
import matplotlib.gridspec as gridspec
matplotlib.rcParams.update({'font.size': 10})
from spike_psvae.triage import run_weighted_triage
from spike_psvae.cluster_utils import make_sorting_from_labels_frames, compute_cluster_centers, relabel_by_depth, remove_duplicate_units, read_waveforms
from spike_psvae.cluster_utils import get_agreement_indices, compute_spiketrain_agreement, get_unit_similarities, compute_shifted_similarity 
from spike_psvae.cluster_utils import get_closest_clusters_kilosort, get_closest_clusters_hdbscan, read_waveforms
from spike_psvae.denoise import denoise_wf_nn_tmp_single_channel
import colorcet as cc

ccolors = cc.glasbey[:31]

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
        ax.scatter(xk, yk, s=s, color=color, alpha=alpha, marker=".")
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

    return fig, axes

def plot_waveforms_geom_unit(geom, first_chans_cluster, mcs_abs_cluster, spike_times, max_ptps_cluster=None, raw_bin=None, residual_bin=None, waveforms_cluster=None, denoiser=None, 
                             device=None, num_spikes_plot=100, t_range=(30,90), num_channels=40, num_rows=3, alpha=.1, h_shift=0, subset_indices=None,
                             scale=None, do_mean=False, annotate=False, ax=None, color='blue'):    
    ax = ax or plt.gca()
    some_in_cluster = np.random.default_rng(0).choice(list(range(len(spike_times))), replace=False, size=min(len(spike_times), num_spikes_plot))
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
    channels_plot = np.flatnonzero((z_ids >= mcid - num_rows) & (z_ids <= mcid + num_rows))
    
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
        #raw data and spike times passed in
        waveforms_read = read_waveforms(spike_times, raw_bin, geom, n_times=121)[0]
        waveforms = []
        for i, waveform in enumerate(waveforms_read):
            waveforms.append(waveform[:,int(first_chans_cluster[i]):int(first_chans_cluster[i])+num_channels].copy())
        waveforms = np.asarray(waveforms)
    elif waveforms_cluster is None:
        #no raw data and no waveforms passed - bad!
        raise ValueError("need to input raw_bin or waveforms")
    else:
        #waveforms passed in
        waveforms = waveforms_cluster[some_in_cluster]
        if residual_bin is not None:
            #add residuals
            residuals_read = read_waveforms(spike_times, residual_bin, geom, n_times=121)[0]
            residuals = []
            for i, residual in enumerate(residuals_read):
                residuals.append(residual[:,int(first_chans_cluster[i]):int(first_chans_cluster[i])+num_channels])
            residuals = np.asarray(residuals)
            waveforms = waveforms + residuals
    if denoiser is not None and device is not None:
        #denoise waveforms
        waveforms = denoise_wf_nn_tmp_single_channel(waveforms, denoiser, device)
    if do_mean:
        #plot the mean rather than the invididual spikes
        waveforms = np.expand_dims(np.mean(waveforms, axis=0),0)
    draw_lines = []
    for i in range(min(len(waveforms), num_spikes_plot)):
        for k, channel in enumerate(range(int(first_chans_cluster[i]),int(first_chans_cluster[i])+waveforms.shape[2])):
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

def plot_waveforms_geom(main_cluster_id, neighbor_clusters, labels, spike_index, firstchans, maxptps, geom, raw_bin=None, residual_bin=None, 
                        waveforms=None, denoiser=None, device=None, num_spikes_plot=100, t_range=(30,90), num_channels=40, num_rows=3, alpha=.1,
                        annotate=False, colors=None, do_mean=False, scale=None, ax=None):    
    ax = ax or plt.gca()
    
    # what channels will we plot?
    vals, counts = np.unique(
        spike_index[np.flatnonzero(labels == main_cluster_id), 1],
        return_counts=True,
    )
    z_uniq, z_ids = np.unique(geom[:, 1], return_inverse=True)
    mcid = z_ids[vals[counts.argmax()]]
    channels_plot = np.flatnonzero((z_ids >= mcid - num_rows) & (z_ids <= mcid + num_rows))
    
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
    
    #plot each cluster
    for j, cluster_id in reversed(list(enumerate((main_cluster_id, *neighbor_clusters)))):
        if colors is None:
            color = get_ccolor(cluster_id)
        else:
            color = colors[j]
        in_cluster = np.flatnonzero(labels == cluster_id)
        some_in_cluster = np.random.default_rng(0).choice(in_cluster, replace=False, size=min(len(in_cluster), num_spikes_plot))
        some_in_cluster.sort()
        first_chans_cluster = firstchans[some_in_cluster]
        spike_times = spike_index[:,0][some_in_cluster]
        if raw_bin is not None:
            #raw data and spike times passed in
            waveforms_read = read_waveforms(spike_times, raw_bin, geom, n_times=121)[0]
            waveforms_in_cluster = []
            for i, waveform in enumerate(waveforms_read):
                waveforms_in_cluster.append(waveform[:,int(first_chans_cluster[i]):int(first_chans_cluster[i])+num_channels].copy())
            waveforms_in_cluster = np.asarray(waveforms_in_cluster)
        elif waveforms is None:
            #no raw data and no waveforms passed - bad!
            raise ValueError("need to input raw_bin or waveforms")
        else:
            #waveforms passed in
            waveforms_in_cluster = waveforms[some_in_cluster]
            if residual_bin is not None:
                #add residuals
                residuals_read = read_waveforms(spike_times, residual_bin, geom, n_times=121)[0]
                residuals = []
                for i, residual in enumerate(residuals_read):
                    residuals.append(residual[:,int(first_chans_cluster[i]):int(first_chans_cluster[i])+num_channels])
                residuals = np.asarray(residuals)
                waveforms_in_cluster = waveforms_in_cluster + residuals
        if denoiser is not None and device is not None:
            #denoise waveforms
            waveforms_in_cluster = denoise_wf_nn_tmp_single_channel(waveforms_in_cluster, denoiser, device)
        if do_mean:
            #plot the mean rather than the invididual spikes
            waveforms_in_cluster = np.expand_dims(np.mean(waveforms_in_cluster, axis=0),0) 
        vertical_lines = set()
        draw_lines = []
        for i in range(min(len(waveforms_in_cluster), num_spikes_plot)):
            for k, channel in enumerate(range(int(first_chans_cluster[i]),int(first_chans_cluster[i])+waveforms_in_cluster.shape[2])):
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

def plot_venn_agreement(cluster_id_1, cluster_id_2, match_ind, not_match_ind_st1, not_match_ind_st2, ax=None):
    if ax is None:
        plt.figure(figsize=(12,12))
        ax = plt.gca()
    lab_st1 = cluster_id_1
    lab_st2 = cluster_id_2
    subsets = [len(not_match_ind_st1), len(not_match_ind_st2), len(match_ind)]
    v = venn2(subsets = subsets, set_labels = ['unit{}'.format(lab_st1),  'unit{}'.format(lab_st2)], ax=ax)
    v.get_patch_by_id('10').set_color('red')
    v.get_patch_by_id('01').set_color('blue')
    v.get_patch_by_id('11').set_color('goldenrod')
    sets = ['10','11','01']
    return ax

def plot_self_agreement(labels, spike_times, fig=None):
    # matplotlib.rcParams.update({'font.size': 22})
    indices_list = []
    labels_list = []
    for cluster_id in np.unique(labels):
        label_ids = np.where(labels==cluster_id)
        indices = spike_times[label_ids]
        num_labels = label_ids[0].shape[0]
        indices_list.append(indices)
        labels_list.append((np.zeros(num_labels) + cluster_id).astype('int'))
    sorting = spikeinterface.numpyextractors.NumpySorting.from_times_labels(times_list=np.concatenate(indices_list), 
                                                                            labels_list=np.concatenate(labels_list), 
                                                                            sampling_frequency=30000)
    sorting_comparison = compare_two_sorters(sorting, sorting)
    if fig is None:
        fig = plt.figure(figsize=(36,36))
    plot_agreement_matrix(sorting_comparison, figure=fig, ordered=False)
    return fig

def plot_isi_distribution(spike_train, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(6,3))
        ax = fig.gca()
    ax.set_xlabel('ms')
    spike_train_diff = np.diff(spike_train)/30000 
    spike_train_diff = spike_train_diff[np.where(spike_train_diff < 0.01)]
    spike_train_diff = spike_train_diff*1000 #convert to ms
    ax.hist(spike_train_diff, bins=np.arange(11))
    ax.set_xticks(range(11))
    ax.set_title('isis')
    ax.set_xlim([-1, 10])
    return ax
    
def plot_single_unit_summary(cluster_id, labels, spike_index, cluster_centers, geom, xs, zs, maxptps, 
                             firstchans, wfs_full_denoise, wfs_subtracted, raw_bin, residual_bin, num_spikes_plot=100, 
                             num_rows_plot=3, t_range=(30,90), plot_all_points=False, num_channels=40):
    matplotlib.rcParams.update({'font.size': 30})
    label_indices = np.where(labels ==cluster_id)
    
    closest_clusters = get_closest_clusters_hdbscan(cluster_id, cluster_centers, num_close_clusters=2)
    # scales = (1,10,1,15,30) #predefined scales for each feature
    features = np.concatenate((np.expand_dims(xs,1), np.expand_dims(zs,1), np.expand_dims(maxptps,1)), axis=1)
    
    close_labels_indices = np.where((labels == cluster_id) | (labels == closest_clusters[0]) | (labels == closest_clusters[1]))
    all_cluster_features_close = features[close_labels_indices]
    all_labels_close = labels[close_labels_indices]
    # print(all_labels_close)

    #buffers for range of scatter plots
    z_buffer = 5
    x_buffer = 5
    ptp_cutoff = .5

    z_cutoff = (np.min(all_cluster_features_close[:,1] - z_buffer), np.max(all_cluster_features_close[:,1] + z_buffer))
    x_cutoff = (np.min(all_cluster_features_close[:,0] - x_buffer), np.max(all_cluster_features_close[:,0] + x_buffer))
    ptps_cutoff = (np.min(all_cluster_features_close[:,2] - ptp_cutoff), np.max(all_cluster_features_close[:,2] + ptp_cutoff))
    
    if plot_all_points:
        plot_features = features
        plot_labels = labels
        cm_plot = np.clip(maxptps, 3, 15)
    else:
        plot_features = all_cluster_features_close
        plot_labels = all_labels_close
        cm_plot = np.clip(maxptps, 3, 15)[close_labels_indices]

    fig = plt.figure(figsize=(24+18*3, 36))
    grid = (6, 6)
    ax_raw = plt.subplot2grid(grid, (0, 0), rowspan=6)
    ax_cleaned  = plt.subplot2grid(grid, (0, 1), rowspan=6)
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
    xs, zs, ids = plot_features[:,0], plot_features[:,1], plot_labels
    cluster_scatter(xs, zs, ids, ax=ax, excluded_ids=set([-1]), s=200, alpha=.3)
    ax.scatter(geom[:, 0], geom[:, 1], s=100, c='orange', marker = "s")
    ax.set_title(f"x vs. z");
    ax.set_ylabel("z");
    ax.set_xlabel("x");
    ax.set_ylim(z_cutoff)
    ax.set_xlim(x_cutoff)

    ax = ax_scatter_sptpz
    ys, zs, ids = plot_features[:,2], plot_features[:,1], plot_labels
    cluster_scatter(ys, zs, ids, ax=ax, excluded_ids=set([-1]), s=200, alpha=.3)
    ax.set_title(f"ptp vs. z");
    ax.set_xlabel("ptp");
    ax.set_yticks([])
    ax.set_ylim(z_cutoff)
    ax.set_xlim(ptps_cutoff)

    ax = ax_scatter_xzptp
    ax.scatter(xs, zs, s=200, c=cm_plot, marker=".", cmap=plt.cm.viridis, alpha=.3)
    ax.scatter(geom[:, 0], geom[:, 1], s=100, c='orange', marker = "s")
    ax.set_title("ptps")
    ax.set_yticks([])
    ax.set_ylim(z_cutoff)
    ax.set_xlim(x_cutoff)
    
    ax = ax_ptp
    ptps_cluster = features[:,2][label_indices]
    spike_train_s = spike_index[:,0][label_indices] / 30000
    ax.plot(spike_train_s, ptps_cluster)
    ax.set_ylabel("ptp");
    ax.set_xlabel("seconds");

    ax = ax_ptp_z
    zs_cluster = features[:,1][label_indices]
    ax.scatter(zs_cluster, ptps_cluster);
    ax.set_xlabel("zs");
    ax.set_ylabel("ptps");

    ax = ax_isi
    spike_train = spike_index[:,0][label_indices]
    ax.set_xlabel('ms')
    spike_train_diff = np.diff(spike_train)/30000 
    spike_train_diff = spike_train_diff[np.where(spike_train_diff < 0.01)]
    spike_train_diff = spike_train_diff*1000
    ax.hist(spike_train_diff, bins=np.arange(11))
    ax.set_xticks(range(11))
    ax.set_title('isis')
    ax.set_xlim([-1, 10])

    axes = [ax_xcorr1, ax_xcorr2]
    for i, cluster_isi_id in enumerate(closest_clusters):
        spike_train_2 = spike_index[:,0][np.flatnonzero(labels==cluster_isi_id)]
        sorting = spikeinterface.numpyextractors.NumpySorting.from_times_labels(times_list=np.concatenate((spike_train,spike_train_2)), 
                                                                                labels_list=np.concatenate((np.zeros(spike_train.shape[0]).astype('int'),
                                                                                                            np.zeros(spike_train_2.shape[0]).astype('int')+1)), 
                                                                                sampling_frequency=30000)
        bin_ms = 1.0
        correlograms, bins = compute_correlograms(sorting, symmetrize=True, window_ms=10.0, bin_ms=bin_ms)
        axes[i].bar(bins[1:], correlograms[0][1], width=bin_ms, align='center')
        axes[i].set_xticks(bins[1:])
        axes[i].set_xlabel('lag (ms)')
        axes[i].set_title(f'{cluster_id}_{cluster_isi_id}_xcorrelogram.png', pad=20)
    
    z_uniq, z_ids = np.unique(geom[:, 1], return_inverse=True)
    all_max_ptp = maxptps[
        np.isin(labels, (*closest_clusters, cluster_id))
    ].max()
    scale = (z_uniq[1] - z_uniq[0]) / max(7, all_max_ptp)

    ax = ax_denoised
    #figure params
    spikes_plot = num_spikes_plot
    waveform_shape=(30, 90)
    num_rows = num_rows_plot
    plot_waveforms_geom(cluster_id, closest_clusters, labels, spike_index, firstchans, maxptps, geom, num_spikes_plot=num_spikes_plot, t_range=t_range, num_channels=num_channels,
                        num_rows=num_rows, alpha=.1, ax=ax, scale=scale, waveforms=wfs_full_denoise)
    ax.set_title("denoised waveforms");
    ax.set_ylabel("z")
    ax.set_xlabel("x")
    ax.set_yticks([])

    ax = ax_raw
    plot_waveforms_geom(cluster_id, closest_clusters, labels, spike_index, firstchans, maxptps, geom, num_spikes_plot=num_spikes_plot, t_range=t_range, num_channels=num_channels,
                        num_rows=num_rows, alpha=.1, ax=ax, scale=scale, raw_bin=raw_bin)
    ax.set_title("raw waveforms")
    ax.set_xlim(ax_denoised.get_xlim())
    ax.set_ylim(ax_denoised.get_ylim())
    ax.set_xlabel("x")
    ax.set_ylabel("z")

    ax = ax_cleaned
    plot_waveforms_geom(cluster_id, closest_clusters, labels, spike_index, firstchans, maxptps, geom, num_spikes_plot=num_spikes_plot, t_range=t_range, num_channels=num_channels,
                        num_rows=num_rows, alpha=.1, ax=ax, scale=scale, waveforms=wfs_subtracted, residual_bin=residual_bin)
    ax.set_title("collision-cleaned waveforms")
    ax.set_yticks([])
    ax.set_xlim(ax_denoised.get_xlim())
    ax.set_ylim(ax_denoised.get_ylim())
    ax.set_xlabel("x")
    matplotlib.rcParams.update({'font.size': 10})
    
    return fig

def plot_agreement_venn(cluster_id_1, cluster_id_2, st_1, st_2, firstchans_cluster_sorting1, mcs_abs_cluster_sorting1, firstchans_cluster_sorting2, mcs_abs_cluster_sorting2, geom, raw_bin, scale=7, sorting1_name="1", sorting2_name="2", num_channels=40, num_spikes_plot=100, t_range=(30,90), num_rows=3, alpha=.1, delta_frames = 12):
    lab_st1 = cluster_id_1
    lab_st2 = cluster_id_2
    ind_st1, ind_st2, not_match_ind_st1, not_match_ind_st2 = compute_spiketrain_agreement(st_1, st_2, delta_frames)
    agreement = len(ind_st1) / (len(st_1) + len(st_2) - len(ind_st1))
    fig = plt.figure(figsize=(12,6))
    grid = (1, 3)
    ax_venn = plt.subplot2grid(grid, (0, 0))
    ax_sorting1  = plt.subplot2grid(grid, (0, 1))
    ax_sorting2 = plt.subplot2grid(grid, (0, 2))

    subsets = [len(not_match_ind_st1), len(not_match_ind_st2), len(ind_st1)]
    v = venn2(subsets = subsets, set_labels = ['unit{}'.format(lab_st1),  'unit{}'.format(lab_st2)], ax=ax_venn)
    v.get_patch_by_id('10').set_color('red')
    v.get_patch_by_id('01').set_color('blue')
    v.get_patch_by_id('11').set_color('goldenrod')
    ax_venn.set_title(f'{sorting1_name}{lab_st1} + {sorting2_name}{lab_st2}, {np.round(agreement, 2)*100}% agreement')
    sets = ['10','11','01']

    # fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,12))
    h_shifts = [-10,10]
    colors = ['goldenrod', 'red']
    indices = [ind_st1, not_match_ind_st1]
    for indices_match, color, h_shift in zip(indices, colors, h_shifts):
        if len(indices_match) > 0:
            firstchans_cluster_sorting = firstchans_cluster_sorting1[indices_match]
            mcs_abs_cluster_sorting = mcs_abs_cluster_sorting1[indices_match]
            spike_times = st_1[indices_match]
            # geom, first_chans_cluster, mcs_abs_cluster, max_ptps_cluster, spike_times, 
            plot_waveforms_geom_unit(geom, firstchans_cluster_sorting, mcs_abs_cluster_sorting, spike_times, raw_bin=raw_bin, num_spikes_plot=num_spikes_plot,
                                     t_range=t_range, num_channels=num_channels, num_rows=num_rows, do_mean=False, scale=scale, h_shift=h_shift, alpha=alpha, ax=ax_sorting1, color=color)  
    colors = ['goldenrod', 'blue']
    indices = [ind_st2, not_match_ind_st2]
    for indices_match, color, h_shift in zip(indices, colors, h_shifts):
        if len(indices_match) > 0:
            mcs_abs_cluster_sorting = mcs_abs_cluster_sorting2[indices_match]
            firstchans_cluster_sorting = firstchans_cluster_sorting2[indices_match]
            spike_times = st_2[indices_match]
            plot_waveforms_geom_unit(geom, firstchans_cluster_sorting, mcs_abs_cluster_sorting, spike_times, raw_bin=raw_bin, num_spikes_plot=num_spikes_plot,
                                     t_range=t_range, num_channels=num_channels, num_rows=num_rows, do_mean=False, scale=scale, h_shift=h_shift, alpha=alpha, ax=ax_sorting2, color=color) 
    return fig


def plot_unit_similarity_heatmaps(cluster_id, st_1, closest_clusters, sorting, geom, raw_data_bin, num_channels_similarity=20, num_close_clusters_plot=10, num_close_clusters=30,
                                  shifts_align=[0], order_by ='similarity', normalize_agreement_by="both", ax_similarity=None, ax_agreement=None):
    if ax_similarity is None:
        plt.figure(figsize=(9,3))
        ax_similarity = plt.gca()
        
    if ax_agreement is None:
        plt.figure(figsize=(9,3))
        ax_agreement = plt.gca()
        
    original_template, closest_clusters, similarities, agreements, templates, shifts = get_unit_similarities(cluster_id, st_1, closest_clusters, sorting, geom, raw_data_bin, 
                                                                                                             num_channels_similarity, num_close_clusters, shifts_align, order_by, 
                                                                                                             normalize_agreement_by)
    
    agreements = agreements[:num_close_clusters_plot]
    similarities = similarities[:num_close_clusters_plot]
    closest_clusters = closest_clusters[:num_close_clusters_plot]
    templates = templates[:num_close_clusters_plot]
    shifts = shifts[:num_close_clusters_plot]

    y_axis_labels = [cluster_id]
    x_axis_labels = closest_clusters
    g = sns.heatmap(np.expand_dims(similarities,0), vmin=0, vmax=max(similarities), cmap='RdYlGn_r', annot=np.expand_dims(similarities,0),xticklabels=x_axis_labels, yticklabels=y_axis_labels, ax=ax_similarity,cbar=False)
    ax_similarity.set_title("Max Abs Norm Similarity");
    g = sns.heatmap(np.expand_dims(agreements,0), vmin=0, vmax=1, cmap='RdYlGn', annot=np.expand_dims(agreements,0),xticklabels=x_axis_labels, yticklabels=y_axis_labels, ax=ax_agreement,cbar=False)
    ax_agreement.set_title(f"Agreement (normalized by {normalize_agreement_by})");
    
    return ax_similarity, ax_agreement, original_template, closest_clusters, similarities, agreements, templates, shifts



def plot_unit_similarities(cluster_id, closest_clusters, sorting1, sorting2, geom, raw_data_bin, recoring_duration, num_channels=40, num_spikes_plot=100, num_channels_similarity=20, 
                           num_close_clusters_plot=10, num_close_clusters=30, shifts_align = np.arange(-3,4), order_by ='similarity', normalize_agreement_by="both",
                           denoised_waveforms=None, cluster_labels=None, non_triaged_idxs=None, triaged_mcs_abs=None, triaged_firstchans=None):
    do_denoised_waveform = denoised_waveforms is not None and cluster_labels is not None and non_triaged_idxs is not None and triaged_mcs_abs is not None and triaged_firstchans is not None
    fig = plt.figure(figsize=(24, 12))
    if do_denoised_waveform:
        gs = gridspec.GridSpec(4,4)
    else:
        gs = gridspec.GridSpec(4,3)
    gs.update(hspace=0.75)
    ax_sim = plt.subplot(gs[0,:2])
    ax_agree  = plt.subplot(gs[1,:2])
    ax_isi = plt.subplot(gs[2,:2])
    ax_raw_wf = plt.subplot(gs[:4,2])
    ax_raw_wf_flat = plt.subplot(gs[3, :2])
    if do_denoised_waveform:
        ax_denoised_wf = plt.subplot(gs[:4,3])

    st_1 = sorting1.get_unit_spike_train(cluster_id)
    firing_rate = len(st_1) / recoring_duration #in seconds

    #compute similarity to closest kilosort clusters
    _, _, original_template, closest_clusters, similarities, agreements, templates, shifts = plot_unit_similarity_heatmaps(cluster_id, st_1, closest_clusters, sorting2, geom, raw_data_bin, 
                                                                                                                           num_channels_similarity=num_channels_similarity, 
                                                                                                                           num_close_clusters_plot=num_close_clusters_plot, 
                                                                                                                           num_close_clusters=num_close_clusters,
                                                                                                                           ax_similarity=ax_sim, ax_agreement=ax_agree, shifts_align=shifts_align,
                                                                                                                           order_by=order_by, normalize_agreement_by=normalize_agreement_by)

    max_ptp_channel = np.argmax(original_template.ptp(0))
    max_ptp = np.max(original_template.ptp(0))
    z_uniq, z_ids = np.unique(geom[:, 1], return_inverse=True)
    scale = (z_uniq[1] - z_uniq[0]) / max(7, max_ptp)

    plot_isi_distribution(st_1, ax=ax_isi);

    most_similar_cluster = closest_clusters[0]
    most_similar_shift = shifts[0]
    h_shifts = [-5,5]
    t_shifts = [0, most_similar_shift]
    colors = [('blue','darkblue'), ('red','darkred')]
    cluster_ids_plot = [cluster_id, most_similar_cluster]
    sortings_plot = [sorting1, sorting2]
    for cluster_id_plot, color, sorting_plot, h_shift, t_shift in zip(cluster_ids_plot, colors, sortings_plot, h_shifts, t_shifts):
        spike_times = sorting_plot.get_unit_spike_train(cluster_id_plot) + t_shift
        mcs_abs_cluster = np.zeros(len(spike_times)).astype('int') + max_ptp_channel
        first_chans_cluster = (mcs_abs_cluster - 20).clip(min=0)
        
        # waveform_scale = 1/25
        
        plot_waveforms_geom_unit(geom, first_chans_cluster, mcs_abs_cluster, spike_times, max_ptps_cluster=None, raw_bin=raw_data_bin, num_spikes_plot=num_spikes_plot,
                                 num_channels=num_channels, alpha=.1, h_shift=h_shift, scale=scale, ax=ax_raw_wf, color=color[0])    
        plot_waveforms_geom_unit(geom, first_chans_cluster, mcs_abs_cluster, spike_times, max_ptps_cluster=None, raw_bin=raw_data_bin, num_spikes_plot=num_spikes_plot,
                                 num_channels=num_channels, alpha=.1, h_shift=h_shift, scale=scale, do_mean=True, ax=ax_raw_wf, color=color[1])   
        
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
            
    ax_raw_wf.set_title(f"cluster {cluster_id}/cluster {most_similar_cluster} raw, shift {most_similar_shift}")
    if do_denoised_waveform:
        ax_denoised_wf.set_title(f"cluster {cluster_id}/cluster {most_similar_cluster} denoised, shift {most_similar_shift}")
    channel_range = (max(max_ptp_channel-num_channels_similarity//2,0),max_ptp_channel+num_channels_similarity//2)
    template1 = original_template[:,channel_range[0]:channel_range[1]]
    most_similar_template = templates[0]
    if most_similar_shift == 0:
        most_similar_template_flattened = most_similar_template.T.flatten()
    elif most_similar_shift < 0:
        most_similar_template_flattened = np.pad(most_similar_template.T.flatten(),((-most_similar_shift,0)), mode='constant')[:most_similar_shift]
    else:    
        most_similar_template_flattened = np.pad(most_similar_template.T.flatten(),((0,most_similar_shift)), mode='constant')[most_similar_shift:]
    ax_raw_wf_flat.plot(template1.T.flatten(),color='blue')
    ax_raw_wf_flat.plot(most_similar_template_flattened,color='red')
    ax_raw_wf_flat.set_title(f"cluster {cluster_id}/cluster {most_similar_cluster} templates flat, shift {most_similar_shift}")
    fig.suptitle(f"cluster {cluster_id}, firing rate: {'%.1f' % round(firing_rate,2)} Hz, max ptp: {'%.1f' % round(max_ptp,2)}");
    return fig

# def plot_unit_similarities_summary(cluster_id, closest_clusters, sorting1, sorting2, geom, raw_data_bin, recoring_duration, num_channels=40, num_spikes_plot=100, num_channels_similarity=20, 
#                                    num_close_clusters_plot=10, num_close_clusters=30, shifts_align = np.arange(-3,4), order_by ='similarity', normalize_agreement_by="both", denoised_waveforms=None, 
#                                    cluster_labels=None, non_triaged_idxs=None, triaged_mcs_abs=None, triaged_firstchans=None):

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

#         st_1 = sorting_kilo.get_unit_spike_train(cluster_id)
#         firing_rate = len(st_1) / recording_duration #in seconds
#         waveforms1 = read_waveforms(st_1, raw_data_bin, geom, n_times=121)[0]
#         template1 = np.mean(waveforms1, axis=0)
#         max_ptp_channel = np.argmax(template1.ptp(0))
#         max_ptp = np.max(template1.ptp(0))
#         channel_range = (max(max_ptp_channel-num_channels_cosine//2,0),max_ptp_channel+num_channels_cosine//2)
#         template1 = template1[:,channel_range[0]:channel_range[1]]

#         #compute K closest clsuters
#         curr_cluster_depth = kilo_cluster_depth_means[cluster_id]
#         dist_to_other_cluster_dict = {cluster_id:abs(mean_depth-curr_cluster_depth) for (cluster_id,mean_depth) in kilo_cluster_depth_means.items()}
#         closest_clusters = [y[0] for y in sorted(dist_to_other_cluster_dict.items(), key = lambda x: x[1])[1:1+num_close_clusters]]

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

#         agreements = np.asarray(agreements).round(2)
#         similarities = np.asarray(similarities).round(2)
#         closest_clusters = np.asarray(closest_clusters)
#         # most_similar_idxs = np.flip(np.argsort(similarities))
#         most_similar_idxs = np.argsort(similarities)
#         agreements = agreements[most_similar_idxs]
#         similarities = similarities[most_similar_idxs]
#         closest_clusters = closest_clusters[most_similar_idxs]

#         y_axis_labels = [f"Unit {cluster_id}"]
#         x_axis_labels = closest_clusters
#         g = sns.heatmap(np.expand_dims(similarities,0), vmin=0, vmax=max(similarities), cmap='RdYlGn_r', annot=np.expand_dims(similarities,0),xticklabels=x_axis_labels, yticklabels=y_axis_labels, ax=ax_cos,cbar=False)
#         ax_cos.set_title("Cosine Similarity");
#         g = sns.heatmap(np.expand_dims(agreements,0), vmin=0, vmax=1, cmap='RdYlGn', annot=np.expand_dims(agreements,0),xticklabels=x_axis_labels, yticklabels=y_axis_labels, ax=ax_agree,cbar=False)
#         ax_agree.set_title("Agreement");

#         plot_isi_distribution(st_1, ax=ax_isi);
#         matplotlib.rcParams.update({'font.size': 22})

#         ax_cid.text(0.15, 0.45, f"Unit id: {cluster_id}")
#         ax_cid.set_xticks([])
#         ax_cid.set_yticks([])

#         ax_fr.text(0.15, 0.45, f"FR: {'%.1f' % round(firing_rate,2)} Hz")
#         ax_fr.set_xticks([])
#         ax_fr.set_yticks([])

#         ax_maxptp.text(0.1, 0.45, f"max ptp: {'%.1f' % round(max_ptp,2)}")
#         ax_maxptp.set_xticks([])
#         ax_maxptp.set_yticks([])
#     plt.close(fig)
#     fig.savefig(f"kilosort_cluster_summaries_norm.png")