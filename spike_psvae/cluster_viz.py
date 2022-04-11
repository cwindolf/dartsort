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
from spike_psvae.cluster_utils import make_sorting_from_labels_frames, compute_cluster_centers, relabel_by_depth, run_weighted_triage, remove_duplicate_units, read_waveforms
from spike_psvae.cluster_utils import get_agreement_indices, compute_spiketrain_agreement, get_unit_similarities, compute_shifted_similarity 
from spike_psvae.cluster_utils import get_closest_clusters_kilosort, get_closest_clusters_hdbscan, read_waveforms

def cluster_scatter(xs, ys, ids, ax=None, n_std=2.0, excluded_ids=set(), s=1, alpha=.5, color_dict=None):
    if color_dict is None:
        raise ValueError("Must pass valid color dict")
    ax = ax or plt.gca()
    # scatter and collect gaussian info
    means = {}
    covs = {}
    for k in np.unique(ids):
        where = np.flatnonzero(ids == k)
        xk = xs[where]
        yk = ys[where]
        color = color_dict[k]
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
        with np.errstate(invalid='ignore'):
            vx, vy = cov[0, 0], cov[1, 1]
            rho = cov[0, 1] / np.sqrt(vx * vy)
        if not np.isfinite([vx, vy, rho]).all():
            continue
        color = color_dict[k]
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
        
def plot_waveforms_geom(main_cluster_id, labels, clusters_to_plot, geom_array, non_triage_indices, wfs, 
                        triaged_firstchans, triaged_mcs_abs, triaged_spike_index=None, bin_file=None, residual_bin_file=None, x_geom_scale = 1/15, 
                        y_geom_scale = 1/10, waveform_scale = .15, spikes_plot = 200, waveform_shape=(30,90), num_rows=3, 
                        alpha=.1, h_shift=.5, raw=False, add_residuals=False, do_mean=False, ax=None, color_dict=None):    
    if color_dict is None:
        raise ValueError("Must pass valid color dict")
    if ax is None:
        plt.figure(figsize=(12,36), dpi=300)
        ax = plt.gca()    
    geom_scale = [x_geom_scale, y_geom_scale]
    geom_plot = geom_array*geom_scale
    first_chans_cluster = triaged_firstchans[labels==main_cluster_id]
    mcs_abs_cluster = triaged_mcs_abs[labels==main_cluster_id]
    mcs_ab_channels_unique = np.unique(mcs_abs_cluster, return_counts=True)
    mc_ab_mode_cluster = int(mcs_ab_channels_unique[0][np.argmax(mcs_ab_channels_unique[1])])
    z_mc_ab = geom_array[mc_ab_mode_cluster][1]
    channel = mc_ab_mode_cluster
    rel_zs = geom_array[:,1] - z_mc_ab
    curr_row_channels = np.where(rel_zs == 0)[0]
    value_change_array = rel_zs[:-1] != rel_zs[1:]
    value_indicator = False
    row_change = 0
    channels_plot = []
    i = np.min(curr_row_channels) - 1
    exit = False
    while ((row_change < num_rows) or (not value_change_array[i])) and (not exit):
        if i < 0:
            exit = True
            break
        channels_plot.append(i)
        if value_change_array[i]:
            row_change += 1
        i -= 1
    for curr_row_channel in curr_row_channels:
        channels_plot.append(curr_row_channel)
    i = np.max(curr_row_channels) + 1 
    row_change = 0
    exit = False
    while ((row_change < num_rows) or (value_change_array[i])) and (not exit):
        if i == value_change_array.shape[0]:
            exit = True
            channels_plot.append(i)
            break
        channels_plot.append(i)
        if value_change_array[i]:
            row_change += 1
        i += 1
    channels_plot = np.sort(channels_plot)

    for i, channel in enumerate(channels_plot):
        ax.scatter(geom_array[channel][0]*geom_scale[0], geom_array[channel][1]*geom_scale[1], s=100, c='orange', marker = "s")
        ax.annotate(channel, (geom_array[channel][0]*geom_scale[0], geom_array[channel][1]*geom_scale[1]))
    for j, cluster_id in enumerate(clusters_to_plot):
        color = color_dict[cluster_id]
        first_chans_cluster = triaged_firstchans[labels==cluster_id]
        if raw:
            num_channels = wfs.shape[2]
            if triaged_spike_index is not None and bin_file is not None:
                spike_times = triaged_spike_index[labels==cluster_id][:,0]
                waveforms_read = read_waveforms(spike_times, bin_file, geom_array, n_times=121)[0]
                waveforms = []
                for i, waveform in enumerate(waveforms_read):
                    waveforms.append(waveform[:,int(first_chans_cluster[i]):int(first_chans_cluster[i])+num_channels])
                waveforms = np.asarray(waveforms)    
            else:
                raise ValueError("Need to specify spike_index and bin_file")
        else:
            waveforms = wfs[non_triage_indices[labels==cluster_id]]
        if add_residuals:
            if triaged_spike_index is not None and residual_bin_file is not None:
                spike_times = triaged_spike_index[labels==cluster_id][:,0]
                residuals_read = read_waveforms(spike_times, residual_bin_file, geom_array, n_times=121)[0]
                num_channels = waveforms.shape[2]
                residuals = []
                for i, residual in enumerate(residuals_read):
                    residuals.append(residual[:,int(first_chans_cluster[i]):int(first_chans_cluster[i])+num_channels])
                residuals = np.asarray(residuals)
                waveforms = waveforms + residuals
            else:
                raise ValueError("Need to specify spike_index and residual_bin_file")
        if do_mean:
            waveforms = np.expand_dims(np.mean(waveforms, axis=0),0)
        vertical_lines = set()
        for i in range(min(spikes_plot, waveforms.shape[0])):
            for k, channel in enumerate(range(int(first_chans_cluster[i]),int(first_chans_cluster[i])+waveforms.shape[2])):
                if channel in channels_plot:
                    channel_position = geom_array[channel]*geom_scale
                    waveform = waveforms[i, waveform_shape[0]:waveform_shape[1],k].T.flatten()*waveform_scale
                    ax.plot(np.linspace(channel_position[0]-.75+h_shift*j, channel_position[0]+.5+h_shift*j, waveform.shape[0]), waveform + channel_position[1], alpha = alpha, c = color)
                    max_vert_line = np.linspace(channel_position[0]-.75+h_shift*j, channel_position[0]+.5+h_shift*j, waveform.shape[0])[12]
                    if max_vert_line not in vertical_lines:
                        vertical_lines.add(max_vert_line)
                        ax.axvline(max_vert_line, linestyle='--')                       
                        
def plot_raw_waveforms_unit_geom(geom_array, num_channels, first_chans_cluster, mcs_abs_cluster, spike_times, bin_file, x_geom_scale = 1/25, 
                                 y_geom_scale = 1/10, waveform_scale = .15, spikes_plot = 200, waveform_shape=(30,90), num_rows=3, 
                                 alpha=.1, h_shift=.5, do_mean=False, ax=None, color='blue'):    
    if ax is None:
        plt.figure(figsize=(6,12))
        ax = plt.gca()
    geom_scale = [x_geom_scale, y_geom_scale]
    geom_plot = geom_array*geom_scale
    mcs_ab_channels_unique = np.unique(mcs_abs_cluster, return_counts=True)
    mc_ab_mode_cluster = int(mcs_ab_channels_unique[0][np.argmax(mcs_ab_channels_unique[1])])
    z_mc_ab = geom_array[mc_ab_mode_cluster][1]
    channel = mc_ab_mode_cluster
    rel_zs = geom_array[:,1] - z_mc_ab
    curr_row_channels = np.where(rel_zs == 0)[0]
    value_change_array = rel_zs[:-1] != rel_zs[1:]
    value_indicator = False
    row_change = 0
    channels_plot = []
    i = np.min(curr_row_channels) - 1
    exit = False
    while ((row_change < num_rows) or (not value_change_array[i])) and (not exit):
        if i < 0:
            exit = True
            break
        channels_plot.append(i)
        if value_change_array[i]:
            row_change += 1
        i -= 1
    for curr_row_channel in curr_row_channels:
        channels_plot.append(curr_row_channel)
    i = np.max(curr_row_channels) + 1 
    row_change = 0
    exit = False
    while ((row_change < num_rows) or (value_change_array[i])) and (not exit):
        if i == value_change_array.shape[0]:
            exit = True
            channels_plot.append(i)
            break
        channels_plot.append(i)
        if value_change_array[i]:
            row_change += 1
        i += 1
    channels_plot = np.sort(channels_plot)
    for i, channel in enumerate(channels_plot):
        ax.scatter(geom_array[channel][0]*geom_scale[0], geom_array[channel][1]*geom_scale[1], s=100, c='orange', marker = "s")
        ax.annotate(channel, (geom_array[channel][0]*geom_scale[0], geom_array[channel][1]*geom_scale[1]))
        waveforms_read = read_waveforms(spike_times, bin_file, geom_array, n_times=121)[0]
        waveforms = []
        for i, waveform in enumerate(waveforms_read):
            waveforms.append(waveform[:,int(first_chans_cluster[i]):int(first_chans_cluster[i])+num_channels])
        waveforms = np.asarray(waveforms)    
    if do_mean:
        waveforms = np.expand_dims(np.mean(waveforms, axis=0),0)
    if waveform_scale is None:
        max_ptp = np.max(np.mean(waveforms, axis=0).flatten().ptp(0))
        waveform_scale = 2/max_ptp
    for i in range(min(spikes_plot, waveforms.shape[0])):
        for k, channel in enumerate(range(int(first_chans_cluster[i]),int(first_chans_cluster[i])+waveforms.shape[2])):
            if channel in channels_plot:
                channel_position = geom_array[channel]*geom_scale
                waveform = waveforms[i, waveform_shape[0]:waveform_shape[1],k].T.flatten()*waveform_scale
                # print(np.abs(waveform).max(), channel)
                ax.plot(np.linspace(channel_position[0]-.75+h_shift, channel_position[0]+.5+h_shift, waveform.shape[0]), waveform + channel_position[1], alpha = alpha, c = color)  
                
def plot_waveforms_unit_geom(geom_array, num_channels, first_chans_cluster, mcs_abs_cluster, waveforms, x_geom_scale = 1/25, 
                             y_geom_scale = 1/10, waveform_scale = .15, spikes_plot = 200, waveform_shape=(30,90), num_rows=3, 
                             alpha=.1, h_shift=.5, do_mean=False, ax=None, color='blue'):    
    if ax is None:
        plt.figure(figsize=(6,12))
        ax = plt.gca()
    geom_scale = [x_geom_scale, y_geom_scale]
    geom_plot = geom_array*geom_scale
    mcs_ab_channels_unique = np.unique(mcs_abs_cluster, return_counts=True)
    mc_ab_mode_cluster = int(mcs_ab_channels_unique[0][np.argmax(mcs_ab_channels_unique[1])])
    z_mc_ab = geom_array[mc_ab_mode_cluster][1]
    channel = mc_ab_mode_cluster
    rel_zs = geom_array[:,1] - z_mc_ab
    curr_row_channels = np.where(rel_zs == 0)[0]
    value_change_array = rel_zs[:-1] != rel_zs[1:]
    value_indicator = False
    row_change = 0
    channels_plot = []
    i = np.min(curr_row_channels) - 1
    exit = False
    while ((row_change < num_rows) or (not value_change_array[i])) and (not exit):
        if i < 0:
            exit = True
            break
        channels_plot.append(i)
        if value_change_array[i]:
            row_change += 1
        i -= 1
    for curr_row_channel in curr_row_channels:
        channels_plot.append(curr_row_channel)
    i = np.max(curr_row_channels) + 1 
    row_change = 0
    exit = False
    while ((row_change < num_rows) or (value_change_array[i])) and (not exit):
        if i == value_change_array.shape[0]:
            exit = True
            channels_plot.append(i)
            break
        channels_plot.append(i)
        if value_change_array[i]:
            row_change += 1
        i += 1
    channels_plot = np.sort(channels_plot)
    for i, channel in enumerate(channels_plot):
        ax.scatter(geom_array[channel][0]*geom_scale[0], geom_array[channel][1]*geom_scale[1], s=100, c='orange', marker = "s")
        ax.annotate(channel, (geom_array[channel][0]*geom_scale[0], geom_array[channel][1]*geom_scale[1]))  
    if do_mean:
        waveforms = np.expand_dims(np.mean(waveforms, axis=0),0)
    if waveform_scale is None:
        max_ptp = np.max(np.mean(waveforms, axis=0).flatten().ptp(0))
        waveform_scale = 2/max_ptp
    for i in range(min(spikes_plot, waveforms.shape[0])):
        for k, channel in enumerate(range(int(first_chans_cluster[i]),int(first_chans_cluster[i])+waveforms.shape[2])):
            if channel in channels_plot:
                channel_position = geom_array[channel]*geom_scale
                waveform = waveforms[i, waveform_shape[0]:waveform_shape[1],k].T.flatten()*waveform_scale
                # print(np.abs(waveform).max(), channel)
                ax.plot(np.linspace(channel_position[0]-.75+h_shift, channel_position[0]+.5+h_shift, waveform.shape[0]), waveform + channel_position[1], alpha = alpha, c = color)    
            

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

def plot_array_scatter(labels, geom_array, triaged_x, triaged_z, triaged_maxptps, cluster_color_dict, color_arr, min_cluster_size, min_samples, z_cutoff = (-50,3900),figsize=(18, 36)):
    #recompute cluster centers for new labels    
    # fig, axes = plt.subplots(1, 3, sharey=True, figsize=(18, 12))
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=figsize, dpi=300)

    # matplotlib.rcParams.update({'font.size': 12})
    xs, zs, ids = triaged_x, triaged_z, labels
    axes[0].set_ylim(z_cutoff)
    cluster_scatter(xs, zs, ids, ax=axes[0], excluded_ids=set([-1]), s=20, alpha=.05, color_dict=cluster_color_dict)
    axes[0].scatter(geom_array[:, 0], geom_array[:, 1], s=20, c='orange', marker = "s")
    # for channel_id, channel in enumerate(geom_array):
    #     axes[0].annotate(str(channel_id), (channel[0], channel[1]))
    axes[0].set_title(f"min_cluster_size {min_cluster_size}, min_samples {min_samples}");
    axes[0].set_ylabel("z");
    axes[0].set_xlabel("x");

    ys, zs, ids = triaged_maxptps, triaged_z, labels
    axes[1].set_ylim(z_cutoff)
    cluster_scatter(ys, zs, ids, ax=axes[1], excluded_ids=set([-1]), s=20, alpha=.05, color_dict=cluster_color_dict)
    axes[1].set_title(f"min_cluster_size {min_cluster_size}, min_samples {min_samples}");
    axes[1].set_xlabel("scaled ptp");

    axes[2].scatter(xs, zs, s=20, c=color_arr, alpha=.1)
    axes[2].scatter(geom_array[:, 0], geom_array[:, 1], s=20, c='orange', marker = "s")
    axes[2].set_ylim(z_cutoff)
    axes[2].set_title("ptps");
    return fig

# def plot_array_scatter(labels, geom_array, x, z, maxptps, cluster_color_dict, color_arr):
#     #recompute cluster centers for new labels    
#     # clusterer_to_be_plotted = clusterer
#     fig, axes = plt.subplots(1, 3, sharey=True, figsize=(16, 24), dpi=300)

#     matplotlib.rcParams.update({'font.size': 12})
#     z_cutoff = (-50,3900)
#     # xs, zs, ids = triaged_x, triaged_z, clusterer_to_be_plotted.labels_
#     axes[0].set_ylim(z_cutoff)
#     cluster_scatter(x, z, labels, ax=axes[0], excluded_ids=set([-1]), s=20, alpha=.05, color_dict=cluster_color_dict)
#     axes[0].scatter(geom_array[:, 0], geom_array[:, 1], s=20, c='orange', marker = "s")
#     # axes[0].set_title(f"min_cluster_size {clusterer_to_be_plotted.min_cluster_size}, min_samples {clusterer_to_be_plotted.min_samples}");
#     axes[0].set_ylabel("z");
#     axes[0].set_xlabel("x");

#     axes[1].set_ylim(z_cutoff)
#     cluster_scatter(maxptps, z, labels, ax=axes[1], excluded_ids=set([-1]), s=20, alpha=.05, color_dict=cluster_color_dict)
#     # axes[1].set_title(f"min_cluster_size {clusterer_to_be_plotted.min_cluster_size}, min_samples {clusterer_to_be_plotted.min_samples}");
#     axes[1].set_xlabel("scaled ptp");

#     axes[2].scatter(x, z, s=20, c=color_arr, alpha=.1)
#     axes[2].scatter(geom_array[:, 0], geom_array[:, 1], s=20, c='orange', marker = "s")
#     axes[2].set_ylim(z_cutoff)
#     axes[2].set_title("ptps");
#     return fig


def plot_self_agreement(labels, triaged_spike_index, fig=None):
    # matplotlib.rcParams.update({'font.size': 22})
    indices_list = []
    labels_list = []
    for cluster_id in np.unique(labels):
        label_ids = np.where(labels==cluster_id)
        indices = triaged_spike_index[label_ids][:,0]
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
    
def plot_single_unit_summary(cluster_id, labels, cluster_centers, geom_array, num_spikes_plot, num_rows_plot, triaged_x, triaged_z, triaged_maxptps, 
                             triaged_firstchans, triaged_mcs_abs, triaged_spike_index, non_triage_indices, wfs_localized, wfs_subtracted, cluster_color_dict, color_arr, 
                             raw_bin_file, residual_bin_file):
    matplotlib.rcParams.update({'font.size': 30})
    label_indices = np.where(labels ==cluster_id)
    
    closest_clusters = get_closest_clusters_hdbscan(cluster_id, cluster_centers, num_close_clusters=2)
    # scales = (1,10,1,15,30) #predefined scales for each feature
    features = np.concatenate((np.expand_dims(triaged_x,1), np.expand_dims(triaged_z,1), np.expand_dims(np.log(triaged_maxptps),1)), axis=1)
    all_cluster_features_close = features[np.where((labels == cluster_id) | (labels == closest_clusters[0]) | (labels == closest_clusters[1]))]

    #buffers for range of scatter plots
    z_buffer = 5
    x_buffer = 5
    ptp_cutoff = .5

    z_cutoff = (np.min(all_cluster_features_close[:,1] - z_buffer), np.max(all_cluster_features_close[:,1] + z_buffer))
    x_cutoff = (np.min(all_cluster_features_close[:,0] - x_buffer), np.max(all_cluster_features_close[:,0] + x_buffer))
    ptps_cutoff = (np.min(all_cluster_features_close[:,2] - ptp_cutoff), np.max(all_cluster_features_close[:,2] + ptp_cutoff))

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

    ax = ax_ptp
    ptps_cluster = features[:,2][label_indices]
    spike_train_s = triaged_spike_index[:,0][label_indices] / 30000
    ax.plot(spike_train_s, ptps_cluster)
    ax.set_ylabel("ptp");
    ax.set_xlabel("seconds");

    ax = ax_ptp_z
    zs_cluster = features[:,1][label_indices]
    ax.scatter(zs_cluster, ptps_cluster);
    ax.set_xlabel("zs");
    ax.set_ylabel("ptps");

    ax = ax_scatter_xz
    xs, zs, ids = features[:,0], features[:,1], labels
    cluster_scatter(xs, zs, ids, ax=ax, excluded_ids=set([-1]), s=100, alpha=.3, color_dict=cluster_color_dict)
    ax.scatter(geom_array[:, 0], geom_array[:, 1], s=100, c='orange', marker = "s")
    ax.set_title(f"x vs. z");
    ax.set_ylabel("z");
    ax.set_xlabel("x");
    ax.set_ylim(z_cutoff)
    ax.set_xlim(x_cutoff)

    ax = ax_scatter_sptpz
    ys, zs, ids = features[:,2], features[:,1], labels
    cluster_scatter(ys, zs, ids, ax=ax, excluded_ids=set([-1]), s=100, alpha=.3, color_dict=cluster_color_dict)
    ax.set_title(f"ptp vs. z");
    ax.set_xlabel("ptp");
    ax.set_yticks([])
    ax.set_ylim(z_cutoff)
    ax.set_xlim(ptps_cutoff)

    ax = ax_scatter_xzptp
    ax.scatter(xs, zs, s=100, c=color_arr, alpha=.3)
    ax.scatter(geom_array[:, 0], geom_array[:, 1], s=100, c='orange', marker = "s")
    ax.set_title("ptps")
    ax.set_yticks([])
    ax.set_ylim(z_cutoff)
    ax.set_xlim(x_cutoff)

    #figure params
    x_geom_scale = 1/20
    y_geom_scale = 1/10
    spikes_plot = num_spikes_plot
    waveform_shape=(30, 70)
    num_rows = num_rows_plot

    ax = ax_isi
    spike_train = triaged_spike_index[:,0][label_indices]
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
        spike_train_2 = triaged_spike_index[:,0][labels==cluster_isi_id]
        sorting = spikeinterface.numpyextractors.NumpySorting.from_times_labels(times_list=np.concatenate((spike_train,spike_train_2)), 
                                                                                labels_list=np.concatenate((np.zeros(spike_train.shape[0]).astype('int'), np.zeros(spike_train_2.shape[0]).astype('int')+1)), 
                                                                                sampling_frequency=30000)
        bin_ms = 1.0
        correlograms, bins = compute_correlograms(sorting, symmetrize=True, window_ms=10.0, bin_ms=bin_ms)
        axes[i].bar(bins[1:], correlograms[0][1], width=bin_ms, align='center')
        axes[i].set_xticks(bins[1:])
        axes[i].set_xlabel('lag (ms)')
        axes[i].set_title(f'{cluster_id}_{cluster_isi_id}_xcorrelogram.png', pad=20)

    clusters_plot = np.concatenate((closest_clusters, [cluster_id]))
    # clusters_plot = [cluster_id]

    max_ptps = []
    for cluster_id_ptp in clusters_plot:
        max_ptps.append(np.max(triaged_maxptps[labels==cluster_id_ptp]))
    max_ptp = np.max(max_ptps)
    waveform_scale = 2/max_ptp

    ax = ax_denoised
    plot_waveforms_geom(cluster_id, labels, clusters_plot, geom_array, non_triage_indices, wfs_localized, triaged_firstchans, 
                        triaged_mcs_abs, x_geom_scale=x_geom_scale, y_geom_scale=y_geom_scale, waveform_scale=waveform_scale, spikes_plot=spikes_plot, 
                        waveform_shape=waveform_shape,  h_shift=0, ax=ax, alpha=.1, num_rows=num_rows, do_mean=False, color_dict=cluster_color_dict)
    ax.set_title("denoised waveforms");
    ax.set_ylabel("z")
    ax.set_xlabel("x")
    ax.set_yticks([])

    ax = ax_raw
    plot_waveforms_geom(cluster_id, labels, clusters_plot, geom_array, non_triage_indices, wfs_subtracted, triaged_firstchans, 
                        triaged_mcs_abs, triaged_spike_index=triaged_spike_index, bin_file=raw_bin_file, x_geom_scale=x_geom_scale, y_geom_scale=y_geom_scale, 
                        waveform_scale=waveform_scale, spikes_plot=spikes_plot, waveform_shape=waveform_shape, h_shift=0, ax=ax, alpha=.1, num_rows=num_rows, raw=True, do_mean=False, color_dict=cluster_color_dict)
    ax.set_title("raw waveforms")
    ax.set_xlim(ax_denoised.get_xlim())
    ax.set_ylim(ax_denoised.get_ylim())
    ax.set_xlabel("x")
    ax.set_ylabel("z")

    ax = ax_cleaned
    plot_waveforms_geom(cluster_id, labels, clusters_plot, geom_array, non_triage_indices, wfs_subtracted, triaged_firstchans, 
                        triaged_mcs_abs, triaged_spike_index=triaged_spike_index, residual_bin_file=residual_bin_file, x_geom_scale=x_geom_scale, 
                        y_geom_scale=y_geom_scale, waveform_scale=waveform_scale, spikes_plot=spikes_plot, waveform_shape=waveform_shape, h_shift=0, ax=ax,alpha=.1, num_rows=num_rows, 
                        do_mean=False, add_residuals=True, color_dict=cluster_color_dict)
    ax.set_title("collision-cleaned waveforms")
    ax.set_yticks([])
    ax.set_xlim(ax_denoised.get_xlim())
    ax.set_ylim(ax_denoised.get_ylim())
    ax.set_xlabel("x")
    matplotlib.rcParams.update({'font.size': 10})
    
    return fig

def plot_agreement_venn(cluster_id, cluster_id_match, cmp, sorting1, sorting2, sorting1_name, sorting2_name, geom_array, num_channels, num_spikes_plot, firstchans_cluster_sorting1, mcs_abs_cluster_sorting1, 
                        firstchans_cluster_sorting2, mcs_abs_cluster_sorting2, raw_bin_file, delta_frames = 12, alpha=.1):
    lab_st1 = cluster_id
    lab_st2 = cluster_id_match
    st_1 = sorting1.get_unit_spike_train(lab_st1)
    st_2 = sorting2.get_unit_spike_train(lab_st2)
    ind_st1, ind_st2, not_match_ind_st1, not_match_ind_st2 = compute_spiketrain_agreement(st_1, st_2, delta_frames)
    fig = plt.figure(figsize=(24,12))
    grid = (1, 3)
    ax_venn = plt.subplot2grid(grid, (0, 0))
    ax_sorting1  = plt.subplot2grid(grid, (0, 1))
    ax_sorting2 = plt.subplot2grid(grid, (0, 2))

    subsets = [len(not_match_ind_st1), len(not_match_ind_st2), len(ind_st1)]
    v = venn2(subsets = subsets, set_labels = ['unit{}'.format(lab_st1),  'unit{}'.format(lab_st2)], ax=ax_venn)
    v.get_patch_by_id('10').set_color('red')
    v.get_patch_by_id('01').set_color('blue')
    v.get_patch_by_id('11').set_color('goldenrod')
    ax_venn.set_title(f'{sorting1_name}{lab_st1} + {sorting2_name}{lab_st2}, {cmp.get_agreement_fraction(lab_st1, lab_st2).round(2)*100}% agreement')
    sets = ['10','11','01']

    # fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,12))
    h_shifts = [.4,.6]
    colors = ['goldenrod', 'red']
    indices = [ind_st1, not_match_ind_st1]
    for indices_match, color, h_shift in zip(indices, colors, h_shifts):
        if len(indices_match) > 0:
            firstchans_cluster_indices = firstchans_cluster_sorting1[indices_match]
            mcs_abs_cluster_indices = mcs_abs_cluster_sorting1[indices_match]
            spike_times = st_1[indices_match]
            plot_raw_waveforms_unit_geom(geom_array, num_channels, firstchans_cluster_indices, mcs_abs_cluster_indices, spike_times=spike_times, bin_file=raw_bin_file, x_geom_scale = 1/20, 
                                         y_geom_scale = 1/10, waveform_scale = .15, spikes_plot = num_spikes_plot, waveform_shape=(30,70), num_rows=3, 
                                         alpha=alpha, h_shift=h_shift, do_mean=False, ax=ax_sorting1, color=color)

    colors = ['goldenrod', 'blue']
    indices = [ind_st2, not_match_ind_st2]
    for indices_match, color, h_shift in zip(indices, colors, h_shifts):
        if len(indices_match) > 0:
            firstchans_cluster_indices = firstchans_cluster_sorting2[indices_match]
            mcs_abs_cluster_indices = mcs_abs_cluster_sorting2[indices_match]
            spike_times = st_2[indices_match]
            plot_raw_waveforms_unit_geom(geom_array, num_channels, firstchans_cluster_indices, mcs_abs_cluster_indices, spike_times=spike_times, bin_file=raw_bin_file, x_geom_scale = 1/20, 
                                         y_geom_scale = 1/10, waveform_scale = .15, spikes_plot = num_spikes_plot, waveform_shape=(30,70), num_rows=3, 
                                         alpha=alpha, h_shift=h_shift, do_mean=False, ax=ax_sorting2, color=color)
    return fig


def plot_unit_similarity_heatmaps(cluster_id, st_1, closest_clusters, sorting, geom_array, raw_data_bin, num_channels_similarity=20, num_close_clusters_plot=10, num_close_clusters=30,
                                  shifts_align=[0], order_by ='similarity', normalize_agreement_by="both", ax_similarity=None, ax_agreement=None):
    if ax_similarity is None:
        plt.figure(figsize=(15,5))
        ax_similarity = plt.gca()
        
    if ax_agreement is None:
        plt.figure(figsize=(15,5))
        ax_agreement = plt.gca()
        
    original_template, closest_clusters, similarities, agreements, templates, shifts = get_unit_similarities(cluster_id, st_1, closest_clusters, sorting, geom_array, raw_data_bin, 
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
    ax_agreement.set_title("Agreement");
    
    return ax_similarity, ax_agreement, original_template, closest_clusters, similarities, agreements, templates, shifts



def plot_unit_similarities(cluster_id, closest_clusters, sorting1, sorting2, geom_array, raw_data_bin, recoring_duration, num_channels=40, num_spikes_plot=100, num_channels_similarity=20, 
                           num_close_clusters_plot=10, num_close_clusters=30, shifts_align = np.arange(-3,4), order_by ='similarity', normalize_agreement_by="both", denoised_waveforms=None, 
                           cluster_labels=None, non_triaged_idxs=None, triaged_mcs_abs=None, triaged_firstchans=None):
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
    _, _, original_template, closest_clusters, similarities, agreements, templates, shifts = plot_unit_similarity_heatmaps(cluster_id, st_1, closest_clusters, sorting2, geom_array, raw_data_bin, 
                                                                                                                           num_channels_similarity=num_channels_similarity, 
                                                                                                                           num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters,
                                                                                                                           ax_similarity=ax_sim, ax_agreement=ax_agree, shifts_align=shifts_align,
                                                                                                                           order_by=order_by, normalize_agreement_by=normalize_agreement_by)

    max_ptp_channel = np.argmax(original_template.ptp(0))
    max_ptp = np.max(original_template.ptp(0))

    plot_isi_distribution(st_1, ax=ax_isi);

    most_similar_cluster = closest_clusters[0]
    most_similar_shift = shifts[0]
    h_shifts = [.2,.8]
    t_shifts = [0, most_similar_shift]
    colors = [('blue','darkblue'), ('red','darkred')]
    cluster_ids_plot = [cluster_id, most_similar_cluster]
    sortings_plot = [sorting1, sorting2]
    for cluster_id_plot, color, sorting_plot, h_shift, t_shift in zip(cluster_ids_plot, colors, sortings_plot, h_shifts, t_shifts):
        spike_times = sorting_plot.get_unit_spike_train(cluster_id_plot) + t_shift
        mcs_abs_cluster = np.zeros(len(spike_times)).astype('int') + max_ptp_channel
        first_chans_cluster = (mcs_abs_cluster - 20).clip(min=0)
        waveform_scale = 2/max_ptp
        # waveform_scale = 1/25
        plot_raw_waveforms_unit_geom(geom_array, num_channels, first_chans_cluster, mcs_abs_cluster, spike_times=spike_times, bin_file=raw_data_bin, x_geom_scale = 1/15, 
                                     y_geom_scale = 1/10, waveform_scale = waveform_scale, spikes_plot = num_spikes_plot, waveform_shape=(30,70), num_rows=3, 
                                     alpha=.2, h_shift=h_shift, do_mean=False, ax=ax_raw_wf, color=color[0])
        plot_raw_waveforms_unit_geom(geom_array, num_channels, first_chans_cluster, mcs_abs_cluster, spike_times=spike_times, bin_file=raw_data_bin, x_geom_scale = 1/15, 
                                     y_geom_scale = 1/10, waveform_scale = waveform_scale, spikes_plot = num_spikes_plot, waveform_shape=(30,70), num_rows=3, 
                                     alpha=1, h_shift=h_shift, do_mean=True, ax=ax_raw_wf, color=color[1])
        
        if do_denoised_waveform:
            mcs_abs_cluster = triaged_mcs_abs[cluster_labels==cluster_id_plot]
            first_chans_cluster = triaged_firstchans[cluster_labels==cluster_id_plot]
            waveforms = denoised_waveforms[non_triaged_idxs[cluster_labels==cluster_id_plot]]
            plot_waveforms_unit_geom(geom_array, num_channels, first_chans_cluster, mcs_abs_cluster, waveforms, x_geom_scale = 1/15, 
                                     y_geom_scale = 1/10, waveform_scale = waveform_scale, spikes_plot = num_spikes_plot, waveform_shape=(30,70), num_rows=3, 
                                     alpha=.2, h_shift=h_shift, do_mean=False, ax=ax_denoised_wf, color=color[0])
            plot_waveforms_unit_geom(geom_array, num_channels, first_chans_cluster, mcs_abs_cluster, waveforms, x_geom_scale = 1/15, 
                                     y_geom_scale = 1/10, waveform_scale = waveform_scale, spikes_plot = num_spikes_plot, waveform_shape=(30,70), num_rows=3, 
                                     alpha=1, h_shift=h_shift, do_mean=True, ax=ax_denoised_wf, color=color[1])
            
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

# def plot_unit_similarities_summary(cluster_id, closest_clusters, sorting1, sorting2, geom_array, raw_data_bin, recoring_duration, num_channels=40, num_spikes_plot=100, num_channels_similarity=20, 
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
#         waveforms1 = read_waveforms(st_1, raw_data_bin, geom_array, n_times=121)[0]
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
#             waveforms2 = read_waveforms(st_2, raw_data_bin, geom_array, n_times=121)[0]
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