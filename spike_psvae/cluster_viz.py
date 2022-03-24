import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import colorcet
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib import cm
matplotlib.use('Agg')
import os
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

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
        vx, vy = cov[0, 0], cov[1, 1]
        rho = cov[0, 1] / np.sqrt(vx * vy)
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
        
def plot_waveforms_geom(main_cluster_id, clusterer_to_be_plotted, clusters_to_plot, cluster_centers, geom_array, triaged_wfs, 
                        triaged_firstchans, triaged_mcs_abs, triaged_spike_index=None, bin_file=None, x_geom_scale = 1/25, 
                        y_geom_scale = 1/10, waveform_scale = .15, spikes_plot = 200, waveform_shape=(30,90), num_rows=3, 
                        alpha=.1, h_shift=.5, raw=False, ax=None, color_dict=None):    
    if color_dict is None:
        raise ValueError("Must pass valid color dict")
    if ax is None:
        plt.figure(figsize=(12,36), dpi=300)
        ax = plt.gca()
        
    cluster_centers_to_plot = cluster_centers[clusters_to_plot]
    geom_scale = [x_geom_scale, y_geom_scale]
    min_cluster_center = np.min(cluster_centers_to_plot[:,1])
    max_cluster_center =  np.max(cluster_centers_to_plot[:,1])
    geom_plot = geom_array*geom_scale
    first_chans_cluster = triaged_firstchans[clusterer_to_be_plotted.labels_==main_cluster_id]
    mcs_abs_cluster = triaged_mcs_abs[clusterer_to_be_plotted.labels_==main_cluster_id]
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
        if i > geom_array.shape[0] - 1:
            exit = True
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
        first_chans_cluster = triaged_firstchans[clusterer_to_be_plotted.labels_==cluster_id]
        if raw:
            if triaged_spike_index is not None and bin_file is not None:
                spike_times = triaged_spike_index[clusterer_to_be_plotted.labels_==cluster_id][:,0]
                waveforms_read = read_waveforms(spike_times, bin_file, geom_array, n_times=121)[0]
                waveforms = []
                for i, waveform in enumerate(waveforms_read):
                    waveforms.append(waveform[:,int(first_chans_cluster[i]):int(first_chans_cluster[i])+20])
                waveforms = np.asarray(waveforms)    
            else:
                raise ValueError("Need to specify spike_index and bin_file")
        else:
            waveforms = triaged_wfs[clusterer_to_be_plotted.labels_==cluster_id]
        vertical_lines = set()
        for i in range(min(spikes_plot, waveforms.shape[0])):
            for k, channel in enumerate(range(int(first_chans_cluster[i]),int(first_chans_cluster[i])+20)):
                if channel in channels_plot:
                    channel_position = geom_array[channel]*geom_scale
                    waveform = waveforms[i, waveform_shape[0]:waveform_shape[1],k].T.flatten()*waveform_scale
                    ax.plot(np.linspace(channel_position[0]-.75+h_shift*j, channel_position[0]+.5+h_shift*j, waveform.shape[0]), waveform + channel_position[1], alpha = alpha, c = color)
                    max_vert_line = np.linspace(channel_position[0]-.75+h_shift*j, channel_position[0]+.5+h_shift*j, waveform.shape[0])[12]
                    if max_vert_line not in vertical_lines:
                        vertical_lines.add(max_vert_line)
                        ax.axvline(max_vert_line, linestyle='--')             