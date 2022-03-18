#!/usr/bin/env python
# coding: utf-8

# # Clustering with localization-derived features with hdbscan
# 
# HDBSCAN is a clustering algorithm developed by Campello, Moulavi, and Sander. It extends DBSCAN by converting it into a hierarchical clustering algorithm, and then using a technique to extract a flat clustering based in the stability of clusters. 
# 
# Steps
# > 1. Transform the space according to the density/sparsity. 
#   2. Build the minimum spanning tree of the distance weighted graph. 
#   3. Construct a cluster hierarchy of connected components. 
#   4. Condense the cluster hierarchy based on minimum cluster size.
#   5. Extract the stable clusters from the condensed tree.
# 
# Important parameters
# 
# > min_cluster_size, int, optional (default=5): The minimum size of clusters; single linkage splits that contain fewer points than this will be considered points “falling out” of a cluster rather than a cluster splitting into two new clusters.
# 
# > min_samples, int, optional (default=None): The number of samples in a neighbourhood for a point to be considered a core point.

import numpy as np
import os
import scipy
import argparse
import hdbscan

from spike_psvae.cluster_viz import cluster_scatter, plot_waveforms_geom

#Helper functions

from scipy.spatial import cKDTree
def run_weighted_triage(x, y, z, alpha, maxptps, pcs, 
                        scales=(1,10,1,15,30,10),
                        threshold=100, ptp_threshold=3, c=1):
    
    ptp_filter = np.where(maxptps>ptp_threshold)
    x = x[ptp_filter]
    y = y[ptp_filter]
    z = z[ptp_filter]
    alpha = alpha[ptp_filter]
    maxptps = maxptps[ptp_filter]
    pcs = pcs[ptp_filter]
    
    feats = np.c_[scales[0]*x,
                  scales[1]*np.log(y),
                  scales[2]*z,
                  scales[3]*np.log(alpha),
                  scales[4]*np.log(maxptps),
                  scales[5]*pcs[:,:3]]
    tree = cKDTree(feats)
    dist, ind = tree.query(feats, k=6)
    dist = dist[:,1:]
    dist = np.sum(c*np.log(dist) + np.log(1/(scales[4]*np.log(maxptps)))[:,None], 1)
    idx_keep = dist <= np.percentile(dist, threshold)
    
    triaged_x = x[idx_keep]
    triaged_y = y[idx_keep]
    triaged_z = z[idx_keep]
    triaged_alpha = alpha[idx_keep]
    triaged_maxptps = maxptps[idx_keep]
    triaged_pcs = pcs[idx_keep]
    
    return triaged_x, triaged_y, triaged_z, triaged_alpha, triaged_maxptps, triaged_pcs, ptp_filter, idx_keep

def main():
    
    ap = argparse.ArgumentParser()

    ap.add_argument("--loc_features_path")
    ap.add_argument("--geom")
    ap.add_argument("--out_folder", type=str, default='clustering_results')
    ap.add_argument("--triage_quantile", type=int, default=75)
    ap.add_argument("--do_infer_ptp", action='store_true')
    ap.add_argument("--num_spikes_cluster", type=int, default=None)
    ap.add_argument("--min_cluster_size", type=int, default=25)
    ap.add_argument("--min_samples", type=int, default=25)
    ap.add_argument("--num_spikes_plot", type=int, default=250)
    ap.add_argument("--num_rows_plot", type=int, default=3)
    ap.add_argument("--do_save_figures", action='store_true')
    ap.add_argument("--no_verbose", action='store_false')
    
    args = ap.parse_args()

    data_path = args.loc_features_path
    data_name = os.path.basename(os.path.normpath(data_path))
    data_dir = data_path + '/'

    spike_index = np.load(data_dir+'spike_index.npy')
    results_localization = np.load(data_dir+'localization_results.npy')
    ptps_localized = np.load(data_dir+'ptps.npy')
    geom = args.geom
    geom_array = np.load(data_dir+geom)

    #Remove indices with 0 spike 
    spike_index = spike_index[results_localization[:, 4]!=0]
    ptps_localized = ptps_localized[results_localization[:, 4]!=0]
    results_localization = results_localization[results_localization[:, 4]!=0]
    displacement = np.load(data_dir+'displacement_array.npy')
    
    #AE features not used at this point.
    ae_features = np.load(data_dir+'ae_features.npy') 

    # register displacement (here starts at sec 50)
    z_abs = results_localization[:, 1] - displacement[spike_index[:, 0]//30000]
    x = results_localization[:, 0]
    y = results_localization[:, 2]
    z = z_abs
    alpha = results_localization[:, 3]
    maxptps = results_localization[:, 4]
    pcs = ae_features
    
    #perform triaging 
    triaged_x, triaged_y, triaged_z, triaged_alpha, triaged_maxptps, triaged_pcs, ptp_filter, idx_keep = run_weighted_triage(x, y, z, alpha, maxptps, pcs, threshold=args.triage_quantile)
    triaged_spike_index = spike_index[ptp_filter][idx_keep]
    
    #can infer ptp if wanted
    do_infer_ptp = args.do_infer_ptp
    if do_infer_ptp:
        def infer_ptp(x, y, z, alpha):
            return (alpha / np.sqrt((geom_array[:, 0] - x)**2 + (geom_array[:, 1] - z)**2 + y**2)).max()
        vinfer_ptp = np.vectorize(infer_ptp)
        triaged_maxptps = vinfer_ptp(triaged_x, triaged_y, triaged_z, triaged_alpha)

    triaged_log_ptp = triaged_maxptps.copy()
    triaged_log_ptp[triaged_log_ptp >= 27.5] = 27.5
    triaged_log_ptp = np.log(triaged_log_ptp+1)
    triaged_log_ptp[triaged_log_ptp<=1.25] = 1.25
    triaged_ptp_rescaled = (triaged_log_ptp - triaged_log_ptp.min())/(triaged_log_ptp.max() - triaged_log_ptp.min())

    triaged_mcs_abs = results_localization[:,6][ptp_filter][idx_keep]
    triaged_firstchans = results_localization[:,5][ptp_filter][idx_keep]

    # ## Create feature set for clustering
    if args.num_spikes_cluster is None:
        num_spikes = triaged_x.shape[0]
    else:
        num_spikes = args.num_spikes_cluster
    if args.no_verbose:
        print(f"clustering {num_spikes} spikes")
    scales = (1,10,1,15,30,10) #predefined scales for each feature
    features = np.concatenate((np.expand_dims(triaged_x,1), np.expand_dims(triaged_z,1), np.expand_dims(np.log(triaged_maxptps)*scales[4],1)), axis=1)[:num_spikes]
    
    #perform hdbscan clustering
    min_cluster_size = args.min_cluster_size
    min_samples = args.min_samples
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusterer.fit(features)
    if args.no_verbose:
        print(clusterer)
    cluster_centers = []
    for label in np.unique(clusterer.labels_):
        if label != -1:
            cluster_centers.append(clusterer.weighted_cluster_centroid(label))
    cluster_centers = np.asarray(cluster_centers)
    
    #re-label each cluster by z-depth
    labels_depth = np.argsort(-cluster_centers[:,1])
    label_to_id = {}
    for i, label in enumerate(labels_depth):
        label_to_id[label] = i
    label_to_id[-1] = -1
    new_labels = np.vectorize(label_to_id.get)(clusterer.labels_) 
    clusterer.labels_ = new_labels

    save_dir_path = args.out_folder + '/' + data_name + '_' + str(num_spikes) + f'infer_ptp_{do_infer_ptp}_' + 'hdbscan_' + 'min_cluster_size' + str(clusterer.min_cluster_size) + '_' + 'min_samples' + str(clusterer.min_samples)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    
    #save cluster indices
    if not os.path.exists(save_dir_path + '/cluster_indices'):
        os.makedirs(save_dir_path + '/cluster_indices')
    for cluster_id in np.unique(clusterer.labels_):
        indices = triaged_spike_index[np.where(clusterer.labels_==cluster_id)]
        np.save(save_dir_path + f"/cluster_indices/cluster_{cluster_id}_indices.npy", indices)
    
    if args.do_save_figures:
        if args.no_verbose:
            print("saving figures...")
        import spikeinterface 
        from spikeinterface.toolkit import compute_correlograms
        from spikeinterface.comparison import compare_two_sorters
        from spikeinterface.widgets import plot_agreement_matrix
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib import cm
        matplotlib.use('Agg')
        plt.rcParams['axes.xmargin'] = 0
        plt.rcParams['axes.ymargin'] = 0
        from tqdm import tqdm
        #recompute cluster centers for new labels
        cluster_centers = []
        for label in np.unique(clusterer.labels_):
            if label != -1:
                cluster_centers.append(clusterer.weighted_cluster_centroid(label))
        cluster_centers = np.asarray(cluster_centers)
        
        vir = cm.get_cmap('viridis')
        clusterer_to_be_plotted = clusterer
        
        #load waveforms (need memmap?)
        wfs_localized = np.load(data_dir+'wfs.npy')
        wfs_raw = np.load(data_dir+'wfs_raws.npy')
        wfs_cleaned = np.load(data_dir+'wfs_cleaned.npy')
        wfs_localized = wfs_localized[results_localization[:, 4]!=0]
        wfs_raw = wfs_raw[results_localization[:, 4]!=0]
        wfs_cleaned = wfs_cleaned[results_localization[:, 4]!=0]
        triaged_wfs_localized = wfs_localized[ptp_filter][idx_keep]
        triaged_wfs_raw = wfs_raw[ptp_filter][idx_keep]
        triaged_wfs_cleaned = wfs_cleaned[ptp_filter][idx_keep]
        triaged_spike_index = spike_index[ptp_filter][idx_keep]

        del wfs_localized
        del wfs_cleaned
        del wfs_raw
        
        color_arr = vir(triaged_ptp_rescaled)
        color_arr[:, 3] = triaged_ptp_rescaled

        # ## Define colors
        unique_colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3', '#808000', '#000075', '#000000']
        
        cluster_color_dict = {}
        for cluster_id in np.unique(clusterer_to_be_plotted.labels_):
            cluster_color_dict[cluster_id] = unique_colors[cluster_id % len(unique_colors)]
        cluster_color_dict[-1] = '#808080' #set outlier color to grey
            
        fig, axes = plt.subplots(1, 3, sharey=True, figsize=(16, 24), dpi=300)

        matplotlib.rcParams.update({'font.size': 12})
        z_cutoff = (0,3000)
        xs, zs, ids = features[:,0], features[:,1], clusterer_to_be_plotted.labels_
        axes[0].set_ylim(z_cutoff)
        cluster_scatter(xs, zs, ids, ax=axes[0], excluded_ids=set([-1]), s=20, alpha=.05, color_dict=cluster_color_dict)
        axes[0].scatter(geom_array[:, 0], geom_array[:, 1], s=20, c='orange', marker = "s")
        axes[0].set_title(f"min_cluster_size {clusterer_to_be_plotted.min_cluster_size}, min_samples {clusterer_to_be_plotted.min_samples}");
        axes[0].set_ylabel("z");
        axes[0].set_xlabel("x");

        ys, zs, ids = features[:,2], features[:,1], clusterer_to_be_plotted.labels_
        axes[1].set_ylim(z_cutoff)
        cluster_scatter(ys, zs, ids, ax=axes[1], excluded_ids=set([-1]), s=20, alpha=.05, color_dict=cluster_color_dict)
        axes[1].set_title(f"min_cluster_size {clusterer_to_be_plotted.min_cluster_size}, min_samples {clusterer_to_be_plotted.min_samples}");
        axes[1].set_xlabel("scaled ptp");

        axes[2].scatter(xs, zs, s=20, c=color_arr[:num_spikes], alpha=.1)
        axes[2].scatter(geom_array[:, 0], geom_array[:, 1], s=20, c='orange', marker = "s")
        axes[2].set_ylim(z_cutoff)
        axes[2].set_title("ptps");

        fig.suptitle(f'x,z,scaled_logptp features, {num_spikes} datapoints');
        plt.close(fig)
        fig.savefig(save_dir_path + '/array_full_scatter.png')

        matplotlib.rcParams.update({'font.size': 22})

        indices_list = []
        labels_list = []
        for cluster_id in np.unique(clusterer_to_be_plotted.labels_):
            label_ids = np.where(clusterer_to_be_plotted.labels_==cluster_id)
            indices = triaged_spike_index[label_ids][:,0]
            num_labels = label_ids[0].shape[0]
            indices_list.append(indices)
            labels_list.append((np.zeros(num_labels) + cluster_id).astype('int'))
        sorting = spikeinterface.numpyextractors.NumpySorting.from_times_labels(times_list=np.concatenate(indices_list), 
                                                                                labels_list=np.concatenate(labels_list), 
                                                                                sampling_frequency=30000)
        sorting_comparison = compare_two_sorters(sorting, sorting)
        fig = plt.figure(figsize=(36,36))
        plot_agreement_matrix(sorting_comparison, figure=fig)
        plt.title("Agreement matrix")
        plt.close(fig)
        fig.savefig(save_dir_path + '/agreement_matrix.png')

        pbar = tqdm(np.unique(clusterer_to_be_plotted.labels_))
        for cluster_id in pbar:
            if cluster_id > -1:
                pbar.set_description("Processing cluster %s" % cluster_id)
                cluster_dir_path = save_dir_path

                curr_cluster_center = cluster_centers[cluster_id]
                num_close_clusters = 2
                dist_other_clusters = np.linalg.norm(curr_cluster_center[:2] - cluster_centers[:,:2], axis=1)
                closest_clusters = np.argsort(dist_other_clusters)[1:num_close_clusters + 1]
                closest_clusters_dist = dist_other_clusters[closest_clusters]
                closest_clusters_features = cluster_centers[closest_clusters]
                all_cluster_features_close = features[np.where((clusterer_to_be_plotted.labels_ == cluster_id) | (clusterer_to_be_plotted.labels_ == closest_clusters[0]) | (clusterer_to_be_plotted.labels_ == closest_clusters[1]))]

                #buffers for range of scatter plots
                z_buffer = 5
                x_buffer = 5
                scaled_ptp_cutoff = 2.5

                z_cutoff = (np.min(all_cluster_features_close[:,1] - z_buffer), np.max(all_cluster_features_close[:,1] + z_buffer))
                x_cutoff = (np.min(all_cluster_features_close[:,0] - x_buffer), np.max(all_cluster_features_close[:,0] + x_buffer))
                scaled_ptps_cutoff = (np.min(all_cluster_features_close[:,2] - scaled_ptp_cutoff), np.max(all_cluster_features_close[:,2] + scaled_ptp_cutoff))

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
                ptps_cluster = features[:num_spikes][:,2][np.where(clusterer_to_be_plotted.labels_ ==cluster_id)]
                spike_train_s = triaged_spike_index[:,0][:num_spikes][clusterer_to_be_plotted.labels_==cluster_id] / 30000
                ax.plot(spike_train_s, ptps_cluster)
                ax.set_title(f"ptps over time");
                ax.set_ylabel("ptp");
                ax.set_xlabel("seconds");

                ax = ax_ptp_z
                zs_cluster = features[:num_spikes][:,1][np.where(clusterer_to_be_plotted.labels_ ==cluster_id)]
                ax.scatter(zs_cluster, ptps_cluster);
                ax.set_title(f"zs vs. ptps");
                ax.set_xlabel("zs");
                ax.set_ylabel("ptps");

                ax = ax_scatter_xz
                xs, zs, ids = features[:,0], features[:,1], clusterer_to_be_plotted.labels_
                cluster_scatter(xs, zs, ids, ax=ax, excluded_ids=set([-1]), s=100, alpha=.3, color_dict=cluster_color_dict)
                ax.scatter(geom_array[:, 0], geom_array[:, 1], s=100, c='orange', marker = "s")
                ax.set_title(f"x vs. z");
                ax.set_ylabel("z");
                ax.set_xlabel("x");
                ax.set_ylim(z_cutoff)
                ax.set_xlim(x_cutoff)

                ax = ax_scatter_sptpz
                ys, zs, ids = features[:,2], features[:,1], clusterer_to_be_plotted.labels_
                cluster_scatter(ys, zs, ids, ax=ax, excluded_ids=set([-1]), s=100, alpha=.3, color_dict=cluster_color_dict)
                ax.set_title(f"scaled ptp vs. z");
                ax.set_xlabel("scaled ptp");
                ax.set_yticks([])
                ax.set_ylim(z_cutoff)
                ax.set_xlim(scaled_ptps_cutoff)

                ax = ax_scatter_xzptp
                ax.scatter(xs, zs, s=100, c=color_arr[:num_spikes], alpha=.3)
                ax.scatter(geom_array[:, 0], geom_array[:, 1], s=100, c='orange', marker = "s")
                ax.set_title("ptps")
                ax.set_yticks([])
                ax.set_ylim(z_cutoff)
                ax.set_xlim(x_cutoff)
                
                #figure params
                x_geom_scale = 1/25
                y_geom_scale = 1/10
                spikes_plot = args.num_spikes_plot
                waveform_shape=(30, 70)
                num_rows = args.num_rows_plot

                ax = ax_isi
                spike_train = triaged_spike_index[:,0][:num_spikes][clusterer_to_be_plotted.labels_==cluster_id]
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
                    spike_train_2 = triaged_spike_index[:,0][:num_spikes][clusterer_to_be_plotted.labels_==cluster_isi_id]
                    sorting = spikeinterface.numpyextractors.NumpySorting.from_times_labels(times_list=np.concatenate((spike_train,spike_train_2)), 
                                                                                            labels_list=np.concatenate((np.zeros(spike_train.shape[0]).astype('int'), np.zeros(spike_train_2.shape[0]).astype('int')+1)), 
                                                                                            sampling_frequency=30000)
                    bin_ms = 1.0
                    correlograms, bins = compute_correlograms(sorting, symmetrize=True, window_ms=10.0, bin_ms=bin_ms)
                    axes[i].bar(bins[1:], correlograms[0][1], width=bin_ms, align='center')
                    axes[i].set_xticks(bins[1:])
                    axes[i].set_xlabel('lag (ms)')
                    axes[i].set_title(f'cluster_{cluster_id}_cluster_{cluster_isi_id}_xcorrelogram.png')

                clusters_plot = np.concatenate(([cluster_id], closest_clusters))

                max_ptps = []
                for cluster_id_ptp in clusters_plot:
                    max_ptps.append(np.max(triaged_maxptps[:num_spikes][clusterer_to_be_plotted.labels_==cluster_id_ptp]))
                max_ptp = np.max(max_ptps)
                waveform_scale = 2/max_ptp

                ax = ax_denoised
                plot_waveforms_geom(cluster_id, clusterer_to_be_plotted, clusters_plot, cluster_centers, geom_array, triaged_wfs_localized[:num_spikes], triaged_firstchans[:num_spikes], 
                                    triaged_mcs_abs[:num_spikes], x_geom_scale=x_geom_scale, y_geom_scale=y_geom_scale, waveform_scale=waveform_scale, spikes_plot=spikes_plot, 
                                    waveform_shape=waveform_shape,  h_shift=0, ax=ax, alpha=.1, num_rows=num_rows, color_dict=cluster_color_dict)
                ax.set_title("denoised waveforms");
                ax.set_ylabel("z")
                ax.set_xlabel("x")
                ax.set_yticks([])

                ax = ax_raw
                plot_waveforms_geom(cluster_id, clusterer_to_be_plotted, clusters_plot, cluster_centers, geom_array, triaged_wfs_raw[:num_spikes], triaged_firstchans[:num_spikes], 
                                    triaged_mcs_abs[:num_spikes], x_geom_scale=x_geom_scale, y_geom_scale=y_geom_scale, waveform_scale=waveform_scale, 
                                    spikes_plot=spikes_plot, waveform_shape=waveform_shape, h_shift=0, ax=ax, alpha=.1, num_rows=num_rows, color_dict=cluster_color_dict)
                ax.set_title("raw waveforms")
                ax.set_xlim(ax_denoised.get_xlim())
                ax.set_ylim(ax_denoised.get_ylim())
                ax.set_xlabel("x")
                ax.set_ylabel("z")

                ax = ax_cleaned
                plot_waveforms_geom(cluster_id, clusterer_to_be_plotted, clusters_plot, cluster_centers, geom_array, triaged_wfs_cleaned[:num_spikes], triaged_firstchans[:num_spikes], 
                                    triaged_mcs_abs[:num_spikes], x_geom_scale=x_geom_scale, y_geom_scale=y_geom_scale, waveform_scale=waveform_scale, spikes_plot=spikes_plot, 
                                    waveform_shape=waveform_shape, h_shift=0, ax=ax,alpha=.1, num_rows=num_rows, color_dict=cluster_color_dict)
                ax.set_title("cleaned waveforms")
                ax.set_yticks([])
                ax.set_xlim(ax_denoised.get_xlim())
                ax.set_ylim(ax_denoised.get_ylim())
                ax.set_xlabel("x")
                
                plt.tight_layout()
                plt.close(fig)
                save_z_int = int(cluster_centers[cluster_id][1])
                save_str = str(save_z_int).zfill(4)
                fig.savefig(cluster_dir_path + f"/Z{save_str}_cluster{cluster_id}.png")

if __name__ == "__main__":
    main()

