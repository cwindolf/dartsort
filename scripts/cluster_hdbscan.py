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
from spike_psvae.cluster_viz import cluster_scatter, plot_waveforms_geom, plot_raw_waveforms_unit_geom, plot_venn_agreement
from spike_psvae.cluster_viz import plot_array_scatter, plot_self_agreement, plot_single_unit_summary, plot_agreement_venn, plot_isi_distribution, plot_waveforms_unit_geom, plot_unit_similarities
from spike_psvae.cluster_viz import plot_unit_similarity_heatmaps
from spike_psvae.cluster_utils import make_sorting_from_labels_frames, compute_cluster_centers, relabel_by_depth, run_weighted_triage, remove_duplicate_units
from spike_psvae.cluster_utils import get_agreement_indices, compute_spiketrain_agreement, get_unit_similarities, compute_shifted_similarity, read_waveforms
from spike_psvae.cluster_utils import get_closest_clusters_hdbscan, get_closest_clusters_kilosort, get_closest_clusters_hdbscan_kilosort, get_closest_clusters_kilosort_hdbscan
from joblib import Parallel, delayed
from tqdm import tqdm
import h5py
from scipy.spatial import cKDTree
import pickle


def main():
    
    ap = argparse.ArgumentParser()

    ap.add_argument("--loc_features_path")
    ap.add_argument("--geom")
    ap.add_argument("--out_folder", type=str, default='clustering_results')
    ap.add_argument("--triage_quantile", type=int, default=75)
    ap.add_argument("--ptp_threshold", type=int, default=3)
    ap.add_argument("--do_infer_ptp", action='store_true')
    ap.add_argument("--num_spikes_cluster", type=int, default=None)
    ap.add_argument("--min_cluster_size", type=int, default=25)
    ap.add_argument("--min_samples", type=int, default=25)
    ap.add_argument("--do_save_figures", action='store_true')
    ap.add_argument("--no_verbose", action='store_false')
    ap.add_argument("--raw_data_bin", type=str, default=None)
    ap.add_argument("--residual_data_bin", type=str, default=None)
    ap.add_argument("--num_spikes_plot", type=int, default=100)
    ap.add_argument("--num_rows_plot", type=int, default=3)
    ap.add_argument("--num_cpus_plot", type=int, default=12)
    
    args = ap.parse_args()

    data_path = args.loc_features_path
    data_name = os.path.basename(os.path.normpath(data_path))
    data_dir = data_path + '/'
    
    #load features
    spike_index = np.load(data_dir+'spike_index.npy')
    spike_index[:,0] = spike_index[:,0] + 18 #only for Hyun's data
    results_localization = np.load(data_dir+'localization_results.npy')
    ptps_localized = np.load(data_dir+'ptps.npy')
    raw_data_bin = data_dir + args.raw_data_bin
    residual_data_bin = data_dir + args.residual_data_bin
    geom = args.geom
    geom_array = np.load(data_dir+geom)
    #AE features not used at this point.
    # ae_features = np.load(data_dir+'ae_features.npy') 
    # register displacement (here starts at sec 50)
    # displacement = np.load(data_dir+'displacement_array.npy' )(if you have displacement)
    # z_abs = results_localization[:, 1] - displacement[spike_index[:, 0]//30000] (if you have displacement)
    z_abs =  np.load(data_dir+'z_reg.npy') #if you already have registered zs
    x = results_localization[:, 0]
    y = results_localization[:, 2]
    z = z_abs
    alpha = results_localization[:, 3]
    maxptps = results_localization[:, 4]
    
    #perform triaging 
    triaged_x, triaged_y, triaged_z, triaged_alpha, triaged_maxptps, _, ptp_filter, idx_keep = run_weighted_triage(x, y, z, alpha, maxptps, threshold=args.triage_quantile, 
                                                                                                                   ptp_threshold=args.ptp_threshold, ptp_weighting=True) #pcs None
    triaged_spike_index = spike_index[ptp_filter][idx_keep]
    triaged_mcs_abs = spike_index[:,1][ptp_filter][idx_keep]
    non_triaged_idxs = ptp_filter[0][idx_keep]

    #can infer ptp
    do_infer_ptp = args.do_infer_ptp
    if do_infer_ptp:
        if args.no_verbose:
            print(f"inferring ptps using registered zs..")
        def infer_ptp(x, y, z, alpha):
            return (alpha / np.sqrt((geom_array[:, 0] - x)**2 + (geom_array[:, 1] - z)**2 + y**2)).max()
        vinfer_ptp = np.vectorize(infer_ptp)
        triaged_maxptps = vinfer_ptp(triaged_x, triaged_y, triaged_z, triaged_alpha)
 
    #load firstchans
    #triaged_firstchans = results_localization[:,5][ptp_filter][idx_keep] #if you saved firstchans
    filename = data_dir + "subtraction_1min_standardized_t_0_None.h5"
    with h5py.File(filename, "r") as f:
        print("Keys: %s" % f.keys())
        firstchans = np.asarray(list(f["first_channels"]))
    triaged_firstchans = firstchans[ptp_filter][idx_keep]

    # ## Create feature set for clustering
    if args.num_spikes_cluster is None:
        num_spikes = triaged_x.shape[0]
    else:
        num_spikes = args.num_spikes_cluster
    if args.no_verbose:
        print(f"clustering {num_spikes} spikes")
    triaged_firstchans = triaged_firstchans[:num_spikes]
    triaged_alpha = triaged_alpha[:num_spikes]
    triaged_spike_index = triaged_spike_index[:num_spikes]
    triaged_x = triaged_x[:num_spikes]
    triaged_y = triaged_y[:num_spikes]
    triaged_z = triaged_z[:num_spikes]
    triaged_maxptps = triaged_maxptps[:num_spikes]
    triaged_mcs_abs = triaged_mcs_abs[:num_spikes]
    
    scales = (1,10,1,15,30) #predefined scales for each feature
    features = np.concatenate((np.expand_dims(triaged_x,1), np.expand_dims(triaged_z,1), np.expand_dims(np.log(triaged_maxptps)*scales[4],1)), axis=1)
    
    #perform hdbscan clustering
    min_cluster_size = args.min_cluster_size
    min_samples = args.min_samples
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusterer.fit(features)
    if args.no_verbose:
        print(clusterer)

    #compute cluster centers
    cluster_centers = compute_cluster_centers(clusterer)

    #re-label each cluster by z-depth
    clusterer = relabel_by_depth(clusterer, cluster_centers)

    #remove duplicate units by spike_times_agreement and ptp
    clusterer = remove_duplicate_units(clusterer, triaged_spike_index[:,0], triaged_maxptps)

    #re-compute cluster centers
    cluster_centers = compute_cluster_centers(clusterer)

    save_dir_path = args.out_folder + '/' + data_name + '_' + str(num_spikes) + 'hdbscan_' + 'min_cluster_size' + str(clusterer.min_cluster_size) + '_' + 'min_samples' + str(clusterer.min_samples)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
        
    #save triaged indices
    mask = np.ones(spike_index[:,1].size, dtype=bool)
    mask[ptp_filter[0][idx_keep]] = False
    triaged_indices = np.where(mask)[0]
    np.save(save_dir_path + '/triaged_indices', triaged_indices)
    
    #save cluster indices
    if not os.path.exists(save_dir_path + '/cluster_indices'):
        os.makedirs(save_dir_path + '/cluster_indices')
    for cluster_id in np.unique(clusterer.labels_):
        indices = triaged_spike_index[np.where(clusterer.labels_==cluster_id)]
        np.save(save_dir_path + f"/cluster_indices/cluster_{cluster_id}_indices.npy", indices)
        
    pickle.dump(clusterer, open(save_dir_path + '/clusterer', 'wb'))
    
    import matplotlib.pyplot as plt
    from matplotlib import cm

    vir = cm.get_cmap('viridis')
    triaged_log_ptp = triaged_maxptps.copy()
    triaged_log_ptp[triaged_log_ptp >= 27.5] = 27.5
    triaged_log_ptp = np.log(triaged_log_ptp+1)
    triaged_log_ptp[triaged_log_ptp<=1.25] = 1.25
    triaged_ptp_rescaled = (triaged_log_ptp - triaged_log_ptp.min())/(triaged_log_ptp.max() - triaged_log_ptp.min())
    color_arr = vir(triaged_ptp_rescaled)
    color_arr[:, 3] = triaged_ptp_rescaled

    # ## Define colors
    unique_colors = ['#e6194b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3', '#808000', '#000075', '#000000']

    cluster_color_dict = {}
    for cluster_id in np.unique(clusterer.labels_):
        cluster_color_dict[cluster_id] = unique_colors[cluster_id % len(unique_colors)]
    cluster_color_dict[-1] = '#808080' #set outlier color to grey

    ##### plot array scatter #####
    fig = plot_array_scatter(clusterer.labels_, geom_array, triaged_x, triaged_z, triaged_maxptps, cluster_color_dict, color_arr, 
                             min_cluster_size=clusterer.min_cluster_size, min_samples=clusterer.min_samples, z_cutoff=(0, 3900))
    fig.suptitle(f'x,z,scaled_logptp features, {num_spikes} datapoints');
    plt.close(fig)
    fig.savefig(save_dir_path + '/array_full_scatter.png')

    ##### plot clusterer self-agreement #####
    fig = plot_self_agreement(clusterer.labels_, triaged_spike_index[:,0])
    plt.title("Agreement matrix")
    plt.close(fig)
    fig.savefig(save_dir_path + '/agreement_matrix.png')
    
    if args.do_save_figures:
        if args.no_verbose:
            print("saving waveform figures...")
        ##### plot individual cluster summaries #####
        wfs_localized = np.load(data_dir+'denoised_wfs.npy', mmap_mode='r') #np.memmap(data_dir+'denoised_waveforms.npy', dtype='float32', shape=(290025, 121, 40))
        wfs_subtracted = np.load(data_dir+'subtracted_wfs.npy', mmap_mode='r')
        
        def job(cluster_id):
            fig = plot_single_unit_summary(
                cluster_id,
                clusterer.labels_,
                cluster_centers,
                geom_array,
                args.num_spikes_plot,
                args.num_rows_plot,
                triaged_x,
                triaged_z,
                triaged_maxptps,
                triaged_firstchans,
                triaged_mcs_abs,
                triaged_spike_index[:,0],
                non_triaged_idxs,
                wfs_localized,
                wfs_subtracted,
                cluster_color_dict,
                color_arr,
                raw_data_bin,
                residual_data_bin,
            )
            save_z_int = int(cluster_centers.loc[cluster_id][1])
            save_str = str(save_z_int).zfill(4)
            fig.savefig(save_dir_path + f"/Z{save_str}_cluster{cluster_id}.png")
            plt.close(fig)
        with Parallel(
            args.num_cpus_plot,
        ) as p:
            unit_ids = np.setdiff1d(np.unique(clusterer.labels_), [-1])
            for res in p(delayed(job)(u) for u in  tqdm(unit_ids)):
                pass

if __name__ == "__main__":
    main()

