import matplotlib
matplotlib.use('Agg')

import sys
from matplotlib.gridspec import GridSpec

import logging
import os
try:
    from pathlib2 import Path
except ImportError:
    from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import notebook

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import numpy as np
from tqdm.auto import tqdm
from mpl_toolkits import mplot3d

import h5py
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib import cm
import hdbscan

from spike_psvae.cluster_viz import plot_array_scatter, plot_self_agreement, plot_single_unit_summary, cluster_scatter, plot_agreement_venn, plot_unit_similarities

import spikeinterface 
from spikeinterface.toolkit import compute_correlograms
from spikeinterface.comparison import compare_two_sorters
from spikeinterface.widgets import plot_agreement_matrix

from spike_psvae.cluster_utils import make_sorting_from_labels_frames, compute_cluster_centers, relabel_by_depth, run_weighted_triage, remove_duplicate_units
from spike_psvae.cluster_utils import get_agreement_indices, compute_spiketrain_agreement, get_unit_similarities, compute_shifted_similarity, read_waveforms
from spike_psvae.cluster_utils import get_closest_clusters_hdbscan, get_closest_clusters_kilosort, get_closest_clusters_hdbscan_kilosort, get_closest_clusters_kilosort_hdbscan

from spike_psvae.merge_split import split_clusters, get_templates, get_merged
from spike_psvae.denoise import SingleChanDenoiser

np.random.seed(0)

standardized_path = '/media/cat/data/CSH_ZAD_026_1800_1860/1min_standardized.bin'
residuals_path = '/media/cat/data/CSH_ZAD_026_1800_1860/residual_1min_standardized_t_0_None.bin'
# standardized_path = '/media/cat/julien//1min_standardized.bin'
standardized_dtype = 'float32'

# denoiser = denoiser.cuda()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

denoiser = SingleChanDenoiser()
denoiser.load()
denoiser.to(device)

geom_path = '/media/cat/cole/CSH_ZAD_026_1800_1860/np1_channel_map.npy'
geom_array = np.load(geom_path)
n_chans = geom_array.shape[0]
sampling_rate = 30000


loc_res = np.load('/media/cat/data/CSH_ZAD_026_1800_1860/localization_results.npy')
spike_index = np.load('/media/cat/data/CSH_ZAD_026_1800_1860/spike_index.npy')
z_registered = np.load('/media/cat/data/CSH_ZAD_026_1800_1860/z_reg.npy')
scales=(1,10,1,15,30,10)



print('triaging + first clustering')

triaged_x, triaged_y, triaged_z, triaged_alpha, triaged_maxptps, triaged_pcs, ptp_filter, idx_keep = run_weighted_triage(loc_res[:, 0], 
                        loc_res[:, 2], z_registered, loc_res[:, 3], loc_res[:, 4], pcs=None, 
                        scales=(1,10,1,15,30,10),
                        threshold=80, ptp_threshold=3, c=1)

non_triaged_idxs = ptp_filter[0][idx_keep]

locs_triaged = loc_res[(loc_res[:, 4] >= 3)][idx_keep]
locs_triaged[:, 1] = triaged_z
spike_index_triaged = spike_index[(loc_res[:, 4] >= 3)][idx_keep]
print(spike_index_triaged.shape)

clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
features = np.concatenate((np.expand_dims(triaged_x,1), np.expand_dims(triaged_z,1), np.expand_dims(np.log(triaged_maxptps)*scales[4],1)), axis=1)
clusterer.fit(features)

#ORDER HDBSCAN LABELS
labels = clusterer.labels_
z_labels = np.zeros(clusterer.labels_.max()+1)
for i in range(clusterer.labels_.max()+1):
    z_labels[i] = triaged_z[clusterer.labels_ == i].mean()
ordered_labels = labels.copy()
z_argsort = z_labels.argsort()[::-1]
for i in range(clusterer.labels_.max()+1):
    ordered_labels[labels == z_argsort[i]] = i
    
#Uncomment for visualizing clusterer output
vir = cm.get_cmap('viridis')
triaged_log_ptp = triaged_maxptps.copy()
triaged_log_ptp[triaged_log_ptp >= 27.5] = 27.5
triaged_log_ptp = np.log(triaged_log_ptp+1)
triaged_log_ptp[triaged_log_ptp<=1.25] = 1.25
triaged_ptp_rescaled = (triaged_log_ptp - triaged_log_ptp.min())/(triaged_log_ptp.max() - triaged_log_ptp.min())
color_arr = vir(triaged_ptp_rescaled)
color_arr[:, 3] = triaged_ptp_rescaled
unique_colors = ['#e6194b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3', '#808000', '#000075', '#000000']
cluster_color_dict = {}
for cluster_id in np.unique(ordered_labels):
    cluster_color_dict[cluster_id] = unique_colors[cluster_id % len(unique_colors)]
cluster_color_dict[-1] = '#808080' #set outlier color to grey
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 12))
z_cutoff = (-50,3900)
min_cluster_size = 25
min_samples = 25
# matplotlib.rcParams.update({'font.size': 12})
xs, zs, ids = triaged_x, triaged_z, ordered_labels
axes[0].set_ylim(z_cutoff)
cluster_scatter(xs, zs, ids, ax=axes[0], excluded_ids=set([-1]), s=20, alpha=.05, color_dict=cluster_color_dict)
axes[0].scatter(geom_array[:, 0], geom_array[:, 1], s=20, c='orange', marker = "s")
axes[0].set_title(f"min_cluster_size {min_cluster_size}, min_samples {min_samples}");
axes[0].set_ylabel("z");
axes[0].set_xlabel("x");
ys, zs, ids = triaged_maxptps, triaged_z, ordered_labels
axes[1].set_ylim(z_cutoff)
cluster_scatter(ys, zs, ids, ax=axes[1], excluded_ids=set([-1]), s=20, alpha=.05, color_dict=cluster_color_dict)
axes[1].set_title(f"min_cluster_size {min_cluster_size}, min_samples {min_samples}");
axes[1].set_xlabel("scaled ptp");
axes[2].scatter(xs, zs, s=20, c=color_arr, alpha=.1)
axes[2].scatter(geom_array[:, 0], geom_array[:, 1], s=20, c='orange', marker = "s")
axes[2].set_ylim(z_cutoff)
axes[2].set_title("ptps");
fig.savefig('first_clustering_pass.png')
plt.close(fig)
    
print(len(np.unique(ordered_labels)))

print('split/merge')
    
temp_clusterer = get_templates(standardized_path, geom_array, ordered_labels.max()+1, spike_index_triaged, ordered_labels)

list_argmin = np.zeros(temp_clusterer.shape[0])
for i in range(temp_clusterer.shape[0]):
    list_argmin[i] = temp_clusterer[i, :, temp_clusterer[i].ptp(0).argmax()].argmin()

idx_not_aligned = np.where(list_argmin!=60)[0]

for unit in idx_not_aligned:
    mc = temp_clusterer[unit].ptp(0).argmax()
    offset = temp_clusterer[unit, :, mc].argmin()
    spike_index_triaged[ordered_labels == unit, 0] += offset-60

idx_sorted = spike_index_triaged[:, 0].argsort()
spike_index_triaged = spike_index_triaged[idx_sorted]
ordered_labels = ordered_labels[idx_sorted]
triaged_x = triaged_x[idx_sorted]
triaged_z = triaged_z[idx_sorted]
triaged_maxptps = triaged_maxptps[idx_sorted]

labels_split = split_clusters(standardized_path, spike_index_triaged, ordered_labels, triaged_x, triaged_z, triaged_maxptps, geom_array, denoiser, device)

print(len(np.unique(labels_split)))
    
temp_clusterer = get_templates(standardized_path, geom_array, labels_split.max()+1, spike_index_triaged, labels_split)

list_argmin = np.zeros(temp_clusterer.shape[0])
for i in range(temp_clusterer.shape[0]):
    list_argmin[i] = temp_clusterer[i, :, temp_clusterer[i].ptp(0).argmax()].argmin()

idx_not_aligned = np.where(list_argmin!=60)[0]

for unit in idx_not_aligned:
    mc = temp_clusterer[unit].ptp(0).argmax()
    offset = temp_clusterer[unit, :, mc].argmin()
    spike_index_triaged[labels_split == unit, 0] += offset-60

idx_sorted = spike_index_triaged[:, 0].argsort()
spike_index_triaged = spike_index_triaged[idx_sorted]
labels_split = labels_split[idx_sorted]
triaged_x = triaged_x[idx_sorted]
triaged_z = triaged_z[idx_sorted]
triaged_maxptps = triaged_maxptps[idx_sorted]

z_labels = np.zeros(labels_split.max()+1)
for i in range(labels_split.max()+1):
    z_labels[i] = triaged_z[labels_split == i].mean()
    
ordered_split_labels = labels_split.copy()
z_argsort = z_labels.argsort()[::-1]
for i in range(labels_split.max()+1):
    ordered_split_labels[labels_split == z_argsort[i]] = i

labels_merged = get_merged(standardized_path, geom_array, ordered_split_labels.max()+1, spike_index_triaged, ordered_split_labels, triaged_x, triaged_z, denoiser, device, n_temp = 20, distance_threshold = 2.5, threshold_diptest = 1)

labels_corrected_merged = np.zeros(labels_merged.shape)
cmp = -1
for i in np.unique(labels_merged):
    labels_corrected_merged[labels_merged == i] = cmp
    cmp += 1
labels_corrected_merged = labels_corrected_merged.astype('int')

z_labels = np.zeros(labels_corrected_merged.max()+1)
for i in range(labels_corrected_merged.max()+1):
    z_labels[i] = triaged_z[labels_corrected_merged == i].mean()
    
ordered_merged_labels = labels_corrected_merged.copy()
z_argsort = z_labels.argsort()[::-1]
for i in range(labels_corrected_merged.max()+1):
    ordered_merged_labels[labels_corrected_merged == z_argsort[i]] = i
    
print(len(np.unique(ordered_merged_labels)))


# print("VISUALIZATION")

#Uncomment for visualizing clusterer output
unique_colors = ['#e6194b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3', '#808000', '#000075', '#000000']
cluster_color_dict = {}
for cluster_id in np.unique(ordered_merged_labels):
    cluster_color_dict[cluster_id] = unique_colors[cluster_id % len(unique_colors)]
cluster_color_dict[-1] = '#808080' #set outlier color to grey
fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 12))
z_cutoff = (-50,3900)
min_cluster_size = 25
min_samples = 25
# matplotlib.rcParams.update({'font.size': 12})
xs, zs, ids = triaged_x, triaged_z, ordered_merged_labels
axes[0].set_ylim(z_cutoff)
cluster_scatter(xs, zs, ids, ax=axes[0], excluded_ids=set([-1]), s=20, alpha=.05, color_dict=cluster_color_dict)
axes[0].scatter(geom_array[:, 0], geom_array[:, 1], s=20, c='orange', marker = "s")
axes[0].set_title(f"min_cluster_size {min_cluster_size}, min_samples {min_samples}");
axes[0].set_ylabel("z");
axes[0].set_xlabel("x");
ys, zs, ids = triaged_maxptps, triaged_z, ordered_merged_labels
axes[1].set_ylim(z_cutoff)
cluster_scatter(ys, zs, ids, ax=axes[1], excluded_ids=set([-1]), s=20, alpha=.05, color_dict=cluster_color_dict)
axes[1].set_title(f"min_cluster_size {min_cluster_size}, min_samples {min_samples}");
axes[1].set_xlabel("scaled ptp");
axes[2].scatter(xs, zs, s=20, c=color_arr, alpha=.1)
axes[2].scatter(geom_array[:, 0], geom_array[:, 1], s=20, c='orange', marker = "s")
axes[2].set_ylim(z_cutoff)
axes[2].set_title("ptps");
fig.savefig('split_merged_clustering.png')
plt.close(fig)
# print("Creating Figures")
# #load kilosort results
# recording_duration = 60
# kilo_spike_samples = np.load('/media/cat/data/CSH_ZAD_026_1800_1860/kilosort_spk_samples.npy')
# kilo_spike_frames = (kilo_spike_samples - 30*recording_duration*30000) + 18 #+18 to match our detection alignment
# kilo_spike_clusters = np.load('/media/cat/data/CSH_ZAD_026_1800_1860/kilsort_spk_clusters.npy')
# kilo_spike_depths = np.load('/media/cat/data/CSH_ZAD_026_1800_1860/kilsort_spk_depths.npy')
# kilo_cluster_depth_means = {}
# for cluster_id in np.unique(kilo_spike_clusters):
#     kilo_cluster_depth_means[cluster_id] = np.mean(kilo_spike_depths[kilo_spike_clusters==cluster_id])
# #create kilosort SpikeInterface sorting
# times_list = []
# labels_list = []
# for cluster_id in np.unique(kilo_spike_clusters):
#     spike_train_kilo = kilo_spike_frames[np.where(kilo_spike_clusters==cluster_id)]
#     times_list.append(spike_train_kilo)
#     labels_list.append(np.zeros(spike_train_kilo.shape[0])+cluster_id)
# times_array = np.concatenate(times_list).astype('int')
# labels_array = np.concatenate(labels_list).astype('int')
# sorting_kilo = spikeinterface.numpyextractors.NumpySorting.from_times_labels(times_list=times_array, 
#                                                                              labels_list=labels_array, 
#                                                                                   sampling_frequency=30000)
# #create hdbscan/localization SpikeInterface sorting (with triage)
# times_list = []
# labels_list = []
# for cluster_id in np.unique(ordered_merged_labels):
#     spike_train_hdbl_t = spike_index_triaged[:,0][np.where(ordered_merged_labels==cluster_id)]+18
#     times_list.append(spike_train_hdbl_t)
#     labels_list.append(np.zeros(spike_train_hdbl_t.shape[0])+cluster_id)
# times_array = np.concatenate(times_list).astype('int')
# labels_array = np.concatenate(labels_list).astype('int')
# sorting_hdbl_t = spikeinterface.numpyextractors.NumpySorting.from_times_labels(times_list=times_array, 
#                                                                                 labels_list=labels_array, 
#                                                                                 sampling_frequency=30000)
# cmp_5 = compare_two_sorters(sorting_hdbl_t, sorting_kilo, sorting1_name='Ours_Split_Merged', sorting2_name='kilosort', match_score=.5)
# matched_units_5 = cmp_5.get_matching()[0].index.to_numpy()[np.where(cmp_5.get_matching()[0] != -1.)]
# matches_kilos_5 = cmp_5.get_best_unit_match1(matched_units_5).values.astype('int')
# cmp_1 = compare_two_sorters(sorting_hdbl_t, sorting_kilo, sorting1_name='Ours_Split_Merged', sorting2_name='kilosort', match_score=.1)
# matched_units_1 = cmp_1.get_matching()[0].index.to_numpy()[np.where(cmp_1.get_matching()[0] != -1.)]
# unmatched_units_1 = cmp_1.get_matching()[0].index.to_numpy()[np.where(cmp_1.get_matching()[0] == -1.)]
# matches_kilos_1 = cmp_1.get_best_unit_match1(matched_units_1).values.astype('int')
# cluster_centers = get_x_z_templates(ordered_merged_labels.max()+1, ordered_merged_labels, triaged_x, triaged_z)
# triaged_mcs_abs = locs_triaged[:, 6].astype('int')
# triaged_firstchans = locs_triaged[:, 5].astype('int')
# num_rows_plot=3
# num_spikes_plot = 100
# num_channels = 40
# wfs_localized = np.load('/media/cat/data/CSH_ZAD_026_1800_1860/denoised_wfs.npy')
# wfs_subtracted = np.load('/media/cat/data/CSH_ZAD_026_1800_1860/subtracted_wfs.npy')
# raw_data_bin = '/media/cat/data/CSH_ZAD_026_1800_1860/1min_standardized.bin'
# residual_data_bin = '/media/cat/data/CSH_ZAD_026_1800_1860/residual_1min_standardized_t_0_None.bin'
# num_channels_similarity = 10
# num_close_clusters = 10
# num_close_clusters_plot = 5
# shifts_align=np.arange(-8,9)
# templates = get_templates(standardized_path, sorting_hdbl_t.get_unit_ids().max()+1, spike_index_triaged, ordered_merged_labels)
# dist_argsort, dist_template = get_proposed_pairs(sorting_hdbl_t.get_unit_ids().max()+1, templates, cluster_centers, n_temp = 10)
# save_dir_path = "oversplit_cluster_summaries_ours_merged"
# if not os.path.exists(save_dir_path):
#     os.makedirs(save_dir_path)
# for cluster_id in range(ordered_merged_labels.max()+1):
#     print(cluster_id)
# # cluster_id = 4
#     if cluster_id in matched_units_5:
#         cmp = cmp_5
#         print(">50% match")
#     elif cluster_id in matched_units_1:
#         cmp = cmp_1
#         print("50%> and >10% match")
#     else:
#         cmp = None
#         print("<10% match")
#     plot cluster summary
#     fig = plot_single_unit_summary(cluster_id, ordered_merged_labels, cluster_centers, geom_array, 50, num_rows_plot, triaged_x, triaged_z, triaged_maxptps, 
#                                    triaged_firstchans, triaged_mcs_abs, spike_index_triaged+18, non_triaged_idxs, wfs_localized, wfs_subtracted, cluster_color_dict, 
#                                    color_arr, raw_data_bin, residual_data_bin)
#     plt.close(fig)
#     fig.savefig('individual_units_plots/unit_'+str(cluster_id)+'.png', dpi = 100)
#     if cmp is not None:
#         num_channels = wfs_localized.shape[2]
#         cluster_id_match = cmp.get_best_unit_match1(cluster_id)
#         sorting1 = sorting_hdbl_t
#         sorting2 = sorting_kilo
#         sorting1_name = "hdb"
#         sorting2_name = "kilo"
#         firstchans_cluster_sorting1 = triaged_firstchans[ordered_merged_labels == cluster_id]
#         mcs_abs_cluster_sorting1 = triaged_mcs_abs[ordered_merged_labels == cluster_id]
#         spike_depths = kilo_spike_depths[np.where(kilo_spike_clusters==cluster_id_match)]
#         mcs_abs_cluster_sorting2 = np.asarray([np.argmin(np.abs(spike_depth - geom_array[:,1])) for spike_depth in spike_depths])
#         firstchans_cluster_sorting2 = (mcs_abs_cluster_sorting2 - 20).clip(min=0)
#         fig = plot_agreement_venn(cluster_id, cluster_id_match, cmp, sorting1, sorting2, sorting1_name, sorting2_name, geom_array, num_channels, num_spikes_plot, firstchans_cluster_sorting1, mcs_abs_cluster_sorting1, 
#                             firstchans_cluster_sorting2, mcs_abs_cluster_sorting2, raw_data_bin, delta_frames = 12)
#         fig.savefig('kilosort_agreement_plots/unit_'+str(cluster_id)+'.png', dpi = 100)
#         plt.close(fig)
#     st_1 = sorting_hdbl_t.get_unit_spike_train(cluster_id)
#     #compute K closest clsuters
#     dist_template_unit = dist_template[cluster_id]
#     closest_clusters = dist_argsort[cluster_id]
#     idx = dist_template_unit.argsort()
#     dist_template_unit = dist_template_unit[idx]
#     closest_clusters = closest_clusters[idx]
#     if dist_template_unit[0] < 2.0: #arbitrary..
#         fig = plot_unit_similarities(cluster_id, closest_clusters, sorting_hdbl_t, sorting_hdbl_t, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
#                                      num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="both")
#         plt.close(fig)
#         fig.savefig(save_dir_path + f"/cluster_{cluster_id}_summary.png")


















