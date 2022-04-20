# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
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

# %%
# %load_ext autoreload
# %autoreload 2

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
import spikeinterface 
from spikeinterface.toolkit import compute_correlograms
from spikeinterface.comparison import compare_two_sorters
from spikeinterface.widgets import plot_agreement_matrix
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib_venn import venn3, venn3_circles, venn2
import matplotlib.gridspec as gridspec
from spike_psvae.merge_split import split_clusters, get_templates, get_merged, align_templates
from spike_psvae.denoise import SingleChanDenoiser
import torch
import torch.multiprocessing as mp
# %matplotlib inline
import pandas
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
import h5py
import pickle
import sklearn
import seaborn as sns
from tqdm import tqdm

#random seed for provenance
np.random.seed(0)

# %%
from spike_psvae.denoise import SingleChanDenoiser
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

denoiser = SingleChanDenoiser()
denoiser.load()
denoiser.to(device)

# %%
geom = 'np1_channel_map.npy'
triage_quantile = 85
do_infer_ptp = False
num_spikes_cluster = None
min_cluster_size = 25
min_samples = 25
num_spikes_plot = 250
num_rows_plot = 3
num_channels = 40
no_verbose = True

data_path = '/media/cat/data/'
data_name = 'CSH_ZAD_026_5min'
data_dir = data_path + data_name + '/'
raw_data_bin = data_dir + 'CSH_ZAD_026_snip.ap.bin'
residual_data_bin = data_dir + 'residual_CSH_ZAD_026_snip.ap_t_0_None.bin'

# %%
filename = data_dir + "subtraction_CSH_ZAD_026_snip.ap_t_0_None.h5"
h5 = h5py.File(filename)
print("Keys: %s" % h5.keys())

# %%
geom_array = np.asarray(list(h5["geom"]))
firstchans = np.asarray(list(h5["first_channels"]))
spike_index = np.asarray(list(h5["spike_index"]))
maxptps = np.asarray(list(h5["maxptps"]))
wfs_localized = h5["cleaned_waveforms"]#np.load(f["cleaned_waveforms"], mmap_mode='r') #np.memmap(data_dir+'denoised_waveforms.npy', dtype='float32', shape=(290025, 121, 40))
wfs_subtracted = h5["subtracted_waveforms"] #np.load(f["subtracted_waveforms"], mmap_mode='r')
x, y, z_abs, alpha = h5["localizations"][:, :4].T
z = np.asarray(list(h5["z_reg"]))
print(h5["end_sample"])
end_sample = h5["end_sample"][()]
start_sample = h5["start_sample"][()]
end_time = end_sample / 30000
start_time = start_sample / 30000
recording_duration = end_time - start_time
print(f"duration of recording: {recording_duration} s")

# %%
#perform triaging
# triaged_x, triaged_y, triaged_z, triaged_alpha, triaged_maxptps, triaged_ae_features, ptp_filter, idx_keep = run_weighted_triage(x, y, z, alpha, maxptps, pcs=ae_features, threshold=75, ptp_threshold=3, ptp_weighting=True) #pcs is None here
triaged_x, triaged_y, triaged_z, triaged_alpha, triaged_maxptps, _, ptp_filter, idx_keep = run_weighted_triage(x, y, z, alpha, maxptps, threshold=80, ptp_threshold=3, ptp_weighting=True) #pcs is None here
# triaged_x, triaged_y, triaged_z, triaged_alpha, triaged_maxptps, _, ptp_filter, idx_keep = run_weighted_triage(x, y, z, alpha, maxptps, threshold=100, ptp_threshold=0, ptp_weighting=False) #pcs is None here
triaged_spike_index = spike_index[ptp_filter][idx_keep]
triaged_mcs_abs = spike_index[:,1][ptp_filter][idx_keep]
non_triaged_idxs = ptp_filter[0][idx_keep]
triaged_firstchans = firstchans[ptp_filter][idx_keep]

mask = np.ones(spike_index[:,1].size, dtype=bool)
mask[ptp_filter[0][idx_keep]] = False
triaged_indices = np.where(mask)[0]
# np.save('triaged_indices', triaged_indices)

# %%
#load kilosort results
kilo_spike_samples = np.load(data_dir + 'kilosort_spk_samples.npy')
kilo_spike_frames = (kilo_spike_samples - 30*recording_duration*30000) #to match our detection alignment
kilo_spike_clusters = np.load(data_dir + 'kilosort_spk_clusters.npy')
kilo_spike_depths = np.load(data_dir + 'kilosort_spk_depths.npy')
kilo_cluster_depth_means = {}
for cluster_id in np.unique(kilo_spike_clusters):
    kilo_cluster_depth_means[cluster_id] = np.mean(kilo_spike_depths[kilo_spike_clusters==cluster_id])

# %%
#create kilosort SpikeInterface sorting
sorting_kilo = make_sorting_from_labels_frames(kilo_spike_clusters, kilo_spike_frames)
    
good_kilo_sort_clusters_all = np.array([  0,  17,  19,  25,  30,  33,  36,  38,  41,  47,  48,  53,  64,
        70,  78,  82,  83,  85,  88,  90,  97, 103, 109, 112, 114, 115,
       117, 119, 120, 131, 132, 133, 141, 142, 153, 158, 169, 172, 185,
       187, 189, 193, 197, 199, 205, 208, 211, 215, 217, 224, 237, 244,
       247, 269, 272, 274, 280, 283, 289, 291, 292, 296, 300, 303, 304,
       308, 309, 320, 328, 331, 336, 341, 349, 350, 380, 382, 386, 400,
       409, 411, 414, 435, 438, 439, 464, 474, 476, 478, 485, 487, 488,
       496, 503, 509, 512, 521, 522, 523, 529, 533, 534, 535, 536, 537,
       539, 544, 545, 547, 548, 551, 552, 555, 557, 570, 583, 596, 598,
       621, 629, 633, 637, 648, 655, 660, 670, 671, 677, 678, 681, 682,
       683, 699, 700, 702, 708, 709])

#remove empty clusters
good_kilo_sort_clusters = []
for good_cluster in good_kilo_sort_clusters_all:
    if good_cluster in sorting_kilo.get_unit_ids():
        good_kilo_sort_clusters.append(good_cluster)
good_kilo_sort_clusters = np.asarray(good_kilo_sort_clusters)

# %%
scales = (1,10,1,15,30) #predefined scales for each feature
features = np.concatenate((np.expand_dims(triaged_x,1), np.expand_dims(triaged_z,1), np.expand_dims(np.log(triaged_maxptps)*scales[4],1)), axis=1)

# %%
#perform hdbscan clustering
min_cluster_size =  min_cluster_size
min_samples = min_samples
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
clusterer.fit(features)
if no_verbose:
    print(clusterer)

#compute cluster centers
cluster_centers = compute_cluster_centers(clusterer)
    
#re-label each cluster by z-depth
clusterer = relabel_by_depth(clusterer, cluster_centers)

#remove duplicate units by spike_times_agreement and ptp
clusterer, duplicate_ids = remove_duplicate_units(clusterer, triaged_spike_index[:,0], triaged_maxptps)

#re-compute cluster centers
cluster_centers = compute_cluster_centers(clusterer)

#re-label each cluster by z-depth
clusterer = relabel_by_depth(clusterer, cluster_centers)

# %%
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
fig = plot_array_scatter(clusterer.labels_, geom_array, triaged_x, triaged_z, triaged_maxptps, cluster_color_dict, color_arr, min_cluster_size=clusterer.min_cluster_size, min_samples=clusterer.min_samples, 
                         z_cutoff=(0, 3900), figsize=(18, 24))
# fig.suptitle(f'x,z,scaled_logptp features," {num_spikes} datapoints');
plt.show()

# %%
####split units####
templates = get_templates(raw_data_bin, geom_array, clusterer.labels_.max()+1, triaged_spike_index, clusterer.labels_)

plt.plot(templates[12]);

#align all templates to 60
triaged_spike_index, idx_sorted = align_templates(clusterer.labels_, templates, triaged_spike_index)

clusterer.labels_ = clusterer.labels_[idx_sorted]
triaged_x = triaged_x[idx_sorted]
triaged_z = triaged_z[idx_sorted]
triaged_maxptps = triaged_maxptps[idx_sorted]
triaged_alpha = triaged_alpha[idx_sorted]
triaged_firstchans = triaged_firstchans[idx_sorted]
triaged_mcs_abs = triaged_mcs_abs[idx_sorted]
non_triaged_idxs = non_triaged_idxs[idx_sorted]

#change clusterer raw data
raw_data = np.concatenate((np.expand_dims(triaged_x,1), np.expand_dims(triaged_z,1), np.expand_dims(np.log(triaged_maxptps)*scales[4],1)), axis=1)
clusterer._raw_data = raw_data

# %%
#split clusters
labels_split = split_clusters(raw_data_bin, triaged_spike_index, clusterer.labels_, triaged_x, triaged_z, triaged_maxptps, geom_array, denoiser, device, n_channels=10)
print(np.unique(labels_split).shape)

# %%
clusterer.labels_ = labels_split

#compute cluster centers
cluster_centers = compute_cluster_centers(clusterer)
    
#re-label each cluster by z-depth
clusterer = relabel_by_depth(clusterer, cluster_centers)

#compute cluster centers
cluster_centers = compute_cluster_centers(clusterer)

# %%
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

# %%
##### plot array scatter #####
fig = plot_array_scatter(clusterer.labels_, geom_array, triaged_x, triaged_z, triaged_maxptps, cluster_color_dict, color_arr, min_cluster_size=clusterer.min_cluster_size, min_samples=clusterer.min_samples, 
                         z_cutoff=(0, 3900), figsize=(18, 24))
# fig.suptitle(f'x,z,scaled_logptp features," {num_spikes} datapoints');
plt.show()

# %%
####merge units####
templates = get_templates(raw_data_bin, geom_array, clusterer.labels_.max()+1, triaged_spike_index, clusterer.labels_)

#align all templates to 60
triaged_spike_index, idx_sorted = align_templates(clusterer.labels_, templates, triaged_spike_index)

clusterer.labels_ = clusterer.labels_[idx_sorted]
triaged_x = triaged_x[idx_sorted]
triaged_z = triaged_z[idx_sorted]
triaged_maxptps = triaged_maxptps[idx_sorted]
triaged_alpha = triaged_alpha[idx_sorted]
triaged_firstchans = triaged_firstchans[idx_sorted]
triaged_mcs_abs = triaged_mcs_abs[idx_sorted]
non_triaged_idxs = non_triaged_idxs[idx_sorted]

#change clusterer raw data
raw_data = np.concatenate((np.expand_dims(triaged_x,1), np.expand_dims(triaged_z,1), np.expand_dims(np.log(triaged_maxptps)*scales[4],1)), axis=1)
clusterer._raw_data = raw_data

labels_merged = get_merged(raw_data_bin, geom_array, clusterer.labels_.max()+1, triaged_spike_index, clusterer.labels_, triaged_x, triaged_z, denoiser, device, n_channels=10, n_temp = 20, distance_threshold = 2.5, threshold_diptest = 1.25)
print(np.unique(labels_merged).shape)
clusterer.labels_ = labels_merged

#compute cluster centers
cluster_centers = compute_cluster_centers(clusterer)
    
#re-label each cluster by z-depth
clusterer = relabel_by_depth(clusterer, cluster_centers)

#compute cluster centers
cluster_centers = compute_cluster_centers(clusterer)

# %%
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

# %%
##### plot array scatter #####
fig = plot_array_scatter(clusterer.labels_, geom_array, triaged_x, triaged_z, triaged_maxptps, cluster_color_dict, color_arr, min_cluster_size=clusterer.min_cluster_size, min_samples=clusterer.min_samples, 
                         z_cutoff=(0, 3900), figsize=(18, 24))
# fig.suptitle(f'x,z,scaled_logptp features," {num_spikes} datapoints');
plt.show()

# %%
cluster_id = 260
num_spikes_plot=50
#plot cluster summary
fig = plot_single_unit_summary(cluster_id, clusterer.labels_, cluster_centers, geom_array, num_spikes_plot, num_rows_plot, triaged_x, triaged_z, triaged_maxptps, 
                               triaged_firstchans, triaged_mcs_abs, triaged_spike_index[:,0], non_triaged_idxs, wfs_localized, wfs_subtracted, cluster_color_dict, 
                               color_arr, raw_data_bin, residual_data_bin)

# %%
#create kilosort SpikeInterface sorting
sorting_kilo = make_sorting_from_labels_frames(kilo_spike_clusters, kilo_spike_frames)

#create hdbscan/localization SpikeInterface sorting (with triage)
sorting_hdbl_t = make_sorting_from_labels_frames(clusterer.labels_, triaged_spike_index[:,0])

cmp_5 = compare_two_sorters(sorting_hdbl_t, sorting_kilo, sorting1_name='ours', sorting2_name='kilosort', match_score=.5)
matched_units_5 = cmp_5.get_matching()[0].index.to_numpy()[np.where(cmp_5.get_matching()[0] != -1.)]
matches_kilos_5 = cmp_5.get_best_unit_match1(matched_units_5).values.astype('int')

cmp_1 = compare_two_sorters(sorting_hdbl_t, sorting_kilo, sorting1_name='ours', sorting2_name='kilosort', match_score=.1)
matched_units_1 = cmp_1.get_matching()[0].index.to_numpy()[np.where(cmp_1.get_matching()[0] != -1.)]
unmatched_units_1 = cmp_1.get_matching()[0].index.to_numpy()[np.where(cmp_1.get_matching()[0] == -1.)]
matches_kilos_1 = cmp_1.get_best_unit_match1(matched_units_1).values.astype('int')

# %%
good_kilo_sort_clusters_all = np.array([  0,  17,  19,  25,  30,  33,  36,  38,  41,  47,  48,  53,  64,
        70,  78,  82,  83,  85,  88,  90,  97, 103, 109, 112, 114, 115,
       117, 119, 120, 131, 132, 133, 141, 142, 153, 158, 169, 172, 185,
       187, 189, 193, 197, 199, 205, 208, 211, 215, 217, 224, 237, 244,
       247, 269, 272, 274, 280, 283, 289, 291, 292, 296, 300, 303, 304,
       308, 309, 320, 328, 331, 336, 341, 349, 350, 380, 382, 386, 400,
       409, 411, 414, 435, 438, 439, 464, 474, 476, 478, 485, 487, 488,
       496, 503, 509, 512, 521, 522, 523, 529, 533, 534, 535, 536, 537,
       539, 544, 545, 547, 548, 551, 552, 555, 557, 570, 583, 596, 598,
       621, 629, 633, 637, 648, 655, 660, 670, 671, 677, 678, 681, 682,
       683, 699, 700, 702, 708, 709])

#remove empty clusters
good_kilo_sort_clusters = []
for good_cluster in good_kilo_sort_clusters_all:
    if good_cluster in sorting_kilo.get_unit_ids():
        good_kilo_sort_clusters.append(good_cluster)
good_kilo_sort_clusters = np.asarray(good_kilo_sort_clusters)

# %%
cmp_kilo_5 = compare_two_sorters(sorting_kilo, sorting_hdbl_t, sorting1_name='kilosort', sorting2_name='ours', match_score=.5)
matched_units_kilo_5 = cmp_kilo_5.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_5.get_matching()[0] != -1.)]
unmatched_units_kilo_5 = cmp_kilo_5.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_5.get_matching()[0] == -1.)]

cmp_kilo_1 = compare_two_sorters(sorting_kilo, sorting_hdbl_t, sorting1_name='kilosort', sorting2_name='ours', match_score=.1)
matched_units_kilo_1 = cmp_kilo_1.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_1.get_matching()[0].to_numpy() != -1.)]
unmatched_units_kilo_1 = cmp_kilo_1.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_1.get_matching()[0].to_numpy() == -1.)]

# %% tags=[]
###hdbscan
save_dir_path = "good_unit_kilo_comparison"
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)

for good_kilo_sort_cluster in good_kilo_sort_clusters:
    cluster_id_match = good_kilo_sort_cluster
    cluster_id = int(cmp_kilo_1.get_best_unit_match1(cluster_id_match))
    depth = int(kilo_cluster_depth_means[cluster_id_match])
    save_str = str(depth).zfill(4)
    if cluster_id != -1:
        sorting1 = sorting_hdbl_t
        sorting2 = sorting_kilo
        sorting1_name = "hdb"
        sorting2_name = "kilo"
        firstchans_cluster_sorting1 = triaged_firstchans[clusterer.labels_ == cluster_id]
        mcs_abs_cluster_sorting1 = triaged_mcs_abs[clusterer.labels_ == cluster_id]
        spike_depths = kilo_spike_depths[np.where(kilo_spike_clusters==cluster_id_match)]
        mcs_abs_cluster_sorting2 = np.asarray([np.argmin(np.abs(spike_depth - geom_array[:,1])) for spike_depth in spike_depths])
        firstchans_cluster_sorting2 = (mcs_abs_cluster_sorting2 - 20).clip(min=0)

        fig = plot_agreement_venn(cluster_id, cluster_id_match, cmp_1, sorting1, sorting2, sorting1_name, sorting2_name, geom_array, num_channels, num_spikes_plot, firstchans_cluster_sorting1, mcs_abs_cluster_sorting1, 
                                  firstchans_cluster_sorting2, mcs_abs_cluster_sorting2, raw_data_bin, delta_frames = 12, alpha=.2)
        plt.close(fig)
        fig.savefig(save_dir_path + f"/Z{save_str}_{cluster_id_match}_{cluster_id}_comparison.png")
    else:
        num_spikes = len(sorting_kilo.get_unit_spike_train(cluster_id_match))
        print(f"skipped {cluster_id_match} with {num_spikes} spikes")
        if num_spikes > 0:
            #plot specific kilosort example
            num_close_clusters = 50
            num_close_clusters_plot=10
            num_channels_similarity = 20
            shifts_align=np.arange(-8,9)

            st_1 = sorting_kilo.get_unit_spike_train(cluster_id_match)

            #compute K closest hdbscan clsuters
            closest_clusters = get_closest_clusters_kilosort_hdbscan(cluster_id_match, kilo_cluster_depth_means, cluster_centers, num_close_clusters)

            fig = plot_unit_similarities(cluster_id_match, closest_clusters, sorting_kilo, sorting_hdbl_t, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
                                         num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="both")
            plt.close(fig)
            fig.savefig(save_dir_path + f"/Z{save_str}_{cluster_id_match}_summary.png")

# %%
from spike_psvae import denoise, subtract, localization, ibme, deconvolve, residual

###hdbscan
save_dir_path = "/media/cat/cole/deconv_results"
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)
output_directory = save_dir_path
geom_path = data_dir+geom

result_file_names = deconvolve.deconvolution(triaged_spike_index[np.not_equal(clusterer.labels_, -1)], clusterer.labels_[np.not_equal(clusterer.labels_, -1)], 
                                             output_directory, raw_data_bin, 
                                             geom_path, multi_processing=True, n_processors=6, threshold=40)

deconv_spike_train_up = np.load(result_file_names[1])
deconv_templates_up = np.load(result_file_names[0])

deconv_spike_train = np.load(result_file_names[3])
deconv_templates = np.load(result_file_names[2])


n_spikes = deconv_spike_train.shape[0]
print(f'number of deconv spikes: {n_spikes}')
print(f'deconv templates shape: {deconv_templates.shape}')

residual_path = residual.run_residual(result_file_names[0], result_file_names[1],
                                      output_directory, raw_data_bin, geom_path)

# channels to extract for each mc
extract_channel_index = []
for c in range(384):
    low = max(0, c - 40 // 2)
    low = min(384 - 40, low)
    extract_channel_index.append(
        np.arange(low, low + 40)
    )
extract_channel_index = np.array(extract_channel_index)

from sklearn.decomposition import PCA
# load tPCA
h5_subtract = data_dir + "subtraction_1min_standardized_t_0_None.h5"
with h5py.File(h5_subtract, "r") as f:
    tpca_components = f['tpca_components'][:]
    tpca_mean = f['tpca_mean'][:]

tpca = PCA(8)
tpca.components_ = tpca_components
tpca.mean_ = tpca_mean


# %%
def temporal_align(waveforms, maxchans, offset=42):
    N, T, C = waveforms.shape
    offsets = waveforms[np.arange(N), :, maxchans].argmin(1)
    rolls = offset - offsets
    out = np.empty_like(waveforms)
    pads = [(0, 0), (0, 0)]
    for i, roll in enumerate(rolls):
        if roll > 0:
            pads[0] = (roll, 0)
            start, end = 0, T
        elif roll < 0:
            pads[0] = (0, -roll)
            start, end = -roll, T - roll
        else:
            out[i] = waveforms[i]
            continue

        pwf = np.pad(waveforms[i], pads, mode="linear_ramp")
        out[i] = pwf[start:end, :]

    return out


# %%
subtracted_waveforms_dir = os.path.join(output_directory, 'subtracted_waveforms')
if not os.path.exists(subtracted_waveforms_dir):
    os.makedirs(subtracted_waveforms_dir)
    
collision_subtracted_waveforms_dir = os.path.join(output_directory, 'collision_subtracted_waveforms')
if not os.path.exists(collision_subtracted_waveforms_dir):
    os.makedirs(collision_subtracted_waveforms_dir)
    
denoised_waveforms_dir = os.path.join(output_directory, 'denoised_waveforms')
if not os.path.exists(denoised_waveforms_dir):
    os.makedirs(denoised_waveforms_dir)
    
geom_array = np.load(geom_path)

# %%
batch_id = 0
batch_size=1024

skipped_count = 0

deconv_spike_index = np.zeros((n_spikes,2)).astype(int)
deconv_labels = np.zeros(n_spikes).astype(int)
for start in tqdm(range(0, n_spikes, batch_size)):
    end = start + batch_size
    
    batch_deconv_spike_train_up = deconv_spike_train_up[start:end]
    batch_t, batch_template_idx = batch_deconv_spike_train_up[:,0], batch_deconv_spike_train_up[:,1]
    batch_subtracted_wfs = deconv_templates_up[batch_template_idx]
    batch_mcs = batch_subtracted_wfs.ptp(1).argmax(1)
    batch_extract_channel_index = extract_channel_index[batch_mcs]

    batch_subtracted_wfs = np.array(list(map(lambda x, idx: x[:,idx], 
                                             batch_subtracted_wfs, 
                                             batch_extract_channel_index)))
    
    deconv_spike_index[start:end,0] = batch_t
    deconv_spike_index[start:end,1] = batch_mcs
    deconv_labels[start:end] = batch_template_idx
    
    # load residual batch
    residual_batch, skipped_idx = deconvolve.read_waveforms(batch_t, residual_path, geom_array)
    residual_batch = np.array(list(map(lambda x, idx: x[:,idx], 
                                         residual_batch, 
                                         batch_extract_channel_index)))
    kept_idx = np.ones(batch_subtracted_wfs.shape[0]).astype(bool)
    kept_idx[skipped_idx] = False
    skipped_count += len(skipped_idx)
    
    batch_collision_subtracted_wfs = batch_subtracted_wfs[kept_idx] + residual_batch
    
    relative_batch_mcs = np.where(batch_extract_channel_index-batch_mcs[:,None]==0)[1]
    aligned_wfs = temporal_align(batch_collision_subtracted_wfs, relative_batch_mcs)

    batch_denoised_wfs = subtract.full_denoising(aligned_wfs, 
                                       batch_mcs,
                                       extract_channel_index,
                                       None,
                                       probe='np1',
                                       tpca=tpca,
                                       device=device,
                                       denoiser=denoiser,
                                       )
    
    np.save(os.path.join(subtracted_waveforms_dir,f'subtracted_{str(batch_id).zfill(6)}.npy'), 
            batch_subtracted_wfs.astype(np.float32))
    np.save(os.path.join(collision_subtracted_waveforms_dir,
                         f'collision_subtracted_{str(batch_id).zfill(6)}.npy'), 
            batch_collision_subtracted_wfs.astype(np.float32))
    np.save(os.path.join(denoised_waveforms_dir,f'denoised_{str(batch_id).zfill(6)}.npy'), 
            batch_denoised_wfs.astype(np.float32))
    batch_id += 1
print(f'number of spikes skipped: {skipped_count}')

# %%
from pathlib import Path
def merge_files_h5(filtered_location, output_h5, dataset_name, shape, delete=False):
    with h5py.File(output_h5, "w") as out:
        wfs = out.create_dataset(dataset_name, shape=shape, dtype=np.float32)
        filenames = os.listdir(filtered_location)
        filenames_sorted = sorted(filenames)
        i = 0
        for fname in tqdm(filenames_sorted):
            if '.ipynb' in fname or '.bin' in fname:
                continue
            res = np.load(os.path.join(filtered_location, fname)).astype('float32')
            n_new = res.shape[0]
            wfs[i:i+n_new] = res
            i += n_new
            
            if delete:
                Path(os.path.join(filtered_location, fname)).unlink()
                
# save deconv spike index and labels
np.save(os.path.join(output_directory, 'spike_index.npy'), deconv_spike_index)
np.save(os.path.join(output_directory, 'spike_labels.npy'), deconv_labels)

shape = (n_spikes-skipped_count, 121, 40)
merge_files_h5(subtracted_waveforms_dir, 
               os.path.join(output_directory,'subtracted_wfs.h5'), 'wfs', shape, delete=True)
merge_files_h5(collision_subtracted_waveforms_dir, 
               os.path.join(output_directory,'collision_subtracted_wfs.h5'), 'wfs', shape, delete=True)
merge_files_h5(denoised_waveforms_dir, 
               os.path.join(output_directory,'denoised_wfs.h5'), 'wfs', shape, delete=True)

h5 = h5py.File(os.path.join(output_directory,'denoised_wfs.h5'))
denoised_wfs = h5["wfs"]
n_spikes = denoised_wfs.shape[0]

deconv_spike_index = np.load(os.path.join(output_directory, 'spike_index.npy'))
assert deconv_spike_index.shape[0] == n_spikes
print(f'number of deconv spikes: {n_spikes}')

# %%
h5_subtract = h5py.File(os.path.join(output_directory,'collision_subtracted_wfs.h5'))
subtracted_wfs = h5_subtract["wfs"]
n_spikes = subtracted_wfs.shape[0]

# %%
n_workers=8
batch_size=16384
times = deconv_spike_train_up[:,0].copy()/30000
xss = []
yss = []
z_relss = []
z_absss = []
alphass = []
maxptpss = []
mcss = []
fcss = []
for start in tqdm(range(0, n_spikes, batch_size)):
    end = start+batch_size
    ptps = denoised_wfs[start:end].copy().ptp(1)
    batch_mcs = deconv_spike_index[start:end,1].copy()
    batch_fcs = extract_channel_index[batch_mcs][:,0]
    xs, ys, z_rels, z_abss, alphas = localization.localize_ptps(ptps, geom_array, batch_fcs, 
                                                                batch_mcs, n_workers=n_workers)
    xss.append(xs)
    yss.append(ys)
    z_relss.append(z_rels)
    z_absss.append(z_abss)
    alphass.append(alphas)
    maxptpss.append(ptps.max(1))
    mcss.append(batch_mcs)
    fcss.append(batch_fcs)
    
xss = np.concatenate(xss)
yss = np.concatenate(yss)
z_relss = np.concatenate(z_relss)
z_absss = np.concatenate(z_absss)
alphass = np.concatenate(alphass)
maxptpss = np.concatenate(maxptpss)
mcss = np.concatenate(mcss)
fcss = np.concatenate(fcss)

# %%
deconv_labels_original = deconv_spike_train[:,1]
deconv_frames = deconv_spike_train[:,0]

#create hdbscan/localization SpikeInterface sorting (with triage)
sorting_hdbl_t_deconv = make_sorting_from_labels_frames(deconv_labels_original, deconv_frames)

cmp_5_deconv = compare_two_sorters(sorting_hdbl_t_deconv, sorting_kilo, sorting1_name='ours', sorting2_name='kilosort', match_score=.5)
matched_units_5_deconv = cmp_5_deconv.get_matching()[0].index.to_numpy()[np.where(cmp_5_deconv.get_matching()[0] != -1.)]
matches_kilos_5_deconv = cmp_5_deconv.get_best_unit_match1(matched_units_5_deconv).values.astype('int')

cmp_1_deconv = compare_two_sorters(sorting_hdbl_t_deconv, sorting_kilo, sorting1_name='ours', sorting2_name='kilosort', match_score=.1)
matched_units_1_deconv = cmp_1_deconv.get_matching()[0].index.to_numpy()[np.where(cmp_1_deconv.get_matching()[0] != -1.)]
unmatched_units_1_deconv = cmp_1_deconv.get_matching()[0].index.to_numpy()[np.where(cmp_1_deconv.get_matching()[0] == -1.)]
matches_kilos_1_deconv = cmp_1_deconv.get_best_unit_match1(matched_units_1_deconv).values.astype('int')

cmp_kilo_5_deconv = compare_two_sorters(sorting_kilo, sorting_hdbl_t_deconv, sorting1_name='kilosort', sorting2_name='ours', match_score=.5)
matched_units_kilo_5_deconv = cmp_kilo_5_deconv.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_5_deconv.get_matching()[0] != -1.)]
unmatched_units_kilo_5_deconv = cmp_kilo_5_deconv.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_5_deconv.get_matching()[0] == -1.)]

cmp_kilo_1_deconv = compare_two_sorters(sorting_kilo, sorting_hdbl_t_deconv, sorting1_name='kilosort', sorting2_name='ours', match_score=.1)
matched_units_kilo_1_deconv = cmp_kilo_1_deconv.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_1_deconv.get_matching()[0].to_numpy() != -1.)]
unmatched_units_kilo_1_deconv = cmp_kilo_1_deconv.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_1_deconv.get_matching()[0].to_numpy() == -1.)]

# %%
cmp_kilo_1_deconv.get_agreement_fraction(17, 276.0)

# %%
vir = cm.get_cmap('viridis')
triaged_log_ptp = maxptpss.copy()
triaged_log_ptp[triaged_log_ptp >= 27.5] = 27.5
triaged_log_ptp = np.log(triaged_log_ptp+1)
triaged_log_ptp[triaged_log_ptp<=1.25] = 1.25
triaged_ptp_rescaled = (triaged_log_ptp - triaged_log_ptp.min())/(triaged_log_ptp.max() - triaged_log_ptp.min())
color_arr = vir(triaged_ptp_rescaled)
color_arr[:, 3] = triaged_ptp_rescaled

# %% tags=[]
###hdbscan
save_dir_path = "good_unit_kilo_comparison_deconv"
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)

for good_kilo_sort_cluster in good_kilo_sort_clusters:
    cluster_id_match = good_kilo_sort_cluster
    cluster_id = int(cmp_kilo_1_deconv.get_best_unit_match1(cluster_id_match))
    depth = int(kilo_cluster_depth_means[cluster_id_match])
    save_str = str(depth).zfill(4)
    if cluster_id != -1:
        sorting1 = sorting_hdbl_t_deconv
        sorting2 = sorting_kilo
        sorting1_name = "hdb"
        sorting2_name = "kilo"
        firstchans_cluster_sorting1 = fcss[deconv_labels_original == cluster_id]
        mcs_abs_cluster_sorting1 = mcss[deconv_labels_original == cluster_id]
        spike_depths = kilo_spike_depths[np.where(kilo_spike_clusters==cluster_id_match)]
        mcs_abs_cluster_sorting2 = np.asarray([np.argmin(np.abs(spike_depth - geom_array[:,1])) for spike_depth in spike_depths])
        firstchans_cluster_sorting2 = (mcs_abs_cluster_sorting2 - 20).clip(min=0)

        fig = plot_agreement_venn(cluster_id, cluster_id_match, cmp_1_deconv, sorting1, sorting2, sorting1_name, sorting2_name, geom_array, num_channels, num_spikes_plot, firstchans_cluster_sorting1, mcs_abs_cluster_sorting1, 
                                  firstchans_cluster_sorting2, mcs_abs_cluster_sorting2, raw_data_bin, delta_frames = 12, alpha=.2)
        plt.close(fig)
        fig.savefig(save_dir_path + f"/Z{save_str}_{cluster_id_match}_{cluster_id}_comparison.png")
        
#         fig = plot_single_unit_summary(cluster_id, deconv_labels_original, cluster_centers, geom_array, num_spikes_plot, num_rows_plot, xss, z_absss, maxptpss, 
#                                fcss, mcss, deconv_frames, np.arange(deconv_frames.shape[0]), denoised_wfs, subtracted_wfs, cluster_color_dict, 
#                                color_arr, raw_data_bin, residual_data_bin)
#         plt.close(fig)
#         fig.savefig(save_dir_path + f"/Z{save_str}_hdb{cluster_id}_full_summary.png")
        
    num_spikes = len(sorting_kilo.get_unit_spike_train(cluster_id_match))
    if num_spikes > 0:
        #plot specific kilosort example
        num_close_clusters = 50
        num_close_clusters_plot=10
        num_channels_similarity = 20
        shifts_align=np.arange(-8,9)

        st_1 = sorting_kilo.get_unit_spike_train(cluster_id_match)

        #compute K closest hdbscan clsuters
        closest_clusters = get_closest_clusters_kilosort_hdbscan(cluster_id_match, kilo_cluster_depth_means, cluster_centers, num_close_clusters)

        fig = plot_unit_similarities(cluster_id_match, closest_clusters, sorting_kilo, sorting_hdbl_t_deconv, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
                                     num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="first")
        plt.close(fig)
        fig.savefig(save_dir_path + f"/Z{save_str}_{cluster_id_match}_summary_similarity.png")
        
        fig = plot_unit_similarities(cluster_id_match, closest_clusters, sorting_kilo, sorting_hdbl_t_deconv, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
                                     num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='agreement', normalize_agreement_by="first")
        plt.close(fig)
        fig.savefig(save_dir_path + f"/Z{save_str}_{cluster_id_match}_summary_agreement.png")

# %%
cluster_id

# %%
templates.shape

# %%
plt.plot(templates[275].T[35])

# %%
plt.figure(figsize=(18,6))
plt.plot(templates[275,:,30].T)

# %%
plt.scatter(z_absss[deconv_labels_original==275], maxptpss[deconv_labels_original==275])

# %%
fig = plot_single_unit_summary(cluster_id, deconv_labels_original, cluster_centers, geom_array, num_spikes_plot, num_rows_plot, xss, z_absss, maxptpss, 
                               fcss, mcss, deconv_frames, np.arange(deconv_frames.shape[0]), denoised_wfs, subtracted_wfs, cluster_color_dict, 
                               color_arr, raw_data_bin, residual_data_bin)

# %%
closest_clusters

# %%
sorting_hdbl_t.get_unit_ids()

# %%
cluster_centers

# %%
curr_cluster_depth = kilo_cluster_depth_means[cluster_id]
closest_cluster_indices = np.argsort(np.abs(cluster_centers.iloc[:,1].to_numpy() - kilo_cluster_depth_means[cluster_id]))[:num_close_clusters]
closest_clusters = cluster_centers.index[closest_cluster_indices]

# %%
#plot specific kilosort example
cluster_id_match = 17
num_close_clusters = 100
num_close_clusters_plot=10
num_channels_similarity = 20
shifts_align=np.arange(-8,9)

st_1 = sorting_kilo.get_unit_spike_train(cluster_id_match)

#compute K closest hdbscan clsuters
closest_clusters = get_closest_clusters_kilosort_hdbscan(cluster_id_match, kilo_cluster_depth_means, cluster_centers, num_close_clusters)

fig = plot_unit_similarities(cluster_id_match, closest_clusters, sorting_kilo, sorting_hdbl_t_deconv, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
                             num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='agreement', normalize_agreement_by="both")

# %%

# %%
# cluster_id = 349
# cluster_id_match = 25
# sorting1 = sorting_hdbl_t_deconv
# sorting2 = sorting_kilo
# sorting1_name = "hdb_deconv"
# sorting2_name = "kilo"
# firstchans_cluster_sorting1 = fcss[deconv_labels_original == cluster_id]
# mcs_abs_cluster_sorting1 = mcss[deconv_labels_original == cluster_id]
# spike_depths = kilo_spike_depths[np.where(kilo_spike_clusters==cluster_id_match)]
# mcs_abs_cluster_sorting2 = np.asarray([np.argmin(np.abs(spike_depth - geom_array[:,1])) for spike_depth in spike_depths])
# firstchans_cluster_sorting2 = (mcs_abs_cluster_sorting2 - 20).clip(min=0)

# fig = plot_agreement_venn(cluster_id, cluster_id_match, cmp_5_deconv, sorting1, sorting2, sorting1_name, sorting2_name, geom_array, num_channels, num_spikes_plot, firstchans_cluster_sorting1, mcs_abs_cluster_sorting1, 
#                           firstchans_cluster_sorting2, mcs_abs_cluster_sorting2, raw_data_bin, delta_frames = 12, alpha=.2)

# %% tags=[]
# cluster_id = 187

# if cluster_id in matched_units_5:
#     cmp = cmp_5
#     print(">50% match")
# elif cluster_id in matched_units_1:
#     cmp = cmp_1
#     print("50%> and >10% match")
# else:
#     cmp = None
#     print("<10% match")
    
# num_spikes_plot=50
# #plot cluster summary
# fig = plot_single_unit_summary(cluster_id, clusterer.labels_, cluster_centers, geom_array, num_spikes_plot, num_rows_plot, triaged_x, triaged_z, triaged_maxptps, 
#                                triaged_firstchans, triaged_mcs_abs, triaged_spike_index, non_triaged_idxs, wfs_localized, wfs_subtracted, cluster_color_dict, 
#                                color_arr, raw_data_bin, residual_data_bin)
# plt.show()

# # plot agreement with kilosort
# if cmp is not None:
#     num_channels = wfs_localized.shape[2]
#     cluster_id_match = cmp.get_best_unit_match1(cluster_id)
#     sorting1 = sorting_hdbl_t
#     sorting2 = sorting_kilo
#     sorting1_name = "hdb"
#     sorting2_name = "kilo"
#     firstchans_cluster_sorting1 = triaged_firstchans[clusterer.labels_ == cluster_id]
#     mcs_abs_cluster_sorting1 = triaged_mcs_abs[clusterer.labels_ == cluster_id]
#     spike_depths = kilo_spike_depths[np.where(kilo_spike_clusters==cluster_id_match)]
#     mcs_abs_cluster_sorting2 = np.asarray([np.argmin(np.abs(spike_depth - geom_array[:,1])) for spike_depth in spike_depths])
#     firstchans_cluster_sorting2 = (mcs_abs_cluster_sorting2 - 20).clip(min=0)
    
#     plot_agreement_venn(cluster_id, cluster_id_match, cmp, sorting1, sorting2, sorting1_name, sorting2_name, geom_array, num_channels, num_spikes_plot, firstchans_cluster_sorting1, mcs_abs_cluster_sorting1, 
#                         firstchans_cluster_sorting2, mcs_abs_cluster_sorting2, raw_data_bin, delta_frames = 12)

# %% tags=[]
# from joblib import Parallel, delayed
# save_dir_parallel = "parallel_cluster_summary_plots"
# if not os.path.exists(save_dir_parallel):
#     os.makedirs(save_dir_parallel)
    
# num_spikes_plot = 50
    
# def job(cluster_id):
#     fig = plot_single_unit_summary(
#         cluster_id,
#         clusterer.labels_,
#         cluster_centers,
#         geom_array,
#         num_spikes_plot,
#         num_rows_plot,
#         triaged_x,
#         triaged_z,
#         triaged_maxptps,
#         triaged_firstchans,
#         triaged_mcs_abs,
#         triaged_spike_index,
#         non_triaged_idxs,
#         wfs_localized,
#         wfs_subtracted,
#         cluster_color_dict,
#         color_arr,
#         raw_data_bin,
#         residual_data_bin,
#     )
#     save_z_int = int(cluster_centers.loc[cluster_id][1])
#     save_str = str(save_z_int).zfill(4)
#     fig.savefig(save_dir_parallel + f"/Z{save_str}_cluster{cluster_id}.png", transparent=False, pad_inches=0)
#     plt.close(fig)
# with Parallel(
#     12,
# ) as p:
#     unit_ids = list(range(0,24))#np.setdiff1d(np.unique(clusterer.labels_), [-1])
#     for res in p(delayed(job)(u) for u in  tqdm(unit_ids)):
#         pass

# %% [markdown]
# # Oversplit Analysis

# %% tags=[]
# ###Kilosort
# save_dir_path = "oversplit_cluster_summaries_kilosort"
# if not os.path.exists(save_dir_path):
#     os.makedirs(save_dir_path)
    
# num_close_clusters = 50
# num_close_clusters_plot=10
# num_channels_similarity = 20
# num_under_threshold = 0
# num_spikes_plot = 50
# shifts_align=np.arange(-8,9)
# for cluster_id in sorting_kilo.get_unit_ids():
#     st_1 = sorting_kilo.get_unit_spike_train(cluster_id)
    
#     #compute K closest clsuters
#     closest_clusters = get_closest_clusters_kilosort(cluster_id, kilo_cluster_depth_means, num_close_clusters=num_close_clusters)
    
#     #compute unit similarties
#     original_template, closest_clusters, similarities, agreements, templates, shifts = get_unit_similarities(cluster_id, st_1, closest_clusters, sorting_kilo, geom_array, raw_data_bin, 
#                                                                                                              num_channels_similarity=num_channels_similarity, 
#                                                                                                              num_close_clusters=num_close_clusters, shifts_align=shifts_align,
#                                                                                                              order_by ='similarity')
#     if similarities[0] < 2.0: #arbitrary..
#         print(similarities[0], closest_clusters[0])
#         fig = plot_unit_similarities(cluster_id, closest_clusters, sorting_kilo, sorting_kilo, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
#                                      num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="both")
#         plt.close(fig)
#         fig.savefig(save_dir_path + f"/cluster_{cluster_id}_summary.png")

# %%
# ###hdbscan
# save_dir_path = "oversplit_cluster_summaries_hdbscan"
# if not os.path.exists(save_dir_path):
#     os.makedirs(save_dir_path)

# num_close_clusters = 50
# num_close_clusters_plot=10
# num_channels_similarity = 20
# num_under_threshold = 0
# num_spikes_plot = 50
# shifts_align=np.arange(-8,9)
# for cluster_id in sorting_hdbl_t.get_unit_ids():
#     if cluster_id != -1:
#         #compute firing rate
#         st_1 = sorting_hdbl_t.get_unit_spike_train(cluster_id)
#         #compute K closest clsuters
#         closest_clusters = get_closest_clusters_hdbscan(cluster_id, cluster_centers, num_close_clusters=num_close_clusters)
#         #compute unit similarties
#         original_template, closest_clusters, similarities, agreements, templates, shifts = get_unit_similarities(cluster_id, st_1, closest_clusters, sorting_hdbl_t, geom_array, raw_data_bin, 
#                                                                                                                  num_channels_similarity=num_channels_similarity, 
#                                                                                                                  num_close_clusters=num_close_clusters, shifts_align=shifts_align,
#                                                                                                                  order_by ='similarity')
#         if similarities[0] < 2.0: #arbitrary..
#             print(similarities[0], closest_clusters[0])
#             fig = plot_unit_similarities(cluster_id, closest_clusters, sorting_hdbl_t, sorting_hdbl_t, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
#                                          num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="both",
#                                          denoised_waveforms=wfs_localized, cluster_labels=clusterer.labels_, non_triaged_idxs=non_triaged_idxs, triaged_mcs_abs=triaged_mcs_abs, 
#                                          triaged_firstchans=triaged_firstchans)
#             plt.close(fig)
#             fig.savefig(save_dir_path + f"/cluster_{cluster_id}_summary.png")

# %% tags=[]
# save_dir_path_hdbscan_kilo = "cluster_summaries_hdbscan_kilo"
# if not os.path.exists(save_dir_path_hdbscan_kilo):
#     os.makedirs(save_dir_path_hdbscan_kilo)

# num_close_clusters = 50
# num_close_clusters_plot=10
# num_channels_similarity = 20
# num_under_threshold = 0
# num_spikes_plot = 50
# shifts_align=np.arange(-8,9)
# for cluster_id in [18]:#tqdm(sorting_hdbl_t.get_unit_ids()):
#     if cluster_id != -1:
#         st_1 = sorting_hdbl_t.get_unit_spike_train(cluster_id)

#         #compute K closest kilosort clsuters
#         closest_clusters = get_closest_clusters_hdbscan_kilosort(cluster_id, cluster_centers, kilo_cluster_depth_means, num_close_clusters)
        
#         fig = plot_unit_similarities(cluster_id, closest_clusters, sorting_hdbl_t, sorting_kilo, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
#                                      num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="second")
#         plt.close(fig)
#         fig.savefig(save_dir_path_hdbscan_kilo + f"/cluster_{cluster_id}_summary.png")

# %%
# save_dir_path_hdbscan_kilo = "cluster_summaries_hdbscan_kilo_agreement"
# if not os.path.exists(save_dir_path_hdbscan_kilo):
#     os.makedirs(save_dir_path_hdbscan_kilo)
    
# for cluster_id in tqdm(sorting_hdbl_t.get_unit_ids()):
#     if cluster_id != -1:
#         st_1 = sorting_hdbl_t.get_unit_spike_train(cluster_id)

#         #compute K closest kilosort clsuters
#         closest_clusters = get_closest_clusters_hdbscan_kilosort(cluster_id, cluster_centers, kilo_cluster_depth_means, num_close_clusters)
        
#         fig = plot_unit_similarities(cluster_id, closest_clusters, sorting_hdbl_t, sorting_kilo, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
#                                      num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='agreement', normalize_agreement_by="second")
#         plt.close(fig)
#         fig.savefig(save_dir_path_hdbscan_kilo + f"/cluster_{cluster_id}_summary.png")

# %% tags=[]
# save_dir_path_kilo_hdbscan = "cluster_summaries_kilo_hdbscan"
# if not os.path.exists(save_dir_path_kilo_hdbscan):
#     os.makedirs(save_dir_path_kilo_hdbscan)
    
# for cluster_id in tqdm(sorting_kilo.get_unit_ids()):

#     st_1 = sorting_kilo.get_unit_spike_train(cluster_id)

#     #compute K closest hdbscan clsuters
#     closest_clusters = get_closest_clusters_kilosort_hdbscan(cluster_id, kilo_cluster_depth_means, cluster_centers, num_close_clusters)

#     fig = plot_unit_similarities(cluster_id, closest_clusters, sorting_kilo, sorting_hdbl_t, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
#                                  num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="second")
#     plt.close(fig)
#     fig.savefig(save_dir_path_kilo_hdbscan + f"/cluster_{cluster_id}_summary.png")

# %%
# save_dir_path_kilo_hdbscan = "cluster_summaries_kilo_hdbscan_agreement"
# if not os.path.exists(save_dir_path_kilo_hdbscan):
#     os.makedirs(save_dir_path_kilo_hdbscan)
    
# for cluster_id in tqdm(sorting_kilo.get_unit_ids()):

#     st_1 = sorting_kilo.get_unit_spike_train(cluster_id)

#     #compute K closest hdbscan clsuters
#     closest_clusters = get_closest_clusters_kilosort_hdbscan(cluster_id, kilo_cluster_depth_means, cluster_centers, num_close_clusters)

#     fig = plot_unit_similarities(cluster_id, closest_clusters, sorting_kilo, sorting_hdbl_t, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
#                                  num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='agreement', normalize_agreement_by="second")
#     plt.close(fig)
#     fig.savefig(save_dir_path_kilo_hdbscan + f"/cluster_{cluster_id}_summary.png")

# %%
# cmp_kilo_5 = compare_two_sorters(sorting_kilo, sorting_hdbl_t, sorting1_name='kilosort', sorting2_name='ours', match_score=.5)
# matched_units_kilo_5 = cmp_kilo_5.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_5.get_matching()[0] != -1.)]
# unmatched_units_kilo_5 = cmp_kilo_5.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_5.get_matching()[0] == -1.)]

# cmp_kilo_1 = compare_two_sorters(sorting_kilo, sorting_hdbl_t, sorting1_name='kilosort', sorting2_name='ours', match_score=.1)
# matched_units_kilo_1 = cmp_kilo_1.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_1.get_matching()[0].to_numpy() != -1.)]
# unmatched_units_kilo_1 = cmp_kilo_1.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_1.get_matching()[0].to_numpy() == -1.)]

# %%
# too_small = 0
# not_in_snippet = 0
# good_match = 0
# bad_match = 0
# y_we_no_catch = []
# for good_cluster in good_kilo_sort_clusters:
#     if good_cluster not in sorting_kilo.get_unit_ids():
#         not_in_snippet += 1
#     elif len(sorting_kilo.get_unit_spike_train(good_cluster)) < 25:
#         too_small += 1 
#     elif good_cluster in matched_units_kilo_5:
#         good_match += 1
#     elif good_cluster in matched_units_kilo_1:
#         bad_match += 1
#     else:
#         y_we_no_catch.append(good_cluster)
# print(f"total understood: {not_in_snippet + too_small + bad_match + good_match}, good_match (>.5): {good_match}, bad_match (<.5): {bad_match}, too_few_spikes_in_snippet: {too_small}, not_in_snippet {not_in_snippet}")
# print(f"total not accounted for: {len(y_we_no_catch)}")

# %%
# #plot specific kilosort example
# cluster_id = 53
# num_close_clusters = 50
# num_close_clusters_plot=10
# num_channels_similarity = 20
# shifts_align=np.arange(-8,9)

# st_1 = sorting_kilo.get_unit_spike_train(cluster_id)

# #compute K closest hdbscan clsuters
# closest_clusters = get_closest_clusters_kilosort_hdbscan(cluster_id, kilo_cluster_depth_means, cluster_centers, num_close_clusters)

# fig = plot_unit_similarities(cluster_id, closest_clusters, sorting_kilo, sorting_hdbl_t, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
#                              num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="second")

# %%
#plot specific hdbscan example
cluster_id = 336
num_close_clusters = 50
num_close_clusters_plot=10
num_channels_similarity = 20
shifts_align=np.arange(-8,9)

st_1 = sorting_hdbl_t.get_unit_spike_train(cluster_id)

#compute K closest kilosort clsuters
closest_clusters = get_closest_clusters_hdbscan(cluster_id, cluster_centers, num_close_clusters)

fig = plot_unit_similarities(cluster_id, closest_clusters, sorting_hdbl_t, sorting_hdbl_t, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
                             num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="second")
# plt.close(fig)
# fig.savefig(save_dir_path_kilo_hdbscan + f"/cluster_{cluster_id}_summary.png")

# %%
wfs_a = wfs_localized[non_triaged_idxs[clusterer.labels_==336]]

# %%
wfs_a.shape

# %%
from

# %%
scipy.stats.mode(triaged_mcs_abs[clusterer.labels_==336])[0][0]

# %%
relmaxa = scipy.stats.mode(triaged_mcs_abs[clusterer.labels_==336])[0][0] - scipy.stats.mode(triaged_firstchans[clusterer.labels_==336])[0][0]


# %%
n_templates

# %%
n_templates = clusterer.labels_.max()+1
labels=clusterer.labels_
templates = get_templates(raw_data_bin, geom_array, n_templates, triaged_spike_index, labels)

from spike_psvae.merge_split import get_n_spikes_templates, get_x_z_templates,get_proposed_pairs
n_spikes_templates = get_n_spikes_templates(n_templates, labels)
x_z_templates = get_x_z_templates(n_templates, labels, triaged_x, triaged_z)
print("GET PROPOSED PAIRS")
dist_argsort, dist_template = get_proposed_pairs(n_templates, templates, x_z_templates, n_temp = 20)

# %%
templates[336].ptp(0).argmax()

# %%
for i in range(30):
    from spike_psvae.merge_split import get_diptest_value
    unit_a = 336
    unit_b = 338
    mc = templates[unit_a].ptp(0).argmax()
    two_units_shift = templates[unit_b, :, mc].argmin() - templates[unit_a, :, mc].argmin()
    unit_shifted = unit_b


    print(get_diptest_value(raw_data_bin, geom_array, triaged_spike_index, labels, unit_a, unit_b, n_spikes_templates, mc, two_units_shift, unit_shifted, denoiser, device, n_channels=40, n_times=121))

# %%

# %%

# %%
relmaxa = scipy.stats.mode(triaged_mcs_abs[clusterer.labels_==336])[0][0] - scipy.stats.mode(triaged_firstchans[clusterer.labels_==336])[0][0]
wfs_a = wfs_localized[non_triaged_idxs[clusterer.labels_==336]]
wfs_b = wfs_localized[non_triaged_idxs[clusterer.labels_==338]]
n_channels = 40
wfs_diptest = np.concatenate((wfs_a, wfs_b)).reshape((-1, n_channels*121))
labels_diptest = np.zeros(wfs_a.shape[0]+wfs_b.shape[0])
labels_diptest[:wfs_a.shape[0]] = 1

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from isosplit import isocut

# %%
lda_model = LDA(n_components = 1)
lda_comps = lda_model.fit_transform(wfs_diptest, labels_diptest)
value_dpt, cut_calue = isocut(lda_comps[:, 0])

# %%
plt.hist(lda_comps[:, 0], bins = 20)
plt.title("Diptest value : " + str(value_dpt))
plt.show()

# %%
# cluster_id = 37
# if cluster_id != -1:
#     curr_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
#     initial_indices = np.nonzero(clusterer.labels_ == cluster_id)[0]
#     cluster_features = np.concatenate((np.expand_dims(triaged_x,1), np.expand_dims(triaged_z,1), np.expand_dims(np.log(triaged_maxptps)*scales[4],1)), axis=1)[np.nonzero(clusterer.labels_ == cluster_id)[0]]
#     curr_clusterer.fit(cluster_features)
#     final_indices_list = []
#     final_labels_list = []
#     indices_to_be_processed = []
#     final_labels_concat = None
#     unique_labels = np.unique(curr_clusterer.labels_)
#     if len(unique_labels) > 1:
#         for label in unique_labels:
#             curr_indices = np.nonzero(curr_clusterer.labels_ == label)[0]
#             if label != -1:
#                 indices_to_be_processed.append(curr_indices)
#             else:
#                 final_indices_list.append(curr_indices)
#                 final_labels = (np.zeros(len(curr_indices)) - 1).astype('int')
#                 final_labels_list.append(final_labels)
#         cluster_id_curr = 0
#         while len(indices_to_be_processed) > 0:
#             indices = indices_to_be_processed.pop()
#             curr_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
#             curr_clusterer.fit(cluster_features[indices])
#             # print(np.unique(curr_clusterer.labels_))
#             if len(np.unique(curr_clusterer.labels_)) > 1:
#                 for label in np.unique(curr_clusterer.labels_):
#                     curr_indices = indices[np.nonzero(curr_clusterer.labels_ == label)[0]]
#                     if label != -1:
#                         indices_to_be_processed.append(curr_indices)
#                     else:
#                         final_indices_list.append(curr_indices)
#                         final_labels = (np.zeros(len(curr_indices)) - 1).astype('int')
#                         final_labels_list.append(final_labels)
#             else:
#                 final_indices_list.append(indices)
#                 final_labels = (np.zeros(len(indices)) + cluster_id_curr).astype('int')
#                 final_labels_list.append(final_labels)
#                 cluster_id_curr += 1
#         final_indices_concat = np.concatenate(final_indices_list)
#         final_labels_concat = np.concatenate(final_labels_list)
#         sort_idxs = np.argsort(final_indices_concat)
#         final_indices_concat = final_indices_concat[sort_idxs]
#         final_labels_concat = final_labels_concat[sort_idxs]
#     else:
#         print("no split")
#     print(np.unique(final_labels_concat))

# fig = plot_array_scatter(final_labels_concat, geom_array, triaged_x[clusterer.labels_ == cluster_id], 
#                          triaged_z[clusterer.labels_ == cluster_id], 
#                          triaged_maxptps[clusterer.labels_ == cluster_id], 
#                          cluster_color_dict, color_arr[clusterer.labels_ == cluster_id], 
#                          min_samples=25, min_cluster_size=25, z_cutoff=(cluster_centers.iloc[cluster_id][1]-100,cluster_centers.iloc[cluster_id][1]+100), figsize=(18, 12))

# fig = plt.figure(figsize=(6,12))
# ax_all = fig.gca()
# hshifts=[0, .2,.4,.6,.8, 1]
# for unit_id, hshift in zip(np.unique(final_labels_concat)[1:],hshifts):
#     first_chans_cluster = triaged_firstchans[clusterer.labels_ == cluster_id][final_labels_concat==unit_id]
#     mcs_abs_cluster = triaged_mcs_abs[clusterer.labels_ == cluster_id][final_labels_concat==unit_id]
#     spike_times = triaged_spike_index[:,0][clusterer.labels_ == cluster_id][final_labels_concat==unit_id]
#     num_channels = 40
#     bin_file = raw_data_bin
#     fig = plot_raw_waveforms_unit_geom(geom_array, num_channels, first_chans_cluster, mcs_abs_cluster, spike_times, bin_file, x_geom_scale = 1/15, 
#                                        y_geom_scale = 1/10, waveform_scale = .15, spikes_plot = 100, waveform_shape=(30,90), num_rows=3, 
#                                        alpha=.05, h_shift=hshift, do_mean=False, ax=ax_all, color=cluster_color_dict[unit_id])

# %%
# save_dir_path_duplicates_hdbscan = "duplicates_hdbscan"
# if not os.path.exists(save_dir_path_duplicates_hdbscan):
#     os.makedirs(save_dir_path_duplicates_hdbscan)
    
# for i in range(0,len(duplicates),2):
#     cmp = cmp_self
#     cluster_id = duplicates[i]
#     cluster_id_match = duplicates[i+1]
#     num_channels = wfs_localized.shape[2]
#     # cluster_id_match = cmp.get_best_unit_match1(cluster_id)
#     sorting1 = sorting_hdbl_t
#     sorting2 = sorting_hdbl_t
#     sorting1_name = "hdb"
#     sorting2_name = "hdb"
#     firstchans_cluster_sorting1 = triaged_firstchans[clusterer.labels_ == cluster_id]
#     mcs_abs_cluster_sorting1 = triaged_mcs_abs[clusterer.labels_ == cluster_id]
#     firstchans_cluster_sorting2 = triaged_firstchans[clusterer.labels_ == cluster_id_match]
#     mcs_abs_cluster_sorting2 = triaged_mcs_abs[clusterer.labels_ == cluster_id_match]

#     fig = plot_agreement_venn(cluster_id, cluster_id_match, cmp, sorting1, sorting2, sorting1_name, sorting2_name, geom_array, num_channels, num_spikes_plot, firstchans_cluster_sorting1, mcs_abs_cluster_sorting1, 
#                               firstchans_cluster_sorting2, mcs_abs_cluster_sorting2, raw_data_bin, delta_frames = 12)
#     plt.close(fig)
#     fig.savefig(save_dir_path_duplicates_hdbscan + f"/cluster_{cluster_id}_cluster_{cluster_id_match}_agreement.png")

#     #plot cluster summary
#     fig = plot_single_unit_summary(cluster_id, clusterer.labels_, cluster_centers, geom_array, 50, num_rows_plot, triaged_x, triaged_z, triaged_maxptps, 
#                                    triaged_firstchans, triaged_mcs_abs, triaged_spike_index, non_triaged_idxs, wfs_localized, wfs_subtracted, cluster_color_dict, 
#                                    color_arr, raw_data_bin, residual_data_bin)
#     plt.close(fig)
#     fig.savefig(save_dir_path_duplicates_hdbscan + f"/cluster_{cluster_id}_summary.png")
#     #plot cluster summary
#     fig = plot_single_unit_summary(cluster_id_match, clusterer.labels_, cluster_centers, geom_array, 50, num_rows_plot, triaged_x, triaged_z, triaged_maxptps, 
#                                    triaged_firstchans, triaged_mcs_abs, triaged_spike_index, non_triaged_idxs, wfs_localized, wfs_subtracted, cluster_color_dict, 
#                                    color_arr, raw_data_bin, residual_data_bin)
#     plt.close(fig)
#     fig.savefig(save_dir_path_duplicates_hdbscan + f"/cluster_{cluster_id_match}_summary.png")
