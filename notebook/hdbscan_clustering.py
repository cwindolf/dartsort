# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
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
# %matplotlib inline
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
import h5py
from scipy.spatial import cKDTree
import pickle
import sklearn
import seaborn as sns
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial.distance import cdist
from isosplit import isocut



# %%
from spike_psvae.denoise import SingleChanDenoiser
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

denoiser = SingleChanDenoiser()
denoiser.load()
denoiser.to(device)


# %%
#Helper functions

def denoise_wf_nn_tmp_single_channel(wf, denoiser, device):
    denoiser = denoiser.to(device)
    n_data, n_times, n_chans = wf.shape
    if wf.shape[0]>0:
        wf_reshaped = wf.transpose(0, 2, 1).reshape(-1, n_times)
        wf_torch = torch.FloatTensor(wf_reshaped).to(device)
        denoised_wf = denoiser(wf_torch).data
        denoised_wf = denoised_wf.reshape(
            n_data, n_chans, n_times)
        denoised_wf = denoised_wf.cpu().data.numpy().transpose(0, 2, 1)

        del wf_torch
    else:
        denoised_wf = np.zeros((wf.shape[0], wf.shape[1]*wf.shape[2]),'float32')

    return denoised_wf

def run_weighted_triage(x, y, z, alpha, maxptps, pcs=None, 
                        scales=(1,10,1,15,30,10),
                        threshold=100, ptp_threshold=3, c=1, ptp_weighting=True):
    
    ptp_filter = np.where(maxptps>ptp_threshold)
    x = x[ptp_filter]
    y = y[ptp_filter]
    z = z[ptp_filter]
    alpha = alpha[ptp_filter]
    maxptps = maxptps[ptp_filter]
    if pcs is not None:
        pcs = pcs[ptp_filter]
        feats = np.c_[scales[0]*x,
                      scales[1]*np.log(y),
                      scales[2]*z,
                      scales[3]*np.log(alpha),
                      scales[4]*np.log(maxptps),
                      scales[5]*pcs[:,:3]]
    else:
        feats = np.c_[scales[0]*x,
                      # scales[1]*np.log(y),
                      scales[2]*z,
                      # scales[3]*np.log(alpha),
                      scales[4]*np.log(maxptps)]
    
    tree = cKDTree(feats)
    dist, ind = tree.query(feats, k=6)
    dist = dist[:,1:]
    # dist = np.sum((c*np.log(dist)),1)
    # print(dist)
    if ptp_weighting:
        dist = np.sum(c*np.log(dist) + np.log(1/(scales[4]*np.log(maxptps)))[:,None], 1)
    else:
        dist = np.sum((c*np.log(dist)),1)
        
    idx_keep = dist <= np.percentile(dist, threshold)
    
    triaged_x = x[idx_keep]
    triaged_y = y[idx_keep]
    triaged_z = z[idx_keep]
    triaged_alpha = alpha[idx_keep]
    triaged_maxptps = maxptps[idx_keep]
    triaged_pcs = None
    if pcs is not None:
        triaged_pcs = pcs[idx_keep]
        
    
    return triaged_x, triaged_y, triaged_z, triaged_alpha, triaged_maxptps, triaged_pcs, ptp_filter, idx_keep

def read_waveforms(spike_times, bin_file, geom_array, n_times=121, offset_denoiser = 42, channels=None, dtype=np.dtype('float32')):
    '''
    read waveforms from recording
    n_times : waveform temporal length 
    channels : channels to read from 
    '''
    # n_times needs to be odd
    if n_times % 2 == 0:
        n_times += 1

    # read all channels
    if channels is None:
        channels = np.arange(geom_array.shape[0])
        
    # ***** LOAD RAW RECORDING *****
    wfs = np.zeros((len(spike_times), n_times, len(channels)),
                   'float32')

    skipped_idx = []
    n_channels = geom_array.shape[0] #len(channels)
    total_size = n_times*n_channels
    # spike_times are the centers of waveforms
    spike_times_shifted = spike_times - (offset_denoiser) #n_times//2
    offsets = spike_times_shifted.astype('int64')*dtype.itemsize*n_channels
    with open(bin_file, "rb") as fin:
        for ctr, spike in enumerate(spike_times_shifted):
            try:
                fin.seek(offsets[ctr], os.SEEK_SET)
                wf = np.fromfile(fin,
                                 dtype=dtype,
                                 count=total_size)
                wfs[ctr] = wf.reshape(
                    n_times, n_channels)[:,channels]
            except:
                print(f"skipped {ctr, spike}")
                skipped_idx.append(ctr)
    wfs=np.delete(wfs, skipped_idx, axis=0)
    fin.close()

    return wfs, skipped_idx


# %%
geom = 'np1_channel_map.npy'
triage_quantile = 85
do_infer_ptp = False
num_spikes_cluster = None
min_cluster_size = 25
min_samples = 25
num_spikes_plot = 250
num_rows_plot = 3
no_verbose = True

data_path = '/media/cat/cole/'
data_name = 'CSH_ZAD_026_1800_1860'
data_dir = data_path + data_name + '/'
raw_data_bin = data_dir + '1min_standardized.bin'
residual_data_bin = data_dir + 'residual_1min_standardized_t_0_None.bin'

#load features
spike_index = np.load(data_dir+'spike_index.npy')
num_spikes = spike_index.shape[0]
spike_index[:,0] = spike_index[:,0] #only for Hyun's data
results_localization = np.load(data_dir+'localization_results.npy')
ptps_localized = np.load(data_dir+'ptps.npy')
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
first_chan = results_localization[:, 5]
subtracted_wfs = np.load(data_dir+'subtracted_wfs.npy')

# %%
displacement_estimate = np.load(data_dir+'displacement.npy')

# %%
displacement_estimate.shape

# %%
#perform triaging 
triaged_x, triaged_y, triaged_z, triaged_alpha, triaged_maxptps, _, ptp_filter, idx_keep = run_weighted_triage(x, y, z, alpha, maxptps, threshold=75, ptp_threshold=3, ptp_weighting=True) #pcs is None here
# triaged_x, triaged_y, triaged_z, triaged_alpha, triaged_maxptps, _, ptp_filter, idx_keep = run_weighted_triage(x, y, z, alpha, maxptps, threshold=100, ptp_threshold=0, ptp_weighting=False) #pcs is None here
triaged_spike_index = spike_index[ptp_filter][idx_keep]
triaged_mcs_abs = spike_index[:,1][ptp_filter][idx_keep]
triaged_sub_wfs = subtracted_wfs[ptp_filter][idx_keep]
triaged_first_chan = first_chan[ptp_filter][idx_keep]

# %%
mask = np.ones(spike_index[:,1].size, dtype=bool)
mask[ptp_filter[0][idx_keep]] = False
triaged_indices = np.where(mask)[0]
# np.save('triaged_indices', triaged_indices)

# %%
#can infer ptp
if do_infer_ptp:
    if no_verbose:
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
    a_group_key = list(f.keys())[2]
    firstchans = np.asarray(list(f["first_channels"]))
    print(f["end_sample"])
    end_sample = f["end_sample"][()]
    start_sample = f["start_sample"][()]
triaged_firstchans = firstchans[ptp_filter][idx_keep]
end_time = end_sample / 30000
start_time = start_sample / 30000
recording_duration = end_time - start_time
print(f"duration of recording: {recording_duration} s")

# %%
#load kilosort results
kilo_spike_samples = np.load(data_dir + 'kilosort_spk_samples.npy')
kilo_spike_frames = (kilo_spike_samples - 30*recording_duration*30000) #+18 to match our detection alignment
kilo_spike_clusters = np.load(data_dir + 'kilsort_spk_clusters.npy')
kilo_spike_depths = np.load(data_dir + 'kilsort_spk_depths.npy')
kilo_cluster_depth_means = {}
for cluster_id in np.unique(kilo_spike_clusters):
    kilo_cluster_depth_means[cluster_id] = np.mean(kilo_spike_depths[kilo_spike_clusters==cluster_id])

# %%
# ## Create feature set for clustering
if num_spikes_cluster is None:
    num_spikes = triaged_x.shape[0]
else:
    num_spikes = num_spikes_cluster
triaged_firstchans = triaged_firstchans[:num_spikes]
triaged_alpha = triaged_alpha[:num_spikes]
triaged_spike_index = triaged_spike_index[:num_spikes]
triaged_x = triaged_x[:num_spikes]
triaged_y = triaged_y[:num_spikes]
triaged_z = triaged_z[:num_spikes]
triaged_maxptps = triaged_maxptps[:num_spikes]
triaged_mcs_abs = triaged_mcs_abs[:num_spikes]
triaged_first_chan = triaged_first_chan[:num_spikes]

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

#re-compute cluster centers
cluster_centers = []
for label in np.unique(clusterer.labels_):
    if label != -1:
        cluster_centers.append(clusterer.weighted_cluster_centroid(label))
cluster_centers = np.asarray(cluster_centers)

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


# %% [markdown]
# ## Split

# %%

def run_LDA_split(wfs_unit_denoised, n_channels = 10, n_times=121):
    lda_model = LDA(n_components = 2)
    arr = wfs_unit_denoised.ptp(1).argmax(1)
    if np.unique(arr).shape[0]<=2:
        arr[-1] = np.unique(arr)[0]-1
        arr[0] = np.unique(arr)[-1]+1
    lda_comps = lda_model.fit_transform(wfs_unit_denoised.reshape((-1, n_times*n_channels)), arr)
    lda_clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
    lda_clusterer.fit(lda_comps)
    return lda_clusterer.labels_

def split_clusters_better(residual_path, waveforms, first_chans, spike_index, labels, x, z, ptps, geom_array, denoiser, device, n_channels=10):
    labels_new = labels.copy()
    labels_original = labels.copy()

    n_clusters = labels.max()
    for unit in tqdm(np.unique(labels)[1:]):
        spike_index_unit = spike_index[labels == unit]
        waveforms_unit = waveforms[labels == unit]
        first_chans_unit = first_chans[labels == unit]
        x_unit, z_unit, ptps_unit = x[labels == unit], z[labels == unit], ptps[labels == unit]
        is_split, unit_new_labels = split_individual_cluster_better(residual_path, waveforms_unit, first_chans_unit, spike_index_unit, x_unit, z_unit, ptps_unit, geom_array, denoiser, device, n_channels)
        if is_split:
            for new_label in np.unique(unit_new_labels):
                if new_label == -1:
                    idx = np.flatnonzero(labels_original == unit)[unit_new_labels == new_label]
                    labels_new[idx] = new_label
                elif new_label > 0:
                    n_clusters += 1
                    idx = np.flatnonzero(labels_original == unit)[unit_new_labels == new_label]
                    labels_new[idx] = n_clusters
    return labels_new

def split_individual_cluster_better(residual_path, waveforms_unit, first_chans_unit, spike_index_unit, x_unit, z_unit, ptps_unit, geom_array, denoiser, device, n_channels = 10):
    total_channels = geom_array.shape[0]
    n_channels_half = n_channels//2
    labels_unit = -1*np.ones(spike_index_unit.shape[0])
    is_split = False
    true_mc = int(np.median(spike_index_unit[:, 1]))
    mc = max(n_channels_half, true_mc)
    mc = min(total_channels - n_channels_half, mc)
    pca_model = PCA(2)
    wfs_unit = np.zeros((waveforms_unit.shape[0], waveforms_unit.shape[1], n_channels))
    for i in range(wfs_unit.shape[0]):
        if mc == n_channels_half:
            wfs_unit[i] = waveforms_unit[i, :, :n_channels]
        elif mc == total_channels - n_channels_half:
            wfs_unit[i] = waveforms_unit[i, :, waveforms_unit.shape[2]-n_channels:]
        else:
            mc_new = int(mc - first_chans_unit[i])
            wfs_unit[i] = waveforms_unit[i, :, mc_new-n_channels_half:mc_new+n_channels_half]
    
    wfs_unit += read_waveforms(spike_index_unit[:, 0], residual_path, geom_array, n_times=121, channels = np.arange(mc-n_channels_half, mc+n_channels_half))[0]
    wfs_unit_denoised = denoise_wf_nn_tmp_single_channel(wfs_unit, denoiser, device)
    
    if true_mc<n_channels_half:
        pcs = pca_model.fit_transform(wfs_unit_denoised[:, :, true_mc])
    if true_mc>total_channels-n_channels_half:
        true_mc = true_mc - (total_channels-n_channels)
        pcs = pca_model.fit_transform(wfs_unit_denoised[:, :, true_mc])
    else:
        pcs = pca_model.fit_transform(wfs_unit_denoised[:, :, n_channels_half])

    alpha1 = (x_unit.max() - x_unit.min())/(pcs[:, 0].max()-pcs[:, 0].min())
    alpha2 = (x_unit.max() - x_unit.min())/(pcs[:, 1].max()-pcs[:, 1].min())
    features = np.concatenate((np.expand_dims(x_unit,1), np.expand_dims(z_unit,1), np.expand_dims(pcs[:, 0], 1)*alpha1, np.expand_dims(pcs[:, 1], 1)*alpha2, np.expand_dims(np.log(ptps_unit)*30,1)), axis=1) #Use scales parameter
    clusterer_herding = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
    clusterer_herding.fit(features)
    
    labels_rec_hdbscan = clusterer_herding.labels_
    if np.unique(labels_rec_hdbscan).shape[0]>1:
        is_split = True
# Don't do diptest Here
#         diptest_comps = lda_diptest.fit_transform(wfs_unit_denoised.reshape((-1, 121*10)), clusterer_herding.labels_)
#         value_dpt, cut_calue = isocut(diptest_comps[:, 0])
#         if value_dpt > 0.5:
#             is_split = True
    
    if is_split:
        labels_unit[labels_rec_hdbscan==-1] = -1
        label_max_temp = labels_rec_hdbscan.max()
        cmp = 0
        for new_unit_id in np.unique(labels_rec_hdbscan)[1:]:
            wfs_new_unit = wfs_unit_denoised[labels_rec_hdbscan == new_unit_id]
            lda_labels = run_LDA_split(wfs_new_unit)
            if np.unique(lda_labels).shape[0]==1:
                labels_unit[labels_rec_hdbscan==new_unit_id] = cmp
                cmp += 1
            else:
                for lda_unit in np.unique(lda_labels):
                    if lda_unit >= 0:
                        labels_unit[np.flatnonzero(labels_rec_hdbscan==new_unit_id)[lda_labels == lda_unit]] = cmp
                        cmp += 1
                    else:
                        labels_unit[np.flatnonzero(labels_rec_hdbscan==new_unit_id)[lda_labels == lda_unit]] = -1
    else:
        lda_labels = run_LDA_split(wfs_unit_denoised)
        if np.unique(lda_labels).shape[0]>1:
            is_split = True
            labels_unit = lda_labels

    return is_split, labels_unit

def split_individual_cluster(standardized_path, spike_index_unit, x_unit, z_unit, ptps_unit, geom_array, denoiser, device, n_channels = 10):
    total_channels = geom_array.shape[0]
    n_channels_half = n_channels//2
    labels_unit = -1*np.ones(spike_index_unit.shape[0])
    is_split = False
    true_mc = int(np.median(spike_index_unit[:, 1]))
    mc = max(n_channels_half, true_mc)
    mc = min(total_channels - n_channels_half, mc)
    pca_model = PCA(2)
    wfs_unit = read_waveforms(spike_index_unit[:, 0], standardized_path, geom_array, n_times=121, channels = np.arange(mc-n_channels_half, mc+n_channels_half))[0]
    wfs_unit_denoised = denoise_wf_nn_tmp_single_channel(wfs_unit, denoiser, device)
    if true_mc<n_channels_half:
        pcs = pca_model.fit_transform(wfs_unit_denoised[:, :, true_mc])
    if true_mc>total_channels-n_channels_half:
        true_mc = true_mc - (total_channels-n_channels)
        pcs = pca_model.fit_transform(wfs_unit_denoised[:, :, true_mc])
    else:
        pcs = pca_model.fit_transform(wfs_unit_denoised[:, :, n_channels_half])

    alpha1 = (x_unit.max() - x_unit.min())/(pcs[:, 0].max()-pcs[:, 0].min())
    alpha2 = (x_unit.max() - x_unit.min())/(pcs[:, 1].max()-pcs[:, 1].min())
    features = np.concatenate((np.expand_dims(x_unit,1), np.expand_dims(z_unit,1), np.expand_dims(pcs[:, 0], 1)*alpha1, np.expand_dims(pcs[:, 1], 1)*alpha2, np.expand_dims(np.log(ptps_unit)*30,1)), axis=1) #Use scales parameter
    clusterer_herding = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
    clusterer_herding.fit(features)
    
    labels_rec_hdbscan = clusterer_herding.labels_
    if np.unique(labels_rec_hdbscan).shape[0]>1:
        is_split = True
# Don't do diptest Here
#         diptest_comps = lda_diptest.fit_transform(wfs_unit_denoised.reshape((-1, 121*10)), clusterer_herding.labels_)
#         value_dpt, cut_calue = isocut(diptest_comps[:, 0])
#         if value_dpt > 0.5:
#             is_split = True
    
    if is_split:
        labels_unit[labels_rec_hdbscan==-1] = -1
        label_max_temp = labels_rec_hdbscan.max()
        cmp = 0
        for new_unit_id in np.unique(labels_rec_hdbscan)[1:]:
            wfs_new_unit = wfs_unit_denoised[labels_rec_hdbscan == new_unit_id]
            lda_labels = run_LDA_split(wfs_new_unit)
            if np.unique(lda_labels).shape[0]==1:
                labels_unit[labels_rec_hdbscan==new_unit_id] = cmp
                cmp += 1
            else:
                for lda_unit in np.unique(lda_labels):
                    if lda_unit >= 0:
                        labels_unit[np.flatnonzero(labels_rec_hdbscan==new_unit_id)[lda_labels == lda_unit]] = cmp
                        cmp += 1
                    else:
                        labels_unit[np.flatnonzero(labels_rec_hdbscan==new_unit_id)[lda_labels == lda_unit]] = -1
    else:
        lda_labels = run_LDA_split(wfs_unit_denoised)
        if np.unique(lda_labels).shape[0]>1:
            is_split = True
            labels_unit = lda_labels

    return is_split, labels_unit

def split_clusters(standardized_path, spike_index, labels, x, z, ptps, geom_array, denoiser, device, n_channels=10):
    labels_new = labels.copy()
    labels_original = labels.copy()

    n_clusters = labels.max()
    for unit in (np.unique(labels)[1:]):
        spike_index_unit = spike_index[labels == unit]
        x_unit, z_unit, ptps_unit = x[labels == unit], z[labels == unit], ptps[labels == unit]
        is_split, unit_new_labels = split_individual_cluster(standardized_path, spike_index_unit, x_unit, z_unit, ptps_unit, geom_array, denoiser, device, n_channels)
        if is_split:
            for new_label in np.unique(unit_new_labels):
                if new_label == -1:
                    idx = np.flatnonzero(labels_original == unit)[unit_new_labels == new_label]
                    labels_new[idx] = new_label
                elif new_label > 0:
                    n_clusters += 1
                    idx = np.flatnonzero(labels_original == unit)[unit_new_labels == new_label]
                    labels_new[idx] = n_clusters
    return labels_new

def get_x_z_templates(n_templates, labels, x, z):
    x_z_templates = np.zeros((n_templates, 2))
    for i in range(n_templates):
        x_z_templates[i, 1] = np.median(z[labels==i])
        x_z_templates[i, 0] = np.median(x[labels==i])
    return x_z_templates

def get_n_spikes_templates(n_templates, labels):
    n_spikes_templates = np.zeros(n_templates)
    for i in range(n_templates):
        n_spikes_templates[i] = (labels==i).sum()
    return n_spikes_templates

def get_templates(standardized_path, geom_array, n_templates, spike_index, labels, max_spikes=250, n_times=121):
    templates = np.zeros((n_templates, n_times, geom_array.shape[0]))
    for unit in tqdm(range(n_templates)):
        spike_times_unit = spike_index[labels==unit, 0]
        if spike_times_unit.shape[0]>max_spikes:
            idx = np.random.choice(np.arange(spike_times_unit.shape[0]), max_spikes, replace = False)
        else:
            idx = np.arange(spike_times_unit.shape[0])

        wfs_unit = read_waveforms(spike_times_unit[idx], standardized_path, geom_array, n_times)[0]
        templates[unit] = wfs_unit.mean(0)
    return templates



# %%

labels = clusterer.labels_
z_labels = np.zeros(clusterer.labels_.max()+1)
for i in range(clusterer.labels_.max()+1):
    z_labels[i] = triaged_z[clusterer.labels_ == i].mean()
ordered_labels = labels.copy()
z_argsort = z_labels.argsort()[::-1]
for i in range(clusterer.labels_.max()+1):
    ordered_labels[labels == z_argsort[i]] = i


# %%

# %%
temp_clusterer = get_templates(raw_data_bin, geom_array, ordered_labels.max()+1, triaged_spike_index, ordered_labels)

list_argmin = np.zeros(temp_clusterer.shape[0])
for i in range(temp_clusterer.shape[0]):
    list_argmin[i] = temp_clusterer[i, :, temp_clusterer[i].ptp(0).argmax()].argmin()

idx_not_aligned = np.where(list_argmin!=42)[0]

for unit in idx_not_aligned:
    mc = temp_clusterer[unit].ptp(0).argmax()
    offset = temp_clusterer[unit, :, mc].argmin()
    triaged_spike_index[ordered_labels == unit, 0] += offset-42

idx_sorted = triaged_spike_index[:, 0].argsort()
spike_index_triaged = triaged_spike_index[idx_sorted]
ordered_labels = ordered_labels[idx_sorted]
triaged_x = triaged_x[idx_sorted]
triaged_z = triaged_z[idx_sorted]
triaged_maxptps = triaged_maxptps[idx_sorted]
triaged_sub_wfs = triaged_sub_wfs[idx_sorted]  
triaged_first_chan = triaged_first_chan[idx_sorted]  

temp_clusterer[94, :, temp_clusterer[94].ptp(0).argmax()].argmin()

# %%
labels_split = split_clusters_better(residual_data_bin, triaged_sub_wfs, triaged_first_chan, spike_index_triaged, ordered_labels, triaged_x, triaged_z, triaged_maxptps, geom_array, denoiser, device)


# %%
# labels_split = split_clusters(raw_data_bin, spike_index_triaged, ordered_labels, triaged_x, triaged_z, triaged_maxptps, geom_array, denoiser, device)
temp_clusterer = get_templates(raw_data_bin, geom_array, labels_split.max()+1, spike_index_triaged, labels_split)

list_argmin = np.zeros(temp_clusterer.shape[0])
for i in range(temp_clusterer.shape[0]):
    list_argmin[i] = temp_clusterer[i, :, temp_clusterer[i].ptp(0).argmax()].argmin()

idx_not_aligned = np.where(list_argmin!=42)[0]

for unit in idx_not_aligned:
    mc = temp_clusterer[unit].ptp(0).argmax()
    offset = temp_clusterer[unit, :, mc].argmin()
    spike_index_triaged[labels_split == unit, 0] += offset-42

idx_sorted = spike_index_triaged[:, 0].argsort()
spike_index_triaged = spike_index_triaged[idx_sorted]
labels_split = labels_split[idx_sorted]
triaged_x = triaged_x[idx_sorted]
triaged_z = triaged_z[idx_sorted]
triaged_maxptps = triaged_maxptps[idx_sorted]
triaged_first_chan = triaged_first_chan[idx_sorted]  
triaged_sub_wfs = triaged_sub_wfs[idx_sorted]  

z_labels = np.zeros(labels_split.max()+1)
for i in range(labels_split.max()+1):
    z_labels[i] = triaged_z[labels_split == i].mean()
    
ordered_split_labels = labels_split.copy()
z_argsort = z_labels.argsort()[::-1]
for i in range(labels_split.max()+1):
    ordered_split_labels[labels_split == z_argsort[i]] = i


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
for cluster_id in np.unique(ordered_split_labels):
    cluster_color_dict[cluster_id] = unique_colors[cluster_id % len(unique_colors)]
cluster_color_dict[-1] = '#808080' #set outlier color to grey

# %%
ordered_split_labels.max()

# %%
fig = plot_array_scatter(ordered_split_labels, geom_array, triaged_x, triaged_z, triaged_maxptps, cluster_color_dict, color_arr, min_cluster_size=clusterer.min_cluster_size, min_samples=clusterer.min_samples, 
                         z_cutoff=(0, 3900), figsize=(18, 24))
# fig.suptitle(f'x,z,scaled_logptp features," {num_spikes} datapoints');
plt.show()


# %% jupyter={"outputs_hidden": true} tags=[]
def compute_shifted_similarity(template1, template2, shifts=np.arange(-8,9)):
    curr_similarities = []
    for shift in shifts:
        if shift == 0:
            similarity = np.max(np.abs(template1 - template2))
        elif shift < 0:
            template2_shifted_flattened = np.pad(template2.T.flatten(),((-shift,0)), mode='constant')[:shift]
            similarity = np.max(np.abs(template1.T.flatten() - template2_shifted_flattened))
        else:    
            template2_shifted_flattened = np.pad(template2.T.flatten(),((0,shift)), mode='constant')[shift:]
            similarity = np.max(np.abs(template1.T.flatten() - template2_shifted_flattened))
        curr_similarities.append(similarity)
    return np.min(curr_similarities), shifts[np.argmin(curr_similarities)]

def get_proposed_pairs(n_templates, templates, x_z_templates, n_temp = 20, n_channels = 10):
    n_channels_half = n_channels//2
    dist = cdist(x_z_templates, x_z_templates)
    dist_argsort = dist.argsort(axis = 1)[:, 1:n_temp+1]
    dist_template = np.zeros((dist_argsort.shape[0], n_temp))
    for i in range(n_templates):
        mc = min(templates[i].ptp(0).argmax(), 384-n_channels_half)
        mc = max(mc, n_channels_half)
        temp_a = templates[i, :, mc-n_channels_half:mc+n_channels_half]
        for j in range(n_temp):
            temp_b = templates[dist_argsort[i, j], :, mc-n_channels_half:mc+n_channels_half]
            dist_template[i, j] = compute_shifted_similarity(temp_a, temp_b)[0]
    return dist_argsort, dist_template

def get_diptest_value_better(residual_path, waveforms, first_chans, geom_array, spike_index, labels, unit_a, unit_b, n_spikes_templates, mc, two_units_shift, unit_shifted, denoiser, device, n_channels = 10, n_times=121, rank_pca=8, nn_denoise = False):

    # ALIGN BASED ON MAX PTP TEMPLATE MC 
    n_channels_half = n_channels//2

    n_wfs_max = int(min(250, min(n_spikes_templates[unit_a], n_spikes_templates[unit_b]))) 

    
    mc = min(384-n_channels_half, mc)
    mc = max(n_channels_half, mc)

    spike_times_unit_a = spike_index[labels == unit_a, 0]
    idx = np.random.choice(np.arange(spike_times_unit_a.shape[0]), n_wfs_max, replace = False)
    spike_times_unit_a = spike_times_unit_a[idx]
    wfs_a = waveforms[labels == unit_a][idx]
    first_chan_a = triaged_first_chan[labels == unit_a][idx]
    
    spike_times_unit_b = spike_index[labels == unit_b, 0]
    idx = np.random.choice(np.arange(spike_times_unit_b.shape[0]), n_wfs_max, replace = False)
    spike_times_unit_b = spike_times_unit_b[idx]
    wfs_b = waveforms[labels == unit_b][idx]
    first_chan_b = triaged_first_chan[labels == unit_b][idx]
    
    wfs_a_bis = np.zeros((wfs_a.shape[0], n_times, n_channels))
    wfs_b_bis = np.zeros((wfs_b.shape[0], n_times, n_channels))
    
    if two_units_shift>0:
        
        if unit_shifted == unit_a:
            for i in range(wfs_a_bis.shape[0]):
                first_chan = int(mc - first_chan_a[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_a.shape[2]-n_channels, int(first_chan))
                wfs_a_bis[i, :-two_units_shift] = wfs_a[i, two_units_shift:, first_chan:first_chan+n_channels]
                first_chan = int(mc - first_chan_b[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_b.shape[2]-n_channels, int(first_chan))
                wfs_b_bis[i, :] = wfs_b[i, :, first_chan:first_chan+n_channels]
            wfs_a_bis += read_waveforms(spike_times_unit_a+two_units_shift, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
            wfs_b_bis += read_waveforms(spike_times_unit_b, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
        else:
            for i in range(wfs_a_bis.shape[0]):
                first_chan = int(mc - first_chan_a[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_a.shape[2]-n_channels, int(first_chan))
                wfs_a_bis[i] = wfs_a[i, :, first_chan:first_chan+n_channels]
                first_chan = int(mc - first_chan_b[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_b.shape[2]-n_channels, int(first_chan))
                wfs_b_bis[i, :-two_units_shift] = wfs_b[i, two_units_shift:, first_chan:first_chan+n_channels]
            wfs_a_bis += read_waveforms(spike_times_unit_a, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
            wfs_b_bis += read_waveforms(spike_times_unit_b+two_units_shift, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
    elif two_units_shift<0:
        if unit_shifted == unit_a:
            for i in range(wfs_a_bis.shape[0]):
                first_chan = int(mc - first_chan_a[i] - 5)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_a.shape[2]-n_channels, int(first_chan))
                wfs_a_bis[i, -two_units_shift:] = wfs_a[i, :two_units_shift, first_chan:first_chan+n_channels]
                first_chan = int(mc - first_chan_b[i] - 5)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_b.shape[2]-n_channels, int(first_chan))
                wfs_b_bis[i, :] = wfs_b[i, :, first_chan:first_chan+n_channels]
            wfs_a_bis += read_waveforms(spike_times_unit_a+two_units_shift, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
            wfs_b_bis += read_waveforms(spike_times_unit_b, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]

        else:
            for i in range(wfs_a_bis.shape[0]):
                first_chan = int(mc - first_chan_a[i] - 5)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_a.shape[2]-n_channels, int(first_chan))
                wfs_a_bis[i] = wfs_a[i, :, first_chan:first_chan+n_channels]
                first_chan = int(mc - first_chan_b[i] - 5)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_b.shape[2]-n_channels, int(first_chan))
                wfs_b_bis[i, -two_units_shift:] = wfs_b[i, :two_units_shift, first_chan:first_chan+n_channels]
            wfs_a_bis += read_waveforms(spike_times_unit_a, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
            wfs_b_bis += read_waveforms(spike_times_unit_b+two_units_shift, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
    else:
        for i in range(wfs_a_bis.shape[0]):
            first_chan = int(mc - first_chan_a[i] - 5)
            first_chan = max(0, int(first_chan))
            first_chan = min(wfs_a.shape[2]-n_channels, int(first_chan))
            wfs_a_bis[i] = wfs_a[i, :, first_chan:first_chan+n_channels]
            first_chan = int(mc - first_chan_b[i] - 5)
            first_chan = max(0, int(first_chan))
            first_chan = min(wfs_b.shape[2]-n_channels, int(first_chan))
            wfs_b_bis[i, :] = wfs_b[i, :, first_chan:first_chan+n_channels]
        wfs_a_bis += read_waveforms(spike_times_unit_a, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
        wfs_b_bis += read_waveforms(spike_times_unit_b, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]      
    
    
    tpca = PCA(rank_pca)
    wfs_diptest = np.concatenate((wfs_a, wfs_b))
    
    if nn_denoise:
        wfs_diptest = denoise_wf_nn_tmp_single_channel(wfs_diptest, denoiser, device)

    N, T, C = wfs_diptest.shape
    wfs_diptest = wfs_diptest.transpose(0, 2, 1).reshape(N*C, T)
    wfs_diptest = tpca.inverse_transform(tpca.fit_transform(wfs_diptest))
    wfs_diptest = wfs_diptest.reshape(N, C, T).transpose(0, 2, 1).reshape((N, C*T))
    
#     wfs_diptest = np.concatenate((wfs_a_bis, wfs_b_bis)).reshape((-1, n_channels*n_times))
    labels_diptest = np.zeros(wfs_a_bis.shape[0]+wfs_b_bis.shape[0])
    labels_diptest[:wfs_a_bis.shape[0]] = 1
    
    lda_model = LDA(n_components = 1)
    lda_comps = lda_model.fit_transform(wfs_diptest, labels_diptest)
    value_dpt, cut_calue = isocut(lda_comps[:, 0])
    return value_dpt

def get_merged_better(residual_path, waveforms, first_chans, geom_array, templates, n_spikes_templates, x_z_templates, n_templates, spike_index, labels, x, z, denoiser, device, n_channels=10, n_temp = 10, distance_threshold = 3., threshold_diptest = .75, rank_pca=8, nn_denoise = False):
     
    n_spikes_templates = get_n_spikes_templates(n_templates, labels)
    x_z_templates = get_x_z_templates(n_templates, labels, x, z)
    print("GET PROPOSED PAIRS")
    dist_argsort, dist_template = get_proposed_pairs(n_templates, templates, x_z_templates, n_temp = n_temp)
    
    labels_updated = labels.copy()
    reference_units = np.unique(labels)[1:]
    
    for unit in tqdm(range(n_templates)): #tqdm
        unit_reference = reference_units[unit]
        to_be_merged = [unit_reference]
        merge_shifts = [0]
        is_merged = False

        for j in range(n_temp):
            if dist_template[unit, j] < distance_threshold:
                unit_bis = dist_argsort[unit, j]
                unit_bis_reference = reference_units[unit_bis]
                if unit_reference != unit_bis_reference:
#                     ALIGN BASED ON MAX PTP TEMPLATE MC 
                    if templates[unit_reference].ptp(0).max() < templates[unit_bis_reference].ptp(0).max():
                        mc = templates[unit_bis_reference].ptp(0).argmax()
                        two_units_shift = templates[unit_reference, :, mc].argmin() - templates[unit_bis_reference, :, mc].argmin()
                        unit_shifted = unit_reference
                    else:
                        mc = templates[unit_reference].ptp(0).argmax()
                        two_units_shift = templates[unit_bis_reference, :, mc].argmin() - templates[unit_reference, :, mc].argmin()
                        unit_shifted = unit_bis_reference
                    dpt_val = get_diptest_value_better(residual_path, waveforms, first_chans, geom_array, spike_index, labels_updated, unit_reference, unit_bis_reference, n_spikes_templates, mc, two_units_shift,unit_shifted, denoiser, device, n_channels, rank_pca=rank_pca, nn_denoise = nn_denoise)
                    if dpt_val<threshold_diptest and np.abs(two_units_shift)<2:
                        to_be_merged.append(unit_bis_reference)
                        if unit_shifted == unit_bis_reference:
                            merge_shifts.append(-two_units_shift)
                        else:
                            merge_shifts.append(two_units_shift)
                        is_merged = True
        if is_merged:
            n_total_spikes = 0
            for unit_merged in np.unique(np.asarray(to_be_merged)):
                n_total_spikes += n_spikes_templates[unit_merged]

            new_reference_unit = np.unique(np.asarray(to_be_merged))[0]

            templates[new_reference_unit] = n_spikes_templates[new_reference_unit]*templates[new_reference_unit]/n_total_spikes
            cmp = 1
            for unit_merged in np.unique(np.asarray(to_be_merged))[1:]:
                shift_ = merge_shifts[cmp]
                templates[new_reference_unit] += n_spikes_templates[unit_merged]*np.roll(templates[unit_merged], shift_, axis = 0)/n_total_spikes
                n_spikes_templates[new_reference_unit] += n_spikes_templates[unit_merged]
                n_spikes_templates[unit_merged] = 0
                labels_updated[labels_updated == unit_merged] = new_reference_unit
                reference_units[unit_merged] = new_reference_unit
                cmp += 1
    return labels_updated


# def get_diptest_value(standardized_path, geom_array, spike_index, labels, unit_a, unit_b, n_spikes_templates, mc, two_units_shift, unit_shifted, denoiser, device, n_channels = 10, n_times=121):

#     # ALIGN BASED ON MAX PTP TEMPLATE MC 
#     n_channels_half = n_channels//2

#     n_wfs_max = int(min(500, min(n_spikes_templates[unit_a], n_spikes_templates[unit_b]))) 
#     if unit_b == unit_shifted:
#         spike_index_unit_a = spike_index[labels == unit_a, 0] #denoiser offset ## SHIFT BASED ON TEMPLATES ARGMIN PN MAX PTP TEMPLATE
#         spike_index_unit_b = spike_index[labels == unit_b, 0]+two_units_shift #denoiser offset
#     else:
#         spike_index_unit_a = spike_index[labels == unit_a, 0]+two_units_shift #denoiser offset ## SHIFT BASED ON TEMPLATES ARGMIN PN MAX PTP TEMPLATE
#         spike_index_unit_b = spike_index[labels == unit_b, 0] #denoiser offset
#     idx = np.random.choice(np.arange(spike_index_unit_a.shape[0]), n_wfs_max, replace = False)
#     spike_times_unit_a = spike_index_unit_a[idx]
#     idx = np.random.choice(np.arange(spike_index_unit_b.shape[0]), n_wfs_max, replace = False)
#     spike_times_unit_b = spike_index_unit_b[idx]
#     mc = min(384-n_channels_half, mc)
#     mc = max(n_channels_half, mc)
#     wfs_a = read_waveforms(spike_times_unit_a, standardized_path, geom_array, n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
#     wfs_b = read_waveforms(spike_times_unit_b, standardized_path, geom_array, n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
#     wfs_a = denoise_wf_nn_tmp_single_channel(wfs_a, denoiser, device)
#     wfs_b = denoise_wf_nn_tmp_single_channel(wfs_b, denoiser, device)
#     wfs_diptest = np.concatenate((wfs_a, wfs_b)).reshape((-1, n_channels*n_times))
#     labels_diptest = np.zeros(wfs_a.shape[0]+wfs_b.shape[0])
#     labels_diptest[:wfs_a.shape[0]] = 1
    
    
#     lda_model = LDA(n_components = 1)
#     lda_comps = lda_model.fit_transform(wfs_diptest, labels_diptest)
#     value_dpt, cut_calue = isocut(lda_comps[:, 0])
#     return value_dpt

    
# def get_merged(standardized_path, geom_array, n_templates, spike_index, labels, x, z, denoiser, device, n_channels=10, n_temp = 10, distance_threshold = 3., threshold_diptest = 1.):
     
#     templates = get_templates(standardized_path, geom_array, n_templates, spike_index, labels)
#     n_spikes_templates = get_n_spikes_templates(n_templates, labels)
#     x_z_templates = get_x_z_templates(n_templates, labels, x, z)
#     print("GET PROPOSED PAIRS")
#     dist_argsort, dist_template = get_proposed_pairs(n_templates, templates, x_z_templates, n_temp = n_temp)
    
#     labels_updated = labels.copy()
#     reference_units = np.unique(labels)[1:]
    
#     for unit in tqdm(range(n_templates)): #tqdm
#         unit_reference = reference_units[unit]
#         to_be_merged = [unit_reference]
#         merge_shifts = [0]
#         is_merged = False

#         for j in range(n_temp):
#             if dist_template[unit, j] < distance_threshold:
#                 unit_bis = dist_argsort[unit, j]
#                 unit_bis_reference = reference_units[unit_bis]
#                 if unit_reference != unit_bis_reference:
# #                     ALIGN BASED ON MAX PTP TEMPLATE MC 
#                     if templates[unit_reference].ptp(0).max() < templates[unit_bis_reference].ptp(0).max():
#                         mc = templates[unit_bis_reference].ptp(0).argmax()
#                         two_units_shift = templates[unit_reference, :, mc].argmin() - templates[unit_bis_reference, :, mc].argmin()
#                         unit_shifted = unit_reference
#                     else:
#                         mc = templates[unit_reference].ptp(0).argmax()
#                         two_units_shift = templates[unit_bis_reference, :, mc].argmin() - templates[unit_reference, :, mc].argmin()
#                         unit_shifted = unit_bis_reference
#                     dpt_val = get_diptest_value(standardized_path, geom_array, spike_index, labels_updated, unit_reference, unit_bis_reference, n_spikes_templates, mc, two_units_shift,unit_shifted, denoiser, device, n_channels)
#                     if dpt_val<threshold_diptest and np.abs(two_units_shift)<4:
#                         to_be_merged.append(unit_bis_reference)
#                         if unit_shifted == unit_bis_reference:
#                             merge_shifts.append(-two_units_shift)
#                         else:
#                             merge_shifts.append(two_units_shift)
#                         is_merged = True
#         if is_merged:
#             n_total_spikes = 0
#             for unit_merged in np.unique(np.asarray(to_be_merged)):
#                 n_total_spikes += n_spikes_templates[unit_merged]

#             new_reference_unit = np.unique(np.asarray(to_be_merged))[0]

#             templates[new_reference_unit] = n_spikes_templates[new_reference_unit]*templates[new_reference_unit]/n_total_spikes
#             cmp = 1
#             for unit_merged in np.unique(np.asarray(to_be_merged))[1:]:
#                 shift_ = merge_shifts[cmp]
#                 templates[new_reference_unit] += n_spikes_templates[unit_merged]*np.roll(templates[unit_merged], shift_, axis = 0)/n_total_spikes
#                 n_spikes_templates[new_reference_unit] += n_spikes_templates[unit_merged]
#                 n_spikes_templates[unit_merged] = 0
#                 labels_updated[labels_updated == unit_merged] = new_reference_unit
#                 reference_units[unit_merged] = new_reference_unit
#                 cmp += 1
#     return labels_updated

# def get_diptest_value_yass(residual_path, waveforms, spatial_cov, temporal_whitener, geom_array, spike_index, labels, unit_a, unit_b, n_spikes_templates, mc, two_units_shift, unit_shifted, denoiser, device, n_channels = 10, n_times=121):

#     # ALIGN BASED ON MAX PTP TEMPLATE MC 
#     n_channels_half = n_channels//2

#     n_wfs_max = int(min(250, min(n_spikes_templates[unit_a], n_spikes_templates[unit_b]))) 

    
#     mc = min(384-n_channels_half, mc)
#     mc = max(n_channels_half, mc)

#     spike_times_unit_a = spike_index[labels == unit_a, 0]
#     idx = np.random.choice(np.arange(spike_times_unit_a.shape[0]), n_wfs_max, replace = False)
#     spike_times_unit_a = spike_times_unit_a[idx]
#     wfs_a = waveforms[labels == unit_a][idx]
#     first_chan_a = triaged_first_chan[labels == unit_a][idx]
    
#     spike_times_unit_b = spike_index[labels == unit_b, 0]
#     idx = np.random.choice(np.arange(spike_times_unit_b.shape[0]), n_wfs_max, replace = False)
#     spike_times_unit_b = spike_times_unit_b[idx]
#     wfs_b = waveforms[labels == unit_b][idx]
#     first_chan_b = triaged_first_chan[labels == unit_b][idx]
    
#     wfs_a_bis = np.zeros((wfs_a.shape[0], n_times, n_channels))
#     wfs_b_bis = np.zeros((wfs_b.shape[0], n_times, n_channels))
    
#     if two_units_shift>0:
        
#         if unit_shifted == unit_a:
#             for i in range(wfs_a_bis.shape[0]):
#                 first_chan = int(mc - first_chan_a[i] - 5)
#                 first_chan = min(384-n_channels, first_chan)
#                 first_chan = max(n_channels, first_chan)
#                 wfs_a_bis[i, :-two_units_shift] = wfs_a[i, two_units_shift:, first_chan:first_chan+n_channels]
#                 first_chan = int(mc - first_chan_b[i] - 5)
#                 first_chan = min(384-n_channels, first_chan)
#                 first_chan = max(n_channels, first_chan)
#                 wfs_b_bis[i, :] = wfs_b[i, :, first_chan:first_chan+n_channels]
#             wfs_a_bis += read_waveforms(spike_times_unit_a+two_units_shift, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
#             wfs_b_bis += read_waveforms(spike_times_unit_b, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
#         else:
#             for i in range(wfs_a_bis.shape[0]):
#                 first_chan = int(mc - first_chan_a[i] - 5)
#                 first_chan = min(384-n_channels, first_chan)
#                 first_chan = max(n_channels, first_chan)
#                 wfs_a_bis[i] = wfs_a[i, :, first_chan:first_chan+n_channels]
#                 first_chan = int(mc - first_chan_b[i] - 5)
#                 first_chan = min(384-n_channels, first_chan)
#                 first_chan = max(n_channels, first_chan)
#                 wfs_b_bis[i, :-two_units_shift] = wfs_b[i, two_units_shift:, first_chan:first_chan+n_channels]
#             wfs_a_bis += read_waveforms(spike_times_unit_a, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
#             wfs_b_bis += read_waveforms(spike_times_unit_b+two_units_shift, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
#     elif two_units_shift<0:
#         if unit_shifted == unit_a:
#             for i in range(wfs_a_bis.shape[0]):
#                 first_chan = int(mc - first_chan_a[i] - 5)
#                 first_chan = min(384-n_channels, first_chan)
#                 first_chan = max(n_channels, first_chan)
#                 wfs_a_bis[i, -two_units_shift:] = wfs_a[i, :two_units_shift, first_chan:first_chan+n_channels]
#                 first_chan = int(mc - first_chan_b[i] - 5)
#                 first_chan = min(384-n_channels, first_chan)
#                 first_chan = max(n_channels, first_chan)
#                 wfs_b_bis[i, :] = wfs_b[i, :, first_chan:first_chan+n_channels]
#             wfs_a_bis += read_waveforms(spike_times_unit_a, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
#             wfs_b_bis += read_waveforms(spike_times_unit_b-two_units_shift, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]

#         else:
#             for i in range(wfs_a_bis.shape[0]):
#                 first_chan = int(mc - first_chan_a[i] - 5)
#                 first_chan = min(384-n_channels, first_chan)
#                 first_chan = max(n_channels, first_chan)
#                 wfs_a_bis[i] = wfs_a[i, :, first_chan:first_chan+n_channels]
#                 first_chan = int(mc - first_chan_b[i] - 5)
#                 first_chan = min(384-n_channels, first_chan)
#                 first_chan = max(n_channels, first_chan)
#                 wfs_b_bis[i, -two_units_shift:] = wfs_b[i, :two_units_shift, first_chan:first_chan+n_channels]
#             wfs_a_bis += read_waveforms(spike_times_unit_a-two_units_shift, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
#             wfs_b_bis += read_waveforms(spike_times_unit_b, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
#     else:
#         for i in range(wfs_a_bis.shape[0]):
#             first_chan = int(mc - first_chan_a[i] - 5)
#             wfs_a_bis[i] = wfs_a[i, :, first_chan:first_chan+n_channels]
#             first_chan = int(mc - first_chan_b[i] - 5)
#             wfs_b_bis[i, :] = wfs_b[i, :, first_chan:first_chan+n_channels]
#         wfs_a_bis += read_waveforms(spike_times_unit_a, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
#         wfs_b_bis += read_waveforms(spike_times_unit_b, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]

#     spatial_whitener = get_spatial_whitener(spatial_cov, np.arange(mc-n_channels_half, mc+n_channels_half), geom_array)

#     wfs1_w = np.matmul(wfs_a_bis, spatial_whitener)
#     wfs2_w = np.matmul(wfs_b_bis, spatial_whitener)
#     wfs1_w = np.matmul(wfs1_w.transpose(0,2,1),
#                       temporal_whitener).transpose(0,2,1)
#     wfs2_w = np.matmul(wfs2_w.transpose(0,2,1),
#                       temporal_whitener).transpose(0,2,1)

#     temp_diff_w = np.mean(wfs1_w, 0) - np.mean(wfs2_w,0)
#     c_w = np.sum(0.5*(np.mean(wfs1_w, 0) + np.mean(wfs2_w,0))*temp_diff_w)
#     dat1_w = np.sum(wfs1_w*temp_diff_w, (1,2))
#     dat2_w = np.sum(wfs2_w*temp_diff_w, (1,2))
#     dat_all = np.hstack((dat1_w, dat2_w))

#     labels_diptest = np.zeros(wfs_a.shape[0]+wfs_b.shape[0])
#     labels_diptest[:wfs_a.shape[0]] = 1
    
#     lda_model = LDA(n_components = 1)
#     lda_comps = lda_model.fit_transform(dat_all.reshape(-1, 1), labels_diptest)
#     value_dpt, cut_calue = isocut(lda_comps[:, 0])
#     return value_dpt




# def get_merged_yass(standardized_path, residual_path, waveforms, spatial_cov, temporal_whitener, geom_array, n_templates, spike_index, labels, x, z, denoiser, device, n_channels=10, n_temp = 10, distance_threshold = 3., threshold_diptest = 0.75):
     
#     templates = get_templates(standardized_path, geom_array, n_templates, spike_index, labels)
#     n_spikes_templates = get_n_spikes_templates(n_templates, labels)
#     x_z_templates = get_x_z_templates(n_templates, labels, x, z)
#     print("GET PROPOSED PAIRS")
#     dist_argsort, dist_template = get_proposed_pairs(n_templates, templates, x_z_templates, n_temp = n_temp)
    
#     labels_updated = labels.copy()
#     reference_units = np.unique(labels)[1:]
    
#     for unit in tqdm(range(n_templates)): #tqdm
#         unit_reference = reference_units[unit]
#         to_be_merged = [unit_reference]
#         merge_shifts = [0]
#         is_merged = False

#         for j in range(n_temp):
#             if dist_template[unit, j] < distance_threshold:
#                 unit_bis = dist_argsort[unit, j]
#                 unit_bis_reference = reference_units[unit_bis]
#                 if unit_reference != unit_bis_reference:
# #                     ALIGN BASED ON MAX PTP TEMPLATE MC 
#                     if templates[unit_reference].ptp(0).max() < templates[unit_bis_reference].ptp(0).max():
#                         mc = templates[unit_bis_reference].ptp(0).argmax()
#                         two_units_shift = templates[unit_reference, :, mc].argmin() - templates[unit_bis_reference, :, mc].argmin()
#                         unit_shifted = unit_reference
#                     else:
#                         mc = templates[unit_reference].ptp(0).argmax()
#                         two_units_shift = templates[unit_bis_reference, :, mc].argmin() - templates[unit_reference, :, mc].argmin()
#                         unit_shifted = unit_bis_reference
#                     if np.abs(two_units_shift)<4:
#                         dpt_val = get_diptest_value_yass(residual_path, waveforms, spatial_cov, temporal_whitener, geom_array, spike_index, labels_updated, unit_reference, unit_bis_reference, n_spikes_templates, mc, two_units_shift,unit_shifted, denoiser, device, n_channels)
#                         if dpt_val<threshold_diptest:
#                             to_be_merged.append(unit_bis_reference)
#                             if unit_shifted == unit_bis_reference:
#                                 merge_shifts.append(-two_units_shift)
#                             else:
#                                 merge_shifts.append(two_units_shift)
#                             is_merged = True
#         if is_merged:
#             n_total_spikes = 0
#             for unit_merged in np.unique(np.asarray(to_be_merged)):
#                 n_total_spikes += n_spikes_templates[unit_merged]

#             new_reference_unit = np.unique(np.asarray(to_be_merged))[0]

#             templates[new_reference_unit] = n_spikes_templates[new_reference_unit]*templates[new_reference_unit]/n_total_spikes
#             cmp = 1
#             for unit_merged in np.unique(np.asarray(to_be_merged))[1:]:
#                 shift_ = merge_shifts[cmp]
#                 templates[new_reference_unit] += n_spikes_templates[unit_merged]*np.roll(templates[unit_merged], shift_, axis = 0)/n_total_spikes
#                 n_spikes_templates[new_reference_unit] += n_spikes_templates[unit_merged]
#                 n_spikes_templates[unit_merged] = 0
#                 labels_updated[labels_updated == unit_merged] = new_reference_unit
#                 reference_units[unit_merged] = new_reference_unit
#                 cmp += 1
#     return labels_updated

# %%
# from scipy.spatial.distance import pdist, squareform
# import random


# def read_data(bin_file, dtype, s_start, s_end, n_channels):
#     """Read a chunk of a binary file"""
#     offset = s_start * np.dtype(dtype).itemsize * n_channels
#     with open(bin_file, "rb") as fin:
#         data = np.fromfile(
#             fin,
#             dtype=dtype,
#             count=(s_end - s_start) * n_channels,
#             offset=offset,
#         )
#     data = data.reshape(-1, n_channels)
#     return data


# def kill_signal(recordings, threshold, window_size):
#     """
#     Thresholds recordings, values above 'threshold' are considered signal
#     (set to 0), a window of size 'window_size' is drawn around the signal
#     points and those observations are also killed
#     Returns
#     -------
#     recordings: numpy.ndarray
#         The modified recordings with values above the threshold set to 0
#     is_noise_idx: numpy.ndarray
#         A boolean array with the same shap as 'recordings' indicating if the
#         observation is noise (1) or was killed (0).
#     """
#     recordings = np.copy(recordings)

#     T, C = recordings.shape
#     R = int((window_size-1)/2)

#     # this will hold a flag 1 (noise), 0 (signal) for every obseration in the
#     # recordings
#     is_noise_idx = np.zeros((T, C))

#     # go through every neighboring channel
#     for c in range(C):

#         # get obserations where observation is above threshold
#         idx_temp = np.where(np.abs(recordings[:, c]) > threshold)[0]

#         if len(idx_temp) == 0:
#             is_noise_idx[:, c] = 1
#             continue

#         # shift every index found
#         for j in range(-R, R+1):

#             # shift
#             idx_temp2 = idx_temp + j

#             # remove indexes outside range [0, T]
#             idx_temp2 = idx_temp2[np.logical_and(idx_temp2 >= 0,
#                                                  idx_temp2 < T)]

#             # set surviving indexes to nan
#             recordings[idx_temp2, c] = np.nan

#         # noise indexes are the ones that are not nan
#         # FIXME: compare to np.nan instead
#         is_noise_idx_temp = (recordings[:, c] == recordings[:, c])

#         # standarize data, ignoring nans
#         recordings[:, c] = recordings[:, c]/np.nanstd(recordings[:, c])

#         # set non noise indexes to 0 in the recordings
#         recordings[~is_noise_idx_temp, c] = 0

#         # save noise indexes
#         is_noise_idx[is_noise_idx_temp, c] = 1

#     return recordings, is_noise_idx

# def search_noise_snippets(recordings, is_noise_idx, sample_size,
#                           temporal_size, channel_choices=None,
#                           max_trials_per_sample=100,
#                           allow_smaller_sample_size=False):
#     """
#     Randomly search noise snippets of 'temporal_size'
#     Parameters
#     ----------
#     channel_choices: list
#         List of sets of channels to select at random on each trial
#     max_trials_per_sample: int, optional
#         Maximum random trials per sample
#     allow_smaller_sample_size: bool, optional
#         If 'max_trials_per_sample' is reached and this is True, the noise
#         snippets found up to that time are returned
#     Raises
#     ------
#     ValueError
#         if after 'max_trials_per_sample' trials, no noise snippet has been
#         found this exception is raised
#     Notes
#     -----
#     Channels selected at random using the random module from the standard
#     library (not using np.random)
#     """

#     T, C = recordings.shape

#     if channel_choices is None:
#         noise_wf = np.zeros((sample_size, temporal_size))
#     else:
#         lenghts = set([len(ch) for ch in channel_choices])

#         if len(lenghts) > 1:
#             raise ValueError('All elements in channel_choices must have '
#                              'the same length, got {}'.format(lenghts))

#         n_channels = len(channel_choices[0])
#         noise_wf = np.zeros((sample_size, temporal_size, n_channels))

#     count = 0


#     trial = 0

#     # repeat until you get sample_size noise snippets
#     while count < sample_size:

#         # random number for the start of the noise snippet
#         t_start = np.random.randint(T-temporal_size)

#         if channel_choices is None:
#             # random channel
#             ch = random.randint(0, C - 1)
#         else:
#             ch = random.choice(channel_choices)

#         t_slice = slice(t_start, t_start+temporal_size)

#         # get a snippet from the recordings and the noise flags for the same
#         # location
#         snippet = recordings[t_slice, ch]
#         snipped_idx_noise = is_noise_idx[t_slice, ch]

#         # check if all observations in snippet are noise
#         if snipped_idx_noise.all():
#             # add the snippet and increase count
#             noise_wf[count] = snippet
#             count += 1
#             trial = 0

#         trial += 1

#         if trial == max_trials_per_sample:
#             if allow_smaller_sample_size:
#                 return noise_wf[:count]
#             else:
#                 raise ValueError("Couldn't find snippet {} of size {} after "
#                                  "{} iterations (only {} found)"
#                                  .format(count + 1, temporal_size,
#                                          max_trials_per_sample,
#                                          count))

#     return noise_wf


# def get_noise_covariance(raw_data, geom_array, dtype = 'float32', n_channels = 384, rec_len = 60, sampling_rate = 30000, spike_size = 121):
#     rec_len = rec_len*sampling_rate
#     # get data chunk
#     chunk_5sec = 5*sampling_rate
#     if rec_len < chunk_5sec:
#         chunk_5sec = rec_len
#     small_batch = read_data(raw_data, dtype,
#                 rec_len//2 - chunk_5sec//2,
#                 rec_len//2 + chunk_5sec//2, n_channels = n_channels)
    

#     # get noise floor of recording
#     noised_killed, is_noise_idx = kill_signal(small_batch, 3, spike_size)
#     print ("small_batch: ", small_batch.shape, ", noised_killed: ", noised_killed.shape)
#     # spatial covariance
#     spatial_cov_all = np.divide(np.matmul(noised_killed.T, noised_killed),
#                         np.matmul(is_noise_idx.T, is_noise_idx))
#     sig = np.sqrt(np.diag(spatial_cov_all))
#     sig[sig == 0] = 1
#     spatial_cov_all = spatial_cov_all/(sig[:,None]*sig[None])
# #     chan_dist = squareform(pdist(geom_array))
# #     chan_dist_unique = np.unique(chan_dist)
# #     cov_by_dist = np.zeros(len(chan_dist_unique))
# #     for ii, d in enumerate(chan_dist_unique):
# #         cov_by_dist[ii] = np.mean(spatial_cov_all[chan_dist == d])
# #     dist_in = cov_by_dist > 0.1
# #     chan_dist_unique = chan_dist_unique[dist_in]
# #     cov_by_dist = cov_by_dist[dist_in]
# #     spatial_cov = np.vstack((cov_by_dist, chan_dist_unique)).T

#     # get noise snippets
#     noise_wf = search_noise_snippets(
#                     noised_killed, is_noise_idx, 1000,
#                     spike_size,
#                     channel_choices=None,
#                     max_trials_per_sample=100,
#                     allow_smaller_sample_size=True)

#     # get temporal covariance
#     temp_cov = np.cov(noise_wf.T)
#     sig = np.sqrt(np.diag(temp_cov))
#     temp_cov = temp_cov/(sig[:,None]*sig[None])


#     return spatial_cov_all, temp_cov

# def get_spatial_whitener(spatial_cov, vis_chan, geom_array):

#     chan_dist = squareform(pdist(geom_array[vis_chan]))
#     spat_cov = np.zeros((len(vis_chan), len(vis_chan)))
#     for ii, c in enumerate(spatial_cov[:,1]):
#         spat_cov[chan_dist == c] = spatial_cov[ii, 0]

#     w, v = np.linalg.eig(spat_cov)
#     w[w<=0] = 1E-10
#     inv_half_spat_cov = np.matmul(np.matmul(v, np.diag(1/np.sqrt(w))), v.T)

#     return inv_half_spat_cov

# def get_temporal_whitener(temporal_cov):
#     w, v = np.linalg.eig(temporal_cov)
#     return np.matmul(np.matmul(v, np.diag(1/np.sqrt(w))), v.T)
        
# spatial_cov, temporal_cov = get_noise_covariance(residual_data_bin, geom_array)
# temporal_whitener = get_temporal_whitener(temporal_cov)

# %%
# labels_merged_yass = get_merged_yass(raw_data_bin, residual_data_bin, triaged_sub_wfs, spatial_cov, temporal_whitener, geom_array, 
#                                      ordered_split_labels.max()+1, spike_index_triaged, ordered_split_labels, triaged_x, triaged_z, denoiser, device)

# %%
templates_split = get_templates(raw_data_bin, geom_array, ordered_split_labels.max()+1, triaged_spike_index, ordered_split_labels)
n_spikes_templates = get_n_spikes_templates(templates_split.shape[0], ordered_split_labels)
x_z_templates = get_x_z_templates(templates_split.shape[0], ordered_split_labels, triaged_x, triaged_z)


# %%
labels_merged = get_merged_better(residual_data_bin, triaged_sub_wfs, triaged_first_chan, geom_array, 
                                  templates_split, n_spikes_templates, x_z_templates, templates_split.shape[0], spike_index_triaged, 
                                  ordered_split_labels, triaged_x, triaged_z, denoiser, device, distance_threshold = 1., 
                                  threshold_diptest = .5, rank_pca=8, nn_denoise = True)
     




# %%

# %%
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


# %%
ordered_merged_labels.max()

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
for cluster_id in np.unique(ordered_merged_labels):
    cluster_color_dict[cluster_id] = unique_colors[cluster_id % len(unique_colors)]
cluster_color_dict[-1] = '#808080' #set outlier color to grey

# %%
fig = plot_array_scatter(ordered_merged_labels, geom_array, triaged_x, triaged_z, triaged_maxptps, cluster_color_dict, color_arr, min_cluster_size=clusterer.min_cluster_size, min_samples=clusterer.min_samples, 
                         z_cutoff=(0, 3900), figsize=(18, 24))
# fig.suptitle(f'x,z,scaled_logptp features," {num_spikes} datapoints');
# fig.savefig('yass_merge.png')

# %%
# from scipy.spatial.distance import pdist, squareform
# import random


# def read_data(bin_file, dtype, s_start, s_end, n_channels):
#     """Read a chunk of a binary file"""
#     offset = s_start * np.dtype(dtype).itemsize * n_channels
#     with open(bin_file, "rb") as fin:
#         data = np.fromfile(
#             fin,
#             dtype=dtype,
#             count=(s_end - s_start) * n_channels,
#             offset=offset,
#         )
#     data = data.reshape(-1, n_channels)
#     return data


# def kill_signal(recordings, threshold, window_size):
#     """
#     Thresholds recordings, values above 'threshold' are considered signal
#     (set to 0), a window of size 'window_size' is drawn around the signal
#     points and those observations are also killed
#     Returns
#     -------
#     recordings: numpy.ndarray
#         The modified recordings with values above the threshold set to 0
#     is_noise_idx: numpy.ndarray
#         A boolean array with the same shap as 'recordings' indicating if the
#         observation is noise (1) or was killed (0).
#     """
#     recordings = np.copy(recordings)

#     T, C = recordings.shape
#     R = int((window_size-1)/2)

#     # this will hold a flag 1 (noise), 0 (signal) for every obseration in the
#     # recordings
#     is_noise_idx = np.zeros((T, C))

#     # go through every neighboring channel
#     for c in range(C):

#         # get obserations where observation is above threshold
#         idx_temp = np.where(np.abs(recordings[:, c]) > threshold)[0]

#         if len(idx_temp) == 0:
#             is_noise_idx[:, c] = 1
#             continue

#         # shift every index found
#         for j in range(-R, R+1):

#             # shift
#             idx_temp2 = idx_temp + j

#             # remove indexes outside range [0, T]
#             idx_temp2 = idx_temp2[np.logical_and(idx_temp2 >= 0,
#                                                  idx_temp2 < T)]

#             # set surviving indexes to nan
#             recordings[idx_temp2, c] = np.nan

#         # noise indexes are the ones that are not nan
#         # FIXME: compare to np.nan instead
#         is_noise_idx_temp = (recordings[:, c] == recordings[:, c])

#         # standarize data, ignoring nans
#         recordings[:, c] = recordings[:, c]/np.nanstd(recordings[:, c])

#         # set non noise indexes to 0 in the recordings
#         recordings[~is_noise_idx_temp, c] = 0

#         # save noise indexes
#         is_noise_idx[is_noise_idx_temp, c] = 1

#     return recordings, is_noise_idx

# def search_noise_snippets(recordings, is_noise_idx, sample_size,
#                           temporal_size, channel_choices=None,
#                           max_trials_per_sample=100,
#                           allow_smaller_sample_size=False):
#     """
#     Randomly search noise snippets of 'temporal_size'
#     Parameters
#     ----------
#     channel_choices: list
#         List of sets of channels to select at random on each trial
#     max_trials_per_sample: int, optional
#         Maximum random trials per sample
#     allow_smaller_sample_size: bool, optional
#         If 'max_trials_per_sample' is reached and this is True, the noise
#         snippets found up to that time are returned
#     Raises
#     ------
#     ValueError
#         if after 'max_trials_per_sample' trials, no noise snippet has been
#         found this exception is raised
#     Notes
#     -----
#     Channels selected at random using the random module from the standard
#     library (not using np.random)
#     """

#     T, C = recordings.shape

#     if channel_choices is None:
#         noise_wf = np.zeros((sample_size, temporal_size))
#     else:
#         lenghts = set([len(ch) for ch in channel_choices])

#         if len(lenghts) > 1:
#             raise ValueError('All elements in channel_choices must have '
#                              'the same length, got {}'.format(lenghts))

#         n_channels = len(channel_choices[0])
#         noise_wf = np.zeros((sample_size, temporal_size, n_channels))

#     count = 0


#     trial = 0

#     # repeat until you get sample_size noise snippets
#     while count < sample_size:

#         # random number for the start of the noise snippet
#         t_start = np.random.randint(T-temporal_size)

#         if channel_choices is None:
#             # random channel
#             ch = random.randint(0, C - 1)
#         else:
#             ch = random.choice(channel_choices)

#         t_slice = slice(t_start, t_start+temporal_size)

#         # get a snippet from the recordings and the noise flags for the same
#         # location
#         snippet = recordings[t_slice, ch]
#         snipped_idx_noise = is_noise_idx[t_slice, ch]

#         # check if all observations in snippet are noise
#         if snipped_idx_noise.all():
#             # add the snippet and increase count
#             noise_wf[count] = snippet
#             count += 1
#             trial = 0

#         trial += 1

#         if trial == max_trials_per_sample:
#             if allow_smaller_sample_size:
#                 return noise_wf[:count]
#             else:
#                 raise ValueError("Couldn't find snippet {} of size {} after "
#                                  "{} iterations (only {} found)"
#                                  .format(count + 1, temporal_size,
#                                          max_trials_per_sample,
#                                          count))

#     return noise_wf


# def get_noise_covariance(raw_data, geom_array, dtype = 'float32', n_channels = 384, rec_len = 60, sampling_rate = 30000, spike_size = 121):
#     rec_len = rec_len*sampling_rate
#     # get data chunk
#     chunk_5sec = 5*sampling_rate
#     if rec_len < chunk_5sec:
#         chunk_5sec = rec_len
#     small_batch = read_data(raw_data, dtype,
#                 rec_len//2 - chunk_5sec//2,
#                 rec_len//2 + chunk_5sec//2, n_channels = n_channels)
    

#     # get noise floor of recording
#     noised_killed, is_noise_idx = kill_signal(small_batch, 3, spike_size)
#     print ("small_batch: ", small_batch.shape, ", noised_killed: ", noised_killed.shape)
#     # spatial covariance
#     spatial_cov_all = np.divide(np.matmul(noised_killed.T, noised_killed),
#                         np.matmul(is_noise_idx.T, is_noise_idx))
#     sig = np.sqrt(np.diag(spatial_cov_all))
#     sig[sig == 0] = 1
#     spatial_cov_all = spatial_cov_all/(sig[:,None]*sig[None])
# #     chan_dist = squareform(pdist(geom_array))
# #     chan_dist_unique = np.unique(chan_dist)
# #     cov_by_dist = np.zeros(len(chan_dist_unique))
# #     for ii, d in enumerate(chan_dist_unique):
# #         cov_by_dist[ii] = np.mean(spatial_cov_all[chan_dist == d])
# #     dist_in = cov_by_dist > 0.1
# #     chan_dist_unique = chan_dist_unique[dist_in]
# #     cov_by_dist = cov_by_dist[dist_in]
# #     spatial_cov = np.vstack((cov_by_dist, chan_dist_unique)).T

#     # get noise snippets
#     noise_wf = search_noise_snippets(
#                     noised_killed, is_noise_idx, 1000,
#                     spike_size,
#                     channel_choices=None,
#                     max_trials_per_sample=100,
#                     allow_smaller_sample_size=True)

#     # get temporal covariance
#     temp_cov = np.cov(noise_wf.T)
#     sig = np.sqrt(np.diag(temp_cov))
#     temp_cov = temp_cov/(sig[:,None]*sig[None])


#     return spatial_cov_all, temp_cov

# def get_spatial_whitener(spatial_cov, vis_chan, geom_array):

#     chan_dist = squareform(pdist(geom_array[vis_chan]))
#     spat_cov = np.zeros((len(vis_chan), len(vis_chan)))
#     for ii, c in enumerate(spatial_cov[:,1]):
#         spat_cov[chan_dist == c] = spatial_cov[ii, 0]

#     w, v = np.linalg.eig(spat_cov)
#     w[w<=0] = 1E-10
#     inv_half_spat_cov = np.matmul(np.matmul(v, np.diag(1/np.sqrt(w))), v.T)

#     return inv_half_spat_cov

# def get_temporal_whitener(temporal_cov):
#     w, v = np.linalg.eig(temporal_cov)
#     return np.matmul(np.matmul(v, np.diag(1/np.sqrt(w))), v.T)
        
# spatial_cov, temporal_cov = get_noise_covariance(residual_data_bin, geom_array)
# temporal_whitener = get_temporal_whitener(temporal_cov)


# %%
# spatial_cov, temporal_cov = get_noise_covariance(residual_data_bin, geom_array)
# temporal_whitener = get_temporal_whitener(temporal_cov)

# %%
templates_merged = get_templates(raw_data_bin, geom_array, ordered_merged_labels.max()+1, triaged_spike_index, ordered_merged_labels)
n_spikes_templates = get_n_spikes_templates(templates_merged.shape[0], ordered_merged_labels)
x_z_templates = get_x_z_templates(templates_merged.shape[0], ordered_merged_labels, triaged_x, triaged_z)
templates_merged[94, :, templates_merged[94].ptp(0).argmax()].argmin()

# %%
unit_a = 309
unit_b = 310

# %%
mc = templates_merged[unit_a].ptp(0).argmax()
templates_merged[unit_a, :, mc].argmin()

# %%
templates_merged[unit_a, :, mc].argmin()-templates_merged[unit_b, :, mc].argmin()

# %%
plt.figure(figsize = (20, 2.5))
plt.plot(templates_merged[unit_a, :, mc-5:mc+5].T.flatten())
plt.plot(templates_merged[unit_b, :, mc-5:mc+5].T.flatten())
plt.show()

# %%
n_channels = 10
n_channels_half = n_channels//2

wfs_a = triaged_sub_wfs[ordered_merged_labels == unit_a]
wfs_b = triaged_sub_wfs[ordered_merged_labels == unit_b]
first_chan_a = triaged_first_chan[ordered_merged_labels == unit_a]
first_chan_b = triaged_first_chan[ordered_merged_labels == unit_b]

wfs_a_bis = np.zeros((wfs_a.shape[0], 121, n_channels))
wfs_b_bis = np.zeros((wfs_b.shape[0], 121, n_channels))
for i in range(wfs_a_bis.shape[0]):
    first_chan = int(mc - first_chan_a[i] - 5)
    wfs_a_bis[i] = wfs_a[i, :, first_chan:first_chan+n_channels]
    first_chan = int(mc - first_chan_b[i] - 5)
    wfs_b_bis[i] = wfs_b[i, :, first_chan:first_chan+n_channels]


spike_index_unit_a = triaged_spike_index[ordered_merged_labels == unit_a, 0]#denoiser offset ## SHIFT BASED ON TEMPLATES ARGMIN PN MAX PTP TEMPLATE
spike_index_unit_b = triaged_spike_index[ordered_merged_labels == unit_b, 0]-4

n_wfs_max = int(min(250, min(n_spikes_templates[unit_a], n_spikes_templates[unit_b]))) 
idx = np.random.choice(np.arange(spike_index_unit_a.shape[0]), n_wfs_max, replace = False)
spike_times_unit_a = spike_index_unit_a[idx]
wfs_a = wfs_a[idx]
first_chan_a = first_chan_a[idx]
idx = np.random.choice(np.arange(spike_index_unit_b.shape[0]), n_wfs_max, replace = False)
spike_times_unit_b = spike_index_unit_b[idx]
wfs_b = wfs_b[idx]
first_chan_b = first_chan_b[idx]

mc = min(384-n_channels_half, mc)
mc = max(n_channels_half, mc)

wfs_a_bis = np.zeros((wfs_a.shape[0], 121, 10))
wfs_b_bis = np.zeros((wfs_b.shape[0], 121, 10))
for i in range(wfs_a_bis.shape[0]):
    first_chan = int(mc - first_chan_a[i] - 5)
    wfs_a_bis[i] = wfs_a[i, :, first_chan:first_chan+10]
    first_chan = int(mc - first_chan_b[i] - 5)
    wfs_b_bis[i] = wfs_b[i, :, first_chan:first_chan+10]

wfs_a = wfs_a_bis + read_waveforms(spike_times_unit_a, residual_data_bin, geom_array, n_times=121, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
wfs_b = wfs_b_bis + read_waveforms(spike_times_unit_b, residual_data_bin, geom_array, n_times=121, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
wfs_a_denoised = denoise_wf_nn_tmp_single_channel(wfs_a, denoiser, device)
wfs_b_denoised = denoise_wf_nn_tmp_single_channel(wfs_b, denoiser, device)

wfs_diptest = np.concatenate((wfs_a_denoised, wfs_b_denoised)).reshape((-1, n_channels*121))
labels_diptest = np.zeros(wfs_a.shape[0]+wfs_b.shape[0])
labels_diptest[:wfs_a.shape[0]] = 1


lda_model = LDA(n_components = 1)
lda_comps = lda_model.fit_transform(wfs_diptest, labels_diptest)
value_dpt, cut_calue = isocut(lda_comps[:, 0])

# %%
value_dpt

# %%
plt.hist(lda_comps, bins = 20)
plt.show()

# %%
tpca = PCA(3)
cleaned = np.concatenate((wfs_a, wfs_b))
N, T, C = cleaned.shape
cleaned = cleaned.transpose(0, 2, 1).reshape(N*C, T)
cleaned = tpca.inverse_transform(tpca.fit_transform(cleaned))
cleaned = cleaned.reshape(N, C, T) #.transpose(0, 2, 1)

plt.figure(figsize = (20, 2.5))
for i in range(cleaned.shape[0]//2):
    plt.plot(cleaned[i].flatten(), c='red', alpha = 0.05)
for i in range(cleaned.shape[0]//2):
    plt.plot(cleaned[cleaned.shape[0]//2+i].flatten(), c='blue', alpha = 0.05)
for j in range(9):
    plt.axvline(121+j*121, c='black')
plt.show()
    
wfs_diptest = cleaned.reshape((N, C*T))
labels_diptest = np.zeros(wfs_a.shape[0]+wfs_b.shape[0])
labels_diptest[:wfs_a.shape[0]] = 1


lda_model = LDA(n_components = 1)
lda_comps = lda_model.fit_transform(wfs_diptest, labels_diptest)
value_dpt, cut_calue = isocut(lda_comps[:, 0])

print(value_dpt)
plt.hist(lda_comps, bins = 20)
plt.show()


# %%
tpca = PCA(5)
cleaned = np.concatenate((wfs_a, wfs_b))
N, T, C = cleaned.shape
cleaned = cleaned.transpose(0, 2, 1).reshape(N*C, T)
cleaned = tpca.inverse_transform(tpca.fit_transform(cleaned))
cleaned = cleaned.reshape(N, C, T) #.transpose(0, 2, 1)

plt.figure(figsize = (20, 2.5))
for i in range(cleaned.shape[0]//2):
    plt.plot(cleaned[i].flatten(), c='red', alpha = 0.05)
for i in range(cleaned.shape[0]//2):
    plt.plot(cleaned[cleaned.shape[0]//2+i].flatten(), c='blue', alpha = 0.05)
for j in range(9):
    plt.axvline(121+j*121, c='black')
plt.show()
    
wfs_diptest = cleaned.reshape((N, C*T))
labels_diptest = np.zeros(wfs_a.shape[0]+wfs_b.shape[0])
labels_diptest[:wfs_a.shape[0]] = 1


lda_model = LDA(n_components = 1)
lda_comps = lda_model.fit_transform(wfs_diptest, labels_diptest)
value_dpt, cut_calue = isocut(lda_comps[:, 0])

print(value_dpt)
plt.hist(lda_comps, bins = 20)
plt.show()

# %%
tpca = PCA(8)
cleaned = np.concatenate((wfs_a, wfs_b))
N, T, C = cleaned.shape
cleaned = cleaned.transpose(0, 2, 1).reshape(N*C, T)
cleaned = tpca.inverse_transform(tpca.fit_transform(cleaned))
cleaned = cleaned.reshape(N, C, T) #.transpose(0, 2, 1)

plt.figure(figsize = (20, 2.5))
for i in range(cleaned.shape[0]//2):
    plt.plot(cleaned[i].flatten(), c='red', alpha = 0.05)
# for i in range(cleaned.shape[0]//2):
#     plt.plot(cleaned[cleaned.shape[0]//2+i].flatten(), c='blue', alpha = 0.05)
for j in range(9):
    plt.axvline(121+j*121, c='black')
plt.show()
    
wfs_diptest = cleaned.reshape((N, C*T))
labels_diptest = np.zeros(wfs_a.shape[0]+wfs_b.shape[0])
labels_diptest[:wfs_a.shape[0]] = 1


lda_model = LDA(n_components = 1)
lda_comps = lda_model.fit_transform(wfs_diptest, labels_diptest)
value_dpt, cut_calue = isocut(lda_comps[:, 0])

print(value_dpt)
plt.hist(lda_comps, bins = 20)
plt.show()

# %%

plt.figure(figsize = (20, 2.5))
for i in range(wfs_a_denoised.shape[0]//2):
    plt.plot(wfs_a_denoised[i].T.flatten(), c='red', alpha = 0.05)
# for i in range(wfs_a_denoised.shape[0]//2):
#     plt.plot(wfs_b_denoised[i].T.flatten(), c='blue', alpha = 0.05)
for j in range(9):
    plt.axvline(121+j*121, c='black')


# %%

plt.figure(figsize = (20, 2.5))
for i in range(wfs_a_denoised.shape[0]//2):
    plt.plot(wfs_a[i].T.flatten(), c='red', alpha = 0.05)
for i in range(wfs_a_denoised.shape[0]//2):
    plt.plot(wfs_b[i].T.flatten(), c='blue', alpha = 0.05)
for j in range(9):
    plt.axvline(121+j*121, c='black')


# %%
plt.figure(figsize = (20, 2.5))
for i in range(wfs_a.shape[0]//2):
    plt.plot(wfs_a[i, :, 3:6].T.flatten(), c='red', alpha = 0.5)
for i in range(wfs_a.shape[0]//2):
    plt.plot(wfs_b[i, :,  3:6].T.flatten(), c='blue', alpha = 0.5)
# for j in range(9):
#     plt.axvline(121+j*121, c='black')

# %%
plt.figure(figsize = (20, 2.5))
for i in range(cleaned.shape[0]//2):
    plt.plot(cleaned[i].flatten(), c='red', alpha = 0.05)
for i in range(cleaned.shape[0]//2):
    plt.plot(cleaned[cleaned.shape[0]//2+i].flatten(), c='blue', alpha = 0.05)
for j in range(9):
    plt.axvline(121+j*121, c='black')


# %%
proj_vector = (lda_model.means_[0] - lda_model.means_[1]).reshape((10, 121))
plt.figure(figsize = (10, 2.5))
for i in range(proj_vector.shape[0]):
    plt.plot(proj_vector[i])
plt.show()


# %%
# from scipy.interpolate import griddata, interp2d

# wfs_a_registered = np.zeros(wfs_a_denoised.shape)
# for i in range(wfs_a_denoised.shape[0]):
#     wfs_a_registered[i] = griddata(geom_array[mc-20:mc+20], wfs_a_denoised[i].T, geom_array[mc-20:mc+20] + [0, displacement_estimate[int(geom_array[mc, 1]), spike_index_unit_a[i]//30000]], fill_value = 0).T


# %%
# wfs_b_registered = np.zeros(wfs_b_denoised.shape)
# for i in range(wfs_a_denoised.shape[0]):
#     wfs_b_registered[i] = griddata(geom_array[mc-20:mc+20], wfs_b_denoised[i].T, geom_array[mc-20:mc+20] + [0, displacement_estimate[int(geom_array[mc, 1]), spike_index_unit_b[i]//30000]], fill_value = 0).T


# %%

# %%
disp_rescaled = displacement_estimate[int(geom_array[mc, 1])] - displacement_estimate[int(geom_array[mc, 1])].min()
disp_rescaled = disp_rescaled/disp_rescaled.max()
plt.figure(figsize = (20, 2.5))
for i in range(wfs_a_denoised.shape[0]):
    plt.plot(wfs_a[i, :80].T.flatten(), c=vir(disp_rescaled[spike_index_unit_a[i]//30000]), alpha = 0.1)
for i in range(wfs_b_denoised.shape[0]):
    plt.plot(wfs_b[i, :80].T.flatten(), c=vir(disp_rescaled[spike_index_unit_b[i]//30000]), alpha = 0.1)
for j in range(9):
    plt.axvline(80+j*80, c='black')

# %%
disp_rescaled = displacement_estimate[int(geom_array[mc, 1])] - displacement_estimate[int(geom_array[mc, 1])].min()
disp_rescaled = disp_rescaled/disp_rescaled.max()
plt.figure(figsize = (20, 2.5))
for i in range(wfs_a_denoised.shape[0]):
    plt.plot(wfs_a_denoised[i, :].T.flatten(), c=vir(disp_rescaled[spike_index_unit_a[i]//30000]), alpha = 0.1)
for i in range(wfs_b_denoised.shape[0]):
    plt.plot(wfs_b_denoised[i, :].T.flatten(), c=vir(disp_rescaled[spike_index_unit_b[i]//30000]), alpha = 0.1)
for j in range(9):
    plt.axvline(121+j*121, c='black')

# %%

# %%

# %% [markdown]
# # Visualization

# %%
idx_sorted = triaged_spike_index[:, 0].argsort()
spike_index_triaged = triaged_spike_index[idx_sorted]
clusterer.labels_ = clusterer.labels_[idx_sorted]
triaged_x = triaged_x[idx_sorted]
triaged_z = triaged_z[idx_sorted]
triaged_maxptps = triaged_maxptps[idx_sorted]
triaged_sub_wfs = triaged_sub_wfs[idx_sorted]  
triaged_first_chan = triaged_first_chan[idx_sorted]  


# %%
#create hdbscan/localization SpikeInterface sorting (with triage)
sorting_hdbl_t = make_sorting_from_labels_frames(clusterer.labels_, spike_index_triaged[:,0])

cmp_5 = compare_two_sorters(sorting_hdbl_t, sorting_kilo, sorting1_name='ours', sorting2_name='kilosort', match_score=.5)
matched_units_5 = cmp_5.get_matching()[0].index.to_numpy()[np.where(cmp_5.get_matching()[0] != -1.)]
matches_kilos_5 = cmp_5.get_best_unit_match1(matched_units_5).values.astype('int')

cmp_1 = compare_two_sorters(sorting_hdbl_t, sorting_kilo, sorting1_name='ours', sorting2_name='kilosort', match_score=.1)
matched_units_1 = cmp_1.get_matching()[0].index.to_numpy()[np.where(cmp_1.get_matching()[0] != -1.)]
unmatched_units_1 = cmp_1.get_matching()[0].index.to_numpy()[np.where(cmp_1.get_matching()[0] == -1.)]
matches_kilos_1 = cmp_1.get_best_unit_match1(matched_units_1).values.astype('int')

cmp_kilo_5 = compare_two_sorters(sorting_kilo, sorting_hdbl_t, sorting1_name='kilosort', sorting2_name='ours', match_score=.5)
matched_units_kilo_5 = cmp_kilo_5.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_5.get_matching()[0] != -1.)]
unmatched_units_kilo_5 = cmp_kilo_5.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_5.get_matching()[0] == -1.)]

cmp_kilo_1 = compare_two_sorters(sorting_kilo, sorting_hdbl_t, sorting1_name='kilosort', sorting2_name='ours', match_score=.1)
matched_units_kilo_1 = cmp_kilo_1.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_1.get_matching()[0].to_numpy() != -1.)]
unmatched_units_kilo_1 = cmp_kilo_1.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_1.get_matching()[0].to_numpy() == -1.)]


# %%

# %%
#remove duplicate units by spike_times_agreement and ptp
# clusterer, duplicate_ids = remove_duplicate_units(clusterer, triaged_spike_index[:,0], triaged_maxptps)

#re-compute cluster centers
cluster_centers = compute_cluster_centers(clusterer)


# %%
##### plot array scatter #####
fig = plot_array_scatter(clusterer.labels_, geom_array, triaged_x, triaged_z, triaged_maxptps, cluster_color_dict, color_arr, min_cluster_size=clusterer.min_cluster_size, min_samples=clusterer.min_samples, 
                         z_cutoff=(1600, 2400), figsize=(18, 12))
# fig.suptitle(f'x,z,scaled_logptp features," {num_spikes} datapoints');
plt.show()


# %%
##### plot array scatter #####
fig = plot_array_scatter(clusterer.labels_, geom_array, triaged_x, triaged_z, triaged_maxptps, cluster_color_dict, color_arr, min_cluster_size=clusterer.min_cluster_size, min_samples=clusterer.min_samples, 
                         z_cutoff=(800, 1600), figsize=(18, 12))
# fig.suptitle(f'x,z,scaled_logptp features," {num_spikes} datapoints');
plt.show()


# %%
##### plot array scatter #####
fig = plot_array_scatter(clusterer.labels_, geom_array, triaged_x, triaged_z, triaged_maxptps, cluster_color_dict, color_arr, min_cluster_size=clusterer.min_cluster_size, min_samples=clusterer.min_samples, 
                         z_cutoff=(0, 800), figsize=(18, 12))
# fig.suptitle(f'x,z,scaled_logptp features," {num_spikes} datapoints');
plt.show()


# %%
##### plot individual cluster summaries #####
#load waveforms as memmap
wfs_localized = np.load(data_dir+'denoised_wfs.npy', mmap_mode='r') #np.memmap(data_dir+'denoised_waveforms.npy', dtype='float32', shape=(290025, 121, 40))
wfs_subtracted = np.load(data_dir+'subtracted_wfs.npy', mmap_mode='r')


# %%

# %%

# %%
#create kilosort SpikeInterface sorting
times_list = []
labels_list = []
for cluster_id in np.unique(kilo_spike_clusters):
    spike_train_kilo = kilo_spike_frames[np.where(kilo_spike_clusters==cluster_id)]
    times_list.append(spike_train_kilo)
    labels_list.append(np.zeros(spike_train_kilo.shape[0])+cluster_id)
times_array = np.concatenate(times_list).astype('int')
labels_array = np.concatenate(labels_list).astype('int')
sorting_kilo = spikeinterface.numpyextractors.NumpySorting.from_times_labels(times_list=times_array, 
                                                                             labels_list=labels_array, 
                                                                                  sampling_frequency=30000)
#create hdbscan/localization SpikeInterface sorting (with triage)
times_list = []
labels_list = []
for cluster_id in np.unique(clusterer.labels_):
    spike_train_hdbl_t = spike_index_triaged[:,0][np.where(clusterer.labels_==cluster_id)]
    times_list.append(spike_train_hdbl_t)
    labels_list.append(np.zeros(spike_train_hdbl_t.shape[0])+cluster_id)
times_array = np.concatenate(times_list).astype('int')
labels_array = np.concatenate(labels_list).astype('int')
sorting_hdbl_t = spikeinterface.numpyextractors.NumpySorting.from_times_labels(times_list=times_array, 
                                                                                labels_list=labels_array, 
                                                                                sampling_frequency=30000)

cmp_5 = compare_two_sorters(sorting_hdbl_t, sorting_kilo, sorting1_name='ours', sorting2_name='kilosort', match_score=.5)
matched_units_5 = cmp_5.get_matching()[0].index.to_numpy()[np.where(cmp_5.get_matching()[0] != -1.)]
matches_kilos_5 = cmp_5.get_best_unit_match1(matched_units_5).values.astype('int')

cmp_1 = compare_two_sorters(sorting_hdbl_t, sorting_kilo, sorting1_name='ours', sorting2_name='kilosort', match_score=.1)
matched_units_1 = cmp_1.get_matching()[0].index.to_numpy()[np.where(cmp_1.get_matching()[0] != -1.)]
unmatched_units_1 = cmp_1.get_matching()[0].index.to_numpy()[np.where(cmp_1.get_matching()[0] == -1.)]
matches_kilos_1 = cmp_1.get_best_unit_match1(matched_units_1).values.astype('int')

# %%
# ##### plot individual cluster summaries #####
#load waveforms as memmap
wfs_localized = np.load(data_dir+'denoised_wfs.npy', mmap_mode='r') #np.memmap(data_dir+'denoised_waveforms.npy', dtype='float32', shape=(290025, 121, 40))
wfs_subtracted = np.load(data_dir+'subtracted_wfs.npy', mmap_mode='r')
non_triaged_idxs = ptp_filter[0][idx_keep]

# %% tags=[]
cluster_id = 16

if cluster_id in matched_units_5:
    cmp = cmp_5
    print(">50% match")
elif cluster_id in matched_units_1:
    cmp = cmp_1
    print("50%> and >10% match")
else:
    cmp = None
    print("<10% match")
    

#plot cluster summary
fig = plot_single_unit_summary(cluster_id, clusterer.labels_, cluster_centers, geom_array, 50, num_rows_plot, triaged_x, triaged_z, triaged_maxptps, 
                               triaged_firstchans, triaged_mcs_abs, triaged_spike_index, non_triaged_idxs, wfs_localized, wfs_subtracted, cluster_color_dict, 
                               color_arr, raw_data_bin, residual_data_bin)
plt.show()

# plot agreement with kilosort
if cmp is not None:
    num_channels = wfs_localized.shape[2]
    cluster_id_match = cmp.get_best_unit_match1(cluster_id)
    sorting1 = sorting_hdbl_t
    sorting2 = sorting_kilo
    sorting1_name = "hdb"
    sorting2_name = "kilo"
    firstchans_cluster_sorting1 = triaged_firstchans[clusterer.labels_ == cluster_id]
    mcs_abs_cluster_sorting1 = triaged_mcs_abs[clusterer.labels_ == cluster_id]
    spike_depths = kilo_spike_depths[np.where(kilo_spike_clusters==cluster_id_match)]
    mcs_abs_cluster_sorting2 = np.asarray([np.argmin(np.abs(spike_depth - geom_array[:,1])) for spike_depth in spike_depths])
    firstchans_cluster_sorting2 = (mcs_abs_cluster_sorting2 - 20).clip(min=0)
    
    plot_agreement_venn(cluster_id, cluster_id_match, cmp, sorting1, sorting2, sorting1_name, sorting2_name, geom_array, num_channels, num_spikes_plot, firstchans_cluster_sorting1, mcs_abs_cluster_sorting1, 
                        firstchans_cluster_sorting2, mcs_abs_cluster_sorting2, raw_data_bin, delta_frames = 12)

# %% [markdown]
# # Oversplit Analysis

# %%
sorting_kilo.get_unit_ids()[:1]

# %%
num_channels = 10
num_spikes_plot = 100

# %%

# %% tags=[]
###Kilosort
# %matplotlib inline
save_dir_path = "oversplit_cluster_summaries_kilosort"
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)
    
num_close_clusters = 50
num_close_clusters_plot=10
num_channels_similarity = 20
num_under_threshold = 0
shifts_align=np.arange(-8,9)
for cluster_id in sorting_kilo.get_unit_ids()[:1]:
    st_1 = sorting_kilo.get_unit_spike_train(cluster_id)
    
    #compute K closest clsuters
    curr_cluster_depth = kilo_cluster_depth_means[cluster_id]
    dist_to_other_cluster_dict = {cluster_id:abs(mean_depth-curr_cluster_depth) for (cluster_id,mean_depth) in kilo_cluster_depth_means.items()}
    closest_clusters = [y[0] for y in sorted(dist_to_other_cluster_dict.items(), key = lambda x: x[1])[1:1+num_close_clusters]]
    
    #compute unit similarties
    original_template, closest_clusters, similarities, agreements, templates, shifts = get_unit_similarities(cluster_id, st_1, closest_clusters, sorting_kilo, geom_array, raw_data_bin, 
                                                                                                             num_channels_similarity=num_channels_similarity, 
                                                                                                             num_close_clusters=num_close_clusters, shifts_align=shifts_align,
                                                                                                             order_by ='similarity')
    if similarities[0] < 2.0: #arbitrary..
        fig = plot_unit_similarities(cluster_id, closest_clusters, sorting_kilo, sorting_kilo, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
                                     num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="both")
#         plt.close(fig)
#         fig.savefig(save_dir_path + f"/cluster_{cluster_id}_summary.png")
        fig.show()

# %%
###hdbscan
# %matplotlib inline
save_dir_path = "oversplit_cluster_summaries_hdbscan"
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)
    
for cluster_id in sorting_hdbl_t.get_unit_ids()[:3]:
    if cluster_id != -1:
        #compute firing rate
        st_1 = sorting_hdbl_t.get_unit_spike_train(cluster_id)
        #compute K closest clsuters
        curr_cluster_center = cluster_centers[cluster_id]
        dist_other_clusters = np.linalg.norm(curr_cluster_center[:2] - cluster_centers[:,:2], axis=1)
        closest_clusters = np.argsort(dist_other_clusters)[1:num_close_clusters + 1]
        #compute unit similarties
        original_template, closest_clusters, similarities, agreements, templates, shifts = get_unit_similarities(cluster_id, st_1, closest_clusters, sorting_hdbl_t, geom_array, raw_data_bin, 
                                                                                                                 num_channels_similarity=num_channels_similarity, 
                                                                                                                 num_close_clusters=num_close_clusters, shifts_align=shifts_align,
                                                                                                                 order_by ='similarity')
#         if similarities[0] < 2.0: #arbitrary..
        fig = plot_unit_similarities(cluster_id, closest_clusters, sorting_hdbl_t, sorting_hdbl_t, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
                                     num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="both",
                                     denoised_waveforms=wfs_localized, cluster_labels=clusterer.labels_, non_triaged_idxs=non_triaged_idxs, triaged_mcs_abs=triaged_mcs_abs, 
                                     triaged_firstchans=triaged_firstchans)
        fig.show()
#         plt.close(fig)
#         fig.savefig(save_dir_path + f"/cluster_{cluster_id}_summary.png")

# %% tags=[]
# save_dir_path_hdbscan_kilo = "cluster_summaries_hdbscan_kilo"
# if not os.path.exists(save_dir_path_hdbscan_kilo):
#     os.makedirs(save_dir_path_hdbscan_kilo)
    
# for cluster_id in tqdm(sorting_hdbl_t.get_unit_ids()):
#     if cluster_id != -1:
#         st_1 = sorting_hdbl_t.get_unit_spike_train(cluster_id)

#         #compute K closest kilosort clsuters
#         cluster_center = cluster_centers[cluster_id]
#         curr_cluster_depth = cluster_center[1]
#         dist_to_other_cluster_dict = {cluster_id:abs(mean_depth-curr_cluster_depth) for (cluster_id,mean_depth) in kilo_cluster_depth_means.items()}
#         closest_clusters = [y[0] for y in sorted(dist_to_other_cluster_dict.items(), key = lambda x: x[1])[:num_close_clusters]]
        
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
#         cluster_center = cluster_centers[cluster_id]
#         curr_cluster_depth = cluster_center[1]
#         dist_to_other_cluster_dict = {cluster_id:abs(mean_depth-curr_cluster_depth) for (cluster_id,mean_depth) in kilo_cluster_depth_means.items()}
#         closest_clusters = [y[0] for y in sorted(dist_to_other_cluster_dict.items(), key = lambda x: x[1])[:num_close_clusters]]
        
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
#     curr_cluster_depth = kilo_cluster_depth_means[cluster_id]
#     closest_clusters = np.argsort(np.abs(cluster_centers[:,1] - kilo_cluster_depth_means[cluster_id]))[:num_close_clusters]

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
#     curr_cluster_depth = kilo_cluster_depth_means[cluster_id]
#     closest_clusters = np.argsort(np.abs(cluster_centers[:,1] - kilo_cluster_depth_means[cluster_id]))[:num_close_clusters]

#     fig = plot_unit_similarities(cluster_id, closest_clusters, sorting_kilo, sorting_hdbl_t, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
#                                  num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='agreement', normalize_agreement_by="second")
#     plt.close(fig)
#     fig.savefig(save_dir_path_kilo_hdbscan + f"/cluster_{cluster_id}_summary.png")

# %%
# #plot specific kilosort example
# cluster_id = 166
# num_close_clusters = 50
# num_close_clusters_plot=10
# num_channels_similarity = 20
# shifts_align=np.arange(-8,9)
# cluster_id = 116

# st_1 = sorting_kilo.get_unit_spike_train(cluster_id)

# #compute K closest hdbscan clsuters
# curr_cluster_depth = kilo_cluster_depth_means[cluster_id]
# closest_clusters = np.argsort(np.abs(cluster_centers[:,1] - kilo_cluster_depth_means[cluster_id]))[:num_close_clusters]

# fig = plot_unit_similarities(cluster_id, closest_clusters, sorting_kilo, sorting_hdbl_t, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
#                              num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="second")

# %%
# #plot specific hdbscan example
# cluster_id = 207
# num_close_clusters = 50
# num_close_clusters_plot=10
# num_channels_similarity = 20
# shifts_align=np.arange(-8,9)

# st_1 = sorting_hdbl_t.get_unit_spike_train(cluster_id)

# #compute K closest kilosort clsuters
# cluster_center = cluster_centers[cluster_id]
# curr_cluster_depth = cluster_center[1]
# dist_to_other_cluster_dict = {cluster_id:abs(mean_depth-curr_cluster_depth) for (cluster_id,mean_depth) in kilo_cluster_depth_means.items()}
# closest_clusters = [y[0] for y in sorted(dist_to_other_cluster_dict.items(), key = lambda x: x[1])[:num_close_clusters]]

# fig = plot_unit_similarities(cluster_id, closest_clusters, sorting_hdbl_t, sorting_kilo, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
#                              num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="second")
# # plt.close(fig)
# # fig.savefig(save_dir_path_kilo_hdbscan + f"/cluster_{cluster_id}_summary.png")
