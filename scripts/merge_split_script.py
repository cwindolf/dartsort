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
# import parmap

from detect.detector import Detect
from localization_pipeline.denoiser import Denoise

from detect.deduplication import deduplicate #, deduplicate #deduplication_torch

from scipy.signal import argrelmin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import scipy.optimize as optim_ls


from detect.run_substraction_faster import run #run_substraction_faster

from tqdm import notebook

import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm
from spike_psvae.waveform_utils import get_local_geom

from spike_psvae.localization import localize_ptps
from mpl_toolkits import mplot3d
import hdbscan

import h5py
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib import cm
from isosplit import isocut

from scipy.spatial.distance import cdist
from spike_psvae.cluster_viz import plot_array_scatter, plot_self_agreement, plot_single_unit_summary, cluster_scatter, plot_agreement_venn, plot_unit_similarities

import spikeinterface 
from spikeinterface.toolkit import compute_correlograms
from spikeinterface.comparison import compare_two_sorters
from spikeinterface.widgets import plot_agreement_matrix

from spike_psvae.cluster_utils import make_sorting_from_labels_frames, compute_cluster_centers, relabel_by_depth, run_weighted_triage, remove_duplicate_units
from spike_psvae.cluster_utils import get_agreement_indices, compute_spiketrain_agreement, get_unit_similarities, compute_shifted_similarity, read_waveforms
from spike_psvae.cluster_utils import get_closest_clusters_hdbscan, get_closest_clusters_kilosort, get_closest_clusters_hdbscan_kilosort, get_closest_clusters_kilosort_hdbscan



print('setting config')

set_config('/media/cat/julien/nick_drift_np1/drift.yaml', 'tmp/')
CONFIG = read_config()
TMP_FOLDER = CONFIG.path_to_output_directory

standardized_path = '/media/cat/data/CSH_ZAD_026_1800_1860/1min_standardized.bin'
residuals_path = '/media/cat/data/CSH_ZAD_026_1800_1860/residual_1min_standardized_t_0_None.bin'
# standardized_path = '/media/cat/julien//1min_standardized.bin'
standardized_dtype = 'float32'

denoiser = Denoise(CONFIG.neuralnetwork.denoise.n_filters,
                   CONFIG.neuralnetwork.denoise.filter_sizes,
                   CONFIG.spike_size_nn)
denoiser.load(CONFIG.neuralnetwork.denoise.filename)
# denoiser = denoiser.cuda()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
def denoise_wf_nn_tmp_single_channel(wf, denoiser, device):
    denoiser = denoiser.to(device)
    n_data, n_times, n_chans = wf.shape
    if wf.shape[0]>0:
        wf_reshaped = wf.transpose(0, 2, 1).reshape(-1, n_times)
        wf_torch = torch.FloatTensor(wf_reshaped).to(device)
        denoised_wf = denoiser(wf_torch)[0].data
        denoised_wf = denoised_wf.reshape(
            n_data, n_chans, n_times)
        denoised_wf = denoised_wf.cpu().data.numpy().transpose(0, 2, 1)

        del wf_torch
    else:
        denoised_wf = np.zeros((wf.shape[0], wf.shape[1]*wf.shape[2]),'float32')

    return denoised_wf

geom_path = '/channels_maps/np1_channel_map.npy'
geom_array = np.load(geom_path)
n_chans = geom_array.shape[0]
sampling_rate = 30000


loc_res = np.load('/data/localization_results.npy')
spike_index = np.load('/data/spike_index.npy')
z_registered = np.load('/data/z_reg.npy')
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

""" Uncomment for visualizing clusterer output
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

"""

print('split/merge')

def run_LDA_split(wfs_unit_denoised, n_channels = 10):
    lda_model = LDA(n_components = 2)
    arr = wfs_unit_denoised.ptp(1).argmax(1)
    if np.unique(arr).shape[0]<=2:
        arr[-1] = np.unique(arr)[0]-1
        arr[0] = np.unique(arr)[-1]+1
    lda_comps = lda_model.fit_transform(wfs_unit_denoised.reshape((-1, 121*10)), arr)
    lda_clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
    lda_clusterer.fit(lda_comps)
    return lda_clusterer.labels_

def split_individual_cluster(spike_index_unit, x_unit, z_unit, ptps_unit, denoiser, device, n_channels = 10):
    n_channels_half = n_channels//2
    labels_unit = -1*np.ones(spike_index_unit.shape[0])
    is_split = False
    true_mc = int(np.median(spike_index_unit[:, 1]))
    mc = max(n_channels_half, true_mc)
    mc = min(384 - n_channels_half, mc)
    
    pca_model = PCA(2)

    wfs_unit = read_waveforms(spike_index_unit[:, 0]+18, standardized_path, geom_array, n_times=121, channels = np.arange(mc-n_channels_half, mc+n_channels_half))[0]
    wfs_unit_denoised = denoise_wf_nn_tmp_single_channel(wfs_unit, denoiser, device)
    if true_mc<n_channels_half:
        pcs = pca_model.fit_transform(wfs_unit_denoised[:, :, true_mc])
    if true_mc>384-n_channels_half:
        true_mc = true_mc - 374
        pcs = pca_model.fit_transform(wfs_unit_denoised[:, :, true_mc])
    else:
        pcs = pca_model.fit_transform(wfs_unit_denoised[:, :, 5])

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

def merge_clusters(spike_index, labels, x, z, ptps, denoiser, device):
    labels_new = labels.copy()
    labels_original = labels.copy()

    n_clusters = labels.max()
    for unit in tqdm(np.unique(labels)[1:]):
        spike_index_unit = spike_index[labels == unit]
        x_unit, z_unit, ptps_unit = x[labels == unit], z[labels == unit], ptps[labels == unit]
        is_split, unit_new_labels = split_individual_cluster(spike_index_unit, x_unit, z_unit, ptps_unit, denoiser, device)
        if is_split:
            for new_label in np.unique(unit_new_labels):
                if new_label == -1:
                    idx = np.flatnonzero(labels_original == unit)[unit_new_labels == new_label]
                    labels_new[idx] = new_label
                elif new_label >= 0:
                    n_clusters += 1
                    idx = np.flatnonzero(labels_original == unit)[unit_new_labels == new_label]
                    labels_new[idx] = n_clusters
    return labels_new
    
def split_clusters(spike_index, labels, x, z, ptps, denoiser, device):
    labels_new = labels.copy()
    labels_original = labels.copy()

    n_clusters = labels.max()
    for unit in (np.unique(labels)[1:]):
        spike_index_unit = spike_index[labels == unit]
        x_unit, z_unit, ptps_unit = x[labels == unit], z[labels == unit], ptps[labels == unit]
        is_split, unit_new_labels = split_individual_cluster(spike_index_unit, x_unit, z_unit, ptps_unit, denoiser, device)
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

def get_templates(standardized_path, n_templates, spike_index, labels):
    templates = np.zeros((n_templates, 121, 384))
    for unit in range(n_templates):
        spike_times_unit = spike_index[labels==unit, 0]
        if spike_times_unit.shape[0]>250:
            idx = np.random.choice(np.arange(spike_times_unit.shape[0]), 250, replace = False)
        else:
            idx = np.arange(spike_times_unit.shape[0])

        wfs_unit = read_waveforms(spike_times_unit[idx], standardized_path, geom_array, n_times=121)[0]
        templates[unit] = wfs_unit.mean(0)
    return templates

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
        
def get_diptest_value(standardized_path, spike_index, labels, unit_a, unit_b, n_spikes_templates, mc, two_units_shift, unit_shifted, n_channels = 10):

    # ALIGN BASED ON MAX PTP TEMPLATE MC 
    n_channels_half = n_channels//2

    n_wfs_max = int(min(250, min(n_spikes_templates[unit_a], n_spikes_templates[unit_b]))) 
    if unit_b == unit_shifted:
        spike_index_unit_a = spike_index[labels == unit_a, 0]+18 #denoiser offset ## SHIFT BASED ON TEMPLATES ARGMIN PN MAX PTP TEMPLATE
        spike_index_unit_b = spike_index[labels == unit_b, 0]+18+two_units_shift #denoiser offset
    else:
        spike_index_unit_a = spike_index[labels == unit_a, 0]+18+two_units_shift #denoiser offset ## SHIFT BASED ON TEMPLATES ARGMIN PN MAX PTP TEMPLATE
        spike_index_unit_b = spike_index[labels == unit_b, 0]+18 #denoiser offset
    idx = np.random.choice(np.arange(spike_index_unit_a.shape[0]), n_wfs_max, replace = False)
    spike_times_unit_a = spike_index_unit_a[idx]
    idx = np.random.choice(np.arange(spike_index_unit_b.shape[0]), n_wfs_max, replace = False)
    spike_times_unit_b = spike_index_unit_b[idx]
    mc = min(384-n_channels_half, mc)
    mc = max(n_channels_half, mc)
    wfs_a = read_waveforms(spike_times_unit_a, standardized_path, geom_array, n_times=121, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
    wfs_a = read_waveforms(spike_times_unit_b, standardized_path, geom_array, n_times=121, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
    wfs_a = denoise_wf_nn_tmp_single_channel(wfs_a, denoiser, device)
    wfs_b = denoise_wf_nn_tmp_single_channel(wfs_b, denoiser, device)

    wfs_diptest = np.concatenate((wfs_a, wfs_b)).reshape((-1, n_channels*121))
    labels_diptest = np.zeros(wfs_a.shape[0]+wfs_b.shape[0])
    labels_diptest[:wfs_a.shape[0]] = 1
    
    
    lda_model = LDA(n_components = 1)
    lda_comps = lda_model.fit_transform(wfs_diptest, labels_diptest)
    value_dpt, cut_calue = isocut(lda_comps[:, 0])
    return value_dpt

    
def get_merged(standardized_path, n_templates, spike_index, labels, x, z, n_temp = 5, distance_threshold = 1., threshold_diptest = 1.):
     
    templates = get_templates(standardized_path, n_templates, spike_index, labels)
    n_spikes_templates = get_n_spikes_templates(n_templates, labels)
    x_z_templates = get_x_z_templates(n_templates, labels, x, z)
    print("GET PROPOSED PAIRS")
    dist_argsort, dist_template = get_proposed_pairs(n_templates, templates, x_z_templates, n_temp = 20)
    
    labels_updated = labels.copy()
    reference_units = np.unique(labels)[1:]
    
    for unit in tqdm(range(n_templates)):
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
                    print("Units " + str(unit_reference) + " " + str(unit_bis_reference))
                    print(two_units_shift)
                    dpt_val = get_diptest_value(standardized_path, spike_index_triaged, labels_updated, unit_reference, unit_bis_reference, n_spikes_templates, mc, two_units_shift,unit_shifted)
                    if dpt_val<threshold_diptest and np.abs(two_units_shift)<4:
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


    
temp_clusterer = get_templates(standardized_path, ordered_labels.max()+1, spike_index_triaged, ordered_labels)

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

labels_split = split_clusters(spike_index_triaged, ordered_labels, triaged_x, triaged_z, triaged_maxptps, denoiser, device)
    
temp_clusterer = get_templates(standardized_path, labels_split.max()+1, spike_index_triaged, labels_split)

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

labels_merged = get_merged(standardized_path, ordered_split_labels.max()+1, spike_index_triaged, ordered_split_labels, triaged_x, triaged_z, n_temp = 20, distance_threshold = 2.5, threshold_diptest = 1)

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

print("VISUALIZATION")

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

print("Creating Figures")

#load kilosort results
recording_duration = 60
kilo_spike_samples = np.load('/media/cat/data/CSH_ZAD_026_1800_1860/kilosort_spk_samples.npy')
kilo_spike_frames = (kilo_spike_samples - 30*recording_duration*30000) + 18 #+18 to match our detection alignment
kilo_spike_clusters = np.load('/media/cat/data/CSH_ZAD_026_1800_1860/kilsort_spk_clusters.npy')
kilo_spike_depths = np.load('/media/cat/data/CSH_ZAD_026_1800_1860/kilsort_spk_depths.npy')
kilo_cluster_depth_means = {}
for cluster_id in np.unique(kilo_spike_clusters):
    kilo_cluster_depth_means[cluster_id] = np.mean(kilo_spike_depths[kilo_spike_clusters==cluster_id])


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
for cluster_id in np.unique(ordered_merged_labels):
    spike_train_hdbl_t = spike_index_triaged[:,0][np.where(ordered_merged_labels==cluster_id)]+18
    times_list.append(spike_train_hdbl_t)
    labels_list.append(np.zeros(spike_train_hdbl_t.shape[0])+cluster_id)
times_array = np.concatenate(times_list).astype('int')
labels_array = np.concatenate(labels_list).astype('int')
sorting_hdbl_t = spikeinterface.numpyextractors.NumpySorting.from_times_labels(times_list=times_array, 
                                                                                labels_list=labels_array, 
                                                                                sampling_frequency=30000)

cmp_5 = compare_two_sorters(sorting_hdbl_t, sorting_kilo, sorting1_name='Ours_Split_Merged', sorting2_name='kilosort', match_score=.5)
matched_units_5 = cmp_5.get_matching()[0].index.to_numpy()[np.where(cmp_5.get_matching()[0] != -1.)]
matches_kilos_5 = cmp_5.get_best_unit_match1(matched_units_5).values.astype('int')

cmp_1 = compare_two_sorters(sorting_hdbl_t, sorting_kilo, sorting1_name='Ours_Split_Merged', sorting2_name='kilosort', match_score=.1)
matched_units_1 = cmp_1.get_matching()[0].index.to_numpy()[np.where(cmp_1.get_matching()[0] != -1.)]
unmatched_units_1 = cmp_1.get_matching()[0].index.to_numpy()[np.where(cmp_1.get_matching()[0] == -1.)]
matches_kilos_1 = cmp_1.get_best_unit_match1(matched_units_1).values.astype('int')

cluster_centers = get_x_z_templates(ordered_merged_labels.max()+1, ordered_merged_labels, triaged_x, triaged_z)
triaged_mcs_abs = locs_triaged[:, 6].astype('int')
triaged_firstchans = locs_triaged[:, 5].astype('int')
num_rows_plot=3
num_spikes_plot = 100
num_channels = 40


wfs_localized = np.load('/media/cat/data/CSH_ZAD_026_1800_1860/denoised_wfs.npy')
wfs_subtracted = np.load('/media/cat/data/CSH_ZAD_026_1800_1860/subtracted_wfs.npy')
raw_data_bin = '/media/cat/data/CSH_ZAD_026_1800_1860/1min_standardized.bin'
residual_data_bin = '/media/cat/data/CSH_ZAD_026_1800_1860/residual_1min_standardized_t_0_None.bin'

num_channels_similarity = 10
num_close_clusters = 10
num_close_clusters_plot = 5
shifts_align=np.arange(-8,9)
templates = get_templates(standardized_path, sorting_hdbl_t.get_unit_ids().max()+1, spike_index_triaged, ordered_merged_labels)
dist_argsort, dist_template = get_proposed_pairs(sorting_hdbl_t.get_unit_ids().max()+1, templates, cluster_centers, n_temp = 10)

save_dir_path = "oversplit_cluster_summaries_ours_merged"
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)

for cluster_id in range(ordered_merged_labels.max()+1):
    print(cluster_id)
# cluster_id = 4

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
    fig = plot_single_unit_summary(cluster_id, ordered_merged_labels, cluster_centers, geom_array, 50, num_rows_plot, triaged_x, triaged_z, triaged_maxptps, 
                                   triaged_firstchans, triaged_mcs_abs, spike_index_triaged+18, non_triaged_idxs, wfs_localized, wfs_subtracted, cluster_color_dict, 
                                   color_arr, raw_data_bin, residual_data_bin)
    plt.close(fig)
    fig.savefig('individual_units_plots/unit_'+str(cluster_id)+'.png', dpi = 100)


    if cmp is not None:
        num_channels = wfs_localized.shape[2]
        cluster_id_match = cmp.get_best_unit_match1(cluster_id)
        sorting1 = sorting_hdbl_t
        sorting2 = sorting_kilo
        sorting1_name = "hdb"
        sorting2_name = "kilo"
        firstchans_cluster_sorting1 = triaged_firstchans[ordered_merged_labels == cluster_id]
        mcs_abs_cluster_sorting1 = triaged_mcs_abs[ordered_merged_labels == cluster_id]
        spike_depths = kilo_spike_depths[np.where(kilo_spike_clusters==cluster_id_match)]
        mcs_abs_cluster_sorting2 = np.asarray([np.argmin(np.abs(spike_depth - geom_array[:,1])) for spike_depth in spike_depths])
        firstchans_cluster_sorting2 = (mcs_abs_cluster_sorting2 - 20).clip(min=0)

        fig = plot_agreement_venn(cluster_id, cluster_id_match, cmp, sorting1, sorting2, sorting1_name, sorting2_name, geom_array, num_channels, num_spikes_plot, firstchans_cluster_sorting1, mcs_abs_cluster_sorting1, 
                            firstchans_cluster_sorting2, mcs_abs_cluster_sorting2, raw_data_bin, delta_frames = 12)
        fig.savefig('kilosort_agreement_plots/unit_'+str(cluster_id)+'.png', dpi = 100)
        plt.close(fig)

    st_1 = sorting_hdbl_t.get_unit_spike_train(cluster_id)

    #compute K closest clsuters
    dist_template_unit = dist_template[cluster_id]
    closest_clusters = dist_argsort[cluster_id]
    idx = dist_template_unit.argsort()
    dist_template_unit = dist_template_unit[idx]
    closest_clusters = closest_clusters[idx]
    if dist_template_unit[0] < 2.0: #arbitrary..
        fig = plot_unit_similarities(cluster_id, closest_clusters, sorting_hdbl_t, sorting_hdbl_t, geom_array, raw_data_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
                                     num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="both")
        plt.close(fig)
        fig.savefig(save_dir_path + f"/cluster_{cluster_id}_summary.png")
























