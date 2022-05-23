import numpy as np
import pickle
import os

import hdbscan
from pathlib import Path
import matplotlib


import torch
import h5py
from sklearn.decomposition import PCA
from spike_psvae import denoise, subtract, localization, ibme, residual, deconvolve
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path

from spike_psvae import cluster, merge_split_cleaned, cluster_viz_index, denoise, cluster_utils, triage, cluster_viz
from spike_psvae.cluster_utils import read_waveforms, compare_two_sorters, make_sorting_from_labels_frames
from spike_psvae.cluster_viz import plot_agreement_venn, plot_unit_similarities
from spike_psvae.cluster_utils import get_closest_clusters_kilosort_hdbscan
from spike_psvae.cluster_viz import plot_single_unit_summary
from spike_psvae.cluster_viz import cluster_scatter, plot_waveforms_geom, plot_venn_agreement #plot_raw_waveforms_unit_geom
from spike_psvae.cluster_viz import array_scatter, plot_self_agreement, plot_single_unit_summary, plot_agreement_venn, plot_isi_distribution, plot_unit_similarities, plot_waveforms_geom_unit
# plot_array_scatter, plot_waveforms_unit_geom
from spike_psvae.cluster_viz import plot_unit_similarity_heatmaps
from spike_psvae.cluster_utils import make_sorting_from_labels_frames, compute_cluster_centers, relabel_by_depth, remove_duplicate_units
# run_weighted_triage
from spike_psvae.triage import run_weighted_triage
from spike_psvae.cluster_utils import get_agreement_indices, compute_spiketrain_agreement, get_unit_similarities, compute_shifted_similarity, read_waveforms
from spike_psvae.cluster_utils import get_closest_clusters_hdbscan, get_closest_clusters_kilosort, get_closest_clusters_hdbscan_kilosort, get_closest_clusters_kilosort_hdbscan

from spike_psvae import relocalize_after_deconv, after_deconv_merge_split

spike_index = np.load('spike_index_before_deconv.npy')
labels = np.load('/labels_before_deconv.npy') #labels equal to -1 when spike not assigned
standardized_path = 'standardized.bin'
residual_path = 'subtraction _residuals.bin'
standardized_dtype = 'float32'

output_directory = 'cpu_deconv_localization_results' # output directory
geom_path = 'np1_channel_map.npy' # path to geom file

# h5 subtract from the first pass of detect/subtract
h5_subtract = 'subtraction_localization_results.h5'

fs = 30000
# Run deconvolution

geom_array = np.load(geom_path)

#get_templates reads so that templates have trough at 42
templates_raw = merge_split_cleaned.get_templates(
    standardized_path,
    geom_array,
    np.unique(labels).shape[0]-1,
    spike_index[labels>=0],
    labels[labels>=0],
    max_spikes=250,
    n_times=121,
)

# Align templates/spike index to 42!! Needed for later? - deconvolve.read_waveforms reads from -60 not -42 
# SHOULD WE CHANGE THIS?
for i in range(templates_raw.shape[0]):
    mc = templates_raw[i].ptp(0).argmax(0)
    if templates_raw[i, :, mc].argmin() != 42:
        spike_index[labels == i, 0] += templates_raw[i, :, mc].argmin() - 42

template_spike_train = np.c_[spike_index[labels>=0][:, 0], labels[labels>=0]]

result_file_names = deconvolve.deconvolution(spike_index[labels>=0],labels[labels>=0], 
                        output_directory, standardized_path, residual_path, template_spike_train,
                         geom_path, multi_processing=False, cleaned_temps=True, n_processors=6, threshold=40)

print(result_file_names)


# Compute residual

residual_path = residual.run_residual(result_file_names[0], result_file_names[1],
                                      output_directory, standardized_path, geom_path)


'''
CODE TO CHECK RESIDUALS LOOK GOOD 

start = 0
viz_len = 1000
n_chans = 384
img = np.fromfile(standardized_path, 
                  dtype=np.float32, 
                  count=n_chans*viz_len, 
                  offset=4*start*n_chans).reshape((viz_len,n_chans))
residual_img = np.fromfile(residual_path, 
                  dtype=np.float32, 
                  count=n_chans*viz_len, 
                           offset=4*start*n_chans).reshape((viz_len,n_chans))

vmin = min(img.min(), residual_img.min())
vmax = max(img.max(), residual_img.max())

fig, axs = plt.subplots(1,2, sharey=True, figsize=(14,6))
axs[0].imshow(img.T, aspect='auto', vmin=vmin, vmax=vmax)
axs[1].imshow(residual_img.T, aspect='auto', vmin=vmin, vmax=vmax)
plt.show()

'''

# Extract subtracted, collision-subtracted, denoised waveforms

# load denoiser
device = torch.device("cuda")
denoiser = denoise.SingleChanDenoiser()
denoiser.load()
denoiser.to(device)

deconv_spike_train_up = np.load(result_file_names[1])
deconv_templates_up = np.load(result_file_names[0])

n_spikes = deconv_spike_train_up.shape[0]
print(f'number of deconv spikes: {n_spikes}')
print(f'deconv templates shape: {deconv_templates_up.shape}')

# 42/60 issue : deconvolve.read_waveforms used in this function reads at t-60:t+60 
# and pass wfs through denoising pipeline

#Save all wfs in output_directory
n_chans_to_extract = 40

relocalize_after_deconv.extract_deconv_wfs(h5_subtract, 
    residual_path, geom_array, deconv_spike_train_up, 
    deconv_templates_up, output_directory,
    denoiser, device, n_chans_to_extract=n_chans_to_extract)

# Merge wfs in h5 fils
shape = (n_spikes-skipped_count, 121, n_chans_to_extract)
relocalize_after_deconv.merge_files_h5(subtracted_waveforms_dir, 
               os.path.join(output_directory,'subtracted_wfs.h5'), 'wfs', shape, delete=True)
relocalize_after_deconv.merge_files_h5(collision_subtracted_waveforms_dir, 
               os.path.join(output_directory,'collision_subtracted_wfs.h5'), 'wfs', shape, delete=True)
relocalize_after_deconv.merge_files_h5(denoised_waveforms_dir, 
               os.path.join(output_directory,'denoised_wfs.h5'), 'wfs', shape, delete=True)


# Relocalize Waveforms 

denoised_wfs_h5 = h5py.File(os.path.join(output_directory,'denoised_wfs.h5')) #usd for split
cleaned_wfs_h5 = h5py.File(os.path.join(output_directory,'collision_subtracted_wfs.h5')) #used for merge
# denoised_wfs = denoised_wfs_h5["wfs"]
# n_spikes = denoised_wfs.shape[0]

deconv_spike_index = np.load(os.path.join(output_directory, 'spike_index.npy'))
# assert deconv_spike_index.shape[0] == n_spikes
print(f'number of deconv spikes: {deconv_spike_index.shape[0]}')

relocalize_after_deconv.relocalize_extracted_wfs(denoised_wfs_h5, 
    deconv_spike_train_up, deconv_spike_index, geom_array, output_directory)

localization_results_path = os.path.join(output_directory, 'localization_results.npy')
maxptpss = np.load(localization_results_path)[:, 4]
z_absss = np.load(localization_results_path)[:, 1]
times = deconv_spike_train_up[:,0].copy()/fs


# # Check localization results output
# raster, dd, tt = ibme.fast_raster(maxptpss, z_absss, times)
# plt.figure(figsize=(16,12))
# plt.imshow(raster, aspect='auto')


# Register 

z_reg, dispmap = ibme.register_nonrigid(
    maxptpss,
    z_absss,
    times,
    robust_sigma=1,
    rigid_disp=200,
    disp=100,
    denoise_sigma=0.1,
    destripe=False,
    n_windows=[5, 10],
    widthmul=0.5,
)
z_reg -= (z_reg - z_absss).mean()
dispmap -= dispmap.mean()
np.save(os.path.join(output_directory, 'z_reg.npy'), z_reg)
np.save(os.path.join(output_directory, 'ptps.npy'), maxptpss)

# # Check registration output
# registered_raster, dd, tt = ibme.fast_raster(maxptpss, z_reg, times)
# plt.figure(figsize=(16,12))
# plt.imshow(registered_raster, aspect='auto')

# After Deconv Split Merge

deconv_spike_index = np.load(os.path.join(output_directory, 'spike_index.npy'))
z_abs = np.load(os.path.join(output_directory, 'localization_results.npy'))[:, 1]
firstchans = np.load(os.path.join(output_directory, 'localization_results.npy'))[:, 5]
maxptps = np.load(os.path.join(output_directory, 'localization_results.npy'))[:, 4]
spike_train_deconv = np.load(os.path.join(output_directory, 'spike_train.npy'))
xs = np.load(os.path.join(output_directory, 'localization_results.npy'))[:, 0]
z_reg = np.load(os.path.join(output_directory, 'z_reg.npy'))

templates_after_deconv= merge_split_cleaned.get_templates(
    standardized_path,
    geom_array,
    np.unique(spike_train_deconv[:, 1]).shape[0]-1,
    spike_train_deconv[:, 0],
    spike_train_deconv[:, 1],
    max_spikes=250,
    n_times=121,
)


for i in range(templates_after_deconv.shape[0]):
    mc = templates_after_deconv[i].ptp(0).argmax(0)
    if templates_after_deconv[i, :, mc].argmin() != 42:
        spike_train_deconv[spike_train_deconv[:, 1] == i, 0] += templates_after_deconv[i, :, mc].argmin() - 42

templates_after_deconv= merge_split_cleaned.get_templates(
    standardized_path,
    geom_array,
    np.unique(spike_train_deconv[:, 1]).shape[0]-1,
    spike_train_deconv[:, 0],
    spike_train_deconv[:, 1],
    max_spikes=250,
    n_times=121,
)


split_labels = after_deconv_merge_split.split(spike_train_deconv[:, 1], templates_after_deconv, 
    maxptps, firstchans, denoised_wfs_h5)

templates_geq_4 = merge_split_cleaned.get_templates(
    standardized_path,
    geom_array,
    np.unique(spike_train_deconv[:, 1]).shape[0]-1,
    spike_train_deconv[maxptps>4, 0],
    spike_train_deconv[maxptps>4, 1],
    max_spikes=250,
    n_times=121,
)

#take ptp > 4 before next step of deconv

merged_labels = after_deconv_merge_split.merge(spike_train_deconv[maxptps>4, 1], templates_geq_4, cleaned_wfs_h5, xs[maxptps>4], z_reg[maxptps>4], maxptps[maxptps>4])


# Additional Deconv

which = np.flatnonzero(np.logical_and(maxptps > 4, split_labels >= 0))
spt_deconv_after_merge= spike_train_deconv[which]
spt_deconv_after_merge[:, 1] = merged_labels[split_labels >= 0]

spike_index_DAM = np.zeros(spt_deconv_after_merge.shape)
spike_index_DAM[:, 0] = spt_deconv_after_merge[:, 0].copy()
for i in range(templates_geq_4.shape[0]):
    spike_index_DAM[spt_deconv_after_merge[:, 1] == i, 1] = templates_geq_4[i].ptp(0).argmax()

output_directory = 'cpu_deconv_results_AFTER_SPLIT_MERGE'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

trough_offset = 42
max_time = 5*60*30000
which = (spt_deconv_after_merge[:, 0] > trough_offset) & (spt_deconv_after_merge[:, 0] < max_time - (121 - trough_offset))

result_file_names = deconvolve.deconvolution(spike_index_DAM[which],spt_deconv_after_merge[which, 1], 
                        output_directory, standardized_path, residual_path, spt_deconv_after_merge[which],
                         geom_path, multi_processing=False, cleaned_temps=True, n_processors=6, threshold=40)


# Following steps are optional (compute residuals and relocalize )

residual_path = residual.run_residual(result_file_names[0], result_file_names[1],
                                      output_directory, standardized_path, geom_path)



deconv_spike_train_up = np.load(result_file_names[1])
deconv_templates_up = np.load(result_file_names[0])

n_spikes = deconv_spike_train_up.shape[0]
print(f'number of deconv spikes: {n_spikes}')
print(f'deconv templates shape: {deconv_templates_up.shape}')

# 42/60 issue : deconvolve.read_waveforms used in this function reads at t-60:t+60 
# and pass wfs through denoising pipeline

#Save all wfs in output_directory
n_chans_to_extract = 40

skipped_count = relocalize_after_deconv.extract_deconv_wfs(h5_subtract, 
    residual_path, geom_array, deconv_spike_train_up, 
    deconv_templates_up, output_directory,
    denoiser, device, n_chans_to_extract=n_chans_to_extract)

# Merge wfs in h5 fils
shape = (n_spikes-skipped_count, 121, n_chans_to_extract)
relocalize_after_deconv.merge_files_h5(subtracted_waveforms_dir, 
               os.path.join(output_directory,'subtracted_wfs.h5'), 'wfs', shape, delete=True)
relocalize_after_deconv.merge_files_h5(collision_subtracted_waveforms_dir, 
               os.path.join(output_directory,'collision_subtracted_wfs.h5'), 'wfs', shape, delete=True)
relocalize_after_deconv.merge_files_h5(denoised_waveforms_dir, 
               os.path.join(output_directory,'denoised_wfs.h5'), 'wfs', shape, delete=True)


# Relocalize Waveforms 

denoised_wfs_h5 = h5py.File(os.path.join(output_directory,'denoised_wfs.h5')) #usd for split
cleaned_wfs_h5 = h5py.File(os.path.join(output_directory,'collision_subtracted_wfs.h5')) #used for merge
# denoised_wfs = denoised_wfs_h5["wfs"]
# n_spikes = denoised_wfs.shape[0]

deconv_spike_index = np.load(os.path.join(output_directory, 'spike_index.npy'))
# assert deconv_spike_index.shape[0] == n_spikes
print(f'number of deconv spikes: {deconv_spike_index.shape[0]}')

relocalize_after_deconv.relocalize_extracted_wfs(denoised_wfs_h5, 
    deconv_spike_train_up, deconv_spike_index, geom_array, output_directory)

localization_results_path = os.path.join(output_directory, 'localization_results.npy')
maxptpss = np.load(localization_results_path)[:, 4]
z_absss = np.load(localization_results_path)[:, 1]
times = deconv_spike_train_up[:,0].copy()/fs


# Register 

z_reg, dispmap = ibme.register_nonrigid(
    maxptpss,
    z_absss,
    times,
    robust_sigma=1,
    rigid_disp=200,
    disp=100,
    denoise_sigma=0.1,
    destripe=False,
    n_windows=[5, 10],
    widthmul=0.5,
)
z_reg -= (z_reg - z_absss).mean()
dispmap -= dispmap.mean()
np.save(os.path.join(output_directory, 'z_reg.npy'), z_reg)
np.save(os.path.join(output_directory, 'ptps.npy'), maxptpss)

# # Check registration output
# registered_raster, dd, tt = ibme.fast_raster(maxptpss, z_reg, times)
# plt.figure(figsize=(16,12))
# plt.imshow(registered_raster, aspect='auto')

# After Deconv Split Merge

deconv_spike_index = np.load(os.path.join(output_directory, 'spike_index.npy'))
z_abs = np.load(os.path.join(output_directory, 'localization_results.npy'))[:, 1]
firstchans = np.load(os.path.join(output_directory, 'localization_results.npy'))[:, 5]
maxptps = np.load(os.path.join(output_directory, 'localization_results.npy'))[:, 4]
spike_train_deconv = np.load(os.path.join(output_directory, 'spike_train.npy'))
xs = np.load(os.path.join(output_directory, 'localization_results.npy'))[:, 0]
z_reg = np.load(os.path.join(output_directory, 'z_reg.npy'))

templates_after_deconv= merge_split_cleaned.get_templates(
    standardized_path,
    geom_array,
    np.unique(spike_train_deconv[:, 1]).shape[0]-1,
    spike_train_deconv[:, 0],
    spike_train_deconv[:, 1],
    max_spikes=250,
    n_times=121,
)
