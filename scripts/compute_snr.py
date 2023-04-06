import os
from pathlib import Path
import numpy as np
import h5py
from tqdm.auto import tqdm, trange
import scipy.io
import time
import torch
import shutil
from sklearn.decomposition import PCA
import subprocess
import pickle
import subprocess
import hdbscan
from spike_psvae import (
#     simdata,
# #     subtract,
#     ibme,
    denoise,
#     template_reassignment,
#     snr_templates,
#     grab_and_localize,
#     localize_index,
      cluster_viz,
#     before_deconv_merge_split,
#     cluster_viz_index,
#     deconvolve,
#     extract_deconv,
#     residual,
#     pre_deconv_merge_split,
#     after_deconv_merge_split,
#     cluster_utils,
#     pipeline,
#     spike_reassignment,
    spikeio
)

# from spike_psvae import (
#     denoise,
#     cluster_utils,
#     pre_deconv_merge_split,
#     after_deconv_merge_split,
#     spike_train_utils,
#     extractors,
#     subtract,
#     deconv_resid_merge,
#     waveform_utils,
#     spike_reassignment,
#     cluster_viz
# )

# from spike_psvae.ibme import register_nonrigid
# from spike_psvae.ibme_corr import calc_corr_decent
# from spike_psvae.ibme import fast_raster
# from spike_psvae.ibme_corr import psolvecorr
# from spike_psvae.waveform_utils import make_channel_index, make_contiguous_channel_index
# from spike_psvae import denoise
# from spike_psvae.subtract import full_denoising

from unidip import UniDip
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import SpectralClustering
from spike_psvae.drifty_deconv import superres_deconv, extract_superres_shifted_deconv
from spike_psvae.waveform_utils import get_pitch, pitch_shift_templates
from spike_psvae.deconvolve import MatchPursuitObjectiveUpsample
from spike_psvae.snr_templates import get_raw_template_single

from spike_psvae.cluster_viz import cluster_scatter, plot_waveforms_geom, plot_venn_agreement, array_scatter
from spike_psvae.cluster_viz import plot_self_agreement, plot_single_unit_summary, plot_agreement_venn, plot_isi_distribution, plot_unit_similarities
from spike_psvae.cluster_viz import plot_unit_similarity_heatmaps
from spike_psvae.cluster_utils import make_sorting_from_labels_frames, compute_cluster_centers
from spike_psvae.cluster_utils import get_agreement_indices, compute_spiketrain_agreement, get_unit_similarities, compute_shifted_similarity, read_waveforms
from spike_psvae.cluster_utils import get_closest_clusters_hdbscan, get_closest_clusters_kilosort, get_closest_clusters_hdbscan_kilosort, get_closest_clusters_kilosort_hdbscan
from spike_psvae.isocut5 import isocut5 as isocut
from spike_psvae.drifty_deconv_uhd import superres_denoised_templates, shift_superres_templates

from matplotlib import gridspec
from celluloid import Camera
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import colorcet as ccet

# Load preprocessed data and geom array
# Here example with pattern 2
dat_pat_2 = "../data_2022_02_04_pat_2/traces_cached_seg0.raw"
geom_pat_2 = np.load("../geom_array_pat2.npy")



"""
LOAD RESULTS FROM INITIAL PIPELINE 
If using other spike times / localizations, 
set tpca to None
"""
sub_dir =  Path("test_script_full/initial_detect_localize")

sub_h5 = next(Path(sub_dir).glob("sub*.h5"))

with h5py.File(sub_h5, "r+") as h5:
    # print(subject)
    # print("-" * len(subject))
    for k in h5:
        print(" - ", k) #, h5[k].shape
    geom = h5["geom"][:]
    cleaned_tpca_group = h5["cleaned_tpca"]
    tpca_mean = cleaned_tpca_group["tpca_mean"][:]
    tpca_components = cleaned_tpca_group["tpca_components"][:]
    localization_results = np.array(h5["localizations"][:]) #f.get('localizations').value
    maxptps = np.array(h5["maxptps"][:])
    spike_index = np.array(h5["spike_index"][:])

tpca = PCA(tpca_components.shape[0])
tpca.mean_ = tpca_mean
tpca.components_ = tpca_components

displacement = np.load("displacement_estimate.npy")

pitch = get_pitch(geom_pat_2)

(
    superres_templates_pat_2,
    superres_label_to_bin_id_pat_2,
    superres_label_to_orig_label_pat_2,
    medians_at_computation_pat_2,
) = superres_denoised_templates(
    spt,
    spt,
    z_abs,
    x,
    z_abs,
    x,
    pitch//3,
    geom_pat_2,
    dat_pat_2,
    None,
    0, # Start of the recording
    4000, #end of the recording
    None,
    True,
    25,
    None, 
    None,
    1000*np.ones(len(spt)),
    500,
    max_spikes_per_unit=200,
    n_spikes_max_recent=2000,
    denoise_templates=True,
    do_temporal_decrease=True,
    zero_radius_um=200,
    reducer=np.mean, #TAKE MEAN 
    snr_threshold=5.0 * np.sqrt(100),
    spike_length_samples=121,
    trough_offset=42,
    do_tpca=True,
    tpca=tpca,
    tpca_rank=5,
    tpca_radius=75,
    tpca_n_wfs=50_000,
    tpca_centered=True,
    do_nn_denoise=False,
    fs=30000,
    seed=0,
    n_jobs=0,
)



n_units = spt[:, 1].max()+1
n_spikes = 2000

SNR_val_wf_pat_2 = np.zeros(n_units)
SNR_val_rand_pat_2 = np.zeros(n_units)
n_spikes_per_unit_pat_2 = np.zeros(n_units)


for unit in range(n_units):
    print(unit)
    idx_spikes_unit = np.flatnonzero(spt[:, 1]==unit)
    n_spikes_unit = min(n_spikes, len(idx_spikes_unit))
    idx_spikes_unit = idx_spikes_unit[np.random.choice(len(idx_spikes_unit), n_spikes_unit, replace=False)]
    
    wfs_pat_2 = spikeio.read_waveforms(spt[idx_spikes_unit, 0], dat_pat_2, geom_pat_2.shape[0])[0]
    wfs_rand_pat_2 = spikeio.read_waveforms(np.random.choice(30000*3000, n_spikes_unit, replace=False), dat_pat_2, geom_pat_2.shape[0])[0]
    
    #Pat 2
    pitch = get_pitch(geom_pat_2)
    bin_size_um = pitch//3
    z_int = np.round(z_abs[idx_spikes_unit]/pitch)
    cmp=0
    for z in np.unique(z_int):
        idx_z = np.flatnonzero(z_int==z)
        shifted_temp = shift_superres_templates(superres_templates_pat_2[superres_label_to_orig_label_pat_2==unit], 
                                superres_label_to_bin_id_pat_2[superres_label_to_orig_label_pat_2==unit],
                                np.zeros((superres_label_to_orig_label_pat_2==unit).sum()).astype('int'),
                                bin_size_um, geom_pat_2, z-np.round(medians_at_computation_pat_2[unit]/pitch),
                                [0], [0])
        for k in idx_z:
            wf = wfs_pat_2[k]
            wf_rand = wfs_rand_pat_2[k]
            z_wf = z_abs[idx_spikes_unit[k]]
            bin_id = (z_wf-z*pitch)//bin_size_um
            if np.isin(bin_id, superres_label_to_bin_id_pat_2[superres_label_to_orig_label_pat_2==unit]):
                temp = shifted_temp[superres_label_to_bin_id_pat_2[superres_label_to_orig_label_pat_2==unit]==bin_id]
                SNR_val_wf_pat_2[unit] += np.abs(np.dot(temp.flatten(), wf.flatten())/(384))
                SNR_val_rand_pat_2[unit] += np.abs(np.dot(temp.flatten(), wf_rand.flatten())/(384))
                cmp+=1
    SNR_val_wf_pat_2[unit] /= cmp
    SNR_val_rand_pat_2[unit] /= cmp
    n_spikes_per_unit_pat_2[unit] = cmp



