import os
from pathlib import Path
import numpy as np
import h5py
import scipy.io
import time
import torch
import shutil

from sklearn.decomposition import PCA
from spike_psvae import filter_standardize
from spike_psvae.cluster_uhd import run_full_clustering
from spike_psvae.drifty_deconv_uhd import full_deconv_with_update
from spike_psvae import subtract, ibme
from spike_psvae.ibme import register_nonrigid
from spike_psvae.ibme_corr import calc_corr_decent
from spike_psvae.ibme import fast_raster
from spike_psvae.ibme_corr import psolvecorr
from spike_psvae.filter_standardize import npSampShifts

from spikeinterface.preprocessing import highpass_filter, common_reference, zscore, phase_shift 
import spikeinterface.core as sc

""""
Set parameters / directories name here
I recommend to keep default parameters if possible
"""

raw_data_name = "binfile.bin" #raw rec location
dtype_raw = 'int16' #dtype of raw rec
output_all = "data_set_name" #everything will be saved here
geom_path = 'geom.npy'
rec_len_sec = 3000 #length of rec in seconds
n_channels = 385 #number of channels (before preprocessing)
sampling_rate = 30000
savefigs = True # To save summary figs at each step 

if savefigs:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import colorcet as ccet


nogpu = False # default is to use gpu when possible - set this to True to keep everything on cpu. Gpu speeds things up a lot. 
trough_offset=42 #Keep these two params for good denoising
spike_length_samples=121

#preprocessing parameters
preprocessing=True
apply_filter=True
n_channels_before_preprocessing=385
channels_to_remove=384 #Typically the reference channel - IMPORTANT: make sure it is not included in the geometry array 
low_frequency=300
high_factor=0.1
order=3
median_subtraction=True,
adcshift_correction=True,
t_start_preproc=0
t_end_preproc=None
#Multi processing params
n_job_preprocessing=-1
n_sec_chunk_preprocessing=1



# Initial Detection - Localization parameters 
detect_localize = True
subh5_name = None #This is in case detection has already been ran and we input the subtraction h5 file name here
overwrite_detect=True
t_start_detect = 0 #This is to run detection on full data, and then sort only a "good" portion (no artefacts)
t_end_detect = None #These params correspond to recording AFTER preprocessing
nn_denoise=True
# I recommend not saving wfs / residuals for memory issue 
save_residual = False
save_subtracted_waveforms = False
save_cleaned_waveforms = False
save_denoised_waveforms = False
save_subtracted_tpca_projs = False
save_cleaned_tpca_projs = True
save_denoised_ptp_vectors = False
thresholds_subtract = [12, 9, 6] #thresholds for subtraction
peak_sign = "both"
nn_detect=False
denoise_detect=False
save_denoised_tpca_projs = False
neighborhood_kind = "box"
enforce_decrease_kind = "radial"
extract_box_radius = 70
tpca_rank = 8
n_sec_pca = 20
n_sec_chunk_detect = 1
nsync = 0
n_jobs_detect = 0 # set to 0 if multiprocessing doesn't work
n_loc_workers = 4
localization_kind = "logbarrier"
localize_radius = 100

# Registration parameters 
sigma_reg=0.1
max_disp=100 # This is not the actual max displacement, we don't use paris of bins with relative disp>max_disp when computing full displacement 
max_dt=250
max_dt=min(max_dt, rec_len_sec)
mincorr=0.6
prior_lambda=1

# Clustering parameters 
clustering=True
t_start_clustering=0
t_end_clustering=None
len_chunks_cluster=300
threshold_ptp_cluster=3
triage_quantile_cluster=100
frame_dedup_cluster=20
log_c=5 #to make clusters isotropic
scales=(1, 1, 50)



#Deconv parameters
deconvolve=True
t_start_deconv=0
t_end_deconv=None
n_sec_temp_update=100
bin_size_um=1
n_jobs_deconv=0
max_upsample=1
refractory_period_frames=10
min_spikes_bin=None
max_spikes_per_unit=200
deconv_threshold=200
su_chan_vis=3, 
deconv_th_for_temp_computation=500
n_sec_train_feats=10
n_sec_chunk_deconv=1
overwrite_deconv=True
remove_final_outliers=True

Path(output_all).mkdir(exist_ok=True)
geom = np.load(geom_path)



if preprocessing:
    print("Preprocessing...")
    preprocessing_dir = Path(output_all) / "preprocessing"
#     Path(preprocessing_dir).mkdir(exist_ok=True)
    if t_end_preproc is None:
        t_end_preproc=rec_len_sec
    
    recording = sc.read_binary(
        raw_data_name,
        sampling_rate,
        n_channels_before_preprocessing,
        dtype_raw,
        time_axis=0,
        is_filtered=False,
    )

    recording = recording._remove_channels(channels_to_remove)

    # set geometry
    recording.set_dummy_probe_from_locations(
        geom, shape_params=dict(radius=10)
    )

    recording = recording.frame_slice(start_frame=int(sampling_rate * t_start_preproc), end_frame=int(sampling_rate * t_end_preproc))

    sampShifts = npSampShifts()
    recording = highpass_filter(recording, freq_min=low_frequency, filter_order=order)
    recording = zscore(recording)
    recording = phase_shift(recording, inter_sample_shift=sampShifts)
    recording = common_reference(recording)
    
    recording.save(folder=preprocessing_dir, n_jobs=n_job_preprocessing, chunk_size=sampling_rate*n_sec_chunk_preprocessing, progressbar=True)

    # Update data name and type if preprocesssed
    raw_data_name = Path(preprocessing_dir) / "traces_cached_seg0.raw"
    dtype_raw = "float32"

# Subtraction 
t_start_detect-=t_start_preproc
t_end_detect-=t_start_preproc

if detect_localize:
    print("Detection...")
    detect_dir = Path(output_all) / "initial_detect_localize"
    Path(detect_dir).mkdir(exist_ok=True)
        
    sub_h5 = subtract.subtraction_binary(
        raw_data_name,
        Path(detect_dir),
        geom=geom,
        overwrite=overwrite_detect,
        sampling_rate=sampling_rate,
        do_nn_denoise=nn_denoise,
        save_residual=save_residual,
        save_subtracted_waveforms=save_subtracted_waveforms,
        save_cleaned_waveforms=save_cleaned_waveforms,
        save_denoised_waveforms=save_denoised_waveforms,
        save_subtracted_tpca_projs=save_subtracted_tpca_projs,
        save_cleaned_tpca_projs=save_cleaned_tpca_projs,
        save_denoised_ptp_vectors=save_denoised_ptp_vectors,
        thresholds=thresholds_subtract,
        peak_sign=peak_sign,
        nn_detect=nn_detect,
        denoise_detect=denoise_detect,
        save_denoised_tpca_projs=save_denoised_tpca_projs,
        neighborhood_kind=neighborhood_kind,
        enforce_decrease_kind=enforce_decrease_kind,
        extract_box_radius=extract_box_radius,
        t_start=t_start_detect,
        t_end=t_end_detect,
        tpca_rank=tpca_rank,
        n_sec_pca=n_sec_pca,
        n_sec_chunk=n_sec_chunk_detect,
        nsync=nsync,
        n_jobs=n_jobs_detect,
        loc_workers=n_loc_workers,
        device="cpu" if nogpu else None,
        localization_kind=localization_kind,
        localize_radius=localize_radius,
    )

else:
    sub_h5=subh5_name
    
with h5py.File(sub_h5, "r+") as h5:
    cleaned_tpca_group = h5["cleaned_tpca"]
    tpca_mean = cleaned_tpca_group["tpca_mean"][:]
    tpca_components = cleaned_tpca_group["tpca_components"][:]
    localization_results = np.array(h5["localizations"][:]) 
    maxptps = np.array(h5["maxptps"][:])
    spike_index = np.array(h5["spike_index"][:])

# Load tpca
tpca = PCA(tpca_components.shape[0])
tpca.mean_ = tpca_mean
tpca.components_ = tpca_components

z = localization_results[:, 2]
x = localization_results[:, 0]

# remove spikes localized at boundaries
x_bound_low = geom[:, 0].min()
x_bound_high = geom[:, 0].max()
idx_remove_too_far = np.logical_and(x>x_bound_low-50, x<x_bound_high+50)

maxptps = maxptps[idx_remove_too_far]
z = z[idx_remove_too_far]
spike_index = spike_index[idx_remove_too_far]
x = x[idx_remove_too_far]
localization_results = localization_results[idx_remove_too_far]

# Rigid Registration
print("Registration...")
raster, dd, tt = fast_raster(
        maxptps, z, spike_index[:, 0]/sampling_rate, sigma=sigma_reg, 
    )
D, C = calc_corr_decent(raster, disp = max_disp)
displacement_rigid = psolvecorr(D, C, mincorr=mincorr, max_dt=max_dt, prior_lambda=prior_lambda)

fname_disp = Path(detect_dir) / "displacement_rigid.npy"
np.save(fname_disp, displacement_rigid)

if savefigs:

    vir = cm.get_cmap('viridis')
    ptp_arr = maxptps.copy()
    ptp_arr = np.log(ptp_arr)
    ptp_arr -= ptp_arr.min()
    ptp_arr /= ptp_arr.max()
    color_array = vir(ptp_arr)
    fname_detect_fig = Path(detect_dir) / "detection_displacement_raster_plot.png"
    plt.figure(figsize = (10, 5))
    plt.scatter(spike_index[:, 0]/sampling_rate, z, color = color_array, s = 1)
    plt.plot(displacement_rigid[t_start_detect:t_end_detect]-displacement_rigid[t_start_detect], color = 'red')
    plt.plot(displacement_rigid[t_start_detect:t_end_detect]-displacement_rigid[t_start_detect]+100, color = 'red')
    plt.plot(displacement_rigid[t_start_detect:t_end_detect]-displacement_rigid[t_start_detect]+200, color = 'red')
    plt.savefig(fname_detect_fig)
    plt.close()


# Clustering 
if clustering:
    print("Clustering...")
    cluster_dir = Path(output_all) / "initial_clustering"
    Path(cluster_dir).mkdir(exist_ok=True)
    if t_end_clustering is None:
        t_end_clustering=rec_len_sec
    t_start_clustering-=t_start_preproc
    t_end_clustering-=t_start_preproc

    spt, maxptps, x, z = run_full_clustering(t_start_clustering, t_end_clustering, cluster_dir, raw_data_name, geom, spike_index,
                                                localization_results, maxptps, displacement_rigid, len_chunks=len_chunks_cluster, threshold_ptp=threshold_ptp_cluster,
                                                fs=sampling_rate, triage_quantile_cluster=triage_quantile_cluster, frame_dedup_cluster=frame_dedup_cluster, 
                                                log_c=log_c, scales=scales, savefigs=savefigs)


if deconvolve:
    print("Deconvolution...")
    deconv_dir_all = Path(output_all) / "deconvolution"
    Path(deconv_dir_all).mkdir(exist_ok=True)
    deconv_dir = Path(deconv_dir_all) / "deconv_results"
    Path(deconv_dir).mkdir(exist_ok=True)
    extract_deconv_dir = Path(deconv_dir_all) / "deconv_extracted"
    Path(extract_deconv_dir).mkdir(exist_ok=True)

    if t_end_deconv is None:
        t_end_deconv=rec_len_sec
    t_start_deconv-=t_start_preproc
    t_end_deconv-=t_start_preproc

    full_deconv_with_update(deconv_dir, extract_deconv_dir,
               raw_data_name, geom, displacement_rigid,
               spt, maxptps, x, z, t_start_deconv, t_end_deconv, sub_h5, 
               n_sec_temp_update=n_sec_temp_update, 
               bin_size_um=bin_size_um,
               pfs=sampling_rate,
               n_jobs=n_jobs_deconv,
               trough_offset=trough_offset,
               spike_length_samples=spike_length_samples,
               max_upsample=max_upsample,
               refractory_period_frames=refractory_period_frames,
               min_spikes_bin=min_spikes_bin,
               max_spikes_per_unit=max_spikes_per_unit,
               tpca=tpca,
               deconv_threshold=deconv_threshold,
               su_chan_vis=su_chan_vis,
               deconv_th_for_temp_computation=deconv_th_for_temp_computation,
               extract_radius_um=extract_box_radius,
               loc_radius=localize_radius,
               n_sec_train_feats=n_sec_train_feats,
               n_sec_chunk=n_sec_chunk_deconv,
               overwrite=overwrite_deconv,
               p_bar=True,
               save_chunk_results=False)
    
    fname_medians = Path(extract_dir) / "registered_medians.npy"
    fname_spread = Path(extract_dir) / "registered_spreads.npy"
    units_spread = np.load(fname_spread)
    units_medians = np.load(fname_medians)


    for unit in range(spt[:, 1].max()+1):
        pos = (units_medians[unit]+disp[spt[spt[:, 1]==unit, 0]//30000])
        z_pos = z_abs[spt[:, 1]==unit]
        idx_outlier = np.flatnonzero(np.abs(pos-z_pos)>10*units_spread[unit])
        idx_outlier = np.flatnonzero(spt[:, 1]==unit)[idx_outlier]
        spt[idx_outlier, 1]=-1

    if savefigs:
        spt = np.load(Path(extract_deconv_dir) / "spike_train_final_deconv.npy")
        z = np.load(Path(extract_deconv_dir) / "z_final_deconv.npy")

        fname_deconv_fig=Path(extract_deconv_dir) / "full_deconv_raster_plot.png"
        ccolors = ccet.glasbey[:spt[:, 1].max()+1]
        plt.figure(figsize = (10, 5))
        for k in range(spt[:, 1].max()+1):
            idx = spt[:, 1]==k
            plt.scatter(spt[idx, 0]/sampling_rate, z[idx], c = ccolors[k], s = 1, alpha = 0.1)
        plt.savefig(fname_deconv_fig)
        plt.close()
