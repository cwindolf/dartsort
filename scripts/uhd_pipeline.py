# %%
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
from spike_psvae.drifty_deconv_uhd import full_deconv_with_update, get_registered_pos
from spike_psvae import subtract, ibme
from spike_psvae.ibme import register_nonrigid
from spike_psvae.ibme_corr import calc_corr_decent
from spike_psvae.ibme import fast_raster
from spike_psvae.ibme_corr import psolvecorr
from spike_psvae.waveform_utils import get_pitch
from spike_psvae.uhd_split_merge import run_full_merge_split
from spike_psvae.post_processing_uhd import full_post_processing
import spike_psvae.newton_motion_est as newt
from spike_psvae.waveform_utils import get_pitch
from spike_psvae.post_processing_uhd import final_split_merge

# %%
from spikeinterface.preprocessing import highpass_filter, common_reference, zscore, phase_shift 
import spikeinterface.core as sc

if __name__ == "__main__":

    # %%
    """"
    Set parameters / directories name here
    I recommend to keep default parameters if possible
    """

    # %%
    raw_data_name = "binfile.bin" #raw rec location
    dtype_raw = 'int16' #dtype of raw rec
    output_all = "data_set_name" #everything will be saved here
    Path(output_all).mkdir(exist_ok=True)
    geom_path = 'geom.npy'
    geom = np.load(geom_path)
    rec_len_sec = 4000 #length of rec in seconds
    n_channels = 385 #number of channels (before preprocessing)
    sampling_rate = 30000
    savefigs = True # To save summary figs at each step 

    # %%
    if savefigs:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import colorcet as ccet


    # %%
    nogpu = False # default is to use gpu when possible - set this to True to keep everything on cpu. Gpu speeds things up a lot. 
    trough_offset=42 #Keep these two params for good denoising
    spike_length_samples=121

    # %%
    #preprocessing parameters
    preprocessing=True
    apply_filter=True
    n_channels_before_preprocessing=n_channels
    channels_to_remove=[] #Typically the reference channel - IMPORTANT: make sure it is not included in the geometry array 
    low_frequency=300
    high_factor=0.1
    order=3
    median_subtraction=True #Set to False on Downsampled datasets
    adcshift_correction=True #Set to False on Downsampled datasets
    t_start_preproc=0
    t_end_preproc=None
    #Multi processing params
    n_job_preprocessing=4
    n_sec_chunk_preprocessing=1

    # Initial Detection - Localization parameters 
    detect_localize = True
    subh5_name = Path(output_all) / "initial_detect_localize/subtraction.h5" #This is in case detection has already been ran and we input the subtraction h5 file name here
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
    peak_sign = "neg" #Important
    nn_detect=False
    denoise_detect=False
    save_denoised_tpca_projs = False
    neighborhood_kind = "box"
    enforce_decrease_kind = "radial"
    extract_box_radius = 100
    tpca_rank = 8
    n_sec_pca = 20
    n_sec_chunk_detect = 1
    nsync = 0
    n_jobs_detect = 4 # set to 0 if multiprocessing doesn't work
    n_loc_workers = 4 #recommend setting these to n_cpus/4
    localization_kind = "logbarrier"
    localize_radius = 100
    loc_feature="peak" # change to "peak" to try new loc idea

    # Registration parameters 
    registration=True
    sigma_reg=0.1
    max_disp=100 # This is not the actual max displacement, we don't use paris of bins with relative disp>max_disp when computing full displacement 
    max_dt=250
    mincorr=0.6
    prior_lambda=1

    # Clustering parameters 
    clustering=True
    t_start_clustering=0
    t_end_clustering=3000 # AVOID areas with artefacts in initial clustering (i.e. strokes etc...)
    len_chunks_cluster=300 # 5 min
    threshold_ptp_cluster=3
    triage_quantile_cluster=100
    frame_dedup_cluster=20
    log_c=5 #to make clusters isotropic
    scales=(1, 1, 50)

    #Deconv parameters
    deconvolve=True
    t_start_deconv=0
    t_end_deconv=None
    n_sec_temp_update=rec_len_sec #Keep that to the full time - does nto work for now :) 
    bin_size_um=3
    adaptive_bin_size_selection=False
    n_jobs_deconv=4
    n_jobs_extract_deconv=2
    max_upsample=8
    refractory_period_frames=10
    min_spikes_bin=None
    max_spikes_per_unit=250
    deconv_threshold=250 #1000
    su_chan_vis=1.5
    su_temp_on=None
    deconv_th_for_temp_computation=1000 #
    adaptive_th_for_temp_computation=False
    poly_params=[500, 200, 20, 1]
    n_sec_train_feats=20
    n_sec_chunk_deconv=1
    overwrite_deconv=True
    augment_low_snr_temps=True
    min_spikes_to_augment=50
    save_cleaned_tpca_projs=True
    save_denoised_tpca_projs=True

    second_deconvolve=True
    third_deconvolve=True
    
    # %%
    preprocessing_dir = Path(output_all) / "preprocessing"
    # Don't trust spikeinterface preprocessing :( ... 
    if preprocessing:
        print("Preprocessing...")
        preprocessing_dir = Path(output_all) / "preprocessing_test"
        Path(preprocessing_dir).mkdir(exist_ok=True)
        if t_end_preproc is None:
            t_end_preproc=rec_len_sec

        filter_standardize.filter_standardize_rec_mp(preprocessing_dir,raw_data_name, dtype_raw,
            rec_len_sec, n_channels = n_channels_before_preprocessing, channels_to_remove=channels_to_remove, 
            t_start=t_start_preproc, t_end=t_end_preproc,
            apply_filter=apply_filter, low_frequency=low_frequency, 
            high_factor=high_factor, order=order, sampling_frequency=sampling_rate,
            median_subtraction=median_subtraction, adcshift_correction=adcshift_correction, n_jobs=-1)

        # Update data name and type if preprocesssed
        raw_data_name = Path(preprocessing_dir) / "standardized.bin"
        dtype_raw = "float32"
    else:
        raw_data_name = Path(preprocessing_dir) / "standardized.bin"
        dtype_raw = "float32"    

        """
        spike interface preprocessing does not work... 
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
        if t_end_preproc is not None:
            recording = recording.frame_slice(start_frame=int(sampling_rate * t_start_preproc), end_frame=int(sampling_rate * t_end_preproc))

        if apply_filter:
            recording = highpass_filter(recording, freq_min=low_frequency, filter_order=order)
        if standardize:
            recording = zscore(recording)
        if adcshift_correction:
            sampShifts = npSampShifts()
            recording = phase_shift(recording, inter_sample_shift=sampShifts)
        if median_subtraction:
            recording = common_reference(recording)

        recording.save(folder=preprocessing_dir, n_jobs=n_job_preprocessing, chunk_size=sampling_rate*n_sec_chunk_preprocessing, progressbar=True)
        """
        
    # %%
    # Subtraction 
    t_start_detect-=t_start_preproc
    if t_end_detect is None:
        t_end_detect=rec_len_sec
    t_end_detect-=t_start_preproc

    # %%
    detect_dir = Path(output_all) / "initial_detect_localize"
    Path(detect_dir).mkdir(exist_ok=True)
    
    if detect_localize:
        print("Detection...")

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
            loc_feature=loc_feature,
        )

    # %%
    else:
        sub_h5=subh5_name

    # %%
    with h5py.File(sub_h5, "r+") as h5:
        cleaned_tpca_group = h5["cleaned_tpca"]
        tpca_mean = cleaned_tpca_group["tpca_mean"][:]
        tpca_components = cleaned_tpca_group["tpca_components"][:]
        localization_results = np.array(h5["localizations{}".format(loc_feature)][:]) 
        maxptps = np.array(h5["maxptps"][:])
        spike_index = np.array(h5["spike_index"][:])
    # %%
    # Load tpca
    tpca = PCA(tpca_components.shape[0])
    tpca.mean_ = tpca_mean
    tpca.components_ = tpca_components

    # %%
    z = localization_results[:, 2]
    x = localization_results[:, 0]

    # %%
    # remove spikes localized at boundaries
    x_bound_low = geom[:, 0].min()
    x_bound_high = geom[:, 0].max()
    z_bound_low = geom[:, 1].min()
    z_bound_high = geom[:, 1].max()
    idx_remove_too_far = np.flatnonzero(np.logical_and(
        np.logical_and(z>z_bound_low-100, z<z_bound_high+100),
        np.logical_and(x>x_bound_low-100, x<x_bound_high+100)))

    # %%
    maxptps = maxptps[idx_remove_too_far]
    z = z[idx_remove_too_far]
    spike_index = spike_index[idx_remove_too_far]
    x = x[idx_remove_too_far]
    localization_results = localization_results[idx_remove_too_far]

    if registration:
        print("Registration...")

        device_reg = "cpu"
        motion_est, extra = newt.register(
                maxptps,
                z,
                spike_index[:, 0]/sampling_rate,
                rigid=True,
                win_step_um=50, 
                win_scale_um=150,
                win_margin_um=-75,
                raster_kw=dict(gaussian_smoothing_sigma_um=1,post_transform=None,amp_scale_fn=None,avg_in_bin=True),
                weights_kw=dict(weights_threshold_low=0.2, weights_threshold_high=0.2, mincorr=0.1, max_dt_s=100),
                thomas_kw=dict(lambda_s=0, lambda_t=1),
                max_disp_um=100,
                device=device_reg,
                pbar=False,
        )

        displacement_rigid = motion_est.displacement

        fname_disp = Path(detect_dir) / "displacement_rigid.npy"
        np.save(fname_disp, displacement_rigid)
    else:
        fname_disp = Path(detect_dir) / "displacement_rigid.npy"
        displacement_rigid = np.load(fname_disp)

    # %%
    if savefigs:
        if t_end_detect is None:
            t_end_detect = t_start_detect + len(displacement_rigid)

        vir = cm.get_cmap('jet')
        ptp_arr = maxptps.copy()
        ptp_arr = np.log(ptp_arr)
        ptp_arr -= ptp_arr.min()
        ptp_arr /= ptp_arr.max()
        color_array = vir(ptp_arr)
        fname_detect_fig = Path(detect_dir) / "detection_displacement_raster_plot.png"
        plt.figure(figsize = (10, 5))
        plt.scatter(spike_index[:, 0]/sampling_rate, z, color = color_array, s = 1, alpha=0.05)
        plt.plot(np.arange(t_start_detect, t_end_detect), displacement_rigid[t_start_detect:t_end_detect]-displacement_rigid[t_start_detect], color = 'red')
        plt.plot(np.arange(t_start_detect, t_end_detect), displacement_rigid[t_start_detect:t_end_detect]-displacement_rigid[t_start_detect]+100, color = 'red')
        plt.plot(np.arange(t_start_detect, t_end_detect), displacement_rigid[t_start_detect:t_end_detect]-displacement_rigid[t_start_detect]+200, color = 'red')
        plt.savefig(fname_detect_fig)
        plt.close()


    # %%
    # Clustering 
    cluster_dir = Path(output_all) / "initial_clustering"
    Path(cluster_dir).mkdir(exist_ok=True)
    if clustering:
        print("Clustering...")
        if t_end_clustering is None:
            t_end_clustering=rec_len_sec
            
        time_temp_computation = t_end_clustering//2 # TODO: remove this later 
        t_start_clustering-=t_start_preproc
        t_end_clustering-=t_start_preproc

        spt, maxptps, x, z, spike_index = run_full_clustering(t_start_clustering, t_end_clustering, cluster_dir, raw_data_name, geom, spike_index,
                                                    localization_results, maxptps, displacement_rigid, len_chunks=len_chunks_cluster, threshold_ptp=threshold_ptp_cluster,
                                                    fs=sampling_rate, triage_quantile_cluster=triage_quantile_cluster, frame_dedup_cluster=frame_dedup_cluster, 
                                                    time_temp_comp_merge=time_temp_computation, log_c=log_c, scales=scales, savefigs=savefigs, deconv_resid_th=0.5)
    
    else:
        fname_spt_cluster = Path(cluster_dir) / "spt_full_cluster.npy"
        fname_x = Path(cluster_dir) / "x_full_cluster.npy"
        fname_z = Path(cluster_dir) / "z_full_cluster.npy"
        fname_maxptps = Path(cluster_dir) / "maxptps_full_cluster.npy"
        fname_spike_index = Path(cluster_dir) / "spike_index_full_cluster.npy"
        
        spt = np.load(fname_spt_cluster)
        x = np.load(fname_x)
        z = np.load(fname_z)
        maxptps = np.load(fname_maxptps)
        spike_index = np.load(fname_spike_index)

    idx_kept = np.flatnonzero(spt[:, 1]>-1)
    x = x[idx_kept]
    z = z[idx_kept]
    maxptps = maxptps[idx_kept]
    spike_index = spike_index[idx_kept]
    spt = spt[idx_kept]
    
    if deconvolve:
        print("First Deconvolution...")
        deconv_dir_all = Path(output_all) / "deconvolution"
        Path(deconv_dir_all).mkdir(exist_ok=True)
        deconv_dir = Path(deconv_dir_all) / "deconv_results"
        Path(deconv_dir).mkdir(exist_ok=True)
        extract_deconv_dir = Path(deconv_dir_all) / "deconv_extracted"
        Path(extract_deconv_dir).mkdir(exist_ok=True)

        if t_end_deconv is None:
            t_end_deconv=rec_len_sec

        if adaptive_bin_size_selection:
            n_units = spt[:, 1].max()+1
            spread_x = np.zeros(n_units)
            for k in range(n_units):
                idx_k = np.flatnonzero(spt[:, 1]==k)
                # Next thing: USE STD INSTEAD OF MAD
                spread_x[k] = 1.65*np.median(np.abs(x[idx_k]-np.median(x[idx_k])))/0.6745

            pitch = get_pitch(np.load(geom_path))
            divisors = np.arange(1, pitch+1)
            divisors = divisors[6 % divisors==0]
            bins_sizes_um=divisors[np.abs(spread_x[:, None] - divisors).argmin(1)].astype('int')
            print(bins_sizes_um)
        else:
            bins_sizes_um=None

    #         Uncomment To start where we were :) 
    #         fname_spt_cluster = extract_deconv_dir / "spike_train_final_deconv.npy"
    #         fname_x = extract_deconv_dir / "x_final_deconv.npy"
    #         fname_z = extract_deconv_dir / "z_final_deconv.npy"
    #         fname_maxptps = extract_deconv_dir / "maxptps_final_deconv.npy"
    #         fname_dist_metric = extract_deconv_dir / "dist_metric_final_deconv.npy"

    #         spt = np.load(fname_spt_cluster)
    #         x = np.load(fname_x)
    #         z = np.load(fname_z)
    #         maxptps = np.load(fname_maxptps)
    #         dist_metric = np.load(fname_dist_metric)

        dist_metric=None

        deconv_h5 = full_deconv_with_update(deconv_dir, extract_deconv_dir,
                   raw_data_name, geom, displacement_rigid,
                   spt, spike_index, maxptps, x, z, t_start_deconv, t_end_deconv, sub_h5, 
                   n_sec_temp_update=n_sec_temp_update, 
                   bin_size_um=bin_size_um,
                   bins_sizes_um=bins_sizes_um,
                   pfs=sampling_rate,
                   n_jobs=n_jobs_deconv,
                   n_jobs_extract_deconv=n_jobs_extract_deconv,
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
                   adaptive_th_for_temp_computation=adaptive_th_for_temp_computation,
                   poly_params=poly_params,
                   extract_radius_um=extract_box_radius,
                   loc_radius=localize_radius,
                   loc_feature=loc_feature,
                   n_sec_train_feats=n_sec_train_feats,
                   n_sec_chunk=n_sec_chunk_deconv,
                   overwrite=overwrite_deconv,
                   p_bar=True,
                   save_chunk_results=False,
                   dist_metric=dist_metric,
                   save_cleaned_tpca_projs=save_cleaned_tpca_projs,
                   save_temps=False)


    #         if savefigs:
    #             spt = np.load(Path(extract_deconv_dir) / "spike_train_final_deconv.npy")
    #             z = np.load(Path(extract_deconv_dir) / "z_final_deconv.npy")

    #             fname_deconv_fig=Path(extract_deconv_dir) / "full_deconv_raster_plot.png"
    #             ccolors = ccet.glasbey[:spt[:, 1].max()+1]
    #             plt.figure(figsize = (10, 5))
    #             for k in range(spt[:, 1].max()+1):
    #                 idx = spt[:, 1]==k
    #                 plt.scatter(spt[idx, 0]//sampling_rate, z[idx], c = ccolors[k], s = 1, alpha = 0.1)
    #             plt.savefig(fname_deconv_fig)
    #             plt.close()
    else:
        deconv_dir_all = Path(output_all) / "deconvolution"
        Path(deconv_dir_all).mkdir(exist_ok=True)
        deconv_dir = Path(deconv_dir_all) / "deconv_results"
        Path(deconv_dir).mkdir(exist_ok=True)
        extract_deconv_dir = Path(deconv_dir_all) / "deconv_extracted"
        Path(extract_deconv_dir).mkdir(exist_ok=True)
    if second_deconvolve:

        deconv_h5 = Path(extract_deconv_dir) / 'deconv_results.h5'

        print("Second Deconvolution...")
        deconv_dir_all = Path(output_all) / "second_deconv"
        Path(deconv_dir_all).mkdir(exist_ok=True)
        deconv_dir = Path(deconv_dir_all) / "deconv_results"
        Path(deconv_dir).mkdir(exist_ok=True)
        extract_deconv_dir = Path(deconv_dir_all) / "deconv_extracted"
        Path(extract_deconv_dir).mkdir(exist_ok=True)
        split_merge_dir = Path(deconv_dir_all) / "split_merge_input"
        Path(split_merge_dir).mkdir(exist_ok=True)

        if t_end_deconv is None:
            t_end_deconv=rec_len_sec

        if adaptive_bin_size_selection:
            n_units = spt[:, 1].max()+1
            spread_x = np.zeros(n_units)
            for k in range(n_units):
                idx_k = np.flatnonzero(spt[:, 1]==k)
                # Next thing: USE STD INSTEAD OF MAD
                spread_x[k] = 1.65*np.median(np.abs(x[idx_k]-np.median(x[idx_k])))/0.6745

            pitch = get_pitch(np.load(geom_path))
            divisors = np.arange(1, pitch+1)
            divisors = divisors[6 % divisors==0]
            bins_sizes_um=divisors[np.abs(spread_x[:, None] - divisors).argmin(1)].astype('int')
            print(bins_sizes_um)
        else:
            bins_sizes_um=None

    #         Uncomment To start where we were :) 
    #         fname_spt_cluster = extract_deconv_dir / "spike_train_final_deconv.npy"
    #         fname_x = extract_deconv_dir / "x_final_deconv.npy"
    #         fname_z = extract_deconv_dir / "z_final_deconv.npy"
    #         fname_maxptps = extract_deconv_dir / "maxptps_final_deconv.npy"
    #         fname_dist_metric = extract_deconv_dir / "dist_metric_final_deconv.npy"

    #         spt = np.load(fname_spt_cluster)
    #         x = np.load(fname_x)
    #         z = np.load(fname_z)
    #         maxptps = np.load(fname_maxptps)
    #         dist_metric = np.load(fname_dist_metric)

        dist_metric=None

        with h5py.File(deconv_h5, "r+") as h5:
            spike_index = np.array(h5["spike_index"][:])
            channel_index = np.array(h5["channel_index"][:])
            spt = np.array(h5["deconv_spike_train"][:])
            maxptps = np.array(h5["maxptps"][:])
            channel_index = np.array(h5["channel_index"][:])
            localizations = np.array(h5["localizations{}".format(loc_feature)][:])
        x = localizations[:, 0]
        z = localizations[:, 2]
        z_reg = z - displacement_rigid[spt[:, 0]//sampling_rate]

        print("Split/Merge...")

        labels_split_merge = run_full_merge_split(deconv_h5, spt, spike_index, 
                             channel_index, geom, raw_data_name,
                             z, z_reg, x)

        np.save(split_merge_dir / "labels_input_split_merged.npy", labels_split_merge)

#         labels_split_merge = np.load(split_merge_dir / "labels_input_split_merged.npy")
        spt[:, 1] = labels_split_merge

        print("Second Deconv...")
        deconv_h5 = full_deconv_with_update(deconv_dir, extract_deconv_dir,
                   raw_data_name, geom, displacement_rigid,
                   spt, spike_index, maxptps, x, z, t_start_deconv, t_end_deconv, deconv_h5, 
                   n_sec_temp_update=n_sec_temp_update, 
                   bin_size_um=bin_size_um,
                   bins_sizes_um=bins_sizes_um,
                   pfs=sampling_rate,
                   n_jobs=n_jobs_deconv,
                   n_jobs_extract_deconv=n_jobs_extract_deconv,
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
                   adaptive_th_for_temp_computation=adaptive_th_for_temp_computation,
                   poly_params=poly_params,
                   extract_radius_um=extract_box_radius,
                   loc_radius=localize_radius,
                   loc_feature=loc_feature,
                   n_sec_train_feats=n_sec_train_feats,
                   n_sec_chunk=n_sec_chunk_deconv,
                   overwrite=overwrite_deconv,
                   p_bar=True,
                   save_chunk_results=False,
                   dist_metric=dist_metric,
                   save_cleaned_tpca_projs=save_cleaned_tpca_projs, 
                   save_temps=False)

    if third_deconvolve:
        print("Third Deconvolution...")
        deconv_dir_all = Path(output_all) / "third_deconv"
        Path(deconv_dir_all).mkdir(exist_ok=True)
        deconv_dir = Path(deconv_dir_all) / "deconv_results"
        Path(deconv_dir).mkdir(exist_ok=True)
        extract_deconv_dir = Path(deconv_dir_all) / "deconv_extracted"
        Path(extract_deconv_dir).mkdir(exist_ok=True)
        split_merge_dir = Path(deconv_dir_all) / "split_merge_input"
        Path(split_merge_dir).mkdir(exist_ok=True)

        if t_end_deconv is None:
            t_end_deconv=rec_len_sec

        if adaptive_bin_size_selection:
            n_units = spt[:, 1].max()+1
            spread_x = np.zeros(n_units)
            for k in range(n_units):
                idx_k = np.flatnonzero(spt[:, 1]==k)
                # Next thing: USE STD INSTEAD OF MAD
                spread_x[k] = 1.65*np.median(np.abs(x[idx_k]-np.median(x[idx_k])))/0.6745

            pitch = get_pitch(np.load(geom_path))
            divisors = np.arange(1, pitch+1)
            divisors = divisors[6 % divisors==0]
            bins_sizes_um=divisors[np.abs(spread_x[:, None] - divisors).argmin(1)].astype('int')
            print(bins_sizes_um)
        else:
            bins_sizes_um=None

        dist_metric=None

        with h5py.File(deconv_h5, "r+") as h5:
            spike_index = np.array(h5["spike_index"][:])
            channel_index = np.array(h5["channel_index"][:])
            spt = np.array(h5["deconv_spike_train"][:])
            maxptps = np.array(h5["maxptps"][:])
            channel_index = np.array(h5["channel_index"][:])
            localizations = np.array(h5["localizations{}".format(loc_feature)][:])
        x = localizations[:, 0]
        z = localizations[:, 2]
        z_reg = z - displacement_rigid[spt[:, 0]//sampling_rate]

        print("Split/Merge...")

        labels_split_merge = run_full_merge_split(deconv_h5, spt, spike_index, 
                             channel_index, geom, raw_data_name,
                             z, z_reg, x)

        np.save(split_merge_dir / "labels_input_split_merged.npy", labels_split_merge)

        spt[:, 1] = labels_split_merge

        print("Third Deconv...")

        deconv_h5 = full_deconv_with_update(deconv_dir, extract_deconv_dir,
                   raw_data_name, geom, displacement_rigid,
                   spt, spike_index, maxptps, x, z, t_start_deconv, t_end_deconv, deconv_h5, 
                   n_sec_temp_update=n_sec_temp_update, 
                   bin_size_um=bin_size_um,
                   bins_sizes_um=bins_sizes_um,
                   pfs=sampling_rate,
                   n_jobs=n_jobs_deconv,
                   n_jobs_extract_deconv=n_jobs_extract_deconv,
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
                   adaptive_th_for_temp_computation=adaptive_th_for_temp_computation,
                   poly_params=poly_params,
                   extract_radius_um=extract_box_radius,
                   loc_radius=localize_radius,
                   loc_feature=loc_feature,
                   n_sec_train_feats=n_sec_train_feats,
                   n_sec_chunk=n_sec_chunk_deconv,
                   overwrite=overwrite_deconv,
                   p_bar=True,
                   save_chunk_results=False,
                   dist_metric=dist_metric,
                   save_cleaned_tpca_projs=save_cleaned_tpca_projs, 
                   save_temps=False)

    with h5py.File(deconv_h5, "r+") as h5:
        # print(subject)
        # print("-" * len(subject))
        for k in h5:
            print(" - ", k) #, h5[k].shape
        geom = h5["geom"][:]
        geom_array = np.array(h5["geom"][:])
        spike_index = np.array(h5["spike_index"][:])
        channel_index = np.array(h5["channel_index"][:])
        spt = np.array(h5["deconv_spike_train"][:])
        maxptps = np.array(h5["maxptps"][:])
        spike_index = np.array(h5["spike_index"][:])    
        localizations = np.array(h5["localizationspeak"][:])
        dist_metric = np.array(h5["deconv_dist_metrics"][:])

    x = localizations[:, 0]
    z_abs = localizations[:, 2]

    labels_final = final_split_merge(spt, z_abs, x, displacement_rigid, geom, raw_data_name)
    np.save(Path(deconv_dir_all) / "labels_final.npy", labels_final)
        
        
        
        
