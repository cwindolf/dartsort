import numpy as np
from pathlib import Path
import os
import h5py
from tqdm.auto import tqdm
import torch
import pickle 

from dartsort.util.data_util import DARTsortSorting
from dartsort import main, config
from dartsort.main import subtract, cluster, match
from dartsort.cluster import merge, split
from dartsort.config import *
from dartsort.vis.scatterplots import get_ptp_order_and_alphas
from dartsort.cluster.merge import merge_iterative_templates_with_multiple_chunks, single_merge_GC_multiple_chunks, merge_templates_across_multiple_chunks
from dartsort.cluster.reassignment import iterative_split_merge_reassignment
from dartsort.util.data_util import (
    chunk_time_ranges, 
    DARTsortSorting,
    check_recording,
    keep_only_most_recent_spikes,
)
from dartsort.templates import TemplateData

from dataclasses import replace
from spike_psvae import filter_standardize # Write this into dartsort

from dredge import dredge_ap, motion_util as mu

import spikeinterface.core as sc
import spikeinterface.full as si

from spike_psvae.cluster_viz import array_scatter

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import colorcet as cc 

from spikeinterface.extractors import read_nwb_recording

ccolors = cc.glasbey[:31]
def get_ccolor(k):
    if k == -1:
        return "#808080"
    else:
        return ccolors[k % len(ccolors)]
jet = cm.get_cmap("jet")

# The following parameters are used to sort UHD and resampled (NP1, NP2, Large-Dense) recordings

subtraction_config=SubtractionConfig(
    detection_thresholds=(12, 9, 6, 5, 4),
    max_waveforms_fit=20_000,
    subtraction_denoising_config=FeaturizationConfig(
        denoise_only=True,
        input_waveforms_name="raw",
        output_waveforms_name="subtracted",
        tpca_fit_radius=75.0,
    ),
    residnorm_decrease_threshold=5,
    extract_radius=75.0,
)
template_config=TemplateConfig(
    superres_templates=False,
    spatial_svdsmoothing = True,
    time_tracking = True,
    spikes_per_unit=200,
    subchunk_time_smoothing=True,
    realign_peaks=True,
)

clustering_config = ClusteringConfig(
    cluster_strategy="density_peaks",
    sigma_local=2.5,
    sigma_regional=25,
    noise_density=0.1,
    n_neighbors_search=50,
    radius_search=50,
    remove_clusters_smaller_than=25,
    revert=True,
    use_y_triaging=True,
    triage_quantile_before_clustering=0.,
    triage_quantile_per_cluster=0.2,
    amp_no_triaging_after_clustering=8,
    ramp_triage_per_cluster=True,
)

split_merge_config = SplitMergeConfig(
      m_iter=1,
      amplitude_scaling_boundary=0.01,
      mask_units_too_far=True,
      merge_distance_threshold=0.33,
      linkage="complete",
      min_channel_amplitude=0.0,
      min_spatial_cosine=0.5,
      sigma_local=1.5,
      radius_search=100,
      n_neighbors_search=300,
      noise_density=0.,
      remove_clusters_smaller_than=10
)
    
motion_estimation_config=MotionEstimationConfig(
    window_scale_um=300,
    window_step_um=200,
    max_disp_um=100,
    correlation_threshold = 0.6
)

# For .bin files coming from older format
# preprocessing_config = default_preprocessing_config
# needs_preprocessing = False
# For .bin files coming from nwb data
preprocessing_config = PreprocessingConfig(
    n_channels_before_preprocessing=384,
    channels_to_remove=None,
)

computation_config = ComputationConfig(n_jobs_gpu=0, n_jobs_cpu=4, device = "cuda:1")
trough_offset = default_waveform_config.trough_offset_samples()
spike_length_samples = default_waveform_config.spike_length_samples()

# SET THE FOLLOWING PARAMETERS: 
needs_preprocessing = False # If you need to preprocess your data: standardize + high-pass filter + ADC shift correction + destriping (median subtraction)
needs_subtraction = False # If you need to run subtraction and DREDge (if this has already been done set it to False)
overwrite_all = False # Set to False if the code broke and you want to start from where it left off

dtype_preprocessed = "float32" #Script -> what if needs preprocessing
sampling_rate = 30_000
template_npz_filename="template_data.npz"
matchh5_name="matching0.h5"

# To run on 10 sessions
data_names = [
    Session names
   "ZYE_0021___2021-05-01___1",
   "ZYE_0031___2021-12-03___1___p0_g0_imec0",
   "ZYE_0040___2021-08-15___3___p1_g0_imec0", 
    "ZYE_0021___2021-05-01___4___p2_g0_imec0",
    "ZYE_0057___2022-02-07",
    "ZYE_0057___2022-02-04___1___p0_g0___p0_g0_imec0",
    "ZYE_0031___2021-12-01___1___p0_g0_imec0",
    "ZYE_0057___2022-02-07",
    "ZYE_0021___2021-05-01___4___p2_g0_imec0",
    "ZYE_0057___2022-02-04___1___p0_g0___p0_g0_imec0",
    "ZYE_0031___2021-12-02___4___p2_g0_imec0",
    "ZYE_0057___2022-02-03___2___p1_g0___p1_g0_imec0",
    "LK_0011___2021-12-06___1___p0_g0_imec0"
]

slices_all = [
    # Here, cut parts of recording with low-quality data    
    [280, None],
    [350, None],
    [None, None],
    [200, None],
    [None, None],
    [None, 3750],
    [None, None],
    [None, None],
    [200, None],
    [None, 3750],
    [None, 3700],
    [None, 3400],
    [None, None],
]

all_patterns = ["2", "3", "1", "4"]
#Patterns to probe relation: 2=NP1, 3=NP2, 1=UHD, 4=Large-Dense
full_dir = Path("UHD_DATA") 

if __name__ == '__main__':

    denoising_tsvd = None

    create_summary_figs_all_steps = True
    for name_recording, slice_s in zip(
        data_names, 
        slices_all):

        for pat, loc_radius, localization_model, match_threshold in zip(all_patterns, [100, 100, 50, 75], ["pointsource", "pointsource", "dipole", "dipole"], [113, 145, 250, 173]):
            # The parameters in this for loop are what differ between patterns

            match_threshold *= 2

            #Matching + Featurization params depend on the probe
            matching_config=MatchingConfig(
                threshold=match_threshold, 
                max_waveforms_fit=20_000,
                extract_radius=75.0,
                conv_ignore_threshold=0.0,
                chunk_length_samples=30_000,
            ) 
            featurization_config=FeaturizationConfig(
                tpca_fit_radius=75.0,
                localization_radius=loc_radius, 
                localization_model=localization_model,
            )
    
            name = name_recording + f"_pat{pat}"
            print(f"Running Dartsort on {name}")
            
            os.makedirs(full_dir, exist_ok = True)
            data_dir = full_dir / name
            os.makedirs(data_dir, exist_ok = True)
            raw_data_name = data_dir / "standardized.bin"

            #Here set the patyh to the correct geometry file
            geom = np.load(full_dir / f"geom_array_pat{pat}.npy")
            
            if create_summary_figs_all_steps:
                zlim = (geom[:, 1].min()-100, geom[:, 1].max()+100)
                xlim = (geom[:, 0].min()-50, geom[:, 0].max()+50)
    
            preprocessing_config = PreprocessingConfig(
                n_channels_before_preprocessing = geom.shape[0],
                channels_to_remove = None,
                adcshift_correction=False,
            )

            
            subtraction_dir = data_dir / "subtraction_results"
            sub_h5 = subtraction_dir / "subtraction.h5"
            motion_estimate_name = subtraction_dir / "motion_estimate.obj"

            data_dir = full_dir / name
            os.makedirs(data_dir, exist_ok = True)
            data_dir_cluster = data_dir / "initial_clustering"
            os.makedirs(data_dir_cluster, exist_ok = True)
            data_dir_first_deconv = data_dir / "first_deconv"
            os.makedirs(data_dir_first_deconv, exist_ok = True)
            data_dir_second_deconv = data_dir / "second_deconv"
            os.makedirs(data_dir_second_deconv, exist_ok = True)
        
        
            if needs_preprocessing:
                rec_len_sec = int(os.path.getsize(raw_data_name)/2/preprocessing_config.n_channels_before_preprocessing/sampling_rate)
                
                filter_standardize.filter_standardize_rec_mp(
                    data_dir, raw_data_name, preprocessing_config.dtype_raw, rec_len_sec, geom,
                    n_channels = preprocessing_config.n_channels_before_preprocessing,
                    channels_to_remove=preprocessing_config.channels_to_remove, 
                    t_start=0, t_end=rec_len_sec,
                    apply_filter=preprocessing_config.apply_filter,
                    low_frequency=preprocessing_config.low_frequency, 
                    high_factor=preprocessing_config.high_factor, 
                    order=preprocessing_config.order, 
                    sampling_frequency=sampling_rate,
                    median_subtraction=preprocessing_config.median_subtraction,
                    adcshift_correction=preprocessing_config.adcshift_correction,
                    n_jobs=preprocessing_config.n_jobs_preprocessing
                )
                if default_preprocessing_config.delete_original_data:
                    os.remove(raw_data_name)
                raw_data_name = data_dir / "standardized.bin"
                
                rec_len_sec_bis = int(os.path.getsize(raw_data_name)/4/preprocessing_config.n_channels_before_preprocessing/sampling_rate)
                print("recording len")
                print(rec_len_sec_bis)
        
                if preprocessing_config.channels_to_remove is not None:
                    assert rec_len_sec_bis == (preprocessing_config.n_channels_before_preprocessing-len(preprocessing_config.channels_to_remove))*rec_len_sec/preprocessing_config.n_channels_before_preprocessing, "Preprocessing gone wrong"
                else:
                    assert rec_len_sec_bis == rec_len_sec, "Preprocessing gone wrong"
            
            recording = sc.read_binary(
                    raw_data_name,
                    sampling_rate,
                    dtype_preprocessed,
                    num_channels=geom.shape[0],
                    is_filtered=True,
                )
                    
            recording.set_dummy_probe_from_locations(
                geom, shape_params=dict(radius=10)
            )
    
            if slice_s[0] is not None:
                start_time = slice_s[0]
            else:
                start_time = 0
            if slice_s[1] is not None:
                end_time = slice_s[1]
            else:
                end_time = recording.get_duration()
            slice_s = (start_time, end_time)
    
            print(f"Running dartsort on {slice_s}")
        
            chunk_starts_samples_sub = np.arange(
                slice_s[0] * recording.sampling_frequency,
                slice_s[1] * recording.sampling_frequency,
                recording.sampling_frequency,
            ).astype("int")
    
        
            print("Preprocessing done")
        
            if needs_subtraction:
                subtract(
                    recording,
                    subtraction_dir,
                    featurization_config=featurization_config,
                    subtraction_config=subtraction_config,
                    chunk_starts_samples=chunk_starts_samples_sub,
                    n_jobs=computation_config.n_jobs_gpu,
                    device = computation_config.device,
                    overwrite=True,
                )
            
            print("Subtraction done")
        
            with h5py.File(sub_h5, "r+") as h5:
                localization_results = np.array(h5["point_source_localizations"][:])
                a = np.array(h5["denoised_ptp_amplitudes"][:])
                times_seconds = np.array(h5["times_seconds"][:])
                times_samples = np.array(h5["times_samples"][:])
                channels = np.array(h5["channels"][:])
                geom = np.array(h5["geom"][:])
            peeling_featurization_pt = subtraction_dir / "subtraction_models/featurization_pipeline.pt"
            tpca_features_dataset_name="collisioncleaned_tpca_features"
            feature_pipeline = torch.load(peeling_featurization_pt)
            tpca_feature = [
                f
                for f in feature_pipeline.transformers
                if f.name == tpca_features_dataset_name
            ]
            tpca = tpca_feature[0].to_sklearn()
    
            if not os.path.exists(motion_estimate_name) or overwrite_all:
        
                z = localization_results[:, 2]
                wh = (z > geom[:,1].min() - 100) & (z < geom[:,1].max() + 100)
                a_dredge = a[wh]
                z = z[wh]
                t = times_seconds[wh]
                
                me, extra = dredge_ap.register(a_dredge, z, t, max_disp_um=motion_estimation_config.max_disp_um, win_scale_um=motion_estimation_config.window_scale_um, win_step_um=motion_estimation_config.window_step_um, mincorr=motion_estimation_config.correlation_threshold)
    
                file_pickle = open(motion_estimate_name, 'wb') 
                pickle.dump(me, file_pickle)
                file_pickle.close()
    
                depth_reg = me.correct_s(times_seconds, localization_results[:, 2])
                if create_summary_figs_all_steps and needs_subtraction:
                    fig_dir = subtraction_dir / "figures"
                    os.makedirs(fig_dir, exist_ok = True)
            
                    color_array = np.log(a)
                    color_array -= color_array.min()
                    color_array /= color_array.max()
                    color_array = jet(color_array)
                    plt.figure(figsize = (10, 5))
                    plt.scatter(times_seconds, localization_results[:, 2], 
                                c=color_array, s=1, alpha = 0.1)
                    plt.ylim((geom.min()-100, geom.max()+100))
                    plt.xlabel("Time (s)")
                    plt.ylabel("Depth (um)")
                    plt.title("Unregistered raster colored by log(ptp)")
                    plt.savefig(fig_dir / "unregistered_raster.png")
                    plt.close()
            
                    plt.figure(figsize = (10, 5))
                    plt.scatter(times_seconds, depth_reg, 
                                c=color_array, s=1, alpha = 0.1)
                    plt.ylim((geom.min()-100, geom.max()+100))
                    plt.xlabel("Time (s)")
                    plt.ylabel("Registered Depth (um)")
                    plt.title("Registered raster colored by log(ptp)")
                    plt.savefig(fig_dir / "registered_raster.png")
                    plt.close()
            else:
                filehandler =open(motion_estimate_name, 'rb') 
                me = pickle.load(filehandler)
                depth_reg = me.correct_s(times_seconds, localization_results[:, 2])
    
            chunk_time_ranges_s = chunk_time_ranges(recording, chunk_length_samples=template_config.chunk_size_s*recording.sampling_frequency, slice_s=slice_s)
            n_chunks = len(chunk_time_ranges_s)
        
            cluster_labels_name = data_dir_cluster / "clustering_labels.npy"
            if not overwrite_all and os.path.exists(cluster_labels_name):
                print("Clustering already done!")
                sorting = DARTsortSorting(
                    times_samples=times_samples,
                    channels=channels,
                    labels=np.load(cluster_labels_name),
                    parent_h5_path=sub_h5,
                    extra_features={
                        "point_source_localizations": localization_results,
                        "denoised_ptp_amplitudes": a,
                        "times_seconds": times_seconds,
                    },
                )
            else:
                sorting = cluster(
                    sub_h5,
                    recording,
                    overwrite=overwrite_all,
                    clustering_config=clustering_config, 
                    motion_est=me,
                    slice_s=slice_s, 
                    tpca=tpca,
                    wfs_name="collisioncleaned_tpca_features",
                    trough_offset=trough_offset,
                )
        
                np.save(cluster_labels_name, sorting.labels)
        
                if create_summary_figs_all_steps:
                    units_plot, alphas = get_ptp_order_and_alphas(sorting)
                    
                    fig_dir = data_dir_cluster / "figures"
                    os.makedirs(fig_dir, exist_ok = True)
                
                    fig, axes = array_scatter(
                      sorting.labels, geom, localization_results[:, 0], depth_reg, 
                      a, zlim=zlim, do_ellipse=True, xlim=xlim,
                    )
                    plt.savefig(fig_dir / "cluster_scatter.png")
                    plt.close()
                
                    plt.figure(figsize=(10, 5))
                    for c, u in enumerate(units_plot):
                        plt.scatter(times_seconds[sorting.labels == u], depth_reg[sorting.labels == u], c = get_ccolor(u), s=1, alpha = alphas[c])
                    plt.ylim(zlim)
                    plt.savefig(fig_dir / "cluster_raster.png")
                    plt.close()

            cluster_split_merge_labels_name = data_dir_cluster / "clustering_split_merge_labels.npy"
            if not overwrite_all and os.path.exists(cluster_split_merge_labels_name):
                sorting = DARTsortSorting(
                    times_samples=times_samples,
                    channels=channels,
                    labels=np.load(cluster_split_merge_labels_name),
                    parent_h5_path=sub_h5,
                    extra_features={
                        "point_source_localizations": localization_results,
                        "denoised_ptp_amplitudes": a,
                        "times_seconds": times_seconds,
                    },
                )
        
            else:
                sorting, denoising_tsvd = merge_iterative_templates_with_multiple_chunks(
                    recording,
                    sorting,
                    sub_h5,
                    split_merge_config = split_merge_config,
                    template_config = template_config,
                    motion_est=me,
                    chunk_time_ranges_s=chunk_time_ranges_s,
                    device=computation_config.device,
                    n_jobs=computation_config.n_jobs_gpu,
                    n_jobs_templates=computation_config.n_jobs_gpu,
                    template_save_folder=data_dir_cluster,
                    overwrite_templates=overwrite_all,
                    template_npz_filename=template_npz_filename,
                    reorder_by_depth=True,
                    trough_offset_samples=trough_offset,
                    spike_length_samples=spike_length_samples,
                    return_denoising_tsvd=True,
                    denoising_tsvd=denoising_tsvd,
                )
        
                np.save(cluster_split_merge_labels_name, sorting.labels)
        
                if create_summary_figs_all_steps:
                    units_plot, alphas = get_ptp_order_and_alphas(sorting)
                    
                    fig_dir = data_dir_cluster / "figures"
                    os.makedirs(fig_dir, exist_ok = True)
            
                    fig, axes = array_scatter(
                      sorting.labels, geom, localization_results[:, 0], depth_reg, 
                      a, zlim=zlim, do_ellipse=True, xlim=xlim,
                    )
                    plt.savefig(fig_dir / "cluster_split_merge_scatter.png")
                    plt.close()
            
                    plt.figure(figsize=(10, 5))
                    for c, u in enumerate(units_plot):
                        plt.scatter(times_seconds[sorting.labels == u], depth_reg[sorting.labels == u], c = get_ccolor(u), s=1, alpha = alphas[c])
                    plt.ylim(zlim)
                    plt.savefig(fig_dir / "cluster_split_merge_raster.png")
                    plt.close()
        
            print("Split Merge done!")
            print(f"N Units {sorting.labels.max()+1}")
        
            if os.path.exists(data_dir_first_deconv / matchh5_name) and not overwrite_all:
                matchh5 = data_dir_first_deconv / matchh5_name
                print(f"Deconvolution already done at {matchh5}!!")
                sorting = DARTsortSorting.from_peeling_hdf5(matchh5)
            else:
                sorting, matchh5 = match(
                    recording,
                    sorting=sorting,
                    output_directory=data_dir_first_deconv,
                    motion_est=me,
                    waveform_config=default_waveform_config,
                    template_config=template_config,
                    featurization_config=featurization_config,
                    matching_config=matching_config,
                    chunk_starts_samples=None,
                    slice_s=slice_s,
                    subsampling_proportion=1.0,
                    n_jobs_templates=computation_config.n_jobs_gpu,
                    n_jobs_match=computation_config.n_jobs_gpu,
                    overwrite=overwrite_all,
                    residual_filename=None,
                    show_progress=True,
                    device=computation_config.device,
                    hdf5_filename=matchh5_name,
                    model_subdir="matching0_models",
                    template_npz_filename=template_npz_filename,
                    templates_precomputed=True,
                    template_dir_precomputed=data_dir_cluster,
                )
        
                if create_summary_figs_all_steps:
                    units_plot, alphas = get_ptp_order_and_alphas(sorting)
                    fig_dir = data_dir_first_deconv / "figures"
                    os.makedirs(fig_dir, exist_ok = True)
            
                    plt.figure(figsize=(10, 5))
                    for c, u in enumerate(units_plot):
                        plt.scatter(sorting.times_seconds[sorting.labels == u], sorting.point_source_localizations[sorting.labels == u, 2], c = get_ccolor(u), s=1, alpha = alphas[c])
                    plt.ylim(zlim)
                    plt.savefig(fig_dir / "deconv_raster.png")
                    plt.close()
            
        
            post_split_merge_labels_name = data_dir_first_deconv / "split_merge_labels.npy"
        
            if os.path.exists(post_split_merge_labels_name) and not overwrite_all:
                print("Split-merge already done!")
                sorting = replace(sorting, labels = np.load(post_split_merge_labels_name))
            else:
                sorting = split.split_clusters(
                        sorting,
                        split_strategy="ZipperSplit",
                        split_strategy_kwargs=dict(
                            sorting=sorting,
                        ),
                        recursive=False,
                        n_jobs=computation_config.n_jobs_gpu,
                        motion_est=None, #doesn't matter here...
                    )
            
                sorting, denoising_tsvd = single_merge_GC_multiple_chunks(
                    recording,
                    sorting,
                    matchh5,
                    split_merge_config=split_merge_config,
                    template_config=template_config,
                    motion_est=me,
                    chunk_time_ranges_s=chunk_time_ranges_s,
                    device=computation_config.device,
                    n_jobs=computation_config.n_jobs_gpu,
                    n_jobs_templates=computation_config.n_jobs_gpu,
                    template_save_folder=data_dir_first_deconv,
                    overwrite_templates=overwrite_all,
                    template_npz_filename=template_npz_filename,
                    reorder_by_depth=True,
                    trough_offset_samples=trough_offset,
                    spike_length_samples=spike_length_samples,
                    denoising_tsvd=denoising_tsvd,
                    return_denoising_tsvd=True,
                )
        
                np.save(post_split_merge_labels_name, sorting.labels)
            
                if create_summary_figs_all_steps:
                    units_plot, alphas = get_ptp_order_and_alphas(sorting)
                    fig_dir = data_dir_first_deconv / "figures"
                    os.makedirs(fig_dir, exist_ok = True)
            
                    plt.figure(figsize=(10, 5))
                    for c, u in enumerate(units_plot):
                        plt.scatter(sorting.times_seconds[sorting.labels == u], sorting.point_source_localizations[sorting.labels == u, 2], c = get_ccolor(u), s=1, alpha = alphas[c])
                    plt.ylim(zlim)
                    plt.savefig(fig_dir / "post_split_merge_raster.png")
                    plt.close()
        
        
            print("First deconv split-merge done!")
            
            if os.path.exists(data_dir_second_deconv / matchh5_name) and not overwrite_all:
                matchh5 = data_dir_second_deconv / matchh5_name
                print(f"Second Deconvolution already done at {matchh5}!!")
                sorting = DARTsortSorting.from_peeling_hdf5(matchh5)
            else:
                sorting, matchh5 = match(
                    recording,
                    sorting=sorting,
                    output_directory=data_dir_second_deconv,
                    motion_est=me,
                    waveform_config=default_waveform_config,
                    template_config=template_config,
                    featurization_config=featurization_config,
                    matching_config=matching_config,
                    chunk_starts_samples=None,
                    slice_s=slice_s,
                    subsampling_proportion=1.0,
                    n_jobs_templates=computation_config.n_jobs_gpu,
                    n_jobs_match=computation_config.n_jobs_gpu,
                    overwrite=overwrite_all,
                    residual_filename=None,
                    show_progress=True,
                    device=computation_config.device,
                    hdf5_filename=matchh5_name,
                    model_subdir="matching0_models",
                    template_npz_filename=template_npz_filename,
                    templates_precomputed=True,
                    template_dir_precomputed=data_dir_first_deconv,
                )
            
            
                if create_summary_figs_all_steps:
                    units_plot, alphas = get_ptp_order_and_alphas(sorting)
                    fig_dir = data_dir_second_deconv / "figures"
                    os.makedirs(fig_dir, exist_ok = True)
            
                    plt.figure(figsize=(10, 5))
                    for c, u in enumerate(units_plot):
                        plt.scatter(sorting.times_seconds[sorting.labels == u], sorting.point_source_localizations[sorting.labels == u, 2], c = get_ccolor(u), s=1, alpha = alphas[c])
                    plt.ylim(zlim)
                    plt.savefig(fig_dir / "second_deconv_raster.png")
                    plt.close()
            
                print("Second deconvolution done!!")
                        
            template_data_list, denoising_tsvd = TemplateData.from_config_multiple_chunks_linear(
                recording,
                chunk_time_ranges_s,
                sorting,
                template_config,
                save_folder=data_dir_second_deconv,
                overwrite=True,
                motion_est=me,
                save_npz_name=template_npz_filename,
                n_jobs=0, 
                device="cuda:0",
                trough_offset_samples=42,
                spike_length_samples=121,
                denoising_tsvd=denoising_tsvd,
                return_denoising_tsvd=True,
            )

            # Then merge with template data list
            name_final_labels = data_dir_second_deconv / "post_merge_labels.npy"
            if os.path.exists(name_final_labels) and not overwrite_all:
                print(f"DARTSORT DONE!!")
            else:
                sorting = merge_templates_across_multiple_chunks(
                    sorting,
                    recording,
                    chunk_time_ranges_s,
                    template_data_list=template_data_list,
                    template_config=template_config,
                    motion_est=me,
                    superres_linkage=split_merge_config.superres_linkage,
                    link="single", #split_merge_config.linkage,
                    sym_function=split_merge_config.sym_function,
                    min_channel_amplitude=split_merge_config.min_channel_amplitude, 
                    min_spatial_cosine=split_merge_config.min_spatial_cosine,
                    max_shift_samples=split_merge_config.max_shift_samples,
                    merge_distance_threshold=0.5, #split_merge_config.merge_distance_threshold,
                    temporal_upsampling_factor=split_merge_config.temporal_upsampling_factor,
                    amplitude_scaling_variance=0.001,# split_merge_config.amplitude_scaling_variance,
                    amplitude_scaling_boundary=0.01, #split_merge_config.amplitude_scaling_boundary,
                    svd_compression_rank=split_merge_config.svd_compression_rank,
                    conv_batch_size=split_merge_config.conv_batch_size,
                    units_batch_size=split_merge_config.units_batch_size, 
                    mask_units_too_far=True, #split_merge_config.mask_units_too_far, 
                    aggregate_func=split_merge_config.aggregate_func,
                    denoising_tsvd=None,
                    return_denoising_tsvd=False,
                    trough_offset_samples=42,
                    spike_length_samples=121,
                )
    
                np.save(data_dir_second_deconv / "post_merge_labels.npy", sorting.labels)
    
                if create_summary_figs_all_steps:
                    units_plot, alphas = get_ptp_order_and_alphas(sorting)
                    fig_dir = data_dir_second_deconv / "figures"
                    os.makedirs(fig_dir, exist_ok = True)
                    depth_reg = me.correct_s(sorting.times_seconds, sorting.point_source_localizations[:, 2])
                    
                    plt.figure(figsize=(10, 5))
                    for c, u in enumerate(units_plot):
                        plt.scatter(sorting.times_seconds[sorting.labels == u], sorting.point_source_localizations[sorting.labels == u, 2], c = get_ccolor(u), s=1, alpha = alphas[c])
                    plt.ylim(zlim)
                    plt.savefig(fig_dir / "post_merge_second_deconv_raster.png")
                    plt.close()
        
                    fig, axes = array_scatter(
                      sorting.labels, geom, sorting.point_source_localizations[:, 0], depth_reg, 
                      sorting.denoised_ptp_amplitudes, zlim=zlim, do_ellipse=False, xlim=xlim,
                    )
                    plt.savefig(fig_dir / "post_merge_second_deconv_scatter.png")
                    plt.close()
                print(f"DARTSORT DONE!!")
    
