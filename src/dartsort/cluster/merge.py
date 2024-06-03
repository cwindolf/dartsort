from dataclasses import replace
from typing import Optional

from pathlib import Path
import numpy as np
from dartsort.config import TemplateConfig
from dartsort.templates import TemplateData, template_util
from dartsort.templates.pairwise_util import (
    construct_shift_indices, iterate_compressed_pairwise_convolutions)
from dartsort.util.data_util import DARTsortSorting, combine_sortings, chunk_time_ranges, keep_only_most_recent_spikes, update_sorting_chunk_spikes_loaded
from dartsort.util import spikeio
from dartsort.cluster.postprocess import chuck_noisy_template_units_with_loaded_spikes_per_chunk, chuck_noisy_template_units_from_merge
from dartsort.cluster.split import split_clusters
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import coo_array
from scipy.sparse.csgraph import maximum_bipartite_matching
from tqdm.auto import tqdm

from spike_psvae.cluster_viz import array_scatter_5_features, array_scatter, array_scatter_4_features
import matplotlib.pyplot as plt

import os

from . import cluster_util

def merge_iterative_templates_with_multiple_chunks(
    recording,
    sorting,
    spike_save_folder,
    sub_h5,
    template_data: Optional[TemplateData] = None,
    template_config: Optional[TemplateConfig] = None,
    motion_est=None,
    chunk_time_ranges_s=None,
    slice_s=[None,None],
    max_shift_samples=20,
    superres_linkage=np.max,
    linkage="complete",
    sym_function=np.maximum,
    merge_distance_threshold=0.25,
    temporal_upsampling_factor=8,
    amplitude_scaling_variance=0.001,
    amplitude_scaling_boundary=0.1,
    svd_compression_rank=20,
    min_channel_amplitude=0.0,
    min_spatial_cosine=0.5,
    conv_batch_size=128,
    units_batch_size=8, #change this :) 
    num_merge_iteration=3,
    iterative_split=True,
    device=None,
    n_jobs=0,
    n_jobs_templates=0,
    template_save_folder=None,
    overwrite_templates=True,
    overwrite_spikes=True,
    show_progress=True,
    template_npz_filename="template_data.npz",
    reorder_by_depth=True,
    delete_spikes=True,
    trough_offset_samples=42,
    spike_length_samples=121,
):
    """Template distance based merge, across chunks and iterative

    Pass in a sorting, recording and template config to make templates,
    and this will merge them (with superres). 
    It will create a template object for each chunk (defined by chunk_time_ranges_s or recording + slice_s), 
    and merge based on all temp data for all chunks
    It loads spikes only once to compute templates, to avoid reading recording back and forth

    Does not support superres templates yet 
    Also enforces n_jobs and n_jobs_templates = 0 since full spikes matrix is super large
    Could think of computing templates for each unit separately and loading spikes for groups of units all at once, on different jobs
    
    Arguments
    ---------
    spike_save_folder: to avoid using a large disk memory, load all spikes for each chunk 
    and save them as .npy rather than keeping everything in memory
    Then, reload .npy array rather than reading bin file and spikes_chunk_j.npy get updated as we merge! 
    same as other function
    
    
    Returns
    -------
    A new DARTsortSorting
    """
    os.makedirs(spike_save_folder, exist_ok=True)

    geom = np.load("/mnt/uhdlacie/geom_array_pat1.npy")
    fig_directory = spike_save_folder.parent / "figures"
    print("FIG DIRECTORY:")
    print(fig_directory)

    n_jobs=0
    n_jobs_templates=0

    if chunk_time_ranges_s is None: 
        chunk_time_ranges_s = chunk_time_ranges(recording, chunk_length_samples=template_config.chunk_size_s*recording.sampling_frequency, slice_s=slice_s)
    n_chunks = len(chunk_time_ranges_s)

    print("Loading spikes for all chunks")

    sortings_all = []
    labels_all = []
    times_all = []
    n_spikes_chunks = []
    for j, chunk_time_range in enumerate(chunk_time_ranges_s):

        save_wfs_name = spike_save_folder / f"spike_wfs_chunk{j}.npy"

        sorting_chunk = keep_only_most_recent_spikes(
            sorting,
            n_min_spikes=template_config.spikes_per_unit,
            latest_time_sample=chunk_time_range[1]
            * recording.sampling_frequency,
        )
        # This is the original - need to keep it to track spikes and labels updates correctly
        sortings_all.append(sorting_chunk)

        times_all = sorting_chunk.times_samples[sorting_chunk.labels>-1]
        times_unique, indices_unique_index, indices_unique_inverse = np.unique(times_all, return_inverse=True, return_index=True)

        # Better to just save and read wfs per chunks becuase too big otherwise
        if not os.path.exists(save_wfs_name) or overwrite_spikes: 
            waveforms_chunk = spikeio.read_full_waveforms(
                recording,
                times_unique,
                trough_offset_samples=trough_offset_samples,
                spike_length_samples=spike_length_samples,
                verbose=True,
            )
    
            np.save(save_wfs_name, waveforms_chunk)
    print("spikes loaded!")
    # TODO Iterate the merge!

    for miter in range(num_merge_iteration):

        print(f"ITERATION {miter}")
    
        dists_all = []
        shifts_all = []
        snrs_all = []
        template_data_all = []
        # unit_ids_all = []
        
        for j, chunk_time_range in tqdm(enumerate(chunk_time_ranges_s), desc = "computing templates and distances", total = len(chunk_time_ranges_s)):
    
            template_save_folder_chunk = template_save_folder / f"chunk_{j}_merge"
            os.makedirs(template_save_folder_chunk, exist_ok=True)
            # Can parallelize here!! 

            idx_spikes = sortings_all[j].labels>-1
            if miter ==0:
                sorting_chunk = sortings_all[j]
                unit_ids_all = np.unique(sorting.labels)
            else:                
                sorting_chunk = update_sorting_chunk_spikes_loaded(
                    sorting_GC,
                    idx_spikes,
                    n_min_spikes=template_config.spikes_per_unit,
                    latest_time_sample=chunk_time_range[1]
                    * recording.sampling_frequency,
                )
                unit_ids_all = np.unique(sorting_GC.labels)
            unit_ids_all = unit_ids_all[unit_ids_all>-1]
    
            waveforms_loaded = np.load(spike_save_folder / f"spike_wfs_chunk{j}.npy")    
            times_chunk = sortings_all[j].times_samples[idx_spikes]
            times_unique, indices_unique_index, indices_unique_inverse = np.unique(times_chunk, return_inverse=True, return_index=True)

            # print("sorting_chunk")
            # print(sorting_chunk.labels.max())
            # print(sorting_chunk.unit_ids.max())
            
            # compute templates
            # Simplify this since we pass in sorting_chunk
            template_data = TemplateData.from_config_with_spikes_loaded(
                recording,
                waveforms_loaded,
                indices_unique_inverse,
                indices_unique_index,
                sorting_chunk,
                template_config,
                idx_spikes,
                motion_est=motion_est,
                n_jobs=n_jobs_templates,
                save_folder=template_save_folder_chunk, #make model dir chunk
                overwrite=True,
                device=device,
                save_npz_name=template_npz_filename,
                trough_offset_samples=trough_offset_samples,
                spike_length_samples=spike_length_samples,
            )

            units, dists, shifts, template_snrs = calculate_merge_distances(
                template_data,
                unit_ids_all=unit_ids_all,
                superres_linkage=np.max,
                sym_function=np.maximum,
                min_channel_amplitude=min_channel_amplitude, # Propagate these arguments 
                min_spatial_cosine=min_spatial_cosine,
                n_jobs=n_jobs,
                show_progress=False,
            )
            
            template_data_all.append(template_data)
            dists_all.append(dists)
            shifts_all.append(shifts)
            snrs_all.append(template_snrs)

        print("merging")

        dists_min_across_chunks = np.nanmax(dists_all, axis=0)
            
        total_pairs = shifts_all[0].shape[0]*shifts_all[0].shape[1]
        shifts_min_across_chunks = np.array(shifts_all).reshape((n_chunks, total_pairs))[np.array(dists_all).reshape((n_chunks, total_pairs)).argmin(0), np.arange(total_pairs)]
        shifts_min_across_chunks = shifts_min_across_chunks.reshape((shifts_all[0].shape[0],shifts_all[0].shape[1]))
        template_snrs = np.array(snrs_all).max(0)

        if miter==0:
            sorting_merge = recluster(
                sorting, # should be sorting in first iteration
                unit_ids_all, # needs to be tracked
                dists_min_across_chunks,
                shifts_min_across_chunks,
                template_snrs,
                merge_distance_threshold=merge_distance_threshold,
                link=linkage,
            )
        else:
            sorting_merge = recluster(
                sorting_GC, # should be sorting GC in later iterations 
                unit_ids_all, # needs to be tracked
                dists_min_across_chunks,
                shifts_min_across_chunks,
                template_snrs,
                merge_distance_threshold=merge_distance_threshold,
                link=linkage,
            )

        # fig, axes = array_scatter(
        #   sorting_merge.labels, geom, sorting_merge.point_source_localizations[:, 0], motion_est.correct_s(sorting_merge.times_seconds, sorting_merge.point_source_localizations[:, 2]), 
        #   sorting_merge.denoised_ptp_amplitudes, zlim=(-100, 382), do_ellipse=True, xlim=(-50, 92), ptplim = (0, 50),
        # )
        # plt.savefig(fig_directory / f"post_merge_iter_{miter}.png")
        # plt.close()

        # print(f"FIRST FIG MADE ITER {miter}")

        if miter==0:
            # np.save(template_save_folder / f"labels_pre_GC_iter_{miter}.npy", sorting_merge.labels)
            sorting_GC = chuck_noisy_template_units_from_merge(
                sorting,
                sorting_merge,
                template_data_all,
                spike_count_max=template_config.spikes_per_unit,
                min_n_spikes=template_config.min_count_at_shift,
                min_template_snr=template_config.denoising_snr_threshold,
                template_npz_filename="template_data.npz",
            )
        elif miter<num_merge_iteration-1:
            # np.save(template_save_folder / f"labels_pre_GC_iter_{miter}.npy", sorting_merge.labels)
            sorting_GC = chuck_noisy_template_units_from_merge(
                sorting_GC,
                sorting_merge,
                template_data_all,
                spike_count_max=template_config.spikes_per_unit,
                min_n_spikes=template_config.min_count_at_shift,
                min_template_snr=template_config.denoising_snr_threshold,
                template_npz_filename="template_data.npz",
            )
        else:
            # NEED TO RECOMPUTE DATA IN THIS CASE...
            template_data_all = []
    
            for j, chunk_time_range in tqdm(enumerate(chunk_time_ranges_s), desc="making templates for GC", total=len(chunk_time_ranges_s)):
                # Can parallelize here!! 
                idx_spikes = sortings_all[j].labels>-1
                sorting_chunk = update_sorting_chunk_spikes_loaded(
                    sorting_merge,
                    idx_spikes,
                    n_min_spikes=template_config.spikes_per_unit,
                    latest_time_sample=chunk_time_range[1]
                    * recording.sampling_frequency,
                )
                
                waveforms_loaded = np.load(spike_save_folder / f"spike_wfs_chunk{j}.npy")
    
                idx_spikes = sortings_all[j].labels>-1
                times_chunk = sortings_all[j].times_samples[idx_spikes]
                times_unique, indices_unique_index, indices_unique_inverse = np.unique(times_chunk, return_inverse=True, return_index=True)
                
                # compute templates
                template_data = TemplateData.from_config_with_spikes_loaded(
                    recording,
                    waveforms_loaded,
                    indices_unique_inverse,
                    indices_unique_index,
                    sorting_chunk,
                    template_config,
                    idx_spikes,
                    motion_est=motion_est,
                    n_jobs=n_jobs_templates,
                    save_folder=template_save_folder_chunk, #make model dir chunk
                    overwrite=overwrite_templates,
                    device=device,
                    save_npz_name=template_npz_filename,
                    trough_offset_samples=trough_offset_samples,
                    spike_length_samples=spike_length_samples,
                )        
                template_data_all.append(template_data)

            
            np.save(template_save_folder / f"labels_pre_GC_iter_{miter}.npy", sorting_merge.labels)
            sorting_GC, template_data_all = chuck_noisy_template_units_with_loaded_spikes_per_chunk(
                sorting_merge,
                template_data_all,
                template_save_folder=template_save_folder, #TODO
                min_template_snr=50, #propagate these arguments as well
                min_n_spikes=25,
                template_npz_filename=template_npz_filename,
            )

        fig, axes = array_scatter(
          sorting_GC.labels, geom, sorting_GC.point_source_localizations[:, 0], motion_est.correct_s(sorting_GC.times_seconds, sorting_GC.point_source_localizations[:, 2]), 
          sorting_GC.denoised_ptp_amplitudes, zlim=(-100, 382), do_ellipse=True, xlim=(-50, 92), ptplim = (0, 50),
        )
        plt.savefig(fig_directory / f"post_mergeGC_iter_{miter}.png")
        plt.close()


        if miter < num_merge_iteration-1 and iterative_split:
            print("splitting")
            sorting_GC = split_clusters(
                sorting_GC,
                split_strategy="MaxChanPCSplit",
                split_strategy_kwargs=dict(
                    peeling_hdf5_filename=sub_h5,
                    # change this here depending on the dataset rearrange so that all is subtraction_models / OK if not relocated :) 
                    peeling_featurization_pt=sub_h5.parent / "subtraction_models/featurization_pipeline.pt",
                    channel_selection_radius=3,
                    use_localization_features=False,
                    use_ptp=False,
                    n_neighbors_search=25,
                    radius_search=5,
                    sigma_local=1,
                    noise_density=0.25,
                    remove_clusters_smaller_than=25,
                    relocated=False,
                    whitened=False,
                    cluster_alg="dpc",
                ),
                recursive=False,
                n_jobs=0,
                motion_est=None, #doesn't matter here...
            )

        fig, axes = array_scatter(
          sorting_GC.labels, geom, sorting_GC.point_source_localizations[:, 0], motion_est.correct_s(sorting_GC.times_seconds, sorting_GC.point_source_localizations[:, 2]), 
          sorting_GC.denoised_ptp_amplitudes, zlim=(-100, 382), do_ellipse=True, xlim=(-50, 92), ptplim = (0, 50),
        )
        plt.savefig(fig_directory / f"post_split_iter_{miter}.png")
        plt.close()

        # new_labels_ids = np.full(sorting.labels.max()+1, -1)
        # units_orig_sorting = np.unique(sorting.labels)
        # units_orig_sorting = units_orig_sorting[units_orig_sorting>-1]
        # for k in np.unique(units_orig_sorting):
        #     assert len(np.unique(sorting_GC.labels[sorting.labels==k]))==1
        #     new_labels_ids[k] = sorting_GC.labels[sorting.labels==k][0]

    # Save and compute templates (for deconv, since spikes are loaded already)
    if delete_spikes:
        os.system(f"rm -r {spike_save_folder}")
    return sorting_GC


def merge_templates_across_multiple_chunks(
    sorting: DARTsortSorting,
    recording,
    chunk_time_ranges_s,
    template_data_list = None,
    template_config: Optional[TemplateConfig] = None,
    motion_est=None,
    max_shift_samples=20,
    superres_linkage=np.max,
    linkage="single",
    sym_function=np.maximum,
    masked_sym_func=np.fmax,
    aggregate_func=np.nanmax,
    mask_units_too_far=True,
    merge_distance_threshold=0.1,
    temporal_upsampling_factor=8,
    amplitude_scaling_variance=0.001,
    amplitude_scaling_boundary=0.1,
    svd_compression_rank=20,
    min_channel_amplitude=1.5,
    min_spatial_cosine=0.0,
    conv_batch_size=128,
    units_batch_size=8,
    N_spikes_overlap = 100,
    device=None,
    n_jobs=0,
    n_jobs_templates=0,
    template_save_folder=None,
    overwrite_templates=False,
    show_progress=False,
    template_npz_filename="template_data.npz",
    end_name_per_chunk="templates",
    reorder_by_depth=True,
    return_dist_matrix=False,
) -> DARTsortSorting:
    """Template distance based merge

    Pass in a sorting, recording and template config to make templates,
    and this will merge them (with superres). Or, if you have templates
    already, pass them into template_data and we can skip the template
    construction.

    Arguments
    ---------
    max_shift_samples
        Max offset during matching
    superres_linkage
        How to combine distances between two units' superres templates
        By default, it's the max.
    amplitude_scaling_*
        Optionally allow scaling during matching

    Returns
    -------
    A new DARTsortSorting
    """
    print(f"merge input {np.unique(sorting.labels).size - 1} units")
    n_chunks = len(chunk_time_ranges_s)
    if template_data_list is None:
        template_data_list=[]
        for j, chunk_time_range in tqdm(enumerate(chunk_time_ranges_s), total = len(chunk_time_ranges_s), desc="Making template list for merging"):
            model_subdir_chunk = f"chunk_{j}_{end_name_per_chunk}"
            if template_save_folder is not None:
                model_dir_chunk = Path(template_save_folder) / model_subdir_chunk
            else:
                model_dir_chunk = None
            assert template_config is not None
            sorting_chunk = keep_only_most_recent_spikes(
                sorting,
                n_min_spikes=template_config.spikes_per_unit,
                latest_time_sample=chunk_time_range[1]
                * recording.sampling_frequency,
            )
            
            template_data = TemplateData.from_config(
                recording,
                sorting_chunk,
                template_config,
                motion_est=motion_est,
                n_jobs=n_jobs_templates,
                save_folder=model_dir_chunk,
                overwrite=overwrite_templates,
                device=device,
                save_npz_name=template_npz_filename,
                # Here should add trough + spike length
            )
            template_data_list.append(template_data)

    unit_ids_all = np.unique(sorting.labels)
    unit_ids_all = unit_ids_all[unit_ids_all>-1]
    
    dists_all = []
    shifts_all = []
    snrs_all = []

    for template_data in tqdm(template_data_list, desc = "Computing template distances", total = len(template_data_list)):

        units, dists, shifts, template_snrs = calculate_merge_distances(
            template_data,
            unit_ids_all=unit_ids_all,
            superres_linkage=superres_linkage,
            sym_function=sym_function,
            max_shift_samples=max_shift_samples,
            temporal_upsampling_factor=temporal_upsampling_factor,
            amplitude_scaling_variance=amplitude_scaling_variance,
            amplitude_scaling_boundary=amplitude_scaling_boundary,
            svd_compression_rank=svd_compression_rank,
            min_channel_amplitude=min_channel_amplitude,
            min_spatial_cosine=min_spatial_cosine,
            conv_batch_size=conv_batch_size,
            units_batch_size=units_batch_size,
            device=device,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )
        
        dists_all.append(dists)
        shifts_all.append(shifts)
        snrs_all.append(template_snrs)

    dists_all = np.array(dists_all)
    
    if mask_units_too_far:
        temporal_mask = compute_temporal_mask_merge(
            sorting,
            chunk_time_ranges_s,
            dists_all,
            N_spikes_overlap = N_spikes_overlap,
        )
        dists_all = dists_all*temporal_mask
        
    dists_all[dists_all<0] = np.nan
    for k in range(len(dists_all)):
        dists_all[k] = masked_sym_func(dists_all[k], dists_all[k].T)
    dists_across_chunks = aggregate_func(dists_all, axis=0)


    total_pairs = shifts_all[0].shape[0]*shifts_all[0].shape[1]
    shifts_across_chunks = np.array(shifts_all).reshape((n_chunks, total_pairs))[np.array(dists_all).reshape((n_chunks, total_pairs)).argmin(0), np.arange(total_pairs)]
    shifts_across_chunks = shifts_across_chunks.reshape((shifts_all[0].shape[0],shifts_all[0].shape[1]))
    template_snrs = np.array(snrs_all).max(0)
    
    # now run hierarchical clustering
    merged_sorting = recluster(
        sorting,
        unit_ids_all,
        dists_across_chunks,
        shifts_across_chunks,
        template_snrs,
        merge_distance_threshold=merge_distance_threshold,
        link=linkage,
    )

    if reorder_by_depth:
        merged_sorting = cluster_util.reorder_by_depth(
            merged_sorting, motion_est=motion_est
        )

    if return_dist_matrix:
        return merged_sorting, dists_all
    return merged_sorting



def merge_templates(
    sorting: DARTsortSorting,
    recording,
    template_data: Optional[TemplateData] = None,
    template_config: Optional[TemplateConfig] = None,
    motion_est=None,
    max_shift_samples=20,
    superres_linkage=np.max,
    linkage="complete",
    sym_function=np.minimum,
    merge_distance_threshold=0.25,
    temporal_upsampling_factor=8,
    amplitude_scaling_variance=0.001,
    amplitude_scaling_boundary=0.1,
    svd_compression_rank=20,
    min_channel_amplitude=1.0,
    min_spatial_cosine=0.0,
    conv_batch_size=128,
    units_batch_size=8,
    device=None,
    n_jobs=0,
    n_jobs_templates=0,
    template_save_folder=None,
    overwrite_templates=False,
    show_progress=True,
    template_npz_filename="template_data.npz",
    reorder_by_depth=True,
) -> DARTsortSorting:
    """Template distance based merge

    Pass in a sorting, recording and template config to make templates,
    and this will merge them (with superres). Or, if you have templates
    already, pass them into template_data and we can skip the template
    construction.

    Arguments
    ---------
    max_shift_samples
        Max offset during matching
    superres_linkage
        How to combine distances between two units' superres templates
        By default, it's the max.
    amplitude_scaling_*
        Optionally allow scaling during matching

    Returns
    -------
    A new DARTsortSorting
    """
    print("zmerge input", np.unique(sorting.labels).size - 1)
    if template_data is None:
        template_data, sorting = TemplateData.from_config(
            recording,
            sorting,
            template_config,
            motion_est=motion_est,
            n_jobs=n_jobs_templates,
            save_folder=template_save_folder,
            overwrite=overwrite_templates,
            device=device,
            save_npz_name=template_npz_filename,
            return_realigned_sorting=True,
        )

    units, dists, shifts, template_snrs = calculate_merge_distances(
        template_data,
        superres_linkage=superres_linkage,
        sym_function=sym_function,
        max_shift_samples=max_shift_samples,
        temporal_upsampling_factor=temporal_upsampling_factor,
        amplitude_scaling_variance=amplitude_scaling_variance,
        amplitude_scaling_boundary=amplitude_scaling_boundary,
        svd_compression_rank=svd_compression_rank,
        min_channel_amplitude=min_channel_amplitude,
        min_spatial_cosine=min_spatial_cosine,
        conv_batch_size=conv_batch_size,
        units_batch_size=units_batch_size,
        device=device,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )

    # now run hierarchical clustering
    merged_sorting = recluster(
        sorting,
        units,
        dists,
        shifts,
        template_snrs,
        merge_distance_threshold=merge_distance_threshold,
        link=linkage,
    )

    if reorder_by_depth:
        merged_sorting = cluster_util.reorder_by_depth(
            merged_sorting, motion_est=motion_est
        )

    return merged_sorting


def merge_across_sortings(
    sortings,
    recording,
    template_config: Optional[TemplateConfig] = None,
    motion_est=None,
    cross_merge_distance_threshold=0.5,
    within_merge_distance_threshold=0.5,
    superres_linkage=np.max,
    sym_function=np.minimum,
    max_shift_samples=20,
    temporal_upsampling_factor=8,
    amplitude_scaling_variance=0.001,
    amplitude_scaling_boundary=0.1,
    svd_compression_rank=20,
    min_channel_amplitude=0.0,
    min_spatial_cosine=0.0,
    conv_batch_size=128,
    units_batch_size=8,
    device=None,
    n_jobs=0,
    n_jobs_templates=0,
    show_progress=True,
):
    # first, merge within chunks
    if within_merge_distance_threshold:
        sortings = [
            merge_templates(
                sorting,
                recording,
                template_config=template_config,
                motion_est=motion_est,
                max_shift_samples=max_shift_samples,
                superres_linkage=superres_linkage,
                sym_function=sym_function,
                merge_distance_threshold=within_merge_distance_threshold,
                temporal_upsampling_factor=temporal_upsampling_factor,
                amplitude_scaling_variance=amplitude_scaling_variance,
                amplitude_scaling_boundary=amplitude_scaling_boundary,
                min_spatial_cosine=min_spatial_cosine,
                svd_compression_rank=svd_compression_rank,
                min_channel_amplitude=min_channel_amplitude,
                conv_batch_size=conv_batch_size,
                units_batch_size=units_batch_size,
                device=device,
                n_jobs=n_jobs,
                n_jobs_templates=n_jobs_templates,
                show_progress=False,
            )
            for sorting in tqdm(sortings, desc="Merge within chunks")
        ]

    # now, across chunks
    for i in range(len(sortings) - 1):
        template_data_a = TemplateData.from_config(
            recording,
            sortings[i],
            template_config,
            motion_est=motion_est,
            n_jobs=n_jobs_templates,
            device=device,
        )
        template_data_b = TemplateData.from_config(
            recording,
            sortings[i + 1],
            template_config,
            motion_est=motion_est,
            n_jobs=n_jobs_templates,
            device=device,
        )
        dists, shifts, snrs_a, snrs_b, units_a, units_b = cross_match_distance_matrix(
            template_data_a,
            template_data_b,
            superres_linkage=superres_linkage,
            sym_function=sym_function,
            max_shift_samples=max_shift_samples,
            temporal_upsampling_factor=temporal_upsampling_factor,
            amplitude_scaling_variance=amplitude_scaling_variance,
            amplitude_scaling_boundary=amplitude_scaling_boundary,
            min_spatial_cosine=min_spatial_cosine,
            svd_compression_rank=svd_compression_rank,
            min_channel_amplitude=min_channel_amplitude,
            conv_batch_size=conv_batch_size,
            units_batch_size=units_batch_size,
            device=device,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )
        sortings[i], sortings[i + 1] = cross_match(
            sortings[i],
            sortings[i + 1],
            dists,
            shifts,
            snrs_a,
            snrs_b,
            units_a,
            units_b,
            merge_distance_threshold=cross_merge_distance_threshold,
        )

    # combine into one big sorting
    return combine_sortings(sortings)


def calculate_merge_distances(
    template_data,
    superres_linkage=np.max,
    sym_function=np.minimum,
    max_shift_samples=20,
    temporal_upsampling_factor=8,
    amplitude_scaling_variance=0.001,
    amplitude_scaling_boundary=0.1,
    svd_compression_rank=20,
    min_channel_amplitude=1.0,
    min_spatial_cosine=0.0,
    cooccurrence_mask=None,
    conv_batch_size=128,
    units_batch_size=8,
    unit_ids_all=None,
    device=None,
    n_jobs=0,
    show_progress=True,
):
    """
    if passed, unit_ids_all contains more units that template_data.unit_ids, and will not compute distance for these missing units but dist matrix will be of the shape len(unit_ids_all)
    """
    # allocate distance + shift matrices. shifts[i,j] is trough[j]-trough[i].

    if unit_ids_all is None:
        unit_ids_all = template_data.unit_ids
        
    n_templates = len(template_data.unit_ids)
    sup_dists = np.full((n_templates, n_templates), np.nan) #NAN???
    sup_shifts = np.zeros((n_templates, n_templates), dtype=int)

    # apply min channel amplitude to templates directly so that it reflects
    # in the template norms used in the distance computation
    if min_channel_amplitude:
        temps = template_data.templates.copy()
        mask = temps.ptp(axis=1, keepdims=True) > min_channel_amplitude
        temps *= mask.astype(temps.dtype)
        template_data = replace(template_data, templates=temps)

    # build distance matrix
    dec_res_iter = get_deconv_resid_norm_iter(
        template_data,
        max_shift_samples=max_shift_samples,
        temporal_upsampling_factor=temporal_upsampling_factor,
        amplitude_scaling_variance=amplitude_scaling_variance,
        amplitude_scaling_boundary=amplitude_scaling_boundary,
        svd_compression_rank=svd_compression_rank,
        min_channel_amplitude=min_channel_amplitude,
        min_spatial_cosine=min_spatial_cosine,
        cooccurrence_mask=cooccurrence_mask,
        conv_batch_size=conv_batch_size,
        units_batch_size=units_batch_size,
        device=device,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )
    for res in dec_res_iter:
        if res is None:
            # all pairs in chunk were ignored for one reason or another
            continue

        tixa = res.template_indices_a
        tixb = res.template_indices_b
        rms_ratio = res.deconv_resid_norms / res.template_a_norms
        sup_dists[tixa, tixb] = rms_ratio
        sup_shifts[tixa, tixb] = res.shifts

    # apply linkage to reduce across superres templates
    units = np.unique(template_data.unit_ids)
    # Chnage this! if less units than number of templates, have a nan
    if units.size < unit_ids_all.size:
        dists = np.full((unit_ids_all.size, unit_ids_all.size), np.nan)
        shifts = np.zeros((unit_ids_all.size, unit_ids_all.size), dtype=int)
        template_snrs = np.zeros(unit_ids_all.size)
        for ua in units:
            in_ua = np.flatnonzero(template_data.unit_ids == ua)
            for ub in units:
                in_ub = np.flatnonzero(template_data.unit_ids == ub)
                in_pair = (in_ua[:, None], in_ub[None, :])
                # TO CHANGE
                dists[ua, ub] = superres_linkage(sup_dists[in_pair])
                shifts[ua, ub] = np.median(sup_shifts[in_pair])
        coarse_td = template_data.coarsen(with_locs=False)
        template_snrs[units] = coarse_td.templates.ptp(1).max(1) / coarse_td.spike_counts
    else:
        dists = sup_dists
        shifts = sup_shifts
        template_snrs = (
            template_data.templates.ptp(1).max(1) / template_data.spike_counts
        )

    dists = sym_function(dists, dists.T)
    return units, dists, shifts, template_snrs


def cross_match_distance_matrix(
    template_data_a,
    template_data_b,
    superres_linkage=np.max,
    sym_function=np.minimum,
    max_shift_samples=20,
    temporal_upsampling_factor=8,
    amplitude_scaling_variance=0.001,
    amplitude_scaling_boundary=0.1,
    svd_compression_rank=20,
    min_channel_amplitude=0.0,
    min_spatial_cosine=0.0,
    conv_batch_size=128,
    units_batch_size=8,
    device=None,
    n_jobs=0,
    show_progress=False,
):
    template_data, cross_mask, ids_a, ids_b = combine_templates(
        template_data_a, template_data_b
    )
    units, dists, shifts, template_snrs = calculate_merge_distances(
        template_data,
        superres_linkage=superres_linkage,
        sym_function=sym_function,
        max_shift_samples=max_shift_samples,
        temporal_upsampling_factor=temporal_upsampling_factor,
        amplitude_scaling_variance=amplitude_scaling_variance,
        amplitude_scaling_boundary=amplitude_scaling_boundary,
        svd_compression_rank=svd_compression_rank,
        min_channel_amplitude=min_channel_amplitude,
        min_spatial_cosine=min_spatial_cosine,
        cooccurrence_mask=cross_mask,
        conv_batch_size=conv_batch_size,
        units_batch_size=units_batch_size,
        device=device,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )

    # symmetrize the merge distance matrix, which right now is block off-diagonal
    # (with infs on main diag blocks)
    a_mask = np.flatnonzero(np.isin(units, ids_a))
    b_mask = np.flatnonzero(np.isin(units, ids_b))
    Dab = dists[a_mask[:, None], b_mask[None, :]]
    Dba = dists[b_mask[:, None], a_mask[None, :]]
    Dstack = np.stack((Dab, Dba.T))
    print(f"{Dstack.shape=} {Dab.shape=} {Dba.shape=}")
    shifts_ab = shifts[a_mask[:, None], b_mask[None, :]]
    shifts_ba = shifts[b_mask[:, None], a_mask[None, :]]
    shifts_stack = np.stack((shifts_ab, -shifts_ba.T))

    # choices[ia, ib] = 0 if Dab[ia, ib] < Dba[ib, ia], else 1
    # but what does that mean? if 0, it means that the upsampled model
    # of B was better able to match A than the other way around. it's not
    # that important, at the end of the day. still, we keep track of it
    # so that we can pick the shift and move on with our lives.
    choices = np.argmin(Dstack, axis=0)
    print(f"{choices.shape=}")
    dists = Dstack[
        choices, np.arange(choices.shape[0])[:, None], np.arange(choices.shape[1])[None]
    ]
    shifts = shifts_stack[
        choices, np.arange(choices.shape[0])[:, None], np.arange(choices.shape[1])[None]
    ]

    snrs_a = template_snrs[a_mask]
    snrs_b = template_snrs[b_mask]

    return (
        dists,
        shifts,
        snrs_a,
        snrs_b,
    )


def recluster(
    sorting,
    units,
    dists,
    shifts,
    template_snrs,
    merge_distance_threshold=0.25,
    link="complete",
):
    # upper triangle not including diagonal, aka condensed distance matrix in scipy
    pdist = dists[np.triu_indices(dists.shape[0], k=1)]
    # scipy hierarchical clustering only supports finite values, so let's just
    # drop in a huge value here
    finite = np.isfinite(pdist)
    if not finite.any():
        print("no merges")
        return sorting

    pdist[~finite] = 1_000_000 + pdist[finite].max()
    # complete linkage: max dist between all pairs across clusters.
    Z = linkage(pdist, method=link)
    # extract flat clustering using our max dist threshold
    new_labels = fcluster(Z, merge_distance_threshold, criterion="distance")

    # update labels
    labels_updated = np.full(sorting.labels.shape, -1)
    kept = np.flatnonzero(np.isin(sorting.labels, np.unique(units)))
    labels_updated[kept] = sorting.labels[kept].copy()
    # FIX THIS!! DOESN"T WORK IF NOT CONTIGUOUS TO BEGIN WITH....
    flat_labels = labels_updated[kept]
    # _, flat_labels = np.unique(labels_updated[kept], return_inverse=True)
    labels_updated[kept] = new_labels[flat_labels]
    labels_updated[labels_updated>-1] -= labels_updated[labels_updated>-1].min()

    # update times according to shifts
    times_updated = sorting.times_samples.copy()

    # find original labels in each cluster
    clust_inverse = {i: [] for i in new_labels}
    for orig_label, new_label in enumerate(new_labels):
        clust_inverse[new_label].append(orig_label)
    print(sum(len(v) - 1 for v in clust_inverse.values()), "merges")

    # align to best snr unit
    for new_label, orig_labels in clust_inverse.items():
        # we don't need to realign clusters which didn't change
        if len(orig_labels) <= 1:
            continue

        orig_snrs = template_snrs[orig_labels]
        best_orig = orig_labels[orig_snrs.argmax()]
        for ogl in np.setdiff1d(orig_labels, [best_orig]):
            in_orig_unit = np.flatnonzero(sorting.labels == ogl)
            # this is like trough[best] - trough[ogl]
            shift_og_best = shifts[ogl, best_orig]
            # if >0, trough of og is behind trough of best.
            # subtracting will move trough of og to the right.
            times_updated[in_orig_unit] -= shift_og_best

    return replace(sorting, times_samples=times_updated, labels=labels_updated)


def cross_match(
    sorting_a,
    sorting_b,
    dists,
    shifts,
    snrs_a,
    snrs_b,
    units_a,
    units_b,
    merge_distance_threshold=0.5,
):
    # assert np.array_equal(units_a, np.arange(units_a.size))
    # assert np.array_equal(units_b, np.arange(units_b.size))
    # print(f"{np.unique(sorting_b.labels)=}")

    ia, ib = np.nonzero(dists <= merge_distance_threshold)
    weights = coo_array(
        (-dists[ia, ib], (ia.astype(np.intc), ib.astype(np.intc))),
        shape=dists.shape,
    )
    b_to_a = maximum_bipartite_matching(weights)
    assert b_to_a.shape == units_b.shape

    # -- update sortings
    # sorting A's labels don't change
    # matched B units are given their A-match's label, and unmatched units get labels
    # starting from A's next cluster label
    matched = b_to_a >= 0
    print(f"{matched.sum()=}")
    next_a_label = units_a.max() + 1
    print(f"{next_a_label=}")
    b_reindex = np.full_like(units_b, -1)
    matched_a_units = units_a[b_to_a[matched]]
    b_reindex[matched] = matched_a_units
    b_reindex[~matched] = next_a_label + np.arange(np.count_nonzero(~matched))
    b_kept = np.flatnonzero(sorting_b.labels >= 0)
    b_labels = np.full_like(sorting_b.labels, -1)
    b_labels[b_kept] = b_reindex[sorting_b.labels[b_kept]]

    # both sortings' times can change. we shift the lower SNR unit.
    # shifts is like trough[a] - trough[b]. if >0, subtract from a or add to b to realign.
    #     matched_b_units = units_b[matched]
    #     shifts = shifts[matched_a_units, matched_b_units]

    #     shifts_a = np.zeros_like(sorting_a.times_samples)
    #     a_matched = np.flatnonzero(np.isin(sorting_a.labels, matched_a_units))
    #     a_match_ix = np.searchsorted(matched_a_units, sorting_a.labels[a_matched])
    #     shifts_a[a_matched] = shifts[a_match_ix]
    #     times_a = sorting_a.times_samples - shifts_a

    #     shifts_b = np.zeros_like(sorting_b.times_samples)
    #     b_matched = np.flatnonzero(np.isin(sorting_b.labels, matched_b_units))
    #     b_match_ix = np.searchsorted(matched_b_units, sorting_b.labels[b_matched])
    #     shifts_b[b_matched] = shifts[b_match_ix]
    #     times_b = sorting_b.times_samples - shifts_b

    # sorting_a = replace(sorting_a)#, times_samples=times_a)
    sorting_b = replace(sorting_b, labels=b_labels)  # , times_samples=times_b)
    return sorting_a, sorting_b


def get_deconv_resid_norm_iter(
    template_data,
    max_shift_samples=20,
    temporal_upsampling_factor=8,
    amplitude_scaling_variance=0.001,
    amplitude_scaling_boundary=0.1,
    svd_compression_rank=20,
    min_channel_amplitude=0.0,
    min_spatial_cosine=0.0,
    cooccurrence_mask=None,
    conv_batch_size=128,
    units_batch_size=8,
    device=None,
    n_jobs=0,
    show_progress=True,
):
    # get template aux data
    low_rank_templates = template_util.svd_compress_templates(
        template_data.templates,
        min_channel_amplitude=min_channel_amplitude,
        rank=svd_compression_rank,
    )
    compressed_upsampled_temporal = template_util.compressed_upsampled_templates(
        low_rank_templates.temporal_components,
        ptps=template_data.templates.ptp(1).max(1),
        max_upsample=temporal_upsampling_factor,
    )

    # construct helper data and run pairwise convolutions
    (
        template_shift_index_a,
        template_shift_index_b,
        upsampled_shifted_template_index,
        cooccurrence,
    ) = construct_shift_indices(
        None,
        None,
        template_data,
        compressed_upsampled_temporal,
        motion_est=None,
    )
    if cooccurrence_mask is not None:
        cooccurrence = cooccurrence & cooccurrence_mask
    yield from iterate_compressed_pairwise_convolutions(
        template_data,
        low_rank_templates,
        template_data,
        low_rank_templates,
        compressed_upsampled_temporal,
        template_shift_index_a,
        template_shift_index_b,
        cooccurrence,
        upsampled_shifted_template_index,
        do_shifting=False,
        reduce_deconv_resid_norm=True,
        geom=template_data.registered_geom,
        conv_ignore_threshold=0.0,
        min_spatial_cosine=min_spatial_cosine,
        coarse_approx_error_threshold=0.0,
        amplitude_scaling_variance=amplitude_scaling_variance,
        amplitude_scaling_boundary=amplitude_scaling_boundary,
        max_shift=max_shift_samples,
        conv_batch_size=conv_batch_size,
        units_batch_size=units_batch_size,
        device=device,
        n_jobs=n_jobs,
        show_progress=show_progress,
    )


def combine_templates(template_data_a, template_data_b):
    rgeom = template_data_a.registered_geom
    if rgeom is not None:
        assert np.array_equal(rgeom, template_data_b.registered_geom)
    locs = template_data_a.registered_template_depths_um
    if locs is not None:
        locs = np.concatenate((locs, template_data_b.registered_template_depths_um))

    ids_a = template_data_a.unit_ids
    ids_b = template_data_b.unit_ids + ids_a.max() + 1
    unit_ids = np.concatenate((ids_a, ids_b))
    templates = np.row_stack((template_data_a.templates, template_data_b.templates))
    spike_counts = np.concatenate(
        (template_data_a.spike_counts, template_data_b.spike_counts)
    )
    template_data = TemplateData(
        templates=templates,
        unit_ids=unit_ids,
        spike_counts=spike_counts,
        registered_geom=rgeom,
        registered_template_depths_um=locs,
    )

    cross_mask = np.zeros((unit_ids.size, unit_ids.size), dtype=bool)
    cross_mask[ids_a.size :, : ids_b.size] = True
    cross_mask[: ids_a.size, ids_b.size :] = True

    return template_data, cross_mask, ids_a, ids_b

def compute_temporal_mask_merge(
    sorting,
    chunk_time_ranges_s,
    dists_all,
    N_spikes_overlap = 100,
):
    n_chunks, n_units = dists_all.shape[0], dists_all.shape[1]
    temporal_mask = -1*np.ones((n_chunks, n_units, n_units))
    for j, chunk_time_range in tqdm(enumerate(chunk_time_ranges_s), total = len(chunk_time_ranges_s)):
        idx_chunk = np.flatnonzero(np.logical_and(
                    sorting.times_seconds>=chunk_time_range[0],
                    sorting.times_seconds<chunk_time_range[1]
                ))
        if j<len(chunk_time_ranges_s)-1:
            idx_next_chunk = np.flatnonzero(np.logical_and(
                        sorting.times_seconds>=chunk_time_range[0],
                        sorting.times_seconds<chunk_time_ranges_s[j+1][1]
                    ))
        for unit_a in range(n_units):
            unit_a_inchunk = (sorting.labels[idx_chunk] == unit_a).sum()>=N_spikes_overlap
            if not unit_a_inchunk:
                temporal_mask[j, unit_a, unit_a]=1
            else:
                for unit_b in range(n_units):
                    if unit_b == unit_a:
                        temporal_mask[j, unit_b, unit_a]=1
                    else:
                        unit_b_inchunk = (sorting.labels[idx_chunk] == unit_b).sum()>=N_spikes_overlap
                        if unit_b_inchunk:
                            temporal_mask[j, unit_a, unit_b]=1
                        elif j<len(chunk_time_ranges_s)-1:
                            unit_b_inchunk = (sorting.labels[idx_next_chunk] == unit_b).sum()>=N_spikes_overlap
                            if unit_a_inchunk and unit_b_inchunk:
                                temporal_mask[j, unit_a, unit_b]=1
    return temporal_mask



