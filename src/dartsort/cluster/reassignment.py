"""
Different functions for reassigning spikes, given a spike train / templates or others
"""

from tqdm.auto import tqdm
import numpy as np
from dartsort.util.data_util import chunk_time_ranges, subchunks_time_ranges
from dartsort.util.drift_util import registered_geometry, get_spike_pitch_shifts, get_waveforms_on_static_channels
from dartsort.util.waveform_util import channel_subset_by_radius
import h5py
from dartsort.templates.templates import TemplateData
from dartsort.util.list_util import create_tpca_templates_list, create_tpca_templates_list_efficient
from dartsort.templates.template_util import smooth_list_templates
from dartsort.cluster.merge import merge_templates_across_multiple_chunks
from dataclasses import replace

def compute_residual_norm(
    recording, 
    sorting, #Here doesn't have to be split
    motion_est,
    chunk_time_ranges_s,
    template_config,
    matching_config,
    tpca,
    matchh5,
    neighbors,
    temp_data_smoothed,
    chunk_belong_wfs,
    wfs_name="collisioncleaned_tpca_features",
    fill_nanvalue=1_000_000,
    norm_operator=np.nanmax,
    return_num_channels=False,
    start_wf=25, 
    end_wf=70,
):
    """
    Only works in tpca space - original wf space not implemented yet
    """
    # Can speed this up!! only look at first sorting (before split)
    geom = recording.get_channel_locations()
    registered_geom = registered_geometry(geom, motion_est)
    labels = sorting.labels


    residual_Linf_norm = fill_nanvalue*np.ones((labels.shape[0], neighbors.shape[1]))
    if return_num_channels:
        num_overlapping_chans = np.zeros((labels.shape[0], 3))
        num_chan_waveforms_nonan = np.zeros(labels.shape[0])

    n_pitches_shift = get_spike_pitch_shifts(
        sorting.point_source_localizations[:, 2],
        geom=geom,
        times_s=sorting.times_seconds,
        motion_est=motion_est)

    with h5py.File(matchh5, "r+") as h5:
        colcleanedwfs = h5[wfs_name]
        channel_index = h5["channel_index"][:]
        for sli, *_ in tqdm(colcleanedwfs.iter_chunks()):
            indices_chunk = np.arange(sli.start, sli.stop)
            wfs = colcleanedwfs[sli]
            labels_chunk = labels[sli]
            
            wfs = get_waveforms_on_static_channels(
                wfs,
                geom,
                sorting.channels[sli], 
                channel_index, 
                registered_geom=registered_geom,
                n_pitches_shift=n_pitches_shift[sli],
            )

            if return_num_channels:
                num_chan_waveforms_nonan[indices_chunk] = (~np.isnan(wfs[:, 0])).sum(1)

            for unit in np.unique(labels_chunk):
                unit_neighbors = neighbors[unit]
                idx_unit = np.flatnonzero(labels_chunk == unit)
                idx_unit_all_chunk = indices_chunk[idx_unit]
                wfs_unit = wfs[idx_unit]
                                 
                for i, neigh in enumerate(unit_neighbors):
                    if neigh>-1:
                        temp_neigh = temp_data_smoothed[chunk_belong_wfs[idx_unit_all_chunk], neigh]
                        residual_Linf_norm[idx_unit_all_chunk, i] = norm_operator(np.abs(wfs_unit - temp_neigh), axis = (1, 2))
                        if return_num_channels:
                            num_overlapping_chans[idx_unit_all_chunk, i] = (~np.isnan((wfs_unit - temp_neigh)[:, 0])).sum(1)

    residual_Linf_norm[np.isnan(residual_Linf_norm)] = fill_nanvalue

    if not return_num_channels:
        return residual_Linf_norm
    else:
        return residual_Linf_norm, num_overlapping_chans, num_chan_waveforms_nonan

def compute_maxchan_tpca_residual(
    recording, 
    sorting,
    motion_est,
    chunk_time_ranges_s,
    template_config,
    matching_config,
    tpca,
    matchh5,
    temp_data_smoothed,
    chunk_belong_wfs,
    wfs_name="collisioncleaned_tpca_features",
    trough_offset=42,
    peak_time_selection="maxstd",
    spike_length_samples=121,
    tpca_rank=8,
):

    # Compute max channels here
    (n_chunks, n_units, _, n_chans) = temp_data_smoothed.shape
    max_channels = np.full((n_chunks, n_units), np.nan)
    time_nan, unit_nan, channel_nan = np.where(np.isnan(temp_data_smoothed[:, :, 0]))
    temp_data_smoothed[np.isnan(temp_data_smoothed)]=0
    max_channels = temp_data_smoothed.ptp(2).argmax(2)
    temp_data_smoothed[time_nan, unit_nan, :, channel_nan]=0

    for k in range(n_chunks):
        temp_data_smoothed_reconstructed = tpca.inverse_transform(temp_data_smoothed[k].transpose(0, 2, 1).reshape(-1, tpca_rank)).reshape(-1, n_chans, spike_length_samples)
        max_channels[k] = temp_data_smoothed_reconstructed.ptp(2).argmax(1)

    max_chan_residual = np.zeros((sorting.labels.shape[0], tpca_rank))
        
    geom = recording.get_channel_locations()
    registered_geom = registered_geometry(geom, motion_est)
    n_channel_templates = registered_geom.shape[0]

    n_pitches_shift = get_spike_pitch_shifts(
        sorting.point_source_localizations[:, 2],
        geom=geom,
        times_s=sorting.times_seconds,
        motion_est=motion_est)

    with h5py.File(matchh5, "r+") as h5:
        channel_index = h5["channel_index"][:]
        colcleanedwfs = h5[wfs_name]
        for sli, *_ in tqdm(colcleanedwfs.iter_chunks()):
            indices_chunk = np.arange(sli.start, sli.stop)
        
            wfs = get_waveforms_on_static_channels(
                colcleanedwfs[sli],
                geom,
                sorting.channels[sli], 
                channel_index, 
                registered_geom=registered_geom,
                n_pitches_shift=n_pitches_shift[sli],
            )

            temp_mc = temp_data_smoothed[chunk_belong_wfs[indices_chunk], sorting.labels[sli]]
            mc = max_channels[chunk_belong_wfs[indices_chunk], sorting.labels[sli]]
            max_chan_residual[indices_chunk] = wfs[np.arange(len(mc)), :, mc] - temp_mc[np.arange(len(mc)), :, mc]
    
    for unit in np.unique(sorting.labels):
        idx_unit = np.flatnonzero(sorting.labels==unit)
        no_nan = ~np.isnan(max_chan_residual[idx_unit, 0])
        if peak_time_selection=="maxstd":
            trough_offset = max_chan_residual[idx_unit][no_nan].std(0).argmax()
        cmp = 0
        for j, chunk_time_range in enumerate(chunk_time_ranges_s):
            sub_chunk_time_range_s = subchunks_time_ranges(recording, chunk_time_range, template_config.subchunk_size_s,
                                                  divider_samples=matching_config.chunk_length_samples)
            n_sub_chunks = len(sub_chunk_time_range_s)
            for k, subchunk_time_range in enumerate(sub_chunk_time_range_s):
                idx_chunk = np.flatnonzero(
                    np.logical_and(
                        sorting.times_seconds[idx_unit]>=subchunk_time_range[0],
                        sorting.times_seconds[idx_unit]<subchunk_time_range[1]
                    )
                )
                temp_mc = temp_data_smoothed[cmp, unit]
                mc = max_channels[cmp, unit]
                # mc = temp_mc.ptp(0).argmax()
                max_chan_residual[idx_unit[idx_chunk]]/=np.abs(temp_mc[trough_offset, mc])
                cmp+=1

    return max_chan_residual

def split_maxchan_resid(
    sorting,
    max_chan_residual,
    peak_time_selection="maxstd",
    trough_offset=42,
    max_value_no_split=0.25,
    min_value_split=0.75,
    min_nspikes_unit=150,
    return_neighbors=True,
):
    """
    peak_time_selection can be either trough or maxstd -> This will choose what point to look at in the residuals
    """
    assert peak_time_selection in ["trough", "maxstd"]
    assert min_value_split > max_value_no_split
    
    labels_split = sorting.labels.copy()
    
    cmp=sorting.labels.max()+1
    for unit in np.unique(sorting.labels):
        idx_unit = np.flatnonzero(sorting.labels==unit)
        no_nan = ~np.isnan(max_chan_residual[idx_unit, 0])
        
        if peak_time_selection=="trough":
            values = max_chan_residual[idx_unit, trough_offset]
        elif peak_time_selection=="maxstd":
            maxstd_timepoint = max_chan_residual[idx_unit][no_nan].std(0).argmax()
            values = max_chan_residual[idx_unit, maxstd_timepoint]
        vec_mid = np.abs(values)<max_value_no_split
        vec_low = values<-min_value_split
        vec_high = values>min_value_split

        if vec_mid.sum()>min_nspikes_unit:
            labels_split[idx_unit[~vec_mid]]=-1
        if vec_low.sum()>min_nspikes_unit:
            labels_split[idx_unit[vec_low]]=cmp
            cmp+=1
        if vec_high.sum()>min_nspikes_unit:
            labels_split[idx_unit[vec_high]]=cmp
            cmp+=1
    
    if not return_neighbors:
        return labels_split
    else:
        neighbors = -1*np.ones((sorting.labels.max()+1, 3))
        neighbors[:, 0] = np.arange(neighbors.shape[0])
        for unit in range(neighbors.shape[0]):
            orig_unit = np.unique(sorting.labels[labels_split == unit])
            if len(orig_unit):
                new_units = np.unique(labels_split[sorting.labels == orig_unit])
                new_units = new_units[new_units>-1]
                new_units = new_units[new_units!=unit]        
                if len(new_units)==2:
                    neighbors[unit, 1:] = new_units
                elif len(new_units)==1:
                    neighbors[unit, 1] = new_units
        return labels_split, neighbors.astype('int')    

def iterative_merge_reassignment(
    sorting,
    recording,
    motion_est,
    chunk_time_ranges_s,
    template_config,
    matching_config,
    matchh5,
    tpca,
    split_merge_config,
    template_save_folder=None,
    template_npz_filename="template_data.npz",
    fill_nanvalue=10_000,
    wfs_name="collisioncleaned_tpca_features",
    spike_length_samples=121,
    trough_offset=42,
):

    for iter in range(split_merge_config.m_iter):

        sorting, neighbors = merge_templates_across_multiple_chunks(
            sorting,
            recording,
            chunk_time_ranges_s,
            template_save_folder=template_save_folder,
            template_npz_filename=template_npz_filename,
            motion_est=motion_est,
            template_config=template_config,
            link=split_merge_config.link,
            superres_linkage=split_merge_config.superres_linkage,
            # sym_function=split_merge_config.sym_function,
            min_channel_amplitude=split_merge_config.min_channel_amplitude, 
            min_spatial_cosine=split_merge_config.min_spatial_cosine,
            max_shift_samples=split_merge_config.max_shift_samples,
            merge_distance_threshold=split_merge_config.merge_distance_threshold, #0.25 
            min_ratio_chan_no_nan=split_merge_config.min_ratio_chan_no_nan, #0.25 
            temporal_upsampling_factor=split_merge_config.temporal_upsampling_factor,
            amplitude_scaling_variance=split_merge_config.amplitude_scaling_variance,
            amplitude_scaling_boundary=split_merge_config.amplitude_scaling_boundary,
            svd_compression_rank=split_merge_config.svd_compression_rank,
            conv_batch_size=split_merge_config.conv_batch_size,
            units_batch_size=split_merge_config.units_batch_size, 
            mask_units_too_far=split_merge_config.mask_units_too_far, #False for now
            aggregate_func=split_merge_config.aggregate_func,
            overwrite_templates=True,
            return_neighbors=True,
        )

        #reassign and triage ---

        # The output of these two functions is now in the pc space 
        tpca_templates_list, spike_count_list, chunk_belong = create_tpca_templates_list_efficient(
            recording, 
            sorting,
            motion_est,
            chunk_time_ranges_s,
            template_config, 
            matching_config,
            matchh5,
            weights=None,
            tpca=tpca,
        )
    
        unit_ids = np.unique(sorting.labels)
        unit_ids = unit_ids[unit_ids>-1]
        
        templates_smoothed = smooth_list_templates(
            tpca_templates_list, spike_count_list, unit_ids, threshold_n_spike=split_merge_config.threshold_n_spike,
        )
    
        residual_norm = compute_residual_norm(
            recording, 
            sorting,
            motion_est,
            chunk_time_ranges_s,
            template_config,
            matching_config,
            tpca,
            matchh5,
            neighbors,
            templates_smoothed,
            chunk_belong,
            wfs_name=wfs_name,
            fill_nanvalue=fill_nanvalue,
            norm_operator=split_merge_config.norm_operator,
        )
    
        new_labels = -1*np.ones(sorting.labels.shape)
        for unit in unit_ids:
            idx_unit = np.flatnonzero(sorting.labels == unit)
            new_labels[idx_unit] = neighbors[unit, residual_norm.argmin(1)[idx_unit]]
        new_labels[residual_norm.min(1)>split_merge_config.norm_triage] = -1
        new_labels = new_labels.astype('int')

        sorting = replace(sorting, labels = new_labels)
        
    return sorting

def iterative_split_merge_reassignment(
    sorting,
    recording,
    motion_est,
    chunk_time_ranges_s,
    template_config,
    matching_config,
    matchh5,
    tpca,
    deconv_scores,
    split_merge_config,
    template_save_folder=None,
    template_npz_filename="template_data.npz",
    fill_nanvalue=10_000,
    wfs_name="collisioncleaned_tpca_features",
    spike_length_samples=121,
    trough_offset=42,
    threshold_n_spike=0.2,
    norm_operator=np.nanmax,
    peak_time_selection="maxstd",
    max_value_no_split=0.25,
    min_value_split=0.75,
    min_nspikes_unit=150,
    triage_spikes_2way=0.55,
    triage_spikes_3way=0.5,
    norm_triage=4.0,
):

    """
    Ths function alternates between a split - reassignment - merge and ends with a reassignment 
    TODO: 
     - implement split / reassignment in tpca space 
     - Propagate all these arguments
    - do we want to merge using all labels or only those who are hard assigned?
    """
    for iter in range(split_merge_config.m_iter):

        if iter > 0:
            deconv_scores = np.ones(deconv_scores.shape)

        # Do we keep spikes that are only well assigned here? 
        new_labels, labels_hardassignments_only = full_reassignment_split(
            sorting,
            recording,
            motion_est,
            chunk_time_ranges_s,
            template_config,
            matching_config,
            matchh5,
            tpca,
            deconv_scores,
            return_triaged_labels=True,
            threshold_n_spike=threshold_n_spike,
            fill_nanvalue=fill_nanvalue,
            norm_operator=norm_operator,
            wfs_name=wfs_name,
            spike_length_samples=spike_length_samples,
            peak_time_selection=peak_time_selection,
            trough_offset=trough_offset,
            max_value_no_split=max_value_no_split,
            min_value_split=min_value_split,
            min_nspikes_unit=min_nspikes_unit,
            triage_spikes_2way=triage_spikes_2way,
            triage_spikes_3way=triage_spikes_3way,
            norm_triage=norm_triage,
        )

        sorting = replace(sorting, labels = new_labels) # Or labels hard assigned only?

        sorting, neighbors = merge_templates_across_multiple_chunks(
            sorting,
            recording,
            chunk_time_ranges_s,
            template_save_folder=template_save_folder,
            template_npz_filename=template_npz_filename,
            motion_est=motion_est,
            template_config=template_config,
            link=split_merge_config.link,
            superres_linkage=split_merge_config.superres_linkage,
            # sym_function=split_merge_config.sym_function,
            min_channel_amplitude=split_merge_config.min_channel_amplitude, 
            min_spatial_cosine=split_merge_config.min_spatial_cosine,
            max_shift_samples=split_merge_config.max_shift_samples,
            merge_distance_threshold=split_merge_config.merge_distance_threshold, #0.25 
            min_ratio_chan_no_nan=split_merge_config.min_ratio_chan_no_nan, #0.25 
            temporal_upsampling_factor=split_merge_config.temporal_upsampling_factor,
            amplitude_scaling_variance=split_merge_config.amplitude_scaling_variance,
            amplitude_scaling_boundary=split_merge_config.amplitude_scaling_boundary,
            svd_compression_rank=split_merge_config.svd_compression_rank,
            conv_batch_size=split_merge_config.conv_batch_size,
            units_batch_size=split_merge_config.units_batch_size, 
            mask_units_too_far=split_merge_config.mask_units_too_far, #False for now
            aggregate_func=split_merge_config.aggregate_func,
            overwrite_templates=True,
            return_neighbors=True,
        )

        #reassign and triage ---

        # The output of these two functions is now in the pc space 
        tpca_templates_list, spike_count_list, chunk_belong = create_tpca_templates_list_efficient(
            recording, 
            sorting,
            motion_est,
            chunk_time_ranges_s,
            template_config, 
            matching_config,
            matchh5,
            weights=None,
            tpca=tpca,
        )
    

        unit_ids = np.unique(sorting.labels)
        unit_ids = unit_ids[unit_ids>-1]
        
        templates_smoothed = smooth_list_templates(
            tpca_templates_list, spike_count_list, unit_ids, threshold_n_spike=split_merge_config.threshold_n_spike,
        )
    
        residual_norm = compute_residual_norm(
            recording, 
            sorting,
            motion_est,
            chunk_time_ranges_s,
            template_config,
            matching_config,
            tpca,
            matchh5,
            neighbors,
            templates_smoothed,
            chunk_belong,
            wfs_name=wfs_name,
            fill_nanvalue=fill_nanvalue,
            norm_operator=split_merge_config.norm_operator,
        )
    
        new_labels = -1*np.ones(sorting.labels.shape)
        for unit in unit_ids:
            idx_unit = np.flatnonzero(sorting.labels == unit)
            new_labels[idx_unit] = neighbors[unit, residual_norm.argmin(1)[idx_unit]]
        new_labels[residual_norm.min(1)>split_merge_config.norm_triage] = -1
        new_labels = new_labels.astype('int')

        # sorting = replace(sorting, labels = new_labels)
        
    return sorting

def full_reassignment_split(
    sorting,
    recording,
    motion_est,
    chunk_time_ranges_s,
    template_config,
    matching_config,
    matchh5,
    tpca,
    deconv_scores,
    return_triaged_labels=True,
    threshold_n_spike=0.2,
    fill_nanvalue=1_000_000,
    norm_operator=np.nanmax,
    wfs_name="collisioncleaned_tpca_features",
    spike_length_samples=121,
    peak_time_selection="maxstd",
    trough_offset=42,
    max_value_no_split=0.25,
    min_value_split=0.75,
    min_nspikes_unit=150,
    triage_spikes_2way=0.55,
    triage_spikes_3way=0.5,
    norm_triage=4.0,
):

    weights_deconv = np.log(1 + np.abs(deconv_scores-deconv_scores.min()))
    
    # The output of these two functions is now in the pc space 
    tpca_templates_list, spike_count_list, chunk_belong_wfs = create_tpca_templates_list_efficient(
        recording, 
        sorting,
        motion_est,
        chunk_time_ranges_s,
        template_config, 
        matching_config,
        matchh5,
        weights=weights_deconv,
        tpca=tpca,
    )

    templates_smoothed = smooth_list_templates(
        tpca_templates_list, spike_count_list, np.unique(sorting.labels), threshold_n_spike=threshold_n_spike,
    )

    max_chan_residual = compute_maxchan_tpca_residual(
        recording, 
        sorting,
        motion_est,
        chunk_time_ranges_s,
        template_config,
        matching_config,
        tpca,
        matchh5,
        templates_smoothed,
        chunk_belong_wfs,
        wfs_name=wfs_name,
        spike_length_samples=spike_length_samples,
    )

    labels_split, neighbors = split_maxchan_resid(
        sorting,
        max_chan_residual,
        max_value_no_split=max_value_no_split,
        min_value_split=min_value_split,
        min_nspikes_unit=min_nspikes_unit,
        return_neighbors=True,
    )
    
    sorting_split = replace(sorting, labels=labels_split)

    tpca_templates_list_split, spike_count_list_split, chunk_belong = create_tpca_templates_list_efficient(
        recording, 
        sorting_split,
        motion_est,
        chunk_time_ranges_s,
        template_config, 
        matching_config,
        matchh5,
        weights=None,
        tpca=tpca,
    )

    templates_smoothed_split = smooth_list_templates(
        tpca_templates_list_split, spike_count_list_split, np.unique(sorting_split.labels), threshold_n_spike=threshold_n_spike,
    )

    residual_norm = compute_residual_norm(
        recording, 
        sorting,
        motion_est,
        chunk_time_ranges_s,
        template_config,
        matching_config,
        tpca,
        matchh5,
        neighbors,
        templates_smoothed_split,
        chunk_belong,
        wfs_name=wfs_name,
        fill_nanvalue=fill_nanvalue,
        norm_operator=norm_operator,
    )

    units = np.unique(sorting.labels)
    units = units[units>-1]
    new_labels = -1*np.ones(sorting.labels.shape)
    for unit in units:
        idx_unit = np.flatnonzero(sorting.labels == unit)
        new_labels[idx_unit] = neighbors[sorting.labels[idx_unit], residual_norm.argmin(1)[idx_unit]]
    new_labels[residual_norm.min(1)>norm_triage] = -1
    new_labels = new_labels.astype('int')

    _, new_labels[new_labels>-1] = np.unique(new_labels[new_labels>-1], return_inverse=True)

    if not return_triaged_labels:
        return new_labels

    else:
        labels_hardassignments_only = new_labels.copy()
        for unit in np.unique(sorting.labels):
            idx_unit = np.flatnonzero(sorting.labels == unit)    
            # HERE SHOULD BE LOOKING AT NEIGHBORS INSTEAD OF NEW LABELS FOR INDEXING
            if len(np.unique(new_labels[idx_unit]))==1:
                pass
            elif len(np.unique(new_labels[idx_unit]))==2:
                array_weighted = residual_norm[idx_unit, :2] / residual_norm[idx_unit, :2].sum(1)[:, None]
                idx_bad = array_weighted.max(1)<triage_spikes_2way
                labels_hardassignments_only[idx_unit[idx_bad]]=-1
            else:
                array_weighted = residual_norm[idx_unit] / residual_norm[idx_unit].sum(1)[:, None]
                idx_bad = array_weighted.max(1)<triage_spikes_3way
                labels_hardassignments_only[idx_unit[idx_bad]]=-1
        return new_labels, labels_hardassignments_only

def square_mean(x, axis=1):
    return np.nanmean(np.square(x), axis=axis)
