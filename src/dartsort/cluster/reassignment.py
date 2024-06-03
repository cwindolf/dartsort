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
from dartsort.util.list_util import create_tpca_templates_list
from dartsort.templates.template_util import smooth_list_templates
from dataclasses import replace

def triage(
    sorting,
    recording,
    motion_est,
    chunk_time_ranges_s,
    template_config,
    matching_config,
    data_dir_chunks,
    tpca,
    deconv_scores,
    return_triaged_labels=True,
    threshold_n_spike=0.2,
    fill_nanvalue=1_000,
    norm_operator=np.nanmax,
    min_nspikes_unit=150,
    min_overlapping_chans_ratio=0.75,
    radius=25,
    min_norm_triage=4,
    n_iter=2,
):

    weights_deconv = np.log(1 + np.abs(deconv_scores-deconv_scores.min()))
    tpca_templates_list, spike_count_list, unit_ids_list = create_tpca_templates_list(
        recording, 
        sorting,
        motion_est,
        chunk_time_ranges_s,
        template_config, 
        matching_config,
        data_dir_chunks,
        weights=weights_deconv,
        tpca=tpca,
    )

    templates_smoothed = smooth_list_templates(
        tpca_templates_list, spike_count_list, unit_ids_list, np.unique(sorting.labels), threshold_n_spike=threshold_n_spike,
    )

    neighbors = -1*np.ones((all_units.max()+1, 2))
    neighbors[:, 0] = np.arange(all_units.max()+1)

    residual_norm, n_overlapping_chans = compute_residual_norm_moving_temps(
        recording, 
        sorting.labels,
        motion_est,
        chunk_time_ranges_s,
        template_config,
        matching_config,
        tpca,
        data_dir_chunks,
        neighbors, # Specify neighbors here -- no neighbors 
        templates_smoothed,
        fill_nanvalue=fill_nanvalue,
        norm_operator=norm_operator,
        min_overlapping_chans_ratio=min_overlapping_chans_ratio,
        radius=radius,
    )

    new_labels = sorting.labels.copy()
    all_units = np.unique(sorting.labels)
    all_units = all_units[all_units>-1]

    new_labels[residual_norm>min_norm_triage] = -1

    return new_labels  

def compute_residual_norm_moving_temps(
    recording, 
    labels, #Here doesn't have to be split
    motion_est,
    chunk_time_ranges_s,
    template_config,
    matching_config,
    tpca,
    data_dir_chunks,
    neighbors_over_time,
    temp_data_smoothed,
    min_overlapping_chans_ratio=1., #0.5,
    fill_nanvalue=1_000_000,
    norm_operator=np.nanmax,
    radius=None,
    # return_num_channels=False,
):
    # Can speed this up!! only look at first sorting (before split)
    geom = recording.get_channel_locations()
    registered_geom = registered_geometry(geom, motion_est)

    residual_Linf_norm = fill_nanvalue*np.ones((labels.shape[0], 3))
    num_overlapping_chans = np.zeros((labels.shape[0], 3))
    num_chan_waveforms_nonan = np.zeros(labels.shape[0])
    
    cmp=0
    for j, chunk_time_range in enumerate(chunk_time_ranges_s):
        sub_chunk_time_range_s = subchunks_time_ranges(recording, chunk_time_range, template_config.subchunk_size_s,
                                              divider_samples=matching_config.chunk_length_samples)
        n_sub_chunks = len(sub_chunk_time_range_s)
        for k, subchunk_time_range in tqdm(enumerate(sub_chunk_time_range_s), total = n_sub_chunks):

            matchh5_chunk = data_dir_chunks / f"chunk_{int(j*n_sub_chunks + k)}_matching0.h5"
            with h5py.File(matchh5_chunk, "r+") as h5:
                times_seconds = h5["times_seconds"][:]
                localizations = h5["point_source_localizations"][:] #[idx]
                channels = h5["channels"][:] #[idx]
                colcleanedwfs = h5["collisioncleaned_tpca_features"][:] #[idx]
                channel_index = h5["channel_index"][:]
                
            n_spikes_chunk = len(times_seconds)
            indices_chunk = np.arange(cmp, cmp+n_spikes_chunk)
            cmp+=n_spikes_chunk

            all_units = np.unique(labels)
            for unit in all_units[all_units>-1]:
                if neighbors_over_time.ndim == 3:
                    unit_neighbors = neighbors_over_time[int(j*n_sub_chunks + k), unit]
                else: 
                    unit_neighbors = neighbors_over_time[unit]
                idx_unit = np.flatnonzero(labels[indices_chunk] == unit)
                idx_unit_all_chunk = indices_chunk[idx_unit]

                if radius is not None: 
                    wfs_unit, new_channel_index = channel_subset_by_radius(
                        colcleanedwfs[idx_unit],
                        channels[idx_unit],
                        channel_index,
                        geom,
                        radius=radius
                    )
                else:
                    wfs_unit = colcleanedwfs[idx_unit],
                n_pitches_shift = get_spike_pitch_shifts(
                    localizations[idx_unit, 2],
                    geom=geom,
                    times_s=times_seconds[idx_unit],
                    motion_est=motion_est)
            
                wfs_unit = get_waveforms_on_static_channels(
                    wfs_unit,
                    geom,
                    channels[idx_unit], 
                    new_channel_index, 
                    registered_geom=registered_geom,
                    n_pitches_shift=n_pitches_shift,
                )
                
                wfs_unit = tpca.inverse_transform(wfs_unit.transpose(0, 2, 1).reshape(-1, 8)).reshape(-1, registered_geom.shape[0], 121).transpose(0, 2, 1) #[:, :, 0] 
                wfs_unit = wfs_unit[:, 25:70, :]

                num_chan_waveforms_nonan[idx_unit_all_chunk] = (~np.isnan(wfs_unit[:, 0])).sum(1)
    
                for i, neigh in enumerate(unit_neighbors):
                    if neigh>-1:
                        temp_neigh = temp_data_smoothed[int(j*n_sub_chunks + k), neigh]
                        temp_neigh = temp_neigh[25:70]
                        n_overlap_chans = (~np.isnan((wfs_unit - temp_neigh[None])[:, 0])).sum(1)
                        idx_enough_overlap = n_overlap_chans >= min_overlapping_chans_ratio*num_chan_waveforms_nonan[idx_unit_all_chunk]

                        if idx_enough_overlap.sum()>0:
                            residual_Linf_norm[idx_unit_all_chunk[idx_enough_overlap], i] = norm_operator(np.abs(wfs_unit[idx_enough_overlap] - temp_neigh[None]), axis = (1, 2))
                            # if unit == 23:
                            #     print("ENOUGH OVERLAP")
                            #     print(idx_enough_overlap)
                            #     print(residual_Linf_norm[idx_unit_all_chunk[idx_enough_overlap], i])
                        num_overlapping_chans[idx_unit_all_chunk, i] = n_overlap_chans
    
    residual_Linf_norm[np.isnan(residual_Linf_norm)] = fill_nanvalue

    return residual_Linf_norm, num_overlapping_chans, num_chan_waveforms_nonan



def compute_residual_norm(
    recording, 
    labels, #Here doesn't have to be split
    motion_est,
    chunk_time_ranges_s,
    template_config,
    matching_config,
    tpca,
    data_dir_chunks,
    neighbors,
    temp_data_smoothed,
    fill_nanvalue=1_000_000,
    norm_operator=np.nanmax,
    return_num_channels=False,
):
    # Can speed this up!! only look at first sorting (before split)
    geom = recording.get_channel_locations()
    registered_geom = registered_geometry(geom, motion_est)

    residual_Linf_norm = fill_nanvalue*np.ones((labels.shape[0], neighbors.shape[1]))
    if return_num_channels:
        num_overlapping_chans = np.zeros((labels.shape[0], 3))
        num_chan_waveforms_nonan = np.zeros(labels.shape[0])
    cmp=0
    for j, chunk_time_range in enumerate(chunk_time_ranges_s):
        sub_chunk_time_range_s = subchunks_time_ranges(recording, chunk_time_range, template_config.subchunk_size_s,
                                              divider_samples=matching_config.chunk_length_samples)
        n_sub_chunks = len(sub_chunk_time_range_s)
        for k, subchunk_time_range in tqdm(enumerate(sub_chunk_time_range_s), total = n_sub_chunks):

            matchh5_chunk = data_dir_chunks / f"chunk_{int(j*n_sub_chunks + k)}_matching0.h5"
            with h5py.File(matchh5_chunk, "r+") as h5:
                times_seconds = h5["times_seconds"][:]
                localizations = h5["point_source_localizations"][:] #[idx]
                channels = h5["channels"][:] #[idx]
                colcleanedwfs = h5["collisioncleaned_tpca_features"][:] #[idx]
                channel_index = h5["channel_index"][:]
                
            n_spikes_chunk = len(times_seconds)
            indices_chunk = np.arange(cmp, cmp+n_spikes_chunk)
            cmp+=n_spikes_chunk
                        
            for unit in np.unique(labels):
                unit_neighbors = neighbors[unit]
                idx_unit = np.flatnonzero(labels[indices_chunk] == unit)
                idx_unit_all_chunk = indices_chunk[idx_unit]
                # Maybe above, indexing is wrong>
                
                n_pitches_shift = get_spike_pitch_shifts(
                    localizations[idx_unit, 2],
                    geom=geom,
                    times_s=times_seconds[idx_unit],
                    motion_est=motion_est)
            
                wfs_unit = get_waveforms_on_static_channels(
                    colcleanedwfs[idx_unit],
                    geom,
                    channels[idx_unit], 
                    channel_index, 
                    registered_geom=registered_geom,
                    n_pitches_shift=n_pitches_shift,
                )
                
                wfs_unit = tpca.inverse_transform(wfs_unit.transpose(0, 2, 1).reshape(-1, 8)).reshape(-1, registered_geom.shape[0], 121).transpose(0, 2, 1) #[:, :, 0] 
                wfs_unit = wfs_unit[:, 25:70, :]

                if return_num_channels:
                    num_chan_waveforms_nonan[idx_unit_all_chunk] = (~np.isnan(wfs_unit[:, 0])).sum(1)
    
                for i, neigh in enumerate(unit_neighbors):
                    if neigh>-1:
                        temp_neigh = temp_data_smoothed[int(j*n_sub_chunks + k), neigh]
                        temp_neigh = temp_neigh[25:70]
                        residual_Linf_norm[idx_unit_all_chunk, i] = norm_operator(np.abs(wfs_unit - temp_neigh[None]), axis = (1, 2))
                        if return_num_channels:
                            num_overlapping_chans[idx_unit_all_chunk, i] = (~np.isnan((wfs_unit - temp_neigh[None])[:, 0])).sum(1)
                            
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
    data_dir_chunks,
    temp_data_smoothed,
    trough_offset=42,
    peak_time_selection="maxstd",
    spike_length_samples=121,
    normalize_by_max_value=True,
):

    temp_data_smoothed[np.isnan(temp_data_smoothed)]=0
    geom = recording.get_channel_locations()
    registered_geom = registered_geometry(geom, motion_est)

    max_chan_residual = np.zeros((sorting.labels.shape[0], spike_length_samples))
    
    cmp=0
    for j, chunk_time_range in enumerate(chunk_time_ranges_s):
        sub_chunk_time_range_s = subchunks_time_ranges(recording, chunk_time_range, template_config.subchunk_size_s,
                                              divider_samples=matching_config.chunk_length_samples)
        n_sub_chunks = len(sub_chunk_time_range_s)
        for k, subchunk_time_range in tqdm(enumerate(sub_chunk_time_range_s), total = n_sub_chunks):

            matchh5_chunk = data_dir_chunks / f"chunk_{int(j*n_sub_chunks + k)}_matching0.h5"
            with h5py.File(matchh5_chunk, "r+") as h5:
                times_seconds = h5["times_seconds"][:]
                localizations = h5["point_source_localizations"][:] #[idx]
                channels = h5["channels"][:] #[idx]
                colcleanedwfs = h5["collisioncleaned_tpca_features"][:] #[idx]
                channel_index = h5["channel_index"][:]
                
            n_spikes_chunk = len(times_seconds)
            indices_chunk = np.arange(cmp, cmp+n_spikes_chunk)
            cmp+=n_spikes_chunk
                        
            for unit in np.unique(sorting.labels):
                idx_unit = np.flatnonzero(sorting.labels[indices_chunk] == unit)
                idx_unit_chunk = indices_chunk[idx_unit]
                
                n_pitches_shift = get_spike_pitch_shifts(
                    localizations[idx_unit, 2],
                    geom=geom,
                    times_s=times_seconds[idx_unit],
                    motion_est=motion_est)
            
                wfs_unit = get_waveforms_on_static_channels(
                    colcleanedwfs[idx_unit],
                    geom,
                    channels[idx_unit], 
                    channel_index, 
                    registered_geom=registered_geom,
                    n_pitches_shift=n_pitches_shift,
                )
                wfs_unit = tpca.inverse_transform(wfs_unit.transpose(0, 2, 1).reshape(-1, 8)).reshape(-1, registered_geom.shape[0], 121).transpose(0, 2, 1) #[:, :, 0] 
    
                temp_mc = temp_data_smoothed[int(j*n_sub_chunks + k), unit]
                mc = temp_mc.ptp(0).argmax()
                max_chan_residual[idx_unit_chunk] = wfs_unit[:, :, mc] - temp_mc[None, :, mc]
    
    if normalize_by_max_value: 
        for unit in np.unique(sorting.labels):
            idx_unit = np.flatnonzero(sorting.labels==unit)
            no_nan = ~np.isnan(max_chan_residual[idx_unit, 0])
            if peak_time_selection=="maxstd":
                trough_offset = max_chan_residual[idx_unit][no_nan].std(0).argmax()
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
                    temp_mc = temp_data_smoothed[int(j*n_sub_chunks + k), unit]
                    max_chan_residual[idx_unit[idx_chunk]]/=np.abs(temp_mc[trough_offset, mc])

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

def full_reassignment_split(
    sorting,
    recording,
    motion_est,
    chunk_time_ranges_s,
    template_config,
    matching_config,
    data_dir_chunks,
    tpca,
    deconv_scores,
    return_triaged_labels=True,
    threshold_n_spike=0.2,
    fill_nanvalue=1_000_000,
    norm_operator=np.nanmax,
    spike_length_samples=121,
    peak_time_selection="maxstd",
    trough_offset=42,
    max_value_no_split=0.25,
    min_value_split=0.75,
    normalize_by_max_value=True,
    min_nspikes_unit=150,
    triage_spikes_2way=0.55,
    triage_spikes_3way=0.5,
):

    weights_deconv = np.log(1 + np.abs(deconv_scores-deconv_scores.min()))
    tpca_templates_list, spike_count_list, unit_ids_list = create_tpca_templates_list(
        recording, 
        sorting,
        motion_est,
        chunk_time_ranges_s,
        template_config, 
        matching_config,
        data_dir_chunks,
        weights=weights_deconv,
        tpca=tpca,
    )

    templates_smoothed = smooth_list_templates(
        tpca_templates_list, spike_count_list, unit_ids_list, np.unique(sorting.labels), threshold_n_spike=threshold_n_spike,
    )

    max_chan_residual = compute_maxchan_tpca_residual(
        recording, 
        sorting,
        motion_est,
        chunk_time_ranges_s,
        template_config,
        matching_config,
        tpca,
        data_dir_chunks,
        templates_smoothed,
        spike_length_samples=spike_length_samples,
        normalize_by_max_value=normalize_by_max_value,
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

    tpca_templates_list_split, spike_count_list_split, unit_ids_list_split = create_tpca_templates_list(
        recording, 
        sorting_split,
        motion_est,
        chunk_time_ranges_s,
        template_config, 
        matching_config,
        data_dir_chunks,
        weights=None,
        tpca=tpca,
    )

    templates_smoothed_split = smooth_list_templates(
        tpca_templates_list_split, spike_count_list_split, unit_ids_list_split, np.unique(sorting_split.labels),
    )

    residual_norm = compute_residual_norm(
        recording, 
        sorting.labels,
        motion_est,
        chunk_time_ranges_s,
        template_config,
        matching_config,
        tpca,
        data_dir_chunks,
        neighbors,
        templates_smoothed_split,
        fill_nanvalue=fill_nanvalue,
        norm_operator=norm_operator,
    )

    units = np.unique(sorting.labels)
    units = units[units>-1]
    new_labels = -1*np.ones(sorting.labels.shape)
    for unit in units:
        idx_unit = np.flatnonzero(sorting.labels == unit)
        new_labels[idx_unit] = neighbors[sorting.labels[idx_unit], residual_norm.argmin(1)[idx_unit]]
    new_labels = new_labels.astype('int')

    if not return_triaged_labels:
        return new_labels, templates_smoothed_split

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
        return new_labels, labels_hardassignments_only, templates_smoothed_split

def square_mean(x, axis=1):
    return np.nanmean(np.square(x), axis=axis)

 # UPDATE TO perform EM iterations
