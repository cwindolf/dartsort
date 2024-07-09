from dataclasses import replace

import numpy as np
import h5py
from .. import config
from ..templates import TemplateData
import os
from tqdm.auto import tqdm
from pathlib import Path
from dataclasses import replace
from scipy.cluster.hierarchy import fcluster, linkage

def get_med_locations_ptps(sorting, depth_reg, max_value, scales=(1, 1, 50), log_c=5):
    units = np.unique(sorting.labels)
    units = units[units>-1]
    loc_ptps = np.zeros((units.max()+1, 3))
    if max_value is not None:
        loc_ptps = np.zeros((units.max()+1, 4))
    else:
        loc_ptps = np.zeros((units.max()+1, 3))
    for u in units:
        idx_unit = np.flatnonzero(sorting.labels == u)
        loc_ptps[u, 0] = scales[0]*np.median(sorting.point_source_localizations[idx_unit, 0])
        loc_ptps[u, 1] = scales[1]*np.median(depth_reg[idx_unit])
        loc_ptps[u, 2] = scales[2]*np.log(log_c+np.median(sorting.denoised_ptp_amplitudes[idx_unit]))
        if max_value is not None:
            loc_ptps[u, 3] = np.median(max_value[idx_unit])
    return loc_ptps

def merge_units_close_in_space(sorting, motion_est = None, max_value = None, merge_threshold = 20, scales=(1, 1, 50), log_c=5, fill_value=10_000, link="complete"):

    if motion_est is not None:
        z_reg = motion_est.correct_s(sorting.times_seconds, sorting.point_source_localizations[:, 2])
    else:
        z_reg = sorting.point_source_localizations[:, 2]

    med_loc_ptps = get_med_locations_ptps(sorting, z_reg, max_value, scales, log_c)

    if max_value is not None:
        is_positive = med_loc_ptps[:, 3] > 0
        same_sign = (is_positive[:, None] & is_positive[None]) + (~is_positive[:, None] & ~is_positive[None])
        med_loc_ptps = med_loc_ptps[:, :3]
        dist_matrix = np.sqrt(((med_loc_ptps[:, None] - med_loc_ptps[None])**2).sum(2))
        dist_matrix[~same_sign] = fill_value
        
    pdist = dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]
    finite = np.isfinite(pdist)
    if not finite.any():
        return sorting
    pdist[~finite] = fill_value + pdist[finite].max()
    Z = linkage(pdist, method=link)
    new_labels = fcluster(Z, merge_threshold, criterion="distance")
    
    units = np.unique(sorting.labels)
    units = units[units>-1]
    labels_updated = np.full(sorting.labels.shape, -1)
    kept = np.flatnonzero(np.isin(sorting.labels, np.unique(units)))
    labels_updated[kept] = sorting.labels[kept].copy()
    flat_labels = labels_updated[kept]
    labels_updated[kept] = new_labels[flat_labels]
    labels_updated[labels_updated>-1] -= labels_updated[labels_updated>-1].min()

    sorting = replace(sorting, labels = labels_updated)
    return sorting

    
def separate_positive_negative_wfs(
    sorting,
    peeling_hdf5_filename,
    tpca=None,
    wfs_name="collisioncleaned_tpca_features",
    trough_offset=42,
    return_max_value=False,
):

    units = np.unique(sorting.labels)
    units = units[units>-1]
    new_labels = sorting.labels.copy()
    cmp = units.max()+1
    max_value = np.zeros(sorting.labels.shape)
    
    with h5py.File(peeling_hdf5_filename, "r+") as h5: 
        channels = h5["channels"][:]
        channel_index = h5["channel_index"][:]
        dataset = h5[wfs_name]
        for sli, *_ in tqdm(dataset.iter_chunks()):
            idx = np.where((channel_index[channels[sli]] == channels[sli][:, None]))
            collisioncleaned_tpca_features = dataset[sli]
            waveforms_maxchan = collisioncleaned_tpca_features[idx[0], :, idx[1]]
            if tpca is not None:
                waveforms_maxchan = tpca.inverse_transform(waveforms_maxchan)
            max_value[sli] = waveforms_maxchan[:, trough_offset]

    for unit in units:
        idx_unit = np.flatnonzero(sorting.labels == unit)
        idx_unit_positive = idx_unit[max_value[idx_unit]>=0]
        new_labels[idx_unit_positive] = cmp
        cmp+=1
    sorting = replace(sorting, labels = new_labels)

    if return_max_value:
        return sorting, max_value
    return sorting


def chuck_noisy_template_units_with_time_tracking(
    recording,
    sorting,
    chunk_time_ranges_s,
    template_config,
    template_data_list=None,
    motion_est=None,
    trough_offset_samples=42,
    spike_length_samples=121,
    tsvd=None,
    device=None,
    n_jobs=0,
    template_save_dir=None,
    template_npz_filename="template_data.npz",
    overwrite=False,
    return_denoising_tsvd=False,
):
    """Get rid of noise units.

    This will reindex the sorting and template data -- unit labels will
    change, and the number of templates will change.
    """
    if template_save_dir is not None:
        os.makedirs(template_save_dir, exist_ok=True)

    if template_data_list is None: 
        res = TemplateData.from_config_multiple_chunks_linear(
            recording,
            chunk_time_ranges_s,
            sorting,
            template_config,
            save_folder=Path(template_save_dir),
            overwrite=overwrite,
            motion_est=motion_est,
            save_npz_name=template_npz_filename,
            n_jobs=n_jobs, 
            device=device,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            denoising_tsvd=tsvd,
            return_denoising_tsvd=return_denoising_tsvd,
        )
        if return_denoising_tsvd:
            template_data_list, tsvd = res
        else:
            template_data_list = res

    units = np.unique(sorting.labels)
    units = units[units>-1]

    good_unit_ids = []

    for template_data in tqdm(template_data_list, desc = "GC with template data"):
        template_ptps = np.nanmax(template_data.templates.ptp(1), 1)
        template_snrs = template_ptps * np.sqrt(template_data.spike_counts)
        good_templates = np.logical_and(template_data.spike_counts >=  template_config.min_count_at_shift, template_snrs>=template_config.denoising_snr_threshold)
        good_unit_ids.append(
            template_data.unit_ids[good_templates]
        )

    good_unit_ids = np.hstack(good_unit_ids)
    unique_good_unit_ids, new_ids = np.unique(good_unit_ids, return_inverse=True)

    # print(unique_good_unit_ids)
    # print(new_ids)

    new_labels = sorting.labels.copy()
    valid = np.isin(new_labels, unique_good_unit_ids)
    new_labels[~valid] = -1
    _, new_labels[valid] = np.unique(new_labels[valid], return_inverse=True)

    new_sorting = replace(sorting, labels=new_labels)

    new_template_data_list = []
    for j, template_data in tqdm(enumerate(template_data_list), total = len(chunk_time_ranges_s), desc = "Updating templates after GC"):
        chunk_good_units = np.isin(template_data.unit_ids, good_unit_ids)
        rtdum = None
        if template_data.registered_template_depths_um is not None:
            rtdum = template_data.registered_template_depths_um[chunk_good_units] 
        new_template_data = TemplateData(
            templates=template_data.templates[chunk_good_units],
            unit_ids=np.unique(new_ids[np.isin(good_unit_ids, template_data.unit_ids[chunk_good_units])]), # IS THIS CORRECT?
            spike_counts=template_data.spike_counts[chunk_good_units],
            registered_geom=template_data.registered_geom,
            registered_template_depths_um=rtdum,
        )
        new_template_data_list.append(new_template_data)
        if template_save_dir is not None:
            template_chunk_npz_file = template_save_dir / f"chunk_{j}_{template_npz_filename}"
            new_template_data.to_npz(template_chunk_npz_file)

    if return_denoising_tsvd:
        return new_sorting, new_template_data_list, tsvd
    return new_sorting, new_template_data_list



def realign_and_chuck_noisy_template_units(
    recording,
    sorting,
    template_data=None,
    motion_est=None,
    min_n_spikes=5,
    min_template_snr=15,
    template_config=config.coarse_template_config,
    trough_offset_samples=42,
    spike_length_samples=121,
    tsvd=None,
    device=None,
    n_jobs=0,
):
    """Get rid of noise units.

    This will reindex the sorting and template data -- unit labels will
    change, and the number of templates will change.
    """
    if template_data is None:
        template_data, sorting = TemplateData.from_config(
            recording,
            sorting,
            template_config,
            motion_est=motion_est,
            n_jobs=n_jobs,
            tsvd=tsvd,
            device=device,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            return_realigned_sorting=True,
        )

    template_ptps = np.nanmax(template_data.templates.ptp(1), 1)
    template_snrs = template_ptps * np.sqrt(template_data.spike_counts)
    good_templates = np.logical_and(
        template_data.spike_counts >= min_n_spikes,
        template_snrs > min_template_snr,
    )

    good_unit_ids = template_data.unit_ids[good_templates]
    assert np.all(np.diff(good_unit_ids) >= 0)
    unique_good_unit_ids, new_template_unit_ids = np.unique(good_unit_ids, return_inverse=True)

    new_labels = sorting.labels.copy()
    valid = np.isin(new_labels, unique_good_unit_ids)
    new_labels[~valid] = -1
    _, new_labels[valid] = np.unique(new_labels[valid], return_inverse=True)

    new_sorting = replace(sorting, labels=new_labels)
    rtdum = None
    if template_data.registered_template_depths_um is not None:
        rtdum = template_data.registered_template_depths_um[good_templates]
    new_template_data = TemplateData(
        templates=template_data.templates[good_templates],
        unit_ids=new_template_unit_ids,
        spike_counts=template_data.spike_counts[good_templates],
        registered_geom=template_data.registered_geom,
        registered_template_depths_um=rtdum,
    )

    return new_sorting, new_template_data


def chuck_noisy_template_units_from_merge(
    sorting_pre_merge,
    sorting_post_merge,
    template_data_list_pre_merge,
    spike_count_max=250,
    min_n_spikes=25,
    min_template_snr=50,
    device=None,
    n_jobs=0,
):
    """Get rid of noise units.

    This will reindex the sorting and template data -- unit labels will
    change, and the number of templates will change.

    This takes as input the pre-merge template data, and the sorting after merge + merge unit mapping to automatically discard unit i.e. without computing new temp data 
    """

    units_postmerge = np.unique(sorting_post_merge.labels)
    units_postmerge = units_postmerge[units_postmerge>-1]

    good_unit_ids = []

    for template_data in tqdm(template_data_list_pre_merge, desc = "GC with pre-merge template data"):
        # no_0_count = template_data.spike_counts>0
        for u in units_postmerge:
            units_premerge = np.unique(sorting_pre_merge.labels[sorting_post_merge.labels==u])
            temp_premerge = template_data.templates[np.isin(template_data.unit_ids, units_premerge)]
            spikecount_premerge = template_data.spike_counts[np.isin(template_data.unit_ids, units_premerge)]
            if temp_premerge.ndim==2:
                template_snrs = spikecount_premerge*np.nanmax(temp_premerge.ptp(0))
            else:
                template_snrs = (spikecount_premerge[:, None, None]*temp_premerge/np.nanmax(spikecount_premerge.sum()).sum(0).ptp(0))*np.min((spikecount_premerge.sum(), spike_count_max))
                spikecount_premerge = np.min((spikecount_premerge.sum(), spike_count_max))
            if spikecount_premerge >= min_n_spikes and template_snrs > min_template_snr:
                good_unit_ids.append(u)
                
    good_unit_ids = np.asarray(good_unit_ids)
    unique_good_unit_ids = np.unique(good_unit_ids)

    new_labels = sorting_post_merge.labels.copy()
    valid = np.isin(new_labels, unique_good_unit_ids)
    new_labels[~valid] = -1
    _, new_labels[valid] = np.unique(new_labels[valid], return_inverse=True)

    new_sorting = replace(sorting_post_merge, labels=new_labels)

    print(f"GC keeps {len(unique_good_unit_ids)} units")
    return new_sorting

# def chuck_noisy_template_units_with_loaded_spikes_per_chunk(
#     sorting,
#     template_data_list,
#     template_save_folder=None,
#     min_n_spikes=25,
#     min_template_snr=50,
#     template_npz_filename="template_data.npz",
#     device=None,
#     n_jobs=0,
# ):
#     """Get rid of noise units.

#     This will reindex the sorting and template data -- unit labels will
#     change, and the number of templates will change.

#     This takes as input a list of template data for each chunk and remove units that are too noisy in ALL chunks
#     """
    
#     good_unit_ids = []
#     good_templates_all = []
#     for k in range(len(template_data_list)):
#         template_data = template_data_list[k]

#         # no_0_count = template_data.spike_counts>0
        
#         template_ptps = template_data.templates.ptp(1).max(1)
#         template_snrs = template_ptps * np.sqrt(template_data.spike_counts)
#         good_templates = np.logical_and(
#             template_data.spike_counts >= min_n_spikes,
#             template_snrs > min_template_snr,
#         )
#         # If good in at least one chunk then keep!
#         good_unit_ids.append(template_data.unit_ids[good_templates])
        
#     good_unit_ids = np.hstack(good_unit_ids)
#     unique_good_unit_ids = np.unique(good_unit_ids)

#     new_labels = sorting.labels.copy()
#     valid = np.isin(new_labels, unique_good_unit_ids)
#     new_labels[~valid] = -1
#     _, new_labels[valid] = np.unique(new_labels[valid], return_inverse=True)

#     new_sorting = replace(sorting, labels=new_labels)

#     template_data_all = []

#     if template_save_folder is not None:
#         for k in range(len(template_data_list)):
#             template_save_folder_chunk = template_save_folder / f"chunk_{k}_merge"
#             os.makedirs(template_save_folder_chunk, exist_ok=True)
#             npz_path = template_save_folder_chunk / template_npz_filename
            
#             template_data = template_data_list[k]
#             # no_0_count = template_data.spike_counts>0
#             # good_templates = good_templates_all[k] 

#             rtdum = None
#             if template_data.registered_template_depths_um is not None:
#                 rtdum = template_data.registered_template_depths_um[np.isin(template_data.unit_ids, unique_good_unit_ids)]
#             new_template_data = TemplateData(
#                 templates=template_data.templates[np.isin(template_data.unit_ids, unique_good_unit_ids)],
#                 unit_ids=template_data.unit_ids[np.isin(template_data.unit_ids, unique_good_unit_ids)],
#                 spike_counts=template_data.spike_counts[np.isin(template_data.unit_ids, unique_good_unit_ids)],
#                 registered_geom=template_data.registered_geom,
#                 registered_template_depths_um=rtdum,
#             )
#             new_template_data.to_npz(npz_path)
#             template_data_all.append(new_template_data)

#     return new_sorting, template_data_all
