from dataclasses import replace

import numpy as np

from .. import config
from ..templates import TemplateData
import os
from tqdm.auto import tqdm
from dartsort.util.data_util import keep_only_most_recent_spikes


def chuck_noisy_template_units_with_time_tracking(
    recording,
    sorting,
    chunk_time_ranges_s,
    template_data_list=None,
    motion_est=None,
    min_n_spikes=50,
    min_template_snr=50,
    template_config=config.coarse_template_config,
    trough_offset_samples=42,
    spike_length_samples=121,
    tsvd=None,
    device=None,
    n_jobs=0,
    template_save_dir=None,
    model_subdir="post_GC",
    template_npz_filename="template_data.npz",
    overwrite=False,
):
    """Get rid of noise units.

    This will reindex the sorting and template data -- unit labels will
    change, and the number of templates will change.
    """
    if template_save_dir is not None:
        os.makedirs(template_save_dir, exist_ok=True)

    if template_data_list is None:
        template_data_list = []
        for j, chunk_time_range in tqdm(enumerate(chunk_time_ranges_s), total = len(chunk_time_ranges_s), desc = "Making templates before GC"):

            sorting_chunk = keep_only_most_recent_spikes(
                sorting,
                n_min_spikes=template_config.spikes_per_unit,
                latest_time_sample=chunk_time_range[1]
                * recording.sampling_frequency,
            )
            template_data = TemplateData.from_config(
                recording,
                sorting_chunk,
                template_config=template_config,
                motion_est=motion_est,
                n_jobs=n_jobs,
                overwrite=overwrite,
                device=device,
                trough_offset_samples=trough_offset_samples,
                spike_length_samples=spike_length_samples,
            )
            # to remove later if works well
            template_data_list.append(template_data)
            if template_save_dir is not None:
                template_save_dir_chunk = template_save_dir / f"chunk_{j}_{model_subdir}_pre_GC"
                os.makedirs(template_save_dir_chunk, exist_ok=True)
                template_data.to_npz(template_save_dir_chunk / template_npz_filename)

    units = np.unique(sorting.labels)
    units = units[units>-1]

    good_unit_ids = []

    for template_data in tqdm(template_data_list, desc = "GC with template data"):
        template_ptps = template_data.templates.ptp(1).max(1)
        template_snrs = template_ptps * np.sqrt(template_data.spike_counts)
        good_templates = np.logical_and(template_data.spike_counts >=  min_n_spikes, template_snrs>=min_template_snr)
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
            template_save_dir_chunk = template_save_dir / f"chunk_{j}_{model_subdir}"
            os.makedirs(template_save_dir_chunk, exist_ok=True)
            new_template_data.to_npz(template_save_dir_chunk / template_npz_filename)
            
    return new_sorting, new_template_data_list



def chuck_noisy_template_units(
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
        template_data = TemplateData.from_config(
            recording,
            sorting,
            template_config,
            motion_est=motion_est,
            n_jobs=n_jobs,
            tsvd=tsvd,
            device=device,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
        )

    template_ptps = template_data.templates.ptp(1).max(1)
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
    template_npz_filename="template_data.npz",
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
                template_snrs = spikecount_premerge*temp_premerge.ptp(0).max()
            else:
                template_snrs = (spikecount_premerge[:, None, None]*temp_premerge/spikecount_premerge.sum()).sum(0).ptp(0).max()*np.min((spikecount_premerge.sum(), spike_count_max))
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

def chuck_noisy_template_units_with_loaded_spikes_per_chunk(
    sorting,
    template_data_list,
    template_save_folder=None,
    min_n_spikes=25,
    min_template_snr=50,
    template_npz_filename="template_data.npz",
    device=None,
    n_jobs=0,
):
    """Get rid of noise units.

    This will reindex the sorting and template data -- unit labels will
    change, and the number of templates will change.

    This takes as input a list of template data for each chunk and remove units that are too noisy in ALL chunks
    """
    
    good_unit_ids = []
    good_templates_all = []
    for k in range(len(template_data_list)):
        template_data = template_data_list[k]

        # no_0_count = template_data.spike_counts>0
        
        template_ptps = template_data.templates.ptp(1).max(1)
        template_snrs = template_ptps * np.sqrt(template_data.spike_counts)
        good_templates = np.logical_and(
            template_data.spike_counts >= min_n_spikes,
            template_snrs > min_template_snr,
        )
        # If good in at least one chunk then keep!
        good_unit_ids.append(template_data.unit_ids[good_templates])
        
    good_unit_ids = np.hstack(good_unit_ids)
    unique_good_unit_ids = np.unique(good_unit_ids)

    new_labels = sorting.labels.copy()
    valid = np.isin(new_labels, unique_good_unit_ids)
    new_labels[~valid] = -1
    _, new_labels[valid] = np.unique(new_labels[valid], return_inverse=True)

    new_sorting = replace(sorting, labels=new_labels)

    template_data_all = []

    if template_save_folder is not None:
        for k in range(len(template_data_list)):
            template_save_folder_chunk = template_save_folder / f"chunk_{k}_merge"
            os.makedirs(template_save_folder_chunk, exist_ok=True)
            npz_path = template_save_folder_chunk / template_npz_filename
            
            template_data = template_data_list[k]
            # no_0_count = template_data.spike_counts>0
            # good_templates = good_templates_all[k] 

            rtdum = None
            if template_data.registered_template_depths_um is not None:
                rtdum = template_data.registered_template_depths_um[np.isin(template_data.unit_ids, unique_good_unit_ids)]
            new_template_data = TemplateData(
                templates=template_data.templates[np.isin(template_data.unit_ids, unique_good_unit_ids)],
                unit_ids=template_data.unit_ids[np.isin(template_data.unit_ids, unique_good_unit_ids)],
                spike_counts=template_data.spike_counts[np.isin(template_data.unit_ids, unique_good_unit_ids)],
                registered_geom=template_data.registered_geom,
                registered_template_depths_um=rtdum,
            )
            new_template_data.to_npz(npz_path)
            template_data_all.append(new_template_data)

    return new_sorting, template_data_all
