"""
Utility functions for dealing with the output of time-tracking deconv, which now stores multiple h5 files
"""

from dataclasses import dataclass
from tqdm.auto import tqdm
import numpy as np
from dartsort.util.data_util import chunk_time_ranges, subchunks_time_ranges
import h5py
from dartsort.templates.templates import TemplateData

def create_tpca_templates_list(
    recording, 
    sorting,
    me,
    chunk_time_ranges_s,
    template_config, 
    matching_config,
    data_dir_chunks,
    tpca_list=None,
    tpca_rank=8,
    n_spike_samples=121,
    weights=None,
):

    """
    TODO: Parallelize this code
    """
    
    cmp=0
    tpca_templates_list = []
    unit_ids_list = []
    spike_count_list = []
    
    for j, chunk_time_range in tqdm(enumerate(chunk_time_ranges_s), total = len(chunk_time_ranges_s), desc="Making tpca colcleaned templates"):
        
        colcleaned_wfs_unit = []
        sub_chunk_time_range_s = subchunks_time_ranges(recording, chunk_time_range, template_config.subchunk_size_s,
                                                  divider_samples=matching_config.chunk_length_samples)
        n_sub_chunks = len(sub_chunk_time_range_s)
        
        for k, subchunk_time_range in enumerate(sub_chunk_time_range_s):
                
            matchh5_chunk = data_dir_chunks / f"chunk_{int(j*n_sub_chunks + k)}_matching0.h5"
    # 
            with h5py.File(matchh5_chunk, "r+") as h5:
                n_spikes_chunk = len(h5["times_samples"][:])

            indices_chunk = np.arange(cmp, cmp+n_spikes_chunk)
            tpca = tpca_list[int(j*n_sub_chunks + k)]
            cmp+=n_spikes_chunk
    
            temp_data_colcleaned = TemplateData.from_h5_with_colcleanedwfs(
                recording,
                data_dir_chunks / f"chunk_{int(j*n_sub_chunks + k)}_matching0.h5",
                sorting,
                template_config,
                indices=indices_chunk,
                weight_wfs=weights,
                save_folder=None,
                motion_est=me,
            )   
            unit_ids_list.append(temp_data_colcleaned.unit_ids)
            spike_count_list.append(temp_data_colcleaned.spike_counts)
            if tpca_list is not None:
                temp_data_colcleaned = tpca_list[int(j*n_sub_chunks + k)].inverse_transform(temp_data_colcleaned.templates.transpose(0, 2, 1).reshape(-1, tpca_rank)).reshape(-1, temp_data_colcleaned.templates.shape[2], n_spike_samples).transpose(0, 2, 1) #[:, :, 0] 
            tpca_templates_list.append(temp_data_colcleaned)

    return tpca_templates_list, spike_count_list, unit_ids_list
