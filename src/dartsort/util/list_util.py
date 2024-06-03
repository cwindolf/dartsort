"""
Utility functions for dealing with the output of time-tracking deconv, which now stores multiple h5 files
"""

from dataclasses import dataclass
from tqdm.auto import tqdm
import numpy as np
from dartsort.util.data_util import chunk_time_ranges, subchunks_time_ranges
import h5py
from dartsort.templates.templates import TemplateData
from pathlib import Path
from dartsort.util.py_util import delay_keyboard_interrupt
from dartsort.util.data_util import SpikeDataset
import os

def merge_allh5_into_one(
    output_directory, 
    recording,
    chunk_time_ranges_s,
    template_config,
    matching_config,
    output_hdf5_filename="matching0.h5",
    name_chunk_h5="matching0.h5", #this corresponds to name f"chunk_{j}_{name_chunk_h5}"
    overwrite=False,
    libver="latest",
    chunk_size=1024,
    remove_previous=False, #set to false by default because maybe a bit dangerous? 
):

    output_hdf5_filename = Path(output_directory) / output_hdf5_filename

    first_h5 = output_directory / f"chunk_0_{name_chunk_h5}"
    out_datasets, fixed_output_data = fixed_and_spike_datasets(first_h5)
    
    output_h5, h5_spike_datasets, cur_n_spikes = initialize_file(
            output_hdf5_filename,
            fixed_output_data,
            out_datasets,
            overwrite=overwrite,
            libver=libver,
            chunk_size=chunk_size,
        )
    
    try:
        for j, chunk_time_range in tqdm(enumerate(chunk_time_ranges_s), total = len(chunk_time_ranges_s), desc="merging h5 files"):
            sub_chunk_time_range_s = subchunks_time_ranges(recording, chunk_time_range, template_config.subchunk_size_s,
                                                      divider_samples=matching_config.chunk_length_samples)
            n_sub_chunks = len(sub_chunk_time_range_s)
            for k, subchunk_time_range in enumerate(sub_chunk_time_range_s):
                matchh5_chunk = output_directory / f"chunk_{int(j*n_sub_chunks + k)}_{name_chunk_h5}"
        
                n_new_spikes = gather_h5files(
                    output_h5, 
                    cur_n_spikes,
                    out_datasets, 
                    h5_spike_datasets,
                    matchh5_chunk,
                )
                cur_n_spikes += n_new_spikes

                if remove_previous:
                    os.remove(matchh5_chunk)
    finally:
        output_h5.close()

    


def fixed_and_spike_datasets(h5_file):
    out_datasets = []
    fixed_output_data = []
    with h5py.File(h5_file, "r+") as h5:
        for k in h5.keys():
            if k in ["last_chunk_start", "sampling_frequency"]:
                fixed_output_data.append(
                    (k, h5[k][()]),
                )
            elif k in ["channel_index", "geom", "registered_geom"]:
                fixed_output_data.append(
                    (k, h5[k][:]),
                )
            elif k not in ["singular_values", "temporal_components", "spatial_components"]:
                dataset = h5[k]
                out_datasets.append(
                    SpikeDataset(name=dataset.name, shape_per_spike=dataset.shape[1:], dtype=dataset.dtype),
                )
    return out_datasets, fixed_output_data

def gather_h5files(
    output_h5, 
    cur_n_spikes,
    out_datasets, 
    h5_spike_datasets,
    new_h5,
):
    with delay_keyboard_interrupt:
        with h5py.File(new_h5, "r+") as h5: 
            output_h5["last_n_spikes"][()] = cur_n_spikes
            n_new_spikes = len(h5["channels"][:])

            if not n_new_spikes:
                return 0
    
            for ds in out_datasets:
                h5_spike_datasets[ds.name].resize(
                    cur_n_spikes + n_new_spikes, axis=0
                )
                h5_spike_datasets[ds.name][cur_n_spikes:] = h5[
                    ds.name
                ][:]

    return n_new_spikes

    

def initialize_file(
    output_hdf5_filename,
    fixed_output_data,
    out_datasets,
    overwrite=False,
    libver="latest",
    chunk_size=1024,
):

    output_hdf5_filename = Path(output_hdf5_filename)
    exists = output_hdf5_filename.exists()
    n_spikes = 0

    if exists and overwrite:
        output_hdf5_filename.unlink()
        output_h5 = h5py.File(output_hdf5_filename, "w", libver=libver)
        output_h5.create_dataset(
            "last_n_spikes", data=0, dtype=np.int64
        )
    elif exists:
        # exists and not overwrite
        output_h5 = h5py.File(output_hdf5_filename, "r+", libver=libver)
        n_spikes = len(output_h5["times_samples"])
    else:
        # didn't exist, so overwrite does not matter
        output_h5 = h5py.File(output_hdf5_filename, "w", libver=libver)
        output_h5.create_dataset(
            "last_n_spikes", data=0, dtype=np.int64
        )
    last_n_spikes = output_h5["last_n_spikes"][()]

    for name, value in fixed_output_data:
        if name not in output_h5:
            output_h5.create_dataset(name, data=value)

    # create per-spike datasets
    # use chunks to support growing the dataset as we find spikes
    h5_spike_datasets = {}
    for ds in out_datasets:
        if ds.name in output_h5:
            h5_spike_datasets[ds.name] = output_h5[ds.name]
        else:
            h5_spike_datasets[ds.name] = output_h5.create_dataset(
                ds.name,
                dtype=ds.dtype,
                shape=(n_spikes, *ds.shape_per_spike),
                maxshape=(None, *ds.shape_per_spike),
                chunks=(chunk_size, *ds.shape_per_spike),
            )

    return output_h5, h5_spike_datasets, last_n_spikes



def create_tpca_templates_list(
    recording, 
    sorting,
    me,
    chunk_time_ranges_s,
    template_config, 
    matching_config,
    data_dir_chunks,
    tpca=None,
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
            if tpca is not None:
                temp_data_colcleaned = tpca.inverse_transform(temp_data_colcleaned.templates.transpose(0, 2, 1).reshape(-1, tpca_rank)).reshape(-1, temp_data_colcleaned.templates.shape[2], n_spike_samples).transpose(0, 2, 1) #[:, :, 0] 
            tpca_templates_list.append(temp_data_colcleaned)

    return tpca_templates_list, spike_count_list, unit_ids_list
