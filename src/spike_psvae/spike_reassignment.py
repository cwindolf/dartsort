# %%
"""
outlier detection & soft assignment

input:
    - residual bin file path
    - template path
    - spike train path
    - geometry path
    - output path
    - tPCA components and mean
    - number of channels: 7 by default
    - number of similar units: 3 by default
    
output:
    - reassignment.npy: N dimensional numpy array (N: number of spikes)
        -1 for outlier    
    - soft_assignment_scores: n_sim_units x N dimensional numpy array
        max norm of tpca'd spikes; scores used to reassign spikes
"""

# %%
import numpy as np
import scipy.spatial.distance as dist
from sklearn.decomposition import PCA
import torch
from tqdm.auto import tqdm
from spike_psvae import deconvolve
from spike_psvae.spikeio import read_data, read_waveforms
import os
from spike_psvae.cluster_utils import (
    compute_shifted_similarity,
    get_closest_clusters_hdbscan,
)
from collections import defaultdict
from spike_psvae.localization import localize_ptp
# from spike_psvae.waveform_utils import (
#     get_local_geom,
#     relativize_waveforms,
#     channel_index_subset,
# )
from spike_psvae.pre_deconv_merge_split import (
    get_proposed_pairs,
    get_x_z_templates,
)
import matplotlib.pyplot as plt
import h5py
from spike_psvae import waveform_utils


# %%
def run_with_cleaned_wfs(
    deconv_h5_path,
    template_path,
    spike_train_path,
    geom,
    tpca,
    channel_index_h5,
    n_chans=8,
    n_sim_units=2,
    num_sigma_outlier=4,
    batch_size=2048,
    output_path=None,
    soft_assignment_scores=None,
):

    # save file
    if output_path is not None:
        reassignment_file_path = os.path.join(output_path, "reassignment.npy")
        reassigned_scores_path = os.path.join(
            output_path, "reassignment_scores.npy"
        )
        soft_assignment_scores_path = os.path.join(
            output_path, "soft_assignment_scores.npy"
        )

    # load templates
    templates = np.load(template_path)
    ptps = templates.ptp(1)
    mcs = ptps.argmax(1)
    

    # load spike train
    spike_train = np.load(spike_train_path)

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # channels to be used to compute soft assignment scores
    extract_channel_index = []
    for c in range(geom.shape[0]):
        low = max(0, c - n_chans // 2)
        low = min(geom.shape[0] - n_chans, low)
        extract_channel_index.append(np.arange(low, low + n_chans))
    extract_channel_index = np.array(extract_channel_index)

    # number of channels used for detect/subtract (hardcoded to 40 for now)
    extract_channel_index_40 = []
    for c in range(geom.shape[0]):
        low = max(0, c - 40 // 2)
        low = min(geom.shape[0] - 40, low)
        extract_channel_index_40.append(np.arange(low, low + 40))
    extract_channel_index_40 = np.array(extract_channel_index_40)

    # get similar templates
    similar_array, _ = get_similar_templates(
        templates, extract_channel_index_40, n_sim_units, n_chans, geom
    )
    similar_array = np.hstack(
        (np.arange(similar_array.shape[0])[:, None], similar_array)
    )  # add self as a similar template

    # initialize spike reassignment array to -2
    spike_reassignment = np.ones(len(spike_train)) * -2

    if soft_assignment_scores is None:
        # initialize soft_assignment_scores to zero
        soft_assignment_scores = np.zeros((n_sim_units + 1, len(spike_train)))
        for unit in tqdm(range(templates.shape[0])):
            similar_units = similar_array[unit]
            spike_idx = np.where(spike_train[:, 1] == unit)
            n_spikes = spike_idx[0].shape[0]
            if n_spikes == 0:
                continue
            for batch in range(0, n_spikes, batch_size):
                spike_idx_batch = spike_idx[0][batch : batch + batch_size]
                with h5py.File(deconv_h5_path, "r+") as h5: 
                    cleaned_waveforms = h5["cleaned_waveforms"][spike_idx][batch : batch + batch_size]
#                     first_channels = h5["first_channels"][spike_idx][batch : batch + batch_size]
                for i, su in enumerate(similar_units):
                    batch_spike_train = spike_train[spike_idx_batch]
                    batch_t, batch_template_idx = (
                        batch_spike_train[:, 0],
                        batch_spike_train[:, 1],
                    )

                    batch_mcs = mcs[batch_template_idx.astype('int')]
                    batch_extract_channel_index = channel_index_h5[
                        batch_mcs
                    ]
                    mc = mcs[unit]
                    extract_channels = extract_channel_index[mc]
                    chan_bool = np.isin(
                        channel_index_h5[mc], extract_channels
                    )

                    # N, T, 20
                    batch_cleaned_waveforms = waveform_utils.channel_subset_by_index(
                        cleaned_waveforms, batch_mcs, channel_index_h5, extract_channel_index)

                    temp_small = templates[su][:, extract_channel_index[batch_mcs]].transpose((1, 0, 2))

                    residual_batch_tpca_max = batch_cleaned_waveforms - temp_small
                        
                    N, T, C = residual_batch_tpca_max.shape
                    # max norm of tPCA'd residuals
                    wf = residual_batch_tpca_max.transpose(0, 2, 1).reshape(
                        -1, T
                    )
                    transformed_wf = tpca.inverse_transform(
                        tpca.fit_transform(wf)
                    )
                    transformed_wf = transformed_wf.reshape(N, C, T).transpose(
                        0, 2, 1
                    )
                    scores = np.abs(transformed_wf).max(axis=(1, 2))
                    soft_assignment_scores[i, spike_idx_batch] = scores
        if output_path is not None:
            np.save(
                soft_assignment_scores_path, soft_assignment_scores
            )  # soft assignment scores

    # reassign spikes to closest templates
    reassigned_scores = np.zeros(len(spike_train))
    assignments = soft_assignment_scores.argmin(0)
    for unit in tqdm(range(templates.shape[0])):
        similar_units = similar_array[unit]
        idx = np.isin(spike_train[:, 1], np.ones(1) * unit)
        for i, su in enumerate(similar_units):
            idxx = np.where(np.logical_and(idx, assignments == i))[0]
            spike_reassignment[idxx] = su
            reassigned_scores[idxx] = soft_assignment_scores[i, idxx]

    # outlier triaging on the reassigned spikes
    for unit in tqdm(range(templates.shape[0])):
        # set outlier thresholds
        scores = reassigned_scores[
            np.isin(spike_reassignment, np.ones(1) * unit)
        ]

        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        sigma = mad / 0.6745
        mu = np.median(scores)  # why use median here
        unit_cut_off = mu + num_sigma_outlier * sigma
        outlier_idx = np.logical_and(
            np.isin(spike_reassignment, np.ones(1) * unit),
            reassigned_scores > unit_cut_off,
        )
        spike_reassignment[outlier_idx] = -1
    if output_path is not None:
        np.save(
            reassigned_scores_path, reassigned_scores
        )  # soft assignment scores
        np.save(
            reassignment_file_path, spike_reassignment.astype(int)
        )  # reassignment based on soft assignment scores

    return soft_assignment_scores, spike_reassignment, reassigned_scores


# %%
def run(
    residual_bin_path,
    template_path,
    spike_train_path,
    geom,
    tpca,
    n_chans=8,
    n_sim_units=2,
    num_sigma_outlier=4,
    batch_size=2048,
    output_path=None,
    soft_assignment_scores=None,
):

    # save file
    if output_path is not None:
        reassignment_file_path = os.path.join(output_path, "reassignment.npy")
        reassigned_scores_path = os.path.join(
            output_path, "reassignment_scores.npy"
        )
        soft_assignment_scores_path = os.path.join(
            output_path, "soft_assignment_scores.npy"
        )

    # load templates
    templates = np.load(template_path)
    ptps = templates.ptp(1)
    mcs = ptps.argmax()

    # load spike train
    spike_train = np.load(spike_train_path)

    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # channels to be used to compute soft assignment scores
    extract_channel_index = []
    for c in range(geom.shape[0]):
        low = max(0, c - n_chans // 2)
        low = min(geom.shape[0] - n_chans, low)
        extract_channel_index.append(np.arange(low, low + n_chans))
    extract_channel_index = np.array(extract_channel_index)

    # number of channels used for detect/subtract (hardcoded to 40 for now)
    extract_channel_index_40 = []
    for c in range(geom.shape[0]):
        low = max(0, c - 40 // 2)
        low = min(geom.shape[0] - 40, low)
        extract_channel_index_40.append(np.arange(low, low + 40))
    extract_channel_index_40 = np.array(extract_channel_index_40)

    # get similar templates
    similar_array, _ = get_similar_templates(
        templates, extract_channel_index_40, n_sim_units, n_chans, geom
    )
    similar_array = np.hstack(
        (np.arange(similar_array.shape[0])[:, None], similar_array)
    )  # add self as a similar template

    # initialize spike reassignment array to -2
    spike_reassignment = np.ones(len(spike_train)) * -2

    if soft_assignment_scores is None:
        # initialize soft_assignment_scores to zero
        soft_assignment_scores = np.zeros((n_sim_units + 1, len(spike_train)))
        for unit in tqdm(range(templates.shape[0])):
            similar_units = similar_array[unit]
            spike_idx = np.where(spike_train[:, 1] == unit)
            n_spikes = spike_idx[0].shape[0]
            if n_spikes == 0:
                continue
            for batch in range(0, n_spikes, batch_size):
                spike_idx_batch = spike_idx[0][batch : batch + batch_size]
                for i, su in enumerate(similar_units):
                    batch_spike_train = spike_train[spike_idx_batch]
                    batch_t, batch_template_idx = (
                        batch_spike_train[:, 0],
                        batch_spike_train[:, 1],
                    )
                    batch_mcs = mcs[batch_template_idx]
                    batch_extract_channel_index = extract_channel_index_40[
                        batch_mcs
                    ]
                    mc = mcs[unit]
                    extract_channels = extract_channel_index[mc]
                    chan_bool = np.isin(
                        extract_channel_index_40[mc], extract_channels
                    )

                    # load residual batch
                    residual_batch, skipped_idx = read_waveforms(
                        batch_t, residual_bin_path, n_channels=geom.shape[0]
                    )

                    if su != unit:
                        # add difference in template
                        diff = templates[unit] - templates[su]
                        residual_batch += diff[None]

                    residual_batch_tpca_max = np.array(
                        list(
                            map(
                                lambda x, idx: x[:, idx][:, chan_bool],
                                residual_batch,
                                batch_extract_channel_index,
                            )
                        )
                    )
                    N, T, C = residual_batch_tpca_max.shape
                    # max norm of tPCA'd residuals
                    wf = residual_batch_tpca_max.transpose(0, 2, 1).reshape(
                        -1, T
                    )
                    transformed_wf = tpca.inverse_transform(
                        tpca.fit_transform(wf)
                    )
                    transformed_wf = transformed_wf.reshape(N, C, T).transpose(
                        0, 2, 1
                    )
                    scores = np.abs(transformed_wf).max(axis=(1, 2))
                    soft_assignment_scores[i, spike_idx_batch] = scores
        if output_path is not None:
            np.save(
                soft_assignment_scores_path, soft_assignment_scores
            )  # soft assignment scores

    # reassign spikes to closest templates
    reassigned_scores = np.zeros(len(spike_train))
    assignments = soft_assignment_scores.argmin(0)
    for unit in tqdm(range(templates.shape[0])):
        similar_units = similar_array[unit]
        idx = np.isin(spike_train[:, 1], np.ones(1) * unit)
        for i, su in enumerate(similar_units):
            idxx = np.where(np.logical_and(idx, assignments == i))[0]
            spike_reassignment[idxx] = su
            reassigned_scores[idxx] = soft_assignment_scores[i, idxx]

    # outlier triaging on the reassigned spikes
    for unit in tqdm(range(templates.shape[0])):
        # set outlier thresholds
        scores = reassigned_scores[
            np.isin(spike_reassignment, np.ones(1) * unit)
        ]

        median = np.median(scores)
        mad = np.median(np.abs(scores - median))
        sigma = mad / 0.6745
        mu = np.median(scores)  # why use median here
        unit_cut_off = mu + num_sigma_outlier * sigma
        outlier_idx = np.logical_and(
            np.isin(spike_reassignment, np.ones(1) * unit),
            reassigned_scores > unit_cut_off,
        )
        spike_reassignment[outlier_idx] = -1
    if output_path is not None:
        np.save(
            reassigned_scores_path, reassigned_scores
        )  # soft assignment scores
        np.save(
            reassignment_file_path, spike_reassignment.astype(int)
        )  # reassignment based on soft assignment scores

    return soft_assignment_scores, spike_reassignment, reassigned_scores


# %%
def get_similar_templates(
    templates, extract_channel_index_40, n_sim_units, n_chans, geom
):
    n_units = templates.shape[0]
    ptps = templates.ptp(1)
    mcs = ptps.argmax(1)

    # localize templates
    x_z_templates = np.zeros((n_units, 2))
    for i, template in enumerate(templates):
        mc = mcs[i]
        channels = extract_channel_index_40[mc]
        template_x, _, template_z_rel, template_z_abs, _, _ = localize_ptp(
            template.ptp(0)[channels], channels[0], mc, geom
        )
        x_z_templates[i, 0] = template_x
        x_z_templates[i, 1] = template_z_abs
    dist_argsort, dist_template = get_proposed_pairs(
        templates.shape[0],
        templates,
        x_z_templates,
        n_temp=n_sim_units,
        n_channels=n_chans,
        shifts=[-2, -1, 0, 1, 2],  # predefined shifts
    )
    return dist_argsort, dist_template
