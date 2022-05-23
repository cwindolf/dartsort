import numpy as np
import pickle
import os

import hdbscan
from pathlib import Path
import matplotlib


import torch
import h5py
from sklearn.decomposition import PCA
from spike_psvae import denoise, subtract, localization, ibme, residual, deconvolve
from tqdm.auto import tqdm
from pathlib import Path

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from isosplit import isocut
import merge_split_cleaned

from spike_psvae import cluster, merge_split_cleaned, cluster_viz_index, denoise, cluster_utils, triage, cluster_viz
from spike_psvae.cluster_utils import read_waveforms, compare_two_sorters, make_sorting_from_labels_frames
from spike_psvae.cluster_viz import plot_agreement_venn, plot_unit_similarities
from spike_psvae.cluster_utils import get_closest_clusters_kilosort_hdbscan
from spike_psvae.cluster_viz import plot_single_unit_summary
from spike_psvae.cluster_viz import cluster_scatter, plot_waveforms_geom, plot_venn_agreement #plot_raw_waveforms_unit_geom
from spike_psvae.cluster_viz import array_scatter, plot_self_agreement, plot_single_unit_summary, plot_agreement_venn, plot_isi_distribution, plot_unit_similarities, plot_waveforms_geom_unit
# plot_array_scatter, plot_waveforms_unit_geom
from spike_psvae.cluster_viz import plot_unit_similarity_heatmaps
from spike_psvae.cluster_utils import make_sorting_from_labels_frames, compute_cluster_centers, relabel_by_depth, remove_duplicate_units
# run_weighted_triage
from spike_psvae.triage import run_weighted_triage
from spike_psvae.cluster_utils import get_agreement_indices, compute_spiketrain_agreement, get_unit_similarities, compute_shifted_similarity, read_waveforms
from spike_psvae.cluster_utils import get_closest_clusters_hdbscan, get_closest_clusters_kilosort, get_closest_clusters_hdbscan_kilosort, get_closest_clusters_kilosort_hdbscan



def split(labels_deconv, templates, maxptps, firstchans, path_denoised_wfs_h5, batch_size = 1000, n_chans_split = 10, min_cluster_size=25, min_samples=25):
    cmp = labels_deconv.max() + 1

    for cluster_id in tqdm(np.unique(labels_deconv)):
        which = np.flatnonzero(np.logical_and(maxptps > 4, labels_deconv == cluster_id))
        with h5py.File(path_wfs_h5, "r") as h5:
            batch_wfs = np.empty((len(which), *h5["wfs"].shape[1:]), dtype=h5["wfs"].dtype)
            for batch_start in range(0, len(which), 1000):
                batch_wfs[batch_start : batch_start + batch_size] = h5["wfs"][which[batch_start : batch_start + batch_size]]

        C = batch_wfs.shape[2]
        if C < n_chans_split:
            n_chans_split = C

        maxchan = templates[cluster_id].ptp(0).argmax()
        firstchan_maxchan = maxchan - firstchans[which]
        firstchan_maxchan = np.maximum(firstchan_maxchan, n_chans_split//2)
        firstchan_maxchan = np.minimum(firstchan_maxchan, C - n_chans_split//2)
        firstchan_maxchan = firstchan_maxchan.astype('int')

        if len(np.unique(firstchan_maxchan))<=1:
            wfs_split = batch_wfs[:, :, firstchan_maxchan[0]-n_chans_split//2:firstchan_maxchan[0]+n_chans_split//2]
        else:
            wfs_split = np.zeros((batch_wfs.shape[0], batch_wfs.shape[1], n_chans_split))
            for j in range(batch_wfs.shape[0]):
                wfs_split[j] = batch_wfs[j, :, firstchan_maxchan[j]-n_chans_split//2:firstchan_maxchan[j]+n_chans_split//2]

        pcs_cluster = pca_model.fit_transform(wfs_split.reshape(wfs_split.shape[0], -1))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
        clusterer.fit(pcs_cluster)

        if len(np.unique(clusterer.labels_))>1:
            labels_deconv[which][clusterer.labels_ == -1] = -1
            for i in np.unique(clusterer.labels_)[2:]:
                labels_deconv[which[clusterer.labels_ == i]] = cmp 
                print(labels_deconv.max())
                cmp += 1

    return labels_deconv





def merge(labels_ptp_geq_4_after_split, templates, path_cleaned_wfs_h5, xs, z_reg, maxptps, n_chan_merge = 10, tpca = PCA(8), n_temp = 10, distance_threshold=3.0, threshold_diptest=0.25, max_spikes = 500):

    """
    labels, locations, maxptps, templates should be for spikes with ptp > 4 only
    """

    labels_updated = labels_ptp_geq_4_after_split.copy()
    n_templates = templates.shape[0]
    n_spikes_templates = merge_split_cleaned.get_n_spikes_templates(n_templates, labels_ptp_geq_4_after_split)
    x_z_templates = merge_split_cleaned.get_x_z_templates(n_templates, labels_ptp_geq_4_after_split, xs, z_reg)
    print("GET PROPOSED PAIRS")
    dist_argsort, dist_template = merge_split_cleaned.get_proposed_pairs(
        n_templates, templates, x_z_templates, n_temp
    )

    reference_units = np.setdiff1d(np.unique(labels_ptp_geq_4_after_split), [-1])

    for unit in tqdm(range(n_templates)):

        unit_reference = reference_units[unit]
        to_be_merged = [unit_reference]
        merge_shifts = [0]
        is_merged = False


        for j in range(n_temp):
            if dist_template[unit, j] < distance_threshold:
                unit_bis = dist_argsort[unit, j]
                unit_bis_reference = reference_units[unit_bis]
                if unit_reference != unit_bis_reference:
                    # ALIGN BASED ON MAX PTP TEMPLATE MC
                    if (
                        templates[unit_reference].ptp(0).max()
                        < templates[unit_bis_reference].ptp(0).max()
                    ):
                        mc = templates[unit_bis_reference].ptp(0).argmax()
                        two_units_shift = (
                            templates[unit_reference, :, mc].argmin()
                            - templates[unit_bis_reference, :, mc].argmin()
                        )
                        unit_shifted = unit_reference
                    else:
                        mc = templates[unit_reference].ptp(0).argmax()
                        two_units_shift = (
                            templates[unit_bis_reference, :, mc].argmin()
                            - templates[unit_reference, :, mc].argmin()
                        )
                        unit_shifted = unit_bis_reference



                    n_wfs_max = int(
                        min(max_spikes, min(n_spikes_templates[unit_reference], n_spikes_templates[unit_bis_reference]))
                    )
                    which = np.flatnonzero(np.logical_and(maxptps > 4, labels_updated == unit_reference))
                    idx = np.random.choice(
                        np.arange(len(which)), n_wfs_max, replace=False
                    )
                    idx.sort()

                    with h5py.File(path_cleaned_wfs_h5, "r") as h5:
                        waveforms_ref = np.empty((n_wfs_max, *h5["wfs"].shape[1:]), dtype=h5["wfs"].dtype)
                        waveforms_ref = h5["wfs"][which[idx]]

                    C = waveforms_ref.shape[2]
                    if C<n_chan_merge:
                        n_chan_merge = C

                    firstchan_maxchan = mc - firstchans[which[idx]]
                    firstchan_maxchan = np.maximum(firstchan_maxchan, n_chan_merge//2)
                    firstchan_maxchan = np.minimum(firstchan_maxchan, C - n_chan_merge//2)
                    firstchan_maxchan = firstchan_maxchan.astype('int')

                    if len(np.unique(firstchan_maxchan))<=1:
                        wfs_merge_ref = waveforms_ref[:, :, firstchan_maxchan[0]-n_chan_merge//2:firstchan_maxchan[0]+n_chan_merge//2]
                    else:
                        wfs_merge_ref = np.zeros((waveforms_ref.shape[0], waveforms_ref.shape[1], n_chan_merge))
                        for j in range(wfs_merge.shape[0]):
                            wfs_merge_ref[j] = waveforms_ref[j, :, firstchan_maxchan[j]-n_chan_merge//2:firstchan_maxchan[j]+n_chan_merge//2]


                    which = np.flatnonzero(np.logical_and(maxptps > 4, labels_updated == unit_bis_reference))

                    idx = np.random.choice(
                        np.arange(len(which)), n_wfs_max, replace=False
                    )
                    idx.sort()

                    firstchan_maxchan = mc - firstchans[which[idx]]
                    firstchan_maxchan = np.maximum(firstchan_maxchan, n_chan_merge//2)
                    firstchan_maxchan = np.minimum(firstchan_maxchan, C - n_chan_merge//2)
                    firstchan_maxchan = firstchan_maxchan.astype('int')

                    with h5py.File(path_cleaned_wfs_h5, "r") as h5:
                        waveforms_ref_bis = np.empty((n_wfs_max, *h5["wfs"].shape[1:]), dtype=h5["wfs"].dtype)
                        waveforms_ref_bis = h5["wfs"][which[idx]]


                    if len(np.unique(firstchan_maxchan))<=1:
                        wfs_merge_ref_bis = waveforms_ref_bis[:, :, firstchan_maxchan[0]-n_chans_split//2:firstchan_maxchan[0]+n_chans_split//2]
                    else:
                        wfs_merge_ref_bis = np.zeros((waveforms_ref_bis.shape[0], waveforms_ref_bis.shape[1], n_chans_split))
                        for j in range(waveforms_ref_bis.shape[0]):
                            wfs_merge_ref_bis[j] = waveforms_ref_bis[j, :, firstchan_maxchan[j]-n_chans_split//2:firstchan_maxchan[j]+n_chans_split//2]

                    if unit_shifted == unit_reference and two_units_shift>0:
                        wfs_merge_ref = wfs_merge_ref[:, two_units_shift:, :]
                        wfs_merge_ref_bis = wfs_merge_ref_bis[:, :-two_units_shift, :]
                    elif unit_shifted == unit_reference and two_units_shift<0:
                        wfs_merge_ref = wfs_merge_ref[:, :two_units_shift, :]
                        wfs_merge_ref_bis = wfs_merge_ref_bis[:, -two_units_shift:, :]
                    elif unit_shifted == unit_bis_reference and two_units_shift>0:
                        wfs_merge_ref = wfs_merge_ref[:, :-two_units_shift, :]
                        wfs_merge_ref_bis = wfs_merge_ref_bis[:, two_units_shift:, :]
                    elif unit_shifted == unit_bis_reference and two_units_shift<0:
                        wfs_merge_ref = wfs_merge_ref[:, -two_units_shift:, :]
                        wfs_merge_ref_bis = wfs_merge_ref_bis[:, :two_units_shift, :]

                    wfs_diptest = np.concatenate((wfs_merge_ref, wfs_merge_ref_bis))
                    N, T, C = wfs_diptest.shape
                    wfs_diptest = wfs_diptest.transpose(0, 2, 1).reshape(N * C, T)
                    
                    wfs_diptest = tpca.fit_transform(wfs_diptest)
                    wfs_diptest = (
                        wfs_diptest.reshape(N, C, tpca.n_components).transpose(0, 2, 1).reshape((N, C * tpca.n_components))
                    )
                    
                    labels_diptest = np.zeros(wfs_merge_ref.shape[0] + wfs_merge_ref_bis.shape[0])
                    labels_diptest[: wfs_merge_ref.shape[0]] = 1

                    lda_model = LDA(n_components=1)
                    lda_comps = lda_model.fit_transform(wfs_diptest, labels_diptest)
                    value_dpt, cut_calue = isocut(lda_comps[:, 0])

                    if (
                        value_dpt < threshold_diptest
                        and np.abs(two_units_shift) < 2
                    ):
                        to_be_merged.append(unit_bis_reference)
                        if unit_shifted == unit_bis_reference:
                            merge_shifts.append(-two_units_shift)
                        else:
                            merge_shifts.append(two_units_shift)
                        is_merged = True

            if is_merged:
                n_total_spikes = 0
                for unit_merged in np.unique(np.asarray(to_be_merged)):
                    n_total_spikes += n_spikes_templates[unit_merged]

                new_reference_unit = to_be_merged[0]

                templates[new_reference_unit] = (
                    n_spikes_templates[new_reference_unit]
                    * templates[new_reference_unit]
                    / n_total_spikes
                )
                cmp = 1
                for unit_merged in to_be_merged[1:]:
                    shift_ = merge_shifts[cmp]
                    templates[new_reference_unit] += (
                        n_spikes_templates[unit_merged]
                        * np.roll(templates[unit_merged], shift_, axis=0)
                        / n_total_spikes
                    )
                    n_spikes_templates[new_reference_unit] += n_spikes_templates[
                        unit_merged
                    ]
                    n_spikes_templates[unit_merged] = 0
                    labels_updated[
                        labels_updated == unit_merged
                    ] = new_reference_unit
                    reference_units[reference_units == unit_merged] = new_reference_unit
                    cmp += 1















