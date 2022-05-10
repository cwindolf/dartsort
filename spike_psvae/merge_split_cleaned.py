# %%
import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.signal import argrelmin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import CCA
import scipy.optimize as optim_ls
import hdbscan
from spike_psvae.cluster_utils import (
    compute_shifted_similarity,
    read_waveforms,
)
from isosplit import isocut
from scipy.spatial.distance import cdist
from tqdm import notebook
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
from spike_psvae.denoise import denoise_wf_nn_tmp_single_channel
from sklearn.cluster import MeanShift

# %%
def align_templates(
    labels, templates, triaged_spike_index, trough_offset=42, copy=False
):
    list_argmin = np.zeros(templates.shape[0])
    for i in range(templates.shape[0]):
        list_argmin[i] = templates[i, :, templates[i].ptp(0).argmax()].argmin()
    idx_not_aligned = np.where(list_argmin != trough_offset)[0]

    if copy:
        triaged_spike_index = triaged_spike_index.copy()
    for unit in idx_not_aligned:
        mc = templates[unit].ptp(0).argmax()
        offset = templates[unit, :, mc].argmin()
        triaged_spike_index[labels == unit, 0] += offset - trough_offset

    idx_sorted = triaged_spike_index[:, 0].argsort()
    triaged_spike_index = triaged_spike_index[idx_sorted]
    
    return triaged_spike_index, idx_sorted


def align_spikes_by_templates(
    labels, templates, spike_index, trough_offset=42
):
    list_argmin = np.zeros(templates.shape[0])
    template_maxchans = []
    for i in range(templates.shape[0]):
        mc = templates[i].ptp(0).argmax()
        list_argmin[i] = templates[i, :, mc].argmin()
        template_maxchans.append(mc)
    idx_not_aligned = np.where(list_argmin != trough_offset)[0]
    template_shifts = np.array(list_argmin, dtype=int) - trough_offset

    shifted_spike_index = spike_index.copy()
    for unit in idx_not_aligned:
        shift = template_shifts[unit]
        shifted_spike_index[labels == unit, 0] += shift
    return template_shifts, template_maxchans, shifted_spike_index

# %%
def run_LDA_split(wfs, max_channels, n_channels=10, n_times=121):
    ncomp = 2
    if np.unique(max_channels).shape[0] < 2:
        return np.zeros(len(max_channels), dtype=int)
    elif np.unique(max_channels).shape[0] == 2:
        # ncomp = 1 #this doesn't work with hdbscan yet
        max_channels[-1] = np.unique(max_channels)[0]-1
        max_channels[0] = np.unique(max_channels)[-1]+1
    try:
        lda_model = LDA(n_components=ncomp)
        lda_comps = lda_model.fit_transform(
            wfs.reshape((-1, n_times * n_channels)), max_channels
        )
    except np.linalg.LinAlgError:
        nmc = np.unique(max_channels).shape[0]
        print("SVD error, skipping this one. N maxchans was", nmc, "n data", len(wfs))
        return np.zeros(len(max_channels), dtype=int)
    lda_clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
    lda_clusterer.fit(lda_comps)
    return lda_clusterer.labels_


def run_CCA_split(wfs, x, z, maxptp,):
    cca = CCA(n_components=2)
    cca_embed, _ = cca.fit_transform(
        wfs.reshape(wfs.shape[0], -1),
        np.c_[tx[twhich], tz[twhich], 30 * np.log(tmaxptps[twhich])],
    )
    cca_hdb = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
    cca_hdb.fit(cca_embed)
    return cca_hdb.labels_


# %%


def split_individual_cluster(
    residual_path,
    true_mc,
    waveforms_unit,
    first_chans_unit,
    spike_index_unit,
    x_unit,
    z_unit,
    # ptps_unit,
    geom_array,
    denoiser,
    device,
    tpca,
    n_channels,
    pca_n_channels,
):
    total_channels = geom_array.shape[0]
    N, T, wf_chans = waveforms_unit.shape
    n_channels_half = n_channels // 2

    labels_unit = np.full(spike_index_unit.shape[0], -1)
    is_split = False
    
    # get waveforms on n_channels chans around the template max chan
    # TODO: residual is not being read on the same chans.
    low = np.maximum(
        0,
        (true_mc - n_channels_half) - first_chans_unit
    )
    low = np.minimum(low, wf_chans - n_channels)
    chan_ix = np.arange(n_channels)
    wfs_unit = waveforms_unit[
        np.arange(len(waveforms_unit))[:, None, None],
        np.arange(T)[None, :, None],
        low[:, None, None] + chan_ix[None, None, :]
    ]

    mc = max(n_channels_half, true_mc)
    mc = min(total_channels - n_channels_half, mc)
    assert mc - n_channels_half >= 0
    assert mc + n_channels_half <= total_channels

    # wfs_unit = waveforms_unit[:, :, mc - n_channels_half : mc + n_channels_half]
    # wfs_unit = np.zeros(
    #     (waveforms_unit.shape[0], waveforms_unit.shape[1], n_channels)
    # )
    # for i in range(wfs_unit.shape[0]):
    #     if mc == n_channels_half:
    #         wfs_unit[i] = waveforms_unit[i, :, :n_channels]
    #     elif mc == total_channels - n_channels_half:
    #         wfs_unit[i] = waveforms_unit[
    #             i, :, waveforms_unit.shape[2] - n_channels :
    #         ]
    #     else:
    #         mc_new = int(mc - first_chans_unit[i])
    #         wfs_unit[i] = waveforms_unit[
    #             i, :, mc_new - n_channels_half : mc_new + n_channels_half
    #         ]
    
    readwfs, skipped = read_waveforms(
        spike_index_unit[:, 0],
        residual_path,
        geom_array,
        n_times=T,
        channels=np.arange(mc - n_channels_half, mc + n_channels_half),
    )
    wfs_unit += readwfs
    wfs_unit_denoised = denoise_wf_nn_tmp_single_channel(
        wfs_unit, denoiser, device
    )
    
    if true_mc < n_channels_half:
        true_mc = true_mc
    elif true_mc > total_channels - n_channels_half:
        true_mc = true_mc - (total_channels - n_channels)
    else:
        true_mc = n_channels_half
        
    #get tpca of wfs
    permuted_wfs_unit_denoised = wfs_unit_denoised.transpose(0, 2, 1)
    tpca_wf_units = tpca.transform(permuted_wfs_unit_denoised.reshape(permuted_wfs_unit_denoised.shape[0]*permuted_wfs_unit_denoised.shape[1], -1))
    tpca_wfs_inverse = tpca.inverse_transform(tpca_wf_units)
    tpca_wfs_inverse = tpca_wfs_inverse.reshape(permuted_wfs_unit_denoised.shape[0], permuted_wfs_unit_denoised.shape[1], -1).transpose(0, 2, 1)
    tpca_wf_units = tpca_wf_units.reshape(permuted_wfs_unit_denoised.shape[0], permuted_wfs_unit_denoised.shape[1], -1).transpose(0, 2, 1)
    
    #get waveforms on max channel and max ptps
    wf_units_mc = wfs_unit_denoised[:, :, true_mc]
    ptps_unit = wf_units_mc.ptp(1)
    
    #get tpca on pca_n_channels
    channels_pca_before = true_mc-pca_n_channels//2
    channels_pca_after = true_mc+pca_n_channels//2
    if channels_pca_before < 0:
        channels_pca_after = channels_pca_after + (-channels_pca_before)
        channels_pca_before = 0
    elif channels_pca_after > n_channels:
        channels_pca_before = channels_pca_before + (n_channels-channels_pca_after)
        channels_pca_after = n_channels
    tpca_wf_units_mcs = tpca_wf_units[:, :, channels_pca_before:channels_pca_after]

    tpca_wf_units_mcs = tpca_wf_units_mcs.transpose(0, 2, 1)
    tpca_wf_units_mcs = tpca_wf_units_mcs.reshape(tpca_wf_units_mcs.shape[0], tpca_wf_units_mcs.shape[1]*tpca_wf_units_mcs.shape[2])

    pca_model = PCA(2)
    
    pcs = pca_model.fit_transform(tpca_wf_units_mcs)
    
    alpha1 = (x_unit.max() - x_unit.min()) / (
        pcs[:, 0].max() - pcs[:, 0].min()
    )
    alpha2 = (x_unit.max() - x_unit.min()) / (
        pcs[:, 1].max() - pcs[:, 1].min()
    )
    features = np.concatenate(
        (
            np.expand_dims(x_unit, 1),
            np.expand_dims(z_unit, 1),
            np.expand_dims(pcs[:, 0], 1) * alpha1,
            np.expand_dims(pcs[:, 1], 1) * alpha2,
            np.expand_dims(np.log(ptps_unit) * 30, 1),
        ),
        axis=1,
    )  # Use scales parameter
    clusterer_herding = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
    clusterer_herding.fit(features)
    
    max_channels_all = wfs_unit_denoised.ptp(1).argmax(1)
    labels_rec_hdbscan = clusterer_herding.labels_
    if len(np.unique(labels_unit)) > 1:
        is_split = True
    else:
        is_split = False
    if np.unique(labels_rec_hdbscan).shape[0] > 1:
        is_split = True
    if is_split:
        labels_unit[labels_rec_hdbscan == -1] = -1
        label_max_temp = labels_rec_hdbscan.max()
        cmp = 0
        for new_unit_id in np.unique(labels_rec_hdbscan)[1:]:
            tpca_wfs_new_unit = tpca_wf_units[labels_rec_hdbscan == new_unit_id]
            #overwrite max_channels
            max_channels = wfs_unit_denoised[labels_rec_hdbscan == new_unit_id].ptp(1).argmax(1)
            
            # LinAlgError
            lda_labels = run_LDA_split(tpca_wfs_new_unit, max_channels, n_times=8, n_channels=n_channels)
            if np.unique(lda_labels).shape[0] == 1:
                labels_unit[labels_rec_hdbscan == new_unit_id] = cmp
                cmp += 1
            else:
                for lda_unit in np.unique(lda_labels):
                    if lda_unit >= 0:
                        labels_unit[
                            np.flatnonzero(labels_rec_hdbscan == new_unit_id)[
                                lda_labels == lda_unit
                            ]
                        ] = cmp
                        cmp += 1
                    else:
                        labels_unit[
                            np.flatnonzero(labels_rec_hdbscan == new_unit_id)[
                                lda_labels == lda_unit
                            ]
                        ] = -1
    else:
        lda_labels = run_LDA_split(tpca_wf_units, max_channels_all, n_times=tpca_wf_units.shape[1], n_channels=n_channels)
        if np.unique(lda_labels).shape[0] > 1:
            is_split = True
            labels_unit = lda_labels
    print("split", is_split, np.unique(labels_unit), len(np.unique(labels_unit)[1:]))
    return is_split, labels_unit


def load_aligned_waveforms(
    waveforms, labels, unit, template_shift, indices=None
):
    if indices is None:
        indices = np.flatnonzero(labels == unit)

    # this is an hdf5 workaround, loading lots of data is slow sometimes
    waveforms_unit = np.empty(
        (len(indices), *waveforms.shape[1:]), dtype=waveforms.dtype
    )
    for s in range(0, len(indices), 1000):
        e = min(len(indices), s + 1000)
        waveforms_unit[s:e] = waveforms[indices[s:e]]

    # roll by the template shift
    # template shift = template argmin - original argmin
    # so, if it's positive, pad on the right and chop on the left
    if template_shift != 0:
        waveforms_unit = np.pad(
            waveforms_unit,
            [
                (0, 0),
                (max(0, -template_shift), max(0, template_shift)),
                (0, 0),
            ],
            mode="edge",
        )
        if template_shift > 0:
            waveforms_unit = waveforms_unit[:, template_shift:, :]
        else:
            waveforms_unit = waveforms_unit[:, :template_shift, :]

    return waveforms_unit


# %%
def split_clusters(
    residual_path,
    waveforms,
    first_chans,
    spike_index,
    template_maxchans,
    template_shifts,
    labels,
    x,
    z,
    geom_array,
    denoiser,
    device,
    tpca,
    n_channels=10,
    pca_n_channels=4,
):
    labels_new = labels.copy()
    labels_original = labels.copy()
    cur_max_label = labels.max()
    for unit in tqdm(np.setdiff1d(np.unique(labels), [-1])):
        print(f"splitting unit {unit}")
        in_unit = np.flatnonzero(labels == unit)
        spike_index_unit = spike_index[in_unit]
        template_shift = template_shifts[unit]
        waveforms_unit = load_aligned_waveforms(
            waveforms, labels, unit, template_shift
        )

        first_chans_unit = first_chans[in_unit]
        # x_unit, z_unit, ptps_unit = x[in_unit], z[in_unit], ptps[in_unit]
        x_unit, z_unit = x[in_unit], z[in_unit]
        is_split, unit_new_labels = split_individual_cluster(
            residual_path,
            template_maxchans[unit],
            waveforms_unit,
            first_chans_unit,
            spike_index_unit,
            x_unit,
            z_unit,
            # ptps_unit,
            geom_array,
            denoiser,
            device,
            tpca,
            n_channels,
            pca_n_channels,
        )
        if is_split:
            for new_label in np.unique(unit_new_labels):
                if new_label == -1:
                    idx = np.flatnonzero(labels_original == unit)[
                        unit_new_labels == new_label
                    ]
                    labels_new[idx] = new_label
                elif new_label > 0:
                    cur_max_label += 1
                    idx = np.flatnonzero(labels_original == unit)[
                        unit_new_labels == new_label
                    ]
                    labels_new[idx] = cur_max_label
    return labels_new


# %%
def get_x_z_templates(n_templates, labels, x, z):
    x_z_templates = np.zeros((n_templates, 2))
    for i in range(n_templates):
        x_z_templates[i, 1] = np.median(z[labels == i])
        x_z_templates[i, 0] = np.median(x[labels == i])
    return x_z_templates


# %%
def get_n_spikes_templates(n_templates, labels):
    n_spikes_templates = np.zeros(n_templates)
    for i in range(n_templates):
        n_spikes_templates[i] = (labels == i).sum()
    return n_spikes_templates


# %%
def get_templates(
    standardized_path,
    geom_array,
    n_templates,
    spike_index,
    labels,
    max_spikes=250,
    n_times=121,
):
    templates = np.zeros((n_templates, n_times, geom_array.shape[0]))
    for unit in trange(n_templates, desc="get templates"):
        spike_times_unit = spike_index[labels == unit, 0]
        if spike_times_unit.shape[0] > max_spikes:
            idx = np.random.choice(
                np.arange(spike_times_unit.shape[0]), max_spikes, replace=False
            )
        else:
            idx = np.arange(spike_times_unit.shape[0])

        wfs_unit = read_waveforms(
            spike_times_unit[idx],
            standardized_path,
            geom_array,
            n_times=n_times,
        )[0]
        templates[unit] = wfs_unit.mean(0)
    return templates


# %%
def get_proposed_pairs(
    n_templates, templates, x_z_templates, n_temp=20, n_channels=10
):
    n_channels_half = n_channels // 2
    dist = cdist(x_z_templates, x_z_templates)
    dist_argsort = dist.argsort(axis=1)[:, 1 : n_temp + 1]
    dist_template = np.zeros((dist_argsort.shape[0], n_temp))
    for i in range(n_templates):
        mc = min(templates[i].ptp(0).argmax(), 384 - n_channels_half)
        mc = max(mc, n_channels_half)
        temp_a = templates[i, :, mc - n_channels_half : mc + n_channels_half]
        for j in range(n_temp):
            temp_b = templates[
                dist_argsort[i, j],
                :,
                mc - n_channels_half : mc + n_channels_half,
            ]
            dist_template[i, j] = compute_shifted_similarity(temp_a, temp_b)[0]
    return dist_argsort, dist_template


# %%
def get_diptest_value(
    residual_path,
    waveforms,
    template_shifts,
    first_chans,
    geom_array,
    spike_index,
    labels,
    unit_a,
    unit_b,
    n_spikes_templates,
    mc,
    two_units_shift,
    unit_shifted,
    denoiser,
    device,
    tpca,
    n_channels=10,
    n_times=121,
    nn_denoise=False,
    max_spikes=250,
):
    # ALIGN BASED ON MAX PTP TEMPLATE MC
    n_channels_half = n_channels // 2

    n_wfs_max = int(
        min(max_spikes, min(n_spikes_templates[unit_a], n_spikes_templates[unit_b]))
    )

    mc = min(384 - n_channels_half, mc)
    mc = max(n_channels_half, mc)

    spike_times_unit_a = spike_index[labels == unit_a, 0]
    idx = np.random.choice(
        np.arange(spike_times_unit_a.shape[0]), n_wfs_max, replace=False
    )
    # print(spike_times_unit_a.shape)
    idx.sort()
    spike_times_unit_a = spike_times_unit_a[idx]
    wfs_a = load_aligned_waveforms(
        waveforms,
        labels,
        unit_a,
        template_shifts[unit_a],
        indices=np.flatnonzero(labels == unit_a)[idx],
    )
    first_chan_a = first_chans[labels == unit_a][idx]

    spike_times_unit_b = spike_index[labels == unit_b, 0]
    # print(spike_times_unit_b.shape)
    idx = np.random.choice(
        np.arange(spike_times_unit_b.shape[0]), n_wfs_max, replace=False
    )
    idx.sort()
    spike_times_unit_b = spike_times_unit_b[idx]
    wfs_b = load_aligned_waveforms(
        waveforms,
        labels,
        unit_b,
        template_shifts[unit_b],
        indices=np.flatnonzero(labels == unit_b)[idx],
    )
    first_chan_b = first_chans[labels == unit_b][idx]

    wfs_a_bis = np.zeros((wfs_a.shape[0], n_times, n_channels))
    wfs_b_bis = np.zeros((wfs_b.shape[0], n_times, n_channels))

    if two_units_shift > 0:

        if unit_shifted == unit_a:
            for i in range(wfs_a_bis.shape[0]):
                first_chan = int(mc - first_chan_a[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_a.shape[2] - n_channels, int(first_chan))
                wfs_a_bis[i, :-two_units_shift] = wfs_a[
                    i, two_units_shift:, first_chan : first_chan + n_channels
                ]
                first_chan = int(mc - first_chan_b[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_b.shape[2] - n_channels, int(first_chan))
                wfs_b_bis[i, :] = wfs_b[
                    i, :, first_chan : first_chan + n_channels
                ]
            wfs_a_bis += read_waveforms(
                spike_times_unit_a + two_units_shift,
                residual_path,
                geom_array,
                n_times=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )[0]
            wfs_b_bis += read_waveforms(
                spike_times_unit_b,
                residual_path,
                geom_array,
                n_times=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )[0]
        else:
            for i in range(wfs_a_bis.shape[0]):
                first_chan = int(mc - first_chan_a[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_a.shape[2] - n_channels, int(first_chan))
                wfs_a_bis[i] = wfs_a[
                    i, :, first_chan : first_chan + n_channels
                ]
                first_chan = int(mc - first_chan_b[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_b.shape[2] - n_channels, int(first_chan))
                wfs_b_bis[i, :-two_units_shift] = wfs_b[
                    i, two_units_shift:, first_chan : first_chan + n_channels
                ]
            wfs_a_bis += read_waveforms(
                spike_times_unit_a,
                residual_path,
                geom_array,
                n_times=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )[0]
            wfs_b_bis += read_waveforms(
                spike_times_unit_b + two_units_shift,
                residual_path,
                geom_array,
                n_times=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )[0]
    elif two_units_shift < 0:
        if unit_shifted == unit_a:
            for i in range(wfs_a_bis.shape[0]):
                first_chan = int(mc - first_chan_a[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_a.shape[2] - n_channels, int(first_chan))
                wfs_a_bis[i, -two_units_shift:] = wfs_a[
                    i, :two_units_shift, first_chan : first_chan + n_channels
                ]
                first_chan = int(mc - first_chan_b[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_b.shape[2] - n_channels, int(first_chan))
                wfs_b_bis[i, :] = wfs_b[
                    i, :, first_chan : first_chan + n_channels
                ]
            wfs_a_bis += read_waveforms(
                spike_times_unit_a + two_units_shift,
                residual_path,
                geom_array,
                n_times=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )[0]
            wfs_b_bis += read_waveforms(
                spike_times_unit_b,
                residual_path,
                geom_array,
                n_times=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )[0]

        else:
            for i in range(wfs_a_bis.shape[0]):
                first_chan = int(mc - first_chan_a[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_a.shape[2] - n_channels, int(first_chan))
                wfs_a_bis[i] = wfs_a[
                    i, :, first_chan : first_chan + n_channels
                ]
                first_chan = int(mc - first_chan_b[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_b.shape[2] - n_channels, int(first_chan))
                wfs_b_bis[i, -two_units_shift:] = wfs_b[
                    i, :two_units_shift, first_chan : first_chan + n_channels
                ]
            wfs_a_bis += read_waveforms(
                spike_times_unit_a,
                residual_path,
                geom_array,
                n_times=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )[0]
            wfs_b_bis += read_waveforms(
                spike_times_unit_b + two_units_shift,
                residual_path,
                geom_array,
                n_times=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )[0]
    else:
        for i in range(wfs_a_bis.shape[0]):
            first_chan = int(mc - first_chan_a[i] - n_channels_half)
            first_chan = max(0, int(first_chan))
            first_chan = min(wfs_a.shape[2] - n_channels, int(first_chan))
            wfs_a_bis[i] = wfs_a[i, :, first_chan : first_chan + n_channels]
            first_chan = int(mc - first_chan_b[i] - n_channels_half)
            first_chan = max(0, int(first_chan))
            first_chan = min(wfs_b.shape[2] - n_channels, int(first_chan))
            wfs_b_bis[i, :] = wfs_b[i, :, first_chan : first_chan + n_channels]
        wfs_a_bis += read_waveforms(
            spike_times_unit_a,
            residual_path,
            geom_array,
            n_times=n_times,
            channels=np.arange(mc - n_channels_half, mc + n_channels_half),
        )[0]
        wfs_b_bis += read_waveforms(
            spike_times_unit_b,
            residual_path,
            geom_array,
            n_times=n_times,
            channels=np.arange(mc - n_channels_half, mc + n_channels_half),
        )[0]
    
    # tpca = PCA(rank_pca)
    wfs_diptest = np.concatenate((wfs_a, wfs_b))

    if nn_denoise:
        wfs_diptest = denoise_wf_nn_tmp_single_channel(
            wfs_diptest, denoiser, device
        )
    # print(wfs_diptest.shape)
    N, T, C = wfs_diptest.shape
    wfs_diptest = wfs_diptest.transpose(0, 2, 1).reshape(N * C, T)
    # wfs_diptest = tpca.inverse_transform(tpca.fit_transform(wfs_diptest))
    # wfs_diptest = (
    #     wfs_diptest.reshape(N, C, T).transpose(0, 2, 1).reshape((N, C * T))
    # )
    wfs_diptest = tpca.fit_transform(wfs_diptest)
    wfs_diptest = (
        wfs_diptest.reshape(N, C, tpca.n_components).transpose(0, 2, 1).reshape((N, C * tpca.n_components))
    )
    labels_diptest = np.zeros(wfs_a_bis.shape[0] + wfs_b_bis.shape[0])
    labels_diptest[: wfs_a_bis.shape[0]] = 1

    lda_model = LDA(n_components=1)
    lda_comps = lda_model.fit_transform(wfs_diptest, labels_diptest)
    value_dpt, cut_calue = isocut(lda_comps[:, 0])
    return value_dpt


# %%
def get_merged(
    residual_path,
    waveforms,
    first_chans,
    geom_array,
    templates,
    template_shifts,
    n_templates,
    spike_index,
    labels,
    x,
    z,
    denoiser,
    device,
    tpca,
    n_channels=10,
    n_temp=10,
    distance_threshold=3.0,
    threshold_diptest=0.75,
    nn_denoise=False,
):
    n_spikes_templates = get_n_spikes_templates(n_templates, labels)
    x_z_templates = get_x_z_templates(n_templates, labels, x, z)
    print("GET PROPOSED PAIRS")
    dist_argsort, dist_template = get_proposed_pairs(
        n_templates, templates, x_z_templates, n_temp=n_temp
    )

    labels_updated = labels.copy()
    reference_units = np.setdiff1d(np.unique(labels), [-1])

    for unit in tqdm(range(n_templates)):  # tqdm
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
                    dpt_val = get_diptest_value(
                        residual_path,
                        waveforms,
                        template_shifts,
                        first_chans,
                        geom_array,
                        spike_index,
                        labels_updated,
                        unit_reference,
                        unit_bis_reference,
                        n_spikes_templates,
                        mc,
                        two_units_shift,
                        unit_shifted,
                        denoiser,
                        device,
                        tpca,
                        n_channels,
                        nn_denoise=nn_denoise,
                    )
                    if (
                        dpt_val < threshold_diptest
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
    return labels_updated
