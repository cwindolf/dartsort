import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import hdbscan
from spike_psvae.cluster_utils import (
    compute_shifted_similarity,
    read_waveforms,
)
from spike_psvae.isocut5 import isocut5 as isocut
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm, trange
from spike_psvae.denoise import denoise_wf_nn_tmp_single_channel, SingleChanDenoiser
from spike_psvae import waveform_utils
from spike_psvae.pyks_ccg import ccg_metrics


# %%
def align_spikes_by_templates(
    labels,
    templates,
    spike_index,
    trough_offset=42,
    shift_max=2,
):
    spike_time_offsets = np.zeros(templates.shape[0])
    template_maxchans = []
    for i in range(templates.shape[0]):
        mc = templates[i].ptp(0).argmax()
        spike_time_offsets[i] = np.abs(templates[i, :, mc]).argmax()
        template_maxchans.append(mc)
    idx_not_aligned = np.where(spike_time_offsets != trough_offset)[0]
    template_shifts = np.array(spike_time_offsets, dtype=int) - trough_offset

    shifted_spike_index = spike_index.copy()
    for unit in idx_not_aligned:
        shift = template_shifts[unit]
        if abs(shift) <= shift_max:
            shifted_spike_index[labels == unit, 0] -= shift
        else:
            # zero out overly large shifts (denoiser issue)
            template_shifts[unit] = 0
    return (
        template_shifts,
        template_maxchans,
        shifted_spike_index,
        idx_not_aligned,
    )


# %%
def run_LDA_split(wfs, max_channels, threshold_diptest=1.0):
    ncomp = 2
    if np.unique(max_channels).shape[0] < 2:
        return np.zeros(len(max_channels), dtype=int)
    elif np.unique(max_channels).shape[0] == 2:
        ncomp = 1
    try:
        lda_model = LDA(n_components=ncomp)
        lda_comps = lda_model.fit_transform(
            wfs.reshape((-1, wfs.shape[1] * wfs.shape[2])), max_channels
        )
    except np.linalg.LinAlgError:
        nmc = np.unique(max_channels).shape[0]
        print(
            "SVD error, skipping this one. N maxchans was",
            nmc,
            "n data",
            len(wfs),
        )
        return np.zeros(len(max_channels), dtype=int)
    except ValueError as e:
        print(
            "Some ValueError during LDA split. Ignoring it and not splitting."
            f"Here is the error message though: {e}"
        )
        return np.zeros(len(max_channels), dtype=int)

    if ncomp == 2:
        lda_clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
        lda_clusterer.fit(lda_comps)
        labels = lda_clusterer.labels_
    else:
        value_dpt, cut_value = isocut(lda_comps[:, 0])
        if value_dpt < threshold_diptest:
            labels = np.zeros(len(max_channels), dtype=int)
        else:
            labels = np.zeros(len(max_channels), dtype=int)
            labels[np.where(lda_comps[:, 0] > cut_value)] = 1

    return labels


# %%
def split_individual_cluster(
    residual_path,
    true_mc,
    waveforms_unit,
    first_chans_unit,
    spike_index_unit,
    x_unit,
    z_unit,
    geom_array,
    denoiser,
    device,
    tpca,
    n_channels,
    pca_n_channels,
    nn_denoise,
    threshold_diptest=1.0,
    min_size_split=25,
):
    total_channels = geom_array.shape[0]
    N, T, wf_chans = waveforms_unit.shape
    n_channels_half = n_channels // 2

    labels_unit = np.full(N, -1)
    is_split = False

    if N < min_size_split:
        return is_split, labels_unit * 0

    mc = max(n_channels_half, true_mc)
    mc = min(total_channels - n_channels_half, mc)
    assert mc - n_channels_half >= 0
    assert mc + n_channels_half <= total_channels

    # get n_channels worth of residuals
    wfs_unit, skipped = read_waveforms(
        spike_index_unit[:, 0],
        residual_path,
        geom_array.shape[0],
        spike_length_samples=T,
        channels=np.arange(mc - n_channels_half, mc + n_channels_half),
    )
    kept = np.setdiff1d(np.arange(N), skipped)
    
    # add in raw waveforms
    # get n_channels of waveforms for each unit with the unit max channel and the firstchan for each spike
    for i, j in enumerate(kept):
        mc_new = int(mc - first_chans_unit[i])
        if mc_new <= n_channels_half:
            wfs_unit[i] += waveforms_unit[j, :, :n_channels]
        elif mc_new >= waveforms_unit.shape[2] - n_channels_half:
            wfs_unit[i] += waveforms_unit[
                j, :, waveforms_unit.shape[2] - n_channels :
            ]
        else:
            wfs_unit[i] += waveforms_unit[
                j, :, mc_new - n_channels_half : mc_new + n_channels_half
            ]

    # denoise optional (False by default)
    if nn_denoise:
        wfs_unit = denoise_wf_nn_tmp_single_channel(wfs_unit, denoiser, device)

    # get true_mc for each spike (only different than mc for edge spikes)
    if true_mc < n_channels_half:
        true_mc = true_mc
    elif true_mc > total_channels - n_channels_half:
        true_mc = true_mc - (total_channels - n_channels)
    else:
        true_mc = n_channels_half

    # get tpca of wfs using pre-trained tpca
    permuted_wfs_unit = wfs_unit.transpose(0, 2, 1)
    tpca_wf_units = tpca.transform(
        permuted_wfs_unit.reshape(
            permuted_wfs_unit.shape[0] * permuted_wfs_unit.shape[1], -1
        )
    )
    tpca_wfs_inverse = tpca.inverse_transform(tpca_wf_units)
    tpca_wfs_inverse = tpca_wfs_inverse.reshape(
        permuted_wfs_unit.shape[0], permuted_wfs_unit.shape[1], -1
    ).transpose(0, 2, 1)
    tpca_wf_units = tpca_wf_units.reshape(
        permuted_wfs_unit.shape[0], permuted_wfs_unit.shape[1], -1
    ).transpose(0, 2, 1)

    # get waveforms on max channel and max ptps
    wf_units_mc = wfs_unit[:, :, true_mc]
    ptps_unit = wf_units_mc.ptp(1)

    # get tpca embeddings for pca_n_channels (edges handled differently)
    channels_pca_before = true_mc - pca_n_channels // 2
    channels_pca_after = true_mc + pca_n_channels // 2
    if channels_pca_before < 0:
        channels_pca_after = channels_pca_after + (-channels_pca_before)
        channels_pca_before = 0
    elif channels_pca_after > n_channels:
        channels_pca_before = channels_pca_before + (
            n_channels - channels_pca_after
        )
        channels_pca_after = n_channels
    tpca_wf_units_mcs = tpca_wf_units[
        :, :, channels_pca_before:channels_pca_after
    ]
    tpca_wf_units_mcs = tpca_wf_units_mcs.transpose(0, 2, 1)
    tpca_wf_units_mcs = tpca_wf_units_mcs.reshape(
        tpca_wf_units_mcs.shape[0],
        tpca_wf_units_mcs.shape[1] * tpca_wf_units_mcs.shape[2],
    )

    # get 2D pc embedding of tpca embeddings
    pca_model = PCA(2)
    try:
        pcs = pca_model.fit_transform(tpca_wf_units_mcs)
    except ValueError:
        print("ERR", tpca_wf_units_mcs.shape, flush=True)
        raise

    # scale pc embeddings to X feature
    alpha1 = (x_unit.max() - x_unit.min()) / (
        pcs[:, 0].max() - pcs[:, 0].min()
    )
    alpha2 = (x_unit.max() - x_unit.min()) / (
        pcs[:, 1].max() - pcs[:, 1].min()
    )

    # create 5D feature set for clustering (herdingspikes)
    features = np.concatenate(
        (
            np.expand_dims(x_unit[kept], 1),
            np.expand_dims(z_unit[kept], 1),
            np.expand_dims(pcs[:, 0], 1) * alpha1,
            np.expand_dims(pcs[:, 1], 1) * alpha2,
            np.expand_dims(np.log(ptps_unit) * 30, 1),
        ),
        axis=1,
    )  # Use scales parameter

    # cluster using herding spikes (parameters could be adjusted)
    clusterer_herding = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
    clusterer_herding.fit(features)
    labels_rec_hdbscan = clusterer_herding.labels_
    # check if cluster split by herdingspikes clustering
    # print("herding", np.unique(labels_rec_hdbscan, return_counts=True))
    if np.unique(labels_rec_hdbscan).shape[0] > 1:
        is_split = True

    # LDA split - split by clustering LDA embeddings: X,y = wfs,max_channels
    max_channels_all = wfs_unit.ptp(1).argmax(1)
    if is_split:
        # split by herdingspikes, run LDA split on new clusters.
        labels_unit[kept[labels_rec_hdbscan == -1]] = -1
        cmp = 0
        for new_unit_id in np.unique(labels_rec_hdbscan)[1:]:
            in_new_unit = np.flatnonzero(labels_rec_hdbscan == new_unit_id)
            tpca_wfs_new_unit = tpca_wf_units[in_new_unit]

            # get max_channels for new unit
            max_channels = wfs_unit[in_new_unit].ptp(1).argmax(1)

            # lda split
            lda_labels = run_LDA_split(
                tpca_wfs_new_unit, max_channels, threshold_diptest
            )
            # print("new unit", new_unit_id, "lda", np.unique(lda_labels))
            if np.unique(lda_labels).shape[0] == 1:
                labels_unit[kept[in_new_unit]] = cmp
                cmp += 1
            else:
                for lda_unit in np.unique(lda_labels):
                    if lda_unit >= 0:
                        labels_unit[kept[in_new_unit[lda_labels == lda_unit]]] = cmp
                        cmp += 1
                    else:
                        labels_unit[kept[in_new_unit[lda_labels == lda_unit]]] = -1
    else:
        # not split by herdingspikes, run LDA split.
        lda_labels = run_LDA_split(
            tpca_wf_units, max_channels_all, threshold_diptest
        )
        if np.unique(lda_labels).shape[0] > 1:
            is_split = True
            labels_unit = lda_labels

    return is_split, labels_unit


# %%
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
    nn_denoise=False,
    threshold_diptest=1.0,
):
    labels_new = labels.copy()
    next_label = labels.max() + 1
    for unit in tqdm(np.setdiff1d(np.unique(labels), [-1])):  # 216
        # for unit in [412]:
        # print(f"splitting unit {unit}")
        in_unit = np.flatnonzero(labels == unit)
        spike_index_unit = spike_index[in_unit]
        template_shift = template_shifts[unit]
        waveforms_unit = load_aligned_waveforms(
            waveforms, labels, unit, template_shift
        )
        # print(f"{in_unit.shape=} {spike_index_unit.shape=} {waveforms_unit.shape=}")
        # print("max channels unit", np.unique(waveforms_unit.ptp(1).argmax(1)))

        first_chans_unit = first_chans[in_unit]
        x_unit, z_unit = x[in_unit], z[in_unit]
        is_split, unit_new_labels = split_individual_cluster(
            residual_path,
            template_maxchans[unit],
            waveforms_unit,
            first_chans_unit,
            spike_index_unit,
            x_unit,
            z_unit,
            geom_array,
            denoiser,
            device,
            tpca,
            n_channels,
            pca_n_channels,
            nn_denoise,
            threshold_diptest,
        )
        # print(f"{in_unit.shape=} {unit_new_labels.shape=}")
        # print("final", is_split)
        if is_split:
            for new_label in np.unique(unit_new_labels):
                if new_label == -1:
                    idx = in_unit[unit_new_labels == new_label]
                    labels_new[idx] = -1
                elif new_label > 0:
                    idx = in_unit[unit_new_labels == new_label]
                    labels_new[idx] = next_label
                    next_label += 1
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
    n_spikes_templates = np.zeros(n_templates, dtype=int)
    unique, count = np.unique(labels, return_counts=True)
    n_spikes_templates[unique] = count
    return n_spikes_templates


# %%
def get_templates(
    standardized_path,
    geom_array,
    n_templates,
    spike_index,
    labels,
    max_spikes=250,
    spike_length_samples=121,
    reducer=np.median,
):
    templates = np.zeros(
        (n_templates, spike_length_samples, geom_array.shape[0])
    )
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
            geom_array.shape[0],
            spike_length_samples=spike_length_samples,
        )[0]
        templates[unit] = reducer(wfs_unit, axis=0)
    return templates


# %%
def get_proposed_pairs(
    n_templates, templates, x_z_templates, n_temp=20, n_channels=10, shifts=[0]
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
            dist_template[i, j], best_shift = compute_shifted_similarity(
                temp_a, temp_b, shifts=shifts
            )
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
    tpca_rank=5,
    n_channels=10,
    n_times=121,
    nn_denoise=False,
    max_spikes=250,
):
    tpca_rank = (
        tpca.n_components
        if tpca_rank is None
        else min(tpca.n_components, tpca_rank)
    )
    # ALIGN BASED ON MAX PTP TEMPLATE MC
    n_channels_half = n_channels // 2

    n_wfs_max = int(
        min(
            max_spikes,
            min(n_spikes_templates[unit_a], n_spikes_templates[unit_b]),
        )
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

            wfs_a_read, skipped_idx = read_waveforms(
                spike_times_unit_a + two_units_shift,
                residual_path,
                geom_array.shape[0],
                spike_length_samples=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )
            kept_idx = np.delete(np.arange(wfs_a_bis.shape[0]), skipped_idx)
            wfs_a_bis = wfs_a_bis[kept_idx] + wfs_a_read

            wfs_b_read, skipped_idx = read_waveforms(
                spike_times_unit_b,
                residual_path,
                geom_array.shape[0],
                spike_length_samples=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )
            kept_idx = np.delete(np.arange(wfs_b_bis.shape[0]), skipped_idx)
            wfs_b_bis = wfs_b_bis[kept_idx] + wfs_b_read

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
            wfs_a_read, skipped_idx = read_waveforms(
                spike_times_unit_a,
                residual_path,
                geom_array.shape[0],
                spike_length_samples=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )
            kept_idx = np.delete(np.arange(wfs_a_bis.shape[0]), skipped_idx)
            wfs_a_bis = wfs_a_bis[kept_idx] + wfs_a_read

            wfs_b_read, skipped_idx = read_waveforms(
                spike_times_unit_b + two_units_shift,
                residual_path,
                geom_array.shape[0],
                spike_length_samples=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )
            kept_idx = np.delete(np.arange(wfs_b_bis.shape[0]), skipped_idx)
            wfs_b_bis = wfs_b_bis[kept_idx] + wfs_b_read
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

            wfs_a_read, skipped_idx = read_waveforms(
                spike_times_unit_a + two_units_shift,
                residual_path,
                geom_array.shape[0],
                spike_length_samples=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )
            kept_idx = np.delete(np.arange(wfs_a_bis.shape[0]), skipped_idx)
            wfs_a_bis = wfs_a_bis[kept_idx] + wfs_a_read

            wfs_b_read, skipped_idx = read_waveforms(
                spike_times_unit_b,
                residual_path,
                geom_array.shape[0],
                spike_length_samples=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )
            kept_idx = np.delete(np.arange(wfs_b_bis.shape[0]), skipped_idx)
            wfs_b_bis = wfs_b_bis[kept_idx] + wfs_b_read

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
            wfs_a_read, skipped_idx = read_waveforms(
                spike_times_unit_a,
                residual_path,
                geom_array.shape[0],
                spike_length_samples=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )
            kept_idx = np.delete(np.arange(wfs_a_bis.shape[0]), skipped_idx)
            wfs_a_bis = wfs_a_bis[kept_idx] + wfs_a_read

            wfs_b_read, skipped_idx = read_waveforms(
                spike_times_unit_b + two_units_shift,
                residual_path,
                geom_array.shape[0],
                spike_length_samples=n_times,
                channels=np.arange(mc - n_channels_half, mc + n_channels_half),
            )
            kept_idx = np.delete(np.arange(wfs_b_bis.shape[0]), skipped_idx)
            wfs_b_bis = wfs_b_bis[kept_idx] + wfs_b_read
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
        wfs_a_read, skipped_idx = read_waveforms(
            spike_times_unit_a,
            residual_path,
            geom_array.shape[0],
            spike_length_samples=n_times,
            channels=np.arange(mc - n_channels_half, mc + n_channels_half),
        )
        kept_idx = np.delete(np.arange(wfs_a_bis.shape[0]), skipped_idx)
        wfs_a_bis = wfs_a_bis[kept_idx] + wfs_a_read

        wfs_b_read, skipped_idx = read_waveforms(
            spike_times_unit_b,
            residual_path,
            geom_array.shape[0],
            spike_length_samples=n_times,
            channels=np.arange(mc - n_channels_half, mc + n_channels_half),
        )
        kept_idx = np.delete(np.arange(wfs_b_bis.shape[0]), skipped_idx)
        wfs_b_bis = wfs_b_bis[kept_idx] + wfs_b_read

    # tpca = PCA(rank_pca)
    wfs_diptest = np.concatenate((wfs_a_bis, wfs_b_bis))

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
    wfs_diptest = tpca.fit_transform(wfs_diptest)[:, :tpca_rank]
    wfs_diptest = (
        wfs_diptest.reshape(N, C, tpca.n_components)
        .transpose(0, 2, 1)
        .reshape((N, C * tpca.n_components))
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
    threshold_diptest=1.0,
    nn_denoise=False,
    isi_veto=False,
    contam_ratio_threshold=0.2,
    contam_alpha=0.05,
    isi_nbins=500,
    isi_bin_nsamples=30,
    shifts=[-2, -1, 0, 1, 2],
):
    n_spikes_templates = get_n_spikes_templates(n_templates, labels)
    x_z_templates = get_x_z_templates(n_templates, labels, x, z)
    print("GET PROPOSED PAIRS")
    dist_argsort, dist_template = get_proposed_pairs(
        n_templates,
        templates,
        x_z_templates,
        n_temp=n_temp,
        shifts=shifts,
    )

    labels_updated = labels.copy()
    reference_units = np.setdiff1d(np.unique(labels), [-1])

    for unit in tqdm(range(n_templates), desc="merge?"):
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
                        np.abs(templates[unit_reference]).max()
                        < np.abs(templates[unit_bis_reference]).max()
                    ):
                        mc = templates[unit_bis_reference].ptp(0).argmax()
                        two_units_shift = (
                            np.abs(templates[unit_reference, :, mc]).argmax()
                            - np.abs(
                                templates[unit_bis_reference, :, mc]
                            ).argmax()
                        )
                        unit_shifted = unit_reference
                    else:
                        mc = templates[unit_reference].ptp(0).argmax()
                        two_units_shift = (
                            np.abs(
                                templates[unit_bis_reference, :, mc]
                            ).argmax()
                            - np.abs(templates[unit_reference, :, mc]).argmax()
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
                    # print(unit_reference, unit_bis_reference, dpt_val)
                    is_merged = (
                        dpt_val < threshold_diptest
                        and np.abs(two_units_shift) < 2
                    )

                    # check isi violation
                    isi_allows_merge = True
                    if isi_veto and is_merged:
                        st1 = spike_index[labels == unit_reference, 0]
                        st2 = spike_index[labels == unit_bis_reference, 0]
                        contam_ratio, p_value = ccg_metrics(
                            st1, st2, isi_nbins, isi_bin_nsamples
                        )
                        contam_ok = contam_ratio < contam_ratio_threshold
                        contam_sig = p_value < contam_alpha
                        isi_allows_merge = contam_ok and contam_sig

                    # apply merge
                    if is_merged and isi_allows_merge:
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
                reference_units[
                    reference_units == unit_merged
                ] = new_reference_unit
                cmp += 1
    return labels_updated


# %%
def ks_bimodal_pursuit(
    unit_features,
    tpca,
    unit_rank=3,
    top_pc_init=True,
    aucsplit=0.85,
    min_size_split=50,
    max_split_corr=0.9,
    min_amp_sim=0.2,
    min_split_prop=0.05,
):
    """Adapted from PyKS"""
    N = len(unit_features)
    full = np.arange(N)
    empty = np.array([])
    if len(unit_features) < min_size_split:
        return False, full, empty

    if unit_rank < unit_features.shape[1]:
        unit_features = PCA(unit_rank).fit_transform(unit_features)

    if top_pc_init:
        # input should be centered so no problem with centered pca?
        w = PCA(1).fit(unit_features).components_.squeeze()
    else:
        # initialize with the mean of NOT drift-corrected trace
        w = unit_features.mean(axis=0)
        w /= np.linalg.norm(w)

    # initial projections of waveform PCs onto 1D vector
    x = unit_features @ w
    x_mean = x.mean()
    # initialize estimates of variance for the first
    # and second gaussian in the mixture of 1D gaussians
    s1 = x[x > x_mean].var()
    s2 = x[x < x_mean].var()
    # initialize the means as well
    mu1 = x[x > x_mean].mean()
    mu2 = x[x < x_mean].mean()
    # and the probability that a spike is assigned to the first Gaussian
    p = (x > x_mean).mean()

    # initialize matrix of log probabilities that each spike is assigned to the first
    # or second cluster
    logp = np.zeros((x.shape[0], 2), order="F")
    # do 50 pursuit iteration
    logP = np.zeros(50)  # used to monitor the cost function

    # TODO: move_to_config - maybe...
    for k in range(50):
        if min(s1, s2) < 1e-6:
            break

        # for each spike, estimate its probability to come from either Gaussian cluster
        logp[:, 0] = np.log(s1) / 2 - ((x - mu1) ** 2) / (2 * s1) + np.log(p)
        logp[:, 1] = (
            np.log(s2) / 2 - ((x - mu2) ** 2) / (2 * s2) + np.log(1 - p)
        )

        lMax = logp.max(axis=1)
        # subtract the max for floating point accuracy
        logp = logp - lMax[:, np.newaxis]
        rs = np.exp(logp)

        # get the normalizer and add back the max
        pval = np.log(np.sum(rs, axis=1)) + lMax
        # this is the cost function: we can monitor its increase
        logP[k] = pval.mean()
        # normalize so that probabilities sum to 1
        rs /= np.sum(rs, axis=1)[:, np.newaxis]
        if rs.sum(0).min() < 1e-6:
            break

        # mean probability to be assigned to Gaussian 1
        p = rs[:, 0].mean()
        # new estimate of mean of cluster 1 (weighted by "responsibilities")
        mu1 = np.dot(rs[:, 0], x) / np.sum(rs[:, 0])
        # new estimate of mean of cluster 2 (weighted by "responsibilities")
        mu2 = np.dot(rs[:, 1], x) / np.sum(rs[:, 1])

        # new estimates of variances
        s1 = np.dot(rs[:, 0], (x - mu1) ** 2) / np.sum(rs[:, 0])
        s2 = np.dot(rs[:, 1], (x - mu2) ** 2) / np.sum(rs[:, 1])

        if min(s1, s2) < 1e-6:
            break

        if (k >= 10) and (k % 2 == 0):
            # starting at iteration 10, we start re-estimating the pursuit direction
            # that is, given the Gaussian cluster assignments, and the mean and variances,
            # we re-estimate w
            # these equations follow from the model
            StS = (
                np.matmul(
                    unit_features.T,
                    unit_features
                    * (rs[:, 0] / s1 + rs[:, 1] / s2)[:, np.newaxis],
                )
                / unit_features.shape[0]
            )
            StMu = (
                np.dot(
                    unit_features.T, rs[:, 0] * mu1 / s1 + rs[:, 1] * mu2 / s2
                )
                / unit_features.shape[0]
            )

            # this is the new estimate of the best pursuit direction
            w = np.linalg.solve(StS.T, StMu)
            w /= np.linalg.norm(w)
            x = unit_features @ w

    # these spikes are assigned to cluster 1
    ilow = rs[:, 0] > rs[:, 1]
    # the smallest cluster has this proportion of all spikes
    nremove = min(ilow.mean(), (~ilow).mean())
    if nremove < min_split_prop:
        return False, full, empty

    # the mean probability of spikes assigned to cluster 1/2
    plow = rs[ilow, 0].mean()
    phigh = rs[~ilow, 1].mean()

    # now decide if the split would result in waveforms that are too similar
    # the reconstructed mean waveforms for putative cluster 1
    # c1 = cp.matmul(wPCA, cp.reshape((mean(clp0[ilow, :], 0), 3, -1), order='F'))
    c1 = tpca.inverse_transform(unit_features[ilow].mean())
    c2 = tpca.inverse_transform(unit_features[~ilow].mean())
    cc = np.corrcoef(c1.ravel(), c2.ravel())[
        0, 1
    ]  # correlation of mean waveforms
    n1 = np.linalg.norm(c1)  # the amplitude estimate 1
    n2 = np.linalg.norm(c2)  # the amplitude estimate 2

    r0 = 2 * abs((n1 - n2) / (n1 + n2))

    # if the templates are correlated, and their amplitudes are similar, stop the split!!!
    if (cc > max_split_corr) and (r0 < min_amp_sim):
        return False, full, empty

    # finaly criteria to continue with the split: if the split piece is more than 5% of all
    # spikes, if the split piece is more than 300 spikes, and if the confidences for
    # assigning spikes to # both clusters exceeds a preset criterion ccsplit
    if (
        (nremove > min_split_prop)
        and (min(plow, phigh) > aucsplit)
        # and (min(cp.sum(ilow), cp.sum(~ilow)) > 300)
    ):
        return True, np.flatnonzero(ilow), np.flatnonzero(~ilow)

    return False, full, empty


# %%
def ks_maxchan_tpca_split(
    tpca_embeddings,
    channel_index,
    maxchans,
    labels,
    tpca,
    recursive=True,
    top_pc_init=False,
    aucsplit=0.85,
    min_size_split=50,
    max_split_corr=0.9,
    min_amp_sim=0.2,
    min_split_prop=0.05,
):
    """
    If `recursive`, we will attempt to split the results of successful splits.
    """
    # initialize labels logic
    labels_new = labels.copy()
    next_label = labels_new.max() + 1
    labels_to_process = list(np.setdiff1d(np.unique(labels_new), [-1]))

    # load up maxchan TPCA loadings, batched in case of H5 dataset
    # these are small enough to fit in memory for short recordings
    N, P, C = tpca_embeddings.shape
    maxchan_loadings = np.empty((N, P), dtype=tpca_embeddings.dtype)
    for bs in range(0, N, 1000):
        be = min(N, bs + 1000)
        maxchan_loadings[bs:be] = waveform_utils.get_maxchan_traces(
            tpca_embeddings[bs:be], channel_index, maxchans[bs:be]
        )

    # store what final labels each original label ends up with
    # so that we can visualize and understand this step's output
    child_to_parent = {i: i for i in labels_to_process}

    # loop to process splits
    pbar = tqdm(total=len(labels_to_process), desc="KSMaxchan")
    while labels_to_process:
        cur_label = labels_to_process.pop()
        in_unit = np.flatnonzero(labels_new == cur_label)
        unit_features = maxchan_loadings[in_unit]
        try:
            is_split, group_a, group_b = ks_bimodal_pursuit(
                unit_features,
                tpca,
                top_pc_init=top_pc_init,
                aucsplit=aucsplit,
                min_size_split=min_size_split,
                max_split_corr=max_split_corr,
                min_amp_sim=min_amp_sim,
                min_split_prop=min_split_prop,
            )
        except np.linalg.LinAlgError as e:
            print(cur_label, "had error", e)
            is_split = False

        if is_split:
            labels_new[in_unit[group_b]] = next_label
            # parent is cur_label's parent (which is itself if cur_label was not split)
            child_to_parent[next_label] = child_to_parent[cur_label]
            next_label += 1
            if recursive:
                labels_to_process += [cur_label, next_label]
                pbar.total += 2

        pbar.update()

    # the parent to child mapping will be more useful for callers
    parent_to_child = {}
    for k, v in child_to_parent.items():
        if k in parent_to_child:
            parent_to_child[k].append(v)
        else:
            parent_to_child[k] = [v]

    return labels_new, parent_to_child
