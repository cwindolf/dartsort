import numpy as np
import spikeinterface
from spikeinterface.comparison import compare_two_sorters
import os
import pandas
import hdbscan
from spike_psvae import triage
from tqdm.auto import tqdm
import scipy


def read_waveforms(
    spike_times,
    bin_file,
    geom_array,
    n_times=121,
    offset_denoiser=42,
    channels=None,
    dtype=np.dtype("float32"),
):
    """
    read waveforms from recording
    n_times : waveform temporal length
    channels : channels to read from
    """
    # n_times needs to be odd
    if n_times % 2 == 0:
        n_times += 1

    # read all channels
    if channels is None:
        channels = np.arange(geom_array.shape[0])

    # ***** LOAD RAW RECORDING *****
    wfs = np.zeros((len(spike_times), n_times, len(channels)), "float32")

    skipped_idx = []
    n_channels = geom_array.shape[0]  # len(channels)
    total_size = n_times * n_channels
    # spike_times are the centers of waveforms
    spike_times_shifted = spike_times - offset_denoiser  # n_times//2
    # print(spike_times, spike_times_shifted)
    offsets = spike_times_shifted.astype(int) * dtype.itemsize * n_channels
    with open(bin_file, "rb") as fin:
        for ctr, spike in enumerate(spike_times_shifted):
            try:
                fin.seek(offsets[ctr], os.SEEK_SET)
                wf = np.fromfile(fin, dtype=dtype, count=total_size)
                wfs[ctr] = wf.reshape(n_times, n_channels)[:, channels]
            except OSError:
                print(f"skipped {ctr, spike}")
                skipped_idx.append(ctr)
    wfs = np.delete(wfs, skipped_idx, axis=0)
    fin.close()

    return wfs, skipped_idx


def compute_shifted_similarity(template1, template2, shifts=[0]):
    # TODO trim instead of padding, it can artificially increase this distance
    curr_similarities = []
    for shift in shifts:
        if shift == 0:
            similarity = np.max(np.abs(template1 - template2))
        elif shift < 0:
            template2_shifted_flattened = np.pad(
                template2.T.flatten(), ((-shift, 0)), mode="constant"
            )[:shift]
            similarity = np.max(
                np.abs(template1.T.flatten() - template2_shifted_flattened)
            )
        else:
            template2_shifted_flattened = np.pad(
                template2.T.flatten(), ((0, shift)), mode="constant"
            )[shift:]
            similarity = np.max(
                np.abs(template1.T.flatten() - template2_shifted_flattened)
            )
        curr_similarities.append(similarity)
    return np.min(curr_similarities), shifts[np.argmin(curr_similarities)]


def get_unit_similarities(
    cluster_id,
    st_1,
    closest_clusters,
    sorting,
    geom_array,
    raw_data_bin,
    num_channels_similarity=20,
    num_close_clusters=30,
    shifts_align=[0],
    order_by="similarity",
    normalize_agreement_by="both",
):
    waveforms1 = read_waveforms(st_1, raw_data_bin, geom_array, n_times=121)[0]
    template1 = np.median(waveforms1, axis=0)
    original_template = np.copy(template1)
    max_ptp_channel = np.argmax(template1.ptp(0))
    max_ptp = np.max(template1.ptp(0))
    channel_range = (
        max(max_ptp_channel - num_channels_similarity // 2, 0),
        max_ptp_channel + num_channels_similarity // 2,
    )
    template1 = template1[:, channel_range[0] : channel_range[1]]

    similarities = []
    agreements = []
    templates = []
    shifts = []
    for closest_cluster in closest_clusters:
        if closest_cluster in sorting.get_unit_ids():
            st_2 = sorting.get_unit_spike_train(closest_cluster)
            waveforms2 = read_waveforms(
                st_2, raw_data_bin, geom_array, n_times=121
            )[0]
            template2 = np.median(waveforms2, axis=0)[
                :, channel_range[0] : channel_range[1]
            ]
            similarity, shift = compute_shifted_similarity(
                template1, template2, shifts_align
            )
            shifts.append(shift)
            similarities.append(similarity)
            # similarities.append(similarity[0][0])
            (
                ind_st1,
                ind_st2,
                not_match_ind_st1,
                not_match_ind_st2,
            ) = compute_spiketrain_agreement(st_1, st_2, delta_frames=12)
            if normalize_agreement_by == "both":
                agreement = len(ind_st1) / (
                    len(st_1) + len(st_2) - len(ind_st1)
                )
            elif normalize_agreement_by == "first":
                agreement = len(ind_st1) / len(st_1)
            elif normalize_agreement_by == "second":
                agreement = len(ind_st1) / len(st_2)
            else:
                raise ValueError(
                    "normalize_agreement_by must be both, first, or second"
                )
            agreements.append(agreement)
            templates.append(template2)
    agreements = np.asarray(agreements).round(2)
    similarities = np.asarray(similarities).round(2)
    closest_clusters = np.asarray(closest_clusters)
    shifts = np.asarray(shifts)
    templates = np.asarray(templates)

    # compute most similar units (with template similarity or spike train agreement)
    if order_by == "similarity":
        most_similar_idxs = np.argsort(
            similarities
        )  # np.flip(np.argsort(similarities))
    elif order_by == "agreement":
        most_similar_idxs = np.flip(np.argsort(agreements))

    agreements = agreements[most_similar_idxs]
    similarities = similarities[most_similar_idxs]
    closest_clusters = closest_clusters[most_similar_idxs]
    templates = templates[most_similar_idxs]
    shifts = shifts[most_similar_idxs]

    return (
        original_template,
        closest_clusters,
        similarities,
        agreements,
        templates,
        shifts,
    )


def compute_spiketrain_agreement(st_1, st_2, delta_frames=12):
    # create figure for each match
    times_concat = np.concatenate((st_1, st_2))
    membership = np.concatenate(
        (np.ones(st_1.shape) * 1, np.ones(st_2.shape) * 2)
    )
    indices = times_concat.argsort()
    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    diffs = times_concat_sorted[1:] - times_concat_sorted[:-1]
    inds = np.where(
        (diffs <= delta_frames)
        & (membership_sorted[:-1] != membership_sorted[1:])
    )[0]
    if len(inds) > 0:
        inds2 = inds[np.where(inds[:-1] + 1 != inds[1:])[0]] + 1
        inds2 = np.concatenate((inds2, [inds[-1]]))
        times_matched = times_concat_sorted[inds2]
        # # find and label closest spikes
        ind_st1 = np.array(
            [np.abs(st_1 - tm).argmin() for tm in times_matched]
        )
        ind_st2 = np.array(
            [np.abs(st_2 - tm).argmin() for tm in times_matched]
        )
        not_match_ind_st1 = np.ones(st_1.shape[0], bool)
        not_match_ind_st1[ind_st1] = False
        not_match_ind_st1 = np.where(not_match_ind_st1)[0]
        not_match_ind_st2 = np.ones(st_2.shape[0], bool)
        not_match_ind_st2[ind_st2] = False
        not_match_ind_st2 = np.where(not_match_ind_st2)[0]
    else:
        ind_st1 = np.asarray([]).astype("int")
        ind_st2 = np.asarray([]).astype("int")
        not_match_ind_st1 = np.asarray([]).astype("int")
        not_match_ind_st2 = np.asarray([]).astype("int")

    return ind_st1, ind_st2, not_match_ind_st1, not_match_ind_st2


def get_agreement_indices(
    cluster_id_1, cluster_id_2, sorting1, sorting2, delta_frames=12
):
    # code borrowed from SpikeInterface
    lab_st1 = cluster_id_1
    lab_st2 = cluster_id_2
    st_1 = sorting1.get_unit_spike_train(lab_st1)
    mapped_st = sorting2.get_unit_spike_train(lab_st2)
    times_concat = np.concatenate((st_1, mapped_st))
    membership = np.concatenate(
        (np.ones(st_1.shape) * 1, np.ones(mapped_st.shape) * 2)
    )
    indices = times_concat.argsort()
    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    diffs = times_concat_sorted[1:] - times_concat_sorted[:-1]
    inds = np.where(
        (diffs <= delta_frames)
        & (membership_sorted[:-1] != membership_sorted[1:])
    )[0]

    if len(inds) > 0:
        inds2 = inds[np.where(inds[:-1] + 1 != inds[1:])[0]] + 1
        inds2 = np.concatenate((inds2, [inds[-1]]))
        times_matched = times_concat_sorted[inds2]
        # # find and label closest spikes
        ind_st1 = np.array(
            [np.abs(st_1 - tm).argmin() for tm in times_matched]
        )
        ind_st2 = np.array(
            [np.abs(mapped_st - tm).argmin() for tm in times_matched]
        )
        not_match_ind_st1 = np.ones(st_1.shape[0], bool)
        not_match_ind_st1[ind_st1] = False
        not_match_ind_st1 = np.where(not_match_ind_st1)[0]
        not_match_ind_st2 = np.ones(mapped_st.shape[0], bool)
        not_match_ind_st2[ind_st2] = False
        not_match_ind_st2 = np.where(not_match_ind_st2)[0]

    return (
        ind_st1,
        ind_st2,
        not_match_ind_st1,
        not_match_ind_st2,
        st_1,
        mapped_st,
    )


def remove_duplicate_units(clusterer, spike_frames, maxptps):
    sorting = make_sorting_from_labels_frames(clusterer.labels_, spike_frames)
    # remove duplicates
    cmp_self = compare_two_sorters(
        sorting, sorting, match_score=0.1, chance_score=0.1
    )
    remove_ids = set()
    for cluster_id in sorting.get_unit_ids():
        possible_matches = cmp_self.possible_match_12[cluster_id]
        if len(possible_matches) > 1:
            mean_ptp_matches = [
                np.mean(maxptps[clusterer.labels_ == cluster_id])
                for cluster_id in possible_matches
            ]
            remove_ids.add(possible_matches[np.argmin(mean_ptp_matches)])
    for remove_id in remove_ids:
        remove_id_indices = np.where(clusterer.labels_ == remove_id)
        clusterer.labels_[remove_id_indices] = -1

    # make sequential
    for i, label in enumerate(
        np.setdiff1d(np.unique(clusterer.labels_), [-1])
    ):
        label_indices = np.where(clusterer.labels_ == label)
        clusterer.labels_[label_indices] = i

    return clusterer, remove_ids


def remove_duplicate_spikes(clusterer, spike_frames, maxptps, frames_dedup):
    # normalize agreement by smaller unit and then remove only the spikes with agreement
    sorting = make_sorting_from_labels_frames(clusterer.labels_, spike_frames)
    # remove duplicates
    cmp_self = compare_two_sorters(
        sorting, sorting, match_score=0.1, chance_score=0.1, verbose=True
    )
    removed_cluster_ids = set()
    remove_spikes = []
    for cluster_id in tqdm(sorting.get_unit_ids(), desc="remove dups"):
        possible_matches = cmp_self.possible_match_12[cluster_id]
        # possible_matches[possible_matches!=cluster_id])
        if len(possible_matches) == 2:
            st_1 = sorting.get_unit_spike_train(possible_matches[0])
            st_2 = sorting.get_unit_spike_train(possible_matches[1])
            (
                ind_st1,
                ind_st2,
                not_match_ind_st1,
                not_match_ind_st2,
            ) = compute_spiketrain_agreement(
                st_1, st_2, delta_frames=frames_dedup
            )
            indices_agreement = [ind_st1, ind_st2]
            mean_ptp_matches = [
                np.mean(maxptps[clusterer.labels_ == cluster_id])
                for cluster_id in possible_matches
            ]
            remove_cluster_id = possible_matches[np.argmin(mean_ptp_matches)]
            remove_spike_indices = indices_agreement[
                np.argmin(mean_ptp_matches)
            ]
            if remove_cluster_id not in removed_cluster_ids:
                remove_spikes.append(
                    (remove_cluster_id, possible_matches, remove_spike_indices)
                )
                removed_cluster_ids.add(remove_cluster_id)
    remove_indices_list = []
    for cluster_id, _, spike_indices in remove_spikes:
        remove_indices = np.where(clusterer.labels_ == cluster_id)[0][
            spike_indices
        ]
        clusterer.labels_[remove_indices] = -1
        remove_indices_list.append(remove_indices)

    # make sequential
    for i, label in enumerate(
        np.setdiff1d(np.unique(clusterer.labels_), [-1])
    ):
        label_indices = np.where(clusterer.labels_ == label)
        clusterer.labels_[label_indices] = i

    return clusterer, remove_indices_list, remove_spikes


def perturb_features(x, z, logmaxptp_scaled, noise_scale):
    x_p = x + np.random.normal(loc=0.0, scale=noise_scale)
    z_p = z + np.random.normal(loc=0.0, scale=noise_scale)
    log_maxptp_scaled_p = logmaxptp_scaled + np.random.normal(
        loc=0.0, scale=noise_scale
    )
    return x_p, z_p, log_maxptp_scaled_p


def copy_spikes(x, z, maxptps, spike_index, scales=(1,1,30), num_duplicates_list=[0,1,2,3,4]):
    true_spike_indices = []
    new_x = []
    new_z = []
    new_maxptps = []
    new_spike_index = []
    new_spike_ids = []
    num_bins = len(num_duplicates_list)
    num_duplicates_array = np.asarray(num_duplicates_list)
    ptp_bins = np.histogram(maxptps, bins=num_bins-1)[1]
    for spike_id in range(len(spike_index)):
        maxptp_i = maxptps[spike_id]
        x_i = x[spike_id]
        z_i = z[spike_id]
        new_x.append(x_i)
        new_z.append(z_i)
        new_maxptps.append(maxptp_i) 
        new_spike_index.append(spike_index[spike_id])
        #add a 1 to spike index to indicate real spike
        true_spike_indices.append([True, spike_id])
        ptp_bin_i = np.sort(ptp_bins[np.argsort(np.abs(maxptp_i - ptp_bins))[:2]])
        duplicates_choice = num_duplicates_array[np.sort(np.argsort(np.abs(maxptp_i - ptp_bins))[:2])]
        p = (maxptp_i - ptp_bin_i.min())/(ptp_bin_i - ptp_bin_i.min()).max()
        num_duplicates = np.random.choice(duplicates_choice, p=[1-p, p])
        for i in range(num_duplicates):
            x_p, z_p, log_maxptp_scaled_p = perturb_features(x_i*scales[0], z_i*scales[1], np.log(maxptp_i)*scales[2], noise_scale=1)
            new_x.append(x_p/scales[0])
            new_z.append(z_p/scales[1])
            new_maxptps.append(np.e**(log_maxptp_scaled_p/scales[2]))
            new_spike_index.append(spike_index[spike_id])
            #add a 0 to spike index to indicate copied spike
            true_spike_indices.append([False, -1])
    new_x = np.asarray(new_x)
    new_z = np.asarray(new_z)
    new_maxptps = np.asarray(new_maxptps)
    new_spike_index = np.asarray(new_spike_index)
    true_spike_indices = np.asarray(true_spike_indices)
    return new_x, new_z, new_maxptps, new_spike_index, true_spike_indices


def cluster_spikes(
    x,
    z,
    maxptps,
    spike_index,
    min_cluster_size=25,
    min_samples=25,
    scales=(1, 1, 30),
    frames_dedup=12,
    triage_quantile=80,
    region_size=25,
    bin_size=5,
    ptp_low_threshold=3,
    ptp_high_threshold=6, #deprecated
    do_copy_spikes=True,
    do_relabel_by_depth=True,
    do_remove_dups=True,
    split_big=False,
    split_big_kw=dict(dx=40, dz=48, min_size_split=50),
    do_subsample=False,
):
    # copy high-ptp spikes
    true_spike_indices = np.stack(
        (np.ones(maxptps.shape[0], dtype=bool), np.arange(maxptps.shape[0])),
        axis=1,
    )
    if do_copy_spikes:
        print(f"copying {z.size} spikes")
        x, z, maxptps, spike_index, true_spike_indices = copy_spikes(
            x,
            z,
            maxptps,
            spike_index,
            scales=scales,
            num_duplicates_list=[0, 1, 2, 3, 4],
        )
        print(f"{z.size} spikes")
        
    if do_subsample:
        print(f"subsampling from {z.size} spikes")
        n_spikes = 2000
        selected_spike_indices = subsample_spikes(n_spikes=n_spikes, spike_index=spike_index, 
                                                  method='smart_sampling_amplitudes', x=x, z=z, maxptps=maxptps)
        x = x[selected_spike_indices]
        z = z[selected_spike_indices]
        maxptps = maxptps[selected_spike_indices]
        spike_index = spike_index[selected_spike_indices]
        true_spike_indices = true_spike_indices[selected_spike_indices]
        print(f"{z.size} spikes")

    # triage low ptp spikes to improve density-based clustering
    if triage_quantile < 100:
        # print(f"triaging from {z.size} spikes")
        # (
        #     idx_keep,
        #     high_ptp_filter,
        #     low_ptp_filter,
        # ) = triage.run_weighted_triage_adaptive(
        #     x, 
        #     z, 
        #     maxptps, 
        #     scales=scales, 
        #     threshold=triage_quantile, 
        #     ptp_low_threshold=ptp_low_threshold, 
        #     ptp_high_threshold=ptp_high_threshold, 
        #     bin_size=bin_size, 
        #     region_size=region_size
        # )
        # high_ptp_mask = np.zeros(maxptps.shape[0], dtype=bool)
        # high_ptp_mask[high_ptp_filter] = True
        # spike_ids = np.arange(maxptps.shape[0])
        # low_ptp_spike_ids = spike_ids[high_ptp_mask]
        # # low ptp spikes keep ids
        # keep_low_ptp_spike_ids = low_ptp_spike_ids[low_ptp_filter][idx_keep]
        # keep_high_ptp_mask = np.ones(maxptps.shape[0], dtype=bool)
        # keep_high_ptp_mask[high_ptp_filter] = False
        # # high ptp spikes keep ids
        # high_ptp_spike_ids = spike_ids[keep_high_ptp_mask]
        # # final indices keep
        # final_keep_indices = np.sort(
        #     np.concatenate((keep_low_ptp_spike_ids, high_ptp_spike_ids))
        # )
        # x = x[final_keep_indices]
        # z = z[final_keep_indices]
        # maxptps = maxptps[final_keep_indices]
        # spike_index = spike_index[final_keep_indices]
        # true_spike_indices = true_spike_indices[final_keep_indices]
        x, z, maxptps, idx_keep, low_ptp_filter = triage.run_weighted_triage_adaptive(x, z, maxptps, scales=scales, threshold=triage_quantile, ptp_low_threshold=ptp_low_threshold, bin_size=bin_size, region_size=region_size)
        spike_index = spike_index[low_ptp_filter][idx_keep]
        true_spike_indices = true_spike_indices[low_ptp_filter][idx_keep]
        print(f"{z.size} spikes")
        # barf
        
        
    # create feature set for clustering
    features = np.c_[x * scales[0], z * scales[1], np.log(maxptps) * scales[2]]

    # features = np.c_[tx*scales[0], tz*scales[2], all_pcs[:,0] * alpha1, all_pcs[:,1] * alpha2]
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, core_dist_n_jobs=-1
    )
    clusterer.fit(features)

    # remove copied spikes
    clusterer.labels_ = clusterer.labels_[np.where(true_spike_indices[:, 0])]
    clusterer._raw_data = clusterer._raw_data[
        np.where(true_spike_indices[:, 0])
    ]
    clusterer.probabilities_ = clusterer.probabilities_[
        np.where(true_spike_indices[:, 0])
    ]
    maxptps = maxptps[np.where(true_spike_indices[:, 0])]
    x = x[np.where(true_spike_indices[:, 0])]
    z = z[np.where(true_spike_indices[:, 0])]
    spike_index = spike_index[np.where(true_spike_indices[:, 0])]
    original_spike_ids = true_spike_indices[
        np.where(true_spike_indices[:, 0])
    ][:, 1]

    # reorder by z
    if do_relabel_by_depth:
        print("relabel", flush=True)
        cluster_centers = compute_cluster_centers(clusterer)
        clusterer = relabel_by_depth(clusterer, cluster_centers)
    else:
        cluster_centers = compute_cluster_centers(clusterer)
    # remove dups (from NN denoise) and reorder by z
    if do_remove_dups:
        print("dups", flush=True)
        (
            clusterer,
            duplicate_indices,
            duplicate_spikes,
        ) = remove_duplicate_spikes(
            clusterer, spike_index[:, 0], maxptps, frames_dedup=frames_dedup
        )
        cluster_centers = compute_cluster_centers(clusterer)

    if split_big:
        print("split big...")
        clusterer.labels_ = split_big_clusters(
            clusterer.labels_,
            x,
            z,
            maxptps,
            spike_index,
            **split_big_kw,
        )
        print("done splitting big", flush=True)
    print("done", flush=True)

    return (
        clusterer,
        cluster_centers,
        spike_index,
        x,
        z,
        maxptps,
        original_spike_ids,
    )


def split_big_clusters(
    labels_original,
    x,
    z_reg,
    maxptps,
    spike_index,
    dz=40,
    dx=48,
    min_size_split=50,
):
    labels_new = labels_original.copy()
    next_label = labels_original.max() + 1
    for unit in tqdm(np.setdiff1d(np.unique(labels_original), [-1])):
        idx = np.flatnonzero(labels_original == unit)

        tall = z_reg[idx].ptp() > dz
        wide = x[idx].ptp() > dx
        if (tall or wide) and idx.sum() > min_size_split:
            (
                clusterer,
                cluster_centers_unit,
                tspike_index_unit,
                tx_unit,
                tz_unit,
                tmaxptps_unit,
                idx_full_unit,
            ) = cluster_spikes(
                x[idx],
                z_reg[idx],
                maxptps[idx],
                spike_index[idx],
                triage_quantile=100,
                do_copy_spikes=True,
                do_relabel_by_depth=False,
                do_remove_dups=False,
            )
            if np.unique(clusterer.labels_).shape[0] <= 1:
                continue

            for k in np.unique(clusterer.labels_):
                which = idx[clusterer.labels_ == k]

                if k == -1:
                    labels_new[which] = -1
                elif k == 0:
                    pass
                else:
                    labels_new[which] = next_label
                    next_label += 1

    return labels_new

def compute_cluster_centers(clusterer):
    cluster_centers_data = []
    cluster_ids = np.setdiff1d(np.unique(clusterer.labels_), [-1])
    for label in cluster_ids:
        cluster_centers_data.append(clusterer.weighted_cluster_centroid(label))
    cluster_centers_data = np.asarray(cluster_centers_data)
    cluster_centers = pandas.DataFrame(
        data=cluster_centers_data, index=cluster_ids
    )
    return cluster_centers


def relabel_by_depth(clusterer, cluster_centers):
    # re-label each cluster by z-depth
    indices_depth = np.argsort(-cluster_centers.iloc[:, 1].to_numpy())
    labels_depth = cluster_centers.index[indices_depth]
    label_to_id = {}
    for i, label in enumerate(labels_depth):
        label_to_id[label] = i
    label_to_id[-1] = -1
    new_labels = np.vectorize(label_to_id.get)(clusterer.labels_)
    clusterer.labels_ = new_labels
    return clusterer


def get_closest_clusters_hdbscan(
    cluster_id, cluster_centers, num_close_clusters=2
):
    curr_cluster_center = cluster_centers.loc[cluster_id].to_numpy()
    dist_other_clusters = np.linalg.norm(
        curr_cluster_center[:2] - cluster_centers.iloc[:, :2].to_numpy(),
        axis=1,
    )
    closest_cluster_indices = np.argsort(dist_other_clusters)[
        1 : num_close_clusters + 1
    ]
    closest_clusters = cluster_centers.index[closest_cluster_indices]
    return closest_clusters


def get_closest_clusters_kilosort(
    cluster_id, kilo_cluster_depth_means, num_close_clusters=2
):
    curr_cluster_depth = kilo_cluster_depth_means[cluster_id]
    dist_to_other_cluster_dict = {
        cluster_id: abs(mean_depth - curr_cluster_depth)
        for (cluster_id, mean_depth) in kilo_cluster_depth_means.items()
    }
    closest_clusters = [
        y[0]
        for y in sorted(
            dist_to_other_cluster_dict.items(), key=lambda x: x[1]
        )[1 : 1 + num_close_clusters]
    ]
    return closest_clusters


def get_closest_clusters_hdbscan_kilosort(
    cluster_id, cluster_centers, kilo_cluster_depth_means, num_close_clusters=2
):
    cluster_center = cluster_centers.loc[cluster_id].to_numpy()
    curr_cluster_depth = cluster_center[1]
    dist_to_other_cluster_dict = {
        cluster_id: abs(mean_depth - curr_cluster_depth)
        for (cluster_id, mean_depth) in kilo_cluster_depth_means.items()
    }
    closest_clusters = [
        y[0]
        for y in sorted(
            dist_to_other_cluster_dict.items(), key=lambda x: x[1]
        )[:num_close_clusters]
    ]
    return closest_clusters


def get_closest_clusters_kilosort_hdbscan(
    cluster_id, kilo_cluster_depth_means, cluster_centers, num_close_clusters=2
):
    curr_cluster_depth = kilo_cluster_depth_means[cluster_id]
    closest_cluster_indices = np.argsort(
        np.abs(
            cluster_centers.iloc[:, 1].to_numpy()
            - kilo_cluster_depth_means[cluster_id]
        )
    )[:num_close_clusters]
    closest_clusters = cluster_centers.index[closest_cluster_indices]
    return closest_clusters


def make_sorting_from_labels_frames(
    labels, spike_frames, sampling_frequency=30000
):
    # times_list = []
    # labels_list = []
    # for cluster_id in np.unique(labels):
    #     spike_train = spike_frames[np.where(labels==cluster_id)]
    #     times_list.append(spike_train)
    #     labels_list.append(np.zeros(spike_train.shape[0])+cluster_id)
    # times_array = np.concatenate(times_list).astype('int')
    # labels_array = np.concatenate(labels_list).astype('int')
    sorting = spikeinterface.numpyextractors.NumpySorting.from_times_labels(
        times_list=spike_frames.astype("int"),
        labels_list=labels.astype("int"),
        sampling_frequency=sampling_frequency,
    )
    return sorting


def make_labels_contiguous(labels, in_place=False, return_unique=False):
    untriaged = np.flatnonzero(labels >= 0)
    unique, contiguous = np.unique(labels[untriaged], return_inverse=True)
    out = labels if in_place else labels.copy()
    out[untriaged] = contiguous
    if return_unique:
        return out, unique
    return out

def subsample_spikes(n_spikes, spike_index, method='uniform', x=None, z=None, maxptps=None, num_channels=384):
    #can subsample with number of spikes (int) or percentage of spikes (0.0-1.0 float)
    assert type(n_spikes) == int or type(n_spikes) == float
    selected_spike_indices = []
    if method == 'uniform':
        for channel in range(num_channels):
            spike_indices_channel = np.where(spike_index[:,1] == channel)[0]
            if type(n_spikes) == float:
                n_spikes_chan = int(n_spikes*len(spike_indices_channel))
            max_peaks = min(spike_indices_channel.size, n_spikes_chan)
            selected_spike_indices += [np.random.choice(spike_indices_channel, size=max_peaks, replace=False)]        
    if method == 'uniform_locations':
        n_bins = (50, 50)
        assert x is not None
        assert z is not None
        xmin, xmax = np.min(x), np.max(x)
        zmin, zmax = np.min(z), np.max(z)

        x_grid = np.linspace(xmin, xmax, n_bins[0])
        z_grid = np.linspace(zmin, zmax, n_bins[1])

        x_idx = np.searchsorted(x_grid, x)
        z_idx = np.searchsorted(z_grid, z)

        for i in range(n_bins[0]):
            for j in range(n_bins[1]):
                spike_indices = np.where((x_idx == i) & (z_idx == j))[0]
                if type(n_spikes) == float:
                    n_spikes_bin = int(n_spikes*len(spike_indices))
                max_peaks = min(spike_indices.size, n_spikes_bin)
                selected_spike_indices += [np.random.choice(spike_indices, size=max_peaks, replace=False)]
                
    if method == 'smart_sampling_amplitudes':
        for channel in range(num_channels):
            n_bins = 50
            assert maxptps is not None
            spike_indices_channel = np.where(spike_index[:,1] == channel)[0]
            sub_maxptps = maxptps[spike_indices_channel]  
            if type(n_spikes) == float:
                n_spikes_chan = int(n_spikes*len(spike_indices_channel))
            valid_indices = get_valid_indices(sub_maxptps, n_bins=n_bins, n_spikes=n_spikes_chan)
            selected_spike_indices += [spike_indices_channel[valid_indices]]
                
    selected_spike_indices = np.sort(np.concatenate(selected_spike_indices))
    return selected_spike_indices

def reject_rate(x, d, a, target, n_bins):
    return (np.mean(n_bins*a*np.clip(1 - d*x, 0, 1)) - target)**2
    
def get_valid_indices(maxptps, n_bins, n_spikes, exponent=1):
    assert n_bins is not None
    bins = np.linspace(maxptps.min(), maxptps.max(), n_bins)
    x, y = np.histogram(maxptps, bins=bins)
    histograms = {'probability' : x/x.sum(), 'maxptps' : y[1:]}
    indices = np.searchsorted(histograms['maxptps'], maxptps)

    probabilities = histograms['probability']
    z = probabilities[probabilities > 0]
    c = 1.0 / np.min(z)
    d = np.ones(len(probabilities))
    d[probabilities > 0] = 1. / (c * z)
    d = np.minimum(1, d)
    d /= np.sum(d)
    twist = np.sum(probabilities * d)
    factor = twist * c

    target_rejection = (1 - max(0, n_spikes/len(indices)))**exponent
    res = scipy.optimize.fmin(reject_rate, factor, args=(d, probabilities, target_rejection, n_bins), disp=False)
    rejection_curve = np.clip(1 - d*res[0], 0, 1)

    acceptation_threshold = rejection_curve[indices]
    valid_indices = acceptation_threshold < np.random.rand(len(indices))

    return valid_indices