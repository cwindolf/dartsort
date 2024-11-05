import dataclasses

import hdbscan
import h5py
import numpy as np
import spikeinterface
from dartsort.util import drift_util, spikeio, data_util, waveform_util
from dredge.motion_util import IdentityMotionEstimate
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import KDTree
from sklearn.neighbors import KNeighborsClassifier


def agglomerate(labels, distances, linkage_method="complete", threshold=1.0):
    """"""
    n = distances.shape[0]
    pdist = distances[np.triu_indices(n, k=1)]
    if pdist.min() > threshold:
        ids = np.unique(labels)
        return labels, ids[ids >= 0]
    finite = np.isfinite(pdist)
    if not finite.all():
        inf = max(0, pdist[finite].max()) + threshold + 1.0
        pdist[np.logical_not(finite)] = inf

    Z = linkage(pdist, method=linkage_method)
    new_ids = fcluster(Z, threshold, criterion="distance")
    # offset by 1, I think always, but I don't want to be wrong?
    new_ids -= new_ids.min()

    kept = labels >= 0
    new_labels = np.full_like(labels, -1)
    new_labels[kept] = new_ids[labels[kept]]

    return new_labels, new_ids


def combine_disjoint(inds_a, labels_a, inds_b, labels_b):
    labels = np.full(labels_a.size + labels_b.size, -1, dtype=labels_a.dtype)
    labels[inds_a] = labels_a
    labels[inds_b] = labels_b
    return labels


def reorder_by_depth(sorting, motion_est=None):
    kept = np.flatnonzero(sorting.labels >= 0)
    kept_labels = sorting.labels[kept]

    units, kept_labels = np.unique(kept_labels, return_inverse=True)

    depths = sorting.point_source_localizations[kept, 2]
    if motion_est is not None:
        depths = motion_est.correct_s(sorting.times_seconds[kept], depths)

    centroids = np.zeros(units.size)
    for u in range(units.size):
        inu = np.flatnonzero(kept_labels == u)
        centroids[u] = np.median(depths[inu])

    labels = sorting.labels.copy()
    # this one is some food for thought, lol.
    labels[kept] = np.argsort(np.argsort(centroids))[kept_labels]

    return dataclasses.replace(sorting, labels=labels)


def closest_registered_channels(times_seconds, x, z_abs, geom, motion_est=None):
    """Assign spikes to the drift-extended channel closest to their registered position"""
    if motion_est is None:
        motion_est == IdentityMotionEstimate()
    registered_geom = drift_util.registered_geometry(geom, motion_est)
    z_reg = motion_est.correct_s(times_seconds, z_abs)
    reg_pos = np.c_[x, z_reg]

    registered_kdt = KDTree(registered_geom)
    distances, reg_channels = registered_kdt.query(reg_pos)

    return reg_channels


def grid_snap(times_seconds, x, z_abs, geom, grid_dx=15, grid_dz=15, motion_est=None):
    if motion_est is None:
        motion_est == IdentityMotionEstimate()
    z_reg = motion_est.correct_s(times_seconds, z_abs)
    reg_pos = np.c_[x, z_reg]

    # make a grid inside the registered geom bounding box
    registered_geom = drift_util.registered_geometry(geom, motion_est)
    min_x, max_x = registered_geom[:, 0].min(), registered_geom[:, 0].max()
    min_z, max_z = registered_geom[:, 1].min(), registered_geom[:, 1].max()
    grid_x = np.arange(min_x, max_x, grid_dx)
    grid_x += (min_x + max_x) / 2 - grid_x.mean()
    grid_z = np.arange(min_z, max_z, grid_dz)
    grid_z += (min_z + max_z) / 2 - grid_z.mean()
    grid_xx, grid_zz = np.meshgrid(grid_x, grid_z, indexing="ij")
    grid = np.c_[grid_xx.ravel(), grid_zz.ravel()]

    # snap to closest grid point
    registered_kdt = KDTree(grid)
    distances, reg_channels = registered_kdt.query(reg_pos)

    return reg_channels


def hdbscan_clustering(
    recording,
    times_seconds,
    times_samples,
    x,
    z_abs,
    amps,
    geom,
    motion_est=None,
    min_cluster_size=25,
    min_samples=25,
    cluster_selection_epsilon=1,
    scales=(1, 1, 50),
    adaptive_feature_scales=False,
    log_c=5,
    recursive=True,
    remove_duplicates=True,
    frames_dedup=12,
    frame_dedup_cluster=20,
    remove_big_units=True,
    zstd_big_units=50,
):
    """
    Run HDBSCAN
    triaging/subsampling/copying/splitting big clusters not implemented since we don't use it (so far)
    """
    if motion_est is None:
        z_reg = z_abs
    else:
        z_reg = motion_est.correct_s(times_seconds, z_abs)

    if adaptive_feature_scales:
        scales = (
            1,
            1,
            np.median(np.abs(x - np.median(x)))
            / np.median(np.abs(np.log(log_c + amps) - np.median(np.log(log_c + amps)))),
        )

    features = np.c_[x * scales[0], z_reg * scales[1], np.log(log_c + amps) * scales[2]]
    if features.shape[1] >= features.shape[0]:
        return -1 * np.ones(features.shape[0])

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        min_samples=min_samples,
        core_dist_n_jobs=-1,
    )
    clusterer.fit(features)

    if remove_duplicates:
        (
            clusterer,
            duplicate_indices,
            duplicate_spikes,
        ) = remove_duplicate_spikes(
            clusterer, times_samples, amps, frames_dedup=frames_dedup
        )

        kept_ix, removed_ix = remove_self_duplicates(
            times_samples,
            clusterer.labels_,
            recording,
            geom.shape[0],
            frame_dedup=frame_dedup_cluster,
        )
        if len(removed_ix):
            clusterer.labels_[removed_ix.astype("int")] = -1

    if not recursive:
        if remove_big_units:
            labels = clusterer.labels_
            _, labels[labels >= 0] = np.unique(labels[labels >= 0], return_inverse=True)
            arr_z_std = np.zeros(labels.max() + 1)
            for k in np.unique(labels[labels >= 0]):
                idx = np.flatnonzero(labels == k)
                arr_z_std[k] = z_reg[idx].std()
            bad_units = np.where(arr_z_std > zstd_big_units)
            labels[np.isin(labels, bad_units)] = -1
            _, labels[labels >= 0] = np.unique(labels[labels >= 0], return_inverse=True)
            return labels
        else:
            return clusterer.labels_

    # -- recursively split clusters as long as HDBSCAN keeps finding more than 1
    # if HDBSCAN only finds one cluster, then be done
    units = np.unique(clusterer.labels_)
    if units[units >= 0].size <= 1:
        # prevent triaging when no split was found
        return np.zeros_like(clusterer.labels_)

    # else, recursively enter all labels and split them
    labels = clusterer.labels_.copy()
    next_label = units.max() + 1
    for unit in units[units >= 0]:
        in_unit = np.flatnonzero(clusterer.labels_ == unit)
        split_labels = hdbscan_clustering(
            recording,
            times_seconds[in_unit],
            times_samples[in_unit],
            x[in_unit],
            z_abs[in_unit],
            amps[in_unit],
            geom,
            motion_est=motion_est,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            scales=scales,
            log_c=log_c,
            recursive=recursive,
            remove_duplicates=remove_duplicates,
            frames_dedup=frames_dedup,
            frame_dedup_cluster=frame_dedup_cluster,
        )
        labels[in_unit[split_labels < 0]] = split_labels[split_labels < 0]
        labels[in_unit[split_labels >= 0]] = (
            split_labels[split_labels >= 0] + next_label
        )
        # next_label += split_labels[split_labels >= 0].max() + 1
        next_label += split_labels.max() + 1  # is that ok

    # reindex
    _, labels[labels >= 0] = np.unique(labels[labels >= 0], return_inverse=True)
    return labels


# How to deal with outliers?


def knn_reassign_outliers(labels, features):
    outliers = labels < 0
    outliers_idx = np.flatnonzero(outliers)
    if not outliers_idx.size:
        return labels
    knn = KNeighborsClassifier()
    knn.fit(features[~outliers], labels[~outliers])
    new_labels = labels.copy()
    new_labels[outliers_idx] = knn.predict(features[outliers_idx])
    return new_labels


"""
Functions below are not  used yet - need some updating before plugging them in 
"""


def remove_self_duplicates(
    spike_times,
    spike_labels,
    recording,
    n_channels,
    frame_dedup=20,
    n_samples=250,
    too_contaminated=0.75,
    search_threshold_lo=0.01,
    search_threshold_switch=500,
    search_threshold_hi=0.05,
    seed=0,
):
    """
    TODO: change how things are read etc... + is this really useful?

    """
    indices_to_remove = []
    N = spike_labels.shape[0]
    assert spike_times.shape == spike_labels.shape == (N,)
    unit_labels = np.unique(spike_labels)
    unit_labels = unit_labels[unit_labels >= 0]
    rg = np.random.default_rng(seed)

    for unit in unit_labels:
        in_unit = np.flatnonzero(spike_labels == unit)

        spike_times_unit = spike_times[in_unit]
        violations = np.diff(spike_times_unit) < frame_dedup

        # if there are few violations, it's not worth trying to keep them
        if violations.mean() < search_threshold_lo or (
            in_unit.size > search_threshold_switch
            and violations.mean() < search_threshold_hi
        ):
            viol_ix = np.flatnonzero(violations)
            ix_remove_unit = np.unique(np.concatenate((viol_ix, viol_ix + 1)))
            indices_to_remove.extend(in_unit[ix_remove_unit])

        elif violations.mean() > too_contaminated:
            # print("super contaminated unit.")
            indices_to_remove.extend(in_unit)

        elif violations.any():
            # print(f"{unit=} {in_unit.size=} {violations.mean()=}")
            # we'll remove either an index in first_viol_ix,
            # or that index + 1, depending on template agreement
            first_viol_ix = np.flatnonzero(violations)
            all_viol_ix = np.concatenate([first_viol_ix, [first_viol_ix[-1] + 1]])
            unviol = np.setdiff1d(np.arange(spike_times_unit.shape[0]), all_viol_ix)

            # load as many unviolated wfs as possible
            if unviol.size > n_samples:
                # we can compute template just from unviolated wfs
                which_unviol = rg.choice(unviol.size, n_samples, replace=False)
                which_unviol.sort()
                wfs_unit = spikeio.read_full_waveforms(
                    recording, spike_times_unit[which_unviol]
                )
            else:
                n_viol_load = min(all_viol_ix.size, n_samples - unviol.size)
                load_ix = np.concatenate(
                    [
                        unviol,
                        rg.choice(all_viol_ix, n_viol_load, replace=False),
                    ]
                )
                load_ix.sort()
                wfs_unit = spikeio.read_full_waveforms(
                    recording, spike_times_unit[load_ix]
                )

            # reshape to NxT
            template = np.median(wfs_unit, axis=0)
            mc = np.ptp(template, 0).argmax()
            template_mc = template[:, mc]
            template_argmin = np.abs(template_mc).argmax()

            # get subsets of wfs -- will we remove leading (wfs_1)
            # or trailing (wfs_2) waveform in each case?
            wfs_1 = spikeio.read_subset_waveforms(
                recording,
                spike_times_unit[first_viol_ix],
                load_channels=np.array([mc]),
            )
            wfs_1 = wfs_1[:, :, 0]
            wfs_2 = spikeio.read_subset_waveforms(
                recording,
                spike_times_unit[first_viol_ix + 1],
                load_channels=np.array([mc]),
            )
            wfs_2 = wfs_2[:, :, 0]

            # first is better will have a 1 where wfs_1 was better
            # aligned then wfs_2, so that first_viol_ix + best_aligned
            # is the index of the waveforms to *remove!*
            argmins_1 = np.abs(wfs_1).argmax(axis=1)
            argmins_2 = np.abs(wfs_2).argmax(axis=1)
            first_is_better = np.abs(argmins_1 - template_argmin) <= (
                argmins_2 - template_argmin
            )

            # it's possible that duplicates could arrive, so that we're
            # not really removing *all* the violations.
            # but it's better than nothing!
            ix_remove_unit = np.unique(first_viol_ix + first_is_better)

            # append and continue to next unit
            indices_to_remove.extend(in_unit[ix_remove_unit])

    indices_to_remove = np.array(indices_to_remove)
    indices_to_keep = np.setdiff1d(np.arange(N), indices_to_remove)

    return indices_to_keep, indices_to_remove


def remove_duplicate_spikes(
    clusterer,
    spike_frames,
    maxptps,
    frames_dedup,
    full_duplicate_fraction=0.95,
    min_result_spikes=10,
):
    from spikeinterface.comparison import compare_two_sorters
    # normalize agreement by smaller unit and then remove only the spikes with agreement
    sorting = make_sorting_from_labels_frames(clusterer.labels_, spike_frames)
    # remove duplicates
    cmp_self = compare_two_sorters(sorting, sorting, match_score=0.1, chance_score=0.1)
    removed_cluster_ids = set()
    remove_spikes = []
    for cluster_id in sorting.get_unit_ids():
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
            ) = compute_spiketrain_agreement(st_1, st_2, delta_frames=frames_dedup)
            mean_ptp_matches = [
                maxptps[clusterer.labels_ == cluster_id].mean()
                for cluster_id in possible_matches
            ]
            which = np.argmin(mean_ptp_matches)
            remove_cluster_id = possible_matches[which]
            remove_ind = [ind_st1, ind_st2][which]
            not_match = [not_match_ind_st1, not_match_ind_st2][which]
            remain_frac = not_match.size / (not_match.size + remove_ind.size)
            if (
                not_match.size < min_result_spikes
                or remain_frac < 1 - full_duplicate_fraction
            ):
                remove_ind = np.concatenate((remove_ind, not_match))

            if remove_cluster_id not in removed_cluster_ids:
                remove_spikes.append((remove_cluster_id, possible_matches, remove_ind))
                removed_cluster_ids.add(remove_cluster_id)

    remove_indices_list = []
    for cluster_id, _, spike_indices in remove_spikes:
        in_unit = np.flatnonzero(clusterer.labels_ == cluster_id)
        remove_indices = in_unit[spike_indices]
        clusterer.labels_[remove_indices] = -1
        remove_indices_list.append(remove_indices)

    # make contiguous
    _, clusterer.labels_ = np.unique(clusterer.labels_, return_inverse=True)

    return clusterer, remove_indices_list, remove_spikes


def compute_spiketrain_agreement(st_1, st_2, delta_frames=12):
    # create figure for each match
    times_concat = np.concatenate((st_1, st_2))
    membership = np.concatenate((np.ones(st_1.shape) * 1, np.ones(st_2.shape) * 2))
    indices = times_concat.argsort()
    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    diffs = times_concat_sorted[1:] - times_concat_sorted[:-1]
    inds = np.flatnonzero(
        (diffs <= delta_frames) & (membership_sorted[:-1] != membership_sorted[1:])
    )

    if len(inds) > 0:
        inds2 = inds[np.where(inds[:-1] + 1 != inds[1:])[0]] + 1
        inds2 = np.concatenate((inds2, [inds[-1]]))
        times_matched = times_concat_sorted[inds2]
        # # find and label closest spikes
        ind_st1 = np.array([np.abs(st_1 - tm).argmin() for tm in times_matched])
        ind_st2 = np.array([np.abs(st_2 - tm).argmin() for tm in times_matched])
        not_match_ind_st1 = np.ones(st_1.shape[0], bool)
        not_match_ind_st1[ind_st1] = False
        not_match_ind_st1 = np.where(not_match_ind_st1)[0]
        not_match_ind_st2 = np.ones(st_2.shape[0], bool)
        not_match_ind_st2[ind_st2] = False
        not_match_ind_st2 = np.where(not_match_ind_st2)[0]
    else:
        ind_st1 = np.array([], dtype=int)
        ind_st2 = np.array([], dtype=int)
        not_match_ind_st1 = np.arange(len(st_1))
        not_match_ind_st2 = np.arange(len(st_2))

    return ind_st1, ind_st2, not_match_ind_st1, not_match_ind_st2


def make_sorting_from_labels_frames(labels, spike_frames, sampling_frequency=30000):
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


def meet(labels_a, labels_b):
    """Sort of an intersection operation for clusterings."""
    labels_ab = np.stack((labels_a, labels_b), axis=1)
    ab_unique, meet_labels = np.unique(labels_ab, axis=0, return_inverse=True)
    return meet_labels


def get_main_channel_pcs(sorting, which=slice(None), rank=1, show_progress=False, dataset_name="collisioncleaned_tpca_features"):
    mask = np.zeros(len(sorting), dtype=bool)
    mask[which] = True
    channels = sorting.channels[which]

    features = np.empty((mask.sum(), rank), dtype=np.float32)
    with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
        feats_dset = h5[dataset_name]
        channel_index = h5["channel_index"][:]
        for ixs, feats in data_util.yield_masked_chunks(mask, feats_dset, show_progress=show_progress, desc_prefix="Main channel"):
            feats = feats[:, :rank]
            feats = waveform_util.grab_main_channels(feats, channels[ixs], channel_index)
            features[ixs] = feats
    return features