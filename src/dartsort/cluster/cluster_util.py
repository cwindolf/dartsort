import numpy as np
from dartsort.util import drift_util
from dredge.motion_util import IdentityMotionEstimate
from scipy.spatial import KDTree
from sklearn.neighbors import KNeighborsClassifier


def closest_registered_channels(
    times_seconds, x, z_abs, geom, motion_est=None
):
    """Assign spikes to the drift-extended channel closest to their registered position"""
    if motion_est is None:
        motion_est == IdentityMotionEstimate()
    registered_geom = drift_util.registered_geometry(geom, motion_est)
    z_reg = motion_est.correct_s(times_seconds, z_abs)
    reg_pos = np.c_[x, z_reg]

    registered_kdt = KDTree(registered_geom)
    distances, reg_channels = registered_kdt.query(reg_pos)

    return reg_channels

def ensembling_hdbscan(
    times_seconds, x, z_abs, geom, amps, motion_est=None,
    chunk_size_s=300, 
    min_cluster_size=25,
    min_samples=25,
    cluster_selection_epsilon=15, 
    scales=(1, 1, 50),
    log_c=5,
    threshold_ptp_cluster=3,
):
    """
    Ensemble over HDBSCAN clustering
    triaging/subsampling/copying/splitting big clusters not implemented since we don't use it (so far)
    """
    
    n_chunks = (times_seconds.max() - times_seconds.min())// chunk_size_s
    if n_chunks == 0 or n_chunks == 1:
        return hdbscan_clustering(
            times_seconds, x, z_abs, geom, amps, motion_est, min_cluster_size, min_samples, 
            cluster_selection_epsilon, scales, log_c, threshold_ptp_cluster,
        )
    else:
        min_time_s = times_seconds.min()
        labels_all_chunks = []
        idx_all_chunks = []
        for k in range(n_chunks):
            idx_chunk = np.flatnonzero(np.logical_and(times_seconds>=min_time_s+k*chunk_size_s, times_seconds<min_time_s+(k+1)*chunk_size_s))
            idx_all_chunks.append(idx_chunk)
            labels_chunk = hdbscan_clustering(
                times_seconds, x, z_abs, geom, amps, motion_est, min_cluster_size, min_samples, 
                cluster_selection_epsilon, scales, log_c, threshold_ptp_cluster,
            )
            labels_all_chunks.append(labels_chunk)
        
        z_reg = motion_est.correct_s(times_seconds, z_abs)
        
        for k in range(n_chunks-1):
            
            #CHANGE THE 1 ---
            idx_1 = np.flatnonzero(np.logical_and(times_seconds>=min_time_s, times_seconds<min_time_s+(k+1)*chunk_size_s))
            idx_2 = idx_all_chunks[k+1]
            x_1 = scales[0]*x[idx_1]
            x_2 = scales[0]*x[idx_2]
            z_1 = scales[1]*z_reg[idx_1]
            z_2 = scales[1]*z_reg[idx_2]
            amps_1 = scales[2]*np.log(log_c+amps[idx_1])
            amps_2 = scales[2]*np.log(log_c+amps[idx_2])
            labels_1 = labels_chunk[k]
            labels_2 = labels_chunk[k+1]
            unit_label_shift = labels_1.max()+1
            labels_2[labels_2>-1]+=unit_label_shift

            units_1 = np.unique(labels_1)
            units_1 = units_1[units_1>-1]
            units_2 = np.unique(labels_2)
            units_2 = units_2[units_2>-1]
            
            # FORWARD PASS
            
            dist_matrix = np.zeros((units_1.shape[0], units_2.shape[0]))
            
            # Speed up this code - this matrix can be sparse (only compute distance for "neighboring" units) - OK for now, still pretty fast
            for i in range(units_1.shape[0]):
                unit_1 = units_1[i]
                for j in range(units_2.shape[0]):
                    unit_2 = units_2[j]
                    feat_1 = np.c_[np.median(x_1[labels_1==unit_1]), np.median(z_1[labels_1==unit_1]), np.median(amps_1[labels_1==unit_1])]
                    feat_2 = np.c_[np.median(x_2[labels_2==unit_2]), np.median(z_2[labels_2==unit_2]), np.median(amps_2[labels_2==unit_2])]
                    dist_matrix[i, j] = ((feat_1-feat_2)**2).sum()

            # find for chunk 2 units the closest units in chunk 1 and split chunk 1 units
            dist_forward = dist_matrix.argmin(0) 
            units_, counts_ = np.unique(dist_forward, return_counts=True)
            
            for unit_to_split in units_[counts_>1]:
                units_to_match_to = np.flatnonzero(dist_forward==unit_to_split)+unit_label_shift
                features_to_match_to = np.c_[np.median(x_2[labels_2==units_to_match_to[0]]), np.median(z_2[labels_2==units_to_match_to[0]]), np.median(amps_2[labels_2==units_to_match_to[0]])]
                for u in units_to_match_to[1:]:
                    features_to_match_to = np.concatenate((features_to_match_to, 
                        np.c_[np.median(x_2[spt_2[:, 1]==u]), 
                        np.median(z_2_reg[spt_2[:, 1]==u]), 
                        np.median(max_ptps_2[spt_2[:, 1]==u])])
                    )
                spikes_to_update = np.flatnonzero(labels_1==unit_to_split)
                x_s_to_update = x_1[spikes_to_update]
                z_s_to_update = z_1[spikes_to_update]
                amps_s_to_update = amps_1[spikes_to_update]
                for j, s in enumerate(spikes_to_update):
                    # Don't update if new distance is too high? 
                    feat_s = np.c_[x_s_to_update[j], z_s_to_update[j], amps_s_to_update[j]]
                    labels_1[s] = units_to_match_to[((feat_s - features_to_match_to)**2).sum(1).argmin()]    

            # Relabel labels_1 and labels_2
            for unit_to_relabel in units_:
                if counts_[np.flatnonzero(units_==unit_to_relabel)][0]==1:
                    idx_to_relabel = np.flatnonzero(labels_1==unit_to_relabel)
                    labels_1[idx_to_relabel] = units_2[dist_forward == unit_to_relabel]

            # BACKWARD PASS
            
            units_not_matched = np.unique(labels_1)
            units_not_matched = units_not_matched[units_not_matched>-1]
            units_not_matched = units_not_matched[units_not_matched<unit_label_shift]
            
            if len(units_not_matched):
                all_units_to_match_to = dist_matrix[units_not_matched].argmin(1)+unit_label_shift 
                for unit_to_split in np.unique(all_units_to_match_to):
                    units_to_match_to = np.concatenate((units_not_matched[all_units_to_match_to==unit_to_split], [unit_to_split]))

                    features_to_match_to = np.c_[np.median(x_1[labels_1==units_to_match_to[0]]), 
                                                 np.median(z_1[labels_1==units_to_match_to[0]]), 
                                                 np.median(amps_1[labels_1==units_to_match_to[0]])]
                    for u in units_to_match_to[1:]:
                        features_to_match_to = np.concatenate((features_to_match_to, 
                            np.c_[np.median(x_1[labels_1==u]), 
                            np.median(z_1[labels_1==u]), 
                            np.median(amps_1[labels_1==u])])
                        )
                    spikes_to_update = np.flatnonzero(labels_2==unit_to_split)
                    x_s_to_update = x_2[spikes_to_update]
                    z_s_to_update = z_2_reg[spikes_to_update]
                    amps_s_to_update = amps_2[spikes_to_update]
                    for j, s in enumerate(spikes_to_update):
                        feat_s = np.c_[x_s_to_update[j], z_s_to_update[j], amps_s_to_update[j]]
                        labels_2[s] = units_to_match_to[((feat_s - features_to_match_to)**2).sum(1).argmin()]

#           Do we need to "regularize" and make sure the distance intra units after merging is smaller than the distance inter units before merging 
#           all_labels_1 = np.unique(labels_1)
#           all_labels_1 = all_labels_1[all_labels_1>-1]
#           features_all_1 = np.c_[np.median(x_1[labels_1==all_labels_1[0]]), #WHY [1]?
#                                  np.median(z_1[labels_1==all_labels_1[0]]), 
#                                  np.median(amps_1[labels_1==all_labels_1[0]])]
#           for u in all_labels_1[1:]:
#               features_all_1 = np.concatenate((features_all_1, 
#                                                     np.c_[np.median(x_1[labels_1==u]), 
#                                                     np.median(z_1_reg[labels_1==u]), 
#                                                     np.median(amps_1[labels_1==u])]))

#           distance_inter = ((features_all_1[:, :, None]-features_all_1.T[None])**2).sum(1)

            labels_12 = np.concatenate((labels_1, labels_2))
            labels_12 = spike_train_utils.make_labels_contiguous(
                labels_12
            )
            idx_all = np.flatnonzero(times_seconds<times_seconds.min()+n_chunks*chunk_size_s)
            labels_all = -1*np.ones(times_seconds.shape[0])
            labels_all[idx_all] = labels_1_2
        
    return labels_all




def hdbscan_clustering(
    times_seconds, x, z_abs, geom, amps, motion_est=None,
    min_cluster_size=25,
    min_samples=25,
    cluster_selection_epsilon=15, 
    scales=(1, 1, 50),
    log_c=5,
    threshold_ptp_cluster=3,
):
    """
    Run HDBSCAN
    triaging/subsampling/copying/splitting big clusters not implemented since we don't use it (so far)
    """
    if motion_est is None:
        motion_est == IdentityMotionEstimate()
    
    z_reg = motion_est.correct_s(times_seconds, z_abs)
    features = np.c_[x * scales[0], z_reg * scales[1], np.log(log_c + maxptps) * scales[2]]
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        min_samples=min_samples,
        core_dist_n_jobs=-1,
    )
    clusterer.fit(features)
    
    # Implement deduplication here cluster_utils.remove_duplicate_spikes then cluster_utils.remove_self_duplicates
    # How to deal with outliers?
    
    return clusterer.labels_


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
    binary_file,
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

    for unit in tqdm(unit_labels, desc="Remove self violations"):
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
            print("super contaminated unit.")
            indices_to_remove.extend(in_unit)

        elif violations.any():
            print(f"{unit=} {in_unit.size=} {violations.mean()=}")
            # we'll remove either an index in first_viol_ix,
            # or that index + 1, depending on template agreement
            first_viol_ix = np.flatnonzero(violations)
            all_viol_ix = np.concatenate(
                [first_viol_ix, [first_viol_ix[-1] + 1]]
            )
            unviol = np.setdiff1d(
                np.arange(spike_times_unit.shape[0]), all_viol_ix
            )

            # load as many unviolated wfs as possible
            if unviol.size > n_samples:
                # we can compute template just from unviolated wfs
                which_unviol = rg.choice(unviol.size, n_samples, replace=False)
                which_unviol.sort()
                wfs_unit, _ = read_waveforms(
                    spike_times_unit[which_unviol], binary_file, n_channels
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
                wfs_unit, _ = read_waveforms(
                    spike_times_unit[load_ix], binary_file, n_channels
                )

            # reshape to NxT
            template = np.median(wfs_unit, axis=0)
            mc = template.ptp(0).argmax()
            template_mc = template[:, mc]
            template_argmin = np.abs(template_mc).argmax()

            # get subsets of wfs -- will we remove leading (wfs_1)
            # or trailing (wfs_2) waveform in each case?
            wfs_1, _ = read_waveforms(
                spike_times_unit[first_viol_ix],
                binary_file,
                n_channels,
                channels=[mc],
            )
            wfs_1 = wfs_1[:, :, 0]
            wfs_2, _ = read_waveforms(
                spike_times_unit[first_viol_ix + 1],
                binary_file,
                n_channels,
                channels=[mc],
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
    # normalize agreement by smaller unit and then remove only the spikes with agreement
    sorting = make_sorting_from_labels_frames(clusterer.labels_, spike_frames)
    # remove duplicates
    cmp_self = compare_two_sorters(
        sorting, sorting, match_score=0.1, chance_score=0.1
    )
    removed_cluster_ids = set()
    remove_spikes = []
    for cluster_id in tqdm(sorting.get_unit_ids(), desc="Remove pair dups"):
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
                remove_spikes.append(
                    (remove_cluster_id, possible_matches, remove_ind)
                )
                removed_cluster_ids.add(remove_cluster_id)

    remove_indices_list = []
    for cluster_id, _, spike_indices in remove_spikes:
        in_unit = np.flatnonzero(clusterer.labels_ == cluster_id)
        remove_indices = in_unit[spike_indices]
        clusterer.labels_[remove_indices] = -1
        remove_indices_list.append(remove_indices)

    # make contiguous
    clusterer.labels_ = make_labels_contiguous(clusterer.labels_)

    return clusterer, remove_indices_list, remove_spikes

