import numpy as np
from .cluster_util import hdbscan_clustering
from tqdm.auto import trange

def ensembling_hdbscan(recording,
    times_seconds, 
    times_samples, 
    x, 
    z_abs, 
    geom, 
    amps,
    clustering_config,
    motion_est=None,
):
    """
    Ensemble over HDBSCAN clustering
    triaging/subsampling/copying/splitting big clusters not implemented since we don't use it (so far)
    """
    chunk_size_s = clustering_config.chunk_size_s
    min_cluster_size = clustering_config.min_cluster_size
    min_samples = clustering_config.min_samples
    cluster_selection_epsilon = clustering_config.cluster_selection_epsilon 
    scales = clustering_config.feature_scales
    log_c = clustering_config.log_c
    n_chunks = int((times_seconds.max() - times_seconds.min())// chunk_size_s)
    leftover_time = int(times_seconds.max())-chunk_size_s*n_chunks
    #if last chunk is at least 2/3 of chunk size, still ensemble
    if leftover_time>chunk_size_s*2/3:
        n_chunks+=1     
    # TODO: extend to overlapping bins
    # shift = (int(times_seconds.max())-chunk_size_s) // n_chunks
    # n_chunks+=1
    if n_chunks == 0 or n_chunks == 1:
        return hdbscan_clustering(recording,
            times_seconds, times_samples, x, z_abs, geom, amps, motion_est, min_cluster_size, min_samples, 
            cluster_selection_epsilon, scales, log_c,
        )
    else:
        min_time_s = int(times_seconds.min())
        labels_all_chunks = []
        idx_all_chunks = []
        labels_all = -1*np.ones(times_seconds.shape[0])
        for k in  trange(n_chunks, desc="Per-chunk clustering"):
            idx_chunk = np.flatnonzero(np.logical_and(times_seconds>=min_time_s+k*chunk_size_s, times_seconds<min_time_s+(k+1)*chunk_size_s))
            # idx_chunk = np.flatnonzero(np.logical_and(times_seconds>=min_time_s+k*shift, times_seconds<min_time_s+k*shift+chunk_size_s))
            idx_all_chunks.append(idx_chunk)
            labels_chunk = hdbscan_clustering(recording,
                times_seconds[idx_chunk], times_samples[idx_chunk], x[idx_chunk], z_abs[idx_chunk], geom, amps[idx_chunk], motion_est, 
                min_cluster_size, min_samples, 
                cluster_selection_epsilon, scales, log_c,
            )
            _, labels_chunk[labels_chunk>-1] = np.unique(labels_chunk[labels_chunk>-1], return_inverse=True)
            labels_all_chunks.append(labels_chunk.astype('int'))
            # if k == 0:
            labels_all[idx_chunk] = labels_chunk
        
        z_reg = motion_est.correct_s(times_seconds, z_abs)
         
        labels_all = labels_all.astype('int')
        
        for k in  trange(n_chunks-1, desc="Ensembling chunks"):
            
            #CHANGE THE 1 ---
            # idx_1 = np.flatnonzero(np.logical_and(times_seconds>=min_time_s, times_seconds<min_time_s+k*shift+chunk_size_s))
            idx_1 = np.flatnonzero(np.logical_and(times_seconds>=min_time_s, times_seconds<min_time_s+(k+1)*chunk_size_s))
            idx_2 = idx_all_chunks[k+1]
            x_1 = scales[0]*x[idx_1]
            x_2 = scales[0]*x[idx_2]
            z_1 = scales[1]*z_reg[idx_1]
            z_2 = scales[1]*z_reg[idx_2]
            amps_1 = scales[2]*np.log(log_c+amps[idx_1])
            amps_2 = scales[2]*np.log(log_c+amps[idx_2])
            labels_1 = labels_all[idx_1].copy().astype('int')
            labels_2 = labels_all_chunks[k+1].copy()
            unit_label_shift = int(labels_1.max()+1)
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
                        np.c_[np.median(x_2[labels_2==u]), 
                        np.median(z_2[labels_2==u]), 
                        np.median(amps_2[labels_2==u])])
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
                    z_s_to_update = z_2[spikes_to_update]
                    amps_s_to_update = amps_2[spikes_to_update]
                    for j, s in enumerate(spikes_to_update):
                        feat_s = np.c_[x_s_to_update[j], z_s_to_update[j], amps_s_to_update[j]]
                        labels_2[s] = units_to_match_to[((feat_s - features_to_match_to)**2).sum(1).argmin()]

#           Do we need to "regularize" and make sure the distance intra units after merging is smaller than the distance inter units before merging 
            all_labels_1 = np.unique(labels_1)
            all_labels_1 = all_labels_1[all_labels_1>-1]
            
            features_all_1 = np.c_[np.median(x_1[labels_1==all_labels_1[0]]), #WHY [1]?
np.median(z_1[labels_1==all_labels_1[0]]), 
np.median(amps_1[labels_1==all_labels_1[0]])]
            for u in all_labels_1[1:]:
                features_all_1 = np.concatenate((features_all_1, np.c_[np.median(x_1[labels_1==u]), np.median(z_1[labels_1==u]), np.median(amps_1[labels_1==u])]))

            distance_inter = ((features_all_1[:, :, None]-features_all_1.T[None])**2).sum(1)
            
            labels_12 = np.concatenate((labels_1, labels_2))
            _, labels_12[labels_12>-1] = np.unique(labels_12[labels_12>-1], return_inverse=True) #Make contiguous
            idx_all = np.flatnonzero(times_seconds<min_time_s+(k+2)*chunk_size_s)
            labels_all = -1*np.ones(times_seconds.shape[0]) #discard all spikes at the end for now
            labels_all[idx_all] = labels_12.astype('int')
        
    return labels_all


def get_indices_in_chunk(times_s, chunk_time_range_s):
    if chunk_time_range_s is None:
        in_chunk = slice(None)
    else:
        in_chunk = np.where((times_s >= chunk_time_range_s[0]) & (times_s < chunk_time_range_s[1]))[0]
    return in_chunk