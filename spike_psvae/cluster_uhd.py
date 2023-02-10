from . import cluster_utils, spike_train_utils, cluster_viz
import numpy as np
import hdbscan
from sklearn.decomposition import PCA
from spike_psvae.isocut5 import isocut5 as isocut
from pathlib import Path
import matplotlib.pyplot as plt

def cluster_5_min(cluster_output_directory, raw_data_bin, geom, T_START, T_END, maxptps, x, z, spike_index, displacement_rigid, 
                  threshold_ptp=3, fs=30000, triage_quantile_cluster=100,
                  frame_dedup_cluster=20, log_c=5, scales=(1, 1, 50)):

    Path(cluster_output_directory).mkdir(exist_ok=True)
    
    fname_spike_train= Path(cluster_output_directory) / "spt_clustering_{}_{}.npy".format(T_START, T_END)
    fname_x_loc= Path(cluster_output_directory) / "x_clustering_{}_{}.npy".format(T_START, T_END)
    fname_z_loc= Path(cluster_output_directory) / "z_abs_clustering_{}_{}.npy".format(T_START, T_END)
    fname_maxptps_loc= Path(cluster_output_directory) / "maxptps_clustering_{}_{}.npy".format(T_START, T_END)
    
    idx_cluster = np.flatnonzero(np.logical_and(maxptps>threshold_ptp, 
                  np.logical_and(spike_index[:, 0]/fs>T_START, spike_index[:, 0]/fs<T_END)))

    x_cluster = x[idx_cluster]
    z_cluster = z[idx_cluster] - displacement_rigid[spike_index[idx_cluster, 0]//fs]
    maxptps_cluster = maxptps[idx_cluster]
    spike_index_cluster = spike_index[idx_cluster]

    (
        clusterer,
        cluster_centers,
        tspike_index,
        tx,
        tz,
        tmaxptps,
        idx_keep_full,
    ) = cluster_utils.cluster_spikes(
        x_cluster,
        z_cluster,
        maxptps_cluster,
        spike_index_cluster,
        triage_quantile = triage_quantile_cluster,
        do_copy_spikes=False,
        split_big=False,
        do_remove_dups=True,
        do_relabel_by_depth=False,
        log_c=log_c, 
        scales=scales
    )

    kept_ix, removed_ix = cluster_utils.remove_self_duplicates(
        tspike_index[:, 0],
        clusterer.labels_,
        raw_data_bin,
        geom.shape[0],
        frame_dedup=frame_dedup_cluster,
    )
    clusterer.labels_[removed_ix] = -1
    clusterer.labels_ = spike_train_utils.make_labels_contiguous(
        clusterer.labels_
    )

    # labels in full index space (not triaged)
    spike_train = spike_index_cluster.copy()
    spike_train[:, 1] = -1
    spike_train[idx_keep_full, 1] = clusterer.labels_

    #SPLIT TEST
    clust = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
    
    cmp = spike_train[:, 1].max()+1
    spike_train_split = spike_train.copy()
    for unit in range(spike_train[:, 1].max()+1):
        idx = spike_train[:, 1] == unit
        clust.fit(np.c_[scales[0]*x_cluster[idx], scales[1]*z_cluster[idx], scales[2]*np.log(log_c+maxptps_cluster[idx])])
        if clust.labels_.max()>-1 and (clust.labels_ == -1).sum()/len(clust.labels_)<0.5:
            for k in np.arange(1, clust.labels_.max()+1):
                which = np.where(idx)[0][clust.labels_ == k]
                spike_train_split[which, 1] = cmp
                cmp+=1
            which = np.where(idx)[0][clust.labels_ == -1]
            spike_train_split[which, 1] = -1
            
    np.save(fname_spike_train, spike_train_split)
    np.save(fname_x_loc, x_cluster)
    np.save(fname_z_loc, z[idx_cluster])
    np.save(fname_maxptps_loc, maxptps_cluster)
    

def gather_all_results_clustering(cluster_output_directory, t_start, t_end, K_LEN):

    """
    K_LEN lengths of chunks 
    """
    
    T_START = t_start
    T_END = t_start + K_LEN

    fname_spike_train=Path(cluster_output_directory) / "spt_clustering_{}_{}.npy".format(T_START, T_END)
    fname_x_loc=Path(cluster_output_directory) / "x_clustering_{}_{}.npy".format(T_START, T_END)
    fname_z_loc=Path(cluster_output_directory) / "z_abs_clustering_{}_{}.npy".format(T_START, T_END)
    fname_maxptps_loc=Path(cluster_output_directory) / "maxptps_clustering_{}_{}.npy".format(T_START, T_END)
    
    spt_all = np.load(fname_spike_train)
    max_ptps_all = np.load(fname_maxptps_loc)
    x_all = np.load(fname_x_loc)
    z_all_abs = np.load(fname_z_loc)
    
    for T_START in np.arange(t_start + K_LEN, t_end, K_LEN):
        
        T_END = T_START + K_LEN
        fname_spike_train="spt_clustering_{}_{}.npy".format(T_START, T_END)
        fname_x_loc="x_clustering_{}_{}.npy".format(T_START, T_END)
        fname_z_loc="z_abs_clustering_{}_{}.npy".format(T_START, T_END)
        fname_maxptps_loc="maxptps_clustering_{}_{}.npy".format(T_START, T_END)
            
        spt_all = np.concatenate((spt_all, np.load(fname_spike_train)))
        max_ptps_all = np.concatenate((max_ptps_all, np.load(fname_maxptps_loc)))
        x_all = np.concatenate((x_all, np.load(fname_x_loc)))
        z_all_abs = np.concatenate((z_all_abs, np.load(fname_z_loc)))
    
    return spt_all, max_ptps_all, x_all, z_all_abs



def ensemble_hdbscan_clustering(t_start, t_end, K_LEN, displacement_rigid, spt_all, max_ptps_all, x_all, z_all_abs, 
                               scales, log_c):
    
    z_reg_all = z_all_abs - displacement_rigid[spt_all[:, 0]//30000]
    
    for T_START in np.arange(t_start, t_end-K_LEN, K_LEN):
        T_END = T_START+K_LEN
        idx_1 = np.flatnonzero(np.logical_and(spt_all[:, 0]/30000>t_start, spt_all[:, 0]/30000<T_END))
        idx_2 = np.flatnonzero(np.logical_and(spt_all[:, 0]/30000>T_START+K_LEN, spt_all[:, 0]/30000<T_END+K_LEN))
        
        spt_1 = spt_all[idx_1]
        max_ptps_1 = scales[2]*np.log(log_c+max_ptps_all[idx_1])
        x_1 = scales[0]*x_all[idx_1]
        z_1_reg = scales[1]*z_reg_all[idx_1]
        
        spt_2 = spt_all[idx_2]
        max_ptps_2 = scales[2]*np.log(log_c+max_ptps_all[idx_2])
        x_2 = scales[0]*x_all[idx_2]
        z_2_reg = scales[1]*z_reg_all[idx_2]

    
        unit_label_shift = spt_1[:, 1].max()+1
        spt_2[spt_2[:, 1]>-1, 1]+=unit_label_shift

        units_1 = np.unique(spt_1[:, 1])[1:]
        
        units_2 = np.unique(spt_2[:, 1])[1:]
        dist_matrix = np.zeros((units_1.shape[0], units_2.shape[0]))

    # Speed up this code - sparse matrix
        for i in range(dist_matrix.shape[0]):
            unit_1 = units_1[i]
            for j in range(dist_matrix.shape[1]):
                unit_2 = units_2[j]
                feat_1 = np.c_[np.median(x_1[spt_1[:, 1]==unit_1]), np.median(z_1_reg[spt_1[:, 1]==unit_1]), np.median(max_ptps_1[spt_1[:, 1]==unit_1])]
                feat_2 = np.c_[np.median(x_2[spt_2[:, 1]==unit_2]), np.median(z_2_reg[spt_2[:, 1]==unit_2]), np.median(max_ptps_2[spt_2[:, 1]==unit_2])]
                dist_matrix[i, j] = ((feat_1-feat_2)**2).sum()

        dist_forward = dist_matrix.argmin(0)
        units_, counts_ = np.unique(dist_forward, return_counts=True)

        for unit_to_split in units_[counts_>1]:
            units_to_match_to = np.flatnonzero(dist_forward==unit_to_split)+unit_label_shift
            features_to_match_to = np.c_[np.median(x_2[spt_2[:, 1]==units_to_match_to[0]]), np.median(z_2_reg[spt_2[:, 1]==units_to_match_to[0]]), np.median(max_ptps_2[spt_2[:, 1]==units_to_match_to[0]])]
            for u in units_to_match_to[1:]:
                features_to_match_to = np.concatenate((features_to_match_to, 
                                                      np.c_[np.median(x_2[spt_2[:, 1]==u]), 
                                                      np.median(z_2_reg[spt_2[:, 1]==u]), 
                                                      np.median(max_ptps_2[spt_2[:, 1]==u])]))

            spikes_to_update = np.flatnonzero(spt_1[:, 1]==unit_to_split)
            x_s_to_update = x_1[spikes_to_update]
            z_s_to_update = z_1_reg[spikes_to_update]
            ptps_s_to_update = max_ptps_1[spikes_to_update]
            for j, s in enumerate(spikes_to_update):
                # Don't update if new distance is too high 
                feat_s = np.c_[x_s_to_update[j], z_s_to_update[j], ptps_s_to_update[j]]
                spt_1[s, 1] = units_to_match_to[((feat_s - features_to_match_to)**2).sum(1).argmin()]    

        # Relabel spt_1 and spt_2
        for unit_to_relabel in units_:
            if counts_[np.flatnonzero(units_==unit_to_relabel)][0]==1:
                idx_to_relabel = np.flatnonzero(spt_1[:, 1]==unit_to_relabel)
                spt_1[idx_to_relabel, 1] = np.unique(spt_2[:, 1])[1:][dist_forward == unit_to_relabel]

        #Backwards pass
        vec_new_units = np.unique(spt_1[:, 1])
        units_not_matched = vec_new_units[np.logical_and(vec_new_units>-1, vec_new_units<unit_label_shift)]
        if len(units_not_matched):
            for unit_to_split in np.unique(dist_matrix[units_not_matched].argmin(1)+unit_label_shift):
                units_to_match_to = np.concatenate((units_not_matched[dist_matrix[units_not_matched].argmin(1)+unit_label_shift==unit_to_split], 
                                                    [unit_to_split]))

                features_to_match_to = np.c_[np.median(x_1[spt_1[:, 1]==units_to_match_to[0]]), 
                                             np.median(z_1_reg[spt_1[:, 1]==units_to_match_to[0]]), 
                                             np.median(max_ptps_1[spt_1[:, 1]==units_to_match_to[0]])]
                for u in units_to_match_to[1:]:
                    features_to_match_to = np.concatenate((features_to_match_to, 
                                                          np.c_[np.median(x_1[spt_1[:, 1]==u]), 
                                                          np.median(z_1_reg[spt_1[:, 1]==u]), 
                                                          np.median(max_ptps_1[spt_1[:, 1]==u])]))
                spikes_to_update = np.flatnonzero(spt_2[:, 1]==unit_to_split)
                x_s_to_update = x_2[spikes_to_update]
                z_s_to_update = z_2_reg[spikes_to_update]
                ptps_s_to_update = max_ptps_2[spikes_to_update]
                for j, s in enumerate(spikes_to_update):
                    feat_s = np.c_[x_s_to_update[j], z_s_to_update[j], ptps_s_to_update[j]]
                    spt_2[s, 1] = units_to_match_to[((feat_s - features_to_match_to)**2).sum(1).argmin()]

        features_all_1 = np.c_[np.median(x_1[spt_1[:, 1]==np.unique(spt_1[:, 1])[1]]), 
                               np.median(z_1_reg[spt_1[:, 1]==np.unique(spt_1[:, 1])[1]]), 
                               np.median(max_ptps_1[spt_1[:, 1]==np.unique(spt_1[:, 1])[1]])]
        for u in np.unique(spt_1[:, 1])[2:]:
            features_all_1 = np.concatenate((features_all_1, 
                                                  np.c_[np.median(x_1[spt_1[:, 1]==u]), 
                                                  np.median(z_1_reg[spt_1[:, 1]==u]), 
                                                  np.median(max_ptps_1[spt_1[:, 1]==u])]))

        distance_inter = ((features_all_1[:, :, None]-features_all_1.T[None])**2).sum(1)

        labels_1_2 = np.concatenate((spt_1[:, 1], spt_2[:, 1]))
        labels_1_2 = spike_train_utils.make_labels_contiguous(
            labels_1_2
        )
        idx_all = np.flatnonzero(np.logical_and(spt_all[:, 0]/30000>t_start, spt_all[:, 0]/30000<T_END+K_LEN))
        spt_all[idx_all, 1] = labels_1_2
        
    return spt_all

def pre_deconv_split(spt_all, max_ptps_all, x_all, z_all_reg, scales, log_c=5):

    spt_after_split = spt_all.copy()
    # Try diptest split 
    pca_features = PCA(1)
    features = np.concatenate((scales[1]*z_all_reg[:, None], scales[0]*x_all[:, None], scales[2]*np.log(log_c+max_ptps_all[:, None])), axis = 1)
    cmp = spt_all[:, 1].max()+1
    for unit in range(spt_all[:, 1].max()+1):
        features_unit = features[spt_all[:, 1]==unit]
        feat_pca = pca_features.fit_transform(features_unit)
        value_dpt, cut_value = isocut(feat_pca[:, 0])
        if value_dpt>1:
            idx_to_update = np.flatnonzero(spt_all[:, 1]==unit)[feat_pca[:, 0]>cut_value]
            spt_after_split[idx_to_update, 1]=cmp
            cmp+=1

    return spt_after_split


def relabel_by_depth(spt, z_abs):
    # re-label each cluster by z-depth
    spt_ordered = spt.copy()
    cluster_centers = np.zeros(spt[:, 1].max()+1)
    for k in range(spt[:, 1].max()+1):
      cluster_centers[k] = np.median(z_abs[spt[:, 1]==k])
    indices_depth = np.argsort(-cluster_centers)
    cmp=0
    for unit in indices_depth:
        spt_ordered[spt[:, 1]==unit, 1] = cmp
        cmp+=1 
    return spt_ordered

def run_full_clustering(t_start, t_end, cluster_output_directory, raw_data_bin, geom, spike_index, 
                        localizations, maxptps, displacement_rigid, len_chunks=300, threshold_ptp=3,
                        fs=30000, triage_quantile_cluster=100, frame_dedup_cluster=20, log_c=5, scales=(1, 1, 50)):

    Path(cluster_output_directory).mkdir(exist_ok=True)
    
    print("Initial clustering")
    for T_START in np.arange(t_start, t_end, len_chunks):
        T_END = T_START+300
        cluster_5_min(cluster_output_directory, raw_data_bin, geom, T_START, T_END, maxptps, 
                      localizations[:, 0], localizations[:, 2], spike_index, displacement_rigid, 
                      threshold_ptp=threshold_ptp, fs=fs, triage_quantile_cluster=triage_quantile_cluster,
                      frame_dedup_cluster=frame_dedup_cluster, log_c=log_c, scales=scales)

    print("Ensembling")
    spt, max_ptps, x, z_abs = gather_all_results_clustering(cluster_output_directory, t_start, t_end, len_chunks)
    spt = ensemble_hdbscan_clustering(t_start, t_end, len_chunks, displacement_rigid, spt, max_ptps, x, z_abs, scales, log_c)

    print("Split")
    spt = pre_deconv_split(spt, max_ptps, x, z_abs - displacement_rigid[spt[:, 0]//30000], scales=scales, log_c=log_c)
    
    print("Relabel by Depth")
    spt = relabel_by_depth(spt, z_abs)
    
    figname = Path(cluster_output_directory) / "final_clustering_scatter_plot.png"
    fig, axes = cluster_viz.array_scatter(
        spt[:, 1], geom, x, z_abs - displacement_rigid[spt[:, 0]//30000], max_ptps,
        zlim=(-45, 325), do_ellipse=True
    )
    plt.savefig(figname)
    plt.close()
    return spt, max_ptps, x, z_abs
