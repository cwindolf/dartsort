import numpy as np
from scipy.spatial import cKDTree
import spikeinterface 
from spikeinterface.comparison import compare_two_sorters
import os
import pandas

def read_waveforms(spike_times, bin_file, geom_array, n_times=121, offset_denoiser = 42, channels=None, dtype=np.dtype('float32')):
    '''
    read waveforms from recording
    n_times : waveform temporal length 
    channels : channels to read from 
    '''
    # n_times needs to be odd
    if n_times % 2 == 0:
        n_times += 1

    # read all channels
    if channels is None:
        channels = np.arange(geom_array.shape[0])
        
    # ***** LOAD RAW RECORDING *****
    wfs = np.zeros((len(spike_times), n_times, len(channels)),
                   'float32')

    skipped_idx = []
    n_channels = geom_array.shape[0] #len(channels)
    total_size = n_times*n_channels
    # spike_times are the centers of waveforms
    spike_times_shifted = spike_times - (offset_denoiser) #n_times//2
    offsets = spike_times_shifted.astype('int64')*dtype.itemsize*n_channels
    with open(bin_file, "rb") as fin:
        for ctr, spike in enumerate(spike_times_shifted):
            try:
                fin.seek(offsets[ctr], os.SEEK_SET)
                wf = np.fromfile(fin,
                                 dtype=dtype,
                                 count=total_size)
                wfs[ctr] = wf.reshape(
                    n_times, n_channels)[:,channels]
            except:
                print(f"skipped {ctr, spike}")
                skipped_idx.append(ctr)
    wfs=np.delete(wfs, skipped_idx, axis=0)
    fin.close()

    return wfs, skipped_idx

def compute_shifted_similarity(template1, template2, shifts=[0]):
    curr_similarities = []
    for shift in shifts:
        if shift == 0:
            similarity = np.max(np.abs(template1 - template2))
        elif shift < 0:
            template2_shifted_flattened = np.pad(template2.T.flatten(),((-shift,0)), mode='constant')[:shift]
            similarity = np.max(np.abs(template1.T.flatten() - template2_shifted_flattened))
        else:    
            template2_shifted_flattened = np.pad(template2.T.flatten(),((0,shift)), mode='constant')[shift:]
            similarity = np.max(np.abs(template1.T.flatten() - template2_shifted_flattened))
        curr_similarities.append(similarity)
    return np.min(curr_similarities), shifts[np.argmin(curr_similarities)]

def get_unit_similarities(cluster_id, st_1, closest_clusters, sorting, geom_array, raw_data_bin, num_channels_similarity=20, num_close_clusters=30, shifts_align=[0], order_by ='similarity',
                          normalize_agreement_by="both"):
    waveforms1 = read_waveforms(st_1, raw_data_bin, geom_array, n_times=121)[0]
    template1 = np.mean(waveforms1, axis=0)
    original_template = np.copy(template1)
    max_ptp_channel = np.argmax(template1.ptp(0))
    max_ptp = np.max(template1.ptp(0))
    channel_range = (max(max_ptp_channel-num_channels_similarity//2,0),max_ptp_channel+num_channels_similarity//2)
    template1 = template1[:,channel_range[0]:channel_range[1]]

    similarities = []
    agreements = []
    templates = []
    shifts = []
    for closest_cluster in closest_clusters:
        if closest_cluster in sorting.get_unit_ids():
            st_2 = sorting.get_unit_spike_train(closest_cluster)
            waveforms2 = read_waveforms(st_2, raw_data_bin, geom_array, n_times=121)[0]
            template2 = np.mean(waveforms2, axis=0)[:,channel_range[0]:channel_range[1]]
            similarity, shift = compute_shifted_similarity(template1, template2, shifts_align)
            shifts.append(shift)
            similarities.append(similarity)
            # similarities.append(similarity[0][0])
            ind_st1, ind_st2, not_match_ind_st1, not_match_ind_st2 = compute_spiketrain_agreement(st_1, st_2, delta_frames=12)
            if normalize_agreement_by == "both":
                agreement = len(ind_st1) / (len(st_1) + len(st_2) - len(ind_st1))
            elif normalize_agreement_by == "first":
                agreement = len(ind_st1) / len(st_1)
            elif normalize_agreement_by == "second":
                agreement = len(ind_st1) / len(st_2)
            else:
                raise ValueError("normalize_agreement_by must be both, first, or second")
            agreements.append(agreement)
            templates.append(template2)
    agreements = np.asarray(agreements).round(2)
    similarities = np.asarray(similarities).round(2)
    closest_clusters = np.asarray(closest_clusters)
    shifts = np.asarray(shifts)
    templates = np.asarray(templates)
    
    #compute most similar units (with template similarity or spike train agreement)
    if order_by == 'similarity':
        most_similar_idxs = np.argsort(similarities) #np.flip(np.argsort(similarities))
    elif order_by == 'agreement':
        most_similar_idxs = np.flip(np.argsort(agreements))
        
    agreements = agreements[most_similar_idxs]
    similarities = similarities[most_similar_idxs]
    closest_clusters = closest_clusters[most_similar_idxs]
    templates = templates[most_similar_idxs]
    shifts = shifts[most_similar_idxs]
    
    return original_template, closest_clusters, similarities, agreements, templates, shifts

def compute_spiketrain_agreement(st_1, st_2, delta_frames=12):
    #create figure for each match
    times_concat = np.concatenate((st_1, st_2))
    membership = np.concatenate((np.ones(st_1.shape) * 1, np.ones(st_2.shape) * 2))
    indices = times_concat.argsort()
    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    diffs = times_concat_sorted[1:] - times_concat_sorted[:-1]
    inds = np.where((diffs <= delta_frames) & (membership_sorted[:-1] != membership_sorted[1:]))[0]
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
        ind_st1 = np.asarray([]).astype('int')
        ind_st2 = np.asarray([]).astype('int')
        not_match_ind_st1 = np.asarray([]).astype('int')
        not_match_ind_st2 = np.asarray([]).astype('int')
        
    return ind_st1, ind_st2, not_match_ind_st1, not_match_ind_st2

def get_agreement_indices(cluster_id_1, cluster_id_2, sorting1, sorting2, delta_frames=12):
    #code borrowed from SpikeInterface
    lab_st1 = cluster_id_1
    lab_st2 = cluster_id_2
    st_1 = sorting1.get_unit_spike_train(lab_st1)
    mapped_st = sorting2.get_unit_spike_train(lab_st2)
    times_concat = np.concatenate((st_1, mapped_st))
    membership = np.concatenate((np.ones(st_1.shape) * 1, np.ones(mapped_st.shape) * 2))
    indices = times_concat.argsort()
    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    diffs = times_concat_sorted[1:] - times_concat_sorted[:-1]
    inds = np.where((diffs <= delta_frames) & (membership_sorted[:-1] != membership_sorted[1:]))[0]

    if len(inds) > 0:
        inds2 = inds[np.where(inds[:-1] + 1 != inds[1:])[0]] + 1
        inds2 = np.concatenate((inds2, [inds[-1]]))
        times_matched = times_concat_sorted[inds2]
        # # find and label closest spikes
        ind_st1 = np.array([np.abs(st_1 - tm).argmin() for tm in times_matched])
        ind_st2 = np.array([np.abs(mapped_st - tm).argmin() for tm in times_matched])
        not_match_ind_st1 = np.ones(st_1.shape[0], bool)
        not_match_ind_st1[ind_st1] = False
        not_match_ind_st1 = np.where(not_match_ind_st1)[0]
        not_match_ind_st2 = np.ones(mapped_st.shape[0], bool)
        not_match_ind_st2[ind_st2] = False
        not_match_ind_st2 = np.where(not_match_ind_st2)[0]
        
    return ind_st1, ind_st2, not_match_ind_st1, not_match_ind_st2, st_1, mapped_st

def remove_duplicate_units(clusterer, spike_frames, maxptps):
    sorting = make_sorting_from_labels_frames(clusterer.labels_, spike_frames)
    #remove duplicates
    cmp_self = compare_two_sorters(sorting,sorting, match_score=.1, chance_score=.1)
    remove_ids = set()
    for cluster_id in sorting.get_unit_ids():
        possible_matches = cmp_self.possible_match_12[cluster_id]
        if len(possible_matches) > 1:
            mean_ptp_matches = [np.mean(maxptps[clusterer.labels_==cluster_id]) for cluster_id in possible_matches]
            remove_ids.add(possible_matches[np.argmin(mean_ptp_matches)])
    for remove_id in remove_ids:
        remove_id_indices = np.where(clusterer.labels_ == remove_id)
        clusterer.labels_[remove_id_indices] = -1
    
    #make sequential
    for i, label in enumerate(np.setdiff1d(np.unique(clusterer.labels_), [-1])):
        label_indices = np.where(clusterer.labels_ == label)
        clusterer.labels_[label_indices] = i 

    return clusterer, remove_ids

def get_closest_clusters_hdbscan(cluster_id, cluster_centers, num_close_clusters=2):
    curr_cluster_center = cluster_centers.loc[cluster_id].to_numpy()
    dist_other_clusters = np.linalg.norm(curr_cluster_center[:2] - cluster_centers.iloc[:,:2].to_numpy(), axis=1)
    closest_cluster_indices = np.argsort(dist_other_clusters)[1:num_close_clusters + 1]
    closest_clusters = cluster_centers.index[closest_cluster_indices]
    return closest_clusters

def get_closest_clusters_kilosort(cluster_id, kilo_cluster_depth_means, num_close_clusters=2):
    curr_cluster_depth = kilo_cluster_depth_means[cluster_id]
    dist_to_other_cluster_dict = {cluster_id:abs(mean_depth-curr_cluster_depth) for (cluster_id,mean_depth) in kilo_cluster_depth_means.items()}
    closest_clusters = [y[0] for y in sorted(dist_to_other_cluster_dict.items(), key = lambda x: x[1])[1:1+num_close_clusters]]
    return closest_clusters

def get_closest_clusters_hdbscan_kilosort(cluster_id, cluster_centers, kilo_cluster_depth_means, num_close_clusters=2):
    cluster_center = cluster_centers.loc[cluster_id].to_numpy()
    curr_cluster_depth = cluster_center[1]
    dist_to_other_cluster_dict = {cluster_id:abs(mean_depth-curr_cluster_depth) for (cluster_id,mean_depth) in kilo_cluster_depth_means.items()}
    closest_clusters = [y[0] for y in sorted(dist_to_other_cluster_dict.items(), key = lambda x: x[1])[:num_close_clusters]]
    return closest_clusters

def get_closest_clusters_kilosort_hdbscan(cluster_id, kilo_cluster_depth_means, cluster_centers, num_close_clusters=2):
    curr_cluster_depth = kilo_cluster_depth_means[cluster_id]
    closest_cluster_indices = np.argsort(np.abs(cluster_centers.iloc[:,1].to_numpy() - kilo_cluster_depth_means[cluster_id]))[:num_close_clusters]
    closest_clusters = cluster_centers.index[closest_cluster_indices]
    return closest_clusters
        
def compute_cluster_centers(clusterer):
    cluster_centers_data = []
    cluster_ids = np.setdiff1d(np.unique(clusterer.labels_), [-1])
    for label in cluster_ids:
        cluster_centers_data.append(clusterer.weighted_cluster_centroid(label))
    cluster_centers_data = np.asarray(cluster_centers_data)
    cluster_centers = pandas.DataFrame(data=cluster_centers_data, index=cluster_ids)
    return cluster_centers

def relabel_by_depth(clusterer, cluster_centers):
    #re-label each cluster by z-depth
    indices_depth = np.argsort(-cluster_centers.iloc[:,1].to_numpy())
    labels_depth = cluster_centers.index[indices_depth]
    label_to_id = {}
    for i, label in enumerate(labels_depth):
        label_to_id[label] = i
    label_to_id[-1] = -1
    new_labels = np.vectorize(label_to_id.get)(clusterer.labels_) 
    clusterer.labels_ = new_labels
    return clusterer

def run_weighted_triage(x, y, z, alpha, maxptps, pcs=None, scales=(1,10,1,15,30,10),threshold=100, ptp_threshold=3, c=1, ptp_weighting=True):
    ptp_filter = np.where(maxptps>ptp_threshold)
    x = x[ptp_filter]
    y = y[ptp_filter]
    z = z[ptp_filter]
    alpha = alpha[ptp_filter]
    maxptps = maxptps[ptp_filter]
    if pcs is not None:
        pcs = pcs[ptp_filter]
        feats = np.c_[scales[0]*x,
                      # scales[1]*np.log(y),
                      scales[2]*z,
                      # scales[3]*np.log(alpha),
                      # scales[4]*np.log(maxptps),
                      scales[5]*pcs[:,:3]]
    else:
        feats = np.c_[scales[0]*x,
                      # scales[1]*np.log(y),
                      scales[2]*z,
                      # scales[3]*np.log(alpha),
                      scales[4]*np.log(maxptps)]
    
    tree = cKDTree(feats)
    dist, ind = tree.query(feats, k=6)
    dist = dist[:,1:]
    # dist = np.sum((c*np.log(dist)),1)
    # print(dist)
    if ptp_weighting:
        dist = np.sum(c*np.log(dist) + np.log(1/(scales[4]*np.log(maxptps)))[:,None], 1)
    else:
        dist = np.sum((c*np.log(dist)),1)
        
    idx_keep = dist <= np.percentile(dist, threshold)
    
    triaged_x = x[idx_keep]
    triaged_y = y[idx_keep]
    triaged_z = z[idx_keep]
    triaged_alpha = alpha[idx_keep]
    triaged_maxptps = maxptps[idx_keep]
    triaged_pcs = None
    if pcs is not None:
        triaged_pcs = pcs[idx_keep]
        
    
    return triaged_x, triaged_y, triaged_z, triaged_alpha, triaged_maxptps, triaged_pcs, ptp_filter, idx_keep

def make_sorting_from_labels_frames(labels, spike_frames, sampling_frequency=30000):
    times_list = []
    labels_list = []
    for cluster_id in np.unique(labels):
        spike_train = spike_frames[np.where(labels==cluster_id)]
        times_list.append(spike_train)
        labels_list.append(np.zeros(spike_train.shape[0])+cluster_id)
    times_array = np.concatenate(times_list).astype('int')
    labels_array = np.concatenate(labels_list).astype('int')
    sorting = spikeinterface.numpyextractors.NumpySorting.from_times_labels(times_list=times_array, 
                                                                            labels_list=labels_array, 
                                                                            sampling_frequency=sampling_frequency)  
    return sorting
