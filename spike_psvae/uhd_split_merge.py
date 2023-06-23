from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
import numpy as np
from spike_psvae.isocut5 import isocut5 as isocut
from spike_psvae.drifty_deconv_uhd import superres_denoised_templates, shift_superres_templates
from spike_psvae.deconv_resid_merge import resid_dist_multiple
from scipy.cluster.hierarchy import complete, fcluster
from spike_psvae.spike_train_utils import make_labels_contiguous, clean_align_and_get_templates
from spike_psvae.waveform_utils import get_pitch, get_maxchan_traces
import hdbscan
import h5py
from tqdm.auto import tqdm, trange

def get_spread_mc_wf(h5_path, spike_index, channel_index, geom, batch_size=10000, waveform_type = "denoised", max_value_dist=70):

    """
    h5_path : deconv output
    """

    C = channel_index.shape[1]
    n_spikes = spike_index.shape[0]

    wfs_mc = np.zeros((n_spikes, 121))
    max_abs_value = np.zeros((n_spikes, C))
    spread = np.zeros(n_spikes)  
        
    n_batches = n_spikes//batch_size+1
    
    print("get {} maxchan wf".format(waveform_type))
    with h5py.File(h5_path, "r+") as h5:
        
        tpca_group = h5["{}_tpca".format(waveform_type)]
        tpca_mean = tpca_group["tpca_mean"][:]
        tpca_components = tpca_group["tpca_components"][:]

        tpca = PCA(tpca_components.shape[0])
        tpca.mean_ = tpca_mean
        tpca.components_ = tpca_components


        for k in tqdm(range(n_batches)):
            
            wfs = np.array(h5["{}_tpca_projs".format(waveform_type)][k*batch_size:(k+1)*batch_size])
            N, T, C = wfs.shape
            wfs = wfs.transpose(0, 2, 1).reshape((N*C, T))
            wfs = tpca.inverse_transform(wfs)
            wfs = wfs.reshape((N, C, 121)).transpose(0, 2, 1)
            
            max_abs_value[k*batch_size:(k+1)*batch_size] = np.abs(wfs).max(1)  
            wfs_mc[k*batch_size:(k+1)*batch_size] = get_maxchan_traces(wfs, channel_index, spike_index[k*batch_size:(k+1)*batch_size, 1])
    
    print("get spread")

    channel_distances_index = np.sqrt(((np.pad(geom, [[0, 1], [0, 0]],mode='constant',constant_values=np.nan)[channel_index]-geom[:, None])**2).sum(2))

    for k in tqdm(range(n_spikes)):
        max_chan = spike_index[k, 1]
        channels_effective = np.flatnonzero(channel_index[max_chan]<384)
        channels_effective = channels_effective[channel_distances_index[max_chan][channels_effective]<max_value_dist]
        spread[k] = np.nansum(max_abs_value[k, channels_effective] * channel_distances_index[max_chan][channels_effective])/np.nansum(max_abs_value[k, channels_effective])

    return wfs_mc, spread


# Split 

def uhd_split(spt, wfs_mc, spread, x, z_reg, min_cluster_size=25, cluster_selection_epsilon=25):

    labels_split = spt[:, 1].copy()
    cmp_all = spt[:, 1].max()+1
    pc_mc = PCA(n_components=2, whiten=True)

    for unit in tqdm(range(spt[:, 1].max()+1)):
        
        in_unit = np.flatnonzero(spt[:, 1]==unit)
        if len(in_unit)>min_cluster_size:
            pcs_unit = pc_mc.fit_transform(wfs_mc[in_unit])

            # pcs_new_mc normalized
            rescale_spread = x[in_unit].std()/spread[in_unit].std()
            rescale_pcs = x[in_unit].std()/pcs_unit[:, 0].std()
            rescale_pcs_2 = x[in_unit].std()/pcs_unit[:, 1].std()
            rescale_z = x[in_unit].std()/z_reg[in_unit].std()

            features = np.c_[x[in_unit], rescale_z*z_reg[in_unit], rescale_pcs*pcs_unit[:, 0], rescale_pcs_2*pcs_unit[:, 1], rescale_spread*spread[in_unit]]
            clusterer=hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon)
            clusterer.fit(features)
            labels = clusterer.labels_.copy()

            if labels.max()>0:
                reiterate = True
                while reiterate:
                    reiterate_bis=False
                    labels_max = labels.max()+1
                    cmp = labels.max()+1
                    values_labels = np.unique(labels[labels>-1])
                    for k in values_labels:
                        idx_k_ = np.flatnonzero(labels==k)
                        in_unit_bis = in_unit[idx_k_]

                        pcs_mc_bis = pc_mc.fit_transform(wfs_mc[in_unit_bis])

                        rescale_pcs = x[in_unit_bis].std()/pcs_mc_bis[:, 0].std()
                        rescale_pcs_2 = x[in_unit_bis].std()/pcs_mc_bis[:, 1].std()
                        rescale_z = x[in_unit_bis].std()/z_reg[in_unit_bis].std()
                        rescale_spread = x[in_unit_bis].std()/spread[in_unit_bis].std()
                        features_bis = np.c_[x[in_unit_bis], rescale_z*z_reg[in_unit_bis], rescale_pcs*pcs_mc_bis[:, 0], rescale_pcs_2*pcs_mc_bis[:, 1], rescale_spread*spread[in_unit_bis]]
                        clusterer.fit(features_bis)
                        if clusterer.labels_.max()>0:
                            reiterate_bis=True
                            idx_labels = idx_k_[clusterer.labels_>=0]
                            labels[idx_labels] = clusterer.labels_[clusterer.labels_>=0]+cmp
                            idx_labels = idx_k_[clusterer.labels_==-1]
                            labels[idx_labels] = -1
                            cmp+=clusterer.labels_.max()
                    if not reiterate_bis:
                        reiterate=False

                idx_outliers = in_unit[labels==-1]
                labels_split[idx_outliers] = -1
                idx_no_outliers = in_unit[labels>-1]
                labels_split[idx_no_outliers] = labels[labels>-1] + cmp_all
                cmp_all = labels_split.max()+1
                
        else:
            labels_split[in_unit] = -1

    labels_split = make_labels_contiguous(labels_split)

    return labels_split

def template_deconv_merge(spt, labels_split, z_abs, z_reg, x, geom, raw_data_bin, threshold_resid=0.25, su_chan_vis=1.5, bin_size_um=None, zero_radius_um=70, n_jobs=-1, sampling_rate=30000, dist_proposed_pairs=None):

    labels_merge = labels_split.copy()
    
    pitch = get_pitch(geom)
    if bin_size_um is None:
        bin_size_um = pitch//2
    print("BIN SIZE : " + str(bin_size_um))
        
    if dist_proposed_pairs is None:
        dist_proposed_pairs=pitch

    n_units = labels_split.max()+1
    std_z = np.zeros(n_units)
    std_x = np.zeros(n_units)
    for k in range(n_units):
        idx_k = np.flatnonzero(labels_split==k)
        std_z[k] = z_reg[idx_k].std()
        std_x[k] = x[idx_k].std()

    std_x = np.minimum(std_x, pitch*2)
    std_z = np.minimum(std_z, pitch*2)

    med_position_units = np.zeros((labels_split.max()+1, 2))
    for k in range(len(med_position_units)):
        in_unit = np.flatnonzero(labels_split==k)
        med_position_units[k, 0] = np.median(x[in_unit])
        med_position_units[k, 1] = np.median(z_reg[in_unit])
    dist_matrix = np.sqrt(((med_position_units[None] - med_position_units[:, None])**2).sum(2))

    units_1 = np.where(dist_matrix<dist_proposed_pairs)[0]
    units_2 = np.where(dist_matrix<dist_proposed_pairs)[1]
    idx = units_1!=units_2
    units_1 = units_1[idx]
    units_2 = units_2[idx]
    n_pairs = len(units_1)
    matrix_all_distances = 100_000*np.ones((dist_matrix.shape)) #drop in huge value here


    print("Get superres temps...")
    (
        superres_templates,
        superres_label_to_bin_id,
        superres_label_to_orig_label,
        medians_at_computation,
        n_spikes_per_bin
    ) = superres_denoised_templates(
        np.c_[spt[labels_split>=0, 0], labels_split[labels_split>=0]],
        np.c_[spt[labels_split>=0, 0], labels_split[labels_split>=0]],
        z_abs[labels_split>=0],
        x[labels_split>=0], 
        z_abs[labels_split>=0],
        x[labels_split>=0], 
        bin_size_um, # Important param - might need to be tuned for other probes (NP 1,2 etc...)
        geom,
        raw_data_bin,
        reducer=np.median,
        zero_radius_um=zero_radius_um, # maybe pass it as param input 
        augment_low_snr_temps=True,
        units_spread=std_z*1.65,
        units_x_spread=std_x*1.65,
        t_end=int(spt[:, 0].max()//sampling_rate),
        dist_metric=1000*np.ones(len(labels_split[labels_split>=0])), #Use all spikes here - higher than dist_metric_threshold
        dist_metric_threshold=500,
        n_jobs=n_jobs,
    )
    
    print("Get distance matrix...")
    for k in tqdm(range(n_pairs)):

        unit_1 = units_1[k]
        unit_2 = units_2[k]
        
        idx_temp_1 = superres_label_to_orig_label==unit_1
        bins_id_1 = superres_label_to_bin_id[idx_temp_1]
        all_temps_1 = superres_templates[idx_temp_1]
        med_1 = medians_at_computation[unit_1]

        idx_temp_2 = superres_label_to_orig_label==unit_2
        bins_id_2 = superres_label_to_bin_id[idx_temp_2]
        all_temps_2 = superres_templates[idx_temp_2]
        med_2 = medians_at_computation[unit_2]

        bins_to_compute_diff = np.intersect1d(bins_id_1, bins_id_2)
        if len(bins_to_compute_diff):
            shift_amount = med_1-med_2

            shifted_temp_1 = all_temps_1[np.isin(bins_id_1, bins_to_compute_diff)]
            shifted_temp_2 = shift_superres_templates(all_temps_2[np.isin(bins_id_2, bins_to_compute_diff)],
                                                      bins_to_compute_diff, np.zeros(len(bins_to_compute_diff), dtype=int),
                                                      bin_size_um, geom, shift_amount, [med_2], [med_2])

            good_channels = np.intersect1d(np.flatnonzero(shifted_temp_1.ptp(1).min(0)>su_chan_vis), 
                                       np.flatnonzero(shifted_temp_2.ptp(1).min(0)>su_chan_vis))

            if len(good_channels):
                bin_rms = np.argmin(np.abs(bins_to_compute_diff))
                rms_a = np.sqrt(np.square(shifted_temp_1[bin_rms, 30:60, good_channels]).sum()/(np.abs(shifted_temp_1[bin_rms, 30:60, good_channels]) > 0).sum())
                dist, shift = resid_dist_multiple(shifted_temp_1[:, :, good_channels], shifted_temp_2[:, :, good_channels])
                if dist>=0:
                #RMS
                    matrix_all_distances[unit_1, unit_2] = dist/rms_a


    matrix_all_distances_sym = np.maximum(matrix_all_distances, matrix_all_distances.T) 
    pdist = matrix_all_distances_sym[np.triu_indices(matrix_all_distances_sym.shape[0], k=1)]
    Z = complete(pdist)
    new_labels = fcluster(Z, threshold_resid, criterion="distance")
    new_labels = new_labels-1

    labels_merge[labels_split>-1] = new_labels[labels_split[labels_split>-1]]

    return labels_merge



def run_full_merge_split(h5_path, spt, spike_index, 
    channel_index, geom, raw_data_bin,
    z_abs, z_reg, x,
    batch_size=10000, waveform_type = "denoised", max_value_dist=70, 
    min_cluster_size=50, cluster_selection_epsilon=25, 
    threshold_resid=0.25, su_chan_vis=1.5, 
    bin_size_um=None, n_jobs=-1):
    
    pitch = get_pitch(geom)
    if bin_size_um is None:
        bin_size_um = pitch//2

    wfs_mc, spread = get_spread_mc_wf(h5_path, spike_index, channel_index, geom, batch_size=batch_size, waveform_type = waveform_type, max_value_dist=max_value_dist)
    print("Split...")
    labels_split = uhd_split(spt, wfs_mc, spread, x, z_reg, min_cluster_size=min_cluster_size, cluster_selection_epsilon=cluster_selection_epsilon)
    #align
    print("Align...")
    (
        spike_train_split,
        order,
        templates_split,
        template_shifts,
    ) = clean_align_and_get_templates(
        np.c_[spt[:, 0], labels_split],
        geom.shape[0],
        raw_data_bin,
        sort_by_time=False,
        min_n_spikes=min_cluster_size, # rewrite it as FR min value 
        trough_offset=42,
        spike_length_samples=121,
    )
    print("Merge...")
    labels_merge = template_deconv_merge(spt, labels_split, z_abs, z_reg, x, geom, raw_data_bin, threshold_resid=threshold_resid, su_chan_vis=su_chan_vis, bin_size_um=bin_size_um, n_jobs=n_jobs)

    return make_labels_contiguous(labels_merge)


def split_half_z(spt, z_reg, th_diptest=1):
    
    
    # Iterative
    # Reassignment

    labels_split = spt[:, 1].copy()
    cmp = spt[:, 1].max()+1
    for unit in tqdm(np.unique(spt[:, 1])):
        in_unit = np.flatnonzero(spt[:, 1]==unit)
        dipscore, cutpoint = isocut5(z_reg[in_unit])
        if dipscore>th_diptest:
            labels_split[in_unit[z_reg[in_unit]>cutpoint]] = cmp
            cmp+=1
    return labels_split




        
