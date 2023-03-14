from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
import numpy as np
from spike_psvae.isocut5 import isocut5 as isocut
from spike_psvae.drifty_deconv_uhd import superres_denoised_templates, shift_superres_templates
from spike_psvae.deconv_resid_merge import resid_dist_multiple
from scipy.cluster.hierarchy import complete, fcluster


def correct_outliers(spt, x, z_reg, disp, units_to_clean = None, prob_min = 0.1):
    
    labels_outliers_corrected = spt[:, 1].copy()
    if units_to_clean is None:
        units_to_clean = np.unique(spt[:, 1])
    for k in units_to_clean:
        idx_k = np.flatnonzero(spt[:, 1]==k)
        features = np.concatenate((x[idx_k][None], z_reg[idx_k][None]), axis=0)
        features = features.T
        GMM_model = GMM(covariance_type='diag')
        GMM_model.fit(features)

        probabilities = GMM_model.score_samples(features)
        probabilities = np.exp(probabilities)
        idx_sort_prob = probabilities.argsort()
        probabilities_sorted = probabilities[idx_sort_prob]
        cum_sum_prob = np.cumsum(probabilities_sorted)
        cum_sum_prob /= cum_sum_prob.max()
        
        idx_quantile = np.where(cum_sum_prob > prob_min)[0].min()

        idx_outliers = idx_sort_prob[:idx_quantile]
        idx_outliers_dist = idx_k[idx_outliers]
        
        labels_outliers_corrected[idx_outliers_dist]=-1
    return labels_outliers_corrected


def post_deconv_split(spt_no_outliers, x, z_reg, pcs, isosplit_th=1, n_iter=2):
    for iter in range(n_iter):
        cmp = spt_no_outliers[:, 1].max()+1
        for unit in range(spt_no_outliers[:, 1].max()+1):
            idx_unit = np.flatnonzero(spt_no_outliers[:, 1]==unit)
            spread_0 = x[idx_unit].std()/pcs[idx_unit, 0].std()
            spread_1 = x[idx_unit].std()/pcs[idx_unit, 1].std()
            features = np.concatenate((z_reg[idx_unit, None], x[idx_unit, None], 
                                      spread_0*pcs[idx_unit, 0][:, None], 
                                       spread_1*pcs[idx_unit, 1][:, None]), axis=1)

            pca_features = PCA(1)
            feat_pca = pca_features.fit_transform(features)[:, 0]
            prob, cut_value = isocut(feat_pca)
            if prob>isosplit_th:
                idx_bigger = idx_unit[feat_pca>cut_value]
                spt_no_outliers[idx_bigger, 1]=cmp
                cmp+=1
    return spt_no_outliers


def post_deconv_merge(raw_data_bin, geom, spt_no_outliers, 
                      z_abs, z_reg, x, time_computation, spread_z,
                      dist_pairs=20, resid_threshold=10, deconv_th=1000,
                      bin_size_um=1, pfs=30000):
    
    (
        superres_templates_no_augment,
        superres_label_to_bin_id,
        superres_label_to_orig_label,
        medians_at_computation,
    ) = superres_denoised_templates(
        spt_no_outliers,
        z_abs,
        x,
        bin_size_um,
        geom,
        raw_data_bin,
        t_end=time_computation,
        augment_low_snr_temps=False,
        units_spread=spread_z,
        dist_metric=1000*np.ones(len(spt_no_outliers)), #Use all spikes here
        dist_metric_threshold=500,
    )

    med_position_units = np.zeros((spt_no_outliers[:, 1].max()+1, 2))
    for k in range(len(med_position_units)):
        idx_unit = np.flatnonzero(spt_no_outliers[:, 1]==k)
        med_position_units[k, 0] = np.median(x[idx_unit])
        med_position_units[k, 1] = np.median(z_reg[idx_unit])
    dist_matrix = ((med_position_units[None] - med_position_units[:, None])**2).sum(2)
    
    units_1 = np.where(dist_matrix<dist_pairs)[0]
    units_2 = np.where(dist_matrix<dist_pairs)[1]
    idx = units_1!=units_2
    units_1 = units_1[idx]
    units_2 = units_2[idx]
    n_pairs = len(units_1)
    
    matrix_all_distances = 100_000*np.ones((dist_matrix.shape)) #drop in huge value here
    
    for k in range(n_pairs):
        unit_1 = units_1[k]
        unit_2 = units_2[k]

        idx_temp_1 = superres_label_to_orig_label==unit_1
        bins_id_1 = superres_label_to_bin_id[idx_temp_1]
        all_temps_1 = superres_templates_no_augment[idx_temp_1]
        med_1 = medians_at_computation[unit_1]

        idx_temp_2 = superres_label_to_orig_label==unit_2
        bins_id_2 = superres_label_to_bin_id[idx_temp_2]
        all_temps_2 = superres_templates_no_augment[idx_temp_2]
        med_2 = medians_at_computation[unit_2]

        bins_to_compute_diff = np.intersect1d(bins_id_1, bins_id_2)
        shift_amount = med_1-med_2

        shifted_temp_1 = all_temps_1[np.isin(bins_id_1, bins_to_compute_diff)]
        shifted_temp_2 = shift_superres_templates(all_temps_2[np.isin(bins_id_2, bins_to_compute_diff)],
                                                  bins_to_compute_diff, np.zeros(len(bins_to_compute_diff), dtype=int),
                                                  bin_size_um, geom, shift_amount, [med_2], [med_2])

        good_channels = np.intersect1d(np.flatnonzero(shifted_temp_1.ptp(1).min(0)>1.5), 
                                   np.flatnonzero(shifted_temp_2.ptp(1).min(0)>1.5))

        if len(good_channels):

            dist, shift = resid_dist_multiple(shifted_temp_1[:, :, good_channels], shifted_temp_2[:, :, good_channels], deconv_th)
            if dist>=0:
                matrix_all_distances[unit_1, unit_2] = dist

    matrix_all_distances = np.minimum(matrix_all_distances, matrix_all_distances.T) 
    pdist = matrix_all_distances[np.triu_indices(matrix_all_distances.shape[0], k=1)]
    Z = complete(pdist)
    new_labels = fcluster(Z, resid_threshold, criterion="distance")

    new_labels -= 1
    labels_merged = spt_no_outliers[:, 1].copy()
    labels_merged[spt_no_outliers[:, 1]>=0] = new_labels[spt_no_outliers[spt_no_outliers[:, 1]>=0, 1]]
    
    return labels_merged


def full_post_processing(raw_data_bin, geom, 
                         spt, x, z_reg, z_abs, 
                         pcs, disp, 
                         prob_min = 0.1, time_temp_computation=0,
                         threshold_to_clean_1=10, threshold_to_clean_2=5, threshold_to_clean_3=4, 
                         isosplit_th=1, n_iter_split=2, 
                         dist_pairs=20, resid_threshold=7, deconv_th=1000,
                         bin_size_um=1, pfs=30000):
    
    # Recommendation: set time_temp_computation to middle of recording
    
    n_units = spt[:, 1].max()+1
    std_z = np.zeros(n_units)
    std_x = np.zeros(n_units)
    for k in range(n_units):
        idx_k = np.flatnonzero(spt[:, 1]==k)
        std_z[k] = z_reg[idx_k].std()
        std_x[k] = x[idx_k].std()

    units_to_clean= np.flatnonzero(np.maximum(std_z, std_x)>threshold_to_clean_1)
    spt_final = spt.copy()
    print("Label Outliers")
    spt_final[:, 1] = correct_outliers(spt, x, z_reg, disp, units_to_clean = units_to_clean, prob_min = prob_min)
    print("Split")
    spt_final = post_deconv_split(spt_final, x, z_reg, pcs, isosplit_th=isosplit_th, n_iter=n_iter_split)

    n_units = spt_final[:, 1].max()+1
    std_z = np.zeros(n_units)
    std_x = np.zeros(n_units)
    for k in range(n_units):
        idx_k = np.flatnonzero(spt_final[:, 1]==k)
        std_z[k] = z_reg[idx_k].std()
        std_x[k] = x[idx_k].std()
    units_to_clean= np.flatnonzero(np.maximum(std_z, std_x)>threshold_to_clean_2)
    spt_final[:, 1] = correct_outliers(spt_final, x, z_reg, disp, units_to_clean = units_to_clean, prob_min = prob_min)
    
    print("Merge")
    spt_final[:, 1] = post_deconv_merge(raw_data_bin, geom, spt_final, 
                          z_abs, z_reg, x, time_temp_computation, std_z*1.65,
                          dist_pairs=dist_pairs, resid_threshold=resid_threshold, deconv_th=deconv_th,
                          bin_size_um=bin_size_um, pfs=pfs)

    n_units = spt_final[:, 1].max()+1
    std_z = np.zeros(n_units)
    std_x = np.zeros(n_units)
    for k in range(n_units):
        idx_k = np.flatnonzero(spt_final[:, 1]==k)
        std_z[k] = z_reg[idx_k].std()
        std_x[k] = x[idx_k].std()
    units_to_clean= np.flatnonzero(np.maximum(std_z, std_x)>threshold_to_clean_3)
    spt_final[:, 1] = correct_outliers(spt_final, x, z_reg, disp, units_to_clean = units_to_clean, prob_min = prob_min/2)

    return spt_final