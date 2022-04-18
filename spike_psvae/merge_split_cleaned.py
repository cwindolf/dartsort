# %%
import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.signal import argrelmin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import scipy.optimize as optim_ls
import hdbscan
from spike_psvae.cluster_utils import compute_shifted_similarity, read_waveforms
from isosplit import isocut
from scipy.spatial.distance import cdist
from tqdm import notebook
from tqdm.auto import tqdm

# %%
def align_templates(labels, templates, wfs, triaged_spike_index, denoiser_offset=42):
    list_argmin = np.zeros(templates.shape[0])
    for i in range(templates.shape[0]):
        list_argmin[i] = templates[i, :, templates[i].ptp(0).argmax()].argmin()
    idx_not_aligned = np.where(list_argmin!=denoiser_offset)[0]

    for unit in idx_not_aligned:
        mc = templates[unit].ptp(0).argmax()
        offset = templates[unit, :, mc].argmin()
        shift = offset-denoiser_offset
        triaged_spike_index[labels == unit, 0] += shift
        if shift>0:
            wfs[labels == unit, :-shift] = wfs[labels == unit, shift:]
            wfs[labels == unit, -shift:] = 0
        elif shift<0:
            wfs[labels == unit, -shift:] = wfs[labels == unit, :shift]
            wfs[labels == unit, :-shift] = 0

    idx_sorted = triaged_spike_index[:, 0].argsort()
    triaged_spike_index = triaged_spike_index[idx_sorted]
    
    return triaged_spike_index, idx_sorted
# %%
def denoise_wf_nn_tmp_single_channel(wf, denoiser, device):
    denoiser = denoiser.to(device)
    n_data, n_times, n_chans = wf.shape
    if wf.shape[0]>0:
        wf_reshaped = wf.transpose(0, 2, 1).reshape(-1, n_times)
        wf_torch = torch.FloatTensor(wf_reshaped).to(device)
        denoised_wf = denoiser(wf_torch).data
        denoised_wf = denoised_wf.reshape(
            n_data, n_chans, n_times)
        denoised_wf = denoised_wf.cpu().data.numpy().transpose(0, 2, 1)

        del wf_torch
    else:
        denoised_wf = np.zeros((wf.shape[0], wf.shape[1]*wf.shape[2]),'float32')

    return denoised_wf

# %%
def run_LDA_split(wfs_unit_denoised, n_channels=10, n_times=121):
    lda_model = LDA(n_components = 2)
    arr = wfs_unit_denoised.ptp(1).argmax(1)
    if np.unique(arr).shape[0]<=2:
        arr[-1] = np.unique(arr)[0]-1
        arr[0] = np.unique(arr)[-1]+1
    lda_comps = lda_model.fit_transform(wfs_unit_denoised.reshape((-1, n_times*n_channels)), arr)
    lda_clusterer = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
    lda_clusterer.fit(lda_comps)
    return lda_clusterer.labels_

# %%

def split_individual_cluster(residual_path, waveforms_unit, first_chans_unit, spike_index_unit, x_unit, z_unit, ptps_unit, geom_array, denoiser, device, n_channels = 10):
    total_channels = geom_array.shape[0]
    n_channels_half = n_channels//2
    labels_unit = -1*np.ones(spike_index_unit.shape[0])
    is_split = False
    true_mc = int(np.median(spike_index_unit[:, 1]))
    mc = max(n_channels_half, true_mc)
    mc = min(total_channels - n_channels_half, mc)
    pca_model = PCA(2)
    wfs_unit = np.zeros((waveforms_unit.shape[0], waveforms_unit.shape[1], n_channels))
    for i in range(wfs_unit.shape[0]):
        if mc == n_channels_half:
            wfs_unit[i] = waveforms_unit[i, :, :n_channels]
        elif mc == total_channels - n_channels_half:
            wfs_unit[i] = waveforms_unit[i, :, waveforms_unit.shape[2]-n_channels:]
        else:
            mc_new = int(mc - first_chans_unit[i])
            wfs_unit[i] = waveforms_unit[i, :, mc_new-n_channels_half:mc_new+n_channels_half]
    
    wfs_unit += read_waveforms(spike_index_unit[:, 0], residual_path, geom_array, n_times=121, channels = np.arange(mc-n_channels_half, mc+n_channels_half))[0]
    wfs_unit_denoised = denoise_wf_nn_tmp_single_channel(wfs_unit, denoiser, device)
    
    if true_mc<n_channels_half:
        pcs = pca_model.fit_transform(wfs_unit_denoised[:, :, true_mc])
    if true_mc>total_channels-n_channels_half:
        true_mc = true_mc - (total_channels-n_channels)
        pcs = pca_model.fit_transform(wfs_unit_denoised[:, :, true_mc])
    else:
        pcs = pca_model.fit_transform(wfs_unit_denoised[:, :, n_channels_half])

    alpha1 = (x_unit.max() - x_unit.min())/(pcs[:, 0].max()-pcs[:, 0].min())
    alpha2 = (x_unit.max() - x_unit.min())/(pcs[:, 1].max()-pcs[:, 1].min())
    features = np.concatenate((np.expand_dims(x_unit,1), np.expand_dims(z_unit,1), np.expand_dims(pcs[:, 0], 1)*alpha1, np.expand_dims(pcs[:, 1], 1)*alpha2, np.expand_dims(np.log(ptps_unit)*30,1)), axis=1) #Use scales parameter
    clusterer_herding = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=25)
    clusterer_herding.fit(features)
    
    labels_rec_hdbscan = clusterer_herding.labels_
    if np.unique(labels_rec_hdbscan).shape[0]>1:
        is_split = True
# Don't do diptest Here
#         diptest_comps = lda_diptest.fit_transform(wfs_unit_denoised.reshape((-1, 121*10)), clusterer_herding.labels_)
#         value_dpt, cut_calue = isocut(diptest_comps[:, 0])
#         if value_dpt > 0.5:
#             is_split = True
    
    if is_split:
        labels_unit[labels_rec_hdbscan==-1] = -1
        label_max_temp = labels_rec_hdbscan.max()
        cmp = 0
        for new_unit_id in np.unique(labels_rec_hdbscan)[1:]:
            wfs_new_unit = wfs_unit_denoised[labels_rec_hdbscan == new_unit_id]
            lda_labels = run_LDA_split(wfs_new_unit)
            if np.unique(lda_labels).shape[0]==1:
                labels_unit[labels_rec_hdbscan==new_unit_id] = cmp
                cmp += 1
            else:
                for lda_unit in np.unique(lda_labels):
                    if lda_unit >= 0:
                        labels_unit[np.flatnonzero(labels_rec_hdbscan==new_unit_id)[lda_labels == lda_unit]] = cmp
                        cmp += 1
                    else:
                        labels_unit[np.flatnonzero(labels_rec_hdbscan==new_unit_id)[lda_labels == lda_unit]] = -1
    else:
        lda_labels = run_LDA_split(wfs_unit_denoised)
        if np.unique(lda_labels).shape[0]>1:
            is_split = True
            labels_unit = lda_labels

    return is_split, labels_unit

# %%
def split_clusters(residual_path, waveforms, first_chans, spike_index, labels, x, z, ptps, geom_array, denoiser, device, n_channels=10):
    labels_new = labels.copy()
    labels_original = labels.copy()

    n_clusters = labels.max()
    for unit in tqdm(np.unique(labels)[1:]):
        spike_index_unit = spike_index[labels == unit]
        waveforms_unit = waveforms[labels == unit]
        first_chans_unit = first_chans[labels == unit]
        x_unit, z_unit, ptps_unit = x[labels == unit], z[labels == unit], ptps[labels == unit]
        is_split, unit_new_labels = split_individual_cluster(residual_path, waveforms_unit, first_chans_unit, spike_index_unit, x_unit, z_unit, ptps_unit, geom_array, denoiser, device, n_channels)
        if is_split:
            for new_label in np.unique(unit_new_labels):
                if new_label == -1:
                    idx = np.flatnonzero(labels_original == unit)[unit_new_labels == new_label]
                    labels_new[idx] = new_label
                elif new_label > 0:
                    n_clusters += 1
                    idx = np.flatnonzero(labels_original == unit)[unit_new_labels == new_label]
                    labels_new[idx] = n_clusters
    return labels_new
# %%
def get_x_z_templates(n_templates, labels, x, z):
    x_z_templates = np.zeros((n_templates, 2))
    for i in range(n_templates):
        x_z_templates[i, 1] = np.median(z[labels==i])
        x_z_templates[i, 0] = np.median(x[labels==i])
    return x_z_templates

# %%
def get_n_spikes_templates(n_templates, labels):
    n_spikes_templates = np.zeros(n_templates)
    for i in range(n_templates):
        n_spikes_templates[i] = (labels==i).sum()
    return n_spikes_templates

# %%
def get_templates(standardized_path, geom_array, n_templates, spike_index, labels, max_spikes=250, n_times=121):
    templates = np.zeros((n_templates, n_times, geom_array.shape[0]))
    for unit in range(n_templates):
        spike_times_unit = spike_index[labels==unit, 0]
        if spike_times_unit.shape[0]>max_spikes:
            idx = np.random.choice(np.arange(spike_times_unit.shape[0]), max_spikes, replace = False)
        else:
            idx = np.arange(spike_times_unit.shape[0])

        wfs_unit = read_waveforms(spike_times_unit[idx], standardized_path, geom_array, n_times=n_times)[0]
        templates[unit] = wfs_unit.mean(0)
    return templates

# %%
def get_proposed_pairs(n_templates, templates, x_z_templates, n_temp = 20, n_channels = 10):
    n_channels_half = n_channels//2
    dist = cdist(x_z_templates, x_z_templates)
    dist_argsort = dist.argsort(axis = 1)[:, 1:n_temp+1]
    dist_template = np.zeros((dist_argsort.shape[0], n_temp))
    for i in range(n_templates):
        mc = min(templates[i].ptp(0).argmax(), 384-n_channels_half)
        mc = max(mc, n_channels_half)
        temp_a = templates[i, :, mc-n_channels_half:mc+n_channels_half]
        for j in range(n_temp):
            temp_b = templates[dist_argsort[i, j], :, mc-n_channels_half:mc+n_channels_half]
            dist_template[i, j] = compute_shifted_similarity(temp_a, temp_b)[0]
    return dist_argsort, dist_template

# %%
def get_diptest_value(residual_path, waveforms, first_chans, geom_array, spike_index, labels, unit_a, unit_b, n_spikes_templates, mc, two_units_shift, unit_shifted, denoiser, device, n_channels = 10, n_times=121, rank_pca=8, nn_denoise = False):

    # ALIGN BASED ON MAX PTP TEMPLATE MC 
    n_channels_half = n_channels//2

    n_wfs_max = int(min(250, min(n_spikes_templates[unit_a], n_spikes_templates[unit_b]))) 

    
    mc = min(384-n_channels_half, mc)
    mc = max(n_channels_half, mc)

    spike_times_unit_a = spike_index[labels == unit_a, 0]
    idx = np.random.choice(np.arange(spike_times_unit_a.shape[0]), n_wfs_max, replace = False)
    spike_times_unit_a = spike_times_unit_a[idx]
    wfs_a = waveforms[labels == unit_a][idx]
    first_chan_a = first_chans[labels == unit_a][idx]
    
    spike_times_unit_b = spike_index[labels == unit_b, 0]
    idx = np.random.choice(np.arange(spike_times_unit_b.shape[0]), n_wfs_max, replace = False)
    spike_times_unit_b = spike_times_unit_b[idx]
    wfs_b = waveforms[labels == unit_b][idx]
    first_chan_b = first_chans[labels == unit_b][idx]
    
    wfs_a_bis = np.zeros((wfs_a.shape[0], n_times, n_channels))
    wfs_b_bis = np.zeros((wfs_b.shape[0], n_times, n_channels))
    
    if two_units_shift>0:
        
        if unit_shifted == unit_a:
            for i in range(wfs_a_bis.shape[0]):
                first_chan = int(mc - first_chan_a[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_a.shape[2]-n_channels, int(first_chan))
                wfs_a_bis[i, :-two_units_shift] = wfs_a[i, two_units_shift:, first_chan:first_chan+n_channels]
                first_chan = int(mc - first_chan_b[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_b.shape[2]-n_channels, int(first_chan))
                wfs_b_bis[i, :] = wfs_b[i, :, first_chan:first_chan+n_channels]
            wfs_a_bis += read_waveforms(spike_times_unit_a+two_units_shift, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
            wfs_b_bis += read_waveforms(spike_times_unit_b, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
        else:
            for i in range(wfs_a_bis.shape[0]):
                first_chan = int(mc - first_chan_a[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_a.shape[2]-n_channels, int(first_chan))
                wfs_a_bis[i] = wfs_a[i, :, first_chan:first_chan+n_channels]
                first_chan = int(mc - first_chan_b[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_b.shape[2]-n_channels, int(first_chan))
                wfs_b_bis[i, :-two_units_shift] = wfs_b[i, two_units_shift:, first_chan:first_chan+n_channels]
            wfs_a_bis += read_waveforms(spike_times_unit_a, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
            wfs_b_bis += read_waveforms(spike_times_unit_b+two_units_shift, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
    elif two_units_shift<0:
        if unit_shifted == unit_a:
            for i in range(wfs_a_bis.shape[0]):
                first_chan = int(mc - first_chan_a[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_a.shape[2]-n_channels, int(first_chan))
                wfs_a_bis[i, -two_units_shift:] = wfs_a[i, :two_units_shift, first_chan:first_chan+n_channels]
                first_chan = int(mc - first_chan_b[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_b.shape[2]-n_channels, int(first_chan))
                wfs_b_bis[i, :] = wfs_b[i, :, first_chan:first_chan+n_channels]
            wfs_a_bis += read_waveforms(spike_times_unit_a+two_units_shift, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
            wfs_b_bis += read_waveforms(spike_times_unit_b, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]

        else:
            for i in range(wfs_a_bis.shape[0]):
                first_chan = int(mc - first_chan_a[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_a.shape[2]-n_channels, int(first_chan))
                wfs_a_bis[i] = wfs_a[i, :, first_chan:first_chan+n_channels]
                first_chan = int(mc - first_chan_b[i] - n_channels_half)
                first_chan = max(0, int(first_chan))
                first_chan = min(wfs_b.shape[2]-n_channels, int(first_chan))
                wfs_b_bis[i, -two_units_shift:] = wfs_b[i, :two_units_shift, first_chan:first_chan+n_channels]
            wfs_a_bis += read_waveforms(spike_times_unit_a, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
            wfs_b_bis += read_waveforms(spike_times_unit_b+two_units_shift, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
    else:
        for i in range(wfs_a_bis.shape[0]):
            first_chan = int(mc - first_chan_a[i] - n_channels_half)
            first_chan = max(0, int(first_chan))
            first_chan = min(wfs_a.shape[2]-n_channels, int(first_chan))
            wfs_a_bis[i] = wfs_a[i, :, first_chan:first_chan+n_channels]
            first_chan = int(mc - first_chan_b[i] - n_channels_half)
            first_chan = max(0, int(first_chan))
            first_chan = min(wfs_b.shape[2]-n_channels, int(first_chan))
            wfs_b_bis[i, :] = wfs_b[i, :, first_chan:first_chan+n_channels]
        wfs_a_bis += read_waveforms(spike_times_unit_a, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]
        wfs_b_bis += read_waveforms(spike_times_unit_b, residual_path, geom_array, n_times=n_times, channels = np.arange(mc-n_channels_half,mc+n_channels_half))[0]      
    
    
    tpca = PCA(rank_pca)
    wfs_diptest = np.concatenate((wfs_a, wfs_b))
    
    if nn_denoise:
        wfs_diptest = denoise_wf_nn_tmp_single_channel(wfs_diptest, denoiser, device)

    N, T, C = wfs_diptest.shape
    wfs_diptest = wfs_diptest.transpose(0, 2, 1).reshape(N*C, T)
    wfs_diptest = tpca.inverse_transform(tpca.fit_transform(wfs_diptest))
    wfs_diptest = wfs_diptest.reshape(N, C, T).transpose(0, 2, 1).reshape((N, C*T))
    
#     wfs_diptest = np.concatenate((wfs_a_bis, wfs_b_bis)).reshape((-1, n_channels*n_times))
    labels_diptest = np.zeros(wfs_a_bis.shape[0]+wfs_b_bis.shape[0])
    labels_diptest[:wfs_a_bis.shape[0]] = 1
    
    lda_model = LDA(n_components = 1)
    lda_comps = lda_model.fit_transform(wfs_diptest, labels_diptest)
    value_dpt, cut_calue = isocut(lda_comps[:, 0])
    return value_dpt

# %%
def get_merged(residual_path, waveforms, first_chans, geom_array, templates, n_spikes_templates, x_z_templates, n_templates, spike_index, labels, x, z, denoiser, device, n_channels=10, n_temp = 10, distance_threshold = 3., threshold_diptest = .75, rank_pca=8, nn_denoise = False):
     
    n_spikes_templates = get_n_spikes_templates(n_templates, labels)
    x_z_templates = get_x_z_templates(n_templates, labels, x, z)
    print("GET PROPOSED PAIRS")
    dist_argsort, dist_template = get_proposed_pairs(n_templates, templates, x_z_templates, n_temp = n_temp)
    
    labels_updated = labels.copy()
    reference_units = np.unique(labels)[1:]
    
    for unit in tqdm(range(n_templates)): #tqdm
        unit_reference = reference_units[unit]
        to_be_merged = [unit_reference]
        merge_shifts = [0]
        is_merged = False

        for j in range(n_temp):
            if dist_template[unit, j] < distance_threshold:
                unit_bis = dist_argsort[unit, j]
                unit_bis_reference = reference_units[unit_bis]
                if unit_reference != unit_bis_reference:
#                     ALIGN BASED ON MAX PTP TEMPLATE MC 
                    if templates[unit_reference].ptp(0).max() < templates[unit_bis_reference].ptp(0).max():
                        mc = templates[unit_bis_reference].ptp(0).argmax()
                        two_units_shift = templates[unit_reference, :, mc].argmin() - templates[unit_bis_reference, :, mc].argmin()
                        unit_shifted = unit_reference
                    else:
                        mc = templates[unit_reference].ptp(0).argmax()
                        two_units_shift = templates[unit_bis_reference, :, mc].argmin() - templates[unit_reference, :, mc].argmin()
                        unit_shifted = unit_bis_reference
                    dpt_val = get_diptest_value(residual_path, waveforms, first_chans, geom_array, spike_index, labels_updated, unit_reference, unit_bis_reference, n_spikes_templates, mc, two_units_shift,unit_shifted, denoiser, device, n_channels, rank_pca=rank_pca, nn_denoise = nn_denoise)
                    if dpt_val<threshold_diptest and np.abs(two_units_shift)<2:
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

            new_reference_unit = np.unique(np.asarray(to_be_merged))[0]

            templates[new_reference_unit] = n_spikes_templates[new_reference_unit]*templates[new_reference_unit]/n_total_spikes
            cmp = 1
            for unit_merged in np.unique(np.asarray(to_be_merged))[1:]:
                shift_ = merge_shifts[cmp]
                templates[new_reference_unit] += n_spikes_templates[unit_merged]*np.roll(templates[unit_merged], shift_, axis = 0)/n_total_spikes
                n_spikes_templates[new_reference_unit] += n_spikes_templates[unit_merged]
                n_spikes_templates[unit_merged] = 0
                labels_updated[labels_updated == unit_merged] = new_reference_unit
                reference_units[unit_merged] = new_reference_unit
                cmp += 1
    return labels_updated


