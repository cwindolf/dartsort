# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.rc("figure", dpi=200)
import hdbscan
from pathlib import Path
import torch
from sklearn.decomposition import PCA
import matplotlib
# from spike_psvae import cluster, merge_split_cleaned, cluster_viz_index, denoise, cluster_utils, triage, cluster_viz
# from spike_psvae.cluster_utils import read_waveforms, compare_two_sorters, make_sorting_from_labels_frames
# from spike_psvae.cluster_viz import plot_agreement_venn, plot_unit_similarities
# from spike_psvae.cluster_utils import get_closest_clusters_kilosort_hdbscan
# from spike_psvae.cluster_viz import plot_single_unit_summary
# from spike_psvae.cluster_viz import cluster_scatter, plot_waveforms_geom, plot_raw_waveforms_unit_geom, plot_venn_agreement
# from spike_psvae.cluster_viz import plot_array_scatter, plot_self_agreement, plot_single_unit_summary, plot_agreement_venn, plot_isi_distribution, plot_waveforms_unit_geom, plot_unit_similarities
# from spike_psvae.cluster_viz import plot_unit_similarity_heatmaps
# from spike_psvae.cluster_utils import make_sorting_from_labels_frames, compute_cluster_centers, relabel_by_depth, run_weighted_triage, remove_duplicate_units
# from spike_psvae.cluster_utils import get_agreement_indices, compute_spiketrain_agreement, get_unit_similarities, compute_shifted_similarity, read_waveforms
# from spike_psvae.cluster_utils import get_closest_clusters_hdbscan, get_closest_clusters_kilosort, get_closest_clusters_hdbscan_kilosort, get_closest_clusters_kilosort_hdbscan

# %%
np.random.seed(0) #for reproducibility (templates use random waveforms)

# %%
data_path = '/media/cat/data/'
data_name = 'CSH_ZAD_026_5min'
data_dir = data_path + data_name + '/'
raw_bin = data_dir + 'CSH_ZAD_026_snip.ap.bin'
residual_bin = data_dir + 'residual_CSH_ZAD_026_snip.ap_t_0_None.bin'
sub_h5 = data_dir + "subtraction_CSH_ZAD_026_snip.ap_t_0_None.h5"

output_dir = Path("/outputs")

# %%
#load features
offset_min = 30 #30 minutes into the recording
with h5py.File(sub_h5, "r") as h5:
    print(h5.keys())
    spike_index = h5["spike_index"][:]
    x, y, z, alpha, z_rel = h5["localizations"][:].T
    maxptps = h5["maxptps"][:]
    z_abs = h5["z_reg"][:]
    geom = h5["geom"][:]
    firstchans = h5["first_channels"][:]
    end_sample = h5["end_sample"][()]
    start_sample = h5["start_sample"][()]
    start_sample += offset_min * 60 * 30000
    end_sample += offset_min * 60 * 30000
    channel_index = h5["channel_index"][:]
    z_reg = h5["z_reg"][:]
    tpca_mean = h5["tpca_mean"][:]
    tpca_components = h5["tpca_components"][:]
    print("Loading TPCA from h5")
    tpca = PCA(tpca_components.shape[0])
    tpca.mean_ = tpca_mean
    tpca.components_ = tpca_components
    
num_spikes = spike_index.shape[0]
end_time = end_sample / 30000
start_time = start_sample / 30000
recording_duration = end_time - start_time
h5 = h5py.File(sub_h5)
wfs_subtracted = h5["subtracted_waveforms"]
wfs_full_denoise = h5["cleaned_waveforms"]
print(f"duration of recording: {recording_duration} s")

# %%
#load kilosort results
data_path = '/media/cat/data/'
data_name = 'CSH_ZAD_026_5min'
data_dir = data_path + data_name + '/'
offset_min = 30

kilo_spike_samples = np.load(data_dir + 'kilosort_spk_samples.npy')
kilo_spike_frames = (kilo_spike_samples - offset_min*60*30000) #to match our detection alignment
kilo_spike_clusters = np.load(data_dir + 'kilosort_spk_clusters.npy')
kilo_spike_depths = np.load(data_dir + 'kilosort_spk_depths.npy')
kilo_cluster_depth_means = {}
kilo_cluster_locations= {}
kilo_cluster_templates = {}
kilo_cluster_maxptps = {}
#create kilosort SpikeInterface sorting
sorting_kilo = cluster_utils.make_sorting_from_labels_frames(kilo_spike_clusters, kilo_spike_frames)

from spike_psvae.localization import localize_ptp
for cluster_id in np.unique(kilo_spike_clusters):
    kilo_cluster_depth_means[cluster_id] = np.mean(kilo_spike_depths[kilo_spike_clusters==cluster_id])
    waveforms = read_waveforms(np.random.choice(sorting_kilo.get_unit_spike_train(cluster_id), 250), raw_bin, geom, n_times=121)[0]
    template = np.mean(waveforms, axis=0)
    kilo_cluster_templates[cluster_id] = template
    max_chan = np.argmin(np.abs((kilo_cluster_depth_means[cluster_id] - geom[:,1])))
    first_chan = max(max_chan-20, 0)
    channels = list(range(first_chan, first_chan+40))
    template_x, _, template_z_rel, template_z_abs, _, _ = localize_ptp(kilo_cluster_templates[cluster_id].ptp(0)[channels], first_chan, max_chan, geom)
    kilo_cluster_locations[cluster_id] = (template_x, template_z_abs)
    kilo_cluster_maxptps[cluster_id] = np.max(waveforms.ptp(1),1)
    
    
good_kilo_sort_clusters_all = np.array([  0,  17,  19,  25,  30,  33,  36,  38,  41,  47,  48,  53,  64,
        70,  78,  82,  83,  85,  88,  90,  97, 103, 109, 112, 114, 115,
       117, 119, 120, 131, 132, 133, 141, 142, 153, 158, 169, 172, 185,
       187, 189, 193, 197, 199, 205, 208, 211, 215, 217, 224, 237, 244,
       247, 269, 272, 274, 280, 283, 289, 291, 292, 296, 300, 303, 304,
       308, 309, 320, 328, 331, 336, 341, 349, 350, 380, 382, 386, 400,
       409, 411, 414, 435, 438, 439, 464, 474, 476, 478, 485, 487, 488,
       496, 503, 509, 512, 521, 522, 523, 529, 533, 534, 535, 536, 537,
       539, 544, 545, 547, 548, 551, 552, 555, 557, 570, 583, 596, 598,
       621, 629, 633, 637, 648, 655, 660, 670, 671, 677, 678, 681, 682,
       683, 699, 700, 702, 708, 709])

#remove empty clusters
good_kilo_sort_clusters = []

for good_cluster in good_kilo_sort_clusters_all:
    if good_cluster in sorting_kilo.get_unit_ids():
        good_kilo_sort_clusters.append(good_cluster) 

# %% [markdown]
# ## triage and cluster

# %%
tx, ty, tz, talpha, tmaxptps, _, ptp_keep, idx_keep  = triage.run_weighted_triage(
    x, y, z_reg, alpha, maxptps, threshold=80
)
idx_keep_full = ptp_keep[idx_keep]

# %%
denoiser = denoise.SingleChanDenoiser().load()
device = "cuda" if torch.cuda.is_available() else "cpu"
denoiser.to(device);

# %%
#clustering parameters
min_cluster_size = 25
min_samples = 25

# this will cluster and relabel by depth
scales = (1,10,1,15,30) #predefined scales for each feature
features = np.c_[tx*scales[0], tz*scales[2], np.log(tmaxptps) * scales[4]]

# features = np.c_[tx*scales[0], tz*scales[2], all_pcs[:,0] * alpha1, all_pcs[:,1] * alpha2]
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
clusterer.fit(features)

# z order
cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
clusterer = cluster_utils.relabel_by_depth(clusterer, cluster_centers)
pre_dup_labels = clusterer.labels_.copy()
# remove dups and re z order
clusterer, duplicate_indices, duplicate_spikes = cluster_utils.remove_duplicate_spikes(clusterer, spike_index[idx_keep_full, 0], tmaxptps, frames_dedup=12)
#recompute cluster centers
cluster_centers = cluster_utils.compute_cluster_centers(clusterer)

# labels in full index space (not triaged)
labels = np.full(x.shape, -1)
labels[idx_keep_full] = clusterer.labels_
labels_original = labels.copy()

# %% tags=[]
# z_cutoffs= [(0,550), (500,1050), (1000,1550), (1500,2050), (2000,2550), (2500,3050), (3000,3550), (3500,4050)]
z_cutoffs= [(500,1000)]#, (1000,1550), (1500,2050), (2000,2550), (2500,3050), (3000,3550), (3500,4050)]
matplotlib.rcParams.update({'font.size': 14})
for z_cutoff in z_cutoffs:
    fig, axes = cluster_viz_index.array_scatter(
            labels_original[idx_keep_full], geom, tx, tz, tmaxptps, 
            zlim=z_cutoff,
    )
    for cluster_id in good_kilo_sort_clusters:
        # axes[0].scatter(-25, kilo_cluster_depth_means[cluster_id], marker='x', color='blue')
        if len(sorting_kilo.get_unit_spike_train(cluster_id)) > 5:
            axes[0].scatter(kilo_cluster_locations[cluster_id][0], kilo_cluster_locations[cluster_id][1], marker='x', color='red', s=150)
            axes[0].annotate(f"{cluster_id}", (kilo_cluster_locations[cluster_id][0]+2.5, kilo_cluster_locations[cluster_id][1]+2.5), color='red')
            axes[1].scatter(np.max(kilo_cluster_templates[cluster_id].ptp(0)), kilo_cluster_locations[cluster_id][1], marker='x', color='red', s=150)
            axes[1].annotate(f"{cluster_id}", (np.max(kilo_cluster_templates[cluster_id].ptp(0))+1, kilo_cluster_locations[cluster_id][1]+2.5), color='red')
            # axes[2].scatter(-25, kilo_cluster_depth_means[cluster_id], marker='x', color='blue')
            axes[2].scatter(kilo_cluster_locations[cluster_id][0], kilo_cluster_locations[cluster_id][1], marker='x', color='red', s=150)
            axes[2].annotate(f"{cluster_id}", (kilo_cluster_locations[cluster_id][0]+2.5, kilo_cluster_locations[cluster_id][1]+2.5), color='red')
            axes[0].set_xlim(-20, 80)
            axes[1].set_xlim(0, 30)
            axes[2].set_xlim(-20, 80)
    # for go
    # fig.savefig(f"{save_dir_path}/full_scatter_{z_cutoff[0]}_{z_cutoff[1]}", dpi=200)
    # plt.close(fig)

# %%
templates = merge_split_cleaned.get_templates(
    raw_bin, geom, clusterer.labels_.max()+1, spike_index[idx_keep_full], clusterer.labels_
)

template_shifts, template_maxchans, shifted_triaged_spike_index, idx_not_aligned = merge_split_cleaned.align_spikes_by_templates(
    clusterer.labels_, templates, spike_index[idx_keep_full]
)

shifted_full_spike_index = spike_index.copy()
shifted_full_spike_index[idx_keep_full] = shifted_triaged_spike_index

# %% tags=[] jupyter={"outputs_hidden": true}
# split
with h5py.File(sub_h5, "r") as h5:
    labels_split = merge_split_cleaned.split_clusters(
        residual_bin, 
        h5["subtracted_waveforms"], 
        firstchans, 
        shifted_full_spike_index,
        template_maxchans,
        template_shifts,
        labels_original, 
        x, 
        z_reg, 
        # maxptps, 
        geom, 
        denoiser, 
        device,
        tpca,
        n_channels=10,
        pca_n_channels=4,
        nn_denoise=False,
        threshold_diptest=.5,
    )    

# re-order again
clusterer.labels_ = labels_split[idx_keep_full]
cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
clusterer = cluster_utils.relabel_by_depth(clusterer, cluster_centers)
cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
labels = np.full(x.shape, -1)
labels[idx_keep_full] = clusterer.labels_

# %% tags=[]
# z_cutoffs= [(0,550), (500,1050), (1000,1550), (1500,2050), (2000,2550), (2500,3050), (3000,3550), (3500,4050)]
z_cutoffs= [(500,1000)]#, (1000,1550), (1500,2050), (2000,2550), (2500,3050), (3000,3550), (3500,4050)]
matplotlib.rcParams.update({'font.size': 14})
for z_cutoff in z_cutoffs:
    fig, axes = cluster_viz.array_scatter(
            labels_split[idx_keep_full], geom, tx, tz, tmaxptps, 
            zlim=z_cutoff,
    )
    for cluster_id in sorting_kilo.get_unit_ids():
        if len(sorting_kilo.get_unit_spike_train(cluster_id)) > 25:
            if cluster_id in good_kilo_sort_clusters:
                color = 'red'
                alpha = 1
            else:
                color = 'blue'
                alpha = .4
            if cluster_id in good_kilo_sort_clusters:
                axes[0].scatter(kilo_cluster_locations[cluster_id][0], kilo_cluster_locations[cluster_id][1], marker='x', color=color, s=100, alpha=alpha)
                text_0 = axes[0].annotate(f"{cluster_id}", (kilo_cluster_locations[cluster_id][0]+2.5, kilo_cluster_locations[cluster_id][1]+2.5), color=color)
                axes[1].scatter(np.max(kilo_cluster_templates[cluster_id].ptp(0)), kilo_cluster_locations[cluster_id][1], marker='x', color=color, s=100, alpha=alpha)
                text_1 = axes[1].annotate(f"{cluster_id}", (np.max(kilo_cluster_templates[cluster_id].ptp(0))+1, kilo_cluster_locations[cluster_id][1]+2.5), color=color)
                axes[2].scatter(kilo_cluster_locations[cluster_id][0], kilo_cluster_locations[cluster_id][1], marker='x', color=color, s=100, alpha=alpha)
                text_2 = axes[2].annotate(f"{cluster_id}", (kilo_cluster_locations[cluster_id][0]+2.5, kilo_cluster_locations[cluster_id][1]+2.5), color=color)
                text_0.set_alpha(alpha)
                text_1.set_alpha(alpha)
                text_2.set_alpha(alpha)
                axes[0].set_xlim(-20, 80)
                axes[1].set_xlim(0, 30)
                axes[2].set_xlim(-20, 80)
            # axes[0].set_xlim(-25, 120)
            # axes[2].set_xlim(-25, 120)
            # axes[2].set_ylim(200, 500)
            # axes[2].set_ylim(200, 500)
    # for go
    # fig.savefig(f"{save_dir_path}/full_scatter_{z_cutoff[0]}_{z_cutoff[1]}", dpi=200)
    # plt.close(fig)

# %% tags=[]
# get templates
templates = merge_split_cleaned.get_templates(
    raw_bin, geom, clusterer.labels_.max()+1, spike_index[idx_keep_full], clusterer.labels_
)

template_shifts, template_maxchans, shifted_triaged_spike_index, idx_not_aligned  = merge_split_cleaned.align_spikes_by_templates(
    clusterer.labels_, templates, spike_index[idx_keep_full]
)
shifted_full_spike_index = spike_index.copy()
shifted_full_spike_index[idx_keep_full] = shifted_triaged_spike_index

# %% tags=[]
# merge
with h5py.File(sub_h5, "r") as h5:
    labels_merged = merge_split_cleaned.get_merged(
        residual_bin,
        h5["subtracted_waveforms"],
        firstchans,
        geom,
        templates,
        template_shifts,
        len(templates),
        shifted_full_spike_index,
        labels,
        x,
        z_reg,
        denoiser,
        device,
        tpca,
        distance_threshold=1.,
        threshold_diptest=.5,
        nn_denoise=False,
    )
    
# re-order again
clusterer.labels_ = labels_merged[idx_keep_full]
cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
clusterer = cluster_utils.relabel_by_depth(clusterer, cluster_centers)
cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
labels = np.full(x.shape, -1)
labels[idx_keep_full] = clusterer.labels_

# %%
# z_cutoffs= [(0,550), (500,1050), (1000,1550), (1500,2050), (2000,2550), (2500,3050), (3000,3550), (3500,4050)]
z_cutoffs= [(500,1000)]#, (1000,1550), (1500,2050), (2000,2550), (2500,3050), (3000,3550), (3500,4050)]
matplotlib.rcParams.update({'font.size': 14})
for z_cutoff in z_cutoffs:
    fig, axes = cluster_viz.array_scatter(
            labels_merged[idx_keep_full], geom, tx, tz, tmaxptps, 
            zlim=z_cutoff,
    )
    for cluster_id in sorting_kilo.get_unit_ids():
        if len(sorting_kilo.get_unit_spike_train(cluster_id)) > 25:
            if cluster_id in good_kilo_sort_clusters:
                color = 'red'
                alpha = 1
            else:
                color = 'blue'
                alpha = .4
            if cluster_id in good_kilo_sort_clusters:
                axes[0].scatter(kilo_cluster_locations[cluster_id][0], kilo_cluster_locations[cluster_id][1], marker='x', color=color, s=100, alpha=alpha)
                text_0 = axes[0].annotate(f"{cluster_id}", (kilo_cluster_locations[cluster_id][0]+2.5, kilo_cluster_locations[cluster_id][1]+2.5), color=color)
                axes[1].scatter(np.max(kilo_cluster_templates[cluster_id].ptp(0)), kilo_cluster_locations[cluster_id][1], marker='x', color=color, s=100, alpha=alpha)
                text_1 = axes[1].annotate(f"{cluster_id}", (np.max(kilo_cluster_templates[cluster_id].ptp(0))+1, kilo_cluster_locations[cluster_id][1]+2.5), color=color)
                axes[2].scatter(kilo_cluster_locations[cluster_id][0], kilo_cluster_locations[cluster_id][1], marker='x', color=color, s=100, alpha=alpha)
                text_2 = axes[2].annotate(f"{cluster_id}", (kilo_cluster_locations[cluster_id][0]+2.5, kilo_cluster_locations[cluster_id][1]+2.5), color=color)
                text_0.set_alpha(alpha)
                text_1.set_alpha(alpha)
                text_2.set_alpha(alpha)
                axes[0].set_xlim(-20, 80)
                axes[1].set_xlim(0, 30)
                axes[2].set_xlim(-20, 80)
    # fig.savefig(f"{save_dir_path}/full_scatter_{z_cutoff[0]}_{z_cutoff[1]}", dpi=200)
    # plt.close(fig)

# %%
###hdbscan
import os
save_dir_path = "clustering_results_split_merge"
if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)

# %%
z_cutoffs= [(0,550), (500,1050), (1000,1550), (1500,2050), (2000,2550), (2500,3050), (3000,3550), (3500,4050)]
for z_cutoff in z_cutoffs:
    fig, axes = cluster_viz.array_scatter(
            clusterer.labels_, geom, tx, tz, tmaxptps, 
            zlim=z_cutoff,
    )
    for cluster_id in sorting_kilo.get_unit_ids():
        if len(sorting_kilo.get_unit_spike_train(cluster_id)) > 25:
            if cluster_id in good_kilo_sort_clusters:
                color = 'red'
                alpha = 1
            else:
                color = 'blue'
                alpha = .4
            if cluster_id in good_kilo_sort_clusters:
                axes[0].scatter(kilo_cluster_locations[cluster_id][0], kilo_cluster_locations[cluster_id][1], marker='x', color=color, s=100, alpha=alpha)
                text_0 = axes[0].annotate(f"{cluster_id}", (kilo_cluster_locations[cluster_id][0]+2.5, kilo_cluster_locations[cluster_id][1]+2.5), color=color)
                axes[1].scatter(np.max(kilo_cluster_templates[cluster_id].ptp(0)), kilo_cluster_locations[cluster_id][1], marker='x', color=color, s=100, alpha=alpha)
                text_1 = axes[1].annotate(f"{cluster_id}", (np.max(kilo_cluster_templates[cluster_id].ptp(0))+1, kilo_cluster_locations[cluster_id][1]+2.5), color=color)
                axes[2].scatter(kilo_cluster_locations[cluster_id][0], kilo_cluster_locations[cluster_id][1], marker='x', color=color, s=100, alpha=alpha)
                text_2 = axes[2].annotate(f"{cluster_id}", (kilo_cluster_locations[cluster_id][0]+2.5, kilo_cluster_locations[cluster_id][1]+2.5), color=color)
                text_0.set_alpha(alpha)
                text_1.set_alpha(alpha)
                text_2.set_alpha(alpha)
                axes[0].set_xlim(-20, 80)
                axes[1].set_xlim(0, 30)
                axes[2].set_xlim(-20, 80)
    fig.savefig(f"{save_dir_path}/full_scatter_{z_cutoff[0]}_{z_cutoff[1]}", dpi=200)
    plt.close(fig)

# %%
from joblib import Parallel, delayed

save_dir_parallel = save_dir_path + "/unit_summaries"
###hdbscan###
if not os.path.exists(save_dir_parallel):
    os.makedirs(save_dir_parallel)

for cluster_id in np.setdiff1d(np.unique(clusterer.labels_), [-1]):
    fig = plot_single_unit_summary(
        cluster_id,
        labels,
        spike_index,
        cluster_centers,
        geom,
        x,
        z,
        maxptps,
        firstchans,
        wfs_full_denoise,
        wfs_subtracted,
        raw_bin,
        residual_bin,
        num_spikes_plot=100, 
        num_rows_plot=3, 
        t_range=(30,90), 
        plot_all_points=False, 
        num_channels=40
    )
    save_z_int = int(cluster_centers.loc[cluster_id][1])
    save_str = str(save_z_int).zfill(4)
    fig.savefig(save_dir_parallel + f"/Z{save_str}_cluster{cluster_id}.png", transparent=False, pad_inches=0)
    plt.close(fig)

# %%
num_close_clusters = 50
num_close_clusters_plot=10
num_channels_similarity = 20
shifts_align=np.arange(-8,9)
num_channels = 40
num_spikes_plot = 100

save_dir_similarity = save_dir_path + "/kilo_venns_similarities"
###hdbscan###
if not os.path.exists(save_dir_similarity):
    os.makedirs(save_dir_similarity)
    
for good_cluster_id in good_kilo_sort_clusters:
    cluster_id_kilo = good_cluster_id
    cluster_id = int(cmp_kilo.get_best_unit_match1(cluster_id_kilo))    
    if cluster_id != -1:
        st_1 = spike_index[:,0][np.where(labels==cluster_id)]
        st_2 = sorting_kilo.get_unit_spike_train(cluster_id_kilo)
        sorting1_name = "hdb"
        sorting2_name = "kilo"

        z_uniq, z_ids = np.unique(geom[:, 1], return_inverse=True)
        all_max_ptp = maxptps[labels==cluster_id].max()
        scale = (z_uniq[1] - z_uniq[0]) / max(7, all_max_ptp)

        firstchans_cluster_sorting1 = firstchans[labels == cluster_id]
        mcs_abs_cluster_sorting1 = spike_index[:,1][labels == cluster_id]

        spike_depths = kilo_spike_depths[np.where(kilo_spike_clusters==cluster_id_kilo)]
        mcs_abs_cluster_sorting2 = np.asarray([np.argmin(np.abs(spike_depth - geom[:,1])) for spike_depth in spike_depths])
        firstchans_cluster_sorting2 = (mcs_abs_cluster_sorting2 - 20).clip(min=0)

        fig = plot_agreement_venn(cluster_id, cluster_id_kilo, st_1, st_2, firstchans_cluster_sorting1, mcs_abs_cluster_sorting1, firstchans_cluster_sorting2, mcs_abs_cluster_sorting2,
                                  geom, raw_bin, scale=scale, sorting1_name=sorting1_name, sorting2_name=sorting2_name, num_channels=40, num_spikes_plot=200, t_range=(30,90), num_rows=3, 
                                  alpha=.1);
        
        save_z_int = int(kilo_spike_depths[cluster_id_kilo])
        save_str = str(save_z_int).zfill(4)
        fig.savefig(save_dir_similarity + f"/Z{save_str}_kscluster{cluster_id_kilo}_hdbcluster{cluster_id}.png", transparent=False, pad_inches=0)
        plt.close(fig)
        
    
    else:
        #compute K closest hdbscan clsuters
        closest_clusters = get_closest_clusters_kilosort_hdbscan(cluster_id_kilo, kilo_cluster_depth_means, cluster_centers, num_close_clusters)
        fig = plot_unit_similarities(cluster_id_kilo, closest_clusters, sorting_kilo, sorting_hdbl_t, geom, raw_bin, recording_duration, num_channels, num_spikes_plot, num_channels_similarity=num_channels_similarity, 
                                     num_close_clusters_plot=num_close_clusters_plot, num_close_clusters=num_close_clusters, shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="both")
        save_z_int = int(kilo_spike_depths[cluster_id_kilo])
        save_str = str(save_z_int).zfill(4)
        fig.savefig(save_dir_similarity + f"/Z{save_str}_kscluster{cluster_id_kilo}.png", transparent=False, pad_inches=0)
        plt.close(fig)
