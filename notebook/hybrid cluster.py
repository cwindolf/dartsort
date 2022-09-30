# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
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
import hdbscan
from pathlib import Path
import torch
from joblib.externals import loky
from tqdm.auto import tqdm, trange

# %%
from spike_psvae import cluster, merge_split_cleaned, cluster_viz_index, denoise, cluster_utils, triage, cluster_viz

# %%
plt.rc("figure", dpi=200)

# %%
offset_min = 30
min_cluster_size = 25
min_samples = 25

# %%
# # raw_data_bin = Path("/mnt/3TB/charlie/re_snips_5min/CSH_ZAD_026_snip.ap.bin")
# raw_data_bin = Path("/mnt/3TB/charlie/re_snips/ibl_witten_27_snip.ap.bin")
# assert raw_data_bin.exists()
# residual_data_bin = Path("/mnt/3TB/charlie/shake_res/re_snips_dndres/residual_ibl_witten_27_snip.ap_t_0_None.bin")
# assert residual_data_bin.exists()
# sub_h5 = Path("/mnt/3TB/charlie/shake_res/re_snips_dndres/subtraction_ibl_witten_27_snip.ap_t_0_None.h5")
# assert sub_h5.exists()

# %%
hybrid_bin_dir = Path("/mnt/3TB/charlie/hybrid_1min_output/")
hybrid_res_dir = Path("/mnt/3TB/charlie/hybrid_1min_subtraction/")
dset = "SWC_054"

# %%
raw_data_bin = hybrid_bin_dir / f"{dset}.ap.bin"
assert raw_data_bin.exists()
residual_data_bin = next((hybrid_res_dir / dset).glob("residual*bin"))
assert residual_data_bin.exists()
sub_h5 = next((hybrid_res_dir / dset).glob("sub*h5"))
assert sub_h5.exists()

# %%
hybrid_gt_h5 = hybrid_bin_dir / f"{dset}_gt.h5"
output_dir = hybrid_res_dir / dset / "gt_comparison"
output_dir.mkdir(exist_ok=True)

# %%
with h5py.File(hybrid_gt_h5, "r") as gt_h5:
    gt_spike_train = gt_h5["spike_train"][:]
    gt_spike_index = gt_h5["spike_index"][:]
    templates = gt_h5["templates"][:]
    wfs = gt_h5["waveforms"][:]

# %%
wfs.ptp(1).max(1).mean()

# %%
n, b, p = plt.hist(wfs.ptp(1).max(1), bins=128, density=True);
# plt.hist(templates.ptp(1).max(1), bins=b, density=True);
plt.axvline(3, color="silver")
plt.axvline(wfs.ptp(1).max(1).mean(), color="k")

# %%
# output_dir = Path("/mnt/3TB/charlie/dnd_pyks_5min/")
# output_dir.mkdir(exist_ok=True)
output_dir = None

# %% tags=[]
#load features
with h5py.File(sub_h5, "r") as h5:
    spike_index = h5["spike_index"][:]
    x, y, z, alpha, z_rel = h5["localizations"][:].T
    maxptps = h5["maxptps"][:]
    z_abs = h5["z_reg"][:]
    geom_array = h5["geom"][:]
    firstchans = h5["first_channels"][:]
    end_sample = h5["end_sample"][()]
    start_sample = h5["start_sample"][()]
    print(start_sample, end_sample)
    
    recording_length = (end_sample - start_sample) // 30000
    
    start_sample += offset_min * 60 * 30000
    end_sample += offset_min * 60 * 30000
    channel_index = h5["channel_index"][:]
    z_reg = h5["z_reg"][:]
    
num_spikes = spike_index.shape[0]
end_time = end_sample / 30000
start_time = start_sample / 30000

# %%
with np.load("/mnt/3TB/charlie/CSH_ZAD_026_pyks.npz") as f:
    kilo_spike_samples = f["samples"]
    which = (start_sample + 60 <= kilo_spike_samples) & (kilo_spike_samples <= end_sample - 61)
    kilo_spike_samples = kilo_spike_samples[which].astype(int)
    kilo_spike_clusters = f["clusters"][which]
    kilo_spike_depths = f["depths"][which]

kilo_spike_frames = kilo_spike_samples - offset_min * 60 * 30000 #to match our detection alignment
kilo_cluster_depth_means = {}
for cluster_id in np.unique(kilo_spike_clusters):
    kilo_cluster_depth_means[cluster_id] = np.mean(kilo_spike_depths[kilo_spike_clusters==cluster_id]) 
    
#create kilosort SpikeInterface sorting
sorting_kilo = cluster_utils.make_sorting_from_labels_frames(kilo_spike_clusters, kilo_spike_frames)
    
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
good_kilo_sort_clusters = np.asarray(good_kilo_sort_clusters)

# %%
triaged_log_ptp = tmaxptps.copy()
triaged_log_ptp[triaged_log_ptp >= 27.5] = 27.5
triaged_log_ptp = np.log(triaged_log_ptp+1)
triaged_log_ptp[triaged_log_ptp<=1.25] = 1.25
triaged_ptp_rescaled = (triaged_log_ptp - triaged_log_ptp.min())/(triaged_log_ptp.max() - triaged_log_ptp.min())
color_arr = plt.cm.viridis(triaged_ptp_rescaled)
color_arr[:, 3] = triaged_ptp_rescaled

# ## Define colors
unique_colors = ['#e6194b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3', '#808000', '#000075', '#000000']

cluster_color_dict = {}
for cluster_id in np.unique(clusterer.labels_):
    cluster_color_dict[cluster_id] = unique_colors[cluster_id % len(unique_colors)]
cluster_color_dict[-1] = '#808080' #set outlier color to grey

# %%
# create hdbscan/localization SpikeInterface sorting (with triage)
sorting_hdbl_t = cluster_utils.make_sorting_from_labels_frames(clusterer.labels_, spike_index[idx_keep_full,0])

# cmp_5 = cluster_utils.compare_two_sorters(sorting_hdbl_t, sorting_kilo, sorting1_name='ours', sorting2_name='kilosort', match_score=.5)
# matched_units_5 = cmp_5.get_matching()[0].index.to_numpy()[np.where(cmp_5.get_matching()[0] != -1.)]
# matches_kilos_5 = cmp_5.get_best_unit_match1(matched_units_5).values.astype('int')

cmp_1 = cluster_utils.compare_two_sorters(sorting_hdbl_t, sorting_kilo, sorting1_name='ours', sorting2_name='kilosort', match_score=.1)
matched_units_1 = cmp_1.get_matching()[0].index.to_numpy()[np.where(cmp_1.get_matching()[0] != -1.)]
unmatched_units_1 = cmp_1.get_matching()[0].index.to_numpy()[np.where(cmp_1.get_matching()[0] == -1.)]
matches_kilos_1 = cmp_1.get_best_unit_match1(matched_units_1).values.astype('int')

# %%
# cmp_kilo_5 = cluster_utils.compare_two_sorters(sorting_kilo, sorting_hdbl_t, sorting1_name='kilosort', sorting2_name='ours', match_score=.5)
# matched_units_kilo_5 = cmp_kilo_5.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_5.get_matching()[0] != -1.)]
# unmatched_units_kilo_5 = cmp_kilo_5.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_5.get_matching()[0] == -1.)]

cmp_kilo_1 = cluster_utils.compare_two_sorters(sorting_kilo, sorting_hdbl_t, sorting1_name='kilosort', sorting2_name='ours', match_score=.1)
matched_units_kilo_1 = cmp_kilo_1.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_1.get_matching()[0].to_numpy() != -1.)]
unmatched_units_kilo_1 = cmp_kilo_1.get_matching()[0].index.to_numpy()[np.where(cmp_kilo_1.get_matching()[0].to_numpy() == -1.)]

# %%
gudir = Path(output_dir / "kilo_comparison")
gudir.mkdir(exist_ok=True)

num_channels = 40

for good_kilo_sort_cluster in tqdm(good_kilo_sort_clusters):
    num_spikes = len(sorting_kilo.get_unit_spike_train(cluster_id_match))
    cluster_id_match = good_kilo_sort_cluster
    cluster_id = int(cmp_kilo_1.get_best_unit_match1(cluster_id_match))
    if cluster_id != -1:
        depth = int(kilo_cluster_depth_means[cluster_id_match])
        save_str = str(depth).zfill(4)
        sorting1 = sorting_hdbl_t
        sorting2 = sorting_kilo
        sorting1_name = "hdb"
        sorting2_name = "kilo"
        firstchans_cluster_sorting1 = firstchans[labels == cluster_id]
        mcs_abs_cluster_sorting1 = spike_index[labels == cluster_id, 1]
        spike_depths = kilo_spike_depths[np.where(kilo_spike_clusters == cluster_id_match)]
        mcs_abs_cluster_sorting2 = np.asarray([np.argmin(np.abs(spike_depth - geom_array[:,1])) for spike_depth in spike_depths])
        firstchans_cluster_sorting2 = (mcs_abs_cluster_sorting2 - 20).clip(min=0)

        fig = cluster_viz.plot_agreement_venn(
            cluster_id, cluster_id_match, cmp_1, sorting1, sorting2, sorting1_name, sorting2_name,
            geom_array, num_channels, 200, firstchans_cluster_sorting1, mcs_abs_cluster_sorting1, 
            firstchans_cluster_sorting2, mcs_abs_cluster_sorting2, raw_data_bin, delta_frames = 12, alpha=.2
        )
        fig.savefig(gudir / f"Z{save_str}_{cluster_id_match}_{cluster_id}_comparison.png", bbox_inches="tight")
        plt.close(fig)
    else:
        print(f"skipped {cluster_id_match} with {num_spikes} spikes")

    if num_spikes > 0:
        #plot specific kilosort example
        num_close_clusters = 50
        num_close_clusters_plot=10
        num_channels_similarity = 20
        shifts_align=np.arange(-8,9)

        st_1 = sorting_kilo.get_unit_spike_train(cluster_id_match)

        #compute K closest hdbscan clsuters
        closest_clusters = cluster_utils.get_closest_clusters_kilosort_hdbscan(
            cluster_id_match, kilo_cluster_depth_means, cluster_centers, num_close_clusters)

        num_close_clusters = 50
        num_close_clusters_plot=10
        num_channels_similarity = 20
        fig = cluster_viz.plot_unit_similarities(
            cluster_id_match, closest_clusters, sorting_kilo, 
            sorting_hdbl_t, geom_array, raw_data_bin, end_time - start_time, 
            num_channels, 200, num_channels_similarity=num_channels_similarity, 
            num_close_clusters_plot=num_close_clusters_plot,
            num_close_clusters=num_close_clusters,
            shifts_align = shifts_align, order_by ='similarity', normalize_agreement_by="both")
        fig.savefig(gudir / f"Z{save_str}_{cluster_id_match}_summary.png", bbox_inches="tight")
        plt.close(fig)

# %%
len(spike_index), idx_keep_full.size

# %%
np.unique(labels).size

# %%
