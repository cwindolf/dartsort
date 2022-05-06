# %%
# %load_ext autoreload
# %autoreload 2

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import argparse
import spikeinterface.comparison as sc
from spikeinterface.extractors import NumpySorting
import spikeinterface.widgets as sw

# %%
from spike_psvae import (
    # cluster,
    # merge_split_cleaned,
    # cluster_viz_index,
    # denoise,
    cluster_utils,
    # triage,
    cluster_viz,
    localization,
)

# ap = argparse.ArgumentParser()
# ap.add_argument("subject")
# args = ap.parse_args()
# subject = args.subject

subject = "SWC_054"

hybrid_bin_dir = Path("/mnt/3TB/charlie/hybrid_1min_output/")
hybrid_res_dir = Path("/mnt/3TB/charlie/hybrid_1min_res/")

raw_data_bin = hybrid_bin_dir / f"{subject}.ap.bin"
assert raw_data_bin.exists()
residual_data_bin = next((hybrid_res_dir / subject).glob("residual*bin"))
assert residual_data_bin.exists()
sub_h5 = next((hybrid_res_dir / subject).glob("sub*h5"))
assert sub_h5.exists()

hybrid_gt_h5 = hybrid_bin_dir / f"{subject}_gt.h5"
output_dir = hybrid_res_dir / subject / "gt_comparison"
output_dir.mkdir(exist_ok=True)

# %%
sub_h5

# %%
hybrid_res_dir / subject / "aligned_spike_index.npy"

# %%
# %ll {{hybrid_res_dir /subject}}

# %% tags=[]
# load features
with h5py.File(sub_h5, "r") as h5:
    geom_array = h5["geom"][:]
    end_sample = h5["end_sample"][()]
    start_sample = h5["start_sample"][()]
    #     print(start_sample, end_sample)
    o_spike_index = h5["spike_index"][:]
    #     # x, y, z, alpha, z_rel = h5["localizations"][:].T
    #     maxptps = h5["maxptps"][:]
    #     z_abs = h5["z_reg"][:]
    #     firstchans = h5["first_channels"][:]

    recording_length = (end_sample - start_sample) / 30000

    #     # start_sample += offset_min * 60 * 30000
    #     # end_sample += offset_min * 60 * 30000
    channel_index = h5["channel_index"][:]
#     z_reg = h5["z_reg"][:]
hdb_spike_index = np.load(hybrid_res_dir / subject / "aligned_spike_index.npy")
print("al", hdb_spike_index.shape)
hdb_spike_index = o_spike_index
print("unal", hdb_spike_index.shape)
hdb_labels = np.load(hybrid_res_dir / subject / "labels.npy")
print("lab", hdb_labels.shape)
cluster_centers = pd.read_csv(hybrid_res_dir / subject / "cluster_centers.csv")
nontriaged = np.flatnonzero(hdb_labels >= 0)
hdb_spike_train = np.c_[
    hdb_spike_index[nontriaged, 0],
    hdb_labels[nontriaged],
]
hdb_spike_index = hdb_spike_index[nontriaged]

# %%
with h5py.File(hybrid_gt_h5, "r") as gt_h5:
    gt_spike_train = gt_h5["spike_train"][:]
    gt_spike_index = gt_h5["spike_index"][:]
    templates = gt_h5["templates"][:]

gt_template_maxchans = templates.ptp(1).argmax(1)
gt_template_locs = localization.localize_ptps_index(
    templates.ptp(1),
    geom_array,
    gt_template_maxchans,
    np.stack([np.arange(len(geom_array))] * len(geom_array), axis=0),
    n_channels=20,
    n_workers=None,
    pbar=True,
)
gt_template_depths = gt_template_locs[3]

# sort gt train
argsort = np.argsort(gt_spike_train[:, 0])
gt_spike_train = gt_spike_train[argsort]
gt_spike_index = gt_spike_index[argsort]
# TODO: return trough times in simulator
gt_spike_train[:, 0] += 42
gt_spike_index[:, 0] += 42

# %%
hdb_units, hdb_counts = np.unique(hdb_spike_train[:, 1], return_counts=True)
plt.hist(hdb_counts, bins=32);

# %%
gt_units, gt_counts = np.unique(gt_spike_train[:, 1], return_counts=True)
plt.hist(gt_counts, bins=32);

# %%
print("gt", gt_spike_train[:10])
sorting_gt = NumpySorting.from_times_labels(
    times_list=gt_spike_train[:, 0],
    labels_list=gt_spike_train[:, 1],
    sampling_frequency=30_000,
)
print("hdb", hdb_spike_train[:10])
sorting_hdb = NumpySorting.from_times_labels(
    times_list=hdb_spike_train[:, 0],
    labels_list=hdb_spike_train[:, 1],
    sampling_frequency=30_000,
)
cmp_1 = sc.compare_two_sorters(
    sorting_hdb,
    sorting_gt,
    sorting1_name="ours",
    sorting2_name="hybrid_gt",
    verbose=True
)
cmp_gt_1 = sc.compare_two_sorters(
    sorting_gt,
    sorting_hdb,
    sorting1_name="hybrid_gt",
    sorting2_name="ours",
    verbose=True
)

# %%
sorting_hdb.get_unit_spike_train(-1)

# %%
fig = plt.figure(figsize=(80, 80))
sw.plot_agreement_matrix(cmp_gt_1, figure=fig, ordered=True)

# %%
cmp_gt_1.agreement_scores.values.max()

# %%
cmp_gt_1.agreement_scores.values[25,190]

# %%
cmp_gt_1.get

# %%
cmp_gt_1.get_best_unit_match1(25)

# %%
num_channels = 40

for gt_unit in tqdm(np.unique(gt_spike_train[:, 1])):
    num_spikes = len(sorting_gt.get_unit_spike_train(gt_unit))
    cluster_id = int(cmp_gt_1.get_best_unit_match1(gt_unit))
    if cluster_id != -1:
        fig = cluster_viz.plot_agreement_venn(
            cluster_id,
            gt_unit,
            cmp_1,
            sorting_hdb,
            sorting_gt,
            "hdb",
            "hybrid_gt",
            geom_array,
            num_channels,
            200,
            channel_index[
                hdb_spike_index[hdb_spike_train[:, 1] == cluster_id, 1],
                0,
            ],
            hdb_spike_index[hdb_spike_train[:, 1] == cluster_id, 1],
            channel_index[
                gt_spike_index[gt_spike_train[:, 1] == cluster_id, 1],
                0,
            ],
            gt_spike_index[gt_spike_train[:, 1] == cluster_id, 1],
            raw_data_bin,
            delta_frames=12,
            alpha=0.2,
        )
        fig.savefig(
            output_dir / f"gt{gt_unit:02d}_hdb{cluster_id}_comparison.png",
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        print(f"skipped {gt_unit} with {num_spikes} spikes")

    if num_spikes > 0:
        # plot specific kilosort example
        num_close_clusters = 50
        num_close_clusters_plot = 10
        num_channels_similarity = 20
        shifts_align = np.arange(-8, 9)

        st_1 = sorting_gt.get_unit_spike_train(gt_unit)

        # compute K closest hdbscan clsuters
        closest_clusters = cluster_utils.get_closest_clusters_kilosort_hdbscan(
            gt_unit,
            gt_template_depths,
            cluster_centers,
            num_close_clusters,
        )

        num_close_clusters = 50
        num_close_clusters_plot = 10
        num_channels_similarity = 20
        fig = cluster_viz.plot_unit_similarities(
            gt_unit,
            closest_clusters,
            sorting_gt,
            sorting_hdb,
            geom_array,
            raw_data_bin,
            recording_length,
            num_channels,
            200,
            num_channels_similarity=num_channels_similarity,
            num_close_clusters_plot=num_close_clusters_plot,
            num_close_clusters=num_close_clusters,
            shifts_align=shifts_align,
            order_by="similarity",
            normalize_agreement_by="both",
        )
        fig.savefig(
            output_dir / f"gt{gt_unit:02d}_summary.png",
            bbox_inches="tight",
        )
        plt.close(fig)

# %%
