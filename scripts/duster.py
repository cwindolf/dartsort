import argparse

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from joblib.externals import loky
from tqdm.auto import tqdm
import pickle
from sklearn.decomposition import PCA
import time
import subprocess

from spike_psvae import (
    pre_deconv_merge_split,
    cluster_viz_index,
    denoise,
    cluster_utils,
    spike_train_utils
    # cluster_viz,
)

ap = argparse.ArgumentParser()

ap.add_argument("raw_data_bin")
ap.add_argument("residual_data_bin")
ap.add_argument("sub_h5")
ap.add_argument("output_dir")
ap.add_argument("--tmpdir", type=Path, default=None)
ap.add_argument("--inmem", action="store_true")
ap.add_argument("--doplot", action="store_true")
ap.add_argument("--doscatter", action="store_true")
ap.add_argument("--noremoveselfdups", action="store_true")
ap.add_argument("--usemean", action="store_true")
ap.add_argument("--plotdir", type=str, default=None)
ap.add_argument("--merge_dipscore", type=float, default=1.0)

args = ap.parse_args()

plotdir = Path(args.plotdir if args.plotdir else args.output_dir)

# %%
np.random.seed(1)
plt.rc("figure", dpi=200)

# %%
offset_min = 30
min_cluster_size = 25
min_samples = 25

# %%
raw_data_bin = Path(args.raw_data_bin)
assert raw_data_bin.exists()
residual_data_bin = Path(args.residual_data_bin)
assert residual_data_bin.exists()
sub_h5 = Path(args.sub_h5)
assert sub_h5.exists()

print(raw_data_bin)
print(residual_data_bin)


class timer:
    def __init__(self, name="timer"):
        self.name = name
        print("start", name, "...")

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.t = time.time() - self.start
        print(self.name, "took", self.t, "s")


print("cooking", flush=True)
if args.tmpdir is not None:
    with timer("copying h5 to scratch"):
        subprocess.run(["rsync", "-avP", sub_h5, args.tmpdir / "sub.h5"])
    sub_h5 = args.tmpdir / "sub.h5"
    with timer("copying bin to scratch"):
        subprocess.run(["rsync", "-avP", raw_data_bin, args.tmpdir / "raw.bin"])
    raw_data_bin = args.tmpdir / "raw.bin"

# %%
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)

# %%
denoiser = denoise.SingleChanDenoiser().load()
device = "cuda"
denoiser.to(device)

# %% tags=[]
# load features
with h5py.File(sub_h5, "r") as h5:
    spike_index = h5["spike_index"][:]
    x, y, z, alpha, z_rel = h5["localizations"][:].T
    maxptps = h5["maxptps"][:]
    z_abs = h5["z_reg"][:]
    geom = h5["geom"][:]
    firstchans = h5["first_channels"][:]
    end_sample = h5["end_sample"][()]
    start_sample = h5["start_sample"][()]
    print(start_sample, end_sample)

    recording_length = (end_sample - start_sample) // 30000

    start_sample += offset_min * 60 * 30000
    end_sample += offset_min * 60 * 30000
    channel_index = h5["channel_index"][:]
    z_reg = h5["z_reg"][:]

    tpca_mean = h5["tpca_mean"][:]
    tpca_components = h5["tpca_components"][:]

num_spikes = spike_index.shape[0]
end_time = end_sample / 30000
start_time = start_sample / 30000

tpca = PCA(tpca_components.shape[0])
tpca.mean_ = tpca_mean
tpca.components_ = tpca_components

reducer = np.mean if args.usemean else np.median

# %%
(
    clusterer,
    cluster_centers,
    tspike_index,
    tx,
    tz,
    tmaxptps,
    idx_keep_full,
) = cluster_utils.cluster_spikes(
    x,
    z,
    maxptps,
    spike_index,
    split_big=True,
)

# remove self-duplicate spikes
if not args.noremoveselfdups:
    kept_ix, removed_ix = cluster_utils.remove_self_duplicates(
        tspike_index[:, 0],
        clusterer.labels_,
        raw_data_bin,
        geom.shape[0],
        frame_dedup=20,
    )
    clusterer.labels_[removed_ix] = -1

# labels in full index space (not triaged)
labels = np.full(x.shape, -1)
labels[idx_keep_full] = clusterer.labels_
labels_original = labels.copy()

# %%
if args.doplot:
    z_cutoff = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    for za, zb in zip(z_cutoff, z_cutoff[1:]):
        fig, _ = cluster_viz_index.array_scatter(
            clusterer.labels_,
            geom,
            tx,
            tz,
            tmaxptps,
            zlim=(za, zb),
        )
        fig.savefig(plotdir / f"B_pre_split_full_scatter_{za}_{zb}", dpi=200)
        plt.close(fig)

# %%
spike_train = np.c_[
    spike_index[:, 0],
    labels,
]
del labels
(
    spike_train,
    order,
    templates,
    template_shifts,
) = spike_train_utils.clean_align_and_get_templates(
    spike_train,
    geom.shape[0],
    raw_data_bin,
    sort_by_time=False,
    reducer=reducer,
    min_n_spikes=0,
    pbar=True,
)
spike_index = np.c_[spike_train[:, 0], spike_index[:, 1]]

# save
print("Save pre merge/split...")
np.save(output_dir / "pre_merge_split_labels.npy", spike_train[:, 1])
np.save(
    output_dir / "pre_merge_split_aligned_spike_index.npy",
    spike_index,
)
np.save(
    output_dir / "pre_merge_split_aligned_templates.npy", templates
)
np.save(
    output_dir / "pre_merge_split_template_shifts.npy", template_shifts
)


# %%
# split
h5 = h5py.File(sub_h5, "r")
sub_wf = h5["subtracted_waveforms"]
if args.inmem:
    sub_wf = sub_wf[:]
spike_train[:, 1] = pre_deconv_merge_split.split_clusters(
    residual_data_bin,
    sub_wf,
    firstchans,
    spike_index,
    templates.ptp(1).argmax(1),
    template_shifts,
    spike_train[:, 1],
    x,
    z_reg,
    # maxptps,
    geom,
    denoiser,
    device,
    tpca,
)

# ks split
print("before ks split", spike_train[:, 1].max() + 1)
spike_train[:, 1], split_map = pre_deconv_merge_split.ks_maxchan_tpca_split(
    h5["subtracted_tpca_projs"],
    channel_index,
    spike_index[:, 1],
    spike_train[:, 1],
    tpca,
    recursive=True,
    top_pc_init=True,
    aucsplit=0.85,
    min_size_split=50,
    max_split_corr=0.9,
    min_amp_sim=0.2,
    min_split_prop=0.05,
)
print("after ks split", spike_train[:, 1].max() + 1)

# %%
# re-order again
clusterer.labels_ = spike_train[:, 1][idx_keep_full]
cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
clusterer = cluster_utils.relabel_by_depth(clusterer, cluster_centers)
cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
spike_train[idx_keep_full, 1] = clusterer.labels_

# %%
if args.doplot:
    z_cutoff = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    for za, zb in zip(z_cutoff, z_cutoff[1:]):
        fig, _ = cluster_viz_index.array_scatter(
            clusterer.labels_,
            geom,
            tx,
            tz,
            tmaxptps,
            zlim=(za, zb),
        )
        fig.savefig(plotdir / f"C_after_split_full_scatter_{za}_{zb}", dpi=200)
        plt.close(fig)

# %%
(
    spike_train,
    order,
    templates,
    template_shifts,
) = spike_train_utils.clean_align_and_get_templates(
    spike_train,
    geom.shape[0],
    raw_data_bin,
    sort_by_time=False,
    reducer=reducer,
    min_n_spikes=0,
    pbar=True,
)
spike_index = np.c_[spike_train[:, 0], spike_index[:, 1]]

# %%
# merge
K_pre = spike_train[:, 1].max() + 1
spike_train[:, 1] = pre_deconv_merge_split.get_merged(
    residual_data_bin,
    sub_wf,
    firstchans,
    geom,
    templates,
    template_shifts,
    len(templates),
    spike_index,
    spike_train[:, 1],
    x,
    z_reg,
    denoiser,
    device,
    tpca,
    threshold_diptest=args.merge_dipscore,
)
print("pre->post merge", K_pre, spike_train[:, 1].max() + 1)

# %%
# re-order again
clusterer.labels_ = spike_train[idx_keep_full, 1]
cluster_centers = cluster_utils.compute_cluster_centers(clusterer)
clusterer = cluster_utils.relabel_by_depth(clusterer, cluster_centers)
spike_train[idx_keep_full, 1] = clusterer.labels_

# %%
# final templates
(
    spike_train,
    order,
    templates,
    template_shifts,
) = spike_train_utils.clean_align_and_get_templates(
    spike_train,
    geom.shape[0],
    raw_data_bin,
    sort_by_time=False,
    reducer=reducer,
    min_n_spikes=0,
    pbar=True,
)
spike_index = np.c_[spike_train[:, 0], spike_index[:, 1]]


# save
print("Save final...")
np.save(output_dir / "labels.npy", spike_train[:, 1])
np.save(output_dir / "aligned_spike_index.npy", spike_index)
np.save(output_dir / "templates.npy", templates)
np.save(output_dir / "aligned_templates.npy", templates)
np.save(output_dir / "template_shifts.npy", template_shifts)


# %%
if args.doplot:

    z_cutoff = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    for za, zb in zip(z_cutoff, z_cutoff[1:]):
        fig, _ = cluster_viz_index.array_scatter(
            clusterer.labels_,
            geom,
            tx,
            tz,
            tmaxptps,
            zlim=(za - 50, zb + 50),
        )
        fig.savefig(plotdir / f"AAA_final_full_scatter_{za}_{zb}", dpi=200)
        plt.close(fig)

    # %%
    triaged_log_ptp = tmaxptps.copy()
    triaged_log_ptp[triaged_log_ptp >= 27.5] = 27.5
    triaged_log_ptp = np.log(triaged_log_ptp + 1)
    triaged_log_ptp[triaged_log_ptp <= 1.25] = 1.25
    triaged_ptp_rescaled = (triaged_log_ptp - triaged_log_ptp.min()) / (
        triaged_log_ptp.max() - triaged_log_ptp.min()
    )
    color_arr = plt.cm.viridis(triaged_ptp_rescaled)
    color_arr[:, 3] = triaged_ptp_rescaled

    # ## Define colors
    unique_colors = (
        "#e6194b,#4363d8,#f58231,#911eb4,#46f0f0,"
        "#f032e6,#008080,#e6beff,#9a6324,#800000,"
        "#aaffc3,#808000,#000075,#000000",
    ).split(",")

    cluster_color_dict = {}
    for cluster_id in np.unique(clusterer.labels_):
        cluster_color_dict[cluster_id] = unique_colors[
            cluster_id % len(unique_colors)
        ]
    cluster_color_dict[-1] = "#808080"  # set outlier color to grey

    # %%
    cluster_centers.index

    # %%
    sudir = Path(plotdir / "singleunit")
    sudir.mkdir(exist_ok=True)

    # plot cluster summary

    def job(cluster_id):
        if (sudir / f"unit_{cluster_id:03d}.png").exists():
            return
        with h5py.File(sub_h5, "r") as d:
            fig = cluster_viz_index.single_unit_summary(
                cluster_id,
                clusterer,
                spike_train[:, 1],
                geom,
                idx_keep_full,
                x,
                z,
                maxptps,
                channel_index,
                spike_index,
                d["cleaned_waveforms"],
                d["subtracted_waveforms"],
                raw_data_bin,
                residual_data_bin,
                spikes_plot=100,
                num_rows=3,
            )
            fig.savefig(sudir / f"unit_{cluster_id:03d}.png")
            plt.close(fig)

    i = 0
    with loky.ProcessPoolExecutor(
        12,
    ) as p:
        units = np.setdiff1d(np.unique(clusterer.labels_), [-1])
        for res in tqdm(p.map(job, units), total=len(units)):
            pass
else:
    print("No single unit figs, bye.")

if args.tmpdir is not None:
    print("Deleting scratch")
    (args.tmpdir / "sub.h5").unlink()
