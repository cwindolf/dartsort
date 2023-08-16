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
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

# %%
from spike_psvae import (
    cluster_viz_index,
    grab_and_localize,
    localize_index,
    point_source_centering,
)

# %%
import torch

# %%
from spike_psvae import denoise
dn = denoise.SingleChanDenoiser().load()

# %%
from scipy.signal import resample

# %%
import colorcet as cc

# %%

# %%
# %matplotlib inline
plt.rc("figure", dpi=200)
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# %%
subject = "CSHL051"

# %%
hybrid_bin_dir = Path("/mnt/3TB/charlie/hybrid_5min/hybrid_5min_output/")

# %%
hybrid_gt_h5 = hybrid_bin_dir / f"{subject}_gt.h5"
raw_data_bin = hybrid_bin_dir / f"{subject}.ap.bin"

with h5py.File(hybrid_gt_h5, "r") as gt_h5:
    gt_spike_train = gt_h5["spike_train"][:]
    gt_spike_index = gt_h5["spike_index"][:]
    gt_templates = gt_h5["templates"][:]
    geom = gt_h5["geom"][:]
    ci = gt_h5["write_channel_index"][:]
    choices = gt_h5["choices"][:]

# %%
tptps = gt_templates.ptp(1).max(1)

# %%
txs, tys, tzs, tzrs, tas = localize_index.localize_ptps_index(
    gt_templates.ptp(1),
    geom,
    gt_templates.ptp(1).argmax(1),
    np.arange(384)[None, :] * np.ones((384,384),dtype=int),
    radius=100,
)

# %%
rg = np.random.default_rng()

# %%
gt_relocalizations, gt_remaxptp = grab_and_localize.grab_and_localize(
    gt_spike_index,
    raw_data_bin,
    geom,
    loc_radius=100,
    nn_denoise=True,
    enforce_decrease=True,
    tpca=None,
    chunk_size=30_000,
    n_jobs=10,
)
gt_xzptp = np.c_[
    gt_relocalizations[:, 0],
    gt_relocalizations[:, 3],
    gt_remaxptp,
]

# %%
gtx, gtz, gtptp = gt_xzptp.T

# %%
cluster_viz_index.array_scatter(gt_spike_train[:,1], geom, gtx, gtz, gtptp, do_log=False);
plt.gcf().suptitle("New denoised localizations + ptps. GT amp scaling std=0.1", y=1.02)
plt.gcf().axes[1].set_xlim([0, 35])

# %%
fig, axes = plt.subplots(8, 5, figsize=(32, 32), sharex=True, sharey=True, gridspec_kw=dict(hspace=0, wspace=0))

for ti in range(40):
    tt = gt_templates[ti]
    tmc = tt.ptp(0).argmax()
    
    tttmc = tt[:, tmc]
    # resamp = resample(tttmc, 8 * 121)
    # versions = resamp.reshape(8, 121, order="F")
    axes.flat[ti].plot(tt[:, tmc])
    axes.flat[ti].plot(dn(torch.tensor(tt[:, tmc][None].astype(np.float32)))[0].detach().numpy())
    # lb = f"argmin: {tt[:, tmc].argmin()}, argmax: {tt[:, tmc].argmax()}, abs argmax: {np.abs(tt[:, tmc]).argmax()}"
    # plt.title(lb)
    # axes.flat[i].plot(versions.T)
fig.suptitle("Denoised versions of max channels of templates used to create hybrid data", y=0.9)
plt.show()

# %%
gtas = []
std = []
stdlog = []
stdlog1p = []
stdx = []
stdz = []
for u in np.unique(gt_spike_train[:,1]):
    in_unit = np.flatnonzero(gt_spike_train[:,1] == u)
    gtas.append(gt_templates[u].ptp(0).max())
    std.append(gtptp[in_unit].std())
    stdlog.append(np.log(gtptp[in_unit]).std())
    stdlog1p.append(np.log(1 + gtptp[in_unit]).std())
    # stdx.append(gtx[in_unit].std())
    # stdz.append(gtz[in_unit].std())
    stdx.append(np.median(np.abs(gtx[in_unit] - np.median(gtx[in_unit]))) / 0.675)
    stdz.append(np.median(np.abs(gtz[in_unit] - np.median(gtz[in_unit]))) / 0.675)
gtas = np.array(gtas)
sort = np.argsort(gtas)
gtas = gtas[sort]
std = np.array(std)[sort]
stdlog = np.array(stdlog)[sort]
stdlog1p = np.array(stdlog1p)[sort]
stdx = np.array(stdx)[sort]
stdz = np.array(stdz)[sort]

# %%
1

# %%
np.save("/mnt/3TB/charlie/template_amplitudes_sorted.npy", gtas)
np.save("/mnt/3TB/charlie/x_mad_scaled.npy", stdx)
np.save("/mnt/3TB/charlie/z_mad_scaled.npy", stdz)

# %%
plt.plot(gtas, stdx, label="x")
plt.plot(gtas, stdz, label="z")
plt.legend()
plt.ylabel("feature MAD/0.675")
plt.xlabel("template amplitude")

# %%
# plt.plot(std)
allthestds = []
for c in [0, 1, 2, 5, 10]:
    thestds = []
    for u in sort:
        in_unit = np.flatnonzero(gt_spike_train[:,1] == u)
        thestds.append(np.log(c + gtptp[in_unit]).std())
    plt.axhline(np.array(thestds).mean())
    plt.plot(gtas, thestds, label=f"log({c}+ptp)")
    allthestds.append(np.array(thestds))
plt.legend()
plt.title("stddev of log(c+ptp) as a function of template ptp")
plt.ylabel("feature std")
plt.xlabel("template amplitude")

# %%
# plt.plot(std)
allthestds = []
for c in [0, 1, 2, 5, 10]:
    thestds = []
    for u in sort:
        in_unit = np.flatnonzero(gt_spike_train[:,1] == u)
        thestds.append(np.log(c + gtptp[in_unit]).std())
    plt.axhline(np.array(thestds).mean())
    plt.plot(gtas, thestds, label=f"log({c}+ptp)")
    allthestds.append(np.array(thestds))
plt.legend()
plt.title("stddev of log(c+ptp) as a function of template ptp")
plt.ylabel("feature std")
plt.xlabel("template amplitude")

# %%
cluster_viz_index.array_scatter(gt_spike_train[:,1], geom, gtx, gtz, gtptp, c=5);
plt.gcf().suptitle("New denoised localizations + ptps. GT amp scaling std=0.1", y=1.02)
# plt.gcf().axes[1].set_xlim([0, 35])

# %%
stdx.mean()

# %%
stdz.mean()

# %%
allthestds[0].mean(), 5 / allthestds[0].mean()

# %%
allthestds[3].mean(), 5 / allthestds[3].mean()

# %%

# %%
# plt.plot(gtas, std, label="usual std")
plt.plot(gtas, stdlog, label="std of log ptp")
plt.plot(gtas, stdlog1p, label="std of log(10+ptp)")
plt.xlabel("template ptp")

# %%
stdx.mean(), stdz.mean(), stdlog.mean(), stdlog1p.mean()

# %%
stdx.mean() / stdlog.mean()

# %%
stdx.mean() / stdlog1p.mean()

# %%
# plt.plot(gtas, std, label="usual std")
plt.plot(gtas, stdx, label="std of x")
plt.plot(gtas, stdz, label="std of z")
plt.xlabel("template ptp")

# %%
labels.shape, x.shape

# %%
cluster_viz_index.array_scatter(labels, geom, x, z, maxptps)

# %%
