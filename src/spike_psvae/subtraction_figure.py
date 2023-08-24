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
from pathlib import Path
from tqdm.auto import tqdm, trange
import pandas as pd
from IPython.display import display, Image
from joblib import Parallel, delayed
import os
import colorcet as cc
import pickle
from matplotlib import colors
import matplotlib
import torch

# %%
from spike_psvae import (
    subtract,
    denoise,
    detect,
    waveform_utils,
)

# %%
plt.rc("figure", dpi=300)
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
# matplotlib.use('pgf')

# %%
figdir = Path("/home/charlie/displayfigs")
figdir.mkdir(exist_ok=True)

# %%
plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "Times",
})
# plt.rc('text.latex', preamble='\\usepackage{stix}\n\\usepackage{newtxtext}')
preamble = r"""
\usepackage{helvet}
\renewcommand\familydefault{\sfdefault}
\usepackage{newtxtext}
\usepackage{newtxmath}
\usepackage{sfmath}
"""
plt.rc('text.latex', preamble=preamble)

# %%
dn = denoise.SingleChanDenoiser().load()
dnd = detect.DenoiserDetect(dn)

# %%
channel_index = waveform_utils.make_contiguous_channel_index(384, 20)

# %%
mmap = np.memmap("/mnt/3TB/charlie/hybrid_5min/hybrid_5min_input/DY_018.ap.bin", dtype=np.float32)
mmap = mmap.reshape(-1, 384)

# %%
snip = mmap[100 * 30_000 : 100 * 30_000 + 1000]
snip_pad = mmap[100 * 30_000 - 42 : 100 * 30_000 + 1000 + 120 - 42]
snip_dnd = dnd.forward_recording(torch.tensor(snip_pad)).detach().numpy()
times, chans, energies = detect.denoiser_detect_dedup(snip_pad, 5, dnd, device="cpu", channel_index=channel_index)

# %%
### PTP DETECTION FIGURE
fig, (aa, ab) = plt.subplots(nrows=2, figsize=(8, 5), sharex=True)

im = aa.imshow(snip.T, cmap=plt.cm.RdGy, vmin=-8, vmax=8)
plt.colorbar(im, ax=aa, shrink=0.8, label="standardized voltage (s.u.)", pad=0.02)
aa.set_ylabel("channel")
aa.set_title("Filtered and standardized recording")

im = ab.imshow(snip_dnd.T, cmap=plt.cm.bone)
plt.colorbar(im, ax=ab, shrink=0.8, label="denoised ptp (s.u.)", pad=0.02)
ab.set_ylabel("channel")
ab.set_xlabel("time (samples)")
ab.set_title("Denoised PTP")

fig.suptitle("Denoised peak-to-peak amplitude detection", y=0.97)

plt.show()

fig.savefig(figdir / "ptp_detection_nopeaks.pdf")
fig.savefig(figdir / "ptp_detection_nopeaks.png", dpi=200)
plt.close(fig)

# %%
### PTP DETECTION FIGURE
fig, (aa, ab) = plt.subplots(nrows=2, figsize=(8, 5), sharex=True)

im = aa.imshow(snip.T, cmap=plt.cm.RdGy, vmin=-8, vmax=8)
plt.colorbar(im, ax=aa, shrink=0.8, label="standardized voltage (s.u.)", pad=0.02)
aa.set_ylabel("channel")
aa.set_title("Filtered and standardized recording with deduplicated denoised-PTP peaks")
sc = aa.scatter(times, chans, marker=".", c=energies, cmap=plt.cm.spring, edgecolor="k", s=8, alpha=0.8)
sc = aa.scatter(times, chans, marker=".", c=energies, cmap=plt.cm.spring, s=4, alpha=0.8)
aa.legend(
    handles=[plt.plot([],color=sc.get_cmap()(sc.norm(c)),ls="", marker=".")[0] for c in [5, 10, 15]],
    labels=["5", "10", "15"],
    title="Denoised PTP",
    framealpha=1,
    fancybox=False,
)

im = ab.imshow(snip_dnd.T, cmap=plt.cm.bone)
plt.colorbar(im, ax=ab, shrink=0.8, label="denoised ptp (s.u.)", pad=0.02)
ab.set_ylabel("channel")
ab.set_xlabel("time (samples)")
ab.set_title("Denoised PTP with deduplicated denoised-PTP peaks")
ab.scatter(times, chans, marker=".", c=energies, cmap=plt.cm.spring, s=5, alpha=0.8)
ab.legend(
    handles=[plt.plot([],color=sc.get_cmap()(sc.norm(c)),ls="", marker=".")[0] for c in [5, 10, 15]],
    labels=["5", "10", "15"],
    title="Denoised PTP",
    framealpha=1,
    fancybox=False,
)

fig.suptitle("Denoised peak-to-peak amplitude detection", y=0.97)

plt.show()

fig.savefig(figdir / "ptp_detection_withpeaks.pdf")
fig.savefig(figdir / "ptp_detection_withpeaks.png", dpi=200)
plt.close(fig)

# %%
raw0 = snip_pad.copy()
wfs0, _, res0, si0 = subtract.detect_and_subtract(
    raw0,
    8,
    None,
    None,
    channel_index,
    channel_index,
    denoiser=dn,
    denoiser_detector=dnd,
    do_enforce_decrease=False,
)
wfs1, _, res1, si1 = subtract.detect_and_subtract(
    res0,
    5,
    None,
    None,
    channel_index,
    channel_index,
    denoiser=dn,
    denoiser_detector=dnd,
    do_enforce_decrease=False,
)

# %%
raw0.shape, res0.shape, res1.shape

# %%
si0.shape, si1.shape

# %%
time_range = np.arange(
    2 * 121 - 42,
    3 * 121 - 42,
)

# %%
sub0 = np.zeros_like(raw0)
time_ix = si0[:, 0, None] + time_range[None, :]
chan_ix = channel_index[si0[:, 1]]
np.add.at(
    sub0, 
    (time_ix[:, :, None], chan_ix[:, None, :]),
    wfs0,
)

# %%
sub1 = np.zeros_like(raw0)
time_ix = si1[:, 0, None] + time_range[None, :]
chan_ix = channel_index[si1[:, 1]]
np.add.at(
    sub1, 
    (time_ix[:, :, None], chan_ix[:, None, :]),
    wfs1,
)

# %%
res1.min(), res1.max()

# %%
norm = colors.CenteredNorm(halfrange=4)

# %%
### SUBTRACTION FIGURE A
# say 2 thresholds? one per column
# each column: raw w/ peaks, subtracted wfs, residual

mos = """\
ad
be
cf
"""
fig, axes = plt.subplot_mosaic(mos, sharex=True, sharey=True, figsize=(7, 6))

im_raw0 = axes["a"].imshow(raw0[160:-460].T, aspect=2/3, cmap=plt.cm.RdGy, vmin=-8, vmax=8)
im_sub0 = axes["b"].imshow(sub0[160:-460].T, aspect=2/3, cmap=plt.cm.RdGy, vmin=-8, vmax=8)
im_res0 = axes["c"].imshow(res0[160:-460].T, aspect=2/3, cmap=plt.cm.RdGy, vmin=-8, vmax=8)

titles = [
    "Original", "Subtracted (threshold=8)", "First residual",
    "First residual", "Subtracted (threshold=5)", "Second residual",
]
for k, t in zip("abcdef", titles):
    axes[k].set_title(t)

for k in "abc":
    axes[k].set_ylabel("channel")
for k in "cf":
    axes[k].set_xlabel("time (samples)")

im_raw1 = axes["d"].imshow(res0[160:-460].T, aspect=2/3, cmap=plt.cm.RdGy, vmin=-8, vmax=8)
im_sub1 = axes["e"].imshow(sub1[160:-460].T, aspect=2/3, cmap=plt.cm.RdGy, vmin=-8, vmax=8)
im_res1 = axes["f"].imshow(res1[160:-460].T, aspect=2/3, cmap=plt.cm.RdGy, vmin=-8, vmax=8)

fig.suptitle("NN-based peeling (`subtraction') overview", y=0.96)

plt.show()

fig.savefig(figdir / "subtraction_overview_a.pdf")
fig.savefig(figdir / "subtraction_overview_a.png", dpi=200)
plt.close(fig)

# %%
im_raw0.norm.halfrange

# %%
