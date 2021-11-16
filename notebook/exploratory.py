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
import matplotlib.pyplot as plt
import scipy.linalg as la
import h5py

# %%
from spike_psvae import waveform_utils, point_source_centering, localization, vis_utils

# %%
plt.rc("figure", dpi=200)

# %%
h5_path = "../data/wfs_locs_b.h5"

# %%
with h5py.File(h5_path, "r+") as h5:
    print(", ".join(h5.keys()))
    alpha = h5["alpha"][:]
    x = h5["x"][:]
    y = h5["y"][:]
    z = h5["z"][:]
    maxchan = h5["max_channels"][:]
    
    if "geom" not in h5:
        h5.create_dataset("geom", data=np.load("../data/np2_channel_map.npy"))
    geom = h5["geom"][:]
    
    if "z_rel" not in h5:
        h5.create_dataset("z_rel", data=waveform_utils.relativize_z(z, maxchan, geom))
    z_rel = h5["z_rel"][:]

# %%

# %% [markdown] tags=[]
# ### are the localizations the same?

# %%
with h5py.File(h5_path, "r") as h5:
    x_, y_, z_, alpha_ = localization.localize_waveforms_batched(
        h5["denoised_waveforms"],
        geom,
        maxchans=maxchan,
        channel_radius=10,
        n_workers=8,
        jac=False,
        batch_size=512,
    )

# %%
z_rel_ = waveform_utils.relativize_z(z_, maxchan, geom)

# %%
plt.plot(x, z, "k.", ms=1)
plt.plot(x_, z_, "g.", ms=1)
plt.show()

# %% [markdown]
# ### data transformations -- can we make y, alpha more gaussian for the NN?
#
# Wilson-Hilferty: cube root of gamma looks more normal?

# %%
plt.hist(y_, bins=128)
plt.show()
plt.hist(np.log1p(y_), bins=128)
plt.show()

# %%
plt.hist(alpha, bins=128)
plt.show()
plt.hist(np.log(alpha), bins=128)
plt.show()

# %%

# %% [markdown]
# ### ptps and waveforms before and after recentering

# %%
# get a batch
with h5py.File(h5_path, "r+") as h5:
    good = np.flatnonzero(h5["y"][:] > 0.1)
    bwf = h5["denoised_waveforms"][good[:16]]
    balpha = h5["alpha"][good[:16]]
    bx = h5["x"][good[:16]]
    by = h5["y"][good[:16]]
    bz = h5["z_rel"][good[:16]]
    bmaxchan = h5["max_channels"][good[:16]]

# %%
reloc, r, q = point_source_centering.relocate_simple(bwf, geom, bmaxchan, bx, by, bz, balpha)
reloc = reloc.numpy()
r = r.numpy()
q = q.numpy()

# %%
bx[2], by[2], bz[2], balpha[2]

# %%
reloc.min(), reloc.max()

# %%
vis_utils.labeledmosaic(
    [bwf, reloc, bwf - reloc],
    ["original", "relocated", "residual"],
    pad=2,
)

# %%
(np.abs(y) < 0.01).mean()

# %%
fig, axes = vis_utils.vis_ptps([bwf.ptp(1), q], ["observed ptp", "predicted ptp"], "bg")
plt.show()
fig, axes = vis_utils.vis_ptps([reloc.ptp(1), r], ["relocated ptp", "standard ptp"], "kr")
plt.show()

# %%
bix = good[:16]
reloc_, r_, q_ = point_source_centering.relocate_simple(bwf, geom, bmaxchan, x_[bix], y_[bix], z_rel_[bix], alpha_[bix])
reloc_ = reloc_.numpy()
r_ = r_.numpy()
q_ = q_.numpy()

# %%
fig, axes = vis_utils.vis_ptps([bwf.ptp(1), q_], ["observed ptp", "predicted ptp"], "bg")
plt.show()
fig, axes = vis_utils.vis_ptps([reloc_.ptp(1), r_], ["relocated ptp", "standard ptp"], "kr")
plt.show()

# %% [markdown]
# ### does recentring help PCA?

# %% [markdown]
# let's try on templates first

# %%
wfs = np.load("../data/spt_yass_templates.npy")
maxchans = wfs.ptp(1).argmax(1)
local_wfs = waveform_utils.get_local_waveforms(wfs, 10, geom, maxchans)
x, y, z_rel, z_abs, alpha = localization.localize_waveforms(wfs, geom, jac=False)
reloc, r, q = point_source_centering.relocate_simple(local_wfs, geom, maxchans, x, y, z_rel, alpha)
reloc = reloc.numpy(); r = r.numpy(); q = q.numpy()

# %%
fig, axes = vis_utils.vis_ptps([local_wfs.ptp(1)[:16], q[:16]], ["observed ptp", "predicted ptp"], "bg")
plt.show()
fig, axes = vis_utils.vis_ptps([reloc.ptp(1)[:16], r[:16]], ["relocated ptp", "standard ptp"], "kr")
plt.show()

# %%
fig, (aa, ab) = plt.subplots(2, 1, sharex=False)
aa.hist(np.square(local_wfs.ptp(1) - q).mean(axis=1), bins=32);
aa.set_title("||wf - pred||")
ab.hist(np.square(reloc.ptp(1) - r).mean(axis=1), bins=32);
ab.set_title("||reloc - std||")
plt.tight_layout()


# %%
def pca_rank_plot(wfs, ax=None, q=0.95, c="bk"):
    s = la.svdvals(wfs.reshape(wfs.shape[0], -1))
    ax = ax or plt.gca()
    sqs = np.square(s)
    seq = np.cumsum(sqs) / np.sum(sqs)
    rank = np.flatnonzero(seq >= q)[0]
    ax.plot(seq, c=c[0])
    ax.axvline(rank, color=c[1])
    plt.xticks(list(range(0, len(s), 50))  + [rank])
    plt.xlim([-5, len(s) + 5])


# %%
pca_rank_plot(local_wfs)
pca_rank_plot(reloc, c="rk")

# %%
