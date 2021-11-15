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

# %%

# %%

# %%
