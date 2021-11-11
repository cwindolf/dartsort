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
from spike_psvae import vis_utils, point_source_centering
import torch
import matplotlib.pyplot as plt
from scipy import linalg

# %%
plt.rc("figure", dpi=200)

# %%
with h5py.File("../data/wfs_locs.h5") as f:
    y = f["y"][:]
    maxchans = f["max_channels"][:]
    good = np.flatnonzero((y >= 1) & (maxchans > 10))[:10000]
    print(len(good))
    y = y[good]
    maxchans = maxchans[good]
    wfs = f["denoised_waveforms"][good]
    x = f["x"][good]
    z = f["z_rel"][good]
    z_abs = f["z"][good]
    alpha = f["alpha"][good]
    max_ptp = f["max_ptp"][good]


# %%
plt.plot(x, z_abs, "k.", ms=1);

# %%
plt.hist(y, bins=128);
plt.axvline(15, color="r");

# %%
plt.hist(alpha, bins=128);
plt.axvline(175, color="r");

# %%
np.corrcoef(y, alpha)[1, 0]

# %%
geom = np.load("../data/np2_channel_map.npy")
geom.shape

# %%
geom[1, 0] - geom[0, 0]

# %%
geom[2, 1] - geom[0, 1]

# %%
wfs[0].shape

# %%
geom = torch.tensor(geom)

# %%
batch = torch.tensor(wfs[:16])
bx = torch.tensor(x[:16])
by = torch.tensor(y[:16])
bz = torch.tensor(z[:16])
bmaxchan = torch.LongTensor(maxchans[:16])
print(bmaxchan)
balpha = torch.tensor(alpha[:16])
reloc = point_source_centering.relocate_simple(batch, geom, bmaxchan, bx, by, bz, balpha)

# %%
vis_utils.labeledmosaic([batch, reloc, torch.abs(batch - reloc)], ["original", "relocated", "|resid|"], pad=2)

# %%
plt.plot(bx, ".", ms=5, label="x")
plt.plot(by, ".", ms=5, label="y")
plt.plot(bz, ".", ms=5, label="z")
plt.plot(balpha, ".", ms=5, label="alpha")
plt.legend()
plt.show()

# %%
maxchans[0], maxchans[4447], maxchans[4448], maxchans[4449]

# %%
batch = torch.tensor(wfs[:])
bx = torch.tensor(x[:])
by = torch.tensor(y[:])
bz = torch.tensor(z[:])
bmaxchan = torch.LongTensor(maxchans[:])
print(bmaxchan)
balpha = torch.tensor(alpha[:])
reloc = point_source_centering.relocate_simple(batch, geom, bmaxchan, bx, by, bz, balpha)

# %%
vals = linalg.svdvals(batch.numpy().reshape(10000, -1))
(np.cumsum(vals) / np.sum(vals) < 0.95).sum()

# %%
vals = linalg.svdvals(reloc.numpy().reshape(10000, -1))
(np.cumsum(vals) / np.sum(vals) < 0.90).sum()

# %%
np.arange(35).reshape(5, 7).mean(axis=1).shape


# %%
def nf(x):
    x = x.numpy().reshape(10000, -1)
    x -= x.mean(axis=1, keepdims=True)
    eigs = np.square(linalg.svdvals(x))
    return eigs


# %%
eigs_n = nf(batch)

# %%
eigs_y = nf(reloc)

# %%
(np.cumsum(eigs_n) / eigs_n.sum() < 0.95).sum()

# %%
(np.cumsum(eigs_y) / eigs_y.sum() < 0.95).sum()

# %%

# %%
plt.plot(np.log(eigs_n)[:100])
plt.plot(np.log(eigs_y)[:100])

# %%
plt.plot(np.cumsum(eigs_n)[:100] / eigs_n.sum())
plt.plot(np.cumsum(eigs_y)[:100] / eigs_y.sum())

# %%
