# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import h5py
import numpy as np
from spike_psvae import vis_utils, point_source_centering, localization, waveform_utils
import torch
import matplotlib.pyplot as plt
from scipy import linalg
import numpy as np

# %%
plt.rc("figure", dpi=200)
rg = np.random.default_rng(0)

# %%
with h5py.File("../data/wfs_locs_tiny.h5") as f:
    y = f["y"][:]
    good = np.flatnonzero(y >= 1)
    y = y[good]
    wfs = f["denoised_waveforms"][good]
    x = f["x"][good]
    z = f["z_rel"][good]
    z_abs = f["z"][good]
    alpha = f["alpha"][good]
    max_ptp = f["max_ptp"][good]
    maxchans = f["max_channels"][good]

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
# batch = torch.tensor(wfs[15:16])
# bx = torch.tensor(x[15:16])
# by = torch.tensor(y[15:16])
# bz = torch.tensor(z[15:16])
# bmaxchan = torch.LongTensor(maxchans[15:16])
# balpha = torch.tensor(alpha[15:16])

# inds = rg.choice(len(good), size=16, replace=False)
inds = np.arange(16)
batch = torch.tensor(wfs[inds])
bx = torch.tensor(x[inds])
by = torch.tensor(y[inds])
bz = torch.tensor(z[inds])
bmaxchan = torch.LongTensor(maxchans[inds])
balpha = torch.tensor(alpha[inds])
reloc, r, q = point_source_centering.relocate_simple(batch, geom, bmaxchan, bx, by, bz, balpha)

# %%
bx, by, bz, balpha

# %%
q.shape, r.shape

# %%
# plt.plot(r.t())
# plt.show()
# plt.plot(q.t())
# plt.show()
mx = torch.max(batch, dim=1)
mn = torch.min(batch, dim=1)
ptp = mx.values - mn.values
# plt.plot(ptp.t())
fig, axes = plt.subplots(4, 4, figsize=(6, 6), sharex=True, sharey=True)
for qq, pp, rr, ax in zip(q, ptp, r, axes.flat):
#     ax.plot(pp - qq, color="k", label="difference");
    # TODO add locs to title
    ax.plot(pp, color="b", label="observed ptp");
    ax.plot(qq, color="g", label="ptp predicted from localization");
#     ax.plot(rr, color="r", label="standard location ptp");
# TODO separate plot with post-reloc ptp and standard loc ptp
axes.flat[3].legend();
plt.show()

# %%
# TODO lineplots
vis_utils.labeledmosaic([batch, reloc, batch - reloc], ["original", "relocated", "difference"], pad=2, cbar=True)

# %%
fig, (aa, ab, ac) = plt.subplots(3, 1, figsize=(6,6), sharex=True)
aa.plot(bx, ".", ms=5, label="x")
aa.plot(bz, ".", ms=5, label="z")
aa.legend()
ab.plot(by, ".", ms=5, label="y")
ab.legend()
ac.plot(balpha, ".", ms=5, label="alpha")
ac.legend()
plt.show()

# %%

# %%

# %%
batch = torch.tensor(wfs[:])
bx = torch.tensor(x[:])
by = torch.tensor(y[:])
bz = torch.tensor(z[:])
bmaxchan = torch.LongTensor(maxchans[:])
balpha = torch.tensor(alpha[:])
reloc, r, q = point_source_centering.relocate_simple(batch, geom, bmaxchan, bx, by, bz, balpha)

# %%

# %%
vals = linalg.svdvals(batch.numpy().reshape(614, -1))
vals = np.square(vals)
(np.cumsum(vals) / np.sum(vals) < 0.95).sum()

# %%
vals = linalg.svdvals(reloc.numpy().reshape(614, -1))
vals = np.square(vals)
(np.cumsum(vals) / np.sum(vals) < 0.95).sum()

# %%
t = np.load("/Users/charlie/Downloads/spt_yass_templates.npy")
good = np.flatnonzero(t.ptp(1).ptp(1))
t = t[good]

# %%
t.shape

# %%
# xt, yt, zt, alphat = localization.localize_waveforms(t, geom, n_workers=1)


# %%
t[:16].shape

# %%
bt, maxchant = waveform_utils.get_local_waveforms(t[rg.choice(t.shape[0], size=16)], 10)
xt, yt, zt, alphat = localization.localize_waveforms(bt, geom, maxchans=maxchant, n_workers=1)
zt_rel = zt - geom[maxchant, 1]
ptpt = bt.ptp(1)
reloct, rt, qt = point_source_centering.relocate_simple(bt, geom, maxchant, xt, yt, zt_rel, alphat)

# %%
xt, yt, zt, alphat

# %%
vis_utils.labeledmosaic(
    [torch.as_tensor(bt), reloct, torch.as_tensor(bt) - reloct],
    ["original", "relocated", "difference"],
    pad=2, cbar=True)

# %%
fig, axes = plt.subplots(3, 3, figsize=(6, 6), sharex=True, sharey=True)
for qq, pp, rr, ax, x_, y_, z_, alpha_, gg in zip(qt, ptpt, rt, axes.flat, xt, yt, zt_rel, alphat, "abcdefghijklmnopqrstuv"):
#     ax.plot(pp - qq, color="k", label="difference");
    # TODO add locs to title
    ax.plot(pp, color="b", label="observed ptp");
    ax.plot(qq, color="g", label="ptp predicted from localization");
    ax.set_title(f"{gg}: (x,y,z,α)=({x_:.2f},{y_:.2f},{z_:.2f},{alpha_:.2f})", fontsize=6)
#     ax.plot(rr, color="r", label="standard location ptp");
# TODO separate plot with post-reloc ptp and standard loc ptp
axes[0, -1].legend();
plt.show()

# %%
# TODO pca temporal vectors, then their spatial loadings
