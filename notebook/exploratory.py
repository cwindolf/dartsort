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
from tensorly.decomposition import parafac

# %%
from spike_psvae import waveform_utils, point_source_centering, localization, vis_utils, decomp

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
plt.hist(alpha[y < 0.1], bins=32);
plt.xlabel("alpha")
plt.ylabel("frequency")
plt.title("histogram of alpha when y==0")

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
    good = np.flatnonzero(h5["y"][:] < 1e-8)
    bwf = h5["denoised_waveforms"][good[:16]]
    bmaxchan = h5["max_channels"][good[:16]]
    # balpha = h5["alpha"][good[:16]]
    # bx = h5["x"][good[:16]]
    # by = h5["y"][good[:16]]
    # bz = h5["z_rel"][good[:16]]
    bx, by, bz, _, balpha = localization.localize_waveforms(bwf, geom, bmaxchan)

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

# %%
vis_utils.labeledmosaic(
    [bwf, reloc, bwf - reloc],
    ["original", "relocated", "residual"],
    pad=2,
    separate_norm=True,
    cbar=False
)

# %%
(np.abs(y) < 0.01).mean()

# %%
by.max()

# %%
fig, axes = vis_utils.vis_ptps([bwf.ptp(1), q], ["observed ptp", "predicted ptp"], "bg", subplots_kwargs=dict(sharex=True, figsize=(5, 5)))
plt.show()
fig, axes = vis_utils.vis_ptps([reloc.ptp(1), r], ["relocated ptp", "standard ptp"], "kr", subplots_kwargs=dict(sharex=True, figsize=(5, 5)))
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
local_wfs = waveform_utils.get_local_waveforms(wfs, 10, geom, maxchans, geomkind="standard")
x, y, z_rel, z_abs, alpha = localization.localize_waveforms(wfs, geom, jac=False, geomkind="standard")
reloc, r, q = point_source_centering.relocate_simple(local_wfs, geom, maxchans, x, y, z_rel, alpha, geomkind="standard")
reloc = reloc.numpy(); r = r.numpy(); q = q.numpy()

# %%
vis_utils.labeledmosaic(
    [local_wfs[:16], reloc[:16]], #, local_wfs[:16] - reloc[:16]],
    ["original", "relocated"], #, "residual"],
    pad=2,
)

# %%
fig, axes = vis_utils.vis_ptps([local_wfs.ptp(1)[big[:16]], q[big[:16]]], ["observed ptp", "predicted ptp"], "bg")
plt.show()
fig, axes = vis_utils.vis_ptps([reloc.ptp(1)[big[:16]], r[big[:16]]], ["relocated ptp", "standard ptp"], "kr")
plt.show()

# %%
fig, (aa, ab) = plt.subplots(2, 1, sharex=False)
aa.hist(np.square(local_wfs.ptp(1) - q).mean(axis=1), bins=32);
aa.set_title("||wf - pred||")
ab.hist(np.square(reloc.ptp(1) - r).mean(axis=1), bins=32);
ab.set_title("||reloc - std||")
plt.tight_layout()


# %%
def pca_rank_plot(wfs, ax=None, q=0.95, c="bk", name=None):
    wfs = wfs.reshape(wfs.shape[0], -1)
    wfs = wfs - wfs.mean(axis=1, keepdims=True)
    s = la.svdvals(wfs)
    ax = ax or plt.gca()
    sqs = np.square(s)
    seq = np.cumsum(sqs) / np.sum(sqs)
    rank = np.flatnonzero(seq >= q)[0]
    ax.axhline(q, color="gray", zorder=-1)
    ax.plot(seq, c=c[0], label=name)
    ax.axvline(rank, color=c[1])
    plt.xlim([-5, len(s) + 5])
    return rank


# %%
well_model = np.flatnonzero((np.square(local_wfs.ptp(1) - q).mean(axis=1)) < 2)
well_reloc = np.flatnonzero((np.square(reloc.ptp(1) - r).mean(axis=1)) < 2)

# %%
rank0 = pca_rank_plot(local_wfs, name="original")
rank1 = pca_rank_plot(reloc, c="rk", name="relocated")
plt.xticks(list(range(0, local_wfs.shape[0], 50)) + [rank0, rank1]);
plt.yticks(list(plt.yticks()[0]) + [0.95])
plt.ylim(0.5, 1.0)
plt.title("pca: does relocating help?")
plt.ylabel("proportion of variance captured")
plt.xlabel("number of components")
plt.legend(fancybox=False)

# %%
np.arange(35).reshape(5, 7).mean(axis=0)


# %%
def pca_resid_plot(wfs, ax=None, q=0.95, c="bk", name=None, dup=False):
    wfs = wfs.reshape(wfs.shape[0], -1)
    wfs = wfs - wfs.mean(axis=0, keepdims=True)
    s = la.svdvals(wfs)
    v = np.square(s) / wfs.shape[0]
    ax = ax or plt.gca()
    sqs = np.square(s)
    totvar = np.square(wfs).mean(0).sum()
    print(totvar)
    residvar = np.concatenate(([totvar], totvar - np.cumsum(v)))
    print(np.cumsum(v)[:5])
    print(np.cumsum(v)[-5:])
    if dup:
        ax.plot(np.array([totvar, totvar, totvar, totvar, *residvar[:50]][:50]) / (wfs.shape[1]), marker=".", c=c[0], label=name)
    else:
        ax.plot(np.array(residvar[:50]) / (wfs.shape[1]), marker=".", c=c[0], label=name)


# %%
pca_resid_plot(local_wfs[:, :, 2:-2], name="original")
pca_resid_plot(reloc[:, :, 2:-2], c="rk", name="relocated", dup=True)
plt.title("pca: does relocating help?")
plt.ylabel("residual variance (s.u.)")
plt.xlabel("number of components (really starts at 0 this time)")
plt.legend(fancybox=False)

# %%
pca_resid_plot(local_wfs[:, :, 2:-2], name="original")
pca_resid_plot(reloc[:, :, 2:-2], c="rk", name="relocated", dup=True)
plt.title("pca: does relocating help?")
plt.ylabel("residual variance (s.u.)")
plt.xlabel("number of components (really starts at 0 this time)")
plt.semilogy()
plt.legend(fancybox=False)


# %%
def parafac_rank_plot(wfs, ax=None, q=0.95, c="bk", name=None):
    wfs = wfs - wfs.mean(axis=0, keepdims=True)
    seq = decomp.cumparafac(wfs, 50)
    seq = np.array(seq)
    seq /= np.square(wfs).sum(axis=(1,2)).mean()
    rank = np.flatnonzero(seq >= q)[0]
    ax.axhline(q, color="gray", zorder=-1)
    ax.plot(seq, c=c[0], label=name)
    ax.axvline(rank, color=c[1])
    plt.xlim([-5, len(seq) + 5])
    return rank


# %%
seq0 = decomp.cumparafac(local_wfs - local_wfs.mean(axis=0, keepdims=True), 50)
seq1 = decomp.cumparafac(reloc - reloc.mean(axis=0, keepdims=True), 50)

# %%
seq0 = np.array(seq0); seq1 = np.array(seq1)

# %%
seq0

# %%
n0, n1

# %%
mwfs = local_wfs - local_wfs.mean(axis=0, keepdims=True)
n0 = np.square(mwfs).mean(axis=0).sum()
v0 = (n0 - seq0) / n0
mreloc = reloc - reloc.mean(axis=0, keepdims=True)
n1 = np.square(mreloc).mean(axis=0).sum()
v1 = (n1 - seq1) / n1

# %%
plt.plot(np.array([n0, *seq0]) / (121 * 22), marker=".", color="b")
plt.plot(np.array([n1, n1, n1, n1, n1, *seq1][:50]) / (121 * 22), marker=".", color="r")
plt.title("parafac: does relocating help?")
plt.ylabel("residual variance (s.u.)")
plt.xlabel("number of components (really starts at 0 this time)")

# %%
plt.plot(np.array([n0, *seq0]) / (121 * 22), marker=".", color="b")
plt.plot(np.array([n1, n1, n1, n1, n1, *seq1][:50]) / (121 * 22), marker=".", color="r")
plt.title("parafac: does relocating help?")
plt.semilogy()
plt.ylabel("residual variance (s.u.)")
plt.xlabel("number of components (really starts at 0 this time)")


# %% [markdown]
# ### images of 5 component PCA and PARAFAC reconstructions, with and without relocating, for the same 16 waveforms

# %%
def recon_plot(wfs, k=5, addmean=True, label="original"):
    means = wfs.mean(axis=0, keepdims=True)
    wfs = wfs - means
    cmeans = int(addmean) * means
    ogshape = wfs.shape
    
    inds = np.random.default_rng(2).choice(wfs.shape[0], size=16, replace=False)
    batch = wfs[inds] + cmeans
    
    # k component PCA reconstruction
    U, s, Vh = la.svd(wfs.reshape(wfs.shape[0], -1), full_matrices=False)
    pca = (U[inds, :k] @ np.diag(s[:k]) @ Vh[:k, :]).reshape((16, *ogshape[1:])) + cmeans
    
    # k component Parafac reconstruction
    weights, factors = parafac(wfs, k)
    pfac = np.einsum("n,in,jn,kn->ijk", weights, factors[0][inds], *factors[1:]) + cmeans
    
    vis_utils.labeledmosaic(
        [batch, pca, pfac],
        [label, "pca recon", "parafac recon"],
        pad=2,
    )


# %%
recon_plot(local_wfs[:, :, 2:-2])
plt.suptitle("original spikes, reconstruction with 5 components", fontsize=8)

# %%
recon_plot(local_wfs[:, :, 2:-2], k=1)
plt.suptitle("original spikes, reconstruction with 1 component", fontsize=8)

# %%
recon_plot(reloc[:, :, 2:-2], label="reloc", addmean=False)
plt.suptitle("relocated spikes, reconstruction with 5 components", fontsize=8)

# %%
recon_plot(reloc[:, :, 2:-2], label="reloc", k=1)
plt.suptitle("relocated spikes, reconstruction with 1 component", fontsize=8)

# %%

# %%

# %%

# %% [markdown] tags=[]
# ### does this relocation remove correlations with the localization features?

# %%

# %%

# %%
