# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.3
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
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.stats import zscore, gamma
import seaborn as sns
import time
from tqdm.auto import tqdm, trange
from sklearn.metrics import adjusted_rand_score
from IPython.display import HTML

# %%
from spike_psvae import waveform_utils, localization, point_source_centering, vis_utils, simdata, cluster, featurize

# %%
rg = lambda k=0: np.random.default_rng(k)

# %% [markdown]
# ### plot kit

# %%
sns.set_style("ticks")

# %%
plt.rc("figure", dpi=200)

# %%
# plt.rc("text", usetex=True)
# plt.rc("font", family="TeX Gyre Pagella")

# %%
plt.rc("axes", titlesize=8)

# %%
which2txt = {"orig": "orig.", "yza": "$yz\\alpha$", "xyza": "$xyz\\alpha$"}

# %%
darkpurple = plt.cm.Purples(0.99)
purple = plt.cm.Purples(0.75)
lightpurple = plt.cm.Purples(0.5)
darkgreen = plt.cm.Greens(0.99)
green = plt.cm.Greens(0.75)
lightgreen = plt.cm.Greens(0.5)
darkblue = plt.cm.Blues(0.99)
blue = plt.cm.Blues(0.75)
lightblue = plt.cm.Blues(0.5)

# %% [markdown] tags=[]
# ## data munging

# %%
# %ll ../data/*{nzy,rigid}*

# %%
# np2 data
ctx_h5 = h5py.File("../data/ks_np2_nzy_cortex.h5", "r")
# hc_h5 = h5py.File("../data/ks_np2_nzy_hippocampus.h5", "r")
# th_h5 = h5py.File("../data/ks_np2_nzy_thalamus.h5", "r")

# # np1 data
# np1_h5 = h5py.File("../data/yass_np1_nzy.h5", "r")

# %%
templates = ctx_h5["templates"][:]

# %%
# h5s = {"NP2 Cortex": ctx_h5, "NP2 Hippocampus": hc_h5, "NP2 Thalamus": th_h5, "NP1": np1_h5}

# %%
# fns = {"NP2 Cortex": "ctx", "NP2 Hippocampus": "hc", "NP2 Thalamus": "th", "NP1": "np1"}

# %%
# load geometry
# geom_np1 = np1_h5["geom"][:]
geom_np2 = ctx_h5["geom"][:]

# rigid disp
# p = np.load("../data/np2_p_rigid.npy")

# %%
noise_segment = np.load("../data/ten_np2_seconds.npy")

# %% tags=[]
(
    loc_templates,
    full_shifted_templates,
    loc_shifted_templates,
    full_noised_waveforms,
    full_denoised_waveforms,
    denoised_waveforms,
    cluster_ids,
    maxchans,
    locs,
    pserrs,
) = simdata.simdata(
    "output_h5",
    templates,
    geom_np2,
    noise_segment=noise_segment,
    centers="original",
    n_clusters=20,
    spikes_per_cluster=100,
)

# %%
denoised_waveforms.shape

# %%
wfs_denoised = denoised_waveforms + 0

# %%
from sklearn.decomposition import PCA

# %%
n_, t_, c_ = wfs_denoised.shape
u, s, vh = np.linalg.svd(wfs_denoised)
temporal_wfs = u[:, :, :3].reshape(n_, t_ * 3)

pca_temporal = PCA(3)
pca_temporal.fit(temporal_wfs)

temp_pcs = pca_temporal.transform(temporal_wfs)
pca_temporal_reconstructed = pca_temporal.inverse_transform(temp_pcs)
pca_temporal_reconstructed = pca_temporal_reconstructed.reshape(n_, t_, 3)

wfs_reconstructed = np.zeros(wfs_denoised.shape)
for i in range(wfs_denoised.shape[0]):
    u, s, vh = np.linalg.svd(wfs_denoised[i], False)
    wfs_reconstructed[i] = np.matmul(pca_temporal_reconstructed[i], s[:3, None] * vh[:3])

# %%
ix = rg().choice(denoised_waveforms.shape[0], size=16, replace=False)
# ix = [18, 19, 20, 21]
ptps = denoised_waveforms[ix].ptp(1)
isos = featurize.isotonic_ptp(ptps)
cisos = featurize.isotonic_ptp(ptps, central=True)
vis_utils.vis_ptps(
    [ptps], 
    ["ptps"],
    "k"
)
plt.show()
vis_utils.vis_ptps(
    [ptps, isos, cisos],
    ["ptps", "up-down", "central up-down"],
    "krb"
)

# %%
feats_yza, rec_yza = featurize.featurize(
    denoised_waveforms, maxchans, geom_np2, return_recons=True, k=3, #iso_ptp=True
)
feats_xyza, rec_xyza = featurize.featurize(
    denoised_waveforms, maxchans, geom_np2, relocate_dims="xyza", return_recons=True, k=3, #iso_ptp=True
)
feats_template_yza, st_z_rel = featurize.featurize(
    loc_shifted_templates, maxchans, geom_np2, return_rel=True, k=3
)
feats_template_xyza, _ = featurize.featurize(
    loc_shifted_templates, maxchans, geom_np2, relocate_dims="xyza", return_rel=True, k=3
)

# %%
vis_utils.labeledmosaic(
    [loc_templates[:10], loc_templates[10:]]
)

# %%
idx = rg().choice(len(denoised_waveforms), size=9, replace=False)
vis_utils.labeledmosaic(
    [
        full_shifted_templates[idx].repeat(2, axis=1),
        full_noised_waveforms[idx].repeat(2, axis=1),
        full_denoised_waveforms[idx].repeat(2, axis=1)
    ],
    rowlabels=["shifted templates", "with noise", "denoised"],
    # separate_norm=True, cbar=False
)


# %%
vis_utils.labeledmosaic(
    [loc_shifted_templates[idx], denoised_waveforms[idx]],
    rowlabels=["local shifted template", "local denoised"],
    separate_norm=True, cbar=False
)


# %%
fig, axes = vis_utils.vis_ptps(
    [loc_shifted_templates[idx].ptp(1), denoised_waveforms[idx].ptp(1)],
    labels=["local shifted template", "local denoised"],
    colors=["k", "silver"],
    legloc="lower center",
)
fig.suptitle("PTP of shifted template vs. noised->denoised", y=1.05)
plt.show()

# %%
z_abss = locs[:, 3]
fig, (aa, ab, ac) = plt.subplots(1, 3, figsize=(6, 4), sharey=True)

vis_utils.cluster_scatter(locs[:, 0], z_abss, cluster_ids, ax=aa, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 1], z_abss, cluster_ids, ax=ab, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 4], z_abss, cluster_ids, ax=ac, alpha=0.2)

aa.set_ylabel("z")
aa.set_xlabel("x")
ab.set_xlabel("y")
ac.set_xlabel("alpha")

fig.suptitle("Ground truth spike positions by true cluster")

plt.show()

# %%
fig, ((aa, ab), (ac, ad)) = plt.subplots(2, 2, figsize=(6, 6))

vis_utils.cluster_scatter(locs[:, 0], feats_xyza[:, 0], cluster_ids, ax=aa, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 1], feats_xyza[:, 1], cluster_ids, ax=ab, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 3], feats_xyza[:, 2], cluster_ids, ax=ac, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 4], feats_xyza[:, 3], cluster_ids, ax=ad, alpha=0.2)

aa.set_title("x")
ab.set_title("y")
ac.set_title("z")
ad.set_title("alpha")

fig.suptitle("True v. denoised est. localizations by true cluster")

plt.show()

# %%
vis_utils.cluster_scatter(locs[:, 0], feats_template_yza[:, 0], cluster_ids,alpha=0.2)

# %%
locs[:, 4]

# %%
feats_template_yza[:, 3]

# %%
vis_utils.cluster_scatter(locs[:, 4], feats_template_yza[:, 3], cluster_ids, alpha=0.2)

# %%
np.sqrt(np.square(locs[:, 3] - feats_template_xyza[:, 2]).mean())

# %%
fig, ((aa, ab), (ac, ad)) = plt.subplots(2, 2, figsize=(6, 6))

vis_utils.cluster_scatter(locs[:, 0], feats_template_xyza[:, 0], cluster_ids, ax=aa, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 1], feats_template_xyza[:, 1], cluster_ids, ax=ab, alpha=0.2)
# vis_utils.cluster_scatter(locs[:, 3], feats_template_xyza[:, 2], cluster_ids, ax=ac, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 2], st_z_rel, cluster_ids, ax=ac, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 4], feats_template_xyza[:, 3], cluster_ids, ax=ad, alpha=0.2)

for ax in (aa, ab, ac, ad):
    # ax.plot(ax.get_xlim(), ax.get_ylim(), c="w", lw=3)
    xa, xb = xlim = np.array(ax.get_xlim())
    ya, yb = ylim = np.array(ax.get_ylim())
    ax.plot((xa - 100, xb + 100), (xa - 100, xb + 100), c="k", lw=1)
    ax.set_xlim([min(xa, ya), max(xb, yb)])
    ax.set_ylim([min(xa, ya), max(xb, yb)])    

aa.set_title("x")
ab.set_title("y")
# ac.set_title("z")
ac.set_title("zrel")
ad.set_title("alpha")

fig.suptitle("True localizations vs. est. from shifted template")

plt.show()

# %%
maxptp = loc_shifted_templates.ptp(1).max(1)

# %%
fig, ((aa, ab), (ac, ad)) = plt.subplots(2, 2, figsize=(6, 6))

vis_utils.cluster_scatter(locs[:, 0], feats_template_xyza[:, 0], cluster_ids, c=np.repeat(pserrs, 100), ax=aa, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 1], feats_template_xyza[:, 1], cluster_ids, c=np.repeat(pserrs, 100), ax=ab, alpha=0.2)
# vis_utils.cluster_scatter(locs[:, 3], feats_template_xyza[:, 2], cluster_ids, ax=ac, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 2], st_z_rel, cluster_ids, c=np.repeat(pserrs, 100), ax=ac, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 4], feats_template_xyza[:, 3], cluster_ids, c=np.repeat(pserrs, 100), ax=ad, alpha=0.2)

for ax in (aa, ab, ac, ad):
    # ax.plot(ax.get_xlim(), ax.get_ylim(), c="w", lw=3)
    xa, xb = xlim = np.array(ax.get_xlim())
    ya, yb = ylim = np.array(ax.get_ylim())
    ax.plot((xa - 100, xb + 100), (xa - 100, xb + 100), c="k", lw=1)
    ax.set_xlim([min(xa, ya), max(xb, yb)])
    ax.set_ylim([min(xa, ya), max(xb, yb)])
    
    

aa.set_title("x")
ab.set_title("y")
# ac.set_title("z")
ac.set_title("zrel")
ad.set_title("alpha")

fig.suptitle("True localizations vs. est. from shifted template by true cluster")

plt.show()

# %%
fig, ((aa, ab), (ac, ad)) = plt.subplots(2, 2, figsize=(6, 6))

vis_utils.cluster_scatter(locs[:, 0], feats_xyza[:, 0], cluster_ids, ax=aa, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 1], feats_xyza[:, 1], cluster_ids, ax=ab, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 3], feats_xyza[:, 2], cluster_ids, ax=ac, alpha=0.2)
# vis_utils.cluster_scatter(locs[:, 2], st_z_rel, cluster_ids, ax=ac, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 4], feats_xyza[:, 3], cluster_ids, ax=ad, alpha=0.2)

for ax in (aa, ab, ac, ad):
    # ax.plot(ax.get_xlim(), ax.get_ylim(), c="w", lw=3)
    xa, xb = xlim = np.array(ax.get_xlim())
    ya, yb = ylim = np.array(ax.get_ylim())
    ax.plot((xa - 100, xb + 100), (xa - 100, xb + 100), c="k", lw=1)
    ax.set_xlim([min(xa, ya), max(xb, yb)])
    ax.set_ylim([min(xa, ya), max(xb, yb)])    

aa.set_title("x")
ab.set_title("y")
ac.set_title("z")
# ac.set_title("zrzel")
ad.set_title("alpha")

fig.suptitle("GT localizations vs. est. after noise+denoise")

plt.show()

# %%
vis_utils.cluster_scatter(locs[:, 2], st_z_rel, cluster_ids, alpha=0.2)

# %%
fig, ((aa, ab, ac), (ad, ae, af)) = plt.subplots(2, 3, figsize=(6, 6), sharey=True)

vis_utils.cluster_scatter(feats_yza[:, 0], feats_yza[:, 2], cluster_ids, ax=aa)
vis_utils.cluster_scatter(feats_yza[:, 1], feats_yza[:, 2], cluster_ids, ax=ab)
vis_utils.cluster_scatter(feats_yza[:, 3], feats_yza[:, 2], cluster_ids, ax=ac)

vis_utils.cluster_scatter(feats_yza[:, 4], feats_yza[:, 2], cluster_ids, ax=ad)
vis_utils.cluster_scatter(feats_yza[:, 5], feats_yza[:, 2], cluster_ids, ax=ae)
vis_utils.cluster_scatter(feats_yza[:, 6], feats_yza[:, 2], cluster_ids, ax=af)

aa.set_ylabel("z")
aa.set_xlabel("x")
ab.set_xlabel("y")
ac.set_xlabel("alpha")
ae.set_ylabel("pc1")
ad.set_xlabel("pc2")
af.set_xlabel("pc3")

fig.suptitle("Spike featurization (yza reloc.) by true cluster", y=0.95)

plt.show()

# %%
fig, ((aa, ab, ac), (ad, ae, af)) = plt.subplots(2, 3, figsize=(6, 6), sharey=True)

vis_utils.cluster_scatter(feats_xyza[:, 0], feats_xyza[:, 2], cluster_ids, ax=aa)
vis_utils.cluster_scatter(feats_xyza[:, 1], feats_xyza[:, 2], cluster_ids, ax=ab)
vis_utils.cluster_scatter(feats_xyza[:, 3], feats_xyza[:, 2], cluster_ids, ax=ac)

vis_utils.cluster_scatter(feats_xyza[:, 4], feats_xyza[:, 2], cluster_ids, ax=ad)
vis_utils.cluster_scatter(feats_xyza[:, 5], feats_xyza[:, 2], cluster_ids, ax=ae)
vis_utils.cluster_scatter(feats_xyza[:, 6], feats_xyza[:, 2], cluster_ids, ax=af)

aa.set_ylabel("z")
aa.set_xlabel("x")
ab.set_xlabel("y")
ac.set_xlabel("alpha")
ae.set_ylabel("pc1")
ad.set_xlabel("pc2")
af.set_xlabel("pc3")

fig.suptitle("Spike featurization (xyza reloc.) by true cluster", y=0.95)

plt.show()

# %%
fig, ((aa, ab, ac), (ad, ae, af)) = plt.subplots(2, 3, figsize=(6, 6), sharey=True, sharex="col")

vis_utils.cluster_scatter(feats_xyza[:, 0], feats_xyza[:, 2], cluster_ids, ax=aa)
vis_utils.cluster_scatter(feats_xyza[:, 1], feats_xyza[:, 2], cluster_ids, ax=ab)
vis_utils.cluster_scatter(feats_xyza[:, 3], feats_xyza[:, 2], cluster_ids, ax=ac)

vis_utils.cluster_scatter(locs[:, 0], z_abss, cluster_ids, ax=ad, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 1], z_abss, cluster_ids, ax=ae, alpha=0.2)
vis_utils.cluster_scatter(locs[:, 4], z_abss, cluster_ids, ax=af, alpha=0.2)

aa.set_xlim([-100, 100])
ab.set_xlim([-10, 100])


ad.set_ylabel("GT z")
ad.set_xlabel("GT x")
ae.set_xlabel("GT y")
af.set_xlabel("GT alpha")

aa.set_ylabel("est z")
aa.set_xlabel("est x")
ab.set_xlabel("est y")
ac.set_xlabel("est alpha")

fig.suptitle("True and est. locs by true cluster", y=0.95)

plt.show()

# %%
scales_yza = cluster.dim_scales_lsq(full_denoised_waveforms, feats_yza)
scales_xyza = cluster.dim_scales_lsq(full_denoised_waveforms, feats_xyza)

# %%
[float(f"{x:0.3f}") for x in scales_yza]

# %%
[float(f"{x:0.3f}") for x in scales_xyza]

# %%
bins = np.arange(300)
plt.hist(cluster.pairdists(full_shifted_templates, square=False), bins=bins, label="Shifted templates", histtype="step");
plt.hist(cluster.pairdists(full_noised_waveforms, square=False), bins=bins, label="Noised", histtype="step");
plt.hist(cluster.pairdists(full_denoised_waveforms, square=False), bins=bins, label="Denoised", histtype="step");
plt.hist(cluster.pdist(feats_yza * scales_yza), bins=bins, label="scaled yza Features", histtype="step");
plt.hist(cluster.pdist(feats_xyza * scales_xyza), bins=bins, label="scaled xyza Features", histtype="step");
plt.xlabel("distance between pairs")
plt.ylabel("frequency")
# plt.title("nelder mead")
plt.legend();

# %%
fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(4, 5))

im = axes[0, 0].imshow(cluster.pairdists(full_shifted_templates))
axes[0, 0].set_title("Shifted templates")
fig.colorbar(im, ax=axes[0, 0], shrink=0.6)
im = axes[0, 1].imshow(cluster.pairdists(full_denoised_waveforms))
fig.colorbar(im, ax=axes[0, 1], shrink=0.6)
axes[0, 1].set_title("Denoised waveforms")

im = axes[1, 0].imshow(cluster.squareform(cluster.pdist(feats_yza)))
fig.colorbar(im, ax=axes[1, 0], shrink=0.6)
axes[1, 0].set_title("yza features")

im = axes[1, 1].imshow(cluster.squareform(cluster.pdist(feats_yza * scales_yza)))
fig.colorbar(im, ax=axes[1, 1], shrink=0.6)
axes[1, 1].set_title("scaled yza features")

im = axes[2, 0].imshow(cluster.squareform(cluster.pdist(feats_xyza)))
fig.colorbar(im, ax=axes[2, 0], shrink=0.6)
axes[2, 0].set_title("xyza features")

im = axes[2, 1].imshow(cluster.squareform(cluster.pdist(feats_xyza * scales_xyza)))
fig.colorbar(im, ax=axes[2, 1], shrink=0.6)
axes[2, 1].set_title("scaled xyza features")

plt.tight_layout(pad=0.1)
plt.show()

# %%
fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(4, 5))

im = axes[0, 0].imshow(cluster.pairdists(full_shifted_templates, log=True))
axes[0, 0].set_title("Shifted templates")
fig.colorbar(im, ax=axes[0, 0], shrink=0.6)
im = axes[0, 1].imshow(cluster.pairdists(full_denoised_waveforms, log=True))
fig.colorbar(im, ax=axes[0, 1], shrink=0.6)
axes[0, 1].set_title("Denoised waveforms")

im = axes[1, 0].imshow(cluster.squareform(np.log(cluster.pdist(feats_yza))))
fig.colorbar(im, ax=axes[1, 0], shrink=0.6)
axes[1, 0].set_title("yza features")

im = axes[1, 1].imshow(cluster.squareform(np.log(cluster.pdist(feats_yza * scales_yza))))
fig.colorbar(im, ax=axes[1, 1], shrink=0.6)
axes[1, 1].set_title("scaled yza features")

im = axes[2, 0].imshow(cluster.squareform(np.log(cluster.pdist(feats_xyza))))
fig.colorbar(im, ax=axes[2, 0], shrink=0.6)
axes[2, 0].set_title("xyza features")

im = axes[2, 1].imshow(cluster.squareform(np.log(cluster.pdist(feats_xyza * scales_xyza))))
fig.colorbar(im, ax=axes[2, 1], shrink=0.6)
axes[2, 1].set_title("scaled xyza features")

plt.tight_layout(pad=0.1)
plt.show()

# %%
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(4, 3.5))

im = axes[0, 0].imshow(cluster.pairdists(full_shifted_templates))
axes[0, 0].set_title("Shifted templates")
fig.colorbar(im, ax=axes[0, 0], shrink=0.6)
im = axes[0, 1].imshow(cluster.pairdists(full_noised_waveforms))
fig.colorbar(im, ax=axes[0, 1], shrink=0.6)
axes[0, 1].set_title("Noised")

im = axes[1, 0].imshow(cluster.pairdists(full_denoised_waveforms))
axes[1, 0].set_title("Denoised")
fig.colorbar(im, ax=axes[1, 0], shrink=0.6)
im = axes[1, 1].imshow(cluster.pairdists(denoised_waveforms))
fig.colorbar(im, ax=axes[1, 1], shrink=0.6)
axes[1, 1].set_title("Denoised maxchan neighborhood")

# im = axes[1, 0].imshow(cluster.pairdists(rec_yza))
# axes[1, 0].set_title("yza reconstructed local denoised")
# fig.colorbar(im, ax=axes[1, 0], shrink=0.6)
# im = axes[1, 1].imshow(cluster.pairdists(rec_xyza))
# fig.colorbar(im, ax=axes[1, 1], shrink=0.6)
# axes[1, 1].set_title("xyza reconstructed local denoised")

plt.tight_layout(pad=0.1)
plt.show()

# %%
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(4, 3.5))

im = axes[0, 0].imshow(cluster.pairdists(full_shifted_templates, log=True))
axes[0, 0].set_title("Shifted templates")
fig.colorbar(im, ax=axes[0, 0], shrink=0.6)
im = axes[0, 1].imshow(cluster.pairdists(full_noised_waveforms, log=True))
fig.colorbar(im, ax=axes[0, 1], shrink=0.6)
axes[0, 1].set_title("Noised")

im = axes[1, 0].imshow(cluster.pairdists(full_denoised_waveforms, log=True))
axes[1, 0].set_title("Denoised")
fig.colorbar(im, ax=axes[1, 0], shrink=0.6)
im = axes[1, 1].imshow(cluster.pairdists(denoised_waveforms, log=True))
fig.colorbar(im, ax=axes[1, 1], shrink=0.6)
axes[1, 1].set_title("Denoised maxchan neighborhood")

# im = axes[1, 0].imshow(cluster.pairdists(rec_yza))
# axes[1, 0].set_title("yza reconstructed local denoised")
# fig.colorbar(im, ax=axes[1, 0], shrink=0.6)
# im = axes[1, 1].imshow(cluster.pairdists(rec_xyza))
# fig.colorbar(im, ax=axes[1, 1], shrink=0.6)
# axes[1, 1].set_title("xyza reconstructed local denoised")

plt.suptitle("Log distances")
plt.tight_layout(pad=0.1)
plt.show()

# %%
