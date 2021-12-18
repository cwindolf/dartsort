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
from scipy.stats import zscore
import seaborn as sns
import time
from tqdm.auto import tqdm, trange
from sklearn.decomposition import IncrementalPCA

# %%
from npx import lib, reg, cuts

# %%
from isosplit import isosplit

# %%
from isosplit5 import isosplit5

# %%
from spike_psvae import waveform_utils, localization, point_source_centering, vis_utils, statistics

# %%
sns.set_style("ticks")

# %%
rg = lambda k=0: np.random.default_rng(k)

# %%
plt.rc("figure", dpi=200)

# %% [markdown]
# ## parameters

# %%
# single channel denoised data
original_h5 = "../data/wfs_locs_b.h5"

# %%
# we will write some waveforms data here
# this one will just have features, cluster ids, lightweight stuff
# that can be rsyncd to local for datoviz
feats_h5 = "../data/story_feats_b.h5"


# %% [markdown]
# ## helpers

# %%
class timer:
    def __init__(self, name="timer"):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.t = time.time() - self.start
        print(self.name, "took", self.t, "s")


# %%

# %%

# %% [markdown]
# ## initial processing: localization, registration

# %%
with h5py.File(original_h5, "r") as orig_f:
    ckw = dict(geomkind="updown")
    if "first_channels" in orig_f:
        ckw = dict(geomkind="firstchan", firstchans=orig_f["first_channels"][:])
        
    N, T, C = orig_f["denoised_waveforms"].shape
    
    xs, ys, z_rels, z_abss, alphas = localization.localize_waveforms_batched(
        orig_f["denoised_waveforms"],
        orig_f["geom"][:],
        maxchans=orig_f["max_channels"][:],
        channel_radius=10,
        n_workers=8,
        jac=True,
        batch_size=2048,
        **ckw,
    )

# %%
with h5py.File(original_h5, "r") as orig_f:
    times = orig_f["spike_index"][:, 0].astype(float) / 30_000
    with timer("maxptp"):
        maxptp = orig_f["denoised_waveforms"][:].ptp(1).ptp(1)

# %%
help(np.array)

# %% tags=[]
with timer("registration"):
    z_reg, dispmap = reg.register_nonrigid(
        maxptp,
        z_abss,
        times,
        robust_sigma=1,
        rigid_disp=200,
        disp=100,
        denoise_sigma=0.1,
        destripe=False,
        n_windows=[5, 30, 60],
        n_iter=1,
        widthmul=0.25,
    )

# %%
with h5py.File(original_h5, "r") as orig_f:
    geom = orig_f["geom"][:]

# %%
plt.figure(figsize=(1, 10))
plt.plot(geom[:, 0], geom[:, 1], "k.")

# %%
R0, _, _ = lib.faster(maxptp, z_abss, times)
Rreg, _, _ = lib.faster(maxptp, z_reg, times)

# %%
fig, (aa, ab) = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
cuts.plot(R0, ax=aa, aspect=0.66)
cuts.plot(Rreg[100:], ax=ab, aspect=0.66)
aa.set_ylabel("depth")
aa.set_xlabel("time", labelpad=-12)
ab.set_xlabel("time", labelpad=-12)
aa.set_title("unregistered")
ab.set_title("registered")
fig.suptitle("raster of localizations after single chan denoising")
plt.tight_layout(pad=1)
plt.show()

# %%
z_disp = z_reg - (z_abss - z_abss.min())
fig, ax = plt.subplots(figsize=(5, 4))
a = ax.scatter(times, z_reg, c=z_disp, s=0.5, alpha=0.5)
cbar = fig.colorbar(a, ax=ax)
cbar.solids.set_alpha(1)
plt.title("Spikes colored by displacement")
plt.ylabel("registered depth")
plt.xlabel("time")
plt.show()

# %%
z_disp = z_reg - (z_abss - z_abss.min())
fig, ax = plt.subplots()
a = ax.scatter(times, z_reg, c=np.abs(z_disp), s=0.5, alpha=0.5)
cbar = fig.colorbar(a, ax=ax)
cbar.solids.set_alpha(1)
plt.title("Spikes colored by absolute displacement")
plt.ylabel("registered depth")
plt.xlabel("time")
plt.show()

# %%
# let's pause to save these somewhere
with h5py.File(feats_h5, "w") as feats_f:
    for k, v in zip(
        ["x", "y", "z_rel", "z_abs", "z_reg", "z_disp", "alpha", "maxptp", "times"],
        [xs, ys, z_rels, z_abss, z_reg, z_disp, alphas, maxptp, times],
    ):
        feats_f.create_dataset(k, data=v)

# %% [markdown]
# ## relocation

# %%
# 9 random spike indices
example_inds = rg(1).choice(N, 9)
example_inds.sort()

# %%
with h5py.File(original_h5, "r") as orig_f:
    exwf = orig_f["denoised_waveforms"][example_inds]
    geom = orig_f["geom"][:]
    exmc = orig_f["max_channels"][example_inds]

# %%
ex_x = xs[example_inds]
ex_y = ys[example_inds]
ex_zr = z_rels[example_inds]
ex_a = alphas[example_inds]

# %%
vis_utils.labeledmosaic(exwf.reshape(3, 3, T, C)[:, :, 15:, :], [1, 2, 3], pad=1);

# %%
exp = exwf.ptp(1)
fig, axes = vis_utils.vis_ptps(
    [exp], ["peak to peak"], "b"
)

# %%
exwf_yza, ex_q_hat_yza, ex_p_hat = point_source_centering.relocate_simple(
    exwf, geom, exmc, ex_x, ex_y, ex_zr, ex_a,
    relocate_dims="yza",
)
exwf_xyza, ex_q_hat_xyza, ex_p_hat_ = point_source_centering.relocate_simple(
    exwf, geom, exmc, ex_x, ex_y, ex_zr, ex_a,
    relocate_dims="xyza",
)
assert (ex_p_hat == ex_p_hat_).all()

# %%
ex_q_yza = exwf_yza.numpy().ptp(1)
ex_q_xyza = exwf_xyza.numpy().ptp(1)

# %%
fig, axes = vis_utils.vis_ptps(
    [exp, ex_p_hat.numpy()], ["observed ptp", "predicted ptp"], "bg"
)
plt.tight_layout(pad=0.25)
plt.show()
fig, axes = vis_utils.vis_ptps(
    [ex_q_yza, ex_q_hat_yza.numpy()], ["yza relocated ptp", "yza standard ptp"], "kr"
)
plt.tight_layout(pad=0.25)
plt.show()
fig, axes = vis_utils.vis_ptps(
    [ex_q_xyza, ex_q_hat_xyza.numpy()], ["xyza relocated ptp", "xyza standard ptp"], "kr"
)
plt.tight_layout(pad=0.25)
plt.show()

# %%
vis_utils.labeledmosaic(
    [exwf[:, 15:, :], exwf_yza[:, 15:, :], exwf_xyza[:, 15:, :]],
    ["original", "yza reloc", "xyza reloc"],
    pad=1,
)


# %%

# %% [markdown]
# ## PCA

# %% [markdown]
# ### test: PCA error on a batch of data

# %%
def pca_resid_plot(wfs, ax=None, c="b", name=None, pad=0, K=25):
    wfs = wfs.reshape(wfs.shape[0], -1)
    wfs = wfs - wfs.mean(axis=0, keepdims=True)
    v = np.square(la.svdvals(wfs)[:K - pad]) / np.prod(wfs.shape)
    ax = ax or plt.gca()
    totvar = np.square(wfs).mean()
    residvar = np.concatenate(([totvar], totvar - np.cumsum(v)))
    if pad:
        ax.plot(([totvar] * pad + [*residvar]), marker=".", c=c, label=name)
    else:
        ax.plot(residvar[:50], marker=".", c=c, label=name)


# %%
B = 50_000
with h5py.File(original_h5, "r") as orig_f:
    batch_wfs = orig_f["denoised_waveforms"][:B]
    batch_mc = orig_f["max_channels"][:B]
    
    batch_wfs_yza, q_hat_yza, p_hat = point_source_centering.relocate_simple(
        batch_wfs, geom, batch_mc, xs[:B], ys[:B], z_rels[:B], alphas[:B],
        relocate_dims="yza",
    )
    batch_wfs_xyza, q_hat_xyza, p_hat_ = point_source_centering.relocate_simple(
        batch_wfs, geom, batch_mc, xs[:B], ys[:B], z_rels[:B], alphas[:B],
        relocate_dims="xyza",
    )

# %%
fig = plt.figure()
pca_resid_plot(batch_wfs, name="No relocation", c="k")
pca_resid_plot(batch_wfs_yza, name="YZa relocated", c="b", pad=3)
pca_resid_plot(batch_wfs_xyza, name="XYZA relocated", c="g", pad=4)
plt.semilogy()
yt = [0.5, 0.1, 0.01]
plt.yticks(yt, list(map(str, yt)))
plt.legend(fancybox=False)
plt.ylabel("PCA remaining variance (s.u.)")
plt.xlabel("number of factors")
plt.title("Does relocation help PCA?")
plt.show()

# %%

# %% [markdown]
# ### fit PCAs on the whole data

# %%
K = 5

# %%
ipca_orig = IncrementalPCA(n_components=K)
ipca_yza = IncrementalPCA(n_components=K)
ipca_xyza = IncrementalPCA(n_components=K)

with h5py.File(original_h5, "r") as orig_f:
    waveforms = orig_f["denoised_waveforms"]
    maxchans = orig_f["max_channels"][:]
    
    ckw = dict(geomkind="updown")
    if "first_channels" in orig_f:
        ckw = dict(geomkind="firstchan", firstchans=orig_f["first_channels"][:])
    
    for b in trange((N + 1) // 2048, desc="fit"):
        start = b * 2048
        end = min(N, (b + 1) * 2048)

        wfs_orig = waveforms[start:end]
        B, _, _ = wfs_orig.shape
        
        wfs_yza, _, _ = point_source_centering.relocate_simple(
            wfs_orig,
            geom,
            maxchans[start:end],
            xs[start:end],
            ys[start:end],
            z_rels[start:end],
            alphas[start:end],
            channel_radius=10,
            relocate_dims="yza",
            **ckw,
        )
        wfs_xyza, _, _ = point_source_centering.relocate_simple(
            wfs_orig,
            geom,
            maxchans[start:end],
            xs[start:end],
            ys[start:end],
            z_rels[start:end],
            alphas[start:end],
            channel_radius=10,
            relocate_dims="xyza",
            **ckw,
        )
        ipca_orig.partial_fit(wfs_orig.reshape(B, -1))
        ipca_yza.partial_fit(wfs_yza.reshape(B, -1))
        ipca_xyza.partial_fit(wfs_xyza.reshape(B, -1))        

# %%
loadings_orig = np.empty((N, K))
loadings_yza = np.empty((N, K))
loadings_xyza = np.empty((N, K))

with h5py.File(original_h5, "r") as orig_f:
    waveforms = orig_f["denoised_waveforms"]
    maxchans = orig_f["max_channels"][:]
    
    ckw = dict(geomkind="updown")
    if "first_channels" in orig_f:
        ckw = dict(geomkind="firstchan", firstchans=orig_f["first_channels"][:])

    for b in trange((N + 1) // 2048, desc="project"):
        start = b * 2048
        end = min(N, (b + 1) * 2048)

        wfs_orig = waveforms[start:end]
        B, _, _ = wfs_orig.shape
        wfs_yza, _, _ = point_source_centering.relocate_simple(
            wfs_orig,
            geom,
            maxchans[start:end],
            xs[start:end],
            ys[start:end],
            z_rels[start:end],
            alphas[start:end],
            channel_radius=10,
            relocate_dims="yza",
            **ckw,
        )
        wfs_xyza, _, _ = point_source_centering.relocate_simple(
            wfs_orig,
            geom,
            maxchans[start:end],
            xs[start:end],
            ys[start:end],
            z_rels[start:end],
            alphas[start:end],
            channel_radius=10,
            relocate_dims="xyza",
            **ckw,
        )
        wfs_orig = wfs_orig.reshape(end - start, -1)
        wfs_yza = wfs_yza.reshape(end - start, -1)
        wfs_xyza = wfs_xyza.reshape(end - start, -1)

        loadings_orig[start:end] = ipca_orig.transform(wfs_orig)
        loadings_yza[start:end] = ipca_yza.transform(wfs_yza)
        loadings_xyza[start:end] = ipca_xyza.transform(wfs_xyza)

# %%
pcs_orig = ipca_orig.components_.reshape(K, T, C)
pcs_yza = ipca_yza.components_.reshape(K, T, C)
pcs_xyza = ipca_xyza.components_.reshape(K, T, C)

# %%
vis_utils.labeledmosaic(
    [pcs_orig[:, 15:, :], pcs_yza[:, 15:, :], pcs_xyza[:, 15:, :]],
    ["orig. PCs", "yza PCs", "xyza PCs"],
    collabels="12345",
    pad=1,
)

# %%
with h5py.File(feats_h5, "r+") as feats_f:
    feats_f.create_dataset("loadings_orig", data=loadings_orig)
    feats_f.create_dataset("loadings_yza", data=loadings_yza)
    feats_f.create_dataset("loadings_xyza", data=loadings_xyza)
    feats_f.create_dataset("pcs_orig", data=pcs_orig)
    feats_f.create_dataset("pcs_yza", data=pcs_yza)
    feats_f.create_dataset("pcs_xyza", data=pcs_xyza)

# %%
with h5py.File(original_h5, "r") as original_f:
    N, T, C = original_f["denoised_waveforms"].shape

# %%
with h5py.File(feats_h5, "r") as feats_f:
    for k in feats_f.keys():
        print(k, feats_f[k].dtype, feats_f[k].shape)
    alphas = feats_f["alpha"][:]
    loadings_orig = feats_f["loadings_orig"][:]
    loadings_yza = feats_f["loadings_yza"][:]
    loadings_xyza = feats_f["loadings_xyza"][:]
    maxptp = feats_f["maxptp"][:]
    pcs_orig = feats_f["pcs_orig"][:]
    pcs_xyza = feats_f["pcs_xyza"][:]
    pcs_yza = feats_f["pcs_yza"][:]
    times = feats_f["times"][:]
    xs = feats_f["x"][:]
    ys = feats_f["y"][:]
    z_abss = feats_f["z_abs"][:]
    z_disp = feats_f["z_disp"][:]
    z_reg = feats_f["z_reg"][:]
    z_rels = feats_f["z_rel"][:]

# %% [markdown]
# ## invariance / clustering

# %% tags=[]
good = np.flatnonzero(ys > 0.05)

# this handles a bug now fixed above, remove after next PCA fit
# good = good[good < 2048 * (N // 2048)]
good = np.setdiff1d(good, np.flatnonzero(loadings_orig[:, 0] == 0))
good = np.setdiff1d(good, np.flatnonzero(loadings_yza[:, 0] == 0))
good = np.setdiff1d(good, np.flatnonzero(loadings_xyza[:, 0] == 0))

# this handles some bug with duplicated spikes, see the bug tracker
# with h5py.File(original_h5, "r") as orig_f:
#     _, unique_inds = np.unique(orig_f["spike_index"][:, 0], return_index=True)
# good = np.intersect1d(good, unique_inds)

# feats_orig = np.ascontiguousarray(np.c_[xs, ys, z_reg, alphas][good]).T #, loadings_orig[:, :5]][good]).T
feats_orig = np.ascontiguousarray(np.c_[xs, ys, z_reg, alphas, loadings_orig[:, :3]][good]).T
feats_yza = np.ascontiguousarray(np.c_[xs, ys, z_reg, alphas, loadings_yza[:, :3]][good]).T
feats_xyza = np.ascontiguousarray(np.c_[xs, ys, z_reg, alphas, loadings_xyza[:, :3]][good]).T

# %%
N - len(unique_inds), (N - len(unique_inds)) / N

# %%
shuffle = rg().permutation(len(good))
invshuf = np.empty_like(shuffle)
for i, j in enumerate(shuffle):
    invshuf[j] = i

# %%
plot = (maxptp >= 6) & good_mask

# %%
with timer("isosplit"):
    print("Running clustering on original ...")
    labels_orig = isosplit(feats_orig[:, shuffle], K_init=1024)
    labels_orig = labels_orig[invshuf]
    print("Got", labels_orig.max() + 1, "clusters")
    labels_orig_full = np.full(N, -1, dtype=np.int32)
    labels_orig_full[good] = labels_orig

# %%
with timer("isosplit"):
    print("Running clustering on yza ...")
    labels_yza = isosplit(feats_yza[:, shuffle], K_init=4096)
    labels_yza = labels_yza[invshuf]
    print("Got", labels_yza.max() + 1, "clusters")
    labels_yza_full = np.full(N, -1, dtype=np.int32)
    labels_yza_full[good] = labels_yza

# %%
with timer("isosplit"):
    print("Running clustering on xyza ...")
    labels_xyza = isosplit(feats_xyza[:, shuffle], K_init=1024)
    labels_xyza = labels_xyza[invshuf]
    print("Got", labels_xyza.max() + 1, "clusters")
    labels_xyza_full = np.full(N, -1, dtype=np.int32)
    labels_xyza_full[good] = labels_xyza

# %%
fig, (aa, ab, ac) = plt.subplots(1, 3, sharey=True)

aa.scatter(xs[plot], z_reg[plot], c=labels_orig_full[plot], cmap=plt.cm.rainbow, s=1, marker="x");
aa.set_ylabel("registered depth")
aa.set_xlabel("x")
aa.set_title(f"no relocation - found {labels_orig.max() + 1}");

ab.scatter(xs[plot], z_reg[plot], c=labels_yza_full[plot], cmap=plt.cm.rainbow, s=1, marker="x");
# ab.set_ylabel("registered depth")
ab.set_xlabel("x")
ab.set_title(f"YZa - found {labels_yza.max() + 1}");

ac.scatter(xs[plot], z_reg[plot], c=labels_xyza_full[plot], cmap=plt.cm.rainbow, s=1, marker="x");
# ac.set_ylabel("registered depth")
ac.set_xlabel("x")
ac.set_title(f"XYZa -- found {labels_xyza.max() + 1}");

fig.suptitle("ISO-SPLIT clusters (x,y,z,a pcs 1,2,3)")
plt.tight_layout()
plt.show()

# %%
with h5py.File(feats_h5, "r+") as feats_f:
    for k in ["good_mask", "labels_orig", "labels_yza", "labels_xyza"]:
        del feats_f[k]
    good_mask = np.zeros(N, dtype=bool)
    good_mask[good] = 1
    feats_f.create_dataset("good_mask", data=good_mask)
    feats_f.create_dataset("labels_orig", data=labels_orig_full)
    feats_f.create_dataset("labels_yza", data=labels_yza_full)
    feats_f.create_dataset("labels_xyza", data=labels_xyza_full)

# %%
with h5py.File(feats_h5, "r") as feats_f:
    good_mask = feats_f["good_mask"][:]
    labels_orig = feats_f["labels_orig"][:]
    labels_yza = feats_f["labels_yza"][:]
    labels_xyza = feats_f["labels_xyza"][:]

# %%
vis_utils.lcorrs(z_disp, ys, alphas, loadings_orig, maxptp, plot);
plt.gcf().suptitle("pairplot: Unrelocated PCs vs. drift/localization")
plt.tight_layout()
plt.show()

# %%
vis_utils.lcorrs(z_disp, ys, alphas, loadings_yza, maxptp, plot);
plt.gcf().suptitle("pairplot: YZa PCs vs. drift/localization")
plt.tight_layout()
plt.show()

# %%
vis_utils.lcorrs(z_disp, ys, alphas, loadings_xyza, maxptp, plot);
plt.gcf().suptitle("pairplot: XYZa PCs vs. drift/localization")
plt.tight_layout()
plt.show()

# %%

# %%

# %%
fig, (aa, ab, ac) = plt.subplots(1, 3, sharey=True)

plt.sca(aa)
vis_utils.gcsboxes(z_disp, loadings_orig, labels_orig_full)
aa.set_xticklabels([f"pc{i}" for i in range(1, 6)])
aa.set_ylabel("GCS with z displacement")
aa.set_title("no reloc")

plt.sca(ab)
vis_utils.gcsboxes(z_disp, loadings_yza, labels_orig_full)
ab.set_xticklabels([f"pc{i}" for i in range(1, 6)])
ab.set_title("YZa")

plt.sca(ac)
vis_utils.gcsboxes(z_disp, loadings_xyza, labels_orig_full)
ac.set_xticklabels([f"pc{i}" for i in range(1, 6)])
ac.set_title("XYZa")

fig.suptitle("GCS(drift, pc loading) within (unrelocated) clusters")
plt.show();

# %%
fig, (aa, ab, ac) = plt.subplots(1, 3, sharey=True)

plt.sca(aa)
vis_utils.spearmanrboxes(z_disp, loadings_orig, labels_orig_full)
aa.set_xticklabels([f"pc{i}" for i in range(1, 6)])
aa.set_ylabel("SpearmanR with z displacement")
aa.set_title("no reloc")

plt.sca(ab)
vis_utils.spearmanrboxes(z_disp, loadings_yza, labels_orig_full)
ab.set_xticklabels([f"pc{i}" for i in range(1, 6)])
ab.set_title("YZa")

plt.sca(ac)
vis_utils.spearmanrboxes(z_disp, loadings_xyza, labels_orig_full)
ac.set_xticklabels([f"pc{i}" for i in range(1, 6)])
ac.set_title("XYZa")

fig.suptitle("SpearmanR(drift, pc loading) within (unrelocated) clusters")
plt.show();

# %%
