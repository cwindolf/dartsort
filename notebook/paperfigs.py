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
from celluloid import Camera

# %%
from npx import lib, reg, cuts

# %%
from isosplit import isosplit

# %%
from isosplit5 import isosplit5

# %%
from spike_psvae import waveform_utils, localization, point_source_centering, vis_utils, statistics

# %%
rg = lambda k=0: np.random.default_rng(k)

# %% [markdown]
# ### plot kit

# %%
sns.set_style("ticks")

# %%
plt.rc("figure", dpi=200)

# %%
plt.rc("text", usetex=True)
plt.rc("font", family="TeX Gyre Pagella")

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

# %%

# %%

# %% [markdown] tags=[]
# ## data munging

# %%
# %ll ../data/*{nzy,rigid}*

# %%
# np2 data
ctx_h5 = h5py.File("../data/ks_np2_nzy_cortex.h5", "r")
hc_h5 = h5py.File("../data/ks_np2_nzy_hippocampus.h5", "r")
th_h5 = h5py.File("../data/ks_np2_nzy_thalamus.h5", "r")

# np1 data
np1_h5 = h5py.File("../data/yass_np1_nzy.h5", "r")

# %%
h5s = {"NP2 Cortex": ctx_h5, "NP2 Hippocampus": hc_h5, "NP2 Thalamus": th_h5, "NP1": np1_h5}

# %%
fns = {"NP2 Cortex": "ctx", "NP2 Hippocampus": "hc", "NP2 Thalamus": "th", "NP1": "np1"}

# %%
# load geometry
geom_np1 = np1_h5["geom"][:]
geom_np2 = ctx_h5["geom"][:]

# rigid disp
p = np.load("../data/np2_p_rigid.npy")

# %%
# load unified loc data for NP2 rasters
times_2 = np.concatenate([
    ctx_h5["spike_index"][:, 0],
    hc_h5["spike_index"][:, 0],
    th_h5["spike_index"][:, 0],
]) / 30_000
z_reg_2 = np.concatenate([
    ctx_h5["z_reg"],
    hc_h5["z_reg"],
    th_h5["z_reg"],
])
x_2 = np.concatenate([
    ctx_h5["x"],
    hc_h5["x"],
    th_h5["x"],
])
maxptp_2 = np.concatenate([
    ctx_h5["maxptp"],
    hc_h5["maxptp"],
    th_h5["maxptp"],
])

# %%
# load NP1 loc data
times_1 = np1_h5["spike_index"][:, 0] / 30_000
z_reg_1 = np1_h5["z_reg"][:]
x_1 = np1_h5["x"][:]
maxptp_1 = np1_h5["maxptp"][:]

# %% [markdown]
# ## basic explanatory plots

# %% [markdown]
# ### local channel geometry

# %%
fig, (aa, ab) = plt.subplots(1, 2, sharey=False)
lgeom1 = geom_np1[:18].copy()
lgeom1[:, 1] -= lgeom1[9, 1]
lgeom2 = geom_np2[:18].copy()
lgeom2[:, 1] -= lgeom2[9, 1]
aa.plot(lgeom1[:, 0], lgeom1[:, 1], color=blue, lw=1, marker=".")
aa.set_xticks(np.sort(lgeom1[:4, 0]))
aa.set_yticks(np.sort(lgeom1[:18:2, 1]))
aa.set_ylabel("z relative to max channel")
aa.set_xlabel("x")
ab.plot(lgeom2[:18, 0], lgeom2[:18, 1], color=blue, lw=1, marker=".")
ab.set_xticks(np.sort(lgeom2[:2, 0]))
ab.set_yticks(np.sort(lgeom2[:18:2, 1]))
ab.set_xlabel("x")
aa.set_title("NP1")
ab.set_title("NP2")
fig.suptitle("Geometry of 18 channel neighborhoods")
plt.savefig("../figs/geometry.pdf")
plt.show()

# %%
# anatomical split plots NP2
Rreg2, _, _ = lib.faster(maxptp_2, z_reg_2, times_2)
meanproj2, _, _ = lib.faster(maxptp_2, z_reg_2, x_2)

# make plot
fig, axes = plt.subplot_mosaic("ab\ncc", figsize=(6, 5))
aa = axes["a"]; ab = axes["b"]; ac = axes["c"]

cuts.plot(Rreg2, ax=aa)
aa.axhline(1175, lw=1, c=blue)
aa.axhline(1870, lw=1, c=blue)
aa.set_ylabel("registered z")
aa.set_xlabel("time", labelpad=-9)
aa.set_title("raster: PTP by time and reg.\\ z")

ab.hist(z_reg_2, facecolor="k", edgecolor="w", linewidth=0, bins=np.arange(3050, step=1), log=True, color="k")
ab.set_box_aspect(1)
ab.axvline(1175, lw=1, c=blue)
ab.axvline(1870, lw=1, c=blue)
# ab.set_xlabel("registered depth")
ab.set_ylabel("spike count")
ab.set_title("histogram of reg.\\ z")

cuts.plot(meanproj2.T, ax=ac, aspect=0.33)
ac.axvline(1175, lw=1, c=blue)
ac.axvline(1870, lw=1, c=blue)
ac.set_ylabel("x", labelpad=-12)
ac.set_xlabel("reg.\\ z", labelpad=-9)
ac.set_title("mean projection of PTP on x/reg.\\ z")

ac.text(587.5, 66, "cortex", color=blue, backgroundcolor=[1,1,1,1], ha="center", va="center")
ac.text(1525, 66, "hippocampus", color=blue, backgroundcolor=[1,1,1,1], ha="center", va="center")
ac.text(2500, 66, "thalamus", color=blue, backgroundcolor=[1,1,1,1], ha="center", va="center")

# plt.tight_layout(pad=0.0)
fig.suptitle("NP2 recording regions (divided at registered z=1175,1870)", y=0.96)
fig.savefig("../figs/np2_regions.pdf")
plt.show()

# %%
# similar plot for NP1
Rreg1, _, _ = lib.faster(maxptp_1, z_reg_1, times_1)
meanproj1, _, _ = lib.faster(maxptp_1, z_reg_1, x_1)

# make plot
fig, axes = plt.subplot_mosaic("ab\ncc", figsize=(6, 5))
aa = axes["a"]; ab = axes["b"]; ac = axes["c"]

cuts.plot(Rreg1, ax=aa)
aa.set_ylabel("registered z")
aa.set_xlabel("time", labelpad=-9)
aa.set_title("raster: PTP by time and reg.\\ z")

ab.hist(z_reg_1, facecolor="k", edgecolor="w", linewidth=0, bins=np.arange(3050, step=1), log=True, color="k")
ab.set_box_aspect(1)
# ab.set_xlabel("registered depth")
ab.set_ylabel("spike count")
ab.set_title("histogram of reg.\\ z")

cuts.plot(meanproj1.T, ax=ac, aspect=0.33)
ac.set_ylabel("x", labelpad=-11)
ac.set_xlabel("reg.\\ z")
ac.set_title("mean projection of PTP on x/reg.\\ z")

# plt.tight_layout(pad=0.0)
fig.suptitle("NP1 recording summary", y=0.96)
plt.savefig("../figs/np1_regions.pdf")
plt.show()

# %% [markdown]
# ### localization / re-location plots

# %%
for k, v in h5s.items():
    fig, axes = vis_utils.locrelocplots(v, name=k, seed=1)
    fig.savefig(f"../figs/{fns[k]}_ptpreloc.pdf")
    plt.show()

# %% [markdown]
# ## PCA explained variance plots

# %%
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for ax, (k, v) in zip(axes.flat, h5s.items()):
    vis_utils.reloc_pcaresidplot(v, name=k, ax=ax, nolabel=True)
plt.suptitle("PCA error before/after relocation")
for ax in axes[1]: ax.set_xlabel("number of factors")
fig.text(0.04, 0.5, 'PCA unexplained variance (s.u.)', va='center', rotation='vertical')
fig.savefig("../figs/pcaerr.pdf")
plt.show()

# %%
print("hi")

# %%
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for ax, (k, v) in zip(axes.flat, h5s.items()):
    vis_utils.reloc_pcaresidplot(v, name=k, ax=ax, nolabel=True, B=50_000, kind="invert")
plt.suptitle("PCA error before/after relocation")
for ax in axes[1]: ax.set_xlabel("number of factors")
fig.text(0.04, 0.5, 'PCA error after inverting relocation (s.u.)', va='center', rotation='vertical')
fig.savefig("../figs/pcainverr.pdf")
plt.show()

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## PCA heatmaps

# %%
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(4, 6))
for ax, (k, v) in zip(axes.flat, h5s.items()):
    vis_utils.labeledmosaic(
        [v["pcs_orig"][:5, 20:80], v["pcs_yza"][:5, 20:80], v["pcs_xyza"][:5, 20:80]],
        ["orig.", "$yz\\alpha$", "$xyz\\alpha$"],
        collabels=range(1, 6),
        ax=ax,
        pad=1,
        cbar=False,
    )
    ax.set_title(k, x=0.63)
fig.suptitle("Principal components before/after relocation", y=0.97, x=0.54)
plt.tight_layout()
plt.savefig("../figs/pcims.pdf")
plt.show()

# %% [markdown]
# ## Wf / PCA recon (3 ways) traceplots

# %%
vis_utils.pcarecontrace(ctx_h5)

# %% [markdown]
# ## drift vs loadings for top units

# %%
for k, v in h5s.items():
    for unit in range(4):
        vis_utils.loadings_vs_disp(v, unit, p, name=k, by="counttrend")
        plt.savefig(f"../figs/{fns[k]}_ldisp_{unit}.png", dpi=300, bbox_inches="tight")
        plt.show()

# %% tags=[]
for k, v in h5s.items():
    for unit in range(2):
        for which in ["orig", "yza", "xyza"]:
            top, grid = vis_utils.pairplot_loadings(v, unit, which, name=k, by="counttrend")
            plt.gcf().suptitle(f"{k} unit {top}, {which}", y=0.98)
            plt.tight_layout()
            plt.savefig(f"../figs/{fns[k]}_pair_{which}_{unit}.pdf", bbox_inches="tight")
            plt.show()

# %%
fits = vis_utils.template_psfit(ctx_h5)

# %%
for k, v in h5s.items():
    fig, axes = plt.subplots(2, 3, sharex=True, sharey="row", figsize=(3, 3))
    for i, which in enumerate(["orig", "yza", "xyza"]):
        vis_utils.uncolboxes(v, which=which, kind="gcs", ax=axes[0, i])
        vis_utils.uncolboxes(v, which=which, kind="spear", ax=axes[1, i])
        axes[0, i].set_title(which2txt[which])
        axes[1, i].set_xticklabels([f"${i}$" for i in range(1, 6)])
    axes[0, 0].set_ylabel("GCS")
    axes[1, 0].set_ylabel("Spearman's $r$")
    axes[1, 1].set_xlabel("principal component number")
    fig.suptitle(f"Corr.\\ of uncollided units with disp.\\ in {k}", fontsize=8, y=0.99)
    plt.tight_layout(pad=0.1)
    plt.savefig(f"../figs/{fns[k]}_box.pdf", bbox_inches="tight")
    plt.show()

# %% [markdown]
# # sorter cluster vis

# %%
np.std(np.arange(35).reshape(5,7), axis=0)

# %%
for k, v in h5s.items():
    zlims = None
    if "NP1" in k:
        zlims = [1100, 2400]
    vis_utils.sortedclust_pcvis(v, k, zlims=zlims)
    plt.savefig(f"../figs/{fns[k]}_sorted.png", dpi=300, bbox_inches="tight")
    plt.show()

# %% [markdown]
#  ## picking number of PCs for clustering by ARI

# %%
pserrs = {}
batch_size = 128
for k, v in h5s.items():
    g = v["geom"][:]
    mcs = v["max_channels"][:]
    N = len(mcs)
    x = v["x"][:]
    y = v["y"][:]
    z_rel = v["z_rel"][:]
    alpha = v["alpha"][:]
    
    pserr = np.empty(len(v["denoised_waveforms"]))
    pserr[:] = np.inf
    for b in trange((N + 1) // batch_size, desc="fit"):
        start = b * batch_size
        end = min(N, (b + 1) * batch_size)

        bwfs = v["denoised_waveforms"][start:end]
        B, _, _ = bwfs.shape
        ptp, ptp_hat = point_source_centering.ptp_fit(
            bwfs,
            g,
            mcs[start:end],
            x[start:end],
            y[start:end],
            z_rel[start:end],
            alpha[start:end],
            channel_radius=8,
            geomkind="standard",
        )
        pserr[start:end] = np.square(ptp - ptp_hat).mean(axis=1)
    
    pserrs[k] = pserr

# %%
well_modeled = {}
for k in pserrs:
    well_modeled[k] = pserrs[k] < np.quantile(pserrs[k], 0.9)
    # throw away some edge times
    times = h5s[k]["spike_index"][:, 0] / 30000
    well_modeled[k] &= (times >= 200) & (times < 800)
    well_modeled[k] = np.flatnonzero(well_modeled[k])
    _, unique_inds = np.unique(h5s[k]["spike_index"][:, 0], return_index=True)
    well_modeled[k] = np.intersect1d(well_modeled[k], unique_inds)

# %%
npcs = 3
clustering = {}
ariss = {}

for k, v in h5s.items():
    print(k, flush=True)
    
    shuffle = rg().permutation(len(well_modeled[k]))
    invshuf = np.empty_like(shuffle)
    for i, j in enumerate(shuffle):
        invshuf[j] = i

    x = v["x"][:][well_modeled[k]]
    y = v["y"][:][well_modeled[k]]
    z = v["z_reg"][:][well_modeled[k]]
    alpha = v["alpha"][:][well_modeled[k]]
    ids = v["spike_train"][:, 1][well_modeled[k]]
    
    lo = v["loadings_orig"][:]
    lo /= np.std(lo, axis=0, keepdims=True) / 16
    lo = lo[well_modeled[k]]
    
    ly = v["loadings_yza"][:]
    ly /= np.std(lo, axis=0, keepdims=True) / 16
    ly = ly[well_modeled[k]]
    
    lx = v["loadings_xyza"][:]
    lx /= np.std(lo, axis=0, keepdims=True) / 16
    lx = lx[well_modeled[k]]
    
    f = np.c_[x, y, z, alpha, lo[:, :npcs]]
    co = isosplit(f[shuffle].T, K_init=1024)
    co = co[invshuf]
    ario = adjusted_rand_score(ids, co)
    f = np.c_[x, y, z, alpha, ly[:, :npcs]]
    cy = isosplit(f[shuffle].T, K_init=1024)
    cy = cy[invshuf]
    ariy = adjusted_rand_score(ids, cy)
    f = np.c_[x, y, z, alpha, lx[:, :npcs]]
    cx = isosplit(f[shuffle].T, K_init=1024)
    cx = cx[invshuf]
    arix = adjusted_rand_score(ids, cx)
    
    aris = {"orig": ario, "yza": ariy, "xyza": arix}
    ariss[k] = aris
    c = {"orig": co, "yza": cy, "xyza": cx}
    clustering[k] = c    

# %%
for k, v in h5s.items():
    x = v["x"][:][well_modeled[k]]
    y = v["y"][:][well_modeled[k]]
    z = v["z_reg"][:][well_modeled[k]]
    alpha = v["alpha"][:][well_modeled[k]]
    ids = v["spike_train"][:, 1][well_modeled[k]]
    
    lo = v["loadings_orig"][:, :3]
    lo /= np.std(lo, axis=0, keepdims=True)
    lo = lo[well_modeled[k]]
    
    ly = v["loadings_yza"][:, :3]
    ly /= np.std(lo, axis=0, keepdims=True)
    ly = ly[well_modeled[k]]
    
    lx = v["loadings_xyza"][:, :3]
    lx /= np.std(lo, axis=0, keepdims=True)
    lx = lx[well_modeled[k]]
    
    
    zlims = None
    if "NP1" in k:
        zlims = [1100, 2400]
    
    vis_utils.relocclusts(
        k,
        x,
        y,
        z,
        alpha,
        lo,
        ly,
        lx,
        clustering[k],
        ariss[k],
        zlims=zlims,
    )
    plt.savefig(f"../figs/{fns[k]}_relocclust.png", dpi=300, bbox_inches="tight")
    plt.show()

# %%

# %%

# %%
templates = ctx_h5["templates"][:]

# %%
twfs, mcs = waveform_utils.get_local_waveforms(templates, 8, ctx_h5["geom"][:], geomkind="standard")

# %%
twfs.shape

# %%
gtz = twfs.ptp(1).ptp(1) > 0
twfs = twfs[gtz]
mcs = mcs[gtz]

# %%
vis_utils.labeledmosaic([twfs[:8], twfs[8:16]], cbar=False)

# %%
x, y, zr, za, alpha = localization.localize_waveforms(twfs, ctx_h5["geom"][:], maxchans=mcs, geomkind="standard", channel_radius=8)

# %%
p, q = point_source_centering.ptp_fit(
    twfs,
    ctx_h5["geom"][:],
    mcs,
    x,
    y,
    zr,
    alpha,
    channel_radius=8,
    geomkind="standard",
)
psfits = np.square(p - q).mean(axis=1)

# %%
byfit = np.argsort(psfits)

# %%
units = byfit[[1, 3, 5, 6, 7]]

# %%
vis_utils.labeledmosaic([twfs[units]], collabels=units, rowlabels="", cbar=True)

# %%
from scipy import signal

# %%
sns.regplot(x=np.arange(len(y)), y=y[rg().permutation(len(y))], scatter_kws=dict(s=1), label="shuffled")
sns.regplot(x=np.arange(len(y)), y=y[byfit], scatter_kws=dict(s=1), label="sorted by PS fit")
plt.ylabel("y")
plt.xlabel("template index (before/after sorting by PS fit)")
plt.legend()
plt.title("Better point source fit means less y=0")


# %%
def simplot(ix):
    wf = twfs[ix]
    mc = mcs[ix]
    x0 = x[ix]
    y0 = y[ix]
    zr0 = zr[ix]
    a0 = alpha[ix]
    print(ix, ":", x0, y0, zr0, a0)
    
    pos = ctx_h5["pcs_orig"][:]
    pys = ctx_h5["pcs_yza"][:]
    pxs = ctx_h5["pcs_xyza"][:]
    print(pos.shape)
    
    lzos = []
    lzys = []
    lzxs = []
    zs = []
    for dz in range(-16, 17):
        zs.append(dz)
        shifted, shift_target_ptp = point_source_centering.shift(
            wf, mc, ctx_h5["geom"][:], dz=dz, channel_radius=8, geomkind="standard"
        )
        shifted_ptp = shifted.ptp(0)
        sx, sy, szr, sza, salpha = localization.localize_ptp(
            shifted_ptp,
            mc + (shifted_ptp.argmax() - wf.ptp(0).argmax()),
            ctx_h5["geom"][:],
            geomkind="standard",
        )
        (std_yza,), _, _ = point_source_centering.relocate_simple(
            shifted[None, :, :],
            ctx_h5["geom"][:],
            [mc],
            sx,
            sy,
            szr,
            salpha,
            channel_radius=8,
            geomkind="standard",
            relocate_dims="yza",
        )
        (std_xyza,), _, _ = point_source_centering.relocate_simple(
            shifted[None, :, :],
            ctx_h5["geom"][:],
            [mc],
            sx,
            sy,
            szr,
            salpha,
            channel_radius=8,
            geomkind="standard",
            relocate_dims="xyza",
        )
        lzos.append(np.einsum("ktc,tc->k", pos, shifted - ctx_h5["mean_orig"][:]))
        lzys.append(np.einsum("ktc,tc->k", pys, std_yza.numpy() - ctx_h5["mean_yza"][:]))
        lzxs.append(np.einsum("ktc,tc->k", pxs, std_xyza.numpy() - ctx_h5["mean_xyza"][:]))
    lzos = np.array(lzos)
    lzys = np.array(lzys)
    lzxs = np.array(lzxs)
    plt.scatter(zs, lzos[:, 0], marker=".", color="k", label="no reloc")
    plt.scatter(zs, lzys[:, 0], marker=".", color=green, label="$yz\\alpha$")
    plt.scatter(zs, lzxs[:, 0], marker=".", color=purple, label="$xyz\\alpha$")
    plt.legend()
    plt.xlabel("$z$ shift")
    plt.ylabel("first PC loading")
    plt.title(f"cortex unit {ix}")
    plt.show()


# %%
for unit in units:
    simplot(unit)

# %%
ctx_h5["geom"][:5]

# %%
list(ctx_h5.keys())


# %%
def simmov(ix=None, spike_ix=None, shiftdim="z", xtitle="", pc=0):
    assert shiftdim in "xyza"
    
    if ix is not None:
        wf = twfs[ix]
        mc = mcs[ix]
        x0 = x[ix]
        y0 = y[ix]
        zr0 = zr[ix]
        a0 = alpha[ix]
    elif spike_ix is not None:
        wf = ctx_h5["denoised_waveforms"][spike_ix]
        mc = ctx_h5["max_channels"][spike_ix]
        x0 = ctx_h5["x"][spike_ix]
        y0 = ctx_h5["y"][spike_ix]
        zr0 = ctx_h5["z_rel"][spike_ix]
        a0 = ctx_h5["alpha"][spike_ix]
        ix = ctx_h5["spike_train"][spike_ix, 1]
    print(ix, ":", x0, y0, zr0, a0)
    print(wf.shape, mc)
    
    pos = ctx_h5["pcs_orig"][:]
    pys = ctx_h5["pcs_yza"][:]
    pxs = ctx_h5["pcs_xyza"][:]
    print(pos.shape)
    
    lzos = []
    lzys = []
    lzxs = []
    shifts = []
    
    fig, axes = plt.subplot_mosaic("aaaabbbb\n..cdef..", figsize=(6,5), gridspec_kw=dict(height_ratios=(4, 2)))
    aa = axes["a"]
    ab = axes["b"]
    dimname = dict(x="x", z="z", y="y", a="\\alpha")[shiftdim]
    camera = Camera(fig)
    if shiftdim in "xz":
        ab.set_xlabel(f"${dimname}$ shift")
    else:
        ab.set_xlabel(f"target ${dimname}$")
    ab.set_ylabel("first PC loading")
    ab.set_title(f"cortex unit {ix}")
    
    fig.suptitle(f"PCA vs. ${dimname}$ shifts {xtitle}")
    
    shift_range = dict(z=range(-16, 17), x=range(-10, 42), y=[1, 2, 3, 4, 5, 6, 7, 8, 9] + list(range(10, 200, 5)), a=range(20, 200, 5))[shiftdim]
    for dd in shift_range:
        shifts.append(dd)

        if shiftdim == "z":
            shift_kwarg = dict(dz=dd)
        elif shiftdim == "x":
            shift_kwarg = dict(dx=dd)
        elif shiftdim == "y":
            shift_kwarg = dict(y1=dd)
        elif shiftdim == "a":
            shift_kwarg = dict(alpha1=dd)
        
        shifted, shift_target_ptp = point_source_centering.shift(
            wf, mc, ctx_h5["geom"][:], channel_radius=8, geomkind="standard", **shift_kwarg
        )
        orig_ptp = wf.ptp(0)
        shifted_ptp = shifted.ptp(0)
        sx, sy, szr, sza, salpha = localization.localize_ptp(
            shifted_ptp,
            mc, # + (shifted_ptp.argmax() - wf.ptp(0).argmax()),
            ctx_h5["geom"][:],
            geomkind="standard",
        )
        (std_yza,), stereo_yza, pred_yza = point_source_centering.relocate_simple(
            shifted[None, :, :],
            ctx_h5["geom"][:],
            [mc],
            sx,
            sy,
            szr,
            salpha,
            channel_radius=8,
            geomkind="standard",
            relocate_dims="yza",
        )
        (std_xyza,), stereo_xyza, pred_xyza = point_source_centering.relocate_simple(
            shifted[None, :, :],
            ctx_h5["geom"][:],
            [mc],
            sx,
            sy,
            szr,
            salpha,
            channel_radius=8,
            geomkind="standard",
            relocate_dims="xyza",
        )
        
        lzos.append(np.einsum("ktc,tc->k", pos, shifted - ctx_h5["mean_orig"][:]))
        lzys.append(np.einsum("ktc,tc->k", pys, std_yza.numpy() - ctx_h5["mean_yza"][:]))
        lzxs.append(np.einsum("ktc,tc->k", pxs, std_xyza.numpy() - ctx_h5["mean_xyza"][:]))
        
        show = np.array([wf, shifted, std_yza.numpy(), std_xyza.numpy()])[:, 20:-40]
        print(show.min(), show.max())
        vmin = -50
        vmax = 10
        show = np.clip(show, vmin, vmax)
        # print(wf.shape, pos.shape, pys.shape, pxs.shape)
        pcshow = np.array([np.full(wf.shape, vmax), 8 * pos[0], 8 * pys[0], 8 * pxs[0]])[:, 20:-40]
        vis_utils.labeledmosaic(
            [show, pcshow],
            rowlabels=["waveform", "pcs"],
            collabels=["orig", "shifted", "yza", "xyza"],
            ax=aa,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
        )
        nrl = ab.scatter(shifts, np.array(lzos)[:, pc], marker=".", color="k", label="no reloc")
        yrl = ab.scatter(shifts, np.array(lzys)[:, pc], marker=".", color=green, label="$yz\\alpha$")
        xrl = ab.scatter(shifts, np.array(lzxs)[:, pc], marker=".", color=purple, label="$xyz\\alpha$")
        ab.legend(labels=["no reloc", "$yz\\alpha$", "$xyz\\alpha$"], handles=[nrl, yrl, xrl])
        for ww, pp, kk, cc, dd, tt in zip([orig_ptp, shifted_ptp, std_yza.numpy().ptp(0), std_xyza.numpy().ptp(0)], [None, shift_target_ptp, stereo_yza, stereo_xyza], "cdef", ["k", "k", darkgreen, darkpurple], ["silver", "silver", lightgreen, lightpurple], ["orig", "shifted", "yza stdized", "xyza stdized"]):
            vis_utils.plot_ptp(ww[None, ...], np.array([axes[kk]]), "", cc, [""])
            if pp is not None:
                vis_utils.plot_ptp(np.array(pp)[None, ...], np.array([axes[kk]]), "", dd, [""])
            axes[kk].set_title(tt)
        plt.tight_layout()
        camera.snap()
    return camera


# %% tags=[]
c = simmov(ix=units[0], shiftdim="a", xtitle="-- cortex template 151", pc=3)
anim = c.animate()
HTML(anim.to_html5_video())

# %%
c = simmov(spike_ix=50000, shiftdim="z", xtitle="-- spike 50,000 (unit 54)", pc=4)
anim = c.animate()
HTML(anim.to_html5_video())

# %%
