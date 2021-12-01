# -*- coding: utf-8 -*-
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

# %% [markdown]
# (Please ignore the code, could not figure out how to hide it in this PDF.)

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# %% tags=[]
import h5py
import numpy as np
from spike_psvae import (
    vis_utils, point_source_centering, localization, waveform_utils, decomp
)
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy as np
from tensorly.decomposition import parafac
from joblib import Memory

# %% tags=[]
mem = Memory("/tmp/reloc")

# %% tags=[]
plt.rc("figure", dpi=200)
rg = np.random.default_rng(0)


# %% tags=[]
def relocation_analysis(waveforms, maxchans, geom, name, K=40, channel_radius=8, do_pfac=True, seed=0, relocate_dims="xyza"):
    # -- localize in standard form
    std_wfs = waveform_utils.as_standard_local(
        waveforms, maxchans, geom, channel_radius=channel_radius
    )
    geomkind = "standard"
    # if waveforms.shape[2] == 4 + 2 * channel_radius:
    #     geomkind = "updown"
    #     channel_radius += 2
    #     std_wfs = waveforms
    # else:
    #     std_wfs = waveform_utils.get_local_waveforms(
    #         waveforms, channel_radius, geom, maxchans=maxchans, geomkind=geomkind
    #     )
    
    plt.figure(figsize=(6, 4))
    vis_utils.labeledmosaic([std_wfs[:16], std_wfs[16:32]], ["0-15", "16-32"], pad=2)
    plt.suptitle(f"{name}: 32 waveforms", fontsize=8)
    plt.tight_layout(pad=0.25)
    plt.show()
    
    x, y, z_rel, z_abs, alpha = localization.localize_waveforms(
        std_wfs, geom, maxchans=maxchans, jac=False, geomkind=geomkind, channel_radius=channel_radius
    )
    
    # -- relocated versions
    reloc, r, q = point_source_centering.relocate_simple(
        std_wfs, geom, maxchans, x, y, z_rel, alpha, channel_radius=channel_radius, geomkind=geomkind, relocate_dims=relocate_dims, interp_xz=True
    )
    reloc = reloc.numpy(); r = r.numpy(); q = q.numpy()
    
    # -- factor analysis
    # PCA
    def pca_resid_plot(wfs, ax=None, q=0.95, c="bk", name=None, pad4=False):
        wfs = wfs.reshape(wfs.shape[0], -1)
        wfs = wfs - wfs.mean(axis=0, keepdims=True)
        v = np.square(la.svdvals(wfs)[:K]) / np.prod(wfs.shape)
        ax = ax or plt.gca()
        totvar = np.square(wfs).mean()
        residvar = np.concatenate(([totvar], totvar - np.cumsum(v)))
        if pad4:
            ax.plot([totvar, totvar, totvar, totvar, *residvar][:K], marker=".", c=c[0], label=name)
        else:
            ax.plot(residvar[:50], marker=".", c=c[0], label=name)
    
    # Parafac
    cumparafac = mem.cache(decomp.cumparafac)
    def pfac_resid_plot(wfs, ax=None, q=0.95, c="bk", name=None, pad4=False):
        mwfs = wfs - wfs.mean(axis=0, keepdims=True)
        sses = cumparafac(mwfs, K)
        mses = sses / np.prod(wfs.shape[1:])
        n0 = np.square(mwfs).mean()
        ax = ax or plt.gca()
        mses = np.concatenate(([n0], mses))
        if pad4:
            ax.plot([n0, n0, n0, n0, *mses][:K], marker=".", c=c[0], label=name)
        else:
            ax.plot(mses, marker=".", c=c[0], label=name)
            
    # -- Plots
    # the ones we will show
    # inds = np.random.default_rng(seed).choice(std_wfs.shape[0], size=16)
    inds = np.arange(16)
    
    
    # Relocation x PTPs
    fig, axes = vis_utils.vis_ptps([std_wfs.ptp(1)[inds], q[inds]], ["observed ptp", "predicted ptp"], "bg")
    plt.suptitle(f"{name}: PTP predictions", fontsize=8)
    plt.tight_layout(pad=0.25)
    plt.show()
    fig, axes = vis_utils.vis_ptps([reloc.ptp(1)[inds], r[inds]], ["relocated ptp", "standard ptp"], "kr")
    plt.suptitle(f"{name}: Relocated PTPs", fontsize=8)
    plt.tight_layout(pad=0.25)
    plt.show()
    
    # PCA resid
    def a(semilogy=False):
        plt.figure(figsize=(6, 4))
        pca_resid_plot(std_wfs, name="original")
        pca_resid_plot(reloc, c="rk", name="relocated", pad4=True)
        if semilogy: plt.semilogy()
        plt.title(f"{name}: does relocating help PCA?")
        plt.ylabel("residual variance (s.u.)")
        plt.xlabel("number of components (0=full data)")
        plt.legend(fancybox=False)
        plt.show()
    a(); a(1)
    
    # Parafac
    if do_pfac:
        def b(semilogy=False):
            plt.figure(figsize=(6, 4))
            pfac_resid_plot(std_wfs, name="original")
            pfac_resid_plot(reloc, c="rk", name="relocated", pad4=True)
            if semilogy: plt.semilogy()
            plt.title(f"{name}: does relocating help Parafac?")
            plt.ylabel("residual variance (s.u.)")
            plt.xlabel("number of components (0=full data)")
            plt.show()
        b(); b(1)
        
    # -- low rank models + residuals
    def modelplot(wfs, model="PCA", flavor="original", add_center=True, k=1):
        mean = wfs.mean(axis=0, keepdims=True)
        mwfs = wfs - mean
        
        if model == "PCA":
            U, s, Vh = la.svd(mwfs.reshape(mwfs.shape[0], -1), full_matrices=False)
            recon = (U[inds, :k] @ np.diag(s[:k]) @ Vh[:k]).reshape(16, *mwfs.shape[1:])
        elif model == "Parafac":
            weights, factors = parafac(mwfs, k)
            recon = np.einsum("l,il,jl,kl->ijk", weights, factors[0][inds], *factors[1:])
        else:
            assert False
        
        batch = mwfs[inds]
        if add_center:
            batch += mean
            recon += mean
        
        plt.figure(figsize=(6, 4))
        vis_utils.labeledmosaic(
            [batch, recon, batch - recon],
            [flavor, f"{model} recon", "residual"],
            pad=2,
        )
        plt.suptitle(f"{name}: {model}({k}) - {flavor} data", fontsize=8)
        plt.tight_layout(pad=0.25)
        plt.show()
    
    modelplot(std_wfs)
    modelplot(reloc, flavor="relocated")
    modelplot(std_wfs, k=5)
    modelplot(reloc, flavor="relocated", k=5)
    
    if do_pfac:
        modelplot(std_wfs, model="Parafac")
        modelplot(reloc, model="Parafac", flavor="relocated")
        modelplot(std_wfs, model="Parafac", k=5)
        modelplot(reloc, model="Parafac", flavor="relocated", k=5)



# %% [markdown]
# # Relocating, PCA and Parafac
#
# This is a huge pile of figures showing the effect of the simple point source relocation on PCA and Parafac. So, does it help these models reconstruct the data? Yes. It does. At least a little bit.
#
# Below, the same set of figures is shown for 3 data sets. First, ~170 nice NP2 templates. Next, the same templates, but with 10 or so weird looking ones removed. Finally, 10,000 denoised spikes from an NP2 probe.

# %% [markdown] tags=[]
# # All Templates

# %% tags=[]
with h5py.File("../data/spt_yasstemplates.h5") as h5:
    wfs = h5["waveforms"][:]
    geom = h5["geom"][:]
    maxchans = h5["maxchans"][:]
    relocation_analysis(wfs, maxchans, geom, "All Templates, Just Y/Z/alpha", K=30, do_pfac=False, relocate_dims="yza")

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# # Culled Templates

# %% tags=[]
with h5py.File("../data/spt_yasstemplates_culled.h5") as h5:
    wfs = h5["waveforms"][:]
    geom = h5["geom"][:]
    maxchans = h5["maxchans"][:]
    relocation_analysis(wfs, maxchans, geom, "Culled Templates", K=30)

# %% [markdown]
# # 10,000 denoised NP2 waveforms
#
# (I did not run Parafac on these because I didn't want to wait around.)

# %%
with h5py.File("../data/wfs_locs_b.h5") as h5:
    wfs = h5["denoised_waveforms"][:10_00]
    geom = h5["geom"][:]
    maxchans = h5["max_channels"][:10_00]
    relocation_analysis(wfs, maxchans, geom, "10k Denoised NP2, Just Y/Z/alpha", do_pfac=False, K=30)

# %%
