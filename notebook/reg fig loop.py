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
from matplotlib import gridspec
from pathlib import Path
import matplotlib as mpl

# %%
from tqdm.auto import tqdm

# %%
from spike_psvae import ibme

# %%
import h5py

# %%
plt.rc("figure", dpi=200)

# %%
data_dir = Path("/mnt/3TB/charlie/re_datasets/")

# %%
# %ll /mnt/3TB/charlie/re_datasets/c17772a9-21b5-49df-ab31-3017addea12e/

# %%
list(data_dir.glob("*"))

# %%
testcases = [
    'ce397420-3cd2-4a55-8fd1-5e28321981f4',  #  0 ok
    'e31b4e39-e350-47a9-aca4-72496d99ff2a',  #  2 ok
    '1e176f17-d00f-49bb-87ff-26d237b525f1',  #  4 ok
    'b25799a5-09e8-4656-9c1b-44bc9cbb5279',  #  5 ok
    'c17772a9-21b5-49df-ab31-3017addea12e',  #  6 ok
    "31f3e083-a324-4b88-b0a4-7788ec37b191"
]

# %%
for dataset in sorted(data_dir.glob("*")):
    print(f'"{dataset.stem}",')


# %%
# testcases = [
#     "1e176f17-d00f-49bb-87ff-26d237b525f1",
#     "31f3e083-a324-4b88-b0a4-7788ec37b191",
#     "c17772a9-21b5-49df-ab31-3017addea12e",
# ]

# %%
fig_dir = Path("../reg_figs_a/").resolve()
fig_dir.mkdir(exist_ok=True)

# %%
savedir = Path("../re_reg_a/").resolve()
savedir.mkdir(exist_ok=True)

# %%
pids = [
 "1e176f17-d00f-49bb-87ff-26d237b525f1",
"31f3e083-a324-4b88-b0a4-7788ec37b191",
"6fc4d73c-2071-43ec-a756-c6c6d8322c8b",
"b25799a5-09e8-4656-9c1b-44bc9cbb5279",
"c17772a9-21b5-49df-ab31-3017addea12e",
"ce397420-3cd2-4a55-8fd1-5e28321981f4",
"e31b4e39-e350-47a9-aca4-72496d99ff2a",
"f03b61b4-6b13-479d-940f-d1608eb275cc",
"f26a6ab1-7e37-4f8d-bb50-295c056e1062",
"f2ee886d-5b9c-4d06-a9be-ee7ae8381114",
"f86e9571-63ff-4116-9c40-aa44d57d2da9",
   
]
savedir = Path("/mnt/3TB/charlie/re_reg_a/").resolve()
pyksdir = Path("/mnt/3TB/charlie/pyks_drift")
todir = Path("/mnt/3TB/charlie/figs/drift_rasters_new_nocorr_rob1_10")
todir.mkdir(parents=True, exist_ok=True)
for pid in tqdm(pids):
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()


    print(pid)
    
    npz = savedir / f"{pid}.npz"
    print(pid, npz.exists())
    if not npz.exists():
        continue
        
    omaxptp = np.load(data_dir / pid / "maxptps.npy")
    # which = maxptp > 6
    which = slice(None)
    ox, oy, oz, oa = np.load(data_dir / pid / "localizations.npy")[which, :4].T
    ot = np.load(data_dir / pid / "spike_index.npy")[which, 0] / 30000
    
        
    # with np.load(npz) as f:
    #     print(list(f.keys()))
    #     x = f["x"]
    #     y = f["y"]
    #     t = f["t"]
    #     maxptp = f["maxptp"]
    #     z_reg = f["z_reg"]
    # print(oz.min(), oz.max())
    # print(z_reg.min(), z_reg.max())
    # if np.isnan(z_reg.max()):
        # continue
        
    z_reg, dispmap = ibme.register_nonrigid(
        omaxptp, oz, ot,
        n_windows=10,
        disp=2000,
        batch_size=64,
        robust_sigma=1,
        corr_threshold=0.0,
        widthmul=1,
        rigid_init=False,
    )
    
    r0, *_ = ibme.fast_raster(omaxptp, oz - oz.min(), ot)
    
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(np.clip(r0, 0, 13), aspect=r0.shape[1] / r0.shape[0], cmap=plt.cm.cubehelix, origin="lower")
    plt.colorbar(shrink=0.3)
    plt.title(f"{pid}  |  PTP raster, unreg.", fontsize=10)
    plt.xlabel("t (s)", )
    plt.ylabel("depth (um)", )
    plt.tick_params(labelsize=8)
    plt.ylim([0, min(oz.max() - oz.min(), z_reg.max() - z_reg.min())])
    fig.savefig(todir / f"{pid}_a_unreg.png", bbox_inches="tight")
    
    plt.close(fig)
    
    r1, *_ = ibme.fast_raster(omaxptp, z_reg - z_reg.min(), ot)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(np.clip(r1, 0, 13), aspect=r0.shape[1] / r0.shape[0], cmap=plt.cm.cubehelix, origin="lower")
    plt.colorbar(shrink=0.3)
    plt.title(f"{pid}  |  PTP>6 raster, reg", fontsize=10)
    plt.xlabel("t (s)", )
    plt.ylabel("depth (um)", )
    plt.tick_params(labelsize=8)
    plt.ylim([0, min(oz.max() - oz.min(), z_reg.max() - z_reg.min())])
    fig.savefig(todir / f"{pid}_b_reg.png", bbox_inches="tight")
    
    plt.close(fig)
    
    pyks = np.load(pyksdir / f"{pid}.npz")
    a = pyks["amps"]
    z = pyks["z"]
    t = pyks["t"]
    which = np.isfinite(z)
    a = a[which]
    z = z[which]
    t = t[which]
    
    r1, *_ = ibme.fast_raster(a, z, t)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(np.clip(r1, 0, 13), aspect=r0.shape[1] / r0.shape[0], cmap=plt.cm.cubehelix, origin="lower")
    plt.colorbar(shrink=0.3)
    plt.title(f"{pid}  |  pyks", fontsize=10)
    plt.xlabel("t (s)", )
    plt.ylabel("depth (um)", )
    plt.tick_params(labelsize=8)
    fig.savefig(todir / f"{pid}_c_pyks.png", bbox_inches="tight")
    
    plt.close(fig)


# %%
1


# %%
from scipy.stats import norm

# %%
z = norm.pdf(np.arange(3840), loc=1200, scale=0.5*3840/10)

# %%
(z > 1e-6).mean()

# %%
for dataset in tqdm(sorted(data_dir.glob("*"))):
    # if not (dataset / "dispmap.npy").exists():
    #     continue
    # if dataset.stem not in testcases:
    #     continue
        
    
    # dispmap = np.load(dataset / "dispmap.npy")
    
    maxptp = np.load(dataset / "maxptps.npy")
    which = maxptp > 6
    # which = slice(None)
    x, y, z, a = np.load(dataset / "localizations.npy")[which, :4].T
    maxptp = maxptp[which]
    # dat = np.load(savedir / f"{dataset.stem}.npz")
    # dat = dict(dat)
    # dat["z"] = z
    # dat["maxptp"] = maxptp[which]
    # np.savez(savedir / f"{dataset.stem}.npz", **dat)
    # z_reg0 = np.load(dataset / "z_reg.npy")[which]
    t = np.load(dataset / "spike_index.npy")[which, 0] / 30000
    z_reg1, dispmap = ibme.register_nonrigid(
        maxptp, z, t, n_windows=10, disp=500, batch_size=8, robust_sigma=1, widthmul=0.5, rigid_init=False
    )
    # r1, *_ = ibme.fast_raster(maxptp, z, t)
    # fig = plt.figure(figsize=(8,6))
    # plt.imshow(np.clip(r1, 3, 13))
    # plt.title(f"{dataset.stem}  |  unregistered localizations")
    # plt.xlabel("time (s)")
    # plt.ylabel("depth")
    # plt.show()
    # plt.close(fig)
    
    
    np.savez(savedir / f"{dataset.stem}.npz", z=z, x=x, y=y,  alpha=a, maxptp=maxptp, z_reg=z_reg1, dispmap=dispmap, t=t)
    
    # fig, (aa, ab, ac) = plt.subplots(1, 3, figsize=(10, 8))
    # aa.imshow(np.clip(r0, 0, 10), aspect=2 * r0.shape[1] / r0.shape[0])
    # aa.set_title("original")
    # im = ab.imshow(dispmap, aspect=2 * r0.shape[1] / r0.shape[0])
    # ab.set_title("disp map")
    # plt.colorbar(im, ax=ab, shrink=0.25)
    # ac.imshow(np.clip(r1, 0, 10), aspect=2 * r0.shape[1] / r0.shape[0])
    # ac.set_title("registered")
    # ab.set_yticks([])
    # ac.set_yticks([])
    # fig.suptitle(dataset.stem + " original", fontsize=8, y=0.85)
    # fig.savefig(fig_dir / f"{dataset.stem}_00_original.png", bbox_inches="tight")
    # plt.close(fig)
    
    

# %%

# %%

# %%
for dataset in sorted(data_dir.glob("*")):
    if not (dataset / "dispmap.npy").exists():
        continue
    if dataset.stem not in testcases:
        continue
        
    
    dispmap = np.load(dataset / "dispmap.npy")
    
    maxptp = np.load(dataset / "maxptps.npy")
    which = maxptp > 6
    x, y, z, a = np.load(dataset / "localizations.npy")[which, :4].T
    z_reg0 = np.load(dataset / "z_reg.npy")[which]
    t = np.load(dataset / "spike_index.npy")[which, 0] / 30000
    
    
            z_reg1, dispmap = ibme.register_nonrigid(maxptp, z, t, n_windows=wins, disp=500, batch_size=8, robust_sigma=zrob, widthmul=w, rigid_init=r, corr_threshold=mincorr)
            r1, *_ = ibme.fast_raster(maxptp, z_reg1, t)
    
    
    r0, *_ = ibme.fast_raster(maxptp, z, t)
    r1, *_ = ibme.fast_raster(maxptp, z_reg0, t)
    
    
    fig, (aa, ab, ac) = plt.subplots(1, 3, figsize=(10, 8))
    aa.imshow(np.clip(r0, 0, 10), aspect=2 * r0.shape[1] / r0.shape[0])
    aa.set_title("original")
    im = ab.imshow(dispmap, aspect=2 * r0.shape[1] / r0.shape[0])
    ab.set_title("disp map")
    plt.colorbar(im, ax=ab, shrink=0.25)
    ac.imshow(np.clip(r1, 0, 10), aspect=2 * r0.shape[1] / r0.shape[0])
    ac.set_title("registered")
    ab.set_yticks([])
    ac.set_yticks([])
    fig.suptitle(dataset.stem + " original", fontsize=8, y=0.85)
    fig.savefig(fig_dir / f"{dataset.stem}_00_original.png", bbox_inches="tight")
    plt.close(fig)

# %%
kwargs = [
    # dict(robust_sigma=0., widthmul=0.5, rigid_init=False),
    # dict(robust_sigma=1., widthmul=0.5, rigid_init=False),
    # dict(robust_sigma=1., widthmul=0.25, rigid_init=False),
    dict(robust_sigma=0.5, corr_threshold=0, widthmul=1, rigid_init=False),
]

# %%
windows = [[], [2], [5], [10], [5, 10], [5, 20], [5, 40]]
windows = [[10], [20]]

# %%
import gc

# %%
for wins in windows:
    for dataset in sorted(data_dir.glob("*")):
        if not (dataset / "dispmap.npy").exists():
            continue
        if dataset.stem not in testcases:
            continue
        
        
    
        maxptp = np.load(dataset / "maxptps.npy")
        which = maxptp > 6
        maxptp = maxptp[which]
        x, y, z, a = np.load(dataset / "localizations.npy")[which, :4].T
        t = np.load(dataset / "spike_index.npy")[which, 0] / 30000
            
        r0, *_ = ibme.fast_raster(maxptp, z, t)

        for kw in kwargs:
            zrob = kw["robust_sigma"]
            w = kw["widthmul"]
            r = kw["rigid_init"]
            mincorr = kw["corr_threshold"]
            if not wins:
                r = True
            
            win_str = "-".join(map(str, wins))
            if not win_str:
                win_str = "0"
            rig_str = "y" if r else "n"
            title = f"{dataset.stem} | windows: {win_str}, z={zrob}, w={w}, init rigid = {rig_str}, mincorr={mincorr}"
            filename = f"{dataset.stem}_02_wins{win_str}_z={zrob:01.1f}_{w:01.2f}_r{rig_str}_mc{mincorr:01.2f}.png"
            print(title)
            print(filename)

            z_reg1, dispmap = ibme.register_nonrigid(maxptp, z, t, n_windows=wins, disp=500, batch_size=8, robust_sigma=zrob, widthmul=w, rigid_init=r, corr_threshold=mincorr)
            r1, *_ = ibme.fast_raster(maxptp, z_reg1, t)
            
    
    
            fig, (aa, ab, ac) = plt.subplots(1, 3, figsize=(10, 8))
            aa.imshow(np.clip(r0, 0, 10), aspect=2 * r0.shape[1] / r0.shape[0])
            aa.set_title("original")
            im = ab.imshow(dispmap, aspect=2 * r0.shape[1] / r0.shape[0])
            ab.set_title("disp map")
            plt.colorbar(im, ax=ab, shrink=0.25)
            ac.imshow(np.clip(r1, 0, 10), aspect=2 * r0.shape[1] / r0.shape[0])
            ac.set_title("registered")
            ab.set_yticks([])
            ac.set_yticks([])
            fig.suptitle(title, fontsize=8, y=0.85)
            fig.savefig(fig_dir / filename, bbox_inches="tight")
            plt.close(fig)
        
        gc.collect()


# %%

# %%

# %%
# pid = "e31b4e39-e350-47a9-aca4-72496d99ff2a"
pid = "ce397420-3cd2-4a55-8fd1-5e28321981f4"
npz = savedir / f"{pid}.npz"
with np.load(npz) as f:
    print(list(f.keys()))
    x = f["x"]
    y = f["y"] 
    z = f["z"]
    alpha = f["alpha"]
    maxptp = f["maxptp"]
    z_reg = f["z_reg"]
    dispmap = f["dispmap"]
    t = f["t"]

# %%
x.shape, z_reg.shape, maxptp.min()

# %%
r0, dd0, tt0 = ibme.fast_raster(maxptp, z, t)
r1, dd1, tt1 = ibme.fast_raster(maxptp, z_reg, t)

# %%
dd0.min(), dd0.max(), tt0.min(), tt0.max()

# %%
dd1.min(), dd1.max(), tt1.min(), tt1.max()

# %%
plt.rc("figure", dpi=300)

# %%
# inset axes....
axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
axins.imshow(Z2, extent=extent, origin="lower")
# sub region of the original image
x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])

# %%
e0 = [tt0.min(), tt0.max(), dd0.min(), dd0.max()]
e1 = [tt1.min(), tt1.max(), dd1.min(), dd1.max()]

# %%

mosaic = """\
a.b.
a.bc
a.b.
"""
fig, axes = plt.subplot_mosaic(mosaic, figsize=(6, 4), gridspec_kw=dict(width_ratios=[1,0,1, 0.05], height_ratios=[1, 2, 1], wspace=0.075, hspace=0))
aa = axes["a"]
ab = axes["b"]
ac = axes["c"]
aa.get_shared_y_axes().join(aa, ab)

aa.imshow(np.clip(r0, 0, 13), extent=e0, aspect=r0.shape[1] / r0.shape[0], origin="lower", cmap=plt.cm.cubehelix)
im = ab.imshow(np.clip(r1, 0, 13), extent=e1, aspect=r1.shape[1] / r1.shape[0], origin="lower", cmap=plt.cm.cubehelix)
ab.set_yticks([])
aa.set_ylim([0, 3840])
aa.set_yticks([0, 1000, 2000, 3000, 3840])
aa.set_ylabel("depth (um)", fontsize=8)
plt.colorbar(im, cax=ac)

aa.set_xticks([0, 1000, 3000, 4000])
ab.set_xticks([0, 1000, 3000, 4000])
aa.set_xlabel("time (s)", fontsize=8, labelpad=-7.5)
ab.set_xlabel("time (s)", fontsize=8, labelpad=-7.5)

ac.set_yticks([0, 5, 10])
ac.set_ylabel("amplitude (s.u.)", fontsize=8)

aa.set_title("unregistered", fontsize=8)
ab.set_title("registered", fontsize=8)

for ax, r, ext in zip([aa, ab], [r0, r1], [e0, e1]):
    ai = ax.inset_axes([0.17, 0.13, 0.37, 0.84])
    ai.imshow(np.clip(r, 0, 13), extent=ext, origin="lower", cmap=plt.cm.cubehelix)
    ai.set_xlim([2500, 3000])
    ai.set_ylim([1000, 2000])
    ai.set_xticks([2500, 3000], fontsize=6)
    ai.set_yticks([1000, 2000], fontsize=6)
    ai.tick_params(color='white', labelcolor='white')
    ai.tick_params(axis='both', which='major', labelsize=6)
    for spine in ai.spines.values():
        spine.set_edgecolor('white')

for ax in axes.values():
    ax.tick_params(axis='both', which='major', labelsize=8)
    
gridspec.GridSpec(1, 2).update(wspace=0)

fig.suptitle(f"{pid}", y=0.84, fontsize=8)
fig.savefig("/mnt/3TB/charlie/figs/motion_ce39.pdf", bbox_inches="tight", dpi=200)
# plt.show()

# %%

# %%
pid = "e31b4e39-e350-47a9-aca4-72496d99ff2a"
# pid = "ce397420-3cd2-4a55-8fd1-5e28321981f4"
npz = savedir / f"{pid}.npz"
with np.load(npz) as f:
    print(list(f.keys()))
    x = f["x"]
    y = f["y"] 
    z = f["z"]
    alpha = f["alpha"]
    maxptp = f["maxptp"]
    z_reg = f["z_reg"]
    dispmap = f["dispmap"]
    t = f["t"]

# %%
r0, dd0, tt0 = ibme.fast_raster(maxptp, z, t)
e0 = [tt0.min(), tt0.max(), dd0.min(), dd0.max()]

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.imshow(np.clip(r0, 0, 13), extent=e0, aspect=0.4 * r0.shape[1] / r0.shape[0], origin="lower", cmap=plt.cm.cubehelix)


ai = ax.inset_axes([0.4, 0.13, 0.37, 0.84])
ai.imshow(np.clip(r0, 0, 13), extent=e0, origin="lower", cmap=plt.cm.cubehelix)
ai.set_xlim([1500, 2000])
ai.set_ylim([1000, 2000])
ai.set_xticks([1500, 2000], fontsize=6)
ai.set_yticks([1000, 2000], fontsize=6)
ai.tick_params(color='white', labelcolor='white')
ai.tick_params(axis='both', which='major', labelsize=6)
for spine in ai.spines.values():
    spine.set_edgecolor('white')


ax.set_ylim([0, 3840])
ax.set_yticks([0, 1000, 2000, 3000, 3840])
ax.set_ylabel("depth (um)", fontsize=8)
cbar = plt.colorbar(im, ax=ax, shrink=0.3)
cbar.ax.set_ylabel("amplitude (s.u.)", fontsize=8)

ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
ax.set_xlabel("time (s)", fontsize=8)
ax.tick_params(axis='both', which='major', labelsize=8)

plt.title(pid, fontsize=8)

fig.savefig("/mnt/3TB/charlie/figs/nonstationary_e31.pdf", bbox_inches="tight", dpi=200)
# plt.show()

# %%

# %%

# %%

# %%
# %ll /mnt/3TB/charlie/steps_5min/

# %%
# with h5py.File("/mnt/3TB/charlie/steps_5min/nosub_justnn_nolb/subtraction_CSH_ZAD_026_snip.ap_t_0_None.h5") as f:
with h5py.File("/mnt/3TB/charlie/steps_1min/nosub_justnn_nolb/subtraction_CSH_ZAD_026_snip.ap_t_0_None.h5") as f:
    ax, ay, azr, aza, aa = f["localizations"][:].T
    geom = f["geom"][:]
    az_reg = f["z_reg"][:]
    amaxptps = f["maxptps"][:]
    at = f["spike_index"][:, 0] / 30000

# %%
with h5py.File("/mnt/3TB/charlie/steps_1min/nolb/subtraction_CSH_ZAD_026_snip.ap_t_0_None.h5") as f:
    bx, by, bzr, bza, ba = f["localizations"][:].T
    bz_reg = f["z_reg"][:]
    bmaxptps = f["maxptps"][:]    
    bt = f["spike_index"][:, 0] / 30000    

# %%
with h5py.File("/mnt/3TB/charlie/steps_1min/yeslb/subtraction_CSH_ZAD_026_snip.ap_t_0_None.h5") as f:
    cx, cy, czr, cza, ca = f["localizations"][:].T
    cz_reg = f["z_reg"][:]
    cmaxptps = f["maxptps"][:]
    ct = f["spike_index"][:, 0] / 30000    

# %%
len(amaxptps), len(bmaxptps), len(cmaxptps)
vmin = min(amaxptps.min(), bmaxptps.min(), cmaxptps.min())
vmax = max(amaxptps.max(), bmaxptps.max(), cmaxptps.max())
vmin, vmax

# %%
mos = """\
ab.cd.ef..
ab.cd.ef.g
ab.cd.ef..
"""
fig, axes = plt.subplot_mosaic(
    mos,
    figsize=(6, 4),
    # sharey=True,
    gridspec_kw=dict(
        width_ratios=[1, 1, 0.1, 1, 1, 0.1, 1, 1, 0.05, 0.1],
        wspace=0.05
    )
)

for (aa, ab, x, y, z, maxptp, t) in zip(
    [axes["a"], axes["c"], axes["e"]],
    [axes["b"], axes["d"], axes["f"]],
    (ax, bx, cx),
    (ay, by, cy),
    (az_reg, bz_reg, cz_reg),
    # (aza, bza, cza),
    (amaxptps, bmaxptps, cmaxptps),
    (at, bt, ct),
):
    nmaxptp = (maxptp - vmin) / (vmax - vmin)
    alpha = 0.1 + 0.3 * nmaxptp
    
    which = np.flatnonzero((500 < z) & (z < 1000) & (t < 60))
    # which = slice(None)
    
    aa.scatter(x[which], z[which], c=nmaxptp[which], alpha=alpha[which], marker=".", s=1)
    ab.scatter(np.log(y[which]), z[which], c=nmaxptp[which], alpha=alpha[which], marker=".", s=1)
    aa.scatter(*geom.T, color="orange", marker="s", s=2)
    # ab.scatter(np.zeros_like(geom[:, 0]), geom[:, 1], color="orange", marker="s")
    
    aa.set_ylim([500, 1000])
    ab.set_ylim([500, 1000])
    aa.set_xlim([geom[:, 0].min() - 35, geom[:, 0].max() + 30])

cbar = plt.colorbar(
    plt.cm.ScalarMappable(
        mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        cmap=plt.cm.viridis,
    ),
    cax=axes["g"],
    # shrink=0.5,
)
axes["g"].set_ylabel("denoised PTP", fontsize=6)
axes["g"].set_yticks([10, 25, 50])

ext = {}
for k, v in axes.items():
    if k in "bdf":
        v.set_yticks([])
        v.set_xlabel(r"$\log y$", fontsize=8)
    else:
        v.set_xlabel(r"$x$ (um)", fontsize=8)
    if k in "ce":
        v.set_yticklabels([""] * len(v.get_yticks()))
    ext[k] = [v.get_window_extent().x0, v.get_window_extent().width]
    v.tick_params(axis='both', which='major', labelsize=8)

axes["a"].set_ylabel("registered $z$ (um)", fontsize=8, labelpad=-3)

fig.subplots_adjust(top=0.91)   

inv = fig.transFigure.inverted()
for group, title in zip(
    ("ab", "cd", "ef"),
    ("voltage det., NN denoising,\nno log barrier", "subtraction, full denoising,\nno log barrier", "subtraction, full denoising,\nlog barrier"),
):
    ka, kb = group
    width = ext[ka][0]+(ext[kb][0]+ext[kb][1]-ext[ka][0])/2.
    center = inv.transform((width, 1))
    plt.figtext(center[0],0.95, title, va="center", ha="center", size=8)


# %%
mos = """\
ab.ef..
ab.ef.g
ab.ef..
"""
fig, axes = plt.subplot_mosaic(
    mos,
    figsize=(6, 4),
    # sharey=True,
    gridspec_kw=dict(
        width_ratios=[1, 1, 0.1, 1, 1, 0.05, 0.1],
        wspace=0.05
    ),
    dpi=300,
)

for (aa, ab, x, y, z, maxptp, t) in zip(
    [axes["a"], axes["e"]],
    [axes["b"], axes["f"]],
    (ax, cx),
    (ay, cy),
    (az_reg, cz_reg),
    # (aza, bza, cza),
    (amaxptps, cmaxptps),
    (at, ct),
):
    nmaxptp = (maxptp - vmin) / (vmax - vmin)
    alpha = 0.1 + 0.3 * nmaxptp
    
    which = np.flatnonzero((500 < z) & (z < 1000) & (t < 60))
    print(len(which))
    # which = slice(None)
    
    aa.scatter(x[which], z[which], c=nmaxptp[which], alpha=alpha[which], marker=".", s=1, rasterized=True)
    ab.scatter(np.log(y[which]), z[which], c=nmaxptp[which], alpha=alpha[which], marker=".", s=1, rasterized=True)
    aa.scatter(*geom.T, color="orange", marker="s", s=2)
    # ab.scatter(np.zeros_like(geom[:, 0]), geom[:, 1], color="orange", marker="s")
    
    aa.set_ylim([500, 1000])
    ab.set_ylim([500, 1000])
    aa.set_xlim([geom[:, 0].min() - 35, geom[:, 0].max() + 30])

cbar = plt.colorbar(
    plt.cm.ScalarMappable(
        mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        cmap=plt.cm.viridis,
    ),
    cax=axes["g"],
    # shrink=0.5,
)
axes["g"].set_ylabel("denoised PTP", fontsize=8)
axes["g"].set_yticks([10, 30, 50])

ext = {}
for k, v in axes.items():
    if k in "bf":
        v.set_yticks([])
        v.set_xlabel(r"$\log y$", fontsize=8)
    else:
        v.set_xlabel(r"$x$ (um)", fontsize=8)
    if k in "e":
        v.set_yticklabels([""] * len(v.get_yticks()))
    ext[k] = [v.get_window_extent().x0, v.get_window_extent().width]
    v.tick_params(axis='both', which='major', labelsize=8)

axes["a"].set_ylabel("registered $z$ (um)", fontsize=8, labelpad=-3)

fig.subplots_adjust(top=0.91)   

inv = fig.transFigure.inverted()
for group, title in zip(
    ("ab", "ef"),
    ("voltage det., NN denoising,\nno log barrier", "subtraction, full denoising,\nlog barrier"),
):
    ka, kb = group
    width = ext[ka][0]+(ext[kb][0]+ext[kb][1]-ext[ka][0])/2.
    center = inv.transform((width, 1))
    plt.figtext(center[0],0.95, title, va="center", ha="center", size=8)

fig.savefig("/mnt/3TB/charlie/figs/old_and_new_pipeline.pdf", dpi=300, bbox_inches="tight")

# %%
