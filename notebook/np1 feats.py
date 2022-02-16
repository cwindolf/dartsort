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
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import zscore

# %%
rg = lambda: np.random.default_rng(0)
plt.rc("figure", dpi=200)

# %%
from spike_psvae import denoise, featurize, localization, point_source_centering, vis_utils

# %%
from npx import reg

# %%
# %ll -h /mnt/3TB/charlie/subtracted_datasets/

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
root = Path("/mnt/3TB/charlie/subtracted_datasets/")

# %%
standardwfs = {}
firstchans = {}
maxchans = {}

for ds in root.glob("*.h5"):
    print(ds.stem)
    with h5py.File(ds, "r") as f:
        cleaned = denoise.cleaned_waveforms(
            f["subtracted_waveforms"],
            f["spike_index"][:],
            f["first_channels"][:],
            f["residual"],
            s_start=f["start_sample"][()]
        )
        stdwfs, firstchans_std, maxchans_std, chans_down = featurize.relativize_waveforms(
            cleaned,
            f["first_channels"][:],
            None,
            f["geom"][:],
            feat_chans=18,
        )
        standardwfs[ds.stem] = stdwfs
        firstchans[ds.stem] = firstchans_std
        maxchans[ds.stem] = maxchans_std
        print(chans_down)


# %%
for ds in root.glob("*.h5"):
    with h5py.File(ds, "r") as f:
        show = rg().choice(f["spike_index"].shape[0], size=8, replace=False)
        fig, axes = plt.subplots(4, 2)
        vis_utils.plot_ptp(standardwfs[ds.stem][show].ptp(1), axes, "", "k", "abcdefgh")
        
        for j in show:
            plt.figure(figsize = (20, 2.5))
            dfc = firstchans[ds.stem][j] - f["first_channels"][j]
            plt.plot(f["subtracted_waveforms"][j][:82, dfc:dfc + 24].T.flatten(), 'blue')
            plt.plot(standardwfs[ds.stem][j, :82].T.flatten(), 'orange')
            for j in range(19):
                plt.axvline(82 + 82*j, color = 'black')
            plt.show() 

# %%
locs = {}
for ds in root.glob("*.h5"):
    with h5py.File(ds, "r") as f:
        maxptps = standardwfs[ds.stem].ptp(1).ptp(1)
        show = rg().choice(np.flatnonzero(maxptps > 6), size=8, replace=False)
        
        locs = localization.localize_waveforms(
            standardwfs[ds.stem][show],
            f["geom"][:],
            maxchans[ds.stem][show],
            channel_radius=chans_down,
            n_workers=1,
            firstchans=firstchans[ds.stem][show],
            geomkind="firstchanstandard",
            logbarrier=True,
        )
        print(locs[1])
        locs = list(zip(*locs))
        print(locs)
        
        fig, axes = plt.subplots(4, 2)
        vis_utils.plot_ptp(standardwfs[ds.stem][show].ptp(1), axes, "", "k", "abcdefgh")
        geom = f["geom"][:]
        lgeoms = np.array([
            geom[firstchans[ds.stem][j] : firstchans[ds.stem][j] + standardwfs[ds.stem].shape[2]]
            for j in show
        ])
        for i, j in enumerate(show):
            lgeoms[i, :, 1] -= geom[maxchans[ds.stem][j], 1]
            
        predptps = [
            localization.ptp_at(
                loc[0], loc[1], loc[2], loc[4], lgeoms[i]
            )
            for i, loc in enumerate(locs)
        ]
        vis_utils.plot_ptp(np.array(predptps), axes, "", "silver", "abcdefgh")        

# %%
locs = {}
for ds in root.glob("*.h5"):
    with h5py.File(ds, "r") as f:
        x, y, zr, za, a = localization.localize_waveforms_batched(
            standardwfs[ds.stem],
            f["geom"][:],
            maxchans[ds.stem],
            channel_radius=chans_down,
            n_workers=15,
            firstchans=firstchans[ds.stem],
            geomkind="firstchanstandard",
            # logbarrier=True,
        )
        locs[ds.stem] = np.c_[x, y, zr, za, a]
        np.save(f"../data/{ds.stem}_locs.npy", locs[ds.stem])

# %% tags=[]
zregs = {}
for ds in root.glob("*.h5"):
    with h5py.File(ds, "r") as f:
        maxptps = standardwfs[ds.stem].ptp(1).ptp(1).astype(float)
        
        # z_rigid_reg, p_rigid = reg.register_rigid(
        #     maxptps,
        #     locs[ds.stem][:, 3],
        #     (f["spike_index"][:, 0] - f["start_sample"][()]) / 30000,
        #     robust_sigma=0,
        #     disp=400,
        #     denoise_sigma=0.1,
        #     destripe=False,
        # )
        z_reg, dispmap = reg.register_nonrigid(
            maxptps,
            locs[ds.stem][:, 3],
            (f["spike_index"][:, 0] - f["start_sample"][()]) / 30000,
            robust_sigma=1,
            rigid_disp=200,
            disp=100,
            denoise_sigma=0.1,
            destripe=False,
            n_windows=[5, 30, 60],
            n_iter=1,
            widthmul=0.25,
        )
        zregs[ds.stem] = z_reg
        np.save(f"../data/{ds.stem}_zreg.npy", locs[ds.stem])

# %%
feats = {}
for ds in root.glob("*.h5"):
    with h5py.File(ds, "r") as f:
        x, y, z_rel, z_abs, alpha = locs[ds.stem].T
        stdwfs_xyza, xyza_target_ptp, original_ptp = point_source_centering.relocate_simple(
            standardwfs[ds.stem],
            f["geom"][:],
            maxchans[ds.stem],
            x, y, z_rel, alpha,
            firstchans=firstchans[ds.stem],
            relocate_dims="xyza",
            geomkind="firstchanstandard",
            channel_radius=chans_down,
        )
        # those are torch but we want numpy
        stdwfs_xyza = stdwfs_xyza.cpu().numpy()
        xyza_target_ptp = xyza_target_ptp.cpu().numpy()
        original_ptp = original_ptp.cpu().numpy()

        ae_feats_xyza, err = featurize.pca_reload(
            standardwfs[ds.stem], stdwfs_xyza, original_ptp, xyza_target_ptp, rank=10, B_updates=2
        )
        feats[ds.stem] = ae_feats_xyza
        np.save(f"../data/{ds.stem}_ae_feats.npy", ae_feats_xyza)

# %%

# %%
for ds in root.glob("*.h5"):
    with h5py.File(ds, "r") as f:
        maxptps = standardwfs[ds.stem].ptp(1).ptp(1).astype(float)
        geom = f["geom"][:]

        fig, (aa, ab, ac, ad, ae, af) = plt.subplots(1, 6, sharey=True, figsize=(16, 8))
        nmaxptps = 0.1 + 0.9 * (maxptps - maxptps.min()) / (maxptps.max() - maxptps.min())
        # nmaxptps = np.ones_like(nmaxptps)
        # (1900, 2600)
        
        x, y, z_rel, z_abs, alpha = locs[ds.stem].T
        z_reg = zregs[ds.stem]
        

        inrange = (1900 < z_reg) & (z_reg < 2600)
        cm = plt.cm.viridis
        aes = feats[ds.stem]


        aa.scatter(x[:], z_reg[:], s=0.1, alpha=nmaxptps[:], c=maxptps[:], cmap=cm)
        aa.scatter(geom[:, 0], geom[:, 1], color="orange", s=1)
        ab.scatter(np.log(y[:]), z_reg[:], s=0.1, alpha=nmaxptps[:], c=maxptps[:], cmap=cm)
        ac.scatter(np.log(alpha[:]), z_reg[:], s=0.1, alpha=nmaxptps[:], c=maxptps[:], cmap=cm)
        ad.scatter(zscore(aes[:, 0]), z_reg[:], s=0.1, alpha=nmaxptps[:], c=maxptps[:], cmap=cm)
        ae.scatter(zscore(aes[:, 1]), z_reg[:], s=0.1, alpha=nmaxptps[:], c=maxptps[:], cmap=cm)
        af.scatter(zscore(aes[:, 2]), z_reg[:], s=0.1, alpha=nmaxptps[:], c=maxptps[:], cmap=cm)
        aa.set_ylabel("z")
        aa.set_xlabel("x")
        ab.set_xlabel("$\\log y$")
        ac.set_xlabel("$\\log \\alpha$")
        ad.set_xlabel("ae1")
        ae.set_xlabel("ae2")
        af.set_xlabel("ae3")
        aa.set_xlim([11 - 50, 59 + 50])
        ab.set_xlim([-1, 5])
        ac.set_xlim([2.5, 6.1])
        aa.set_ylim([0 - 10, geom[:, 1].max() + 10])
        fig.suptitle(ds.stem, y=0.925)

# %%
# %ll -h /mnt/3TB/charlie/ibl_feats

# %% [markdown]
# for ds in root.glob("*.h5"):
#     with h5py.File(ds, "r") as f:
#         maxptps = standardwfs[ds.stem].ptp(1).ptp(1).astype(float)
#         geom = f["geom"][:]
#         x, y, z_rel, z_abs, alpha = locs[ds.stem].T
#         z_reg = zregs[ds.stem]
#         np.savez(
#             f"/mnt/3TB/charlie/ibl_feats/{ds.stem}.npz",
#             locs=np.c_[x, y, z_reg, alpha],
#             maxptps=maxptps,
#             times=f["spike_index"][:, 0] - f["start_sample"][()],
#             feats=feats[ds.stem],
#         )

# %%
