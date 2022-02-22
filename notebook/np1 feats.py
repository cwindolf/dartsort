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
import colorcet as cc

# %%
rg = lambda: np.random.default_rng(0)
plt.rc("figure", dpi=200)

# %%
from spike_psvae import denoise, featurize, localization, point_source_centering, vis_utils, triage, waveform_utils


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
    if "p7_t_2000" not in ds.stem:
        print("bye")
        continue
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
    print(ds.stem)
    if "p7_t_2000" not in ds.stem:
        print("bye")
        continue
    with h5py.File(ds, "r") as f:
        wfs = f["subtracted_waveforms"]
        print(wfs.shape)
        show = rg().choice(f["spike_index"].shape[0], size=16, replace=False)
        show.sort()
        cwfs = standardwfs[ds.stem]
        fcs = f["first_channels"][:]
        cfcs = firstchans[ds.stem]
        mcs = f["spike_index"][:, 1]
        cmcs = maxchans[ds.stem]
        print((cfcs - fcs).min(), (cfcs - fcs).max())
        print((cmcs - mcs).min(), (cmcs - mcs).max())
        

        fig, axes = plt.subplots(4, 4)
        vis_utils.plot_ptp(wfs[show].ptp(1), axes, "", "k", "abcdefghijklmnop")
        crelptps = []
        for ix in show:
            fcrel = cfcs[ix] - fcs[ix]
            print(fcrel, cfcs[ix], fcs[ix])
            crelptps.append(np.pad(cwfs[ix].ptp(0), (fcrel, 22 - fcrel)))
        vis_utils.plot_ptp(crelptps, axes, "", "purple", "abcdefghijklmnop")

# %%
for ds in root.glob("*.h5"):
    if "p7_t_2000" not in ds.stem:
        print("bye")
        continue
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
from scipy.optimize import minimize


# %%
def cost(xy):
    return np.square(xy).sum()
print(
    minimize(cost, x0=[10, 10], bounds=[(-100, 100), (-100, 100)])
)


# %%

# %%

# %%

def ptp_at(x, y, z, alpha, local_geom):
    return alpha / np.sqrt(
        np.square(x - local_geom[:, 0])
        + np.square(z - local_geom[:, 1])
        + np.square(y)
    )


# %% tags=[]
locs = {}
for ds in root.glob("*.h5"):
    if "p7_t_2000" not in ds.stem:
        print("bye")
        continue
    with h5py.File(ds, "r") as f:
        geom = f["geom"][:]
        maxptps = standardwfs[ds.stem].ptp(1).ptp(1)
        plt.figure()
        plt.hist(maxptps, bins=100)
        plt.show()
        
        show = rg().choice(maxptps.shape[0], size=8, replace=False)
        
        locs = localization.localize_waveforms(
            standardwfs[ds.stem][show],
            f["geom"][:],
            firstchans[ds.stem][show],
            maxchans[ds.stem][show],
            n_workers=1,
        )
        print(locs[2].mean(), locs[2].std())
        locs = np.array(list(zip(*locs)))
        
        gridlocs = np.zeros((8, 5))
        ugridlocs = np.zeros((8, 5))
        for ind, (ptp, mc, fc) in enumerate(zip(standardwfs[ds.stem][show].ptp(1), maxchans[ds.stem][show], firstchans[ds.stem][show])):
            print("-" * 80)
            local_geom, z_maxchan = waveform_utils.get_local_geom(
                geom,
                fc,
                mc,
                ptp.shape[0],
                return_z_maxchan=True,
            )

            ptp = ptp.astype(float)
            local_geom = local_geom.astype(float)
            ptp_p = ptp / ptp.sum()
            xcom, zcom = (ptp_p[:, None] * local_geom).sum(axis=0)
            maxptp = ptp.max()
            
            def penalty(y):
                return -np.log1p(y * 6) / 10000

            def mse(x, y, z, pen=1):
                q = ptp_at(x, y, z, 1., local_geom)
                alpha = (q * ptp / maxptp).sum() / (q*q).sum()
                return np.square(ptp / maxptp - ptp_at(x, y, z, alpha, local_geom)).mean() + pen * penalty(y)

            def residual(loc):
                x, y, z = loc
                return mse(x, y, z)

            def unpenresidual(loc):
                x, y, z = loc
                return mse(x, y, z, pen=0)

            result = minimize(
                residual,
                x0=[xcom, 21., zcom],
                bounds=[(-100, 132), (1e-4, 250), (-100, 100)]
            )

            print(result)
            bx, by, bz_rel = result.x
            q = ptp_at(bx, by, bz_rel, 1., local_geom)
            balpha = (ptp * q).sum() / np.square(q).sum()
            gridlocs[ind, 0] = bx
            gridlocs[ind, 1] = by
            gridlocs[ind, 2] = bz_rel
            gridlocs[ind, 4] = balpha
            print(by, locs[ind, 1], "hi")

            upres = minimize(
                unpenresidual,
                x0=[xcom, 21., zcom],
                bounds=[(-100, 132), (1e-4, 250), (-100, 100)]
            )
            print(upres)
            ubx, uby, ubz_rel = upres.x
            q = ptp_at(ubx, uby, ubz_rel, 1., local_geom)
            ubalpha = (ptp * q).sum() / np.square(q).sum()
            ugridlocs[ind, 0] = ubx
            ugridlocs[ind, 1] = uby
            ugridlocs[ind, 2] = ubz_rel
            ugridlocs[ind, 4] = ubalpha

            ys = np.linspace(1e-4, 100, num=151)
            xs = np.linspace(-100, 132, num=24)
            zs = np.linspace(-100, 100, num=26)

            xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")

            xzunpen = np.zeros((24, 26))
            for i in range(24):
                for k in range(26):
                    xzunpen[i, k] = mse(xs[i], 21, zs[k], pen=0)

            xyunpen = np.zeros((24, 151))
            for i in range(24):
                for j in range(151):
                    xyunpen[i, j] = mse(xs[i], ys[j], zcom, pen=0)
            zyunpen = np.zeros((26, 151))
            for i in range(26):
                for j in range(151):
                    zyunpen[i, j] = mse(xcom, ys[j], zs[i], pen=0)

            xymse = np.zeros((24, 151))
            for i in range(24):
                for j in range(151):
                    xymse[i, j] = mse(xs[i], ys[j], zcom)
            zymse = np.zeros((26, 151))
            for i in range(26):
                for j in range(151):
                    zymse[i, j] = mse(xcom, ys[j], zs[i])
            xyzmse = np.zeros((24, 151, 26))
            for i in range(24):
                for j in range(151):
                    for k in range(26):
                        xyzmse[i, j, k] = mse(xs[i], ys[j], zs[k])
            ii, jj, kk = np.unravel_index(np.argmin(xyzmse), xyzmse.shape)
            xyzunpen = np.zeros((24, 151, 26))
            for i in range(24):
                for j in range(151):
                    for k in range(26):
                        xyzunpen[i, j, k] = mse(xs[i], ys[j], zs[k], pen=0)
            iiu, jju, kku = np.unravel_index(np.argmin(xyzunpen), xyzunpen.shape)

            plt.figure()
            plt.pcolormesh(xs, zs, xzunpen.T)
            plt.scatter([xcom], [zcom], color="green", label="com")
            plt.scatter([xs[ii]], [zs[kk]], color="orange", label="xyz grid search (penalized)")
            plt.scatter([xs[iiu]], [zs[kku]], color="blue", label="xyz grid search (unpenalized)")
            plt.scatter([bx], [bz_rel], color="red", label="bfgs opt (penalized)")
            plt.scatter([ubx], [ubz_rel], color="purple", label="bfgs opt (unpenalized)")
            plt.xlabel("x")
            plt.ylabel("z")
            plt.legend()
            plt.show()

            fig, (aa, ab) = plt.subplots(1, 2)
            aa.pcolormesh(xs, ys, xyunpen.T)
            aa.scatter([xs[ii]], [ys[jj]], color="orange", label="grid search (pen)", marker="x")
            aa.scatter([xs[iiu]], [ys[jju]], color="blue", label="grid search (unpen)", marker="x")
            aa.scatter([bx], [by], color="red", label="bfgs (pen)", marker="x")
            aa.scatter([ubx], [uby], color="purple", label="bfgs (unpen)", marker="x")
            aa.set_title(f"unpenalized -- y={uby:0.3f}")

            ab.pcolormesh(xs, ys, xymse.T)
            ab.scatter([xs[ii]], [ys[jj]], color="orange", label="grid search (pen)", marker="x")
            ab.scatter([xs[iiu]], [ys[jju]], color="blue", label="grid search (unpen)", marker="x")
            ab.scatter([bx], [by], color="red", label="bfgs (pen)", marker="x")
            ab.scatter([ubx], [uby], color="purple", label="bfgs (unpen)", marker="x")

            ab.set_title(f"penalized -- y={by:0.3f}")

            aa.set_xlabel("x")
            ab.set_xlabel("x")
            aa.set_ylabel("y")
            ab.legend()
            plt.show()
            
            
            fig, (aa, ab) = plt.subplots(1, 2)
            aa.pcolormesh(zs, ys, zyunpen.T)
            aa.scatter([zs[kk]], [ys[jj]], color="orange", label="grid search (pen)", marker="x")
            aa.scatter([zs[kku]], [ys[jju]], color="blue", label="grid search (unpen)", marker="x")
            aa.scatter([bz_rel], [by], color="red", label="bfgs (pen)", marker="x")
            aa.scatter([ubz_rel], [uby], color="purple", label="bfgs (unpen)", marker="x")

            aa.set_title(f"unpenalized -- y={uby:0.3f}")

            ab.pcolormesh(zs, ys, zymse.T)
            ab.scatter([zs[kk]], [ys[jj]], color="orange", label="grid search (pen)", marker="x")
            ab.scatter([zs[kku]], [ys[jju]], color="blue", label="grid search (unpen)", marker="x")
            ab.scatter([bz_rel], [by], color="red", label="bfgs (pen)", marker="x")
            ab.scatter([ubz_rel], [uby], color="purple", label="bfgs (unpen)", marker="x")
            ab.set_title(f"penalized -- y={by:0.3f}")


            aa.set_xlabel("z")
            ab.set_xlabel("z")
            aa.set_ylabel("y")
            ab.legend()
            plt.show()

            plt.figure()
            plt.plot(ys, [mse(xcom, y, zcom) for y in ys], color="k", label="total cost")
            plt.plot(ys, [mse(xcom, y, zcom, pen=0) for y in ys], color="b", label="mse")
            plt.plot(ys, [penalty(y) for y in ys], color="g", label="penalty")
            plt.xlabel("y (xz=com)")
            plt.ylabel("cost")
            plt.semilogx()
            plt.legend()
            plt.show()
        
        print("max y diff", (gridlocs[:, 1] - locs[:, 1]).max())
            
        plt.figure()
        plt.hist(gridlocs[:, 1], bins=128)
        plt.show()
        
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
        vis_utils.plot_ptp(np.array(predptps), axes, "old bfgs", "silver", "abcdefgh")  
            
        predptps = [
            localization.ptp_at(
                loc[0], loc[1], loc[2], loc[4], lgeoms[i]
            )
            for i, loc in enumerate(gridlocs)
        ]
        vis_utils.plot_ptp(np.array(predptps), axes, "new bfgs (pen)", "red", "abcdefgh")    
            
        upredptps = [
            localization.ptp_at(
                loc[0], loc[1], loc[2], loc[4], lgeoms[i]
            )
            for i, loc in enumerate(ugridlocs)
        ]
        vis_utils.plot_ptp(np.array(upredptps), axes, "new bfgs (unpen)", "purple", "abcdefgh")
        axes.flat[-1].legend()
        plt.show()


# %%
def regline(x, y, ax=None, yx=True):
    b = ((x - x.mean()) * (y - y.mean())).sum() / np.square(x - x.mean()).sum()
    a = y.mean() - b * x.mean()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    r = np.corrcoef(x, y)[0, 1]
    if yx:
        lim = [min(x0, y0), max(x1, y1)]
        ax.plot(lim, lim, lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    else:
        ax.plot([x0, x1], [a + b * x0, a + b * x1], lw=1)
    ax.text(
        0.1,
        0.9,
        f"$\\rho={r:.2f}$",
        horizontalalignment="left",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize=6,
    )


# %% tags=[]
olocs = {}
locs = {}
for ds in root.glob("*.h5"):
    if "p7_t_2000" not in ds.stem:
        print("bye")
        continue
    with h5py.File(ds, "r") as f:
        maxptps = standardwfs[ds.stem].ptp(1).ptp(1).astype(float)
        show = rg().choice(np.flatnonzero(maxptps > 6), size=8, replace=False)
        show = slice(None)
        
        # ox, oy, ozr, oza, oa = localization.localize_waveforms(
        #     standardwfs[ds.stem][show],
        #     f["geom"][:],
        #     maxchans[ds.stem][show],
        #     channel_radius=chans_down,
        #     n_workers=15,
        #     firstchans=firstchans[ds.stem][show],
        #     geomkind="firstchanstandard",
        #     logbarrier=False,
        # )
        # olocs[ds.stem] = np.c_[ox, oy, ozr, oza, oa]
        # ox, oy, ozr, oza, oa = olocs[ds.stem].T
        
        x, y, zr, za, a = localization.localize_waveforms(
            standardwfs[ds.stem][show],
            f["geom"][:],
            firstchans[ds.stem][show],
            maxchans[ds.stem][show],
            n_workers=15,
        )
        print(y.min())
        print(zr.mean(), zr.std())
        locs[ds.stem] = np.c_[x, y, zr, za, a]
        # x, y, zr, za, a = locs[ds.stem].T
        z_reg = za
        
        nmaxptps = 0.25 + 0.74 * (maxptps - maxptps.min()) / (maxptps.max() - maxptps.min())
        geom = f["geom"][:]
        
        
        with np.load(f"/mnt/3TB/charlie/ibl_feats/{ds.stem}.npz") as npz:
            locsb = npz["locs"]
        print(np.abs(ox - locsb[:, 0]).max())
        print(np.abs(oy - locsb[:, 1]).max())    
            
        fig, (aa, ab, ac) = plt.subplots(1, 3, figsize=(10, 5))
        aa.scatter(ox, x, s=1, alpha=nmaxptps, c=maxptps, cmap=cm)
        aa.set_xlabel("prev x")
        aa.set_ylabel("x")
        regline(ox, x, ax=aa)
        ab.scatter(ozr, zr, s=1, alpha=nmaxptps, c=maxptps, cmap=cm)
        ab.set_xlabel("prev z (relative to max chan)")
        ab.set_ylabel("z")
        regline(ozr, zr, ax=ab)
        ac.scatter(oa, a, s=1, alpha=nmaxptps, c=maxptps, cmap=cm)
        ac.set_xlabel("prev alpha")
        ac.set_ylabel("alpha")
        regline(oa, a, ax=ac)
        ab.set_title(ds.stem)
        plt.show()
        
        plt.figure()
        plt.hist(y, bins=np.arange(50))
        plt.xlabel("y")
        plt.title(ds.stem)
        plt.show()
        
        plt.figure()
        plt.hist(y[y<5], bins=128)
        plt.xlabel("y")
        plt.title(ds.stem)
        plt.show()
        
        plt.figure()
        plt.hist(np.log(y), bins=np.arange(50))
        plt.xlabel("log y")
        plt.title(ds.stem)
        plt.show()

        # nmaxptps = np.ones_like(nmaxptps)
        # (1900, 2600)
        

        inrange = (1900 < z_reg) & (z_reg < 2600)
        cm = plt.cm.viridis

        fig, (aa, ab, ac) = plt.subplots(1, 3, sharey=True, figsize=(8, 8))
        aa.scatter(x[:], za[:], s=0.1, alpha=nmaxptps[show], c=maxptps[show], cmap=cm)
        aa.scatter(geom[:, 0], geom[:, 1], color="orange", s=1)
        ab.scatter(np.log(y[:]), za[:], s=0.1, alpha=nmaxptps[show], c=maxptps[show], cmap=cm)
        ac.scatter(np.log(a[:]), za[:], s=0.1, alpha=nmaxptps[show], c=maxptps[show], cmap=cm)
        aa.set_ylabel("z")
        aa.set_xlabel("x")
        ab.set_xlabel("$\\log y$")
        ac.set_xlabel("$\\log \\alpha$")
        aa.set_xlim([11 - 50, 59 + 50])
        ab.set_xlim([-1, 5])
        ac.set_xlim([2.5, 6.1])
        aa.set_ylim([0 - 10, geom[:, 1].max() + 10])
        fig.suptitle(ds.stem + " trust-ncg", y=0.925)
        plt.show()
        
        

        fig, (aa, ab, ac) = plt.subplots(1, 3, sharey=True, figsize=(8, 8))
        aa.scatter(ox[:], oza[:], s=0.1, alpha=nmaxptps[show], c=maxptps[show], cmap=cm)
        aa.scatter(geom[:, 0], geom[:, 1], color="orange", s=1)
        ab.scatter(np.log(oy[:]), oza[:], s=0.1, alpha=nmaxptps[show], c=maxptps[show], cmap=cm)
        ac.scatter(np.log(oa[:]), oza[:], s=0.1, alpha=nmaxptps[show], c=maxptps[show], cmap=cm)
        aa.set_ylabel("oz")
        aa.set_xlabel("ox")
        ab.set_xlabel("$\\log oy$")
        ac.set_xlabel("$\\log o\\alpha$")
        aa.set_xlim([11 - 50, 59 + 50])
        ab.set_xlim([-1, 5])
        ac.set_xlim([2.5, 6.1])
        aa.set_ylim([0 - 10, geom[:, 1].max() + 10])
        fig.suptitle(ds.stem + " BFGS", y=0.925)
        plt.show()

# %%
for ds in root.glob("*.h5"):
    with h5py.File(ds, "r") as f:
        ptps = standardwfs[ds.stem].ptp(1)

        fcs = firstchans[ds.stem]
        mcs = maxchans[ds.stem]
        N, C = ptps.shape
        
        maxptps = ptps.ptp(1).astype(float)
        show = rg().choice(np.flatnonzero(maxptps > 6), size=8, replace=False)
        show = slice(None)
        
        ox, oy, ozr, oza, oa = olocs[ds.stem].T
        
        x, y, zr, za, a = locs[ds.stem].T
        z_reg = za
        
        nmaxptps = 0.25 + 0.74 * (maxptps - maxptps.min()) / (maxptps.max() - maxptps.min())
        geom = f["geom"][:]
        
        opserrs = np.array([np.square(ptps[i] - ptp_at(ox[i], oy[i], ozr[i], oa[i], geom[fcs[i]:fcs[i] + C] - geom[mcs[i], 1])).mean() for i in range(N)])
        pserrs = np.array([np.square(ptps[i] - ptp_at(x[i], y[i], zr[i], a[i], geom[fcs[i]:fcs[i] + C] - geom[mcs[i], 1])).mean() for i in range(N)])
        
        fig, (aa, ab, ac, ad, ae) = plt.subplots(1, 5, figsize=(20, 5))
        aa.scatter(ox, x, s=1, alpha=nmaxptps, c=maxptps, cmap=cm)
        aa.set_xlabel("bfgs x")
        aa.set_ylabel("trust-ncg x")
        regline(ox, x, ax=aa)
        ab.scatter(ozr, zr, s=1, alpha=nmaxptps, c=maxptps, cmap=cm)
        ab.set_xlabel("bfgs z (relative to max chan)")
        ab.set_ylabel("trust-ncg z")
        regline(ozr, zr, ax=ab)
        ac.scatter(np.log(oy), np.log(y), s=1, alpha=nmaxptps, c=maxptps, cmap=cm)
        ac.set_xlabel("bfgs log y")
        ac.set_ylabel("trust-ncg log y")
        regline(oa, a, ax=ac)
        ad.scatter(np.log(oa), np.log(a), s=1, alpha=nmaxptps, c=maxptps, cmap=cm)
        ad.set_xlabel("bfgs log alpha")
        ad.set_ylabel("trust-ncg log alpha")
        regline(oa, a, ax=ad)
        ae.scatter(opserrs, pserrs, s=1, alpha=nmaxptps, c=maxptps, cmap=cm)
        ae.set_xlabel("bfgs point source mse")
        ae.set_ylabel("trust-ncg point source mse")
        regline(opserrs, pserrs, ax=ae)
        fig.suptitle(ds.stem)
        plt.show()

# %%
for ds in root.glob("*.h5"):
    with h5py.File(ds, "r") as f:
        maxptps = standardwfs[ds.stem].ptp(1).ptp(1).astype(float)
        show = rg().choice(np.flatnonzero(maxptps > 6), size=8, replace=False)
        show = slice(None)
        
        ox, oy, ozr, oza, oa = olocs[ds.stem].T
        
        x, y, zr, za, a = locs[ds.stem].T
        z_reg = za
        
        nmaxptps = 0.25 + 0.74 * (maxptps - maxptps.min()) / (maxptps.max() - maxptps.min())
        geom = f["geom"][:]
        
        
        fig, (aa, ab, ac) = plt.subplots(1, 3, sharey=True, figsize=(8, 8))
        cm = plt.cm.viridis
        aa.scatter(x[:], za[:], s=0.1, alpha=nmaxptps[show], c=maxptps[show], cmap=cm)
        aa.scatter(geom[:, 0], geom[:, 1], color="orange", s=1)
        ab.scatter(np.log(y[:]), za[:], s=0.1, alpha=nmaxptps[show], c=maxptps[show], cmap=cm)
        ac.scatter(np.log(a[:]), za[:], s=0.1, alpha=nmaxptps[show], c=maxptps[show], cmap=cm)
        aa.set_ylabel("z")
        aa.set_xlabel("x")
        ab.set_xlabel("$\\log y$")
        ac.set_xlabel("$\\log \\alpha$")
        aa.set_xlim([11 - 50, 59 + 50])
        ab.set_xlim([-1, 5])
        ac.set_xlim([2.5, 6.1])
        aa.set_ylim([0 - 10, geom[:, 1].max() + 10])
        fig.suptitle(ds.stem, y=0.925)
        plt.show()

# %%
for ds in root.glob("*.h5"):
    with h5py.File(ds, "r") as f:
        maxptps = standardwfs[ds.stem].ptp(1).ptp(1).astype(float)
        show = rg().choice(np.flatnonzero(maxptps > 6), size=8, replace=False)
        show = slice(None)
        
        ox, oy, ozr, oza, oa = olocs[ds.stem].T
        
        x, y, zr, za, a = locs[ds.stem].T
        z_reg = za
        
        nmaxptps = 0.25 + 0.74 * (maxptps - maxptps.min()) / (maxptps.max() - maxptps.min())
        geom = f["geom"][:]
        
        
        fig, (aa, ab, ac) = plt.subplots(1, 3, sharey=True, figsize=(8, 8))
        cm = plt.cm.viridis
        aa.scatter(ox[:], oza[:], s=0.1, alpha=nmaxptps[show], c=maxptps[show], cmap=cm)
        aa.scatter(geom[:, 0], geom[:, 1], color="orange", s=1)
        ab.scatter(np.log(oy[:]), oza[:], s=0.1, alpha=nmaxptps[show], c=maxptps[show], cmap=cm)
        ac.scatter(np.log(oa[:]), oza[:], s=0.1, alpha=nmaxptps[show], c=maxptps[show], cmap=cm)
        aa.set_ylabel("z")
        aa.set_xlabel("x")
        ab.set_xlabel("$\\log y$")
        ac.set_xlabel("$\\log \\alpha$")
        aa.set_xlim([11 - 50, 59 + 50])
        ab.set_xlim([-1, 5])
        ac.set_xlim([2.5, 6.1])
        aa.set_ylim([0 - 10, geom[:, 1].max() + 10])
        fig.suptitle(ds.stem, y=0.925)
        plt.show()

# %% tags=[]
# zregs = {}
# for ds in root.glob("*.h5"):
#     with h5py.File(ds, "r") as f:
#         maxptps = standardwfs[ds.stem].ptp(1).ptp(1).astype(float)
        
#         # z_rigid_reg, p_rigid = reg.register_rigid(
#         #     maxptps,
#         #     locs[ds.stem][:, 3],
#         #     (f["spike_index"][:, 0] - f["start_sample"][()]) / 30000,
#         #     robust_sigma=0,
#         #     disp=400,
#         #     denoise_sigma=0.1,
#         #     destripe=False,
#         # )
#         z_reg, dispmap = reg.register_nonrigid(
#             maxptps,
#             locs[ds.stem][:, 3],
#             (f["spike_index"][:, 0] - f["start_sample"][()]) / 30000,
#             robust_sigma=1,
#             rigid_disp=200,
#             disp=100,
#             denoise_sigma=0.1,
#             destripe=False,
#             n_windows=[5, 30, 60],
#             n_iter=1,
#             widthmul=0.25,
#         )
#         zregs[ds.stem] = z_reg
#         np.save(f"../data/{ds.stem}_zreg.npy", locs[ds.stem])

# %%
for ds in root.glob("*.h5"):
    with h5py.File(ds, "r") as f:
        maxptps = standardwfs[ds.stem].ptp(1).ptp(1).astype(float)
        nmaxptps = 0.1 + 0.89 * (maxptps - maxptps.min()) / (maxptps.max() - maxptps.min())
        geom = f["geom"][:]
        
        x, y, z_rel, z_abs, alpha = locs[ds.stem].T
        # z_reg = zregs[ds.stem]
        z_reg = z_abs
        
        plt.figure()
        plt.hist(y, bins=np.arange(50))
        plt.xlabel("y")
        plt.title(ds.stem)
        plt.show()

        fig, (aa, ab, ac) = plt.subplots(1, 3, sharey=True, figsize=(8, 8))
        # nmaxptps = np.ones_like(nmaxptps)
        # (1900, 2600)
        

        inrange = (1900 < z_reg) & (z_reg < 2600)
        cm = plt.cm.viridis

        aa.scatter(x[:], z_reg[:], s=0.1, alpha=nmaxptps[:], c=maxptps[:], cmap=cm)
        aa.scatter(geom[:, 0], geom[:, 1], color="orange", s=1)
        ab.scatter(np.log(y[:]), z_reg[:], s=0.1, alpha=nmaxptps[:], c=maxptps[:], cmap=cm)
        ac.scatter(np.log(alpha[:]), z_reg[:], s=0.1, alpha=nmaxptps[:], c=maxptps[:], cmap=cm)
        aa.set_ylabel("z")
        aa.set_xlabel("x")
        ab.set_xlabel("$\\log y$")
        ac.set_xlabel("$\\log \\alpha$")
        aa.set_xlim([11 - 50, 59 + 50])
        ab.set_xlim([-1, 5])
        ac.set_xlim([2.5, 6.1])
        aa.set_ylim([0 - 10, geom[:, 1].max() + 10])
        fig.suptitle(ds.stem, y=0.925)
        plt.show()

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
        nmaxptps = 0.1 + 0.89 * (maxptps - maxptps.min()) / (maxptps.max() - maxptps.min())
        geom = f["geom"][:]
        
        x, y, z_rel, z_abs, alpha = locs[ds.stem].T
        z_reg = zregs[ds.stem]
        
        plt.figure()
        plt.hist(y, bins=np.arange(50))
        plt.xlabel("y")
        plt.title(ds.stem)
        plt.show()

        fig, (aa, ab, ac, ad, ae, af) = plt.subplots(1, 6, sharey=True, figsize=(16, 8))
        # nmaxptps = np.ones_like(nmaxptps)
        # (1900, 2600)
        

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
        plt.show()

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
for ds in root.glob("*.h5"):
    with h5py.File(ds, "r") as f:
        maxptps = standardwfs[ds.stem].ptp(1).ptp(1).astype(float)
        geom = f["geom"][:]

        fig, axes = plt.subplots(1, 7, sharey=True, figsize=(16, 8))
        (aa, ab, ac, ag, ad, ae, af) = axes.flat
        nmaxptps = 0.1 + 0.89 * (maxptps - maxptps.min()) / (maxptps.max() - maxptps.min())
        # nmaxptps = np.ones_like(nmaxptps)
        # (1900, 2600)
        
        x, y, z_rel, z_abs, alpha = locs[ds.stem].T
        z_reg = zregs[ds.stem]
        
        low = 3100
        high = 3600
        ginrange = (low <= geom[:, 1]) & (geom[:, 1] <= high)
        inrange = (low <= z_reg) & (z_reg <= high)
        cm = plt.cm.viridis
        aes = feats[ds.stem]
        
        xscale = 1
        yscale = 10
        ascale = 15
        pscale = 30
        fscale = 8


        aa.scatter(x[inrange], z_reg[inrange], s=0.1, alpha=nmaxptps[inrange], c=maxptps[inrange], cmap=cm)
        aa.scatter(geom[ginrange, 0], geom[ginrange, 1], color="orange", s=1)
        ab.scatter(yscale * np.log(y[inrange]), z_reg[inrange], s=0.1, alpha=nmaxptps[inrange], c=maxptps[inrange], cmap=cm)
        ac.scatter(ascale * np.log(alpha[inrange]), z_reg[inrange], s=0.1, alpha=nmaxptps[inrange], c=maxptps[inrange], cmap=cm)
        ag.scatter(pscale * np.log(maxptps[inrange]), z_reg[inrange], s=0.1, alpha=nmaxptps[inrange], c=maxptps[inrange], cmap=cm)
        ad.scatter(fscale * zscore(aes[inrange, 0]), z_reg[inrange], s=0.1, alpha=nmaxptps[inrange], c=maxptps[inrange], cmap=cm)
        ae.scatter(fscale * zscore(aes[inrange, 1]), z_reg[inrange], s=0.1, alpha=nmaxptps[inrange], c=maxptps[inrange], cmap=cm)
        af.scatter(fscale * zscore(aes[inrange, 2]), z_reg[inrange], s=0.1, alpha=nmaxptps[inrange], c=maxptps[inrange], cmap=cm)
        aa.set_ylabel("z")
        aa.set_xlabel("x")
        ab.set_xlabel(f"${yscale}\cdot\\log y$")
        ac.set_xlabel(f"${ascale}\cdot\\log \\alpha$")
        ag.set_xlabel(f"${pscale}\cdot\\log \\mathrm{{maxptp}}$")
        ad.set_xlabel(f"{fscale} * ae1")
        ae.set_xlabel(f"{fscale} * ae2")
        af.set_xlabel(f"{fscale} * ae3")
        aa.set_xlim([11 - 50, 59 + 50])
        ab.set_xlim([-1 * yscale, 5 * yscale])
        ac.set_xlim([2.5 * ascale, 6.1 * ascale])
        for ax in axes.flat:
            ax.set_ylim([low - 10, high + 10])
            ax.set_aspect(1)
        fig.suptitle(ds.stem, y=0.925)
        plt.show()

# %%
for ds in root.glob("*.h5"):
    with h5py.File(ds, "r") as f:
        maxptps = standardwfs[ds.stem].ptp(1).ptp(1).astype(float)
        geom = f["geom"][:]

        fig, (aa, ab, ac, ad, ae, af) = plt.subplots(1, 6, sharey=True, figsize=(16, 8))
        nmaxptps = 0.1 + 0.89 * (maxptps - maxptps.min()) / (maxptps.max() - maxptps.min())
        # nmaxptps = np.ones_like(nmaxptps)
        # (1900, 2600)
        
        x, y, z_rel, z_abs, alpha = locs[ds.stem].T
        z_reg = zregs[ds.stem]
        

        inrange = (1900 < z_reg) & (z_reg < 2600)
        cm = plt.cm.viridis
        aes = feats[ds.stem]
        tm = triage.weighted_knn_triage(np.c_[x, np.log(y), z_reg, np.log(alpha), np.log(maxptps)], zscore(aes[:, :3]), percentile=70)


        aa.scatter(x[tm], z_reg[tm], s=0.1, alpha=nmaxptps[tm], c=maxptps[tm], cmap=cm)
        aa.scatter(geom[:, 0], geom[:, 1], color="orange", s=1)
        ab.scatter(np.log(y[tm]), z_reg[tm], s=0.1, alpha=nmaxptps[tm], c=maxptps[tm], cmap=cm)
        ac.scatter(np.log(alpha[tm]), z_reg[tm], s=0.1, alpha=nmaxptps[tm], c=maxptps[tm], cmap=cm)
        ad.scatter(zscore(aes[tm, 0]), z_reg[tm], s=0.1, alpha=nmaxptps[tm], c=maxptps[tm], cmap=cm)
        ae.scatter(zscore(aes[tm, 1]), z_reg[tm], s=0.1, alpha=nmaxptps[tm], c=maxptps[tm], cmap=cm)
        af.scatter(zscore(aes[tm, 2]), z_reg[tm], s=0.1, alpha=nmaxptps[tm], c=maxptps[tm], cmap=cm)
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
        fig.suptitle(f"{ds.stem} triaged", y=0.925)

# %%
for ds in root.glob("*.h5"):
    with h5py.File(ds, "r") as f:
        maxptps = standardwfs[ds.stem].ptp(1).ptp(1).astype(float)
        geom = f["geom"][:]

        fig, (aa, ab, ac, ad, ae, af) = plt.subplots(1, 6, sharey=True, figsize=(16, 8))
        nmaxptps = 0.1 + 0.89 * (maxptps - maxptps.min()) / (maxptps.max() - maxptps.min())
        # nmaxptps = np.ones_like(nmaxptps)
        # (1900, 2600)
        
        x, y, z_rel, z_abs, alpha = locs[ds.stem].T
        z_reg = zregs[ds.stem]
        

        inrange = (1900 < z_reg) & (z_reg < 2600)
        cm = plt.cm.viridis
        aes = feats[ds.stem]
        tm = triage.weighted_knn_triage(np.c_[x, np.log(y), z_reg, np.log(alpha), np.log(maxptps)], zscore(aes[:, :3]), percentile=70)

        comps = triage.coarse_split(np.c_[x, np.log(y), z_reg, np.log(alpha), np.log(maxptps)][tm], zscore(aes[:, :3])[tm])
        cs = np.array(cc.glasbey_hv)[comps % 256]

        aa.scatter(x[tm], z_reg[tm], s=0.1, c=cs)
        aa.scatter(geom[:, 0], geom[:, 1], color="orange", s=1)
        ab.scatter(np.log(y[tm]), z_reg[tm], s=0.1, c=cs)
        ac.scatter(np.log(alpha[tm]), z_reg[tm], s=0.1, c=cs)
        ad.scatter(zscore(aes[tm, 0]), z_reg[tm], s=0.1, c=cs)
        ae.scatter(zscore(aes[tm, 1]), z_reg[tm], s=0.1, c=cs)
        af.scatter(zscore(aes[tm, 2]), z_reg[tm], s=0.1, c=cs)
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
        fig.suptitle(f"{ds.stem} triaged", y=0.925)

# %%
