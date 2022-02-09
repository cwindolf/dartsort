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
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from IPython.display import display
import colorcet

# %%
from spike_psvae import simdata, vis_utils, waveform_utils, point_source_centering, localization, laplace, posterior, denoise

# %%
plt.rc("figure", dpi=200)

# %%
rg = lambda k=0: np.random.default_rng(k)

# %%
ctx_h5 = h5py.File("../data/ks_np2_nzy_cortex.h5", "r")
geom_np2 = ctx_h5["geom"][:]

# %%
print(list(ctx_h5.keys()))

# %%
big_y = True
threshy = 0.5

# %%
(
    choice_units,
    choice_loc_templates,
    tmaxchans,
    tfirstchans,
    txs,
    tys,
    tz_rels,
    talphas,
    pserrs
) = simdata.cull_templates(ctx_h5["templates"][:], geom_np2, 30, y_min=threshy if big_y else -threshy, rg=rg(), channel_radius=20)
C = choice_loc_templates.shape[-1]

# %%
lgeoms = [geom_np2[f : f + C] - np.array([[0, geom_np2[m, 1]]])  for f, m in zip(tfirstchans, tmaxchans)]

# %%
tpca = denoise.fit_temporal_pca(ctx_h5["denoised_waveforms"])

# %%
print(choice_loc_templates[0].ptp(0).shape, lgeoms[0].shape)

# %%
figdir = Path("../figs/stanspikes")
figdir.mkdir(exist_ok=True)
figdir, figdir.resolve()


# %%
def stan_diagnostic(
    name, waveform, firstchan, maxchan
):
    orig_ptp = waveform.ptp(0)
    
    waveform = denoise.apply_temporal_pca(tpca, waveform[None, ...])[0]
    waveform = denoise.enforce_decrease(waveform)
    ptp = waveform.ptp(0)
    C = ptp.shape[0]
    lgeom = geom_np2[firstchan : firstchan + C].copy()
    lgeom[:, 1] -= geom_np2[maxchan, 1]
    
    res, x, y, z, alpha = posterior.sample(ptp, lgeom)
    ix = rg().choice(x.size, replace=False, size=50)
    
    lx, ly, lzr, lza, la = localization.localize_ptp(ptp, maxchan, geom_np2, firstchan=firstchan, geomkind="firstchan")
    lapx, lapy = laplace.laplace_correct(lx, ly, lzr, la, ptp, lgeom)
    
    # print(name)
    # display(res)
    
    fig, (aa, ab) = plt.subplots(1, 2)
    for i in ix:
        ha, _ = vis_utils.plot_single_ptp(
            localization.ptp_at(x[i], y[i], z[i], alpha[i], lgeom),
            aa, "sample", "silver", None
        )
    ho, _ = vis_utils.plot_single_ptp(orig_ptp, aa, "before", "purple", None)
    hb, _ = vis_utils.plot_single_ptp(ptp, aa, "ex. denoise", "k", None)
    hc, _ = vis_utils.plot_single_ptp(
        localization.ptp_at(lx, ly, lzr, la, lgeom),
        aa, "LS", "r", None
    )
    hd, _ = vis_utils.plot_single_ptp(
        localization.ptp_at(x.mean(), y.mean(), z.mean(), alpha.mean(), lgeom),
        aa, "Stan", "g", None
    )
    he, _ = vis_utils.plot_single_ptp(
        localization.ptp_at(lapx, lapy, lzr, la, lgeom),
        aa, "Lap", "b", None
    )
    
    aa.legend([ho, hb, ha, hc, hd, he], ["singlechan", "ex. denoised", "sampled", "LS", "Stan", "Lap LS"])
    aa.set_ylabel("ptp")
    
    xx, yy = np.meshgrid(
        np.linspace(x.min(), x.max(), num=100),
        np.linspace(y.min(), y.max(), num=100),
        indexing="ij",
    )
    post = np.zeros_like(xx)
    for i in range(100):
        for j in range(100):
            post[i, j] = laplace.point_source_post_lpdf(xx[i, j], yy[i, j], lzr, la, ptp, lgeom, ptp_sigma=0.1)
    ab.contourf(xx, yy, post, levels=30)
    

    ab.scatter(x, y, s=1, alpha=0.1, color="k", label="post samples")
    ab.scatter([lx], [ly], s=10, marker="x", color="r", label="LS")
    ab.scatter([x.mean()], [y.mean()], s=10, marker="x", color="g", label="Stan mean")
    ab.scatter([lapx], [lapy], s=10, marker="x", color="b", label="Lap LS")
    ab.set_xlabel("x")
    ab.set_ylabel("y")
    ab.legend()
    
    fig.suptitle(f"{name} maxptp={ptp.max():0.2f} LS=({lx:0.2f},{ly:0.2f},{lzr:0.2f},{la:0.2f})", fontsize=10)
    plt.tight_layout()
    
    return fig
    

# %%
small_y = ctx_h5["y"][:] < 0.5
big_y = ctx_h5["y"][:] > 4
big_ptp = ctx_h5["maxptp"][:] > 6
good = (16 < ctx_h5["max_channels"][:]) & (ctx_h5["max_channels"][:] < 360)
units = ctx_h5["spike_train"][:, 1]

# %% tags=[]
ix = rg().choice(np.flatnonzero(good & small_y & big_ptp), replace=False, size=20)
for i in ix:
    fig = stan_diagnostic(
        f"[small y] unit {units[i]} spike {i} --",
        ctx_h5["denoised_waveforms"][i],
        ctx_h5["first_channels"][i],
        ctx_h5["max_channels"][i],
    )
    fig.savefig(figdir / f"smally_{i}.png")

# %%
ix = rg().choice(np.flatnonzero(good & big_y & big_ptp), replace=False, size=20)
for i in ix:
    fig = stan_diagnostic(
        f"[big y] unit {units[i]} spike {i} --",
        ctx_h5["denoised_waveforms"][i],
        ctx_h5["first_channels"][i],
        ctx_h5["max_channels"][i],
    )
    fig.savefig(figdir / f"bigyptp_{i}.png")

# %%
