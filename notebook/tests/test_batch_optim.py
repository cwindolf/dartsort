# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python [conda env:a]
#     language: python
#     name: conda-env-a-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np

# %%
from spike_psvae import (
    optutils,
    localize_torch,
    localize_index,
    subtract,
    waveform_utils,
)

# %%
from scipy.optimize import minimize

# %%
import h5py

# %%
import matplotlib.pyplot as plt

# %%
import torch
import torch.nn.functional as F

# %%
from torch import vmap
from torch.func import grad, hessian, grad_and_value

# %%
# with h5py.File("/Users/charlie/data/test/subtraction_CSHL051.ap_t_0_None.h5") as h5:
#     geom = h5["geom"][:]

# %%
import spikeinterface.full as si

# %%
rec = si.read_binary_folder("/mnt/home/cwindolf/ceph/subtraction_fig_data/zad5/")

# %%
with h5py.File("/tmp/loctest/subtraction.h5", "r+") as h5:
    print(list(h5.keys()))
    if "spike_index" in h5:
        del h5["ptp_vectors"]
        del h5["spike_index"]
        if "localizations" in h5:
            del h5["localizations"]
    print(
        [
            (k, h5["denoised_ptpvector_info"][k][()])
            for k in h5["denoised_ptpvector_info"].keys()
        ]
    )
    h5.create_dataset(
        "spike_index",
        shape=(0, 2),
        chunks=(1024, 2),
        maxshape=(None, 2),
        dtype=np.int64,
    )
    h5.create_dataset(
        "ptp_vectors",
        shape=(0, 40),
        chunks=(1024, 40),
        maxshape=(None, 40),
        dtype=np.float64,
    )
    h5.create_dataset(
        "localizations",
        shape=(0, 5),
        chunks=(1024, 5),
        maxshape=(None, 5),
        dtype=np.float64,
    )

# %%
# get example data
# # %%snakeviz
sub_h5_none = subtract.subtraction(
    rec.frame_slice(0, 100 * 30_000),
    out_folder="/tmp/loctestnone",
    thresholds=[12, 10, 8, 6, 5],
    n_jobs=2,
    save_denoised_ptp_vectors=True,
    save_subtracted_tpca_projs=False,
    save_cleaned_tpca_projs=False,
    save_denoised_tpca_projs=False,
    overwrite=True,
    n_sec_pca=40,
    localization_kind=None,
    extra_features=[],
)

# %%
# get example data
# # %%snakeviz
sub_h5_orig = subtract.subtraction(
    rec.frame_slice(0, 100 * 30_000),
    out_folder="/tmp/loctestorig",
    thresholds=[12, 10, 8, 6, 5],
    n_jobs=2,
    save_denoised_ptp_vectors=False,
    save_subtracted_tpca_projs=False,
    save_cleaned_tpca_projs=False,
    save_denoised_tpca_projs=False,
    overwrite=False,
    n_sec_pca=40,
    loc_feature="ptp",
    extra_features=[],
)

# %%
# get example data
# # %%snakeviz
sub_h5_lm = subtract.subtraction(
    rec.frame_slice(0, 100 * 30_000),
    out_folder="/tmp/loctestgpu",
    thresholds=[12, 10, 8, 6, 5],
    n_jobs=2,
    save_denoised_ptp_vectors=False,
    save_subtracted_tpca_projs=False,
    save_cleaned_tpca_projs=False,
    save_denoised_tpca_projs=False,
    overwrite=False,
    n_sec_pca=40,
    loc_feature="ptpgpu",
    extra_features=[],
)

# %%
# get example data
# # %%snakeviz
sub_h5_orig = subtract.subtraction(
    rec.frame_slice(0, 100 * 30_000),
    out_folder="/tmp/loctestorigpar",
    thresholds=[12, 10, 8, 6, 5],
    n_jobs=2,
    save_denoised_ptp_vectors=False,
    save_subtracted_tpca_projs=False,
    save_cleaned_tpca_projs=False,
    save_denoised_tpca_projs=False,
    overwrite=False,
    n_sec_pca=40,
    loc_feature="ptp",
    extra_features=[],
    loc_workers=4,
)

# %%
# get example data
# # %%snakeviz
sub_h5_lm = subtract.subtraction(
    rec.frame_slice(0, 100 * 30_000),
    out_folder="/tmp/loctestgpumorejobs",
    thresholds=[12, 10, 8, 6, 5],
    n_jobs=4,
    save_denoised_ptp_vectors=False,
    save_subtracted_tpca_projs=False,
    save_cleaned_tpca_projs=False,
    save_denoised_tpca_projs=False,
    overwrite=False,
    n_sec_pca=40,
    loc_feature="ptpgpu",
    extra_features=[],
)

# %%
# get example data
# # %%snakeviz
sub_h5_lm_basic = subtract.subtraction(
    rec.frame_slice(0, 100 * 30_000),
    out_folder="/tmp/loctestbasic",
    thresholds=[5],
    n_jobs=4,
    save_denoised_ptp_vectors=False,
    save_subtracted_tpca_projs=False,
    save_cleaned_tpca_projs=False,
    save_denoised_tpca_projs=False,
    overwrite=False,
    n_sec_pca=40,
    loc_feature="ptpgpu",
    extra_features=[],
)

# %%
with h5py.File(sub_h5_none, "r") as h5:
    ptps = h5["ptp_vectors"][:]
    geom = h5["geom"][:]
    ci = h5["channel_index"][:]
    maxchans = h5["spike_index"][:, 1]
    t = h5["spike_index"][:, 0] / h5["fs"][()]
    a = np.nanmax(ptps, 1)
with h5py.File(sub_h5_orig, "r") as h5:
    x_orig, y_orig, z_orig = h5["localizations"][:, :3].T
with h5py.File(sub_h5_lm, "r") as h5:
    x_lm, y_lm, z_lm = h5["localizationsptpgpu"][:, :3].T

# %%
fig, (aa, ab) = plt.subplots(ncols=2, figsize=(10, 10), sharey=True)
aa.scatter(x_orig, z_orig, c=np.minimum(a, 15), lw=0, s=1)
ab.scatter(x_lm, z_lm, c=np.minimum(a, 15), lw=0, s=1)
aa.set_ylim([geom[:, 1].min(), geom[:, 1].max()])
ab.set_ylim([geom[:, 1].min(), geom[:, 1].max()])
aa.set_xlim([geom[:, 0].min() - 100, geom[:, 0].max() + 100])
ab.set_xlim([geom[:, 0].min() - 100, geom[:, 0].max() + 100])
aa.set_title("Previous (scipy LBFGS)")
ab.set_title("New (torch batched LM)")

# %%
ptps.shape

# %%
ptps.size * 4 / 1e6

# %%
x_orig, y_orig, zr_orig, za_orig, alpha_orig = localize_index.localize_ptps_index(
    ptps, geom, maxchans, ci, radius=100, pbar=True, n_workers=8
)

# %%

# %%
tptps = torch.as_tensor(ptps, device="cuda", dtype=torch.double)
tgeom = torch.as_tensor(geom, device="cuda", dtype=torch.double)
tmcs = torch.tensor(maxchans, device="cuda", dtype=torch.long)
tci = torch.tensor(ci, device="cuda", dtype=torch.long)

# %%
N, C = ptps.shape
maxchans = maxchans.astype(int)
local_geoms = np.pad(geom, [(0, 1), (0, 0)])[ci[maxchans]]
local_geoms[:, :, 1] -= geom[maxchans, 1][:, None]

# %%
nanmask = ~np.isnan(ptps)

# %%
tptps = torch.nan_to_num(tptps)
tlgs = torch.as_tensor(local_geoms)
tnms = torch.as_tensor(nanmask)

# %%
nptps = tptps / torch.as_tensor(a[:, None], device=tptps.device)

# %%
tptps.shape

# %%
torch.isnan(tptps).sum()

# %%
a.shape, nptps.shape

# %%
xs_orig = x_orig, y_orig, zr_orig, alpha_orig
x_orig, y_orig, zr_orig, alpha_orig = map(
    lambda x: torch.as_tensor(x, device="cpu"), xs_orig
)
ptps_orig = (
    localize_torch.bptp_at(
        x_orig, y_orig, zr_orig, torch.ones_like(alpha_orig), tlgs.cpu()
    )
    * tnms
)
res_orig = (
    torch.square(nptps.cpu() - ptps_orig).mean(1) - torch.log(10.0 * y_orig) / 10000.0
)
res_orig.min(), res_orig.median(), res_orig.mean(), res_orig.max()

# %%
tic = time.time()
x_lm, y_lm, zr_lm, za_lm, alpha_lm = localize_torch.localize_ptps_index_lm(
    tptps,
    geom,
    tmcs,
    ci,
    radius=100,
    convergence_err=1e-20,
    convergence_g=1e-20,
    max_steps=250,
    nu=10.0,
    lambd=1e-2,
    min_scale=1e-5,
)
toc = time.time()
print(toc - tic)

xs_lm = x_lm, y_lm, zr_lm, alpha_lm
x_lm, y_lm, zr_lm, alpha_lm = map(lambda x: torch.as_tensor(x, device="cpu"), xs_lm)
ptps_lm = (
    localize_torch.bptp_at(x_lm, y_lm, zr_lm, torch.ones_like(alpha_lm), tlgs.cpu())
    * tnms
)
res_lm = torch.square(nptps.cpu() - ptps_lm).mean(1) - torch.log(10.0 * y_lm) / 10000.0
res_lm.min(), res_lm.median(), res_lm.mean(), res_lm.max(), (res_lm < res_orig).to(
    torch.double
).mean()

# %%

# %%
((df > 5) & (a >= 6)).sum()

# %%
fig, (aa, ab, ac) = plt.subplots(ncols=3, figsize=(15, 5))
aa.scatter(res_orig, res_lm, lw=0, s=2, c=np.minimum(a, 15))
aa.set_xlabel("original objective")
aa.set_ylabel("new objective")
mn = min(res_orig.min(), res_lm.min())
mx = max(res_orig.max(), res_lm.max())
aa.plot([mn, mx], [mn, mx], lw=0.8, color="k", label="new=orig")
aa.legend(frameon=False, loc="upper left")
aa.loglog()

ab.scatter(a, res_lm - res_orig, lw=0, s=2, c=np.minimum(a, 15))
ab.set_ylabel("new - original objective")
ab.set_xlabel("max ptp")
ab.semilogx()

df = res_lm - res_orig
# ac.hist(-df[df < 0], label="new < orig", histtype="step", bins=np.arange(0, np.abs(df).max(), 10), log=True)
# ac.hist(df[df >= 0], label="new >= orig", histtype="step", bins=np.arange(0, np.abs(df).max(), 10), log=True)
# vvv = 0.1 * (np.abs(df).max() // 0.1 + 1)
# vvv = 75

ac.hist(df[a < 6], label="maxptp < 6", histtype="step", log=True)
ac.hist(df[a >= 6], label="maxptp >= 6", histtype="step", log=True)
ac.axvline(0, lw=0.8, color="k")
ac.legend()
# ac.set_xlabel("absolute change in objective")
ac.set_xlabel("new - original objective")
ac.set_ylabel("number of spikes")
fig.tight_layout()

# %%

# %%

# %%
# get example data
# # %%snakeviz
sub_h5_lm = subtract.subtraction(
    rec.frame_slice(0, 100 * 30_000),
    out_folder="/tmp/loctestgpumorejobs",
    thresholds=[12, 10, 8, 6, 5],
    n_jobs=4,
    save_denoised_ptp_vectors=False,
    save_subtracted_tpca_projs=False,
    save_cleaned_tpca_projs=False,
    save_denoised_tpca_projs=False,
    overwrite=False,
    n_sec_pca=40,
    loc_feature="ptpgpu",
    extra_features=[],
)

# %%
import time

# %%

# %%
sub_h5

# %%
with h5py.File(sub_h5, "r", locking=False) as h5:
    ptps = h5["ptp_vectors"][:][:1000]
    maxchans = h5["spike_index"][:, 1][:1000]
    locs0 = h5["localizations"][:][:1000]
    x0, y0, z0, zr0 = locs0[:, :4].T
    ci = h5["channel_index"][:]
    geom = h5["geom"][()]

# %%
plt.plot(ptps.T)

# %%
plt.scatter(x0, zr0)

# %%
N, C = ptps.shape
maxchans = maxchans.astype(int)

local_geoms = np.pad(geom, [(0, 1), (0, 0)])[ci[maxchans]]
local_geoms[:, :, 1] -= geom[maxchans, 1][:, None]

# %%
nanmask = ~np.isnan(ptps)

# %%
# %%timeit
[localize_index.localize_ptp_index(ptp, lg) for ptp, lg in zip(ptps, local_geoms)]

# %%
# %%timeit
localize_index.localize_ptps_index(ptps, geom, maxchans, ci, pbar=False)

# %%
# %%timeit
localize_index.localize_ptps_index(ptps, geom, maxchans, ci, n_workers=8, pbar=False)

# %%
tptps = torch.as_tensor(ptps)
tptps = torch.nan_to_num(tptps)
tlgs = torch.as_tensor(local_geoms)
tnms = torch.as_tensor(nanmask)

# %%
# %%timeit
localize_torch.localize_ptps_index_lm(tptps, geom, maxchans, ci)

# %%
# %%timeit
localize_torch.localize_ptps_index_newton(tptps, geom, maxchans, ci)

# %%
x0, y0, zr0, za0, a0 = localize_index.localize_ptps_index(
    ptps, geom, maxchans, ci, n_workers=8, pbar=False
)


# %%
def localize_single_torch(ptp, local_geom, nan_mask):
    ptp = torch.nan_to_num(ptp)
    nptp = ptp / ptp.max()
    xcom = (ptp * local_geom[:, 0]).sum() / ptp.sum()
    zcom = (ptp * local_geom[:, 1]).sum() / ptp.sum()
    y0 = 20.0
    xinit = torch.tensor([xcom, y0, zcom])

    def obj(loc):
        return localize_torch.mse(loc, nptp, nan_mask, local_geom)

    grad_and_func = grad_and_value(obj)
    hess = hessian(obj)

    x, nevals, i = optutils.single_newton(
        xinit, grad_and_func, hess, nsteps=15000, max_ls=100, c1=1e-6
    )

    alpha = localize_torch.find_alpha(ptp, nan_mask, *x, local_geom)

    return [*x, alpha], i


# %%
ressingle = [
    localize_single_torch(ptp, lg, nm)
    for ptp, lg, nm in zip(*map(torch.as_tensor, [ptps, local_geoms, nanmask]))
]

# %%
xs1, i1 = zip(*ressingle)

# %%
x1, y1, zr1, a1 = map(torch.tensor, zip(*xs1))

# %%
vgrad_and_func = vmap(grad_and_value(localize_torch.mse))
vhess = vmap(hessian(localize_torch.mse))


# %%
def localize_multi_torch(ptps, local_geoms, nan_masks, init_y="taylor"):
    maxptps, _ = torch.max(torch.nan_to_num(ptps, -1.0), dim=1)
    ptps = torch.nan_to_num(ptps)
    nptps = ptps / maxptps[:, None]
    com = (ptps[:, :, None] * local_geoms).sum(1) / ptps.sum(1)[:, None]
    xcom, zcom = com.T
    y0 = torch.full_like(xcom, init_y)
    xinit = torch.column_stack([xcom, y0, zcom])

    x, nevals, i = optutils.batched_newton(
        xinit,
        vgrad_and_func,
        vhess,
        extra_args=(nptps, nan_masks, local_geoms),
        max_steps=150,
        convergence_x=1e-4,
    )
    alpha = localize_torch.bfind_alpha(ptps, nan_masks, *x.T, local_geoms)
    return torch.column_stack((x, alpha)), i


# %%
ptps0 = localize_torch.bptp_at(*map(torch.as_tensor, [x0, y0, zr0, a0]), tlgs) * tnms
res0 = torch.square(tptps - ptps0).sum(1)
res0.min(), res0.median(), res0.mean(), res0.max()

# %%
# # %%timeit
# localize_multi_torch(tptps, tlgs, tnms, init_y=1.)

# %%
xs_bn, ii = localize_multi_torch(tptps, tlgs, tnms, init_y=1.0)
x_bn, y_bn, zr_bn, a_bn = map(torch.tensor, zip(*xs_bn))
ptps_bn = localize_torch.bptp_at(x_bn, y_bn, zr_bn, a_bn, tlgs) * tnms
res_bn = torch.square(tptps - ptps_bn).sum(1)
res_bn.min(), res_bn.median(), res_bn.mean(), res_bn.max()


# %%
def localize_multi_lm(
    ptps, local_geoms, nan_masks, init_y=1.0, scale_problem="none", tikhonov=0.0
):
    maxptps, _ = torch.max(torch.nan_to_num(ptps, -1.0), dim=1)
    ptps = torch.nan_to_num(ptps)
    nptps = ptps / maxptps[:, None]
    com = (ptps[:, :, None] * local_geoms).sum(1) / ptps.sum(1)[:, None]
    xcom, zcom = com.T
    y0 = torch.full_like(xcom, init_y)
    xinit = torch.column_stack([xcom, y0, zcom])
    print(f"{xinit.dtype=}")

    x, i = optutils.batched_levenberg_marquardt(
        xinit,
        vgrad_and_func,
        vhess,
        extra_args=(nptps, nan_masks, local_geoms),
        max_steps=150,
        scale_problem=scale_problem,
        convergence_g=1e-10,
        convergence_err=1e-10,
        tikhonov=tikhonov,
    )
    alpha = localize_torch.bfind_alpha(ptps, nan_masks, *x.T, local_geoms)
    return torch.column_stack((x, alpha)), i


# %%
xs_lm, ii_lm = localize_multi_lm(
    tptps, tlgs, tnms, init_y=1.0, scale_problem="hessian", tikhonov=0.0
)
x_lm, y_lm, zr_lm, a_lm = map(torch.tensor, zip(*xs_lm))
ptps_lm = localize_torch.bptp_at(x_lm, y_lm, zr_lm, a_lm, tlgs) * tnms
res_lm = torch.square(tptps - ptps_lm).sum(1)
res_lm.min(), res_lm.median(), res_lm.mean(), res_lm.max()

# %%
ii_lm

# %%
(res_lm - res0)[ii_lm == 150]

# %%
(res_lm - res0)[ii_lm < 150]

# %%
(res_lm - res0)

# %%
(res_lm - res0).mean()

# %%
(res_lm - res0).median()

# %%
(res_lm < res0).to(torch.double).mean()

# %%

# %%

# %%

# %%
x3, y3, zr3, za3, a3 = localize_torch.localize_ptps_index(
    tptps,
    geom,
    maxchans,
    ci,
    # max_steps=500,
    # convergence_x=1e-10,
)
ptps3 = localize_torch.bptp_at(x3, y3, zr3, a3, tlgs) * tnms
res3 = torch.square(tptps - ptps3).sum(1)
res3.min(), res3.median(), res3.mean(), res3.max()

# %%
ptps0 = localize_torch.bptp_at(*map(torch.as_tensor, [x0, y0, zr0, a0]), tlgs) * tnms
res0 = torch.square(tptps - ptps0).sum(1)
res0.min(), res0.median(), res0.mean(), res0.max()

# %%
(res3 < res0).to(float).mean()

# %%
plt.hist(res_lm.numpy(), bins=np.arange(0, 100), histtype="step")
plt.hist(res0, bins=np.arange(0, 100), histtype="step")

# %%
(res3 - res0)[res3 < res0].mean()

# %%
(res3 - res0)[res3 < res0].min()

# %%
(res3 - res0)[res3 > res0].mean()

# %%
(res3 - res0)[res3 > res0].max()

# %%
x3.min(), x3.max()

# %%
geom[:, 0].min(), geom[:, 0].max()

# %%

# %%
ptps1 = localize_torch.bptp_at(x1, y1, zr1, a1, tlgs) * tnms
res1 = torch.square(tptps - ptps1).sum(1)
res1.min(), res1.median(), res1.mean(), res1.max()

# %%
ptps2 = localize_torch.bptp_at(x2, y2, zr2, a2, tlgs) * tnms
res2 = torch.square(tptps - ptps2).sum(1)
res2.min(), res2.median(), res2.mean(), res2.max()

# %%
mn = min(res0.min(), res1.min())
mx = max(res0.max(), res1.max())
plt.plot([mn, mx], [mn, mx], "k")
plt.scatter(res0, res3, s=1, lw=0)
plt.loglog()

# %%

# %%

# %%
N, C = ptps.shape

# handle channel subsetting
nc = len(channel_index)
subset = channel_index_subset(geom, channel_index, n_channels=n_channels, radius=radius)
subset = binary_subset_to_relative(subset)
channel_index_pad = F.pad(torch.as_tensor(channel_index), (0, 1, 0, 0), value=nc)
channel_index = channel_index_pad[torch.arange(nc)[:, None], subset]
# pad with 0s rather than nans, we will mask below.
ptps = F.pad(ptps, (0, 1, 0, 0))[torch.arange(N)[:, None], subset[maxchans]]

# torch everyone
device = ptps.device
ptps = torch.as_tensor(ptps, dtype=dtype, device=device)
geom = torch.as_tensor(geom, dtype=dtype, device=device)
channel_index = torch.as_tensor(channel_index, device=device)

# figure out which chans are outside the probe
in_probe_channel_index = (channel_index < nc).to(torch.double)
nan_mask = in_probe_channel_index[maxchans]

# local geometries in each ptp
geom_pad = F.pad(geom, (0, 0, 0, 1))
local_geoms = geom_pad[channel_index[maxchans]]
local_geoms[:, :, 1] -= geom[maxchans, 1][:, None]

# center of mass initialization
com = (ptps[:, :, None] * local_geoms).sum(1) / ptps.sum(1)[:, None]
xcom, zcom = com.T
