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
from tqdm.auto import trange, tqdm
from pathlib import Path
import torch

# %%
import time

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# %%
from spike_psvae import denoise, vis_utils, waveform_utils, localization, point_source_centering, linear_ae, ptp_vae, stacks
from npx import reg

# %%
dn = denoise.SingleChanDenoiser().load().eval()

# %%
(np.abs(geom - geom[100]).sum(axis=1) <= 200).sum()

# %%
from scipy.spatial.distance import squareform, pdist

# %%
(squareform(pdist(geom, metric="minkowski", p=1)) <= 200)[100].sum()

# %%
from joblib import Parallel, delayed
from spike_psvae.waveform_utils import get_local_geom
from scipy.optimize import minimize

BOUNDS_NP1 = [(-100, 132), (1e-4, 250), (-100, 100)]
BOUNDS = {20: BOUNDS_NP1, 15: BOUNDS_NP1}

# how to initialize y, alpha?
Y0, ALPHA0 = 20.0, 1000.0

def localize_ptp(
    ptp,
    firstchan,
    maxchan,
    geom,
):
    """Find the localization result for a single ptp vector

    Arguments
    ---------
    ptp : np.array (2 * channel_radius,)
    maxchan : int
    geom : np.array (total_channels, 2)

    Returns
    -------
    x, y, z_rel, z_abs, alpha
    """
    n_channels = ptp.size
    local_geom, z_maxchan = get_local_geom(
        geom,
        firstchan,
        maxchan,
        n_channels,
        return_z_maxchan=True,
    )

    # initialize x, z with CoM
    ptp = ptp.astype(float)
    local_geom = local_geom.astype(float)
    ptp_p = ptp / ptp.sum()
    xcom, zcom = (ptp_p[:, None] * local_geom).sum(axis=0)
    maxptp = ptp.max()

    def ptp_at(x, y, z, alpha):
        return alpha / np.sqrt(
            np.square(x - local_geom[:, 0])
            + np.square(z - local_geom[:, 1])
            + np.square(y)
        )

    def mse(loc):
        x, y, z = loc
        # q = ptp_at(x, y, z, 1.0)
        # alpha = (q * (ptp / maxptp - delta)).sum() / (q * q).sum()
        duv = np.c_[x - local_geom[:, 0], np.broadcast_to(y, ptp.shape), z - local_geom[:, 1]]
        X = duv / np.square(duv).sum(axis=1, keepdims=True)
        beta = np.linalg.solve(X.T @ X, X.T @ (ptp / maxptp))
        qtq = X @ beta
        return (
            np.square(ptp / maxptp - qtq).mean()
            # np.square(ptp / maxptp - delta - ptp_at(x, y, z, alpha)).mean()
            # np.square(np.maximum(0, ptp / maxptp - ptp_at(x, y, z, alpha))).mean()
            # - np.log1p(10.0 * y) / 10000.0
        )

    result = minimize(
        mse,
        x0=[xcom, Y0, zcom],
        bounds=[(-150, 209), (1e-4, 500), (-200, 200)],
    )

    # print(result)
    bx, by, bz_rel = result.x
    # q = ptp_at(bx, by, bz_rel, 1.0)
    # balpha = (ptp * q).sum() / np.square(q).sum()
    duv = np.c_[bx - local_geom[:, 0], np.broadcast_to(by, ptp.shape), bz_rel - local_geom[:, 1]]
    X = duv / np.square(duv).sum(axis=1, keepdims=True)
    beta = np.linalg.solve(X.T @ X, X.T @ ptp)
    balpha = np.linalg.norm(beta)
    return bx, by, bz_rel, geom[maxchan, 1] + bz_rel, balpha


def localize_ptps(
    ptps,
    geom,
    firstchans,
    maxchans,
    n_workers=None,
    pbar=True,
    dipole=False,
):
    """Localize a bunch of waveforms

    waveforms is N x T x C, where C can be either 2 * channel_radius, or
    C can be 384 (or geom.shape[0]). If the latter, the 2 * channel_radius
    bits will be extracted.

    Returns
    -------
    xs, ys, z_rels, z_abss, alphas
    """
    N, C = ptps.shape

    maxchans = maxchans.astype(int)
    firstchans = firstchans.astype(int)

    # handle pbars
    xqdm = tqdm if pbar else lambda a, total, desc: a

    # -- run the least squares
    xs = np.empty(N)
    ys = np.empty(N)
    z_rels = np.empty(N)
    z_abss = np.empty(N)
    alphas = np.empty(N)
    locptp = localize_ptp if dipole else localization.localize_ptp
    with Parallel(n_workers) as pool:
        for n, (x, y, z_rel, z_abs, alpha) in enumerate(
            pool(
                delayed(locptp)(
                    ptp,
                    firstchan,
                    maxchan,
                    geom,
                )
                for ptp, maxchan, firstchan in xqdm(
                    zip(ptps, maxchans, firstchans), total=N, desc="lsq"
                )
            )
        ):
            xs[n] = x
            ys[n] = y
            z_rels[n] = z_rel
            z_abss[n] = z_abs
            alphas[n] = alpha

    return xs, ys, z_rels, z_abss, alphas


# %%
plt.rc("figure", dpi=200)
rg = lambda: np.random.default_rng(0)

# %%
ddir = "/mnt/3TB/charlie/upload_feats/"

# %%
# %ll {{ddir}}

# %%
dset = "cshl048_t_250_300"
root = Path(ddir) / dset

# %%
# %ll {{root}}

# %%
wfs = np.load(root / "wfs.npy")
geom = np.load(root / "np1_channel_map.npy")
res = np.load(root / "localization_results.npy")
firstchans = res[:, 5].astype(int)
maxchans = res[:, 5].astype(int)

# %%
subh5 = "/mnt/3TB/charlie/subtracted_datasets/subtraction__spikeglx_ephysData_g0_t0.imec.ap.normalized_t_250_300.h5"
with h5py.File(subh5) as f:
    owfs = f["cleaned_waveforms"][:]
    owfs, ofcs, _, chans_down = waveform_utils.relativize_waveforms(owfs, f["first_channels"][:], None, geom, maxchans_orig=maxchans, feat_chans=20)


# %%
ptps = wfs.ptp(1)
del wfs

# %%
optps = owfs.ptp(1)

# %%
ox, oy, oz_rel, oz_abs, oalpha = localize_ptps(
    optps,
    geom,
    ofcs,
    maxchans,
    n_workers=15,
    pbar=True,
    dipole=False,
)

# %%
dx, dy, dz_rel, dz_abs, dalpha = localize_ptps(
    ptps,
    geom,
    firstchans,
    maxchans,
    n_workers=15,
    pbar=True,
    dipole=True,
)

# %%
oz_abs.max()

# %%
dz_abs.max()

# %%
ptps_ = torch.tensor(ptps, device="cuda")
loader = torch.utils.data.DataLoader(
    ptps_,
    batch_size=64,
    shuffle=True,
    drop_last=True,
)
cgeom = geom[:20].copy()
cgeom -= cgeom.mean(0)

# %%
zfc = geom[firstchans, 1]

# %%
encoder = stacks.linear_encoder(20, [64, 32, 16], 3, batchnorm=False)
mpae = ptp_vae.PTPVAE(encoder, cgeom, variational=False, dipole=False)
optimizer = torch.optim.RAdam(mpae.parameters(), lr=1e-3)
device = torch.device("cuda")
mpae.to(device)

# %%
global_step = 0
n_epochs = 100
for e in trange(n_epochs):
    tic = time.time()
    losses = []
    for batch_idx, x in enumerate(loader):
        x = x.to(device)

        optimizer.zero_grad()

        recon_x, mu, logvar = mpae(x)
        loss, loss_dict = mpae.loss(x, recon_x, mu, logvar)

        loss.backward()
        optimizer.step()
        loss_ = loss.cpu().detach().numpy()
        losses.append(loss_)
        
        if np.isnan(loss_).any():
            print("NaN")

    if not e % 10:
        gsps = len(loader) / (time.time() - tic)
        print(
            f"Epoch {e}, batch {batch_idx}. Loss {np.array(losses).mean()}, "
            f"Global steps per sec: {gsps}",
            flush=True,
        )

# %%
with torch.no_grad():
    mpx, mpy, mpz_rel, mpalpha = mpae.encode_with_alpha(ptps_)
    mpx = mpx.cpu().numpy()
    mpx += geom[:20, 0].mean()
    mpy = mpy.cpu().numpy()
    mpz_abs = mpz_rel.cpu().numpy() - cgeom[0,1] + zfc
    mpalpha = mpalpha.cpu().numpy()

# %%
encoder = stacks.linear_encoder(20, [64, 32, 16], 3, batchnorm=False)
dpae = ptp_vae.PTPVAE(encoder, cgeom, variational=False, dipole=True)
optimizer = torch.optim.RAdam(dpae.parameters(), lr=1e-3)
device = torch.device("cuda")
dpae.to(device)

# %%
global_step = 0
n_epochs = 100
for e in trange(n_epochs):
    tic = time.time()
    losses = []
    for batch_idx, x in enumerate(loader):
        x = x.to(device)

        optimizer.zero_grad()

        recon_x, mu, logvar = dpae(x)
        loss, loss_dict = dpae.loss(x, recon_x, mu, logvar)

        loss.backward()
        optimizer.step()
        loss_ = loss.cpu().detach().numpy()
        losses.append(loss_)
        
        if np.isnan(loss_).any():
            print("NaN")

    if not e % 10:
        gsps = len(loader) / (time.time() - tic)
        print(
            f"Epoch {e}, batch {batch_idx}. Loss {np.array(losses).mean()}, "
            f"Global steps per sec: {gsps}",
            flush=True,
        )

# %%
plt.figure(figsize=(3, 2))
plt.hist(np.log(dy), bins=100);

# %%
with torch.no_grad():
    dpx, dpy, dpz_rel, dpalpha = dpae.encode_with_alpha(ptps_)
    dpx = dpx.cpu().numpy()
    dpx += geom[:20, 0].mean()
    dpy = dpy.cpu().numpy()
    dpz_abs = dpz_rel.cpu().numpy() - cgeom[0,1] + zfc
    dpalpha = dpalpha.cpu().numpy()

# %%
locs = {
    "monopole bfgs": np.c_[ox, oy, oz_abs, oalpha],
    "dipole bfgs": np.c_[dx, dy, dz_abs, dalpha],
    "monopole ae": np.c_[mpx, mpy, mpz_abs, mpalpha],
    "dipole ae": np.c_[dpx, dpy, dpz_abs, dpalpha],
}

# %%
maxptp = ptps.max(1)

# %%
for name, ls in locs.items():
    plt.figure(figsize=(4, 3))
    vis_utils.plotlocs(*ls.T, maxptp, geom, suptitle=name, ylim=[-2, 6], xlim=[-160,160], zlim=[-20, 3860])
    plt.show()

# %%
for name, ls in locs.items():
    which = ls[:, 2] > 2750
    plt.figure(figsize=(5, 4))
    vis_utils.plotlocs(*ls[which].T, maxptp[which], geom, suptitle=name + " ROI", ylim=[-2, 6], xlim=[-160,160], zlim=[2750, 3860])
    plt.show()

# %%
fig, axes = plt.subplots(3, 4, figsize=(10, 8))
for i, (name, ls) in enumerate((n, l) for n, l in locs.items() if n != "monopole bfgs"):
    x_, y_, za_, alpha_ = ls.T
    zr_ = za_ - geom[maxchans, 1]
    vis_utils.corr_scatter(
        np.c_[x_, np.log(y_), zr_, np.log(alpha_)],
        np.c_[ox, np.log(oy), oz_rel, np.log(oalpha)],
        [f"{name} {dim}" for dim in ("x", "log y", "z rel", "log alpha")],
        [f"mono. bfgs {dim}" for dim in ("x", "log y", "z rel", "log alpha")],
        maxptp,
        1,
        axes=axes[i],
        grid=False,
    )

# %%
