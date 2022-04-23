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
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import time

# %%
from spike_psvae import denoise, vis_utils, waveform_utils, localization, point_source_centering, linear_ae, ptp_vae, stacks
from npx import reg

# %%
plt.rc("figure", dpi=200)
rg = lambda: np.random.default_rng(0)

# %%
sub_h5_path = "/mnt/3TB/charlie/subtracted_datasets/subtraction__spikeglx_ephysData_g0_t0.imec.ap.normalized_t_250_300.h5"
res_bin_path = "/mnt/3TB/charlie/subtracted_datasets/residual__spikeglx_ephysData_g0_t0.imec.ap.normalized_t_250_300.bin"
raw_bin_path = "/mnt/3TB/charlie/.one/openalyx.internationalbrainlab.org/churchlandlab/Subjects/CSHL049/2020-01-08/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec.ap.normalized.bin"
loc_npz_path = "/mnt/3TB/charlie/subtracted_datasets/locs__spikeglx_ephysData_g0_t0.imec.ap.normalized_t_250_300.npz"
feat_npz_path = "/mnt/3TB/charlie/subtracted_datasets/feats__spikeglx_ephysData_g0_t0.imec.ap.normalized_t_250_300_r3.npz"

# %%
subh5 = h5py.File(sub_h5_path, "r")
firstchans = subh5["first_channels"][:]
spike_index = subh5["spike_index"][:]
maxchans = spike_index[:, 1]
geom = subh5["geom"][:]
wfs = subh5["subtracted_waveforms"]
cwfs = subh5["cleaned_waveforms"]
residual = np.memmap(res_bin_path, dtype=np.float32)
residual = residual.reshape(-1, geom.shape[0])
feat_chans = cwfs.shape[2]
cfirstchans = firstchans
cmaxchans = maxchans

# %%
locs = np.load(loc_npz_path)
list(locs.keys())

# %%
maxptp = locs["maxptp"]
# print(which.sum())
ox = locs["locs"][:, 0]
oy = locs["locs"][:, 1]
oalpha = locs["locs"][:, 4]
oza = locs["locs"][:, 3]
ozr = locs["z_reg"][:]
t = locs["t"][:]


# %%
stdwfs, firstchans_std, chans_down = waveform_utils.relativize_waveforms_np1(cwfs[:], cfirstchans, geom, cmaxchans)

# %%
ptps = stdwfs[:].ptp(1)

# %%
ptps.shape

# %%
ptps_ = torch.tensor(ptps, device="cuda")

# %%
loader = torch.utils.data.DataLoader(
    ptps_,
    batch_size=64,
    shuffle=True,
    drop_last=True,
)

# %%
cgeom = geom[:20].copy()
cgeom -= cgeom.mean(0)

# %%
cgeom

# %%
zfc = geom[firstchans_std, 1]
ozfr = oza + cgeom[0,1] - zfc

# %%
encoder = stacks.linear_encoder(20, [64, 32, 16], 3, batchnorm=False)

# %%
ae = ptp_vae.PTPVAE(encoder, cgeom, variational=False)

# %%
optimizer = torch.optim.RAdam(ae.parameters(), lr=1e-3)

# %%
device = torch.device("cuda")

# %%
ae.to(device);

# %%
pgeom = geom.copy()
pgeom[:, 0] -= pgeom[:, 0].mean()

# %%
global_step = 0
n_epochs = 5000
for e in range(n_epochs):
    tic = time.time()
    losses = []
    for batch_idx, x in enumerate(loader):
        x = x.to(device)

        optimizer.zero_grad()

        recon_x, mu, logvar = ae(x)
        loss, loss_dict = ae.loss(x, recon_x, mu, logvar)

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
        
        fig, axes = plt.subplots(5, 5)
        print(x.shape, recon_x.shape, mu.shape)
        vis_utils.plot_ptp(x.cpu().detach().numpy()[:25], axes, "", "k", range(25))
        vis_utils.plot_ptp(recon_x.cpu().detach().numpy()[:25], axes, "", "silver", range(25))
        plt.show()
        
        x, log_y, zz = ae.localize(ptps_)
        x = x.cpu().detach().numpy()
        log_y = log_y.cpu().detach().numpy()
        y = np.exp(log_y)
        zr = zz.cpu().detach().numpy()
        z_abs = zr - cgeom[0,1] + zfc
        plt.figure()
        vis_utils.plotlocs(x, y, z_abs, None, maxptp, pgeom)
        plt.show()
        
        vis_utils.corr_scatter(np.c_[x, log_y, zr], np.c_[ox, np.log(oy), ozfr], ["AE x", "AE logy", "AE zrel"], ["BFGS x", "BFGS logy", "BFGS zrel"], maxptp, 1, "AE Localizations vs. BFGS", grid=False)
        plt.show()

# %%
      
x, log_y, zz = ae.localize(ptps_)
x = x.cpu().detach().numpy()
log_y = log_y.cpu().detach().numpy()
y = np.exp(log_y)
zr = zz.cpu().detach().numpy()
z_abs = zr - cgeom[0,1] + zfc
plt.figure()
vis_utils.plotlocs(x, y, z_abs, None, maxptp, pgeom)

# %%
vis_utils.plotlocs(ox - geom[:,0].mean(), oy, oza, None, maxptp, pgeom)

# %%
