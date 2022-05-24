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
import h5py
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
import scipy
from scipy.io import loadmat
from scipy.stats import zscore
import scipy.linalg as la

# %%
from scipy.spatial.distance import squareform, pdist

# %%
from ibllib.ephys import neuropixel
from ibllib.dsp import voltage
# from ibllib.dsp.voltage import destripe, decompress_destripe_cbin
from ibllib.dsp.utils import rms
from ibllib.io.spikeglx import Reader, read_meta_data, _map_channels_from_meta, _geometry_from_meta, _get_neuropixel_major_version_from_meta

# %%

# %%
from spike_psvae import subtract, ibme, ibme_corr

# %%
plt.rc("figure", dpi=200)

# %%
ap_bin = "/mnt/3TB/charlie/et/pacman-task_c_220218_neu_insertion_g0_imec0/pacman-task_c_220218_neu_insertion_g0_t0.imec0.ap.bin"

# %%
std_bin = ap_bin = "/mnt/3TB/charlie/et/pacman-task_c_220218_neu_insertion_g0_imec0/pacman-task_c_220218_neu_insertion_g0_t0.imec0.normalized.ap.bin"

# %% [markdown]
# # preprocessing

# %%
raw = np.memmap(ap_bin, dtype=np.int16)
raw = raw.reshape(-1, 385)[:, :-1]
raw_shape = raw.shape
del raw

# %%
raw_shape, raw_shape[0] // 30000


# %%
def destripe(x, fs, neuropixel_version=1, butter_kwargs=None, k_kwargs=None, channel_labels=None, k_filter=True):
    butter_kwargs, k_kwargs, spatial_fcn = voltage._get_destripe_parameters(fs, butter_kwargs, k_kwargs, k_filter)
    if channel_labels is True:
        channel_labels, _ = voltage.detect_bad_channels(x, fs)
    # butterworth
    sos = scipy.signal.butter(**butter_kwargs, output='sos')
    x = scipy.signal.sosfiltfilt(sos, x)
    # channel interpolation
    # apply spatial filter only on channels that are inside of the brain
    if channel_labels is not None:
        x = voltage.interpolate_bad_channels(x, channel_labels, h)
        inside_brain = np.where(channel_labels != 3)[0]
        x[inside_brain, :] = spatial_fcn(x[inside_brain, :])  # apply the k-filter
    else:
        x = spatial_fcn(x)
    return x


# %%
with open(std_bin, "wb") as f:
    for i, s in enumerate(trange(0, raw_shape[0], 30000)):
        se = min(raw_shape[0], s+30000)
        x = np.fromfile(
            raw_data_bin,
            dtype=np.int16,
            count=385 * (se - s),
            offset=np.dtype(np.int16).itemsize * 385 * s,
        )
        x = x.reshape(se - s, 385)
        x = x[:, :384].astype(np.float32)
        z = destripe(x.T, fs=30000, channel_labels=None, neuropixel_version=None)
        # chunk = z * intnorm
        # for i in range(4):
        #     z = zscore(z, axis=1)
        #     z = zscore(z, axis=0)
        #     chunk[:, :] = np.dot(wrot, chunk[:, :])
        zscore(z[:, :], axis=1).T.astype(np.float32).tofile(f)


# %%
std = np.memmap(std_bin, mode="r", dtype=np.float32)
std = std.reshape(-1, 384)
offset = 1000
plt.imshow(std[offset * 30000 + 20000:offset * 30000 + 21000].T);
plt.colorbar(shrink=0.3)
plt.title("Filtered AP band");

# %%
chanmap = loadmat("/mnt/3TB/charlie/et/neuropixNHP_kilosortChanMap_v1.mat")
geom = np.c_[chanmap["xcoords"], chanmap["ycoords"]]

# %%
sub_h5 = subtract.subtraction(
    std_bin,
    "/mnt/3TB/charlie/et/deadreckoning/",
    geom=geom,
    n_sec_pca=20,
    # t_start=10,
    # t_end=1010,
    sampling_rate=30_000,
    thresholds=[12, 10, 8, 6, 5],
    denoise_detect=True,
    neighborhood_kind="box",
    extract_box_radius=200,
    dedup_spatial_radius=70,
    enforce_decrease_kind="radial",
    n_jobs=2,
    save_residual=False,
    save_waveforms=False,
    do_clean=True,
    localization_kind="logbarrier",
    localize_radius=100,
    loc_workers=4,
    # overwrite=True,
    random_seed=0,
)

# %%
sub_h5 = "/mnt/3TB/charlie/et/deadreckoning/subtraction_pacman-task_c_220218_neu_insertion_g0_t0.imec0.normalized.ap_t_0_None.h5"

# %%
with h5py.File(sub_h5) as h5:
    for k in h5:
        print(k, h5[k].dtype, h5[k].shape)

# %%
with h5py.File(sub_h5) as h5:
    z_abs = h5["localizations"][:, 2]
    x = h5["localizations"][:, 0]
    y = h5["localizations"][:, 1]
    times = (h5["spike_index"][:, 0] - h5["start_sample"][()]) / 30_000
    maxptps = h5["maxptps"][:]

# %%
r0, dd, tt = ibme.fast_raster(maxptps, z_abs, times)

# %%
plt.imshow(np.clip(r0, 0, 13), aspect=0.5 * r0.shape[1] / r0.shape[0], origin="lower", cmap=plt.cm.cubehelix)
plt.ylabel("

# %%
gc.collect()

# %%
torch.cuda.empty_cache()

# %%
d_reg, p, extra = ibme.register_rigid(
    maxptps,
    z_abs,
    times,
    denoise_sigma=0.01,
    disp=1000,
    batch_size=8,
    corr_threshold=0.6,
    return_extra=True,
)
D = extra["D"]
C = extra["C"]

# %%
fig, (aa, ab) = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
im = aa.imshow(D, cmap=plt.cm.seismic)
plt.colorbar(im, ax=aa, shrink=0.3)
aa.set_title("Displacement estimates", fontsize=10)
im = plt.imshow(C, cmap=plt.cm.cubehelix)
plt.colorbar(im, ax=ab, shrink=0.3)
ab.set_title("Norm. xcorr at displacement", fontsize=10)

for ax in (aa, ab):
    plt.setp(ax.get_xticklabels(), fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
plt.show()
    

# %%
# 

# %%
thresh = 0.6

S = la.toeplitz(np.r_[np.ones(100), np.zeros(D.shape[0] - 100)])
S *= C > thresh

fig, (aa, ab) = plt.subplots(1, 2, figsize=(6, 3), sharey=True)
im = aa.imshow(D * S, cmap=plt.cm.seismic)
plt.colorbar(im, ax=aa, shrink=0.3)
aa.set_title("Displacement estimates", fontsize=10)
im = plt.imshow(C * S, cmap=plt.cm.cubehelix)
plt.colorbar(im, ax=ab, shrink=0.3)
ab.set_title("Norm. xcorr at displacement", fontsize=10)

for ax in (aa, ab):
    plt.setp(ax.get_xticklabels(), fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
plt.show()
    

# %%
for dt in [30, 50, 100]:
    S = la.toeplitz(np.r_[np.ones(dt), np.zeros(D.shape[0] - dt)])
    plt.scatter(np.arange(D.shape[0]), ibme_corr.psolvecorr(D, C * S, 0.6), s=0.1, label=dt)
plt.legend()

# %%
dt = 30
S = la.toeplitz(np.r_[np.ones(dt), np.zeros(D.shape[0] - dt)])
p = ibme_corr.psolvecorr(D, C * S, 0.6)


# %%
print(p.shape)
p_ = scipy.signal.medfilt(p, 9)
print(p_.shape)

# %%
plt.scatter(np.arange(len(p)), p, s=1)
plt.scatter(np.arange(len(p)), p_, s=0.1)

# %%
warps = scipy.interpolate.interp1d(tt + 0.5, p_, fill_value="extrapolate")(times)
z_reg = z_abs - warps
z_reg -= z_reg.min()

# %%
z_reg_nr, total_shift = ibme.register_nonrigid(
    maxptps,
    z_reg,
    times,
    robust_sigma=0.5,
    denoise_sigma=0.01,
    disp=24000,
    n_windows=100,
    batch_size=8,
    corr_threshold=0.6,
    rigid_init=False,
)

# %%

# %%
r1, *_ = ibme.fast_raster(maxptps, z_reg, times)

# %%
# np.savez("tmp.npz", maxptps=maxptps, z_reg_nr=z_reg_nr, times=times)
with np.load("tmp.npz") as npz:
    maxptps = npz["maxptps"]
    z_reg_nr = npz["z_reg_nr"]
    times = npz["times"]

# %%
r2, *_ = ibme.fast_raster(maxptps, z_reg_nr, times)

# %%

# %%
plt.figure(figsize=(5, 25))
plt.imshow(np.clip(r2, 0, 13), aspect=5 * r2.shape[1] / r2.shape[0], origin="lower", cmap=plt.cm.cubehelix)

# %%
z_reg.shape

# %%
fig = plt.figure(figsize=(5, 250))
cmps = np.clip(maxptps, 3, 13)
nmps = 0.25 + 0.74 * (cmps - cmps.min()) / (cmps.max() - cmps.min())
plt.scatter(x, z_reg, c=cmps, alpha=nmps, s=0.1)
plt.axis("on")
plt.xlim([-95, 95])
plt.yticks(list(range(0, 24000, 200)))
fig.savefig("/mnt/3TB/charlie/figs/corticalcolumn.png", dpi=200, bbox_inches="tight", transparent=False, facecolor="white")

# %%
mos = """\
aaaa.e
.....e
bbbb.e
.....e
p.dc.e
"""
fig, axes = plt.subplot_mosaic(
    mos,
    figsize=(10, 8),
    gridspec_kw=dict(
        height_ratios=[1, 0.05, 1.5, 0.25, 0.95],
        width_ratios=[1, 0.0, 1, 1, 0.15, 1],
        hspace=0.15,
        wspace=0.15,
    )
)

axes["a"].get_shared_x_axes().join(axes["a"], axes["b"])
axes["a"].imshow(
    np.clip(r0, 0, 13), aspect=0.25 * r0.shape[1] / r0.shape[0], origin="lower", cmap=plt.cm.cubehelix
)
axes["a"].set_xticks([])
axes["a"].set_title("Original raster")
axes["a"].set_ylabel("probe depth (um)")
axes["b"].imshow(
    np.clip(r1, 0, 13), aspect=1.5 * 0.25 * r1.shape[1] / r1.shape[0], origin="lower", cmap=plt.cm.cubehelix
)
axes["b"].set_title("Registered raster")
axes["b"].set_ylabel("registered depth (um)")
axes["b"].set_xlabel("time (s)")

axes["p"].scatter(np.arange(len(p_)), p_, s=0.2)
axes["p"].set_ylabel("est. displacement")
axes["p"].set_xlabel("time (s)")
axes["p"].set_box_aspect(1)

im = axes["d"].imshow(D, cmap=plt.cm.seismic)
plt.colorbar(im, ax=axes["d"], shrink=0.3, pad=0.05)
axes["d"].set_title("Pair displacements", fontsize=10)
im = axes["c"].imshow(C, cmap=plt.cm.cubehelix)
plt.colorbar(im, ax=axes["c"], shrink=0.3, pad=0.05)
axes["c"].set_title("Max norm. xcorr", fontsize=10)    

axes["d"].get_shared_y_axes().join(axes["d"], axes["c"])
axes["c"].set_yticks([])

axes["e"].scatter(x[::10], z_reg[::10], c=cmps[::10], alpha=nmps[::10], s=0.1, rasterized=True)
axes["e"].set_ylabel("registered depth (um)", labelpad=-16)
axes["e"].set_xlabel("x (um)")
axes["e"].set_title("Registered xz locations")
axes["e"].set_ylim([z_reg.min(), z_reg.max()])
axes["e"].set_yticks([0, 8000, 16000, 24000])
axes["e"].set_xlim([-95, 95])

for ax in axes.values():
    plt.setp(ax.get_xticklabels(), fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    
fig.savefig("/mnt/3TB/charlie/figs/et_insertion_panel.pdf", dpi=300, bbox_inches="tight")

# %%
