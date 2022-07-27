# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
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
import scipy.linalg as la
from scipy import signal, stats
import spikeinterface.full as si
import neuropixel
import h5py
from pathlib import Path
from scipy.io import loadmat
from spikeglx import Reader
from spike_psvae import subtract, ibme, ibme_corr, ap_filter
import torch
from tqdm.auto import tqdm, trange
import sys; sys.path.append(str(Path("~/neuropixelsLFPregistration/python").expanduser()))
import pixelCSD, lfpreg, batchreg


# %%
# %matplotlib inline
plt.rc("figure", dpi=200)
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# %% [markdown]
# # Params

# %%
ds_root = Path("/local/MotionCorrectionWithNeuropixels/data/NeuropixelsHumanData/Pt02/")
raw_dir = ds_root / "raw"
ppx_dir = ds_root / "ppx"
ppx_dir.mkdir(exist_ok=True)

# %%
# %ll {raw_dir}

# %%
ptname = next(raw_dir.glob("*.ap.bin")).name.split(".")[0]
ptname

# %%
raw_lfp_bin = next(raw_dir.glob("*.lf.bin"))
ppx_lfp_bin = ppx_dir / raw_lfp_bin.name
ppx_lfp_bin

# %%
h = neuropixel.dense_layout()
geom = np.c_[h["x"], h["y"]]

# %%
lfsr = Reader(next(raw_dir.glob("*.lf.bin")))
apsr = Reader(next(raw_dir.glob("*.ap.bin")))

# %%
lfsr.fs, apsr.fs

# %%
lfp0 = np.memmap(raw_lfp_bin, dtype=np.int16).reshape(-1, 385)[:, :-1]
lfp0.shape, lfp0.shape[0] / lfsr.fs

# %%
start = int(250 * lfsr.fs)
plt.imshow(lfp0[start:start + 10 * int(lfsr.fs)].T, aspect=10);

# %% [markdown]
# # KS drift

# %%
with h5py.File(Path("/home/ciw2107/shifts_pat1_4.mat")) as h5:
    for k in h5:
        print(k, h5[k].shape)
    ks_fs = h5["block_size"][()].squeeze()
    pt02_shift = h5["pt02_shift"][:].squeeze()

# %% [markdown]
# # Relevant region

# %%
t_start = 230
s_start = t_start * int(lfsr.fs)
t_end = t_start + pt02_shift.size * ks_fs / apsr.fs
s_end = int(t_end * lfsr.fs)
s_start, s_end, (s_end - s_start) / lfsr.fs


# %% [markdown]
# # LFP

# %%
def zs(x, it=4):
    for _ in range(it):
        x = stats.zscore(x, axis=0)
        x = stats.zscore(x, axis=1)
    return x


# %%
def downsample(x, factor=10):
    return signal.resample(x, x.shape[0] // factor)


# %%
2 / 30_000 * 2

# %%
2 / 2500 * 2

# %%
rmss = ap_filter.run_preprocessing(
    raw_lfp_bin,
    ppx_lfp_bin,
    fs=2500,
    n_channels=384,
    extra_channels=1,
    ptp_len=0,
    bp=(0.5, 250),
    order=3,
    decorr_iter=0,
    # resample_to=30_000,
    resample_to=250,
    lfp_destripe=True,
    csd=False,
    pad_for_filter=2500 * 2,
    # chunk_seconds=1,
    chunk_seconds=12,
    standardize=None,
    in_dtype=np.int16,
    rmss=None,
    geom=geom,
    avg_depth=True,
    debug_imshow=5,
    t_start=230.0,
    t_end=t_end,
    # t_start=230.0 * (2500/30000),
)

# %%
lfp = np.memmap(ppx_lfp_bin, dtype=np.float32).reshape(-1, 384 // 2)
lfp.shape

# %%
# lfp_chunk = downsample(lfp[s_start + 150 * 2500 : s_start + 180 * 2500])
lfp_chunk = lfp[s_start // 10 + 250 * 250 : s_start // 10 + 280 * 250]
plt.imshow(lfp_chunk.T, aspect=5);

# %%
D, C = ibme_corr.calc_corr_decent(lfp_chunk.T, disp=100, device="cuda:1")

# %%
seismic = plt.get_cmap("seismic").copy()
gray = np.array([0.25, 0.25, 0.25, 1])
seismic.set_bad(color=gray)
cividis = plt.get_cmap("cividis").copy()
gray = np.array([0.25, 0.25, 0.25, 1])
cividis.set_bad(color=gray)

# %%
fig, ((aa, ab), (ac, ad)) = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(8, 8))

aa.imshow(D, cmap=seismic, interpolation="nearest")
ab.imshow(C, cmap=cividis, interpolation="nearest");
threshold = 0.4
where = C <= threshold
cD = np.ma.masked_where(C <= threshold, D)
cC = np.ma.masked_where(C <= threshold, C)
ac.imshow(cD, cmap=seismic, interpolation="nearest")
ad.imshow(cC, cmap=cividis, interpolation="nearest");

# %%
for c in tqdm([0.4, 0.5, 0.6, 0.7, 0.8]):
    p = ibme_corr.psolvecorr(D, C, mincorr=c, robust_sigma=0)
    plt.plot(p, label=c)
plt.legend()

# %%
plt.imshow(lfp_chunk.T, aspect=30);
for c in tqdm([0.75, 0.8, 0.85, 0.9]):
    p = ibme_corr.psolvecorr(D, C, mincorr=c, robust_sigma=0)
    plt.plot(p + 192 / 2, label=c, lw=1)
plt.legend()
# plt.plot(p + 192 / 2, color="w")

# %%
p_lfp_chunk = ibme_corr.psolvecorr(D, C, mincorr=0.8, robust_sigma=0)

# %%
csd_chunk = 2 * lfp_chunk[:, 1:-1] - lfp_chunk[:, :-2] - lfp_chunk[:, 2:]
plt.imshow(csd_chunk.T, aspect=10);

# %%
Dc, Cc = ibme_corr.calc_corr_decent(csd_chunk.T, disp=100)

# %%
fig, ((aa, ab), (ac, ad)) = plt.subplots(2, 2, sharey=True, sharex=True, figsize=(4, 4))

aa.imshow(Dc, cmap=seismic, interpolation="nearest")
ab.imshow(Cc, cmap=cividis, interpolation="nearest");
threshold = 0.4
where = Cc <= threshold
cDc = np.ma.masked_where(Cc <= threshold, Dc)
cCc = np.ma.masked_where(Cc <= threshold, Cc)
ac.imshow(cDc, cmap=seismic, interpolation="nearest")
ad.imshow(cCc, cmap=cividis, interpolation="nearest");

# %%
for c in tqdm([0.4, 0.5, 0.6, 0.7, 0.8]):
    p = ibme_corr.psolvecorr(Dc, Cc, mincorr=c, robust_sigma=0)
    plt.plot(p, label=c)
plt.legend()

# %%
plt.imshow(lfp_chunk.T, aspect=30);
for c in tqdm([0.75, 0.8, 0.85, 0.9]):
    p = ibme_corr.psolvecorr(Dc, Cc, mincorr=c, robust_sigma=0)
    plt.plot(p + 192 / 2, label=c, lw=1)
plt.plot(p_lfp_chunk + 192 / 2, color="w", label="lfp", lw=0.5)
plt.legend()

# %%
p_global = ibme_corr.online_register_rigid(lfp.T, mincorr=0.8, disp=100, batch_length=5000, batch_size=128, device="cuda:1")

# %%
p_global.shape

# %%
ks_end = pt02_shift.shape[0] * 30016/30000

# %%
ks_end

# %%
T_samples = 2183292

# %%
our_end = p_global.shape[0] / 250
our_end

# %%
ks_end_ix = int(np.floor((ks_end / our_end) * p_global.shape[0]))
ks_end_ix

# %%
from scipy.interpolate import interp1d

# %%
old_domain = np.linspace(0, 100, num=pt02_shift.shape[0])
new_domain = np.linspace(0, 100, num=ks_end_ix)

# %%
p_ks = interp1d(old_domain, pt02_shift, assume_sorted=True)(new_domain)

# %%
lfp.shape, p_global.shape, p_ks.shape

# %%
ks_end_ix // 30000

# %%
p_ks.min(), p_ks.max(), pt02_shift.min(), pt02_shift.max()

# %%
p_global.min(), p_global.max()

# %%
np.arange(5000, ks_end_ix - 6000, 5000)

# %%
starts = np.arange(5000, ks_end_ix - 6000, 30_000)
ends = starts + 5000

fig, axes = plt.subplots(len(starts), 1, figsize=(5, 10), sharex=True)

for ax, start, end in zip(axes, starts, ends):
    print(start, end)
    ax.imshow(lfp[start:end].T, aspect=0.5 * (end - start) / lfp.shape[1])
    ax.plot(p_global[start:end] - p_global.mean() + 192 / 2, color="w", lw=1)
    ax.plot(p_ks[start:end] - p_ks.mean() + 192 / 2, color="k", lw=1)    

# %%
np.save("p_global_pt02.npy", p_global)

# %% [markdown]
# # AP

# %%
# %ll {raw_dir}

# %%
raw_dir

# %%
ap_dir = Path('/local/MotionCorrectionWithNeuropixels/data/NeuropixelsHumanData/Pt02/ap')
ap_dir.mkdir(exist_ok=True)

# %%
# %rm {ap_dir}/*

# %%
1

# %%
# %ll {ap_dir}

# %%
# !ln -s {raw_dir / "Pt02.imec0.ap.bin"} {ap_dir / "pt02_g0_t0.imec0.ap.bin"}

# %%
# !ln -s {raw_dir / "Pt02.imec0.ap.meta"} {ap_dir / "pt02_g0_t0.imec0.ap.meta"}

# %%
# global kwargs for parallel computing
job_kwargs = dict(
    n_jobs=8,
    chunk_size=30_000,
    progress_bar=True,
)
# read the file
rec = si.read_spikeglx(ap_dir)
rec

# %%
rec = rec.frame_slice(230 * 30_000, None)

# %%
rec

# %%
rec_filtered = si.bandpass_filter(rec, freq_min=300., freq_max=6000.)
rec_cmr = si.common_reference(rec_filtered, reference='global', operator='median')
rec_preprocessed = si.zscore(rec_cmr)

# %%
rec_preprocessed

# %%
si.plot_timeseries(rec_preprocessed, time_range=(100, 110), channel_ids=rec.channel_ids[50:60])

# %%
noise_levels = si.get_noise_levels(rec_preprocessed, return_scaled=False)
fig, ax = plt.subplots(figsize=(3, 2))
ax.hist(noise_levels, bins=10)
ax.set_title('noise across channel')

# %%
preprocess_folder = Path("/local/ppx")
rec_preprocessed.save(folder=preprocess_folder, **job_kwargs)
rec_preprocessed = si.load_extractor(preprocess_folder)

# %%
from spikeinterface.sortingcomponents.peak_detection import detect_peaks

# %%
peaks = detect_peaks(
    rec_preprocessed,
    method='locally_exclusive',
    local_radius_um=100,
    peak_sign='both',
    detect_threshold=5,
    noise_levels=noise_levels,
    **job_kwargs,
)

# %%
peaks.shape, peaks.dtype

# %%
from spikeinterface.sortingcomponents.peak_localization import localize_peaks

# %%
# %rm -rf /local/ppx

# %%
peak_locations = localize_peaks(
    rec_preprocessed,
    peaks,
    ms_before=0.3,
    ms_after=0.6,
    method='monopolar_triangulation',
    method_kwargs={
        'local_radius_um': 100.,
        'max_distance_um': 1000.,
        'optimizer': 'minimize_with_log_penality',
    },
    **job_kwargs,
)


# %%
def clip_values_for_cmap(x):
    low, high = np.percentile(x, [5, 95])
    return np.clip(x, low, high)


# %%
fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(15, 10))
ax = axs[0]
si.plot_probe_map(rec_preprocessed, ax=ax)
ax.scatter(peak_locations['x'], peak_locations['y'], c=clip_values_for_cmap(peaks['amplitude']), s=1, alpha=0.002, cmap=plt.cm.plasma)
ax.set_xlabel('x')
ax.set_ylabel('y')
if 'z' in peak_locations.dtype.fields:
    ax = axs[1]
    ax.scatter(peak_locations['z'], peak_locations['y'], c=clip_values_for_cmap(peaks['amplitude']), s=1, alpha=0.002, cmap=plt.cm.plasma)
    ax.set_xlabel('z')
    ax.set_xlim(0, 150)
ax.set_ylim(1800, 2500)

# %%
fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(15, 10))
ax = axs[0]
si.plot_probe_map(rec_preprocessed, ax=ax)
ax.scatter(peak_locations['x'], peak_locations['y'], c=clip_values_for_cmap(peaks['amplitude']), s=1, alpha=0.002, cmap=plt.cm.plasma)
ax.set_xlabel('x')
ax.set_ylabel('y')
if 'z' in peak_locations.dtype.fields:
    ax = axs[1]
    ax.scatter(peak_locations['z'], peak_locations['y'], c=clip_values_for_cmap(peaks['amplitude']), s=1, alpha=0.002, cmap=plt.cm.plasma)
    ax.set_xlabel('z')
    ax.set_xlim(0, 150)
# ax.set_ylim(1800, 2500)

# %%
fig, ax = plt.subplots()
x = peaks['sample_ind'] / rec_preprocessed.get_sampling_frequency()
y = peak_locations['y']
ax.scatter(x, y, s=1, c=clip_values_for_cmap(peaks['amplitude']), cmap=plt.cm.plasma, alpha=0.25)
# ax.set_ylim(1300, 2500)

# %%
fig, ax = plt.subplots()
which = np.flatnonzero(peaks['amplitude'] > 10)
x = peaks[which]['sample_ind'] / rec_preprocessed.get_sampling_frequency()
y = peak_locations['y'][which]
ax.scatter(x, y, s=1, c=clip_values_for_cmap(peaks[which]['amplitude']), cmap=plt.cm.plasma, alpha=0.25)
# ax.set_ylim(1300, 2500)

# %%
