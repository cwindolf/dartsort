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
from pathlib import Path

# %%
from tqdm.auto import trange, tqdm

# %%
from spike_psvae import denoise, vis_utils, waveform_utils, localization, point_source_centering, linear_ae, ibme, detect, subtract
from npx import reg

# %%
plt.rc("figure", dpi=200)
rg = lambda: np.random.default_rng(0)

# %%
# %ll -h /mnt/3TB/charlie/subtracted_datasets/CSHL049/

# %%
dsdir = Path("/mnt/3TB/charlie/subtracted_datasets/CSHL049_dnd/")
openalyx = Path("/mnt/3TB/charlie/.one/openalyx.internationalbrainlab.org/")

sub_h5_path = dsdir / "subtraction__spikeglx_ephysData_g0_t0.imec.ap.normalized_t_250_350.h5"
res_bin_path = dsdir / "residual__spikeglx_ephysData_g0_t0.imec.ap.normalized_t_250_350.bin"
raw_bin_path = openalyx / "churchlandlab/Subjects/CSHL049/2020-01-08/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec.ap.normalized.bin"

# %%
# subh5 = h5py.File("/mnt/3TB/charlie/subtracted_datasets/churchlandlab_CSHL049_p7_t_2000_2010.h5", "r")
subh5 = h5py.File(sub_h5_path, "r")
spike_index = subh5["spike_index"][:]
maxchans = spike_index[:, 1]
geom = subh5["geom"][:]
channel_index = subh5["channel_index"][:]
wfs = subh5["subtracted_waveforms"]
cwfs = subh5["cleaned_waveforms"]
locs = subh5["localizations"][:]
maxptps = subh5["maxptps"][:]
z_reg = subh5["z_reg"][:]
dispmap = subh5["dispmap"]

residual = np.memmap(res_bin_path, dtype=np.float32)
residual = residual.reshape(-1, geom.shape[0])
feat_chans = cwfs.shape[2]

N, T, C = cwfs.shape
num_channels = len(geom)

# feat_chans = 20
# if "cleaned_first_channels" in subh5:
#     cfirstchans = subh5["first_channels"][:]
#     cmaxchans = subh5["spike_index"][:, 1]
#     feat_chans = cwfs.shape[-1]
# else:
#     cwfs, cfirstchans, cmaxchans, chans_down = waveform_utils.relativize_waveforms(
#         cwfs,
#         firstchans,
#         None,
#         geom,
#         feat_chans=feat_chans,
#     )

# relativize time
spike_index[:, 0] -= subh5["start_sample"][()]

# %%
(spike_index[1:, 0] >= spike_index[:-1, 0]).all()

# %%
residual.shape, subh5["end_sample"][()] - subh5["start_sample"][()]

# %%
raw = np.memmap(
    raw_bin_path,
    dtype=np.float32,
    mode="r",
)
raw = raw.reshape(-1, 384)
raw = raw[subh5["start_sample"][()]:subh5["end_sample"][()]]

# %%

# %%

# %%
snip = raw[30000:60000]

# %%
snip.mean(axis=1)

# %%
ddci = dci = subtract.make_channel_index(geom, 70, steps=2)

# %%
dci = subtract.make_channel_index(geom, 70)
d = detect.Detect(dci)
d.load("../pretrained/detect_np1.pt")

# %%
dn = denoise.SingleChanDenoiser()
dn.load()

# %%
dn.conv1[0].kernel_size

# %%
import torch

# %%
si = d.get_spike_times(torch.as_tensor(snip[400:1000].copy()))

# %%
si21, _ = detect.voltage_threshold(snip[400:1000], 1)
si23, _ = detect.voltage_threshold(snip[400:1000], 3)
si2, _ = detect.voltage_threshold(snip[400:1000], 2)

# %%
v = snip[400:1000][si[:, 0], si[:, 1]]

# %%
v.mean()

# %%
dci.shape


# %%
def shift_channels(signal, shifts):
    """Shifts each channel of the signal according to given shifts.
    params:
    -------
    signal: np.ndarray with shape (#channels, #time)
    shifts: np.array with size #channels
    returns:
    --------
    a copy of the shifted signal according to the given shifts.
    """
    n_chan, size = signal.shape
    max_shift = shifts.max()
    shifted_signal_size = size + max_shift
    shifted_signal = np.zeros([n_chan, shifted_signal_size], dtype=signal.dtype)
    # Getting shifted indices.
    ix1 = np.tile(np.arange(n_chan)[:, None], size)
    ix2 = np.arange(size) + shifts[:, None]
    shifted_signal[ix1, ix2] = signal
    return shifted_signal


# %%
import torch.nn.functional as F

# %%
F.pad(torch.arange(35).reshape(5, 7), (0, 1)).shape

# %%
x = torch.tensor([[1, 1], [0, 2], [0, 2], [0, 3]])
x

# %%
torch.unique(x, dim=0)

# %%
which = which = (5 < (si[:, 0])) & (4 < (600 - si[:, 0]))
shiftsi = si[which].clone()
tix = shiftsi[:, 0, None] + np.arange(-5, 4)[None, :]
dts = snip[400:1000][tix, shiftsi[:, 1, None]]

# %%
shifts = dts.argmin(1) - 5
shifts

# %%
shiftsi[:, 0] += shifts

# %%
which = (42 < (si[:, 0])) & (79 < (600 - si[:, 0]))
si = si[which]

# %%
# mct = 
# mct.argmin(1)

# %%
wf = torch.as_tensor(np.pad(snip[400:1000], [(0, 0), (0, 1)])[
    si[:, 0, None, None] + np.arange(-42, 79)[None, :, None],
    dci[si[:, None, 1]],
])
v = dn(
    wf.permute(0, 2, 1).reshape(-1, 121)
).detach().numpy()
v = v.reshape(wf.shape).ptp(1).ptp(1)

# %%
shiftv = snip[400:1000][shiftsi[:, 0], shiftsi[:, 1]]

# %%
v

# %%
v.mean()

# %%
plt.figure(figsize=(8, 8))
c = plt.imshow(snip[400:1000].T, cmap=plt.cm.viridis, aspect=3, interpolation="none")
# plt.scatter(*si[:][v > 4].T, s=1, c="r", label="nn detect ptp>4")
# plt.scatter(*si21.T, s=1, c=plt.cm.Reds(0.1), label="voltage detect threshold 1")
plt.scatter(*si2.T, s=1, c=plt.cm.Reds(0.5), label="voltage detect threshold 2")
plt.scatter(*si23.T, s=1, c=plt.cm.Reds(0.9), label="voltage detect threshold 3")
plt.scatter(*si.T, s=1, c="b", label="nn detections")
# plt.scatter(*shiftsi[shiftv < -3].T, s=1, c="purple", label="nn detections")
plt.colorbar(c)
plt.legend(loc="upper center", bbox_to_anchor=[0.5, 1.15])

# %%

# %%
eci = []
for c in range(384):
    low = max(0, c - 40 // 2)
    low = min(384 - 40, low)
    eci.append(
        np.arange(low, low + 40)
    )
eci = np.array(eci)

# %%
eci

# %%
n_channels=20

# %% tags=[]
subset = np.empty(shape=eci.shape, dtype=bool)
pgeom = np.pad(geom, [(0, 1), (0, 0)], constant_values=-2 * geom.max())
for c in range(len(geom)):
    if n_channels is not None:
        print(c, c - n_channels // 2)
        low = max(0, c - n_channels // 2)
        low = min(len(geom) - n_channels, low)
        high = min(len(geom), low + n_channels)
        print(c, low, high)
        subset[c] = (low <= eci[c]) & (eci[c] < high)
    elif radius is not None:
        dists = cdist([geom[c]], pgeom[eci[c]]).ravel()
        subset[c] = dists <= radius
    else:
        subset[c] = True

# %%
subset.sum(axis=1)

# %%
plt.figure(figsize=(8, 8))
c = plt.imshow(snip[400:1000].T, cmap=plt.cm.viridis, aspect=3, interpolation="none")
# plt.scatter(*si[:][v > 4].T, s=1, c="r", label="nn detect ptp>4")
# plt.scatter(*si21.T, s=1, c=plt.cm.Reds(0.1), label="voltage detect threshold 1")
# plt.scatter(*si2.T, s=1, c=plt.cm.Reds(0.5), label="voltage detect threshold 2")
# plt.scatter(*si.T, s=1, c="b", label="nn detections")
plt.scatter(*shiftsi[shiftv < -3].T, s=5, c="magenta", label="nn detections, trough aligned, threshold 3", marker="x")
plt.scatter(*si23.T, s=1, c=plt.cm.Reds(0.9), label="voltage detect threshold 3")
plt.colorbar(c)
plt.legend(loc="upper center", bbox_to_anchor=[0.5, 1.15])

# %%
plt.figure(figsize=(8, 8))
c = plt.imshow(snip[400:1000].T, cmap=plt.cm.viridis, aspect=3, interpolation="none")
# plt.scatter(*si[:][v > 4].T, s=1, c="r", label="nn detect ptp>4")
# plt.scatter(*si21.T, s=1, c=plt.cm.Reds(0.1), label="voltage detect threshold 1")
# plt.scatter(*si2.T, s=1, c=plt.cm.Reds(0.5), label="voltage detect threshold 2")
# plt.scatter(*si.T, s=1, c="b", label="nn detections")
plt.scatter(*shiftsi[shiftv < -2].T, s=5, c="magenta", label="nn detections, trough aligned, threshold 2", marker="x")
plt.scatter(*si2.T, s=1, c=plt.cm.Reds(0.9), label="voltage detect threshold 2")
plt.colorbar(c)
plt.legend(loc="upper center", bbox_to_anchor=[0.5, 1.15])

# %%
rrr = torch.as_tensor(snip[400:1000].copy())
plt.imshow(d.forward_recording(rrr).detach().numpy().T)
plt.scatter(*(d.get_spike_times(rrr).T), c="b", s=1)

# %%
sia, _ = detect.voltage_detect_and_deduplicate(snip[400:1000].copy(), 4, ddci, 0)
sib, _ = detect.nn_detect_and_deduplicate(snip[400:1000].copy(), 4, ddci, 0, d, dn)

plt.figure(figsize=(8, 8))
c = plt.imshow(snip[400:1000].T, cmap=plt.cm.viridis, aspect=3, interpolation="none")
plt.scatter(*sib.T, s=5, c="magenta", label="deduplicated nn detections, trough aligned, threshold 2", marker="x")
plt.scatter(*sia.T, s=1, c=plt.cm.Reds(0.9), label="deduplicated voltage detect threshold 2")
plt.colorbar(c)
plt.legend(loc="upper center", bbox_to_anchor=[0.5, 1.15])

# %%
snip[400:1000].shape

# %%
dtdn = detect.DenoiserDetect(dn)

# %%
dtdn.to("cuda")

# %%
import os; os.environ["CUDA_LAUNCH_BLOCKING"]="1"

# %%
ptp = dtdn.forward_recording(torch.as_tensor(snip[400:1000].copy(), device="cuda"))

# %%
ptp.shape

# %%
ptp = ptp.detach().cpu().numpy()

# %%
plt.imshow(ptp.T)
plt.colorbar()

# %%

# %%
sia, _ = detect.voltage_detect_and_deduplicate(snip[400:1000].copy(), 4, ddci, 0)
sib, _ = detect.nn_detect_and_deduplicate(snip[400:1000].copy(), 4, ddci, 0, d, dn)
t, c, _ = detect.denoiser_detect_dedup(snip[400 - 42:1000 + 78].copy(), 3, dtdn, channel_index=ddci)
sic = np.c_[t.cpu().numpy(), c.cpu().numpy()]

plt.figure(figsize=(8, 8))
c = plt.imshow(snip[400:1000].T, cmap=plt.cm.viridis, aspect=3, interpolation="none")
plt.scatter(*sib.T, s=5, c="magenta", label="deduplicated nn detections, trough aligned, threshold 2", marker="x")
plt.scatter(*sia.T, s=1, c=plt.cm.Reds(0.9), label="deduplicated voltage detect threshold 2")
plt.scatter(*sic.T, s=1, c=plt.cm.Blues(0.9), label="denoiser-detect PTP threshold 2")
plt.colorbar(c)
plt.legend(loc="upper center", bbox_to_anchor=[0.5, 1.15])

# %%

# %%
show = rg().choice(N, replace=False, size=16)
show.sort()

# %%
fig, axes = plt.subplots(4, 4)
vis_utils.plot_ptp(wfs[show].ptp(1), axes, "", "k", "abcdefghijklmnop")
vis_utils.plot_ptp(cwfs[show].ptp(1), axes, "", "purple", "abcdefghijklmnop")

# %%
for ix in show:
    fig = plt.figure(figsize=(20, 2.5))
    cwf = cwfs[ix, :82]
    plt.plot(wfs[ix, :82].T.ravel())
    plt.plot(cwf.T.ravel())
    for j in range(39):
        plt.axvline(82 + 82*j, color = 'black')
    plt.show()

# %%
for ix in show:
    print(ix, channel_index.shape, wfs[ix].shape, cwfs[ix].shape)

# %%
mosaic = """\
adbcx
yyyyy
zzzzz
"""
def subfig(ix):
    t, mc = spike_index[ix]
    print(t, mc)
    
    chans = channel_index[mc]
    goodchans = chans < num_channels
    chans = chans[goodchans]
    wf = wfs[ix][:, goodchans]
    T, C = wf.shape
    dn2 = cwfs[ix][:, goodchans]
    T_, C_ = dn2.shape
    
    
    raw_ix = raw[t - 42 : t + 79, chans]
    res_ix = residual[t - 42 : t + 79, chans]
    mcr = np.flatnonzero(chans == mc)[0]
    print(raw_ix.shape, res_ix.shape)
    print((res_ix[:, mcr] + wf[:, mcr]).argmin())
    
    vmin = min([v.min() for v in (raw_ix, res_ix, dn2, wf)])
    vmax = max([v.max() for v in (raw_ix, res_ix, dn2, wf)])
    
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(6, 5), gridspec_kw=dict(height_ratios=[2, 0.5, 0.5]))
    for k in "dbc":
        axes[k].set_yticks([])
    axes["a"].imshow(raw_ix[20:-20], cmap="RdBu_r", vmin=min(-vmax, vmin), vmax=max(-vmin, vmax))
    axes["d"].imshow(wf[20:-20], cmap="RdBu_r", vmin=min(-vmax, vmin), vmax=max(-vmin, vmax))
    axes["b"].imshow(res_ix[20:-20], cmap="RdBu_r", vmin=min(-vmax, vmin), vmax=max(-vmin, vmax))
    im = axes["c"].imshow(dn2[20:-20], cmap="RdBu_r", vmin=min(-vmax, vmin), vmax=max(-vmin, vmax))
    cbar = plt.colorbar(im, ax=[axes[k] for k in "abcd"], shrink=0.5)
    cpos = cbar.ax.get_position()
    cpos.x0 = cpos.x0 - 0.02
    cbar.ax.set_position(cpos)
    axes["a"].set_title("raw", fontsize=8)
    axes["b"].set_title("residual", fontsize=8)
    axes["c"].set_title("denoised", fontsize=8)
    axes["d"].set_title("subtracted", fontsize=8)
    for k in "abcd":
        axes[k].set_xticks([0, C_])
    axes["a"].axhline(22, lw=1, c="k")
        
    vis_utils.plot_single_ptp_np2(raw_ix.ptp(0), axes["x"], "raw", "k", "")
    vis_utils.plot_single_ptp_np2(res_ix.ptp(0), axes["x"], "residual", "silver", "")
    vis_utils.plot_single_ptp_np2(wf.ptp(0), axes["x"], "subtracted", "g", "")
    vis_utils.plot_single_ptp_np2((wf + res_ix).ptp(0), axes["x"], "cleaned", "b", "")
    vis_utils.plot_single_ptp_np2(dn2.ptp(0), axes["x"], "denoised", "r", "")
    axes["x"].set_ylabel("ptp", labelpad=0)
    axes["x"].set_xticks([0, C_//2])
    axes["x"].set_box_aspect(1)
    pos = axes["x"].get_position()
    print(pos)
    pos.y0 = axes["c"].get_position().y0 - 0.075
    print(pos)
    axes["x"].set_position(pos)
    axes["x"].legend(loc="upper center", bbox_to_anchor=(0.5, 1.9), fancybox=False, frameon=False)
    
    cshow = 6
    axes["y"].plot(raw_ix[:82, :].T.flatten(), "k", lw=0.5)
    axes["y"].plot(wf[:82, :].T.flatten(), "g", lw=0.5)
    axes["y"].plot(res_ix[:82, :].T.flatten(), "silver", lw=0.5)
    axes["y"].set_xlim([0, dn2[:82, :].size])
    for j in range(C):
        axes["y"].axvline(82 + 82*j, color="k", lw=0.5)
    axes["y"].set_xticks([])
    
    # axes["z"].plot(raw_ix[:82, :].T.flatten(), "k", lw=0.5)
    axes["z"].plot((res_ix + wf)[:82, :].T.flatten(), "b", lw=0.5)
    axes["z"].plot(dn2[:82, :].T.flatten(), "r", lw=0.5)
    # axes["z"].plot(bcwfs[ix, :82, :].T.flatten(), "orange", lw=0.5)
    axes["z"].set_xlim([0, dn2[:82, :].size])
    for j in range(C):
        axes["z"].axvline(82 + 82*j, color="k", lw=0.5)
    axes["z"].set_xticks([])
    return fig

# %% tags=[]
for ix in show:
    print(ix)
    subfig(ix)
    plt.show()

# %%
x = locs[:, 0]
y = locs[:, 1]
z_abs = locs[:, 2]
alpha = locs[:, 3]
z_rel = locs[:, 4]
times = spike_index[:, 0] / 30000

# %%
cm = plt.cm.viridis
plt.figure()
plt.hist(maxptps, bins=100)
plt.show()
which = slice(None)

vis_utils.plotlocs(x, y, z_abs, alpha, maxptps, geom, which=maxptps > 3, suptitle="CSHL049")

# %%
vis_utils.plotlocs(x, y, z_abs, alpha, maxptps, geom, which=z_abs > 2800, suptitle="CSHL049")

# %%
vis_utils.plotlocs(x, y, z_reg, alpha, maxptps, geom, suptitle="CSHL049")

# %%
vis_utils.plotlocs(x, y, z_reg, alpha, maxptps, geom, which=z_reg > 2800, suptitle="CSHL049 (NN detect trough threshold 3)")

# %%
subset.sum(axis=1)

# %%
subset = waveform_utils.channel_index_subset(geom, channel_index, radius=100)
c = plt.scatter(*geom.T, c=subset.sum(axis=1))
plt.colorbar(c)

# %%
