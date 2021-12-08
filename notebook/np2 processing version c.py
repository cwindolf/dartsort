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
from glob import glob
from tqdm.auto import trange, tqdm

# %%
import matplotlib.pyplot as plt
plt.rc("figure", dpi=300)

# %%
from spike_psvae import waveform_utils
from npx import cuts, reg, lib

# %% [markdown]
# ### sort data by spike time (in samples)

# %%
# %ll -h /mnt/3TB/charlie/multi_denoised_ls_localized/*_41_1.npy

# %%
root = "/mnt/3TB/charlie/multi_denoised_ls_localized"

# %%
all_times = []
for fcfn in sorted(glob(f"{root}/first_chan_*.npy")):
    *_, mc, batchno = fcfn.split("/")[-1].split(".")[0].split("_")
    mcfn = f"{root}/max_chan_{mc}_{batchno}.npy"
    mcfn = f"{root}/max_chan_{mc}_{batchno}.npy"
    tfn = f"{root}/times_{mc}_{batchno}.npy"
    locfn = f"{root}/localization_features_{mc}_{batchno}.npy"
    wffn = f"{root}/waveforms_{mc}_{batchno}.npy"
    times_ = np.load(tfn)
    wf_ = np.load(wffn)
    if not wf_.shape[0]:
        continue
        
    if wf_.shape[0] != times_.shape[0]:
        if times_[0] < 50:
            print(np.load(wffn).shape, np.load(locfn).shape, times_.shape, np.load(fcfn).shape, np.load(mcfn).shape)
            times_ = times_[1:]
        elif times_[-1] > 30000000 - 121:
            print(np.load(wffn).shape, np.load(locfn).shape, times_.shape, np.load(fcfn).shape, np.load(mcfn).shape)
            times_ = times_[:-1]
        
    all_times.extend(times_)

# %%
fcfn = "/mnt/3TB/charlie/multi_denoised_ls_localized/first_chan_41_1.npy"
*_, mc, batchno = fcfn.split("/")[-1].split(".")[0].split("_")
mcfn = f"{root}/max_chan_{mc}_{batchno}.npy"
tfn = f"{root}/times_{mc}_{batchno}.npy"
locfn = f"{root}/localization_features_{mc}_{batchno}.npy"
wffn = f"{root}/waveforms_{mc}_{batchno}.npy"
fcfn, mcfn, tfn, locfn, wffn

# %%
mct = np.load(mcfn); tt = np.load(tfn); loct = np.load(locfn); wft = np.load(wffn)
for k, v in zip(["mc", "t", "loc", "wf"], [mct, tt, loct, wft]):
    print(k, v.shape, v.dtype)

# %%
len(all_times), all_times[:20]

# %%
sort_order = np.argsort(all_times)

# %%
sort_order[:20]

# %%
sorted_times = np.array(all_times)[sort_order]
sorted_times[:20]

# %%
np.all(sorted_times[1:] >= sorted_times[:-1])

# %%
N = len(all_times); N

# %%
sort_dest = np.zeros_like(sort_order)
for i, k in enumerate(sort_order):
    sort_dest[k] = i
sort_dest[:20]

# %%
with h5py.File("../data/wfs_locs_c.h5", "w") as out:
    out.create_dataset("fs", data=30_000)
    out.create_dataset("geom", data=np.load("../data/np2_channel_map.npy"))
    
    spike_index = out.create_dataset("spike_index", dtype=np.int64, shape=(N, 2))
    maxchans = out.create_dataset("max_channels", dtype=np.int64, shape=N)
    firstchans = out.create_dataset("first_channels", dtype=np.int64, shape=N)
    x = out.create_dataset("x", dtype=np.float64, shape=N)
    y = out.create_dataset("y", dtype=np.float64, shape=N)
    z = out.create_dataset("z", dtype=np.float64, shape=N)
    alpha = out.create_dataset("alpha", dtype=np.float64, shape=N)
    ptp = out.create_dataset("maxptp", dtype=np.float64, shape=N)
    waveforms = out.create_dataset("denoised_waveforms", dtype=np.float64, shape=(N, 82, 20), chunks=(512, 82, 20))
    
    index = 0
    
    for fcfn in tqdm(list(sorted(glob(f"{root}/first_chan_*.npy")))):
        *_, mc, batchno = fcfn.split("/")[-1].split(".")[0].split("_")
        mcfn = f"{root}/max_chan_{mc}_{batchno}.npy"
        tfn = f"{root}/times_{mc}_{batchno}.npy"
        locfn = f"{root}/localization_features_{mc}_{batchno}.npy"
        wffn = f"{root}/waveforms_{mc}_{batchno}.npy"
        
        fc = np.load(fcfn)
        mc = np.load(mcfn)
        ts = np.load(tfn)
        locs = np.load(locfn)
        wfs = np.load(wffn).astype(np.float64)
        
        if not wfs.shape[0]:
            continue
            
        if wf_.shape[0] != times_.shape[0]:
            if ts[0] < 50:
                ts = ts[1:]
            elif ts[-1] > 30000000 - 121:
                ts = ts[:-1]
        
        for t, loc, wf in zip(ts, locs, wfs):
            n = sort_dest[index]
            index += 1
            spike_index[n, 0] = t
            spike_index[n, 1] = mc
            maxchans[n] = mc
            firstchans[n] = fc
            x[n], z[n], y[n], alpha[n] = loc
            waveforms[n] = wf
            amp = wf.ptp(0).ptp()
            ptp[n] = amp


# %% [markdown]
# ### check result

# %%
with h5py.File("../data/wfs_locs_c.h5", "r") as f:
    spike_index = f["spike_index"][:]
    maxchans = f["max_channels"][:]
    # firstchans = f["first_channels"][:]
    x = f["x"][:]
    y = f["y"][:]
    z = f["z"][:]
    alpha = f["alpha"][:]
    ptp = f["maxptp"][:]
    geom = f["geom"][:]
    print((ptp == 0).sum(), np.flatnonzero(ptp == 0))
    print(ptp.dtype)
    print(alpha[np.flatnonzero(ptp == 0)])
    
    # check sorted times
    print(np.all(spike_index[1:, 0] >= spike_index[:-1, 0]))
    
    # check everyone looks normal
    # plt.scatter(firstchans, maxchans, s=1); plt.show()
    plt.scatter(x, z, s=1, c=np.abs(zrel) >= 32); plt.show()
    plt.hist(x, bins=128); plt.title("x"); plt.show()
    plt.hist(y, bins=128); plt.title("y"); plt.show()
    plt.hist(z, bins=128); plt.title("z"); plt.show()
    plt.hist(ptp, bins=128); plt.title("ptp"); plt.show()
    plt.hist(alpha, bins=128); plt.title("alpha"); plt.show()

# %%
# zrel = waveform_utils.relativize_z(z, maxchans, geom)
# pct = 100 * (np.abs(zrel) >= 32).mean()
# print(f"{pct:0.2f}%")
# print(f"{100-pct:0.2f}%")
# plt.hist(zrel, bins=np.arange(np.floor(zrel.min()), np.ceil(zrel.max())), label=f"<32 micron ({100-pct:0.2f}%)", log=True)
# plt.hist(zrel[np.abs(zrel) >= 32], bins=np.arange(np.floor(zrel.min()), np.ceil(zrel.max())), color="r", label=f">=32 micron ({pct:0.2f}%)", log=True)
# plt.xlabel("localized z relative to max channel z")
# plt.ylabel("frequency (log scale)")
# plt.legend(fancybox=False)
# plt.title("do we need to relocate by more than one Z unit often? (no)")
# plt.show()

# %%
# with h5py.File("../data/feats_c_xyza.h5") as f:
#     zrel = f["z_rel"][:]
#     pct = 100 * (np.abs(zrel) >= 32).mean()
#     print(f"{pct:0.2f}%")
#     print(f"{100-pct:0.2f}%")
#     plt.hist(zrel, bins=np.arange(np.floor(zrel.min()), np.ceil(zrel.max())), label=f"<32 micron ({100-pct:0.2f}%)", log=False)
#     plt.hist(zrel[np.abs(zrel) >= 32], bins=np.arange(np.floor(zrel.min()), np.ceil(zrel.max())), color="r", label=f">=32 micron ({pct:0.2f}%)", log=False)
#     plt.xlabel("localized z relative to max channel z")
#     plt.ylabel("frequency")
#     plt.legend(fancybox=False)
#     plt.title("do we need to relocate by more than one Z unit often? (no)")
#     plt.show()

# %%
# with h5py.File("../data/wfs_locs_c.h5", "r+") as f:
#     z = f["z"][:]
#     geom = f["geom"][:]
#     maxchans = f["max_channels"][:]
#     zrel = waveform_utils.relativize_z(z, maxchans, geom)
#     f.create_dataset("z_rel", data=zrel)

# %%
# pct = 100 * (np.abs(zrel) >= 32).mean()
# print(f"{pct:0.2f}%")
# print(f"{100-pct:0.2f}%")
# zrel = waveform_utils.relativize_z(z, maxchans, geom)
# plt.hist(zrel, bins=np.arange(np.floor(zrel.min()), np.ceil(zrel.max())), log=True, label=f"<32 micron ({100-pct:0.2f}%)")
# plt.hist(zrel[np.abs(zrel) >= 32], bins=np.arange(np.floor(zrel.min()), np.ceil(zrel.max())), log=True, color="r", label=f">=32 micron ({pct:0.2f}%)")
# plt.xlabel("localized z relative to max channel z")
# plt.ylabel("frequency (log scale)")
# plt.legend(fancybox=False)
# plt.title("do we need to relocate by more than one Z unit often? (no)")
# plt.show()

# %%

# %% [markdown] tags=[]
# ### registration

# %%
with h5py.File("../data/feats_c_xyza.h5", "r+") as f:
    if "fs" not in f:
        f.create_dataset("fs", data=30_000)
    amps = f["maxptp"][:]
    depths = f["z_abs"][:]
    times = f["spike_index"][:, 0].astype(float) / f["fs"][()]
R, dd, tt = lib.faster(amps, depths, times)
cuts.plot(R)

# %%
plt.hist(amps, bins=128);

# %%
plt.plot(times, depths, "k.", ms=0.1, alpha=0.1);

# %%
regres = reg.register_nonrigid(amps, depths, times, robust_sigma=1, disp=100, destripe=False, n_windows=[30], widthmul=0.25, denoise_sigma=0.075)

# %%
z_reg, total_shift = regres

# %%
R, dd, tt = lib.faster(amps, z_reg, times)
cuts.plot(R)

# %%
for featfn in ["feats_c_xyza", "feats_c_yza"]:
    with h5py.File(f"../data/{featfn}.h5", "r+") as f:
        f.create_dataset("z_reg", data=z_reg)

# %%
with h5py.File("../data/feats_b.h5") as f:
    print((f["y"][:] < 1e-10).mean())
    print(f["y"].shape)

# %%
with h5py.File("../data/feats_c_yza.h5") as f:
    print((f["y"][:] < 1e-10).mean())
    print(f["y"].shape)

# %%
