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
import numpy as np
import glob
import matplotlib.pyplot as plt
import h5py
from tqdm.auto import trange, tqdm

# %%

# %%
# %ll -h /mnt/3TB/charlie/features

# %%
# %ll -h /mnt/3TB/charlie/features/denoised_wfs_ps_vae | head

# %%
# %ll /mnt/3TB/charlie/features/latest_positions_charlie

# %%
with h5py.File("../data/wfs_locs.h5", "r") as orig:
    print(orig.keys())
    geom = orig["geom"]

# %%
pos = "/mnt/3TB/charlie/features/latest_positions_charlie"
with h5py.File("../data/wfs_locs_b.h5", "w") as h5:
    for f in glob.glob(f"{pos}/*.npy"):
        dset = f.split("/")[-1].split(".npy")[0]
        print(dset)
        x = np.load(f)
        print(x.shape)
        print(x[:10])
        h5.create_dataset(dset, data=x)

# %%
wfs = "/mnt/3TB/charlie/features/denoised_wfs_ps_vae"

# %%
x.shape[0], np.load(f"{wfs}/wfs_batch_000004.npy").shape

# %%
with h5py.File("../data/wfs_locs_b.h5", "r+") as h5:
    dwf = h5.create_dataset("denoised_waveforms", shape=(x.shape[0], 121, 20), dtype=np.float64)
    start_ix = 0
    for f in tqdm(sorted(glob.glob(f"{wfs}/wfs_batch_*.npy"))):
        batch = np.load(f)
        b, _, __ = batch.shape
        dwf[start_ix:start_ix + b] = batch
        start_ix += b

# %%

# %%

# %%
pos = "/mnt/3TB/charlie/features/position_results_files_charlie_merged"
wfs = "/mnt/3TB/charlie/features/denoised_wfs_ps_vae"

# %%
good_times = np.flatnonzero(np.load(f"{pos}/times_read.npy") == 1)

# %%
alpha_merged = np.load(f"{pos}/results_alpha_merged.npy")[good_times]
alpha_merged, alpha_merged.shape

# %%
max_channels = np.load(f"{pos}/results_max_channels.npy")[good_times]
max_channels, max_channels.shape

# %%
max_ptp_merged = np.load(f"{pos}/results_max_ptp_merged.npy")[good_times]
max_ptp_merged, max_ptp_merged.shape

# %%
plt.scatter(alpha_merged, max_ptp_merged, s=1)

# %%
spread_merged = np.load(f"{pos}/results_spread_merged.npy")[good_times]
spread_merged, spread_merged.shape

# %%
plt.scatter(spread_merged, max_ptp_merged, s=1)

# %%
width = np.load(f"{pos}/results_width.npy")[good_times]
width, width.shape

# %%
x_mean_merged = np.load(f"{pos}/results_x_mean_merged.npy")[good_times]
x_mean_merged, x_mean_merged.shape

# %%
x_merged = np.load(f"{pos}/results_x_merged.npy")[good_times]
x_merged, x_merged.shape

# %%
x_mean_merged = np.load(f"{pos}/results_x_mean_merged.npy")[good_times]
x_mean_merged, x_mean_merged.shape

# %%
np.load(f"{pos}/results_x_mean_merged.npy").shape

# %%
b0 = np.load(f"{wfs}/wfs_batch_000000.npy")

# %%
b0.shape

# %%
b1 = np.load(f"{wfs}/wfs_batch_000001.npy")
b1.shape, b1.dtype

# %%
bs = []
for f in glob.glob(f"{wfs}/wfs_batch_*.npy"):
    bs.append(np.load(f).shape[0])

# %%
sum(bs)

# %%
len(bs)

# %% [markdown]
# ### write to h5

# %%
pos = "/mnt/3TB/charlie/features/position_results_files_charlie_merged"
wfs = "/mnt/3TB/charlie/features/denoised_wfs_ps_vae"
good_times = np.flatnonzero(np.load(f"{pos}/times_read.npy") == 1)

# %%
with h5py.File("/mnt/3TB/charlie/features/wfs_locs.h5", "w") as h5:
    for f in glob.glob(f"{pos}/results*"):
        n = f.split("results_")[2]
        if "_merged" in n:
            n = n.split("_merged")[0]
        if ".npy" in n:
            n = n.split(".npy")[0]
        x = np.load(f)
        print(n, x[good_times].shape)
        h5[n] = x[good_times]

# %%
good_times.shape[0]

# %%
with h5py.File("/mnt/3TB/charlie/features/wfs_locs.h5", "r+") as h5:
    if "denoised_waveforms" in h5:
        del h5["denoised_waveforms"]
    dwf = h5.create_dataset("denoised_waveforms", shape=(good_times.shape[0], 121, 20), dtype=np.float64)
    
    start_ix = 0
    for f in tqdm(glob.glob(f"{wfs}/wfs_batch_*.npy")):
        batch = np.load(f)
        b, _, __ = batch.shape
        dwf[start_ix:start_ix + b] = batch
        start_ix += b

# %%
with h5py.File("/mnt/3TB/charlie/features/wfs_locs.h5", "r") as h5:
    plt.imshow(h5["denoised_waveforms"][10001])
    plt.show()
    plt.imshow(h5["denoised_waveforms"][20001]) 
    plt.show()
    plt.imshow(h5["denoised_waveforms"][30001])  
    plt.show()
    plt.imshow(h5["denoised_waveforms"][40001])    
    plt.show()

# %%
y = np.load(f"{pos}/results_y_merged.npy")[good_times]

# %%
y.min(), y.max()

# %%
y

# %%
z = np.load(f"{pos}/results_z_merged.npy")[good_times]

# %%
z.min(), z.max()

# %%
z

# %%
x = np.load(f"{pos}/results_x_merged.npy")[good_times]

# %%
x.min(), x.max()

# %%
# add y_rel -- geom is np2, we can use max channel's location
cm = np.load("/home/charlie/spikes_localization_registration/channels_maps/np2_channel_map.npy")
max_channels = np.load(f"{pos}/results_max_channels.npy")[good_times]
cm.min(axis=0), cm.max(axis=0), max_channels.min(), max_channels.max()

# %%
zc = cm[max_channels.astype(int)]

# %%
plt.hist(z - zc[:, 1], bins=128, log=True)
plt.show()

# %%
with h5py.File("/mnt/3TB/charlie/features/wfs_locs.h5", "r+") as h5:
    h5.create_dataset("z_rel", data=z - zc[:, 1])

# %%
# see waveform range
mins = np.full((121, 20), np.inf)
maxs = np.full((121, 20), -np.inf)
with h5py.File("/mnt/3TB/charlie/features/wfs_locs.h5", "r") as h5:
    xs = h5["denoised_waveforms"]
    for ix in trange(0, xs.shape[0], 1000):
        x_ = xs[ix:min(ix + 1000, xs.shape[0])]
        min_ = x_.min(axis=0)
        max_ = x_.max(axis=0)
        mins = np.minimum(min_, mins)
        maxs = np.maximum(max_, maxs)

# %%
mins.min(), mins.max()

# %%
maxs.min(), maxs.max()

# %%
plt.imshow(mins); plt.colorbar()

# %%
plt.imshow(maxs); plt.colorbar()

# %%
