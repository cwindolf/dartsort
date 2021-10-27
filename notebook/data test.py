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
# %ll -h /mnt/3TB/charlie/features/denoised_wfs_ps_vae | head -5

# %%
# %ll /mnt/3TB/charlie/features/position_results_files_charlie_merged

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
