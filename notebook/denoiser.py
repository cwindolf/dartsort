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
from spike_psvae import extract

# %%
spike_train_np1 = np.load("/media/peter/2TB/NP1/final_sps_final.npy")
templates_np1 = np.load("/media/peter/2TB/NP1/templates_np1.npy")
spike_index_np1 = extract.spike_train_to_index(spike_train_np1, templates_np1)

# %%
geom_np1 = np.load("/media/peter/2TB/NP1/geom_np1.npy")

# %%
denoised_np1 = extract.get_denoised_waveforms("/media/peter/2TB/NP1/standardized.bin", spike_index_np1, geom_np1, batch_size=512)

# %%

# %%
spike_train_np2 = np.load("/mnt/3TB/charlie/NP2/spike_train_ks.npy")
templates_np2 = np.load("/mnt/3TB/charlie/NP2/templates_ks.npy")
spike_index_np2 = extract.spike_train_to_index(spike_train_np2, templates_np2)

# %%
geom_np2 = np.load("/mnt/3TB/charlie/NP2/np2_channel_map.npy")

# %%
denoised_np2 = extract.get_denoised_waveforms("/media/peter/2TB/NP1/standardized.bin", spike_index_np2, geom_np2, batch_size=512)

# %%
