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
from spike_psvae import extract, vis_utils, waveform_utils
import matplotlib.pyplot as plt

# %%
plt.rc("figure", dpi=200)

# %% [markdown]
# ### test np1

# %%
spike_train_np1 = np.load("/media/peter/2TB/NP1/final_sps_final.npy")
templates_np1 = np.load("/media/peter/2TB/NP1/templates_np1.npy")
spike_index_np1 = extract.spike_train_to_index(spike_train_np1, templates_np1)

# %%
geom_np1 = np.load("/media/peter/2TB/NP1/geom_np1.npy")

# %%
raw_np1_test, denoised_np1_test, inds_np1_test, fcs_np1_test = extract.get_denoised_waveforms("/media/peter/2TB/NP1/standardized.bin", spike_index_np1[:150], geom_np1, batch_size=512)

# %%
vis_utils.labeledmosaic(raw_np1_test[:100].reshape(5, 20, 121, 20), rowlabels="ab", pad=1)

# %%
vis_utils.labeledmosaic(denoised_np1_test[:100].reshape(5, 20, 121, 20), rowlabels="ab", pad=1)

# %%
fig, axes = plt.subplots(1, 16, sharey=True, figsize=(15, 2.5))
vis_utils.traceplot(raw_np1_test[5, :80, 2:-2], c="b", axes=axes, label="raw", strip=False)
vis_utils.traceplot(denoised_np1_test[5, :80, 2:-2], c="orange", axes=axes, label="denoised", strip=False)
plt.legend();

# %% [markdown]
# <!-- ### test np2 -->

# %%
spike_train_np2 = np.load("/mnt/3TB/charlie/NP2/spike_train_ks.npy")
templates_np2 = np.load("/mnt/3TB/charlie/NP2/templates_ks.npy")
spike_index_np2 = extract.spike_train_to_index(spike_train_np2, templates_np2)

# %%
geom_np2 = np.load("/mnt/3TB/charlie/NP2/np2_channel_map.npy")

# %%
raw_np2_test, denoised_np2_test, inds_np2_test, fcs_np2_test = extract.get_denoised_waveforms("/media/peter/2TB/NP1/standardized.bin", spike_index_np2[:5000], geom_np2, dtype=np.float32, ghost_channel=True, pad_for_denoiser=10)

# %%
raw_np2_test.shape, denoised_np2_test.shape

# %%
vis_utils.labeledmosaic(raw_np2_test[:100].reshape(5, 20, 121, 20), rowlabels="abcde", pad=1)

# %%
vis_utils.labeledmosaic(denoised_np2_test[:100].reshape(5, 20, 121, 20), rowlabels="abcde", pad=1)

# %%
local_templates_np2, template_maxchans_np2 = waveform_utils.get_local_waveforms(templates_np2, 10, geom_np2)

# %%
vis_utils.labeledmosaic(local_templates_np2[:100].reshape(5, 20, 121, 20), rowlabels="abcde", pad=1)

# %%
fig, axes = plt.subplots(10, 16, sharey="row", sharex=True, figsize=(10, 10))
for i in range(10):
    vis_utils.traceplot(raw_np2_test[i, :80, 2:-2], c="b", axes=axes[i], label="raw", strip=True)
    vis_utils.traceplot(denoised_np2_test[i, :80, 2:-2], c="g", axes=axes[i], label="denoised", strip=True)
    axes[i, 0].set_ylabel(f"spike {i}")
# plt.legend();
plt.show()

# %%
raw_np1, denoised_np1, indices_np1, firstchans_np1 = extract.get_denoised_waveforms(
    "/media/peter/2TB/NP1/standardized.bin", spike_index_np1, geom_np1,
)

# %%

# %%

# %%

# %%
raw_np2, denoised_np2, indices_np2, firstchans_np2 = extract.get_denoised_waveforms(
    "/media/peter/2TB/NP1/standardized.bin", spike_index_np2, geom_np2, ghost_channel=True, pad_for_denoiser=10
)

# %%
