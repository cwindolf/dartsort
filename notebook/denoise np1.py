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
import h5py
from tqdm.auto import trange, tqdm

# %%
from npx import lib, cuts, reg

# %%
plt.rc("figure", dpi=200)

# %%

# %% [markdown] tags=[]
# ### test np1

# %%
spike_train_np1 = np.load("/media/peter/2TB/NP1/final_sps_final.npy")
templates_np1 = np.load("/media/peter/2TB/NP1/templates_np1.npy")
spike_index_np1 = extract.spike_train_to_index(spike_train_np1, templates_np1)

# %%
geom_np1 = np.load("/media/peter/2TB/NP1/geom_np1.npy")

# %%
raw_np1_test, denoised_np1_test, inds_np1_test, fcs_np1_test = extract.get_denoised_waveforms("/media/peter/2TB/NP1/standardized.bin", spike_index_np1[:1050], geom_np1, batch_size=128)

# %%
vis_utils.labeledmosaic(raw_np1_test[:100].reshape(5, 20, 121, 20), rowlabels="ab", pad=1)

# %%
m1 = raw_np1_test[:100].ptp(1).argmax(1)
raw_np1_test[np.arange(100), :, m1].argmin(1)

# %%
vis_utils.labeledmosaic(denoised_np1_test[:100].reshape(5, 20, 121, 20), rowlabels="ab", pad=1)

# %%
fig, axes = plt.subplots(10, 16, sharey="row", sharex=True, figsize=(10, 10))
for i in range(10):
    vis_utils.traceplot(raw_np1_test[i, :80, 2:-2], c="b", axes=axes[i], label="raw", strip=True)
    vis_utils.traceplot(denoised_np1_test[i, :80, 2:-2], c="g", axes=axes[i], label="denoised", strip=True)
    axes[i, 0].set_ylabel(f"spike {i}")
# plt.legend();
plt.show()

# %% [markdown]
# <!-- ### test np2 -->

# %%
raw_np1, denoised_np1, indices_np1, firstchans_np1 = extract.get_denoised_waveforms(
    "/media/peter/2TB/NP1/standardized.bin", spike_index_np1, geom_np1, threshold=6.
)

# %%
selected_train_np1 = spike_train_np1[indices_np1]
selected_index_np1 = spike_index_np1[indices_np1]
with h5py.File("../data/yass_np1.h5", "w") as np1h5:
    np1h5.create_dataset("spike_index", data=selected_index_np1)
    np1h5.create_dataset("spike_train", data=selected_train_np1)
    np1h5.create_dataset("templates", data=templates_np1)
    np1h5.create_dataset("geom", data=geom_np1)
    np1h5.create_dataset("raw_waveforms", data=raw_np1)
    np1h5.create_dataset("denoised_waveforms", data=denoised_np1)
    np1h5.create_dataset("first_channels", data=firstchans_np1)
    np1h5.create_dataset("selection_indices", data=indices_np1)
    
    for k in np1h5:
        print(k, np1h5[k].shape, np1h5[k].dtype)

# %%
del raw_np1, denoised_np1

# %%

# %% [markdown]
# ## add maxchans

# %%
with h5py.File("../data/yass_np1.h5", "r+") as np1h5:
    wfs = np1h5["denoised_waveforms"]
    maxchans = []
    for fc, wf in tqdm(zip(np1h5["first_channels"][:], wfs), total=len(wfs)):
        maxchans.append(waveform_utils.maxchan_from_firstchan(fc, wf))
    maxchans = np.array(maxchans)
    np1h5.create_dataset("max_channels", data=maxchans)
    
    for k in np1h5:
        print(k, np1h5[k].shape, np1h5[k].dtype)

# %% [markdown]
# ## localization

# %%
from spike_psvae import localization

# %%
with h5py.File("../data/yass_np1.h5", "r+") as np1h5:
    xs_np1, ys_np1, z_rels_np1, z_abss_np1, alphas_np1 = localization.localize_waveforms_batched(
        np1h5["denoised_waveforms"],
        np1h5["geom"][:],
        maxchans=np1h5["max_channels"][:],
        channel_radius=10,
        n_workers=10,
        jac=False,
        firstchans=np1h5["first_channels"][:],
        geomkind="firstchan",
        batch_size=128,
    )
    
    np1h5.create_dataset("x", data=xs_np1)
    np1h5.create_dataset("y", data=ys_np1)
    np1h5.create_dataset("z_abs", data=z_abss_np1)
    np1h5.create_dataset("z_rel", data=z_rels_np1)
    np1h5.create_dataset("alpha", data=alphas_np1)
    
    for k in np1h5:
        print(k, np1h5[k].shape, np1h5[k].dtype)

# %%
with h5py.File("../data/yass_np1.h5", "r+") as np1h5:
    wfs = np1h5["denoised_waveforms"]
    maxptps_np1 = []
    for fc, wf in tqdm(zip(np1h5["first_channels"][:], wfs), total=len(wfs)):
        maxptps_np1.append(wf.astype(float).ptp(0).ptp())
    maxptps_np1 = np.array(maxptps_np1)
    np1h5.create_dataset("maxptp", data=maxptps_np1)
    
    for k in np1h5:
        print(k, np1h5[k].shape, np1h5[k].dtype)

# %%

# %%
with h5py.File("../data/yass_np1.h5", "r+") as np1h5:
    R, _, _ = lib.faster(np1h5["maxptp"][:], np1h5["z_abs"][:], np1h5["spike_index"][:, 0] / 30000)

# %%
cuts.plot(R)

# %%

# %%
from npx import reg

# %%
with h5py.File("../data/yass_np1.h5", "r+") as np1h5:
    if "z_reg" in np1h5:
        del np1h5["z_reg"]
    z_reg, dispmap = reg.register_nonrigid(
        np1h5["maxptp"][:],
        np1h5["z_abs"][:],
        np1h5["spike_index"][:, 0] / 30_000,
        robust_sigma=1,
        rigid_disp=200,
        disp=100,
        denoise_sigma=0.1,
        destripe=False,
        n_windows=[60],
        n_iter=1,
        widthmul=0.25,
    )
    np1h5.create_dataset("z_reg", data=z_reg)

# %%
with h5py.File("../data/yass_np1.h5", "r") as np1h5:
    Rreg, _, _ = lib.faster(np1h5["maxptp"][:], z_reg, np1h5["spike_index"][:, 0] / 30000)
    cuts.plot(Rreg)

# %%
