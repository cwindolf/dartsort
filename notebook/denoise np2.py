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
dest = "../data/ks_np2_all.h5"

# %% [markdown]
# <!-- ### test np2 -->

# %%
spike_train_np2 = np.load("/mnt/3TB/charlie/NP2/spike_train_ks.npy")
templates_np2 = np.load("/mnt/3TB/charlie/NP2/templates_ks.npy")
spike_index_np2 = extract.spike_train_to_index(spike_train_np2, templates_np2)

# %%
template_mcs = templates_np2.ptp(1).argmax(1)
maxptps = templates_np2[np.arange(len(template_mcs)), :, template_mcs].ptp(1)

# %%
num_channels = 384
standardized = np.memmap("/mnt/3TB/charlie/NP2/standardized.bin", dtype=np.float32, mode="r")
standardized = standardized.reshape(-1, num_channels)
# standardized = standardized.reshape(num_channels, -1).T
# standardized = standardized[:, :num_channels]

# %%
times = spike_train_np2[spike_train_np2[:, 1]==54, 0][:100]
wfs = [standardized[t-60:t-60+121] for t in times]

# %%
times

# %% tags=[]
mc = templates_np2[54].ptp(0).argmax()
for i in range(100):
    plt.figure(figsize = (20, 2.5))
    plt.plot(wfs[i][:82, mc-10:mc+10].T.flatten(), 'blue')
    plt.plot(templates_np2[54, :82, mc-10:mc+10].T.flatten(), 'orange')
    for j in range(19):
        plt.axvline(82 + 82*j, color = 'black')
    plt.show() 

# %%

# %%

# %%
geom_np2 = np.load("/mnt/3TB/charlie/NP2/np2_channel_map.npy")

# %%
raw_np2_test, denoised_np2_test, inds_np2_test, fcs_np2_test = extract.get_denoised_waveforms("/mnt/3TB/charlie/NP2/standardized.bin", spike_index_np2[:5000], geom_np2, dtype=np.float32, threshold=8.0, device="cpu")

# %%
mc2t = raw_np2_test[:100].ptp(1).argmax(1)
raw_np2_test[np.arange(len(mc2t)), :, mc2t].argmin(1)

# %%
vis_utils.labeledmosaic(raw_np2_test[:100].reshape(5, 20, 121, 20), rowlabels="abcde", pad=1)

# %%
vis_utils.labeledmosaic(denoised_np2_test[:100].reshape(5, 20, 121, 20), rowlabels="abcde", pad=1)

# %%
fig, axes = plt.subplots(10, 16, sharey="row", sharex=True, figsize=(10, 10))
for i in range(10):
    vis_utils.traceplot(raw_np2_test[i, :80, 2:-2], c="b", axes=axes[i], label="raw", strip=True)
    vis_utils.traceplot(denoised_np2_test[i, :80, 2:-2], c="g", axes=axes[i], label="denoised", strip=True)
    axes[i, 0].set_ylabel(f"spike {i}")
# plt.legend();
plt.show()

# %%
local_templates_np2, template_maxchans_np2 = waveform_utils.get_local_waveforms(templates_np2, 10, geom_np2)

# %%
vis_utils.labeledmosaic(local_templates_np2[:100].reshape(5, 20, 121, 20), rowlabels="abcde", pad=1)

# %%
raw_np2, denoised_np2, indices_np2, firstchans_np2 = extract.get_denoised_waveforms(
    "/mnt/3TB/charlie/NP2/standardized.bin", spike_index_np2, geom_np2, threshold=6., device="cpu"
)

# %%
selected_train_np2 = spike_train_np2[indices_np2]
selected_index_np2 = spike_index_np2[indices_np2]
with h5py.File("../data/ks_np2.h5", "w") as np2h5:
    np2h5.create_dataset("spike_index", data=selected_index_np2)
    np2h5.create_dataset("spike_train", data=selected_train_np2)
    np2h5.create_dataset("templates", data=templates_np2)
    np2h5.create_dataset("geom", data=geom_np2)
    np2h5.create_dataset("raw_waveforms", data=raw_np2)
    np2h5.create_dataset("denoised_waveforms", data=denoised_np2)
    np2h5.create_dataset("first_channels", data=firstchans_np2)
    np2h5.create_dataset("selection_indices", data=indices_np2)
    
    for k in np2h5:
        print(k, np2h5[k].shape, np2h5[k].dtype)

# %%
del raw_np2, denoised_np2

# %% [markdown]
# ## add maxchans

# %%
with h5py.File("../data/ks_np2.h5", "r+") as np2h5:
    wfs = np2h5["denoised_waveforms"]
    maxchans = []
    for fc, wf in tqdm(zip(np2h5["first_channels"][:], wfs), total=len(wfs)):
        maxchans.append(waveform_utils.maxchan_from_firstchan(fc, wf))
    maxchans = np.array(maxchans)
    np2h5.create_dataset("max_channels", data=maxchans)
    
    for k in np2h5:
        print(k, np2h5[k].shape, np2h5[k].dtype)

# %% [markdown]
# ## localization

# %%
from spike_psvae import localization

# %%
with h5py.File("../data/ks_np2.h5", "r+") as np2h5:
    xs_np2, ys_np2, z_rels_np2, z_abss_np2, alphas_np2 = localization.localize_waveforms_batched(
        np2h5["denoised_waveforms"],
        np2h5["geom"][:],
        maxchans=np2h5["max_channels"][:],
        channel_radius=10,
        n_workers=15,
        jac=False,
        firstchans=np2h5["first_channels"][:],
        geomkind="firstchan",
        batch_size=128,
    )
    
    np2h5.create_dataset("x", data=xs_np2)
    np2h5.create_dataset("y", data=ys_np2)
    np2h5.create_dataset("z_abs", data=z_abss_np2)
    np2h5.create_dataset("z_rel", data=z_rels_np2)
    np2h5.create_dataset("alpha", data=alphas_np2)
    
    for k in np2h5:
        print(k, np2h5[k].shape, np2h5[k].dtype)

# %%
with h5py.File("../data/ks_np2.h5", "r+") as np2h5:
    wfs = np2h5["denoised_waveforms"]
    maxptps_np2 = []
    for fc, wf in tqdm(zip(np2h5["first_channels"][:], wfs), total=len(wfs)):
        maxptps_np2.append(wf.astype(float).ptp(0).ptp())
    maxptps_np2 = np.array(maxptps_np2)
    np2h5.create_dataset("maxptp", data=maxptps_np2)
    
    for k in np2h5:
        print(k, np2h5[k].shape, np2h5[k].dtype)

# %%
with h5py.File("../data/ks_np2.h5", "r+") as np2h5:
    R, _, _ = lib.faster(np2h5["maxptp"][:], np2h5["z_abs"][:], np2h5["spike_index"][:, 0] / 30000)

# %%
cuts.plot(R)

# %%
from npx import reg

# %%
with h5py.File("../data/ks_np2.h5", "r+") as np2h5:
    z_reg, dispmap = reg.register_nonrigid(
        np2h5["maxptp"][:],
        np2h5["z_abs"][:],
        np2h5["spike_index"][:, 0] / 30_000,
        robust_sigma=1,
        rigid_disp=200,
        disp=100,
        denoise_sigma=0.1,
        destripe=False,
        n_windows=[5, 30, 60],
        n_iter=1,
        widthmul=0.25,
    )
    np2h5.create_dataset("z_reg", data=z_reg)

# %%
with h5py.File("../data/ks_np2.h5", "r") as np2h5:
    Rreg, _, _ = lib.faster(np2h5["maxptp"][:], z_reg, np2h5["spike_index"][:, 0] / 30000)
    cuts.plot(Rreg)

# %%

# %%

# %%
randcm = plt.cm.colors.ListedColormap(
  plt.cm.rainbow(
    np.random.default_rng(0).permutation(np.linspace(0, 1, 256))
  )
)

# %%
with h5py.File("../data/ks_np2.h5", "r+") as np2h5:
    plt.scatter(np2h5["spike_index"][:, 0] / 30000, np2h5["z_abs"][:], c=np2h5["spike_train"][:,1], cmap=randcm, s=1)

# %%
