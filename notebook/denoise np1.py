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
template_maxptp = templates_np1.ptp(1).max(1)
bigtemps = template_maxptp >= 50.0
bigtemps.sum()

# %%
templates_np1.shape

# %%
spike_index_np1.shape

# %%
geom_np1 = np.load("/media/peter/2TB/NP1/geom_np1.npy")

# %%
geom_np1[:20]

# %%
plt.plot(geom_np1[:20, 0], geom_np1[:20, 1], lw=0.1)

# %%
rg = np.random.default_rng(0)

# %%
raw_np1_test, denoised_np1_test, inds_np1_test, fcs_np1_test = extract.get_denoised_waveforms("/media/peter/2TB/NP1/standardized.bin", spike_index_np1[rg.choice(1530553, 1000, replace=False)], geom_np1, batch_size=128, threshold=6, geomkind="standard", channel_radius=8)

# %%
raw_np1_test.shape

# %%
vis_utils.labeledmosaic(raw_np1_test[:100].reshape(5, 20, 121, 18), rowlabels="ab", pad=1)

# %%
m1 = raw_np1_test[:100].ptp(1).argmax(1)
raw_np1_test[np.arange(100), :, m1].argmin(1)

# %%
vis_utils.labeledmosaic(denoised_np1_test[:100].reshape(5, 20, 121, 18), rowlabels="ab", pad=1)

# %%
fig, axes = plt.subplots(5, 18, sharey="row", sharex=True, figsize=(10, 10))
for i in range(5):
    vis_utils.traceplot(raw_np1_test[i, :80, :], c="b", axes=axes[i], label="raw", strip=True)
    vis_utils.traceplot(denoised_np1_test[i, :80, :], c="g", axes=axes[i], label="denoised", strip=True)
    axes[i, 0].set_ylabel(f"spike {i}")
# plt.legend();
plt.show()

# %% [markdown]
# <!-- ### test np2 -->

# %% tags=[]
raw_np1, denoised_np1, indices_np1, firstchans_np1 = extract.get_denoised_waveforms(
    "/media/peter/2TB/NP1/standardized.bin", spike_index_np1, geom_np1, threshold=0, device="cpu", inmem=False, geomkind="standard", channel_radius=8, pad_for_denoiser=8,
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
# %reset -f

# %%
# %rm ___tmp.h5

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
    np1h5["spike_index"][:, 1] = maxchans
    
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
        channel_radius=8,
        n_workers=5,
        jac=False,
        firstchans=np1h5["first_channels"][:],
        geomkind="firstchanstandard",
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
print("hi")

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
with h5py.File("../data/yass_np1.h5", "r") as h5:
    y = h5["y"][:]
    big_y = y > 1e-4
big_y_inds = np.flatnonzero(big_y)

# %%
big_y.mean()

# %%
with h5py.File("../data/yass_np1.h5", "r") as h5:
    N, T, C = h5["denoised_waveforms"].shape
    print(T, C, flush=True)
        
    with h5py.File(f"../data/yass_np1_nzy.h5", "w") as out:
        # non per-spike data
        for k in ["geom", "templates"]:
            print(k, flush=True)
            out.create_dataset(k, data=h5[k][:])

        # small datasets can just be sliced directly
        for k in (k for k in h5.keys() if h5[k].shape in ((N,), (N, 2))):
            print(k, flush=True)
            out.create_dataset(k, data=h5[k][:][big_y_inds])

        # spikes in a for loop
        print("spikes", flush=True)
        wfs = out.create_dataset("denoised_waveforms", shape=(len(big_y_inds), T, C))
        in_dnwf = h5["denoised_waveforms"]
        for i, j in tqdm(enumerate(big_y_inds), total=len(big_y_inds)):
            wfs[i] = in_dnwf[j]

# %%
with h5py.File("../data/yass_np1_nzy.h5", "r") as h5:
    vis_utils.locrelocplots(h5, seed=1)

# %%

# %%

# %%

# %%

# %%

# %%
rg = np.random.default_rng(1)

# %%
q = rg.normal(size=(10000000, 3))

# %%
q[:5]

# %%
qq = np.cumsum(q, axis=1)

# %%
qq[:5]

# %%

# %%
b1 = qq[:, 0]
b2 = qq[:, 1]
b3 = qq[:, 2]

# %%
qqm = (b1 <= 0) & (b2 <= 0) & (b3 >= 0)

# %%
1/16

# %%
qqm.mean()

# %%
(np.pi / 2 - np.arctan(np.sqrt(2))) * 3 / (8 * np.pi)

# %%
a = (b1 <= 0) & (b2 <= 0)
b = (b2 <= 0) & (b3 >= 0)
a.mean() * b.mean() * 2, (a & b).mean()

# %%
((qq[:,0] <= 0) & (qq[:,1] <= 0)).mean()

# %%
3/8

# %%
((qq[:,1] <= 0) & (qq[:,2] >= 0)).mean()

# %%
np.arctan(np.sqrt(2)) / (2*np.pi)

# %%
ix = (b2 <= 0) & (b3 >= 0)
plt.scatter(b2[ix], b3[ix] - b2[ix], marker=".", s=1)
plt.gca().set_aspect(1)

# %%
((np.pi / 2) - np.arctan(np.sqrt(2))) / (2 * np.pi)

# %%
((np.pi / 2) - np.arctan(np.sqrt(2))) * 3 / (8 * np.pi)

# %%
