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
geom_np2 = np.load("/mnt/3TB/charlie/NP2/np2_channel_map.npy")

# %%

# %%

# %%
num_channels = 384
standardized = np.memmap("/mnt/3TB/charlie/NP2/standardized.bin", dtype=np.float32, mode="r")
standardized = standardized.reshape(-1, num_channels)
# standardized = standardized.reshape(num_channels, -1).T
# standardized = standardized[:, :num_channels]

# %%

# %%
times = spike_train_np2[spike_train_np2[:, 1]==54, 0][:100]
wfs = [standardized[t-60:t-60+121] for t in times]

# %% tags=[]
mc = templates_np2[54].ptp(0).argmax()
for i in range(5):
    plt.figure(figsize = (20, 2.5))
    plt.plot(wfs[i][:82, mc-10:mc+10].T.flatten(), 'blue')
    plt.plot(templates_np2[54, :82, mc-10:mc+10].T.flatten(), 'orange')
    for j in range(19):
        plt.axvline(82 + 82*j, color = 'black')
    plt.show() 

# %%

# %%

# %%
raw_np2_test, denoised_np2_test, inds_np2_test, fcs_np2_test = extract.get_denoised_waveforms("/mnt/3TB/charlie/NP2/standardized.bin", spike_index_np2[:5000], geom_np2, dtype=np.float32, threshold=0, geomkind="standard", channel_radius=8, device="cpu")

# %%
mc2t = raw_np2_test[:100].ptp(1).argmax(1)
raw_np2_test[np.arange(len(mc2t)), :, mc2t].argmin(1)

# %%
vis_utils.labeledmosaic(raw_np2_test[:100].reshape(5, 20, 121, 18), rowlabels="abcde", pad=1)

# %%
vis_utils.labeledmosaic(denoised_np2_test[:100].reshape(5, 20, 121, 18), rowlabels="abcde", pad=1)

# %%
fig, axes = plt.subplots(10, 18, sharey="row", sharex=True, figsize=(10, 10))
for i in range(10):
    vis_utils.traceplot(raw_np2_test[i, :80, :], c="b", axes=axes[i], label="raw", strip=True)
    vis_utils.traceplot(denoised_np2_test[i, :80, :], c="g", axes=axes[i], label="denoised", strip=True)
    axes[i, 0].set_ylabel(f"spike {i}")
# plt.legend();
plt.show()

# %%
local_templates_np2, template_maxchans_np2 = waveform_utils.get_local_waveforms(templates_np2, 10, geom_np2)

# %%
vis_utils.labeledmosaic(local_templates_np2[:100].reshape(5, 20, 121, 20), rowlabels="abcde", pad=1)

# %%
raw_np2, denoised_np2, indices_np2, firstchans_np2 = extract.get_denoised_waveforms(
    "/mnt/3TB/charlie/NP2/standardized.bin", spike_index_np2, geom_np2, threshold=0, geomkind="standard", channel_radius=8, pad_for_denoiser=8,
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
    np2h5["spike_index"][:, 1] = maxchans
    
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
        channel_radius=8,
        n_workers=10,
        jac=False,
        firstchans=np2h5["first_channels"][:],
        geomkind="firstchanstandard",
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
maxptps_np2.min()

# %%
with h5py.File("../data/ks_np2.h5", "r+") as np2h5:
    R, _, _ = lib.faster(np2h5["maxptp"][:], np2h5["z_abs"][:], np2h5["spike_index"][:, 0] / 30000)

# %%
cuts.plot(R)

# %%
from npx import reg

# %%
with h5py.File("../data/ks_np2.h5", "r") as np2h5:
    z_rigid_reg, p_rigid = reg.register_rigid(
        np2h5["maxptp"][:],
        np2h5["z_abs"][:],
        np2h5["spike_index"][:, 0] / 30_000,
        robust_sigma=0,
        disp=400,
        denoise_sigma=0.1,
        destripe=False,
    )
    Rreg, _, _ = lib.faster(np2h5["maxptp"][:], z_rigid_reg, np2h5["spike_index"][:, 0] / 30000)
    cuts.plot(Rreg); plt.show()
plt.plot(p_rigid)
np.save("../data/np2_p_rigid.npy", p_rigid)

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

# %% [markdown]
# ## cull data
#
# we'll pick: 

# %%
with h5py.File("../data/ks_np2.h5") as h5:
    for k in h5:
        print(k, h5[k].dtype, h5[k].shape)

# %%
with h5py.File("../data/ks_np2.h5", "r") as h5:
    templates = h5["templates"][:]
    tptps = templates.ptp(1).max(1)
    big_units = np.flatnonzero(tptps >= 8)
    print(len(big_units))
    cluster_ids = h5["spike_train"][:, 1]
    is_big_unit = np.flatnonzero(np.isin(cluster_ids, big_units))
    print(is_big_unit.shape)
    ys = h5["y"][:]
    print(ys[is_big_unit].min(), (ys[is_big_unit] < 0.0001).mean())
    big_y = np.flatnonzero(ys > 1e-4)
    keepers = np.intersect1d(is_big_unit, big_y)

# %%
len(keepers)

# %%
bins = np.arange(np.ceil(ys.max()), step=4)
plt.hist(ys, bins=bins, log=True, label="all KS detected spikes");
plt.hist(ys[is_big_unit], bins=bins, log=True, label="unit has KS template ptp >= 8");
plt.ylabel("log frequency")
plt.xlabel("y localization")
plt.legend()
plt.title("Not a surprise in hindsight: bright units seem to lie closer to the probe", fontsize=10)
plt.show()

# %%
_, big_unit_counts = np.unique(cluster_ids[is_big_unit], return_counts=True)
_, keeper_counts = np.unique(cluster_ids[keepers], return_counts=True)
plt.bar(np.arange(len(big_units)), big_unit_counts, label="before removing spikes with small y")
plt.bar(np.arange(len(big_units)), keeper_counts, label="after removing spikes with small y")
plt.gca().set_xticks([])
plt.semilogy()
plt.legend()
plt.ylabel("number of spikes in unit")
plt.xlabel("unit (KS units with ptp >= 8)")
plt.title("Removing small y spikes does not seem to disproportionately affect certain units", fontsize=8)
plt.show()

# %%
with h5py.File("../data/ks_np2.h5", "r") as np2h5:
    z_reg = np2h5["z_reg"][:]
    Rreg, _, _ = lib.faster(np2h5["maxptp"][:], z_reg, np2h5["spike_index"][:, 0] / 30000)
    meanproj, _, _ = lib.faster(np2h5["maxptp"][:], z_reg, np2h5["x"][:])
print(meanproj.shape)

# %%
plt.hist(z_reg, bins=np.arange(np.ceil(z_reg.max())), lw=0);
plt.xlabel("registered depths")
plt.ylabel("frequency")

# %%
fig, axes = plt.subplot_mosaic("ab\ncc", figsize=(6, 5.5))
aa = axes["a"]
ab = axes["b"]
ac = axes["c"]

cuts.plot(Rreg, ax=aa)
aa.axhline(1175, lw=1, c="r")
aa.axhline(1870, lw=1, c="r")
aa.set_ylabel("registered depth")
# aa.set_xlabel("time")
aa.set_title("raster: depth vs. time")

ab.hist(z_reg, facecolor="k", edgecolor="w", linewidth=0.2, bins=np.arange(3050, step=50), log=True, color="k")
ab.set_box_aspect(1)
ab.axvline(1175, lw=1, c="r")
ab.axvline(1870, lw=1, c="r")
# ab.set_xlabel("registered depth")
# ab.set_ylabel("frequency")
ab.set_title("histogram of depths")

cuts.plot(meanproj.T, ax=ac, aspect=0.33)
ac.axvline(1175, lw=1, c="r")
ac.axvline(1870, lw=1, c="r")
ac.set_title("mean y projection of ptp on x/z")

ac.text(587.5, 66, "cortex", color="r", backgroundcolor=[1,1,1,0.5], ha="center", va="center")
ac.text(1525, 66, "hippocampus", color="r", backgroundcolor=[1,1,1,0.5], ha="center", va="center")
ac.text(2500, 66, "thalamus", color="r", backgroundcolor=[1,1,1,0.5], ha="center", va="center")

# plt.tight_layout(pad=0.0)
fig.suptitle("cutting the NP2 recording at z=1175,1870")
plt.show()

# %%
with h5py.File("../data/ks_np2.h5", "r") as h5:
    z_reg = h5["z_reg"][:]
    y = h5["y"][:]
    big_y = y > 1e-4
    
    cortex_inds = np.flatnonzero(big_y & (z_reg < 1175))
    hippocampus_inds = np.flatnonzero(big_y & (1175 <= z_reg) & (z_reg < 1870))
    thalamus_inds = np.flatnonzero(big_y & (1870 <= z_reg))
cortex_inds.shape, hippocampus_inds.shape, thalamus_inds.shape, 

# %%
cortex_inds.shape[0] + hippocampus_inds.shape[0] + thalamus_inds.shape[0] == big_y.sum()

# %%
big_y.sum(), len(y)

# %%
regions = {
    "cortex": cortex_inds,
    "hippocampus": hippocampus_inds,
    "thalamus": thalamus_inds,
}

# %%
with h5py.File("../data/ks_np2.h5", "r") as h5:
    N, T, C = h5["denoised_waveforms"].shape
    print(T, C, flush=True)
    
    for region, region_inds in regions.items():
        N_region = len(region_inds)
        print(region, N_region, flush=True)
        
        with h5py.File(f"../data/ks_np2_nzy_{region}.h5", "w") as out:
            # non per-spike data
            for k in ["geom", "templates"]:
                print(k, flush=True)
                out.create_dataset(k, data=h5[k][:])
                
            # small datasets can just be sliced directly
            for k in (k for k in h5.keys() if h5[k].shape in ((1090917,), (1090917, 2))):
                print(k, flush=True)
                out.create_dataset(k, data=h5[k][:][region_inds])
            
            # spikes in a for loop
            print("spikes", flush=True)
            wfs = out.create_dataset("denoised_waveforms", shape=(N_region, T, C))
            in_dnwf = h5["denoised_waveforms"]
            for i, j in tqdm(enumerate(region_inds), total=N_region):
                wfs[i] = in_dnwf[j]

# %%
from spike_psvae import vis_utils

# %%
with h5py.File("../data/ks_np2_nzy_cortex.h5", "r") as ctx:
    vis_utils.locrelocplots(ctx, seed=1)

# %%
