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
import matplotlib.pyplot as plt
from spike_psvae import vis_utils, waveform_utils, localization
import h5py

# %%
plt.rc("figure", dpi=200)

# %%
wfs = np.load("../data/spt_yass_templates.npy")
geom = np.load("../data/np2_channel_map.npy")

# %%
maxchans = wfs.ptp(1).argmax(1)

# %%
local_wfs = waveform_utils.get_local_waveforms(wfs, 10, geom, maxchans, geomkind="standard")

# %%
fig = plt.figure(figsize=(4, 10))
vis_utils.labeledmosaic(
    [local_wfs[i : i + 13] for i in range(0, local_wfs.shape[0] // 2, 13)],
    rowlabels=range(0, local_wfs.shape[0] // 2, 13),
    collabels=range(200),
    pad=5,
    cbar=False,
    separate_norm=True,
)

# %%
end = max(range(0, local_wfs.shape[0] // 2, 13)) + 13
end

# %%
fig = plt.figure(figsize=(4, 10))
vis_utils.labeledmosaic(
    [local_wfs[i : i + 13] for i in range(end, local_wfs.shape[0], 13)],
    rowlabels=range(end, local_wfs.shape[0], 13),
    collabels=range(200),
    pad=5,
    cbar=False,
    separate_norm=True,
)

# %%
rid = [3, 35, 55, 58, 79, 100, 109, 120, 144, 163, 164]
out = np.setdiff1d(np.arange(wfs.shape[0]), rid)

# %%
with h5py.File("../data/spt_yasstemplates_culled.h5", "w") as h5:
    h5.create_dataset("waveforms", data=wfs[out])
    h5.create_dataset("geom", data=geom)
    h5.create_dataset("maxchans", data=maxchans[out])

# %%
with h5py.File("../data/spt_yasstemplates.h5", "w") as h5:
    h5.create_dataset("waveforms", data=wfs)
    h5.create_dataset("geom", data=geom)
    h5.create_dataset("maxchans", data=maxchans)

# %%
xs, ys, z_rels, z_abss, alphas = localization.localize_waveforms(local_wfs, geom, maxchans, geomkind="standard")

# %%
(ys < 0.01).mean()

# %%
smally = np.flatnonzero(ys < 0.01)
outout = np.setdiff1d(out, smally)

# %%
with h5py.File("../data/spt_yasstemplates_culled_ymin0.01.h5", "w") as h5:
    h5.create_dataset("waveforms", data=wfs[outout])
    h5.create_dataset("geom", data=geom)
    h5.create_dataset("maxchans", data=maxchans[outout])

# %%
