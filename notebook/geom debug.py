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
from spike_psvae import vis_utils, waveform_utils
import h5py

# %%
plt.rc("figure", dpi=200)

# %%
wfs = np.load("../data/spt_yass_templates.npy")[:16]
geom = np.load("../data/np2_channel_map.npy")

# %%
maxchans = wfs.ptp(1).argmax(1)


# %%
def norm(x):
    return (x - x.min()) / (x.max() - x.min())


# %%
print(); print(); print("run standard")
std_wfs = waveform_utils.get_local_waveforms(wfs, 8, geom, maxchans, geomkind="standard")
print(std_wfs.shape)

print(); print(); print("run updown")
ud_wfs = waveform_utils.get_local_waveforms(wfs, 10, geom, maxchans, geomkind="updown")
print(ud_wfs.shape)

print(); print(); print("run convert")
uds_wfs = waveform_utils.as_standard_local(ud_wfs, maxchans, geom)
print(uds_wfs.shape)

# %%
(uds_wfs == std_wfs).mean(axis=(1,2)), (uds_wfs == std_wfs).mean()

# %%
vis_utils.labeledmosaic(
    [norm(uds_wfs), norm(std_wfs), (uds_wfs == std_wfs).astype(float),],
    rowlabels=range(3),
    collabels=range(200),
    pad=5, cbar=False,
)

# %%
fig, axes = vis_utils.vis_ptps([std_wfs.ptp(1)[inds], q[inds]], ["observed ptp", "predicted ptp"], "bg")
plt.suptitle(f"{name}: PTP predictions", fontsize=8)
plt.tight_layout(pad=0.25)
plt.show()
fig, axes = vis_utils.vis_ptps([reloc.ptp(1)[inds], r[inds]], ["relocated ptp", "standard ptp"], "kr")
plt.suptitle(f"{name}: Relocated PTPs", fontsize=8)
plt.tight_layout(pad=0.25)
plt.show()

# %%
