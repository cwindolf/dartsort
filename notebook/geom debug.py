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
from spike_psvae import vis_utils, waveform_utils, localization, point_source_centering
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
std_x, std_y, std_z_rel, std_z_abs, std_alpha = localization.localize_waveforms(std_wfs, geom, maxchans, channel_radius=8, jac=False, geomkind="standard")
std_reloc, std_r, std_q = point_source_centering.relocate_simple(std_wfs, geom, maxchans, std_x, std_y, std_z_rel, std_alpha, channel_radius=8, geomkind="standard")

ud_x, ud_y, ud_z_rel, ud_z_abs, ud_alpha = localization.localize_waveforms(ud_wfs, geom, maxchans, jac=False, geomkind="updown")
ud_reloc, ud_r, ud_q = point_source_centering.relocate_simple(ud_wfs, geom, maxchans, ud_x, ud_y, ud_z_rel, ud_alpha, geomkind="updown")

uds_x, uds_y, uds_z_rel, uds_z_abs, uds_alpha = localization.localize_waveforms(uds_wfs, geom, maxchans, channel_radius=8, jac=False, geomkind="standard")
uds_reloc, uds_r, uds_q = point_source_centering.relocate_simple(uds_wfs, geom, maxchans, uds_x, uds_y, uds_z_rel, uds_alpha, channel_radius=8, geomkind="standard")

# %%
vis_utils.vis_ptps([std_wfs.ptp(1), std_q], ["std", "pred"], "kr");

# %%
vis_utils.vis_ptps([ud_wfs.ptp(1), ud_q], ["ud", "pred"], "kr");

# %%
vis_utils.vis_ptps([uds_wfs.ptp(1), uds_q], ["uds", "pred"], "kr");

# %%

# %%
