# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import h5py
from spike_psvae import vis_utils
import torch
import matplotlib.pyplot as plt

# %%
plt.rc("figure", dpi=200)

# %%
with h5py.File("../data/wfs_locs_tiny.h5") as f:
    wfs = f["denoised_waveforms"][:]
print(wfs.shape)

# %%
x1 = torch.tensor(wfs[:8])
x2 = torch.tensor(wfs[8:16])
x3 = torch.tensor(wfs[16:24])

# %%
vis_utils.labeledmosaic([x1, x2, x3], ["first", "second", "third"], pad=2)

# %%
