# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.3
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
from scipy.io import loadmat
from scipy import stats, linalg
import matplotlib.pyplot as plt

# %%
from spike_psvae import ibme

# %%
import sys
sys.path.append("/Users/charlie/neuropixelsLFPregistration/python/")
import lfpreg, batchreg

# %%
import colorcet as cc

# %%
plt.rc("figure", dpi=200)

# %%
spikes = loadmat("/Users/charlie/data/raw2.5/spikes.mat")
print(list(spikes.keys()))

# %%
glasbey = np.array(cc.glasbey).squeeze()

# %%
amps = spikes["spikeAmps"].squeeze()
depths = spikes["spikeDepths"].squeeze()
sites = spikes["spikeSites"].squeeze()
times = spikes["spikeTimes"].squeeze()
units = np.load("/Users/charlie/data/raw2.5/spike_clusters.npy").squeeze()

# %%
t_range = times > 110

# %%
which = ~np.isin(units, [449])
plt.figure(figsize=(6, 6))
plt.scatter(times[which & t_range], depths[which & t_range], c=glasbey[units[which & t_range] % 256], s=0.1, alpha=0.2)
plt.show()
plt.figure(figsize=(6, 6))
plt.scatter(times[~which & t_range], depths[~which & t_range], c=glasbey[units[~which & t_range] % 256], s=0.1, alpha=0.2)
plt.show()

# %%
u, c = np.unique(units[t_range], return_counts=True)

# %%
u[np.argsort(c)[::-1]]

# %%
u

# %%
np.savez("/Users/charlie/data/raw2.5/spikes_curated.npz", times=times[which], amps=amps[which], depths=depths[which], units=units[which])

# %%
