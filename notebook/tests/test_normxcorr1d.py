# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python [conda env:a]
#     language: python
#     name: conda-env-a-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import torch
from spike_psvae.ibme_corr import normxcorr1d

# %%
import torch.nn.functional as F

# %%
import matplotlib.pyplot as plt

# %%
ones = torch.ones(11)
up = torch.arange(11.)
tri = 5 - torch.abs(up - 5)
tri

# %%
tri2 = torch.cat((tri, tri))
tri2.shape

# %%
normxcorr1d(up[None], up[None])

# %%
normxcorr1d(tri[None], tri[None])

# %%
normxcorr1d(up[None], up[None], weights=tri)

# %%
normxcorr1d(tri[None], tri[None], weights=tri)

# %%
plt.plot(normxcorr1d(tri[None], tri[None], weights=tri).squeeze())

# %%
normxcorr1d(torch.arange(22.).reshape(2, 11), up[None], weights=tri)

# %%
normxcorr1d(tri[None], tri2[None], weights=tri)

# %%
plt.plot(normxcorr1d(tri[None], tri2[None], weights=tri).squeeze())

# %%
