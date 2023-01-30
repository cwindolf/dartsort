# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
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
ones = torch.ones(11)
up = torch.arange(11.)
tri = 5 - torch.abs(up - 5)
tri

# %%

# %%
normxcorr1d(up[None], up[None])

# %%
normxcorr1d(tri[None], tri[None])

# %%
normxcorr1d(up[None], up[None], weights=tri)

# %%
normxcorr1d(tri[None], tri[None], weights=tri)

# %%
normxcorr1d(torch.arange(22.).reshape(2, 11), up[None], weights=tri)

# %%
