# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python [conda env:mysi]
#     language: python
#     name: conda-env-mysi-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
from spike_psvae import ibme

# %%
with h5py.File("/Users/charlie/data/subtraction_KS055_std.ap_t_0_None.h5", "r") as h5:
    z = h5["localizations"][:, 2].T
    wh = z < 800
    z = z[wh]
    a = h5["maxptps"][:][wh]
    t = h5["spike_index"][:, 0][wh] / 30000
    geom = h5["geom"][:]

# %%
r, se, te = ibme.fast_raster(a, z, t)
assert r.shape == (se.size - 1, te.size - 1)

# %%
assert se[0] <= z.min() < z.max() <= se[-1]
assert te[0] <= t.min() < t.max() <= te[-1]

# %%
r3, se, te = ibme.fast_raster(a, z, t, gaussian_smoothing_sigma_um=3)
assert r.shape == (se.size - 1, te.size - 1)

# %%
r.shape

# %%
plt.imshow(r3, aspect="auto", vmax=15, cmap=plt.cm.cubehelix)

# %%
rme, extra = ibme.register_rigid(a, z, t, disp=100, batch_size=256)

# %%
tbc = (te[1:] + te[:-1]) / 2

# %%
plt.imshow(r, aspect="auto", vmax=15, cmap=plt.cm.cubehelix)
plt.plot(rme.disp_at_s(tbc) + 500, color="r")

# %%

# %%
tme, extra = ibme.register_nonrigid(
    a,
    z,
    t,
    geom[geom[:, 1] <= 860],
    disp=100,
    batch_size=512,
    win_step_um=300,
    win_sigma_um=300,
    window_shape="rect",
)

# %%
extra["window_centers"]

# %%
plt.imshow(r, aspect="auto", vmax=15, cmap=plt.cm.cubehelix)
for pos in extra["window_centers"]:
    plt.plot(pos + rme.disp_at_s(tbc, depth_um=pos), color="r")

# %%
