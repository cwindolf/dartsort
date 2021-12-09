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
import h5py
import matplotlib.pyplot as plt
import scipy.linalg as la
import seaborn as sns
import time

# %%
from npx import lib, reg, cuts

# %%
from isosplit import isosplit

# %%
from spike_psvae import waveform_utils, localization, point_source_centering, vis_utils

# %%
rg = lambda k=0: np.random.default_rng(k)

# %%
plt.rc("figure", dpi=200)

# %% [markdown]
# ## parameters

# %%
# single channel denoised data
original_h5 = "../data/wfs_locs_c.h5"

# %%
# we will write some waveforms data here
full_h5 = "../data/story_full_c.h5"
# this one will just have features, cluster ids, lightweight stuff
# that can be rsyncd to local for datoviz
feats_h5 = "../data/story_feats_c.h5"


# %% [markdown]
# ## helpers

# %%
class timer:
    def __init__(self, name="timer"):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.t = time.time() - self.start
        print(self.name, "took", self.t, "s")


# %%

# %%

# %% [markdown]
# ## initial processing: localization, registration

# %%
with h5py.File(original_h5, "r") as orig_f:
    xs, ys, z_rels, z_abss, alphas = localization.localize_waveforms_batched(
        orig_f["denoised_waveforms"],
        orig_f["geom"][:],
        maxchans=orig_f["max_channels"][:],
        channel_radius=10,
        n_workers=10,
        jac=True,
        firstchans=orig_f["first_channels"][:],
        geomkind="firstchan",
        batch_size=1024,
    )

# %%
with h5py.File(original_h5, "r") as orig_f:
    times = orig_f["spike_index"][:, 0].astype(float) / 30_000
    with timer("maxptp"):
        maxptp = orig_f["denoised_waveforms"][:].ptp(1).ptp(1)

# %%
with timer("registration"):
    z_reg, dispmap = reg.register_nonrigid(
        maxptp,
        z_abss,
        times,
        robust_sigma=1,
        rigid_disp=200,
        disp=100,
        denoise_sigma=0.1,
        destripe=False,
        n_windows=[5, 30, 60],
        n_iter=1,
        widthmul=0.25,
    )

# %%
R0, _, _ = lib.faster(maxptp, z_abss, times)
Rreg, _, _ = lib.faster(maxptp, z_reg, times)

# %%
fig, (aa, ab) = plt.subplots(1, 2, figsize=(5, 10))
cuts.plot(R0, ax=aa)
cuts.plot(Rreg, ax=ab)
plt.tight_layout()
plt.show()

# %%
z_disp = z_reg - (z_abss - z_abss.min())
fig, ax = plt.subplots()
a = ax.scatter(times, z_reg, c=np.abs(z_disp), s=0.5, alpha=0.5)
cbar = fig.colorbar(a, ax=ax)
cbar.solids.set_alpha(1)
plt.title("Spikes colored by absolute displacement")
plt.ylabel("registered depth")
plt.xlabel("time")
plt.show()

# %%
# let's pause to save these somewhere
with h5py.File(feats_h5, "w") as feats_f:
    for k, v in zip(
        ["x", "y", "z_rel", "z_abs", "z_reg", "z_disp", "alpha", "maxptp", "times"],
        [xs, ys, z_rels, z_abss, z_reg, z_disp, alphas, maxptp, times],
    ):
        feats_f.create_dataset(k, data=v)

# %% [markdown]
# ## relocation

# %%
with h5py.File(feats_h5, "r") as feats_f:
    N = len(feats_f["x"])
# 9 random spike indices
example_inds = rg().choice(N, 9)
example_inds.sort()

# %%
with h5py.File(original_h5, "r") as orig_f:
    exwf = orig_f["denoised_waveforms"][example_inds]
    geom = orig_f["geom"][:]
    exmc = orig_f["max_channels"][example_inds]

# %%
ex_x = xs[example_inds]
ex_y = ys[example_inds]
ex_zr = z_rels[example_inds]
ex_a = alphas[example_inds]

# %%
vis_utils.labeledmosaic(exwf.reshape(3, 3, 121, 20)[:, :, 20:-40, :], [1, 2, 3], pad=1);

# %%
p = exwf.ptp(1)
exwf_yza, q_hat_yza, p_hat = point_source_centering.relocate_simple(
    exwf, geom, exmc, ex_x, ex_y, ex_zr, ex_a,
    relocate_dims="yza",
)
exwf_xyza, q_hat_xyza, p_hat_ = point_source_centering.relocate_simple(
    exwf, geom, exmc, ex_x, ex_y, ex_zr, ex_a,
    relocate_dims="yza",
)
assert (p_hat == p_hat_).all()

# %%
