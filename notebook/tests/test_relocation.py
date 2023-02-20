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
import numpy as np
import matplotlib.pyplot as plt

# %%
from spike_psvae import relocation

# %%
# a synthetic example on 10 chans

# %%
geom = np.c_[
    np.zeros(10),
    np.arange(10),
]
geom

# %%
spikes = np.zeros((3, 4, 10))
spikes[0, 0, 5] = 1
spikes[1, 0, 5] = 1
spikes[2, 0, 7] = 1
mcs = np.array([5, 5, 7])

# %%
y_from = np.ones(3)
a_from = np.ones(3)
xz_from = geom[mcs]
xyza_from = np.c_[xz_from[:, 0], y_from, xz_from[:, 1], a_from]

# %%
full_ci = np.arange(10)[None, :] * np.ones(10, dtype=int)[:, None]

# %%
# test "simple relocation"

# %%
xyza_to_half = xyza_from.copy()
xyza_to_half[:, 2] += 0.5
reloc_half = relocation.relocate_simple(
    spikes,
    xyza_from,
    xyza_to_half,
    geom,
    max_channels=mcs,
    channel_index=full_ci,
)
np.nonzero(reloc_half), reloc_half[np.nonzero(reloc_half)]

# %%
(1/(1+0.5**2))/(1/1)

# %%
xyza_to_full = xyza_from.copy()
xyza_to_full[:, 2] += 1
reloc_full = relocation.relocate_simple(
    spikes,
    xyza_from,
    xyza_to_full,
    geom,
    max_channels=mcs,
    channel_index=full_ci,
)
np.nonzero(reloc_full), reloc_full[np.nonzero(reloc_full)]

# %%
(1/(1+1))/(1/1)

# %%
# test a helper for fancy relocation

# %%

# %%
# test the fancier one

# %%
# first with full set of chans

# %%
shifted_half = relocation.get_relocated_waveforms_on_channel_subset(
    mcs,
    spikes,
    xyza_from,
    xyza_from[:, 2] + 0.5,
    full_ci,
    geom,
    target_channels=np.arange(10),
    fill_value=np.nan,
)
np.nonzero(shifted_half), shifted_half[np.nonzero(shifted_half)]

# %%
shifted_full = relocation.get_relocated_waveforms_on_channel_subset(
    mcs,
    spikes,
    xyza_from,
    xyza_from[:, 2] + 1.0,
    full_ci,
    geom,
    target_channels=np.arange(10),
    fill_value=np.nan,
)
nans = np.isnan(shifted_full)
nzs = shifted_full != 0
nz_not_nan = nzs & ~nans
print(f"{np.nonzero(nans)=}")
print(f"{np.nonzero(nz_not_nan)=} {shifted_full[np.nonzero(nz_not_nan)]=}")

# %%
shifted_fullhalf = relocation.get_relocated_waveforms_on_channel_subset(
    mcs,
    spikes,
    xyza_from,
    xyza_from[:, 2] + 1.5,
    full_ci,
    geom,
    target_channels=np.arange(10),
    fill_value=np.nan,
)
nans = np.isnan(shifted_fullhalf)
nzs = shifted_fullhalf != 0
nz_not_nan = nzs & ~nans
print(f"{np.nonzero(nans)=}")
print(f"{np.nonzero(nz_not_nan)=} {shifted_fullhalf[np.nonzero(nz_not_nan)]=}")

# %%
shifted_fullhalf = relocation.get_relocated_waveforms_on_channel_subset(
    mcs,
    spikes,
    xyza_from,
    xyza_from[:, 2] + 2.5,
    full_ci,
    geom,
    target_channels=np.arange(10),
    fill_value=np.nan,
)
nans = np.isnan(shifted_fullhalf)
nzs = shifted_fullhalf != 0
nz_not_nan = nzs & ~nans
print(f"{np.nonzero(nans)=}")
print(f"{np.nonzero(nz_not_nan)=} {shifted_fullhalf[np.nonzero(nz_not_nan)]=}")

# %%
shifted_fullhalf = relocation.get_relocated_waveforms_on_channel_subset(
    mcs,
    spikes,
    xyza_from,
    xyza_from[:, 2] + 3.5,
    full_ci,
    geom,
    target_channels=np.arange(10),
    fill_value=np.nan,
)
nans = np.isnan(shifted_fullhalf)
nzs = shifted_fullhalf != 0
nz_not_nan = nzs & ~nans
print(f"{np.nonzero(nans)=}")
print(f"{np.nonzero(nz_not_nan)=} {shifted_fullhalf[np.nonzero(nz_not_nan)]=}")

# %%
shifted_fullhalf = relocation.get_relocated_waveforms_on_channel_subset(
    mcs,
    spikes,
    xyza_from,
    xyza_from[:, 2] - 5,
    full_ci,
    geom,
    target_channels=np.arange(10),
    fill_value=np.nan,
)
nans = np.isnan(shifted_fullhalf)
nzs = shifted_fullhalf != 0
nz_not_nan = nzs & ~nans
print(f"{np.nonzero(nans)=}")
print(f"{np.nonzero(nz_not_nan)=} {shifted_fullhalf[np.nonzero(nz_not_nan)]=}")

# %%
shifted_fullhalf = relocation.get_relocated_waveforms_on_channel_subset(
    mcs,
    spikes,
    xyza_from,
    xyza_from[:, 2] - 5.5,
    full_ci,
    geom,
    target_channels=np.arange(10),
    fill_value=np.nan,
)
nans = np.isnan(shifted_fullhalf)
nzs = shifted_fullhalf != 0
nz_not_nan = nzs & ~nans
print(f"{np.nonzero(nans)=}")
print(f"{np.nonzero(nz_not_nan)=} {shifted_fullhalf[np.nonzero(nz_not_nan)]=}")

# %%
shifted_fullhalf = relocation.get_relocated_waveforms_on_channel_subset(
    mcs,
    spikes,
    xyza_from,
    xyza_from[:, 2] - 6,
    full_ci,
    geom,
    target_channels=np.arange(10),
    fill_value=np.nan,
)
nans = np.isnan(shifted_fullhalf)
nzs = shifted_fullhalf != 0
nz_not_nan = nzs & ~nans
print(f"{np.nonzero(nans)=}")
print(f"{np.nonzero(nz_not_nan)=} {shifted_fullhalf[np.nonzero(nz_not_nan)]=}")

# %%
