# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python [conda env:dartsort]
#     language: python
#     name: conda-env-dartsort-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2
# %config InlineBackend.figure_format = 'retina'

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import spikeinterface.full as si

import dartsort
from dartsort.evaluate import simkit
import dartsort.vis as dartvis
from dartsort.util.testing_util import matching_debug_util

from cycler import cycler
plt.rc('axes', prop_cycle=cycler(color=list(map(tuple, dartvis.glasbey1024))))


# %%
# test_dir = dartsort.resolve_path("~/scratch/test")
test_dir = dartsort.resolve_path("~/data/test")

# %% [markdown]
# # Sim

# %%
# make a single channel recording
res = simkit.generate_simulation(
    folder=test_dir / "tmprec",
    noise_recording_folder=test_dir / "noi",
    n_units=5,
    min_fr_hz=100.0,
    max_fr_hz=150.0,
    duration_seconds=0.5,
    geom=np.zeros((1, 2)),
    noise_kind="white",
    white_noise_scale=0.0,
    # amplitude_jitter=0.0,
    temporal_jitter=1,
    amplitude_jitter=0.0,
    globally_refractory=True,
    # refractory_ms=1.0,
    refractory_ms=5.0,
    # refractory_ms=1.0,
    drift_speed=0.0,
    # noise_kind="zero",
    # noise_in_memory=True,
    overwrite=True,
)
rec = res["recording"]
traces = rec.get_traces()
np.linalg.norm(res["templates"].templates, axis=(1, 2))

# %%
fig, axes = plt.subplots(ncols=2, figsize=(5, 2.5))
axes[0].plot(res["templates"].templates[:, :, 0].T);
axes[1].plot(traces[:1000])

# %%
res["sorting"]

# %% [markdown]
# # Lo

# %%
matcher = dartsort.ObjectiveUpdateTemplateMatchingPeeler.from_config(
    rec,
    waveform_cfg=dartsort.default_waveform_cfg,
    featurization_cfg=dartsort.default_featurization_cfg,
    matching_cfg=dartsort.MatchingConfig(
        threshold=5.,
        template_type="drifty",
        up_method="keys4",
        
        template_temporal_upsampling_factor=1,
        amplitude_scaling_variance=0.0,
        
        # template_temporal_upsampling_factor=16,
        # amplitude_scaling_variance=0.1,

        template_svd_compression_rank=121,
        # cd_iter=5,
        cd_iter=0,
    ),
    template_data=res["templates"],
)
matcher.precompute_models(test_dir / 'tmp', overwrite=True)
if torch.cuda.is_available():
    matcher = matcher.cuda()

# %%
# ctd = matcher.matching_templates.data_at_time(0.0, scaling=False, inv_lambda=float('inf'), scale_min=1.0, scale_max=1.0)
ctd = matcher.matching_templates.data_at_time(0.0, scaling=True, inv_lambda=100.0, scale_min=0.5, scale_max=1.5)

# %%
res["sorting"].times_samples[:10] - 42

# %%
ctd.temporal_comps.shape, ctd.spatial_sing.shape

# %%
matcher.to(ctd.spatial_sing.device);

# %%
matcher.max_iter = 100
chk = matcher.match_chunk(
    torch.asarray(traces.copy()).float().to(ctd.spatial_sing),
    ctd, return_conv=True, return_residual=True)

# %%
chk.keys()

# %%
ctd.temporal_comps.shape, ctd.temporal_comps_up.shape, ctd.spatial_sing.shape

# %%
# plt.plot(res["templates"].templates[1, :, 0])
# plt.plot((ctd.temporal_comps * ctd.spatial_sing[1]).sum(0))
# for u in range(16):
#     plt.plot((ctd.temporal_comps_up[u] * ctd.spatial_sing[1]).sum(0), c=plt.cm.viridis(u/16))

# %%
# u = 3
# plt.plot(
#     res["templates"].templates[1, :, 0]
#     - (ctd.temporal_comps_up[u] * ctd.spatial_sing[1]).sum(0).numpy(), c=plt.cm.viridis(u/16)
# )

# %%
chk['conv'].shape

# %%
fig, (aa, ab, ac) = plt.subplots(figsize=(15, 4), sharex=True, nrows=3, layout='constrained')
t0 = 0
t1 = 1000
aa.plot(traces[t0:t1])
aa.plot(chk['residual'][t0:t1].numpy(force=True))
ab.plot(chk['conv'].T[t0:t1].numpy(force=True))
ac.plot(chk['residual'][t0:t1].numpy(force=True))

for ax, sh in zip((aa, ab, ac), (0, 42, 0)):
    ax.grid();
    ax.set_xticks(np.arange(t0, t1, 100))
    for t, l in zip(res["sorting"].times_samples, res["sorting"].labels):
        if t < t0:
            continue
        if t > t1:
            break
        ax.axvline(t - t0 + sh, c=dartvis.glasbey1024[l])
    for t, l in zip(chk['times_samples'].numpy(force=True), chk['labels'].numpy(force=True)):
        if t < t0:
            continue
        if t > t1:
            break
        ax.axvline(t - t0 + sh, c='k', ls="--")
        ax.axvline(t - t0 + sh, c=dartvis.glasbey1024[l], ls=":")

# %%

# %%

# %%
fig, ax = plt.subplots(figsize=(15, 4), layout='constrained')
ax.plot(chk['residual'][t0:t1].numpy(force=True))

# %%
conv2 = ctd.convolve(chk['residual'].T, padding=120)

# %%
(2 * conv2 - ctd.obj_normsq[:, None]).max()

# %%
fig, ax = plt.subplots(figsize=(15, 4), layout='constrained')
ax.plot((chk['conv'] - conv2)[:, t0:t1].T.numpy(force=True))

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # Hi

# %%
st = dartsort.match(
    test_dir / "himatch",
    rec,
    template_data=res["templates"],
    overwrite=True,
    matching_cfg=dartsort.MatchingConfig(
        threshold=5.,
        template_type="drifty",
        up_method="keys4",
        # template_temporal_upsampling_factor=1,
        amplitude_scaling_variance=0.00001,
        # template_svd_compression_rank=100,
        # cd_iter=5,
    ),
)
st

# %%
viol = np.flatnonzero(np.diff(st.times_samples) < 10)
viol = np.unique(np.concatenate((viol, viol + 1)))
viol

# %%
t0 = st.times_samples[viol][0] - 100
t0

# %%
res["sorting"].times_samples

# %%
st.times_samples[viol]

# %%
st.labels[viol]

# %%
for j, temp in enumerate(res["templates"].templates):
    plt.plot(temp, c=dartvis.glasbey1024[j], label=j)
plt.legend()

# %%
fig, ax = plt.subplots(figsize=(15, 4), layout='constrained')
t0 = t0
t1 = t0 +1000
ax.plot(traces[t0:t1])
for t in res["sorting"].times_samples:
    if t < t0:
        continue
    if t > t1:
        break
    ax.axvline(t - t0, c='k')
for t in st.times_samples:
    if t < t0:
        continue
    if t > t1:
        break
    ax.axvline(t - t0, c='r', ls="--")

# %%
fig, ax = plt.subplots(figsize=(15, 4), layout='constrained')
t0 = t0
t1 = t0 + 1000
ax.plot(traces[t0:t1])
for t in res["sorting"].times_samples:
    if t < t0:
        continue
    if t > t1:
        break
    ax.axvline(t - t0, c='k')
for t in st.times_samples:
    if t < t0:
        continue
    if t > t1:
        break
    ax.axvline(t - t0, c='r', ls="--")

# %%
plt.hist(np.diff(st.times_samples), bins=np.arange(100));

# %%

# %% [markdown]
# # Real data

# %%
iblscr = dartsort.resolve_path("~/scratch/ibl")
iblscr.mkdir(exist_ok=True)
asset_path = "sub-CSH-ZAD-026/sub-CSH-ZAD-026_ses-15763234-d21e-491f-a01b-1238eb96d389_behavior+ecephys+image.nwb"
rec = si.read_binary_folder(iblscr / f"ppx_{asset_path.split("/")[0]}")

# %%
tag = "v0.3.4_1217_driftykeys4_dfix_scaley_spl"
template_data = dartsort.TemplateData.from_npz(
    iblscr / f"ds_{tag}_{asset_path.split("/")[0]}" / "matching1_models" / "template_data.npz"
)

# %%
matcher = dartsort.ObjectiveUpdateTemplateMatchingPeeler.from_config(
    rec,
    waveform_cfg=dartsort.default_waveform_cfg,
    featurization_cfg=dartsort.default_featurization_cfg,
    matching_cfg=dartsort.MatchingConfig(
        template_type="drifty",
        up_method="keys4",
        
        # template_temporal_upsampling_factor=1,
        # amplitude_scaling_variance=0.0,
        
        template_temporal_upsampling_factor=4,
        amplitude_scaling_variance=0.1,

        template_svd_compression_rank=10,
        cd_iter=0,
    ),
    template_data=template_data,
)
matcher.precompute_models(test_dir / 'ibltesttmp', overwrite=True)
matcher = matcher.cuda()

# %%
matching_debug_util.visualize_step_results(
    matcher=matcher,
    chunk=rec.get_traces(0, 100_000, 101_500).copy(),
    t_s=0.1,
    max_iter=500,
    s=20,
    cmap="berlin",
    vis_start=500,
    vis_end=1000,
    obj_mode=True,
)

# %% [markdown]
# # Tail

# %%
