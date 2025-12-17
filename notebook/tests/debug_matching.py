# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python [conda env:dart]
#     language: python
#     name: conda-env-dart-py
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import dartsort
from dartsort.evaluate import simkit
import dartsort.vis as dartvis

# %%
test_dir = dartsort.resolve_path("~/scratch/test")

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
    # temporal_jitter=1,
    refractory_ms=4.6,
    # refractory_ms=1.0,
    drift_speed=0.0,
    globally_refractory=True,
    # noise_kind="zero",
    # noise_in_memory=True,
    overwrite=True,
)
rec = res["recording"]

# %%
np.linalg.norm(res["templates"].templates, axis=(1, 2))

# %%
traces = rec.get_traces()
traces.shape

# %%
plt.plot(res["templates"].templates[:, :, 0].T)

# %%
plt.plot(traces[:1000])

# %%
res["sorting"]

# %%
matcher = dartsort.ObjectiveUpdateTemplateMatchingPeeler.from_config(
    rec,
    waveform_cfg=dartsort.default_waveform_cfg,
    featurization_cfg=dartsort.default_featurization_cfg,
    matching_cfg=dartsort.MatchingConfig(
        threshold=5.2,
        template_type="drifty",
        up_method="keys4",

        
        # template_temporal_upsampling_factor=1,
        # amplitude_scaling_variance=0.0,
        
        template_temporal_upsampling_factor=16,
        amplitude_scaling_variance=0.1,

        template_svd_compression_rank=121,
        # cd_iter=5,
        cd_iter=0,
    ),
    template_data=res["templates"],
)

# %%
matcher.precompute_models(test_dir / 'tmp', overwrite=True)

# %%
# ctd = matcher.matching_templates.data_at_time(0.0, scaling=False, inv_lambda=float('inf'), scale_min=1.0, scale_max=1.0)
ctd = matcher.matching_templates.data_at_time(0.0, scaling=True, inv_lambda=100.0, scale_min=0.5, scale_max=1.5)

# %%
res["sorting"].times_samples[:10] - 42

# %%
ctd.temporal_comps.shape, ctd.spatial_sing.shape

# %%
matcher.to(ctd.spatial_sing.device)

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
(2 * chk['conv'].T - ctd.obj_normsq).max()

# %%
chk['times_samples'][:10] - 42

# %%
chk["scalings"][:10]

# %%
chk["time_shifts"][:10]

# %%
chk["upsampling_indices"][:5]

# %%
chk["template_indices"][:5]

# %%
chk["labels"][:5]

# %%
res["sorting"].labels[:5]

# %%
res["sorting"].times_samples[:5] - 42

# %%
res["sorting"].jitter_ix[:5]

# %%
res["sorting"].scalings[:5]

# %%
res["sorting"].time_shifts[:5]

# %%
fig, ax = plt.subplots(figsize=(15, 4), layout='constrained')
t0 = 0
t1 = 1000
ax.plot(traces[t0:t1])
ax.plot(chk['residual'][t0:t1].numpy(force=True))
for t, l in zip(res["sorting"].times_samples, res["sorting"].labels):
    if t < t0:
        continue
    if t > t1:
        break
    ax.axvline(t - t0, c=dartvis.glasbey1024[l])
for t, l in zip(chk['times_samples'].numpy(force=True), chk['labels'].numpy(force=True)):
    if t < t0:
        continue
    if t > t1:
        break
    ax.axvline(t - t0, c='k', ls="--")
    ax.axvline(t - t0, c=dartvis.glasbey1024[l], ls=":")
plt.xticks(np.arange(t0, t1, 100))
plt.grid();

# %%
fig, ax = plt.subplots(figsize=(15, 4), layout='constrained')
ax.plot(chk['residual'][t0:t1])

# %%
conv2 = ctd.convolve(chk['residual'].T, padding=120)

# %%
(2 * conv2 - ctd.obj_normsq[:, None]).max()

# %%
fig, ax = plt.subplots(figsize=(15, 4), layout='constrained')
ax.plot((chk['conv'] - conv2)[:, t0:t1].T)

# %%
fig, ax = plt.subplots(figsize=(15, 4), layout='constrained')
ax.plot(2 * chk['conv'][:, t0:t1].T - ctd.obj_normsq)

# %%

# %%
st = dartsort.match(test_dir / "himatch", rec, template_data=res["templates"], overwrite=True,
                   matching_cfg=dartsort.MatchingConfig(
                       threshold=5.,
                       # template_type="drifty",
                       # up_method="keys4",
                       # template_temporal_upsampling_factor=1,
                       # amplitude_scaling_variance=0.0,
                       # template_svd_compression_rank=100,
                       # cd_iter=5,
                   ),
                   )
st

# %%
fig, ax = plt.subplots(figsize=(15, 4), layout='constrained')
t0 = 0
t1 = 1000
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
t0 = 0
t1 = 1000
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
