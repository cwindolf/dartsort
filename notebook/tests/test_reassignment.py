# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
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
import h5py

# %%
import matplotlib.pyplot as plt

# %%
plt.rc("figure", dpi=300)

# %%
from spike_psvae import deconvolve, extract_deconv, reassignment, cluster_viz_index, drifty_deconv, snr_templates

# %%
spikelen = 21
reclen = 5000
trough = 0
nchan = 10

# %%
write_chan_ix = np.arange(nchan)[None, :] * np.ones(nchan, dtype=int)[:, None]

# %%
# basic geom
geom = np.c_[
    np.zeros(nchan),
    np.arange(nchan),
]
geom

# %%
# fake bank of osc templates
domain = np.linspace(0, 2 * np.pi, num=spikelen)
sin = 10 * np.sin(domain)
sin2 = 10 * np.sin(2 * domain)

t0 = np.zeros((spikelen, nchan))
ta = t0.copy()
ta[:, 2] = sin
tb = t0.copy()
tb[:, 2] = sin2
tc = t0.copy()
tc[:, 5] = sin

templates = np.r_[ta[None], tb[None], tc[None]]
templates.shape

# %%
rg = np.random.default_rng(0)
def getst(tmin, tmax, size, ref=10):
    while True:
        st = rg.choice(
            np.arange(tmin, tmax), size=size, replace=False
        )
        st.sort()
        if np.diff(st).min() >= ref:
            return st


# %%
sta = getst(0, reclen - spikelen, size=50)
sta = np.c_[sta, np.zeros_like(sta)]
stb = getst(0, reclen - spikelen, size=50)
stb = np.c_[stb, 1 + np.zeros_like(stb)]
stc = getst(0, reclen - spikelen, size=50)
stc = np.c_[stc, 2 + np.zeros_like(stc)]
st = np.r_[sta, stb, stc]
st.shape

# %%
time_range = np.arange(-trough, spikelen - trough)
time_ix = st[:, 0, None] + time_range[None, :]
chan_ix = write_chan_ix[st[:, 1]]
raw = rg.normal(size=(reclen, nchan), scale=0.1).astype(np.float32)
np.add.at(
    raw,
    (time_ix[:, :, None], chan_ix[:, None, :]),
    templates[st[:, 1]],
)

# %%
raw.tofile("/tmp/testaaa.bin")

# %%
waveforms = raw[time_ix[:, :, None], chan_ix[:, None, :]]

# %%
waveforms.shape

# %%
fake_proposed_pairs = [np.arange(3)] * 3

# %%
plt.hist(np.abs(sta[:, 0][None, :] - stb[:, 0][:, None]).ravel(), bins=np.arange(50));

# %%
temps_loc = reassignment.reassignment_templates_local(templates, fake_proposed_pairs, write_chan_ix)
[t.shape for t in temps_loc]

# %% [markdown]
# # reas no deconv

# %%
new_labels, outlier_scores = reassignment.reassign_waveforms(st[:, 1], waveforms, fake_proposed_pairs, temps_loc, norm_p=2)

# %%
cluster_viz_index.reassignments_viz(
    st,
    new_labels,
    "/tmp/testaaa.bin",
    "/Users/charlie/data/testreas",
    geom,
    radius=200,
    z_extension=1.5,
    trough_offset=trough,
    spike_length_samples=spikelen,
    templates=templates,
)

# %% [markdown]
# # reas deconv

# %%
np.square(templates).sum(axis=(1, 2))

# %%
deconv_res = deconvolve.deconv(
    "/tmp/testaaa.bin",
    "/tmp/deconvaaa",
    templates,
    threshold=50,
    # max_upsample=1,
    trough_offset=trough,
)

# %%
dst = deconv_res['deconv_spike_train']
dstu = deconv_res['deconv_spike_train_upsampled']

# %%
dst[:,0].min(), dst[:, 0].max()

# %%
deconv_res.keys()

# %%
tu = deconv_res["templates_up"]

# %%
tu.shape

# %%
te, si, sp = deconv_res["temporal"], deconv_res["singular"], deconv_res["spatial"]
te.shape, si.shape, sp.shape

# %%
tsvd = np.einsum("ntk,nk,nks->nts", te, si, sp)

# %%
tsvd.shape

# %%
for t, c in zip(tsvd, "rgb"):
    cluster_viz_index.pgeom(t[None], geom=geom, color=c, max_abs_amp=10);

# %%
np.unique(dst[:, 1], return_counts=True)

# %%
deconv_res['sparse_id_to_orig_id']

# %%
deconv_res['deconv_id_sparse_temp_map']

# %%
deconv_res['deconv_id_sparse_temp_map'][dstu[:, 1]] == dstu[:, 1]

# %%
fake_proposed_pairs_up = [np.arange(len(deconv_res['templates_up']))] * len(deconv_res['templates_up'])

# %%
h5 = extract_deconv.extract_deconv(
    deconv_res['templates_up'],
    deconv_res['deconv_spike_train_upsampled'],
    "/tmp/deconvaaa",
    "/tmp/testaaa.bin",
    geom=geom,
    channel_index=write_chan_ix,
    reassignment_proposed_pairs_up=fake_proposed_pairs_up,
    save_cleaned_waveforms=True,
    do_reassignment_tpca=False,
    save_cleaned_tpca_projs=False,
    localize=False,
    trough_offset=0,
)

# %%
with h5py.File(h5) as h:
    print(list(h.keys()))
    reas_labels_up = h["reassigned_labels_up"][:]
    reas_labels = deconv_res["sparse_id_to_orig_id"][reas_labels_up]

# %%
deconv_res['deconv_spike_train'].shape

# %%
reas_labels.shape

# %%
dst.min(), dst.max()

# %%
cluster_viz_index.reassignments_viz(
    deconv_res['deconv_spike_train'],
    reas_labels,
    "/tmp/testaaa.bin",
    "/Users/charlie/data/testreasdc",
    geom,
    radius=200,
    z_extension=1.5,
    trough_offset=trough,
    spike_length_samples=spikelen,
    templates=templates,
)

# %% [markdown]
# # test with drifty

# %%
drift = 2

# %%
reclen_drift = 2 * reclen
drift_p = np.zeros(reclen_drift)
drift_p[reclen_drift:] = drift
drift_p_byfs = np.array([0., 2])

# %%
# fake bank of osc templates
domain = np.linspace(0, 2 * np.pi, num=spikelen)
sin = 10 * np.sin(domain)
sin2 = 10 * np.sin(2 * domain)

t0 = np.zeros((spikelen, nchan))
ta = t0.copy()
ta[:, 2] = sin
tad = t0.copy()
tad[:, 2 + drift] = sin
tb = t0.copy()
tb[:, 2] = sin2
tbd = t0.copy()
tbd[:, 2 + drift] = sin2
tc = t0.copy()
tc[:, 5] = sin
tcd = t0.copy()
tcd[:, 5 + drift] = sin

xyza_orig = np.c_[
    geom[[2, 2, 5]][:, 0],
    np.ones(3),
    geom[[2, 2, 5]][:, 1],
    np.ones(3),
]
xyza_drift = np.r_[xyza_orig, xyza_orig + [0, 0, drift, 0]]

templates_drift = np.r_[ta[None], tb[None], tc[None], tad[None], tbd[None], tcd[None]]
templates_drift.shape, xyza_orig.shape, xyza_drift.shape

# %%
rg = np.random.default_rng(0)
def getst(tmin, tmax, size, ref=10):
    while True:
        st = rg.choice(
            np.arange(tmin, tmax), size=size, replace=False
        )
        st.sort()
        if np.diff(st).min() >= ref:
            return st


# %%
sta = getst(0, reclen - spikelen, size=50)
sta = np.c_[sta, np.zeros_like(sta)]
stb = getst(0, reclen - spikelen, size=50)
stb = np.c_[stb, 1 + np.zeros_like(stb)]
stc = getst(0, reclen - spikelen, size=50)
stc = np.c_[stc, 2 + np.zeros_like(stc)]
st0 = np.r_[sta, stb, stc]
st1 = np.r_[st0, st0 + [reclen, 0]]
st = np.r_[st0, st0 + [reclen, 3]]
st1.shape, st.shape

# %%
xyza_abs = xyza_drift[st[:, 1]]
xyza_reg = xyza_drift[st1[:, 1]]

# %%
time_range = np.arange(-trough, spikelen - trough)
time_ix = st[:, 0, None] + time_range[None, :]
chan_ix = write_chan_ix[st[:, 1]]
raw = rg.normal(size=(reclen_drift, nchan), scale=0.1).astype(np.float32)
np.add.at(
    raw,
    (time_ix[:, :, None], chan_ix[:, None, :]),
    templates_drift[st[:, 1]],
)

# %%
raw.tofile("/tmp/testbbb.bin")

# %%

# %%
rts = snr_templates.get_raw_templates(st, geom, "/tmp/testbbb.bin")

# %%
rts.ptp(1).max(1)

# %%
superres = drifty_deconv.superres_deconv(
    "/tmp/testbbb.bin",
    geom,
    xyza_abs[:, 2],
    drift_p_byfs,
    st1,
    bin_size_um=1,
    deconv_dir="/tmp/supertest",
    pfs=5000,
    reference_displacement=0,
    t_start=0,
    t_end=None,
    n_jobs=1,
    trough_offset=trough,
    spike_length_samples=spikelen,
    max_upsample=8,
    refractory_period_frames=10,
    denoise_templates=False,
)

# %%
dh5, extra = drifty_deconv.extract_superres_shifted_deconv(
    superres,
    geom=geom,
    save_cleaned_waveforms=True,
    do_reassignment_tpca=False,
    save_cleaned_tpca_projs=False,
    localize=False,
)

# %%
with h5py.File(dh5) as hh:
    cluster_viz_index.reassignments_viz(
        superres['deconv_spike_train'],
        hh["reassigned_unit_labels"][:],
        "/tmp/testbbb.bin",
        "/Users/charlie/data/testreasdriftydc",
        geom,
        radius=200,
        z_extension=1.5,
        trough_offset=trough,
        spike_length_samples=spikelen,
        templates=templates_drift[:3],
    )
