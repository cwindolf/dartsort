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
import h5py

# %%
import matplotlib.pyplot as plt

# %%
plt.rc("figure", dpi=200)

# %%
from spike_psvae import deconvolve, extract_deconv, waveform_utils

# %%
spikelen = 21
reclen = 50000
trough = 0
nchan = 10
refractory = 11

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
def getst(tmin, tmax, size, ref=refractory):
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
scales = np.ones(len(st))
scales = rg.normal(size=len(st), loc=1, scale=0.1)
scales = np.clip(scales, 0.8, 1.3)

# %%
time_range = np.arange(-trough, spikelen - trough)
time_ix = st[:, 0, None] + time_range[None, :]
chan_ix = write_chan_ix[st[:, 1]]
raw = rg.normal(size=(reclen, nchan), scale=0.1).astype(np.float32)
np.add.at(
    raw,
    (time_ix[:, :, None], chan_ix[:, None, :]),
    scales[:, None, None] * templates[st[:, 1]],
)

# %%
raw.tofile("/tmp/testaaa.bin")

# %%
raw_waveforms = raw[time_ix[:, :, None], chan_ix[:, None, :]]

# %%
np.abs(raw_waveforms).max()

# %%
np.abs(raw).max()

# %%
raw_mcts = waveform_utils.get_maxchan_traces(raw_waveforms, waveform_utils.full_channel_index(geom.shape[0]), templates.ptp(1).argmax(1)[st[:, 1]])

# %%
true_wfs = scales[:, None, None] * templates[st[:, 1]]
true_mcts = waveform_utils.get_maxchan_traces(true_wfs, waveform_utils.full_channel_index(geom.shape[0]), templates.ptp(1).argmax(1)[st[:, 1]])

# %%
np.abs(templates.max())

# %%
res_noscale = deconvolve.deconv(
    "/tmp/testaaa.bin", "/tmp/testdeconv", templates, max_upsample=1
)

# %%
plt.figure(figsize=(2, 1))
for t, c in zip(res_noscale["templates_up"], "rgb"):
    plt.plot(t, color=c, alpha=0.5)

# %%
mcts = waveform_utils.get_maxchan_traces(templates, waveform_utils.full_channel_index(geom.shape[0]), templates.ptp(1).argmax(1))
plt.figure(figsize=(2, 1))
for mct, c in zip(mcts, "rgb"):
    plt.plot(mct, alpha=0.5, c=c)


# %%
def check_deconv_res(dec_temps, dec_st, dec_scales):
    plt.figure(figsize=(3,2))
    plt.hist(dec_scales, bins=64);
    plt.show()
    plt.close("all")
    
    exh5, residpath = extract_deconv.extract_deconv(
        # res_scale["templates_up"],
        # res_scale['deconv_spike_train_upsampled'],
        dec_temps,
        dec_st,
        "/tmp/testdeconv",
        "/tmp/testaaa.bin",
        save_subtracted_waveforms=True,
        save_cleaned_waveforms=True,
        scalings=dec_scales,
        geom=geom,
        do_reassignment=False,
        sampling_rate=150,
        do_denoised_tpca=False,
        nn_denoise=False,
        trough_offset=0,
        save_residual=True,
    )
    with h5py.File(exh5) as h5:
        print(h5.keys())
        sub_wfs = h5["subtracted_waveforms"][:]
        cleaned_wfs = h5["cleaned_waveforms"][:]
        ci = h5["channel_index"][:]
        mcs = h5["spike_index"][:, 1]

        print(f"{ci=}")

        sub_mcts = waveform_utils.get_maxchan_traces(sub_wfs, ci, mcs)
        cleaned_mcts = waveform_utils.get_maxchan_traces(cleaned_wfs, ci, mcs)
        # cleaned_mcts = cleaned_wfs[np.arange(len(cleaned_wfs)), :, cleaned_wfs.ptp(1).argmax(1)]
        print(f"{(cleaned_wfs == 0).all(axis=(1, 2)).sum()=}")
        print(f"{cleaned_mcts.shape=} {true_mcts.shape=} {cleaned_wfs.shape=} {true_wfs.shape=}")
        print(f"{(cleaned_mcts == 0).all(axis=(1,)).sum()=}")
        print(f"{(np.abs(cleaned_wfs) < 3).all(axis=(1,2)).sum()=}")
        print(f"{(np.abs(cleaned_mcts) < 3).all(axis=(1,)).sum()=}")
    

    plt.figure(figsize=(3,2))
    for rwf, swf, cwf, twf in zip(raw_mcts, sub_mcts, cleaned_mcts, true_mcts):
        plt.plot(rwf, color="k", alpha=0.1)
        plt.plot(twf, color="b", alpha=0.1)
    for rwf, swf, cwf, twf in zip(raw_mcts, sub_mcts, cleaned_mcts, true_mcts):
        plt.plot(swf, color="r", ls=":", alpha=0.1)
        # plt.plot(cwf, color="g")
    plt.show()
    plt.close("all")
    plt.figure(figsize=(3,2))
    for rwf, swf, cwf, twf in zip(raw_mcts, sub_mcts, cleaned_mcts, true_mcts):
        plt.plot(rwf, color="k", alpha=0.1)
        plt.plot(twf, color="b", alpha=0.1)
        # plt.plot(swf, color="r")
    for rwf, swf, cwf, twf in zip(raw_mcts, sub_mcts, cleaned_mcts, true_mcts):
        plt.plot(cwf, color="g", ls=":", alpha=0.1)
    plt.show()
    plt.close("all")
    
    # resid_mcts = cleaned_mcts - sub_mcts
    # resid_norms = np.linalg.norm(resid_mcts, axis=1)
    resid = np.fromfile(residpath, dtype=raw.dtype, count=raw.size).reshape(raw.shape)
    resid_waveforms = resid[time_ix[:, :, None], chan_ix[:, None, :]]
    resid_mcts = waveform_utils.get_maxchan_traces(resid_waveforms, waveform_utils.full_channel_index(geom.shape[0]), templates.ptp(1).argmax(1)[st[:, 1]])
    resid_norms = np.linalg.norm(resid_mcts, axis=1)
    plt.figure(figsize=(3,2))
    plt.hist(resid_norms, bins=32);
    plt.show()
    plt.close("all")
    
    return resid_norms

# %%
gt_resid_norms = check_deconv_res(templates, st, scales)

# %%
gt_resid_norms.min()

# %%
for lambd in (0, 0.0001, 0.01, 0.1, 10):
    print(f"{lambd=}")
    res_scale = deconvolve.deconv(
        "/tmp/testaaa.bin",
        "/tmp/testdeconv",
        templates,
        lambd=lambd,
        allowed_scale=1,
        max_upsample=1,
        trough_offset=0,
    )
    
    rns = check_deconv_res(res_scale["templates_up"], res_scale['deconv_spike_train_upsampled'], res_scale["deconv_scalings"])
    if lambd == 0:
        unscaled_rns = rns.copy()
    
    plt.figure(figsize=(3, 3))
    mng = gt_resid_norms.min()
    mxg = gt_resid_norms.max()
    mnt = rns.min()
    mxt = rns.max()
    plt.plot([min(mng, mnt), max(mxg, mxt)], [min(mng, mnt), max(mxg, mxt)], color="k", lw=1)
    plt.scatter(gt_resid_norms, rns, s=5, lw=0, color="orange", zorder=11)
    plt.xlabel("gt")
    plt.ylabel("this")
    plt.show()
    plt.close("all")
    
    plt.figure(figsize=(3, 2))
    plt.hist(rns - gt_resid_norms, bins=32)
    plt.xlabel("this resid norm - gt")
    plt.show()
    plt.close("all")
    
    plt.figure(figsize=(3, 2))
    plt.hist(rns - unscaled_rns, bins=32)
    plt.xlabel("this resid norm - unscaled")
    plt.show()
    plt.close("all")

# %%
