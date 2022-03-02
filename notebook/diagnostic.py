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
import h5py
import numpy as np
import matplotlib.pyplot as plt

# %%
from spike_psvae import denoise, vis_utils, waveform_utils, localization
from npx import reg

# %%
plt.rc("figure", dpi=200)
rg = lambda: np.random.default_rng(0)

# %%
# subh5 = h5py.File("/mnt/3TB/charlie/subtracted_datasets/churchlandlab_CSHL049_p7_t_2000_2010.h5", "r")
subh5 = h5py.File("/mnt/3TB/charlie/subtracted_datasets/subtraction_standardized_t_250_300.h5", "r")
firstchans = subh5["first_channels"][:]
spike_index = subh5["spike_index"][:]
maxchans = spike_index[:, 1]
geom = subh5["geom"][:]
wfs = subh5["subtracted_waveforms"]
cwfs = subh5["cleaned_waveforms"]
residual = np.memmap("/mnt/3TB/charlie/subtracted_datasets/residual_standardized_t_250_300.bin", dtype=np.float32)
residual = residual.reshape(-1, geom.shape[0])
feat_chans = cwfs.shape[2]
cfirstchans = firstchans
cmaxchans = maxchans


# feat_chans = 20
# if "cleaned_first_channels" in subh5:
#     cfirstchans = subh5["first_channels"][:]
#     cmaxchans = subh5["spike_index"][:, 1]
#     feat_chans = cwfs.shape[-1]
# else:
#     cwfs, cfirstchans, cmaxchans, chans_down = waveform_utils.relativize_waveforms(
#         cwfs,
#         firstchans,
#         None,
#         geom,
#         feat_chans=feat_chans,
#     )

# relativize time
spike_index[:, 0] -= subh5["start_sample"][()]

# %%
(spike_index[1:, 0] >= spike_index[:-1, 0]).all()

# %%
residual.shape

# %%
subh5["end_sample"][()] - subh5["start_sample"][()]

# %%
raw = np.memmap(
    # "/mnt/3TB/charlie/.one/openalyx.internationalbrainlab.org/churchlandlab/Subjects/CSHL049/2020-01-08/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec.ap.normalized.bin",
    "/mnt/3TB/charlie/NP2/standardized.bin",
    dtype=np.float32,
    mode="r",
)
raw = raw.reshape(-1, 384)
raw = raw[subh5["start_sample"][()]:subh5["end_sample"][()]]

# %%
assert (cfirstchans >= firstchans).all()
assert (maxchans >= firstchans).all()
assert (cmaxchans >= cfirstchans).all()
(cmaxchans - maxchans).min(), (cmaxchans - maxchans).max()

# %%
(cfirstchans - firstchans).min(), (cfirstchans - firstchans).max()

# %%
assert (cmaxchans >= firstchans).all()

# %%
N = len(spike_index)
N

# %%
show = rg().choice(N, replace=False, size=16)
show.sort()

# %%

# %%
print(firstchans[show])

# %%
cleaned = denoise.cleaned_waveforms(
    subh5["subtracted_waveforms"],
    subh5["spike_index"][:],
    subh5["first_channels"][:],
    residual,
    s_start=subh5["start_sample"][()]
)
stdwfs, firstchans_std, maxchans_std, chans_down = waveform_utils.relativize_waveforms(
    cleaned,
    subh5["first_channels"][:],
    None,
    subh5["geom"][:],
    feat_chans=feat_chans,
)

# %%
np.abs(cwfs - stdwfs).max()

# %%
cptp = cwfs[:].ptp(1).ptp(1)
sptp = stdwfs.ptp(1).ptp(1)

# %%
cptp.max(), sptp.max()

# %%
np.abs(cptp - sptp).max()

# %%
plt.hist(cptp - sptp, bins=20)
plt.show()

# %%
plt.hist(cptp, bins=100);

# %%
plt.hist(sptp, bins=100);

# %%
fig, axes = plt.subplots(4, 4)
vis_utils.plot_ptp(wfs[show].ptp(1), axes, "", "k", "abcdefghijklmnop")
crelptps = []
srelptps = []

for ix in show:
    fcrel = cfirstchans[ix] - firstchans[ix]
    print(fcrel, cfirstchans[ix], firstchans[ix])
    crelptps.append(np.pad(cwfs[ix].ptp(0), (fcrel, (40 - feat_chans) - fcrel)))
    srelptps.append(np.pad(stdwfs[ix].ptp(0), (fcrel, (40 - feat_chans) - fcrel)))
vis_utils.plot_ptp(crelptps, axes, "", "purple", "abcdefghijklmnop")
vis_utils.plot_ptp(srelptps, axes, "", "green", "abcdefghijklmnop")

# %%
wfs.shape, cwfs.shape, stdwfs.shape

# %%
crelptps = []
for ix in show:
    print(ix)
    fcrel = cfirstchans[ix] - firstchans[ix]
    
    fig = plt.figure(figsize=(20, 2.5))
    cwf = np.pad(cwfs[ix, :82], [(0, 0), (fcrel, (22 - fcrel))])
    swf = np.pad(stdwfs[ix, :82], [(0, 0), (fcrel, (22 - fcrel))])
    plt.plot(wfs[ix, :82].T.ravel())
    plt.plot(cwf.T.ravel())
    plt.plot(swf.T.ravel())
    for j in range(39):
        plt.axvline(82 + 82*j, color = 'black')
    
    plt.show()

# %%
mosaic = """\
adbcx
yyyyy
zzzzz
"""
def subfig(ix):
    t, mc = spike_index[ix]
    mcr = cmaxchans[ix] - cfirstchans[ix]
    mcrix = mcr - mcr % 2
    print(t, mc)
    fc = firstchans[ix]
    fcr = cfirstchans[ix] - fc
    wf = wfs[ix]
    T, C = wf.shape
    dn2 = cwfs[ix]
    C_ = dn2.shape[1]
    print(T, C, C_)
    raw_ix = raw[t - 42 : t + 79, fc : fc + C]
    res_ix = residual[t - 42 : t + 79, fc : fc + C]
    print(raw_ix.shape, res_ix.shape)
    print(raw_ix.min(), raw_ix.max(), res_ix.min(), res_ix.max())
    wf = wf[:, fcr:fcr + C_]
    raw_ix = raw_ix[:, fcr:fcr + C_]
    res_ix = res_ix[:, fcr:fcr + C_]
    print(raw_ix.shape, res_ix.shape)
    
    print((res_ix[:, mcr] + wf[:, mcr]).argmin())
    # dn2 = dnfull(res_ix + wf)
    
    vmin = min([v.min() for v in (raw_ix, res_ix, dn2, wf)])
    vmax = max([v.max() for v in (raw_ix, res_ix, dn2, wf)])
    
    
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(6, 5), gridspec_kw=dict(height_ratios=[2, 0.5, 0.5]))
    for k in "dbc":
        axes[k].set_yticks([])
    axes["a"].imshow(raw_ix[20:-20], cmap="RdBu_r", vmin=min(-vmax, vmin), vmax=max(-vmin, vmax))
    axes["d"].imshow(wf[20:-20], cmap="RdBu_r", vmin=min(-vmax, vmin), vmax=max(-vmin, vmax))
    axes["b"].imshow(res_ix[20:-20], cmap="RdBu_r", vmin=min(-vmax, vmin), vmax=max(-vmin, vmax))
    im = axes["c"].imshow(dn2[20:-20], cmap="RdBu_r", vmin=min(-vmax, vmin), vmax=max(-vmin, vmax))
    cbar = plt.colorbar(im, ax=[axes[k] for k in "abcd"], shrink=0.5)
    cpos = cbar.ax.get_position()
    cpos.x0 = cpos.x0 - 0.02
    cbar.ax.set_position(cpos)
    axes["a"].set_title("raw", fontsize=8)
    axes["b"].set_title("residual", fontsize=8)
    axes["c"].set_title("denoised", fontsize=8)
    axes["d"].set_title("subtracted", fontsize=8)
    for k in "abcd":
        axes[k].set_xticks([0, C_])
    axes["a"].axhline(22, lw=1, c="k")
        
    vis_utils.plot_single_ptp(raw_ix.ptp(0), axes["x"], "raw", "k", "")
    vis_utils.plot_single_ptp(res_ix.ptp(0), axes["x"], "residual", "silver", "")
    vis_utils.plot_single_ptp(wf.ptp(0), axes["x"], "subtracted", "g", "")
    vis_utils.plot_single_ptp((wf + res_ix).ptp(0), axes["x"], "cleaned", "b", "")
    vis_utils.plot_single_ptp(dn2.ptp(0), axes["x"], "denoised", "r", "")
    axes["x"].set_ylabel("ptp", labelpad=0)
    axes["x"].set_xticks([0, C_//2])
    axes["x"].set_box_aspect(1)
    pos = axes["x"].get_position()
    print(pos)
    pos.y0 = axes["c"].get_position().y0 - 0.075
    print(pos)
    axes["x"].set_position(pos)
    axes["x"].legend(loc="upper center", bbox_to_anchor=(0.5, 1.9), fancybox=False, frameon=False)
    
    cshow = 6
    axes["y"].plot(raw_ix[:82, mcrix - cshow : mcrix + cshow].T.flatten(), "k", lw=0.5)
    axes["y"].plot(wf[:82, mcrix - cshow : mcrix + cshow].T.flatten(), "g", lw=0.5)
    axes["y"].plot(res_ix[:82, mcrix - cshow : mcrix + cshow].T.flatten(), "silver", lw=0.5)
    axes["y"].set_xlim([0, dn2[:82, mcrix - cshow : mcrix + cshow].size])
    for j in range(2 * cshow):
        axes["y"].axvline(82 + 82*j, color="k", lw=0.5)
    axes["y"].set_xticks([])
    
    # axes["z"].plot(raw_ix[:82, mcrix - cshow : mcrix + cshow].T.flatten(), "k", lw=0.5)
    axes["z"].plot((res_ix + wf)[:82, mcrix - cshow : mcrix + cshow].T.flatten(), "b", lw=0.5)
    axes["z"].plot(dn2[:82, mcrix - cshow : mcrix + cshow].T.flatten(), "r", lw=0.5)
    # axes["z"].plot(bcwfs[ix, :82, mcrix - cshow : mcrix + cshow].T.flatten(), "orange", lw=0.5)
    axes["z"].set_xlim([0, dn2[:82, mcrix - cshow : mcrix + cshow].size])
    for j in range(2 * cshow):
        axes["z"].axvline(82 + 82*j, color="k", lw=0.5)
    axes["z"].set_xticks([])
    return fig

# %%
for ix in show:
    print(ix)
    subfig(ix)
    plt.show()

# %%

# %%
# locs = np.load("/mnt/3TB/charlie/ibl_feats/churchlandlab_CSHL049_p7_t_1500_2500_locs.npz")
locs = np.load("/mnt/3TB/charlie/subtracted_datasets/locs_standardized_t_250_300.h5.npz")

list(locs.keys())

# %%
plt.figure(figsize=(10, 8))
maxptps = locs["maxptp"]
# which = maxptps > 5
which = slice(None)
maxptps = maxptps[which]
nmaxptps = 0.25 + 0.74 * (maxptps - maxptps.min()) / (maxptps.max() - maxptps.min())
# print(which.sum())
x = locs["locs"][which, 0]
y = locs["locs"][which, 1]
a = locs["locs"][which, 4]
za = locs["locs"][which, 3]
zr = locs["z_reg"][which]
t = locs["t"][which]

z_rigid_reg, p_rigid = reg.register_rigid(
    maxptps,
    za + 0,
    t,
    robust_sigma=0,
    disp=400,
    denoise_sigma=0.1,
    destripe=False,
)
z_reg, dispmap = reg.register_nonrigid(
    maxptps,
    z_rigid_reg,
    t,
    robust_sigma=1,
    rigid_disp=200,
    disp=100,
    denoise_sigma=0.1,
    destripe=False,
    n_windows=[5, 30],#, 60],
    n_iter=1,
    widthmul=0.5,
)
D = p_rigid[None, :] + dispmap
D -= D.mean()
plt.figure(figsize=(6, 5))
plt.imshow(D, aspect=0.2, cmap=plt.cm.bwr)
plt.colorbar()

# moves = (zr - za) - (zr - za).mean()
# moves = np.abs(moves)
# moves /= moves.max()
# moves = 0.25 + 0.74 * moves
# plt.scatter(t, za, s=0.1, alpha=moves, c=(zr - za) - (zr - za).mean(), cmap=plt.cm.seismic)
# plt.colorbar()
# plt.show()

# %%
v = max(D.max(), -D.min())
plt.figure(figsize=(6, 5))
plt.imshow(D, aspect=0.2, cmap=plt.cm.bwr, vmin=-v, vmax=v)
plt.colorbar()

# %%
plt.hist((zr - za) - (zr - za).mean(), bins=128);


# %%

# %%
def plotlocs(x, y, z, alpha, maxptps, geom, which=slice(None)):
    maxptps = maxptps[which]
    nmaxptps = 0.25 + 0.74 * (maxptps - maxptps.min()) / (maxptps.max() - maxptps.min())
    cmaxptps = np.clip(maxptps, 3, 13)

    x = x[which]
    y = y[which]
    alpha = alpha[which]
    z = z[which]
    cm = plt.cm.viridis
    fig, (aa, ab, ac) = plt.subplots(1, 3, sharey=True, figsize=(8, 8))
    aa.scatter(x, z, s=0.1, alpha=nmaxptps, c=cmaxptps, cmap=cm)
    aa.scatter(geom[:, 0], geom[:, 1], color="orange", s=1)
    logy = np.log(y)
    ab.scatter(logy, z, s=0.1, alpha=nmaxptps, c=cmaxptps, cmap=cm)
    loga = np.log(alpha)
    ac.scatter(loga, z, s=0.1, alpha=nmaxptps, c=cmaxptps, cmap=cm)
    aa.set_ylabel("z")
    aa.set_xlabel("x")
    ab.set_xlabel("$\\log y$")
    ac.set_xlabel("$\\log \\alpha$")
    aa.set_xlim(np.percentile(x, [0, 100]))
    ab.set_xlim(np.percentile(logy, [0, 100]))
    ac.set_xlim(np.percentile(loga, [0, 100]))
    aa.set_ylim([0 - 10, geom[:, 1].max() + 10])
    plt.show()


# %%
cm = plt.cm.viridis

t = locs["t"]
which = slice(None)

# print(which.sum())
print(t.min(), t.max())
x = locs["locs"][:, 0]
y = locs["locs"][:, 1]
a = locs["locs"][:, 4]
za = locs["z_reg"]
maxptps = locs["maxptp"]
plt.figure()
plt.hist(maxptps, bins=100)
plt.show()

plotlocs(x, y, z_reg, a, maxptps, geom)

# %%
sx, sy, szr, sza, sa = localization.localize_waveforms(
stdwfs,
geom,
firstchans_std,
maxchans_std,
n_workers=1,
)

# %%
szrr, _ = reg.register_rigid(
    stdwfs.ptp(1).max(1).astype(float),
    sza,
    t,
)
sz_reg, _ = reg.register_nonrigid(
    stdwfs.ptp(1).max(1).astype(float),
    szrr,
    t,
)

# %%

# %%
plotlocs(sx, sy, sz_reg, sa, stdwfs.ptp(1).max(1), geom)

# %%

# %%
np.abs(firstchans_std - cfirstchans).max()

# %%
bx, by, bzr, bza, ba = localization.localize_waveforms_batched(
    cwfs,
    geom,
    cfirstchans,
    cmaxchans,
    n_workers=10,
)

# %% tags=[]
bzrr, _ = reg.register_rigid(
    cwfs[:].ptp(1).max(1).astype(float),
    bza,
    t,
)
bz_reg, _ = reg.register_nonrigid(
    cwfs[:].ptp(1).max(1).astype(float),
    bzrr,
    t,
)

# %%
plotlocs(bx, by, bz_reg, ba, cwfs[:].ptp(1).max(1), geom)

# %%
# 
