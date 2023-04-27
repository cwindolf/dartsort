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
# %config InlineBackend.figure_format = 'retina'

# %%
import numpy as np
import pickle
import h5py
from pathlib import Path
from spike_psvae import motion_utils as mu, newton_motion_est as newt

# %%
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)

# %%
import time


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
import scipy.linalg as la
from scipy.linalg import solve

# %%
from scipy import sparse

# %%
from one.api import ONE
import one.alf.io as alfio
from brainbox.io.one import SpikeSortingLoader

# %%
import matplotlib.pyplot as plt

# %%
one = ONE()

# %%
locdir = Path("/mnt/sdceph/users/cwindolf/re_ds_localizations")

# %%

# %%
# nice decent snr test case with various motions
pid0 = "8abf098f-d4f6-4957-9c0a-f53685db74cc"

# pretty stable with SNR fails on edges
pid1 = "7d999a68-0215-4e45-8e6c-879c6ca2b771"

# super stable one that we should be getting!!
pid2 = "45e7731f-4a43-45d5-9029-c080150bc596"

# one where KS kinda messes up
pid3 = "9e44ddb5-7c7c-48f1-954a-6cec2ad26088"

pids = [pid0, pid1, pid2, pid3]

# %%
# several pids where we look worse
pids = """\
0b8ea3ec-e75b-41a1-9442-64f5fbc11a5a
1a60a6e1-da99-4d4e-a734-39b1d4544fad
02cc03e4-8015-4050-bb42-6c832091febb
7f3dddf8-637f-47bb-a7b7-e303277b2107
8b1b285d-c176-421b-b564-3b2c6404fb1e
41a3b948-13f4-4be7-90b9-150705d39005
75d5e438-b08c-47d3-8ac0-992ec63c9c1b
80f6ffdd-f692-450f-ab19-cd6d45bfd73e
82a42cdf-3140-427b-8ad0-0d504716c871
642a3373-6fee-4f2a-b8d6-38dd42ada763
8185f1e9-cfe0-4fd6-8d7e-446a8051c588
""".split()

# %%
feats = {}

for pid in pids:
    print(pid)
    subdir = locdir / f"pid{pid}"
    sub_h5 = subdir / "subtraction.h5"
    meta_pkl = subdir / "metadata.pkl"

    with h5py.File(sub_h5) as h5:
        z = h5["localizations"][:, 2]
        maxptp = h5["maxptps"][:]
        t_samples = h5["spike_index"][:, 0]
        fs = h5["fs"][()]
        t = t_samples / fs
        geom = h5["geom"][:]

    feats[pid] = dict(z=z, a=maxptp, t=t, geom=geom)

# %%
for pid, ff in feats.items():
    print(pid, ff.keys())

# %%
pyks_regs = {}

for pid in pids:
    ssl = SpikeSortingLoader(one=one, pid=pid)
    ssl.download_spike_sorting_object("drift")
    drift = alfio.load_object(ssl.files["drift"], wildcards=ssl.one.wildcards)
    drift_samples = ssl.samples2times(drift["times"], direction="reverse")

    sbe, tbe = mu.get_bins(feats[pid]["z"], feats[pid]["t"], 1, 1)
    sbc = (sbe[1:] + sbe[:-1]) / 2

    # code from pyks. get the centers of the bins that they used
    nblocks = (drift["um"].shape[1] + 1) // 2
    yl = np.floor(geom[:, 1].max() / nblocks).astype("int") - 1
    mins = np.linspace(0, geom[:, 1].max() - yl - 1, 2 * nblocks - 1)
    maxs = mins + yl
    ks_windows = np.zeros((len(mins), len(sbe) - 1))
    for j, (mn, mx) in enumerate(zip(mins, maxs)):
        ks_windows[j, (mn <= sbc) & (sbc <= mx)] = 1
    ks_windows /= ks_windows.sum(axis=1, keepdims=True)
    centers = (mins + maxs) / 2

    ksme = mu.NonrigidMotionEstimate(
        -drift["um"].T,
        time_bin_centers_s=drift_samples / fs,
        spatial_bin_centers_um=centers,
    )
    kextra = dict(window_centers=centers, windows=ks_windows)

    pyks_regs[pid] = dict(me=ksme, extra=kextra)

# %%
np.log1p(0)

# %%
me_yw22_ns_mc1_md100_dt1000, extra_yw2_ns_mc1_md100_dt1000 = newt.register(
    maxptp,
    z,
    t,
    weights_kw=dict(weights_threshold_low=0.2, weights_threshold_high=0.2, mincorr=0.1, max_dt_s=1000),
    precomputed_D_C_maxdisp=(D, C, maxdisp),
    thomas_kw=dict(lambda_s=0),
    # pbar=False,
)

# %%
me_yw22_ns_mc1_md100_dt1000 = {}
for pid in pids:
    ff = feats[pid]
    me, extra = newt.register(
        ff["a"], ff["z"], ff["t"],
        weights_kw=dict(weights_threshold_low=0.2, weights_threshold_high=0.2, mincorr=0.1, max_dt_s=1000),
        thomas_kw=dict(lambda_s=0),
    )
    me_yw22_ns_mc1_md100_dt1000[pid] = dict(me=me, extra=extra)

# %%
me_yw22_mc1_md50_dt1000_altraster = {}
for pid in pids:
    ff = feats[pid]
    me, extra = newt.register(
        ff["a"], ff["z"], ff["t"],
        raster_kw=dict(gaussian_smoothing_sigma_um=1, amp_scale_fn=np.log1p, avg_in_bin=True, post_transform=None),
        weights_kw=dict(weights_threshold_low=0.2, weights_threshold_high=0.2, mincorr=0.1, max_dt_s=1000),
        thomas_kw=dict(lambda_s=0, lambda_t=1),
        max_disp_um=50,
        device="cuda:1",
    )
    me_yw22_mc1_md50_dt1000_altraster[pid] = dict(me=me, extra=extra)

# %%
me_yw22_mc1_md50_dt1000_braster = {}
for pid in pids:
    ff = feats[pid]
    me, extra = newt.register(
        ff["a"], ff["z"], ff["t"],
        raster_kw=dict(gaussian_smoothing_sigma_um=1),
        weights_kw=dict(weights_threshold_low=0.2, weights_threshold_high=0.2, mincorr=0.1, max_dt_s=1000),
        thomas_kw=dict(lambda_s=0, lambda_t=1),
        max_disp_um=50,
        device="cuda:1",
    )
    me_yw22_mc1_md50_dt1000_braster[pid] = dict(me=me, extra=extra)

# %%
thomas_unthresh = {}
for pid in pids:
    ff = feats[pid]
    me, extra = newt.register(ff["a"], ff["z"], ff["t"], device="cuda:1", save_full=True)
    thomas_unthresh[pid] = dict(me=me, extra=extra)

# %%
thomas_unthresh_dt_mc = {}
for pid in pids:
    ff = feats[pid]
    me, extra = newt.register(
        ff["a"],
        ff["z"],
        ff["t"],
        thomas_kw=dict(mincorr=0.1, max_dt_s=1000),
        device="cuda:1",
    )
    thomas_unthresh_dt_mc[pid] = dict(me=me, extra=extra)

# %%

# %%
thomas_thresh_dt3_mc3 = {}
thomas_thresh_dt3_mc3_ind = {}

for pid in pids:
    ff = feats[pid]
    tdb = thomas_unthresh[pid]
    me, extra = newt.register(
        ff["a"],
        ff["z"],
        ff["t"],
        weights_kw=dict(mincorr=0.3, weights_threshold_low=0.3, max_dt_s=1000),
        device="cuda:1",
        _DCmd_precomputed=(
            tdb["extra"]["D"],
            tdb["extra"]["C"],
            tdb["extra"]["max_disp_um"],
        ),
    )
    thomas_thresh_dt3_mc3[pid] = dict(me=me, extra=extra)
    me2 = mu.NonrigidMotionEstimate(
        np.array(extra["Pind"]),
        time_bin_centers_s=me.time_bin_centers_s,
        spatial_bin_centers_um=me.spatial_bin_centers_um,
    )
    thomas_thresh_dt3_mc3_ind[pid] = dict(me=me2, extra=extra)

# %%
print("x")

# %%
toplot = [
    ("pyks", pyks_regs, "r"),
    ("thomas_unthresh", thomas_unthresh, "b"),
    # ("thomas_unthresh_dt_mc", thomas_unthresh_dt_mc, "orange"),
    # ("thomas_thresh_dt_unmc", thomas_thresh_dt_unmc, "purple"),
    # ("thomas_thresh_dt_mc", thomas_thresh_dt_mc, "g"),
    # ("thomas_thresh2_dt_mc", thomas_thresh2_dt_mc, "navy"),
    # ("thomas_thresh_dt_mc2", thomas_thresh_dt_mc2, "yellow"),
    # ("thomas_thresh2_dt_mc2", thomas_thresh2_dt_mc2, "k"),
    ("thomas_thresh_dt_mc3", thomas_thresh_dt_mc3, "purple"),
    ("mc3_ind", thomas_thresh_dt_mc3_ind, "k"),
    ("thomas_thresh_dt2_mc2", thomas_thresh_dt2_mc2, "g"),
    ("mc2_ind", thomas_thresh_dt2_mc2_ind, "gray"),
    ("thomas_thresh_dt3_mc3", thomas_thresh_dt3_mc3, "orange"),
    ("mc3_ind", thomas_thresh_dt3_mc3_ind, "w"),
    # ("thomas_mc1", thomas_mc1, "g"),
    # ("thomas_dt_thresh", thomas_dt_thresh, "b")
    # ("thomas_mc1_thresh", thomas_mc1_thresh, "orange"),
    # ("thomas_mc1_thresh2", thomas_mc1_thresh2, "g"),
    # ("thomas_mc1_thresh3", thomas_mc1_thresh3, "b"),
    # ("thomas_mc1_thresh4", thomas_mc1_thresh4, "purple"),
    # ("thomas_uncent", thomas_uncent, "g"),
]
print("quux", flush=True)
for pid in pids:
    print(pid)
    

    fig, aa = plt.subplots(figsize=(20, 20))
    r, dd, tt = mu.fast_raster(feats[pid]["a"], feats[pid]["z"], feats[pid]["t"])
    aa.imshow(r, aspect="auto", vmax=15, cmap=plt.cm.binary)

    weights_orig = thomas_unthresh[pid]["extra"]["weights_orig"]
    locs = thomas_unthresh[pid]["me"].spatial_bin_centers_um

    overlay = np.zeros_like(r)
    dbc = (dd[1:] + dd[:-1]) / 2
    for i, loc in enumerate(locs):
        which = np.argmin(np.abs(dbc[:, None] - locs[None, :]), axis=1) == i
        overlay[which] = weights_orig[i]
    print("o", flush=True)

    imo = aa.imshow(overlay, aspect="auto", cmap=plt.cm.rainbow, alpha=0.5)
    plt.colorbar(imo, ax=aa, shrink=0.5, label="Unthresholded weight")

    names = []
    handles = []
    offset = -40 * len(toplot) / 2
    for name, meme, color in toplot:
        if "ind" not in name:
            continue
        print(name)
        me = meme[pid]["me"]
        for pos in me.spatial_bin_centers_um:
            (ls,) = aa.plot(
                tt[:-1], pos + offset + me.disp_at_s(tt[:-1], depth_um=pos), color=color
            )
        offset += 40
        names.append(name)
        handles.append(ls)

    plt.legend(handles, names, ncol=5, loc="lower left")

    plt.show()
    plt.close(fig)

# %%
1

# %%
for pid in pids:
    fig, aa = plt.subplots(figsize=(10, 10))
    r, dd, tt = mu.fast_raster(feats[pid]["a"], feats[pid]["z"], feats[pid]["t"])
    aa.imshow(r, aspect="auto", vmax=15, cmap=plt.cm.binary)

    names = []
    handles = []
    offset = 0

    weights_orig = thomas_unthresh[pid]["extra"]["weights_orig"]
    locs = thomas_unthresh[pid]["me"].spatial_bin_centers_um
    print(f"{weights_orig.shape=} {weights_orig.min()=} {weights_orig.max()=}")

    overlay = np.zeros_like(r)
    dbc = (dd[1:] + dd[:-1]) / 2
    for i, loc in enumerate(locs):
        which = np.argmin(np.abs(dbc[:, None] - locs[None, :]), axis=1) == i
        print(f"{loc=} {which.mean()=}")
        overlay[which] = weights_orig[i]

    me = thomas_dt_thresh[pid]["me"]
    for pos in me.spatial_bin_centers_um:
        (ls,) = aa.plot(
            tt[:-1], pos + offset + me.disp_at_s(tt[:-1], depth_um=pos), color="k"
        )

    imo = aa.imshow(overlay, cmap=plt.cm.rainbow, alpha=0.5)
    plt.colorbar(imo, ax=aa, shrink=0.5)

    plt.show()
    plt.close(fig)

# %%

# %%
toplot = [
    ("pyks", pyks_regs, "r"),
    # ("thomas_unthresh", thomas_unthresh, "b"),
    ("thomas_mc1", thomas_mc1, "g"),
    ("thomas_dt_thresh", thomas_dt_thresh, "b")
    # ("thomas_mc1_thresh", thomas_mc1_thresh, "orange"),
    # ("thomas_mc1_thresh2", thomas_mc1_thresh2, "g"),
    # ("thomas_mc1_thresh3", thomas_mc1_thresh3, "b"),
    # ("thomas_mc1_thresh4", thomas_mc1_thresh4, "purple"),
    # ("thomas_uncent", thomas_uncent, "g"),
]

for pid in pids:
    fig, aa = plt.subplots(figsize=(10, 10))
    r, dd, tt = mu.fast_raster(feats[pid]["a"], feats[pid]["z"], feats[pid]["t"])
    aa.imshow(r, aspect="auto", vmax=15, cmap=plt.cm.binary)

    weights_orig = thomas_unthresh[pid]["extra"]["weights_orig"]
    locs = thomas_unthresh[pid]["me"].spatial_bin_centers_um
    print(f"{weights_orig.shape=} {weights_orig.min()=} {weights_orig.max()=}")

    overlay = np.zeros_like(r)
    dbc = (dd[1:] + dd[:-1]) / 2
    for i, loc in enumerate(locs):
        which = np.argmin(np.abs(dbc[:, None] - locs[None, :]), axis=1) == i
        print(f"{loc=} {which.mean()=}")
        overlay[which] = weights_orig[i]

    imo = aa.imshow(overlay, aspect="auto", cmap=plt.cm.rainbow, alpha=0.5)
    plt.colorbar(imo, ax=aa, shrink=0.5, label="Unthresholded weight")

    names = []
    handles = []
    offset = 0
    for name, meme, color in toplot:
        me = meme[pid]["me"]
        for pos in me.spatial_bin_centers_um:
            (ls,) = aa.plot(
                tt[:-1], pos + offset + me.disp_at_s(tt[:-1], depth_um=pos), color=color
            )
        offset += 40
        names.append(name)
        handles.append(ls)

    plt.legend(handles, names, ncol=len(names), loc="lower left")

    plt.show()
    plt.close(fig)

# %%
1

# %%
toplot = [
    ("pyks", pyks_regs, "r"),
    ("yw22+ns+mc1+md100+dt1000", me_yw22_ns_mc1_md100_dt1000, "b"),
    ("yw22_mc1_md50_dt1000_altraster", me_yw22_mc1_md50_dt1000_altraster, "g"),
    ("me_yw22_mc1_md50_dt1000_braster", me_yw22_mc1_md50_dt1000_braster, "orange"),
    # ("thomas_unthresh", thomas_unthresh, "b"),
    # ("thomas_mc1", thomas_mc1, "g"),
    # ("thomas_dt_thresh", thomas_dt_thresh, "b")
    # ("thomas_mc1_thresh", thomas_mc1_thresh, "orange"),
    # ("thomas_mc1_thresh2", thomas_mc1_thresh2, "g"),
    # ("thomas_mc1_thresh3", thomas_mc1_thresh3, "b"),
    # ("thomas_mc1_thresh4", thomas_mc1_thresh4, "purple"),
    # ("thomas_uncent", thomas_uncent, "g"),
]

for pid in pids:
    fig, aa = plt.subplots(figsize=(20, 20))
    r, dd, tt = mu.fast_raster(feats[pid]["a"], feats[pid]["z"], feats[pid]["t"])
    aa.imshow(r, aspect="auto", vmax=15, cmap=plt.cm.binary)

    names = []
    handles = []
    offset = 0
    for name, meme, color in toplot:
        me = meme[pid]["me"]
        for pos in me.spatial_bin_centers_um:
            (ls,) = aa.plot(
                tt[:-1], pos + offset + me.disp_at_s(tt[:-1], depth_um=pos), color=color, lw=1
            )
        offset += 40
        names.append(name)
        handles.append(ls)

    plt.legend(handles, names, ncol=len(names), loc="lower left")

    plt.show()
    plt.close(fig)

# %%
regs_compare = [
    ("pyks", pyks_regs, "r"),
    ("yw22+ns+mc1+md100+dt1000", me_yw22_ns_mc1_md100_dt1000, "b")
    # ("thomas_unthresh", thomas_unthresh, "b"),
    # ("thomas_mc1", thomas_mc1, "g"),
    # ("thomas_dt_thresh", thomas_dt_thresh, "b")
    # ("thomas_mc1_thresh", thomas_mc1_thresh, "orange"),
    # ("thomas_mc1_thresh2", thomas_mc1_thresh2, "g"),
    # ("thomas_mc1_thresh3", thomas_mc1_thresh3, "b"),
    # ("thomas_mc1_thresh4", thomas_mc1_thresh4, "purple"),
    # ("thomas_uncent", thomas_uncent, "g"),
]

for pid in pids:
    for regname, theregs, color in regs_compare:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f"{regname}  {pid}", color=color, fontsize=12)

        me = theregs[pid]["me"]
        z = feats[pid]["z"]
        a = feats[pid]["a"]
        t = feats[pid]["t"]
        zr = me.correct_s(t, z)

        r, dd, tt = mu.fast_raster(a, zr, t)

        ax.imshow(r, aspect="auto", vmax=15, cmap=plt.cm.cubehelix)

        plt.show()

        plt.close(fig)

# %%

# %%

# %%

# %%

# %%

# %%

# %%
# solve original problem to get weights
robust_sigma = 0
corr_threshold = 0.1
soft_weights = True
window_shape = "gaussian"
adaptive_mincorr_percentile = None
prior_lambda = 1
spatial_prior = False
normalized = True
disp = 100
win_step_um = 400
win_sigma_um = 400
max_dt = None
device = "cuda:0"
batch_size = 512
bin_um = 1
bin_s = 1
amp_scale_fn = None
post_transform = np.log1p
gaussian_smoothing_sigma_um = 3
upsample_to_histogram_bin = False
pbar = True
reference = 0

origregs = {}
for pid, ff in feats.items():
    tme, textra = ibme.register_nonrigid(
        ff["a"],
        ff["z"],
        ff["t"],
        ff["geom"],
        robust_sigma=robust_sigma,
        corr_threshold=corr_threshold,
        soft_weights=soft_weights,
        window_shape=window_shape,
        adaptive_mincorr_percentile=adaptive_mincorr_percentile,
        prior_lambda=prior_lambda,
        spatial_prior=spatial_prior,
        normalized=normalized,
        disp=disp,
        win_step_um=win_step_um,
        win_sigma_um=win_sigma_um,
        max_dt=max_dt,
        device=device,
        batch_size=batch_size,
        bin_um=bin_um,
        bin_s=bin_s,
        amp_scale_fn=amp_scale_fn,
        post_transform=post_transform,
        gaussian_smoothing_sigma_um=gaussian_smoothing_sigma_um,
        upsample_to_histogram_bin=upsample_to_histogram_bin,
        pbar=pbar,
        reference=reference,
        return_CD=True,
    )
    origregs[pid] = dict(me=tme, extra=textra)


# %%
def get_weights_in_window(window, db_unreg, db_reg, raster_reg):
    ilow, ihigh = np.flatnonzero(window)[[0, -1]]
    window_sliced = window[ilow : ihigh + 1]
    tbcu = 0.5 * (db_unreg[1:] + db_unreg[:-1])
    dlow, dhigh = tbcu[[ilow, ihigh]]
    tbcr = 0.5 * (db_reg[1:] + db_reg[:-1])
    rilow, rihigh = np.flatnonzero((tbcr >= dlow) & (tbcr <= dhigh))[[0, -1]]
    return window_sliced @ raster_reg[rilow : rihigh + 1]


def get_weights(windows, amps, depths, times, me=None, depths_reg=None):
    r, db, tb = ibme.fast_raster(
        amps, depths, times, avg_in_bin=False, amp_scale_fn=np.log1p
    )
    if depths_reg is None:
        depths_reg = me.correct_s(times, depths)
    rr, dbr, tbr = ibme.fast_raster(
        amps, depths_reg, times, avg_in_bin=False, amp_scale_fn=np.log1p
    )
    weights = np.array([get_weights_in_window(win, db, dbr, rr) for win in windows])
    return weights


# %%
for pid in pids:
    ff = feats[pid]
    reg = origregs[pid]
    me = reg["me"]
    extra = reg["extra"]
    a = ff["a"]
    z = ff["z"]
    t = ff["t"]
    z_reg = me.correct_s(t, z)
    ff["z_reg_orig"] = z_reg

    weights = get_weights(extra["windows"], a, z, t, depths_reg=z_reg)

    unif = np.ones_like(extra["windows"][0])
    unif /= unif.sum()

    threshold_low = (np.log1p(5) * extra["windows"] @ (500 * unif))[:, None]
    threshold_high = (np.log1p(5) * extra["windows"] @ (2000 * unif))[:, None]
    threshold_low, threshold_high

    weights_thresh = np.maximum(weights, threshold_low) - threshold_low
    weights_thresh[weights_thresh > threshold_high] = np.inf

    reg["weights"] = weights
    reg["weights_thresh"] = weights_thresh

# %%
cdsrs = {}

for pid in pids:
    extra = origregs[pid]["extra"]
    weights_thresh = origregs[pid]["weights_thresh"]
    Cs = np.array(extra["C"])
    Ds = np.array(extra["D"])
    Ss = []
    Rs = []
    for i in range(len(weights_thresh)):
        D = Ds[i]
        C = Cs[i]

        wt = weights_thresh[i]
        invwtt = 1 / wt[:, None] + 1 / wt[None, :]
        np.fill_diagonal(invwtt, 0)

        # abs fixes a strange thing that happens with some -0.0s that end up in here
        S = np.abs((C > 0.1) * C)
        S_ = 1 / (invwtt + 1 / S)
        Rs.append(S_)
        Ss.append(S)

    cdsrs[pid] = {}
    cdsrs[pid]["C"] = Cs
    cdsrs[pid]["D"] = Ds
    cdsrs[pid]["S"] = np.array(Ss)
    cdsrs[pid]["R"] = np.array(Rs)


# %%
def newt_solve(D, S, Sigma0inv, normalize=None):
    """D is TxT displacement, S is TxT subsampling or soft weights matrix"""

    if normalize == "sym":
        uu = 1 / np.sqrt((S + S.T).sum(1))
        S = np.einsum("i,j,ij->ij", uu, uu, S)

    # forget the factor of 2, we'll put it in later
    # HS = (S + S.T) - np.diag((S + S.T).sum(1))
    HS = S.copy()
    HS += S.T
    np.fill_diagonal(HS, np.diagonal(HS) - S.sum(1) - S.sum(0))
    # grad_at_0 = (S * D - S.T * D.T).sum(1)
    SD = S * D
    grad_at_0 = SD.sum(1) - SD.sum(0)
    # Next line would be (Sigma0inv ./ 2 .- HS) \ grad in matlab
    p = la.solve(Sigma0inv - HS, grad_at_0, assume_a="pos")
    # p = la.solve(Sigma0inv - 2 * HS, 2 * grad_at_0, assume_a="sym")
    # p = la.lstsq(Sigma0inv - 2 * HS, 2 * grad_at_0)[0]
    return p, HS


# %%

# %%
def laplacian(T):
    return (
        np.eye(T)
        - np.diag(0.5 * np.ones(T - 1), k=1)
        - np.diag(0.5 * np.ones(T - 1), k=-1)
    )


# %%
newt_tsmooth = {}
for pid, cdsr in cdsrs.items():
    ps = []
    for C, D, S, R in zip(*[cdsr[k] for k in "CDSR"]):
        Sigma0inv = laplacian(C.shape[0])
        p = newt_solve(D, np.square(R), Sigma0inv)[0]
        p -= p[0]
        ps.append(p)
    ps = np.array(ps)
    win_centers = origregs[pid]["extra"]["window_centers"]
    tbe = origregs[pid]["extra"]["time_bin_edges_s"]
    me = motion_est.NonrigidMotionEstimate(
        ps, spatial_bin_centers_um=win_centers, time_bin_edges_s=tbe
    )
    newt_tsmooth[pid] = dict(me=me)


# %%
def solve_spatial(wt, pt, lambd=1):
    k = wt.size
    assert wt.shape == pt.shape == (k,)
    finite = np.isfinite(wt)
    finite_inds = np.flatnonzero(finite)
    if not finite_inds.size:
        return pt

    coefts = np.diag(wt[finite_inds] + lambd)
    target = 2 * wt[finite_inds] * pt[finite_inds]
    for i, j in enumerate(finite_inds):
        if j > 0:
            if finite[j - 1]:
                coefts[i, i - 1] = coefts[i - 1, i] = -lambd / 2
            else:
                target[i] += lambd * pt[j - 1]
        if j < k - 1:
            if finite[j + 1]:
                coefts[i, i + 1] = coefts[i + 1, i] = -lambd / 2
            else:
                target[i] += lambd * pt[j + 1]
    try:
        r_finite = solve(coefts, target)
    except np.linalg.LinAlgError:
        print(
            f"{np.array2string(coefts, precision=2, max_line_width=100)} {target=} {coefts.shape=} {target.shape=}"
        )
        raise
    r = pt.copy()
    r[finite_inds] = r_finite
    return r


# %%
newt_tsmooth_sposthoc = {}
for pid in pids:
    weights_thresh = origregs[pid]["weights_thresh"]
    newt_me = newt_tsmooth[pid]["me"]
    new_ps = newt_me.displacement
    rs = np.zeros_like(new_ps)
    for tt in range(rs.shape[1]):
        rs[:, tt] = solve_spatial(weights_thresh[:, tt], new_ps[:, tt], lambd=1)
    me = motion_est.NonrigidMotionEstimate(
        rs,
        time_bin_edges_s=newt_me.time_bin_edges_s,
        spatial_bin_centers_um=newt_me.spatial_bin_centers_um,
    )
    newt_tsmooth_sposthoc[pid] = dict(me=me)


# %%
def multigrid_solve(Ds, Rs, lambda_t=1.0, lambda_s=1.0):
    B, T, T_ = Ds.shape
    assert T == T_
    assert Rs.shape == Ds.shape
    lap = laplacian(T)
    L_t = lambda_t * lap
    eye = np.eye(T)

    # build our sparse guy
    blocks = np.full((B, B), None, dtype=object)
    rhs = []

    # the diagonal and RHS
    with timer("build"):
        for b in range(B):
            R = Rs[b].astype("float64")

            # the likelihood terms
            block = -R.copy()
            block -= R.T
            np.fill_diagonal(block, np.diagonal(block) + R.sum(1) + R.sum(0))

            # the prior terms
            block += L_t
            block += lambda_s * eye

            blocks[b][b] = block

            # target
            SD = R * Ds[b].astype("float64")
            grad_at_0 = SD.sum(1) - SD.sum(0)
            rhs.append(grad_at_0)

        # off diagonal
        for b in range(B - 1):
            blocks[b + 1][b] = blocks[b][b + 1] = -(lambda_s / 2.0) * eye

        coefts = sparse.bmat(blocks, format="csr")

    with timer("construct"):
        ml = ruge_stuben_solver(coefts)
    print(ml)
    with timer("solve"):
        res = ml.solve(np.concatenate(rhs)).reshape(B, T)
    return res


# %%
ruge_stubens = {}
for pid, cdsr in cdsrs.items():
    P = multigrid_solve(cdsr["D"], np.square(cdsr["R"]))
    P -= P[:, 0, None]
    win_centers = origregs[pid]["extra"]["window_centers"]
    tbe = origregs[pid]["extra"]["time_bin_edges_s"]
    me = motion_est.NonrigidMotionEstimate(
        P, spatial_bin_centers_um=win_centers, time_bin_edges_s=tbe
    )
    ruge_stubens[pid] = dict(me=me)


# %%
def negH(R):
    R = R.astype("float64")
    # the likelihood tearms
    negHR = -R.copy()
    negHR -= R.T
    np.fill_diagonal(negHR, np.diagonal(negHR) + R.sum(1) + R.sum(0))
    return negHR


def rhs(R, D):
    SD = R * D.astype("float64")
    grad_at_0 = SD.sum(1) - SD.sum(0)
    return grad_at_0


def thomas_solve(Ds, Rs, lambda_t=1.0, lambda_s=1.0):
    B, T, T_ = Ds.shape
    assert T == T_
    assert Rs.shape == Ds.shape
    lap = laplacian(T)
    L_t = lambda_t * lap
    eye = np.eye(T)
    diag_prior_terms = L_t + lambda_s * eye
    offdiag_prior_terms = -(lambda_s / 2) * eye

    # initialize
    A1 = diag_prior_terms + negH(Rs[0])
    d1 = rhs(Rs[0], Ds[0])
    alpha_hats = [A1]
    res = solve(alpha_hats[0], np.c_[offdiag_prior_terms, d1])
    assert res.shape == (T, T + 1)
    gamma_hats = [res[:, :T]]
    ys = [res[:, T]]

    # forward pass
    with timer("forward pass"):
        for b in range(1, B):
            Ab = diag_prior_terms + negH(Rs[b])
            alpha_hat_b = Ab - offdiag_prior_terms @ gamma_hats[b - 1]
            res = solve(alpha_hat_b, np.c_[offdiag_prior_terms, rhs(Rs[b], Ds[b])])
            gamma_hats.append(res[:, :T])
            ys.append(res[:, T])

    # back substitution
    with timer("backward pass"):
        xs = [None] * B
        xs[-1] = ys[-1]
        for b in range(B - 2, -1, -1):
            xs[b] = ys[b] - gamma_hats[b] @ xs[b + 1]

    return np.concatenate(xs).reshape(B, T)


# %%
thomases = {}
for pid, cdsr in cdsrs.items():
    P = thomas_solve(cdsr["D"], np.square(cdsr["R"]))
    P -= P[:, 0, None]
    win_centers = origregs[pid]["extra"]["window_centers"]
    tbe = origregs[pid]["extra"]["time_bin_edges_s"]
    me = motion_est.NonrigidMotionEstimate(
        P, spatial_bin_centers_um=win_centers, time_bin_edges_s=tbe
    )
    thomases[pid] = dict(me=me)

# %%
toplot = [
    ("t only", newt_tsmooth, "b"),
    ("post hoc", newt_tsmooth_sposthoc, "r"),
    ("multigrid", ruge_stubens, "g"),
    ("thomas", thomases, "orange"),
]

for pid in pids:
    win_centers = origregs[pid]["extra"]["window_centers"]

    fig, aa = plt.subplots(figsize=(10, 10))
    r, dd, tt = ibme.fast_raster(feats[pid]["a"], feats[pid]["z"], feats[pid]["t"])
    aa.imshow(r, aspect="auto", vmax=15, cmap=plt.cm.binary)

    names = []
    handles = []
    offset = 0
    for name, meme, color in toplot:
        me = meme[pid]["me"]
        for pos in win_centers:
            (ls,) = aa.plot(
                tt[:-1], pos + offset + me.disp_at_s(tt[:-1], depth_um=pos), color=color
            )
        offset += 20
        names.append(name)
        handles.append(ls)

    plt.legend(handles, names, ncol=len(names), loc="lower left")

    plt.show()
    plt.close(fig)

# %%
pyks_regs = {}

for pid in pids:
    ssl = SpikeSortingLoader(one=one, pid=pid)
    ssl.download_spike_sorting_object("drift")
    drift = alfio.load_object(ssl.files["drift"], wildcards=ssl.one.wildcards)
    drift_samples = ssl.samples2times(drift["times"], direction="reverse")

    # code from pyks. get the centers of the bins that they used
    nblocks = (drift["um"].shape[1] + 1) // 2
    yl = np.floor(geom[:, 1].max() / nblocks).astype("int") - 1
    mins = np.linspace(0, geom[:, 1].max() - yl - 1, 2 * nblocks - 1)
    maxs = mins + yl
    ks_windows = np.zeros((len(mins), len(tt) - 1))
    for j, (mn, mx) in enumerate(zip(mins, maxs)):
        ks_windows[j, int(np.floor(mn)) : int(np.ceil(mx))] = 1
    ks_windows /= ks_windows.sum(axis=1, keepdims=True)
    centers = (mins + maxs) / 2
    print(f"{centers.shape=} {drift_samples.shape=} {drift['um'].shape=}")

    ksme = motion_est.NonrigidMotionEstimate(
        -drift["um"].T,
        time_bin_centers_s=drift_samples / fs,
        spatial_bin_centers_um=centers,
    )
    kextra = dict(window_centers=centers, windows=ks_windows)

    pyks_regs[pid] = dict(me=ksme, extra=kextra)

# %%
regs_compare = [
    ("pyks", pyks_regs),
    ("thomas", thomases),
]

for pid in pids:
    for regname, theregs in regs_compare:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f"{regname}  {pid}", fontsize=12)

        me = theregs[pid]["me"]
        z = feats[pid]["z"]
        a = feats[pid]["a"]
        t = feats[pid]["t"]
        zr = me.correct_s(t, z)

        r, dd, tt = ibme.fast_raster(a, zr, t)

        ax.imshow(r, aspect="auto", vmax=15, cmap=plt.cm.cubehelix)

        plt.show()

        plt.close(fig)

# %%
