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
import spikeinterface.full as si

# %%
import matplotlib.pyplot as plt

# %%
import seaborn as sns

# %%
plt.rc("figure", dpi=200)

# %%
from spike_psvae import deconvolve, extract_deconv, waveform_utils

# %%
spikelen = 21
reclen = 10000
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
stsort = np.argsort(st[:, 0], kind="stable")
st = st[stsort]
st.shape

# %%
collided = np.zeros(len(st), dtype=bool)
tmcs = templates.ptp(1).argmax(1)
for i in range(len(st)):
    if i == 0:
        collided[i] = abs(st[i + 1, 0] - st[i, 0]) < spikelen and (tmcs[st[i + 1, 1]] == tmcs[st[i, 1]])
    elif i == len(st) - 1:
        collided[i] = (abs(st[i, 0] - st[i - 1, 0]) < spikelen) and (tmcs[st[i, 1]] == tmcs[st[i - 1, 1]])
    else:
        collided[i] = abs(st[i + 1, 0] - st[i, 0]) < spikelen and (tmcs[st[i + 1, 1]] == tmcs[st[i, 1]])
        collided[i] |= (abs(st[i, 0] - st[i - 1, 0]) < spikelen) and (tmcs[st[i, 1]] == tmcs[st[i - 1, 1]])

# %%
collided.sum()

# %%
scales = np.ones(len(st))
scales = rg.normal(size=len(st), loc=1, scale=0.1)
scales = np.clip(scales, 0.8, 1.3)

# %%
time_range = np.arange(-trough, spikelen - trough)
time_ix = st[:, 0, None] + time_range[None, :]
chan_ix = write_chan_ix[st[:, 1]]
raw = rg.normal(size=(reclen, nchan), scale=1.0).astype(np.float64)
raw2 = raw.copy()
raw3 = raw.copy()

np.add.at(
    raw,
    (time_ix[:, :, None], chan_ix[:, None, :]),
    scales[:, None, None] * templates[st[:, 1]],
)

# %%
import torch

# %%
np.add.at(
    raw2.reshape(-1),
    np.ravel_multi_index(
        np.broadcast_arrays(time_ix[:, :, None], chan_ix[:, None, :]),
        # (scales[:, None, None] * templates[st[:, 1]]).shape,
        raw2.shape,
    ),
    scales[:, None, None] * templates[st[:, 1]],
)

# %%
raw3.ptp(1)

# %%
torch.nan

# %%
np.array_equal(raw2, raw)

# %%
raw3 = torch.as_tensor(raw3)

# %%
raw3.dtype

# %%
torch.as_tensor(scales[:, None, None] * templates[st[:, 1]]).dtype

# %%
torch.ravel_m

# %%
raw3.reshape(-1).scatter_add_(
    0,
    torch.as_tensor(np.ravel_multi_index(
        np.broadcast_arrays(time_ix[:, :, None], chan_ix[:, None, :]),
        # (scales[:, None, None] * templates[st[:, 1]]).shape,
        raw3.shape,
    )).reshape(-1),
    torch.as_tensor(scales[:, None, None] * templates[st[:, 1]]).reshape(-1),
)

# %%
raw3[time_ix[:, :, None], chan_ix[:, None, :]].shape

# %%
np.array_equal(raw3, raw)

# %%
np.array_equal(raw3[time_ix[:, :, None], chan_ix[:, None, :]], raw[time_ix[:, :, None], chan_ix[:, None, :]])

# %%
# torch.stack?

# %%
torch.stack((torch.arange(10), torch.arange(10)), 1).shape

# %%
torch.scatter_add(
    torch.as_tensor(scales[:, None, None] * templates[st[:, 1]], dtype=torch.float32),
    torch.as_tensor(np.ravel_multi_index(
        np.broadcast_arrays(time_ix[:, :, None], chan_ix[:, None, :]),
        # (scales[:, None, None] * templates[st[:, 1]]).shape,
        raw3.shape,
    )),
    raw3.reshape(-1),
)

# %%

# %%

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
coco = sparse.dok_matrix((3,3))

# %%
coco

# %%
coco[1, 1] = 2

# %%
1 + coco

# %%
from scipy.optimize import linear_sum_assignment
from scipy import sparse
from scipy.sparse.csgraph import maximum_bipartite_matching, min_weight_full_bipartite_matching

def timesagree_dense(times1, times2, max_dt=21):
    # C = np.full((len(times1), len(times2)), fill_value=np.inf)
    # for i, t1 in enumerate(times1):
    #     for j, t2 in enumerate(times2):
    #         if abs(t1 - t2) <= max_dt:
    C = np.abs(times1[:, None] - times2[None, :]).astype(float)
    # valid = C > max_dt
    # valid_1 = valid.any(axis=1)
    # valid_2 = valid.any(axis=1)
    # C[~valid] = 1000 + C.max()
    # print(valid_1.sum(), valid_2.sum())
    
    ii, jj = linear_sum_assignment(C)
    costs = C[ii, jj]
    valid = costs <= max_dt
    
    # tp: matched spikes
    tp = valid.sum()
    # fn: true spikes not found
    fn = len(times1) - tp
    # fp: new spikes not matched
    fp = len(times2) - tp
    
    accuracy = tp / (tp + fn + fp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fdr = fp / (tp + fp)
    miss_rate = fn / len(times1)
    
    return dict(accuracy=accuracy, recall=recall, precision=precision, fdr=fdr, miss_rate=miss_rate)

def timesagree_sparse(times1, times2, max_dt=21):
    searchrad = 5 * max_dt
    if len(times1) > len(times2):
        times1, times2 = times2, times1
    
    dtdt = sparse.dok_matrix((len(times1), len(times2)))
    C = sparse.dok_matrix((len(times1), len(times2)))
    min_j = 0
    max_j = 1
    for i, t1 in enumerate(times1):
        for j, t2 in enumerate(times2):
            t2 = times2[j]
            dtdt[i, j] = abs(t1 - t2)
            if abs(t1 - t2) <= searchrad:
                C[i, j] = -(1+abs(t1 - t2))
    dtdt = dtdt.tocsr()
    perm = maximum_bipartite_matching(-C.tocsr())
    # ii, jj = min_weight_full_bipartite_matching(C.tocsr())
    # cost = (C[perm].diagonal() <= max_dt).sum()
    
    # tp: matched spikes
    tp = ((dtdt[perm[perm >= 0]].diagonal()) <= max_dt).sum()
    # tp = (dtdt[ii, jj] <= max_dt).sum()
    # fn: true spikes not found
    fn = len(times1) - tp
    # fp: new spikes not matched
    fp = len(times2) - tp
    
    accuracy = tp / (tp + fn + fp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fdr = fp / (tp + fp)
    miss_rate = fn / len(times1)
    
    return dict(accuracy=accuracy, recall=recall, precision=precision, fdr=fdr, miss_rate=miss_rate)

def timesagree_sparse_fast(times1, times2, max_dt=21):
    # if np.array_equal(times1, times2):
    #     return dict(accuracy=1, recall=1, precision=1, fdr=0, miss_rate=0)
    # print("----------------")
    # print("----------------")
    # print("----------------")
    # print(f"{times1=}")
    # print(f"{times2=}")
    searchrad = 5 * max_dt
    if len(times1) > len(times2):
        times1, times2 = times2, times1
    
    dtdt = sparse.dok_matrix((len(times1), len(times2)))
    C = sparse.dok_matrix((len(times1), len(times2)))
    min_j = 0
    max_j = 1
    for i, t1 in enumerate(times1):
        while max_j + 1 < len(times2) and times2[max_j] <= t1 + searchrad:
            # print(f"{t1=} {max_j=} {times2[max_j]=} {(t1 + searchrad)=}")
            max_j += 1
        # print(f"final A {t1=} {max_j=} {times2[max_j]=} {(t1 + searchrad)=}")
        while min_j < max_j and times2[min_j + 1] < t1 - searchrad:
            # print(f"{t1=} {min_j=} {max_j=} {times2[min_j]=} {times2[min_j + 1]=} {(t1 - searchrad)=}")
            min_j += 1
        # print(f"final B {t1=} {min_j=} {times2[min_j]=} {(t1 - searchrad)=}")
        # for j, t2 in enumerate(times2):
        # print("start loop")
        for j in range(min_j, max_j + 1):
            t2 = times2[j]
            # print(f"{j=} {t1=} {t2=}")
            dtdt[i, j] = abs(t1 - t2)
            # print(f"{abs(t1 - t2)=} {searchrad=} {(abs(t1 - t2) <= searchrad)=}")
            if abs(t1 - t2) <= searchrad:
                # print(f"comp {t1=} {t2=}")
                C[i, j] = -(1+abs(t1 - t2))
    dtdt = dtdt.tocsr()
    perm = maximum_bipartite_matching(-C.tocsr())
    # ii, jj = min_weight_full_bipartite_matching(C.tocsr())
    # cost = (C[perm].diagonal() <= max_dt).sum()
    
    # tp: matched spikes
    tp = ((dtdt[perm[perm >= 0]].diagonal()) <= max_dt).sum()
    # tp = (dtdt[ii, jj] <= max_dt).sum()
    # fn: true spikes not found
    fn = len(times1) - tp
    # fp: new spikes not matched
    fp = len(times2) - tp
    
    accuracy = tp / (tp + fn + fp)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fdr = fp / (tp + fp)
    miss_rate = fn / len(times1)
    
    return dict(accuracy=accuracy, recall=recall, precision=precision, fdr=fdr, miss_rate=miss_rate)
  
timesagree = timesagree_dense
timesagree = timesagree_sparse

def metrics_matrix(st1, st2, max_dt=21):
    k1 = st1[:, 1].max() + 1
    k2 = st2[:, 1].max() + 1
    metrics = {k: np.zeros((k1, k2)) for k in ("accuracy", "recall", "precision", "fdr", "miss_rate")}
    
    for u1 in range(k1):
        times1 = st1[st1[:, 1] == u1, 0]
        if not len(times1):
            continue
        for u2 in range(k2):
            times2 = st2[st2[:, 1] == u2, 0]
            if not len(times2):
                continue
            
            ag12 = timesagree(times1, times2, max_dt=max_dt)
            ag21 = timesagree(times2, times1, max_dt=max_dt)
            ag = ag12 if ag12["accuracy"] > ag21["accuracy"] else ag21
            for k in metrics.keys():
                metrics[k][u1, u2] = ag[k]

    return metrics

def hungarian_metrics(metrics):
    best_match_i, best_match_j = linear_sum_assignment(-metrics["accuracy"])
    assert np.array_equal(best_match_i, np.arange(len(best_match_i)))  # o/w need sort
    print(best_match_i)
    return {k: metrics[k][best_match_i, best_match_j] for k in metrics.keys()}


# %%
# SI weirdness??
for dt in (0, 1, 5, 11, 21, 31, 100):
    print(f"{dt=}")
    sorting_gt = si.NumpySorting.from_times_labels(*st.T, sampling_frequency=1000)
    cmp = si.compare_sorter_to_ground_truth(sorting_gt, sorting_gt, exhaustive_gt=True, delta_time=2*dt, match_mode="hungarian")
    fig, (aa, ab) = plt.subplots(ncols=2, figsize=(4, 2))
    # si.plot_agreement_matrix(cmp, ordered=True, ax=aa)
    
    mets = metrics_matrix(st, st, max_dt=dt)
    perf = hungarian_metrics(mets)

    # To have an overview of the match we can use the ordered agreement matrix
    # si.plot_agreement_matrix(cmp, ordered=True, figure=fig)
    siperf = cmp.get_performance()
    sns.heatmap(cmp.agreement_scores, annot=True, cmap=plt.cm.Greens, ax=aa)
    sns.heatmap(mets["accuracy"], annot=True, cmap=plt.cm.Reds, ax=ab)
    plt.show()
    plt.close("all")


# %%
def check_deconv_res(dec_temps_up, dec_st_up, dec_scales, dec_temps, dec_st):
    plt.figure(figsize=(3,2))
    plt.hist(dec_scales, bins=64);
    plt.xlabel("recovered scalings")
    plt.show()
    plt.close("all")
    
    exh5, residpath = extract_deconv.extract_deconv(
        # res_scale["templates_up"],
        # res_scale['deconv_spike_train_upsampled'],
        dec_temps_up,
        dec_st_up,
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

        # print(f"{ci=}")

        sub_mcts = waveform_utils.get_maxchan_traces(sub_wfs, ci, mcs)
        cleaned_mcts = waveform_utils.get_maxchan_traces(cleaned_wfs, ci, mcs)
        # cleaned_mcts = cleaned_wfs[np.arange(len(cleaned_wfs)), :, cleaned_wfs.ptp(1).argmax(1)]
        # print(f"{(cleaned_wfs == 0).all(axis=(1, 2)).sum()=}")
        # print(f"{cleaned_mcts.shape=} {true_mcts.shape=} {cleaned_wfs.shape=} {true_wfs.shape=}")
        # print(f"{(cleaned_mcts == 0).all(axis=(1,)).sum()=}")
        # print(f"{(np.abs(cleaned_wfs) < 3).all(axis=(1,2)).sum()=}")
        # print(f"{(np.abs(cleaned_mcts) < 3).all(axis=(1,)).sum()=}")
    

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
    print(f"total resid norm: {np.linalg.norm(resid):0.2f}")
    resid_waveforms = resid[time_ix[:, :, None], chan_ix[:, None, :]]
    resid_mcts = waveform_utils.get_maxchan_traces(resid_waveforms, waveform_utils.full_channel_index(geom.shape[0]), templates.ptp(1).argmax(1)[st[:, 1]])
    resid_norms = np.linalg.norm(resid_mcts, axis=1)
    # plt.figure(figsize=(3,2))
    # plt.hist(resid_norms, bins=32);
    # plt.show()
    # plt.close("all")
    
    # check accuracy
    # sorting_gt = si.NumpySorting.from_times_labels(*st.T, sampling_frequency=1000)
    # sorting_test = si.NumpySorting.from_times_labels(*dec_st.T, sampling_frequency=1000)
    # cmp = si.compare_sorter_to_ground_truth(sorting_gt, sorting_test, exhaustive_gt=True, delta_time=2 * spikelen + 1, match_mode="hungarian")
    # agreement = agreement_matrix(st, dec_st)
    mets = metrics_matrix(st, dec_st)
    perf = hungarian_metrics(mets)

    # To have an overview of the match we can use the ordered agreement matrix
    fig = plt.figure(figsize=(3, 3))
    # si.plot_agreement_matrix(cmp, ordered=True, figure=fig)
    sns.heatmap(mets["accuracy"], annot=True, cmap=plt.cm.Greens)
    plt.show()
    plt.close("all")

    # This function first matches the ground-truth and spike sorted units, and
    # then it computes several performance metrics: accuracy, recall, precision
    # perf = cmp.get_performance()
    
    return resid_norms, np.linalg.norm(resid), perf

# %%
spikelen

# %%
gt_resid_norms, gt_totalresidnorm, gt_perf = check_deconv_res(templates, st, scales, templates, st)

# %% tags=[]
lambds = (0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10)
residnorms = []
mean_maxchan_residnorms = []
mean_maxchan_residnorms_collided = []
perfs = []
for lambd in lambds:
    print(f"{lambd=}")
    res_scale = deconvolve.deconv(
        "/tmp/testaaa.bin",
        "/tmp/testdeconv",
        templates,
        lambd=lambd,
        allowed_scale=0.4,
        max_upsample=2,
        trough_offset=0,
    )
    
    rns, totalrn, perf = check_deconv_res(res_scale["templates_up"], res_scale['deconv_spike_train_upsampled'], res_scale["deconv_scalings"], templates, res_scale["deconv_spike_train"])
    residnorms.append(totalrn)
    if lambd == 0:
        unscaled_rns = rns.copy()
    perfs.append(perf)
    
    plt.figure(figsize=(3, 3))
    mng = gt_resid_norms.min()
    mxg = gt_resid_norms.max()
    mnt = rns.min()
    mxt = rns.max()
    plt.plot([min(mng, mnt), max(mxg, mxt)], [min(mng, mnt), max(mxg, mxt)], color="k", lw=1)
    plt.scatter(gt_resid_norms[collided], rns[collided], s=25, marker="*", lw=0, color="orange", zorder=11, label="gt coll")
    plt.scatter(gt_resid_norms, rns, s=5, lw=0, color="orange", zorder=11, label="gt")
    plt.scatter(unscaled_rns[collided], rns[collided], s=25, marker="*", lw=0, color="blue", zorder=11, label="unscaled coll")
    plt.scatter(unscaled_rns, rns, s=5, lw=0, color="blue", zorder=11, label="unscaled")
    plt.xlabel("baseline")
    plt.ylabel("this")
    plt.legend(bbox_to_anchor=(1, 0.5), loc="center left")
    plt.show()
    plt.close("all")
    
    mean_maxchan_residnorms.append(np.mean(rns))
    mean_maxchan_residnorms_collided.append(np.mean(rns[collided]))
    
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
plt.plot(lambds, residnorms, label="deconv($\\lambda$) spike train")
plt.axhline(gt_totalresidnorm, color="k", label="GT spike train + scalings")
plt.legend()
plt.semilogx()
plt.ylabel("total resid norm")
plt.xlabel("$\\lambda$")

# %%
plt.plot(lambds, [p["accuracy"].mean() for p in perfs], color="b", alpha=0.5, label="mean acc")
plt.plot(lambds, [p["recall"].mean() for p in perfs], color="r", alpha=0.5, label="mean recall")
plt.plot(lambds, [p["precision"].mean() for p in perfs], color="g", alpha=0.5, label="mean prec")
plt.legend()
plt.semilogx()
plt.ylim([0, 1.05])
plt.xlabel("$\\lambda$")

# %%
plt.plot(lambds, mean_maxchan_residnorms, label="mean maxchan resid norm")
plt.plot(lambds, mean_maxchan_residnorms_collided, label="mean maxchan resid norm (collided)")
plt.axhline(np.mean(gt_resid_norms), color="k", label="GT mean")
plt.axhline(np.mean(gt_resid_norms[collided]), color="gray", label="GT collided mean")
plt.semilogx()
plt.legend()
plt.xlabel("$\\lambda$")

# %%
