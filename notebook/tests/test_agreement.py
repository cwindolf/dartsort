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
import spikeinterface.full as si
import matplotlib.pyplot as plt
import seaborn as sns

# %%
plt.rc("figure", dpi=200)

# %%

spikelen = 21
reclen = 50_000_000
trough = 0
nchan = 10
nspikes_unit = 100_000
refractory = 11

# %%
rg = np.random.default_rng(0)
# refractory spike train generator
def getst(tmin, tmax, size, ref=refractory):
    # rejection sampling is too slow
    # while True:
    #     st = rg.choice(
    #         np.arange(tmin, tmax), size=size, replace=False
    #     )
    #     st.sort()
    #     if np.diff(st).min() >= ref:
    #         return st
    
    # stars and bars solution
    # we want size stars spaced out by at least `refractory` bars,
    # out of tmax - tmin total stars+bars
    total_refractory_bars = (size - 1) * refractory
    n_other_bars = tmax - tmin - size - total_refractory_bars
    
    # randomly distribute the stars among the non-refractory bars
    stars_and_other_bars = np.zeros(size + n_other_bars, dtype=bool)
    stars_and_other_bars[:size] = True
    rg.shuffle(stars_and_other_bars)
    
    # find their positions and space them out
    spike_times = np.flatnonzero(stars_and_other_bars)
    spike_times += refractory * np.arange(size)
    
    return spike_times + tmin


# %%
# generate spike trains for 3 units
sta = getst(0, reclen - spikelen, size=nspikes_unit)
sta = np.c_[sta, np.zeros_like(sta)]
stb = getst(0, reclen - spikelen, size=nspikes_unit)
stb = np.c_[stb, 1 + np.zeros_like(stb)]
stc = getst(0, reclen - spikelen, size=nspikes_unit)
stc = np.c_[stc, 2 + np.zeros_like(stc)]
st = np.r_[sta, stb, stc]
stsort = np.argsort(st[:, 0], kind="stable")
st = st[stsort]
st.shape

# %%
np.diff(st[:,0]).mean()

# %%
np.diff(sta[:,0]).min(), np.diff(stb[:,0]).min(), np.diff(stc[:,0]).min()

# %%
np.diff(sta[:,0]).mean(), np.diff(stb[:,0]).mean(), np.diff(stc[:,0]).mean()

# %%
from scipy.optimize import linear_sum_assignment
from scipy import sparse
from scipy.sparse.csgraph import maximum_bipartite_matching, min_weight_full_bipartite_matching

def timesagree_dense(times1, times2, max_dt=21):
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


def timesagree_dense_sparsified(times1, times2, max_dt=21):
    C = np.abs(times1[:, None] - times2[None, :]).astype(float)
    # valid = C > max_dt
    # valid_1 = valid.any(axis=1)
    # valid_2 = valid.any(axis=1)
    # C[~valid] = 1000 + C.max()
    # print(valid_1.sum(), valid_2.sum())
    
    searchrad = 10 * max_dt
    for i, t1 in enumerate(times1):
        for j, t2 in enumerate(times2):
            C[i, j] = 100 + searchrad
            t2 = times2[j]
            if abs(t1 - t2) <= searchrad:
                C[i, j] = abs(t1 - t2)

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
    searchrad = 50 * max_dt
    if len(times1) > len(times2):
        times1, times2 = times2, times1
    
    dtdt = sparse.dok_matrix((len(times1), len(times2)))
    C = sparse.dok_matrix((len(times1), len(times2)))
    min_j = 0
    max_j = 1
    for i, t1 in enumerate(times1):
        while max_j + 1 < len(times2) and times2[max_j] <= t1 + searchrad:
            max_j += 1
        while min_j < max_j and times2[min_j + 1] < t1 - searchrad:
            min_j += 1
        for j in range(min_j, max_j + 1):
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
  
def metrics_matrix(st1, st2, max_dt=21, fn=timesagree_sparse):
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
            
            ag12 = fn(times1, times2, max_dt=max_dt)
            ag21 = fn(times2, times1, max_dt=max_dt)
            ag = ag12 if ag12["accuracy"] > ag21["accuracy"] else ag21
            for k in metrics.keys():
                metrics[k][u1, u2] = ag[k]

    return metrics

def hungarian_metrics(metrics):
    best_match_i, best_match_j = linear_sum_assignment(-metrics["accuracy"])
    assert np.array_equal(best_match_i, np.arange(len(best_match_i)))  # o/w need sort
    return {k: metrics[k][best_match_i, best_match_j] for k in metrics.keys()}


# %%
import time

# %%
for dt in (0, 1, 5, 11, 21, 31, 100):
    print(f"----- {dt=}")
    
    # spikeinterface matching
    sorting_gt = si.NumpySorting.from_times_labels(*st.T, sampling_frequency=1000)
    
    toc = time.time()
    cmp = si.compare_sorter_to_ground_truth(
        sorting_gt,
        sorting_gt,
        exhaustive_gt=True,
        # spikeinterface has a //2 in the spike train matching function
        # so, multiply by 2 here for a fair comparison to the method above
        # delta_time=2 * dt,
        delta_time=dt,
        match_mode="hungarian",
    )
    siperf = cmp.get_performance()
    sitic = time.time() - toc
    
    # this matching
    toc = time.time()
    mets_sparse = metrics_matrix(st, st, max_dt=dt, fn=timesagree_sparse_fast)
    perf_sparse = hungarian_metrics(mets_sparse)
    sftic = time.time() - toc
    
#     toc = time.time()
#     mets_slow = metrics_matrix(st, st, max_dt=dt, fn=timesagree_dense_sparsified)
#     perf_slow = hungarian_metrics(mets_slow)
#     stic = time.time() - toc
    
#     toc = time.time()
#     mets_dense = metrics_matrix(st, st, max_dt=dt, fn=timesagree_dense)
#     perf_dense = hungarian_metrics(mets_dense)
#     dtic = time.time() - toc
    
    # plot both
    fig, ((aa, ab), (ac, ad)) = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    sns.heatmap(cmp.agreement_scores, annot=True, cmap=plt.cm.Greens, ax=aa)
    sns.heatmap(mets_sparse["accuracy"], annot=True, cmap=plt.cm.Reds, ax=ab)
    # sns.heatmap(mets_slow["accuracy"], annot=True, cmap=plt.cm.Purples, ax=ac)
    # sns.heatmap(mets_dense["accuracy"], annot=True, cmap=plt.cm.Blues, ax=ad)
    
    aa.set_title(f"si -- {sitic:0.2f}s")
    ab.set_title(f"sp fast -- {sftic:0.2f}s")
    # ac.set_title(f"sp slow -- {stic:0.2f}s")
    # ad.set_title(f"exact -- {dtic:0.2f}s")
    
    plt.show()
    plt.close("all")

# %%
