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
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import argparse
import spikeinterface.comparison as sc
from spikeinterface.extractors import NumpySorting
import spikeinterface.widgets as sw
from sklearn.decomposition import PCA
from scipy.optimize import least_squares

# %%
from scipy.spatial import KDTree
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
from IPython.display import display

# %%
from spike_psvae import (
    # cluster,
    # merge_split_cleaned,
    # cluster_viz_index,
    # denoise,
    cluster_utils,
    triage,
    cluster_viz,
    cluster_viz_index,
    localize_index,
    grab_and_localize,
)

# ap = argparse.ArgumentParser()
# ap.add_argument("subject")
# args = ap.parse_args()
# subject = args.subject


hybrid_bin_dir = Path("/mnt/3TB/charlie/hybrid_1min_output/")
hybrid_res_dir = Path("/mnt/3TB/charlie/hybrid_1min_subtraction/")
hybrid_ks_dir = Path("/mnt/3TB/charlie/hybrid_1min_kilosort/")
hybrid_deconv_dir = Path("/mnt/3TB/charlie/hybrid_1min_deconv/")


# %%
hybrid_fig_dir = Path("/mnt/3TB/charlie/figs/hybrid_figs/")
hybrid_fig_dir.mkdir(exist_ok=True)

# %%
# %matplotlib inline

# %%
plt.rc("figure", dpi=200)

# %%
subjects = [d.stem for d in hybrid_res_dir.glob("*") if d.is_dir()]
len(subjects), subjects

# %% [markdown]
# ### for each subject, get the following
#
# per clustering:
#  - agreement matrix
#  - number of units
#  - percentage spikes which were non-GT
#  - per unit:
#    - accuracy
#    - GT average maxptp
#    - detected average max ptp
#    - GT firing rate
#    - detected firing rate
#   
# global:
#  - percent of GT spikes detected
#  - percent surviving triage
#  - percent in an initial clustering unit
#  - percent in a final clustering unit
#  

# %%
unit_records = []
subject_records = []
spike_dfs = []
gtunit_info_by_subj = {}
reloc_results = {}

per_sort_unit_records = []


def sorting_and_cmp(spike_train, sorting_gt, name):
    sorting = NumpySorting.from_times_labels(
        times_list=spike_train[:, 0],
        labels_list=spike_train[:, 1],
        sampling_frequency=30_000,
    )

    cmp_gt = sc.compare_two_sorters(
        sorting_gt,
        sorting,
        sorting1_name="hybrid_gt",
        sorting2_name=name,
        verbose=True,
        match_score=0.1,
    )
    return sorting, cmp_gt


def unit_match(gt_unit, cmp_gt_sorter):
    match = cmp_gt_sorter(gt_unit)
    acc = 0
    n_spikes = 0

    if match >= 0:
        acc = cmp_gt_sorter.get_agreement_fraction(gt_unit, match)
        n_spikes = cmp_gt_sorter.get_agreement_fraction(gt_unit, match)

    return match, acc, n_spikes


def unsorted_detection(gt_spike_index, spike_index, n_samples=12, n_channels=4):
    cmul = n_samples / n_channels

    gt_kdt = KDTree(np.c_[gt_spike_index[:, 0], gt_spike_index[:, 1] * cmul])
    sorter_kdt = KDTree(np.c_[spike_index[:, 0], cmul * spike_index[:, 1]])
    query = gt_kdt.query_ball_tree(sorter_kdt, n_samples + 0.1)
    detected = np.array([len(lst) > 0 for lst in query], dtype=bool)
    n_detected = detected.sum()

    return n_detected, detected


for subject in tqdm(subjects):
    # if subject in ("SWC_054", "ZM_2241", "NYU-21", "SWC_060", "ZFM-01592"):
    # continue
    print(subject)
    raw_data_bin = hybrid_bin_dir / f"{subject}.ap.bin"
    assert raw_data_bin.exists()
    residual_data_bin = next((hybrid_res_dir / subject).glob("residual*bin"))
    assert residual_data_bin.exists()
    sub_h5 = next((hybrid_res_dir / subject).glob("sub*h5"))
    assert sub_h5.exists()

    hybrid_gt_h5 = hybrid_bin_dir / f"{subject}_gt.h5"
    output_dir = hybrid_res_dir / subject / "gt_comparison"

    try:
        deconv_spike_train = np.load(
            hybrid_deconv_dir / subject / "spike_train.npy"
        )
        deconv_templates = np.load(
            hybrid_deconv_dir / subject / "templates.npy"
        )
        deconv_samples, deconv_labels = deconv_spike_train.T
        deconv_template_mc = deconv_templates.ptp(1).argmax(1)
        deconv_spike_index = np.c_[
            deconv_samples,
            deconv_template_mc[deconv_labels],
        ]
    except:
        print("no deconv yet for", subject)
        continue

    ks_samples = (
        np.load(hybrid_ks_dir / subject / "spike_times.npy")
        .squeeze()
        .astype(int)
    )
    ks_labels = (
        np.load(hybrid_ks_dir / subject / "spike_clusters.npy")
        .squeeze()
        .astype(int)
    )
    ksall_spike_train = np.c_[ks_samples, ks_labels]
    ks_templates = np.load(hybrid_ks_dir / subject / "templates.npy")
    with h5py.File(hybrid_ks_dir / subject / "rez2.mat") as h5:
        ks_good = h5["rez"]["good"][:].squeeze().astype(int)
    ksgood_spike_train = ksall_spike_train[np.isin(ks_labels, ks_good)]
    ks_template_mc = ks_templates.ptp(1).argmax(1)
    ksall_spike_index = np.c_[ksall_spike_train[:, 0], ks_template_mc[ksall_spike_train[:, 1]]]
    ksgood_spike_index = np.c_[ksgood_spike_train[:, 0], ks_template_mc[ksgood_spike_train[:, 1]]]

    # -- Load HDB
    with h5py.File(sub_h5, "r") as h5:
        geom_array = h5["geom"][:]
        end_sample = h5["end_sample"][()]
        start_sample = h5["start_sample"][()]
        recording_length = (end_sample - start_sample) / 30000
        channel_index = h5["channel_index"][:]
        n_detections = h5["spike_index"].shape[0]
        o_spike_index = h5["spike_index"][:]
        locs = h5["localizations"][:]
        z_reg = h5["z_reg"][:]
        maxptps = h5["maxptps"][:]

    *_, ptp_keep, idx_keep = triage.run_weighted_triage(
        locs[:, 0], locs[:, 1], z_reg, locs[:, 4], maxptps, threshold=85
    )
    original_triage = ptp_keep[idx_keep]

    # pre merge/split
    pmhdb_spike_index_ = np.load(
        hybrid_res_dir / subject / "pre_merge_split_aligned_spike_index.npy"
    )
    pmhdb_labels = np.load(
        hybrid_res_dir / subject / "pre_merge_split_labels.npy"
    )
    pmhdb_spike_train = np.c_[
        pmhdb_spike_index_[pmhdb_labels >= 0, 0],
        pmhdb_labels[pmhdb_labels >= 0],
    ]
    pmhdb_spike_index = pmhdb_spike_index_[pmhdb_labels >= 0]
    print("__", pmhdb_spike_index_.shape, pmhdb_spike_index.shape)

    # post merge/split
    hdb_spike_index_ = np.load(
        hybrid_res_dir / subject / "aligned_spike_index.npy"
    )
    hdb_labels = np.load(hybrid_res_dir / subject / "labels.npy")
    hdb_spike_train = np.c_[
        hdb_spike_index_[hdb_labels >= 0, 0],
        hdb_labels[hdb_labels >= 0],
    ]
    hdb_spike_index = hdb_spike_index_[hdb_labels >= 0]

    # -- Load GT
    with h5py.File(hybrid_gt_h5, "r") as gt_h5:
        gt_spike_train = gt_h5["spike_train"][:]
        gt_spike_index = gt_h5["spike_index"][:]
        templates = gt_h5["templates"][:]

    gt_template_maxchans = templates.ptp(1).argmax(1)
    gt_template_locs = localize_index.localize_ptps_index(
        templates.ptp(1),
        geom_array,
        gt_template_maxchans,
        np.stack([np.arange(len(geom_array))] * len(geom_array), axis=0),
        n_channels=20,
        n_workers=None,
        pbar=True,
    )
    gt_template_depths = gt_template_locs[3]
    # sort gt train
    argsort = np.argsort(gt_spike_train[:, 0])
    gt_spike_train = gt_spike_train[argsort]
    gt_spike_index = gt_spike_index[argsort]

    gt_relocalizations, gt_remaxptp = grab_and_localize.grab_and_localize(
        gt_spike_index,
        raw_data_bin,
        geom_array,
        loc_radius=100,
        nn_denoise=True,
        enforce_decrease=True,
        tpca=tpca,
        chunk_size=30_000,
        n_jobs=4,
    )
    reloc_results[subject] = dict(
        gt_relocalizations=gt_relocalizations,
        gt_remaxptp=gt_remaxptp,
        gt_spike_index=gt_spike_index,
        gt_spike_train=gt_spike_train,
    )

    # -- comparisons
    sorting_gt = NumpySorting.from_times_labels(
        times_list=gt_spike_train[:, 0],
        labels_list=gt_spike_train[:, 1],
        sampling_frequency=30_000,
    )
    sorting_hdb, cmp_gt_hdb = sorting_and_cmp(hdb_spike_train, sorting_gt, "hdb")
    sorting_pmhdb, cmp_gt_pmhdb = sorting_and_cmp(pmhdb_spike_train, sorting_gt, "pmhdb")
    sorting_deconv, cmp_gt_deconv = sorting_and_cmp(deconv_spike_train, sorting_gt, "deconv")
    sorting_ksall, cmp_gt_ksall = sorting_and_cmp(ksall_spike_train, sorting_gt, "ksall")
    sorting_ksgood, cmp_gt_ksgood = sorting_and_cmp(ksgood_spike_train, sorting_gt, "ksgood")

    gt_units, gt_counts = np.unique(gt_spike_train[:, 1], return_counts=True)
    units_matched_pmhdb = np.zeros(gt_units.shape, dtype=bool)
    units_matched_hdb = np.zeros(gt_units.shape, dtype=bool)
    units_matched_deconv = np.zeros(gt_units.shape, dtype=bool)
    units_matched_ksall = np.zeros(gt_units.shape, dtype=bool)
    units_matched_ksgood = np.zeros(gt_units.shape, dtype=bool)

    n_matched_hdb = 0
    n_matched_pmhdb = 0
    n_matched_ksgood = 0
    n_matched_ksall = 0
    n_matched_deconv = 0

    this_subject_gtunit_info_by_subjs = []

    for i, (unit, tloc) in enumerate(zip(gt_units, zip(*gt_template_locs))):
        assert i == unit
        gt_ptp = templates[unit].ptp(1).max()
        hdb_match, hdb_acc, hdb_n_spikes = unit_match(unit, cmp_gt_hdb)
        pmhdb_match, pmhdb_acc, pmhdb_n_spikes = unit_match(unit, cmp_gt_pmhdb)
        deconv_match, deconv_acc, deconv_n_spikes = unit_match(unit, cmp_gt_deconv)
        ksall_match, ksall_acc, ksall_n_spikes = unit_match(unit, cmp_gt_ksall)
        ksgood_match, ksgood_acc, ksgood_n_spikes = unit_match(unit, cmp_gt_ksgood)

        n_matched_hdb += hdb_n_spikes
        n_matched_pmhdb += pmhdb_n_spikes
        n_matched_deconv += deconv_n_spikes
        n_matched_ksall += ksall_n_spikes
        n_matched_ksgood += ksgood_n_spikes

        units_matched_pmhdb[unit] = pmhdb_match >= 0
        units_matched_hdb[unit] = hdb_match >= 0
        units_matched_deconv[unit] = deconv_match >= 0
        units_matched_ksall[unit] = ksall_match >= 0
        units_matched_ksgood[unit] = ksgood_match >= 0

        unit_records.append(
            dict(
                subject=subject,
                gt_ptp=gt_ptp,
                gt_firing_rate=gt_counts[i] / 60,
                hdb_firing_rate=hdb_n_spikes / 60,
                pmhdb_firing_rate=pmhdb_n_spikes / 60,
                hdb_match=hdb_match,
                pmhdb_match=pmhdb_match,
                deconv_match=deconv_match,
                ksall_match=ksall_match,
                ksgood_match=ksgood_match,
                hdb_acc=hdb_acc,
                pmhdb_acc=pmhdb_acc,
                deconv_acc=deconv_acc,
                ksall_acc=ksall_acc,
                ksgood_acc=ksgood_acc,
            )
        )

        this_subject_gtunit_info_by_subjs.append(
            dict(
                gt_ptp=gt_ptp,
                unit=unit,
                hdb_match=hdb_match,
                hdb_acc=hdb_acc,
                ksall_match=ksall_match,
                ksall_acc=ksall_acc,
                gt_firing_rate=gt_counts[i] / 60,
                gt_x=tloc[0],
                gt_y=tloc[1],
                gt_z=tloc[3],
                gt_alpha=tloc[4],
            )
        )
    gtunit_info_by_subj[subject] = this_subject_gtunit_info_by_subjs

    # what percent were detected?
    n_detected, detected = unsorted_detection(
        gt_spike_index, hdb_spike_index_
    )
    n_detected_triage, triage_detected = unsorted_detection(
        gt_spike_index, hdb_spike_index_[original_triage]
    )
    n_detected_pmhdb, pmhdb_detected = unsorted_detection(
        gt_spike_index, hdb_spike_index_[pmhdb_labels >= 0]
    )
    n_detected_deconv, deconv_detected = unsorted_detection(
        gt_spike_index, deconv_spike_index
    )
    n_detected_ksall, ksall_detected = unsorted_detection(
        gt_spike_index, ksall_spike_index
    )
    n_detected_ksgood, ksgood_detected = unsorted_detection(
        gt_spike_index, ksgood_spike_index
    )

    # of these spikes, which were part of matched units?
    # this is a "sorted" version of the above... although not sure it's so great
    pmhdb_matched = pmhdb_detected & units_matched_pmhdb[gt_spike_train[:, 1]]
    hdb_matched = hdb_detected & units_matched_hdb[gt_spike_train[:, 1]]
    deconv_matched = deconv_detected & units_matched_deconv[gt_spike_train[:, 1]]
    ksall_matched = ksall_detected & units_matched_ksall[gt_spike_train[:, 1]]
    ksgood_matched = ksgood_detected & units_matched_ksgood[gt_spike_train[:, 1]]

    subject_records.append(
        dict(
            n_gt=len(gt_spike_train),
            n_detections=n_detections,
            n_triage=len(original_triage),
            pct_detected=n_detected / len(gt_spike_train),
            pct_kept=n_detected_triaged / len(gt_spike_train),
            pct_pmhdb=n_matched_pmhdb / len(gt_spike_train),
            pct_deconv=n_matched_deconv / len(gt_spike_train),
            pct_hdb=n_matched_hdb / len(gt_spike_train),
            pct_ksall=n_matched_ksall / len(gt_spike_train),
            pct_ksgood=n_matched_ksgood / len(gt_spike_train),
        )
    )

    gt_maxptps = templates.ptp(1).max(1)
    spike_dfs.append(
        pd.DataFrame(
            data=dict(
                gt_samples=gt_spike_index[:, 0],
                gt_maxchans=gt_spike_index[:, 1],
                gt_labels=gt_spike_train[:, 1],
                gt_ptp=gt_maxptps[gt_spike_train[:, 1]],
                detected=detected,
                triaged=triage_detected,
                pmhdb_detected=pmhdb_detected,
                hdb_detected=hdb_detected,
                deconv_detected=deconv_detected,
                ksall_detected=ksall_detected,
                ksgood_detected=ksgood_detected,
                subjects=np.array([subject] * len(gt_spike_index[:, 0])),
                pmhdb_matched=pmhdb_matched,
                hdb_matched=hdb_matched,
                deconv_matched=deconv_matched,
                ksall_matched=ksall_matched,
                ksgood_matched=ksgood_matched,
            )
        )
    )


# %%
per_sort_unit_df = pd.DataFrame.from_records(per_sort_unit_records)
per_sort_unit_df

# %%
subject_df = pd.DataFrame.from_records(subject_records)
subject_df

# %%
unit_df = pd.DataFrame.from_records(unit_records)
unit_df["hdb_detected"] = (unit_df["hdb_match"] >= 0).astype(int)
unit_df["pmhdb_detected"] = (unit_df["pmhdb_match"] >= 0).astype(int)
unit_df["deconv_detected"] = (unit_df["deconv_match"] >= 0).astype(int)
unit_df.head()

# %%
for i, row in subject_df.iterrows():
    print(row["pct_detected"])
    plt.plot(row[["pct_detected", "pct_kept", "pct_pmhdb", "pct_hdb", "pct_deconv"]], color="k")
plt.gca().set_xticklabels(["Detected", "Triage", "HDBSCAN", "Split/Merge", "Deconv"])
plt.title("Fraction of GT spikes recovered through the pipeline")
plt.show()

# %%
for i, row in subject_df.iterrows():
    plt.plot(row[["pct_ksall", "pct_ksgood"]], color="k")
plt.gca().set_xticklabels(["KS (all units)", "KS (good units)"])
plt.title("Fraction of GT spikes recovered by KS")
plt.show()

# %%
spikes_df = pd.concat(spike_dfs)

# %%
spikes_df

# %%
for spike_df in spike_dfs:
    print(spike_df["triaged"].any())

# %%
thresholds = [0, 6, 9, 12, np.inf]
thresholds = [0, 6, 9, np.inf]

colors = plt.cm.viridis(np.array(thresholds[:3]) / 9)
for spike_df in spike_dfs:
    ls = []
    for ti in range(3):
        which = (spike_df["gt_ptp"] >= thresholds[ti]) & (spike_df["gt_ptp"] < thresholds[ti + 1])
        thisdf = spike_df[which]
        ls.append(plt.plot(
            thisdf[["detected", "triaged", "pmhdb_detected", "hdb_detected", "deconv_detected", "ksall_detected"]].mean(axis=0),
            color=colors[ti],
        )[0])
    # plt.plot(row[["pct_detected", "pct_kept", "pct_pmhdb", "pct_hdb", "pct_deconv"]], color="k")
plt.gca().set_xticklabels(["Detect", "Triage", "HDBSCAN", "Split/Merge", "Deconv", "KSall"])
plt.legend(ls, ["<6", "6-9", ">9"], title="GT PTP")
plt.title("Unsorted detection % by PTP")
plt.ylim([0, 1.1])
plt.show()

# %%
thresholds = [0, 6, 9, np.inf]
colors = plt.cm.viridis(np.array(thresholds[:3]) / 9)
for spike_df in spike_dfs:
    ls = []
    for ti in range(3):
        which = (spike_df["gt_ptp"] >= thresholds[ti]) & (spike_df["gt_ptp"] < thresholds[ti + 1])
        thisdf = spike_df[which]
        ls.append(plt.plot(
            thisdf[["pmhdb_matched", "hdb_matched", "deconv_matched", "ksall_matched"]].mean(axis=0),
            color=colors[ti],
        )[0])
    # plt.plot(row[["pct_detected", "pct_kept", "pct_pmhdb", "pct_hdb", "pct_deconv"]], color="k")
plt.gca().set_xticklabels(["HDBSCAN", "Split/Merge", "Deconv", "KSall"])
# plt.legend(ls, ["<6", "6-9", "9-12", ">12"], title="GT PTP")
plt.legend(ls, ["<6", "6-9", ">9"], title="GT PTP")
plt.title("Sorted detection % by PTP")
plt.ylim([0, 1.1])
plt.show()

# %%

# %%
import gc; gc.collect()


# %%
def plotgistic(x="gt_ptp", y=None, c="gt_firing_rate", title=None, cmap=plt.cm.plasma):
    ylab = y
    xlab = x
    clab = c
    y = unit_df[y].values
    x = unit_df[x].values
    X = sm.add_constant(x)
    c = unit_df[c].values
    
    def resids(beta):
        return y - 1 / (1 + np.exp(-X @ beta))
    res = least_squares(resids, np.array([1, 1]))
    b = res.x
    
    fig, ax = plt.subplots()
    sort = np.argsort(x)
    domain = np.linspace(x.min(), x.max())
    l, = ax.plot(domain, 1 / (1 + np.exp(-sm.add_constant(domain) @ b)), color="k")
    
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([x.min() - 0.5, x.max() + 0.5])


    
    leg = ax.scatter(x, y, marker="x", c=c, cmap=cmap, alpha=0.75)
    
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    h, labs = leg.legend_elements(num=4)
    ax.legend(
        (*h, l), (*labs, "logistic fit"),
        title=clab, loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    if title:
        n_missed = (y < 1e-8).sum()
        plt.title(title + f" -- {n_missed} missed")
    
    return fig, ax

# %%
(hybrid_fig_dir / "ptp_v_acc").mkdir(exist_ok=True)
for i, (k, title) in enumerate([
    ("pmhdb_acc", "Per unit HDBSCAN accuracy"),
    ("hdb_acc", "Per unit Split/Merge accuracy"),
    ("deconv_acc", "Per unit Deconv accuracy"),
    ("ksall_acc", "Per unit KSall accuracy"),
]):
    fig, ax = plotgistic(y=k, title=title)
    fig.savefig(hybrid_fig_dir / hybrid_fig_dir / "ptp_v_acc" / f"{i}_{k}_ptp_v_acc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # plt.show()

# %%
(hybrid_fig_dir / "fr_v_acc").mkdir(exist_ok=True)
for i, (k, title) in enumerate([
    ("pmhdb_acc", "Per unit HDBSCAN accuracy"),
    ("hdb_acc", "Per unit Split/Merge accuracy"),
    ("deconv_acc", "Per unit Deconv accuracy"),
    ("ksall_acc", "Per unit KSall accuracy"),
]):
    fig, ax = plotgistic(x="gt_firing_rate", y=k, c="gt_ptp", title=title, cmap=plt.cm.viridis)
    fig.savefig(hybrid_fig_dir / hybrid_fig_dir / "fr_v_acc" / f"{i}_{k}_fr_v_acc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# %%
for k in reloc_results.keys():
    rr = reloc_results[k]
    ti = pd.DataFrame.from_records(template_info[k])
    display(ti.head())

# %% tags=[]
for k in tqdm(reloc_results.keys()):
    rr = reloc_results[k]
    ti = pd.DataFrame.from_records(template_info[k])
    urelocs = rr["gt_relocalizations"]
    umaxptp = rr["gt_remaxptp"]
    ulabels = rr["gt_spike_train"][:, 1]
    
    fig, axes = cluster_viz_index.array_scatter(
        ulabels,
        geom_array,
        urelocs[:, 0],
        urelocs[:, 3],
        umaxptp,
    )
    
    hdbmatch = ti["hdb_acc"].values > 0
    ksmatch = ti["ksall_acc"].values > 0
    match = np.zeros(hdbmatch.shape)
    match[hdbmatch] = 1
    match[ksmatch] = 2
    match[ksmatch & hdbmatch] = 3
    colors = ["k", "b", "r", "purple"]
    
    axes[0].scatter(*geom_array.T, marker="s", s=2, color="orange")
    ls = []
    for i in range(4):
        axes[0].scatter(
            ti["gt_x"][match == i], ti["gt_z"][match == i], color=colors[i], marker="x", s=15
        )
        axes[2].scatter(
            ti["gt_x"][match == i], ti["gt_z"][match == i], color=colors[i], marker="x", s=15
        )
        l = axes[1].scatter(
            np.log(ti["gt_ptp"][match == i]), ti["gt_z"][match == i], color=colors[i], marker="x", s=15
        )
        ls.append(l)
    plt.figlegend(
        ls,
        ["no match", "our match", "ks match", "both"],
        loc="lower center",
        ncol=4, frameon=False, borderaxespad=-0.65)
    
    fig.suptitle(f"{k} template locs + re-localizations", y=0.95)
    fig.tight_layout()
    fig.savefig(hybrid_fig_dir / "matched_scatter" / f"{k}_A_gt_reloc.png", dpi=300)
    plt.close(fig)
    # print(rr)
    # display(ti)

# %%
for k in tqdm(reloc_results.keys()):
    rr = reloc_results[k]
    ti = pd.DataFrame.from_records(template_info[k])
    urelocs = rr["gt_relocalizations"]
    umaxptp = rr["gt_remaxptp"]
    ulabels = rr["gt_spike_train"][:, 1]
    hdb_labels = np.load(hybrid_res_dir / subject / "labels.npy")
    
    
    with h5py.File(sub_h5, "r") as h5:
        geom_array = h5["geom"][:]
        # end_sample = h5["end_sample"][()]
        # start_sample = h5["start_sample"][()]
        # recording_length = (end_sample - start_sample) / 30000
        # channel_index = h5["channel_index"][:]
        # n_detections = h5["spike_index"].shape[0]
        # o_spike_index = h5["spike_index"][:]
        locs = h5["localizations"][:]
        z_reg = h5["z_reg"][:]
        maxptps = h5["maxptps"][:]
    
    fig, axes = cluster_viz_index.array_scatter(
        hdb_labels,
        geom_array,
        locs[:, 0],
        z_reg,
        maxptps,
        annotate=False,
    )
    
    hdbmatch = ti["hdb_acc"].values > 0
    ksmatch = ti["ksall_acc"].values > 0
    match = np.zeros(hdbmatch.shape)
    match[hdbmatch] = 1
    match[ksmatch] = 2
    match[ksmatch & hdbmatch] = 3
    colors = ["k", "b", "r", "purple"]
    
    axes[0].scatter(*geom_array.T, marker="s", s=2, color="orange")
    ls = []
    for i in range(4):
        axes[0].scatter(
            ti["gt_x"][match == i], ti["gt_z"][match == i], color=colors[i], marker="x", s=15
        )
        axes[2].scatter(
            ti["gt_x"][match == i], ti["gt_z"][match == i], color=colors[i], marker="x", s=15
        )
        l = axes[1].scatter(
            np.log(ti["gt_ptp"][match == i]), ti["gt_z"][match == i], color=colors[i], marker="x", s=15
        )
        ls.append(l)
    plt.figlegend(
        ls,
        ["no match", "our match", "ks match", "both"],
        loc="lower center",
        ncol=4, frameon=False, borderaxespad=-0.65)
    
    fig.suptitle(f"{k} template locs + final clustering localization", y=0.95)
    fig.tight_layout()
    fig.savefig(hybrid_fig_dir / "matched_scatter" / f"{k}_B_hdb.png", dpi=300)
    plt.close(fig)
    # print(rr)
    # display(ti)

# %%
