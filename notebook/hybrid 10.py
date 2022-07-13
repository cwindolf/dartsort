# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
1

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm, trange
import pandas as pd
import argparse
import spikeinterface.comparison as sc
from spikeinterface.extractors import NumpySorting
import spikeinterface.widgets as sw
from sklearn.decomposition import PCA
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist

# %%
from scipy.spatial import KDTree
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# %%
from IPython.display import display

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
    deconvolve,
)

# ap = argparse.ArgumentParser()
# ap.add_argument("subject")
# args = ap.parse_args()
# subject = args.subject

hybrid_bin_dir = Path("/share/ctn/users/ciw2107/hybrid_5min/hybrid_5min_output/")
hybrid_res_dir = Path("/share/ctn/users/ciw2107/hybrid_5min/hybrid_5min_subtraction/")
hybrid_ks_dir = Path("/share/ctn/users/ciw2107/hybrid_5min/hybrid_5min_kilosort/")
hybrid_deconv_dir = Path("/share/ctn/users/ciw2107/hybrid_5min/hybrid_5min_deconv/")
# hybrid_raw_sub_dir = Path("/share/ctn/users/ciw2107/hybrid_1min/hybrid_1min_raw_res/")
# hybrid_bin_dir = Path("/share/ctn/users/ciw2107/hybrid_1min/hybrid_1min_output/")
# hybrid_res_dir = Path("/share/ctn/users/ciw2107/hybrid_1min/hybrid_1min_subtraction/")
# hybrid_ks_dir = Path("/share/ctn/users/ciw2107/hybrid_1min/hybrid_1min_kilosort/")
# hybrid_deconv_dir = Path("/share/ctn/users/ciw2107/hybrid_1min/hybrid_1min_deconv/")
hybrid_bin_dir.exists(), hybrid_res_dir.exists(), hybrid_ks_dir.exists(), hybrid_deconv_dir.exists()


# %%
hybrid_fig_dir = Path("/share/ctn/users/ciw2107/hybrid_5min/figs_79/")
# hybrid_fig_dir = Path("/share/ctn/users/ciw2107/hybrid_1min/figs/")


hybrid_fig_dir.mkdir(exist_ok=True)
hybrid_fig_dir.exists()

# %%
# %ll {hybrid_fig_dir}

# %%
# %matplotlib inline

# %%
plt.rc("figure", dpi=200)
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=SMALL_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# %%
# subjects = [d.stem for d in hybrid_res_dir.glob("*") if d.is_dir()]
subjects = ["DY_018", "CSHL051"]
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
# %ll /share/ctn/users/ciw2107/hybrid_5min/hybrid_5min_deconv/DY_018/deconv2

# %%
1

# %%
unit_records = []
subject_records = []
spike_dfs = []
gtunit_info_by_subj = {}
reloc_results = {}

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
    match = cmp_gt_sorter.get_best_unit_match1(gt_unit)
    acc = 0
    n_spikes = 0

    if match >= 0:
        acc = cmp_gt_sorter.get_agreement_fraction(gt_unit, match)
        n_spikes = cmp_gt_sorter.get_matching_event_count(gt_unit, match)

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
    # if subject not in ("SWC_054", "ZM_2241", "NYU-21", "SWC_060", "ZFM-01592"):
    if subject not in ("DY_018", "CSHL051"):
        continue
    print(subject)
    raw_data_bin = hybrid_bin_dir / f"{subject}.ap.bin"
    assert raw_data_bin.exists()
    residual_data_bin = next((hybrid_res_dir / subject).glob("residual*bin"))
    assert residual_data_bin.exists()
    sub_h5 = next((hybrid_res_dir / subject).glob("sub*h5"))
    assert sub_h5.exists()

    hybrid_gt_h5 = hybrid_bin_dir / f"{subject}_gt.h5"
    output_dir = hybrid_res_dir / subject / "gt_comparison"

    # deconv_spike_train = np.load(
    #     hybrid_deconv_dir / subject / "spike_train.npy"
    # )
    # deconv_templates = np.load(
    #     hybrid_deconv_dir / subject / "templates.npy"
    # )
    # deconv_samples, deconv_labels = deconv_spike_train.T
    # deconv_template_mc = deconv_templates.ptp(1).argmax(1)
    # deconv_spike_index = deconvsplit_spike_index = deconv2_spike_index = np.c_[
    #     deconv_samples,
    #     deconv_template_mc[deconv_labels],
    # ]
    geom = np.load(hybrid_deconv_dir / subject / "geom.npy")
    deconv1_samples = np.load(hybrid_deconv_dir / subject / "postdeconv_split_times.npy")
    deconv1_labels = np.load(hybrid_deconv_dir / subject / "postdeconv_merge_labels.npy")
    deconv1_spike_train = np.c_[
        deconv1_samples,
        deconv1_labels,
    ]
    deconv1_templates = deconvolve.get_templates(
        raw_data_bin,
        deconv1_spike_train,          # asks for spike index but only uses first axis
        deconv1_labels,
        geom,
        n_times=121,
        n_samples=250,
        trough_offset=42,
    )
    deconv1_mc = deconv1_templates.ptp(1).argmax(1)
    deconv1_spike_index = np.c_[
        deconv1_samples,
        deconv1_mc[deconv1_labels],
    ]
    
    deconv2_samples = np.load(hybrid_deconv_dir / subject / "deconv2" / "postdeconv_split_times.npy")
    deconv2_labels = np.load(hybrid_deconv_dir / subject / "deconv2" / "postdeconv_merge_labels.npy")
    deconv2_spike_train = np.c_[
        deconv2_samples,
        deconv2_labels,
    ]
    deconv2_templates = deconvolve.get_templates(
        raw_data_bin,
        deconv2_spike_train,          # asks for spike index but only uses first axis
        deconv2_labels,
        geom,
        n_times=121,
        n_samples=250,
        trough_offset=42,
    )
    deconv2_mc = deconv2_templates.ptp(1).argmax(1)
    deconv2_spike_index = np.c_[
        deconv2_samples,
        deconv2_mc[deconv2_labels],
    ]
    #     print("no deconv yet for", subject)
    #     continue

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
        geom = geom_array = h5["geom"][:]
        end_sample = h5["end_sample"][()]
        start_sample = h5["start_sample"][()]
        recording_length = (end_sample - start_sample) / 30000
        channel_index = h5["channel_index"][:]
        n_detections = h5["spike_index"].shape[0]
        o_spike_index = h5["spike_index"][:]
        locs = h5["localizations"][:]
        z_reg = h5["z_reg"][:]
        maxptps = h5["maxptps"][:]
        tpca_mean = h5["tpca_mean"][:]
        tpca_components = h5["tpca_components"][:]
        tpca = PCA(tpca_components.shape[0])
        tpca.mean_ = tpca_mean
        tpca.components_ = tpca_components

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
    gt_labels = gt_spike_train[:, 1]
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
        n_jobs=10,
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
    sorting_deconv1, cmp_gt_deconv1 = sorting_and_cmp(deconv1_spike_train, sorting_gt, "deconv1")
    sorting_deconv2, cmp_gt_deconv2 = sorting_and_cmp(deconv2_spike_train, sorting_gt, "deconv2")
    sorting_ksall, cmp_gt_ksall = sorting_and_cmp(ksall_spike_train, sorting_gt, "ksall")
    sorting_ksgood, cmp_gt_ksgood = sorting_and_cmp(ksgood_spike_train, sorting_gt, "ksgood")

    gt_units, gt_counts = np.unique(gt_spike_train[:, 1], return_counts=True)
    units_matched_pmhdb = np.zeros(gt_units.shape, dtype=bool)
    units_matched_hdb = np.zeros(gt_units.shape, dtype=bool)
    units_matched_deconv1 = np.zeros(gt_units.shape, dtype=bool)
    units_matched_deconv2 = np.zeros(gt_units.shape, dtype=bool)
    units_matched_ksall = np.zeros(gt_units.shape, dtype=bool)
    units_matched_ksgood = np.zeros(gt_units.shape, dtype=bool)

    n_matched_hdb = 0
    n_matched_pmhdb = 0
    n_matched_ksgood = 0
    n_matched_ksall = 0
    n_matched_deconv1 = 0
    n_matched_deconv2 = 0
    
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
    n_detected_hdb, hdb_detected = unsorted_detection(
        gt_spike_index, hdb_spike_index_[hdb_labels >= 0]
    )
    n_detected_deconv1, deconv1_detected = unsorted_detection(
        gt_spike_index, deconv1_spike_index
    )
    n_detected_deconv2, deconv2_detected = unsorted_detection(
        gt_spike_index, deconv2_spike_index
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
    deconv1_matched = deconv1_detected & units_matched_deconv1[gt_spike_train[:, 1]]
    deconv2_matched = deconv2_detected & units_matched_deconv2[gt_spike_train[:, 1]]
    ksall_matched = ksall_detected & units_matched_ksall[gt_spike_train[:, 1]]
    ksgood_matched = ksgood_detected & units_matched_ksgood[gt_spike_train[:, 1]]

    this_subject_gtunit_info_by_subjs = []

    for i, (unit, tloc) in enumerate(zip(gt_units, zip(*gt_template_locs))):
        assert i == unit
        gt_ptp = templates[unit].ptp(1).max()
        hdb_match, hdb_acc, hdb_n_spikes = unit_match(unit, cmp_gt_hdb)
        pmhdb_match, pmhdb_acc, pmhdb_n_spikes = unit_match(unit, cmp_gt_pmhdb)
        deconv1_match, deconv1_acc, deconv1_n_spikes = unit_match(unit, cmp_gt_deconv1)
        deconv2_match, deconv2_acc, deconv2_n_spikes = unit_match(unit, cmp_gt_deconv2)
        print(f"{deconv2_match=} {deconv2_spike_train[:,1].max()=}")
        ksall_match, ksall_acc, ksall_n_spikes = unit_match(unit, cmp_gt_ksall)
        ksgood_match, ksgood_acc, ksgood_n_spikes = unit_match(unit, cmp_gt_ksgood)

        n_matched_hdb += hdb_n_spikes
        n_matched_pmhdb += pmhdb_n_spikes
        n_matched_deconv1 += deconv1_n_spikes
        n_matched_deconv2 += deconv2_n_spikes
        n_matched_ksall += ksall_n_spikes
        n_matched_ksgood += ksgood_n_spikes

        units_matched_pmhdb[unit] = pmhdb_match >= 0
        units_matched_hdb[unit] = hdb_match >= 0
        units_matched_deconv1[unit] = deconv1_match >= 0
        units_matched_deconv2[unit] = deconv2_match >= 0
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
                deconv1_match=deconv1_match,
                deconv2_match=deconv2_match,
                ksall_match=ksall_match,
                ksgood_match=ksgood_match,
                hdb_acc=hdb_acc,
                pmhdb_acc=pmhdb_acc,
                deconv1_acc=deconv1_acc,
                deconv2_acc=deconv2_acc,
                ksall_acc=ksall_acc,
                ksgood_acc=ksgood_acc,
                pmhdb_det=pmhdb_detected[gt_labels == unit].mean(),
                hdb_det=hdb_detected[gt_labels == unit].mean(),
                deconv1_det=deconv1_detected[gt_labels == unit].mean(),
                deconv2_det=deconv2_detected[gt_labels == unit].mean(),
                ksall_det=ksall_detected[gt_labels == unit].mean(),
                ksgood_det=ksgood_detected[gt_labels == unit].mean(),
            )
        )
        
        pmhdb_match_n_spikes = 0
        if pmhdb_match > 0:
            pmhdb_match_n_spikes = (pmhdb_spike_train[:, 1] == pmhdb_match).sum()
        
        hdb_match_n_spikes = 0
        if hdb_match > 0:
            hdb_match_n_spikes = (hdb_spike_train[:, 1] == hdb_match).sum()
        
        deconv1_match_n_spikes = 0
        if deconv1_match > 0:
            deconv1_match_n_spikes = (deconv1_spike_train[:, 1] == deconv1_match).sum()
    
        deconv2_match_n_spikes = 0
        if deconv2_match > 0:
            deconv2_match_n_spikes = (deconv2_spike_train[:, 1] == deconv2_match).sum()
        
        ksall_match_n_spikes = 0
        if ksall_match > 0:
            ksall_match_n_spikes = (ksall_spike_train[:, 1] == ksall_match).sum()

        this_subject_gtunit_info_by_subjs.append(
            dict(
                gt_ptp=gt_ptp,
                unit=unit,
                pmhdb_match=pmhdb_match,
                pmhdb_acc=pmhdb_acc,
                pmhdb_match_event_count=pmhdb_n_spikes,
                pmhdb_match_n_spikes=pmhdb_match_n_spikes,
                hdb_match=hdb_match,
                hdb_acc=hdb_acc,
                hdb_match_event_count=hdb_n_spikes,
                hdb_match_n_spikes=hdb_match_n_spikes,
                deconv1_match=deconv1_match,
                deconv1_acc=deconv1_acc,
                deconv1_match_event_count=deconv1_n_spikes,
                deconv1_match_n_spikes=deconv1_match_n_spikes,
                deconv2_match=deconv2_match,
                deconv2_acc=deconv2_acc,
                deconv2_match_event_count=deconv2_n_spikes,
                deconv2_match_n_spikes=deconv2_match_n_spikes,
                ksall_match=ksall_match,
                ksall_acc=ksall_acc,
                ksall_match_event_count=ksall_n_spikes,
                ksall_match_n_spikes=ksall_match_n_spikes,
                gt_firing_rate=gt_counts[i] / 60,
                gt_n_spikes=gt_counts[i],
                gt_x=tloc[0],
                gt_y=tloc[1],
                gt_z=tloc[3],
                gt_alpha=tloc[4],
                pmhdb_det=pmhdb_detected[gt_labels == unit].mean(),
                hdb_det=hdb_detected[gt_labels == unit].mean(),
                deconv1_det=deconv1_detected[gt_labels == unit].mean(),
                deconv2_det=deconv2_detected[gt_labels == unit].mean(),
                ksall_det=ksall_detected[gt_labels == unit].mean(),
                ksgood_det=ksgood_detected[gt_labels == unit].mean(),
            )
        )
    gtunit_info_by_subj[subject] = this_subject_gtunit_info_by_subjs

    subject_records.append(
        dict(
            n_gt=len(gt_spike_train),
            n_detections=n_detections,
            n_triage=len(original_triage),
            pct_detected=n_detected / len(gt_spike_train),
            pct_triage=n_detected_triage / len(gt_spike_train),
            pct_det_pmhdb=n_detected_pmhdb / len(gt_spike_train),
            pct_det_deconv1=n_detected_deconv1 / len(gt_spike_train),
            pct_det_deconv2=n_detected_deconv2 / len(gt_spike_train),
            pct_det_hdb=n_detected_hdb / len(gt_spike_train),
            pct_det_ksall=n_detected_ksall / len(gt_spike_train),
            pct_det_ksgood=n_detected_ksgood / len(gt_spike_train),
            pct_detmatch_pmhdb=pmhdb_matched.sum() / len(gt_spike_train),
            pct_detmatch_deconv1=deconv1_matched.sum() / len(gt_spike_train),
            pct_detmatch_deconv2=deconv2_matched.sum() / len(gt_spike_train),
            pct_detmatch_hdb=hdb_matched.sum() / len(gt_spike_train),
            pct_detmatch_ksall=ksall_matched.sum() / len(gt_spike_train),
            pct_detmatch_ksgood=ksgood_matched.sum() / len(gt_spike_train),
            pct_pmhdb=n_matched_pmhdb / len(gt_spike_train),
            pct_deconv1=n_matched_deconv1 / len(gt_spike_train),
            pct_deconv2=n_matched_deconv2 / len(gt_spike_train),
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
                triage_detected=triage_detected,
                pmhdb_detected=pmhdb_detected,
                hdb_detected=hdb_detected,
                deconv1_detected=deconv1_detected,
                deconv2_detected=deconv2_detected,
                ksall_detected=ksall_detected,
                ksgood_detected=ksgood_detected,
                subjects=np.array([subject] * len(gt_spike_index[:, 0])),
                pmhdb_matched=pmhdb_matched,
                hdb_matched=hdb_matched,
                deconv1_matched=deconv1_matched,
                deconv2_matched=deconv2_matched,
                ksall_matched=ksall_matched,
                ksgood_matched=ksgood_matched,
            )
        )
    )


# %%
subject_df = pd.DataFrame.from_records(subject_records)
subject_df

# %%
unit_df = pd.DataFrame.from_records(unit_records)
unit_df["hdb_detected"] = (unit_df["hdb_match"] >= 0).astype(int)
unit_df["pmhdb_detected"] = (unit_df["pmhdb_match"] >= 0).astype(int)
unit_df["deconv1_detected"] = (unit_df["deconv1_match"] >= 0).astype(int)
unit_df["deconv2_detected"] = (unit_df["deconv2_match"] >= 0).astype(int)
unit_df["ksall_detected"] = (unit_df["ksall_match"] >= 0).astype(int)
unit_df["ksgood_detected"] = (unit_df["ksgood_match"] >= 0).astype(int)
unit_df.head()

# %%
for (i, row), spike_df in zip(subject_df.iterrows(), spike_dfs):
    # plt.plot([spike_df[k].mean() for k in ["pmhdb_matched", "deconv_matched", "ksall_matched"]], color="red")
    # la, = plt.plot(
    #     list(row[["pct_detected", "pct_triage"]])
    #     + [
    #         spike_df[k].mean()
    #         for k in ["pmhdb_detected", "hdb_detected", "deconv_detected", "ksall_detected"]
    #     ],
    #     color="blue",
    #     label="unsorted",
    # ) 
    la, = plt.plot(
        row[["pct_detected", "pct_triage", "pct_det_pmhdb", "pct_det_hdb", "pct_det_deconv1", "pct_det_deconv2", "pct_det_ksall"]].values,
        color="b",
    ) 
    lb, = plt.plot(
        row[["pct_detected", "pct_triage", "pct_pmhdb", "pct_hdb", "pct_deconv1", "pct_deconv2", "pct_ksall"]].values,
        color="green",
    )
    # plt.plot(
    #     row[["pct_detected", "pct_triage", "pct_pmhdb", "pct_hdb", "pct_deconv", "pct_ksall"]],
    #     color="k",
    # )
plt.gca().set_xticks(range(7),["Detected", "Triage", "HDBSCAN", "Split/Merge", "Deconv1", "Deconv2", "KSall"])
plt.title("Recovery through the pipeline")
plt.legend([la, lb], ["unsorted", "sorted"], frameon=False)
plt.gcf().savefig(hybrid_fig_dir / "summary_recovery.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# thresholds = [0, 6, 9, 12, np.inf]
# thresholds = [0, 6, 9, np.inf]

# colors = plt.cm.viridis(np.array(thresholds[:3]) / 9)
# for spike_df in spike_dfs:
#     ls = []
#     for ti in range(3):
#         which = (spike_df["gt_ptp"] >= thresholds[ti]) & (spike_df["gt_ptp"] < thresholds[ti + 1])
#         thisdf = spike_df[which]
#         ls.append(plt.plot(
#             thisdf[["detected", "triage_detected", "pmhdb_detected", "hdb_detected", "deconv_detected", "ksall_detected"]].mean(axis=0),
#             color=colors[ti],
#         )[0])
#     # plt.plot(row[["pct_detected", "pct_kept", "pct_pmhdb", "pct_hdb", "pct_deconv"]], color="k")
# plt.gca().set_xticklabels(["Detect", "Triage", "HDBSCAN", "Split/Merge", "Deconv", "KSall"])
# plt.legend(ls, ["<6", "6-9", ">9"], title="GT PTP")
# plt.title("Unsorted detection % by PTP")
# plt.ylim([0, 1.1])
# plt.show()

# %%
# thresholds = [0, 6, 9, np.inf]
# colors = plt.cm.viridis(np.array(thresholds[:3]) / 9)
# for spike_df in spike_dfs:
#     ls = []
#     for ti in range(3):
#         which = (spike_df["gt_ptp"] >= thresholds[ti]) & (spike_df["gt_ptp"] < thresholds[ti + 1])
#         thisdf = spike_df[which]
#         ls.append(plt.plot(
#             thisdf[["pmhdb_matched", "hdb_matched", "deconv_matched", "ksall_matched"]].mean(axis=0),
#             color=colors[ti],
#         )[0])
#     # plt.plot(row[["pct_detected", "pct_kept", "pct_pmhdb", "pct_hdb", "pct_deconv"]], color="k")
# plt.gca().set_xticklabels(["HDBSCAN", "Split/Merge", "Deconv", "KSall"])
# # plt.legend(ls, ["<6", "6-9", "9-12", ">12"], title="GT PTP")
# plt.legend(ls, ["<6", "6-9", ">9"], title="GT PTP")
# plt.title("Sorted detection % by PTP")
# plt.ylim([0, 1.1])
# plt.show()

# %%
import gc; gc.collect()

# %%
unit_df.columns


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
    ("deconv1_acc", "Per unit Deconv1 accuracy"),
    ("deconv2_acc", "Per unit Deconv2 accuracy"),
    ("ksall_acc", "Per unit KSall accuracy"),
]):
    fig, ax = plotgistic(y=k, title=title)
    fig.savefig(hybrid_fig_dir / "ptp_v_acc" / f"{i}_{k}_ptp_v_acc.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# %%
(hybrid_fig_dir / "ptp_v_unsorted_acc").mkdir(exist_ok=True)
for i, (k, title) in enumerate([
    ("pmhdb_det", "Per unit HDBSCAN detection rate"),
    ("hdb_det", "Per unit Split/Merge detection rate"),
    ("deconv1_det", "Per unit Deconv1 detection rate"),
    ("deconv2_det", "Per unit Deconv2 detection rate"),
    ("ksall_det", "Per unit KSall detection rate"),
]):
    fig, ax = plotgistic(y=k, title=title)
    fig.savefig(hybrid_fig_dir / "ptp_v_unsorted_acc" / f"{i}_{k}_ptp_v_unsorted_acc.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)



# %%
(hybrid_fig_dir / "fr_v_acc").mkdir(exist_ok=True)
for i, (k, title) in enumerate([
    ("pmhdb_acc", "Per unit HDBSCAN accuracy"),
    ("hdb_acc", "Per unit Split/Merge accuracy"),
    ("deconv1_acc", "Per unit Deconv1 accuracy"),
    ("deconv2_acc", "Per unit Deconv2 accuracy"),
    ("ksall_acc", "Per unit KSall accuracy"),
]):
    fig, ax = plotgistic(x="gt_firing_rate", y=k, c="gt_ptp", title=title, cmap=plt.cm.viridis)
    fig.savefig(hybrid_fig_dir / "fr_v_acc" / f"{i}_{k}_fr_v_acc.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# %%
(hybrid_fig_dir / "fr_v_unsorted_acc").mkdir(exist_ok=True)
for i, (k, title) in enumerate([
    ("pmhdb_det", "Per unit HDBSCAN detection rate"),
    ("hdb_det", "Per unit Split/Merge detection rate"),
    ("deconv1_det", "Per unit Deconv1 detection rate"),
    ("deconv2_det", "Per unit Deconv2 detection rate"),
    ("ksall_det", "Per unit KSall detection rate"),
]):
    fig, ax = plotgistic(x="gt_firing_rate", y=k, c="gt_ptp", title=title, cmap=plt.cm.viridis)
    fig.savefig(hybrid_fig_dir / "fr_v_unsorted_acc" / f"{i}_{k}_fr_v_unsorted_acc.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# %%
# for k in reloc_results.keys():
#     rr = reloc_results[k]
#     ti = pd.DataFrame.from_records(gtunit_info_by_subj[k])
#     display(ti.head())

# %% tags=[]
(hybrid_fig_dir / "matched_scatter").mkdir(exist_ok=True)
for k in tqdm(reloc_results.keys()):
    rr = reloc_results[k]
    ti = pd.DataFrame.from_records(gtunit_info_by_subj[k])
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
    ea = plt.figlegend(
        ls,
        ["no match", "our match", "ks match", "both"],
        loc="lower center",
        ncol=4, frameon=False, borderaxespad=-0.65)
    
    fig.suptitle(f"{k} template locs + re-localizations", y=0.95)
    fig.tight_layout()
    fig.savefig(hybrid_fig_dir / "matched_scatter" / f"{k}_A_gt_reloc.png", dpi=300, bbox_extra_artists=[ea])
    plt.show()
    plt.close(fig)
    # print(rr)
    # display(ti)

# %%
for k in tqdm(reloc_results.keys()):
    rr = reloc_results[k]
    ti = pd.DataFrame.from_records(gtunit_info_by_subj[k])
    urelocs = rr["gt_relocalizations"]
    umaxptp = rr["gt_remaxptp"]
    ulabels = rr["gt_spike_train"][:, 1]
    hdb_labels = np.load(hybrid_res_dir / k / "labels.npy")
    sub_h5 = next((hybrid_res_dir / k).glob("sub*h5"))
    
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
    ea = plt.figlegend(
        ls,
        ["no match", "our match", "ks match", "both"],
        loc="lower center",
        ncol=4, frameon=False, borderaxespad=-10)
    
    fig.suptitle(f"{k} template locs + final clustering localization", y=1.01)
    # fig.tight_layout()
    fig.savefig(hybrid_fig_dir / "matched_scatter" / f"{k}_B_hdb.png", dpi=300)#, bbox_extra_artists=[ea])
    plt.show()
    plt.close(fig)
    # print(rr)
    # display(ti)

# %%
hybrid_res_dir

# %%
mos = """\
abc
def
hij
lmn
pqr
"""

def vis_near_gt(subject, unit, dz=100):
    rr = reloc_results[subject]
    ti = pd.DataFrame.from_records(gtunit_info_by_subj[subject])
    
    gt_x, gt_y, gt_z, gt_ptp = ti[["gt_x", "gt_y", "gt_z", "gt_ptp"]].values[unit]
    z_low = gt_z - dz
    z_high = gt_z + dz
    
    urelocs = rr["gt_relocalizations"]
    umaxptp = rr["gt_remaxptp"]
    ulabels = rr["gt_spike_train"][:, 1]   
    
    sub_h5 = next((hybrid_res_dir / subject).glob("sub*h5"))
    with h5py.File(sub_h5, "r") as h5:
        geom_array = h5["geom"][:]
        z_reg = h5["z_reg"][:]
        which = np.flatnonzero((z_reg > z_low) & (z_reg < z_high))
        z_reg = z_reg[which]
        locs = h5["localizations"][:][which]
        maxptps = h5["maxptps"][:][which]
    hdb_labels = np.load(hybrid_res_dir / subject / "labels.npy")[which]
    pmhdb_labels = np.load(hybrid_res_dir / subject / "pre_merge_split_labels.npy")[which]
    
    with h5py.File(hybrid_deconv_dir / subject / "deconv_results.h5") as h5:
        deconv_x, _, _, deconv_z = h5["localizations"][:, :4].T
        deconv_ptp = h5["maxptps"][:]
    # deconv_x = np.load(hybrid_deconv_dir / subject / "localization_results.npy")[:, 0]
    # deconv_z = np.load(hybrid_deconv_dir / subject / "z_reg.npy")
    # deconv_ptp = np.load(hybrid_deconv_dir / subject / "ptps.npy")
    deconv_labels = np.load(hybrid_deconv_dir / subject / "spike_train.npy")[:, 1]
    deconvmerge_labels = np.load(hybrid_deconv_dir / subject / "postdeconv_merge_labels.npy")

    deconv_match = ti["deconv_acc"].values > 0
    deconvmerge_match = ti["deconvmerge_acc"].values > 0
    hdbmatch = ti["hdb_acc"].values > 0
    ksmatch = ti["ksall_acc"].values > 0
    match = np.zeros(hdbmatch.shape)
    match[deconvmerge_match] = 1
    match[ksmatch] = 2
    match[ksmatch & deconvmerge_match] = 3
    colors = ["k", "b", "r", "purple"]
    
    this_match = ti["deconvmerge_match"].values[unit]
        
    fig, axes = plt.subplot_mosaic(
        mos,
        gridspec_kw=dict(wspace=0.05, hspace=0.15),
        figsize=(6, 11),
    )
    
    cluster_viz_index.array_scatter(
        ulabels,
        geom_array,
        urelocs[:, 0],
        urelocs[:, 3],
        umaxptp,
        annotate=True,
        zlim=(z_low, z_high),
        axes=[axes[k] for k in "abc"],
    )
    
    cluster_viz_index.array_scatter(
        pmhdb_labels,
        geom_array,
        locs[:, 0],
        z_reg,
        maxptps,
        annotate=True,
        zlim=(z_low, z_high),
        axes=[axes[k] for k in "def"],
    )
    
    cluster_viz_index.array_scatter(
        hdb_labels,
        geom_array,
        locs[:, 0],
        z_reg,
        maxptps,
        annotate=True,
        zlim=(z_low, z_high),
        axes=[axes[k] for k in "hij"],
    )
    
    cluster_viz_index.array_scatter(
        deconv_labels,
        geom_array,
        deconv_x,
        deconv_z,
        deconv_ptp,
        annotate=True,
        zlim=(z_low, z_high),
        axes=[axes[k] for k in "lmn"],
    )
    
    cluster_viz_index.array_scatter(
        deconvmerge_labels,
        geom_array,
        deconv_x,
        deconv_z,
        deconv_ptp,
        annotate=True,
        zlim=(z_low, z_high),
        axes=[axes[k] for k in "pqr"],
    )
    
    for axs, key, name in zip(
        ["abc", "def", "hij", "lmn", "pqr"],
        ["deconv", "pmhdb", "hdb", "deconv", "deconvmerge"],
        ["GT reloc", "HDBSCAN", "HDBSCAN+Split/Merge", "Deconv", "Deconv+Split/Merge"],
    ):
        hdbmatch = ti[f"{key}_acc"].values > 0
        hdbmatch_unit = int(ti[f"{key}_match"][unit])
        ksmatch = ti["ksall_acc"].values > 0
        match = np.zeros(hdbmatch.shape)
        match[hdbmatch] = 1
        match[ksmatch] = 2
        match[ksmatch & hdbmatch] = 3
        whicht = (ti["gt_z"].values > z_low) & (ti["gt_z"].values < z_high)
        colors = ["k", "b", "r", "purple"]
        
        if hdbmatch[unit] and name != "GT reloc":
            axes[axs[1]].set_title(f"{name}, matched by unit {hdbmatch_unit}")
        elif name == "GT reloc":
            axes[axs[1]].set_title(name)
        else:
            axes[axs[1]].set_title(f"{name}, unmatched")

        ls = []
        for i in range(4):
            these = (match == i) & whicht
            for k in (axs[0], axs[2]):
                axes[k].scatter(
                    ti["gt_x"][these], ti["gt_z"][these], color="w", marker=".", s=25
                )
                axes[k].scatter(
                    ti["gt_x"][these], ti["gt_z"][these], color=colors[i], marker="x", s=15
                )
            l = axes[axs[1]].scatter(
                np.log(ti["gt_ptp"][these]), ti["gt_z"][these], color="w", marker=".", s=25
            )
            l = axes[axs[1]].scatter(
                np.log(ti["gt_ptp"][these]), ti["gt_z"][these], color=colors[i], marker="x", s=15
            )
            ls.append(l)
    ea = plt.figlegend(
        ls,
        ["no match", "our match", "ks match", "both"],
        loc="lower center",
        ncol=4, frameon=False, borderaxespad=1)
    
    for ka, kb in ["ad", "be", "cf", "dh", "ei", "fj", "hl", "im", "jn", "lp", "mq", "nr"]:
        axes[ka].get_shared_x_axes().join(axes[ka], axes[kb])
        axes[ka].set_xticks([])
        axes[ka].set_xlabel("")
    for k in "bcefijmnqr":
        axes[k].set_yticks([])
    matchstr = "unmatched in our sort"
    if this_match >= 0:
        matchstr = f"our deconv+split/merge match: {int(this_match)}"
    fig.suptitle(f"{subject}: GT unit {unit}, {matchstr}", fontsize=MEDIUM_SIZE, y=0.95)

    we_matched = hdbmatch[unit]
    ks_matched = ksmatch[unit]
    return fig, axes, we_matched, ks_matched, gt_ptp


# %%
vis_near_gt("DY_018", 30)

# %% tags=[]
# (hybrid_fig_dir / "match_scatter_zoom").mkdir(exist_ok=True)
(hybrid_fig_dir / "match_scatter_zoom_by_ptp").mkdir(exist_ok=True)
# prefixes = ["0_nomatch_", "1_ksmatch", "1_ksmatch"
def genjobs(k):
    # (hybrid_fig_dir / "match_scatter_zoom" / k).mkdir(exist_ok=True)
    # (hybrid_fig_dir / "match_scatter_zoom_by_ptp" / k).mkdir(exist_ok=True)
    for unit in trange(len(gtunit_info_by_subj[k])):
        # late binding closures be careful!
        def job(u=unit):
            print(k, unit, u, hybrid_fig_dir)
            fig, axes, we_matched, ks_matched, gt_ptp = vis_near_gt(k, u)
            prefix = "0_nomatch_"
            if ks_matched and not we_matched:
                prefix = "1_ksmatch"
            if not ks_matched and we_matched:
                prefix = "2_ourmatch"
            if ks_matched and we_matched:
                prefix = "3_bothmatch"
            # fig.savefig(hybrid_fig_dir / "match_scatter_zoom" / k / f"{prefix}_{u}.png", dpi=300)
            fig.savefig(hybrid_fig_dir / "match_scatter_zoom_by_ptp" / f"{prefix}_gtptp_{gt_ptp:05.2f}_{k}_{u:02d}.png", dpi=300)
            plt.close(fig)
            return k, u, hybrid_fig_dir / "match_scatter_zoom_by_ptp" / f"{prefix}_gtptp_{gt_ptp:05.2f}_{k}_{u:02d}.png"
        yield job

from joblib import Parallel, delayed

jobs = []
for k in tqdm(reloc_results.keys()):
    jobs.extend(delayed(job)() for job in genjobs(k))

for res in Parallel(14)(tqdm(jobs)):
    print(res)

# %%
1

# %%
res


# %%
def plot_venns(
    subject,
    gt_unit,
    gt_spike_train,
    gt_spike_index,
    ks_spike_train,
    ks_spike_index,
    deconv_spike_train,
    deconv_spike_index,
    gt_template_locs,
    ks_template_locs,
    deconv_template_locs,
    gt_cluster_fc,
    gt_cluster_mc,
    ks_cluster_fc,
    ks_cluster_mc,
    deconv_cluster_fc,
    deconv_cluster_mc,
    geom,
    deconv_templates,
    gt_templates,
    which_sort="deconv2",
):
    out_bin = hybrid_bin_dir / f"{subject}.ap.bin"
    unit_info = gtunit_info_by_subj[subject][gt_unit]
    gt_ptp = unit_info["gt_ptp"]
    
    print(f"{deconv_spike_train[:,1].max()=} {deconv_template_locs.shape=} {deconv_cluster_fc.shape=} {deconv_cluster_mc.shape=} {deconv_templates.shape=}")

    # get gt stuff
    gt_loc = gt_template_locs[gt_unit]
    unit_gt_spike_train = gt_spike_train[gt_spike_train[:, 1] == gt_unit, 0]
    
    # get ks stuff
    ks_match = int(unit_info["ksall_match"])
    print(ks_match)
    ks_matched = ks_match >= 0
    if not ks_matched:
        print(gt_loc.shape, ks_template_locs.shape)
        ks_match = np.argmin(cdist(gt_loc[None], ks_template_locs).squeeze())
    
    # ks_str = f"KSall match {ks_match}"
    # if ks_matched:
    #     ks_match_spike_train = ks_spike_train[ks_spike_train[:, 1] == ks_match, 0]
    # if ks_matched and not (hybrid_fig_dir / "gt_v_ks" / subject / f"gt{gt_unit}.png").exists():
    #     fig = cluster_viz.plot_agreement_venn_better(
    #         ks_match,
    #         gt_unit,
    #         ks_match_spike_train,
    #         unit_gt_spike_train,
    #         np.full(len(ks_match_spike_train), ks_cluster_fc[ks_match]),
    #         np.full(len(ks_match_spike_train), ks_cluster_mc[ks_match]),
    #         np.full(len(unit_gt_spike_train), gt_cluster_fc[gt_unit]),
    #         np.full(len(unit_gt_spike_train), gt_cluster_mc[gt_unit]),
    #         geom,
    #         out_bin,
    #         dict(zip(range(len(ks_template_locs[:, 2])), ks_template_locs[:, 2])),
    #         dict(zip(range(len(gt_template_locs[:, 2])), gt_template_locs[:, 2])),
    #         ks_spike_index,
    #         gt_spike_index,
    #         ks_spike_train[:, 1],
    #         gt_spike_train[:, 1],
    #         scale=7,
    #         sorting1_name="KSall",
    #         sorting2_name="GT",
    #         num_channels=40,
    #         num_spikes_plot=100,
    #         t_range=(30, 90),
    #         num_rows=3,
    #         alpha=0.1,
    #         delta_frames=12,
    #         num_close_clusters=5,
    #     )
    #     fig.suptitle(f"GT unit {gt_unit}. {ks_str}")
    #     # plt.show()
    #     fig.savefig(hybrid_fig_dir / "gt_v_ks" / subject / f"gt{gt_unit}.png")
    #     plt.close(fig)
    
    # get our stuff
    deconv_match = int(unit_info["deconv2_match"])
    print(deconv_match)
    deconv_matched = deconv_match >= 0
    if not deconv_matched:
        print(gt_loc.shape, deconv_template_locs.shape)
        deconv_match = np.argmin(cdist(gt_loc[None], deconv_template_locs).squeeze())
        deconv_matched = False

    
    deconv_str = f"{which_sort} match {deconv_match}"
    if deconv_matched:
        deconv_match_spike_train = deconv_spike_train[deconv_spike_train[:, 1] == deconv_match, 0]
    else:
        deconv_match_spike_train = deconv_spike_train[deconv_spike_train[:, 1] == deconv_match, 0]
        deconv_str = f"No deconv match. Using closest unit: {deconv_match}."
    
    if True or not (hybrid_fig_dir / "venn_by_ptp" / f"ptp{gt_ptp:05.2f}_{subject}_unit{gt_unit:02d}.png").exists():
        print("hi")
        # fig = cluster_viz.plot_agreement_venn_better(
        #     deconv_match,
        #     gt_unit,
        #     deconv_match_spike_train,
        #     unit_gt_spike_train,
        #     np.full(max(1, len(deconv_match_spike_train)), deconv_cluster_fc[deconv_match]),
        #     np.full(max(1, len(deconv_match_spike_train)), deconv_cluster_mc[deconv_match]),
        #     np.full(max(1, len(unit_gt_spike_train)), gt_cluster_fc[gt_unit]),
        #     np.full(max(1, len(unit_gt_spike_train)), gt_cluster_mc[gt_unit]),
        #     geom,
        #     out_bin,
        #     dict(zip(range(len(deconv_template_locs[:, 2])), deconv_template_locs[:, 2])),
        #     dict(zip(range(len(gt_template_locs[:, 2])), gt_template_locs[:, 2])),
        #     deconv_spike_index,
        #     gt_spike_index,
        #     deconv_spike_train[:, 1],
        #     gt_spike_train[:, 1],
        #     scale=7,
        #     sorting1_name="Deconv",
        #     sorting2_name="GT",
        #     num_channels=40,
        #     num_spikes_plot=100,
        #     t_range=(30, 90),
        #     num_rows=3,
        #     alpha=0.1,
        #     delta_frames=12,
        #     num_close_clusters=5,
        # )
        print("deconv match, gt unit", deconv_match, gt_unit)
        print("nspikes", len(deconv_match_spike_train), len(unit_gt_spike_train), flush=True)
        fig = cluster_viz.diagnostic_plots(
            deconv_match,
            gt_unit,
            deconv_match_spike_train,
            unit_gt_spike_train,
            deconv_templates,
            gt_templates,
            np.full(max(1, len(deconv_match_spike_train)), deconv_cluster_fc[deconv_match]),
            np.full(max(1, len(deconv_match_spike_train)), deconv_cluster_mc[deconv_match]),
            np.full(max(1, len(unit_gt_spike_train)), gt_cluster_fc[gt_unit]),
            np.full(max(1, len(unit_gt_spike_train)), gt_cluster_mc[gt_unit]),
            geom,
            out_bin,
            dict(zip(range(len(deconv_template_locs[:, 2])), deconv_template_locs[:, 2])),
            dict(zip(range(len(gt_template_locs[:, 2])), gt_template_locs[:, 2])),
            deconv_spike_index,
            gt_spike_index,
            deconv_spike_train[:, 1],
            gt_spike_train[:, 1],
            scale=7,
            sorting1_name=f"{which_sort}",
            sorting2_name="GT",
            num_channels=40,
            num_spikes_plot=100,
            t_range=(30, 90),
            num_rows=3,
            alpha=0.1,
            delta_frames=12,
            num_close_clusters=5,
        )
        
        
        
        fig.suptitle(f"GT unit {gt_unit}. {deconv_str}")
        # fig.savefig(hybrid_fig_dir / "gt_v_deconv" / subject / f"gt{gt_unit}.png")
        fig.savefig(hybrid_fig_dir / "venn_by_ptp" / f"ptp{gt_ptp:05.2f}_{subject}_unit{gt_unit:02d}.png")
        plt.close(fig)
    
    # deconv_match = unit_info["ks_match"]
    # ks_which = np.flatnonzero(ks_spike_train[:, 1] == ks_match)
    # deconv_which = np.flatnonzero(deconv_spike_train[:, 1] == deconv_match)
    


# %% tags=[]
# (hybrid_fig_dir / "gt_v_deconv").mkdir(exist_ok=True)
# (hybrid_fig_dir / "gt_v_ks").mkdir(exist_ok=True)
(hybrid_fig_dir / "venn_by_ptp").mkdir(exist_ok=True)

# prefixes = ["0_nomatch_", "1_ksmatch", "1_ksmatch"
def job(subject):
    geom = np.load(hybrid_deconv_dir / subject / "geom.npy")
    
    # (hybrid_fig_dir / "gt_v_deconv" / subject).mkdir(exist_ok=True)
    # (hybrid_fig_dir / "gt_v_ks" / subject).mkdir(exist_ok=True)

    ti = pd.DataFrame.from_records(gtunit_info_by_subj[subject])
    gt_template_locs = ti[["gt_x", "gt_y", "gt_z"]].values
    # gt_template_locs = np.c_[gt_x, gt_y, gt
    with h5py.File(hybrid_bin_dir / f"{subject}_gt.h5", "r") as gt_h5:
        gt_spike_train = gt_h5["spike_train"][:]
        gt_spike_index = gt_h5["spike_index"][:]
        gt_templates = gt_h5["templates"][:]
        tmax = gt_spike_index[:,0].max()
        gtwh = (gt_spike_index[:,0] > 60) & (gt_spike_index[:,0] < tmax - 60)
        gt_spike_train = gt_spike_train[gtwh]
        gt_spike_index = gt_spike_index[gtwh]
        tmax = gt_spike_index[:,0].max()
    gt_cluster_maxchans = gt_templates.ptp(1).argmax(1)
    gt_cluster_firstchans = np.maximum(0, gt_cluster_maxchans - 10)
    
    ks_templates = np.load(hybrid_ks_dir / subject / "templates.npy")
    ks_templates[~np.isfinite(ks_templates)] = 0
    ks_labels = np.load(hybrid_ks_dir / subject / "spike_clusters.npy").astype(int).squeeze()
    ks_samples = np.load(hybrid_ks_dir / subject / "spike_times.npy").astype(int).squeeze()
    ks_tptp = ks_templates.ptp(1)
    ks_cluster_maxchans = ks_tptp.argmax(1)
    ks_cluster_firstchans = np.maximum(0, ks_cluster_maxchans - 10)
    ks_template_locs = localize_index.localize_ptps_index(
        ks_tptp,
        geom,
        ks_cluster_maxchans,
        np.stack([np.arange(len(geom_array))] * len(geom_array), axis=0),
        n_channels=20,
        n_workers=None,
        pbar=True,
    )
    ks_template_locs = np.c_[ks_template_locs[0], ks_template_locs[1], ks_template_locs[3]]
    ks_spike_train = np.c_[ks_samples, ks_labels]
    ks_spike_index = np.c_[ks_samples, ks_cluster_maxchans[ks_labels]]
    kswh = (ks_samples > 60) & (ks_samples < tmax - 60)
    ks_spike_train = ks_spike_train[kswh]
    ks_spike_index = ks_spike_index[kswh]
    
    # deconv_templates = np.load(hybrid_deconv_dir / subject / "templates.npy")
    deconv_samples = np.load(hybrid_deconv_dir / subject / "deconv2" / "postdeconv_split_times.npy")
    deconv_labels = np.load(hybrid_deconv_dir / subject / "deconv2" / "postdeconv_merge_labels.npy")
    deconv_spike_train = np.c_[deconv_samples, deconv_labels]
    deconv_templates = deconvolve.get_templates(
        hybrid_bin_dir / f"{subject}.ap.bin",
        deconv_spike_train,          # asks for spike index but only uses first axis
        deconv_labels,
        geom,
        n_times=121,
        n_samples=250,
        trough_offset=42,
    )
    print("max deconv label, template shape", deconv_labels.max(), deconv_templates.shape)
    
    deconv_tptp = deconv_templates.ptp(1)
    deconv_cluster_maxchans = deconv_tptp.argmax(1)
    deconv_cluster_firstchans = np.maximum(0, deconv_cluster_maxchans - 10)
    deconv_template_locs = localize_index.localize_ptps_index(
        deconv_tptp,
        geom,
        deconv_cluster_maxchans,
        np.stack([np.arange(len(geom_array))] * len(geom_array), axis=0),
        n_channels=20,
        n_workers=None,
        pbar=True,
    )
    deconv_template_locs = np.c_[deconv_template_locs[0], deconv_template_locs[1], deconv_template_locs[3]]
    deconv_spike_index = np.c_[deconv_samples, deconv_cluster_maxchans[deconv_labels]]
    deconvwh = (deconv_samples > 60) & (deconv_samples < tmax - 60)
    deconv_spike_train = deconv_spike_train[deconvwh]
    deconv_spike_index = deconv_spike_index[deconvwh]
    
    for unit in trange(len(gtunit_info_by_subj[subject])):
    # for unit in [18]:
        plot_venns(
            subject,
            unit,
            gt_spike_train,
            gt_spike_index,
            ks_spike_train,
            ks_spike_index,
            deconv_spike_train,
            deconv_spike_index,
            gt_template_locs,
            ks_template_locs,
            deconv_template_locs,
            gt_cluster_firstchans,
            gt_cluster_maxchans,
            ks_cluster_firstchans,
            ks_cluster_maxchans,
            deconv_cluster_firstchans,
            deconv_cluster_maxchans,
            geom,
            deconv_templates,
            gt_templates,
        )
    return subject

from joblib import Parallel, delayed

# job(k)
jobs = []
for k in tqdm(reloc_results.keys()):
    jobs.append(delayed(job)(k))

for res in Parallel(1)(tqdm(jobs)):
    print(res)
# print("x", flush=True)
# fig = job("DY_018")
# print("z", flush=True)
# plt.show()


# %%
1

# %%
a, b = fig
a.size, b.size

# %%
a.min(), a.max()

# %%
b.min(), b.max()

# %%
np.isin(a, b).mean()

# %%
np.unique(fig).size

# %%
np.diff(fig)

# %%
1

# %%

# %%

# %%

# %%
