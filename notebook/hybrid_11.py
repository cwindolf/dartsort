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
1

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import time

# %%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm, trange
import pandas as pd
from IPython.display import display, Image
from joblib import Parallel, delayed
import os
import colorcet as cc
import pickle
from matplotlib import colors

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# %%
from spike_psvae import (
    subtract,
    cluster_utils,
    cluster_viz,
    cluster_viz_index,
    grab_and_localize,
    pyks_ccg,
    denoise,
)
from spike_psvae.hybrid_analysis import (
    Sorting,
    HybridComparison,
    plotgistic,
    make_diagnostic_plot,
    array_scatter_vs,
    near_gt_scatter_vs,
    density_near_gt,
    plot_agreement_matrix,
    gtunit_resid_study,
)


# %%
# %matplotlib inline
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
hybrid_bin_dir = Path("/mnt/3TB/charlie/hybrid_5min/hybrid_5min_output/")
hybrid_res_dir = Path("/mnt/3TB/charlie/hybrid_5min/hybrid_5min_subtraction/")
hybrid_ks_dir = Path("/mnt/3TB/charlie/hybrid_5min/hybrid_5min_kilosort/")
hybrid_deconv1_dir = Path("/mnt/3TB/charlie/hybrid_5min/hybrid_5min_deconv1/")
hybrid_deconv2_dir = Path("/mnt/3TB/charlie/hybrid_5min/hybrid_5min_deconv2/")
hybrid_deconv3_dir = Path("/mnt/3TB/charlie/hybrid_5min/hybrid_5min_deconv3/")
assert all((hybrid_bin_dir.exists(), hybrid_res_dir.exists(), hybrid_ks_dir.exists(), hybrid_deconv1_dir.exists(), hybrid_deconv2_dir.exists()))


# %%

# %%
subjects = ("CSHL051", "DY_018")
subjects = ("CSHL051",)

# %%
while not all(
    (hybrid_deconv2_dir / subj / "postdeconv_cleaned_templates.npy").exists()
    for subj in subjects
):
    print(".", end="")
    time.sleep(5 * 60)

# %%

# %% [raw]
#

# %%
hybrid_fig_dir = Path("/mnt/3TB/charlie/hybrid_5min/figs_10_6_newdn_log10")
hybrid_fig_dir.mkdir(exist_ok=True)

# %%
# %rm -rf {hybrid_fig_dir}/*

# %%
hybrid_fig_cache_dir = Path("/tmp/hybrid_cache")
hybrid_fig_cache_dir.mkdir(exist_ok=True)

# %%
# # %rm -rf {hybrid_fig_cache_dir}/*

# %%
# !rsync -avP /mnt/3TB/charlie/hybrid_5min/hybrid_5min_output /tmp/

# %%
hybrid_bin_dir = Path("/tmp/hybrid_5min_output")

# %% tags=[]
# load gt sortings
gt_sortings = {}
for subject in tqdm(subjects):
    print(subject)
    hybrid_gt_h5 = hybrid_bin_dir / f"{subject}_gt.h5"
    raw_data_bin = hybrid_bin_dir / f"{subject}.ap.bin"
    
    sub_h5 = next((hybrid_res_dir / subject).glob("sub*h5"))
    with h5py.File(sub_h5, "r") as h5:
        tpca = subtract.tpca_from_h5(h5)
    
    with h5py.File(hybrid_gt_h5, "r") as gt_h5:
        gt_spike_train = gt_h5["spike_train"][:]
        gt_spike_index = gt_h5["spike_index"][:]
        gt_templates = gt_h5["templates"][:]
        geom = gt_h5["geom"][:]
    
    gt_relocalizations, gt_remaxptp = grab_and_localize.grab_and_localize(
        gt_spike_index,
        raw_data_bin,
        geom,
        loc_radius=100,
        nn_denoise=True,
        enforce_decrease=True,
        tpca=tpca,
        chunk_size=30_000,
        n_jobs=10,
    )
    gt_xzptp = np.c_[
        gt_relocalizations[:, 0],
        gt_relocalizations[:, 3],
        gt_remaxptp,
    ]

    gt_sortings[subject] = Sorting(
        raw_data_bin,
        geom,
        gt_spike_train[:, 0],
        gt_spike_train[:, 1],
        "GT",
        spike_maxchans=gt_spike_index[:, 1],
        templates=gt_templates,
        cache_dir= hybrid_fig_cache_dir / subject,
        spike_xzptp=gt_xzptp,
    )

# %%
# we will populate these dicts
# this cell is for caching.
hybrid_sortings = {}
hybrid_comparisons = {}
for subject in tqdm(subjects):
    print(subject)
    if subject not in hybrid_comparisons:
        hybrid_sortings[subject] = {}
        hybrid_comparisons[subject] = {}

# %%
deconv2 = True
# deconv2 = False

# %%

# %%
for s in subjects:
    print(s)
    print(hybrid_comparisons[subject].keys())
    # del hybrid_comparisons[subject]["Deconv2"]
    # print(s, hybrid_comparisons[subject]["Deconv2-Cleaned"].new_sorting.cleaned_templates is None)

# %% tags=[]
for subject in tqdm(subjects):
    print(subject)
    print("-" * len(subject))

    hybrid_gt_h5 = hybrid_bin_dir / f"{subject}_gt.h5"
    raw_data_bin = hybrid_bin_dir / f"{subject}.ap.bin"
    sub_h5 = next((hybrid_res_dir / subject).glob("sub*h5"))
    
    # unsorted comparison with detection
    name = "Detection"
    if name not in hybrid_comparisons[subject]:
        print("//", name)
        with h5py.File(sub_h5, "r") as h5:
            det_spike_index = h5["spike_index"][:]
            x = h5["localizations"][:, 0]
            z_reg = h5["z_reg"][:]
            maxptps = h5["maxptps"][:]
            det_xzptp = np.c_[x, z_reg, maxptps]
        hybrid_sortings[subject][name] = Sorting(
            raw_data_bin,
            geom,
            det_spike_index[:, 0],
            np.zeros_like(det_spike_index[:, 0]),
            name,
            spike_maxchans=det_spike_index[:, 1],
            unsorted=True,
            spike_xzptp=det_xzptp,
            cache_dir=hybrid_fig_cache_dir / subject,
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )
    
    # original clustering
    name = "Cluster"
    if name not in hybrid_comparisons[subject]:
        print("//", name)
        cluster_spike_index = np.load(
            hybrid_res_dir / subject / "pre_merge_split_aligned_spike_index.npy"
        )
        cluster_labels = np.load(
            hybrid_res_dir / subject / "pre_merge_split_labels.npy"
        )
        hybrid_sortings[subject][name] = Sorting(
            raw_data_bin,
            geom,
            cluster_spike_index[:, 0],
            cluster_labels,
            name,
            spike_maxchans=cluster_spike_index[:, 1],
            spike_xzptp=det_xzptp,
            cache_dir=hybrid_fig_cache_dir / subject,
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )
    
    # original clustering -> split/merge
    name = "Split/Merge"
    if name not in hybrid_comparisons[subject]:
        print("//", name)
        splitmerge_spike_index = np.load(
            hybrid_res_dir / subject / "aligned_spike_index.npy"
        )
        splitmerge_labels = np.load(hybrid_res_dir / subject / "labels.npy")
        hybrid_sortings[subject][name] = Sorting(
            raw_data_bin,
            geom,
            splitmerge_spike_index[:, 0],
            splitmerge_labels,
            name,
            spike_maxchans=splitmerge_spike_index[:, 1],
            spike_xzptp=det_xzptp,
            cache_dir= hybrid_fig_cache_dir / subject,
            do_cleaned_templates=True,
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )
    
    # deconv1
    name = "Deconv1"
    if name not in hybrid_comparisons[subject]:
        print("//", name)
        orig_deconv1_st = np.load(hybrid_deconv1_dir / subject / "spike_train.npy")
        with h5py.File(hybrid_deconv1_dir / subject / "deconv_results.h5") as h5:
            locs = h5["localizations"][:]
            deconv1_xzptp = np.c_[locs[:, 0], locs[:, 3], h5["maxptps"][:]]
        hybrid_sortings[subject][name] = Sorting(
            raw_data_bin,
            geom,
            orig_deconv1_st[:, 0],
            cluster_utils.make_labels_contiguous(orig_deconv1_st[:, 1]),
            name,
            spike_xzptp=deconv1_xzptp,
            cache_dir=hybrid_fig_cache_dir / subject,
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )
    
    name = "Deconv1-Split"
    if name not in hybrid_comparisons[subject]:
        print("//", name)
        samples = np.load(hybrid_deconv1_dir / subject / "postdeconv_split_times.npy")
        labels = np.load(hybrid_deconv1_dir / subject / "postdeconv_split_labels.npy")
        order = np.load(hybrid_deconv1_dir / subject / "postdeconv_split_order.npy")
        templates = np.load(hybrid_deconv1_dir / subject / "postdeconv_split_templates.npy")
        with h5py.File(hybrid_deconv1_dir / subject / "deconv_results.h5") as h5:
            locs = h5["localizations"][:]
            deconv1_xzptp = np.c_[locs[:, 0], locs[:, 3], h5["maxptps"][:]]
        hybrid_sortings[subject][name] = Sorting(
            raw_data_bin,
            geom,
            samples,
            labels,
            name,
            templates=templates,
            spike_xzptp=deconv1_xzptp[order],
            cache_dir=hybrid_fig_cache_dir / subject,
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )
    
    name = "Deconv1-SplitMerge"
    if name not in hybrid_comparisons[subject]:
        print("//", name)
        samples = np.load(hybrid_deconv1_dir / subject / "postdeconv_merge_times.npy")
        labels = np.load(hybrid_deconv1_dir / subject / "postdeconv_merge_labels.npy")
        order = np.load(hybrid_deconv1_dir / subject / "postdeconv_merge_order.npy")
        templates = np.load(hybrid_deconv1_dir / subject / "postdeconv_merge_templates.npy")
        with h5py.File(hybrid_deconv1_dir / subject / "deconv_results.h5") as h5:
            locs = h5["localizations"][:]
            deconv1_xzptp = np.c_[locs[:, 0], locs[:, 3], h5["maxptps"][:]]
        hybrid_sortings[subject][name] = Sorting(
            raw_data_bin,
            geom,
            samples,
            labels,
            name,
            templates=templates,
            spike_xzptp=deconv1_xzptp[order],
            cache_dir= hybrid_fig_cache_dir / subject,
            do_cleaned_templates=True,
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )
    
    # deconv2
    name = "Deconv2"
    if deconv2 and name not in hybrid_comparisons[subject]:
        print("//", name)
        deconv2_st = np.load(hybrid_deconv2_dir / subject / "spike_train.npy")
        with h5py.File(hybrid_deconv2_dir / subject / "deconv_results.h5") as h5:
            locs = h5["localizations"][:]
            deconv2_xzptp = np.c_[locs[:, 0], locs[:, 3], h5["maxptps"][:]]
        hybrid_sortings[subject][name] = Sorting(
            raw_data_bin,
            geom,
            deconv2_st[:, 0],
            cluster_utils.make_labels_contiguous(deconv2_st[:, 1]),
            name,
            spike_xzptp=deconv2_xzptp,
            cache_dir=hybrid_fig_cache_dir / subject,
            do_cleaned_templates=True,
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )
    
    # for lambd in (0.1, 0.01, 0.001):
    #     name = f"Deconv2-Lambda{lambd}"
    #     if deconv2 and name not in hybrid_comparisons[subject]:
    #         print("//", name)
    #         deconv2_st = np.load(hybrid_deconv_dir / subject / f"deconv2_lambd{lambd}_allowed1.0/spike_train.npy")
    #         # with h5py.File(hybrid_deconv2_dir / subject / "deconv_results.h5") as h5:
    #             # locs = h5["localizations"][:]
    #             # deconv2_xzptp = np.c_[locs[:, 0], locs[:, 3], h5["maxptps"][:]]
    #         hybrid_sortings[subject][name] = Sorting(
    #             raw_data_bin,
    #             geom,
    #             deconv2_st[:, 0],
    #             cluster_utils.make_labels_contiguous(deconv2_st[:, 1]),
    #             name,
    #             # spike_xzptp=deconv2_xzptp,
    #             cache_dir= hybrid_fig_cache_dir / subject,
    #         )
    #         hybrid_comparisons[subject][name] = HybridComparison(
    #             gt_sortings[subject], hybrid_sortings[subject][name], geom
    #         )
    
    # name = "Deconv2-CleanBig"
    # if deconv2 and name not in hybrid_comparisons[subject]:
    #     print("//", name)
    #     samples = np.load(hybrid_deconv2_dir / subject / "postdeconv_cleanbig_times.npy")
    #     labels = np.load(hybrid_deconv2_dir / subject / "postdeconv_cleanbig_labels.npy")
    #     order = np.load(hybrid_deconv2_dir / subject / "postdeconv_cleanbig_order.npy")
    #     templates = np.load(hybrid_deconv2_dir / subject / "postdeconv_cleanbig_templates.npy")
    #     hybrid_sortings[subject][name] = Sorting(
    #         raw_data_bin,
    #         geom,
    #         samples,
    #         labels,
    #         name,
    #         templates=templates,
    #         spike_xzptp=deconv2_xzptp[order],
    #         cache_dir= hybrid_fig_cache_dir / subject,
    #         do_cleaned_templates=True,
    #     )
    #     hybrid_comparisons[subject][name] = HybridComparison(
    #         gt_sortings[subject], hybrid_sortings[subject][name], geom
    #     )
    
    name = "Deconv2-Cleaned"
    if deconv2 and name not in hybrid_comparisons[subject]:
        print("//", name)
        samples = np.load(hybrid_deconv2_dir / subject / "postdeconv_cleaned_times.npy")
        labels = np.load(hybrid_deconv2_dir / subject / "postdeconv_cleaned_labels.npy")
        order = np.load(hybrid_deconv2_dir / subject / "postdeconv_cleaned_order.npy")
        templates = np.load(hybrid_deconv2_dir / subject / "postdeconv_cleaned_templates.npy")
        hybrid_sortings[subject][name] = Sorting(
            raw_data_bin,
            geom,
            samples,
            labels,
            name,
            templates=templates,
            spike_xzptp=deconv2_xzptp[order],
            cache_dir= hybrid_fig_cache_dir / subject,
            do_cleaned_templates=True,
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )
        
    
    name = "Deconv3"
    if deconv2 and name not in hybrid_comparisons[subject]:
        print("//", name)
        deconv3_st = np.load(hybrid_deconv3_dir / subject / "spike_train.npy")
        with h5py.File(hybrid_deconv3_dir / subject / "deconv_results.h5") as h5:
            locs = h5["localizations"][:]
            deconv3_xzptp = np.c_[locs[:, 0], locs[:, 3], h5["maxptps"][:]]
        hybrid_sortings[subject][name] = Sorting(
            raw_data_bin,
            geom,
            deconv3_st[:, 0],
            cluster_utils.make_labels_contiguous(deconv3_st[:, 1]),
            name,
            spike_xzptp=deconv3_xzptp,
            cache_dir=hybrid_fig_cache_dir / subject,
            do_cleaned_templates=True,
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )
    
    
    name = "Deconv3-Reassign"
    if deconv2 and name not in hybrid_comparisons[subject]:
        print("//", name)
        samples, labels = np.load(hybrid_deconv3_dir / subject / "spike_train.npy").T
        reassign = np.load(hybrid_deconv3_dir / subject / "reassignment.npy")
        reassign = cluster_utils.make_labels_contiguous(reassign.astype(int))
        hybrid_sortings[subject][name] = Sorting(
            raw_data_bin,
            geom,
            samples,
            reassign,
            name,
            # templates=templates,
            spike_xzptp=deconv3_xzptp,
            cache_dir= hybrid_fig_cache_dir / subject,
            do_cleaned_templates=True,
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )
    
    # KSall
    name = "KSAll"
    if name not in hybrid_comparisons[subject]:
        print("//", name)
        ks_samples = np.load(hybrid_ks_dir / subject / "spike_times.npy")
        ks_labels = np.load(hybrid_ks_dir / subject / "spike_clusters.npy")
        ks_labels = cluster_utils.make_labels_contiguous(ks_labels.squeeze().astype(int))
        hybrid_sortings[subject][name] = Sorting(
            raw_data_bin,
            geom,
            ks_samples.squeeze().astype(int),
            ks_labels.squeeze().astype(int),
            name,
            cache_dir= hybrid_fig_cache_dir / subject,
            # do_cleaned_templates=True,
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )

    print()

# %%
1

# %%
unit_dfs = []
for subject, subject_comparisons in hybrid_comparisons.items():
    if "Detection" in subject_comparisons:
        local_density = density_near_gt(subject_comparisons["Detection"])
    
    for i, (sorter_name, comparison) in enumerate(subject_comparisons.items()):
        if comparison.unsorted:
            continue
        print(sorter_name)
        df = comparison.performance_by_unit.copy()
        df["Subject"] = subject
        df["Sort"] = sorter_name
        df["step"] = i
        df["sort_lo"] = comparison.new_sorting.name_lo
        df["gt_unit_id"] = df.index
        df["gt_ptp"] = comparison.gt_sorting.template_maxptps
        df["gt_firing_rate"] = comparison.gt_sorting.unit_firing_rates
        if "Detection" in subject_comparisons:
            df["gt_local_detection_density"] = local_density
        df["unsorted_recall"] = comparison.unsorted_recall_by_unit
        
        unit_dfs.append(df)
unit_df = pd.concat(unit_dfs, ignore_index=True)
unit_df

# %%
new_unit_dfs = []
for subject, subject_comparisons in hybrid_comparisons.items():
    sorting = comparison.gt_sorting
    df = dict(
        Subject=subject,
        Sort=sorting.name,
        step=-1,
        sort_lo=sorting.name_lo,
        unit_label=sorting.unit_labels,
        contam_ratio=sorting.contam_ratios,
        log10_contam_ratio_pluseneg10=np.log10(sorting.contam_ratios + 1e-10),
        contam_p_value=sorting.contam_p_values,
        template_ptp=sorting.template_xzptp[:, 2],
    )
    df = pd.DataFrame.from_dict(df)

    new_unit_dfs.append(df)
    
    for i, (sorter_name, comparison) in enumerate(subject_comparisons.items()):
        if comparison.unsorted:
            continue
        print(sorting.name, sorting.name_lo)
        sorting = comparison.new_sorting
        df = dict(
            Subject=subject,
            Sort=sorting.name,
            step=i,
            sort_lo=sorting.name_lo,
            unit_label=sorting.unit_labels,
            contam_ratio=sorting.contam_ratios,
            log10_contam_ratio_pluseneg10=np.log10(sorting.contam_ratios + 1e-10),
            contam_p_value=sorting.contam_p_values,
            template_ptp=sorting.template_xzptp[:, 2],
        )
        df = pd.DataFrame.from_dict(df)
        
        new_unit_dfs.append(df)

new_unit_df = pd.concat(new_unit_dfs, ignore_index=True)
new_unit_df

# %%
plt.close("all")

# %%
hybrid_comparisons["DY_018"].keys()

# %%
for subject in subjects:
    comparisons = hybrid_comparisons[subject]
    names = list(comparisons.keys())
    unsorted_recalls = [c.unsorted_recall for c in comparisons.values()]
    sorted_recalls = [c.weighted_average_performance["recall"] for c in comparisons.values()]
    la, = plt.plot(unsorted_recalls, "b") 
    lb, = plt.plot(sorted_recalls, "green")
plt.gca().set_xticks(range(len(names)), names, rotation=45)
plt.title("Recall through the pipeline")
plt.legend([la, lb], ["unsorted", "sorted"], frameon=False)
plt.gcf().savefig(hybrid_fig_dir / "summary_recall.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
for subject in subjects:
    comparisons = hybrid_comparisons[subject]
    names = [k for k, c in comparisons.items() if not c.unsorted]
    acc = [c.weighted_average_performance["accuracy"] for c in comparisons.values() if not c.unsorted]
    rec = [c.weighted_average_performance["recall"] for c in comparisons.values() if not c.unsorted]
    prec = [c.weighted_average_performance["precision"] for c in comparisons.values() if not c.unsorted]
    la, = plt.plot(acc, "b") 
    lb, = plt.plot(rec, "g")
    lc, = plt.plot(prec, "r")
plt.grid(axis="y")
plt.gca().set_xticks(range(len(names)), names, rotation=45)
plt.title("Sorted metrics through the pipeline")
plt.legend([la, lb, lc], ["accuracy", "recall", "precision"], fancybox=False, framealpha=1)
plt.gcf().savefig(hybrid_fig_dir / "summary_perf.png", dpi=300, bbox_inches="tight")
plt.show()

# %% tags=[]
(hybrid_fig_dir / "template_maxchan_traces").mkdir(exist_ok=True)

jobs = []
for subject, comparisons in hybrid_comparisons.items():
    for step, (sorting, comp) in enumerate(comparisons.items()):
        if comp.unsorted:
            continue
        
        fig = comp.new_sorting.template_maxchan_vis()
        name_lo = comp.new_sorting.name_lo
        plt.show()
        fig.savefig(hybrid_fig_dir / "template_maxchan_traces" / f"{subject}_{step}_{name_lo}.png", dpi=300)
        plt.close(fig)

# %%
# (hybrid_fig_dir / "cleaned_temp_vis").mkdir(exist_ok=True)

# def job(step_savedir, new_sorting, i, unit):
#     count = new_sorting.unit_spike_counts[i]
#     fig, ax, maxsnr, raw_maxptp, cleaned_maxptp = new_sorting.cleaned_temp_vis(
#         unit
#     )
#     fig.suptitle(
#         f"unit {unit}: raw maxptp {raw_maxptp:.2f}, cleaned maxptp {cleaned_maxptp:.2f}, "
#         f"nspikes {count}, max chan snr {maxsnr:.2f}.",
#         fontsize=8,
#         y=0.91,
#     )
#     # plt.show()
#     fig.savefig(step_savedir / f"rawmaxptp{raw_maxptp:05.2f}_unit{unit:03d}.pdf")
#     plt.close(fig)


# jobs = []
# for subject, comparisons in hybrid_comparisons.items():
#     for step, (sorting, comp) in enumerate(comparisons.items()):
        
#         if comp.unsorted:
#             continue
        
#         if comp.new_sorting.cleaned_templates is None:
#             continue
        
#         name_lo = comp.new_sorting.name_lo
#         step_savedir = hybrid_fig_dir / "cleaned_temp_vis" / subject / f"{step}_{name_lo}"
#         step_savedir.mkdir(exist_ok=True, parents=True)

#         for i, unit in enumerate(comp.new_sorting.unit_labels):
#             jobs.append(delayed(job)(step_savedir, comp.new_sorting, i, unit))
#             # job(step_savedir, comp.new_sorting, i, unit)

# for res in Parallel(14)(tqdm(jobs)):
#     pass

# %% tags=[]
(hybrid_fig_dir / "perf_by_ptp").mkdir(exist_ok=True)

for step, df in unit_df.groupby("step"):
    # display(df)
    sort = df["Sort"].values[0]
    name_lo = df["sort_lo"].values[0]

    fig, ((aa, ab), (ac, ad)) = plt.subplots(2, 2, sharey=True, figsize=(8, 8))
    plotgistic(df, x="gt_ptp", y="accuracy", ax=aa, legend=False)
    plotgistic(df, x="gt_ptp", y="precision", ax=ab)
    plotgistic(df, x="gt_ptp", y="recall", ax=ac, legend=False)
    plotgistic(df, x="gt_ptp", y="unsorted_recall", ax=ad, legend=False)
    n_missed = (df["accuracy"] < 1e-8).sum()
    
    fig.suptitle(f"Step {step}: {sort}.   {n_missed=}", y=0.925)
    fig.savefig(hybrid_fig_dir / "perf_by_ptp" / f"{step}_{name_lo}.png")
    # plt.show()
    plt.close(fig)

# %% tags=[]
(hybrid_fig_dir / "perf_by_firing_rate").mkdir(exist_ok=True)

for step, df in unit_df.groupby("step"):
    # display(df)
    sort = df["Sort"].values[0]
    name_lo = df["sort_lo"].values[0]    

    fig, ((aa, ab), (ac, ad)) = plt.subplots(2, 2, sharey=True, figsize=(8, 8))
    plotgistic(df, x="gt_firing_rate", y="accuracy", c="gt_ptp", ax=aa, legend=False)
    plotgistic(df, x="gt_firing_rate", y="precision", c="gt_ptp", ax=ab)
    plotgistic(df, x="gt_firing_rate", y="recall", c="gt_ptp", ax=ac, legend=False)
    plotgistic(df, x="gt_firing_rate", y="unsorted_recall", c="gt_ptp", ax=ad, legend=False)
    n_missed = (df["accuracy"] < 1e-8).sum()
    
    fig.suptitle(f"Step {step}: {sort}.    {n_missed=}", y=0.925)
    fig.savefig(hybrid_fig_dir / "perf_by_firing_rate" / f"{step}_{name_lo}.png")
    # plt.show()
    plt.close(fig)

# %% tags=[]
(hybrid_fig_dir / "perf_by_local_detection_density").mkdir(exist_ok=True)

for step, df in unit_df.groupby("step"):
    # display(df)
    sort = df["Sort"].values[0]
    name_lo = df["sort_lo"].values[0]    

    fig, ((aa, ab), (ac, ad)) = plt.subplots(2, 2, sharey=True, figsize=(8, 8))
    plotgistic(df, x="gt_local_detection_density", y="accuracy", c="gt_ptp", ax=aa, legend=False)
    plotgistic(df, x="gt_local_detection_density", y="precision", c="gt_ptp", ax=ab)
    plotgistic(df, x="gt_local_detection_density", y="recall", c="gt_ptp", ax=ac, legend=False)
    plotgistic(df, x="gt_local_detection_density", y="unsorted_recall", c="gt_ptp", ax=ad, legend=False)
    n_missed = (df["accuracy"] < 1e-8).sum()
    
    fig.suptitle(f"Step {step}: {sort}.    {n_missed=}", y=0.925)
    fig.savefig(hybrid_fig_dir / "perf_by_local_detection_density" / f"{step}_{name_lo}.png")
    # plt.show()
    plt.close(fig)

# %% tags=[]
(hybrid_fig_dir / "single_unit_metrics").mkdir(exist_ok=True)

for step, df in new_unit_df.groupby("step"):
    # display(df)
    sort = df["Sort"].values[0]
    name_lo = df["sort_lo"].values[0]    

    fig, (aa, ab) = plt.subplots(1, 2, sharey=False, figsize=(8, 4.5))
    plotgistic(df, x="template_ptp", y="contam_ratio", c="contam_p_value", ax=aa, legend=False, ylim=None)
    plotgistic(df, x="template_ptp", y="log10_contam_ratio_pluseneg10", c="contam_p_value", ax=ab, legend=True, ylim=None)
    ab.axhline(np.log10(0.2 + 1e-10), color="g", ls=":", lw=1)
    mean_contam_ratio = df["contam_ratio"].mean()
    
    fig.suptitle(f"Step {step}: {sort}.    {mean_contam_ratio=:0.2f}", y=0.925)
    step = step if step >= 0 else "_gt"
    fig.savefig(hybrid_fig_dir / "single_unit_metrics" / f"{step}_{name_lo}.png")
    # plt.show()
    plt.close(fig)

# %% tags=[]
(hybrid_fig_dir / "array_scatter").mkdir(exist_ok=True)

def job(step, subject, comp_a, ks_comp):
    name = comp_a.new_sorting.name
    name_lo = comp_a.new_sorting.name_lo
    fig, axes, ea, pct = array_scatter_vs(comp_a, ks_comp, do_ellipse=True)
    n_units = comp_a.new_sorting.unit_labels.size
    st = fig.suptitle(f"{subject}, {name}. {pct}% of spikes shown from {n_units} units. ", y=1.01)
    fig.savefig(hybrid_fig_dir / "array_scatter" / f"{subject}_{step}_{name_lo}.png", dpi=300, bbox_inches="tight", bbox_extra_artists=[st])
    plt.close(fig)

jobs = []
for subject, comparisons in hybrid_comparisons.items():
    ks_comp = comparisons["KSAll"] if "KSAll" in comparisons else None
    for step, (sorting, comp) in enumerate(comparisons.items()):
        if comp.unsorted or "ks" in comp.new_sorting.name_lo:
            continue
        print(subject, comp.new_sorting.name)
        jobs.append(delayed(job)(step, subject, comp, ks_comp))

for res in Parallel(4)(tqdm(jobs, total=len(jobs))):
    pass

# %%
# compose colormap with sqrt to enhance dynamic range on the low end
cmap = colors.LinearSegmentedColormap.from_list("cbhlx", plt.cm.magma(np.sqrt(np.linspace(0, 1, num=256))))

outdir = hybrid_fig_dir / "agreement"
outdir.mkdir(exist_ok=True)

for subject, comparisons in hybrid_comparisons.items():
    for name, comp in comparisons.items():
        if comp.unsorted:
            continue
        fig, ax = plot_agreement_matrix(comp, cmap=cmap)
        ax.set_title(f"{subject}: {name} agreement matrix")
        # plt.show()
        fig.savefig(
            outdir / f"{subject}_{comp.new_sorting.name_lo}_agreement.png",
            dpi=300,
        )
        plt.close(fig)

# %%
outdir = hybrid_fig_dir / "gtunit_resid_norm"
outdir.mkdir(exist_ok=True)

def job(subject, gt_unit, savedir, comp):
    acc = comp.performance_by_unit["accuracy"][gt_unit] * 100
    # if (savedir / f"acc{acc:04.1f}_gtu{gt_unit:03d}.png").exists():
        # return
    fig, axes = gtunit_resid_study(
        comp,
        gt_unit,
        lambd=0.005
    )
    if fig is None:
        return
    axes["a"].set_title(f"{subject}: GT unit {gt_unit}")
    fig.savefig(savedir / f"acc{acc:04.1f}_gtu{gt_unit:03d}.png", dpi=300)
    plt.close(fig)

jobs = []
for subject, comparisons in hybrid_comparisons.items():
    for step, (sorting, comp) in enumerate(comparisons.items()):
        if comp.unsorted:
            continue
        if comp.new_sorting.cleaned_templates is None:
            continue

        savedir = outdir / f"{subject}_{step}_{comp.new_sorting.name_lo}"
        savedir.mkdir(exist_ok=True)
            
        for gtu in comp.gt_sorting.unit_labels:
            jobs.append(delayed(job)(subject, gtu, savedir, comp))

for res in Parallel(2)(tqdm(jobs, total=len(jobs))):
    pass

# %%
# compose colormap with sqrt to enhance dynamic range on the low end
cmap = colors.LinearSegmentedColormap.from_list("cbhlx", plt.cm.magma(np.sqrt(np.linspace(0, 1, num=256))))

outdir = hybrid_fig_dir / "overlap"
outdir.mkdir(exist_ok=True)

for subject, comparisons in hybrid_comparisons.items():
    for name, comp in comparisons.items():
        if comp.unsorted:
            continue
        if comp.new_sorting.cleaned_templates is None:
            continue
        fig, ax = plot_agreement_matrix(comp, cmap=cmap, with_resid=True)
        fig.suptitle(f"{subject}: {name}")
        plt.show()
        fig.savefig(
            outdir / f"{subject}_{comp.new_sorting.name_lo}_overlap.png",
            dpi=300,
        )
        plt.close(fig)

# %% tags=[]
outdir = hybrid_fig_dir / "zoom_scatter"
outdir.mkdir(exist_ok=True)

def job(step_comparisons, vs_comparison, subject, gt_unit):
    fig, axes, leg_artist, gt_ptp = near_gt_scatter_vs(step_comparisons, vs_comparison, gt_unit)
    fig.suptitle(f"{subject} GT unit {gt_unit}", y=0.95)
    fig.savefig(
        outdir / f"ptp{gt_ptp:05.2f}_{subject}_unit{gt_unit:02d}.png",
        dpi=300,
        bbox_extra_artists=[leg_artist],
    )
    plt.close(fig)
        
jobs = []
for subject, comparisons in hybrid_comparisons.items():
    ks_comp = comparisons["KSAll"]
    step_comparisons = [
        comp
        for comp in comparisons.values()
        if not (comp.unsorted or "ks" in comp.new_sorting.name_lo)
    ]
    for gt_unit in ks_comp.gt_sorting.unit_labels:
        jobs.append(delayed(job)(step_comparisons, ks_comp, subject, gt_unit))

for res in Parallel(28)(tqdm(jobs, total=len(jobs))):
    pass

# %%
1

# %% tags=[]
sorting = "Deconv1"
outdir = hybrid_fig_dir / "venn_gt_v_deconv1"
outdir.mkdir(exist_ok=True)

def job(hybrid_comparison, subject, gt_unit):
    import warnings
    with warnings.catch_warnings():
        # try:
        fig, gt_ptp, acc = make_diagnostic_plot(hybrid_comparison, gt_unit)
        fig.savefig(outdir / f"ptp{gt_ptp:05.2f}_acc{acc}_{subject}_unit{gt_unit:02d}.png")
        plt.close(fig)
        # except ValueError as e:
            # print(e)

jobs = []
for subject, comparisons in hybrid_comparisons.items():
    comp = comparisons[sorting]
    for gt_unit in comp.gt_sorting.unit_labels:
        jobs.append(delayed(job)(comp, subject, gt_unit))

for res in Parallel(13)(tqdm(jobs, total=len(jobs))):
    pass

# %%
sorting = "Deconv1-SplitMerge"
outdir = hybrid_fig_dir / "venn_gt_v_deconv1_splitmerge"
outdir.mkdir(exist_ok=True)

def job(hybrid_comparison, subject, gt_unit):
    import warnings
    with warnings.catch_warnings():
        # try:
        fig, gt_ptp, acc = make_diagnostic_plot(hybrid_comparison, gt_unit)
        fig.savefig(outdir / f"ptp{gt_ptp:05.2f}_acc{acc}_{subject}_unit{gt_unit:02d}.png")
        plt.close(fig)
        # except ValueError as e:
            # print(e)

jobs = []
for subject, comparisons in hybrid_comparisons.items():
    comp = comparisons[sorting]
    for gt_unit in comp.gt_sorting.unit_labels:
        jobs.append(delayed(job)(comp, subject, gt_unit))

for res in Parallel(13)(tqdm(jobs, total=len(jobs))):
    pass

# %% tags=[]
sorting = "Deconv2-Cleaned"
outdir = hybrid_fig_dir / "venn_gt_v_deconv2_cleaned"
outdir.mkdir(exist_ok=True)

def job(hybrid_comparison, subject, gt_unit):
    import warnings
    with warnings.catch_warnings():
        try:
            fig, gt_ptp, acc = make_diagnostic_plot(hybrid_comparison, gt_unit)
            fig.savefig(outdir / f"ptp{gt_ptp:05.2f}_acc{acc}_{subject}_unit{gt_unit:02d}.png")
            plt.close(fig)
        except ValueError as e:
            print(e)

jobs = []
for subject, comparisons in hybrid_comparisons.items():
    comp = comparisons[sorting]
    for gt_unit in comp.gt_sorting.unit_labels:
        jobs.append(delayed(job)(comp, subject, gt_unit))

for res in Parallel(13)(tqdm(jobs, total=len(jobs))):
    pass

# %% tags=[]
sorting = "Deconv2"
outdir = hybrid_fig_dir / "venn_gt_v_deconv2"
outdir.mkdir(exist_ok=True)

def job(hybrid_comparison, subject, gt_unit):
    import warnings
    with warnings.catch_warnings():
        try:
            fig, gt_ptp, acc = make_diagnostic_plot(hybrid_comparison, gt_unit)
            fig.savefig(outdir / f"ptp{gt_ptp:05.2f}_acc{acc}_{subject}_unit{gt_unit:02d}.png")
            plt.close(fig)
        except ValueError as e:
            print(e)

jobs = []
for subject, comparisons in hybrid_comparisons.items():
    comp = comparisons[sorting]
    for gt_unit in comp.gt_sorting.unit_labels:
        jobs.append(delayed(job)(comp, subject, gt_unit))

for res in Parallel(13)(tqdm(jobs, total=len(jobs))):
    pass

# %%
sorting = "Deconv3-Reassign"
outdir = hybrid_fig_dir / "venn_gt_v_deconv3_reassign"
outdir.mkdir(exist_ok=True)

def job(hybrid_comparison, subject, gt_unit):
    import warnings
    with warnings.catch_warnings():
        try:
            fig, gt_ptp, acc = make_diagnostic_plot(hybrid_comparison, gt_unit)
            fig.savefig(outdir / f"ptp{gt_ptp:05.2f}_acc{acc}_{subject}_unit{gt_unit:02d}.png")
            plt.close(fig)
        except ValueError as e:
            print(e)

jobs = []
for subject, comparisons in hybrid_comparisons.items():
    comp = comparisons[sorting]
    for gt_unit in comp.gt_sorting.unit_labels:
        jobs.append(delayed(job)(comp, subject, gt_unit))

for res in Parallel(13)(tqdm(jobs, total=len(jobs))):
    pass

# %% tags=[]
sorting = "KSAll"
outdir = hybrid_fig_dir / "venn_gt_v_ksall"
outdir.mkdir(exist_ok=True)

def job(hybrid_comparison, subject, gt_unit):
    try:
        fig, gt_ptp, acc = make_diagnostic_plot(hybrid_comparison, gt_unit)
        fig.savefig(outdir / f"ptp{gt_ptp:05.2f}_acc{acc}_{subject}_unit{gt_unit:02d}.png")
        plt.close(fig)
    except:
        return 1
    return 0

jobs = []
for subject, comparisons in hybrid_comparisons.items():
    comp = comparisons[sorting]
    for gt_unit in comp.gt_sorting.unit_labels:
        jobs.append(delayed(job)(comp, subject, gt_unit))

failed = 0
for res in Parallel(13)(tqdm(jobs, total=len(jobs))):
    failed += res
print(failed)

# %%
# (hybrid_fig_dir / "unit_summary_fig").mkdir(exist_ok=True)

# def job(step_savedir, new_sorting, i, unit):
#     count = new_sorting.unit_spike_counts[i]
#     fig, ax, raw_maxptp = new_sorting.unit_summary_fig(unit, show_chan_label=False)
#     # plt.show()
#     fig.savefig(step_savedir / f"rawmaxptp{raw_maxptp:05.2f}_unit{unit:03d}.png")
#     plt.close(fig)


# jobs = []
# for subject, comparisons in hybrid_comparisons.items():
#     for step, (sorting, comp) in enumerate(comparisons.items()):
        
#         if comp.unsorted:
#             continue
        
#         # if "ks" not in comp.new_sorting.name_lo and comp.new_sorting.cleaned_templates is None:
#         #     continue
#         if not any(
#             k in comp.new_sorting.name_lo for k in ("ks", "cleaned")
#         ):
#             continue
        
#         name_lo = comp.new_sorting.name_lo
#         print(subject, step, name_lo)

#         step_savedir = hybrid_fig_dir / "unit_summary_fig" / subject / f"{step}_{name_lo}"
#         step_savedir.mkdir(exist_ok=True, parents=True)

#         for i, unit in enumerate(comp.new_sorting.unit_labels):
#             jobs.append(delayed(job)(step_savedir, comp.new_sorting, i, unit))
#             # job(step_savedir, comp.new_sorting, i, unit)

# for res in Parallel(14)(tqdm(jobs)):
#     pass

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
