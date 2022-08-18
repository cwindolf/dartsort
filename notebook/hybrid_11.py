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
# time.sleep(13 * 60)

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
hybrid_deconv_dir = Path("/mnt/3TB/charlie/hybrid_5min/hybrid_5min_deconv/")
assert all((hybrid_bin_dir.exists(), hybrid_res_dir.exists(), hybrid_ks_dir.exists(), hybrid_deconv_dir.exists()))


# %%
subjects = ("DY_018", "CSHL051")

# %%
hybrid_fig_dir = Path("/mnt/3TB/charlie/hybrid_5min/figs_8_17_wfsbugfix/")
hybrid_fig_dir.mkdir(exist_ok=True)

# %%
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
        # spike_xzptp=gt_xzptp,
    )

# %%
1

# %%
hybrid_sortings = {}
hybrid_comparisons = {}
for subject in tqdm(subjects):
    print(subject)

    # we will populate these dicts
    hybrid_sortings[subject] = {}
    hybrid_comparisons[subject] = {}

# %% tags=[]
for subject in tqdm(subjects):
    print(subject)
    print("-" * len(subject))

    hybrid_gt_h5 = hybrid_bin_dir / f"{subject}_gt.h5"
    raw_data_bin = hybrid_bin_dir / f"{subject}.ap.bin"
    sub_h5 = next((hybrid_res_dir / subject).glob("sub*h5"))
    
    with h5py.File(sub_h5, "r") as h5:
        det_spike_index = h5["spike_index"][:]
        x = h5["localizations"][:, 0]
        z_reg = h5["z_reg"][:]
        maxptps = h5["maxptps"][:]
        det_xzptp = np.c_[x, z_reg, maxptps]
    
    # unsorted comparison with detection
    name = "Detection"
    if name not in hybrid_comparisons[subject]:
        print("//", name)
        hybrid_sortings[subject][name] = Sorting(
            raw_data_bin,
            geom,
            det_spike_index[:, 0],
            np.zeros_like(det_spike_index[:, 0]),
            name,
            spike_maxchans=det_spike_index[:, 1],
            unsorted=True,
            spike_xzptp=det_xzptp,
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
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )
    
    # deconv1
    name = "Deconv1"
    if name not in hybrid_comparisons[subject]:
        print("//", name)
        orig_deconv1_st = np.load(hybrid_deconv_dir / subject / "spike_train.npy")
        with h5py.File(hybrid_deconv_dir / subject / "deconv_results.h5") as h5:
            locs = h5["localizations"][:]
            deconv1_xzptp = np.c_[locs[:, 0], locs[:, 3], h5["maxptps"][:]]
        hybrid_sortings[subject][name] = Sorting(
            raw_data_bin,
            geom,
            orig_deconv1_st[:, 0],
            cluster_utils.make_labels_contiguous(orig_deconv1_st[:, 1]),
            name,
            spike_xzptp=deconv1_xzptp,
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )
    
    name = "Deconv1-Split"
    if name not in hybrid_comparisons[subject]:
        print("//", name)
        samples = np.load(hybrid_deconv_dir / subject / "postdeconv_split_times.npy")
        labels = np.load(hybrid_deconv_dir / subject / "postdeconv_split_labels.npy")
        order = np.load(hybrid_deconv_dir / subject / "postdeconv_split_order.npy")
        templates = np.load(hybrid_deconv_dir / subject / "postdeconv_split_templates.npy")
        with h5py.File(hybrid_deconv_dir / subject / "deconv_results.h5") as h5:
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
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )
    
    name = "Deconv1-SplitMerge"
    if name not in hybrid_comparisons[subject]:
        print("//", name)
        samples = np.load(hybrid_deconv_dir / subject / "postdeconv_merge_times.npy")
        labels = np.load(hybrid_deconv_dir / subject / "postdeconv_merge_labels.npy")
        order = np.load(hybrid_deconv_dir / subject / "postdeconv_merge_order.npy")
        templates = np.load(hybrid_deconv_dir / subject / "postdeconv_merge_templates.npy")
        with h5py.File(hybrid_deconv_dir / subject / "deconv_results.h5") as h5:
            locs = h5["localizations"][:]
            deconv1_xzptp = np.c_[locs[:, 0], locs[:, 3], h5["maxptps"][:]]
        hybrid_sortings[subject]["Deconv1-SplitMerge"] = Sorting(
            raw_data_bin,
            geom,
            samples,
            labels,
            "Deconv1-SplitMerge",
            templates=templates,
            spike_xzptp=deconv1_xzptp[order],
        )
        hybrid_comparisons[subject]["Deconv1-SplitMerge"] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject]["Deconv1-SplitMerge"], geom
        )
    
    # deconv2
    name = "Deconv2"
    if name not in hybrid_comparisons[subject]:
        print("//", name)
        deconv2_st = np.load(hybrid_deconv_dir / subject / "deconv2/spike_train.npy")
        with h5py.File(hybrid_deconv_dir / subject / "deconv2/deconv_results.h5") as h5:
            locs = h5["localizations"][:]
            deconv2_xzptp = np.c_[locs[:, 0], locs[:, 3], h5["maxptps"][:]]
        hybrid_sortings[subject][name] = Sorting(
            raw_data_bin,
            geom,
            deconv2_st[:, 0],
            cluster_utils.make_labels_contiguous(deconv2_st[:, 1]),
            name,
            spike_xzptp=deconv2_xzptp,
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )
    
    name = "Deconv2-Cleaned"
    if name not in hybrid_comparisons[subject]:
        print("//", name)
        samples = np.load(hybrid_deconv_dir / subject / "deconv2/postdeconv_cleaned_times.npy")
        labels = np.load(hybrid_deconv_dir / subject / "deconv2/postdeconv_cleaned_labels.npy")
        order = np.load(hybrid_deconv_dir / subject / "deconv2/postdeconv_cleaned_order.npy")
        templates = np.load(hybrid_deconv_dir / subject / "deconv2/postdeconv_cleaned_templates.npy")
        hybrid_sortings[subject][name] = Sorting(
            raw_data_bin,
            geom,
            samples,
            labels,
            name,
            templates=templates,
            spike_xzptp=deconv2_xzptp[order],
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
        )
        hybrid_comparisons[subject][name] = HybridComparison(
            gt_sortings[subject], hybrid_sortings[subject][name], geom
        )

    print()

# %%
unit_dfs = []
for subject, subject_comparisons in hybrid_comparisons.items():
    local_density = density_near_gt(subject_comparisons["Detection"])
    
    for i, (sorter_name, comparison) in enumerate(subject_comparisons.items()):
        if comparison.unsorted:
            continue
        df = comparison.performance_by_unit.copy()
        df["Subject"] = subject
        df["Sort"] = sorter_name
        df["step"] = i
        df["sort_lo"] = comparison.new_sorting.name_lo
        df["gt_unit_id"] = df.index
        df["gt_ptp"] = comparison.gt_sorting.template_maxptps
        df["gt_firing_rate"] = comparison.gt_sorting.unit_firing_rates
        df["gt_local_detection_density"] = local_density
        df["unsorted_recall"] = comparison.unsorted_recall_by_unit
        
        unit_dfs.append(df)
unit_df = pd.concat(unit_dfs, ignore_index=True)
unit_df

# %%
new_unit_dfs = []
for subject, subject_comparisons in hybrid_comparisons.items():
    local_density = density_near_gt(subject_comparisons["Detection"])
    
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
plt.gca().set_xticks(range(len(names)), names, rotation=45)
plt.title("Sorted metrics through the pipeline")
plt.legend([la, lb, lc], ["accuracy", "recall", "precision"], frameon=False)
plt.gcf().savefig(hybrid_fig_dir / "summary_perf.png", dpi=300, bbox_inches="tight")
plt.show()

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
    plt.show()
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
    plt.show()
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
    plt.show()
    plt.close(fig)

# %%
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
    plt.show()
    plt.close(fig)

# %%
(hybrid_fig_dir / "template_maxchan_traces").mkdir(exist_ok=True)

def job(step, subject, new_sorting):
    fig = new_sorting.template_maxchan_vis()
    name_lo = new_sorting.name_lo
    fig.savefig(hybrid_fig_dir / "template_maxchan_traces" / f"{subject}_{step}_{name_lo}.png", dpi=300)
    plt.close(fig)

jobs = []
for subject, comparisons in hybrid_comparisons.items():
    for step, (sorting, comp) in enumerate(comparisons.items()):
        if comp.unsorted:
            continue
        jobs.append(delayed(job)(step, subject, comp.new_sorting))

for res in Parallel(8)(tqdm(jobs, total=len(jobs))):
    pass

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
    ks_comp = comparisons["KSAll"]
    for step, (sorting, comp) in enumerate(comparisons.items()):
        if comp.unsorted or "ks" in comp.new_sorting.name_lo:
            continue
        print(subject, comp.new_sorting.name)
        jobs.append(delayed(job)(step, subject, comp, ks_comp))

for res in Parallel(14)(tqdm(jobs, total=len(jobs))):
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
        ax = plot_agreement_matrix(comp, cmap=cmap)
        fig = ax.figure
        ax.set_title(f"{subject}: {name} agreement matrix")
        plt.show()
        fig.savefig(
            outdir / f"{subject}_{comp.new_sorting.name_lo}_agreement.png",
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
sorting = "Deconv2"
outdir = hybrid_fig_dir / "venn_gt_v_deconv2"
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

# %%
