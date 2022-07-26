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
from IPython.display import display, Image
from joblib import Parallel, delayed
import os
import colorcet as cc
from matplotlib import colors

os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %%
from spike_psvae import (
    subtract,
    cluster_utils,
    cluster_viz,
    cluster_viz_index,
    grab_and_localize,
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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %%
hybrid_bin_dir = Path("/share/ctn/users/ciw2107/hybrid_5min/hybrid_5min_output/")
hybrid_res_dir = Path("/share/ctn/users/ciw2107/hybrid_5min/hybrid_5min_subtraction/")
hybrid_ks_dir = Path("/share/ctn/users/ciw2107/hybrid_5min/hybrid_5min_kilosort/")
hybrid_deconv_dir = Path("/share/ctn/users/ciw2107/hybrid_5min/hybrid_5min_deconv/")
hybrid_bin_dir.exists(), hybrid_res_dir.exists(), hybrid_ks_dir.exists(), hybrid_deconv_dir.exists()


# %%
subjects = ("DY_018", "CSHL051")

# %%
hybrid_fig_dir = Path("/share/ctn/users/ciw2107/hybrid_5min/figs_7_19/")
hybrid_fig_dir.mkdir(exist_ok=True)

# %%
# %rm {hybrid_fig_dir}/*

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

# %% tags=[]
hybrid_sortings = {}
hybrid_comparisons = {}

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
    
    # we will populate these dicts
    hybrid_sortings[subject] = {}
    hybrid_comparisons[subject] = {}
    
    # unsorted comparison with detection
    print("//Detection...")
    hybrid_sortings[subject]["Detection"] = Sorting(
        raw_data_bin,
        geom,
        det_spike_index[:, 0],
        np.zeros_like(det_spike_index[:, 0]),
        "Detection",
        spike_maxchans=det_spike_index[:, 1],
        unsorted=True,
        spike_xzptp=det_xzptp,
    )
    hybrid_comparisons[subject]["Detection"] = HybridComparison(
        gt_sortings[subject], hybrid_sortings[subject]["Detection"], geom
    )
    
    # original clustering
    print("//Cluster...")
    cluster_spike_index = np.load(
        hybrid_res_dir / subject / "pre_merge_split_aligned_spike_index.npy"
    )
    cluster_labels = np.load(
        hybrid_res_dir / subject / "pre_merge_split_labels.npy"
    )
    hybrid_sortings[subject]["Cluster"] = Sorting(
        raw_data_bin,
        geom,
        cluster_spike_index[:, 0],
        cluster_labels,
        "Cluster",
        spike_maxchans=cluster_spike_index[:, 1],
        spike_xzptp=det_xzptp,
    )
    hybrid_comparisons[subject]["Cluster"] = HybridComparison(
        gt_sortings[subject], hybrid_sortings[subject]["Cluster"], geom
    )
    
    # original clustering -> split/merge
    print("//Split/merge...")
    splitmerge_spike_index = np.load(
        hybrid_res_dir / subject / "aligned_spike_index.npy"
    )
    splitmerge_labels = np.load(hybrid_res_dir / subject / "labels.npy")
    hybrid_sortings[subject]["Split/Merge"] = Sorting(
        raw_data_bin,
        geom,
        splitmerge_spike_index[:, 0],
        splitmerge_labels,
        "Split/Merge",
        spike_maxchans=splitmerge_spike_index[:, 1],
        spike_xzptp=det_xzptp,
    )
    hybrid_comparisons[subject]["Split/Merge"] = HybridComparison(
        gt_sortings[subject], hybrid_sortings[subject]["Split/Merge"], geom
    )
    
    # deconv1
    print("//Deconv1...")
    deconv1_samples = np.load(hybrid_deconv_dir / subject / "postdeconv_split_times.npy")
    deconv1_labels = np.load(hybrid_deconv_dir / subject / "postdeconv_merge_labels.npy")
    deconv1_sort = np.load(hybrid_deconv_dir / subject / "postdeconv_split_order.npy")
    with h5py.File(hybrid_deconv_dir / subject / "deconv_results.h5") as h5:
        locs = h5["localizations"][:]
        deconv1_xzptp = np.c_[locs[:, 0], locs[:, 3], h5["maxptps"][:]]
    hybrid_sortings[subject]["Deconv1"] = Sorting(
        raw_data_bin,
        geom,
        deconv1_samples,
        deconv1_labels,
        "Deconv1",
        spike_xzptp=deconv1_xzptp[deconv1_sort],
    )
    hybrid_comparisons[subject]["Deconv1"] = HybridComparison(
        gt_sortings[subject], hybrid_sortings[subject]["Deconv1"], geom
    )
    
    # deconv2
    print("//Deconv2...")
    deconv2_samples = np.load(hybrid_deconv_dir / subject / "deconv2/postdeconv_cleaned_times.npy")
    deconv2_labels = np.load(hybrid_deconv_dir / subject / "deconv2/postdeconv_cleaned_labels.npy")
    deconv2_sort = np.load(hybrid_deconv_dir / subject / "deconv2/postdeconv_cleaned_order.npy")
    with h5py.File(hybrid_deconv_dir / subject / "deconv2/deconv_results.h5") as h5:
        locs = h5["localizations"][:]
        deconv2_xzptp = np.c_[locs[:, 0], locs[:, 3], h5["maxptps"][:]]
    hybrid_sortings[subject]["Deconv2"] = Sorting(
        raw_data_bin,
        geom,
        deconv2_samples,
        deconv2_labels,
        "Deconv2",
        spike_xzptp=deconv2_xzptp[deconv2_sort],
    )
    hybrid_comparisons[subject]["Deconv2"] = HybridComparison(
        gt_sortings[subject], hybrid_sortings[subject]["Deconv2"], geom
    )
    
    # KSall
    print("//Kilosort...")
    ks_samples = np.load(hybrid_ks_dir / subject / "spike_times.npy")
    ks_labels = np.load(hybrid_ks_dir / subject / "spike_clusters.npy")
    ks_labels = cluster_utils.make_labels_contiguous(ks_labels.squeeze().astype(int))
    hybrid_sortings[subject]["KSAll"] = Sorting(
        raw_data_bin,
        geom,
        ks_samples.squeeze().astype(int),
        ks_labels.squeeze().astype(int),
        "KSAll",
    )
    hybrid_comparisons[subject]["KSAll"] = HybridComparison(
        gt_sortings[subject], hybrid_sortings[subject]["KSAll"], geom
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
        df["Subject"] = "CSHL051"
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
for subject in subjects:
    comparisons = hybrid_comparisons[subject]
    names = list(comparisons.keys())
    unsorted_recalls = [c.unsorted_recall for c in comparisons.values()]
    sorted_recalls = [c.weighted_average_performance["recall"] for c in comparisons.values()]
    la, = plt.plot(unsorted_recalls, "b") 
    lb, = plt.plot(sorted_recalls, "green")
plt.gca().set_xticks(range(len(names)), names)
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
plt.gca().set_xticks(range(len(names)), names)
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

# %% tags=[]
(hybrid_fig_dir / "array_scatter").mkdir(exist_ok=True)

def job(step, subject, comp_a, ks_comp):
    name = comp_a.new_sorting.name
    name_lo = comp_a.new_sorting.name_lo
    fig, axes, ea = array_scatter_vs(comp_a, ks_comp, do_ellipse=True)
    fig.suptitle(f"{subject}: GT template locs + {name} locs", y=1.01)
    fig.savefig(hybrid_fig_dir / "array_scatter" / f"{subject}_{step}_{name_lo}.png", dpi=300)
    plt.close(fig)

jobs = []
for subject, comparisons in hybrid_comparisons.items():
    ks_comp = comparisons["KSAll"]
    for step, (sorting, comp) in enumerate(comparisons.items()):
        if comp.unsorted or "ks" in comp.new_sorting.name_lo:
            continue
        jobs.append(delayed(job)(step, subject, comp, ks_comp))

for res in Parallel(8)(tqdm(jobs, total=len(jobs))):
    pass

# %% tags=[]
sorting = "Deconv2"
outdir = hybrid_fig_dir / "venn_gt_v_deconv2"
outdir.mkdir(exist_ok=True)

def job(hybrid_comparison, subject, gt_unit):
    import warnings
    with warnings.catch_warnings():
        fig, gt_ptp = make_diagnostic_plot(hybrid_comparison, gt_unit)
        fig.savefig(outdir / f"ptp{gt_ptp:05.2f}_{subject}_unit{gt_unit:02d}.png")
        plt.close(fig)

jobs = []
for subject, comparisons in hybrid_comparisons.items():
    comp = comparisons[sorting]
    for gt_unit in comp.gt_sorting.unit_labels:
        jobs.append(delayed(job)(comp, subject, gt_unit))

for res in Parallel(8)(tqdm(jobs, total=len(jobs))):
    pass

# %% tags=[]
sorting = "KSAll"
outdir = hybrid_fig_dir / "venn_gt_v_ksall"
outdir.mkdir(exist_ok=True)

def job(hybrid_comparison, subject, gt_unit):
    try:
        fig, gt_ptp = make_diagnostic_plot(hybrid_comparison, gt_unit)
        fig.savefig(outdir / f"ptp{gt_ptp:05.2f}_{subject}_unit{gt_unit:02d}.png")
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
for res in Parallel(8)(tqdm(jobs, total=len(jobs))):
    failed += res
print(failed)

# %% tags=[]
sorting = "KSAll"
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

# %%
comp.unsorted

# %%
a = gt_comparison.get_ordered_agreement_scores()

# %%
axes = sns.heatmap(a, cmap=plt.cm.cubehelix)


# %%

# %%
from spikeinterface.
