"""Hybrid analysis helpers

These are just a couple of classes which are basically bags of
data that would be used when making summary plots, and which compute
some metrics etc in the constructor. They help make wrangling a bunch
of sorts a little easier.
"""
import re
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist

from spikeinterface.extractors import NumpySorting
from spikeinterface.comparison import compare_sorter_to_ground_truth

from spike_psvae.spikeio import get_binary_length
from spike_psvae.deconvolve import get_templates
from spike_psvae import localize_index, cluster_viz, cluster_viz_index


class Sorting:
    """
    An object to hold onto a spike train and associated localizations
    and templates, and to compute basic statistics about the spike train,
    which will be used by the HybridSorting below.
    """

    def __init__(
        self,
        raw_bin,
        geom,
        spike_times,
        spike_labels,
        name,
        templates=None,
        spike_maxchans=None,
        spike_xzptp=None,
        unsorted=False,
        fs=30_000,
    ):
        n_spikes_full = spike_labels.shape[0]
        assert spike_labels.shape == spike_times.shape == (n_spikes_full,)
        T_samples, T_sec = get_binary_length(raw_bin, len(geom), fs)

        self.name = name
        self.geom = geom
        self.name_lo = re.sub("[^a-z0-9]+", "_", name.lower())
        self.fs = fs

        which = np.flatnonzero(spike_labels >= 0)
        which = which[np.argsort(spike_times[which])]

        self.unsorted = unsorted
        self.raw_bin = raw_bin
        self.spike_times = spike_times[which]
        self.spike_labels = spike_labels[which]
        self.unit_labels, self.unit_spike_counts = np.unique(
            self.spike_labels, return_counts=True
        )
        self.unit_firing_rates = self.unit_spike_counts / T_sec
        self.contiguous_labels = (
            self.unit_labels.size == self.unit_labels.max() + 1
        )

        self.templates = templates
        if templates is None and not unsorted:
            self.templates = get_templates(
                raw_bin,
                self.spike_times[:, None],
                self.spike_labels,
                geom,
            )

        if self.templates is not None:
            self.template_ptps = self.templates.ptp(1)
            self.template_maxptps = self.template_ptps.max(1)
            self.template_maxchans = self.template_ptps.argmax(1)
            self.template_locs = localize_index.localize_ptps_index(
                self.template_ptps,
                geom,
                self.template_maxchans,
                np.stack([np.arange(len(geom))] * len(geom), axis=0),
                n_channels=20,
                n_workers=None,
                pbar=True,
            )
            self.template_xzptp = np.c_[
                self.template_locs[0],
                self.template_locs[3],
                self.template_maxptps,
            ]

        if spike_maxchans is None:
            assert not unsorted
            print(
                f"Sorting {name} has no intrinsic maxchans. "
                "Using template maxchans."
            )
            self.spike_maxchans = self.template_maxchans[self.spike_labels]
        else:
            assert spike_maxchans.shape == (n_spikes_full,)
            self.spike_maxchans = spike_maxchans[which]

        self.spike_index = np.c_[self.spike_times, self.spike_maxchans]
        self.spike_train = np.c_[self.spike_times, self.spike_labels]
        self.n_spikes = len(self.spike_index)

        self.spike_xzptp = None
        if spike_xzptp is not None:
            assert spike_xzptp.shape == (n_spikes_full, 3)
            self.spike_xzptp = spike_xzptp[which]

    def get_unit_spike_train(self, unit):
        return self.spike_times[self.spike_labels == unit]

    def get_unit_maxchans(self, unit):
        return self.spike_maxchans[self.spike_labels == unit]

    @property
    def np_sorting(self):
        return NumpySorting.from_times_labels(
            times_list=self.spike_times,
            labels_list=self.spike_labels,
            sampling_frequency=self.fs,
        )

    def array_scatter(self):
        fig, axes = cluster_viz_index.array_scatter(
            self.spike_labels,
            self.geom,
            self.spike_xzptp[:, 0],
            self.spike_xzptp[:, 1],
            self.spike_xzptp[:, 2],
            annotate=False,
        )
        axes[0].scatter(*self.geom.T, marker="s", s=2, color="orange")
        return fig, axes


class HybridComparison:
    """
    An object which computes some hybrid metrics and stores references
    to the ground truth and compared sortings, so that everything is
    in one place for later plotting / analysis code.
    """

    def __init__(self, gt_sorting, new_sorting, geom):
        assert gt_sorting.contiguous_labels

        self.gt_sorting = gt_sorting
        self.new_sorting = new_sorting
        self.unsorted = new_sorting.unsorted
        self.geom = geom

        self.average_performance = (
            self.weighted_average_performance
        ) = _na_avg_performance
        if not new_sorting.unsorted:
            gt_comparison = compare_sorter_to_ground_truth(
                gt_sorting.np_sorting,
                new_sorting.np_sorting,
                gt_name=gt_sorting.name,
                tested_name=new_sorting.name,
                sampling_frequency=30_000,
                exhaustive_gt=False,
                match_score=0.1,
                verbose=True,
            )

            self.best_match_12 = gt_comparison.best_match_12.values.astype(int)
            self.gt_matched = self.best_match_12 >= 0

            # matching units and accuracies
            self.performance_by_unit = gt_comparison.get_performance().astype(
                float
            )
            # average the metrics over units
            self.average_performance = gt_comparison.get_performance(
                method="pooled_with_average"
            ).astype(float)
            # average metrics, weighting each unit by its spike count
            self.weighted_average_performance = (
                self.performance_by_unit
                * gt_sorting.unit_spike_counts[:, None]
            ).sum(0) / gt_sorting.unit_spike_counts.sum()

        # unsorted performance
        tp, fn, fp, num_gt = unsorted_confusion(
            gt_sorting.spike_index, new_sorting.spike_index
        )
        # as in spikeinterface, the idea of a true negative does not make sense here
        # accuracy with tn=0 is called threat score or critical success index, apparently
        self.unsorted_accuracy = tp / (tp + fn + fp)
        # this is what I was originally calling the unsorted accuracy
        self.unsorted_recall = tp / (tp + fn)
        self.unsorted_precision = tp / (tp + fp)
        self.unsorted_false_discovery_rate = fp / (tp + fp)
        self.unsorted_miss_rate = fn / num_gt

    def get_best_new_match(self, gt_unit):
        return int(self.best_match_12[gt_unit])

    def get_closest_new_unit(self, gt_unit):
        gt_loc = self.gt_sorting.template_xzptp[gt_unit]
        new_template_locs = self.new_sorting.template_xzptp
        return np.argmin(cdist(gt_loc[None], new_template_locs).squeeze())


# -- library


def unsorted_confusion(
    gt_spike_index, new_spike_index, n_samples=12, n_channels=4
):
    cmul = n_samples / n_channels
    n_gt_spikes = len(gt_spike_index)
    n_new_spikes = len(gt_spike_index)

    gt_kdt = KDTree(np.c_[gt_spike_index[:, 0], gt_spike_index[:, 1] * cmul])
    sorter_kdt = KDTree(
        np.c_[new_spike_index[:, 0], cmul * new_spike_index[:, 1]]
    )
    query = gt_kdt.query_ball_tree(sorter_kdt, n_samples + 0.1)

    # this is a boolean array of length n_gt_spikes
    detected = np.array([len(lst) > 0 for lst in query], dtype=bool)

    # from the above, we can compute a couple of metrics
    true_positives = detected.sum()
    false_negatives = n_gt_spikes - true_positives
    false_positives = n_new_spikes - true_positives

    return true_positives, false_negatives, false_positives, n_gt_spikes


_na_avg_performance = pd.Series(
    index=[
        "accuracy",
        "recall",
        "precision",
        "false_discovery_rate",
        "miss_rate",
    ],
    data=[np.nan] * 5,
)


# -- plotting helpers


def plotgistic(
    df,
    x="gt_ptp",
    y=None,
    c="gt_firing_rate",
    title=None,
    cmap=plt.cm.plasma,
    legend=True,
    ax=None,
    ylim=[-0.05, 1.05],
):
    ylab = y
    xlab = x
    clab = c
    y = df[y].values
    x = df[x].values
    X = sm.add_constant(x)
    c = df[c].values

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # fit the logistic function to the data
    def resids(beta):
        return y - 1 / (1 + np.exp(-X @ beta))

    res = least_squares(resids, np.array([1.0, 1]))
    b = res.x

    # plot the logistic line
    domain = np.linspace(x.min(), x.max())
    (l,) = ax.plot(
        domain, 1 / (1 + np.exp(-sm.add_constant(domain) @ b)), color="k"
    )

    # scatter with legend
    leg = ax.scatter(x, y, marker="x", c=c, cmap=cmap, alpha=0.75)
    h, labs = leg.legend_elements(num=4)
    if legend:
        ax.legend(
            (*h, l),
            (*labs, "logistic fit"),
            title=clab,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
        )

    if title:
        n_missed = (y < 1e-8).sum()
        plt.title(title + f" -- {n_missed} missed")

    ax.set_ylim(ylim)
    ax.set_xlim([x.min() - 0.5, x.max() + 0.5])
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)

    return fig, ax


def make_diagnostic_plot(hybrid_comparison, gt_unit):
    new_unit = hybrid_comparison.get_best_new_match(gt_unit)
    new_str = f"{hybrid_comparison.new_sorting.name} match {new_unit}"
    if new_unit < 0:
        new_unit = hybrid_comparison.get_closest_new_unit(gt_unit)
        new_str = f"No {hybrid_comparison.new_sorting.name} match, using closest unit {new_unit}."

    gt_np_sorting = hybrid_comparison.gt_sorting.np_sorting
    new_np_sorting = hybrid_comparison.new_sorting.np_sorting
    gt_spike_train = gt_np_sorting.get_unit_spike_train(gt_unit)
    new_spike_train = new_np_sorting.get_unit_spike_train(new_unit)
    gt_maxchans = hybrid_comparison.gt_sorting.get_unit_maxchans(gt_unit)
    new_maxchans = hybrid_comparison.new_sorting.get_unit_maxchans(new_unit)
    gt_template_zs = hybrid_comparison.gt_sorting.template_locs[2]
    new_template_zs = hybrid_comparison.new_sorting.template_locs[2]

    gt_ptp = hybrid_comparison.gt_sorting.template_maxptps[gt_unit]

    fig = cluster_viz.diagnostic_plots(
        new_unit,
        gt_unit,
        new_spike_train,
        gt_spike_train,
        hybrid_comparison.new_sorting.templates,
        hybrid_comparison.gt_sorting.templates,
        new_maxchans,
        gt_maxchans,
        hybrid_comparison.geom,
        hybrid_comparison.gt_sorting.raw_bin,
        dict(enumerate(new_template_zs)),
        dict(enumerate(gt_template_zs)),
        hybrid_comparison.new_sorting.spike_index,
        hybrid_comparison.gt_sorting.spike_index,
        hybrid_comparison.new_sorting.spike_labels,
        hybrid_comparison.gt_sorting.spike_labels,
        scale=7,
        sorting1_name=hybrid_comparison.new_sorting.name,
        sorting2_name=hybrid_comparison.gt_sorting.name,
        num_channels=40,
        num_spikes_plot=100,
        t_range=(30, 90),
        num_rows=3,
        alpha=0.1,
        delta_frames=12,
        num_close_clusters=5,
    )

    fig.suptitle(f"GT unit {gt_unit}. {new_str}")

    return fig, gt_ptp


def array_scatter_vs(scatter_comparison, vs_comparison):
    fig, axes = scatter_comparison.new_sorting.array_scatter()
    scatter_match = scatter_comparison.gt_matched
    vs_match = vs_comparison.gt_matched
    match = scatter_match + 2 * vs_match
    colors = ["k", "b", "r", "purple"]

    gt_x, gt_z, gt_ptp = scatter_comparison.gt_sorting.template_xzptp.T
    log_gt_ptp = np.log(gt_ptp)

    ls = []
    for i, c in enumerate(colors):
        matchix = match == i
        gtxix = gt_x[matchix]
        gtzix = gt_z[matchix]
        gtpix = log_gt_ptp[matchix]
        axes[0].scatter(gtxix, gtzix, color=c, marker="x", s=15)
        axes[2].scatter(gtxix, gtzix, color=c, marker="x", s=15)
        l = axes[1].scatter(gtpix, gtzix, color=c, marker="x", s=15)
        ls.append(l)

    leg_artist = plt.figlegend(
        ls,
        [
            "no match",
            f"{scatter_comparison.new_sorting.name} match",
            f"{vs_comparison.new_sorting.name} match",
            "both",
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        borderaxespad=-10,
    )

    return fig, axes, leg_artist
