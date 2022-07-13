"""Hybrid analysis helpers

These are just a couple of classes which are basically bags of
data that would be used when making summary plots, and which compute
some metrics etc in the constructor. They help make wrangling a bunch
of sorts a little easier.
"""
import string
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from spikeinterface.extractors import NumpySorting
from spikeinterface.comparison import compare_sorter_to_ground_truth

from spike_psvae.deconvolve import get_templates
from spike_psvae import localize_index


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
    ):
        n_spikes_full = spike_labels.shape[0]
        assert spike_labels.shape == spike_times.shape == (n_spikes_full,)

        self.name = name
        self.name_lo = name.lower().replace(string.whitespace + string.punctuation, "_")

        which = np.flatnonzero(spike_labels >= 0)
        which = which[np.argsort(spike_times[which])]

        self.unsorted = unsorted
        self.raw_bin = raw_bin
        self.spike_times = spike_times[which]
        self.spike_labels = spike_labels[which]
        self.unit_labels, self.unit_spike_counts = np.unique(self.spike_labels, return_counts=True)
        self.contiguous_labels = self.unit_labels.size == self.unit_labels.max() + 1

        self.np_sorting = NumpySorting.from_times_labels(
            times_list=self.spike_times,
            labels_list=self.spike_labels,
            sampling_frequency=30_000,
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
                self.template_ptps.max(1),
            ]

        if spike_maxchans is None:
            assert not unsorted
            print("Sorting", name, "has no intrinsic maxchans. Using template maxchans.")
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
            
class HybridComparison:
    def __init__(self, gt_sorting, new_sorting):
        assert gt_sorting.contiguous_labels
        
        self.gt_sorting = gt_sorting
        self.new_sorting = new_sorting
        self.unsorted = new_sorting.unsorted
        
        self.average_performance = self.weighted_average_performance = _na_avg_performance
        if not new_sorting.unsorted:
            self.gt_comparison = compare_sorter_to_ground_truth(
                gt_sorting.np_sorting,
                new_sorting.np_sorting,
                gt_name=gt_sorting.name,
                tested_name=new_sorting.name,
                sampling_frequency=30_000,
                exhaustive_gt=False,
                match_score=0.1,
                verbose=True,
            )

            # matching units and accuracies
            self.performance_by_unit = self.gt_comparison.get_performance()
            # average the metrics over units
            self.average_performance = self.gt_comparison.get_performance(
                method="pooled_with_average"
            )
            # average metrics, weighting each unit by its spike count
            self.weighted_average_performance = (
                (self.performance_by_unit * gt_sorting.unit_spike_counts[:, None]).sum(0)
                / gt_sorting.unit_spike_counts.sum()
            )

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


# -- library


def unsorted_confusion(gt_spike_index, new_spike_index, n_samples=12, n_channels=4):
    cmul = n_samples / n_channels
    n_gt_spikes = len(gt_spike_index)
    n_new_spikes = len(gt_spike_index)

    gt_kdt = KDTree(np.c_[gt_spike_index[:, 0], gt_spike_index[:, 1] * cmul])
    sorter_kdt = KDTree(np.c_[new_spike_index[:, 0], cmul * new_spike_index[:, 1]])
    query = gt_kdt.query_ball_tree(sorter_kdt, n_samples + 0.1)

    # this is a boolean array of length n_gt_spikes
    detected = np.array([len(lst) > 0 for lst in query], dtype=bool)
    
    # from the above, we can compute a couple of metrics
    true_positives = detected.sum()
    false_negatives = n_gt_spikes - true_positives
    false_positives = n_new_spikes - true_positives

    return true_positives, false_negatives, false_positives, n_gt_spikes

_na_avg_performance = pd.Series(
    index=["accuracy", "recall", "precision", "false_discovery_rate", "miss_rate"],
    data=[np.nan] * 5,
)


# -- plotting helpers


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
