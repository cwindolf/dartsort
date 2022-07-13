import numpy as np
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
    ):
        n_spikes_full = spike_labels.shape[0]
        assert spike_labels.shape == spike_times.shape == (n_spikes_full,)

        self.name = name

        which = np.flatnonzero(spike_labels >= 0)
        which = which[np.argsort(spike_times[which])]

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
        if templates is None:
            self.templates = get_templates(
                raw_bin,
                self.spike_times[:, None],
                self.spike_labels,
                geom,
            )
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
            self.template_locs[:, 0],
            self.template_locs[:, 3],
            self.template_ptps.max(1),
        ]

        if spike_maxchans is None:
            print("Sorting", name, "has no intrinsic maxchans. Using template maxchans.")
            assert spike_maxchans.shape == (n_spikes_full,)
            self.spike_maxchans = self.template_maxchans[self.spike_labels]
        else:
            self.spike_maxchans = spike_maxchans[which]

        self.spike_index = np.c_[self.spike_times, self.spike_maxchans]
        self.spike_train = np.c_[self.spike_times, self.spike_labels]
        self.n_spikes = len(self.spike_index)

        self.spike_xzptp = None
        if spike_xzptp is not None:
            assert spike_xzptp.shape == (n_spikes_full, 3)
            self.spike_xzptp = spike_xzptp[which]


class HybridSorting:
    def __init__(self, gt_sorting, new_sorting):
        assert gt_sorting.contiguous_labels

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
        self.performance = self.gt_comparison.get_performance()
