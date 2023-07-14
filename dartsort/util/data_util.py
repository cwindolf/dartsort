from collections import namedtuple

import h5py
import numpy as np
from spikeinterface.core import NumpySorting

# this is a data type used in the peeling code to store info about
# the datasets which are being computed
# the featurizers in transform have a .spike_dataset property which
# is this type
SpikeDataset = namedtuple("SpikeDataset", ["name", "shape_per_spike", "dtype"])


class DARTsortSorting:
    """Class which holds spike times, channels, and labels

    This class holds our algorithm state.
    Initially the sorter doesn't have unit labels, so these are optional.
    Export me to a SpikeInterface NumpySorting with .to_numpy_sorting()
    """

    def __init__(self, times, channels, labels=None):
        """
        Arguments
        ---------
        times : np.array
            Array of spike times in samples
        channels : np.array
            Array of spike detection channel indices
        labels : optional, np.array
            Array of unit labels
        """
        self.times = np.array(times, dtype=int)
        self.channels = channels
        self.labels = labels
        if labels is None:
            self.labels = np.zeros_like(times)

        if channels is not None:
            self.channels = np.array(channels, dtype=int)
            assert self.times.shape == self.channels.shape

    def to_numpy_sorting(self):
        return NumpySorting.from_times_labels(
            times=self.times, labels=self.labels
        )

    def __str__(self):
        name = self.__class__.__name__
        nspikes = self.times.size
        labels_str = "unlabeled" if self.labels is None else "labeled"
        return f"{name}: {nspikes} {labels_str} spikes."

    @classmethod
    def from_peeling_hdf5(
        cls,
        peeling_hdf5_filename,
        times_dataset="times",
        channels_dataset="channels",
        labels_dataset="labels",
    ):
        channels = labels = None
        with h5py.File(peeling_hdf5_filename, "r") as h5:
            times = h5[times_dataset][()]
            if channels_dataset in h5:
                channels = h5[channels_dataset][()]
            if labels_dataset in h5:
                labels = h5[labels_dataset][()]
        return cls(times, channels=channels, labels=labels)
