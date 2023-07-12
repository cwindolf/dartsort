from collections import namedtuple

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
