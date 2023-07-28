from collections import namedtuple

import h5py
import numpy as np
from spikeinterface.core import NumpySorting, get_random_data_chunks
from dartsort.detect import detect_and_deduplicate
from warnings import warn
import torch

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

    def __init__(self, times, channels, labels=None, sampling_frequency=30000):
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
        self.sampling_frequency = sampling_frequency
        if labels is None:
            self.labels = np.zeros_like(times)

        if channels is not None:
            self.channels = np.array(channels, dtype=int)
            assert self.times.shape == self.channels.shape

    def to_numpy_sorting(self):
        return NumpySorting.from_times_labels(
            times_list=self.times,
            labels_list=self.labels,
            sampling_frequency=self.sampling_frequency,
        )

    def __str__(self):
        name = self.__class__.__name__
        nspikes = self.times.size
        units = np.unique(self.labels)
        units = units[units >= 0]
        unit_str = f"{units.size} unit" + ("s" if units.size > 1 else "")
        return f"{name}: {nspikes} spikes, {unit_str}."

    def __len__(self):
        return self.times.size

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
            sampling_frequency = h5["sampling_frequency"][()]
            if channels_dataset in h5:
                channels = h5[channels_dataset][()]
            if labels_dataset in h5:
                labels = h5[labels_dataset][()]
        return cls(
            times,
            channels=channels,
            labels=labels,
            sampling_frequency=sampling_frequency,
        )


def check_recording(rec, threshold=5, 
                    expected_value_range=1e4,
                    expected_spikes_per_sec=1e4):
    """
    Sanity check spike detection rate and data range of input recording.
    """
    
    # grab random traces from throughout rec
    random_chunks = get_random_data_chunks(rec, num_chunks_per_segment=5,
                                           chunk_size=int(rec.sampling_frequency),
                                           concatenated=False)
    
    # run detection and compute spike detection rate and data range
    spike_rates = []
    for chunk in random_chunks:
        times, _ = detect_and_deduplicate(torch.tensor(chunk), threshold=threshold,
                                                 peak_sign="both")
        spike_rates.append(times.shape[0])

    avg_detections_per_second = sum(spike_rates) / 5
    max_abs = np.max(random_chunks)

    if avg_detections_per_second > expected_spikes_per_sec:
        warn(f"Average spike detections per second: {avg_detections_per_second}."
            "Running on a full dataset could lead to an out-of-memory error."
            "(Is this data normalized?)", RuntimeWarning)

    if max_abs > expected_value_range:
        warn(f"Data range exceeds |1e4|.", RuntimeWarning)

    return avg_detections_per_second, max_abs
