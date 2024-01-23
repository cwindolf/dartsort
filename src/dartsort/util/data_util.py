from collections import namedtuple
from dataclasses import dataclass, field, replace
from typing import Optional
from warnings import warn

import h5py
import numpy as np
import torch
from dartsort.detect import detect_and_deduplicate
from spikeinterface.core import NumpySorting, get_random_data_chunks

from .waveform_util import make_channel_index

# this is a data type used in the peeling code to store info about
# the datasets which are being computed
# the featurizers in transform have a .spike_dataset property which
# is this type
SpikeDataset = namedtuple("SpikeDataset", ["name", "shape_per_spike", "dtype"])


@dataclass
class DARTsortSorting:
    """Class which holds spike times, channels, and labels

    This class holds our algorithm state.
    Initially the sorter doesn't have unit labels, so these are optional.
    Export me to a SpikeInterface NumpySorting with .to_numpy_sorting()

    When you instantiate this with from_peeling_hdf5, if the
    flag load_simple_features is True (default), then additional
    features of spikes will be loaded into memory -- like localizations,
    which you can access like `sorting.point_source_localizations[...]`
    If you instantiate with __init__(), you can pass these in with
    `extra_features`, and they will also be available as .properties.
    """

    times_samples: np.ndarray
    channels: np.ndarray
    labels: Optional[np.ndarray] = None
    sampling_frequency: Optional[float] = 30_000.0
    # entries in this dictionary will also be set as properties
    extra_features: Optional[dict[str, np.ndarray]] = None

    # automatically set
    n_spikes: int = field(init=False)

    def __post_init__(self):
        self.times_samples = np.asarray(self.times_samples, dtype=int)

        if self.labels is None:
            self.labels = np.zeros_like(self.times_samples)
        self.labels = np.asarray(self.labels, dtype=int)

        self.channels = np.asarray(self.channels, dtype=int)
        assert self.times_samples.shape == self.channels.shape

        if self.extra_features:
            for k in self.extra_features:
                v = self.extra_features[k] = np.asarray(self.extra_features[k])
                assert v.shape[0] == len(self.times_samples)
                assert not hasattr(self, k)
                self.__dict__[k] = v

        self.n_spikes = self.times_samples.size

    def to_numpy_sorting(self):
        return NumpySorting.from_times_labels(
            times_list=self.times_samples,
            labels_list=self.labels,
            sampling_frequency=self.sampling_frequency,
        )

    def __str__(self):
        name = self.__class__.__name__
        nspikes = self.times_samples.size
        nunits = (np.unique(self.labels) >= 0).sum()
        unit_str = f"{nunits} unit" + "s" * (nunits > 1)
        feat_str = ""
        if self.extra_features:
            feat_str = ", ".join(self.extra_features.keys())
            feat_str = f" extra features: {feat_str}."
        return f"{name}: {nspikes} spikes, {unit_str}.{feat_str}"

    def __repr__(self):
        return str(self)

    def __len__(self):
        return self.n_spikes

    @classmethod
    def from_peeling_hdf5(
        cls,
        peeling_hdf5_filename,
        times_samples_dataset="times_samples",
        channels_dataset="channels",
        labels_dataset="labels",
        load_simple_features=True,
        labels=None,
    ):
        channels = None
        with h5py.File(peeling_hdf5_filename, "r") as h5:
            times_samples = h5[times_samples_dataset][()]
            sampling_frequency = h5["sampling_frequency"][()]
            if channels_dataset in h5:
                channels = h5[channels_dataset][()]
            if labels_dataset in h5 and labels is None:
                labels = h5[labels_dataset][()]

            n_spikes = len(times_samples)
            extra_features = None
            if load_simple_features:
                extra_features = {}
                loaded = (
                    times_samples_dataset,
                    channels_dataset,
                    labels_dataset,
                )
                for k in h5:
                    if (
                        k not in loaded
                        and 1 <= h5[k].ndim <= 2
                        and h5[k].shape[0] == n_spikes
                    ):
                        extra_features[k] = h5[k][:]

        return cls(
            times_samples,
            channels=channels,
            labels=labels,
            sampling_frequency=sampling_frequency,
            extra_features=extra_features,
        )


def check_recording(
    rec,
    threshold=5,
    dedup_spatial_radius=75,
    expected_value_range=1e4,
    expected_spikes_per_sec=10_000,
    num_chunks_per_segment=5,
):
    """Sanity check spike detection rate and data range of input recording."""

    # grab random traces from throughout rec
    random_chunks = get_random_data_chunks(
        rec,
        num_chunks_per_segment=num_chunks_per_segment,
        chunk_size=int(rec.sampling_frequency),
        concatenated=False,
    )
    dedup_channel_index = None
    if dedup_spatial_radius:
        dedup_channel_index = make_channel_index(
            rec.get_channel_locations(), dedup_spatial_radius
        )
    failed = False

    # run detection and compute spike detection rate and data range
    spike_rates = []
    for chunk in random_chunks:
        times, _ = detect_and_deduplicate(
            torch.tensor(chunk),
            threshold=threshold,
            peak_sign="both",
            dedup_channel_index=torch.tensor(dedup_channel_index),
        )
        spike_rates.append(times.shape[0])

    avg_detections_per_second = np.mean(spike_rates)
    max_abs = np.max(random_chunks)

    if avg_detections_per_second > expected_spikes_per_sec:
        warn(
            f"Detected {avg_detections_per_second:0.1f} spikes/s, which is "
            "large. You may want to check that your data has been preprocessed, "
            "including standardization. If it seems right, then you may need to "
            "shrink the chunk_length_samples parameters in the configuration if "
            "you experience memory issues.",
            RuntimeWarning,
        )
        failed = True

    if max_abs > expected_value_range:
        warn(
            f"Recording values exceed |{expected_value_range}|. You may want to "
            "check that your data has been preprocessed, including standardization.",
            RuntimeWarning,
        )
        failed = True

    return failed, avg_detections_per_second, max_abs


def subset_sorting_by_spike_count(sorting, min_spikes=0):
    if not min_spikes:
        return sorting

    units, counts = np.unique(sorting.labels, return_counts=True)
    small_units = units[counts < min_spikes]

    new_labels = np.where(
        np.isin(sorting.labels, small_units), -1, sorting.labels
    )

    return replace(sorting, labels=new_labels)


def subset_sorting_by_time_samples(
    sorting, start_sample=0, end_sample=np.inf, reference_to_start_sample=True
):
    new_times = sorting.times_samples.copy()
    new_labels = sorting.labels.copy()

    in_range = (new_times >= start_sample) & (new_times < end_sample)
    print(in_range.sum())
    new_labels[~in_range] = -1

    if reference_to_start_sample:
        new_times -= start_sample

    return replace(sorting, labels=new_labels, times_samples=new_times)


def reindex_sorting_labels(sorting):
    new_labels = sorting.labels.copy()
    kept = np.flatnonzero(new_labels >= 0)
    _, new_labels[kept] = np.unique(new_labels[kept], return_inverse=True)
    return replace(sorting, labels=new_labels)
