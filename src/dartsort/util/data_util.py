from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
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
    parent_h5_path: Optional[str] = None
    extra_features: Optional[dict[str, np.ndarray]] = None

    def __post_init__(self):
        self.times_samples = np.asarray(self.times_samples, dtype=int)

        if self.labels is None:
            self.labels = np.zeros_like(self.times_samples)
        self.labels = np.asarray(self.labels, dtype=int)
        self._n_units = None
        if self.parent_h5_path is not None:
            self.parent_h5_path = Path(self.parent_h5_path).absolute()

        self.channels = np.asarray(self.channels, dtype=int)
        assert self.times_samples.shape == self.channels.shape

        if self.extra_features:
            for k in self.extra_features:
                v = self.extra_features[k] = np.asarray(self.extra_features[k])
                assert v.shape[0] == len(self.times_samples)
                assert not hasattr(self, k)
                self.__dict__[k] = v

    def to_numpy_sorting(self):
        return NumpySorting.from_times_labels(
            times_list=self.times_samples,
            labels_list=self.labels,
            sampling_frequency=self.sampling_frequency,
        )

    def save(self, sorting_npz):
        data = dict(
            times_samples=self.times_samples,
            channels=self.channels,
            labels=self.labels,
            sampling_frequency=self.sampling_frequency,
        )
        if self.parent_h5_path:
            data["parent_h5_path"] = np.array(str(self.parent_h5_path))
            data["feature_keys"] = np.array(list(self.extra_features.keys()))
        np.savez(sorting_npz, **data)

    @classmethod
    def load(cls, sorting_npz):
        with np.load(sorting_npz) as data:
            times_samples = data["times_samples"]
            channels = data["channels"]
            labels = data["labels"]
            sampling_frequency = data["sampling_frequency"]
            parent_h5_path = feature_keys = None
            if "parent_h5_path" in data:
                parent_h5_path = str(data["parent_h5_path"])
                feature_keys = list(map(str, data["feature_keys"]))

        extra_features = None
        if parent_h5_path:
            with h5py.File(parent_h5_path, "r") as h5:
                extra_features = {k: h5[k][()] for k in feature_keys}

        return cls(
            times_samples=times_samples,
            channels=channels,
            labels=labels,
            sampling_frequency=sampling_frequency,
            parent_h5_path=parent_h5_path,
            extra_features=extra_features,
        )

    @property
    def n_spikes(self):
        return self.times_samples.size

    @property
    def n_units(self):
        if self._n_units is None:
            u = np.unique(self.labels)
            self._n_units = (u >= 0).sum()
        return self._n_units

    def __str__(self):
        name = self.__class__.__name__
        ns = self.n_spikes
        nu = self.n_units
        unit_str = f"{nu} unit" + "s" * (nu > 1)
        feat_str = ""
        if self.extra_features:
            feat_str = ", ".join(self.extra_features.keys())
            feat_str = f" extra features: {feat_str}."
        h5_str = ""
        if self.parent_h5_path:
            h5_str = f" from parent h5 file {self.parent_h5_path}."
        return f"{name}: {ns} spikes, {unit_str}.{feat_str}{h5_str}"

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
                loaded = (times_samples_dataset, channels_dataset, labels_dataset)
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
            parent_h5_path=str(peeling_hdf5_filename),
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
