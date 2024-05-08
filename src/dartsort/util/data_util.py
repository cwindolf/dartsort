from collections import namedtuple
from dataclasses import dataclass, field, replace
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
            if isinstance(self.parent_h5_path, str):
                self.parent_h5_path = Path(self.parent_h5_path).absolute()

        self.channels = np.asarray(self.channels, dtype=int)
        assert self.times_samples.shape == self.channels.shape

        unit_ids = np.unique(self.labels)
        self.unit_ids = unit_ids[unit_ids >= 0]

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
            with h5py.File(parent_h5_path, "r", libver="latest", locking=False) as h5:
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
        return self.unit_ids.size

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
        with h5py.File(peeling_hdf5_filename, "r", libver="latest", locking=False) as h5:
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
            parent_h5_path=str(peeling_hdf5_filename),
            extra_features=extra_features,
        )


    @classmethod
    def from_list_peeling_hdf5(
        cls,
        list_peeling_hdf5_filename,
        times_samples_dataset="times_samples",
        channels_dataset="channels",
        labels_dataset="labels",
        load_simple_features=True,
        labels=None,
        chunk_time_ranges_s=None,
    ):
        sorting_list = []
        for peeling_hdf5_filename in list_peeling_hdf5_filename:
            sorting_list.append(DARTsortSorting.from_peeling_hdf5(
                peeling_hdf5_filename, 
                times_samples_dataset=times_samples_dataset,
                channels_dataset=channels_dataset,
                labels_dataset=labels_dataset,
                load_simple_features=load_simple_features,
                labels=labels,
            ))
        return combine_chunked_sortings(sorting_list, chunk_time_ranges_s=chunk_time_ranges_s)




def keep_only_most_recent_spikes(
    sorting,
    n_min_spikes=250,
    latest_time_sample=90_000_000,
):
    """
    This function selects the n_min_spikes before latest_time (or most recent after latest_time)
    """
    new_labels = np.full(sorting.labels.shape, -1)
    units = np.unique(sorting.labels)
    units = units[units > -1]
    for k in units:
        idx_k = np.flatnonzero(sorting.labels == k)
        before_time = sorting.times_samples[idx_k] < latest_time_sample
        if before_time.sum() <= n_min_spikes:
            idx_k = idx_k[:n_min_spikes]
            new_labels[idx_k] = k
        else:
            idx_k = idx_k[before_time][-n_min_spikes:]
            new_labels[idx_k] = k
    new_sorting = replace(sorting, labels=new_labels)
    return new_sorting

def update_sorting_chunk_spikes_loaded(
    sorting_chunk,
    idx_spikes,
    n_min_spikes=250,
    latest_time_sample=90_000_000,
):
    """
    This function selects the n_min_spikes before latest_time from a sorting that has been updated (by merge)
    """

    new_labels_chunks = np.full(sorting_chunk.labels.shape, -1)
    new_labels_chunks[idx_spikes] = sorting_chunk.labels[idx_spikes]
    sorting_chunk = replace(sorting_chunk, labels=new_labels_chunks)

    new_labels = np.full(sorting_chunk.labels.shape, -1)
    units = np.unique(sorting_chunk.labels)
    units = units[units > -1]
    for k in units:
        idx_k = np.flatnonzero(sorting_chunk.labels==k)
        before_time = sorting_chunk.times_samples[idx_k] < latest_time_sample
        if before_time.sum() <= n_min_spikes:
            idx_k = idx_k[:n_min_spikes]
            new_labels[idx_k] = k
        else:
            idx_k = idx_k[before_time][-n_min_spikes:]
            new_labels[idx_k] = k
    new_sorting = replace(sorting_chunk, labels=new_labels)
    return new_sorting


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

    new_labels = np.where(np.isin(sorting.labels, small_units), -1, sorting.labels)

    return replace(sorting, labels=new_labels)


def subset_sorting_by_time_samples(
    sorting, start_sample=0, end_sample=np.inf, reference_to_start_sample=True
):
    new_times = sorting.times_samples.copy()
    new_labels = sorting.labels.copy()

    in_range = (new_times >= start_sample) & (new_times < end_sample)
    new_labels[~in_range] = -1

    if reference_to_start_sample:
        new_times -= start_sample

    return replace(sorting, labels=new_labels, times_samples=new_times)


def subset_sorting_by_time_seconds(
    sorting, t_start=0, t_end=np.inf
):
    new_labels = sorting.labels.copy()
    t_s = sorting.times_seconds
    in_range = t_s == t_s.clip(t_start, t_end)
    new_labels[~in_range] = -1

    return replace(sorting, labels=new_labels)


def time_chunk_sortings(
    sorting, recording=None, chunk_length_samples=None, chunk_time_ranges_s=None
):
    if chunk_time_ranges_s is None:
        chunk_time_ranges_s = chunk_time_ranges(recording, chunk_length_samples)
    chunk_sortings = [
        subset_sorting_by_time_seconds(sorting, *tt) for tt in chunk_time_ranges_s
    ]
    return chunk_time_ranges_s, chunk_sortings


def reindex_sorting_labels(sorting):
    new_labels = sorting.labels.copy()
    kept = np.flatnonzero(new_labels >= 0)
    _, new_labels[kept] = np.unique(new_labels[kept], return_inverse=True)
    return replace(sorting, labels=new_labels)


def combine_sortings(sortings, dodge=False):
    labels = np.full_like(sortings[0].labels, -1)
    times_samples = sortings[0].times_samples.copy()
    assert all(s.labels.size == sortings[0].labels.size for s in sortings)

    if dodge:
        label_to_sorting_index = []
        label_to_original_label = []
    next_label = 0
    for j, sorting in enumerate(sortings):
        kept = np.flatnonzero(sorting.labels >= 0)
        assert np.all(labels[kept] < 0)
        labels[kept] = sorting.labels[kept] + next_label
        if dodge:
            n_new_labels = 1 + sorting.labels[kept].max()
            next_label += n_new_labels
            label_to_sorting_index.append(np.full(n_new_labels, j))
            label_to_original_label.append(np.arange(n_new_labels))
        times_samples[kept] = sorting.times_samples[kept]

    sorting = replace(sortings[0], labels=labels, times_samples=times_samples)

    if dodge:
        label_to_sorting_index = np.array(label_to_sorting_index)
        label_to_original_label = np.array(label_to_original_label)
        return label_to_sorting_index, label_to_original_label, sorting
    return sorting

def combine_chunked_sortings(sortings, n_spikes=None, chunk_time_ranges_s=None):
    """
    This function combines sortings that contain info about non overlapping chunks of data 
    """
    if n_spikes is None:
        n_spikes=0
        if chunk_time_ranges_s is None:
            for sorting in sortings:
                n_spikes+= sorting.labels.size
        else:
            for sorting, chunk_time_range in zip(sortings, chunk_time_ranges_s):
                idx = np.flatnonzero(np.logical_and(
                    sorting.times_seconds >= chunk_time_range[0],
                    sorting.times_seconds <= chunk_time_range[1]
                ))
                n_spikes += len(idx)
    print(f"N SPIKES {n_spikes}")
            
    sorting_final = DARTsortSorting(
        times_samples=np.zeros(n_spikes),
        channels=np.zeros(n_spikes),
        labels=np.zeros(n_spikes),
        sampling_frequency=sortings[0].sampling_frequency,
        parent_h5_path=None,
    )

    labels = np.full(n_spikes, -1, sortings[0].labels.dtype)
    times_samples = np.full(n_spikes, -1, sortings[0].times_samples.dtype)
    channels = np.full(n_spikes, -1, sortings[0].channels.dtype)
    extra_features={}

    list_parent_h5_path = []

    cmp = 0
    if chunk_time_ranges_s is None:
        for j, sorting in enumerate(sortings):
            list_parent_h5_path.append(sorting.parent_h5_path)
            labels[cmp:cmp + sorting.labels.size] = sorting.labels
            times_samples[cmp:cmp + sorting.labels.size] = sorting.times_samples
            channels[cmp:cmp + sorting.labels.size] = sorting.channels
            cmp+=sorting.labels.size
    
        for feat in sortings[0].extra_features:
            if sortings[0].extra_features[feat].ndim==1:   
                feat_array = np.full(n_spikes, -1, sortings[0].extra_features[feat].dtype)
            else:
                feat_array = np.full((n_spikes, sortings[0].extra_features[feat].shape[1]), -1, sortings[0].extra_features[feat].dtype)
            cmp = 0
            for j, sorting in enumerate(sortings):
                feat_array[cmp:cmp + sorting.labels.size] = sorting.extra_features[feat]
                cmp+=sorting.labels.size
            extra_features[feat]=feat_array
    else:
        for j, sorting in enumerate(sortings):
            chunk_time_range = chunk_time_ranges_s[j]
            idx = np.flatnonzero(np.logical_and(
                sorting.times_seconds >= chunk_time_range[0],
                sorting.times_seconds <= chunk_time_range[1]
            ))
            list_parent_h5_path.append(sorting.parent_h5_path)
            labels[cmp:cmp + len(idx)] = sorting.labels[idx]
            times_samples[cmp:cmp + len(idx)] = sorting.times_samples[idx]
            channels[cmp:cmp + len(idx)] = sorting.channels[idx]
            cmp+=len(idx)
    
        for feat in sortings[0].extra_features:
            if sortings[0].extra_features[feat].ndim==1:   
                feat_array = np.full(n_spikes, -1, sortings[0].extra_features[feat].dtype)
            else:
                feat_array = np.full((n_spikes, sortings[0].extra_features[feat].shape[1]), -1, sortings[0].extra_features[feat].dtype)
            cmp = 0
            for j, sorting in enumerate(sortings):
                chunk_time_range = chunk_time_ranges_s[j]
                idx = np.flatnonzero(np.logical_and(
                    sorting.times_seconds >= chunk_time_range[0],
                    sorting.times_seconds <= chunk_time_range[1]
                ))
                feat_array[cmp:cmp + len(idx)] = sorting.extra_features[feat][idx]
                cmp+=len(idx)
            extra_features[feat]=feat_array
    
    sorting_final = DARTsortSorting(
        times_samples=times_samples,
        channels=channels,
        labels=labels,
        sampling_frequency=sortings[0].sampling_frequency,
        parent_h5_path=list_parent_h5_path,
        extra_features=extra_features,
    )

    return sorting_final


# -- timing

def chunk_time_ranges(recording, chunk_length_samples=None, slice_s=None, divider_samples=None):
    if chunk_length_samples is None or chunk_length_samples == np.inf:
        n_chunks = 1
    elif slice is None:
        n_chunks = recording.get_num_samples() / chunk_length_samples
        # we'll count the remainder as a chunk if it's at least 2/3 of one
        n_chunks = np.floor(n_chunks) + (n_chunks - np.floor(n_chunks) > 0.66)
        n_chunks = int(max(1, n_chunks))
    else:
        if slice_s[0] is None:
            slice_s[0] = 0.
        if slice_s[1] is None:
            slice_s[1] = recording.get_num_samples() / recording.sampling_frequency
        n_chunks = (slice_s[1] - slice_s[0]) * recording.sampling_frequency/ chunk_length_samples 
        # we'll count the remainder as a chunk if it's at least 2/3 of one
        n_chunks = np.floor(n_chunks) + (n_chunks - np.floor(n_chunks) > 0.66)
        n_chunks = int(max(1, n_chunks))

    # evenly divide the recording into chunks
    assert recording.get_num_segments() == 1
    if slice is None:
        start_time_s, end_time_s = recording._recording_segments[
            0
        ].sample_index_to_time(np.array([0, recording.get_num_samples() - 1]))
        chunk_times_s = np.linspace(start_time_s, end_time_s, num=n_chunks + 1)
    else:
        chunk_times_s = np.linspace(slice_s[0], slice_s[1], num=n_chunks + 1)
    if divider_samples is not None:
        divider_s = divider_samples / recording.sampling_frequency
        chunk_times_s[:-1] = (chunk_times_s[:-1] // divider_s) * divider_s
    chunk_time_ranges_s = list(zip(chunk_times_s[:-1], chunk_times_s[1:]))
        
    return chunk_time_ranges_s

def subchunks_time_ranges(recording, chunk_range_s, subchunk_size_s, divider_samples=None):
    n_chunks = (chunk_range_s[1] - chunk_range_s[0]) / subchunk_size_s 
    n_chunks = np.floor(n_chunks) + (n_chunks - np.floor(n_chunks) > 0.66)
    n_chunks = int(max(1, n_chunks))
    chunk_times_s = np.linspace(chunk_range_s[0], chunk_range_s[1], num=n_chunks + 1)
    if divider_samples is not None: 
        divider_s = divider_samples / recording.sampling_frequency
        chunk_times_s[:-1] = (chunk_times_s[:-1] // divider_s) * divider_s
    return list(zip(chunk_times_s[:-1], chunk_times_s[1:]))

# -- hdf5 util


def batched_h5_read(dataset, indices, batch_size=128):
    if indices.size < batch_size:
        return dataset[indices]
    else:
        out = np.empty((indices.size, *dataset.shape[1:]), dtype=dataset.dtype)
        for bs in range(0, indices.size, batch_size):
            be = min(indices.size, bs + batch_size)
            out[bs:be] = dataset[indices[bs:be]]
        return out
