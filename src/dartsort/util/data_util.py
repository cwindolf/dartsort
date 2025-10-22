from collections import namedtuple
from dataclasses import dataclass, replace
from logging import getLogger
from pathlib import Path
from typing import Generator
import warnings

import h5py
import numpy as np
import torch
from spikeinterface.core import NumpySorting, get_random_data_chunks
from tqdm.auto import tqdm

from ..detect import detect_and_deduplicate
from ..util.py_util import resolve_path
from .waveform_util import make_channel_index

logger = getLogger(__name__)

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
    labels: np.ndarray | None = None
    sampling_frequency: float = 30000.0
    parent_h5_path: str | None = None

    # entries in this dictionary will also be set as properties
    extra_features: dict[str, np.ndarray] | None = None

    def __post_init__(self):
        self.times_samples = np.asarray(self.times_samples, dtype=np.int64)
        n_spikes = self.times_samples.shape[0]
        assert self.times_samples.ndim == 1

        if self.labels is None:
            self.labels = np.zeros_like(self.times_samples)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        assert self.labels.shape == (n_spikes,)
        self._n_units = None
        if self.parent_h5_path is not None:
            self.parent_h5_path = Path(self.parent_h5_path).absolute()

        self.channels = np.asarray(self.channels, dtype=np.int64)
        assert self.times_samples.shape == self.channels.shape

        unit_ids = np.unique(self.labels)
        self.unit_ids = unit_ids[unit_ids >= 0]

        if self.extra_features:
            for k in self.extra_features:
                v = self.extra_features[k] = np.asarray(self.extra_features[k])
                if not (k.endswith("channel_index") or k == "geom"):
                    if not v.shape[0] == n_spikes:
                        raise ValueError(
                            f"Feature {k} has strange shape {v.shape}, since {n_spikes=}."
                        )
                assert not hasattr(self, k)
                self.__dict__[k] = v

    def setattr(self, name, value):
        if torch.is_tensor(value):
            value = value.numpy(force=True)
        if isinstance(value, np.ndarray):
            assert len(value) == len(self)
            self.extra_features[name] = value
            self.__dict__[name] = value
        else:
            super().__setattr__(name, value)

    def to_numpy_sorting(self):
        return NumpySorting.from_samples_and_labels(
            samples_list=self.times_samples,
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
        elif self.extra_features:
            data.update(self.extra_features)
            data["feature_keys"] = np.array(list(self.extra_features.keys()))
        np.savez(sorting_npz, **data)

    def mask(self, mask):
        if np.dtype(mask.dtype).kind == "b":
            mask = mask.nonzero()

        extra_features = dict(mask_indices=mask)
        if self.extra_features:
            n = self.n_spikes
            for k, v in self.extra_features.items():
                assert k != "mask_indices"  # no recursion...
                if v.shape[0] != n:
                    continue
                extra_features[k] = v[mask]

        return replace(
            self,
            times_samples=self.times_samples[mask],
            channels=self.channels[mask],
            labels=self.labels[mask] if self.labels is not None else None,
            parent_h5_path=self.parent_h5_path,
            extra_features=extra_features,
        )

    def drop_missing(self):
        valid = np.flatnonzero(self.labels >= 0)
        return self.mask(valid)

    @classmethod
    def load(cls, sorting_npz, feature_keys=None):
        extra_features = None
        with np.load(sorting_npz) as data:
            times_samples = data["times_samples"]
            channels = data["channels"]
            labels = data["labels"]
            sampling_frequency = data["sampling_frequency"]
            parent_h5_path = None
            if "parent_h5_path" in data:
                parent_h5_path = str(data["parent_h5_path"])
                if feature_keys is None:
                    feature_keys = list(map(str, data["feature_keys"]))
            elif "feature_keys" in data and feature_keys is None:
                feature_keys = list(map(str, data["feature_keys"]))
                extra_features = {k: data[k] for k in feature_keys}

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
            feat_str = f" Extras: {feat_str}."
        h5_str = ""
        if self.parent_h5_path:
            h5_str = f" From HDF5 file {self.parent_h5_path}."
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
        load_feature_names=None,
        simple_feature_maxshape=1000,
        load_all_features=False,
        labels=None,
    ):
        channels = None
        with h5py.File(
            peeling_hdf5_filename, "r", libver="latest", locking=False
        ) as h5:
            times_samples = h5[times_samples_dataset][()]
            sampling_frequency = h5["sampling_frequency"][()]
            if channels_dataset in h5:
                channels = h5[channels_dataset][()]
            if labels_dataset in h5 and labels is None:
                labels = h5[labels_dataset][()]

            n_spikes = len(times_samples)
            extra_features = None
            if load_simple_features or load_all_features:
                extra_features = {}
                loaded = (
                    times_samples_dataset,
                    channels_dataset,
                    labels_dataset,
                )
                if load_feature_names is None:
                    load_feature_names = h5.keys()
                else:
                    load_all_features = True
                for k in load_feature_names:
                    is_loadable = (
                        k not in loaded
                        and 1 <= h5[k].ndim
                        and h5[k].shape[0] == n_spikes
                    )
                    is_simple = (
                        is_loadable
                        and h5[k].ndim <= 2
                        and h5[k].shape[0] == n_spikes
                        and (h5[k].ndim < 2 or h5[k].shape[1] < simple_feature_maxshape)
                    )
                    if (load_all_features and is_loadable) or is_simple:
                        extra_features[k] = h5[k][()]
                    elif k not in loaded and (
                        k.endswith("channel_index") or k == "geom"
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

    def _stored_datasets(self):
        if self.parent_h5_path is None:
            return []
        with h5py.File(
            self.parent_h5_path, "r", libver="latest", locking=False
        ) as h5:
            return list(h5.keys())

    def _masked_load(self, dset, mask=None, indices=None, batch_transition=1000):
        assert self.parent_h5_path is not None
        with h5py.File(
            self.parent_h5_path, "r", libver="latest", locking=False
        ) as h5:
            if indices is not None and len(indices) < batch_transition:
                return h5[dset][indices]
            return batched_h5_read(h5[dset], mask=mask, indices=indices)

    def _chunked_map_with_indices(self, dset, fn):
        """fn should take args slice_, chunk"""
        with h5py.File(
            self.parent_h5_path, "r", libver="latest", locking=False
        ) as h5:
            g = h5[dset]
            slices_and_chunks = yield_chunks(g)

            # run once on the first chunk to figure out shapes
            s, c = next(slices_and_chunks)
            res = fn(s, c)
            assert len(res) == len(c)
            shape_per_spike = res.shape[1:]

            # allocate output and save first chunk result
            total_shape = (len(g), *shape_per_spike)
            out = np.empty(total_shape, dtype=res.dtype)
            out[s] = res
            del res

            # loop the rest
            for s, c in slices_and_chunks:
                out[s] = fn(s, c)

        return out


def get_featurization_pipeline(sorting, featurization_pipeline_pt=None):
    """Look for the pipeline in the usual place."""
    from dartsort.transform import WaveformPipeline

    if isinstance(sorting, Path):
        base_dir = sorting.parent
        stem = sorting.stem
    else:
        base_dir = sorting.parent_h5_path.parent
        stem = sorting.parent_h5_path.stem
    model_dir = base_dir / f"{stem}_models"
    if hasattr(sorting, "geom"):
        geom = sorting.geom
        channel_index = sorting.channel_index
    else:
        with h5py.File(base_dir / f"{stem}.h5", "r", locking=False) as h5:
            geom = h5["geom"][:]
            channel_index = h5["channel_index"][:]
    if featurization_pipeline_pt is None:
        featurization_pipeline_pt = model_dir / "featurization_pipeline.pt"
    if not featurization_pipeline_pt.exists():
        raise ValueError(f"No file at {featurization_pipeline_pt=}")
    pipeline = WaveformPipeline.from_state_dict_pt(
        geom, channel_index, featurization_pipeline_pt
    )
    return pipeline, featurization_pipeline_pt


def get_tpca(sorting, tpca_name="collisioncleaned_tpca_features"):
    """Look for the TemporalPCAFeaturizer in the usual place."""
    pipeline, _ = get_featurization_pipeline(sorting)
    return pipeline.get_transformer(tpca_name)


def load_stored_tsvd(sorting, tsvd_name="collisioncleaned_basis", to_sklearn=True):
    if sorting.parent_h5_path is None:
        logger.info("Couldn't load stored basis.")
        return None
    pipeline, pt_path = get_featurization_pipeline(sorting)
    tsvd = pipeline.get_transformer(tsvd_name)
    assert tsvd.name == tsvd_name
    if to_sklearn:
        tsvd = tsvd.to_sklearn()
    logger.info(
        f"Loaded stored basis from %s (%s; components shape: %s).",
        pt_path,
        tsvd_name,
        tsvd.components_.shape
    )
    return tsvd


def get_labels(h5_path):
    with h5py.File(h5_path, "r") as h5:
        return h5["labels"][:]


def get_residual_snips(h5_path):
    with h5py.File(h5_path, "r", locking=False) as h5:
        return h5["residual"][:]


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


def check_recording(
    rec,
    threshold=5,
    dedup_spatial_radius=75,
    expected_value_range=1e4,
    expected_spikes_per_sec=10_000,
    num_chunks_per_segment=5,
    dtype=torch.float,
):
    """Sanity check spike detection rate and data range of input recording."""

    # grab random traces from throughout rec
    random_chunks = get_random_data_chunks(
        rec,
        num_chunks_per_segment=num_chunks_per_segment,
        chunk_size=min(rec.get_num_samples(), int(rec.sampling_frequency)),
        concatenated=False,
    )
    dedup_channel_index = None
    if dedup_spatial_radius:
        dedup_channel_index = make_channel_index(
            rec.get_channel_locations(), dedup_spatial_radius
        )

    # run detection and compute spike detection rate and data range
    spike_rates = []
    for chunk in random_chunks:
        times, _ = detect_and_deduplicate(
            torch.tensor(chunk, dtype=dtype),
            threshold=threshold,
            peak_sign="both",
            dedup_channel_index=torch.tensor(dedup_channel_index),
        )
        chunk_len_s = rec.sampling_frequency / chunk.shape[0]
        spike_rates.append(times.shape[0] / chunk_len_s)

    avg_detections_per_second = np.mean(spike_rates)
    max_abs = np.max(random_chunks)

    failed = False
    if avg_detections_per_second > expected_spikes_per_sec:
        warnings.warn(
            f"Detected {avg_detections_per_second:0.1f} spikes/s, which is "
            "large. You may want to check that your data has been preprocessed, "
            "including standardization. If it seems right, then you may need to "
            "shrink the chunk_length_samples parameters in the configuration if "
            "you experience memory issues.",
            RuntimeWarning,
        )
        failed = True

    if max_abs > expected_value_range:
        warnings.warn(
            f"Recording values exceed |{expected_value_range}|. You may want to "
            "check that your data has been preprocessed, including standardization.",
            RuntimeWarning,
        )
        failed = True

    return failed, avg_detections_per_second, max_abs


def subset_sorting_by_spike_count(sorting, min_spikes=0, max_spikes=np.inf):
    if not min_spikes:
        return sorting

    units, counts = np.unique(sorting.labels, return_counts=True)
    invalid = np.logical_or(counts < min_spikes, counts > max_spikes)
    bad_units = units[invalid]

    new_labels = np.where(np.isin(sorting.labels, bad_units), -1, sorting.labels)

    extra_features = (sorting.extra_features or {}).copy()
    extra_features["original_labels"] = sorting.labels

    return replace(sorting, labels=new_labels, extra_features=extra_features)


def subsample_to_max_count(
    sorting, max_spikes=256, seed: int | np.random.Generator = 0, discard_triaged=False
):
    units, counts = np.unique(sorting.labels, return_counts=True)
    if counts.max() <= max_spikes:
        return sorting

    rg = np.random.default_rng(seed)
    new_labels = sorting.labels.copy()
    for u in units[counts > max_spikes]:
        in_u = np.flatnonzero(sorting.labels == u)
        new_labels[in_u] = -1
        in_u = rg.choice(in_u, size=max_spikes, replace=False)
        new_labels[in_u] = u

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


def subset_sorting_by_time_seconds(sorting, t_start=0, t_end=np.inf):
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
            n_new_labels = 0
            if kept.size:
                n_new_labels = 1 + sorting.labels[kept].max()
                next_label += n_new_labels
            label_to_sorting_index.append(np.full(n_new_labels, j))
            label_to_original_label.append(np.arange(n_new_labels))
        times_samples[kept] = sorting.times_samples[kept]

    sorting = replace(sortings[0], labels=labels, times_samples=times_samples)

    if dodge:
        label_to_sorting_index = np.concatenate(label_to_sorting_index)
        label_to_original_label = np.concatenate(label_to_original_label)
        return label_to_sorting_index, label_to_original_label, sorting
    return sorting


# -- timing


def chunk_time_ranges(recording, chunk_length_samples=None):
    if chunk_length_samples is None or chunk_length_samples == np.inf:
        n_chunks = 1
    else:
        n_chunks = recording.get_num_samples() / chunk_length_samples
        # we'll count the remainder as a chunk if it's at least 2/3 of one
        n_chunks = np.floor(n_chunks) + (n_chunks - np.floor(n_chunks) > 0.66)
        n_chunks = int(max(1, n_chunks))

    # evenly divide the recording into chunks
    assert recording.get_num_segments() == 1
    start_time_s, end_time_s = recording._recording_segments[0].sample_index_to_time(
        np.array([0, recording.get_num_samples() - 1])
    )
    chunk_times_s = np.linspace(start_time_s, end_time_s, num=n_chunks + 1)
    chunk_time_ranges_s = list(zip(chunk_times_s[:-1], chunk_times_s[1:]))

    return chunk_time_ranges_s


# -- hdf5 util


def batched_h5_read(dataset, indices=None, mask=None, show_progress=False):
    if mask is None:
        mask = np.zeros(len(dataset), dtype=bool)
        mask[indices] = 1
    return _read_by_chunk(mask, dataset, show_progress=show_progress)


def _read_by_chunk(mask, dataset, show_progress=True):
    """
    mask : boolean array of shape dataset.shape[:1]
    dataset : chunked h5py.Dataset
    """
    out = np.empty((mask.sum(), *dataset.shape[1:]), dtype=dataset.dtype)
    n = 0
    for sli, dsli in yield_chunks(dataset, show_progress):
        m = np.flatnonzero(mask[sli])
        nm = m.size
        if not nm:
            continue
        x = dsli[m]
        # x = dataset[np.arange(sli.start, sli.stop)[m]]
        out[n : n + nm] = x
        n += nm
    return out


def yield_chunks(
    dataset, show_progress=True, desc_prefix=None, fallback_chunk_length=4096
) -> Generator[tuple[slice, np.ndarray], None, None]:
    """Iterate chunks of an h5py dataset

    The dataset can either not be chunked or chunked only on the first axis.
    """
    if dataset.chunks is None:
        chunks = (
            slice(s, min(s + fallback_chunk_length, len(dataset)))
            for s in range(0, len(dataset), fallback_chunk_length)
        )
    else:
        for c, s in zip(dataset.chunks[1:], dataset.shape[1:]):
            if c == s:
                continue
            raise ValueError(
                f"Dataset {dataset} can only be chunked on the first axis. "
                f"Found {dataset.chunks=} with {dataset.shape=}."
            )

        chunks = dataset.iter_chunks()
        # throw away slices along other axes
        chunks = (chunk[0] for chunk in chunks)

    if show_progress:
        desc = dataset.name
        if desc_prefix:
            desc = f"{desc_prefix} {desc}"
        if dataset.chunks is None:
            n_chunks = int(np.ceil(dataset.shape[0] / fallback_chunk_length))
        else:
            n_chunks = int(np.ceil(dataset.shape[0] / dataset.chunks[0]))
        chunks = tqdm(chunks, total=n_chunks, desc=desc)

    for sli in chunks:
        yield sli, dataset[sli]


def yield_masked_chunks(mask, dataset, show_progress=True, desc_prefix=None):
    offset = 0
    for sli, data in yield_chunks(
        dataset, show_progress=show_progress, desc_prefix=desc_prefix
    ):
        source_ixs = np.flatnonzero(mask[sli])
        dest_ixs = slice(offset, offset + source_ixs.size)
        yield dest_ixs, data[source_ixs]
        offset += source_ixs.size


# -- residual


def extract_random_snips(rg, chunk, n, sniplen):
    rg = np.random.default_rng(rg)
    if sniplen * n > chunk.shape[0]:
        warnings.warn("Can't extract this many non-overlapping snips.")
        times = rg.choice(chunk.shape[0] - sniplen, size=n, replace=False)
        times.sort()
    else:
        empty_len = chunk.shape[0] - sniplen * n
        times = rg.choice(empty_len, size=n, replace=False)
        times += np.arange(n) * sniplen
    tixs = times[:, None] + np.arange(sniplen)
    return chunk[tixs], times


# -- dataset subsampling


def subsample_waveforms(
    hdf5_filename=None,
    fit_sampling="random",
    random_state: int | np.random.Generator = 0,
    n_waveforms_fit=10_000,
    voltages_dataset_name="collisioncleaned_voltages",
    waveforms_dataset_name="collisioncleaned_waveforms",
    fit_max_reweighting=4.0,
    log_voltages=True,
    subsample_by_weighting=False,
    fixed_property_keys=("channels",),
    replace=True,
    h5=None,
    device="cpu",
):
    random_state = np.random.default_rng(random_state)

    need_open = h5 is None
    if need_open:
        hdf5_filename = resolve_path(hdf5_filename, strict=True)
        h5 = h5py.File(hdf5_filename)

    try:
        channels: np.ndarray = h5["channels"][:]
        n_wf = channels.shape[0]
        if not n_wf:
            emptyi = torch.tensor([], dtype=torch.long)
            emptywf = torch.zeros(h5[waveforms_dataset_name].shape)
            return emptywf, dict(channels=emptyi)
        weights = fit_reweighting(
            h5=h5,
            log_voltages=log_voltages,
            fit_sampling=fit_sampling,
            fit_max_reweighting=fit_max_reweighting,
            voltages_dataset_name=voltages_dataset_name,
        )
        if n_wf > n_waveforms_fit and not subsample_by_weighting:
            choices = random_state.choice(
                n_wf, p=weights, size=n_waveforms_fit, replace=replace
            )
            if not replace:
                choices.sort()
                waveforms = batched_h5_read(h5[waveforms_dataset_name], choices)
                fixed_properties = {k: h5[k][choices] for k in fixed_property_keys}
            else:
                uchoices, ichoices = np.unique(choices, return_inverse=True)
                waveforms = batched_h5_read(h5[waveforms_dataset_name], uchoices)[
                    ichoices
                ]
                fixed_properties = {
                    k: h5[k][uchoices][ichoices] for k in fixed_property_keys
                }
        else:
            waveforms: np.ndarray = h5[waveforms_dataset_name][:]
            fixed_properties = {k: h5[k][:] for k in fixed_property_keys}
    finally:
        if need_open:
            h5.close()
        del h5

    device = torch.device(device)
    waveforms = torch.as_tensor(waveforms, device=device)
    fixed_properties = {
        k: torch.as_tensor(v, device=device) for k, v in fixed_properties.items()
    }
    if subsample_by_weighting:
        fixed_properties["weights"] = torch.as_tensor(weights, device=device)

    return waveforms, fixed_properties


def fit_reweighting(
    voltages=None,
    h5=None,
    hdf5_path=None,
    log_voltages=True,
    fit_sampling="random",
    fit_max_reweighting=4.0,
    voltages_dataset_name="voltages",
):
    if fit_sampling == "random":
        return None
    assert fit_sampling == "amp_reweighted"

    if voltages is None:
        if h5 is not None:
            voltages = h5[voltages_dataset_name][:]
        elif hdf5_path is not None:
            with h5py.File(hdf5_path) as h5:
                voltages = h5[voltages_dataset_name][:]
        else:
            assert False

    from ..cluster.density import get_smoothed_densities

    if torch.is_tensor(voltages):
        voltages = voltages.numpy(force=True)
    if log_voltages:
        sign = voltages / np.abs(voltages)
        voltages = sign * np.log(np.abs(voltages))
    sigma = 1.06 * voltages.std() * np.power(len(voltages), -0.2)
    dens = get_smoothed_densities(voltages[:, None], sigmas=sigma)
    sample_p = dens.mean() / dens
    sample_p = sample_p.clip(1.0 / fit_max_reweighting, fit_max_reweighting)
    sample_p = sample_p.astype(float)  # ensure double before normalizing
    sample_p /= sample_p.sum()
    return sample_p


def divide_randomly(n_things, n_bins, rg):
    things_per_bin = np.zeros(n_bins, dtype=np.int64)
    n_even_split = n_things // n_bins
    things_per_bin += n_even_split
    n_things_remaining = n_things - n_bins * n_even_split
    assert n_things_remaining >= 0
    if n_things_remaining:
        rg = np.random.default_rng(rg)
        choices = rg.choice(n_bins, size=n_things_remaining)
        np.add.at(things_per_bin, choices, 1)
    assert things_per_bin.sum() == n_things
    return things_per_bin
