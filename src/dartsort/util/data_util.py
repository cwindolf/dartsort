from collections import namedtuple
from copy import copy
from dataclasses import replace
from pathlib import Path
from typing import Generator, Sequence, cast, Self, Literal
import warnings

import h5py
import numpy as np
import torch
from spikeinterface.core import NumpySorting, get_random_data_chunks
from tqdm.auto import tqdm

from ..detect import detect_and_deduplicate
from ..util.logging_util import get_logger
from ..util.py_util import resolve_path
from .waveform_util import make_channel_index

logger = get_logger(__name__)

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

    When you instantiate this with from_peeling_hdf5, if the
    flag load_simple_features is True (default), then additional
    features of spikes will be loaded into memory -- like localizations,
    which you can access like `sorting.point_source_localizations[...]`.
    """

    def __init__(
        self,
        *,
        times_samples: np.ndarray,
        channels: np.ndarray,
        labels: np.ndarray | None,
        parent_h5_path: str | Path | None = None,
        sampling_frequency: float | int = 30000.0,
        persistent_features: dict[str, np.ndarray] | None = None,
        ephemeral_features: dict[str, np.ndarray] | None = None,
    ):
        """Construct a DARTsortSorting directly from times, channels, labels, et cetera.

        It's more common to construct from an HDF5 file with .from_peeling_hdf5() or from
        a .npz with .load().
        """
        self.n_spikes = times_samples.shape[0]
        if parent_h5_path is not None:
            parent_h5_path = resolve_path(parent_h5_path)
        self.parent_h5_path = parent_h5_path
        self.sampling_frequency = float(sampling_frequency)

        assert times_samples.dtype.kind == "i"
        self.times_samples = times_samples
        assert channels.dtype.kind == "i"
        assert channels.shape == (self.n_spikes,)
        self.channels = channels
        if labels is not None:
            assert labels.shape == (self.n_spikes,)
            assert labels.dtype.kind == "i"
        self.labels = labels

        self._ephemeral_feature_names = []
        if ephemeral_features is not None:
            for k, v in ephemeral_features.items():
                check_shape = not self._no_check_needed(k)
                self.add_ephemeral_feature(k, v, check_shape=check_shape)

        self._loaded_persistent_features = []
        if persistent_features is not None:
            for k, v in persistent_features.items():
                check_shape = not self._no_check_needed(k)
                self._register_persistent_feature(k, v, check_shape=check_shape)

    @property
    def unit_ids(self) -> np.ndarray:
        if self.labels is None:
            return np.array([], dtype=np.int64)
        u = np.unique(self.labels)
        return u[u >= 0]

    @property
    def n_units(self) -> int:
        return self.unit_ids.shape[0]

    def copy(self) -> Self:
        """Shallow copy. Doesn't copy data, but copies references and internal state."""
        other = copy(self)
        other._ephemeral_feature_names = self._ephemeral_feature_names.copy()
        other._loaded_persistent_features = self._loaded_persistent_features.copy()
        return other

    def ephemeral_replace(
        self, *, check_shapes=True, **new_features: np.ndarray
    ) -> Self:
        """Return a shallow copy of self with certain datasets/features replaced by new_features."""
        other = self.copy()
        for k, v in new_features.items():
            if k in ("times_samples", "channels", "labels"):
                if check_shapes:
                    other._check_shape(k, v)
                setattr(other, k, v)
            else:
                check = check_shapes and not self._no_check_needed(k)
                other.add_ephemeral_feature(k, v, check_shape=check, overwrite=True)
        return other

    def has_persistent_labels(self) -> bool:
        """Are my .labels those from the hdf5 file?"""
        if self.parent_h5_path is None:
            return False
        if self.labels is None:
            return False
        if "labels" in self._ephemeral_feature_names:
            assert "labels" not in self._loaded_persistent_features
            return False
        try:
            return np.array_equal(self.labels, self._load_dataset("labels"))
        except KeyError:
            return False

    # interface for setting features

    def add_ephemeral_feature(
        self,
        feature_name: str,
        feature: np.ndarray,
        check_shape: bool | None = None,
        overwrite=False,
    ):
        """
        Ephemeral features are accessible as properties and persisted to/from .npz,
        but not saved in the .h5.
        """
        if check_shape is None:
            check_shape = not self._no_check_needed(feature_name)
        if check_shape:
            self._check_shape(feature_name, feature)

        already_ephemeral = feature_name in self._ephemeral_feature_names
        already_attr = hasattr(self, feature_name)
        if already_ephemeral:
            assert already_attr
        if already_attr and not overwrite:
            raise ValueError(
                f"Can't add feature {feature_name}, since it already exists."
            )
        if not already_ephemeral:
            self._ephemeral_feature_names.append(feature_name)
        setattr(self, feature_name, feature)

    def remove_ephemeral_feature(self, feature_name: str):
        assert feature_name in self._ephemeral_feature_names
        assert hasattr(self, feature_name)
        self._ephemeral_feature_names = [
            k for k in self._ephemeral_feature_names if k != feature_name
        ]
        delattr(self, feature_name)

    def unload_persistent_feature(self, feature_name: str):
        assert feature_name in self._loaded_persistent_features
        assert hasattr(self, feature_name)
        self._loaded_persistent_features = [
            k for k in self._loaded_persistent_features if k != feature_name
        ]
        delattr(self, feature_name)

    def add_feature(self, feature_name: str, feature: np.ndarray, check_shape=True):
        """Try to save a feature to h5, else register as ephemeral."""
        if self.parent_h5_path is None:
            self.add_ephemeral_feature(feature_name, feature, check_shape)
        else:
            self._register_persistent_feature(feature_name, feature, check_shape)

    def remove_feature(self, feature_name: str):
        if feature_name in self._loaded_persistent_features:
            self.unload_persistent_feature(feature_name)
        elif feature_name in self._ephemeral_feature_names:
            self.remove_ephemeral_feature(feature_name)
        else:
            raise ValueError(f"Sorting doesn't have {feature_name}.")

    def _register_persistent_feature(
        self, feature_name: str, feature: np.ndarray, check_shape=True
    ):
        """Persistent features are written to the h5."""
        if self.parent_h5_path is None:
            raise ValueError(
                f"Can't register persistent feature {feature_name}, because "
                f"there is no .hdf5 file.",
            )
        if check_shape:
            self._check_shape(feature_name, feature)
        self._loaded_persistent_features.append(feature_name)
        setattr(self, feature_name, feature)
        try:
            with h5py.File(
                self.parent_h5_path, "r", libver="latest", locking=False
            ) as h5:
                if feature_name not in h5:
                    logger.dartsortdebug(
                        "Registering persistent feature %s to %s.",
                        feature_name,
                        self.parent_h5_path,
                    )
                    h5.create_dataset(feature_name, data=feature)
        except FileNotFoundError:
            logger.warning(
                f"Sorting's parent h5 file {self.parent_h5_path} is gone when registering "
                f"persistent feature {feature_name}. Will continue, but this sorting won't "
                "persist correctly.",
                stacklevel=3,
            )

    # save / load

    @classmethod
    def from_peeling_hdf5(
        cls,
        h5_path: str | Path,
        *,
        times_samples_dataset="times_samples",
        channels_dataset="channels",
        labels_dataset="labels",
        load_feature_names: Sequence[str] | None = None,
        load_simple_features=True,
        load_all_features=False,
    ) -> Self:
        """Load sorting from .hdf5 format saved by peelers

        Arguments
        ---------
        load_feature_names : optional list of str
            Load exactly these features, plus geom/channel index.
        load_simple_features : bool
            If load_feature_names unspecified, load all scalar or vector
            features (per spike), but no matrix-valued features like waveforms
            or multi-channel PCA features.
        load_all_features : bool
        """
        h5_path = resolve_path(h5_path, strict=True)

        with h5py.File(h5_path, "r", libver="latest", locking=False) as h5:
            times_samples = cast(h5py.Dataset, h5[times_samples_dataset])[:]
            n = times_samples.shape[0]
            channels = cast(h5py.Dataset, h5[channels_dataset])[:]
            sampling_frequency = float(
                cast(h5py.Dataset, h5["sampling_frequency"])[()].item()
            )
            if labels_dataset in h5:
                labels = cast(h5py.Dataset, h5[labels_dataset])[:]
            else:
                labels = None

            already_loaded = [
                times_samples_dataset,
                channels_dataset,
                labels_dataset,
                "sampling_frequency",
            ]
            if load_feature_names is None and load_all_features:
                load_feature_names = list(h5.keys())
            elif load_feature_names is None and load_simple_features:
                load_feature_names = []
                for k in h5.keys():
                    if k in already_loaded:
                        continue
                    if cls._no_check_needed(k):
                        load_feature_names.append(k)
                        continue
                    dset = cast(h5py.Dataset, h5[k])
                    is_simple = 1 <= dset.ndim <= 2 and dset.shape[0] == n
                    if is_simple:
                        load_feature_names.append(k)
            elif load_feature_names is None:
                load_feature_names = [k for k in h5.keys() if cls._no_check_needed(k)]
            assert load_feature_names is not None
            load_feature_names = [
                k for k in load_feature_names if k not in already_loaded
            ]
            persistent_features = {
                k: cast(h5py.Dataset, h5[k])[:] for k in load_feature_names
            }

        return cls(
            times_samples=times_samples,
            channels=channels,
            labels=labels,
            sampling_frequency=sampling_frequency,
            persistent_features=persistent_features,
            parent_h5_path=h5_path,
        )

    def save(self, sorting_npz: str | Path):
        """Save to npz (usually dartsort_sorting.npz)

        Support persisting myself in non-h5-supportable cases
        Cases:
         - When there is no h5!
         - When I have new labels.
        This is done by saving to .npz, with a pointer (like a relative symlink)
        to the .h5 file if it exists.
        """
        sorting_npz = resolve_path(sorting_npz)
        data = dict(
            times_samples=self.times_samples,
            channels=self.channels,
            sampling_frequency=np.array(self.sampling_frequency),
        )
        if self.labels is not None:
            data["labels"] = self.labels

        have_hdf5 = self.parent_h5_path is not None
        if have_hdf5:
            # path needs to be relative to npz path's parent in case user moves stuff
            h5p = resolve_path(self.parent_h5_path, strict=True)
            h5p = h5p.relative_to(sorting_npz.parent, walk_up=True)
            data["parent_h5_path"] = np.array(str(h5p))
        for k in self._ephemeral_feature_names:
            data[k] = getattr(self, k)
        data["ephemeral_feature_names"] = np.array(self._ephemeral_feature_names)
        data["loaded_persistent_features"] = np.array(self._loaded_persistent_features)
        np.savez(sorting_npz, **data, allow_pickle=False)

    @classmethod
    def load(cls, sorting_npz, additional_persistent_features=None) -> Self:
        """Load from npz (usually dartsort_sorting.npz)."""
        sorting_npz = resolve_path(sorting_npz, strict=True)
        with np.load(sorting_npz) as data:
            times_samples = data["times_samples"]
            channels = data["channels"]
            labels = data.get("labels", None)
            sampling_frequency = data["sampling_frequency"]
            if isinstance(sampling_frequency, np.ndarray):
                sampling_frequency = sampling_frequency.item()
            parent_h5_path = data.get("parent_h5_path", None)
            if "ephemeral_feature_names" in data:
                ephemeral_features = {
                    k: data[k] for k in data["ephemeral_feature_names"]
                }
            else:
                ephemeral_features = {}
            loaded_persistent_features = data.get("loaded_persistent_features", [])

        if parent_h5_path is not None:
            parent_h5_path = parent_h5_path.item()
            assert isinstance(parent_h5_path, str)
            parent_h5_path = sorting_npz.parent / Path(parent_h5_path)
            parent_h5_path = resolve_path(parent_h5_path, strict=True)
            if additional_persistent_features:
                loaded_persistent_features = set(
                    loaded_persistent_features + additional_persistent_features
                )
            self = cls.from_peeling_hdf5(
                parent_h5_path,
                load_feature_names=list(loaded_persistent_features),
            )
            if labels is not None:
                ephemeral_features["labels"] = labels
            return self.ephemeral_replace(
                times_samples=times_samples, channels=channels, **ephemeral_features
            )
        assert not loaded_persistent_features

        return cls(
            times_samples=times_samples,
            channels=channels,
            labels=labels,
            sampling_frequency=sampling_frequency,
            parent_h5_path=parent_h5_path,
            ephemeral_features=ephemeral_features,
        )

    def to_numpy_sorting(self) -> NumpySorting:
        """Clean up and produce a spikeinterface NumpySorting object."""
        st = self.drop_missing()
        return NumpySorting.from_samples_and_labels(
            samples_list=st.times_samples,
            labels_list=st.labels,
            sampling_frequency=st.sampling_frequency,
        )

    def mask(self, mask: np.ndarray) -> Self:
        assert mask.ndim == 1
        if np.dtype(mask.dtype).kind == "b":
            assert mask.shape == (self.n_spikes,)
            mask = np.flatnonzero(mask)
        assert mask.max() < self.n_spikes

        if self.labels is None:
            labels = None
        else:
            labels = self.labels[mask]

        eph = {}
        for k in self._ephemeral_feature_names:
            assert k != "mask_indices"  # no recursion...
            v = getattr(self, k)
            if self._no_check_needed(k):
                eph[k] = v
            else:
                eph[k] = v[mask]
        eph["mask_indices"] = mask

        per = {}
        for k in self._loaded_persistent_features:
            assert k != "mask_indices"  # no recursion...
            v = getattr(self, k)
            if self._no_check_needed(k):
                per[k] = v
            else:
                per[k] = v[mask]

        return self.__class__(
            times_samples=self.times_samples[mask],
            channels=self.channels[mask],
            labels=labels,
            parent_h5_path=self.parent_h5_path,
            sampling_frequency=self.sampling_frequency,
            persistent_features=per,
            ephemeral_features=eph,
        )

    def drop_missing(self) -> Self:
        """Remove spikes with -1 labels."""
        assert self.labels is not None
        return self.mask(self.labels >= 0)

    def drop_doubles(self):
        """Remove spikes detected at the exact same time assigned to the same unit."""
        assert self.labels is not None
        viol_ixs = []
        for uid in self.unit_ids:
            inu = np.flatnonzero(self.labels == uid)
            tu = self.times_samples[inu]
            # this has the first indices of each pair. +1 has the second indices.
            viol = np.flatnonzero(np.diff(tu) == 0)
            viol_ixs.append(viol + 1)
        viol_ixs = np.concatenate(viol_ixs)
        if viol_ixs.size:
            logger.dartsortdebug(f"Dropping {viol_ixs.size} duplicates.")
            labels = self.labels.copy()
            labels[viol_ixs] = -1
            return self.ephemeral_replace(labels=labels)
        else:
            return self

    def flatten(self) -> Self:
        """Flatten the unit IDs so that there are no gaps in the sorted unique label set."""
        assert self.labels is not None
        valid = np.flatnonzero(self.labels >= 0)
        _, flat_labels = np.unique(self.labels[valid], return_inverse=True)
        new_labels = np.full_like(self.labels, -1)
        new_labels[valid] = flat_labels
        return self.ephemeral_replace(labels=new_labels)

    def __str__(self):
        name = self.__class__.__name__
        ns = self.n_spikes
        nu = self.n_units
        unit_str = f"{nu} unit" + "s" * (nu > 1)
        feat_str = " "
        if self._loaded_persistent_features:
            s = ", ".join(self._loaded_persistent_features)
            feat_str += f"Loaded HDF5 features: {s}. "
        if self._ephemeral_feature_names:
            s = ", ".join(self._ephemeral_feature_names)
            feat_str += f"Features: {s}. "
        h5_str = ""
        if self.parent_h5_path:
            h5_str = f"From HDF5 file {self.parent_h5_path}."
        if self.labels is not None:
            noise_prop = (self.labels < 0).mean().item()
            noise_pct = 100 * noise_prop
            noise_str = f" ({noise_pct:.2f}% noise)"
        else:
            noise_str = ""
        return f"{name}: {ns} spikes{noise_str}, {unit_str}.{feat_str}{h5_str}"

    def __repr__(self):
        return str(self)

    def __len__(self):
        return self.n_spikes

    @staticmethod
    def _is_geom_related(k) -> bool:
        return k == "geom" or k.endswith("channel_index")

    @staticmethod
    def _is_unit_related(k) -> bool:
        return k.startswith("unit")

    @classmethod
    def _no_check_needed(cls, k) -> bool:
        return cls._is_geom_related(k) or cls._is_unit_related(k)

    def _check_shape(self, feature_name: str, feature: np.ndarray):
        if feature.shape[0] != self.n_spikes:
            raise ValueError(
                f"Feature {feature_name}'s shape {feature.shape} didn't agree with spike count {self.n_spikes}."
            )

    def _load_dataset(self, dataset_name: str) -> np.ndarray:
        assert self.parent_h5_path is not None
        with h5py.File(self.parent_h5_path, "r", locking=False) as h5:
            dset = h5[dataset_name]
            assert isinstance(dset, h5py.Dataset)
            return dset[:]

    def slice_feature_by_name(
        self, dataset_name: str, mask: np.ndarray | slice = slice(None)
    ) -> np.ndarray:
        if hasattr(self, dataset_name):
            return getattr(self, dataset_name)[mask]

        # otherwise, we don't have it loaded
        assert dataset_name not in self._ephemeral_feature_names
        assert dataset_name not in self._loaded_persistent_features

        # but we can try to load it
        if self.parent_h5_path is None:
            raise ValueError(f"Can't load feature {dataset_name} with no HDF5.")
        if isinstance(mask, slice) and mask == slice(None):
            return self._load_dataset(dataset_name)

        # h5 direct read is fine for a few indices
        if isinstance(mask, np.ndarray) and mask.dtype.kind != "b":
            if mask.size <= 768:
                with h5py.File(self.parent_h5_path, "r", locking=False) as h5:
                    return cast(h5py.Dataset, h5[dataset_name])[mask]

        # mask needs to be boolean for _read_by_chunk
        if not isinstance(mask, np.ndarray) or mask.dtype.kind != "b":
            h5_mask = np.zeros(self.n_spikes, dtype=np.bool_)
            h5_mask[mask] = True
        else:
            h5_mask = mask

        with h5py.File(self.parent_h5_path, "r", locking=False) as h5:
            dset = h5[dataset_name]
            assert isinstance(dset, h5py.Dataset)
            assert dset.shape[0] == self.n_spikes == h5_mask.shape[0]
            return _read_by_chunk(h5_mask, dset, show_progress=False)


def load_h5(f: str | Path) -> DARTsortSorting:
    return DARTsortSorting.from_peeling_hdf5(h5_path=f)


def try_get_model_dir(sorting: DARTsortSorting) -> Path | None:
    if sorting.parent_h5_path is None:
        return None
    h5_path = resolve_path(sorting.parent_h5_path)
    model_dir = h5_path.parent / f"{h5_path.stem}_models"
    if model_dir.exists():
        assert model_dir.is_dir()
        return model_dir
    else:
        return None


def try_get_denoising_pipeline(sorting: DARTsortSorting):
    m_dir = try_get_model_dir(sorting)
    if m_dir is None:
        return None, None, None

    candidates = list(m_dir.glob("*denoising_pipeline.pt"))
    if len(candidates) == 0:
        return None, None, None
    elif len(candidates) > 1:
        raise ValueError(f"Not sure which to load of {candidates}.")
    assert len(candidates) == 1

    from dartsort.transform import WaveformPipeline

    geom = torch.asarray(getattr(sorting, "geom"))
    channel_index = torch.asarray(getattr(sorting, "subtract_channel_index"))
    dn = WaveformPipeline.from_state_dict_pt(geom, channel_index, candidates[0])
    dn = dn.eval()
    return dn, geom, channel_index


def get_featurization_pipeline(sorting, featurization_pipeline_pt=None):
    """Look for the pipeline in the usual place."""
    from dartsort.transform import WaveformPipeline

    if isinstance(sorting, Path):
        base_dir = sorting.parent
        stem = sorting.stem
        # TODO how to type this better... don't understand.
        geom = channel_index = None  # type: ignore
    else:
        assert isinstance(sorting, DARTsortSorting)
        if sorting.parent_h5_path is None:
            raise ValueError("Can't load featurization pipeline.")

        h5_path = resolve_path(sorting.parent_h5_path)
        base_dir = h5_path.parent
        stem = h5_path.stem

        geom = getattr(sorting, "geom", None)  # type: ignore
        channel_index = getattr(sorting, "channel_index", None)  # type: ignore

    model_dir = base_dir / f"{stem}_models"
    if geom is None or channel_index is None:
        with h5py.File(base_dir / f"{stem}.h5", "r", locking=False) as h5:
            geom: np.ndarray = h5["geom"][:]  # type: ignore
            channel_index: np.ndarray = h5["channel_index"][:]  # type: ignore
    assert geom is not None
    assert channel_index is not None

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
    from ..transform import BaseTemporalPCA

    if sorting.parent_h5_path is None:
        logger.info("Couldn't load stored basis.")
        return None
    pipeline, pt_path = get_featurization_pipeline(sorting)
    tsvd = pipeline.get_transformer(tsvd_name)
    assert tsvd is not None
    assert tsvd.name == tsvd_name
    assert isinstance(tsvd, BaseTemporalPCA)
    if to_sklearn:
        tsvd = tsvd.to_sklearn()
    logger.info(
        f"Loaded stored basis from %s (%s; components shape: %s).",
        pt_path,
        tsvd_name,
        tsvd.components_.shape,
    )
    return tsvd


def get_labels(h5_path) -> np.ndarray:
    with h5py.File(h5_path, "r") as h5:
        return h5["labels"][:]  # type: ignore


def get_residual_snips(h5_path) -> np.ndarray:
    with h5py.File(h5_path, "r", locking=False) as h5:
        return h5["residual"][:]  # type: ignore


def sorting_isis(sorting: DARTsortSorting):
    assert sorting.labels is not None
    isis_ms = np.zeros(len(sorting))
    for uid in sorting.unit_ids:
        inu = np.flatnonzero(sorting.labels == uid)
        t_ms = sorting.times_seconds[inu] * 1000  # type: ignore
        isi = np.diff(t_ms)
        isi = np.concatenate([[np.inf], np.abs(isi), [np.inf]])
        isi = np.minimum(isi[1:], isi[:-1])
        isis_ms[inu] = isi
    return isis_ms


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
    new_sorting = sorting.ephemeral_replace(labels=new_labels)
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
        dres = detect_and_deduplicate(
            torch.tensor(chunk, dtype=dtype),
            threshold=threshold,
            peak_sign="both",
            dedup_channel_index=torch.tensor(dedup_channel_index),
        )
        times = dres[0]
        del dres
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

    return sorting.ephemeral_replace(labels=new_labels)


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

    return sorting.ephemeral_replace(labels=new_labels)


def restrict_to_valid_times(sorting, recording, waveform_cfg, pad=0):
    trough = waveform_cfg.trough_offset_samples(recording.sampling_frequency)
    total = waveform_cfg.spike_length_samples(recording.sampling_frequency)
    t_min = trough + pad
    t_max = recording.get_total_samples() - (total - trough) - pad
    new_labels = sorting.labels.copy()
    new_labels[sorting.times_samples < t_min] = -1
    new_labels[sorting.times_samples > t_max] = -1
    return sorting.ephemeral_replace(labels=new_labels)


def subset_sorting_by_time_samples(
    sorting, start_sample=0, end_sample=np.inf, reference_to_start_sample=True
):
    new_times = sorting.times_samples.copy()
    new_labels = sorting.labels.copy()

    in_range = (new_times >= start_sample) & (new_times < end_sample)
    new_labels[~in_range] = -1

    if reference_to_start_sample:
        new_times -= start_sample

    return sorting.ephemeral_replace(labels=new_labels, times_samples=new_times)


def subset_sorting_by_time_seconds(sorting, t_start=0, t_end=np.inf):
    new_labels = sorting.labels.copy()
    t_s = sorting.times_seconds
    in_range = t_s == t_s.clip(t_start, t_end)
    new_labels[~in_range] = -1

    return sorting.ephemeral_replace(labels=new_labels)


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
    return sorting.ephemeral_replace(labels=new_labels)


def combine_sortings(sortings, dodge=False):
    labels = np.full_like(sortings[0].labels, -1)
    times_samples = sortings[0].times_samples.copy()
    assert all(s.labels.size == sortings[0].labels.size for s in sortings)

    if dodge:
        label_to_sorting_index = []
        label_to_original_label = []
    else:
        label_to_sorting_index = None
        label_to_original_label = None

    next_label = 0
    for j, sorting in enumerate(sortings):
        kept = np.flatnonzero(sorting.labels >= 0)
        assert np.all(labels[kept] < 0)
        labels[kept] = sorting.labels[kept] + next_label
        if dodge:
            assert label_to_sorting_index is not None
            assert label_to_original_label is not None
            n_new_labels = 0
            if kept.size:
                n_new_labels = 1 + sorting.labels[kept].max()
                next_label += n_new_labels
            label_to_sorting_index.append(np.full(n_new_labels, j))
            label_to_original_label.append(np.arange(n_new_labels))
        times_samples[kept] = sorting.times_samples[kept]

    sorting = sortings[0].ephemeral_replace(labels=labels, times_samples=times_samples)

    if dodge:
        assert label_to_sorting_index is not None
        assert label_to_original_label is not None
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
    assert mask.dtype.kind == "b"
    assert mask.shape[0] == dataset.shape[0]
    out = np.empty((mask.sum(), *dataset.shape[1:]), dtype=dataset.dtype)
    n = 0
    for sli, dsli in yield_chunks(dataset, show_progress):
        m = np.flatnonzero(mask[sli])
        nm = m.size
        if not nm:
            continue
        x = dsli[m]
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


def extract_random_snips(
    rg: int | np.random.Generator,
    chunk: np.ndarray | torch.Tensor,
    n: int,
    sniplen: int,
):
    """Grab n (or as many as can fit) random non-overlapping snips from chunk."""
    rg = np.random.default_rng(rg)

    # we can extract at most this many snips
    n = min(n, max(1, chunk.shape[0] // sniplen))

    # how many samples will not be covered
    empty_len = chunk.shape[0] - sniplen * n
    assert empty_len >= 0

    # sample points spaced by sniplen + deltaj, where sum of
    # delta is <= empty_len. use an extra slack diff for the <.
    if empty_len == 0:
        delta = np.zeros((n + 1,), dtype=np.int64)
    else:
        delta = rg.integers(0, empty_len, size=n + 1)
    delta.sort()
    delta = np.diff(delta)[:n]
    times = np.cumsum(delta + sniplen) - sniplen
    tixs = times[:, None] + np.arange(sniplen)
    return chunk[tixs], times


# -- dataset subsampling


def subsample_waveforms(
    hdf5_filename: str | Path | None = None,
    fit_sampling: Literal["random", "amp_reweighted"] = "random",
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
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    random_state = np.random.default_rng(random_state)

    need_open = h5 is None
    if need_open and hdf5_filename is not None:
        hdf5_filename = resolve_path(hdf5_filename, strict=True)
        h5 = h5py.File(hdf5_filename)
    elif need_open:
        raise ValueError("Need h5 or hdf5_filename.")

    try:
        channels: np.ndarray = h5["channels"][:]  # type: ignore
        n_wf = channels.shape[0]
        if not n_wf:
            emptyi = torch.tensor([], dtype=torch.long)
            wfshape = h5[waveforms_dataset_name].shape  # type: ignore
            emptywf = torch.zeros(wfshape)
            return emptywf, dict(channels=emptyi)
        weights = fit_reweighting(
            h5=h5,
            log_voltages=log_voltages,
            fit_sampling=fit_sampling,
            fit_max_reweighting=fit_max_reweighting,
            voltages_dataset_name=voltages_dataset_name,
        )
        fixed_property_keys = [k for k in fixed_property_keys if k in h5]
        if n_wf > n_waveforms_fit and not subsample_by_weighting:
            choices = random_state.choice(
                n_wf, p=weights, size=n_waveforms_fit, replace=replace
            )
            if not replace:
                choices.sort()
                waveforms = batched_h5_read(h5[waveforms_dataset_name], choices)
                fixed_properties = {k: h5[k][choices] for k in fixed_property_keys}  # type: ignore
            else:
                uchoices, ichoices = np.unique(choices, return_inverse=True)
                waveforms = batched_h5_read(h5[waveforms_dataset_name], uchoices)[
                    ichoices
                ]
                fixed_properties = {
                    k: h5[k][uchoices][ichoices]  # type: ignore
                    for k in fixed_property_keys
                }
        else:
            waveforms: np.ndarray = h5[waveforms_dataset_name][:]  # type: ignore
            fixed_properties = {k: h5[k][:] for k in fixed_property_keys}  # type: ignore
    finally:
        if need_open:
            h5.close()
        del h5

    device = torch.device(device)
    waveformsr = torch.as_tensor(waveforms, device=device)
    fixed_properties = {
        k: torch.as_tensor(v, device=device) for k, v in fixed_properties.items()
    }
    if subsample_by_weighting and weights is not None:
        fixed_properties["weights"] = torch.as_tensor(weights, device=device)
    elif subsample_by_weighting:
        fixed_properties["weights"] = torch.ones(waveforms.shape[0], device=device)

    return waveformsr, fixed_properties


def fit_reweighting(
    voltages: np.ndarray | None = None,  # type: ignore
    h5=None,
    hdf5_path=None,
    log_voltages=True,
    fit_sampling: Literal["random", "amp_reweighted"] = "random",
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
                voltages: np.ndarray = h5[voltages_dataset_name][:]  # type: ignore
        else:
            assert False
    assert isinstance(voltages, np.ndarray)

    from ..clustering.density import get_smoothed_density

    if torch.is_tensor(voltages):
        voltages = voltages.numpy(force=True)
    if log_voltages:
        sign = np.sign(voltages)
        voltages = sign * np.log(np.abs(voltages))
    voltages = np.nan_to_num(voltages)
    sigma = 1.06 * voltages.std() * np.power(len(voltages), -0.2)
    assert np.isfinite(sigma)
    dens = get_smoothed_density(voltages[:, None], sigma=sigma)
    assert isinstance(dens, np.ndarray)
    sample_p = dens.mean() / dens
    sample_p = sample_p.clip(1.0 / fit_max_reweighting, fit_max_reweighting)
    sample_p = sample_p.astype(np.float64)  # ensure double before normalizing
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
