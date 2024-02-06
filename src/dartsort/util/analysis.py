"""Deeper object-oriented interaction with sorter data

This is meant to make implementing plotting code easier: this
code becomes the model in a MVC framework, and vis/unit.py can
implement a view and controller.

This should also make it easier to compute drift-aware metrics
(e.g., d' using registered templates and shifted waveforms).
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional
import pickle

import h5py
import numpy as np
import spikeinterface.core as sc
import torch
from dredge.motion_util import MotionEstimate
from sklearn.decomposition import PCA
from spikeinterface.comparison import GroundTruthComparison

from ..cluster import merge, relocate
from ..config import default_template_config
from ..templates import TemplateData
from ..transform import WaveformPipeline
from .data_util import DARTsortSorting
from .drift_util import (get_spike_pitch_shifts,
                         get_waveforms_on_static_channels, registered_average)
from .spikeio import read_waveforms_channel_index
from .waveform_util import make_channel_index


@dataclass
class DARTsortAnalysis:
    """Stores all relevant properties for a drift-aware waveform analysis

    If motion_est is None, there is no motion correction applied.

    If motion_est is not None but relocated is False, waveforms are shifted
    across channel neighborhoods to account for drift.

    If additionally relocated is True, point-source relocation is applied
    to change around the amplitudes on each channel.
    """

    sorting: DARTsortSorting
    recording: sc.BaseRecording
    template_data: TemplateData
    hdf5_path: Optional[Path] = None
    featurization_pipeline: Optional[WaveformPipeline] = None
    motion_est: Optional[MotionEstimate] = None

    # hdf5 keys
    localizations_dataset = "point_source_localizations"
    amplitudes_dataset = "denoised_ptp_amplitudes"
    amplitude_vectors_dataset = "denoised_ptp_amplitude_vectors"
    tpca_features_dataset = "collisioncleaned_tpca_features"
    template_indices_dataset = "collisioncleaned_tpca_features"

    # configuration for analysis computations not included in above objects
    device: Optional[torch.device] = None
    merge_distance_templates_kind: str = "coarse"
    merge_superres_linkage: Callable[[np.ndarray], float] = np.max

    # helper constructors

    def from_sorting(
        cls,
        recording,
        sorting,
        motion_est=None,
        template_config=default_template_config,
        n_jobs_templates=0,
    ):
        """Try to re-load as much info as possible from the sorting itself

        Templates are re-computed if labels are not the same as in h5
        or if the template npz does not exist.
        """
        assert hasattr(sorting, "parent_h5_path")
        hdf5_path = sorting.parent_hdf5_path
        model_dir = hdf5_path.parent / f"{hdf5_path.stem}_models"
        assert model_dir.exists()

        featurization_pipeline = torch.load(model_dir / "featurization_pipeline.pt")

        template_npz = model_dir / "template_data.npz"
        have_templates = template_npz.exists()
        if have_templates:
            with h5py.File(hdf5_path, "r") as h5:
                same_labels = np.array_equal(sorting.labels, h5["labels"][:])
            have_templates = have_templates and same_labels

        if not have_templates:
            template_data = TemplateData.from_config(
                recording,
                sorting,
                template_config,
                overwrite=False,
                motion_est=motion_est,
                n_jobs=n_jobs_templates,
            )

        return cls(
            sorting=sorting,
            recording=recording,
            template_data=template_data,
            hdf5_path=hdf5_path,
            featurization_pipeline=featurization_pipeline,
            motion_est=motion_est,
        )

    @classmethod
    def from_peeling_hdf5_and_recording(
        cls,
        hdf5_path,
        recording,
        template_data,
        featurization_pipeline=None,
        motion_est=None,
        **kwargs,
    ):
        return cls(
            DARTsortSorting.from_peeling_hdf5(hdf5_path, load_simple_features=False),
            Path(hdf5_path),
            recording,
            template_data=template_data,
            featurization_pipeline=featurization_pipeline,
            motion_est=motion_est,
            **kwargs,
        )

    @classmethod
    def from_peeling_paths(
        cls,
        recording,
        hdf5_path,
        model_dir=None,
        motion_est=None,
        template_data_npz="template_data.npz",
        template_data=None,
        motion_est_pkl="motion_est.pkl",
        sorting=None,
        **kwargs,
    ):
        hdf5_path = Path(hdf5_path)
        if model_dir is None:
            model_dir = hdf5_path.parent / f"{hdf5_path.stem}_models"
            assert model_dir.exists()
        if sorting is None:
            sorting = DARTsortSorting.from_peeling_hdf5(
                hdf5_path, load_simple_features=False
            )
        if template_data is None:
            template_data = TemplateData.from_npz(Path(model_dir) / template_data_npz)
        if motion_est is None:
            if (hdf5_path.parent / motion_est_pkl).exists():
                with open(hdf5_path.parent / motion_est_pkl, "rb") as jar:
                    motion_est = pickle.load(jar)
        pipeline = torch.load(model_dir / "featurization_pipeline.pt")
        return cls(
            sorting, recording, template_data, hdf5_path, pipeline, motion_est, **kwargs
        )

    # pickle/h5py gizmos

    def __post_init__(self):
        self.clear_cache()
    
        if self.featurization_pipeline is not None:
            assert not self.featurization_pipeline.needs_fit()
        assert np.isin(
            self.template_data.unit_ids, np.unique(self.sorting.labels)
        ).all()

        assert self.hdf5_path.exists()
        self.coarse_template_data = self.template_data.coarsen()

        # this obj will be pickled and we don't use these, let's save ourselves the ram
        if self.sorting.extra_features:
            self.sorting = replace(self.sorting, extra_features=None)
        self.shifting = (
            self.motion_est is not None
            or self.template_data.registered_geom is not None
        )
        if self.shifting:
            assert (
                self.motion_est is not None
                and self.template_data.registered_geom is not None
            )
        assert np.array_equal(self.coarse_template_data.unit_ids, self.unit_ids)

        # cached hdf5 pointer
        self._h5 = None

    def clear_cache(self):
        self._unit_ids = None
        self._xyza = None
        self._max_chan_amplitudes = None
        self._template_indices = None
        self._amplitude_vectors = None
        self._channel_index = None
        self._geom = None
        self._tpca_features = None
        self._sklearn_tpca = None
        self._unit_ids = None
        self._spike_counts = None
        self._merge_dist = None
        self._feats = {}

    def __getstate__(self):
        # remove cached stuff before pickling
        return {
            k: v if not k.startswith("_") else None for k, v in self.__dict__.items()
        }

    # cache gizmos

    @property
    def h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    @property
    def xyza(self):
        if self._xyza is None:
            self._xyza = self.h5[self.localizations_dataset][:]
        return self._xyza

    @property
    def template_indices(self):
        if self._template_indices is None:
            self._template_indices = self.h5[self.template_indices_dataset][:]
        return self._template_indices

    @property
    def max_chan_amplitudes(self):
        if self._max_chan_amplitudes is None:
            self._max_chan_amplitudes = self.h5[self.amplitudes_dataset][:]
        return self._max_chan_amplitudes

    @property
    def amplitude_vectors(self):
        if self._amplitude_vectors is None:
            self._amplitude_vectors = self.h5[self.amplitude_vectors_dataset][:]
        return self._amplitude_vectors

    @property
    def geom(self):
        if self._geom is None:
            self._geom = self.h5["geom"][:]
        return self._geom

    @property
    def channel_index(self):
        if self._channel_index is None:
            self._channel_index = self.h5["channel_index"][:]
        return self._channel_index

    @property
    def sklearn_tpca(self):
        if self._sklearn_tpca is None:
            tpca_feature = [
                f
                for f in self.featurization_pipeline.transformers
                if f.name == self.tpca_features_dataset
            ]
            assert len(tpca_feature) == 1
            self._sklearn_tpca = tpca_feature[0].to_sklearn()
        return self._sklearn_tpca

    @property
    def merge_dist(self):
        if self._merge_dist is None:
            self._merge_dist = self._calc_merge_dist()
        return self._merge_dist

    # spike train helpers

    @property
    def unit_ids(self):
        if self._unit_ids is None:
            allunits, counts = np.unique(self.sorting.labels, return_counts=True)
            self._unit_ids = allunits[allunits >= 0]
            self._spike_counts = counts[allunits >= 0]
        return self._unit_ids

    @property
    def spike_counts(self):
        if self._spike_counts is None:
            allunits, counts = np.unique(self.sorting.labels, return_counts=True)
            self._unit_ids = allunits[allunits >= 0]
            self._spike_counts = counts[allunits >= 0]
        return self._spike_counts

    def in_unit(self, unit_id):
        return np.flatnonzero(np.isin(self.sorting.labels, unit_id))

    def in_template(self, template_index):
        return np.flatnonzero(np.isin(self.template_indices, template_index))

    def unit_template_indices(self, unit_id):
        return np.flatnonzero(self.template_data.unit_ids == self.unit_id)

    @property
    def show_geom(self):
        show_geom = self.template_data.registered_geom
        if show_geom is None:
            show_geom = self.recording.get_channel_locations()
        return show_geom

    def show_channel_index(self, radius_um=50):
        return make_channel_index(self.show_geom, radius_um)

    # spike feature loading methods

    def named_feature(self, name, which=slice(None)):
        if name not in self._feats:
            self._feats[name] = self.h5[name][:]
        return self._feats[name][which]

    def x(self, which=slice(None)):
        return self.xyza[which, 0]

    def z(self, which=slice(None), registered=True):
        z = self.xyza[which, 2]
        if registered and self.motion_est is not None:
            z = self.motion_est.correct_s(self.times_seconds(which=which), z)
        return z

    def times_seconds(self, which=slice(None)):
        return self.recording._recording_segments[0].sample_index_to_time(
            self.times_samples(which=which)
        )

    def times_samples(self, which=slice(None)):
        return self.sorting.times_samples[which]

    def amplitudes(self, which=slice(None), relocated=False):
        if not relocated or self.motion_est is None:
            return self.max_chan_amplitudes[which]

        reloc_amp_vecs = relocate.relocated_waveforms_on_static_channels(
            self.amplitude_vectors[which],
            main_channels=self.sorting.channels[which],
            channel_index=self.channel_index,
            xyza_from=self.xyza[which],
            z_to=self.z(which),
            geom=self.geom,
            registered_geom=self.template_data.registered_geom,
            target_channels=slice(None),
        )
        return np.nanmax(reloc_amp_vecs, axis=1)

    def tpca_features(self, which=slice(None)):
        if self._tpca_features is None:
            self._tpca_features = self.h5[self.tpca_features_dataset]
        if isinstance(which, slice):
            which = np.arange(len(self.sorting))[which]
        return batched_h5_read(self._tpca_features, which)

    # cluster-dependent feature loading methods

    def unit_raw_waveforms(
        self,
        unit_id,
        which=None,
        template_index=None,
        max_count=250,
        random_seed=0,
        show_radius_um=75,
        trough_offset_samples=42,
        spike_length_samples=121,
        relocated=False,
    ):
        if which is None:
            which = self.in_unit(unit_id)
        if template_index is not None:
            assert template_index in self.unit_template_indices(unit_id)
            which = self.in_template(template_index)
        if max_count is None:
            max_count = which.size
        if which.size > max_count:
            rg = np.random.default_rng(0)
            which = rg.choice(which, size=max_count, replace=False)
            which.sort()
        if not which.size:
            return which, None, None, None, None

        # read waveforms from disk
        if self.shifting:
            load_ci = self.channel_index
        if self.shifting:
            load_ci = make_channel_index(
                self.recording.get_channel_locations(), show_radius_um
            )
        waveforms = read_waveforms_channel_index(
            self.recording,
            self.times_samples(which=which),
            load_ci,
            self.sorting.channels[which],
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            fill_value=np.nan,
        )
        if not self.shifting:
            return which, waveforms

        (
            waveforms,
            max_chan,
            show_geom,
            show_channel_index,
        ) = self.unit_shift_or_relocate_channels(
            unit_id,
            which,
            waveforms,
            load_ci,
            show_radius_um=show_radius_um,
            relocated=relocated,
        )
        return which, waveforms, max_chan, show_geom, show_channel_index

    def unit_tpca_waveforms(
        self,
        unit_id,
        template_index=None,
        max_count=250,
        random_seed=0,
        show_radius_um=75,
        relocated=False,
    ):
        which = self.in_unit(unit_id)
        if template_index is not None:
            assert template_index in self.unit_template_indices(unit_id)
            which = self.in_template(template_index)
        if which.size > max_count:
            rg = np.random.default_rng(random_seed)
            which = rg.choice(which, size=max_count, replace=False)
            which.sort()
        if not which.size:
            return which, None, None, None, None

        tpca_embeds = self.tpca_features(which=which)
        n, rank, c = tpca_embeds.shape
        waveforms = tpca_embeds.transpose(0, 2, 1).reshape(n * c, rank)
        waveforms = self.sklearn_tpca.inverse_transform(waveforms)
        t = waveforms.shape[1]
        waveforms = waveforms.reshape(n, c, t).transpose(0, 2, 1)

        (
            waveforms,
            max_chan,
            show_geom,
            show_channel_index,
        ) = self.unit_shift_or_relocate_channels(
            unit_id,
            which,
            waveforms,
            self.channel_index,
            show_radius_um=show_radius_um,
            relocated=relocated,
        )
        return which, waveforms, max_chan, show_geom, show_channel_index

    def unit_pca_features(
        self,
        unit_id,
        relocated=True,
        rank=2,
        pca_radius_um=75,
        random_seed=0,
        max_count=500,
    ):
        (
            which,
            waveforms,
            max_chan,
            show_geom,
            show_channel_index,
        ) = self.unit_tpca_waveforms(
            unit_id,
            relocated=relocated,
            show_radius_um=pca_radius_um,
            random_seed=random_seed,
            max_count=max_count,
        )
        waveforms = waveforms.reshape(len(waveforms), -1)

        no_nan = np.flatnonzero(~np.isnan(waveforms).any(axis=1))
        features = np.full((len(waveforms), rank), np.nan, dtype=waveforms.dtype)
        if no_nan.size < rank:
            return which, features

        pca = PCA(rank, random_state=random_seed, whiten=True)
        features[no_nan] = pca.fit_transform(waveforms[no_nan])
        return which, features

    def unit_shift_or_relocate_channels(
        self,
        unit_id,
        which,
        waveforms,
        load_channel_index,
        show_radius_um=75,
        relocated=False,
    ):
        geom = self.recording.get_channel_locations()
        show_geom = self.template_data.registered_geom
        if show_geom is None:
            show_geom = geom
        temp = self.coarse_template_data.unit_templates(unit_id)
        n_pitches_shift = None
        if temp.shape[0]:
            max_chan = temp.squeeze().ptp(0).argmax()
        else:
            amps = waveforms.ptp(1)
            if self.shifting:
                n_pitches_shift = get_spike_pitch_shifts(
                    self.z(which=which, registered=False),
                    geom=geom,
                    registered_depths_um=self.z(which=which, registered=True),
                    times_s=self.times_seconds(which=which),
                    motion_est=self.motion_est,
                )
                amp_template = registered_average(
                    amps[:, None, :],
                    n_pitches_shift,
                    geom,
                    show_geom,
                    main_channels=self.sorting.channels[which],
                    channel_index=load_channel_index,
                )[0]
            else:
                amp_template = np.nanmean(amps, axis=0)
            max_chan = np.nanargmax(amp_template)
        show_channel_index = make_channel_index(show_geom, show_radius_um)
        show_chans = show_channel_index[max_chan]
        show_chans = show_chans[show_chans < len(show_geom)]
        show_channel_index = np.broadcast_to(
            show_chans[None], (len(show_geom), show_chans.size)
        )

        if not self.shifting:
            return waveforms, max_chan, show_geom, show_channel_index

        if relocated:
            waveforms = relocate.relocated_waveforms_on_static_channels(
                waveforms,
                main_channels=self.sorting.channels[which],
                channel_index=load_channel_index,
                xyza_from=self.xyza[which],
                target_channels=show_chans,
                z_to=self.z(which=which, registered=True),
                geom=geom,
                registered_geom=show_geom,
            )
            return waveforms, max_chan, show_geom, show_channel_index

        if n_pitches_shift is None:
            n_pitches_shift = get_spike_pitch_shifts(
                self.z(which=which, registered=False),
                geom=geom,
                registered_depths_um=self.z(which=which, registered=True),
                times_s=self.times_seconds(which=which),
                motion_est=self.motion_est,
            )

        waveforms = get_waveforms_on_static_channels(
            waveforms,
            geom=geom,
            n_pitches_shift=n_pitches_shift,
            main_channels=self.sorting.channels[which],
            channel_index=load_channel_index,
            target_channels=show_chans,
            registered_geom=show_geom,
        )

        return waveforms, max_chan, show_geom, show_channel_index

    def nearby_coarse_templates(self, unit_id, n_neighbors=5):
        unit_ix = np.searchsorted(self.unit_ids, unit_id)
        unit_dists = self.merge_dist[unit_ix]
        distance_order = np.argsort(unit_dists)
        assert distance_order[0] == unit_ix
        neighbor_ixs = distance_order[:n_neighbors]
        neighbor_ids = self.unit_ids[neighbor_ixs]
        neighbor_dists = self.merge_dist[neighbor_ixs[:, None], neighbor_ixs[None, :]]
        neighbor_coarse_templates = self.coarse_template_data.templates[neighbor_ixs]
        return neighbor_ids, neighbor_dists, neighbor_coarse_templates

    # computation

    def _calc_merge_dist(self):
        """Compute the merge distance matrix"""
        merge_td = self.template_data
        if self.merge_distance_templates_kind == "coarse":
            merge_td = self.coarse_template_data

        units, dists, shifts, template_snrs = merge.calculate_merge_distances(
            merge_td,
            superres_linkage=self.merge_superres_linkage,
            device=self.device,
            n_jobs=1,
        )
        assert np.array_equal(units, self.unit_ids)
        return dists


@dataclass
class DARTsortGroundTruthComparison:
    gt_analysis: DARTsortAnalysis
    predicted_analysis: DARTsortAnalysis
    gt_name: Optional[str] = None
    predicted_name: Optional[str] = None
    delta_time: float = 0.4
    match_score: float = 0.1
    well_detected_score: float = 0.8
    exhaustive_gt: bool = False
    n_jobs: int = -1
    match_mode: str = "hungarian"

    def __post_init__(self):
        self.comparison = GroundTruthComparison(
            gt_sorting=self.gt_analysis.sorting.to_numpy_sorting(),
            tested_sorting=self.predicted_analysis.sorting.to_numpy_sorting(),
            gt_name=self.gt_name,
            predicted_name=self.predicted_name,
            delta_time=self.delta_time,
            match_score=self.match_score,
            well_detected_score=self.well_detected_score,
            exhaustive_gt=self.exhaustive_gt,
            n_jobs=self.n_jobs,
            match_mode=self.match_mode,
        )

    def get_match(self, gt_unit):
        pass

    def get_spikes_by_category(self, gt_unit, predicted_unit=None):
        if predicted_unit is None:
            predicted_unit = self.get_match(gt_unit)

        return dict(
            matched_predicted_indices=...,
            matched_gt_indices=...,
            only_gt_indices=...,
            only_predicted_indices=...,
        )

    def get_performance(self, gt_unit):
        pass

    def get_waveforms_by_category(self, gt_unit, predicted_unit=None):
        return ...


# -- h5 helper... slow reading...


def batched_h5_read(dataset, indices, batch_size=1000):
    if indices.size < batch_size:
        return dataset[indices]
    else:
        out = np.empty((indices.size, *dataset.shape[1:]), dtype=dataset.dtype)
        for bs in range(0, indices.size, batch_size):
            be = min(indices.size, bs + batch_size)
            out[bs:be] = dataset[indices[bs:be]]
        return out
