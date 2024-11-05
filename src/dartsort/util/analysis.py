"""Deeper object-oriented interaction with sorter data

This is meant to make implementing plotting code easier: this
code becomes the model in a MVC framework, and vis/unit.py can
implement a view and controller.

This should also make it easier to compute drift-aware metrics
(e.g., d' using registered templates and shifted waveforms).
"""

import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import spikeinterface.core as sc
import torch
from dredge.motion_util import MotionEstimate
from sklearn.decomposition import PCA

from ..cluster import merge, relocate
from ..config import TemplateConfig
from ..templates import TemplateData
from ..transform import WaveformPipeline
from .data_util import DARTsortSorting, batched_h5_read
from .drift_util import (get_spike_pitch_shifts,
                         get_waveforms_on_static_channels, registered_average)
from .spikeio import read_waveforms_channel_index
from .waveform_util import make_channel_index

no_realign_template_config = TemplateConfig(realign_peaks=False)
basic_template_config = TemplateConfig(realign_peaks=False, superres_templates=False)


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
    name: Optional[str] = None

    # hdf5 keys
    localizations_dataset: str = "point_source_localizations"
    amplitudes_dataset_name: str = "denoised_ptp_amplitudes"
    amplitude_vectors_dataset: str = "denoised_ptp_amplitude_vectors"
    tpca_features_dataset: str = "collisioncleaned_tpca_features"
    template_indices_dataset: str = "template_indices"

    # configuration for analysis computations not included in above objects
    device: Optional[torch.device] = None
    merge_distance_templates_kind: str = "coarse"
    merge_distance_kind: str = "rms"
    merge_distance_spatial_radius_a: Optional[float] = None
    merge_distance_min_channel_amplitude: float = 0.0
    merge_distance_min_spatial_cosine: float = 0.5
    merge_temporal_upsampling: int = 1
    merge_superres_linkage: Callable[[np.ndarray], float] = np.max
    compute_distances: bool = "if_hdf5"
    n_jobs: int = 0
    default_channel_index_radius: float = 100.0

    # helper constructors

    @classmethod
    def from_sorting(
        cls,
        recording,
        sorting,
        motion_est=None,
        name=None,
        template_config=no_realign_template_config,
        allow_template_reload=False,
        n_jobs=0,
        denoising_tsvd=None,
        **kwargs,
    ):
        """Try to re-load as much info as possible from the sorting itself

        Templates are re-computed if labels are not the same as in h5
        or if the template npz does not exist.
        """
        assert hasattr(sorting, "parent_h5_path")
        hdf5_path = sorting.parent_h5_path
        model_dir = hdf5_path.parent / f"{hdf5_path.stem}_models"
        assert model_dir.exists()

        featurization_pipeline = torch.load(
            model_dir / "featurization_pipeline.pt",
            weights_only=True,
        )

        have_templates = False
        if allow_template_reload:
            template_npz = model_dir / "template_data.npz"
            have_templates = template_npz.exists()
            if have_templates:
                print(f"Reloading templates from {template_npz}...")
                with h5py.File(hdf5_path, "r", libver="latest", locking=False) as h5:
                    same_labels = np.array_equal(sorting.labels, h5["labels"][:])
                have_templates = have_templates and same_labels
                template_data = TemplateData.from_npz(template_npz)

        if not have_templates:
            template_data = TemplateData.from_config(
                recording,
                sorting,
                template_config,
                overwrite=False,
                motion_est=motion_est,
                n_jobs=n_jobs,
                tsvd=denoising_tsvd,
            )

        return cls(
            sorting=sorting,
            recording=recording,
            template_data=template_data,
            hdf5_path=hdf5_path,
            featurization_pipeline=featurization_pipeline,
            motion_est=motion_est,
            name=name,
            n_jobs=n_jobs,
            **kwargs,
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
            DARTsortSorting.from_peeling_hdf5(
                hdf5_path, load_simple_features=False
            ),
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
            template_data = TemplateData.from_npz(
                Path(model_dir) / template_data_npz
            )
        if motion_est is None:
            if (hdf5_path.parent / motion_est_pkl).exists():
                with open(hdf5_path.parent / motion_est_pkl, "rb") as jar:
                    motion_est = pickle.load(jar)
        pipeline = torch.load(model_dir / "featurization_pipeline.pt")
        return cls(
            sorting,
            recording,
            template_data,
            hdf5_path,
            pipeline,
            motion_est,
            **kwargs,
        )

    # pickle/h5py gizmos

    def __post_init__(self):
        self.clear_cache()

        if self.featurization_pipeline is not None:
            assert not self.featurization_pipeline.needs_fit()

        if self.hdf5_path is not None:
            assert self.hdf5_path.exists()
        self.coarse_template_data = self.template_data.coarsen()

        self.shifting = (
            self.motion_est is not None
            or self.template_data.registered_geom is not None
        )
        if self.shifting:
            assert (
                self.motion_est is not None
                and self.template_data.registered_geom is not None
            )

        # cached hdf5 pointer
        self._h5 = None
        compute_distances = self.compute_distances
        if self.compute_distances == "if_hdf5":
            compute_distances = self.hdf5_path is not None
        if compute_distances:
            self._calc_merge_dist()

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
        self._feats = {}

    def __getstate__(self):
        # remove cached stuff before pickling
        return {
            k: v if not k.startswith("_") else None
            for k, v in self.__dict__.items()
        }

    # cache gizmos

    @property
    def h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, "r", libver="latest", locking=False)
        return self._h5

    @property
    def xyza(self):
        if self._xyza is None:
            if hasattr(self.sorting, self.localizations_dataset):
                return getattr(self.sorting, self.localizations_dataset)
            else:
                self._xyza = self.h5[self.localizations_dataset][:]
        return self._xyza

    @property
    def template_indices(self):
        if self._template_indices is None:
            if hasattr(self.sorting, self.template_indices_dataset):
                return getattr(self.sorting, self.template_indices_dataset)
            self._template_indices = self.h5[self.template_indices_dataset][:]
        return self._template_indices

    @property
    def max_chan_amplitudes(self):
        if self._max_chan_amplitudes is None:
            if hasattr(self.sorting, self.amplitudes_dataset_name):
                return getattr(self.sorting, self.amplitudes_dataset_name)
            self._max_chan_amplitudes = self.h5[self.amplitudes_dataset_name][:]
        return self._max_chan_amplitudes

    @property
    def amplitude_vectors(self):
        if self._amplitude_vectors is None:
            if hasattr(self.sorting, self.amplitude_vectors_dataset):
                return getattr(self.sorting, self.amplitude_vectors_dataset)
            self._amplitude_vectors = self.h5[self.amplitude_vectors_dataset][:]
        return self._amplitude_vectors

    @property
    def geom(self):
        if self._geom is None:
            if self.hdf5_path is not None:
                self._geom = self.h5["geom"][:]
            else:
                self._geom = self.recording.get_channel_locations()
        return self._geom

    @property
    def channel_index(self):
        if self._channel_index is None:
            if self.hdf5_path is not None:
                self._channel_index = self.h5["channel_index"][:]
            else:
                self._channel_index = make_channel_index(self.geom, self.default_channel_index_radius)
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

    # spike train helpers

    @property
    def unit_ids(self):
        if self._unit_ids is None:
            allunits, counts = np.unique(
                self.sorting.labels, return_counts=True
            )
            self._unit_ids = allunits[allunits >= 0]
            self._spike_counts = counts[allunits >= 0]
        return self._unit_ids

    @property
    def spike_counts(self):
        if self._spike_counts is None:
            allunits, counts = np.unique(
                self.sorting.labels, return_counts=True
            )
            self._unit_ids = allunits[allunits >= 0]
            self._spike_counts = counts[allunits >= 0]
        return self._spike_counts

    def in_unit(self, unit_id):
        return np.flatnonzero(np.isin(self.sorting.labels, unit_id))

    def in_template(self, template_index):
        return np.flatnonzero(np.isin(self.template_indices, template_index))

    def unit_template_indices(self, unit_id):
        return np.flatnonzero(self.template_data.unit_ids == unit_id)

    @property
    def show_geom(self):
        show_geom = self.template_data.registered_geom
        if show_geom is None:
            show_geom = self.recording.get_channel_locations()
        return show_geom

    def show_channel_index(
        self, channel_show_radius_um=50, channel_dist_p=np.inf
    ):
        return make_channel_index(
            self.show_geom, channel_show_radius_um, p=channel_dist_p
        )

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
            amplitude_vectors=self.amplitude_vectors[which],
            xyza_from=self.xyza[which],
            z_to=self.z(which),
            geom=self.geom,
            registered_geom=self.template_data.registered_geom,
            target_channels=None,
        )
        return np.nanmax(reloc_amp_vecs, axis=1)

    def tpca_features(self, which=slice(None)):
        if self._tpca_features is None:
            self._tpca_features = self.h5[self.tpca_features_dataset]
        if isinstance(which, slice):
            which = np.arange(len(self.sorting))[which]
        return batched_h5_read(self._tpca_features, which)

    # unit level properties

    def unit_amplitudes(self, unit_ids=None):
        if unit_ids is None:
            unit_ids = self.unit_ids
        amplitudes = np.zeros(unit_ids.shape)
        for j, unit_id in enumerate(unit_ids):
            temps = self.template_data.unit_templates(unit_id)
            amplitudes[j] = np.ptp(np.nan_to_num(temps))
        return amplitudes

    def firing_rates(self, unit_ids=None):
        if unit_ids is None:
            counts = self.spike_counts
        else:
            counts = np.zeros(unit_ids.shape)
            for j, unit_id in enumerate(unit_ids):
                (isu,) = np.flatnonzero(self.unit_ids == unit_id)
                counts[j] = self.spike_counts[isu]
        seconds = self.recording.get_duration()
        frs = counts / seconds
        return frs

    # cluster-dependent feature loading methods

    def unit_raw_waveforms(
        self,
        unit_id,
        which=None,
        template_index=None,
        max_count=250,
        random_seed=0,
        channel_show_radius_um=75,
        trough_offset_samples=42,
        spike_length_samples=121,
        channel_dist_p=np.inf,
        max_chan=None,
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
            return (
                which,
                None,
                max_chan,
                self.show_geom,
                self.show_channel_index(
                    channel_show_radius_um=channel_show_radius_um,
                    channel_dist_p=channel_dist_p,
                ),
            )

        # read waveforms from disk
        read_chans = self.sorting.channels[which]
        if max_chan is not None:
            read_chans = np.full_like(read_chans, max_chan)
        waveforms = read_waveforms_channel_index(
            self.recording,
            self.times_samples(which=which),
            self.channel_index,
            read_chans,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            fill_value=np.nan,
        )
        waveforms = waveforms.astype(self.template_data.templates.dtype)

        (
            waveforms,
            max_chan,
            show_geom,
            show_channel_index,
        ) = self.unit_select_channels(
            unit_id,
            which,
            waveforms,
            self.channel_index,
            read_chans=read_chans,
            max_chan=max_chan,
            channel_show_radius_um=channel_show_radius_um,
            channel_dist_p=channel_dist_p,
            relocated=relocated,
        )
        return which, waveforms, max_chan, show_geom, show_channel_index

    def unit_tpca_waveforms(
        self,
        unit_id,
        template_index=None,
        max_count=250,
        random_seed=0,
        channel_show_radius_um=75,
        channel_dist_p=np.inf,
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
            return (
                which,
                None,
                None,
                None,
                self.show_channel_index(
                    channel_show_radius_um=channel_show_radius_um,
                    channel_dist_p=channel_dist_p,
                ),
            )

        tpca_embeds = self.tpca_features(which=which)
        n, rank, c = tpca_embeds.shape
        tpca_embeds = tpca_embeds.transpose(0, 2, 1).reshape(n * c, rank)
        waveforms = np.full(
            (n * c, self.sklearn_tpca.components_.shape[1]),
            np.nan,
            dtype=tpca_embeds.dtype,
        )
        valid = np.flatnonzero(np.isfinite(tpca_embeds[:, 0]))
        waveforms[valid] = self.sklearn_tpca.inverse_transform(
            tpca_embeds[valid]
        )
        t = waveforms.shape[1]
        waveforms = waveforms.reshape(n, c, t).transpose(0, 2, 1)

        (
            waveforms,
            max_chan,
            show_geom,
            show_channel_index,
        ) = self.unit_select_channels(
            unit_id,
            which,
            waveforms,
            self.channel_index,
            channel_show_radius_um=channel_show_radius_um,
            channel_dist_p=channel_dist_p,
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
        max_wfs_fit=10_000,
        random_state=0,
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
            channel_show_radius_um=pca_radius_um,
            random_seed=random_seed,
            max_count=max_count,
            channel_dist_p=2,
        )
        if waveforms is None:
            return None, None

        # remove chans with no signal at all
        not_entirely_nan_channels = np.flatnonzero(
            np.isfinite(waveforms[:, 0]).any(axis=0)
        )
        if not_entirely_nan_channels.size and not_entirely_nan_channels.size < waveforms.shape[2]:
            waveforms = waveforms[:, :, not_entirely_nan_channels]

        waveforms = waveforms.reshape(len(waveforms), -1)
        no_nan = np.flatnonzero(~np.isnan(waveforms).any(axis=1))

        features = np.full(
            (len(waveforms), rank), np.nan, dtype=waveforms.dtype
        )
        if no_nan.size < rank:
            return which, features

        pca = PCA(rank, random_state=random_seed, whiten=True)
        if no_nan.size > max_wfs_fit:
            rg = np.random.default_rng(random_state)
            choices = rg.choice(no_nan, size=max_wfs_fit, replace=False)
            choices.sort()
            pca.fit(waveforms[choices])
            # features[no_nan] = pca.transform(waveforms[no_nan])
        else:
            # features[no_nan] = pca.fit_transform(waveforms[no_nan])
            pca.fit(waveforms[no_nan])
        features = pca.transform(np.where(np.isfinite(waveforms), waveforms, pca.mean_[None]))
        return which, features

    def unit_max_channel(self, unit_id):
        temp = self.coarse_template_data.unit_templates(unit_id)
        assert temp.ndim == 3 and temp.shape[0] == np.atleast_1d(unit_id).size
        max_chan = np.ptp(temp.mean(0), 0).argmax()
        return max_chan

    def get_registered_channels(self, in_unit, n_samples=1000, random_state=0):
        amp_samples = slice(None)
        if in_unit.size > n_samples:
            rg = np.random.default_rng(random_state)
            amp_samples = rg.choice(in_unit.size, size=n_samples, replace=False)
            amp_samples.sort()
        n_pitches_shift = get_spike_pitch_shifts(
            self.z(in_unit[amp_samples], registered=False),
            geom=self.geom,
            registered_depths_um=self.z(in_unit[amp_samples], registered=False),
        )
        amp_vecs = self.amplitude_vectors[in_unit[amp_samples]]
        rgeom = self.template_data.registered_geom
        if rgeom is None:
            rgeom = self.geom
        amplitude_template = registered_average(
            amp_vecs,
            n_pitches_shift,
            self.geom,
            rgeom,
            main_channels=self.sorting.channels[in_unit],
            channel_index=self.channel_index,
        )
        max_registered_channel = amplitude_template.argmax()
        return max_registered_channel

    def unit_select_channels(
        self,
        unit_id,
        which,
        waveforms,
        load_channel_index,
        read_chans=None,
        max_chan=None,
        channel_show_radius_um=75,
        channel_dist_p=np.inf,
        relocated=False,
    ):
        geom = self.recording.get_channel_locations()
        show_geom = self.show_geom
        show_channel_index = self.show_channel_index(
            channel_show_radius_um=channel_show_radius_um,
            channel_dist_p=channel_dist_p,
        )

        if read_chans is None:
            read_chans = self.sorting.channels[which]
        if max_chan is None:
            if self.shifting:
                max_chan = self.get_registered_channels(which)
            else:
                max_chan = self.unit_max_channel(unit_id)

        show_chans = show_channel_index[max_chan]
        show_chans = show_chans[show_chans < len(show_geom)]
        show_channel_index = np.broadcast_to(
            show_chans[None], (len(show_geom), show_chans.size)
        )

        if relocated:
            waveforms = relocate.relocated_waveforms_on_static_channels(
                waveforms,
                main_channels=read_chans,
                channel_index=load_channel_index,
                xyza_from=self.xyza[which],
                target_channels=show_chans,
                amplitude_vectors=self.amplitude_vectors[which],
                z_to=self.z(which=which, registered=True),
                geom=geom,
                registered_geom=show_geom,
            )
            return waveforms, max_chan, show_geom, show_channel_index

        n_pitches_shift = None
        if self.shifting:
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
            main_channels=read_chans,
            channel_index=load_channel_index,
            target_channels=show_chans,
            registered_geom=show_geom,
        )

        return waveforms, max_chan, show_geom, show_channel_index

    def nearby_coarse_templates(self, unit_id, n_neighbors=5):
        td = self.coarse_template_data

        unit_ix = np.searchsorted(td.unit_ids, unit_id)
        unit_dists = self.merge_dist[unit_ix]
        distance_order = np.argsort(unit_dists)
        distance_order = np.concatenate(
            ([unit_ix], distance_order[distance_order != unit_ix])
        )
        # assert distance_order[0] == unit_ix
        neighb_ixs = distance_order[:n_neighbors]
        neighb_ids = td.unit_ids[neighb_ixs]
        neighb_dists = self.merge_dist[neighb_ixs[:, None], neighb_ixs[None, :]]
        neighb_coarse_templates = td.templates[neighb_ixs]
        return neighb_ids, neighb_dists, neighb_coarse_templates

    # computation

    def _calc_merge_dist(self):
        """Compute the merge distance matrix"""
        if hasattr(self, "merge_dist"):
            return
        merge_td = self.template_data
        if self.merge_distance_templates_kind == "coarse":
            merge_td = self.coarse_template_data

        if merge_td.templates.shape[0] <= 1:
            self.merge_dist = np.zeros((1, 1))

        units, dists, shifts, template_snrs = merge.calculate_merge_distances(
            merge_td,
            sym_function=np.maximum,
            superres_linkage=self.merge_superres_linkage,
            distance_kind=self.merge_distance_kind,
            spatial_radius_a=self.merge_distance_spatial_radius_a,
            min_channel_amplitude=self.merge_distance_min_channel_amplitude,
            min_spatial_cosine=self.merge_distance_min_spatial_cosine,
            temporal_upsampling_factor=self.merge_temporal_upsampling,
            device=self.device,
            n_jobs=self.n_jobs,
        )
        assert np.array_equal(units, self.coarse_template_data.unit_ids)
        self.merge_dist = dists
