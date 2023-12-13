"""Deeper object-oriented interaction with sorter data

This is meant to make implementing plotting code easier: this
code becomes the model in a MVC framework, and vis/unit.py can
implement a view and controller.

This should also make it easier to compute drift-aware metrics
(e.g., d' using registered templates and shifted waveforms).
"""
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import spikeinterface.core as sc
from dredge.motion_util import MotionEstimate
from sklearn.decomposition import PCA

from ..cluster import relocate
from ..templates import TemplateData
from ..transform import WaveformPipeline
from .data_util import DARTsortSorting
from .drift_util import (get_spike_pitch_shifts,
                         get_waveforms_on_static_channels)
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
    hdf5_path: Path
    recording: sc.BaseRecording
    template_data: TemplateData
    featurization_pipeline: Optional[WaveformPipeline] = None
    motion_est: Optional[MotionEstimate] = None

    # hdf5 keys
    localizations_dataset = "point_source_localizations"
    amplitudes_dataset = "denoised_amplitudes"
    amplitude_vectors_dataset = "denoised_amplitude_vectors"
    tpca_features_dataset = "collisioncleaned_tpca_features"

    # helper constructors

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
        **kwargs,
    ):
        hdf5_path = Path(hdf5_path)
        if model_dir is None:
            model_dir = hdf5_path.parent / f"{hdf5_path.stem}_models"
            assert model_dir.exists()
        sorting = DARTsortSorting.from_peeling_hdf5(
            hdf5_path, load_simple_features=False
        )
        template_data = TemplateData.from_npz(Path(model_dir) / template_data_npz)
        pipeline = torch.load(model_dir / "featurization_pipeline.pt")
        return cls(
            sorting, hdf5_path, recording, template_data, pipeline, motion_est, **kwargs
        )

    # pickle/h5py gizmos

    def __post_init__(self):
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

        # cached hdf5 pointer
        self._h5 = None

        # cached arrays
        self.clear_cache()

    def clear_cache(self):
        self._xyza = None
        self._max_chan_amplitudes = None
        self._amplitude_vectors = None
        self._channel_index = None
        self._geom = None
        self._tpca_features = None
        self._sklearn_tpca = None
        self._feats = {}

    def __getstate__(self):
        # remove cached stuff before pickling
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

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

    # spike train helpers

    def unit_ids(self):
        allunits = np.unique(self.sorting.labels)
        return allunits[allunits >= 0]

    def in_unit(self, unit_id):
        return np.flatnonzero(np.isin(self.sorting.labels, unit_id))

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
            z = self.motion_est.correct_s(self.sorting.times_seconds, z)
        return z

    def times_seconds(self, which=slice(None)):
        return self.sorting.times_seconds[which]

    def times_samples(self, which=slice(None)):
        return self.sorting.times_samples[which]

    def amplitudes(self, which=slice(None), relocated=False):
        if not relocated or self.motion_est is None:
            return self.max_chan_amplitudes[which]

        reloc_amp_vecs = relocate.relocated_waveforms_on_static_channels(
            self.amplitude_vectors[which],
            main_channels=self.channels[which],
            channel_index=self.channel_index,
            xyza_from=self.xyza[which],
            z_to=self.z(which),
            geom=self.geom,
            registered_geom=self.template_data.registered_geom,
        )
        return reloc_amp_vecs.max(1)

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
        max_count=250,
        random_seed=0,
        show_radius_um=75,
        trough_offset_samples=42,
        spike_length_samples=121,
        relocated=False,
    ):
        which = self.in_unit(unit_id)
        if which.size > max_count:
            rg = np.random.default_rng(0)
            which = rg.choice(which, size=max_count, replace=False)
        if not which.size:
            return np.zeros((0, spike_length_samples, 1))

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
            return waveforms

        return self.unit_shift_or_relocate_channels(
            unit_id,
            which,
            waveforms,
            load_ci,
            show_radius_um=show_radius_um,
            relocated=relocated,
        )

    def unit_tpca_waveforms(
        self,
        unit_id,
        max_count=250,
        random_seed=0,
        show_radius_um=75,
        relocated=False,
    ):
        which = self.in_unit(unit_id)
        if not which.size:
            return np.zeros((0, 1, self.channel_index.shape[1]))

        tpca_embeds = self.tpca_features(which=which)
        n, rank, c = tpca_embeds.shape
        waveforms = tpca_embeds.transpose(0, 2, 1).reshape(n * c, rank)
        waveforms = self.sklearn_tpca.inverse_transform(waveforms)
        t = waveforms.shape[1]
        waveforms = waveforms.reshape(n, c, t).transpose(0, 2, 1)

        return self.unit_shift_or_relocate_channels(
            unit_id,
            which,
            waveforms,
            self.channel_index,
            show_radius_um=show_radius_um,
            relocate=relocated,
        )

    def unit_pca_features(
        self, unit_id, relocated=True, rank=2, pca_radius_um=75, random_seed=0
    ):
        waveforms = self.unit_tpca_waveforms(
            unit_id,
            relocated=relocated,
            show_radius_um=pca_radius_um,
            random_seed=random_seed,
        )

        no_nan = np.flatnonzero(~np.isnan(waveforms).any(axis=1))
        features = np.full((len(waveforms), rank), np.nan, dtype=waveforms.dtype)
        if no_nan.size < max(self.min_cluster_size, self.n_pca_features):
            return features

        pca = PCA(self.n_pca_features, random_state=random_seed, whiten=True)
        features[no_nan] = pca.fit_transform(waveforms[no_nan])
        return features

    def unit_shift_or_relocate_channels(
        self,
        unit_id,
        which,
        waveforms,
        load_channel_index,
        show_radius_um=75,
        relocate=False,
    ):
        geom = self.recording.get_channel_locations()
        show_geom = self.recording.registered_geom
        if show_geom is None:
            show_geom is geom
        temp = self.coarse_templates.templates[
            self.coarse_templates.unit_ids == unit_id
        ]
        assert temp.shape[0] == 1
        max_chan = temp.squeeze().ptp(0).argmax()
        show_channel_index = make_channel_index(show_geom, show_radius_um)
        show_chans = show_channel_index[max_chan]

        if relocate:
            return relocate.relocated_waveforms_on_static_channels(
                waveforms,
                main_channels=self.sorting.channels[which],
                channel_index=load_channel_index,
                xyza_from=self.xyza[which],
                target_channels=show_chans,
                z_to=self.z(which=which, registered=True),
                geom=geom,
                registered_geom=show_geom,
            )

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
