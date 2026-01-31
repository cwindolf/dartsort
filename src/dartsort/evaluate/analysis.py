"""Deeper object-oriented interaction with sorter data

This is meant to make implementing plotting code easier: this
code becomes the model in a MVC framework, and vis/unit.py can
implement a view and controller.

This should also make it easier to compute drift-aware metrics
(e.g., d' using registered templates and shifted waveforms).
"""

from dataclasses import dataclass

import numpy as np
import spikeinterface.core as sc
import torch
from dredge.motion_util import MotionEstimate
from sklearn.decomposition import PCA

from ..clustering import merge
from ..util.internal_config import (
    TemplateConfig,
    TemplateMergeConfig,
    ComputationConfig,
    ClusteringFeaturesConfig,
    default_clustering_features_cfg,
)
from ..templates import TemplateData
from ..util.data_util import (
    DARTsortSorting,
    get_tpca,
    try_get_model_dir,
    sorting_isis,
)
from ..util.drift_util import (
    get_spike_pitch_shifts,
    get_waveforms_on_static_channels,
    get_stable_channels,
)
from ..util.interpolation_util import StableFeaturesInterpolator, pad_geom
from ..util.py_util import databag
from ..util.spikeio import read_waveforms_channel_index
from ..util.waveform_util import make_channel_index
from ..util import job_util, logging_util


logger = logging_util.get_logger(__name__)


@dataclass
class DARTsortAnalysis:
    """Stores all relevant properties for a drift-aware waveform analysis"""

    sorting: DARTsortSorting
    recording: sc.BaseRecording
    template_data: TemplateData | None
    coarse_template_data: TemplateData | None
    motion_est: MotionEstimate | None
    merge_distances: np.ndarray | None
    geom: np.ndarray
    registered_geom: np.ndarray
    extract_channel_index: np.ndarray | None
    vis_channel_index: np.ndarray
    xyza: np.ndarray | None
    x: np.ndarray | None
    z: np.ndarray | None
    registered_z: np.ndarray | None
    shifting: bool
    times_seconds: np.ndarray | None
    amplitudes: np.ndarray | None
    unit_ids: np.ndarray
    spike_counts: np.ndarray
    amplitude_vectors: np.ndarray | None
    erp: StableFeaturesInterpolator | None
    sklearn_tpca: PCA | None
    tpca_temporal_slice: slice
    name: str | None = None
    vis_radius: float = 50.0
    trough_offset_samples: int = 42
    spike_length_samples: int = 121
    tpca_features_dset: str = "collisioncleaned_tpca_features"

    @classmethod
    def from_sorting(
        cls,
        recording: sc.BaseRecording,
        sorting: DARTsortSorting,
        motion_est=None,
        name: str | None = None,
        template_data: TemplateData | None = None,
        template_cfg: TemplateConfig | None = TemplateConfig(denoising_method="none"),
        template_merge_cfg: TemplateMergeConfig = TemplateMergeConfig(
            min_spatial_cosine=0.8
        ),
        clustering_features_cfg: ClusteringFeaturesConfig = default_clustering_features_cfg,
        computation_cfg: ComputationConfig | None = None,
        vis_radius: float = 50.0,
        vis_neighborhood_p: float = np.inf,
    ):
        """Try to re-load as much info as possible from the sorting itself

        Templates are re-computed if labels are not the same as in h5
        or if the template npz does not exist.
        """
        computation_cfg = job_util.ensure_computation_config(computation_cfg)
        has_hdf5 = sorting.parent_h5_path is not None

        if has_hdf5 and vis_radius and (tpca := get_tpca(sorting)) is not None:
            sklearn_tpca = tpca.to_sklearn()  # type: ignore
            tpca_temporal_slice = sklearn_tpca.temporal_slice
        else:
            sklearn_tpca = None
            tpca_temporal_slice = slice(None)

        model_dir = try_get_model_dir(sorting)
        if template_data is not None:
            pass
        elif has_hdf5 and model_dir is not None:
            template_npz = model_dir / "template_data.npz"
            can_reload = sorting.has_persistent_labels()
            if can_reload and template_npz.exists():
                logger.info(f"Reloading templates from {template_npz}...")
                template_data = TemplateData.from_npz(template_npz)

        if template_data is None and template_cfg is not None:
            template_data = TemplateData.from_config(
                recording,
                sorting,
                template_cfg,
                overwrite=False,
                motion_est=motion_est,
                computation_cfg=computation_cfg,
            )

        if template_data is not None:
            coarse_template_data = template_data.coarsen()
            merge_distances = merge.get_merge_distances(
                template_data=coarse_template_data,
                template_merge_cfg=template_merge_cfg,
                computation_cfg=computation_cfg,
                sampling_frequency=recording.sampling_frequency,
            )[1]
            trough_offset_samples = template_data.trough_offset_samples
            spike_length_samples = template_data.spike_length_samples
        else:
            trough_offset_samples = spike_length_samples = 0
            coarse_template_data = merge_distances = None

        channel_index = getattr(sorting, "channel_index", None)
        amplitudes = getattr(
            sorting, clustering_features_cfg.amplitudes_dataset_name, None
        )
        amplitude_vecs = getattr(
            sorting, clustering_features_cfg.amplitude_vectors_dataset_name, None
        )
        xyza = getattr(
            sorting, clustering_features_cfg.localizations_dataset_name, None
        )
        tpca_features_dset = clustering_features_cfg.pca_dataset_name
        times_seconds = getattr(sorting, "times_seconds", None)
        assert times_seconds is not None
        if motion_est is None and xyza is not None:
            reg_z = xyza[:, 2]
        elif xyza is not None:
            assert motion_est is not None
            reg_z = motion_est.correct_s(times_seconds, xyza[:, 2])
        else:
            reg_z = None
        if xyza is not None:
            x = xyza[:, 0]
            z = xyza[:, 2]
        else:
            x = z = None

        geom = recording.get_channel_locations()
        if motion_est is None:
            rgeom = geom
            if template_data is not None:
                trg = template_data.registered_geom
                assert (trg is None) or np.array_equal(rgeom, trg)
        else:
            assert template_data is not None
            rgeom = template_data.registered_geom
        assert rgeom is not None

        device = computation_cfg.actual_device()
        if vis_radius and channel_index is not None:
            # interping to geom with shifts on fly rather than rgeom.
            erp = StableFeaturesInterpolator(
                source_geom=pad_geom(geom, device=device),
                target_geom=pad_geom(geom, device=device),
                channel_index=torch.asarray(channel_index, device=device),
                params=clustering_features_cfg.interp_params,
            )
        else:
            erp = None

        unit_ids, spike_counts = np.unique(sorting.labels, return_counts=True)  # type: ignore
        spike_counts = spike_counts[unit_ids >= 0]
        unit_ids = unit_ids[unit_ids >= 0]

        return cls(
            sorting=sorting,
            recording=recording,
            template_data=template_data,
            coarse_template_data=coarse_template_data,
            motion_est=motion_est,
            merge_distances=merge_distances,
            geom=geom,
            registered_geom=rgeom,
            extract_channel_index=channel_index,
            vis_channel_index=make_channel_index(
                geom=rgeom, radius=vis_radius, p=vis_neighborhood_p, to_torch=False
            ),
            xyza=xyza,
            x=x,
            z=z,
            registered_z=reg_z,
            shifting=motion_est is not None,
            times_seconds=times_seconds,
            amplitudes=amplitudes,
            amplitude_vectors=amplitude_vecs,
            erp=erp,
            sklearn_tpca=sklearn_tpca,
            tpca_temporal_slice=tpca_temporal_slice,
            name=name,
            vis_radius=vis_radius,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            tpca_features_dset=tpca_features_dset,
            unit_ids=unit_ids,
            spike_counts=spike_counts,
        )

    def in_unit(self, unit_id):
        assert self.sorting.labels is not None
        return np.flatnonzero(np.isin(self.sorting.labels, unit_id))

    def in_template(self, template_index):
        template_indices = getattr(self.sorting, "template_inds", None)
        assert template_indices is not None
        return np.flatnonzero(np.isin(template_indices, template_index))

    def unit_template_indices(self, unit_id):
        assert self.template_data is not None
        return np.flatnonzero(self.template_data.unit_ids == unit_id)

    def unit_amplitudes(self, unit_id=None):
        assert self.template_data is not None
        if unit_id is not None:
            return np.ptp(self.template_data.unit_templates(unit_id))
        amplitudes = np.zeros(self.sorting.unit_ids.shape)
        for j, unit_id in enumerate(self.sorting.unit_ids):
            temps = self.template_data.unit_templates(unit_id)
            amplitudes[j] = np.ptp(np.nan_to_num(temps))
        return amplitudes

    def firing_rates(self):
        assert self.sorting.labels is not None
        assert np.array_equal(self.unit_ids, self.sorting.unit_ids)
        frs = self.spike_counts / self.recording.get_duration()
        return frs

    def named_feature(self, fname: str, which: slice | np.ndarray):
        if hasattr(self, fname):
            return getattr(self, fname)[which]
        else:
            return self.sorting.slice_feature_by_name(fname, mask=which)

    # cluster-dependent feature loading methods

    def unit_raw_waveforms(
        self,
        unit_id: int | None = None,
        which: np.ndarray | None = None,
        template_index: int | None = None,
        max_count=250,
        main_channel=None,
        random_seed=0,
        dtype=np.float32,
    ) -> "WaveformsBag | None":
        """Load raw waveforms for visualization"""
        if which is not None:
            pass
        elif template_index is not None:
            assert template_index in self.unit_template_indices(unit_id)
            which = self.in_template(template_index)
        elif unit_id is not None:
            which = self.in_unit(unit_id)
        else:
            assert False

        if max_count is not None and which.size > max_count:
            rg = np.random.default_rng(random_seed)
            which = rg.choice(which, size=max_count, replace=False)
            which.sort()

        times = self.sorting.times_samples[which]
        mnt = self.trough_offset_samples
        mxt = self.recording.get_num_samples() - self.spike_length_samples
        which = which[times == times.clip(mnt, mxt)]

        if not which.size:
            return None

        # read waveforms from disk
        read_chans = self.sorting.channels[which]
        if self.extract_channel_index is None:
            read_channel_index = self.vis_channel_index
        else:
            read_channel_index = self.extract_channel_index
        waveforms = read_waveforms_channel_index(
            recording=self.recording,
            times_samples=self.sorting.times_samples[which],
            channel_index=read_channel_index,
            main_channels=read_chans,
            trough_offset_samples=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
            fill_value=np.nan,
        )
        waveforms = waveforms.astype(dtype)

        waveforms, main_channel = self.unit_select_channels(
            unit_id=unit_id,
            which=which,
            waveforms=waveforms,
            read_chans=read_chans,
            main_channel=main_channel,
        )
        return WaveformsBag(
            which=which,
            waveforms=waveforms,
            main_channel=main_channel,
            geom=self.registered_geom,
            channel_index=self.vis_channel_index,
            temporal_slice=None,
        )

    def tpca_features(self, which: np.ndarray):
        assert self.erp is not None
        features = self.sorting.slice_feature_by_name(
            self.tpca_features_dset, mask=which
        )
        device = self.erp.b.source_geom.device
        channels = torch.asarray(self.sorting.channels[which], device=device)
        if self.motion_est is None:
            shifts = torch.zeros(channels.shape, device=device)
        else:
            assert self.z is not None
            assert self.registered_z is not None
            # target will be geom - shift
            # want to move from original position to reg pos = z - disp
            # so, should take shift=disp=z-reg_z
            shifts = self.z[which] - self.registered_z[which]
            shifts = torch.asarray(shifts, device=device).float()
        features = self.erp.interp(
            features=torch.asarray(features, device=device),
            source_main_channels=channels,
            target_channels=self.erp.b.channel_index[channels],
            source_shifts=shifts,
        )
        return features.numpy(force=True)

    def unit_tpca_waveforms(
        self,
        unit_id,
        which: np.ndarray | None = None,
        template_index=None,
        max_count=250,
        random_seed=0,
    ) -> "WaveformsBag | None":
        assert self.sklearn_tpca is not None
        if which is not None:
            pass
        elif template_index is not None:
            assert template_index in self.unit_template_indices(unit_id)
            which = self.in_template(template_index)
        elif unit_id is not None:
            which = self.in_unit(unit_id)
        else:
            assert False

        if max_count is not None and which.size > max_count:
            rg = np.random.default_rng(random_seed)
            which = rg.choice(which, size=max_count, replace=False)
            which.sort()

        if not which.size:
            return None

        tpca_embeds = self.tpca_features(which=which)
        n, rank, c = tpca_embeds.shape
        tpca_embeds = tpca_embeds.transpose(0, 2, 1).reshape(n * c, rank)
        waveforms = np.full(
            (n * c, self.sklearn_tpca.components_.shape[1]),
            np.nan,
            dtype=tpca_embeds.dtype,
        )
        valid = np.flatnonzero(np.isfinite(tpca_embeds[:, 0]))
        waveforms[valid] = self.sklearn_tpca.inverse_transform(tpca_embeds[valid])
        t = waveforms.shape[1]
        waveforms = waveforms.reshape(n, c, t).transpose(0, 2, 1)

        waveforms, main_channel = self.unit_select_channels(
            unit_id=unit_id, which=which, waveforms=waveforms
        )
        return WaveformsBag(
            which=which,
            waveforms=waveforms,
            main_channel=main_channel,
            geom=self.registered_geom,
            channel_index=self.vis_channel_index,
            temporal_slice=self.tpca_temporal_slice,
        )

    def unit_pca_features(
        self,
        unit_id,
        rank=2,
        random_seed=0,
        max_count=500,
        max_wfs_fit=10_000,
        random_state=0,
    ):
        tpca_waves = self.unit_tpca_waveforms(
            unit_id,
            random_seed=random_seed,
            max_count=max_count,
        )
        if tpca_waves is None:
            return None, None

        waveforms = tpca_waves.waveforms

        # remove chans with no signal at all
        not_entirely_nan_channels = np.flatnonzero(
            np.isfinite(waveforms[:, 0]).any(axis=0)
        )
        if (
            not_entirely_nan_channels.size
            and not_entirely_nan_channels.size < waveforms.shape[2]
        ):
            waveforms = waveforms[:, :, not_entirely_nan_channels]

        waveforms = waveforms.reshape(len(waveforms), -1)
        no_nan = np.flatnonzero(~np.isnan(waveforms).any(axis=1))

        features = np.full((len(waveforms), rank), np.nan, dtype=waveforms.dtype)
        if no_nan.size < rank:
            return tpca_waves.which, features

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
        features = pca.transform(
            np.where(np.isfinite(waveforms), waveforms, pca.mean_[None])
        )
        return tpca_waves.which, features

    def unit_max_channel(self, unit_id) -> int:
        assert self.coarse_template_data is not None
        temp = self.coarse_template_data.unit_templates(unit_id)
        assert temp.ndim == 3 and temp.shape[0] == np.atleast_1d(unit_id).size

        which = self.in_unit(unit_id)
        if self.motion_est is not None:
            assert self.z is not None
            assert self.registered_z is not None
            times_seconds = getattr(self.sorting, "times_seconds", None)
            assert times_seconds is not None
            n_pitches_shift = get_spike_pitch_shifts(
                self.z[which],
                geom=self.geom,
                registered_depths_um=self.registered_z[which],
                times_s=times_seconds[which],
                motion_est=self.motion_est,
            )
            covered_chans = get_stable_channels(
                geom=self.geom,
                registered_geom=self.registered_geom,
                channels=self.sorting.channels[which],
                channel_index=self.vis_channel_index,
                n_pitches_shift=n_pitches_shift,
            )[0]
        else:
            covered_chans = self.vis_channel_index[self.sorting.channels[which]]
        covered_chans = np.unique(covered_chans)
        covered_chans = covered_chans[covered_chans < len(self.vis_channel_index)]
        amp_vec = np.ptp(temp.mean(0), 0)
        max_chan = covered_chans[amp_vec[covered_chans].argmax()]
        return max_chan.item()

    def unit_select_channels(
        self,
        unit_id,
        which,
        waveforms,
        read_chans=None,
        main_channel=None,
    ):
        if read_chans is None:
            read_chans = self.sorting.channels[which]
        if main_channel is None:
            main_channel = self.unit_max_channel(unit_id)

        show_chans = self.vis_channel_index[main_channel]
        show_valid = show_chans < len(self.registered_geom)
        show_chans = show_chans[show_valid]

        if self.shifting:
            assert self.z is not None
            assert self.registered_z is not None
            times_seconds = getattr(self.sorting, "times_seconds", None)
            assert times_seconds is not None
            n_pitches_shift = get_spike_pitch_shifts(
                self.z[which],
                geom=self.geom,
                registered_depths_um=self.registered_z[which],
                times_s=times_seconds[which],
                motion_est=self.motion_est,
            )
        else:
            n_pitches_shift = None

        if self.extract_channel_index is None:
            read_channel_index = self.vis_channel_index
        else:
            read_channel_index = self.extract_channel_index

        waveforms_valid = get_waveforms_on_static_channels(
            waveforms=waveforms,
            geom=self.geom,
            n_pitches_shift=n_pitches_shift,
            main_channels=read_chans,
            channel_index=read_channel_index,
            target_channels=show_chans,
            registered_geom=self.registered_geom,
        )
        waveforms = np.full_like(
            waveforms_valid,
            shape=(*waveforms_valid.shape[:2], show_valid.size),
            fill_value=np.nan,
        )
        waveforms[:, :, show_valid] = waveforms_valid

        return waveforms, main_channel

    def nearby_coarse_templates(self, unit_id, n_neighbors=5):
        td = self.coarse_template_data
        assert td is not None
        assert self.merge_distances is not None

        unit_ix = np.searchsorted(td.unit_ids, unit_id)
        unit_dists = self.merge_distances[unit_ix]
        distance_order = np.argsort(unit_dists)
        distance_order = np.concatenate(
            ([unit_ix], distance_order[distance_order != unit_ix])
        )
        # assert distance_order[0] == unit_ix
        neighb_ixs = distance_order[:n_neighbors]
        neighb_ids = td.unit_ids[neighb_ixs]
        neighb_dists = self.merge_distances[neighb_ixs[:, None], neighb_ixs[None, :]]
        neighb_coarse_templates = td.templates[neighb_ixs]
        return neighb_ixs, neighb_ids, neighb_dists, neighb_coarse_templates

    def spike_isis(self):
        return sorting_isis(self.sorting)

    def viol_rate(self, dt_ms=0.8):
        viol_rates = np.zeros(self.unit_ids.shape)
        for j, uid in enumerate(self.unit_ids):
            inu = self.in_unit(uid)
            if not inu.size > 1:
                continue
            t_ms = self.sorting.times_seconds[inu] * 1000  # type: ignore
            isi = np.diff(t_ms)
            viol_rates[j] = (np.abs(isi) < dt_ms).mean()
        return viol_rates


@databag
class WaveformsBag:
    temporal_slice: slice | None
    which: np.ndarray
    waveforms: np.ndarray
    main_channel: int
    geom: np.ndarray
    channel_index: np.ndarray
