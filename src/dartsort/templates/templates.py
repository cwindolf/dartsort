import gc
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from spikeinterface.core import BaseRecording

from ..localize.localize_util import localize_waveforms
from ..util.data_util import DARTsortSorting
from ..util.internal_config import (
    ComputationConfig,
    TemplateConfig,
    WaveformConfig,
    WhiteningStrategy,
    default_waveform_cfg,
)
from ..util.logging_util import get_logger
from ..util.motion import MotionInfo
from ..util.noise_util import Whitener
from ..util.py_util import databag
from .template_util import weighted_average

logger = get_logger(__name__)


@databag
class TemplateData:
    # (n_templates, spike_length_samples, n_registered_channels or n_channels)
    templates: np.ndarray
    # (n_templates,) maps template index to unit index (multiple templates can share a unit index)
    unit_ids: np.ndarray
    # (n_templates,) spike count for each template
    spike_counts: np.ndarray
    # (n_templates, n_registered_channels or n_channels) spike count for each channel
    spike_counts_by_channel: np.ndarray | None = None
    # (n_templates, spike_length_samples, n_registered_channels or n_channels)
    raw_std_dev: np.ndarray | None = None

    registered_geom: np.ndarray | None = None
    trough_offset_samples: int
    sampling_frequency: float

    # stores (n_templates, *) arrays of template properties
    # possibilities:
    #  - motion_estimate_bin_centers
    #    For motion_estimate binned superres sortings, these are the drift
    #    bin centers for each template.
    properties: dict[str, np.ndarray] | None = None
    tsvd: TruncatedSVD | PCA | None = None
    whitener: np.ndarray | None = None
    covariance: np.ndarray | None = None
    temporal_kernel: np.ndarray | None = None
    whiten_strategy: WhiteningStrategy = "none"
    featurization_basis: np.ndarray | None = None

    # plugin registry for classes which actually estimate templates to hook into
    _registry: ClassVar = {}
    _algorithm: ClassVar = "base"

    def __post_init__(self):
        assert self.trough_offset_samples < self.spike_length_samples

        ntemp = len(self.templates)
        assert self.unit_ids.shape == (ntemp,)
        assert self.spike_counts.shape == (ntemp,)
        if self.spike_counts_by_channel is not None:
            assert self.spike_counts_by_channel.ndim == 2
            assert self.spike_counts_by_channel.shape[0] == ntemp
        if self.raw_std_dev is not None:
            assert self.raw_std_dev.shape == self.templates.shape

        nc = self.templates.shape[2]
        if self.spike_counts_by_channel is not None:
            assert self.spike_counts_by_channel.shape[1] == nc
        if self.registered_geom is not None:
            assert self.registered_geom.ndim == 2
            assert self.registered_geom.shape[0] == nc
        if self.properties:
            for v in self.properties.values():
                assert v.shape[0] == ntemp

    @property
    def spike_length_samples(self):
        return self.templates.shape[1]

    def snrs_by_channel(self):
        amp_vecs = np.nan_to_num(np.ptp(self.templates, axis=1), nan=-np.inf)
        if self.spike_counts_by_channel is not None:
            amp_vecs *= np.sqrt(self.spike_counts_by_channel)
        return amp_vecs

    def main_channels(self):
        return self.snrs_by_channel().argmax(axis=1)

    def template_locations(self, mode="channel", radius=100.0):
        assert mode in ("localization", "channel")

        if mode == "channel":
            assert self.registered_geom is not None
            return self.registered_geom[self.main_channels()]

        assert mode == "localization"
        rdepths = localize_waveforms(
            waveforms=self.templates,
            geom=self.registered_geom,
            main_channels=self.main_channels(),
            radius=radius,
        )
        rdepths = np.c_[rdepths["x"], rdepths["z_abs"]]
        return rdepths

    def registered_depths_um(self, mode="channel", radius=100.0):
        return self.template_locations(mode=mode, radius=radius)[:, 1]

    @classmethod
    def from_npz(cls, npz_path):
        with np.load(npz_path, allow_pickle=True) as data:
            data = dict(**data)
            data["whiten_strategy"] = str(data["whiten_strategy"])
            if "spike_length_samples" in data:
                del data["spike_length_samples"]  # todo: remove
            if "parent_sorting_hdf5_path" in data:
                del data["parent_sorting_hdf5_path"]  # todo: remove

            properties = {}
            for k, v in data.items():
                if k.startswith("__prop_"):
                    properties[k.removeprefix("__prop_")] = v
            for k in properties:
                del data[f"__prop_{k}"]
            if not properties:
                properties = None
            if "tsvd_components" in data:
                components_ = data.pop("tsvd_components")
                whiten = data.pop("tsvd_whiten")
                tsvd = PCA(components_.shape[0], whiten=whiten)
                tsvd.components_ = components_
                tsvd.mean_ = data.pop("tsvd_mean")
                tsvd.explained_variance_ = data.pop("tsvd_explained_variance")
            else:
                tsvd = None

            return cls(**data, tsvd=tsvd, properties=properties)

    def to_npz(self, npz_path):
        to_save = dict(
            templates=self.templates,
            unit_ids=self.unit_ids,
            spike_counts=self.spike_counts,
            trough_offset_samples=self.trough_offset_samples,
            sampling_frequency=self.sampling_frequency,
            whiten_strategy=self.whiten_strategy,
        )
        if self.registered_geom is not None:
            to_save["registered_geom"] = self.registered_geom
        if self.spike_counts_by_channel is not None:
            to_save["spike_counts_by_channel"] = self.spike_counts_by_channel
        if self.raw_std_dev is not None:
            to_save["raw_std_dev"] = self.raw_std_dev
        if self.whitener is not None:
            to_save["whitener"] = self.whitener
            if self.covariance is not None:
                to_save["covariance"] = self.covariance
            if self.temporal_kernel is not None:
                to_save["temporal_kernel"] = self.temporal_kernel
        if self.featurization_basis is not None:
            to_save["featurization_basis"] = self.featurization_basis
        if not npz_path.parent.exists():
            npz_path.parent.mkdir()
        if self.tsvd is not None:
            whiten = isinstance(self.tsvd, PCA) and self.tsvd.whiten
            to_save["tsvd_whiten"] = whiten
            to_save["tsvd_components"] = self.tsvd.components_
            to_save["tsvd_explained_variance"] = self.tsvd.explained_variance_
            if isinstance(self.tsvd, PCA):
                to_save["tsvd_mean"] = self.tsvd.mean_
            else:
                to_save["tsvd_mean"] = np.zeros_like(self.tsvd.components_[0])
        if self.properties is not None:
            for k, p in self.properties.items():
                to_save[f"__prop_{k}"] = p
        np.savez(npz_path, **to_save)  # type: ignore

    def __getitem__(self, subset):
        if not np.array_equal(self.unit_ids, np.arange(len(self.unit_ids))):
            subset_ixs = np.searchsorted(self.unit_ids, subset, side="right") - 1
            matched = self.unit_ids[subset_ixs] == subset
            assert matched.all()
            subset = subset_ixs
        if self.properties is not None:
            properties = {k: p[subset] for k, p in self.properties.items()}
        else:
            properties = None
        return self.__class__(
            templates=self.templates[subset],
            unit_ids=self.unit_ids[subset],
            spike_counts=self.spike_counts[subset],
            spike_counts_by_channel=(
                self.spike_counts_by_channel[subset]
                if self.spike_counts_by_channel is not None
                else None
            ),
            raw_std_dev=(
                self.raw_std_dev[subset] if self.raw_std_dev is not None else None
            ),
            registered_geom=self.registered_geom,
            trough_offset_samples=self.trough_offset_samples,
            properties=properties,
            tsvd=self.tsvd,
            sampling_frequency=self.sampling_frequency,
            whiten_strategy=self.whiten_strategy,
        )

    def coarsen(self):
        """Weighted average all templates that share a unit id."""
        # update templates
        unit_ids_unique, flat_ids = np.unique(self.unit_ids, return_inverse=True)
        templates = weighted_average(flat_ids, self.templates, self.spike_counts)

        # collect spike counts
        spike_counts = np.zeros(len(templates))
        np.add.at(spike_counts, flat_ids, self.spike_counts)

        return replace(
            self,
            templates=templates,
            unit_ids=unit_ids_unique,
            spike_counts=spike_counts,
            properties=None,
            tsvd=self.tsvd,
        )

    def unit_mask(self, unit_id):
        return np.isin(self.unit_ids, unit_id)

    def unit_templates(self, unit_id):
        return self.templates[self.unit_mask(unit_id)]

    def __init_subclass__(cls):
        logger.dartsortverbose("Register template engine: %s", cls._algorithm)
        cls._registry[cls._algorithm] = cls

    @classmethod
    def from_config(
        cls,
        *,
        recording: BaseRecording,
        sorting: DARTsortSorting | None,
        template_cfg: TemplateConfig,
        waveform_cfg: WaveformConfig = default_waveform_cfg,
        save_folder: Path | None = None,
        overwrite=False,
        motion: MotionInfo | None = None,
        whitener: Whitener | None = None,
        save_npz_name: str | None = "template_data.npz",
        tsvd=None,
        featurization_basis=None,
        computation_cfg: ComputationConfig | None = None,
        show_progress: bool = True,
    ) -> "TemplateData":
        # load if saved already and not overwriting
        if save_folder is not None:
            save_folder = Path(save_folder)
            if not save_folder.exists():
                save_folder.mkdir()
            assert save_npz_name is not None
            npz_path = save_folder / save_npz_name
            if npz_path.exists() and not overwrite:
                return cls.from_npz(npz_path)
        else:
            npz_path = None
        if motion is None:
            motion = MotionInfo.from_motion_est(geom=recording.get_channel_locations())
        if template_cfg.whitening.strategy != "none" and whitener is None:
            assert sorting is not None
            whitener = Whitener.from_config(
                sorting=sorting,
                motion=motion,
                whiten_cfg=template_cfg.whitening,
                computation_cfg=computation_cfg,
            )
        else:
            whitener = None

        if tsvd is None and template_cfg.try_reload_svd and sorting is not None:
            tsvd = _try_reload_svd(sorting, rank=template_cfg.denoising_rank)

        if sorting is None:
            raise ValueError(
                "TemplateData.from_config needs a sorting when its .npz file "
                "does not exist."
            )

        self = cls._registry[template_cfg.actual_algorithm()]._from_config(
            recording=recording,
            sorting=sorting,
            template_cfg=template_cfg,
            waveform_cfg=waveform_cfg,
            motion=motion,
            tsvd=tsvd,
            whitener=whitener,
            computation_cfg=computation_cfg,
            show_progress=show_progress,
        )
        self = replace(self, featurization_basis=featurization_basis)

        gc.collect()
        torch.cuda.empty_cache()

        if save_folder is not None:
            assert npz_path is not None
            self.to_npz(npz_path)

        return self

    @classmethod
    def _from_config(
        cls,
        *,
        recording: BaseRecording,
        sorting: DARTsortSorting,
        template_cfg: TemplateConfig,
        waveform_cfg: WaveformConfig = default_waveform_cfg,
        motion: MotionInfo,
        whitener: Whitener | None = None,
        tsvd=None,
        computation_cfg: ComputationConfig | None = None,
    ) -> "TemplateData":
        raise NotImplementedError


def _try_reload_svd(
    sorting: DARTsortSorting, npz_name="template_data.npz", rank: int = 5
) -> PCA | None:
    from ..util.data_util import try_get_model_dir

    mdir = try_get_model_dir(sorting)
    if not mdir or not mdir.exists():
        return None
    tnpz = mdir / npz_name
    if not tnpz.exists():
        return None
    td = TemplateData.from_npz(tnpz)
    tsvd = td.tsvd
    if tsvd is not None:
        rank_ = tsvd.components_.shape[0]
        if rank_ != rank:
            logger.dartsortdebug(
                f"TSVD from {tnpz} had wrong rank {rank_}, wanted {rank}."
            )
            return None
        logger.dartsortdebug(f"Reloading TSVD from {tnpz}")
    else:
        logger.dartsortdebug(f"No TSVD to reload in {tnpz}")
    return tsvd
