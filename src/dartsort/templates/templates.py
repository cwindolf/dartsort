from dataclasses import dataclass
from typing import Optional

import numpy as np
from pathlib import Path

from .get_templates import get_templates
from .superres_util import superres_sorting
from .template_util import (
    get_registered_templates,
    get_realigned_sorting,
    get_template_depths,
)
from dartsort.util import drift_util

_motion_error_prefix = (
    "If template_config has registered_templates==True "
    "or superres_templates=True, then "
)
_aware_error = "motion_est must be passed to TemplateData.from_config()"


@dataclass
class TemplateData:
    # (n_templates, spike_length_samples, n_registered_channels or n_channels)
    templates: np.ndarray
    # (n_templates,) maps template index to unit index (multiple templates can share a unit index)
    unit_ids: np.ndarray
    # (n_templates,) spike count for each template
    spike_counts: np.ndarray

    registered_geom: Optional[np.ndarray] = None
    registered_template_depths_um: Optional[np.ndarray] = None

    @classmethod
    def from_npz(cls, npz_path):
        with np.load(npz_path) as npz:
            templates = npz["templates"]
            unit_ids = npz["unit_ids"]
            spike_counts = npz["spike_counts"]
            registered_geom = registered_template_depths_um = None
            if "registered_geom" in npz:
                registered_geom = npz["registered_geom"]
            if "registered_template_depths_um" in npz:
                registered_template_depths_um = npz["registered_template_depths_um"]
        return cls(
            templates,
            unit_ids,
            spike_counts,
            registered_geom,
            registered_template_depths_um,
        )

    def to_npz(self, npz_path):
        to_save = dict(
            templates=self.templates,
            unit_ids=self.unit_ids,
            spike_counts=self.spike_counts,
        )
        if self.registered_geom is not None:
            to_save["registered_geom"] = self.registered_geom
        if self.registered_template_depths_um is not None:
            to_save[
                "registered_template_depths_um"
            ] = self.registered_template_depths_um
        np.savez(npz_path, **to_save)

    @classmethod
    def from_config(
        cls,
        recording,
        sorting,
        template_config,
        save_folder=None,
        overwrite=False,
        motion_est=None,
        save_npz_name="template_data.npz",
        localizations_dataset_name="point_source_localizations",
        n_jobs=0,
        device=None,
    ):
        if save_folder is not None:
            save_folder = Path(save_folder)
            if not save_folder.exists():
                save_folder.mkdir()
            npz_path = save_folder / save_npz_name
            if npz_path.exists() and not overwrite:
                return cls.from_npz(npz_path)

        motion_aware = (
            template_config.registered_templates or template_config.superres_templates
        )
        if motion_aware and motion_est is None:
            raise ValueError(_motion_error_prefix + _aware_error)
        has_localizations = hasattr(sorting, localizations_dataset_name)
        if motion_aware and not has_localizations:
            raise ValueError(
                f"{_motion_error_prefix}"
                "sorting must contain localizations in the attribute "
                f"{localizations_dataset_name=}. Using load_simple_features"
                "=True in DARTsortSorting.from_peeling_hdf5() would put them "
                "there."
            )

        # load motion features if necessary
        if motion_aware and has_localizations:
            # load spike depths
            # TODO: relying on this index feels wrong
            spike_depths_um = sorting.extra_features[localizations_dataset_name][:, 2]
            geom = recording.get_channel_locations()

        kwargs = dict(
            trough_offset_samples=template_config.trough_offset_samples,
            spike_length_samples=template_config.spike_length_samples,
            spikes_per_unit=template_config.spikes_per_unit,
            # realign_peaks=template_config.realign_peaks,
            realign_max_sample_shift=template_config.realign_max_sample_shift,
            denoising_rank=template_config.denoising_rank,
            denoising_fit_radius=template_config.denoising_fit_radius,
            denoising_snr_threshold=template_config.denoising_snr_threshold,
            device=device,
        )
        if template_config.registered_templates:
            kwargs["registered_geom"] = drift_util.registered_geometry(
                geom, motion_est=motion_est
            )
            kwargs["pitch_shifts"] = drift_util.get_spike_pitch_shifts(
                spike_depths_um,
                geom,
                times_s=sorting.times_seconds,
                motion_est=motion_est,
            )

        # realign before superres
        if template_config.realign_peaks:
            sorting = get_realigned_sorting(
                recording,
                sorting,
                **kwargs,
                realign_peaks=True,
                low_rank_denoising=False,
                n_jobs=n_jobs,
            )
        kwargs["low_rank_denoising"] = template_config.low_rank_denoising
        kwargs["realign_peaks"] = False

        # handle superresolved templates
        if template_config.superres_templates:
            unit_ids, sorting = superres_sorting(
                sorting,
                sorting.times_seconds,
                spike_depths_um,
                geom,
                motion_est=motion_est,
                strategy=template_config.superres_strategy,
                superres_bin_size_um=template_config.superres_bin_size_um,
                min_spikes_per_bin=template_config.superres_bin_min_spikes,
            )
        else:
            unit_ids = np.arange(sorting.labels.max() + 1)

        # count spikes in each template
        spike_counts = np.zeros_like(unit_ids)
        ix, counts = np.unique(sorting.labels, return_counts=True)
        spike_counts[ix[ix >= 0]] = counts[ix >= 0]

        # main!
        results = get_templates(recording, sorting, **kwargs)

        # handle registered templates
        if template_config.registered_templates:
            registered_template_depths_um = get_template_depths(
                results["templates"],
                kwargs["registered_geom"],
                localization_radius_um=template_config.registered_template_localization_radius_um,
            )
            obj = cls(
                results["templates"],
                unit_ids,
                spike_counts,
                kwargs["registered_geom"],
                registered_template_depths_um,
            )
        else:
            obj = cls(
                results["templates"],
                unit_ids,
                spike_counts,
            )

        if save_folder is not None:
            obj.to_npz(npz_path)

        return obj
