from dataclasses import dataclass
from typing import Optional

import numpy as np

from .get_templates import get_templates
from .superres_util import superres_sorting
from .template_util import get_registered_templates

_motion_error_prefix = (
    "If template_config has registered_templates==True "
    "or superres_templates=True, then "
)


@dataclass
class TemplateData:
    templates: np.ndarray
    unit_ids: np.ndarray
    registered_geom: Optional[np.ndarray] = None
    registered_template_depths_um: Optional[np.ndarray] = None

    @classmethod
    def from_config(
        cls,
        recording,
        sorting,
        template_config,
        motion_est=None,
        localizations_dataset_name="point_source_localizations",
        n_jobs=0,
    ):
        motion_aware = (
            template_config.registered_templates or template_config.superres_templates
        )
        if motion_aware and motion_est is None:
            raise ValueError(
                f"{_motion_error_prefix}"
                "motion_est must be passed to TemplateData.from_config()"
            )
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
            locs = sorting.extra_features[localizations_dataset_name]
            # TODO: relying on this index feels wrong
            spike_depths_um = locs[:, 2]
            geom = recording.get_channel_locations()

        # handle superresolved templates
        # TODO: should we re-align the original spike train or the superres?
        if template_config.superres_templates:
            unit_ids, sorting = superres_sorting(
                sorting,
                sorting.times_seconds,
                spike_depths_um,
                geom,
                motion_est=motion_est,
                strategy=template_config.superres_strategy,
                superres_bin_size_um=template_config.superres_bin_size_um,
            )
        else:
            unit_ids = np.arange(sorting.labels.max() + 1)

        common_kwargs = dict(
            trough_offset_samples=template_config.trough_offset_samples,
            spike_length_samples=template_config.spike_length_samples,
            spikes_per_unit=template_config.spikes_per_unit,
            realign_peaks=template_config.realign_peaks,
            realign_max_sample_shift=template_config.realign_max_sample_shift,
            low_rank_denoising=template_config.low_rank_denoising,
            denoising_rank=template_config.denoising_rank,
            denoising_fit_radius=template_config.denoising_fit_radius,
            denoising_snr_threshold=template_config.denoising_snr_threshold,
        )

        # handle registered templates
        if template_config.registered_templates:
            results = get_registered_templates(
                recording,
                sorting,
                sorting.times_seconds,
                spike_depths_um,
                geom,
                motion_est,
                localization_radius_um=template_config.registered_template_localization_radius_um,
                **common_kwargs,
                random_seed=0,
                n_jobs=n_jobs,
                show_progress=True,
            )
            return cls(
                results["templates"],
                unit_ids,
                results["registered_geom"],
                results["registered_template_depths_um"],
            )

        # rest of cases handled by get_templates
        results = get_templates(recording, sorting, **common_kwargs)
        return cls(
            results["templates"],
            unit_ids,
        )
