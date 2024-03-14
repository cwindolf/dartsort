from dataclasses import replace

import numpy as np

from .. import config
from ..templates import TemplateData


def realign_and_chuck_noisy_template_units(
    recording,
    sorting,
    template_data=None,
    motion_est=None,
    min_n_spikes=5,
    min_template_snr=15,
    template_config=config.coarse_template_config,
    trough_offset_samples=42,
    spike_length_samples=121,
    tsvd=None,
    device=None,
    n_jobs=0,
):
    """Get rid of noise units.

    This will reindex the sorting and template data -- unit labels will
    change, and the number of templates will change.
    """
    if template_data is None:
        template_data, sorting = TemplateData.from_config(
            recording,
            sorting,
            template_config,
            motion_est=motion_est,
            n_jobs=n_jobs,
            tsvd=tsvd,
            device=device,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            return_realigned_sorting=True,
        )

    template_ptps = template_data.templates.ptp(1).max(1)
    template_snrs = template_ptps * np.sqrt(template_data.spike_counts)
    good_templates = np.logical_and(
        template_data.spike_counts >= min_n_spikes,
        template_snrs > min_template_snr,
    )

    good_unit_ids = template_data.unit_ids[good_templates]
    assert np.all(np.diff(good_unit_ids) >= 0)
    unique_good_unit_ids, new_template_unit_ids = np.unique(good_unit_ids, return_inverse=True)

    new_labels = sorting.labels.copy()
    valid = np.isin(new_labels, unique_good_unit_ids)
    new_labels[~valid] = -1
    _, new_labels[valid] = np.unique(new_labels[valid], return_inverse=True)

    new_sorting = replace(sorting, labels=new_labels)
    rtdum = None
    if template_data.registered_template_depths_um is not None:
        rtdum = template_data.registered_template_depths_um[good_templates]
    new_template_data = TemplateData(
        templates=template_data.templates[good_templates],
        unit_ids=new_template_unit_ids,
        spike_counts=template_data.spike_counts[good_templates],
        registered_geom=template_data.registered_geom,
        registered_template_depths_um=rtdum,
    )

    return new_sorting, new_template_data
