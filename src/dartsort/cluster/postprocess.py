from dataclasses import replace

import numpy as np
import torch.nn.functional as F

from ..util.internal_config import default_waveform_config, coarse_template_config
from ..templates import TemplateData


def realign_and_chuck_noisy_template_units(
    recording,
    sorting,
    template_data=None,
    motion_est=None,
    min_n_spikes=5,
    min_template_snr=15,
    waveform_config=default_waveform_config,
    template_config=coarse_template_config,
    tsvd=None,
    computation_config=None,
):
    """Get rid of noise units.

    This will reindex the sorting and template data -- unit labels will
    change, and the number of templates will change.
    """
    if template_data is None:
        template_data, sorting = TemplateData.from_config_with_realigned_sorting(
            recording,
            sorting,
            template_config,
            motion_est=motion_est,
            tsvd=tsvd,
            waveform_config=waveform_config,
            computation_config=computation_config,
        )
        assert sorting is not None

    template_ptps = np.ptp(template_data.templates, 1).max(1)
    template_snrs = template_ptps * np.sqrt(template_data.spike_counts)
    good_templates = np.logical_and(
        template_data.spike_counts >= min_n_spikes,
        template_snrs > min_template_snr,
    )

    good_unit_ids = template_data.unit_ids[good_templates]
    assert np.all(np.diff(good_unit_ids) >= 0)
    unique_good_unit_ids, new_template_unit_ids = np.unique(
        good_unit_ids, return_inverse=True
    )

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
        spike_counts_by_channel=template_data.spike_counts_by_channel[good_templates],
        registered_geom=template_data.registered_geom,
        registered_template_depths_um=rtdum,
    )

    return new_sorting, new_template_data
