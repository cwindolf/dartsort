from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

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

    template_ptps = np.ptp(template_data.templates, 1).max(1)
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
        spike_counts_by_channel=template_data.spike_counts_by_channel[good_templates],
        registered_geom=template_data.registered_geom,
        registered_template_depths_um=rtdum,
    )

    return new_sorting, new_template_data


def template_collision_scores(
    recording,
    template_data,
    svd_compression_rank=20,
    temporal_upsampling_factor=8,
    min_channel_amplitude=0.0,
    amplitude_scaling_variance=0.0,
    amplitude_scaling_boundary=0.5,
    trough_offset_samples=42,
    device=None,
    max_n_colliding=5,
    threshold=None,
    save_folder=None,
):
    from ..peel.matching import ObjectiveUpdateTemplateMatchingPeeler

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if threshold is None or threshold < 1:
        factor = 0.9
        if threshold < 1:
            factor = threshold
        threshold = factor * np.square(template_data.templates).sum((1, 2)).min()
        print(f"Using {threshold=:0.3f} for decollision")

    matcher = ObjectiveUpdateTemplateMatchingPeeler(
        recording,
        template_data,
        channel_index=None,
        featurization_pipeline=None,
        motion_est=None,
        svd_compression_rank=svd_compression_rank,
        temporal_upsampling_factor=temporal_upsampling_factor,
        min_channel_amplitude=min_channel_amplitude,
        refractory_radius_frames=10,
        amplitude_scaling_variance=amplitude_scaling_variance,
        amplitude_scaling_boundary=amplitude_scaling_boundary,
        conv_ignore_threshold=0.0,
        coarse_approx_error_threshold=0.0,
        trough_offset_samples=template_data.trough_offset_samples,
        threshold=threshold,
        # chunk_length_samples=30_000,
        # n_chunks_fit=40,
        # max_waveforms_fit=50_000,
        # n_waveforms_fit=20_000,
        # fit_subsampling_random_state=0,
        # fit_sampling="random",
        max_iter=max_n_colliding,
        dtype=torch.float,
    )
    matcher.to(device)
    save_folder.mkdir(exist_ok=True)
    matcher.precompute_peeling_data(save_folder)
    matcher.to(device)

    n = len(template_data.templates)
    scores = np.zeros(n)
    matches = []
    unit_mask = torch.arange(n, device=device)
    for j in range(n):
        mask = template_data.spike_counts_by_channel[j] > 0
        template = template_data.templates[j][:, mask]

        mask = torch.from_numpy(mask)
        compressed_template_data = matcher.templates_at_time(0.0, spatial_mask=mask)
        traces = F.pad(
            torch.from_numpy(template).to(device),
            (0, 0, *(2 * [template_data.spike_length_samples])),
        )
        res = matcher.match_chunk(
            traces,
            compressed_template_data,
            trough_offset_samples=42,
            left_margin=0,
            right_margin=0,
            threshold=30,
            return_collisioncleaned_waveforms=False,
            return_residual=True,
            unit_mask=unit_mask != j,
        )
        resid = res["residual"][template_data.spike_length_samples:2*template_data.spike_length_samples]
        scores[j] = resid.square().sum() / traces.square().sum()
        matches.append(res["labels"].numpy(force=True))
    return scores, matches