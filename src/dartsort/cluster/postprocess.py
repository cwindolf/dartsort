from dataclasses import replace
from pathlib import Path
from logging import getLogger

import numpy as np

from ..util.internal_config import (
    default_matching_config,
    default_waveform_config,
    coarse_template_config,
)
from ..templates import TemplateData
from ..templates.get_templates import fit_tsvd


logger = getLogger(__name__)


def realign_and_chuck_noisy_template_units(
    recording,
    sorting,
    template_data=None,
    motion_est=None,
    min_n_spikes=50,
    min_template_snr=15.0,
    waveform_config=default_waveform_config,
    template_config=coarse_template_config,
    tsvd=None,
    computation_config=None,
    template_save_folder=None,
    template_npz_filename=None,
):
    """Get rid of noise units.

    This will reindex the sorting and template data -- unit labels will
    change, and the number of templates will change.
    """
    if template_save_folder is not None:
        if template_npz_filename is not None:
            npz = Path(template_save_folder) / template_npz_filename
            if npz.exists():
                return sorting, TemplateData.from_npz(npz)
    h5_path = sorting.parent_h5_path

    if template_data is None:
        template_data, sorting = TemplateData.from_config_with_realigned_sorting(
            recording,
            sorting,
            template_config,
            motion_est=motion_est,
            tsvd=tsvd,
            waveform_config=waveform_config,
            computation_config=computation_config,
            save_folder=None,
            save_npz_name=None,
        )
        assert sorting is not None
    assert template_data.spike_counts_by_channel is not None
    assert template_data.parent_sorting_hdf5_path == h5_path

    template_ptps = np.ptp(template_data.templates, 1).max(1)
    template_snrs = template_ptps * np.sqrt(template_data.spike_counts)
    good_templates = np.logical_and(
        template_data.spike_counts >= min_n_spikes,
        template_snrs > min_template_snr,
    )
    logger.dartsortdebug(f"Discard {np.logical_not(good_templates).sum()} low-signal templates.")

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
    new_template_data = TemplateData(
        templates=template_data.templates[good_templates],
        unit_ids=new_template_unit_ids,
        spike_counts=template_data.spike_counts[good_templates],
        spike_counts_by_channel=template_data.spike_counts_by_channel[good_templates],
        registered_geom=template_data.registered_geom,
        trough_offset_samples=template_data.trough_offset_samples,
        spike_length_samples=template_data.spike_length_samples,
        parent_sorting_hdf5_path=h5_path,
    )
    if template_save_folder is not None:
        if template_npz_filename is not None:
            npz = Path(template_save_folder) / template_npz_filename
            new_template_data.to_npz(npz)

    return new_sorting, new_template_data


def process_templates_for_matching(
    recording,
    sorting,
    motion_est=None,
    matching_config=default_matching_config,
    waveform_config=default_waveform_config,
    template_config=coarse_template_config,
    tsvd=None,
    computation_config=None,
    template_save_folder=None,
    template_npz_filename=None,
):
    from .merge import merge_templates

    # get tsvd to share across steps
    if tsvd is None and template_config.low_rank_denoising:
        tsvd = fit_tsvd(
            recording,
            sorting,
            denoising_rank=template_config.denoising_rank,
            denoising_fit_radius=template_config.denoising_fit_radius,
            trough_offset_samples=waveform_config.trough_offset_samples(recording.sampling_frequency),
            spike_length_samples=waveform_config.spike_length_samples(recording.sampling_frequency),
        )
    h5_path = sorting.parent_h5_path

    sorting, template_data = realign_and_chuck_noisy_template_units(
        recording,
        sorting,
        motion_est=motion_est,
        min_n_spikes=matching_config.min_template_count,
        min_template_snr=matching_config.min_template_snr,
        waveform_config=waveform_config,
        template_config=template_config,
        tsvd=tsvd,
        computation_config=computation_config,
    )
    assert sorting.parent_h5_path == h5_path == template_data.parent_sorting_hdf5_path
    fs_ms = recording.sampling_frequency / 1000
    max_shift_samples = int(template_config.realign_shift_ms * fs_ms)

    # merge
    merge_config = matching_config.template_merge_config
    if merge_config is None or not merge_config.merge_distance_threshold:
        return template_data

    merge_res = merge_templates(
        sorting=sorting,
        recording=recording,
        template_data=template_data,
        motion_est=motion_est,
        max_shift_samples=max_shift_samples,
        linkage=merge_config.linkage,
        merge_distance_threshold=merge_config.merge_distance_threshold,
        temporal_upsampling_factor=merge_config.temporal_upsampling_factor,
        amplitude_scaling_variance=merge_config.amplitude_scaling_variance,
        amplitude_scaling_boundary=merge_config.amplitude_scaling_boundary,
        svd_compression_rank=merge_config.svd_compression_rank,
        min_spatial_cosine=merge_config.min_spatial_cosine,
        denoising_tsvd=tsvd,
        computation_config=computation_config,
        show_progress=True,
    )
    sorting = merge_res["sorting"]
    new_unit_ids = merge_res["new_unit_ids"]
    assert sorting.parent_h5_path == h5_path

    # determine which units were merged and recompute only those templates
    ul, uc = np.unique(new_unit_ids, return_counts=True)
    needs_recompute = ul[uc > 1]
    if not needs_recompute.size:
        return template_data
    recompute_labels = np.where(
        np.isin(sorting.labels, needs_recompute), sorting.labels, -1
    )
    recompute_sorting = replace(sorting, labels=recompute_labels)
    recompute_sorting, recompute_template_data = realign_and_chuck_noisy_template_units(
        recording,
        recompute_sorting,
        motion_est=motion_est,
        min_n_spikes=matching_config.min_template_count,
        min_template_snr=matching_config.min_template_snr,
        waveform_config=waveform_config,
        template_config=template_config,
        tsvd=tsvd,
        computation_config=computation_config,
    )
    assert len(recompute_template_data.templates) == needs_recompute.size
    assert (
        recompute_template_data.trough_offset_samples
        == template_data.trough_offset_samples
    )
    assert (
        recompute_template_data.spike_length_samples
        == template_data.spike_length_samples
    )

    # pack up the the templates
    original_keep = np.isin(new_unit_ids, ul[uc <= 1])
    templates = np.concatenate(
        [template_data.templates[original_keep], recompute_template_data.templates],
        axis=0,
    )
    spike_counts = np.concatenate(
        [template_data.spike_counts[original_keep], recompute_template_data.spike_counts],
        axis=0,
    )
    spike_counts_by_channel = None
    if template_data.spike_counts_by_channel is not None:
        spike_counts_by_channel = np.concatenate(
            [
                template_data.spike_counts_by_channel[original_keep],
                recompute_template_data.spike_counts_by_channel,
            ],
            axis=0,
        )
    raw_std_dev = None
    if template_data.raw_std_dev is not None:
        raw_std_dev = np.concatenate(
            [
                template_data.raw_std_dev[original_keep],
                recompute_template_data.raw_std_dev,
            ],
            axis=0,
        )
    template_data = TemplateData(
        templates,
        unit_ids=np.arange(len(templates)),
        spike_counts=spike_counts,
        spike_counts_by_channel=spike_counts_by_channel,
        raw_std_dev=raw_std_dev,
        registered_geom=template_data.registered_geom,
        trough_offset_samples=template_data.trough_offset_samples,
        spike_length_samples=template_data.spike_length_samples,
        parent_sorting_hdf5_path=sorting.parent_h5_path,
    )
    return template_data
