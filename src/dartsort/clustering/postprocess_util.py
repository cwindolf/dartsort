from dataclasses import replace
from pathlib import Path

import numpy as np

from ..templates import TemplateData
from ..templates.get_templates import fit_tsvd
from ..util.internal_config import (
    coarse_template_cfg,
    default_matching_cfg,
    default_waveform_cfg,
)
from ..util.logging_util import get_logger
from ..util.py_util import resolve_path
from ..util.spiketorch import ptp

logger = get_logger(__name__)


def realign_and_chuck_noisy_template_units(
    recording,
    sorting,
    template_data=None,
    motion_est=None,
    min_n_spikes=50,
    min_template_snr=15.0,
    waveform_cfg=default_waveform_cfg,
    template_cfg=coarse_template_cfg,
    tsvd=None,
    computation_cfg=None,
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

    if template_data is None:
        template_data, sorting = TemplateData.from_config_with_realigned_sorting(
            recording,
            sorting,
            template_cfg=template_cfg,
            motion_est=motion_est,
            tsvd=tsvd,
            waveform_cfg=waveform_cfg,
            computation_cfg=computation_cfg,
            save_folder=None,
            save_npz_name=None,
        )
        assert sorting is not None
    assert template_data.spike_counts_by_channel is not None

    good_templates = snr_mask(
        template_data, min_n_spikes=min_n_spikes, min_template_snr=min_template_snr
    )
    logger.dartsortdebug(
        f"Discard {np.logical_not(good_templates).sum()} low-signal templates."
    )
    good_unit_ids = template_data.unit_ids[good_templates]
    assert np.all(np.diff(good_unit_ids) >= 0)
    unique_good_unit_ids, new_template_unit_ids = np.unique(
        good_unit_ids, return_inverse=True
    )

    if template_data.properties:
        properties = {k: v[good_templates] for k, v in template_data.properties.items()}
    else:
        properties = None

    assert sorting.labels is not None
    new_labels = sorting.labels.copy()
    valid = np.isin(new_labels, unique_good_unit_ids)
    new_labels[~valid] = -1
    _, new_labels[valid] = np.unique(new_labels[valid], return_inverse=True)

    new_sorting = sorting.ephemeral_replace(labels=new_labels)
    new_template_data = TemplateData(
        templates=template_data.templates[good_templates],
        unit_ids=new_template_unit_ids,
        spike_counts=template_data.spike_counts[good_templates],
        spike_counts_by_channel=template_data.spike_counts_by_channel[good_templates],
        registered_geom=template_data.registered_geom,
        trough_offset_samples=template_data.trough_offset_samples,
        properties=properties,
    )
    if template_save_folder is not None:
        if template_npz_filename is not None:
            npz = Path(template_save_folder) / template_npz_filename
            new_template_data.to_npz(npz)

    return new_sorting, new_template_data


def snr_mask(template_data, min_n_spikes=50, min_template_snr=15.0):
    template_ptps = np.ptp(template_data.templates, 1).max(1)
    template_snrs = template_ptps * np.sqrt(template_data.spike_counts)
    good_templates = np.logical_and(
        template_data.spike_counts >= min_n_spikes,
        template_snrs > min_template_snr,
    )
    return good_templates


def reorder_by_depth(sorting, template_data):
    assert template_data.registered_geom is not None
    w = ptp(template_data.templates, dim=1)
    if template_data.spike_counts_by_channel is not None:
        w *= np.sqrt(template_data.spike_counts_by_channel)
    w /= w.sum(axis=1, keepdims=True)
    meanz = np.sum(template_data.registered_geom[:, 1] * w, axis=1)

    # new_to_old[i] = old id for new id i
    new_to_old = np.argsort(meanz, stable=True)
    # old_to_new[i] = new id for old id i
    old_to_new = np.argsort(new_to_old, stable=True)

    if template_data.properties:
        properties = {k: v[new_to_old] for k, v in template_data.properties.items()}
    else:
        properties = {}

    valid = np.flatnonzero(sorting.labels >= 0)
    labels = np.full_like(sorting.labels, -1)
    labels[valid] = old_to_new[sorting.labels[valid]]
    sorting = sorting.ephemeral_replace(labels=labels)

    uids = np.arange(len(new_to_old))
    scbc = template_data.raw_std_dev
    if scbc is not None:
        scbc = scbc[new_to_old]
    rsd = template_data.raw_std_dev
    if rsd is not None:
        rsd = rsd[new_to_old]
    template_data = TemplateData(
        templates=template_data.templates[new_to_old],
        unit_ids=uids,
        spike_counts=template_data.spike_counts[new_to_old],
        spike_counts_by_channel=scbc,
        raw_std_dev=rsd,
        registered_geom=template_data.registered_geom,
        trough_offset_samples=template_data.trough_offset_samples,
        properties=properties,
    )
    return sorting, template_data


def postprocess(
    recording,
    sorting,
    motion_est=None,
    matching_cfg=default_matching_cfg,
    waveform_cfg=default_waveform_cfg,
    template_cfg=coarse_template_cfg,
    tsvd=None,
    computation_cfg=None,
    depth_order=True,
    template_npz_path=None,
):
    from .merge import merge_templates

    if template_npz_path is not None:
        template_npz_path = resolve_path(template_npz_path)
        if template_npz_path.exists():
            return sorting, TemplateData.from_npz(template_npz_path)

    assert sorting.labels is not None
    if (sorting.labels < 0).all():
        raise ValueError("No labels in sorting input to template postprocessing.")

    # get tsvd to share across steps
    if tsvd is None and template_cfg.denoising_method not in (None, "none"):
        trough = waveform_cfg.trough_offset_samples(recording.sampling_frequency)
        full = waveform_cfg.spike_length_samples(recording.sampling_frequency)
        tsvd = fit_tsvd(
            recording,
            sorting,
            denoising_rank=template_cfg.denoising_rank,
            denoising_fit_radius=template_cfg.denoising_fit_radius,
            trough_offset_samples=trough,
            spike_length_samples=full,
            recompute_tsvd=template_cfg.recompute_tsvd,
        )
    h5_path = sorting.parent_h5_path

    sorting, template_data = realign_and_chuck_noisy_template_units(
        recording,
        sorting,
        motion_est=motion_est,
        min_n_spikes=matching_cfg.min_template_count,
        min_template_snr=matching_cfg.min_template_snr,
        waveform_cfg=waveform_cfg,
        template_cfg=template_cfg,
        tsvd=tsvd,
        computation_cfg=computation_cfg,
    )
    assert sorting.parent_h5_path == h5_path
    fs_ms = recording.sampling_frequency / 1000
    max_shift_samples = int(template_cfg.realign_shift_ms * fs_ms)
    assert sorting.labels is not None
    if (sorting.labels < 0).all():
        raise ValueError("All units were thrown away during template postprocessing.")

    # merge
    merge_cfg = matching_cfg.template_merge_cfg
    if merge_cfg is None or not merge_cfg.merge_distance_threshold:
        return sorting, ensure_save(template_data, template_npz_path)
    merge_res = merge_templates(
        sorting=sorting,
        recording=recording,
        template_data=template_data,
        motion_est=motion_est,
        max_shift_samples=max_shift_samples,
        linkage=merge_cfg.linkage,
        merge_distance_threshold=merge_cfg.merge_distance_threshold,
        temporal_upsampling_factor=merge_cfg.temporal_upsampling_factor,
        amplitude_scaling_variance=merge_cfg.amplitude_scaling_variance,
        amplitude_scaling_boundary=merge_cfg.amplitude_scaling_boundary,
        svd_compression_rank=merge_cfg.svd_compression_rank,
        min_spatial_cosine=merge_cfg.min_spatial_cosine,
        denoising_tsvd=tsvd,
        computation_cfg=computation_cfg,
        show_progress=True,
    )
    sorting = merge_res["sorting"]
    new_unit_ids = merge_res["new_unit_ids"]
    del merge_res
    assert sorting.parent_h5_path == h5_path

    # determine which units were merged and recompute only those templates
    ul, ui, uc = np.unique(new_unit_ids, return_index=True, return_counts=True)
    n_merged_units = ul.shape[0]
    assert np.array_equal(ul, np.arange(len(ul)))
    needs_recompute = ul[uc > 1]
    if needs_recompute.size:
        recompute_labels = np.where(
            np.isin(sorting.labels, needs_recompute), sorting.labels, -1
        )
        recompute_sorting = sorting.ephemeral_replace(labels=recompute_labels)
        recompute_sorting, recompute_template_data = realign_and_chuck_noisy_template_units(
            recording,
            recompute_sorting,
            motion_est=motion_est,
            min_n_spikes=matching_cfg.min_template_count,
            min_template_snr=matching_cfg.min_template_snr,
            waveform_cfg=waveform_cfg,
            template_cfg=template_cfg,
            tsvd=tsvd,
            computation_cfg=computation_cfg,
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
    else:
        recompute_template_data = None

    # new indices corresponding to kept units
    new_kept_ixs = np.flatnonzero(uc <= 1)
    # original indices corresponding to kept units
    old_kept_ixs = ui[new_kept_ixs]
    # new indices for recomputed units
    new_recompute_ix = np.flatnonzero(uc > 1)
    # original indices corresponding to recomputed units
    old_recompute_ix = ui[new_recompute_ix]

    # pack up the the templates
    templates = np.empty(
        (n_merged_units, *template_data.templates.shape[1:]),
        dtype=template_data.templates.dtype,
    )
    templates[new_kept_ixs] = template_data.templates[old_kept_ixs]
    if recompute_template_data is not None:
        templates[new_recompute_ix] = recompute_template_data.templates

    spike_counts = np.empty(
        (n_merged_units, *template_data.spike_counts.shape[1:]),
        dtype=template_data.spike_counts.dtype,
    )
    spike_counts[new_kept_ixs] = template_data.spike_counts[old_kept_ixs]
    if recompute_template_data is not None:
        spike_counts[new_recompute_ix] = recompute_template_data.spike_counts

    if template_data.spike_counts_by_channel is not None:
        spike_counts_by_channel = np.empty(
            (n_merged_units, *template_data.spike_counts_by_channel.shape[1:]),
            dtype=template_data.spike_counts_by_channel.dtype,
        )
        spike_counts_by_channel[new_kept_ixs] = template_data.spike_counts_by_channel[old_kept_ixs]
        if recompute_template_data is not None:
            spike_counts_by_channel[new_recompute_ix] = recompute_template_data.spike_counts_by_channel
    else:
        spike_counts_by_channel = None

    if template_data.raw_std_dev is not None:
        raw_std_dev = np.empty(
            (n_merged_units, *template_data.raw_std_dev.shape[1:]),
            dtype=template_data.raw_std_dev.dtype,
        )
        raw_std_dev[new_kept_ixs] = template_data.raw_std_dev[old_kept_ixs]
        if recompute_template_data is not None:
            raw_std_dev[new_recompute_ix] = recompute_template_data.raw_std_dev
    else:
        raw_std_dev = None

    template_data = TemplateData(
        templates,
        unit_ids=np.arange(len(templates)),
        spike_counts=spike_counts,
        spike_counts_by_channel=spike_counts_by_channel,
        raw_std_dev=raw_std_dev,
        registered_geom=template_data.registered_geom,
        trough_offset_samples=template_data.trough_offset_samples,
    )
    if depth_order:
        sorting, template_data = reorder_by_depth(sorting, template_data)

    return sorting, ensure_save(template_data, template_npz_path)


def ensure_save(template_data, template_npz_path):
    if template_npz_path is not None:
        template_npz_path.parent.parent.mkdir(exist_ok=True)
        template_npz_path.parent.mkdir(exist_ok=True)
        template_data.to_npz(template_npz_path)
    return template_data
