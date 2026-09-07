import gc
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from spikeinterface.core import BaseRecording

from ..util.data_util import DARTsortSorting, apply_label_remapping_in_place
from ..util.internal_config import (
    ComputationConfig,
    FeaturizationConfig,
    TemplateConfig,
    TemplateMergeConfig,
    TemplateRealignmentConfig,
    WaveformConfig,
    default_template_cfg,
    default_waveform_cfg,
)
from ..util.job_util import ensure_computation_config
from ..util.logging_util import get_logger
from ..util.motion import MotionInfo
from ..util.noise_util import Whitener
from ..util.py_util import ensure_path
from ..util.spiketorch import ptp
from . import TemplateData, realign
from .templib import fit_tsvd, pca_from_templates, quick_mean_templates

logger = get_logger(__name__)


def estimate_template_library(
    recording: BaseRecording,
    sorting: DARTsortSorting,
    motion: MotionInfo | None = None,
    min_template_snr: float = 0.0,
    min_template_ptp: float = 0.0,
    always_keep_ptp: float = 0.0,
    min_template_count: int = 0,
    waveform_cfg: WaveformConfig = default_waveform_cfg,
    template_cfg: TemplateConfig = default_template_cfg,
    realign_cfg: TemplateRealignmentConfig | None = None,
    template_merge_cfg: TemplateMergeConfig | None = None,
    tsvd: PCA | TruncatedSVD | None = None,
    whitener: Whitener | None = None,
    computation_cfg: ComputationConfig | None = None,
    fit_featurization_tsvd: bool = False,
    featurization_cfg: FeaturizationConfig | None = None,
    depth_order: bool = False,
    template_npz_path=None,
) -> tuple[DARTsortSorting, TemplateData]:
    """Postprocess spike train and estimate a TemplateData."""
    if template_npz_path is not None:
        template_npz_path = ensure_path(template_npz_path)
        if template_npz_path.exists():
            return sorting, TemplateData.from_npz(template_npz_path)

    if (sorting.labels is None) or (sorting.labels < 0).all():
        raise ValueError("No labels in sorting input to template postprocessing.")
    computation_cfg = ensure_computation_config(computation_cfg)

    if motion is None:
        motion = MotionInfo.from_motion_est(geom=recording.get_channel_locations())

    # avoid blanks down the line
    if min_template_count:
        from ..clustering.cluster_util import decrumb_labels

        sorting = sorting.ephemeral_replace(
            labels=decrumb_labels(sorting.labels, min_size=min_template_count)
        )

    # realign sorting and estimate template snr
    sorting, templates0 = realign(
        recording=recording,
        sorting=sorting,
        realign_cfg=realign_cfg,
        waveform_cfg=waveform_cfg,
        computation_cfg=computation_cfg,
        motion=motion,
    )

    # filter out low-count/snr units
    need_templates = (
        min_template_count or min_template_snr or min_template_ptp or always_keep_ptp
    )
    if templates0 is None and need_templates:
        templates0 = quick_mean_templates(
            recording=recording,
            sorting=sorting,
            waveform_cfg=waveform_cfg,
            computation_cfg=computation_cfg,
            motion=motion,
        )

    sorting, templates0 = mask_out_units(
        sorting,
        templates0,
        min_template_count=min_template_count,
        min_template_snr=min_template_snr,
        min_template_ptp=min_template_ptp,
        always_keep_ptp=always_keep_ptp,
        template_cfg=template_cfg,
    )

    # use templates0 to fit tsvd if relevant
    need_tsvd = template_cfg.use_svd and tsvd is None
    if need_tsvd and template_cfg.svd_method == "raw_template":
        tsvd = fit_tsvd(
            recording=recording,
            sorting=sorting,
            motion=motion,
            template_cfg=template_cfg,
            waveform_cfg=waveform_cfg,
            computation_cfg=computation_cfg,
            svd_input_templates=templates0,
        )

    if fit_featurization_tsvd:
        assert featurization_cfg is not None
        assert templates0 is not None
        featurization_basis = featurization_basis_from_templates(
            templates0,
            featurization_cfg=featurization_cfg,
            waveform_cfg=waveform_cfg,
            rank=featurization_cfg.tpca_rank,
            min_channel_amplitude=template_cfg.template_min_channel_amplitude,
            sampling_frequency=recording.sampling_frequency,
            computation_cfg=computation_cfg,
        )
    else:
        featurization_basis = None
    del templates0

    _check_still_valid(sorting)
    gc.collect()
    torch.cuda.empty_cache()

    # main task: get denoised templates from aligned spike train
    templates = TemplateData.from_config(
        recording=recording,
        sorting=sorting,
        motion=motion,
        waveform_cfg=waveform_cfg,
        template_cfg=template_cfg,
        computation_cfg=computation_cfg,
        tsvd=tsvd,
        whitener=whitener,
        featurization_basis=featurization_basis,
    )
    gc.collect()
    torch.cuda.empty_cache()

    # merge units by template distance
    sorting, templates = _handle_merge(
        recording=recording,
        sorting=sorting,
        template_data=templates,
        motion=motion,
        merge_cfg=template_merge_cfg,
        computation_cfg=computation_cfg,
        waveform_cfg=waveform_cfg,
        template_cfg=template_cfg,
    )
    gc.collect()
    torch.cuda.empty_cache()

    # re-order along probe length
    if depth_order:
        sorting, templates = reorder_templates_by_depth(sorting, templates)

    return sorting, ensure_save(templates, template_npz_path)


def realign_and_chuck_noisy_template_units(
    recording,
    sorting,
    template_data=None,
    motion=None,
    min_n_spikes=50,
    min_template_snr=15.0,
    waveform_cfg=default_waveform_cfg,
    template_cfg=default_template_cfg,
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
        template_data = TemplateData.from_config(
            recording=recording,
            sorting=sorting,
            template_cfg=template_cfg,
            motion=motion,
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
    label_remapping = np.full((unique_good_unit_ids.max() + 1,), -1)
    label_remapping[unique_good_unit_ids] = np.arange(len(unique_good_unit_ids))
    apply_label_remapping_in_place(new_labels, label_remapping, allow_over=True)

    new_sorting = sorting.ephemeral_replace(labels=new_labels)
    new_template_data = TemplateData(
        templates=template_data.templates[good_templates],
        unit_ids=new_template_unit_ids,
        spike_counts=template_data.spike_counts[good_templates],
        spike_counts_by_channel=template_data.spike_counts_by_channel[good_templates],
        registered_geom=template_data.registered_geom,
        trough_offset_samples=template_data.trough_offset_samples,
        whitener=template_data.whitener,
        tsvd=template_data.tsvd,
        properties=properties,
        sampling_frequency=template_data.sampling_frequency,
        whiten_strategy=template_data.whiten_strategy,
    )
    if template_save_folder is not None:
        if template_npz_filename is not None:
            npz = Path(template_save_folder) / template_npz_filename
            new_template_data.to_npz(npz)

    return new_sorting, new_template_data


def mask_out_units(
    sorting: DARTsortSorting,
    templates0: TemplateData | None,
    min_template_count: int,
    min_template_snr: float,
    min_template_ptp: float,
    always_keep_ptp: float | None,
    template_cfg,
):
    mask = None

    if min_template_count:
        assert templates0 is not None
        m = templates0.spike_counts >= min_template_count
        mask = np.logical_and(mask, m) if mask is not None else m.copy()

    if min_template_snr:
        assert templates0 is not None
        m = templates0.snrs_by_channel().max(1) >= min_template_snr
        mask = np.logical_and(mask, m) if mask is not None else m.copy()

    if min_template_ptp:
        assert templates0 is not None
        amp = ptp(templates0.templates).max(1)
        m = amp >= min_template_ptp
        mask = np.logical_and(mask, m) if mask is not None else m.copy()

    if mask is not None and always_keep_ptp is not None:
        assert templates0 is not None
        amp = ptp(templates0.templates).max(1)
        mask |= amp >= always_keep_ptp

    if mask is None:
        return sorting, templates0

    if templates0 is not None:
        sorting = filter_by_unit_mask(sorting, mask, mask_ids=templates0.unit_ids)
        templates0 = templates0[mask]
    else:
        sorting = filter_by_unit_mask(sorting, mask)

    return sorting, templates0


def snr_mask(template_data, min_n_spikes=50, min_template_snr=15.0):
    template_ptps = np.ptp(template_data.templates, 1).max(1)
    template_snrs = template_ptps * np.sqrt(template_data.spike_counts)
    good_templates = np.logical_and(
        template_data.spike_counts >= min_n_spikes,
        template_snrs > min_template_snr,
    )
    return good_templates


def reorder_templates_by_depth(sorting, template_data):
    assert template_data.registered_geom is not None
    w = template_data.snrs_by_channel()
    w /= w.sum(axis=1, keepdims=True)
    meanz = np.sum(template_data.registered_geom[:, 1] * w, axis=1)

    # new_to_old[i] = old id for new id i
    new_to_old = np.argsort(meanz, kind="stable")
    # old_to_new[i] = new id for old id i
    old_to_new = np.argsort(new_to_old, kind="stable")

    if template_data.properties:
        properties = {k: v[new_to_old] for k, v in template_data.properties.items()}
    else:
        properties = {}

    valid = np.flatnonzero(sorting.labels >= 0)
    labels = np.full_like(sorting.labels, -1)
    labels[valid] = old_to_new[sorting.labels[valid]]
    sorting = sorting.ephemeral_replace(labels=labels)

    uids = np.arange(len(new_to_old))
    scbc = template_data.spike_counts_by_channel
    if scbc is not None:
        scbc = scbc[new_to_old]
    rsd = template_data.raw_std_dev
    if rsd is not None:
        rsd = rsd[new_to_old]
    template_data = replace(
        template_data,
        templates=template_data.templates[new_to_old],
        unit_ids=uids,
        spike_counts=template_data.spike_counts[new_to_old],
        spike_counts_by_channel=scbc,
        raw_std_dev=rsd,
        properties=properties,
    )
    return sorting, template_data


def ensure_save(template_data, template_npz_path):
    if template_npz_path is not None:
        template_npz_path.parent.parent.mkdir(exist_ok=True)
        template_npz_path.parent.mkdir(exist_ok=True)
        template_data.to_npz(template_npz_path)
    return template_data


def featurization_basis_from_templates(
    templates0: TemplateData,
    featurization_cfg: FeaturizationConfig,
    waveform_cfg: WaveformConfig,
    sampling_frequency: float,
    rank: int,
    min_channel_amplitude: float,
    computation_cfg: ComputationConfig,
    random_seed: int = 0,
):
    if featurization_cfg.input_tpca_waveform_cfg is None:
        templates = templates0
    else:
        tslice = featurization_cfg.input_tpca_waveform_cfg.relative_slice(
            waveform_cfg, sampling_frequency
        )
        trough = templates0.trough_offset_samples - tslice.start
        templates = TemplateData(
            templates=templates0.templates[:, tslice],
            trough_offset_samples=trough,
            unit_ids=templates0.unit_ids,
            spike_counts=templates0.spike_counts,
            spike_counts_by_channel=templates0.spike_counts_by_channel,
            sampling_frequency=templates0.sampling_frequency,
            whiten_strategy=templates0.whiten_strategy,
        )
    pca = pca_from_templates(
        templates,
        rank=rank,
        min_channel_amplitude=min_channel_amplitude,
        random_seed=random_seed,
        computation_cfg=computation_cfg,
    )
    return pca.components_


def _check_still_valid(sorting: DARTsortSorting):
    assert sorting.labels is not None
    if (sorting.labels < 0).all():
        raise ValueError("All units were thrown away during template postprocessing.")


def _handle_merge(
    *,
    recording: BaseRecording,
    sorting: DARTsortSorting,
    motion,
    template_data: TemplateData,
    merge_cfg: TemplateMergeConfig | None,
    computation_cfg: ComputationConfig,
    waveform_cfg: WaveformConfig,
    template_cfg: TemplateConfig,
) -> tuple[DARTsortSorting, TemplateData]:
    if merge_cfg is None or not merge_cfg.merge_distance_threshold:
        return sorting, template_data

    if template_cfg.denoising_method == "svd":
        # use new shared basis stuff
        from ..clustering.agglomerate import agglomerate

        agg = agglomerate(
            sorting=sorting,
            recording=recording,
            motion=motion,
            template_data=template_data,
            template_merge_cfg=merge_cfg,
            computation_cfg=computation_cfg,
            waveform_cfg=waveform_cfg,
            refinement_cfg=None,
        )
        new_unit_ids = agg.merge_mapping
        sorting = agg.agglomerated_sorting
        assert sorting.labels is not None
        del agg
    else:
        # TODO: remove old impl?
        from ..clustering.merge import merge_templates

        merge_shift_samples = waveform_cfg.ms_to_samples(merge_cfg.max_shift_ms)
        merge_res = merge_templates(
            sorting=sorting,
            template_data=template_data,
            max_shift_samples=merge_shift_samples,
            linkage=merge_cfg.linkage,
            merge_distance_threshold=merge_cfg.merge_distance_threshold,
            temporal_upsampling_factor=merge_cfg.temporal_upsampling_factor,
            amplitude_scaling_variance=merge_cfg.amplitude_scaling_variance,
            amplitude_scaling_boundary=merge_cfg.amplitude_scaling_boundary,
            svd_compression_rank=merge_cfg.svd_compression_rank,
            min_spatial_cosine=merge_cfg.min_spatial_cosine,
            computation_cfg=computation_cfg,
            show_progress=True,
        )
        sorting = cast(DARTsortSorting, merge_res["sorting"])
        new_unit_ids = merge_res["new_unit_ids"]
        del merge_res
        assert sorting.labels is not None

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
        # turn off merge here
        recompute_sorting, recompute_template_data = estimate_template_library(
            recording=recording,
            sorting=recompute_sorting,
            motion=motion,
            min_template_snr=0.0,
            min_template_count=0,
            waveform_cfg=waveform_cfg,
            template_cfg=template_cfg,
            realign_cfg=None,
            template_merge_cfg=None,
            tsvd=template_data.tsvd,
            computation_cfg=computation_cfg,
            depth_order=False,
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
    # old_recompute_ix = ui[new_recompute_ix]

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
        spike_counts_by_channel[new_kept_ixs] = template_data.spike_counts_by_channel[
            old_kept_ixs
        ]
        if recompute_template_data is not None:
            spike_counts_by_channel[new_recompute_ix] = (
                recompute_template_data.spike_counts_by_channel
            )
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

    template_data = replace(
        template_data,
        templates=templates,
        unit_ids=np.arange(len(templates)),
        spike_counts=spike_counts,
        spike_counts_by_channel=spike_counts_by_channel,
        raw_std_dev=raw_std_dev,
        registered_geom=template_data.registered_geom,
        trough_offset_samples=template_data.trough_offset_samples,
    )
    return sorting, template_data


def filter_by_unit_mask(
    sorting: DARTsortSorting, keep_mask: np.ndarray, mask_ids: np.ndarray | None = None
) -> DARTsortSorting:
    assert sorting.labels is not None

    if mask_ids is not None:
        assert mask_ids.shape == keep_mask.shape
        k_full = mask_ids.max() + 1
        assert k_full >= keep_mask.shape[0]
        mask = np.zeros(k_full, dtype=bool)
        mask[mask_ids[keep_mask]] = True
        keep_mask = mask

    discard_mask = np.logical_not(keep_mask)
    if not discard_mask.any():
        return sorting

    valid = np.flatnonzero(sorting.labels >= 0)
    chuck = valid[discard_mask[sorting.labels[valid]]]
    sorting.labels[chuck] = -1

    return sorting.flatten()
