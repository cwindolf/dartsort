from pathlib import Path
from typing import Any
import gc

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from spikeinterface.core import BaseRecording
from scipy.spatial import KDTree
from scipy.sparse import coo_array
import torch

from . import TemplateData, realign
from ..util.data_util import DARTsortSorting
from ..util.internal_config import (
    ComputationConfig,
    TemplateConfig,
    TemplateMergeConfig,
    TemplateRealignmentConfig,
    WaveformConfig,
    SubtractionConfig,
    default_template_cfg,
    default_waveform_cfg,
)
from ..util.job_util import ensure_computation_config
from ..util.logging_util import get_logger
from ..util.py_util import resolve_path
from ..util.spiketorch import ptp

logger = get_logger(__name__)


def estimate_template_library(
    recording: BaseRecording,
    sorting: DARTsortSorting,
    motion_est=None,
    min_template_snr: float = 0.0,
    min_template_ptp: float = 0.0,
    min_template_count: int = 0,
    max_cc_flag_rate: float = 1.0,
    cc_flag_entropy_cutoff: float = 0.0,
    waveform_cfg: WaveformConfig = default_waveform_cfg,
    template_cfg: TemplateConfig = default_template_cfg,
    realign_cfg: TemplateRealignmentConfig | None = None,
    template_merge_cfg: TemplateMergeConfig | None = None,
    tsvd: PCA | TruncatedSVD | None = None,
    computation_cfg: ComputationConfig | None = None,
    detection_cfg: Any | None = None,
    depth_order: bool = False,
    template_npz_path=None,
) -> tuple[DARTsortSorting, TemplateData]:
    """Postprocess spike train and estimate a TemplateData."""
    if template_npz_path is not None:
        template_npz_path = resolve_path(template_npz_path)
        if template_npz_path.exists():
            return sorting, TemplateData.from_npz(template_npz_path)

    if (sorting.labels is None) or (sorting.labels < 0).all():
        raise ValueError("No labels in sorting input to template postprocessing.")
    computation_cfg = ensure_computation_config(computation_cfg)

    # realign sorting and estimate template snr
    sorting, templates0 = realign(
        recording=recording,
        sorting=sorting,
        realign_cfg=realign_cfg,
        waveform_cfg=waveform_cfg,
        computation_cfg=computation_cfg,
        motion_est=motion_est,
    )

    # filter out low-count/snr units
    if templates0 is None and (min_template_count or min_template_snr):
        templates0 = _quick_mean_templates(
            recording=recording,
            sorting=sorting,
            waveform_cfg=waveform_cfg,
            computation_cfg=computation_cfg,
            motion_est=motion_est,
        )
    if min_template_count or min_template_snr or (max_cc_flag_rate < 1.0):
        assert templates0 is not None
        count_mask = templates0.spike_counts >= min_template_count
        snr_mask = templates0.snrs_by_channel().max(1) >= min_template_snr
        amp_mask = ptp(templates0.templates).max(1) >= min_template_ptp
        mask = count_mask & snr_mask & amp_mask
        flag_mask = cc_flag_criterion(
            sorting,
            detection_cfg,
            max_cc_flag_rate,
            cc_flag_entropy_cutoff,
            amplitudes_dataset_name=template_cfg.amplitudes_dataset_name,
        )
        if flag_mask is not None:
            mask = mask & flag_mask
        sorting = filter_by_unit_mask(sorting, mask, mask_ids=templates0.unit_ids)
    del templates0

    _check_still_valid(sorting)
    gc.collect()
    torch.cuda.empty_cache()

    # main task: get denoised templates from aligned spike train
    templates = TemplateData.from_config(
        recording=recording,
        sorting=sorting,
        motion_est=motion_est,
        waveform_cfg=waveform_cfg,
        template_cfg=template_cfg,
        computation_cfg=computation_cfg,
        tsvd=tsvd,
    )
    gc.collect()
    torch.cuda.empty_cache()

    # merge units by template distance
    sorting, templates = _handle_merge(
        recording=recording,
        sorting=sorting,
        template_data=templates,
        motion_est=motion_est,
        merge_cfg=template_merge_cfg,
        computation_cfg=computation_cfg,
        waveform_cfg=waveform_cfg,
        template_cfg=template_cfg,
    )
    gc.collect()
    torch.cuda.empty_cache()

    # re-order along probe length
    if depth_order:
        sorting, templates = reorder_by_depth(sorting, templates)

    return sorting, ensure_save(templates, template_npz_path)


def realign_and_chuck_noisy_template_units(
    recording,
    sorting,
    template_data=None,
    motion_est=None,
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
    w = template_data.snrs_by_channel()
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
    scbc = template_data.spike_counts_by_channel
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


def ensure_save(template_data, template_npz_path):
    if template_npz_path is not None:
        template_npz_path.parent.parent.mkdir(exist_ok=True)
        template_npz_path.parent.mkdir(exist_ok=True)
        template_data.to_npz(template_npz_path)
    return template_data


def _check_still_valid(sorting: DARTsortSorting):
    assert sorting.labels is not None
    if (sorting.labels < 0).all():
        raise ValueError("All units were thrown away during template postprocessing.")


def _handle_merge(
    *,
    recording: BaseRecording,
    sorting: DARTsortSorting,
    motion_est,
    template_data: TemplateData,
    merge_cfg: TemplateMergeConfig | None,
    computation_cfg: ComputationConfig,
    waveform_cfg: WaveformConfig,
    template_cfg: TemplateConfig,
) -> tuple[DARTsortSorting, TemplateData]:
    if merge_cfg is None or not merge_cfg.merge_distance_threshold:
        return sorting, template_data

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
    sorting = merge_res["sorting"]
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
            motion_est=motion_est,
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
    old_recompute_ix = ui[new_recompute_ix]

    logger.dartsortverbose(f"Merged template indices: {new_recompute_ix.tolist()}")

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

    template_data = TemplateData(
        templates,
        unit_ids=np.arange(len(templates)),
        spike_counts=spike_counts,
        spike_counts_by_channel=spike_counts_by_channel,
        raw_std_dev=raw_std_dev,
        registered_geom=template_data.registered_geom,
        trough_offset_samples=template_data.trough_offset_samples,
    )
    return sorting, template_data


def _quick_mean_templates(
    recording, sorting, waveform_cfg, computation_cfg, motion_est
):
    return TemplateData.from_config(
        recording=recording,
        sorting=sorting,
        motion_est=motion_est,
        waveform_cfg=waveform_cfg,
        template_cfg=TemplateConfig(denoising_method="none"),
        computation_cfg=computation_cfg,
    )


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


def flag_possible_cc_error_spikes(
    sorting: DARTsortSorting,
    subtraction_cfg,
    amplitudes_dataset_name="denoised_ptp_amplitudes",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ..util.py_util import timer

    times = sorting.times_samples
    channels = sorting.channels
    amps = getattr(sorting, amplitudes_dataset_name)
    xy = sorting.geom[channels]  # type: ignore
    n = len(times)

    # rescale units so that the max allowed temporal and spatial dists are 10
    times = times / subtraction_cfg.temporal_dedup_radius_samples
    xy_subtract = xy / subtraction_cfg.subtract_radius
    kdt_subtract = KDTree(np.c_[times, xy_subtract])

    # get all possible neighbors
    with timer("a"):
        sdm = kdt_subtract.sparse_distance_matrix(
            kdt_subtract, max_distance=1.0, p=np.inf, output_type="ndarray"
        )
    # this is more an adjacency matrix. 0s will be discarded later.
    v = np.ones(sdm["i"].shape, dtype=np.float32)
    coo = coo_array((v, (sdm["i"], sdm["j"])), shape=(n, n), dtype=v.dtype)

    # remove the exclusions by marking their distance as large
    if subtraction_cfg.spatial_dedup_radius:
        xy_dedup = xy / subtraction_cfg.spatial_dedup_radius
        kdt_dedup = KDTree(np.c_[times, xy_dedup])
        with timer("b"):
            sdm_dedup = kdt_dedup.sparse_distance_matrix(
                kdt_dedup, max_distance=1.0, p=np.inf, output_type="ndarray"
            )
        del kdt_dedup, xy_dedup
        v = np.ones(sdm_dedup["i"].shape, dtype=np.float32)
        coo_dedup = coo_array(
            (v, (sdm_dedup["i"], sdm_dedup["j"])), shape=(n, n), dtype=v.dtype
        )
        coo = coo - coo_dedup
        coo = coo.tocoo()

        # dedup neighbors are removed
        coo.sum_duplicates()
        coo.eliminate_zeros()

    # apply higher amplitude criterion
    bigger = np.flatnonzero(amps[coo.coords[0]] > amps[coo.coords[1]])
    coo.data[bigger] = 0.0
    coo.eliminate_zeros()

    # i am suspicious if i have any neighbors. obviously many spikes do.
    # it's just that a unit consisting entirely of such spikes is suspicious.
    flagged = np.zeros(len(times), dtype=bool)
    ii, jj = coo.coords
    flagged[ii] = True

    return flagged, ii, jj


def cc_flag_criterion(
    sorting: DARTsortSorting,
    detection_cfg: Any | None = None,
    max_cc_flag_rate=1.0,
    cc_flag_entropy_cutoff=0.0,
    amplitudes_dataset_name="denoised_ptp_amplitudes",
) -> np.ndarray | None:
    if max_cc_flag_rate == 1.0:
        return None
    if not isinstance(detection_cfg, SubtractionConfig):
        # this step applied only at subtraction to clean up nn error units
        return None

    flagged, ii, jj = flag_possible_cc_error_spikes(
        sorting=sorting,
        subtraction_cfg=detection_cfg,
        amplitudes_dataset_name=amplitudes_dataset_name,
    )
    assert sorting.labels is not None
    li = sorting.labels[ii]
    lj = sorting.labels[jj]

    rate = np.zeros(sorting.unit_ids.shape)
    entropy = np.zeros_like(rate)
    for j, u in enumerate(sorting.unit_ids):
        inu = np.flatnonzero(sorting.labels == u)
        rate[j] = flagged[inu].mean()
        _, friend_count = np.unique(lj[li == u], return_counts=True)
        friend_p = friend_count / friend_count.sum()
        entropy[j] = -(np.log(friend_p) * friend_p).sum()

    bad = (rate > max_cc_flag_rate) & (entropy < cc_flag_entropy_cutoff)
    logger.dartsortdebug(
        f"CCG peak criterion flagged {bad.sum().item()} "
        f"units ({sorting.unit_ids[np.flatnonzero(bad)].tolist()})."
    )
    return np.logical_not(bad)
