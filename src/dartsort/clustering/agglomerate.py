"""Agglomeration of clusters to fix up GMM oversplits."""

from threading import local
from typing import cast

import numba
import numpy as np
import torch
from spikeinterface.core import BaseRecording
from tqdm.auto import tqdm

from ..templates.template_util import shared_basis_compress_templates
from ..templates.templates import TemplateData
from ..util.data_util import DARTsortSorting
from ..util.internal_config import (
    ComputationConfig,
    RefinementConfig,
    TemplateConfig,
    TemplateMergeConfig,
    WaveformConfig,
    default_waveform_cfg,
)
from ..util.job_util import ensure_computation_config
from ..util.logging_util import get_logger
from ..util.motion import MotionInfo
from ..util.multiprocessing_util import pool_from_cfg
from ..util.py_util import databag
from ..util.spiketorch import (
    best_shared_pconv,
    scaled_normeuc_from_dots,
    shared_temporal_pconv,
)
from .cluster_util import linkage_mask, recluster, sparsify_labels
from .gmm.mixture import Scores

logger = get_logger(__name__)


@databag
class Agglomeration:
    agglomerated_sorting: DARTsortSorting
    merge_mapping: np.ndarray
    distances: np.ndarray
    shifts: np.ndarray


def agglomerate(
    *,
    sorting: DARTsortSorting,
    recording: BaseRecording | None,
    template_merge_cfg: TemplateMergeConfig | None,
    refinement_cfg: RefinementConfig | None,
    motion: MotionInfo,
    template_data: TemplateData | None = None,
    computation_cfg: ComputationConfig | None = None,
    waveform_cfg: WaveformConfig,
    show_progress: bool = True,
) -> Agglomeration:
    computation_cfg = ensure_computation_config(computation_cfg)

    if template_merge_cfg is None:
        assert refinement_cfg is not None
        template_merge_cfg = refinement_cfg.template_merge_cfg

    tdist = template_distances(
        sorting=sorting,
        recording=recording,
        motion=motion,
        template_data=template_data,
        waveform_cfg=waveform_cfg,
        template_merge_cfg=template_merge_cfg,
        computation_cfg=computation_cfg,
    )

    # if not doing any QDA, be done now.
    if refinement_cfg is None or not refinement_cfg.qda_threshold:
        agg_sorting, new_ids = recluster(
            sorting=sorting,
            unit_ids=tdist.template_data.unit_ids,
            dists=tdist.distances,
            shifts=tdist.shifts,
            unit_snrs=tdist.template_data.snrs_by_channel().max(1),
            threshold=template_merge_cfg.merge_distance_threshold,
            link=template_merge_cfg.linkage,
        )
        return Agglomeration(
            agglomerated_sorting=agg_sorting,
            merge_mapping=new_ids,
            distances=tdist.distances,
            shifts=tdist.shifts,
        )

    # tdist tells us the possible merges
    mask = linkage_mask(
        tdist.distances,
        linkage_method=template_merge_cfg.linkage,
        threshold=template_merge_cfg.merge_distance_threshold,
    )

    # reconstruct scores from sorting attached data (exclude train_ix?)
    ntlabels, ntscores = _get_non_train_scores(sorting)

    # restrict mask by overlap criteria
    qda_res = qda(
        mask=mask,
        labels=ntlabels,
        scores=ntscores,
        min_iou=refinement_cfg.qda_min_iou,
        min_cov=refinement_cfg.qda_min_coverage,
        show_progress=show_progress,
        computation_cfg=computation_cfg,
    )

    qda_mask = np.all(
        [
            qda_res.coverage >= refinement_cfg.qda_min_coverage,
            qda_res.iou >= refinement_cfg.qda_min_iou,
            qda_res.score >= refinement_cfg.qda_threshold,
            qda_res.min_ratio >= refinement_cfg.qda_min_ratio,
        ],
        axis=0,
    )
    np.fill_diagonal(qda_mask, True)
    assert np.all(qda_mask <= mask)
    qda_as_dist = np.logical_not(qda_mask).astype(np.float32)

    agg_sorting, new_ids = recluster(
        sorting=sorting,
        unit_ids=tdist.template_data.unit_ids,
        dists=qda_as_dist,
        shifts=tdist.shifts,
        unit_snrs=tdist.template_data.snrs_by_channel().max(1),
        threshold=0.5,
        link=template_merge_cfg.linkage,
    )
    return Agglomeration(
        agglomerated_sorting=agg_sorting,
        merge_mapping=new_ids,
        distances=tdist.distances,
        shifts=tdist.shifts,
    )


@databag
class TemplateDistanceResult:
    distances: np.ndarray
    shifts: np.ndarray
    r2: np.ndarray
    template_data: TemplateData


def template_distances(
    *,
    template_data: TemplateData | None,
    template_merge_cfg: TemplateMergeConfig,
    sorting: DARTsortSorting | None = None,
    recording: BaseRecording | None = None,
    motion: MotionInfo | None = None,
    template_cfg: TemplateConfig | None = None,
    waveform_cfg: WaveformConfig = default_waveform_cfg,
    computation_cfg: ComputationConfig | None = None,
    allow_whitening_fail: bool = False,
) -> TemplateDistanceResult:
    computation_cfg = ensure_computation_config(computation_cfg)
    device = computation_cfg.actual_device()

    if template_merge_cfg.whitening.strategy == "prewhiten_postapply":
        raise ValueError(
            "prewhiten_postapply does not make sense for template distance."
        )

    need_whitening = template_merge_cfg.whitening.strategy != "none"
    if template_data is None:
        need_templates = True
    elif need_whitening and template_data.whitener is None:
        logger.dartsortdebug(
            "Need to recompute templates for distances since they were not whitened."
        )
        need_templates = True
        if allow_whitening_fail:
            # this path is useful for visualization, when we don't care too much.
            if sorting is None or sorting.parent_h5_path is None:
                logger.info("Can't whiten, sorting doesn't have the data.")
                need_templates = False
    else:
        need_templates = False

    if need_templates:
        assert sorting is not None
        assert recording is not None
        assert motion is not None
        assert template_cfg is not None
        template_data = TemplateData.from_config(
            recording=recording,
            sorting=sorting,
            template_cfg=template_merge_cfg.to_template_config(template_cfg),
            motion=motion,
            waveform_cfg=waveform_cfg,
            computation_cfg=computation_cfg,
        )
    assert template_data is not None

    if template_data.tsvd is not None:
        basis = template_data.tsvd.components_
    else:
        basis = None

    sbt = shared_basis_compress_templates(
        template_data,
        rank=template_merge_cfg.svd_compression_rank,
        precomputed_basis=basis,
        computation_cfg=computation_cfg,
        with_r2=True,
    )
    tcomp = torch.asarray(sbt.temporal_components, device=device)
    spatial_sing = torch.asarray(sbt.spatial_singular, device=device)

    need_whiten_mul = template_merge_cfg.whitening.strategy == "postwhiten"
    if need_whiten_mul:
        ww = torch.asarray(template_data.whitener).to(spatial_sing)
        k, r, c = spatial_sing.shape
        spatial_sing = (spatial_sing.view(k * r, c) @ ww.T).view(k, r, c)

    tconv = shared_temporal_pconv(
        temporal_comps=tcomp, up_temporal_comps=tcomp[:, None]
    )
    tconv = tconv[:, :, 0, :]

    # trim tconv to shift range
    max_shift = WaveformConfig.ms_to_samples(
        ms=template_merge_cfg.max_shift_ms,
        sampling_frequency=template_data.sampling_frequency,
    )
    conv_len = tconv.shape[2]
    center = conv_len // 2
    assert conv_len == 2 * center + 1
    assert center >= max_shift
    tconv = tconv[:, :, center - max_shift : center + 1 + max_shift]
    tconv = tconv.contiguous()

    best_conv, best_lag = best_shared_pconv(tconv, spatial_sing)

    # convert conv to distance
    if template_merge_cfg.distance_kind == "scaled_normeuc":
        dist = scaled_normeuc_from_dots(best_conv)
    else:
        raise ValueError(f"{template_merge_cfg.distance_kind=} not implemented.")

    # okay then
    return TemplateDistanceResult(
        distances=dist.numpy(force=True),
        shifts=best_lag.numpy(force=True),
        r2=cast(np.ndarray, sbt.r2),
        template_data=template_data,
    )


def _get_non_train_scores(sorting: DARTsortSorting) -> tuple[np.ndarray, Scores]:
    is_train = getattr(sorting, "gmm_train", None)
    cand = getattr(sorting, "gmm_candidates", None)
    log_liks = getattr(sorting, "gmm_log_liks", None)
    resp = getattr(sorting, "gmm_responsibilities", None)

    assert is_train is not None
    assert cand is not None
    assert log_liks is not None
    assert resp is not None

    not_train = torch.asarray(np.flatnonzero(np.logical_not(is_train)))
    cand = torch.asarray(cand[not_train])
    log_liks = torch.asarray(log_liks[not_train])
    resp = torch.asarray(resp[not_train])

    scores = Scores(
        candidates=cand, log_liks=log_liks, responsibilities=resp, duties=None
    )
    assert sorting.labels is not None
    labels = sorting.labels[not_train]
    return labels, scores


@databag
class QDAResult:
    score: np.ndarray
    min_ratio: np.ndarray
    iou: np.ndarray
    coverage: np.ndarray


def qda(
    *,
    mask: np.ndarray,
    labels: np.ndarray,
    scores: Scores,
    min_iou: float = 0.5,
    min_cov: float = 0.3,
    min_count: int = 20,
    show_progress: bool,
    computation_cfg: ComputationConfig,
) -> QDAResult:
    iou = np.zeros(mask.shape, dtype=np.float32)
    ctx = QDACtx(
        inus=sparsify_labels(labels),
        cand=scores.candidates.numpy(force=True),
        log_liks=scores.log_liks.numpy(force=True),
        min_iou=min_iou,
        min_cov=min_cov,
        min_count=min_count,
        iou=iou,
        cov=iou.copy(),
        score=iou.copy(),
        min_ratio=iou.copy(),
    )

    n_jobs, Executor, context, *_ = pool_from_cfg(
        computation_cfg, check_local=True, small=True
    )
    with Executor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=_qda_init,
        initargs=(ctx,),
    ) as pool:
        ii, jj = np.triu_indices_from(mask, k=1)
        kk = np.flatnonzero(mask[ii, jj])
        ii = ii[kk]
        jj = jj[kk]

        results = pool.map(_qda_job, np.c_[ii, jj])
        if show_progress:
            results = tqdm(results, desc=f"QDA:{n_jobs}")

        for _ in results:
            pass

    return QDAResult(
        score=ctx.score, min_ratio=ctx.min_ratio, iou=ctx.iou, coverage=ctx.cov
    )


_qda_context = local()
_qda_context.ctx = None


@databag
class QDACtx:
    inus: dict[int, np.ndarray]
    cand: np.ndarray
    log_liks: np.ndarray
    min_iou: float
    min_cov: float
    min_count: int
    iou: np.ndarray
    cov: np.ndarray
    score: np.ndarray
    min_ratio: np.ndarray


def _qda_init(ctx):
    global _qda_context
    _qda_context.ctx = ctx


def _qda_job(ij):
    p = _qda_context.ctx
    assert p is not None

    i, j = ij

    ini = p.inus[i]
    inj = p.inus[j]
    inij, overlap, imask, jmask, iou, cov = _ioucov(i, j, ini, inj, p.cand)
    p.iou[i, j] = p.iou[j, i] = iou
    p.cov[i, j] = p.cov[j, i] = cov
    if iou < p.min_iou:
        return
    if cov < p.min_cov:
        return
    if ini.size + inj.size < p.min_count:
        return
    dll = _dll(inij, overlap, imask, jmask, p.log_liks)


@numba.jit(nopython=True, nogil=True)
def _ioucov(i: int, j: int, ini: np.ndarray, inj: np.ndarray, cand: np.ndarray):
    inij = np.concatenate([ini, inj])
    ni = ini.size
    nj = inj.size
    nij = inij.size

    candij = cand[inij]
    imask = candij == i
    jmask = candij == j

    icov = imask.any(axis=1)
    jcov = jmask.any(axis=1)
    overlap = np.logical_and(icov, jcov)

    noi = overlap[:ni].sum()
    noj = overlap[ni:].sum()
    iou = (noi + noj) / nij
    cov = min(noi / ni, noj / nj)
    return inij, overlap, imask, jmask, iou, cov


@numba.jit(nopython=True, nogil=True)
def _dll(
    inij,
    overlap: np.ndarray,
    imask: np.ndarray,
    jmask: np.ndarray,
    log_liks: np.ndarray,
):
    ll = log_liks[inij]
    _, ixi = np.nonzero(imask[overlap])
    lli = np.take_along_axis(ll[overlap], ixi[:, None], axis=1)[:, 0]
    _, ixj = np.nonzero(jmask[overlap])
    llj = np.take_along_axis(ll[overlap], ixj[:, None], axis=1)[:, 0]
    return lli - llj
