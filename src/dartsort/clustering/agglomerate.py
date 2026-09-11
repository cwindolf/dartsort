"""Agglomeration of clusters to fix up GMM oversplits."""

from threading import local
from typing import cast

import numba
import numpy as np
import torch
from KDEpy import FFTKDE
from spikeinterface.core import BaseRecording

from ..templates.template_util import shared_basis_compress_templates
from ..templates.templates import TemplateData
from ..util.data_util import (
    DARTsortSorting,
    apply_label_remapping_in_place,
    count_not_sorted,
    pos_int_unique_and_counts,
)
from ..util.internal_config import (
    ComputationConfig,
    RefinementConfig,
    TemplateConfig,
    TemplateMergeConfig,
    WaveformConfig,
    default_waveform_cfg,
)
from ..util.job_util import ensure_computation_config
from ..util.logging_util import get_logger, progbar
from ..util.motion import MotionInfo
from ..util.multiprocessing_util import pool_from_cfg
from ..util.py_util import databag
from ..util.spiketorch import (
    best_shared_pconv,
    scaled_normeuc_from_dots,
    shared_temporal_pconv,
    weighted_best_lagged_scaled_normeuc_dist,
)
from ..util.waveform_util import make_channel_index
from .cluster_util import (
    ViolationInfo,
    apply_reclustering,
    closest_registered_channels,
    hierarchical_cluster,
    meet,
    reorder_by_depth,
    sparsify_labels,
    violation_statistics,
)

logger = get_logger(__name__)


@databag
class Agglomeration:
    agglomerated_sorting: DARTsortSorting
    merge_mapping: np.ndarray
    template_distances: np.ndarray | None
    template_shifts: np.ndarray | None
    violation: ViolationInfo | None
    glom_cost: np.ndarray | None


def agglomerate(
    *,
    sorting: DARTsortSorting,
    recording: BaseRecording,
    template_merge_cfg: TemplateMergeConfig | None,
    refinement_cfg: RefinementConfig | None,
    motion: MotionInfo,
    template_data: TemplateData | None = None,
    computation_cfg: ComputationConfig | None = None,
    waveform_cfg: WaveformConfig,
    in_place: bool = True,
) -> Agglomeration:
    """Postprocessing merge step

    By default, template distances and a refractoriness statistic are combined
    with hierarchical clustering to perform the merge.

    If refinement_cfg is not set, the merge is just a hierarchical clustering
    of the template distances.

    The algorithm is like this.
     - Pair i,j is allowed to be merged if any of:
        - Template distance < merge_distance_threshold (and, a non-default
          QDA/overlap condition holds if specified)
        - Template distance < glom_force_merge_template_distance
     - Merges are decided within allowed groups by average linkage on a chance-
       corrected measure of violation within the groups
        - Optionally, the criterion can be restricted by a "worst pair" violation
          rather than average if glom_veto_threshold is set.
        - Pairs with low overlap (glom_min_violation_evidence) are merged only
          under the force_merge_template_distance

    This code might read a little weird, because it's used both as a clustering pass
    and to apply a distance-based merge to the template library. In the latter case
    template_data is supplied. So there are some conditions that depend on that.
    """
    computation_cfg = ensure_computation_config(computation_cfg)

    if template_merge_cfg is None:
        assert refinement_cfg is not None
        template_merge_cfg = refinement_cfg.template_merge_cfg

    if template_data is None:
        sorting = sorting.flatten(include_gmm_properties=True, in_place=in_place)

    if template_merge_cfg is None:
        agg_sorting, merge_mapping = clean_final_sorting(
            sorting, motion=motion, merge_mapping=None, in_place=in_place
        )
        return Agglomeration(
            agglomerated_sorting=agg_sorting,
            merge_mapping=merge_mapping,
            template_distances=None,
            template_shifts=None,
            violation=None,
            glom_cost=None,
        )

    tdist = template_distances(
        sorting=sorting,
        recording=recording,
        motion=motion,
        template_data=template_data,
        waveform_cfg=waveform_cfg,
        template_merge_cfg=template_merge_cfg,
        computation_cfg=computation_cfg,
    )
    distances = np.minimum(tdist.distances, tdist.distances.T)
    np.fill_diagonal(distances, 0.0)
    assert np.array_equal(tdist.template_data.unit_ids, np.arange(distances.shape[0]))

    # early out
    if refinement_cfg is None:
        glom_cost = distances
        veto_cost = veto_threshold = violation = None
        linkage_method = template_merge_cfg.linkage
        threshold = template_merge_cfg.merge_distance_threshold
        violation = None
    else:
        res = _agglomerate_violation_merge(
            sorting, distances, refinement_cfg, template_merge_cfg, computation_cfg
        )
        (
            _mask,
            glom_cost,
            veto_cost,
            linkage_method,
            threshold,
            veto_threshold,
            violation,
        ) = res

    _, merge_mapping = hierarchical_cluster(
        None, glom_cost, linkage_method=linkage_method, threshold=threshold
    )
    if veto_cost is not None:
        assert veto_threshold is not None
        _, veto_mapping = hierarchical_cluster(
            None, veto_cost, linkage_method="complete", threshold=veto_threshold
        )
        merge_mapping = meet(merge_mapping, veto_mapping)

    agg_sorting = apply_reclustering(
        sorting=sorting,
        merge_mapping=merge_mapping,
        shifts=tdist.shifts,
        unit_snrs=tdist.template_data.snrs_by_channel().max(1),
        in_place=in_place,
    )
    agg_sorting = combine_gmm_scores(
        agg_sorting, new_ids=merge_mapping, in_place=in_place
    )
    agg_sorting, merge_mapping = clean_final_sorting(
        agg_sorting, motion=motion, merge_mapping=merge_mapping, in_place=in_place
    )

    return Agglomeration(
        agglomerated_sorting=agg_sorting,
        merge_mapping=merge_mapping,
        template_distances=distances,
        template_shifts=tdist.shifts,
        violation=violation,
        glom_cost=glom_cost,
    )


def _agglomerate_violation_merge(
    sorting: DARTsortSorting,
    distances: np.ndarray,
    refinement_cfg: RefinementConfig,
    template_merge_cfg: TemplateMergeConfig,
    computation_cfg: ComputationConfig,
    huge=1e8,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    str,
    float,
    float | None,
    ViolationInfo | None,
]:
    mask = distances < template_merge_cfg.merge_distance_threshold

    if refinement_cfg.glom_qda_overlap or refinement_cfg.glom_qda_bimodality:
        qda_res = qda(
            mask=mask,
            sorting=sorting,
            min_iou=refinement_cfg.qda_min_iou
            if refinement_cfg.glom_qda_overlap
            else 0.0,
            min_cov=refinement_cfg.qda_min_coverage
            if refinement_cfg.glom_qda_overlap
            else 0.0,
            bimodality=refinement_cfg.glom_qda_bimodality,
            show_progress=False,
            computation_cfg=computation_cfg,
        )
        if refinement_cfg.glom_qda_overlap:
            mask &= np.logical_and(
                qda_res.coverage >= refinement_cfg.qda_min_coverage,
                qda_res.iou >= refinement_cfg.qda_min_iou,
            )
        if refinement_cfg.glom_qda_bimodality:
            mask &= np.logical_or(
                qda_res.score >= refinement_cfg.qda_uni_score,
                np.logical_and(
                    qda_res.score >= refinement_cfg.qda_threshold,
                    qda_res.min_ratio >= refinement_cfg.qda_min_ratio,
                ),
            )

    force_mask = distances < refinement_cfg.glom_force_merge_template_distance
    mask |= force_mask
    np.fill_diagonal(mask, True)

    # early out: no violation stuff. just distance mask connected components.
    if refinement_cfg.glom_violation_threshold is None:
        glom_cost = np.logical_not(mask).astype(np.float32)
        linkage_method = template_merge_cfg.linkage
        threshold = 0.5
        veto_cost = veto_threshold = None
        return (
            mask,
            glom_cost,
            veto_cost,
            linkage_method,
            threshold,
            veto_threshold,
            None,
        )

    violation = violation_statistics(
        sorting,
        censor_ms=refinement_cfg.censor_ms,
        viol_ms=refinement_cfg.glom_violation_ms,
        jitter_ms=refinement_cfg.glom_jitter_ms,
    )
    assert violation.jitter_counts is not None
    assert violation.jitter_counts.shape == distances.shape

    # main mask:
    # it's basically forcing where there's low evidence, and it's the ratio
    # of observed violation counts to their jitter average elsewhere.
    glom_cost = violation.jitter_viol_ratio(
        refinement_cfg.glom_min_violation_evidence,
        fill_value=np.where(force_mask, 0.0, huge),
    )
    # mask out distance enemies
    glom_cost[np.logical_not(mask)] = huge
    glom_cost = np.minimum(glom_cost, glom_cost.T)
    np.fill_diagonal(glom_cost, 0.0)

    # last thing: optionally, be extremely finnicky about merging into violated groups
    # i am not sure if this will be a good idea or not; it may cost too many merges
    # to be worthwhile.
    if refinement_cfg.glom_veto_threshold is not None:
        veto_cost = violation.jitter_viol_ratio(
            refinement_cfg.glom_veto_min_evidence, fill_value=0.0
        )
        veto_cost[np.logical_not(mask)] = huge
        veto_cost = np.minimum(veto_cost, veto_cost.T)
        np.fill_diagonal(veto_cost, 0.0)
    else:
        veto_cost = None

    return (
        mask,
        glom_cost,
        veto_cost,
        refinement_cfg.glom_violation_linkage,
        refinement_cfg.glom_violation_threshold,
        refinement_cfg.glom_veto_threshold,
        violation,
    )


@databag
class TemplateDistanceResult:
    distances: np.ndarray
    shifts: np.ndarray
    r2: np.ndarray
    template_data: TemplateData
    spatial_weights: np.ndarray | None
    spatial_iou: np.ndarray | None


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

    if template_data is not None and template_cfg is not None:
        if template_merge_cfg.whitening.strategy == "none":
            assert template_cfg.whitening.strategy in ("none", "prewhiten_postapply")
        else:
            assert template_cfg.whitening == template_merge_cfg.whitening

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

    spatial_weights = spatial_iou = None
    if template_merge_cfg.distance_kind == "scaled_normeuc":
        best_conv, best_lag = best_shared_pconv(tconv, spatial_sing)
        dist = scaled_normeuc_from_dots(
            best_conv,
            scale_var=template_merge_cfg.amplitude_scaling_variance,
            scale_boundary=template_merge_cfg.amplitude_scaling_boundary,
        )
    elif template_merge_cfg.distance_kind == "weighted_scaled_normeuc":
        assert sorting is not None
        assert motion is not None
        spatial_weights = count_radial_weights(
            sorting=sorting,
            motion=motion,
            radius=template_merge_cfg.weighted_dist_radius,
        )
        assert np.isfinite(spatial_weights).all()
        dist, best_lag, iou = weighted_best_lagged_scaled_normeuc_dist(
            tconv=tconv,
            spatial_sing=spatial_sing,
            weights=torch.asarray(spatial_weights).to(spatial_sing),
            scale_var=template_merge_cfg.amplitude_scaling_variance,
            scale_boundary=template_merge_cfg.amplitude_scaling_boundary,
        )
        dist.masked_fill_(iou < template_merge_cfg.weighted_dist_min_iou, torch.inf)
        spatial_iou = iou.numpy(force=True)
    else:
        raise ValueError(f"{template_merge_cfg.distance_kind=} not implemented.")

    # okay then
    return TemplateDistanceResult(
        distances=dist.numpy(force=True),
        shifts=best_lag.numpy(force=True),
        r2=cast(np.ndarray, sbt.r2),
        template_data=template_data,
        spatial_weights=spatial_weights,
        spatial_iou=spatial_iou,
    )


@databag
class QDAResult:
    """Unit pair QDA metrics

    Algorithm:
     - For a pair of units i,j, grab all the spikes which both units
       assign a likelihood to (their candidate set intersection)
     - Compute coverage statistics:
        - Let #i be the number of spikes for which i is a candidate, sim #j.
        - Let #union be the number of spikes for which either is a candidate
        - Let #inter be the number of spikes in the intersection
        - Let `iou[i,j]` be #inter / #union
        - Let `cov[i,j]` be min(#inter / #i, #inter / #j)
     - Use a 1d KDE to estimate the density of the difference in likelihoods
       of the intersection spikes, lik[j] - lik[i]. Note that 0 is the decision
       boundary above which a spike comes from unit j, below which i.
       Call that KDE f(l)
     - Compute bimodality statistics
        - Let fi = max_{l<0} f(l), fj = max_{l>0} f(l)
        - Let `score[i,j]` = f(0) / min(fi,fj)
        - Let `min_ratio[i,j]` = f(0) / max(fi,fj)
    """

    score: np.ndarray
    min_ratio: np.ndarray
    iou: np.ndarray
    coverage: np.ndarray


def qda(
    *,
    mask: np.ndarray | None,
    sorting: DARTsortSorting,
    min_iou: float = 0.5,
    min_cov: float = 0.35,
    min_count: int = 20,
    dx: float = 1.0,
    bimodality: bool = True,
    show_progress: bool,
    computation_cfg: ComputationConfig,
) -> QDAResult:
    from ..util.data_util import get_gmm_scores

    # reconstruct scores from sorting attached data (exclude train_ix?)
    gscores = get_gmm_scores(sorting)
    glabels = sorting.labels
    assert glabels is not None

    if mask is None:
        mask = np.ones((sorting.n_units, sorting.n_units), dtype=bool)

    iou = np.zeros(mask.shape, dtype=np.float32)
    ctx = QDACtx(
        inus=sparsify_labels(glabels),
        cand=gscores.candidates.numpy(force=True),
        log_liks=gscores.log_liks.numpy(force=True),
        min_iou=min_iou,
        min_cov=min_cov,
        min_count=min_count,
        iou=iou,
        cov=iou.copy(),
        score=iou.copy(),
        min_ratio=iou.copy(),
        dx=dx,
        bimodality=bimodality,
    )

    n_jobs, Executor, context, *_ = pool_from_cfg(
        computation_cfg, check_local=True, small=True, cpu=True
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
            results = progbar(
                results,
                desc=f"QDA:{n_jobs}",
                total=ii.shape[0],
                mininterval=0.5,
                smoothing=0.0,
            )

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
    dx: float
    bimodality: bool


def _qda_init(ctx):
    _qda_context.ctx = ctx


def _qda_job(ij):
    p = _qda_context.ctx
    assert p is not None

    i, j = ij

    ini = p.inus.get(i)
    inj = p.inus.get(j)
    if ini is None or inj is None:
        return
    inij = np.concatenate((ini, inj), axis=0)
    overlap, imask, jmask, iou, cov = _ioucov(i, j, ini, inj, inij, p.cand)
    p.iou[i, j] = p.iou[j, i] = iou
    p.cov[i, j] = p.cov[j, i] = cov

    if not p.bimodality:
        return
    if iou < p.min_iou:
        return
    if cov < p.min_cov:
        return
    if ini.size + inj.size < p.min_count:
        return

    dll = _dll(inij, overlap, imask, jmask, p.log_liks)
    if not dll.size:
        p.score[i, j] = p.score[j, i] = 0.0
        p.min_ratio[i, j] = p.min_ratio[j, i] = 0.0
        return

    vmn, vmx = torch.aminmax(torch.asarray(dll))
    vm = max(-vmn, vmx)
    nbins = (vm + p.dx) // p.dx
    binc = np.arange(-nbins * p.dx, (nbins + 1) * p.dx, p.dx)
    bc = binc.shape[0] // 2
    assert np.isclose(binc[bc], 0.0)
    assert binc.shape[0] == 2 * bc + 1

    try:
        kde = FFTKDE(bw="ISJ").fit(dll)
    except ValueError as e:
        logger.dartsortdebug(f"KDEpy error: {e}")
        p.score[i, j] = p.score[j, i] = 0.0
        p.min_ratio[i, j] = p.min_ratio[j, i] = 0.0
        return

    kde = cast(np.ndarray, kde.evaluate(binc))
    score, min_ratio = bimod_stats(kde)
    p.score[i, j] = p.score[j, i] = score
    p.min_ratio[i, j] = p.min_ratio[j, i] = min_ratio


@numba.jit("b1[:](b1[:,:])", nopython=True, nogil=True)
def np_any_axis1(x):
    out = x[:, 0]
    for i in range(1, x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out


@numba.jit(
    "Tuple((b1[:],b1[:,:],b1[:,:],f8,f8))(i8,i8,i8[:],i8[:],i8[:],i4[:,:])",
    nopython=True,
    nogil=True,
)
def _ioucov(i, j, ini: np.ndarray, inj: np.ndarray, inij: np.ndarray, cand: np.ndarray):
    ni = ini.size
    nj = inj.size
    nij = inij.size

    candij = cand[inij]
    imask = candij == i
    jmask = candij == j

    icov = np_any_axis1(imask)
    jcov = np_any_axis1(jmask)
    overlap = np.logical_and(icov, jcov)

    noi = overlap[:ni].sum()
    noj = overlap[ni:].sum()
    iou = (noi + noj).item() / nij
    cov = min(noi / ni, noj / nj)
    return overlap, imask, jmask, iou, cov


@numba.jit("f4[:](i8[:],b1[:],b1[:,:],b1[:,:],f4[:,:])", nopython=True, nogil=True)
def _dll(
    inij: np.ndarray,
    overlap: np.ndarray,
    imask: np.ndarray,
    jmask: np.ndarray,
    log_liks: np.ndarray,
):
    ll = log_liks[inij]
    olap = np.flatnonzero(overlap)
    _, ixi = np.nonzero(imask[olap])
    lli = np.take_along_axis(ll[olap], ixi[:, None], axis=1)[:, 0]
    _, ixj = np.nonzero(jmask[olap])
    llj = np.take_along_axis(ll[olap], ixj[:, None], axis=1)[:, 0]
    return lli - llj


def bimod_stats(h):
    assert h.ndim == 1
    assert h.size % 2
    cix = h.shape[0] // 2
    h0 = h[cix]
    da = h[:cix].max()
    db = h[cix + 1 :].max()
    dd = min(da, db)
    if np.isclose(dd, 0.0) and np.isclose(h0, 0.0):
        a = 0.0
    elif np.isclose(dd, 0.0):
        a = np.inf
    else:
        a = h0 / dd
    b = h0 / max(da, db)
    return a, b


def count_radial_weights(sorting: DARTsortSorting, motion: MotionInfo, radius: float):
    assert sorting.labels is not None
    kept = np.flatnonzero(sorting.labels >= 0)

    # which reg chans do the spikes land on?
    x, z = motion.geom[sorting.channels[kept]].T
    cc = closest_registered_channels(
        times_seconds=sorting.times_seconds[kept], x=x, z_abs=z, motion=motion
    )

    # count by label
    ll = sorting.labels[kept]
    counts = np.zeros((ll.max() + 1, motion.rgeom.shape[0]), dtype=np.int64)
    np.add.at(counts, (ll, cc), 1)

    # get radial neighborhoods
    ci = make_channel_index(motion.rgeom, radius, to_torch=False)

    # puff out with radial neighborhood and sum up the counts by label
    weights = np.zeros(counts.shape)
    for uu in range(counts.shape[0]):
        row = counts[uu]
        ii = np.flatnonzero(row)
        if not ii.size:
            continue
        vv = row[ii]
        vv = vv / vv.sum()

        for channel, value in zip(ii, vv, strict=True):
            cixs = ci[channel]
            cixs = cixs[cixs < motion.rgeom.shape[0]]
            weights[uu, cixs] += value

    denom = weights.max(axis=1, keepdims=True).clip(min=1e-10)  # avoid div by 0
    weights /= denom
    return weights


def combine_gmm_scores(
    sorting: DARTsortSorting,
    new_ids: np.ndarray,
    prefix: str = "gmm",
    in_place: bool = True,
) -> DARTsortSorting:
    """new_ids is a label remapping array."""
    candidates = getattr(sorting, f"{prefix}_candidates", None)
    responsibilities = getattr(sorting, f"{prefix}_responsibilities", None)
    logliks = getattr(sorting, f"{prefix}_log_liks", None)

    havec = candidates is not None
    haver = responsibilities is not None
    havel = logliks is not None
    assert all([havec, haver, havel]) or not any([havec, havel, haver])
    if not havec:
        return sorting
    assert candidates is not None
    assert responsibilities is not None
    assert logliks is not None
    n_cand = candidates.shape[1]
    assert logliks.shape[1] == responsibilities.shape[1] == n_cand + 1

    # check that new_ids is a merge
    assert (new_ids >= 0).all()
    unique_new_ids, new_id_counts = np.unique(new_ids, return_counts=True)
    assert unique_new_ids.shape[0] == unique_new_ids.max() + 1 <= new_ids.shape[0]
    if np.array_equal(new_ids, np.arange(new_ids.shape[0])):
        return sorting

    if not in_place:
        candidates = candidates.copy()
        responsibilities = responsibilities.copy()
        logliks = logliks.copy()

    # check invariants at the top
    nbye, maxdiff, n_neginf_viol = _check_soft_assign_invariants(
        candidates, responsibilities, logliks
    )
    assert maxdiff <= 1e-3, maxdiff
    assert not n_neginf_viol
    if sorting.labels is not None:
        not_noise = np.flatnonzero(sorting.labels >= 0)
        assert np.array_equal(
            sorting.labels[not_noise], new_ids[candidates[not_noise, 0]]
        )

    # merge candidates, deduplicate, and re-sort by likelihood
    apply_label_remapping_in_place(candidates, new_ids, allow_over=True)
    _combine_loop(candidates, new_id_counts, responsibilities, logliks)

    # check invariants at the bottom
    new_nbye, maxdiff, n_neginf_viol = _check_soft_assign_invariants(
        candidates, responsibilities, logliks
    )
    assert maxdiff <= 1e-3, maxdiff
    assert not n_neginf_viol
    assert new_nbye >= nbye

    labels = sorting.labels
    if labels is None:
        labels = np.full(candidates.shape[0], -1, dtype=candidates.dtype)
    elif not in_place:
        labels = labels.copy()
    n_changed, n_labeled = _assign_labels(candidates, logliks, labels)
    if n_labeled:
        logger.dartsortdebug(
            f"Mixture component aggregation changes {100 * n_changed / n_labeled:0.2f}"
            f"% of spike labels ({n_changed} spikes)."
        )

    return sorting.ephemeral_replace(
        labels=labels,
        **{
            f"{prefix}_candidates": candidates,
            f"{prefix}_responsibilities": responsibilities,
            f"{prefix}_log_liks": logliks,
        },
    )


@numba.njit(parallel=True, nogil=True)
def _check_soft_assign_invariants(
    cand: np.ndarray, resp: np.ndarray, logliks: np.ndarray
) -> tuple[int, float, int]:
    n_cand = cand.shape[1]
    nbye = 0
    maxdiff = -np.inf
    n_neginf_viol = 0

    for s in numba.prange(cand.shape[0]):  # ty: ignore[not-iterable]
        for j in range(n_cand):
            if cand[s, j] < 0:
                nbye += 1
                if logliks[s, j] != -np.inf:
                    n_neginf_viol += 1
            if j + 1 < n_cand:
                maxdiff = max(maxdiff, resp[s, j + 1] - resp[s, j])

    return nbye, maxdiff, n_neginf_viol


@numba.njit(parallel=True, nogil=True)
def _assign_labels(
    cand: np.ndarray, logliks: np.ndarray, labels: np.ndarray
) -> tuple[int, int]:
    """Update labels in-place, including some -1s."""
    n_changed = 0
    n_labeled = 0

    for s in numba.prange(cand.shape[0]):  # ty: ignore[not-iterable]
        if labels[s] >= 0:
            n_labeled += 1
            if labels[s] != cand[s, 0]:
                n_changed += 1
        labels[s] = cand[s, 0] if logliks[s, 0] >= logliks[s, -1] else -1

    return n_changed, n_labeled


@numba.njit(parallel=True, nogil=True)
def _combine_loop(
    cand: np.ndarray,
    new_id_counts: np.ndarray,
    mergedr: np.ndarray,
    mergedl: np.ndarray,
):
    n_cand = cand.shape[1]

    for s in numba.prange(cand.shape[0]):  # ty: ignore
        spike_cand = cand[s]
        for j in range(n_cand - 1):
            spike_candj = spike_cand[j]

            # noise or not a merge
            if spike_candj < 0 or new_id_counts[spike_candj] <= 1:
                continue

            # what later indices are equal to me?
            eq_spike_candj = spike_cand[j + 1 :] == spike_candj
            if eq_spike_candj.sum() < 1:
                continue

            # loop through and combine liks/resps, and -1 out the cands
            rsum = mergedr[s, j]
            lsum = mergedl[s, j]
            for i, k in enumerate(range(j + 1, n_cand)):
                if not eq_spike_candj[i]:
                    continue
                cand[s, k] = -1

                if mergedl[s, k] == -np.inf:
                    continue
                rsum += mergedr[s, k]
                lsum = np.logaddexp(lsum, mergedl[s, k])

                mergedr[s, k] = 0.0
                mergedl[s, k] = -np.inf

            mergedr[s, j] = rsum
            mergedl[s, j] = lsum

        # vacuum into noise component
        # this is partly to handle stuff that was missed before getting here
        # from flatten, for example
        for j in range(n_cand):
            if 0 <= spike_cand[j] < new_id_counts.shape[0]:
                continue
            rsj = mergedr[s, j]
            if rsj == 0:
                continue
            mergedr[s, -1] += rsj
            mergedr[s, j] = 0.0

        # stable descending insertion sort by log likelihood
        for j in range(1, n_cand):
            cj = cand[s, j]
            rj = mergedr[s, j]
            lj = mergedl[s, j]
            i = j - 1
            while i >= 0 and mergedl[s, i] < lj:
                cand[s, i + 1] = cand[s, i]
                mergedr[s, i + 1] = mergedr[s, i]
                mergedl[s, i + 1] = mergedl[s, i]
                i -= 1
            cand[s, i + 1] = cj
            mergedr[s, i + 1] = rj
            mergedl[s, i + 1] = lj


def clean_final_sorting(
    sorting: DARTsortSorting,
    *,
    motion: MotionInfo,
    dedup_ms: float = -1.0,
    merge_mapping: np.ndarray | None = None,
    score_by=("merged_log_liks", "gmm_log_liks", "scores"),
    in_place: bool = True,
) -> tuple[DARTsortSorting, np.ndarray]:
    """Deduplicate, flatten, depth-order

    Parameters
    ----------
    sorting : DARTsortSorting
    motion : MotionInfo
    dedup_ms : float
    merge_mapping : np.ndarray | None
        If this sorting was the result of a merge, caller might want to know
        what happened to those ids.
        the depth reordering.
    in_place : bool

    Returns
    -------
    clean_sorting : DARTsortSorting
    mapping : np.ndarray
    """
    if dedup_ms >= 0 and not any(sorting.has_dataset(k) for k in score_by):
        logger.warning(
            f"Not deduplicating: sorting has none of {score_by}. "
            "Set dedup_ms<0 to silence this."
        )
        dedup_ms = -1.0

    sorting = deduplicate_spikes(sorting, dedup_ms, in_place=in_place)
    sorting, reorder = reorder_by_depth(sorting, motion=motion, in_place=in_place)
    if merge_mapping is None:
        return sorting, reorder
    return sorting, reorder[np.unique(merge_mapping, return_inverse=True)[1]]


def deduplicate_spikes(
    sorting: DARTsortSorting,
    radius_ms: float = -1.0,
    score_by=("merged_log_liks", "gmm_log_liks", "scores"),
    in_place: bool = False,
) -> DARTsortSorting:
    if radius_ms < 0 or sorting.labels is None:
        return sorting

    radius_samples = WaveformConfig.ms_to_samples(
        ms=radius_ms,
        sampling_frequency=sorting.sampling_frequency,
    )
    assert radius_samples >= 0

    new_labels = sorting.labels if in_place else sorting.labels.copy()
    scores = None
    for sck in score_by:
        if not sorting.has_dataset(sck):
            continue
        sl = (slice(None), 0) if sck.endswith("log_liks") else ()
        scores = sorting.load_dataset(sck, sl=sl)
        if scores is not None:
            logger.dartsortdebug(f"deduplicate by score {sck}")
            break
    if scores is None:
        raise ValueError(f"sorting had none of {score_by}.")
    if scores.ndim >= 2:
        scores = scores[:, 0]
    assert scores.ndim == 1
    assert scores.shape == new_labels.shape

    # handle unsorted times
    if count_not_sorted(sorting.times_samples) > 0:
        tsort = np.argsort(sorting.times_samples, kind="stable")
        labels_by_time = new_labels[tsort]
        times_samples = sorting.times_samples[tsort]
        scores = scores[tsort]
    else:
        tsort = None
        labels_by_time = new_labels
        times_samples = sorting.times_samples

    unit_ids, _, _ = pos_int_unique_and_counts(labels_by_time)
    ndrop = 0
    unit_mask_tmp = np.empty(labels_by_time.shape, dtype=bool)
    for unit_id in unit_ids:
        in_unit = np.flatnonzero(np.equal(labels_by_time, unit_id, out=unit_mask_tmp))
        if in_unit.size <= 1:
            continue
        t = times_samples[in_unit]
        dt = np.diff(t)
        if dt.min() > radius_samples:
            continue
        discard = dedup_unit(t, dt, scores[in_unit], radius_samples)
        ndrop += discard.sum()
        labels_by_time[in_unit[discard]] = -1

    logger.dartsortdebug(f"drop {ndrop}/{len(sorting)} isi violator spikes")
    if tsort is not None:
        new_labels[tsort] = labels_by_time

    return sorting.ephemeral_replace(labels=new_labels)


def dedup_unit(
    t: np.ndarray, dt: np.ndarray, scores: np.ndarray, radius: int
) -> np.ndarray:
    """Deduplicate a single unit's spike train."""
    discard = np.zeros(t.shape, dtype=bool)
    _dedup_unit_loop(dt, scores, radius, discard)
    return discard


@numba.njit(nogil=True)
def _dedup_unit_loop(
    dt: np.ndarray, scores: np.ndarray, radius: int, discard: np.ndarray
):
    n = scores.shape[0]
    i0 = 0
    while i0 < n - 1:
        if dt[i0] > radius:
            i0 += 1
            continue

        i1 = i0 + 1
        while i1 < n - 1 and dt[i1] <= radius:
            i1 += 1

        # now, i0:i1 + 1 is a slice of violators
        # discard until all valid
        score_slice = scores[i0 : i1 + 1]
        dt_slice = dt[i0:i1]
        order = np.argsort(score_slice)
        # putting the -1 there ensures at least one spike is kept
        # that case is only relevant when 0 in dt_slice
        for oo in order[:-1]:
            # discard spike i0 + oo
            ii = i0 + oo
            discard[ii] = True

            # figure out isi after bridging the gap, handle edges
            bridge_isi = 0.0
            if ii < n - 1:
                bridge_isi += dt[ii]
            else:
                bridge_isi = np.inf
            if ii > 0:
                bridge_isi += dt[ii - 1]
            else:
                bridge_isi = np.inf

            # update dt with bridge isi
            if ii < n - 1:
                dt[ii] = bridge_isi
            if ii > 0:
                dt[ii - 1] = bridge_isi

            # check if done
            if dt_slice.min() > radius:
                break

        i0 = i1
