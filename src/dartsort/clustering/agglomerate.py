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
    closest_registered_channels,
    linkage_mask,
    recluster,
    reorder_by_depth,
    sparsify_labels,
)

logger = get_logger(__name__)


@databag
class Agglomeration:
    agglomerated_sorting: DARTsortSorting
    merge_mapping: np.ndarray
    distances: np.ndarray | None
    shifts: np.ndarray | None
    firing_corr: np.ndarray | None


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
    show_progress: bool = True,
) -> Agglomeration:
    computation_cfg = ensure_computation_config(computation_cfg)

    if template_merge_cfg is None:
        assert refinement_cfg is not None
        template_merge_cfg = refinement_cfg.template_merge_cfg

    if template_data is None:
        did_flatten = True
        sorting = sorting.flatten(include_gmm_properties=True)
    else:
        did_flatten = False

    if template_merge_cfg is not None:
        tdist = template_distances(
            sorting=sorting,
            recording=recording,
            motion=motion,
            template_data=template_data,
            waveform_cfg=waveform_cfg,
            template_merge_cfg=template_merge_cfg,
            computation_cfg=computation_cfg,
        )
    else:
        tdist = None

    # if not doing any QDA, be done now.
    if refinement_cfg is None or not refinement_cfg.qda_threshold:
        if tdist is not None:
            assert template_merge_cfg is not None
            agg_sorting, new_ids = recluster(
                sorting=sorting,
                unit_ids=tdist.template_data.unit_ids,
                dists=tdist.distances,
                shifts=tdist.shifts,
                unit_snrs=tdist.template_data.snrs_by_channel().max(1),
                threshold=template_merge_cfg.merge_distance_threshold,
                link=template_merge_cfg.linkage,
            )
        elif not did_flatten:
            agg_sorting = sorting.flatten(include_gmm_properties=True)
            new_ids = None
        else:
            agg_sorting = sorting
            new_ids = None

        if refinement_cfg is not None:
            agg_sorting = deduplicate_spikes(agg_sorting, refinement_cfg.dedup_ms)

        agg_sorting, reorder = reorder_by_depth(agg_sorting, motion=motion)
        if new_ids is None:
            new_ids = reorder
        else:
            new_ids = reorder[np.unique(new_ids, return_inverse=True)[1]]

        return Agglomeration(
            agglomerated_sorting=agg_sorting,
            merge_mapping=new_ids,
            distances=None if tdist is None else tdist.distances,
            shifts=None if tdist is None else tdist.shifts,
            firing_corr=None,
        )

    # tdist tells us the possible merges
    assert tdist is not None
    assert template_merge_cfg is not None
    distance_mask = linkage_mask(
        tdist.distances,
        linkage_method=template_merge_cfg.linkage,
        threshold=template_merge_cfg.merge_distance_threshold,
    )

    # only QDA within negatively correlated firing enemies
    if refinement_cfg.glom_max_firing_corr is not None:
        fcorr = firing_corr(
            sorting,
            dt=refinement_cfg.glom_firing_corr_dt,
            method=refinement_cfg.glom_firing_corr_method,
        )
        _oldsum = distance_mask[np.triu_indices_from(distance_mask)].sum()
        fcorr_mask = fcorr <= refinement_cfg.glom_max_firing_corr
        mask = np.logical_and(distance_mask, fcorr_mask)
        np.fill_diagonal(mask, True)
        _newsum = mask[np.triu_indices_from(mask)].sum()
        logger.dartsortdebug(
            f"Firing corr dropped QDA candidate count from {_oldsum} -> {_newsum}."
        )
    else:
        fcorr = fcorr_mask = None
        mask = distance_mask

    # restrict mask by overlap criteria
    qda_res = qda(
        mask=mask,
        sorting=sorting,
        min_iou=refinement_cfg.qda_min_iou,
        min_cov=refinement_cfg.qda_min_coverage,
        show_progress=show_progress,
        computation_cfg=computation_cfg,
    )

    coverage_mask = np.logical_and(
        qda_res.coverage >= refinement_cfg.qda_min_coverage,
        qda_res.iou >= refinement_cfg.qda_min_iou,
    )
    qda_mask_uni = np.logical_and(
        coverage_mask,
        qda_res.score >= refinement_cfg.qda_uni_score,
    )
    qda_mask_bi = np.all(
        [
            coverage_mask,
            qda_res.score >= refinement_cfg.qda_threshold,
            qda_res.min_ratio >= refinement_cfg.qda_min_ratio,
        ],
        axis=0,
    )
    qda_mask = np.logical_or(qda_mask_uni, qda_mask_bi)
    assert np.all(qda_mask <= mask)
    if fcorr_mask is not None:
        assert np.all(np.logical_and(qda_mask, fcorr_mask) <= mask)

    simg = refinement_cfg.spikeinterface_merge_preset
    if simg is not None and simg != "none":
        pair_mask = tdist.distances < refinement_cfg.spikeinterface_merge_max_distance
        if refinement_cfg.spikeinterface_merge_min_coentropy is not None:
            cmask, _ = coentropy_merge_mask(
                sorting=sorting,
                min_coentropy=refinement_cfg.spikeinterface_merge_min_coentropy,
                coverage_threshold=refinement_cfg.spikeinterface_merge_coent_coverage,
                iou_threshold=refinement_cfg.spikeinterface_merge_coent_iou,
            )
            pair_mask = np.logical_or(cmask, pair_mask)

        si_mask = spikeinterface_merge_mask(
            recording=recording,
            sorting=sorting,
            preset=refinement_cfg.spikeinterface_merge_preset,
            censor_ms=refinement_cfg.censor_ms,
            template_data=tdist.template_data,
            pair_mask=pair_mask,
        )
    else:
        si_mask = None

    # force merges for very close neighbors
    force_mask = linkage_mask(
        tdist.distances,
        linkage_method=template_merge_cfg.linkage,
        threshold=refinement_cfg.qda_force_merge_for_temp_dist_below,
    )

    # extract final mask
    final_mask = np.logical_or(qda_mask, force_mask)
    if si_mask is not None:
        final_mask = np.logical_or(final_mask, si_mask)
    np.fill_diagonal(final_mask, True)
    final_mask_as_distance = np.logical_not(final_mask).astype(np.float32)

    agg_sorting, new_ids = recluster(
        sorting=sorting,
        unit_ids=tdist.template_data.unit_ids,
        dists=final_mask_as_distance,
        shifts=tdist.shifts,
        unit_snrs=tdist.template_data.snrs_by_channel().max(1),
        threshold=0.5,  # binary input here
        link=template_merge_cfg.linkage,
    )

    agg_sorting = combine_gmm_scores(agg_sorting, new_ids=new_ids)

    agg_sorting = deduplicate_spikes(agg_sorting, refinement_cfg.dedup_ms)

    agg_sorting, reorder = reorder_by_depth(agg_sorting, motion=motion)
    new_ids = reorder[np.unique(new_ids, return_inverse=True)[1]]

    return Agglomeration(
        agglomerated_sorting=agg_sorting,
        merge_mapping=new_ids,
        distances=tdist.distances,
        shifts=tdist.shifts,
        firing_corr=fcorr,
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


def spikeinterface_merge_mask(
    *,
    recording: BaseRecording,
    sorting: DARTsortSorting,
    preset: str | None,
    censor_ms: float = 0.0,
    template_data: TemplateData,
    pair_mask: np.ndarray,
    min_count: int = 100,
):
    from spikeinterface.curation.auto_merge import compute_merge_unit_groups
    from spikeinterface.postprocessing import ComputeTemplateSimilarity

    # censor first
    if censor_ms:
        sorting = deduplicate_spikes(sorting, censor_ms)

    # analyzer (lightweight one)
    analyzer = sorting.to_sorting_analyzer(
        recording=recording,
        template_data=template_data,
        compute_extensions=None,
        compute_extensions_if_templates=None,
        estimate_si_sparsity=False,
        compute_template_similarity=False,
    )

    # register the mask as the template similarity extension
    tsim_ext = ComputeTemplateSimilarity(analyzer)
    tsim_ext.data = {"similarity": pair_mask.astype(np.float32)}
    tsim_ext.params = {"method": "dartsort"}
    tsim_ext.run_info = {"run_completed": True}
    analyzer.extensions["template_similarity"] = tsim_ext

    # handle custom presets
    if preset == "dartsort_slay_xc":
        steps = [
            "num_spikes",
            "remove_contaminated",
            "unit_locations",
            "template_similarity",
            "slay_score",
            "cross_contamination",
            "quality_score",
        ]
        preset = None
        analyzer.compute_one_extension("correlograms")
    elif preset == "dartsort_slay_ccg":
        steps = [
            "num_spikes",
            "remove_contaminated",
            "unit_locations",
            "template_similarity",
            "correlogram",
            "slay_score",
            "quality_score",
        ]
        preset = None
        analyzer.compute_one_extension("correlograms")
    elif preset == "dartsort_slay_xc_ccg":
        steps = [
            "num_spikes",
            "remove_contaminated",
            "unit_locations",
            "template_similarity",
            "correlogram",
            "cross_contamination",
            "slay_score",
            "quality_score",
        ]
        preset = None
        analyzer.compute_one_extension("correlograms")
    else:
        assert preset is not None
        steps = None

    # make parameters aware of censorship and other params
    my_step_params = {
        "num_spikes": {"min_spikes": min_count},
        "remove_contaminated": {"censored_period_ms": censor_ms},
        "template_similarity": {"similarity_method": "dartsort"},
        "correlogram": {"censor_correlograms_ms": censor_ms},
        "cross_contamination": {"censored_period_ms": censor_ms},
        "quality_score": {"censored_period_ms": censor_ms},
    }
    groups = compute_merge_unit_groups(
        preset=preset,
        steps=steps,
        sorting_analyzer=analyzer,
        steps_params=my_step_params,
        force_copy=False,
    )
    mask = np.zeros_like(pair_mask)
    for g in groups:
        g = np.array(g)
        mask[g[:, None], g[None, :]] = True
    return mask


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


def _qda_init(ctx):
    global _qda_context
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

    if iou < p.min_iou:
        return
    if cov < p.min_cov:
        return
    if ini.size + inj.size < p.min_count:
        return

    dll = _dll(inij, overlap, imask, jmask, p.log_liks)

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
        logger.dartsortdebug(f"KDEpy error: {str(e)}")
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

        for channel, value in zip(ii, vv):
            cixs = ci[channel]
            cixs = cixs[cixs < motion.rgeom.shape[0]]
            weights[uu, cixs] += value

    denom = weights.max(axis=1, keepdims=True).clip(min=1e-10)  # avoid div by 0
    weights /= denom
    return weights


def firing_corr(sorting: DARTsortSorting, dt: float, method="binsqrt"):
    if method != "binsqrt":
        assert False

    tsg = sorting.to_tsgroup()
    fr = tsg.count(bin_size=dt) / dt
    fr = np.sqrt(fr.values)

    return np.corrcoef(fr, rowvar=False)


def combine_gmm_scores(
    sorting: DARTsortSorting, new_ids: np.ndarray, old_prefix="gmm", new_prefix="merged"
) -> DARTsortSorting:
    """If new_ids merges units, return a sorting with merged likelihoods."""
    candidates = getattr(sorting, f"{old_prefix}_candidates", None)
    responsibilities = getattr(sorting, f"{old_prefix}_responsibilities", None)
    logliks = getattr(sorting, f"{old_prefix}_log_liks", None)

    assert not hasattr(sorting, f"{new_prefix}_responsibilities")

    havec = candidates is not None
    haver = responsibilities is not None
    havel = logliks is not None
    assert all([havec, haver, havel]) or not any([havec, havel, haver])
    if not havec:
        return sorting
    assert candidates is not None
    assert responsibilities is not None
    assert logliks is not None

    # check that new_ids is a merge
    assert (new_ids >= 0).all()
    unique_new_ids, new_id_counts = np.unique(new_ids, return_counts=True)
    assert unique_new_ids.shape[0] == unique_new_ids.max() + 1 <= new_ids.shape[0]
    if unique_new_ids.shape == new_ids.shape:
        return sorting.ephemeral_replace(
            **{
                f"{new_prefix}_candidates": candidates,
                f"{new_prefix}_responsibilities": responsibilities,
                f"{new_prefix}_log_liks": logliks,
            }
        )

    # check invariants at the top
    if responsibilities.shape[1] > 2:
        _maxdiff = np.diff(responsibilities[:, :-1], axis=1).max()
        assert _maxdiff <= 1e-3, _maxdiff
    assert np.greater_equal(np.isneginf(logliks[:, :-1]), candidates == -1).all()
    if sorting.labels is not None:
        not_noise = np.flatnonzero(sorting.labels >= 0)
        assert np.array_equal(
            sorting.labels[not_noise], new_ids[candidates[not_noise, 0]]
        )

    # two steps: first merge, then sort
    # merge candidates
    new_ids_ = np.pad(new_ids, [(0, 1)], constant_values=-1)
    orig_bye = candidates < 0
    nbye = orig_bye.sum()
    cand = np.where(orig_bye, new_ids.shape[0], candidates)
    cand = new_ids_[cand]
    assert (cand < 0).sum() >= nbye

    # deduplicate
    mergedr = responsibilities[:, : cand.shape[1]].copy()
    mergedl = logliks[:, : cand.shape[1]].copy()
    _combine_loop(cand, new_id_counts, mergedr, mergedl)

    # now re-sort
    order = np.argsort(-mergedl, axis=1, kind="stable")
    cand = np.take_along_axis(cand, axis=1, indices=order)
    mergedr = np.take_along_axis(mergedr, axis=1, indices=order)
    mergedl = np.take_along_axis(mergedl, axis=1, indices=order)
    mergedl = np.concatenate([mergedl, logliks[:, cand.shape[1] :]], axis=1)
    mergedr = np.concatenate([mergedr, responsibilities[:, cand.shape[1] :]], axis=1)

    # check invariants at the bottom
    if mergedr.shape[1] > 2:
        _maxdiff = np.diff(mergedr[:, : cand.shape[1]], axis=1).max()
        assert _maxdiff <= 1e-3, _maxdiff
    assert np.greater_equal(np.isneginf(mergedl[:, : cand.shape[1]]), cand == -1).all()
    assert (cand < 0).sum() >= nbye
    if sorting.labels is not None:
        changed = sorting.labels != cand[:, 0]
        changed = changed[sorting.labels >= 0]
        logger.dartsortdebug(
            f"Mixture component aggregation changes {100 * changed.mean():0.2f}"
            f"% of spike labels ({changed.sum().item()} spikes)."
        )
    return sorting.ephemeral_replace(
        labels=np.where(mergedl[:, 0] >= mergedl[:, -1], cand[:, 0], -1),
        **{
            f"{new_prefix}_candidates": cand,
            f"{new_prefix}_responsibilities": mergedr,
            f"{new_prefix}_log_liks": mergedl,
        },
    )


@numba.njit(parallel=True)
def _combine_loop(
    cand: np.ndarray,
    new_id_counts: np.ndarray,
    mergedr: np.ndarray,
    mergedl: np.ndarray,
):
    for s in numba.prange(cand.shape[0]):  # ty: ignore
        spike_cand = cand[s]
        for j in range(cand.shape[1] - 1):
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
            for i, k in enumerate(range(j + 1, cand.shape[1])):
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
        for j in range(cand.shape[1]):
            if 0 <= spike_cand[j] < new_id_counts.shape[0]:
                continue
            rsj = mergedr[s, j]
            if rsj == 0:
                continue
            mergedr[s, -1] += rsj
            mergedr[s, j] = 0.0


def deduplicate_spikes(
    sorting: DARTsortSorting,
    radius_ms: float = -1.0,
    score_by=("merged_log_liks", "gmm_log_liks", "scores"),
) -> DARTsortSorting:
    if radius_ms < 0 or sorting.labels is None:
        return sorting

    radius_samples = WaveformConfig.ms_to_samples(
        ms=radius_ms,
        sampling_frequency=sorting.sampling_frequency,
    )
    assert radius_samples >= 0

    new_labels = sorting.labels.copy()
    scores = None
    for sck in score_by:
        scores = getattr(sorting, sck, None)
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
    tsort = np.argsort(sorting.times_samples)
    new_labels = new_labels[tsort]
    times_samples = sorting.times_samples[tsort]
    scores = scores[tsort]

    unit_ids = np.unique(new_labels)
    unit_ids = unit_ids[unit_ids >= 0]
    ndrop = 0
    for unit_id in unit_ids:
        in_unit = np.flatnonzero(new_labels == unit_id)
        if in_unit.size <= 1:
            continue
        t = times_samples[in_unit]
        dt = np.diff(t)
        if dt.min() > radius_samples:
            continue
        discard = _dedup_unit(t, dt, scores[in_unit], radius_samples)
        ndrop += discard.sum()
        new_labels[in_unit[discard]] = -1

    logger.dartsortdebug(f"drop {ndrop}/{len(sorting)} isi violator spikes")
    new_labels = new_labels[np.argsort(tsort)]

    return sorting.ephemeral_replace(labels=new_labels)


def _dedup_unit(
    t: np.ndarray, dt: np.ndarray, scores: np.ndarray, radius: int
) -> np.ndarray:
    """Deduplicate a single unit's spike train."""
    discard = np.zeros(t.shape, dtype=bool)
    _dedup_unit_loop(dt, scores, radius, discard)
    return discard


@numba.njit
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


@databag
class CoentropyResult:
    coentropy: np.ndarray
    """KxK; reduction of entropy per cooccurrence due to merging pair"""

    cooccurrence: np.ndarray
    """KxK; number of times these units score the same spike"""

    rival_count: np.ndarray
    """KxK; number of times one unit scores a spike where the other is top"""

    occurrence: np.ndarray
    """K; number of times the unit appears in the candidates at all"""

    cov: np.ndarray
    """KxK; rival count / max pair count (rival diag)"""

    iou: np.ndarray
    """KxK; rival count over pair sum"""


def coentropy_merge_mask(
    sorting: DARTsortSorting,
    min_coentropy: float,
    coverage_threshold: float,
    iou_threshold: float,
    gmm_prefix=("merged", "gmm"),
) -> tuple[np.ndarray, CoentropyResult]:
    """
    Parameters
    ----------
    sorting : DARTsortSorting
    min_coentropy : float
        Must be met by pair for mask=True
    min_coverage : float
        Pairs such that at least one unit in each pair has
        rival_count/count > mincov are allowed
    iou_threshold: float
        Pairs with rival iou > iouthresh are allowed
    """
    c = coentropy(sorting, gmm_prefix=gmm_prefix)
    assert c is not None

    mask = np.logical_or(c.cov >= coverage_threshold, c.iou >= iou_threshold)
    mask = np.logical_and(c.coentropy >= min_coentropy, mask)
    np.fill_diagonal(mask, True)
    return mask, c


def coentropy(
    sorting: DARTsortSorting,
    gmm_prefix=("merged", "gmm"),
) -> CoentropyResult | None:
    """Calculate entropy reduction due to merging pairs."""
    for k in gmm_prefix:
        cands = getattr(sorting, f"{k}_candidates", None)
        resps = getattr(sorting, f"{k}_responsibilities", None)
        if cands is not None:
            assert resps is not None
            break
    else:
        return None

    k = sorting.n_units
    resps = resps[:, : cands.shape[1]].astype(np.float64)
    coentropy = np.zeros((k, k))
    cooccurrence = np.zeros((k, k), dtype=np.int64)
    rival_count = np.zeros((k, k), dtype=np.int64)
    occurrence = np.zeros((k,), dtype=np.int64)
    _calc_coentropy(coentropy, cooccurrence, rival_count, occurrence, cands, resps)
    rival_count += rival_count.T
    cdiag = np.diagonal(rival_count)
    assert (cdiag % 2 == 0).all()
    np.fill_diagonal(rival_count, cdiag // 2)
    coentropy += coentropy.T
    cooccurrence += cooccurrence.T

    # rival count diagonal is just unit top count (not exactly label count,
    # since it doesn't account for noise assignments)
    counts = np.diagonal(rival_count)
    counts = np.maximum(counts, 1)

    cov = rival_count / counts
    cov = np.minimum(cov, cov.T)

    # this is a disjoint union, since it's the top-label count
    union = counts[:, None] + counts[None, :]
    iou = rival_count / union

    return CoentropyResult(
        coentropy=coentropy,
        cooccurrence=cooccurrence,
        rival_count=rival_count,
        occurrence=occurrence,
        cov=cov,
        iou=iou,
    )


@numba.njit(parallel=True)
def _calc_coentropy(
    coentropy: np.ndarray,
    cooccurrence: np.ndarray,
    rival_count: np.ndarray,
    occurrence: np.ndarray,
    cands: np.ndarray,
    resps: np.ndarray,
):
    for i in numba.prange(cands.shape[0]):  # ty: ignore
        u = cands[i]
        q = resps[i]
        log_q = np.log(q)
        np.nan_to_num(log_q, copy=False, neginf=0.0)
        dh = q * log_q

        ui0 = u[0]
        qi0 = q[0]
        dhi0 = dh[0]

        occurrence[ui0] += 1
        rival_count[ui0, ui0] += 1

        for j in range(1, cands.shape[1]):
            uj = u[j]
            if uj < 0:
                break

            ii = min(ui0, uj)
            jj = max(ui0, uj)

            occurrence[uj] += 1
            rival_count[ui0, uj] += 1

            cij = cooccurrence[ii, jj] + 1
            cooccurrence[ii, jj] = cij

            # change in entropy due to merging uj, uk:
            # subtract their current contribution, add the new contribution
            # we want reduction of entropy, so this is the negative of that!
            qij = q[j] + qi0
            dhij = dh[j] + dhi0
            if qij > 0:
                dhij -= qij * np.log(qij)

            # Welford mean of -dh
            cur_coent = coentropy[ii, jj]
            coentropy[ii, jj] = cur_coent + (-dhij - cur_coent) / cij
