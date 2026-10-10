"""Agglomeration of clusters to fix up GMM oversplits."""

from threading import local
from typing import cast

import numba
import numpy as np
import torch
from spikeinterface.core import BaseRecording

try:
    import KDEpy

    HAVE_KDEPY = True
except ImportError:
    HAVE_KDEPY = False

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

    The algorithm is like this (violation_linkage). It's a greedy procedure building
    up a custom linkage as in hierarchical clustering based on chance-corrected refractory
    violations and template distances both.
     - For groups A and B, let d be the distance between their merged templates.
       Let E, O be their expected and observed violation counts summed over member
       pairs with E under the jitter model of violation_statistics.
     - A pair of groups is a candidate for merging if either of:
        - d < glom_force_merge_template_distance
        - d < merge_distance_threshold and E >= glom_min_violation_evidence.
     - Additionally, each member unit u of A must have, against the whole of B,
       at least one of:
         - E(u, B) >= glom_min_violation_evidence
         - Distance from u's template to B's < glom_force_merge_template_distance
     - Groups are merged greedily in chance-corrected violation ratio (O/E) order
       up to glom_violation_threshold
     - Optionally, groups are then split so that no pair of units within a
       group has violation ratio >= glom_veto_threshold.
     - If glom_violation_threshold is None, groups are the connected components
       of the allowed-pair graph by template distance according to the linkage parameter.

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
    unit_snrs = tdist.template_data.snrs_by_channel().max(1)

    if refinement_cfg is None:
        _, merge_mapping = hierarchical_cluster(
            None,
            distances,
            linkage_method=template_merge_cfg.linkage,
            threshold=template_merge_cfg.merge_distance_threshold,
        )
        violation = None
    else:
        merge_mapping, violation = _agglomerate_violation_merge(
            sorting=sorting,
            distances=distances,
            tdist=tdist,
            unit_snrs=unit_snrs,
            refinement_cfg=refinement_cfg,
            template_merge_cfg=template_merge_cfg,
        )

    agg_sorting = apply_reclustering(
        sorting=sorting,
        merge_mapping=merge_mapping,
        shifts=tdist.shifts,
        unit_snrs=unit_snrs,
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
    )


def _agglomerate_violation_merge(
    *,
    sorting: DARTsortSorting,
    distances: np.ndarray,
    tdist: "TemplateDistanceResult",
    unit_snrs: np.ndarray,
    refinement_cfg: RefinementConfig,
    template_merge_cfg: TemplateMergeConfig,
) -> tuple[np.ndarray, ViolationInfo | None]:
    # early out: no violation stuff. just distance mask connected components.
    if refinement_cfg.glom_violation_threshold is None:
        mask = distances < template_merge_cfg.merge_distance_threshold
        mask |= distances < refinement_cfg.glom_force_merge_template_distance
        np.fill_diagonal(mask, True)
        _, merge_mapping = hierarchical_cluster(
            None,
            np.logical_not(mask).astype(np.float32),
            linkage_method=template_merge_cfg.linkage,
            threshold=0.5,
        )
        return merge_mapping, None

    violation = violation_statistics(
        sorting,
        censor_ms=refinement_cfg.censor_ms,
        viol_ms=refinement_cfg.glom_violation_ms,
        jitter_ms=refinement_cfg.glom_jitter_ms,
    )
    assert sorting.labels is not None
    spike_counts = np.bincount(
        sorting.labels[sorting.labels >= 0], minlength=distances.shape[0]
    )
    merge_mapping = violation_linkage(
        distances=distances,
        violation=violation,
        tdist=tdist,
        spike_counts=spike_counts,
        unit_snrs=unit_snrs,
        template_merge_cfg=template_merge_cfg,
        force_distance=refinement_cfg.glom_force_merge_template_distance,
        min_evidence=refinement_cfg.glom_min_violation_evidence,
        violation_threshold=refinement_cfg.glom_violation_threshold,
    )

    # last thing: optionally, be extremely finnicky about merging into violated groups
    # i am not sure if this will be a good idea or not; it may cost too many merges
    # to be worthwhile.
    if refinement_cfg.glom_veto_threshold is not None:
        veto_cost = violation.jitter_viol_ratio(
            refinement_cfg.glom_min_violation_evidence, fill_value=0.0
        )
        veto_cost = np.minimum(veto_cost, veto_cost.T)
        np.fill_diagonal(veto_cost, 0.0)
        _, veto_mapping = hierarchical_cluster(
            None,
            veto_cost,
            linkage_method="complete",
            threshold=refinement_cfg.glom_veto_threshold,
        )
        merge_mapping = meet(merge_mapping, veto_mapping)

    return merge_mapping, violation


def violation_linkage(
    *,
    distances: np.ndarray,
    violation: ViolationInfo,
    tdist: "TemplateDistanceResult",
    spike_counts: np.ndarray,
    unit_snrs: np.ndarray,
    template_merge_cfg: TemplateMergeConfig,
    force_distance: float,
    min_evidence: float,
    violation_threshold: float,
) -> np.ndarray:
    """Greedy agglomeration on chance-corrected violations and merged templates

    Groups' merged templates are the spike-count-weighted average of their
    members' templates, aligned to the group's highest-SNR member as in
    apply_reclustering.
     - For groups A and B, let d be the distance between their merged templates.
       Let E, O be their expected and observed violation counts summed over member
       pairs.
     - A pair of groups is a candidate for merging if either of:
        - d < force_distance
        - d < merge_distance_threshold and E >= min_evidence.
     - Additionally, each member unit u of A must have, against the whole of B,
       at least one of:
         - E(u, B) >= min_evidence
         - Distance from u's template to B's < force_distance
     - Groups are merged greedily in O/E order (0 if E < min_evidence) up to
       violation_threshold

    Returns the merge mapping (group label of each unit).
    """
    n = distances.shape[0]
    assert violation.jitter_counts is not None
    assert distances.shape == violation.viol_counts.shape == (n, n)
    assert np.array_equal(distances, distances.T)
    assert not np.isnan(distances).any()
    assert (np.diagonal(distances) == 0).all()
    assert spike_counts.shape == unit_snrs.shape == (n,)
    assert (spike_counts > 0).all()
    assert np.isfinite(unit_snrs).all()
    assert tdist.radial_counts is not None

    templates = tdist.template_data.templates
    assert templates.shape[0] == n
    assert tdist.temporal_components.shape[1] == templates.shape[1]
    assert tdist.radial_counts.shape == (n, templates.shape[2])
    assert tdist.shifts.shape == (n, n)
    unit_spatial_sing = np.einsum("rt,ntc->nrc", tdist.temporal_components, templates)
    unit_weights = radial_weights(tdist.radial_counts)

    # groups are identified by their lowest unit id. every unit starts as its own
    # group, so group stats start as unit stats.
    members = {u: [u] for u in range(n)}
    group_spatial_sing = unit_spatial_sing.copy()
    group_weights = unit_weights.copy()
    group_dist = distances.copy()
    group_obs_viol = violation.viol_counts.astype(np.float64)
    group_exp_viol = violation.jitter_counts.astype(np.float64)
    # rows: units, cols: current groups (members.keys()). a unit's entry for its
    # own group is stale; it's never read.
    unit_to_group_dist = distances.copy()
    unit_to_group_exp_viol = group_exp_viol.copy()

    while True:
        group_ids = np.array(sorted(members))
        dist = group_dist[group_ids][:, group_ids]
        exp_viol = group_exp_viol[group_ids][:, group_ids]
        obs_viol = group_obs_viol[group_ids][:, group_ids]

        # O/E for candidate pairs (0 if E < min_evidence), inf for the rest
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = obs_viol / exp_viol
        ratio[exp_viol < min_evidence] = 0.0
        allow = dist < force_distance
        allow |= np.logical_and(
            dist < template_merge_cfg.merge_distance_threshold,
            exp_viol >= min_evidence,
        )
        ratio[np.logical_not(allow, out=allow)] = np.inf
        del allow

        # find lowest-ratio group pair below violation_threshold which passes the
        # per-member evidence check (or dist forcing)
        ii, jj = np.nonzero(np.triu(ratio < violation_threshold, k=1))
        ratio_pair_order = np.argsort(ratio[ii, jj], kind="stable")
        for i, j in zip(ii[ratio_pair_order], jj[ratio_pair_order], strict=True):
            a = group_ids[i]
            b = group_ids[j]
            assert a < b
            if _all_group_members_have_evidence(
                a,
                b,
                members=members,
                unit_to_group_exp_viol=unit_to_group_exp_viol,
                unit_to_group_dist=unit_to_group_dist,
                min_evidence=min_evidence,
                force_distance=force_distance,
            ):
                break
        else:  # no pair passed: done
            break

        # merge B into A
        # combine violation stats over the groups
        members[a] += members.pop(b)
        group_obs_viol[a] += group_obs_viol[b]
        group_obs_viol[:, a] += group_obs_viol[:, b]
        group_exp_viol[a] += group_exp_viol[b]
        group_exp_viol[:, a] += group_exp_viol[:, b]
        unit_to_group_exp_viol[:, a] += unit_to_group_exp_viol[:, b]

        # recompute A's template and distances to other groups and units
        group_spatial_sing[a], group_weights[a] = _merged_template(
            members[a],
            templates=templates,
            spike_counts=spike_counts,
            unit_snrs=unit_snrs,
            shifts=tdist.shifts,
            temporal_components=tdist.temporal_components,
            radial_counts=tdist.radial_counts,
        )
        other_groups = [g for g in members if g != a]
        a_dist = template_distances_to(
            group_spatial_sing[a],
            group_weights[a],
            group_spatial_sing[other_groups],
            group_weights[other_groups],
            trimmed_tconv=tdist.trimmed_tconv,
            template_merge_cfg=template_merge_cfg,
        )
        group_dist[a, other_groups] = a_dist
        group_dist[other_groups, a] = a_dist
        nonmembers = np.setdiff1d(np.arange(n), members[a])
        unit_to_group_dist[nonmembers, a] = template_distances_to(
            group_spatial_sing[a],
            group_weights[a],
            unit_spatial_sing[nonmembers],
            unit_weights[nonmembers],
            trimmed_tconv=tdist.trimmed_tconv,
            template_merge_cfg=template_merge_cfg,
        )

    merge_mapping = np.full(n, -1, dtype=np.int64)
    for label, unit_ids in enumerate(members.values()):
        merge_mapping[unit_ids] = label
    assert np.min(merge_mapping, initial=0) >= 0
    return merge_mapping


def _all_group_members_have_evidence(
    a: int,
    b: int,
    *,
    members: dict[int, list[int]],
    unit_to_group_exp_viol: np.ndarray,
    unit_to_group_dist: np.ndarray,
    min_evidence: float,
    force_distance: float,
) -> bool:
    """Each unit u in A has E(u, B) >= min_evidence or d(u, B) < force_distance; and vice versa"""
    return all(
        np.logical_or(
            unit_to_group_exp_viol[members[src], dst] >= min_evidence,
            unit_to_group_dist[members[src], dst] < force_distance,
        ).all()
        for src, dst in ((a, b), (b, a))
    )


def _merged_template(
    unit_ids,
    *,
    templates: np.ndarray,
    spike_counts: np.ndarray,
    unit_snrs: np.ndarray,
    shifts: np.ndarray,
    temporal_components: np.ndarray,
    radial_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Shift, weighted sum in temporal mask, project onto known shared basis"""
    unit_ids = np.sort(unit_ids)
    ref = unit_ids[np.argmax(unit_snrs[unit_ids])]
    n_samples = templates.shape[1]
    total = np.zeros(templates.shape[1:])
    coverage = np.zeros(n_samples)
    for u in unit_ids:
        s = shifts[ref, u]
        assert abs(s) < n_samples
        src = slice(max(0, -s), n_samples - max(0, s))
        dst = slice(max(0, s), n_samples - max(0, -s))
        total[dst] += spike_counts[u] * templates[u, src]
        coverage[dst] += spike_counts[u]
    assert (coverage > 0).all()
    template = np.divide(total, coverage[:, None], out=total)
    spatial = temporal_components @ template
    weights = radial_weights(radial_counts[unit_ids].sum(0))
    return spatial, weights


@databag
class TemplateDistanceResult:
    distances: np.ndarray
    shifts: np.ndarray
    """shifts[i, j] is like trough[i] - trough[j]"""
    r2: np.ndarray
    template_data: TemplateData
    temporal_components: np.ndarray
    trimmed_tconv: torch.Tensor
    radial_counts: np.ndarray | None
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
    trimmed_tconv = tconv[:, :, center - max_shift : center + 1 + max_shift]
    trimmed_tconv = trimmed_tconv.contiguous()

    radial_counts = spatial_iou = None
    if template_merge_cfg.distance_kind == "scaled_normeuc":
        best_conv, best_lag = best_shared_pconv(trimmed_tconv, spatial_sing)
        dist = scaled_normeuc_from_dots(
            best_conv,
            scale_var=template_merge_cfg.amplitude_scaling_variance,
            scale_boundary=template_merge_cfg.amplitude_scaling_boundary,
        )
    elif template_merge_cfg.distance_kind == "weighted_scaled_normeuc":
        assert sorting is not None
        assert motion is not None
        radial_counts = radial_spike_counts(
            sorting=sorting,
            motion=motion,
            radius=template_merge_cfg.weighted_dist_radius,
        )
        spatial_weights = radial_weights(radial_counts)
        assert np.isfinite(spatial_weights).all()
        dist, best_lag, iou = weighted_best_lagged_scaled_normeuc_dist(
            tconv=trimmed_tconv,
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
        shifts=best_lag.T.numpy(force=True),
        r2=cast(np.ndarray, sbt.r2),
        template_data=template_data,
        temporal_components=sbt.temporal_components,
        trimmed_tconv=trimmed_tconv,
        radial_counts=radial_counts,
        spatial_iou=spatial_iou,
    )


def template_distances_to(
    spatial_sing_a: np.ndarray,
    weights_a: np.ndarray,
    spatial_sing_b: np.ndarray,
    weights_b: np.ndarray,
    *,
    trimmed_tconv: torch.Tensor,
    template_merge_cfg: TemplateMergeConfig,
) -> np.ndarray:
    assert template_merge_cfg.distance_kind == "weighted_scaled_normeuc"
    n = spatial_sing_b.shape[0]
    assert spatial_sing_b.shape[1:] == spatial_sing_a.shape
    assert weights_a.shape == spatial_sing_a.shape[1:]
    assert weights_b.shape == (n, *weights_a.shape)
    assert weights_a.max() > 0 and (weights_b.max(1) > 0).all()

    dist = np.full(n, np.inf)
    iou = np.minimum(weights_a, weights_b).sum(1) / np.maximum(
        weights_a, weights_b
    ).sum(1)
    (ix,) = np.nonzero(iou >= template_merge_cfg.weighted_dist_min_iou)
    if not ix.size:
        return dist

    spatial_sing = np.concatenate([spatial_sing_a[None], spatial_sing_b[ix]])
    weights = np.concatenate([weights_a[None], weights_b[ix]])
    jj = torch.arange(1, ix.size + 1)
    d, _, pair_iou = weighted_best_lagged_scaled_normeuc_dist(
        tconv=trimmed_tconv,
        spatial_sing=torch.asarray(spatial_sing).to(trimmed_tconv),
        weights=torch.asarray(weights).to(trimmed_tconv),
        scale_var=template_merge_cfg.amplitude_scaling_variance,
        scale_boundary=template_merge_cfg.amplitude_scaling_boundary,
        pairs=(torch.zeros_like(jj), jj),
    )
    d.masked_fill_(pair_iou < template_merge_cfg.weighted_dist_min_iou, torch.inf)
    dist[ix] = d.numpy(force=True)
    return dist


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

    assert HAVE_KDEPY or not bimodality

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

    from KDEpy import FFTKDE

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


def radial_spike_counts(sorting: DARTsortSorting, motion: MotionInfo, radius: float):
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
    radial_counts = np.zeros(counts.shape)
    for uu in range(counts.shape[0]):
        row = counts[uu]
        ii = np.flatnonzero(row)
        for channel, value in zip(ii, row[ii], strict=True):
            cixs = ci[channel]
            cixs = cixs[cixs < motion.rgeom.shape[0]]
            radial_counts[uu, cixs] += value

    return radial_counts


def radial_weights(radial_counts: np.ndarray) -> np.ndarray:
    denom = radial_counts.max(axis=-1, keepdims=True).clip(min=1e-10)
    return radial_counts / denom


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

    for s in numba.prange(cand.shape[0]):
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

    for s in numba.prange(cand.shape[0]):
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

    for s in numba.prange(cand.shape[0]):
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
    score_by=("gmm_log_liks", "scores"),
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
    score_by=("gmm_log_liks", "scores"),
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
