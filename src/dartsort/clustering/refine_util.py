from threading import local
from typing import cast

import numba
import numpy as np
import torch
from KDEpy import FFTKDE
from KDEpy.bw_selection import improved_sheather_jones
from scipy.spatial import KDTree
from scipy.stats import norm
from spikeinterface.core.baserecording import BaseRecording

from dartsort.clustering.mixture import Scores

from ..util import data_util, spiketorch
from ..util.internal_config import ComputationConfig, RefinementConfig
from ..util.logging_util import DARTSORTDEBUG, DARTSORTVERBOSE, get_logger, progbar
from ..util.motion import MotionInfo
from ..util.multiprocessing_util import pool_from_cfg
from ..util.py_util import databag
from .cluster_util import hierarchical_cluster, reorder_by_depth
from .clustering_features import StableWaveformFeatures

logger = get_logger(__name__)


@databag
class PCMergeResult:
    sorting: data_util.DARTsortSorting
    means: torch.Tensor | None = None
    counts: torch.Tensor | None = None
    dists: torch.Tensor | None = None
    merge_ids: np.ndarray | None = None
    x: torch.Tensor | None = None
    xlabels: torch.Tensor | None = None


def pc_merge(
    *,
    sorting: data_util.DARTsortSorting,
    stable_features: StableWaveformFeatures,
    refinement_cfg: RefinementConfig,
    motion: MotionInfo,
    computation_cfg: ComputationConfig | None = None,
    debug: bool = False,
) -> PCMergeResult:
    assert refinement_cfg.refinement_strategy == "pcmerge"
    if not refinement_cfg.pc_merge_threshold:
        return PCMergeResult(sorting=sorting)

    # remove blank labels just in case
    sorting = sorting.flatten()
    assert sorting.labels is not None
    nu0 = sorting.labels.max() + 1
    if not nu0:
        return PCMergeResult(sorting=sorting)

    # subset the sorting to count per unit
    subset_sorting = data_util.subsample_to_max_count(
        sorting, max_spikes=refinement_cfg.pc_merge_spikes_per_unit
    )
    assert subset_sorting.labels is not None

    # make stable features, no need for core features though.
    kept = np.flatnonzero(subset_sorting.labels >= 0)
    x = stable_features.features[kept]
    x = x[:, : refinement_cfg.pc_merge_rank]
    xlabels = torch.from_numpy(subset_sorting.labels[kept]).to(x.device)
    n_reg_chans = motion.rgeom.shape[0]
    means, counts = spiketorch.average_by_label(
        x, xlabels, stable_features.channels[kept], n_reg_chans
    )

    # compute distances
    if refinement_cfg.pc_merge_metric == "cosine":
        dists = spiketorch.cosine_distance(means)
    elif refinement_cfg.pc_merge_metric == "maxz":
        x = x.square_()
        meansq, _ = spiketorch.average_by_label(
            x, xlabels, stable_features.channels, n_reg_chans
        )
        stddev = meansq.sub_(means.square()).sqrt_()
        stddev = stddev.clamp_(min=torch.finfo(stddev.dtype).tiny)
        stderr = stddev.div_(counts.sqrt()[:, None])
        dists = spiketorch.maxz_distance(
            means, stderr, counts, min_iou=refinement_cfg.pc_merge_min_iou
        )
    elif refinement_cfg.pc_merge_metric.endswith("normeuc"):
        dists = spiketorch.weighted_normeuc_distance(
            means, counts, min_iou=refinement_cfg.pc_merge_min_iou
        )
    elif refinement_cfg.pc_merge_metric == "normsup":
        dists = spiketorch.weighted_normsup_distance(
            means, counts, min_iou=refinement_cfg.pc_merge_min_iou
        )
    elif refinement_cfg.pc_merge_metric == "euclidean":
        means = means.reshape(len(means), -1)
        dists = torch.cdist(means, means).numpy(force=True)
    else:
        raise ValueError(f"Have not implemented {refinement_cfg.pc_merge_metric=}.")

    # linkage
    labels, ids = hierarchical_cluster(
        labels=sorting.labels,
        distances=np.asarray(dists),
        linkage_method=refinement_cfg.pc_merge_linkage,
        threshold=refinement_cfg.pc_merge_threshold,
    )
    assert labels is not None
    labels = np.atleast_1d(labels)
    k = ids.max() + 1
    ul, _, _ = data_util.pos_int_unique_and_counts(labels)
    assert np.array_equal(ul, np.unique(ids))
    assert ul.shape == (k,)
    assert k == ul.max() + 1
    logger.dartsortdebug(f"pc_merge: Unit count {nu0}->{k}.")

    sorting = sorting.ephemeral_replace(labels=labels)
    if debug:
        return PCMergeResult(
            sorting=sorting,
            means=means,
            counts=counts,
            merge_ids=ids,
            x=x,
            xlabels=xlabels,
            dists=torch.asarray(dists),
        )

    xlabels = torch.from_numpy(labels[kept]).to(x.device)
    means, counts = spiketorch.average_by_label(
        x, xlabels, stable_features.channels[kept], n_reg_chans
    )
    sf = means.square_().sum(dim=1).sqrt_() * counts.sqrt()
    sf = sf.numpy(force=True)
    sorting, _new_ids = reorder_by_depth(
        sorting, motion=motion, spatial_footprints=sf, geom=motion.rgeom
    )
    return PCMergeResult(sorting=sorting)


@databag
class CCErrorResult:
    sorting: data_util.DARTsortSorting

    kept: np.ndarray | None = None
    """Which of the input units survived, indexed before the output was flattened."""

    errors: np.ndarray | None = None
    flag_scores: "CCFlagScores | None" = None


def flag_possible_cc_error_spikes(
    sorting: data_util.DARTsortSorting,
    temporal_radius_samples: int,
    radius_um: float,
    spatial_dedup_radius_um: float | None,
    dedup_temporal_radius_samples: int | None = None,
    amplitudes_dataset_name: str = "denoised_ptp_amplitudes",
    time_shifts_dataset_name: str = "time_shifts",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # both zones describe detection-time events, before realignment moved the times
    times = sorting.times_samples
    if sorting.has_dataset(time_shifts_dataset_name):
        times = times - np.asarray(getattr(sorting, time_shifts_dataset_name))
    channels = sorting.channels
    amps = getattr(sorting, amplitudes_dataset_name)
    xy = sorting.geom[channels]
    n = len(times)
    if dedup_temporal_radius_samples is None:
        dedup_temporal_radius_samples = temporal_radius_samples
    assert temporal_radius_samples > 0
    assert radius_um > 0
    assert dedup_temporal_radius_samples > 0
    assert amps.shape == (n,)
    assert xy.shape == (n, 2)

    assert dedup_temporal_radius_samples <= temporal_radius_samples
    if spatial_dedup_radius_um:
        assert spatial_dedup_radius_um <= radius_um

    # cylinders! spatial l2, max with time
    kdt = KDTree(np.c_[times / temporal_radius_samples, xy / radius_um])
    sdm = kdt.sparse_distance_matrix(
        kdt, max_distance=1.0, p=np.inf, output_type="ndarray"
    )
    ii = sdm["i"]
    jj = sdm["j"]

    dt = np.abs(times[ii] - times[jj])
    dxy = np.linalg.norm(xy[ii] - xy[jj], axis=1)

    in_candidate = (dt <= temporal_radius_samples) & (dxy <= radius_um)
    if spatial_dedup_radius_um:
        in_dedup = (dt <= dedup_temporal_radius_samples) & (
            dxy <= spatial_dedup_radius_um
        )
    else:
        in_dedup = np.zeros(ii.shape, dtype=bool)

    partner_at_least_as_large = amps[ii] <= amps[jj]
    keep = in_candidate & ~in_dedup & partner_at_least_as_large & (ii != jj)
    ii = ii[keep]
    jj = jj[keep]

    flagged = np.zeros(n, dtype=bool)
    flagged[ii] = True

    return flagged, ii, jj


@databag
class CCFlagScores:
    rate: np.ndarray
    entropy: np.ndarray
    partner_fraction: np.ndarray
    chance_rate: np.ndarray | None = None
    excess_rate: np.ndarray | None = None


def cc_flag_scores(
    sorting: data_util.DARTsortSorting,
    n_units: int,
    temporal_radius_samples: int,
    radius_um: float,
    spatial_dedup_radius_um: float | None,
    dedup_temporal_radius_samples: int | None = None,
    chance_jitter_samples: int = 0,
    chance_draws: int = 0,
    seed: int = 0,
    amplitudes_dataset_name: str = "denoised_ptp_amplitudes",
) -> CCFlagScores:
    rate, entropy, partner_fraction = _cc_flag_rate(
        sorting=sorting,
        n_units=n_units,
        temporal_radius_samples=temporal_radius_samples,
        radius_um=radius_um,
        spatial_dedup_radius_um=spatial_dedup_radius_um,
        dedup_temporal_radius_samples=dedup_temporal_radius_samples,
        amplitudes_dataset_name=amplitudes_dataset_name,
    )

    if not (chance_jitter_samples and chance_draws):
        return CCFlagScores(
            rate=rate, entropy=entropy, partner_fraction=partner_fraction
        )

    assert chance_jitter_samples > temporal_radius_samples
    assert chance_draws > 0

    # correct for chance
    rg = np.random.default_rng(seed)
    times = sorting.times_samples
    chance = np.zeros(n_units)
    for _ in range(chance_draws):
        jittered = rg.integers(
            -chance_jitter_samples, chance_jitter_samples + 1, size=times.size
        )
        jittered += times
        jittered.sort()
        chance += _cc_flag_rate(
            sorting=sorting.ephemeral_replace(times_samples=jittered),
            n_units=n_units,
            temporal_radius_samples=temporal_radius_samples,
            radius_um=radius_um,
            spatial_dedup_radius_um=spatial_dedup_radius_um,
            dedup_temporal_radius_samples=dedup_temporal_radius_samples,
            amplitudes_dataset_name=amplitudes_dataset_name,
        )[0]
    chance /= chance_draws

    return CCFlagScores(
        rate=rate,
        entropy=entropy,
        partner_fraction=partner_fraction,
        chance_rate=chance,
        excess_rate=rate - chance,
    )


def _cc_flag_rate(
    sorting: data_util.DARTsortSorting,
    n_units: int,
    temporal_radius_samples: int,
    radius_um: float,
    spatial_dedup_radius_um: float | None,
    dedup_temporal_radius_samples: int | None,
    amplitudes_dataset_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flagged, ii, jj = flag_possible_cc_error_spikes(
        sorting=sorting,
        temporal_radius_samples=temporal_radius_samples,
        radius_um=radius_um,
        spatial_dedup_radius_um=spatial_dedup_radius_um,
        dedup_temporal_radius_samples=dedup_temporal_radius_samples,
        amplitudes_dataset_name=amplitudes_dataset_name,
    )
    assert sorting.labels is not None
    assert sorting.labels.max() < n_units
    li = sorting.labels[ii]
    lj = sorting.labels[jj]

    rate = np.zeros(n_units)
    entropy = np.full(n_units, np.inf)
    partner_fraction = np.zeros(n_units)
    for u in range(n_units):
        inu = np.flatnonzero(sorting.labels == u)
        if not inu.size:
            continue
        rate[u] = flagged[inu].mean()
        friends = lj[li == u]
        friends = friends[friends >= 0]
        if not friends.size:
            continue
        _, friend_count = np.unique(friends, return_counts=True)
        friend_p = friend_count / friend_count.sum()
        entropy[u] = -(np.log(friend_p) * friend_p).sum()
        partner_fraction[u] = friend_p.max()

    return rate, entropy, partner_fraction


def collision_cleaning_error_filter(
    *,
    recording: BaseRecording | None,
    sorting: data_util.DARTsortSorting,
    stable_features: StableWaveformFeatures | None,
    refinement_cfg: RefinementConfig,
    motion: MotionInfo,
    computation_cfg: ComputationConfig | None = None,
) -> CCErrorResult:
    use_error = refinement_cfg.collision_cleaning_error_threshold is not None
    use_flag = (
        refinement_cfg.max_cc_flag_rate < 1.0
        or refinement_cfg.cc_flag_excess_rate is not None
    )
    if not (use_error or use_flag):
        return CCErrorResult(sorting=sorting)

    from ..clustering.mixture import drop_units_and_update_scores
    from ..util.data_util import get_gmm_scores

    # remove blank labels just in case
    sorting = sorting.flatten(include_gmm_properties=True)
    assert sorting.labels is not None
    nu0 = sorting.labels.max() + 1
    if not nu0:
        return CCErrorResult(sorting=sorting)

    keep_mask = np.ones(nu0, dtype=np.bool)

    dist = None
    if use_error:
        assert recording is not None
        assert stable_features is not None
        dist = collision_cleaning_errors(
            recording=recording,
            sorting=sorting,
            stable_features=stable_features,
            refinement_cfg=refinement_cfg,
            motion=motion,
            computation_cfg=computation_cfg,
        )
        error_bad = dist > refinement_cfg.collision_cleaning_error_threshold
        logger.dartsortdebug(
            f"Collision-cleaning error filter dropped {error_bad.sum()} / {nu0} units."
        )
        keep_mask &= np.logical_not(error_bad)

    flag_scores = None
    if use_flag:
        need_chance = refinement_cfg.cc_flag_excess_rate is not None
        flag_scores = cc_flag_scores(
            sorting=sorting,
            n_units=nu0,
            temporal_radius_samples=refinement_cfg.cc_flag_temporal_radius_samples,
            radius_um=refinement_cfg.cc_flag_radius_um,
            spatial_dedup_radius_um=refinement_cfg.cc_flag_spatial_dedup_radius_um,
            dedup_temporal_radius_samples=refinement_cfg.cc_flag_dedup_temporal_radius_samples,
            chance_jitter_samples=(
                refinement_cfg.cc_flag_chance_jitter_samples if need_chance else 0
            ),
            chance_draws=refinement_cfg.cc_flag_chance_draws if need_chance else 0,
        )

        flag_bad = np.zeros(nu0, dtype=np.bool)
        if refinement_cfg.max_cc_flag_rate < 1.0:
            flag_bad |= (flag_scores.rate > refinement_cfg.max_cc_flag_rate) & (
                flag_scores.entropy < refinement_cfg.cc_flag_entropy_cutoff
            )
        if refinement_cfg.cc_flag_excess_rate is not None:
            assert flag_scores.excess_rate is not None
            flag_bad |= flag_scores.excess_rate > refinement_cfg.cc_flag_excess_rate

        logger.dartsortdebug(f"CC flag criterion dropped {flag_bad.sum()} / {nu0} units.")
        keep_mask &= np.logical_not(flag_bad)

    bad_ids = np.flatnonzero(np.logical_not(keep_mask))
    spike_keep_mask = keep_mask[sorting.labels]
    new_labels = np.where(spike_keep_mask, sorting.labels, -1)

    new_props = dict(labels=new_labels)
    if bad_ids.size:
        try:
            scores = get_gmm_scores(sorting, prefixes=["gmm"])
            scores, _ = drop_units_and_update_scores(
                train_scores=scores,
                scores=None,
                n_units=nu0,
                remove_ids=torch.tensor(bad_ids, dtype=torch.long),
            )

            # but also fully delete those spikes.
            cand = scores.candidates.numpy(force=True)
            ll = scores.log_liks.numpy(force=True)
            del_mask = np.logical_not(spike_keep_mask)
            cand[del_mask] = -1
            ll[del_mask, : cand.shape[1]] = -np.inf
            new_props["gmm_candidates"] = cand
            new_props["gmm_log_liks"] = ll
            if scores.responsibilities is not None:
                resp = scores.responsibilities.numpy(force=True)
                resp[del_mask, : cand.shape[1]] = 0.0
                resp[del_mask, -1] = 1.0
                new_props["gmm_responsibilities"] = resp
        except AttributeError:
            pass

    sorting = sorting.ephemeral_replace(**new_props)
    sorting = sorting.flatten(include_gmm_properties=True)
    return CCErrorResult(
        sorting=sorting, kept=keep_mask, errors=dist, flag_scores=flag_scores
    )


def collision_cleaning_errors(
    *,
    recording: BaseRecording,
    sorting: data_util.DARTsortSorting,
    stable_features: StableWaveformFeatures,
    refinement_cfg: RefinementConfig,
    motion: MotionInfo,
    computation_cfg: ComputationConfig | None = None,
) -> np.ndarray:
    from ..templates.templates import TemplateConfig, TemplateData

    assert sorting.labels is not None

    # subset the sorting to count per unit
    subset_sorting = data_util.subsample_to_max_count(
        sorting, max_spikes=refinement_cfg.pc_merge_spikes_per_unit
    )
    assert subset_sorting.labels is not None

    # average feature
    kept = np.flatnonzero(subset_sorting.labels >= 0)
    x = stable_features.features[kept]
    xlabels = torch.from_numpy(subset_sorting.labels[kept]).to(x.device)
    n_reg_chans = motion.rgeom.shape[0]
    means, counts = spiketorch.average_by_label(
        x, xlabels, stable_features.channels[kept], n_reg_chans
    )
    weights = counts / counts.amax(dim=1, keepdim=True)

    # median
    tpca = data_util.get_tpca(sorting)
    wf_cfg = tpca.waveform_cfg
    if tpca.temporal_slice is not None:
        wf_cfg = wf_cfg.relative_cfg(tpca.temporal_slice, recording.sampling_frequency)
    template_data = TemplateData.from_config(
        recording=recording,
        sorting=sorting,
        template_cfg=TemplateConfig(
            denoising_method="svd",
            reduction="median",
            denoising_rank=tpca.rank,
        ),
        tsvd=tpca.to_sklearn(),
        motion=motion,
        waveform_cfg=wf_cfg,
        computation_cfg=computation_cfg,
    )
    # the template engine flattens its sorting, so rows are indexed by unit_ids
    unit_ids = np.asarray(template_data.unit_ids)
    assert unit_ids.shape[0] == template_data.templates.shape[0]
    assert unit_ids.max() < means.shape[0]
    templates = means.new_zeros(means.shape[0], *template_data.templates.shape[1:])
    templates[unit_ids] = means.new_tensor(template_data.templates)
    templates = tpca.force_embed(templates)
    assert templates.shape == means.shape

    # check difference
    x = means * weights[:, None]
    K = x.shape[0]
    x = x.view(K, -1)
    y = (templates * weights[:, None]).view(K, -1)
    xnorm = torch.linalg.vector_norm(x, dim=1)
    ynorm = torch.linalg.vector_norm(y, dim=1)
    dist = torch.linalg.vector_norm(x - y, dim=1).div_((xnorm * ynorm).sqrt_())
    return dist.numpy(force=True)


@databag
class GMMIsolationResult:
    sorting: data_util.DARTsortSorting
    """sorting after discarding badly isolated units"""

    scores: Scores | None
    """Soft assignment likelihoods after discarding"""

    isolation: np.ndarray | None
    """K: measure of unit isolation for original units before discarding."""

    keep_mask: np.ndarray | None
    """K: which units were retained? isolation[keep_mask] gives remaining iso scores."""


def gmm_isolation_filter(
    *,
    sorting: data_util.DARTsortSorting,
    refinement_cfg: RefinementConfig,
    computation_cfg: ComputationConfig | None,
    show_progress: bool = False,
):
    from ..clustering.mixture import drop_units_and_update_scores, labels_from_scores
    from ..util.data_util import get_gmm_scores

    assert sorting.labels is not None
    try:
        scores = get_gmm_scores(sorting, prefixes=["gmm"])
    except AttributeError:
        logger.dartsortdebug("No GMM scores attached to sorting, no isolation filter.")
        return GMMIsolationResult(
            sorting=sorting, isolation=None, scores=None, keep_mask=None
        )

    gi = gmm_isolation_scores(
        scores=scores,
        unit_ids=sorting.unit_ids,
        show_progress=show_progress or logger.isEnabledFor(DARTSORTVERBOSE),
        neighbor_fraction=refinement_cfg.gmm_isolation_neighbor_fraction,
        min_count=refinement_cfg.min_count,
        computation_cfg=computation_cfg,
    )
    orig_isolation = gi.isolation.copy()

    unit_ids = sorting.unit_ids
    removed_ids = []
    if refinement_cfg.gmm_isolation_threshold:
        # sequential filter:
        # remove the baddest non-isolated unit until none remain
        update = np.zeros(len(unit_ids), dtype=bool)
        while True:
            assert scores.candidates is not None
            gi_update = gmm_isolation_scores(
                scores,
                unit_ids=unit_ids[update],
                show_progress=show_progress,
                neighbor_fraction=refinement_cfg.gmm_isolation_neighbor_fraction,
                min_count=refinement_cfg.min_count,
                computation_cfg=computation_cfg,
            )
            gi.isolation[update] = gi_update.isolation

            # nb: nan is not <= anything. hence nan2num. but nans retained in return val.
            good = np.nan_to_num(gi.isolation) < refinement_cfg.gmm_isolation_threshold
            ngood = good.sum()
            if ngood == gi.isolation.shape[0]:
                break

            # TODO: why is bad_guy coming through in sorted order?
            bad_guy = np.argmax(np.where(good, -np.inf, gi.isolation))
            removed_ids.append(bad_guy)

            # try to update as little as possible
            in_cand = (scores.candidates == bad_guy).any(dim=1)  # type: ignore
            needs_update = scores.candidates[in_cand].unique()
            update[:] = False
            update[needs_update[needs_update >= 0]] = True
            update[good] = False

            logger.dartsortverbose(
                f"Drop {bad_guy} (%s / %s total, %s good).",
                len(removed_ids),
                update.size,
                ngood,
            )
            scores, _ = drop_units_and_update_scores(
                train_scores=scores,
                scores=None,
                n_units=unit_ids.shape[0],
                remove_ids=torch.tensor([bad_guy], dtype=torch.long),
            )

    keep_mask = np.ones(unit_ids.max() + 1)
    removed_ids = np.array(removed_ids, dtype=np.int64)
    removed_ids.sort()
    keep_mask[removed_ids] = 0

    if logger.isEnabledFor(DARTSORTDEBUG):
        logger.dartsortdebug(
            "GMM isolation at threshold %s: %s finite scores "
            "(min/mean/max=%s/%s/%s), keep %s (%s %%).",
            refinement_cfg.gmm_isolation_threshold,
            np.isfinite(orig_isolation).sum().item(),
            np.nanmin(orig_isolation).item(),
            np.nanmean(orig_isolation).item(),
            np.nanmax(orig_isolation).item(),
            keep_mask.sum().item(),
            f"{100 * keep_mask.mean():0.2f}",
        )

    new_labels = labels_from_scores(scores)
    new_props = dict(
        labels=new_labels,
        gmm_candidates=scores.candidates.numpy(force=True),
        gmm_log_liks=scores.log_liks.numpy(force=True),
    )
    if scores.responsibilities is not None:
        new_props["gmm_responsibilities"] = scores.responsibilities.numpy(force=True)
    else:
        sorting.remove_ephemeral_feature("gmm_responsibilities")
    sorting = sorting.ephemeral_replace(**new_props)
    sorting = sorting.flatten(include_gmm_properties=True)
    return GMMIsolationResult(
        sorting=sorting, scores=scores, keep_mask=keep_mask, isolation=orig_isolation
    )


@databag
class GMMIsolationScores:
    isolation: np.ndarray
    kde_domain: np.ndarray | None = None
    kdes: np.ndarray | None = None


def gmm_isolation_scores(
    scores: Scores,
    unit_ids: np.ndarray | None = None,
    show_progress=False,
    computation_cfg: ComputationConfig | None = None,
    neighbor_fraction: float = 0.9,
    min_count: int = 5,
    allow_parallel=False,
    return_kdes=False,
    kde_rhs=50.0,
    kde_dx: float = 0.5,
) -> GMMIsolationScores:
    if unit_ids is None:
        unit_ids = np.arange(scores.candidates[:, 0].max().item() + 1)
    assert unit_ids is not None
    if not unit_ids.size:
        _e = np.zeros((0,))
        return GMMIsolationScores(isolation=_e)

    ctx = _GMMIsolationContext(
        cand=scores.candidates,
        log_liks=scores.log_liks,
        neighbor_fraction=neighbor_fraction,
        min_count=min_count,
        return_kdes=return_kdes,
        kde_rhs=kde_rhs,
        dx=kde_dx,
    )

    if not allow_parallel:
        # seems not to help here.
        computation_cfg = ComputationConfig.from_n_jobs(0, 0)
    n_jobs, Executor, context, *_ = pool_from_cfg(
        computation_cfg, check_local=True, small=True, cpu=True
    )
    with Executor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=_iso_init,
        initargs=(ctx,),
    ) as pool:
        isolation = np.full(unit_ids.size, np.nan)
        domain = None
        kdes = None

        results = pool.map(_iso_job, unit_ids)
        if show_progress:
            results = progbar(
                results,
                desc=f"GMMiso:{n_jobs}",
                total=unit_ids.shape[0],
                mininterval=0.5,
                smoothing=0.0,
            )

        for j, (iso, dd, kde) in enumerate(results):
            isolation[j] = iso
            if dd is None:
                continue

            assert return_kdes
            assert kde is not None
            if domain is None:
                domain = dd
                assert kdes is None
                kdes = np.full((unit_ids.size, domain.shape[0]), np.nan)
            else:
                assert np.array_equal(domain, dd)
            assert kdes is not None
            kdes[j] = kde

    return GMMIsolationScores(isolation=isolation, kde_domain=domain, kdes=kdes)


_iso_ctx = local()
_iso_ctx.ctx = None


@databag
class _GMMIsolationContext:
    cand: torch.Tensor
    log_liks: torch.Tensor
    dx: float
    neighbor_fraction: float
    min_count: int
    return_kdes: bool
    kde_rhs: float


def _iso_init(ctx):
    _iso_ctx.ctx = ctx


def _iso_job(unit_id) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    p = cast(_GMMIsolationContext, _iso_ctx.ctx)
    assert p is not None

    # top indices
    (inu,) = (p.cand[:, 0] == unit_id).nonzero(as_tuple=True)
    ninu = inu.numel()
    if ninu <= p.min_count:
        return np.nan, None, None

    # likelihood ratio when unit comes first vs second
    in_ll = p.log_liks[inu, :-1]
    # in_denom = in_ll[:, 1:].logsumexp(dim=1)
    in_denom = in_ll[:, 1]
    min_denom, max_denom = in_denom.aminmax()
    in_lr = in_ll[:, 0] - in_denom
    if torch.isneginf(max_denom):
        # this unit is super isolated.
        return 0.0, None, None
    if torch.isneginf(min_denom):
        # first check that enough are finite
        (finite,) = torch.isfinite(in_lr).nonzero(as_tuple=True)
        nfinite = finite.numel()
        if nfinite < p.min_count:
            return np.nan, None, None
        if nfinite / ninu < p.neighbor_fraction:
            return np.nan, None, None

        # handle +inf lrs by replacing with neg log liks
        in_lr.clamp_(max=in_ll[:, 0].abs_())

    # fit kde. will symmetrize around 0 to avoid boundary issues (since lr>=0)
    amax = in_lr.abs().amax().item()
    lr = in_lr.double().numpy(force=True)
    try:
        # select bandwidth based on positive part
        bw = improved_sheather_jones(lr[:, None])
        if not np.isfinite(bw):
            return np.nan, None, None
        # reweight near 0
        weights = norm.sf(0, loc=lr, scale=bw)
        # reflect and stack
        weights = np.concatenate([weights, weights])
        lr = np.concatenate([-lr, lr])

        kde = FFTKDE(bw=bw).fit(lr, weights=weights)
    except ValueError:
        return np.nan, None, None

    # evaluate kde. grid needs to cover data.
    end = p.dx * np.ceil((amax + p.dx) / p.dx)
    grid = np.arange(0, end + p.dx / 2, p.dx)
    grid_left = -(grid[1:][::-1])
    grid = np.concatenate([grid_left, grid])
    evf = kde.evaluate(grid) / 2
    # fold
    nc = grid.shape[0] // 2
    assert np.isclose(grid[nc], 0.0)
    ev = evf[nc:]
    ev += evf[: nc + 1][::-1]
    grid = grid[nc:]
    ev0 = ev[0]

    # find the HIGHEST local max right of 0, if any
    peak_ix_right = _find_right_peak_index(ev, 1)
    no_right_peak = peak_ix_right == ev.shape[0] or ev0 > ev[peak_ix_right]

    if no_right_peak:
        # no isolation
        iso = 1.0
    else:
        # main case
        rad = int(np.ceil(10 / p.dx))
        dip = np.min(ev[: min(rad + 1, peak_ix_right)])
        peak = ev[peak_ix_right]
        iso = dip / peak

    if p.return_kdes:
        # ev = ev[nc:]
        if grid[-1] < p.kde_rhs:
            domain = np.arange(0, p.kde_rhs + p.dx / 2, p.dx)
            npad = domain.size - grid.size
            kde = np.pad(ev, (0, npad))
        else:
            mask = grid <= p.kde_rhs
            kde = ev[mask]
            domain = grid[mask]
    else:
        kde = domain = None

    # compare value at 0 to peak value
    return iso, domain, kde


@numba.njit
def _find_right_peak_index(x: np.ndarray, i: int) -> int:
    n = x.shape[0]

    # while decreasing, move right
    for j in range(i, n - 1):
        if x[j + 1] > x[j]:
            break
    else:
        return n

    return j + np.argmax(x[j:]).item()


@numba.njit
def _find_left_peak_index(x: np.ndarray, i: int):
    # if 0 is a peak, okay then
    if x[i] > x[i - 1] and x[i] > x[i + 1]:
        return i

    # move left while decreasing
    j = i
    while j > 0:
        if x[j - 1] > x[j]:
            break
        j -= 1
    else:
        return -1

    # move left while increasing, return once that stops
    while j > 0:
        if x[j - 1] <= x[j]:
            return j
        j -= 1

    # I don't think this is possible
    return -2
