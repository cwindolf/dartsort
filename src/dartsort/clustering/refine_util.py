from threading import local
from typing import cast

import h5py
import numba
import numpy as np
import torch
from KDEpy import FFTKDE
from KDEpy.bw_selection import improved_sheather_jones
from scipy.stats import norm
from spikeinterface.core.baserecording import BaseRecording

from dartsort.clustering.mixture import Scores

from ..transform.temporal_pca import BaseTemporalPCA
from ..util import data_util, spiketorch
from ..util.internal_config import ComputationConfig, RefinementConfig
from ..util.logging_util import DARTSORTDEBUG, DARTSORTVERBOSE, get_logger, progbar
from ..util.motion import MotionInfo
from ..util.multiprocessing_util import pool_from_cfg
from ..util.py_util import databag
from .cluster_util import hierarchical_cluster, reorder_by_depth
from .clustering_features import StableWaveformFeatures

logger = get_logger(__name__)


def get_noise_log_priors(noise, sorting, refinement_cfg):
    from dartsort.templates import TemplateData

    if not refinement_cfg.noise_fp_correction:
        return None

    h5_name = sorting.parent_h5_path
    if h5_name is None:
        return None
    stem = h5_name.stem

    if stem.startswith("matching"):
        model_dir = h5_name.parent / f"{stem}_models"
        templates_npz = model_dir / "template_data.npz"
        if not templates_npz.exists():
            raise ValueError(f"{templates_npz} is not there?")

        with h5py.File(h5_name, "r", locking=False) as h5:
            matching_labels = h5["labels"][:]

        template_data = TemplateData.from_npz(templates_npz)
        tpca = data_util.get_tpca(sorting)
        assert isinstance(tpca, BaseTemporalPCA)
        if (sl := getattr(tpca, "temporal_slice", None)) is not None:
            temps_tpca = torch.asarray(template_data.templates[:, sl])
        else:
            temps_tpca = torch.asarray(template_data.templates)

        n, t, c = temps_tpca.shape
        temps_tpca = temps_tpca.permute(0, 2, 1).reshape(n * c, t)
        temps_tpca = tpca._transform_in_probe(temps_tpca)
        temps_tpca = temps_tpca.reshape(n, c, -1).permute(0, 2, 1)

        noise_log_priors = noise.detection_prior_log_prob(temps_tpca)
        logger.dartsortdebug(
            f"Got log priors ranging {noise_log_priors.min()}-{noise_log_priors.max()}."
        )
        noise_log_priors = noise_log_priors[matching_labels]

        return noise_log_priors
    elif stem.startswith("subtract"):
        noise_log_priors = noise.channelwise_detection_prior_log_prob()
        logger.dartsortdebug(
            f"Got log priors ranging {noise_log_priors.min()}-{noise_log_priors.max()}."
        )
        noise_log_priors = noise_log_priors[sorting.channels]
        return noise_log_priors
    else:
        return None


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
    ul = np.unique(labels)
    ul = ul[ul >= 0]
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
    sorting, new_ids = reorder_by_depth(
        sorting, motion=motion, spatial_footprints=sf, geom=motion.rgeom
    )
    return PCMergeResult(sorting=sorting)


@databag
class CCErrorResult:
    sorting: data_util.DARTsortSorting
    errors: np.ndarray | None = None


def collision_cleaning_error_filter(
    *,
    recording: BaseRecording,
    sorting: data_util.DARTsortSorting,
    stable_features: StableWaveformFeatures,
    refinement_cfg: RefinementConfig,
    motion: MotionInfo,
    computation_cfg: ComputationConfig | None = None,
) -> CCErrorResult:
    if refinement_cfg.collision_cleaning_error_threshold is None:
        return CCErrorResult(sorting=sorting)

    from ..clustering.mixture import drop_units_and_update_scores
    from ..templates.templates import TemplateConfig, TemplateData
    from ..util.data_util import get_gmm_scores

    # remove blank labels just in case
    sorting = sorting.flatten(include_gmm_properties=True)
    assert sorting.labels is not None
    nu0 = sorting.labels.max() + 1
    if not nu0:
        return CCErrorResult(sorting=sorting)

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
    templates = TemplateData.from_config(
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
    templates = means.new_tensor(templates.templates)
    templates = tpca.force_embed(templates)

    # check difference
    x = means * weights[:, None]
    K = x.shape[0]
    x = x.view(K, -1)
    y = (templates * weights[:, None]).view(K, -1)
    xnorm = torch.linalg.vector_norm(x, dim=1)
    ynorm = torch.linalg.vector_norm(y, dim=1)
    dist = torch.linalg.vector_norm(x - y, dim=1).div_((xnorm * ynorm).sqrt_())
    dist = dist.numpy(force=True)

    # discard bad units
    keep_mask = np.ones(nu0, dtype=np.bool)
    bad_ids = np.flatnonzero(dist > refinement_cfg.collision_cleaning_error_threshold)
    logger.dartsortdebug(
        f"Collision-cleaning error filter drops {bad_ids.size} / {nu0} units."
    )
    keep_mask[bad_ids] = False
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
    return CCErrorResult(sorting=sorting, errors=dist)


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
        unit_ids = unit_ids
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

            assert return_kdes, 1
            assert kde is not None, 2
            if domain is None:
                domain = dd
                assert kdes is None, 3
                kdes = np.full((unit_ids.size, domain.shape[0]), np.nan)
            else:
                assert np.array_equal(domain, dd), 4
            assert kdes is not None, 5
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
    global _iso_ctx
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
