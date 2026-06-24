from threading import local
from typing import cast

import h5py
import numpy as np
import torch
from KDEpy import FFTKDE

from ..transform.temporal_pca import BaseTemporalPCA
from ..util import data_util, spiketorch
from ..util.internal_config import ComputationConfig, RefinementConfig
from ..util.logging_util import DARTSORTVERBOSE, get_logger, progbar
from ..util.motion import MotionInfo
from ..util.multiprocessing_util import pool_from_cfg
from ..util.py_util import databag
from .cluster_util import hierarchical_cluster, reorder_by_depth, sparsify_labels
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
class GMMIsolationResult:
    sorting: data_util.DARTsortSorting
    """sorting after discarding badly isolated units"""

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
    from ..util.data_util import get_gmm_scores

    assert sorting.labels is not None
    try:
        scores = get_gmm_scores(sorting)
    except AttributeError:
        logger.dartsortdebug("No GMM scores attached to sorting, no isolation filter.")
        return GMMIsolationResult(sorting=sorting, isolation=None, keep_mask=None)

    glabels = scores.candidates[:, 0].numpy(force=True)

    ctx = _GMMIsolationContext(
        inus=sparsify_labels(glabels),
        cand=scores.candidates,
        log_liks=scores.log_liks,
        cfg=refinement_cfg,
    )

    n_jobs, Executor, context, *_ = pool_from_cfg(
        computation_cfg, check_local=True, small=True, cpu=True
    )
    with Executor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=_iso_init,
        initargs=(ctx,),
    ) as pool:
        unit_ids = sorting.unit_ids
        iso_scores = np.full(unit_ids.size, np.nan)

        results = pool.map(_iso_job, unit_ids)
        if show_progress or logger.isEnabledFor(DARTSORTVERBOSE):
            results = progbar(
                results,
                desc=f"GMMiso:{n_jobs}",
                total=unit_ids.shape[0],
                mininterval=0.5,
                smoothing=0.0,
            )

        for j, res in enumerate(results):
            iso_scores[j] = res

    # nb: nan is not <= anything. hence nan2num. but nans retained in return val.
    keep_mask = np.nan_to_num(iso_scores) <= refinement_cfg.gmm_isolation_threshold
    new_labels = np.where(keep_mask[sorting.labels], sorting.labels, -1)
    sorting = sorting.ephemeral_replace(labels=new_labels).flatten()
    return GMMIsolationResult(
        sorting=sorting, keep_mask=keep_mask, isolation=iso_scores
    )


_iso_ctx = local()
_iso_ctx.ctx = None


@databag
class _GMMIsolationContext:
    inus: dict[int, np.ndarray]
    cand: torch.Tensor
    log_liks: torch.Tensor
    dx: float = 1.0
    cfg: RefinementConfig


def _iso_init(ctx):
    global _iso_ctx
    _iso_ctx.ctx = ctx


def _iso_job(unit_id):
    p = cast(_GMMIsolationContext, _iso_ctx.ctx)
    assert p is not None
    inu = p.inus[unit_id]
    if inu.size < p.cfg.min_count:
        return np.nan

    # get my log likelihood ratio vs the rest of the mixture (not noise unit)
    ll = p.log_liks[inu]
    log_num = ll[:, 0]
    log_denom = ll[:, 1 : p.cand.shape[1]].logsumexp(dim=1)
    min_denom, max_denom = log_denom.aminmax()
    min_num, max_num = log_num.aminmax()
    assert torch.isfinite(min_num)
    assert torch.isfinite(max_num)
    if torch.isneginf(max_denom):
        # this unit is super isolated.
        return 0.0
    lr = log_num - log_denom
    if torch.isneginf(min_denom):
        # first check that enough are finite
        (finite,) = torch.isfinite(lr).nonzero(as_tuple=True)
        nfinite = finite.numel()
        if nfinite < p.cfg.min_count:
            return np.nan
        if nfinite / inu.size < p.cfg.gmm_isolation_neighbor_fraction:
            return np.nan

        # handle +inf lrs by clamping to a big value
        lr.clamp_(max=log_num[finite].abs_().add_(lr[finite].amax() + 10.0))

    # fit kde. will symmetrize around 0 to avoid boundary issues (since lr>=0)
    amax = lr.amax().item()
    lr = lr.numpy(force=True)
    lr = np.concatenate([-lr, lr])
    kde = FFTKDE(bw="ISJ").fit(lr)

    # evaluate kde. grid needs to cover data.
    end = p.dx * np.ceil((amax + p.dx) / p.dx)
    grid = np.arange(-end, end + p.dx / 2, p.dx)
    ev = kde.evaluate(grid)

    # compare value at 0 to peak value
    nc = grid.shape[0] // 2
    assert grid.shape == (2 * nc + 1,)
    assert np.isclose(grid[nc], 0.0)
    return ev[nc] / ev[nc:].max()
