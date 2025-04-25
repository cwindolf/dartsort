import gc
from logging import getLogger

from .. import config
from ..util import job_util, noise_util
from .split import split_clusters
from .merge import merge_templates
from .stable_features import StableSpikeDataset
from .gaussian_mixture import SpikeMixtureModel


logger = getLogger(__name__)


def refine_clustering(
    recording,
    sorting,
    motion_est=None,
    refinement_config=config.default_refinement_config,
    computation_config=None,
    return_step_labels=False,
):
    """Refine a clustering using the strategy specified by the config."""
    if refinement_config.refinement_stragegy == "splitmerge":
        assert refinement_config.split_merge_config is not None
        ref = split_merge(
            recording,
            sorting,
            motion_est=motion_est,
            split_merge_config=refinement_config.split_merge_config,
            computation_config=computation_config,
        )
        return ref, {}

    # below is all gmm stuff
    assert refinement_config.refinement_stragegy == "gmm"
    if computation_config is None:
        computation_config = job_util.get_global_computation_config()

    logger.dartsortdebug(f"Refine clustering from {sorting.parent_h5_path}")

    noise = noise_util.EmbeddedNoise.estimate_from_hdf5(
        sorting.parent_h5_path,
        cov_kind=refinement_config.cov_kind,
        motion_est=motion_est,
        sigma=refinement_config.interpolation_sigma,
        device=computation_config.actual_device(),
        glasso_alpha=refinement_config.glasso_alpha,
    )
    data = StableSpikeDataset.from_sorting(
        sorting,
        motion_est=motion_est,
        core_radius=refinement_config.core_radius,
        max_n_spikes=refinement_config.max_n_spikes,
        interpolation_sigma=refinement_config.interpolation_sigma,
        split_proportions=(
            1.0 - refinement_config.val_proportion,
            refinement_config.val_proportion,
        ),
        device=computation_config.actual_device(),
    )
    gmm = SpikeMixtureModel(
        data,
        noise,
        min_count=refinement_config.min_count,
        n_threads=computation_config.actual_n_jobs(),
        n_spikes_fit=refinement_config.n_spikes_fit,
        ppca_rank=refinement_config.signal_rank,
        ppca_inner_em_iter=refinement_config.ppca_inner_em_iter,
        n_em_iters=refinement_config.n_em_iters,
        distance_metric=refinement_config.distance_metric,
        distance_normalization_kind=refinement_config.distance_normalization_kind,
        merge_distance_threshold=refinement_config.merge_distance_threshold,
        merge_criterion_threshold=refinement_config.merge_criterion_threshold,
        merge_criterion=refinement_config.merge_criterion,
        merge_bimodality_threshold=refinement_config.merge_bimodality_threshold,
        em_converged_prop=refinement_config.em_converged_prop,
        em_converged_churn=refinement_config.em_converged_churn,
        em_converged_atol=refinement_config.em_converged_atol,
        channels_strategy=refinement_config.channels_strategy,
        hard_noise=refinement_config.hard_noise,
        split_decision_algorithm=refinement_config.split_decision_algorithm,
        merge_decision_algorithm=refinement_config.merge_decision_algorithm,
        prior_pseudocount=refinement_config.prior_pseudocount,
        laplace_ard=refinement_config.laplace_ard,
    )

    # these are for benchmarking
    step_labels = {}
    intermediate_split = "full" if return_step_labels else "kept"
    gmm.log_liks = None  # TODO
    for it in range(refinement_config.n_total_iters):
        if refinement_config.truncated:
            res = gmm.tvi(final_split=intermediate_split)
            gmm.log_liks = res["log_liks"]
        else:
            gmm.log_liks = gmm.em(final_split=intermediate_split)
        if return_step_labels:
            step_labels[f"refstepaem{it}"] = gmm.labels.numpy(force=True).copy()

        assert gmm.log_liks is not None
        # TODO: if split is self-consistent enough, we don't need this.
        if (
            gmm.log_liks.shape[0]
            > refinement_config.max_avg_units * recording.get_num_channels()
        ):
            logger.dartsortdebug(f"{gmm.log_liks.shape=}, skipping split.")
        else:
            # TODO: not this.
            gmm.split()
            gmm.log_liks = None

            gc.collect()
            if refinement_config.truncated:
                res = gmm.tvi(final_split=intermediate_split)
                gmm.log_liks = res["log_liks"]
            else:
                gmm.log_liks = gmm.em(final_split=intermediate_split)
            if return_step_labels:
                step_labels[f"refstepbsplit{it}"] = gmm.labels.numpy(force=True).copy()
        assert gmm.log_liks is not None
        gmm.merge(gmm.log_liks)
        gmm.log_liks = None

        gc.collect()
        if return_step_labels:
            step_labels[f"refstepcmerge{it}"] = gmm.labels.numpy(force=True).copy()

    if refinement_config.truncated:
        res = gmm.tvi(final_split="full")
        gmm.log_liks = res  # not actually! but just to del it later.
    else:
        gmm.log_liks = gmm.em(final_split="full")
    gmm.log_liks = None

    gc.collect()
    gmm.cpu()
    sorting = gmm.to_sorting()
    del gmm

    gc.collect()
    return sorting, step_labels


def split_merge(
    recording,
    sorting,
    motion_est=None,
    split_merge_config=config.default_split_merge_config,
    computation_config=None,
):
    if computation_config is None:
        computation_config = job_util.get_global_computation_config()

    split_sorting = split_clusters(
        sorting,
        split_strategy=split_merge_config.split_strategy,
        recursive=split_merge_config.recursive_split,
        n_jobs=computation_config.actual_n_jobs(),
        motion_est=motion_est,
    )

    merge_sorting = merge_templates(
        split_sorting,
        recording,
        motion_est=motion_est,
        template_config=split_merge_config.merge_template_config,
        merge_distance_threshold=split_merge_config.merge_distance_threshold,
        min_spatial_cosine=split_merge_config.min_spatial_cosine,
        linkage=split_merge_config.linkage,
        computation_config=computation_config,
    )

    return merge_sorting
