import gc
from logging import getLogger

from ..util.internal_config import default_refinement_config, default_split_merge_config
from ..util import job_util, noise_util
from ..util.main_util import ds_save_intermediate_labels
from .split import split_clusters
from .merge import merge_templates
from .stable_features import StableSpikeDataset
from .gaussian_mixture import SpikeMixtureModel


logger = getLogger(__name__)


def refine_clustering(
    recording,
    sorting,
    motion_est=None,
    refinement_config=default_refinement_config,
    computation_config=None,
    return_step_labels=False,
    save_step_labels_format=None,
    save_step_labels_dir=None,
    save_cfg=None,
):
    """Refine a clustering using the strategy specified by the config."""
    if refinement_config.refinement_strategy == "splitmerge":
        assert refinement_config.split_merge_config is not None
        ref = split_merge(
            recording,
            sorting,
            motion_est=motion_est,
            split_merge_config=refinement_config.split_merge_config,
            computation_config=computation_config,
        )
        return ref, {}
    elif refinement_config.refinement_strategy == "gmm":
        return gmm_refine(
            recording,
            sorting,
            motion_est=motion_est,
            refinement_config=refinement_config,
            computation_config=computation_config,
            return_step_labels=return_step_labels,
            save_step_labels_format=save_step_labels_format,
            save_step_labels_dir=save_step_labels_dir,
            save_cfg=save_cfg,
        )
    else:
        assert False


def gmm_refine(
    recording,
    sorting,
    motion_est=None,
    refinement_config=default_refinement_config,
    computation_config=None,
    return_step_labels=False,
    save_step_labels_format=None,
    save_step_labels_dir=None,
    save_cfg=None,
):
    saving = (save_step_labels_format is not None) and (
        save_step_labels_dir is not None
    )
    assert refinement_config.refinement_strategy == "gmm"
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
        interpolation_method=refinement_config.interpolation_method,
    )
    data = StableSpikeDataset.from_sorting(
        sorting,
        motion_est=motion_est,
        core_radius=refinement_config.core_radius,
        max_n_spikes=refinement_config.max_n_spikes,
        interpolation_method=refinement_config.interpolation_method,
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
        criterion_threshold=refinement_config.criterion_threshold,
        criterion=refinement_config.criterion,
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

    step_labels = {}
    intermediate_split = "full" if return_step_labels else "kept"
    gmm.log_liks = None

    for it in range(refinement_config.n_total_iters):
        if refinement_config.truncated:
            res = gmm.tvi(final_split=intermediate_split)
            gmm.log_liks = res["log_liks"]
        else:
            gmm.log_liks = gmm.em(final_split=intermediate_split)
        if return_step_labels:
            step_labels[f"refstep{it}aem"] = gmm.labels.numpy(force=True).copy()
        if saving:
            ds_save_intermediate_labels(
                save_step_labels_format.format(stepname=f"refstep{it}aem"),
                gmm.to_sorting(),
                save_step_labels_dir,
                save_cfg,
            )

        assert gmm.log_liks is not None
        if (
            gmm.log_liks.shape[0]
            > refinement_config.max_avg_units * recording.get_num_channels()
        ):
            logger.dartsortdebug(
                f"Skipping split ({gmm.log_liks.shape[0]} is too many units already)."
            )
            if refinement_config.one_split_only:
                break
        else:
            gmm.em(n_iter=1, force_refit=True)
            gmm.split()
            gmm.log_liks = None

            if refinement_config.one_split_only:
                break

            gc.collect()
            if refinement_config.truncated:
                res = gmm.tvi(final_split=intermediate_split)
                gmm.log_liks = res["log_liks"]
            else:
                gmm.log_liks = gmm.em(final_split=intermediate_split)
            if return_step_labels:
                step_labels[f"refstep{it}bsplit"] = gmm.labels.numpy(force=True).copy()
            if saving:
                ds_save_intermediate_labels(
                    save_step_labels_format.format(stepname=f"refstep{it}bsplit"),
                    gmm.to_sorting(),
                    save_step_labels_dir,
                    save_cfg,
                )

        assert gmm.log_liks is not None
        gmm.em(n_iter=1, force_refit=True)
        gmm.merge(gmm.log_liks)
        gmm.log_liks = None

        gc.collect()
        if return_step_labels:
            step_labels[f"refstep{it}cmerge"] = gmm.labels.numpy(force=True).copy()
        if saving:
            ds_save_intermediate_labels(
                save_step_labels_format.format(stepname=f"refstep{it}cmerge"),
                gmm.to_sorting(),
                save_step_labels_dir,
                save_cfg,
            )

    if refinement_config.truncated:
        gmm.tvi(final_split="full")
    else:
        gmm.em(final_split="full")
    sorting = gmm.to_sorting()
    return sorting, step_labels


def split_merge(
    recording,
    sorting,
    motion_est=None,
    split_merge_config=default_split_merge_config,
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
