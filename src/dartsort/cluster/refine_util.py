from dataclasses import replace
from logging import getLogger

import torch
import h5py

from ..util.internal_config import default_refinement_cfg
from ..util import job_util, noise_util, data_util, spiketorch
from ..util.main_util import ds_save_intermediate_labels
from .cluster_util import agglomerate
from .split import split_clusters
from .merge import merge_templates
from .stable_features import StableSpikeDataset
from .gaussian_mixture import SpikeMixtureModel


logger = getLogger(__name__)


def gmm_refine(
    recording,
    sorting,
    motion_est=None,
    refinement_cfg=default_refinement_cfg,
    computation_cfg=None,
    return_step_labels=False,
    save_step_labels_format=None,
    save_step_labels_dir=None,
    save_cfg=None,
):
    saving = (save_step_labels_format is not None) and (
        save_step_labels_dir is not None
    )
    assert refinement_cfg.refinement_strategy == "gmm"
    if computation_cfg is None:
        computation_cfg = job_util.get_global_computation_config()

    logger.dartsortdebug(f"Refine clustering from {sorting.parent_h5_path}")

    data = StableSpikeDataset.from_sorting(
        sorting,
        motion_est=motion_est,
        core_radius=refinement_cfg.core_radius,
        max_n_spikes=refinement_cfg.max_n_spikes,
        interpolation_method=refinement_cfg.interpolation_method,
        extrap_method=refinement_cfg.extrapolation_method,
        extrap_kernel=refinement_cfg.extrapolation_kernel,
        kernel_name=refinement_cfg.kernel_name,
        sigma=refinement_cfg.interpolation_sigma,
        rq_alpha=refinement_cfg.rq_alpha,
        kriging_poly_degree=refinement_cfg.kriging_poly_degree,
        split_proportions=(
            1.0 - refinement_cfg.val_proportion,
            refinement_cfg.val_proportion,
        ),
        device=computation_cfg.actual_device(),
    )
    noise = noise_util.EmbeddedNoise.estimate_from_hdf5(
        sorting.parent_h5_path,
        cov_kind=refinement_cfg.cov_kind,
        motion_est=motion_est,
        device=computation_cfg.actual_device(),
        glasso_alpha=refinement_cfg.glasso_alpha,
        interpolation_method=refinement_cfg.interpolation_method,
        kernel_name=refinement_cfg.kernel_name,
        sigma=refinement_cfg.interpolation_sigma,
        rq_alpha=refinement_cfg.rq_alpha,
        kriging_poly_degree=refinement_cfg.kriging_poly_degree,
        zero_radius=refinement_cfg.cov_radius,
        rgeom=data.prgeom[:-1].numpy(force=True),
    )
    noise_log_priors = get_noise_log_priors(noise, sorting, refinement_cfg)
    gmm = SpikeMixtureModel(
        data,
        noise,
        min_count=refinement_cfg.min_count,
        n_threads=computation_cfg.actual_n_jobs(),
        n_spikes_fit=refinement_cfg.n_spikes_fit,
        ppca_rank=refinement_cfg.signal_rank,
        ppca_inner_em_iter=refinement_cfg.ppca_inner_em_iter,
        n_em_iters=refinement_cfg.n_em_iters,
        distance_metric=refinement_cfg.distance_metric,
        search_type=refinement_cfg.search_type,
        lls_keep_k=refinement_cfg.n_candidates,
        tvi_n_candidates=refinement_cfg.n_candidates,
        tvi_n_search=refinement_cfg.n_search,
        distance_normalization_kind=refinement_cfg.distance_normalization_kind,
        merge_distance_threshold=refinement_cfg.merge_distance_threshold,
        criterion_threshold=refinement_cfg.criterion_threshold,
        criterion=refinement_cfg.criterion,
        merge_bimodality_threshold=refinement_cfg.merge_bimodality_threshold,
        em_converged_prop=refinement_cfg.em_converged_prop,
        em_converged_churn=refinement_cfg.em_converged_churn,
        em_converged_atol=refinement_cfg.em_converged_atol,
        channels_strategy=refinement_cfg.channels_strategy,
        hard_noise=refinement_cfg.hard_noise,
        split_decision_algorithm=refinement_cfg.split_decision_algorithm,
        merge_decision_algorithm=refinement_cfg.merge_decision_algorithm,
        prior_pseudocount=refinement_cfg.prior_pseudocount,
        prior_scales_mean=refinement_cfg.prior_scales_mean,
        laplace_ard=refinement_cfg.laplace_ard,
        kmeans_k=refinement_cfg.kmeansk,
        noise_log_priors=noise_log_priors,
    )

    step_labels = {}
    intermediate_split = "full" if return_step_labels else "kept"
    gmm.log_liks = None

    for it in range(refinement_cfg.n_total_iters):
        if refinement_cfg.truncated:
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
        skip_this_one = refinement_cfg.skip_first_split and not it
        too_many_units = (
            gmm.log_liks.shape[0]
            > refinement_cfg.max_avg_units * recording.get_num_channels()
        )
        if skip_this_one or too_many_units:
            if too_many_units:
                logger.dartsortdebug(
                    f"Skipping split ({gmm.log_liks.shape[0]} is too many units already)."
                )
            if refinement_cfg.one_split_only:
                break
        else:
            if refinement_cfg.refit_before_criteria:
                gmm.log_liks = gmm.em(
                    n_iter=1, force_refit=True, final_split=intermediate_split
                )
            gmm.split()
            gmm.log_liks = None

            if refinement_cfg.one_split_only:
                break

            if refinement_cfg.truncated:
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
        if refinement_cfg.refit_before_criteria:
            gmm.log_liks = gmm.em(
                n_iter=1, force_refit=True, final_split=intermediate_split
            )
        gmm.merge(gmm.log_liks)
        gmm.log_liks = None

        if return_step_labels:
            step_labels[f"refstep{it}cmerge"] = gmm.labels.numpy(force=True).copy()
        if saving:
            ds_save_intermediate_labels(
                save_step_labels_format.format(stepname=f"refstep{it}cmerge"),
                gmm.to_sorting(),
                save_step_labels_dir,
                save_cfg,
            )

    if refinement_cfg.truncated:
        gmm.tvi(final_split="full")
    else:
        gmm.em(final_split="full")
    sorting = gmm.to_sorting()
    gmm.tmm.processor.pool.shutdown()
    return sorting, step_labels


def split_merge(
    recording,
    sorting,
    motion_est=None,
    split_cfg=None,
    merge_cfg=None,
    merge_template_cfg=None,
    computation_cfg=None,
):
    if computation_cfg is None:
        computation_cfg = job_util.get_global_computation_config()

    split_sorting = split_clusters(
        sorting,
        split_strategy=split_cfg.split_strategy,
        recursive=split_cfg.recursive_split,
        n_jobs=computation_cfg.actual_n_jobs(),
        motion_est=motion_est,
    )

    merge_sorting = merge_templates(
        split_sorting,
        recording,
        motion_est=motion_est,
        template_config=merge_template_cfg,
        merge_distance_threshold=merge_cfg.merge_distance_threshold,
        min_spatial_cosine=merge_cfg.min_spatial_cosine,
        linkage=merge_cfg.linkage,
        computation_cfg=computation_cfg,
    )

    return merge_sorting


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
        temps_tpca = torch.asarray(template_data.templates[:, tpca.temporal_slice])

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


def pc_merge(sorting, refinement_cfg, motion_est=None, computation_cfg=None):
    assert refinement_cfg.refinement_strategy == "pcmerge"
    if computation_cfg is None:
        computation_cfg = job_util.get_global_computation_config()
    if not refinement_cfg.pc_merge_threshold:
        return sortings

    # remove blank labels just in case
    sorting = data_util.subset_sorting_by_spike_count(sorting)
    nu0 = sorting.labels.max() + 1
    if not nu0:
        return sorting

    # subset the sorting to count per unit
    subset_sorting = data_util.subsample_to_max_count(
        sorting, max_spikes=refinement_cfg.pc_merge_spikes_per_unit
    )

    # make stable features, no need for core features though.
    data = StableSpikeDataset.from_sorting(
        subset_sorting,
        motion_est=motion_est,
        core_radius=None,
        discard_triaged=True,
        interpolation_method=refinement_cfg.interpolation_method,
        extrap_method=refinement_cfg.extrapolation_method,
        extrap_kernel=refinement_cfg.extrapolation_kernel,
        kernel_name=refinement_cfg.kernel_name,
        sigma=refinement_cfg.interpolation_sigma,
        rq_alpha=refinement_cfg.rq_alpha,
        kriging_poly_degree=refinement_cfg.kriging_poly_degree,
        split_proportions=None,
        split_names=("train",),
        device=computation_cfg.actual_device(),
    )

    # average by unit
    kept = data.kept_indices
    n = len(kept)
    x = data._train_extract_features.view(n, -1, data.n_channels_extract)
    labels = torch.from_numpy(subset_sorting.labels[kept]).to(x.device)
    means = spiketorch.average_by_label(
        x, labels, data._train_extract_channels, data.n_channels
    )

    # compute distances
    if refinement_cfg.pc_merge_metric == "cosine":
        dists = spiketorch.cosine_distance(means)
    else:
        raise ValueError(f"Have not implemented {refinement_cfg.pc_merge_metric=}.")
    dists = dists.numpy(force=True)

    # linkage
    labels, ids = agglomerate(
        sorting.labels,
        dists,
        linkage_method=refinement_cfg.pc_merge_linkage,
        threshold=refinement_cfg.pc_merge_threshold,
    )
    logger.dartsortdebug(f"pc_merge: Unit count {nu0}->{ids.max() + 1}.")

    return replace(sorting, labels=labels)
