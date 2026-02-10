from dataclasses import replace
import gc

import torch
import h5py

from ..util.internal_config import default_refinement_cfg
from ..util import job_util, noise_util, data_util, spiketorch
from ..util.logging_util import get_logger
from ..util.main_util import ds_save_intermediate_labels
from ..transform.temporal_pca import BaseTemporalPCA
from ..templates import TemplateData
from .cluster_util import agglomerate, reorder_by_depth
from .split import split_clusters
from .merge import merge_templates
from .gmm.stable_features import StableSpikeDataset


logger = get_logger(__name__)


def split_merge(
    *,
    recording,
    sorting,
    motion_est=None,
    split_cfg,
    merge_cfg,
    merge_template_cfg,
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
    template_data = TemplateData.from_config(
        recording=recording,
        sorting=sorting,
        motion_est=motion_est,
        template_cfg=merge_template_cfg,
    )
    merge_sorting = merge_templates(
        sorting=split_sorting,
        template_data=template_data,
        merge_distance_threshold=merge_cfg.merge_distance_threshold,
        min_spatial_cosine=merge_cfg.min_spatial_cosine,
        linkage=merge_cfg.linkage,
        computation_cfg=computation_cfg,
    )["sorting"]

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
            matching_labels = h5["labels"][:]  # type: ignore

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


def pc_merge(sorting, refinement_cfg, motion_est=None, computation_cfg=None):
    assert refinement_cfg.refinement_strategy == "pcmerge"
    if computation_cfg is None:
        computation_cfg = job_util.get_global_computation_config()
    if not refinement_cfg.pc_merge_threshold:
        return sorting

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
        interp_params=refinement_cfg.interp_params.normalize(),
        split_proportions=None,
        split_names=("train",),
        device=computation_cfg.actual_device(),
    )

    # average by unit
    kept = data.kept_indices
    n = len(kept)
    x = data._train_extract_features.view(n, -1, data.n_channels_extract)
    x = x[:, : refinement_cfg.pc_merge_rank]
    labels = torch.from_numpy(subset_sorting.labels[kept]).to(x.device)
    means, counts = spiketorch.average_by_label(
        x, labels, data._train_extract_channels, data.n_channels
    )

    # compute distances
    if refinement_cfg.pc_merge_metric == "cosine":
        dists = spiketorch.cosine_distance(means)
    elif refinement_cfg.pc_merge_metric == "maxz":
        x = x.square_()
        meansq, _ = spiketorch.average_by_label(
            x, labels, data._train_extract_channels, data.n_channels
        )
        stddev = meansq.sub_(means.square()).sqrt_()
        stddev = stddev.clamp_(min=torch.finfo(stddev.dtype).tiny)
        stderr = stddev.div_(counts.sqrt()[:, None])
        dists = spiketorch.maxz_distance(
            means, stderr, counts, min_iou=refinement_cfg.pc_merge_min_iou
        )
    elif refinement_cfg.pc_merge_metric == "normeuc":
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
    labels, ids = agglomerate(
        sorting.labels,
        dists,
        linkage_method=refinement_cfg.pc_merge_linkage,
        threshold=refinement_cfg.pc_merge_threshold,
    )
    logger.dartsortdebug(f"pc_merge: Unit count {nu0}->{ids.max() + 1}.")

    sorting = sorting.ephemeral_replace(labels=labels)
    sorting = reorder_by_depth(sorting, motion_est=motion_est)
    return sorting

