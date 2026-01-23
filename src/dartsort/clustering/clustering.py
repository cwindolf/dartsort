import gc
from typing import cast, Literal, Self

import numpy as np
import sklearn.cluster
import torch
from spikeinterface.core import BaseRecording

from ..util.data_util import chunk_time_ranges, fit_reweighting, DARTsortSorting
from ..util.main_util import ds_save_intermediate_labels
from ..util.internal_config import (
    ClusteringConfig,
    RefinementConfig,
    FitSamplingConfig,
    ComputationConfig,
)
from .clustering_features import SimpleMatrixFeatures
from ..util import job_util
from . import cluster_util, density, forward_backward, refine_util
from .gmm import mixture


clustering_strategies: dict[str, "type[Clusterer]"] = {}
refinement_strategies: dict[str, "type[Refinement]"] = {}


def get_clusterer(
    clustering_cfg: ClusteringConfig | None = None,
    refinement_cfg: RefinementConfig | None = None,
    pre_refinement_cfg: RefinementConfig | None = None,
    computation_cfg: ComputationConfig | None = None,
    save_cfg=None,
    save_labels_dir=None,
    initial_name=None,
    refine_labels_fmt=None,
):
    if clustering_cfg is not None:
        if clustering_cfg.cluster_strategy not in clustering_strategies:
            raise ValueError(
                f"Unknown cluster_strategy={clustering_cfg.cluster_strategy}. "
                f"Options are: {', '.join(clustering_strategies.keys())}."
            )
        clus_strategy = clustering_cfg.cluster_strategy
    else:
        clus_strategy = "none"

    saving_labels = save_cfg is not None and save_cfg.save_intermediate_labels
    if saving_labels:
        shared_save_kw = dict(
            save_labels_dir=save_labels_dir,
            save_cfg=save_cfg,
        )
    else:
        shared_save_kw = dict(
            save_labels_dir=None,
            save_cfg=None,
        )

    C = clustering_strategies[clus_strategy]
    init_fmt = initial_name if (saving_labels and clustering_cfg is not None) else None
    clusterer = C.from_config(
        clustering_cfg,
        labels_fmt=init_fmt,
        computation_cfg=computation_cfg,
        **shared_save_kw,
    )

    if pre_refinement_cfg is not None:
        pr_strategy = pre_refinement_cfg.refinement_strategy
        if pr_strategy not in refinement_strategies:
            raise ValueError(
                f"Unknown refinement_strategy={pre_refinement_cfg.refinement_strategy}. "
                f"Options are: {', '.join(refinement_strategies.keys())}."
            )
        R = refinement_strategies[pre_refinement_cfg.refinement_strategy]
        if saving_labels and clustering_cfg is not None:
            mid_fmt = f"{initial_name}_preref{pr_strategy}"
        else:
            mid_fmt = None
        clusterer = R(
            clusterer,
            refinement_cfg=pre_refinement_cfg,
            labels_fmt=mid_fmt,
            computation_cfg=computation_cfg,
            **shared_save_kw,
        )

    if refinement_cfg is not None:
        if refinement_cfg.refinement_strategy not in refinement_strategies:
            raise ValueError(
                f"Unknown refinement_strategy={refinement_cfg.refinement_strategy}. "
                f"Options are: {', '.join(refinement_strategies.keys())}."
            )
        R = refinement_strategies[refinement_cfg.refinement_strategy]
        rsave = saving_labels and refinement_cfg.refinement_strategy != "none"
        clusterer = R(
            clusterer,
            refinement_cfg=refinement_cfg,
            labels_fmt=refine_labels_fmt if rsave else None,
            computation_cfg=computation_cfg,
            **shared_save_kw,
        )

    return clusterer


class Clusterer:
    def __init__(
        self,
        computation_cfg: ComputationConfig | None = None,
        sampling_cfg: FitSamplingConfig | None = None,
        save_cfg=None,
        save_labels_dir=None,
        labels_fmt=None,
    ):
        self.computation_cfg = computation_cfg
        self.sampling_cfg = sampling_cfg
        if computation_cfg is None:
            self.computation_cfg = job_util.get_global_computation_config()
        self.save_cfg = save_cfg
        self.save_labels_dir = save_labels_dir
        self.labels_fmt = labels_fmt

    @classmethod
    def from_config(
        cls,
        clustering_cfg: ClusteringConfig | None,
        computation_cfg: ComputationConfig | None = None,
        save_cfg=None,
        save_labels_dir=None,
        labels_fmt=None,
    ) -> Self:
        if clustering_cfg is None:
            sampling_cfg = None
        else:
            sampling_cfg = clustering_cfg.sampling_cfg
        return cls(
            computation_cfg=computation_cfg,
            save_cfg=save_cfg,
            save_labels_dir=save_labels_dir,
            labels_fmt=labels_fmt,
            sampling_cfg=sampling_cfg,
        )

    def handle_sampling(
        self, features: SimpleMatrixFeatures
    ) -> tuple[Literal[False], slice] | tuple[Literal[True], np.ndarray]:
        if self.sampling_cfg is None:
            return False, slice(None)
        elif features.features.shape[0] <= self.sampling_cfg.n_waveforms_fit:
            return False, slice(None)

        weights = fit_reweighting(
            voltages=features.signed_amplitudes,
            fit_sampling=self.sampling_cfg.fit_sampling,
            fit_max_reweighting=self.sampling_cfg.fit_max_reweighting,
        )
        rg = np.random.default_rng(self.sampling_cfg.fit_subsampling_random_state)
        ixs = rg.choice(
            features.features.shape[0],
            size=self.sampling_cfg.n_waveforms_fit,
            p=weights,
            replace=False,
        )
        ixs.sort()
        return True, ixs

    def cluster(
        self,
        features: SimpleMatrixFeatures | None,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ):
        if features is None:
            pass
        else:
            labels = self._cluster(features, sorting, recording, motion_est)
            sorting = sorting.ephemeral_replace(labels=labels)
        if self.labels_fmt and self.save_labels_dir is not None:
            assert "{" not in self.labels_fmt
            assert "}" not in self.labels_fmt
            ds_save_intermediate_labels(
                self.labels_fmt, sorting, self.save_labels_dir, self.save_cfg
            )
        return sorting

    def _cluster(
        self,
        features: SimpleMatrixFeatures,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ) -> np.ndarray:
        """Unused method but shows API."""
        del features, recording, motion_est
        assert sorting.labels is not None
        return sorting.labels


clustering_strategies["none"] = Clusterer


class ChannelSnapClusterer(Clusterer):
    def _cluster(
        self,
        features: SimpleMatrixFeatures,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ) -> np.ndarray:
        return cluster_util.closest_registered_channels(
            times_seconds=sorting.times_seconds,  # type: ignore
            x=features.x,
            z_abs=features.z,
            z_reg=features.z_reg,
            geom=recording.get_channel_locations(),
            motion_est=motion_est,
        )


clustering_strategies["channel_snap"] = ChannelSnapClusterer


class GridSnapClusterer(Clusterer):
    def __init__(self, grid_dx=15.0, grid_dz=15.0, **kwargs):
        super().__init__(**kwargs)
        self.grid_dx = grid_dx
        self.grid_dz = grid_dz

    @classmethod
    def from_config(
        cls,
        clustering_cfg: ClusteringConfig | None,
        computation_cfg: ComputationConfig | None = None,
        save_cfg=None,
        save_labels_dir=None,
        labels_fmt=None,
    ) -> Self:
        assert clustering_cfg is not None
        return cls(
            grid_dx=clustering_cfg.grid_dx,
            grid_dz=clustering_cfg.grid_dz,
            computation_cfg=computation_cfg,
            save_cfg=save_cfg,
            save_labels_dir=save_labels_dir,
            labels_fmt=labels_fmt,
            sampling_cfg=clustering_cfg.sampling_cfg,
        )

    def _cluster(
        self,
        features: SimpleMatrixFeatures,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ) -> np.ndarray:
        return cluster_util.grid_snap(
            times_seconds=sorting.times_seconds,  # type: ignore
            x=features.x,
            z_abs=features.z,
            z_reg=features.z_reg,
            grid_dx=self.grid_dx,
            grid_dz=self.grid_dz,
            geom=recording.get_channel_locations(),
            motion_est=motion_est,
        )


clustering_strategies["grid_snap"] = GridSnapClusterer


class DensityPeaksClusterer(Clusterer):
    def __init__(
        self,
        knn_k=None,
        sigma_local=5.0,
        sigma_regional: float | None = 25.0,
        n_neighbors_search=20,
        radius_search=25.0,
        remove_clusters_smaller_than=50,
        noise_density=0.0,
        outlier_radius=5.0,
        outlier_neighbor_count=5,
        workers=-1,
        uhdversion=False,
        random_seed=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.knn_k = knn_k
        self.uhdversion = uhdversion
        self.sigma_local = sigma_local
        self.sigma_regional = sigma_regional
        self.n_neighbors_search = n_neighbors_search
        self.radius_search = radius_search
        self.remove_clusters_smaller_than = remove_clusters_smaller_than
        self.noise_density = noise_density
        self.outlier_radius = outlier_radius
        self.outlier_neighbor_count = outlier_neighbor_count
        self.workers = workers
        self.uhdversion = uhdversion
        self.random_seed = random_seed

    @classmethod
    def from_config(
        cls,
        clustering_cfg: ClusteringConfig | None,
        computation_cfg: ComputationConfig | None = None,
        save_cfg=None,
        save_labels_dir=None,
        labels_fmt=None,
    ) -> Self:
        assert clustering_cfg is not None
        uhdversion = clustering_cfg.cluster_strategy == "density_peaks_uhdversion"
        return cls(
            knn_k=clustering_cfg.knn_k,
            sigma_local=clustering_cfg.sigma_local,
            sigma_regional=clustering_cfg.sigma_regional,
            n_neighbors_search=clustering_cfg.n_neighbors_search,
            radius_search=clustering_cfg.radius_search,
            remove_clusters_smaller_than=clustering_cfg.min_cluster_size,
            noise_density=clustering_cfg.noise_density,
            random_seed=clustering_cfg.random_seed,
            outlier_radius=clustering_cfg.outlier_radius,
            outlier_neighbor_count=clustering_cfg.outlier_neighbor_count,
            workers=clustering_cfg.workers,
            uhdversion=uhdversion,
            computation_cfg=computation_cfg,
            save_cfg=save_cfg,
            save_labels_dir=save_labels_dir,
            labels_fmt=labels_fmt,
            sampling_cfg=clustering_cfg.sampling_cfg,
        )

    def _cluster(
        self,
        features: SimpleMatrixFeatures,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ) -> np.ndarray:
        subsampling, ixs = self.handle_sampling(features)
        X = features.features
        X_fit = X[ixs]

        if not self.uhdversion:
            res = density.density_peaks(
                X_fit,
                knn_k=self.knn_k,
                sigma_local=self.sigma_local,
                sigma_regional=self.sigma_regional,
                n_neighbors_search=self.n_neighbors_search,
                radius_search=self.radius_search,
                remove_clusters_smaller_than=0,
                noise_density=self.noise_density,
                outlier_radius=self.outlier_radius,
                outlier_neighbor_count=self.outlier_neighbor_count,
                workers=self.workers,
            )
        else:
            res = density.density_peaks_fancy(
                features.xyza,
                features.amplitudes,
                sorting,
                motion_est,
                recording.get_channel_locations(),
                sigma_local=self.sigma_local,
                sigma_regional=self.sigma_regional,
                n_neighbors_search=self.n_neighbors_search,
                radius_search=self.radius_search,
                remove_clusters_smaller_than=self.remove_clusters_smaller_than,
                noise_density=self.noise_density,
                outlier_radius=self.outlier_radius,
                outlier_neighbor_count=self.outlier_neighbor_count,
                workers=self.workers,
            )

        if subsampling:
            kdtree = res["kdtree"]
            assert isinstance(ixs, np.ndarray)
            rest = np.setdiff1d(np.arange(len(X)), ixs)
            other_labels = density.nearest_neighbor_assign(
                kdtree,
                res["labels"],
                X[rest],
                radius_search=self.radius_search,
                workers=self.workers,
            )
            labels = cluster_util.combine_disjoint(
                ixs, res["labels"], rest, other_labels
            )
        else:
            labels = res["labels"]

        labels = cluster_util.decrumb(
            labels, min_size=self.remove_clusters_smaller_than, in_place=True
        )

        return labels


clustering_strategies["dpc"] = DensityPeaksClusterer
clustering_strategies["density_peaks_uhdversion"] = DensityPeaksClusterer


class GMMDensityPeaksClusterer(Clusterer):
    def __init__(
        self,
        outlier_neighbor_count=10,
        outlier_radius=25.0,
        remove_clusters_smaller_than=50,
        workers=-1,
        n_initializations=5,
        n_iter=50,
        max_components_per_channel=20,
        min_spikes_per_component=10,
        random_state=0,
        use_hellinger=True,
        kmeanspp_min_dist=5.0,
        hellinger_cutoff=0.8,
        hellinger_strong=0.0,
        hellinger_weak=0.0,
        mop=False,
        n_neighbors_search=20,
        max_sigma=5.0,
        max_samples=2_000_000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.outlier_neighbor_count = outlier_neighbor_count
        self.outlier_radius = outlier_radius
        self.remove_clusters_smaller_than = remove_clusters_smaller_than
        self.workers = workers
        self.n_initializations = n_initializations
        self.n_iter = n_iter
        self.max_components_per_channel = max_components_per_channel
        self.min_spikes_per_component = min_spikes_per_component
        self.random_state = random_state
        self.use_hellinger = use_hellinger
        self.hellinger_cutoff = hellinger_cutoff
        self.hellinger_strong = hellinger_strong
        self.hellinger_weak = hellinger_weak
        self.mop = mop
        self.max_samples = max_samples
        self.n_neighbors_search = n_neighbors_search

        # unconfigured params
        self.kmeanspp_min_dist = kmeanspp_min_dist
        self.max_sigma = max_sigma

    @classmethod
    def from_config(
        cls,
        clustering_cfg: ClusteringConfig | None,
        computation_cfg: ComputationConfig | None = None,
        save_cfg=None,
        save_labels_dir=None,
        labels_fmt=None,
    ) -> Self:
        assert clustering_cfg is not None
        return cls(
            outlier_neighbor_count=clustering_cfg.outlier_neighbor_count,
            outlier_radius=clustering_cfg.outlier_radius,
            remove_clusters_smaller_than=clustering_cfg.min_cluster_size,
            workers=clustering_cfg.workers,
            n_initializations=clustering_cfg.kmeanspp_initializations,
            n_iter=clustering_cfg.kmeans_iter,
            max_components_per_channel=clustering_cfg.components_per_channel,
            random_state=clustering_cfg.random_seed,
            use_hellinger=clustering_cfg.use_hellinger,
            hellinger_cutoff=clustering_cfg.component_overlap,
            hellinger_strong=clustering_cfg.hellinger_strong,
            hellinger_weak=clustering_cfg.hellinger_weak,
            mop=clustering_cfg.mop,
            max_sigma=clustering_cfg.gmmdpc_max_sigma,
            max_samples=clustering_cfg.sampling_cfg.n_waveforms_fit,
            n_neighbors_search=clustering_cfg.n_neighbors_search,
            computation_cfg=computation_cfg,
            save_cfg=save_cfg,
            save_labels_dir=save_labels_dir,
            labels_fmt=labels_fmt,
            sampling_cfg=clustering_cfg.sampling_cfg,
        )

    def _cluster(
        self,
        features: SimpleMatrixFeatures,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ) -> np.ndarray:
        res = density.gmm_density_peaks(
            X=features.features,
            channels=sorting.channels,
            outlier_neighbor_count=self.outlier_neighbor_count,
            outlier_radius=self.outlier_radius,
            remove_clusters_smaller_than=self.remove_clusters_smaller_than,
            workers=self.workers,
            n_initializations=self.n_initializations,
            n_iter=self.n_iter,
            max_components_per_channel=self.max_components_per_channel,
            min_spikes_per_component=self.min_spikes_per_component,
            random_state=self.random_state,
            kmeanspp_min_dist=self.kmeanspp_min_dist,
            use_hellinger=self.use_hellinger,
            hellinger_cutoff=self.hellinger_cutoff,
            hellinger_strong=self.hellinger_strong,
            hellinger_weak=self.hellinger_weak,
            n_neighbors_search=self.n_neighbors_search,
            mop=self.mop,
            max_sigma=self.max_sigma,
            max_samples=self.max_samples,
        )
        return res["labels"]


clustering_strategies["gmmdpc"] = GMMDensityPeaksClusterer


class RecursiveHDBSCANClusterer(Clusterer):
    def __init__(
        self,
        min_cluster_size=25,
        min_samples=25,
        cluster_selection_epsilon=1,
        recursive=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.recursive = recursive

    @classmethod
    def from_config(
        cls,
        clustering_cfg: ClusteringConfig | None,
        computation_cfg: ComputationConfig | None = None,
        save_cfg=None,
        save_labels_dir=None,
        labels_fmt=None,
    ) -> Self:
        assert clustering_cfg is not None
        return cls(
            min_cluster_size=clustering_cfg.min_cluster_size,
            min_samples=clustering_cfg.min_samples,
            cluster_selection_epsilon=clustering_cfg.cluster_selection_epsilon,
            recursive=clustering_cfg.recursive,
            computation_cfg=computation_cfg,
            save_cfg=save_cfg,
            save_labels_dir=save_labels_dir,
            labels_fmt=labels_fmt,
            sampling_cfg=clustering_cfg.sampling_cfg,
        )

    def _cluster(
        self,
        features: SimpleMatrixFeatures,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ) -> np.ndarray:
        return cluster_util.recursive_hdbscan_clustering(
            features.features,
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            recursive=self.recursive,
        )


class ScikitLearnClusterer(Clusterer):
    def __init__(self, sklearn_class_name="DBSCAN", sklearn_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.sklearn_class_name = sklearn_class_name
        self.sklearn_kwargs = sklearn_kwargs or {}

    @classmethod
    def from_config(
        cls,
        clustering_cfg: ClusteringConfig | None,
        computation_cfg: ComputationConfig | None = None,
        save_cfg=None,
        save_labels_dir=None,
        labels_fmt=None,
    ) -> Self:
        assert clustering_cfg is not None
        return cls(
            sklearn_class_name=clustering_cfg.sklearn_class_name,
            sklearn_kwargs=clustering_cfg.sklearn_kwargs,
            computation_cfg=computation_cfg,
            save_cfg=save_cfg,
            save_labels_dir=save_labels_dir,
            labels_fmt=labels_fmt,
            sampling_cfg=clustering_cfg.sampling_cfg,
        )

    def _cluster(
        self,
        features: SimpleMatrixFeatures,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ) -> np.ndarray:
        skcls = getattr(sklearn.cluster, self.sklearn_class_name)
        clus = skcls(**self.sklearn_kwargs)
        return clus.fit_predict(features.features)


clustering_strategies["sklearn"] = ScikitLearnClusterer


class Refinement(Clusterer):
    def __init__(
        self, clusterer: Clusterer, refinement_cfg: RefinementConfig, **kwargs
    ):
        super().__init__(**kwargs)
        self.clusterer = clusterer
        self.refinement_cfg = refinement_cfg
        self.sampling_cfg = refinement_cfg.sampling_cfg

    def cluster(
        self,
        features: SimpleMatrixFeatures | None,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ):
        sorting = self.clusterer.cluster(features, sorting, recording, motion_est)
        sorting = self.refine(features, sorting, recording, motion_est)
        return sorting

    def _refine(
        self,
        features: SimpleMatrixFeatures | None,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ):
        del features, recording, motion_est
        return sorting

    def refine(
        self,
        features: SimpleMatrixFeatures | None,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ):
        sorting = self._refine(features, sorting, recording, motion_est)
        if self.labels_fmt and self.save_labels_dir is not None:
            labels_fmt = self.labels_fmt.format(stepname="")
            ds_save_intermediate_labels(
                labels_fmt, sorting, self.save_labels_dir, self.save_cfg
            )
        return sorting


refinement_strategies["none"] = Refinement


class GMMRefinement(Refinement):
    def _refine(
        self,
        features: SimpleMatrixFeatures | None,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ):
        sorting, _ = refine_util.gmm_refine(
            recording,
            sorting,
            motion_est=motion_est,
            refinement_cfg=self.refinement_cfg,
            computation_cfg=self.computation_cfg,
            save_step_labels_format=self.labels_fmt,
            save_step_labels_dir=self.save_labels_dir,
            save_cfg=self.save_cfg,
        )
        return sorting


refinement_strategies["gmm"] = GMMRefinement


class TMMRefinement(Refinement):
    def _refine(
        self,
        features: SimpleMatrixFeatures | None,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ):
        assert features is not None
        subsampling, ixs = self.handle_sampling(features)
        ixs = cast(np.ndarray, ixs) if subsampling else None
        sorting = mixture.tmm_demix(
            sorting=sorting,
            motion_est=motion_est,
            refinement_cfg=self.refinement_cfg,
            computation_cfg=self.computation_cfg,
            fit_indices=ixs,
            save_step_labels_format=self.labels_fmt,
            save_step_labels_dir=self.save_labels_dir,
            save_cfg=self.save_cfg,
        )
        gc.collect()
        torch.cuda.empty_cache()
        return sorting


refinement_strategies["tmm"] = TMMRefinement


class SplitMergeRefinement(Refinement):
    def _refine(
        self,
        features: SimpleMatrixFeatures | None,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ):
        return refine_util.split_merge(
            recording=recording,
            sorting=sorting,
            motion_est=motion_est,
            split_cfg=self.refinement_cfg.split_cfg,
            merge_cfg=self.refinement_cfg.merge_cfg,
            merge_template_cfg=self.refinement_cfg.merge_template_cfg,
            computation_cfg=self.computation_cfg,
        )


# code too old. todo.
# refinement_strategies["splitmerge"] = SplitMergeRefinement


class PCMergeRefinement(Refinement):
    def _refine(
        self,
        features: SimpleMatrixFeatures | None,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ):
        return refine_util.pc_merge(
            sorting,
            refinement_cfg=self.refinement_cfg,
            motion_est=motion_est,
            computation_cfg=self.computation_cfg,
        )


refinement_strategies["pcmerge"] = PCMergeRefinement


class ForwardBackwardEnsembler(Refinement):
    """If there are more time chunk ones, make a new ABC with this logic."""

    def cluster(
        self,
        features: SimpleMatrixFeatures | None,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        motion_est=None,
    ):
        chunk_length_samples = (
            recording.sampling_frequency * self.refinement_cfg.chunk_size_s
        )
        chunk_time_ranges_s = chunk_time_ranges(recording, chunk_length_samples)
        times_seconds = sorting.times_seconds  # type: ignore
        assert features is not None

        chunk_sortings = []
        for lo, hi in chunk_time_ranges_s:
            mask = np.flatnonzero(times_seconds == times_seconds.clip(lo, hi))
            s = sorting.mask(mask)
            f = features.mask(mask)
            l = self.clusterer._cluster(f, s, recording, motion_est)
            labels = np.full_like(sorting.labels, -1)
            labels[mask] = l
            chunk_sortings.append(sorting.ephemeral_replace(labels=labels))

        labels = forward_backward.forward_backward(
            chunk_time_ranges_s,
            chunk_sortings,
            log_c=self.refinement_cfg.log_c,
            feature_scales=self.refinement_cfg.feature_scales,
            adaptive_feature_scales=self.refinement_cfg.adaptive_feature_scales,
            motion_est=motion_est,
        )
        sorting = sorting.ephemeral_replace(labels=labels)
        return sorting


refinement_strategies["forwardbackward"] = ForwardBackwardEnsembler
