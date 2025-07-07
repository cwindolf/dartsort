import dataclasses
from dataclasses import field, fields
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict
import torch

from .py_util import int_or_inf, float_or_none
from .cli_util import argfield

try:
    from importlib.resources import files
except ImportError:
    try:
        from importlib_resources import files
    except ImportError:
        raise ValueError("Need python>=3.10 or pip install importlib_resources.")

default_pretrained_path = files("dartsort.pretrained")
default_pretrained_path = default_pretrained_path.joinpath("single_chan_denoiser.pt")
default_pretrained_path = str(default_pretrained_path)


_pydantic_strict_cfg = ConfigDict(strict=True, extra="forbid")


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class WaveformConfig:
    """Defaults yield 42 sample trough offset and 121 total at 30kHz."""

    ms_before: float = 1.4
    ms_after: float = 2.6 + 0.1 / 3

    @classmethod
    def from_samples(cls, samples_before, samples_after, sampling_frequency=30_000.0):
        samples_per_ms = sampling_frequency / 1000
        return cls(
            ms_before=samples_before / samples_per_ms,
            ms_after=samples_after / samples_per_ms,
        )

    def trough_offset_samples(self, sampling_frequency=30_000):
        sampling_frequency = np.round(sampling_frequency)
        return int(self.ms_before * (sampling_frequency / 1000))

    def spike_length_samples(self, sampling_frequency=30_000):
        spike_len_ms = self.ms_before + self.ms_after
        sampling_frequency = np.round(sampling_frequency)
        length = int(spike_len_ms * (sampling_frequency / 1000))
        # odd is better for convolution arithmetic elsewhere
        length = 2 * (length // 2) + 1
        return length

    def relative_slice(self, other, sampling_frequency=30_000):
        """My trough-aligned subset of samples in other, which contains me."""
        assert other.ms_before >= self.ms_before
        assert other.ms_after >= self.ms_after
        my_trough = self.trough_offset_samples(sampling_frequency)
        my_len = self.spike_length_samples(sampling_frequency)
        other_trough = other.trough_offset_samples(sampling_frequency)
        other_len = other.spike_length_samples(sampling_frequency)
        start_offset = other_trough - my_trough
        end_offset = (other_len - other_trough) - (my_len - my_trough)
        if start_offset == end_offset == 0:
            return slice(None)
        return slice(start_offset, other_len - end_offset)


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class FeaturizationConfig:
    """Featurization and denoising configuration

    Parameters for a featurization and denoising pipeline
    which has the flow:
    [input waveforms]
        -> [featurization of input waveforms]
        -> [denoising]
        -> [featurization of output waveforms]

    The flags below allow users to control which features
    are computed for the input waveforms, what denoising
    operations are applied, and what features are computed
    for the output (post-denoising) waveforms.

    Users who'd rather do something not covered by this
    typical case can manually instantiate a WaveformPipeline
    and pass it into their peeler.
    """

    skip: bool = False
    extract_radius: float = 100.0
    stop_after_n: int | None = None
    shuffle: bool = False

    # -- denoising configuration
    do_nn_denoise: bool = False
    do_tpca_denoise: bool = True
    do_enforce_decrease: bool = True
    # turn off features below
    denoise_only: bool = False

    # -- residual snips
    n_residual_snips: int = 4096
    residual_later: bool = False

    # -- featurization configuration
    save_input_voltages: bool = False
    save_input_waveforms: bool = False
    save_input_tpca_projs: bool = True
    save_output_waveforms: bool = False
    save_output_tpca_projs: bool = False
    save_amplitudes: bool = True
    save_all_amplitudes: bool = False
    # localization runs on output waveforms
    do_localization: bool = True
    localization_radius: float = 100.0
    # these are saved always if do_localization
    localization_amplitude_type: Literal["peak", "ptp"] = "peak"
    localization_model: Literal["pointsource", "dipole"] = "pointsource"
    nn_localization: bool = True
    additional_com_localization: bool = False
    localization_noise_floor: bool = False

    # -- further info about denoising
    # in the future we may add multi-channel or other nns
    nn_denoiser_class_name: str = "SingleChannelWaveformDenoiser"
    nn_denoiser_pretrained_path: str | None = default_pretrained_path
    nn_denoiser_train_epochs: int = 100
    nn_denoiser_epoch_size: int = 200 * 256
    nn_denoiser_extra_kwargs: dict | None = argfield(None, cli=False)

    # optionally restrict how many channels TPCA are fit on
    tpca_fit_radius: float = 75.0
    tpca_rank: int = 8
    tpca_centered: bool = False
    learn_cleaned_tpca_basis: bool = True
    input_tpca_waveform_cfg: WaveformConfig | None = WaveformConfig(
        ms_before=0.75, ms_after=1.25
    )
    tpca_max_waveforms: int = 20_000

    # used when naming datasets saved to h5 files
    input_waveforms_name: str = "collisioncleaned"
    output_waveforms_name: str = "denoised"


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class SubtractionConfig:
    # peeling common
    chunk_length_samples: int = 30_000
    n_seconds_fit: int = 100
    max_waveforms_fit: int = 50_000
    n_waveforms_fit: int = 20_000
    fit_subsampling_random_state: int = 0
    fit_sampling: str = "amp_reweighted"
    fit_max_reweighting: float = 4.0
    fit_only: bool = False

    # subtraction
    detection_threshold: float = 4.0
    peak_sign: Literal["pos", "neg", "both"] = "both"
    relative_peak_radius_um: float | None = 35.0
    spatial_dedup_radius: float | None = 100.0
    temporal_dedup_radius_samples: int = 11
    positive_temporal_dedup_radius_samples: int = 41
    subtract_radius: float = 200.0
    residnorm_decrease_threshold: float = 0.1 * 10**2
    growth_tolerance: float | None = 0.5
    trough_priority: float | None = 2.0
    use_singlechan_templates: bool = False
    singlechan_threshold: float = 50.0
    n_singlechan_templates: int = 10
    singlechan_alignment_padding_ms: float = 1.5

    # how will waveforms be denoised before subtraction?
    # users can also save waveforms/features during subtraction
    subtraction_denoising_cfg: FeaturizationConfig = FeaturizationConfig(
        denoise_only=True,
        do_nn_denoise=True,
        extract_radius=200.0,
        input_waveforms_name="raw",
        output_waveforms_name="subtracted",
    )

    # initial denoiser fitting parameters
    first_denoiser_max_waveforms_fit: int = 250_000
    first_denoiser_thinning: float = 0.5
    first_denoiser_temporal_jitter: int = 3
    first_denoiser_spatial_jitter: float = 35.0

    # for debugging / vis
    save_iteration: bool = False


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class TemplateConfig:
    spikes_per_unit: int = 500

    # -- template construction parameters
    # registered templates?
    registered_templates: bool = True

    # superresolved templates
    superres_templates: bool = False
    superres_bin_size_um: float = 10.0
    superres_bin_min_spikes: int = 5
    superres_strategy: str = "drift_pitch_loc_bin"
    adaptive_bin_size: bool = False

    # low rank denoising?
    low_rank_denoising: bool = True
    denoising_rank: int = 5
    denoising_snr_threshold: float = 50.0
    denoising_fit_radius: float = 75.0

    # realignment
    realign_peaks: bool = True
    realign_shift_ms: float = 1.5

    # track template over time
    time_tracking: bool = False
    chunk_size_s: int = 300


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class TemplateMergeConfig:
    linkage: str = "complete"
    merge_distance_threshold: float = 0.25
    cross_merge_distance_threshold: float = 0.5
    min_spatial_cosine: float = 0.5
    temporal_upsampling_factor: int = 4
    amplitude_scaling_variance: float = 0.1**2
    amplitude_scaling_boundary: float = 1.0
    svd_compression_rank: int = 20


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class MatchingConfig:
    # peeling common
    chunk_length_samples: int = 30_000
    n_seconds_fit: int = 100
    max_waveforms_fit: int = 50_000
    n_waveforms_fit: int = 20_000
    fit_subsampling_random_state: int = 0
    fit_sampling: str = "random"
    fit_max_reweighting: float = 4.0

    # template matching parameters
    threshold: float | Literal["fp_control"] = 10.0  # norm, not normsq
    template_svd_compression_rank: int = 10
    template_temporal_upsampling_factor: int = 4
    template_min_channel_amplitude: float = 0.0
    refractory_radius_frames: int = 10
    amplitude_scaling_variance: float = 0.1**2
    amplitude_scaling_boundary: float = 1.0
    max_iter: int = 1000
    conv_ignore_threshold: float = 5.0
    coarse_approx_error_threshold: float = 0.0
    coarse_objective: bool = True
    channel_selection_radius: float | None = 50.0

    # template postprocessing parameters
    min_template_snr: float = 15.0
    min_template_count: int = 50
    template_merge_cfg: TemplateMergeConfig | None = TemplateMergeConfig(
        merge_distance_threshold=0.025
    )
    precomputed_templates_npz: str | None = None


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class ThresholdingConfig:
    # peeling common
    chunk_length_samples: int = 30_000
    n_seconds_fit: int = 100
    max_waveforms_fit: int = 50_000
    n_waveforms_fit: int = 20_000
    fit_subsampling_random_state: int = 0
    fit_sampling: str = "random"
    fit_max_reweighting: float = 4.0

    # thresholding
    detection_threshold: float = 5.0
    max_spikes_per_chunk: int | None = None
    peak_sign: Literal["pos", "neg", "both"] = "both"
    spatial_dedup_radius: float = 150.0
    relative_peak_radius_um: float = 35.0
    relative_peak_radius_samples: int = 5
    dedup_temporal_radius_samples: int = 7
    thinning: float = 0.0
    time_jitter: int = 0
    spatial_jitter_radius: float = 0.0
    trough_priority: float | None = 2.0


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class UniversalMatchingConfig:
    # peeling common
    chunk_length_samples: int = 1_000
    n_seconds_fit: int = 100
    max_waveforms_fit: int = 50_000
    n_waveforms_fit: int = 20_000
    fit_subsampling_random_state: int = 0
    fit_sampling: str = "random"
    fit_max_reweighting: float = 4.0

    n_sigmas: int = 5
    n_centroids: int = 6
    threshold: float = 100.0
    detection_threshold: float = 6.0
    alignment_padding_ms: float = 1.5

    waveform_cfg: WaveformConfig = WaveformConfig(ms_before=0.75, ms_after=1.25)


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class MotionEstimationConfig:
    """Configure motion estimation."""

    do_motion_estimation: bool = True

    # sometimes spikes can be localized far away from the probe, causing
    # issues with motion estimation, we will ignore such spikes
    probe_boundary_padding_um: float = 100.0

    # DREDge parameters
    spatial_bin_length_um: float = 1.0
    temporal_bin_length_s: float = 1.0
    window_step_um: float = 400.0
    window_scale_um: float = 450.0
    window_margin_um: float | None = argfield(default=None, arg_type=float)
    max_dt_s: float = 1000.0
    max_disp_um: float | None = argfield(default=None, arg_type=float)
    correlation_threshold: float = 0.1
    min_amplitude: float | None = argfield(default=None, arg_type=float)
    rigid: bool = False


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class SplitConfig:
    split_strategy: str = "FeatureSplit"
    recursive_split: bool = True
    split_strategy_kwargs: dict | None = field(
        default_factory=lambda: dict(max_spikes=20_000)
    )


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class ClusteringFeaturesConfig:
    features_type: Literal["simple_matrix", "stable_waveforms"] = "simple_matrix"

    # simple matrix feature controls
    use_x: bool = True
    use_z: bool = True
    motion_aware: bool = True
    use_amplitude: bool = False
    log_transform_amplitude: bool = True
    amp_log_c: float = 5.0
    amp_scale: float = 50.0
    n_main_channel_pcs: int = 2
    pc_scale: float = 2.5
    adaptive_feature_scales: bool = False
    workers: int = 5

    amplitudes_dataset_name: str = "denoised_ptp_amplitudes"
    localizations_dataset_name: str = "point_source_localizations"
    pca_dataset_name: str = "collisioncleaned_tpca_features"

    interpolation_method: str = "kriging"
    kernel_name: str = "thinplate"
    interpolation_sigma: float = 10.0
    rq_alpha: float = 0.5
    kriging_poly_degree: int = 0


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class ClusteringConfig:
    cluster_strategy: str = "gmmdpc"

    # global parameters
    workers: int = 5
    random_seed: int = 0

    # density peaks parameters
    sigma_local: float = 5.0
    sigma_regional: float | None = argfield(default=25.0, arg_type=float_or_none)
    n_neighbors_search: int = 20
    radius_search: float = 25.0
    remove_clusters_smaller_than: int = 50
    noise_density: float = 0.0
    outlier_radius: float = 25.0
    outlier_neighbor_count: int = 10
    kdtree_subsample_max_size: int = 1_000_000

    # gmm density peaks additional parameters
    kmeanspp_initializations: int = 5
    kmeans_iter: int = 50
    components_per_channel: int = 20
    component_overlap: float = 0.8
    hellinger_strong: float = 0.0

    # hdbscan parameters
    min_cluster_size: int = 25
    min_samples: int = 25
    cluster_selection_epsilon: int = 1
    recursive: bool = False

    # grid snap parameters
    grid_dx: float = 15.0
    grid_dz: float = 15.0

    # sklearn clusterer params
    sklearn_class_name: str = "DBSCAN"
    sklearn_kwargs: dict | None = None


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class RefinementConfig:
    refinement_strategy: str = "gmm"

    # -- gmm parameters
    # noise params
    cov_kind: str = "factorizednoise"
    glasso_alpha: float | int | None = None

    # feature params
    max_avg_units: int = 3

    # model params
    channels_strategy: Literal["count", "all"] = "count"
    min_count: int = 50
    signal_rank: int = 0
    n_spikes_fit: int = 4096
    ppca_inner_em_iter: int = 5
    distance_metric: Literal["noise_metric", "kl", "reverse_kl", "symkl", "cosine"] = (
        "kl"
    )
    search_type: Literal["topk", "random"] = "topk"
    n_candidates: int = 3
    n_search: int | None = None
    distance_normalization_kind: Literal["none", "noise", "channels"] = "noise"
    merge_distance_threshold: float = 2.0
    # if None, switches to bimodality
    criterion_threshold: float | None = 0.0
    criterion: Literal["heldout_loglik", "heldout_elbo", "loglik", "elbo"] = (
        "heldout_elbo"
    )
    merge_bimodality_threshold: float = 0.05
    refit_before_criteria: bool = False
    n_em_iters: int = 50
    em_converged_prop: float = 0.02
    em_converged_churn: float = 0.01
    em_converged_atol: float = 1e-3
    n_total_iters: int = 3
    one_split_only: bool = False
    skip_first_split: bool = False
    hard_noise: bool = False
    truncated: bool = True
    split_decision_algorithm: str = "brute"
    merge_decision_algorithm: str = "brute"
    prior_pseudocount: float = 10.0
    prior_scales_mean: bool = False
    laplace_ard: bool = False
    kmeansk: int = 4

    # TODO... reintroduce this if wanted. or remove
    split_cfg: SplitConfig | None = None
    merge_cfg: TemplateMergeConfig | None = None
    merge_template_cfg: TemplateConfig | None = None

    # forward_backward
    chunk_size_s: float = 300.0
    log_c: float = 5.0
    feature_scales: tuple[float, float, float] = (1.0, 1.0, 50.0)
    adaptive_feature_scales: bool = False

    # stable waveform feature controls
    cov_radius: float = 500.0
    core_radius: float = 35.0
    val_proportion: float = 0.25
    max_n_spikes: float | int = argfield(default=2_000_000, arg_type=int_or_inf)
    interpolation_method: str = "kriging"
    extrapolation_method: str | None = "kernel"
    kernel_name: str = "thinplate"
    extrapolation_kernel: str | None = "rq"
    interpolation_sigma: float = 10.0
    rq_alpha: float = 0.5
    kriging_poly_degree: int = 0


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class ComputationConfig:
    n_jobs_cpu: int = 0
    n_jobs_gpu: int = 0
    executor: str = "threading_unless_multigpu"
    device: str | None = argfield(default=None, arg_type=str)

    @classmethod
    def from_n_jobs(cls, n_jobs):
        return cls(n_jobs_cpu=n_jobs, n_jobs_gpu=n_jobs)

    def actual_device(self):
        if self.device is None:
            have_cuda = torch.cuda.is_available()
            if have_cuda:
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.device)

    def actual_n_jobs(self):
        if self.actual_device().type == "cuda":
            return self.n_jobs_gpu
        return self.n_jobs_cpu

    def is_multi_gpu(self):
        if self.n_jobs_gpu in (0, 1):
            return False
        dev = self.actual_device()
        if dev.type != "cuda":
            return False
        if dev.index is not None:
            return False
        return torch.cuda.device_count() > 1


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class DARTsortInternalConfig:
    """This is an internal object. Make a DARTsortUserConfig, not one of these."""

    waveform_cfg: WaveformConfig = WaveformConfig()
    featurization_cfg: FeaturizationConfig = FeaturizationConfig(
        learn_cleaned_tpca_basis=True
    )
    initial_detection_cfg: (
        SubtractionConfig
        | MatchingConfig
        | ThresholdingConfig
        | UniversalMatchingConfig
    ) = SubtractionConfig()
    template_cfg: TemplateConfig = TemplateConfig()
    clustering_cfg: ClusteringConfig = ClusteringConfig()
    clustering_features_cfg: ClusteringFeaturesConfig = ClusteringFeaturesConfig()
    initial_refinement_cfg: RefinementConfig = RefinementConfig(one_split_only=True)
    refinement_cfg: RefinementConfig = RefinementConfig(skip_first_split=True)
    matching_cfg: MatchingConfig = MatchingConfig()
    motion_estimation_cfg: MotionEstimationConfig = MotionEstimationConfig()
    computation_cfg: ComputationConfig = ComputationConfig()

    # high level behavior
    detect_only: bool = False
    dredge_only: bool = False
    detection_type: Literal["subtract", "match", "threshold", "universal"] = "subtract"
    final_refinement: bool = True
    matching_iterations: int = 1
    recluster_after_first_matching: bool = False
    intermediate_matching_subsampling: float = 1.0
    overwrite_matching: bool = False

    # development / debugging flags
    work_in_tmpdir: bool = False
    tmpdir_parent: str | Path | None = None
    save_intermediate_labels: bool = False
    save_intermediate_features: bool = True
    save_final_features: bool = True
    save_everything_on_error: bool = False


def to_internal_config(cfg):
    from dartsort.config import DARTsortUserConfig, DeveloperConfig

    if isinstance(cfg, DARTsortInternalConfig):
        return cfg
    else:
        assert isinstance(cfg, (DARTsortUserConfig, DeveloperConfig))

    # if we have a user cfg, dump into dev cfg, and work from there
    if isinstance(cfg, DARTsortUserConfig):
        cfg = DeveloperConfig(**dataclasses.asdict(cfg))

    waveform_cfg = WaveformConfig(ms_before=cfg.ms_before, ms_after=cfg.ms_after)
    tpca_waveform_cfg = WaveformConfig(
        ms_before=cfg.feature_ms_before, ms_after=cfg.feature_ms_after
    )
    featurization_cfg = FeaturizationConfig(
        tpca_rank=cfg.temporal_pca_rank,
        extract_radius=cfg.featurization_radius_um,
        input_tpca_waveform_cfg=tpca_waveform_cfg,
        localization_radius=cfg.localization_radius_um,
        tpca_fit_radius=cfg.fit_radius_um,
        tpca_max_waveforms=cfg.n_waveforms_fit,
        save_input_waveforms=cfg.save_collisioncleaned_waveforms,
        learn_cleaned_tpca_basis=True,
    )

    if cfg.detection_type == "subtract":
        subtraction_denoising_cfg = FeaturizationConfig(
            denoise_only=True,
            extract_radius=cfg.subtraction_radius_um,
            do_nn_denoise=cfg.use_nn_in_subtraction,
            do_tpca_denoise=cfg.do_tpca_denoise,
            tpca_rank=cfg.temporal_pca_rank,
            tpca_fit_radius=cfg.fit_radius_um,
            input_waveforms_name="raw",
            output_waveforms_name="subtracted",
            save_input_waveforms=cfg.save_subtracted_waveforms,
            nn_denoiser_class_name=cfg.nn_denoiser_class_name,
            nn_denoiser_pretrained_path=cfg.nn_denoiser_pretrained_path,
        )
        initial_detection_cfg = SubtractionConfig(
            detection_threshold=cfg.initial_threshold,
            spatial_dedup_radius=cfg.deduplication_radius_um,
            subtract_radius=cfg.subtraction_radius_um,
            singlechan_alignment_padding_ms=cfg.alignment_ms,
            use_singlechan_templates=cfg.use_singlechan_templates,
            residnorm_decrease_threshold=cfg.denoiser_badness_factor
            * (cfg.matching_threshold**2),
            chunk_length_samples=cfg.chunk_length_samples,
            first_denoiser_thinning=cfg.first_denoiser_thinning,
            first_denoiser_max_waveforms_fit=cfg.nn_denoiser_max_waveforms_fit,
            subtraction_denoising_cfg=subtraction_denoising_cfg,
        )
    elif cfg.detection_type == "threshold":
        initial_detection_cfg = ThresholdingConfig(
            detection_threshold=cfg.initial_threshold,
            spatial_dedup_radius=cfg.deduplication_radius_um,
            chunk_length_samples=cfg.chunk_length_samples,
        )
    elif cfg.detection_type == "match":
        assert cfg.precomputed_templates_npz is not None
        initial_detection_cfg = MatchingConfig(
            threshold=cfg.matching_threshold,
            amplitude_scaling_variance=cfg.amplitude_scaling_stddev**2,
            amplitude_scaling_boundary=cfg.amplitude_scaling_limit,
            template_temporal_upsampling_factor=cfg.temporal_upsamples,
            chunk_length_samples=cfg.chunk_length_samples,
            precomputed_templates_npz=cfg.precomputed_templates_npz,
        )
    elif cfg.detection_type == "universal":
        initial_detection_cfg = UniversalMatchingConfig(
            waveform_cfg=tpca_waveform_cfg,
            threshold=cfg.denoiser_badness_factor * (cfg.matching_threshold**2),
        )
    else:
        raise ValueError(f"Unknown detection_type {cfg.detection_type}.")

    template_cfg = TemplateConfig(
        denoising_fit_radius=cfg.fit_radius_um, realign_shift_ms=cfg.alignment_ms
    )
    clustering_cfg = ClusteringConfig(
        cluster_strategy=cfg.cluster_strategy,
        sigma_local=cfg.density_bandwidth,
        sigma_regional=5 * cfg.density_bandwidth,
        outlier_radius=5 * cfg.density_bandwidth,
        radius_search=5 * cfg.density_bandwidth,
        remove_clusters_smaller_than=cfg.min_cluster_size,
        workers=cfg.clustering_workers,
        hellinger_strong=cfg.hellinger_strong,
        kdtree_subsample_max_size=cfg.clustering_max_spikes,
    )
    clustering_features_cfg = ClusteringFeaturesConfig(
        use_amplitude=cfg.initial_amp_feat,
        n_main_channel_pcs=cfg.initial_pc_feats,
        pc_scale=cfg.initial_pc_scale,
        motion_aware=cfg.motion_aware_clustering,
        workers=cfg.clustering_workers,
    )

    skip_step1_first_split = cfg.initial_split_only and not cfg.resume_with_split
    dist_thresh = 0.5 if cfg.gmm_metric == "cosine" else 2.0
    refinement_cfg = RefinementConfig(
        refinement_strategy=cfg.refinement_strategy,
        min_count=cfg.min_cluster_size,
        signal_rank=cfg.signal_rank,
        criterion=cfg.criterion,
        criterion_threshold=cfg.criterion_threshold,
        merge_bimodality_threshold=cfg.merge_bimodality_threshold,
        n_total_iters=cfg.n_refinement_iters,
        n_em_iters=cfg.n_em_iters,
        max_n_spikes=cfg.gmm_max_spikes,
        val_proportion=cfg.gmm_val_proportion,
        channels_strategy=cfg.channels_strategy,
        truncated=cfg.truncated,
        distance_metric=cfg.gmm_metric,
        search_type=cfg.gmm_search,
        n_candidates=cfg.gmm_n_candidates,
        n_search=cfg.gmm_n_search,
        merge_distance_threshold=dist_thresh,
        split_decision_algorithm=cfg.gmm_split_decision_algorithm,
        merge_decision_algorithm=cfg.gmm_merge_decision_algorithm,
        prior_pseudocount=cfg.prior_pseudocount,
        laplace_ard=cfg.laplace_ard,
        cov_kind=cfg.cov_kind,
        glasso_alpha=cfg.glasso_alpha,
        core_radius=cfg.core_radius,
        interpolation_method=cfg.interpolation_method,
        extrapolation_method=cfg.extrapolation_method,
        kernel_name=cfg.interpolation_kernel,
        interpolation_sigma=cfg.interpolation_bandwidth,
        rq_alpha=cfg.interpolation_rq_alpha,
        kriging_poly_degree=cfg.interpolation_degree,
        skip_first_split=skip_step1_first_split,
        kmeansk=cfg.kmeansk,
        prior_scales_mean=cfg.prior_scales_mean,
    )
    if cfg.initial_rank is None:
        irank = refinement_cfg.signal_rank
    else:
        irank = cfg.initial_rank
    initial_refinement_cfg = dataclasses.replace(
        refinement_cfg,
        one_split_only=cfg.initial_split_only,
        skip_first_split=False,
        signal_rank=irank,
    )
    motion_estimation_cfg = MotionEstimationConfig(
        **{k.name: getattr(cfg, k.name) for k in fields(MotionEstimationConfig)}
    )
    matching_cfg = MatchingConfig(
        threshold="fp_control" if cfg.matching_fp_control else cfg.matching_threshold,
        amplitude_scaling_variance=cfg.amplitude_scaling_stddev**2,
        amplitude_scaling_boundary=cfg.amplitude_scaling_limit,
        template_temporal_upsampling_factor=cfg.temporal_upsamples,
        chunk_length_samples=cfg.chunk_length_samples,
        template_merge_cfg=TemplateMergeConfig(
            merge_distance_threshold=cfg.postprocessing_merge_threshold,
        ),
    )
    computation_cfg = ComputationConfig(
        n_jobs_cpu=cfg.n_jobs_cpu,
        n_jobs_gpu=cfg.n_jobs_gpu,
        device=cfg.device,
        executor=cfg.executor,
    )

    return DARTsortInternalConfig(
        waveform_cfg=waveform_cfg,
        featurization_cfg=featurization_cfg,
        initial_detection_cfg=initial_detection_cfg,
        template_cfg=template_cfg,
        clustering_cfg=clustering_cfg,
        initial_refinement_cfg=initial_refinement_cfg,
        refinement_cfg=refinement_cfg,
        matching_cfg=matching_cfg,
        clustering_features_cfg=clustering_features_cfg,
        motion_estimation_cfg=motion_estimation_cfg,
        computation_cfg=computation_cfg,
        detection_type=cfg.detection_type,
        dredge_only=cfg.dredge_only,
        matching_iterations=cfg.matching_iterations,
        recluster_after_first_matching=cfg.recluster_after_first_matching,
        overwrite_matching=cfg.overwrite_matching,
        work_in_tmpdir=cfg.work_in_tmpdir,
        tmpdir_parent=cfg.tmpdir_parent,
        save_intermediate_labels=cfg.save_intermediate_labels,
        save_intermediate_features=cfg.save_intermediate_features,
        save_final_features=cfg.save_final_features,
        save_everything_on_error=cfg.save_everything_on_error,
    )


# default configs, used as defaults for kwargs in main.py etc
default_waveform_cfg = WaveformConfig()
default_featurization_cfg = FeaturizationConfig(learn_cleaned_tpca_basis=True)
default_subtraction_cfg = SubtractionConfig()
default_thresholding_cfg = ThresholdingConfig()
default_template_cfg = TemplateConfig()
default_clustering_cfg = ClusteringConfig()
default_clustering_features_cfg = ClusteringFeaturesConfig()
default_matching_cfg = MatchingConfig()
default_motion_estimation_cfg = MotionEstimationConfig()
default_computation_cfg = ComputationConfig()
default_dartsort_cfg = DARTsortInternalConfig()
default_refinement_cfg = RefinementConfig()
default_universal_cfg = UniversalMatchingConfig()

# configs which are commonly used for specific tasks
coarse_template_cfg = TemplateConfig(superres_templates=False)
raw_template_cfg = TemplateConfig(
    realign_peaks=False, low_rank_denoising=False, superres_templates=False
)
unshifted_raw_template_cfg = TemplateConfig(
    registered_templates=False,
    realign_peaks=False,
    low_rank_denoising=False,
    superres_templates=False,
)
unaligned_coarse_denoised_template_cfg = TemplateConfig(
    realign_peaks=False, low_rank_denoising=True, superres_templates=False
)
waveforms_only_featurization_cfg = FeaturizationConfig(
    do_tpca_denoise=False,
    do_enforce_decrease=False,
    n_residual_snips=0,
    save_input_tpca_projs=False,
    save_amplitudes=False,
    do_localization=False,
    input_waveforms_name="raw",
    save_input_voltages=True,
    save_input_waveforms=True,
)
