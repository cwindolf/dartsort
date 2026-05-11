from collections.abc import Sequence
from dataclasses import asdict, fields, replace
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Literal, Self

import numpy as np
import torch

from .cli_util import argfield, dataclass_from_toml
from .py_util import cfg_dataclass, float_or_none, resolve_path

try:
    from importlib.resources import files
except ImportError:
    try:
        from importlib_resources import files
    except ImportError:
        raise ValueError("Need python>=3.10 or pip install importlib_resources.")

_default_pretrained_path: Traversable = files("dartsort.pretrained")
_default_pretrained_path = _default_pretrained_path.joinpath("single_chan_denoiser.pt")
default_pretrained_path = str(_default_pretrained_path)


PreprocessingStrategy = Literal["none", "ibllike", "ibllikecmr"] | str


@cfg_dataclass
class WaveformConfig:
    """Defaults yield 42 sample trough offset and 121 total at 30kHz."""

    ms_before: float = 1.4
    ms_after: float = 2.6 + 0.1 / 3

    @classmethod
    def from_samples(
        cls,
        samples_before: int,
        samples_after: int,
        sampling_frequency: float = 30_000.0,
    ) -> Self:
        sampling_frequency = float(sampling_frequency)
        samples_per_ms = sampling_frequency / 1000
        self = cls(
            ms_before=samples_before / samples_per_ms,
            ms_after=samples_after / samples_per_ms,
        )
        assert self.trough_offset_samples(sampling_frequency) == samples_before
        samples_total = samples_before + samples_after
        if not samples_total % 2:
            raise ValueError(f"{samples_before=} plus {samples_after=} should be odd.")
        assert self.spike_length_samples(sampling_frequency) == samples_total
        return self

    @staticmethod
    def ms_to_samples(ms, sampling_frequency: float = 30_000.0):
        if ms > sampling_frequency:
            return int((ms / 1000.0) * sampling_frequency)
        else:
            return int(ms * (sampling_frequency / 1000.0))

    def trough_offset_samples(self, sampling_frequency: float = 30_000.0):
        sampling_frequency = np.round(sampling_frequency)
        return self.ms_to_samples(self.ms_before, sampling_frequency=sampling_frequency)

    def spike_length_samples(self, sampling_frequency: float = 30_000.0):
        spike_len_ms = self.ms_before + self.ms_after
        sampling_frequency = np.round(sampling_frequency)
        length = self.ms_to_samples(spike_len_ms, sampling_frequency=sampling_frequency)
        # odd is better for convolution arithmetic elsewhere
        length = 2 * (length // 2) + 1
        return length

    def relative_slice(
        self, other: Self, sampling_frequency: float = 30_000.0
    ) -> slice:
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

    def pad(self, padding_ms: float) -> Self:
        return self.__class__(
            ms_before=self.ms_before + padding_ms,
            ms_after=self.ms_after + padding_ms,
        )


InterpMethod = Literal[
    "kriging",
    "kernel",
    "normalized",
    "krigingnormalized",
    "zero",
    "nearest",
    "nan",
    "clampna",
]
InterpKernel = Literal[
    "zero",
    "nearest",
    "idw",
    "rbf",
    "multiquadric",
    "rq",
    "thinplate",
    "nan",
    "clampna",
    "polyharmonic",
]
_kmethods = {"zero", "nearest", "nan", "clampna"}


@cfg_dataclass
class InterpolationParams:
    method: InterpMethod = "kriging"
    kernel: InterpKernel = "thinplate"
    extrap_method: InterpMethod | None = None
    extrap_kernel: InterpKernel | None = None
    kriging_poly_degree: int = 1
    sigma: float = 10.0
    rq_alpha: float = 0.5
    smoothing_lambda: float = 0.0
    neighborhood_radius: float = 200.0
    polyharmonic_order: int | float = 2

    @property
    def actual_extrap_method(self):
        if self.extrap_method is None:
            return self.method
        return self.extrap_method

    @property
    def actual_extrap_kernel(self):
        if self.extrap_kernel is None:
            return self.kernel
        return self.extrap_kernel

    def extrap_diff(self):
        if self.actual_extrap_method != self.method:
            return True
        if self.actual_extrap_kernel != self.kernel:
            return True
        return False

    def normalize(self) -> Self:
        method = self.method
        kernel = self.kernel
        if method in _kmethods:
            kernel = method
            method = "kernel"

        extrap_method = self.extrap_method
        extrap_kernel = self.extrap_kernel
        if extrap_method in _kmethods:
            extrap_kernel = extrap_method
            extrap_method = "kernel"

        return self.__class__(
            method=method,
            kernel=kernel,  # type: ignore
            extrap_method=extrap_method,
            extrap_kernel=extrap_kernel,  # type: ignore
            kriging_poly_degree=self.kriging_poly_degree,
            sigma=self.sigma,
            rq_alpha=self.rq_alpha,
            smoothing_lambda=self.smoothing_lambda,
            neighborhood_radius=self.neighborhood_radius,
            polyharmonic_order=self.polyharmonic_order,
        )


tps_interp_params = InterpolationParams()
clampna_interp_params = InterpolationParams(method="clampna")
tps_interp_clampna_extrap_params = InterpolationParams(extrap_method="clampna")
default_extrapolation_params = InterpolationParams(
    method="kernel", kernel="rq", sigma=10.0
)

FitSamplingMethod = Literal["random", "amp_reweighted"]
default_fit_sampling_method = "amp_reweighted"
default_fit_max_reweighting = 4.0


@cfg_dataclass
class FitSamplingConfig:
    max_waveforms_fit: int = 50_000
    n_waveforms_fit: int = 40_000
    more_waveforms_fit: int = 2000 * 1024
    n_residual_snips: int = 4 * 4096
    residual_sampling_target_density: float = 0.25
    seed: int = 0
    chunk_sampling: Literal["random", "kmeanspp"] = "kmeanspp"
    fit_sampling: FitSamplingMethod = "amp_reweighted"
    fit_max_reweighting: float = default_fit_max_reweighting
    n_seconds_fit: int = 100


default_peeling_fit_sampling_cfg = FitSamplingConfig()
default_clustering_fit_sampling_cfg = FitSamplingConfig(
    max_waveforms_fit=500_000, n_waveforms_fit=500_000
)


@cfg_dataclass
class ClusteringFeaturesConfig:
    # simple matrix feature controls
    use_x: bool = True
    use_z: bool = True
    motion_aware: bool = True
    use_amplitude: bool = False
    use_signed_amplitude: bool = True
    log_transform_amplitude: bool = True
    amp_log_c: float = 5.0
    amp_scale: float = 3.0
    x_scale: float = 1.0
    n_main_channel_pcs: int = 5
    pc_scale: float = 2.0
    pc_transform: Literal["log", "sqrt", "none"] | None = "none"
    pc_pre_transform_scale: float = 0.5
    adaptive_feature_scales: bool = False

    # stable feature controls
    feature_rank: int = 8

    # interpolation, drift handling
    interp_params: InterpolationParams = tps_interp_params
    motion_depth_mode: Literal["channel", "localization"] = "channel"

    # attribute name registry
    amplitudes_dataset_name: str = "denoised_ptp_amplitudes"
    voltages_dataset_name: str = "collisioncleaned_voltages"
    amplitude_vectors_dataset_name: str = "denoised_ptp_amplitude_vectors"
    localizations_dataset_name: str = "point_source_localizations"
    pca_dataset_name: str = "collisioncleaned_tpca_features"


@cfg_dataclass
class ClusteringConfig:
    cluster_strategy: str = "dpc"
    sampling_cfg: FitSamplingConfig = default_clustering_fit_sampling_cfg

    # global parameters
    random_seed: int = 0
    min_cluster_size: int = 25

    # density peaks parameters
    knn_k: int | None = None
    sigma_local: float = 5.0
    sigma_regional: float | None = argfield(default=25.0, arg_type=float_or_none)
    n_neighbors_search: int = 50
    radius_search: float = 25.0
    noise_density: float = 0.0
    outlier_radius: float = 25.0
    outlier_neighbor_count: int = 10

    # gmm density peaks additional parameters
    kmeanspp_initializations: int = 10
    kmeans_iter: int = 100
    components_per_channel: int = 20
    component_overlap: float = 0.95
    hellinger_strong: float = 0.0
    hellinger_weak: float = 0.0
    use_hellinger: bool = True
    gmmdpc_max_sigma: float = 5.0
    mop: bool = True

    # hdbscan parameters
    min_samples: int = 25
    cluster_selection_epsilon: int = 1
    recursive: bool = False

    # grid snap parameters
    grid_dx: float = 15.0
    grid_dz: float = 15.0

    # sklearn clusterer params
    sklearn_class_name: str = "DBSCAN"
    sklearn_kwargs: dict | None = None


WhiteningStrategy = Literal["none", "prewhiten", "prewhiten_postapply", "postwhiten"]
WhiteningEstimator = Literal["fullzca", "localzca", "sparsechol"]


@cfg_dataclass
class WhiteningConfig:
    strategy: WhiteningStrategy = "none"
    estimator: WhiteningEstimator = "localzca"
    interp_params: InterpolationParams = tps_interp_clampna_extrap_params
    radius: float = 200.0


TemplateSVDMethod = Literal[
    "collisioncleaned", "spike_sklearn", "peeler", "raw_template"
]


@cfg_dataclass
class TemplateConfig:
    spikes_per_unit: int = 500
    with_raw_std_dev: bool = False
    reduction: Literal["median", "mean"] = "median"
    algorithm: (
        Literal[
            "unitextract",
            "peelreduce",
            "peelreduce_if_mean",
        ]
        | str
    ) = "peelreduce"
    denoising_method: Literal["none", "exp_weighted", "svd"] = "svd"
    weighted: bool = False
    grab_chunk_length_samples: int = 30_000
    units_per_job: int = 8
    whitening: WhiteningConfig = WhiteningConfig()

    # -- template construction parameters
    # registered templates?
    registered_templates: bool = True
    min_fraction_at_shift: float = 0.25
    min_count_at_shift: int = 25
    template_interp_params: InterpolationParams = tps_interp_clampna_extrap_params

    # low rank denoising?
    denoising_rank: int = 5
    denoising_fit_radius: float = 75.0
    denoising_fit_sampling_cfg: FitSamplingConfig = replace(
        default_peeling_fit_sampling_cfg, n_residual_snips=0
    )
    template_min_channel_amplitude: float = 1.0
    svd_method: TemplateSVDMethod = "raw_template"
    svd_alignment_iterations: int = 0
    svd_alignment_ms: float = 0.75

    # exp weight denoising
    exp_weight_snr_threshold: float = 50.0

    # where to find data if needed
    amplitudes_dataset_name: str = "denoised_ptp_amplitudes"
    localizations_dataset_name: str = "point_source_localizations"

    @property
    def use_svd(self) -> bool:
        return self.denoising_method in ("svd", "exp_weighted")

    def actual_algorithm(self) -> str:
        if self.algorithm.endswith("_if_mean") and self.reduction == "mean":
            return self.algorithm.removesuffix("_if_mean")
        elif self.algorithm.endswith("_if_mean"):
            return "unitextract"
        else:
            return self.algorithm


raw_template_cfg = TemplateConfig(
    reduction="mean",
    denoising_method="none",
    template_interp_params=clampna_interp_params,
)

RealignStrategy = Literal[
    "mainchan_trough_factor",
    "dredge",
    "snr_weighted_trough_factor",
    "normsq_weighted_trough_factor",
    "ampsq_weighted_trough_factor",
    "mainchan_svd_trough_factor",
    "snr_weighted_svd_trough_factor",
    "normsq_weighted_svd_trough_factor",
    "ampsq_weighted_svd_trough_factor",
]


@cfg_dataclass
class TemplateRealignmentConfig:
    realign_peaks: bool = True
    realign_strategy: RealignStrategy = "snr_weighted_trough_factor"
    realign_shift_ms: float = 1.5
    trough_factor: float = 3.0
    template_cfg: TemplateConfig = raw_template_cfg
    min_pair_corr: float = 0.8


@cfg_dataclass
class TemplateMergeConfig:
    distance_kind: Literal[
        "scaled_normeuc", "deconv", "max", "weighted_scaled_normeuc"
    ] = "weighted_scaled_normeuc"
    linkage: str = "complete"
    merge_distance_threshold: float = 0.05
    cross_merge_distance_threshold: float = 0.5
    min_spatial_cosine: float = 0.75
    temporal_upsampling_factor: int = 4
    amplitude_scaling_variance: float = 0.01**2
    amplitude_scaling_boundary: float = 1.0 / 3.0
    svd_compression_rank: int = 10
    max_shift_ms: float = 1.5
    weighted_dist_min_iou: float = 0.75
    weighted_dist_radius: float = 100.0

    template_cfg: TemplateConfig | None = TemplateConfig()
    waveform_cfg: WaveformConfig = WaveformConfig()
    whitening: WhiteningConfig = WhiteningConfig()

    def to_template_config(self, template_cfg: TemplateConfig | None = None):
        if template_cfg is None:
            assert self.template_cfg is not None
            template_cfg = self.template_cfg
        return replace(
            template_cfg,
            denoising_method="svd",
            denoising_rank=self.svd_compression_rank,
            whitening=self.whitening,
        )


MixtureStep = Literal["split", "merge", "demolish"]
ComponentDistanceMetric = Literal["cosine", "normeuc", "scaled_normeuc"]


@cfg_dataclass
class RefinementConfig:
    refinement_strategy: str = "tmm"
    sampling_cfg: FitSamplingConfig = default_clustering_fit_sampling_cfg

    # pcmerge
    pc_merge_threshold: float = 0.1
    pc_merge_metric: str = "normeuc"
    pc_merge_spikes_per_unit: int = 4096
    pc_merge_linkage: str = "complete"
    pc_merge_rank: int = 5
    pc_merge_min_iou: float = 0.95

    # -- gmm parameters
    # noise params
    cov_kind: str = "factorizednoise"
    glasso_alpha: float | int | None = None

    # model params
    neighb_overlap: float = 0.75
    explore_neighb_steps: int = 0
    min_count: int = 25
    split_min_count: int = 8
    channels_count_min: int = 1
    signal_rank: int = 3
    initialize_at_rank_0: bool = False
    cl_alpha: float = 0.05
    cl_split_only: bool = True
    latent_prior_std: float = 1.0
    initial_basis_shrinkage: float = 1.0
    n_spikes_fit: int = 4096
    distance_metric: ComponentDistanceMetric = "scaled_normeuc"
    n_candidates: int = 5
    merge_group_size: int = 5
    n_search: int | None = 3
    n_explore: int | None = None
    train_batch_size: int = 512
    eval_batch_size: int = 512
    split_friend_distance: float = 0.8
    split_distance_threshold: float = 1.5
    merge_distance_threshold: float = 1.5
    criterion_em_iters: int = 3
    hold_out_criterion: bool = True
    n_em_iters: int = 250
    em_converged_atol: float = 5e-3
    n_total_iters: int = 1
    mixture_steps: Sequence[MixtureStep] = ("split", "merge", "demolish")
    prior_pseudocount: float = 0.0
    kmeansk: int = 4
    kmeans_tries: int = 5
    kmeanspp_tries: int = 5
    full_proposal_every: int = 10
    main_min_iters: int = 20
    search_adj: Literal["top", "explore"] = "top"
    robust_strategy: Literal["none", "fixed"] = "none"
    robust_fixed_std_dataset: str = "collidedness"
    robust_fixed_power: float = 40.0
    robust_df: float = 4.0
    demolition_min_resp_ratio: float = 1.1
    demolish_during_selection: bool = False
    em_after_demolish: bool = False
    whiten_split: bool = True
    scale_dist_args: tuple[float, float, float] = (0.01, 3.0 / 4.0, 4.0 / 3.0)
    whiten_dist: bool = True

    # template merge parameters
    template_merge_cfg: TemplateMergeConfig = TemplateMergeConfig(linkage="single")

    # other agglomeration parameters
    glom_max_firing_corr: float | None = -0.1
    glom_firing_corr_dt: float = 0.5
    glom_firing_corr_method: Literal["binsqrt"] = "binsqrt"
    qda_link: Literal["single", "complete"] = "single"
    qda_uni_score: float = 0.95
    qda_threshold: float = 0.35
    qda_min_ratio: float = 0.1
    qda_min_coverage: float = 0.35
    qda_min_iou: float = 0.5
    qda_force_merge_for_temp_dist_below: float = 0.3

    # forward_backward parameters
    chunk_size_s: float = 300.0
    log_c: float = 5.0
    feature_scales: tuple[float, float, float] = (1.0, 1.0, 50.0)
    adaptive_feature_scales: bool = False

    # stable waveform feature controls
    cov_radius: float = 200.0
    val_proportion: float = 0.5
    impute_kind: Literal["interp", "impute"] = "impute"
    noise_interp_params: InterpolationParams = tps_interp_clampna_extrap_params


@cfg_dataclass
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

    # -- denoising configuration
    do_nn_denoise: bool = False
    do_tpca_denoise: bool = True
    do_enforce_decrease: bool | Literal["loc_only"] = "loc_only"
    # turn off features below
    denoise_only: bool = False

    # -- featurization configuration
    save_input_voltages: bool = True
    save_input_waveforms: bool = False
    save_input_tpca_projs: bool = True
    compute_input_tpca_projs_regardless: bool = False
    save_input_tpca_projs: bool = True
    save_output_waveforms: bool = False
    save_output_tpca_projs: bool = False
    save_collidedness: bool = False
    save_amplitudes: bool = True
    save_all_amplitudes: bool = False
    # localization runs on output waveforms
    do_localization: bool = True
    localization_radius: float = 100.0
    # these are saved always if do_localization
    localization_amplitude_type: Literal["peak", "ptp"] = "peak"
    localization_decay_power: int = 1
    localization_model: Literal["pointsource", "dipole"] = "pointsource"
    nn_localization: bool = True
    additional_com_localization: bool = False
    localization_noise_floor: bool = False

    # -- further info about denoising
    nn_denoiser_class_name: str = "Decollider"
    nn_denoiser_pretrained_path: str | None = None
    nn_denoiser_train_epochs: int = 100
    nn_denoiser_epoch_size: int = 200 * 256
    nn_denoiser_extra_kwargs: dict | None = argfield(None, cli=False)

    # optionally restrict how many channels TPCA are fit on
    tpca_fit_radius: float = 75.0
    tpca_rank: int = 8
    tpca_centered: bool = False
    learn_cleaned_tpca_basis: bool = False
    input_tpca_waveform_cfg: WaveformConfig | None = WaveformConfig(
        ms_before=0.75, ms_after=1.25
    )
    tpca_max_waveforms: int = 40_000
    tpca_from_templates: bool = True

    # mixture model
    use_gmm_classifier: bool = False
    pre_gmm_clustering_cfg: ClusteringConfig | None = None
    pre_gmm_refinement_cfgs: Sequence[RefinementConfig | None] | None = None
    gmm_refinement_cfg: RefinementConfig | None = None
    gmm_clustering_features_cfg: ClusteringFeaturesConfig | None = None

    # used when naming datasets saved to h5 files
    input_waveforms_name: str = "collisioncleaned"
    output_waveforms_name: str = "denoised"


PeakSign = Literal["pos", "neg", "both"]


@cfg_dataclass
class SubtractionConfig:
    # peeling common
    chunk_length_samples: int = 30_000
    fit_only: bool = False

    # subtraction
    detection_threshold: float = 3.0
    peak_sign: PeakSign = "both"
    realign_to_denoiser: bool = True
    denoiser_realignment_channel: Literal["detection", "denoised"] = "detection"
    denoiser_realignment_shift: int = 5
    relative_peak_radius_samples: int = 5
    relative_peak_radius_um: float | None = 35.0
    spatial_dedup_radius_um: float | None = 50.0
    temporal_dedup_radius_samples: int = 11
    remove_exact_duplicates: bool = True
    positive_temporal_dedup_radius_samples: int = 41
    subtract_radius_um: float = 200.0
    residnorm_decrease_threshold: float = 10.0
    decrease_objective: Literal["norm", "normsq", "deconv"] = "deconv"
    growth_tolerance: float | None = None
    trough_priority: float | None = 2.0
    cumulant_order: int | None = None
    convexity_threshold: float | None = None
    convexity_radius: int = 7
    max_iter: int = 100

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
    first_denoiser_thinning: float = 0.0
    first_denoiser_temporal_jitter: int = 3
    first_denoiser_spatial_jitter: float = 35.0
    first_denoiser_spatial_dedup_radius: float = 100.0

    # for debugging / vis
    save_iteration: bool = False
    save_residnorm_decrease: bool = False


@cfg_dataclass
class ThresholdingConfig:
    # peeling common
    chunk_length_samples: int = 30_000

    # thresholding
    detection_threshold: float = 4.0
    max_spikes_per_chunk: int | None = None
    peak_sign: Literal["pos", "neg", "both"] = "both"
    spatial_dedup_radius_um: float = 150.0
    relative_peak_radius_um: float = 35.0
    relative_peak_radius_samples: int = 5
    temporal_dedup_radius_samples: int = 11
    remove_exact_duplicates: bool = True
    cumulant_order: int | None = None
    convexity_threshold: float | None = None
    convexity_radius: int = 7

    thinning: float = 0.0
    time_jitter: int = 0
    spatial_jitter_radius: float = 0.0
    trough_priority: float | None = 2.0
    shave_score: float = 10.0


@cfg_dataclass
class MatchingConfig:
    # peeling common
    chunk_length_samples: int = 30_000
    max_spikes_per_second: int = 16384
    cd_iter: int = 0
    coarse_cd: bool = True

    # template matching parameters
    threshold: float | Literal["fp_control"] = 8.0
    template_svd_compression_rank: int = 5
    up_factor: int = 4
    upsampling_radius: int = 8
    template_min_channel_amplitude: float = 1.0
    refractory_radius_frames: int = 0
    amplitude_scaling_variance: float = 0.01**2
    amplitude_scaling_boundary: float = 1.0 / 3.0
    max_iter: int = 100
    conv_ignore_threshold: float = 0.0
    coarse_approx_error_threshold: float = 0.0
    coarse_objective: bool = True
    channel_selection: Literal["template", "amplitude"] = "template"
    channel_selection_radius: float | None = None
    template_type: Literal["individual_compressed_upsampled", "drifty", "debug"] = (
        "drifty"
    )
    up_method: Literal["interpolation", "keys3", "keys4", "direct"] = "keys4"
    drift_interp_params: InterpolationParams = tps_interp_clampna_extrap_params
    upsampling_compression_map: Literal["yass", "none"] = "yass"
    whitening: WhiteningConfig = WhiteningConfig(strategy="prewhiten_postapply")
    whiten_features: bool = False
    margin_factor: int = 2
    max_fp_per_input_spike: float = 2.5

    # template postprocessing parameters
    min_template_ptp: float = 1.0
    always_keep_ptp: float = 10.0
    min_template_snr: float = 0.0
    min_template_count: int = 20
    max_cc_flag_rate: float = 0.4
    cc_flag_entropy_cutoff: float = 2.0
    depth_order: bool = True
    template_merge_cfg: TemplateMergeConfig | None = TemplateMergeConfig()
    template_realignment_cfg: TemplateRealignmentConfig = TemplateRealignmentConfig()
    precomputed_templates_npz: str | None = None
    delete_pconv: bool = True


@cfg_dataclass
class MotionEstimationConfig:
    """Configure motion estimation."""

    do_motion_estimation: bool = True

    # DREDge parameters
    probe_boundary_padding_um: float = 100.0
    spatial_bin_length_um: float = 1.0
    temporal_bin_length_s: float = 1.0
    smoothing_um: float | None = 3.0
    smoothing_s: float | None = None
    window_step_um: float = 400.0
    window_scale_um: float = 600.0
    window_margin_um: float | None = argfield(default=None, arg_type=float)
    max_dt_s: float = 500.0
    max_disp_um: float | None = argfield(
        default=None,
        arg_type=float,
        doc="Will be set to win_scale_um / 4 if left blank.",
    )
    correlation_threshold: float = 0.1
    weight_threshold: float = 0.2
    min_amplitude: float | None = argfield(default=None, arg_type=float)
    rigid: bool = False
    speed_limit_um_per_s: float = argfield(
        default=500.0,
        arg_type=float,
        doc="Motion bins exceeding this speed will be replaced by interpolation.",
    )
    max_dist_from_median_um: float = argfield(
        default=250.0,
        arg_type=float,
        doc="Motion bins farther than this from the local median will be replaced by interpolation.",
    )
    median_neighborhood_bins: int = 51

    # if spikes are needed, a thresholding detection is run
    tpca_rank: int = 8
    localization_radius_um: float = 100.0
    threshold_cfg: ThresholdingConfig = ThresholdingConfig()
    spike_denoising_score: float = 10.0


@cfg_dataclass
class ComputationConfig:
    n_jobs_cpu: int = 0
    n_jobs_gpu: int = 0
    n_jobs_small: int = -2
    n_jobs_small_gpu: int = 4
    executor: str = "threading_unless_multigpu"
    device: str | None = argfield(default=None, arg_type=str)

    @classmethod
    def from_n_jobs(cls, n_jobs, n_jobs_small=8):
        return cls(n_jobs_cpu=n_jobs, n_jobs_gpu=n_jobs, n_jobs_small=n_jobs_small)

    def actual_device(self):
        if self.device is None:
            have_cuda = torch.cuda.is_available()
            if have_cuda:
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.device)

    def actual_n_jobs(self, small: bool = False, cpu: bool = False):
        if cpu or self.actual_device().type == "cpu":
            if small:
                return self.n_jobs_small
            else:
                return self.n_jobs_cpu
        else:
            if small:
                return self.n_jobs_small_gpu
            else:
                return self.n_jobs_gpu

    def is_multi_gpu(self):
        if self.n_jobs_gpu in (0, 1):
            return False
        dev = self.actual_device()
        if dev.type != "cuda":
            return False
        if dev.index is not None:
            return False
        return torch.cuda.device_count() > 1


# default configs, used as defaults for kwargs in main.py etc
default_waveform_cfg = WaveformConfig()
default_featurization_cfg = FeaturizationConfig()
default_subtraction_cfg = SubtractionConfig()
default_thresholding_cfg = ThresholdingConfig()
default_template_cfg = TemplateConfig()
default_matching_template_cfg = TemplateConfig(
    whitening=WhiteningConfig(strategy="prewhiten_postapply")
)
default_clustering_cfg = ClusteringConfig()
default_clustering_features_cfg = ClusteringFeaturesConfig()
default_matching_cfg = MatchingConfig()
default_motion_estimation_cfg = MotionEstimationConfig()
default_computation_cfg = ComputationConfig()
default_refinement_cfg = RefinementConfig()
default_initial_refinement_cfg = RefinementConfig(mixture_steps=("split", "demolish"))
default_pre_refinement_cfg = RefinementConfig(refinement_strategy="pcmerge")
default_agglomerate_cfg = RefinementConfig(
    refinement_strategy="agglomerate",
    template_merge_cfg=TemplateMergeConfig(
        merge_distance_threshold=0.6, linkage="single"
    ),
)


@cfg_dataclass
class DARTsortInternalConfig:
    """This is an internal object. Make a DARTsortUserConfig, not one of these."""

    waveform_cfg: WaveformConfig = default_waveform_cfg
    featurization_cfg: FeaturizationConfig = default_featurization_cfg
    peeler_sampling_cfg: FitSamplingConfig = default_peeling_fit_sampling_cfg
    initial_detection_cfg: SubtractionConfig | MatchingConfig | ThresholdingConfig = (
        default_subtraction_cfg
    )
    template_cfg: TemplateConfig = default_matching_template_cfg
    clustering_cfg: ClusteringConfig = default_clustering_cfg
    clustering_features_cfg: ClusteringFeaturesConfig = default_clustering_features_cfg
    initial_refinement_cfg: RefinementConfig = default_initial_refinement_cfg
    pre_refinement_cfg: RefinementConfig | None = default_pre_refinement_cfg
    refinement_cfg: RefinementConfig = default_refinement_cfg
    post_refinement_cfg: RefinementConfig | None = default_pre_refinement_cfg
    agglomerate_cfg: RefinementConfig | None = default_agglomerate_cfg
    matching_cfg: MatchingConfig = default_matching_cfg
    motion_estimation_cfg: MotionEstimationConfig = default_motion_estimation_cfg
    computation_cfg: ComputationConfig = default_computation_cfg

    # high level behavior
    detect_only: bool = False
    dredge_only: bool = False
    detection_type: Literal["subtract", "match", "threshold"] = "subtract"
    preprocessing: PreprocessingStrategy = "none"
    preprocessing_dtype: Literal["float16", "float32"] = "float32"
    final_refinement: bool = True
    matching_iterations: int = 1
    recluster_after_first_matching: bool = False
    # subsampling: intermediate peels will continue until both criteria satisfied
    # need at least this many spikes
    subsampling_spikes: int | None = 2_048_000
    # need to cover at least this fraction of chunks
    subsampling_presence: float = 0.1

    # development / debugging flags
    work_in_tmpdir: bool = False
    copy_recording_to_tmpdir: bool = False
    workdir_follow_symlinks: bool = False
    workdir_copier: Literal["shutil", "rsync"] = "shutil"
    tmpdir_parent: str | None = None
    link_from: str | None = None
    link_step: Literal["denoising", "detection", "refined0", "matching1"] = "refined0"
    save_intermediate_labels: bool = False
    save_intermediate_features: bool = False
    save_final_features: bool = True
    always_save_final_tpca_feature: bool = False
    save_everything_on_error: bool = False


def to_internal_config(cfg) -> DARTsortInternalConfig:
    """Laundromat of configuration formats

    Arguments
    ---------
    cfg : str | Path | DARTsortUserConfig | DeveloperConfig
        If str or Path, it should point to a .toml file.

    Returns
    -------
    DARTsortInternalConfig
    """
    from dartsort.config import DARTsortUserConfig, DeveloperConfig

    if isinstance(cfg, (str, Path)):
        # load toml config
        cfg0 = cfg

        try:
            cfg = resolve_path(cfg, strict=True)
        except OSError as e:
            raise ValueError(f"Configuration file {cfg0} does not exist.") from e

        try:
            cfg = dataclass_from_toml((DeveloperConfig,), cfg)
        except Exception as e:
            raise ValueError(
                f"Could not read configuration from {cfg0}. More error info above."
            ) from e

    if isinstance(cfg, DARTsortInternalConfig):
        return cfg
    else:
        assert isinstance(cfg, (DARTsortUserConfig, DeveloperConfig))

    # if we have a user cfg, dump into dev cfg, and work from there
    if isinstance(cfg, DARTsortUserConfig):
        cfg = DeveloperConfig(**asdict(cfg))

    waveform_cfg = WaveformConfig(ms_before=cfg.ms_before, ms_after=cfg.ms_after)
    tpca_waveform_cfg = WaveformConfig(
        ms_before=cfg.feature_ms_before, ms_after=cfg.feature_ms_after
    )
    save_collidedness = cfg.robust_strategy == "fixed"
    featurization_cfg = FeaturizationConfig(
        tpca_rank=cfg.temporal_pca_rank,
        extract_radius=cfg.featurization_radius_um,
        input_tpca_waveform_cfg=tpca_waveform_cfg,
        localization_radius=cfg.localization_radius_um,
        tpca_fit_radius=cfg.fit_radius_um,
        tpca_max_waveforms=cfg.n_waveforms_fit,
        save_input_waveforms=cfg.save_collisioncleaned_waveforms,
        save_collidedness=save_collidedness,
    )
    if cfg.dredge_only:
        n_residual_snips = 0
    else:
        n_residual_snips = cfg.n_residual_snips
    peeler_fit_sampling_cfg = FitSamplingConfig(
        n_waveforms_fit=cfg.n_waveforms_fit,
        max_waveforms_fit=cfg.max_waveforms_fit,
        more_waveforms_fit=cfg.gmm_max_spikes,
        fit_sampling=cfg.fit_sampling,
        n_residual_snips=n_residual_snips,
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
            save_output_waveforms=cfg.save_subtracted_waveforms,
            nn_denoiser_class_name=cfg.nn_denoiser_class_name,
            nn_denoiser_pretrained_path=cfg.nn_denoiser_pretrained_path,
        )
        initial_detection_cfg = SubtractionConfig(
            peak_sign=cfg.peak_sign,
            detection_threshold=cfg.voltage_threshold,
            spatial_dedup_radius_um=cfg.deduplication_radius_um,
            subtract_radius_um=cfg.subtraction_radius_um,
            realign_to_denoiser=cfg.realign_to_denoiser,
            residnorm_decrease_threshold=cfg.initial_threshold,
            chunk_length_samples=cfg.chunk_length_samples,
            first_denoiser_thinning=cfg.first_denoiser_thinning,
            first_denoiser_max_waveforms_fit=cfg.nn_denoiser_max_waveforms_fit,
            first_denoiser_spatial_dedup_radius=cfg.first_denoiser_spatial_dedup_radius,
            subtraction_denoising_cfg=subtraction_denoising_cfg,
        )
    elif cfg.detection_type == "threshold":
        initial_detection_cfg = ThresholdingConfig(
            peak_sign=cfg.peak_sign,
            detection_threshold=cfg.voltage_threshold,
            spatial_dedup_radius_um=cfg.deduplication_radius_um,
            chunk_length_samples=cfg.chunk_length_samples,
        )
    elif cfg.detection_type == "match":
        assert cfg.precomputed_templates_npz is not None
        initial_detection_cfg = MatchingConfig(
            threshold=cfg.matching_threshold,
            amplitude_scaling_variance=cfg.amplitude_scaling_stddev**2,
            amplitude_scaling_boundary=cfg.amplitude_scaling_boundary,
            up_factor=cfg.temporal_upsamples,
            chunk_length_samples=cfg.chunk_length_samples,
            precomputed_templates_npz=cfg.precomputed_templates_npz,
            channel_selection_radius=cfg.channel_selection_radius,
            template_type=cfg.matching_template_type,
            up_method=cfg.matching_up_method,
            template_min_channel_amplitude=cfg.matching_template_min_amplitude,
            refractory_radius_frames=cfg.refractory_radius_frames,
            template_svd_compression_rank=cfg.matching_svd_rank,
            whitening=WhiteningConfig(),  # we don't know how to whiten yet
        )
    else:
        raise ValueError(f"Unknown detection_type {cfg.detection_type}.")

    if cfg.template_interp_kind == "tps":
        temp_interp_params = tps_interp_clampna_extrap_params
    elif cfg.template_interp_kind == "clampna":
        temp_interp_params = clampna_interp_params
    else:
        assert False
    if cfg.matching_interp_kind == "tps":
        match_interp_params = tps_interp_clampna_extrap_params
    elif cfg.matching_interp_kind == "clampna":
        match_interp_params = clampna_interp_params
    else:
        assert False
    whiten_cfg = WhiteningConfig(
        strategy=cfg.whiten_strategy,
        estimator=cfg.whiten_estimator,
        radius=cfg.subtraction_radius_um,
        interp_params=temp_interp_params,
    )
    template_cfg = TemplateConfig(
        denoising_fit_radius=cfg.fit_radius_um,
        spikes_per_unit=cfg.template_spikes_per_unit,
        reduction=cfg.template_reduction,
        denoising_method=cfg.template_denoising_method,
        svd_method=cfg.template_svd_method,
        whitening=whiten_cfg,
        template_interp_params=temp_interp_params,
        svd_alignment_iterations=cfg.svd_alignment_iterations,
        svd_alignment_ms=cfg.alignment_ms / 2,
    )
    clus_sampling_cfg = FitSamplingConfig(
        n_waveforms_fit=cfg.clustering_max_spikes,
        max_waveforms_fit=cfg.clustering_max_spikes,
        more_waveforms_fit=cfg.gmm_max_spikes,
        fit_sampling=cfg.fit_sampling,
    )
    clustering_cfg = ClusteringConfig(
        cluster_strategy=cfg.cluster_strategy,
        sigma_local=cfg.density_bandwidth,
        sigma_regional=5 * cfg.density_bandwidth,
        n_neighbors_search=cfg.n_neighbors_search or cfg.min_cluster_size,
        outlier_radius=5 * cfg.density_bandwidth,
        radius_search=5 * cfg.density_bandwidth,
        min_cluster_size=cfg.min_cluster_size,
        use_hellinger=cfg.use_hellinger,
        component_overlap=cfg.component_overlap,
        hellinger_strong=cfg.hellinger_strong,
        hellinger_weak=cfg.hellinger_weak,
        mop=cfg.dpc_mop,
        sampling_cfg=clus_sampling_cfg,
    )

    if cfg.gmm_metric == "cosine":
        dist_thresh = cfg.gmm_cosine_threshold
    elif cfg.gmm_metric == "normeuc":
        dist_thresh = cfg.gmm_normeuc_threshold
    elif cfg.gmm_metric == "scaled_normeuc":
        dist_thresh = cfg.gmm_scaled_normeuc_threshold
    else:
        assert False
    interp_params = InterpolationParams(
        method=cfg.interp_method,
        kernel=cfg.interp_kernel,
        extrap_method=cfg.extrap_method,
        extrap_kernel=cfg.extrap_kernel,
        kriging_poly_degree=cfg.kriging_poly_degree,
        sigma=cfg.interp_sigma,
        rq_alpha=cfg.rq_alpha,
        smoothing_lambda=cfg.smoothing_lambda,
        polyharmonic_order=cfg.polyharmonic_order,
    ).normalize()
    clustering_features_cfg = ClusteringFeaturesConfig(
        use_amplitude=cfg.initial_amp_feat,
        use_signed_amplitude=cfg.initial_signed_amp_feat,
        n_main_channel_pcs=cfg.initial_pc_feats,
        pc_transform=cfg.initial_pc_transform,
        pc_scale=cfg.initial_pc_scale,
        pc_pre_transform_scale=cfg.initial_pc_pre_scale,
        motion_aware=cfg.motion_aware_clustering,
        interp_params=interp_params,
        feature_rank=cfg.temporal_pca_rank,
    )
    sb = 1.0 + cfg.amplitude_scaling_boundary
    refinement_cfg = RefinementConfig(
        refinement_strategy=cfg.refinement_strategy,
        min_count=cfg.min_cluster_size,
        signal_rank=cfg.signal_rank,
        initialize_at_rank_0=cfg.initialize_at_rank_0,
        n_total_iters=cfg.n_later_refinement_iters,
        n_em_iters=cfg.n_em_iters,
        sampling_cfg=clus_sampling_cfg,
        em_converged_atol=cfg.gmm_em_atol,
        val_proportion=cfg.gmm_val_proportion,
        distance_metric=cfg.gmm_metric,
        n_candidates=cfg.gmm_n_candidates,
        n_search=cfg.gmm_n_search,
        merge_distance_threshold=dist_thresh,
        prior_pseudocount=cfg.prior_pseudocount,
        initial_basis_shrinkage=cfg.initial_basis_shrinkage,
        cov_kind=cfg.cov_kind,
        mixture_steps=cfg.later_steps,
        kmeansk=cfg.kmeansk,
        cl_alpha=cfg.gmm_cl_alpha,
        cl_split_only=cfg.gmm_cl_split_only,
        robust_strategy=cfg.robust_strategy,
        robust_fixed_std_dataset=cfg.robust_fixed_std_dataset,
        robust_fixed_power=cfg.robust_fixed_power,
        robust_df=cfg.robust_df,
        demolish_during_selection=cfg.demolish_during_selection,
        em_after_demolish=cfg.em_after_demolish,
        scale_dist_args=(cfg.amplitude_scaling_stddev, 1.0 / sb, sb / 1.0),
    )
    if cfg.initial_rank is None:
        irank = refinement_cfg.signal_rank
    else:
        irank = cfg.initial_rank
    initial_refinement_cfg = replace(
        refinement_cfg,
        mixture_steps=cfg.initial_steps,
        signal_rank=irank,
        n_total_iters=cfg.n_refinement_iters,
    )
    if cfg.pre_refinement_merge:
        pre_refinement_cfg = RefinementConfig(
            refinement_strategy="pcmerge",
            pc_merge_metric=cfg.pre_refinement_merge_metric,
            pc_merge_threshold=cfg.pre_refinement_merge_threshold,
            pc_merge_rank=cfg.initial_pc_feats,
        )
    else:
        pre_refinement_cfg = None
    motion_kw = {
        k.name: getattr(cfg, k.name)
        for k in fields(MotionEstimationConfig)
        if hasattr(cfg, k.name)
    }
    motion_threshold_cfg = ThresholdingConfig(
        detection_threshold=cfg.motion_voltage_threshold,
        chunk_length_samples=cfg.chunk_length_samples,
        peak_sign=cfg.peak_sign,
        shave_score=cfg.initial_threshold,
    )
    motion_estimation_cfg = MotionEstimationConfig(
        **motion_kw,
        tpca_rank=cfg.temporal_pca_rank,
        threshold_cfg=motion_threshold_cfg,
        spike_denoising_score=cfg.initial_threshold,
    )
    matching_cfg = MatchingConfig(
        threshold="fp_control" if cfg.matching_fp_control else cfg.matching_threshold,
        amplitude_scaling_variance=cfg.amplitude_scaling_stddev**2,
        amplitude_scaling_boundary=cfg.amplitude_scaling_boundary,
        up_factor=cfg.temporal_upsamples,
        chunk_length_samples=cfg.chunk_length_samples,
        template_merge_cfg=TemplateMergeConfig(
            merge_distance_threshold=cfg.postprocessing_merge_threshold,
        ),
        cd_iter=cfg.matching_cd_iter,
        coarse_cd=cfg.matching_coarse_cd,
        channel_selection_radius=cfg.channel_selection_radius,
        template_type=cfg.matching_template_type,
        up_method=cfg.matching_up_method,
        template_min_channel_amplitude=cfg.matching_template_min_amplitude,
        min_template_snr=cfg.min_template_snr,
        min_template_count=cfg.min_template_count,
        whitening=whiten_cfg,
        whiten_features=cfg.whiten_features,
        template_realignment_cfg=TemplateRealignmentConfig(
            trough_factor=cfg.trough_factor,
            realign_strategy=cfg.realign_strategy,
            realign_shift_ms=cfg.alignment_ms,
            template_cfg=TemplateConfig(
                denoising_method="none",
                spikes_per_unit=cfg.template_spikes_per_unit,
                reduction="mean",
                template_interp_params=clampna_interp_params,
            ),
        ),
        template_svd_compression_rank=cfg.matching_svd_rank,
        drift_interp_params=match_interp_params,
        refractory_radius_frames=cfg.refractory_radius_frames,
    )
    computation_cfg = ComputationConfig(
        n_jobs_cpu=cfg.n_jobs_cpu,
        n_jobs_gpu=cfg.n_jobs_gpu,
        device=cfg.device,
        executor=cfg.executor,
    )

    # final aggregation
    if cfg.agg_kind == "none":
        agg_cfg = None
    elif cfg.agg_kind == "template_distance":
        agg_whiten_cfg = WhiteningConfig(
            strategy=cfg.agg_template_whiten_strategy,
            estimator=cfg.whiten_estimator,
            radius=cfg.subtraction_radius_um,
            interp_params=temp_interp_params,
        )
        agg_tmcfg = TemplateMergeConfig(
            linkage=cfg.agg_template_linkage,
            merge_distance_threshold=cfg.agg_no_qda_template_distance,
            waveform_cfg=waveform_cfg,
            whitening=agg_whiten_cfg,
            template_cfg=replace(template_cfg, whitening=agg_whiten_cfg),
        )
        agg_cfg = RefinementConfig(
            refinement_strategy="agglomerate",
            template_merge_cfg=agg_tmcfg,
            qda_threshold=0.0,
        )
    elif cfg.agg_kind == "qda":
        agg_whiten_cfg = WhiteningConfig(
            strategy=cfg.agg_template_whiten_strategy,
            estimator=cfg.whiten_estimator,
            radius=cfg.subtraction_radius_um,
            interp_params=temp_interp_params,
        )
        agg_tmcfg = TemplateMergeConfig(
            linkage=cfg.agg_qda_linkage,
            merge_distance_threshold=cfg.agg_qda_max_template_distance,
            waveform_cfg=waveform_cfg,
            whitening=agg_whiten_cfg,
            template_cfg=replace(template_cfg, whitening=agg_whiten_cfg),
        )
        agg_cfg = RefinementConfig(
            refinement_strategy="agglomerate",
            template_merge_cfg=agg_tmcfg,
            qda_force_merge_for_temp_dist_below=cfg.agg_no_qda_template_distance,
        )
    else:
        assert False

    return DARTsortInternalConfig(
        waveform_cfg=waveform_cfg,
        featurization_cfg=featurization_cfg,
        initial_detection_cfg=initial_detection_cfg,
        peeler_sampling_cfg=peeler_fit_sampling_cfg,
        template_cfg=template_cfg,
        clustering_cfg=clustering_cfg,
        pre_refinement_cfg=pre_refinement_cfg,
        initial_refinement_cfg=initial_refinement_cfg,
        post_refinement_cfg=pre_refinement_cfg,
        agglomerate_cfg=agg_cfg,
        refinement_cfg=refinement_cfg,
        matching_cfg=matching_cfg,
        clustering_features_cfg=clustering_features_cfg,
        motion_estimation_cfg=motion_estimation_cfg,
        computation_cfg=computation_cfg,
        preprocessing=cfg.preprocessing,
        preprocessing_dtype=cfg.preprocessing_dtype,
        detection_type=cfg.detection_type,
        dredge_only=cfg.dredge_only,
        matching_iterations=cfg.matching_iterations,
        recluster_after_first_matching=cfg.recluster_after_first_matching,
        work_in_tmpdir=cfg.work_in_tmpdir,
        copy_recording_to_tmpdir=cfg.copy_recording_to_tmpdir,
        workdir_copier=cfg.workdir_copier,
        workdir_follow_symlinks=cfg.workdir_follow_symlinks,
        tmpdir_parent=cfg.tmpdir_parent,
        save_intermediate_labels=cfg.save_intermediates,
        save_intermediate_features=cfg.save_intermediates,
        save_final_features=cfg.save_final_features,
        save_everything_on_error=cfg.save_everything_on_error,
        link_from=cfg.link_from,
        link_step=cfg.link_step,
        subsampling_spikes=cfg.subsampling_spikes,
        subsampling_presence=cfg.subsampling_presence,
        always_save_final_tpca_feature=cfg.always_save_final_tpca_feature,
    )


default_dartsort_cfg = DARTsortInternalConfig()

# configs which are commonly used for specific tasks
unshifted_raw_template_cfg = TemplateConfig(
    registered_templates=False, denoising_method="none"
)
waveforms_only_featurization_cfg = FeaturizationConfig(
    do_tpca_denoise=False,
    do_enforce_decrease=False,
    save_input_tpca_projs=False,
    save_amplitudes=False,
    do_localization=False,
    input_waveforms_name="raw",
    save_input_voltages=True,
    save_input_waveforms=True,
)
skip_featurization_cfg = FeaturizationConfig(skip=True)


if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(
        [
            WaveformConfig,
            FitSamplingConfig,
            ClusteringConfig,
            ClusteringFeaturesConfig,
            FeaturizationConfig,
            RefinementConfig,
            InterpolationParams,
            TemplateMergeConfig,
            TemplateConfig,
            TemplateRealignmentConfig,
            WhiteningConfig,
        ]
    )
