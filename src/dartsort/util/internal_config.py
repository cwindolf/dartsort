import dataclasses
from dataclasses import field, fields
import math
from typing import Literal

import numpy as np
from pydantic.dataclasses import dataclass
import torch

from .py_util import int_or_inf
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


@dataclass(frozen=True, kw_only=True)
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


@dataclass(frozen=True, kw_only=True)
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

    # -- further info about denoising
    # in the future we may add multi-channel or other nns
    nn_denoiser_class_name: str = "SingleChannelWaveformDenoiser"
    nn_denoiser_pretrained_path: str = default_pretrained_path
    nn_denoiser_train_epochs: int = 50
    nn_denoiser_extra_kwargs: dict | None = argfield(None, cli=False)

    # optionally restrict how many channels TPCA are fit on
    tpca_fit_radius: float = 75.0
    tpca_rank: int = 8
    tpca_centered: bool = False
    learn_cleaned_tpca_basis: bool = False
    input_tpca_waveform_config: WaveformConfig | None = WaveformConfig(
        ms_before=0.75, ms_after=1.25
    )

    # used when naming datasets saved to h5 files
    input_waveforms_name: str = "collisioncleaned"
    output_waveforms_name: str = "denoised"


@dataclass(frozen=True, kw_only=True)
class SubtractionConfig:
    # peeling common
    chunk_length_samples: int = 30_000
    n_chunks_fit: int = 100
    max_waveforms_fit: int = 50_000
    n_waveforms_fit: int = 20_000
    fit_subsampling_random_state: int = 0
    fit_sampling: str = "random"
    fit_max_reweighting: float = 4.0

    # subtraction
    detection_threshold: float = 4.0
    peak_sign: Literal["pos", "neg", "both"] = "both"
    spatial_dedup_radius: float | None = 150.0
    subtract_radius: float = 200.0
    residnorm_decrease_threshold: float = math.sqrt(0.1 * 15**2)  # sqrt(10)
    use_singlechan_templates: bool = False
    singlechan_threshold: float = 50.0
    n_singlechan_templates: int = 10
    singlechan_alignment_padding_ms: float = 1.0
    use_universal_templates: bool = False
    universal_threshold: float = 50.0

    # how will waveforms be denoised before subtraction?
    # users can also save waveforms/features during subtraction
    subtraction_denoising_config: FeaturizationConfig = FeaturizationConfig(
        denoise_only=True,
        do_nn_denoise=True,
        input_waveforms_name="raw",
        output_waveforms_name="subtracted",
    )


@dataclass(frozen=True, kw_only=True)
class MatchingConfig:
    # peeling common
    chunk_length_samples: int = 30_000
    n_chunks_fit: int = 100
    max_waveforms_fit: int = 50_000
    n_waveforms_fit: int = 20_000
    fit_subsampling_random_state: int = 0
    fit_sampling: str = "random"
    fit_max_reweighting: float = 4.0

    # template matching parameters
    threshold: float | Literal["fp_control"] = 15.0  # norm, not normsq
    template_svd_compression_rank: int = 10
    template_temporal_upsampling_factor: int = 4
    template_min_channel_amplitude: float = 1.0
    refractory_radius_frames: int = 10
    amplitude_scaling_variance: float = 0.1**2
    amplitude_scaling_boundary: float = 1.0
    max_iter: int = 1000
    conv_ignore_threshold: float = 5.0
    coarse_approx_error_threshold: float = 0.0
    coarse_objective: bool = True


@dataclass(frozen=True, kw_only=True)
class ThresholdingConfig:
    # peeling common
    chunk_length_samples: int = 30_000
    n_chunks_fit: int = 100
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
    relative_peak_radius_samples: int = 5
    dedup_temporal_radius_samples: int = 7
    thinning: float = 0.0
    time_jitter: int = 0
    spatial_jitter_radius: float = 0.0


@dataclass(frozen=True, kw_only=True)
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


@dataclass(frozen=True, kw_only=True)
class TemplateConfig:
    spikes_per_unit: int = 500

    # -- template construction parameters
    # registered templates?
    registered_templates: bool = True
    registered_template_localization_radius_um: float = 100.0

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
    realign_shift_ms: float = 1.0

    # track template over time
    time_tracking: bool = False
    chunk_size_s: int = 300


@dataclass(frozen=True, kw_only=True)
class SplitMergeConfig:
    # -- split
    split_strategy: str = "FeatureSplit"
    recursive_split: bool = True
    split_strategy_kwargs: dict | None = field(
        default_factory=lambda: dict(max_spikes=20_000)
    )

    # -- merge
    merge_template_config: TemplateConfig = TemplateConfig(superres_templates=False)
    linkage: str = "complete"
    merge_distance_threshold: float = 0.25
    cross_merge_distance_threshold: float = 0.5
    min_spatial_cosine: float = 0.0


@dataclass(frozen=True, kw_only=True)
class ClusteringConfig:
    # -- initial clustering
    cluster_strategy: str = "dpc"

    # initial clustering features
    use_amplitude: bool = True
    amp_log_c: float = 5.0
    amp_scale: float = 50.0
    n_main_channel_pcs: int = 0
    pc_scale: float = 10.0
    adaptive_feature_scales: bool = False

    # density peaks parameters
    sigma_local: float = 5.0
    sigma_regional: float | None = argfield(default=25.0, arg_type=float)
    workers: int = -1
    n_neighbors_search: int = 20
    radius_search: float = 5.0
    remove_clusters_smaller_than: int = 10
    noise_density: float = 0.0
    outlier_radius: float = 5.0
    outlier_neighbor_count: int = 5
    kdtree_subsample_max_size: int = 2_500_000

    # hdbscan parameters
    min_cluster_size: int = 25
    min_samples: int = 25
    cluster_selection_epsilon: int = 1
    recursive: bool = False
    remove_duplicates: bool = False

    # remove large clusters in hdbscan?
    remove_big_units: bool = False
    zstd_big_units: float = 50.0

    # grid snap parameters
    grid_dx: float = 15.0
    grid_dz: float = 15.0

    # uhd version of density peaks parameters
    sigma_local_low: float | None = argfield(default=None, arg_type=float)
    sigma_regional_low: float | None = argfield(default=None, arg_type=float)
    distance_dependent_noise_density: bool = False
    attach_density_feature: bool = False
    triage_quantile_per_cluster: float = 0.0
    revert: bool = False
    ramp_triage_per_cluster: bool = False
    triage_quantile_before_clustering: float = 0.0
    amp_no_triaging_before_clustering: float = 6.0
    amp_no_triaging_after_clustering: float = 8.0
    use_y_triaging: bool = False
    remove_small_far_clusters: bool = False

    # -- ensembling
    ensemble_strategy: str | None = argfield(default=None, arg_type=str)
    chunk_size_s: float = 300.0
    split_merge_ensemble_config: SplitMergeConfig | None = None


@dataclass(frozen=True, kw_only=True)
class RefinementConfig:
    refinement_stragegy: Literal["gmm", "splitmerge"] = "gmm"

    # -- gmm parameters
    # noise params
    cov_kind = "factorized"
    glasso_alpha: float = 0.01

    # feature params
    core_radius: float = 35.0
    interpolation_sigma: float = 20.0
    val_proportion: float = 0.25
    max_n_spikes: float | int = argfield(default=4_000_000, arg_type=int_or_inf)
    max_avg_units: int = 3

    # model params
    channels_strategy: str = "count"
    min_count: int = 50
    signal_rank: int = 0
    n_spikes_fit: int = 4096
    ppca_inner_em_iter: int = 5
    distance_metric: Literal["noise_metric", "kl", "reverse_kl", "symkl"] = (
        "noise_metric"
    )
    distance_normalization_kind: Literal["none", "noise", "channels"] = "noise"
    merge_distance_threshold: float = 2.0
    # if None, switches to bimodality
    merge_criterion_threshold: float | None = 0.0
    merge_criterion: Literal[
        "heldout_loglik",
        "heldout_elbo",
        "loglik",
        "elbo",
        "old_heldout_loglik",
        "old_heldout_ccl",
        "old_loglik",
        "old_ccl",
        "old_aic",
        "old_bic",
        "old_icl",
        "bimodality",
    ] = "heldout_elbo"
    merge_bimodality_threshold: float = 0.05
    n_em_iters: int = 50
    em_converged_prop: float = 0.02
    em_converged_churn: float = 0.01
    em_converged_atol: float = 1e-2
    n_total_iters: int = 3
    hard_noise: bool = False
    truncated: bool = True
    split_decision_algorithm: str = "tree"
    merge_decision_algorithm: str = "brute"
    prior_pseudocount: float = 5.0

    # if someone wants this
    split_merge_config: SplitMergeConfig | None = None


@dataclass(frozen=True, kw_only=True)
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


@dataclass(frozen=True, kw_only=True)
class DARTsortInternalConfig:
    """This is an internal object. Make a DARTsortUserConfig, not one of these."""

    waveform_config: WaveformConfig = WaveformConfig()
    featurization_config: FeaturizationConfig = FeaturizationConfig()
    subtraction_config: SubtractionConfig = SubtractionConfig()
    template_config: TemplateConfig = TemplateConfig()
    clustering_config: ClusteringConfig = ClusteringConfig()
    refinement_config: RefinementConfig = RefinementConfig()
    matching_config: MatchingConfig = MatchingConfig()
    motion_estimation_config: MotionEstimationConfig = MotionEstimationConfig()
    computation_config: ComputationConfig = ComputationConfig()

    # high level behavior
    subtract_only: bool = False
    dredge_only: bool = False
    final_refinement: bool = True
    matching_iterations: int = 1
    intermediate_matching_subsampling: float = 1.0
    overwrite_matching: bool = False

    # development / debugging flags
    save_intermediate_labels: bool = False


default_waveform_config = WaveformConfig()
default_featurization_config = FeaturizationConfig()
default_subtraction_config = SubtractionConfig()
default_thresholding_config = ThresholdingConfig()
default_template_config = TemplateConfig()
default_clustering_config = ClusteringConfig()
default_split_merge_config = SplitMergeConfig()
coarse_template_config = TemplateConfig(superres_templates=False)
raw_template_config = TemplateConfig(
    realign_peaks=False, low_rank_denoising=False, superres_templates=False
)
unshifted_raw_template_config = TemplateConfig(
    registered_templates=False,
    realign_peaks=False,
    low_rank_denoising=False,
    superres_templates=False,
)
unaligned_coarse_denoised_template_config = TemplateConfig(
    realign_peaks=False, low_rank_denoising=True, superres_templates=False
)
default_matching_config = MatchingConfig()
default_motion_estimation_config = MotionEstimationConfig()
default_computation_config = ComputationConfig()
default_dartsort_config = DARTsortInternalConfig()
default_refinement_config = RefinementConfig()

waveforms_only_featurization_config = FeaturizationConfig(
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


def to_internal_config(cfg):
    from dartsort.config import DARTsortUserConfig, DeveloperConfig

    if isinstance(cfg, DARTsortInternalConfig):
        return cfg
    else:
        assert isinstance(cfg, (DARTsortUserConfig, DeveloperConfig))

    # if we have a user cfg, dump into dev cfg, and work from there
    if isinstance(cfg, DARTsortUserConfig):
        cfg = DeveloperConfig(**dataclasses.asdict(cfg))

    waveform_config = WaveformConfig(ms_before=cfg.ms_before, ms_after=cfg.ms_after)
    tpca_waveform_config = WaveformConfig(
        ms_before=cfg.feature_ms_before, ms_after=cfg.feature_ms_after
    )
    featurization_config = FeaturizationConfig(
        tpca_rank=cfg.temporal_pca_rank,
        extract_radius=cfg.featurization_radius_um,
        input_tpca_waveform_config=tpca_waveform_config,
        localization_radius=cfg.localization_radius_um,
        tpca_fit_radius=cfg.fit_radius_um,
    )
    subtraction_denoising_config = FeaturizationConfig(
        denoise_only=True,
        do_nn_denoise=cfg.use_nn_in_subtraction,
        tpca_rank=cfg.temporal_pca_rank,
        tpca_fit_radius=cfg.fit_radius_um,
        input_waveforms_name="raw",
        output_waveforms_name="subtracted",
    )
    subtraction_config = SubtractionConfig(
        detection_threshold=cfg.initial_threshold,
        spatial_dedup_radius=cfg.deduplication_radius_um,
        subtract_radius=cfg.subtraction_radius_um,
        singlechan_alignment_padding_ms=cfg.alignment_ms,
        use_singlechan_templates=cfg.use_singlechan_templates,
        use_universal_templates=cfg.use_universal_templates,
        subtraction_denoising_config=subtraction_denoising_config,
        residnorm_decrease_threshold=np.sqrt(
            cfg.denoiser_badness_factor * cfg.matching_threshold**2
        ),
        chunk_length_samples=cfg.chunk_length_samples,
    )
    template_config = TemplateConfig(
        registered_template_localization_radius_um=cfg.localization_radius_um,
        denoising_fit_radius=cfg.fit_radius_um,
        realign_shift_ms=cfg.alignment_ms,
    )
    clustering_config = ClusteringConfig(
        sigma_local=cfg.density_bandwidth,
        sigma_regional=5 * cfg.density_bandwidth,
        outlier_radius=cfg.density_bandwidth,
        radius_search=cfg.density_bandwidth,
    )
    refinement_config = RefinementConfig(
        signal_rank=cfg.signal_rank,
        interpolation_sigma=cfg.interpolation_bandwidth,
        merge_criterion=cfg.merge_criterion,
        merge_criterion_threshold=cfg.merge_criterion_threshold,
        merge_bimodality_threshold=cfg.merge_bimodality_threshold,
        n_total_iters=cfg.n_refinement_iters,
        n_em_iters=cfg.n_em_iters,
        max_n_spikes=cfg.gmm_max_spikes,
        val_proportion=cfg.gmm_val_proportion,
        channels_strategy=cfg.channels_strategy,
        truncated=cfg.truncated,
        split_decision_algorithm=cfg.gmm_split_decision_algorithm,
        merge_decision_algorithm=cfg.gmm_merge_decision_algorithm,
        prior_pseudocount=cfg.prior_pseudocount,
        cov_kind=cfg.cov_kind,
        glasso_alpha=cfg.glasso_alpha,
    )
    motion_estimation_config = MotionEstimationConfig(
        **{k.name: getattr(cfg, k.name) for k in fields(MotionEstimationConfig)}
    )
    matching_config = MatchingConfig(
        threshold=cfg.matching_threshold,
        amplitude_scaling_variance=cfg.amplitude_scaling_stddev**2,
        amplitude_scaling_boundary=cfg.amplitude_scaling_limit,
        template_temporal_upsampling_factor=cfg.temporal_upsamples,
        chunk_length_samples=cfg.chunk_length_samples,
    )
    computation_config = ComputationConfig(
        n_jobs_cpu=cfg.n_jobs_cpu,
        n_jobs_gpu=cfg.n_jobs_gpu,
        device=cfg.device,
        executor=cfg.executor,
    )

    return DARTsortInternalConfig(
        waveform_config=waveform_config,
        featurization_config=featurization_config,
        subtraction_config=subtraction_config,
        template_config=template_config,
        clustering_config=clustering_config,
        refinement_config=refinement_config,
        matching_config=matching_config,
        motion_estimation_config=motion_estimation_config,
        computation_config=computation_config,
        dredge_only=cfg.dredge_only,
        matching_iterations=cfg.matching_iterations,
        overwrite_matching=cfg.overwrite_matching,
        save_intermediate_labels=cfg.save_intermediate_labels,
    )
