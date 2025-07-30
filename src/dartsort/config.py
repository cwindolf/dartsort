"""Knobs"""

from typing import Annotated, Literal
from pathlib import Path

from pydantic import Field
from pydantic.dataclasses import dataclass

from .util.internal_config import _pydantic_strict_cfg, default_pretrained_path
from .util.py_util import int_or_none, str_or_none, int_or_float_or_none
from .util.cli_util import argfield


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class DARTsortUserConfig:
    """User-facing configuration options"""

    # -- high level behavior
    dredge_only: bool = argfield(
        False, doc="Whether to stop after initial localization and motion tracking."
    )
    matching_iterations: int = 1

    # -- computer options
    n_jobs_cpu: int = argfield(
        default=0,
        doc="Number of parallel workers to use when running on CPU. "
        "0 means everything runs on the main thread.",
    )
    n_jobs_gpu: int = argfield(
        default=0,
        doc="Number of parallel workers to use when running on GPU. "
        "0 means everything runs on the main thread.",
    )
    device: str | None = argfield(
        default=None,
        arg_type=str,
        doc="The name of the PyTorch device to use. For example, 'cpu' "
        "or 'cuda' or 'cuda:1'. If unset, uses n_jobs_gpu of your CUDA "
        "GPUs if you have multiple, or else just the one, or your CPU.",
    )
    executor: str = "threading_unless_multigpu"
    chunk_length_samples: int = 30_000

    # -- storage behavior
    work_in_tmpdir: bool = False
    tmpdir_parent: str | Path | None = argfield(default=None, arg_type=str_or_none)
    save_intermediate_labels: bool = False
    save_intermediate_features: bool = True
    save_final_features: bool = True

    # -- waveform snippet length parameters
    ms_before: Annotated[float, Field(gt=0)] = argfield(
        default=1.4,
        doc="Length of time (ms) before trough (or peak) in waveform snippets. "
        "Default value corresponds to 42 samples at 30kHz.",
    )
    ms_after: Annotated[float, Field(gt=0)] = argfield(
        default=2.6 + 0.1 / 3,
        doc="Length of time (ms) after trough (or peak) in waveform snippets. "
        "Default value corresponds to 79 samples at 30kHz.",
    )
    alignment_ms: Annotated[float, Field(gt=0)] = argfield(
        default=1.5,
        doc="Time shift allowed when aligning events.",
    )

    # -- thresholds
    initial_threshold: Annotated[float, Field(gt=0)] = argfield(
        default=4.0,
        doc="Threshold in standardized voltage units for initial detection; "
        "peaks or troughs larger than this value will be grabbed.",
    )
    matching_threshold: Annotated[float, Field(gt=0)] = argfield(
        default=10.0,
        doc="Template matching threshold. If subtracting a template leads "
        "to at least this great of a decrease in the norm of the residual, "
        "that match will be used.",
    )
    matching_fp_control: bool = False
    denoiser_badness_factor: Annotated[float, Field(ge=0, le=1)] = argfield(
        default=0.15,
        doc="In initial detection, subtracting clean waveforms inferred "
        "by the NN denoiser need only decrease the residual norm squared "
        "by this multiple of the squared matching threshold to be accepted.",
    )

    # -- featurization length, radius, rank parameters
    temporal_pca_rank: Annotated[int, Field(gt=0)] = argfield(
        default=8, doc="Rank of global temporal PCA."
    )
    feature_ms_before: Annotated[float, Field(gt=0)] = argfield(
        default=0.75,
        doc="As ms_before, but used only when computing PCA features in clustering.",
    )
    feature_ms_after: Annotated[float, Field(gt=0)] = argfield(
        default=1.25,
        doc="As ms_after, but used only when computing PCA features in clustering.",
    )
    subtraction_radius_um: Annotated[float, Field(gt=0)] = argfield(
        default=200.0,
        doc="Radius of neighborhoods around spike events extracted "
        "when denoising and subtracting NN-denoised events.",
    )
    deduplication_radius_um: Annotated[float, Field(gt=0)] = argfield(
        default=100.0,
        doc="During initial detection, if two spike events occur at the "
        "same time within this radius, then the smaller of the two is "
        "ignored. But also all of the secondary channels of the big one, "
        "which is important.",
    )
    featurization_radius_um: Annotated[float, Field(gt=0)] = argfield(
        default=100.0,
        doc="Radius around detection channel or template peak channel used "
        "to extract spike features for clustering.",
    )
    fit_radius_um: Annotated[float, Field(gt=0)] = argfield(
        default=75.0,
        doc="Extraction radius when fitting features like PCA; "
        "smaller than other radii to include less noise.",
    )
    localization_radius_um: Annotated[float, Field(gt=0)] = argfield(
        default=100.0,
        doc="Radius around main channel used when localizing spikes.",
    )

    # -- clustering parameters
    density_bandwidth: Annotated[float, Field(gt=0)] = 5.0
    interpolation_bandwidth: Annotated[float, Field(gt=0)] = 10.0

    # -- matching parameters
    amplitude_scaling_stddev: Annotated[float, Field(ge=0)] = 0.1
    amplitude_scaling_limit: Annotated[float, Field(ge=0)] = 1.0
    temporal_upsamples: Annotated[int, Field(ge=1)] = 4

    # -- motion estimation parameters
    do_motion_estimation: bool = argfield(
        default=True,
        doc="Set this to false if your data is super stable or already motion-corrected.",
    )

    # DREDge parameters
    rigid: bool = argfield(
        default=False, doc="Use rigid registration and ignore the window parameters."
    )
    probe_boundary_padding_um: float = 100.0
    spatial_bin_length_um: Annotated[float, Field(gt=0)] = 1.0
    temporal_bin_length_s: Annotated[float, Field(gt=0)] = 1.0
    window_step_um: Annotated[float, Field(gt=0)] = 400.0
    window_scale_um: Annotated[float, Field(gt=0)] = 450.0
    window_margin_um: Annotated[float, Field(gt=0)] | None = None
    max_dt_s: Annotated[float, Field(gt=0)] = 1000.0
    max_disp_um: Annotated[float, Field(gt=0)] | None = None
    correlation_threshold: Annotated[float, Field(gt=0, lt=1)] = 0.1
    min_amplitude: float | None = argfield(default=None, arg_type=float)


@dataclass(frozen=True, kw_only=True, config=_pydantic_strict_cfg)
class DeveloperConfig(DARTsortUserConfig):
    """Additional parameters for experiments. This API will never be stable."""

    detection_type: str = "subtract"
    initial_split_only: bool = True
    resume_with_split: bool = False
    cluster_strategy: str = "gmmdpc"
    refinement_strategy: str = "gmm"
    recluster_after_first_matching: bool = False
    initial_rank: int | None = argfield(default=None, arg_type=int_or_none)
    signal_rank: Annotated[int, Field(ge=0)] = 0
    gmm_euclidean_threshold: float = 5.0
    gmm_kl_threshold: float = 2.0
    gmm_cosine_threshold: float = 0.75
    initial_euclidean_complete_only: bool = False
    initial_cosine_complete_only: bool = False
    gmm_noise_fp_correction: bool = False

    pre_refinement_merge: bool = False
    pre_refinement_merge_metric: str = "cosine"
    pre_refinement_merge_threshold: float = 0.025

    use_nn_in_subtraction: bool = True
    use_singlechan_templates: bool = False
    truncated: bool = True
    overwrite_matching: bool = False

    cumulant_order: int = 0

    criterion_threshold: float = 0.0
    criterion: Literal[
        "heldout_loglik", "heldout_elbo", "loglik", "elbo"
    ] = "heldout_elbo"
    merge_bimodality_threshold: float = 0.05
    n_refinement_iters: int = 3
    n_em_iters: int = 50
    channels_strategy: str = "count"
    hard_noise: bool = False
    gmm_metric: Literal["kl", "cosine"] = "cosine"
    gmm_search: Literal["topk", "random"] = "topk"
    gmm_n_candidates: int = 5
    gmm_n_search: int | None = argfield(default=None, arg_type=int_or_none)

    initial_amp_feat: bool = False
    initial_pc_feats: int = 3
    initial_pc_transform: str = "none"
    initial_pc_scale: float = 2.5
    initial_pc_pre_scale: float = 0.5
    motion_aware_clustering: bool = True
    clustering_workers: int = 5
    clustering_max_spikes: Annotated[int, Field(gt=0)] = 100_000

    n_waveforms_fit: int = 20_000
    max_waveforms_fit: int = 50_000
    nn_denoiser_max_waveforms_fit: int = 250_000
    nn_denoiser_class_name: str = "SingleChannelWaveformDenoiser"
    nn_denoiser_pretrained_path: str | None = argfield(
        default=default_pretrained_path, arg_type=str_or_none
    )
    do_tpca_denoise: bool = True
    first_denoiser_thinning: float = 0.5
    postprocessing_merge_threshold: float = 0.025

    gmm_max_spikes: Annotated[int, Field(gt=0)] = 2_000_000
    gmm_val_proportion: Annotated[float, Field(gt=0)] = 0.25
    gmm_split_decision_algorithm: str = "brute"
    gmm_merge_decision_algorithm: str = "brute"
    kmeansk: int = 3
    prior_pseudocount: float = 10.0
    prior_scales_mean: bool = False
    cov_kind: str = "factorizednoise"
    interpolation_method: str = "kriging"
    extrapolation_method: str | None = argfield(default="kernel", arg_type=str_or_none)
    interpolation_kernel: str = "thinplate"
    interpolation_rq_alpha: float = 0.5
    interpolation_degree: int = 0
    glasso_alpha: float | int | None = argfield(default=None, arg_type=int_or_float_or_none)
    laplace_ard: bool = False
    core_radius: float = 35.0
    min_cluster_size: int = 50

    use_hellinger: bool = False
    component_overlap: float = 0.95
    hellinger_strong: float = 0.0
    hellinger_weak: float = 0.0
    dpc_mop: bool = False
    n_neighbors_search: int | None = argfield(default=50, arg_type=int_or_none)

    save_subtracted_waveforms: bool = False
    save_collisioncleaned_waveforms: bool = False
    precomputed_templates_npz: str | None = argfield(default=None, arg_type=str_or_none)
    save_everything_on_error: bool = False
