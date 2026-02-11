from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from .util.cli_util import argfield
from .util.internal_config import (
    default_pretrained_path,
    InterpMethod,
    InterpKernel,
    RealignStrategy,
)
from .util.py_util import (
    cfg_dataclass,
    float_or_none,
    int_or_float_or_none,
    int_or_none,
    str_or_none,
)


@cfg_dataclass
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
    copy_recording_to_tmpdir: bool = False
    workdir_copier: Literal["shutil", "rsync"] = "shutil"
    workdir_follow_symlinks: bool = False
    tmpdir_parent: str | Path | None = argfield(default=None, arg_type=str_or_none)
    save_intermediates: bool = False
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
        doc="Largest time shift allowed when re-aligning events.",
    )

    # -- thresholds
    peak_sign: Literal["neg", "both", "pos"] = argfield(
        default="both",
        doc="Allow only troughs or events of both signs when detecting threshold "
        "crossings during initialization. Or positive only, if that's your thing.",
    )
    voltage_threshold: Annotated[float, Field(gt=0)] = argfield(
        default=4.0,
        doc="Threshold in standardized (SNR) voltage units for initial detection; "
        "peaks or troughs larger than this value will be grabbed.",
    )
    matching_threshold: Annotated[float, Field(gt=0)] = argfield(
        default=10.0,
        doc="Template matching threshold. If subtracting a template leads "
        "to at least this great of a decrease in the norm of the residual, "
        "that match will be used.",
    )
    initial_threshold: Annotated[float, Field(gt=0)] = argfield(
        default=12.0,
        doc="Initial detection's neural net matching threshold. Same as "
        "matching_threshold, except that a neural net is trying to guess "
        "the true waveforms here, rather than using cluster templates.",
    )

    # -- featurization length, radius, rank parameters
    temporal_pca_rank: Annotated[int, Field(gt=0)] = argfield(
        default=8, doc="Rank of temporal PCAs used in denoising and featurization."
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

    # -- matching parameters
    amplitude_scaling_stddev: Annotated[float, Field(ge=0)] = 0.01
    amplitude_scaling_boundary: Annotated[float, Field(ge=0)] = 0.333
    temporal_upsamples: Annotated[int, Field(ge=1)] = 8

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
    window_margin_um: Annotated[float, Field(gt=0)] | None = argfield(
        default=None, arg_type=float_or_none
    )
    max_dt_s: Annotated[float, Field(gt=0)] = 1000.0
    max_disp_um: Annotated[float, Field(gt=0)] | None = argfield(
        default=None, arg_type=float_or_none
    )
    correlation_threshold: Annotated[float, Field(gt=0, lt=1)] = 0.1
    min_amplitude: float | None = argfield(default=None, arg_type=float_or_none)


@cfg_dataclass
class DeveloperConfig(DARTsortUserConfig):
    """Additional parameters for experiments. This API will never be stable."""

    # high level behavior
    initial_steps: Literal["neither", "split", "merge", "both"] = "split"
    later_steps: Literal["neither", "split", "merge", "both"] = "merge"
    detection_type: str = "subtract"
    cluster_strategy: str = "dpc"
    refinement_strategy: str = "tmm"
    recluster_after_first_matching: bool = True

    # general peeling
    n_waveforms_fit: int = 40_000
    max_waveforms_fit: int = 50_000
    fit_sampling: Literal["random", "amp_reweighted"] = "amp_reweighted"

    # initial detection
    nn_denoiser_max_waveforms_fit: int = 250_000
    nn_denoiser_class_name: str = "SingleChannelWaveformDenoiser"
    nn_denoiser_pretrained_path: str | None = argfield(
        default=default_pretrained_path, arg_type=str_or_none
    )
    do_tpca_denoise: bool = True
    first_denoiser_thinning: float = 0.5
    realign_to_denoiser: bool = True
    use_nn_in_subtraction: bool = True
    use_singlechan_templates: bool = False

    # matching
    matching_template_type: Literal["individual_compressed_upsampled", "drifty"] = (
        "drifty"
    )
    matching_up_method: Literal["interpolation", "keys3", "keys4", "direct"] = "keys4"
    matching_cd_iter: int = 0
    matching_coarse_cd: bool = True
    postprocessing_merge_threshold: float = 0.025
    template_spikes_per_unit: int = 500
    template_reduction: Literal["mean", "median"] = "mean"
    template_denoising_method: Literal["none", "exp_weighted", "t", "loot"] = (
        "exp_weighted"
    )
    template_mix_zero: bool = False
    template_mix_svd: bool = True
    min_template_snr: float = 40.0
    min_template_count: int = 50
    channel_selection_radius: float | None = argfield(
        default=None, arg_type=float_or_none
    )
    always_recompute_tsvd: bool = True
    matching_template_min_amplitude: float = 0.0
    realign_strategy: RealignStrategy = "snr_weighted_trough_factor"
    trough_factor: float = 3.0
    whiten_matching: bool = False
    matching_fp_control: bool = False

    # interpolation for features
    interp_method: InterpMethod = "kriging"
    interp_kernel: InterpKernel = "thinplate"
    extrap_method: InterpMethod | None = argfield(default=None, arg_type=str_or_none)
    extrap_kernel: InterpKernel | None = argfield(default=None, arg_type=str_or_none)
    kriging_poly_degree: int = 1
    interp_sigma: float = 10.0
    rq_alpha: float = 0.5
    smoothing_lambda: float = 0.0

    # initial clustering
    initial_euclidean_complete_only: bool = False
    initial_cosine_complete_only: bool = False
    initial_amp_feat: bool = False
    initial_signed_amp_feat: bool = True
    initial_pc_feats: int = 5
    initial_pc_transform: Literal["log", "sqrt", "none"] = "none"
    initial_pc_scale: float = 2.0
    initial_pc_pre_scale: float = 0.5
    motion_aware_clustering: bool = True
    clustering_workers: int = 5
    clustering_max_spikes: Annotated[int, Field(gt=0)] = 500_000
    pre_refinement_merge: bool = True
    pre_refinement_merge_metric: str = "normeuc"
    pre_refinement_merge_threshold: float = 0.1
    use_hellinger: bool = True
    density_bandwidth: Annotated[float, Field(gt=0)] = 5.0
    component_overlap: float = 0.95
    hellinger_strong: float = 0.0
    hellinger_weak: float = 0.0
    dpc_mop: bool = True
    n_neighbors_search: int | None = argfield(default=50, arg_type=int_or_none)

    # gaussian mixture high level
    truncated: bool = True
    initial_rank: int | None = argfield(default=None, arg_type=int_or_none)
    initialize_at_rank_0: bool = False
    signal_rank: Annotated[int, Field(ge=0)] = 5
    gmm_max_spikes: Annotated[int, Field(gt=0)] = 1000 * 1024
    kmeansk: int = 4
    min_cluster_size: int = 25

    # gausian mixture low level
    n_refinement_iters: int = 1
    n_em_iters: int = 250
    channels_strategy: Literal["count", "all"] = "count"
    gmm_cl_alpha: float = 1.0
    gmm_em_atol: float = 5e-3
    gmm_metric: Literal["cosine", "normeuc"] = "normeuc"
    gmm_n_candidates: int = 3
    gmm_n_search: int | None = argfield(default=None, arg_type=int_or_none)
    gmm_val_proportion: Annotated[float, Field(gt=0)] = 0.25
    initial_basis_shrinkage: float = 1.0
    prior_pseudocount: float = 0.0
    cov_kind: str = "factorizednoise"
    gmm_euclidean_threshold: float = 5.0
    gmm_kl_threshold: float = 2.0
    gmm_cosine_threshold: float = 0.8
    gmm_normeuc_threshold: float = 1.0

    # store extra intermediates
    save_subtracted_waveforms: bool = False
    save_collisioncleaned_waveforms: bool = False
    precomputed_templates_npz: str | None = argfield(default=None, arg_type=str_or_none)
    save_everything_on_error: bool = False
