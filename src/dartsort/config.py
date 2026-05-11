from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import Field

from .util.cli_util import argfield
from .util.internal_config import (
    InterpKernel,
    InterpMethod,
    MixtureStep,
    PreprocessingStrategy,
    RealignStrategy,
    TemplateSVDMethod,
    WhiteningEstimator,
    WhiteningStrategy,
    default_pretrained_path,
)
from .util.py_util import cfg_dataclass, float_or_none, int_or_none, str_or_none


@cfg_dataclass
class DARTsortUserConfig:
    """User-facing configuration options"""

    # -- high level behavior
    do_motion_estimation: bool = argfield(
        default=True,
        doc="Set this to false if your data is super stable or already motion-corrected.",
    )
    preprocessing: PreprocessingStrategy = argfield(
        default="none",
        arg_type=str,
        doc="If other than 'none', dartsort will apply a standard preprocessing "
        "to the recording. Leave as 'none' if you'd prefer to control preprocessing. "
        "If so, be aware that dartsort expects its input to be standardized on "
        "each channel in addition to the usual highpass filtering, but that "
        "whitening is handled internally. See util/preprocess_util.py if you're "
        "curious about the details of the methods.",
    )
    preprocessing_dtype: Literal["float16", "float32"] = argfield(
        default="float32",
        arg_type=str,
        doc="If you have a lot of data and you're using a workflow where it is important "
        "to save a preprocessed copy of the recording, float16 is a good option. Only "
        "relevant if preprocessing != 'none'. If the recording isn't getting saved, "
        "stick to float32.",
    )
    subsampling_spikes: int | None = argfield(
        default=2_048_000,
        arg_type=int_or_none,
        doc="Detection steps before the final matching round will run until at least "
        "this many spikes are found or the whole recording is covered, to make sure "
        "that there is enough data for clustering. See also subsampling_fraction. "
        "Set to None to disable subsampling.",
    )
    subsampling_presence: Annotated[float, Field(gt=0.0, le=1.0)] = argfield(
        default=0.1,
        doc="Early detection steps which have already found `subsampling_spikes` "
        "spikes are only allowed to end early if they additionally cover this "
        "fraction of the recording, to make sure there's good coverage of "
        "conditions for template estimation.",
    )
    matching_iterations: int = argfield(
        default=1,
        doc="By default, 1 template matching step is carried out using templates "
        "estimated from the initial detection round.",
    )

    # -- computer options
    n_jobs_cpu: int = argfield(
        default=0,
        doc="Number of parallel workers to use when running on CPU. "
        "0 means everything runs on the main thread; negative means "
        "#cpu - (val+1) so that -1 is all cores, -2 is all less 1, etc.",
    )
    n_jobs_gpu: int = argfield(
        default=0, doc="Number of parallel workers to use when running on GPU."
    )
    n_jobs_small: int = argfield(
        default=-2,
        doc="Max workers to use for small jobs.",
    )
    n_jobs_small_gpu: int = argfield(
        default=4,
        doc="Max workers to use for small jobs running on GPU.",
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
    # TODO: document
    work_in_tmpdir: bool = False
    copy_recording_to_tmpdir: bool = False
    workdir_copier: Literal["shutil", "rsync"] = "shutil"
    workdir_follow_symlinks: bool = False
    tmpdir_parent: str | None = argfield(default=None, arg_type=str_or_none)
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
        default=3.0,
        doc="Threshold in standardized (SNR) voltage units for initial detection; "
        "peaks or troughs larger than this value will be grabbed.",
    )
    matching_threshold: Annotated[float, Field(gt=0)] = argfield(
        default=8.0,
        doc="Template matching threshold. If subtracting a template leads "
        "to at least this great of a decrease in the norm of the residual, "
        "that match will be used.",
    )
    initial_threshold: Annotated[float, Field(gt=0)] = argfield(
        default=10.0,
        doc="Initial detection's neural net matching threshold. Same as "
        "matching_threshold, except that a neural net is trying to guess "
        "the true waveforms here, rather than using cluster templates.",
    )
    motion_voltage_threshold: Annotated[float, Field(gt=0)] = argfield(
        default=4.0,
        doc="If subsampling, a quick thresholding will be run at this voltage "
        "threshold to grab spikes for motion estimation purposes.",
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
        default=50.0,
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

    # -- subtraction neural net
    nn_denoiser_class_name: Literal["SingleChannelWaveformDenoiser", "Decollider"] = (
        argfield(
            default="Decollider",
            doc="Which neural net to use in initial detection? Set to Decollider (and set the pretrained "
            "path to None to train a  brand-new unsupervised denoiser.",
        )
    )
    nn_denoiser_pretrained_path: str | None = argfield(
        default=None,
        arg_type=str_or_none,
        doc="Path to a pytorch saved model (.pt file as dumped by torch.save()). If this is None, the "
        "model will be fit.",
    )

    # -- matching parameters
    # TODO: document
    amplitude_scaling_stddev: Annotated[float, Field(ge=0)] = 0.01
    amplitude_scaling_boundary: Annotated[float, Field(ge=0)] = 1.0 / 3.0
    temporal_upsamples: Annotated[int, Field(ge=1)] = 4

    # -- motion estimation parameters
    rigid: bool = argfield(
        default=False, doc="Use rigid registration and ignore the window parameters."
    )
    probe_boundary_padding_um: float = 100.0
    spatial_bin_length_um: Annotated[float, Field(gt=0)] = 1.0
    temporal_bin_length_s: Annotated[float, Field(gt=0)] = 1.0
    smoothing_um: Annotated[float, Field(gt=0)] | None = argfield(
        default=3.0, arg_type=float_or_none
    )
    smoothing_s: Annotated[float, Field(gt=0)] | None = argfield(
        default=None, arg_type=float_or_none
    )
    window_step_um: Annotated[float, Field(gt=0)] = 400.0
    window_scale_um: Annotated[float, Field(gt=0)] = 600.0
    window_margin_um: Annotated[float, Field(gt=0)] | None = argfield(
        default=None, arg_type=float_or_none
    )
    max_dt_s: Annotated[float, Field(gt=0)] = 500.0
    max_disp_um: Annotated[float, Field(gt=0)] | None = argfield(
        default=None, arg_type=float_or_none
    )
    correlation_threshold: Annotated[float, Field(gt=0, lt=1)] = 0.1
    min_amplitude: float | None = argfield(default=None, arg_type=float_or_none)
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
    dredge_only: bool = argfield(
        False, doc="Whether to stop after initial localization and motion tracking."
    )


@cfg_dataclass
class DeveloperConfig(DARTsortUserConfig):
    """Additional parameters for experiments. This API will never be stable."""

    # high level behavior
    initial_steps: Sequence[MixtureStep] = argfield(
        default=("split", "demolish"), arg_type=tuple
    )
    later_steps: Sequence[MixtureStep] = argfield(
        default=("split", "merge", "demolish"), arg_type=tuple
    )
    detection_type: Literal["subtract", "match", "threshold"] = "subtract"
    cluster_strategy: str = "dpc"
    refinement_strategy: str = "tmm"
    recluster_after_first_matching: bool = False

    # general peeling
    n_waveforms_fit: int = 40_000
    max_waveforms_fit: int = 50_000
    fit_sampling: Literal["random", "amp_reweighted"] = "amp_reweighted"
    n_residual_snips: int = 4 * 4096

    # initial detection
    nn_denoiser_max_waveforms_fit: int = 250_000
    do_tpca_denoise: bool = True
    first_denoiser_thinning: float = 0.0
    first_denoiser_spatial_dedup_radius: float = 100.0
    realign_to_denoiser: bool = True
    use_nn_in_subtraction: bool = True

    # matching
    matching_template_type: Literal["individual_compressed_upsampled", "drifty"] = (
        "drifty"
    )
    matching_up_method: Literal["interpolation", "keys3", "keys4", "direct"] = "keys4"
    matching_cd_iter: int = 0
    matching_coarse_cd: bool = True
    postprocessing_merge_threshold: float = 0.05
    template_spikes_per_unit: int = 500
    template_reduction: Literal["mean", "median"] = "median"
    template_denoising_method: Literal["none", "exp_weighted", "svd"] = "svd"
    min_template_snr: float = 0.0
    min_template_count: int = 20
    template_interp_kind: Literal["tps", "clampna"] = "tps"
    matching_interp_kind: Literal["tps", "clampna"] = "tps"
    matching_svd_rank: int = 5
    channel_selection_radius: float | None = argfield(
        default=None, arg_type=float_or_none
    )
    template_svd_method: TemplateSVDMethod = "raw_template"
    matching_template_min_amplitude: float = 1.0
    realign_strategy: RealignStrategy = "snr_weighted_trough_factor"
    trough_factor: float = 3.0
    whiten_strategy: WhiteningStrategy = "prewhiten_postapply"
    whiten_estimator: WhiteningEstimator = "localzca"
    whiten_features: bool = False
    matching_fp_control: bool = False
    refractory_radius_frames: int = 0
    svd_alignment_iterations: int = 0

    # interpolation for features
    interp_method: InterpMethod = "kriging"
    interp_kernel: InterpKernel = "thinplate"
    extrap_method: InterpMethod | None = argfield(default=None, arg_type=str_or_none)
    extrap_kernel: InterpKernel | None = argfield(default=None, arg_type=str_or_none)
    kriging_poly_degree: int = 1
    interp_sigma: float = 10.0
    rq_alpha: float = 0.5
    smoothing_lambda: float = 0.0
    polyharmonic_order: int = 2

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
    initial_rank: int | None = argfield(default=None, arg_type=int_or_none)
    initialize_at_rank_0: bool = False
    signal_rank: Annotated[int, Field(ge=0)] = 3
    gmm_max_spikes: Annotated[int, Field(gt=0)] = 2_048_000
    kmeansk: int = 4
    min_cluster_size: int = 25

    # gausian mixture low level
    n_refinement_iters: int = 1
    n_later_refinement_iters: int = 1
    n_em_iters: int = 250
    gmm_cl_alpha: float = 0.05
    gmm_cl_split_only: bool = True
    gmm_em_atol: float = 5e-3
    gmm_metric: Literal["cosine", "normeuc", "scaled_normeuc"] = "scaled_normeuc"
    gmm_n_candidates: int = 5
    gmm_n_search: int | None = argfield(default=3, arg_type=int_or_none)
    gmm_val_proportion: Annotated[float, Field(gt=0)] = 0.5
    initial_basis_shrinkage: float = 1.0
    prior_pseudocount: float = 0.0
    cov_kind: str = "factorizednoise"
    gmm_cosine_threshold: float = 0.8
    gmm_normeuc_threshold: float = 1.0
    gmm_scaled_normeuc_threshold: float = 1.5
    robust_strategy: Literal["none", "fixed"] = "none"
    robust_fixed_std_dataset: str = "collidedness"
    robust_fixed_power: float = 40.0
    robust_df: float = 4.0
    demolish_during_selection: bool = False
    em_after_demolish: bool = False

    # agglomeration
    agg_kind: Literal["none", "template_distance", "qda"] = "qda"
    agg_qda_max_template_distance: float = 0.6
    agg_no_qda_template_distance: float = 0.3
    agg_qda_linkage: Literal["single", "complete"] = "single"
    agg_template_linkage: Literal["single", "complete"] = "complete"
    agg_template_whiten_strategy: WhiteningStrategy = "none"

    # store extra intermediates@
    save_subtracted_waveforms: bool = False
    save_collisioncleaned_waveforms: bool = False
    always_save_final_tpca_feature: bool = False
    precomputed_templates_npz: str | None = argfield(default=None, arg_type=str_or_none)
    save_everything_on_error: bool = False
    link_from: str | None = argfield(default=None, arg_type=str_or_none)
    link_step: Literal["denoising", "detection", "refined0", "matching1"] = "refined0"
