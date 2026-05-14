from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import Field
from typing_extensions import Doc

from .util.internal_config import (
    InterpKernel,
    InterpMethod,
    MixtureStep,
    PreprocessingStrategy,
    RealignStrategy,
    TemplateSVDMethod,
    WhiteningEstimator,
    WhiteningStrategy,
    default_pretrained_path,  # noqa
)
from .util.py_util import cfg_dataclass


@cfg_dataclass
class DARTsortUserConfig:
    """User-facing configuration options"""

    # -- high level behavior
    do_motion_estimation: Annotated[
        bool,
        Doc(
            "Set this to false if your data is super stable or already motion-corrected."
        ),
    ] = True
    preprocessing: Annotated[
        PreprocessingStrategy,
        Doc(
            "If other than 'none', dartsort will apply a standard preprocessing "
            "to the recording. Leave as 'none' if you'd prefer to control preprocessing. "
            "If so, be aware that dartsort expects its input to be standardized on "
            "each channel in addition to the usual highpass filtering, but that "
            "whitening is handled internally. See util/preprocess_util.py if you're "
            "curious about the details of the methods."
        ),
    ] = "none"
    preprocessing_dtype: Annotated[
        Literal["float16", "float32"],
        Doc(
            "If you have a lot of data and you're using a workflow where it is important "
            "to save a preprocessed copy of the recording, float16 is a good option. Only "
            "relevant if preprocessing != 'none'. If the recording isn't getting saved, "
            "stick to float32."
        ),
    ] = "float32"
    subsampling_spikes: Annotated[
        int | None,
        Doc(
            "Detection steps before the final matching round will run until at least "
            "this many spikes are found or the whole recording is covered, to make sure "
            "that there is enough data for clustering. See also subsampling_fraction. "
            "Set to None to disable subsampling."
        ),
    ] = 2_048_000
    subsampling_presence: Annotated[
        float,
        Field(gt=0.0, le=1.0),
        Doc(
            "Early detection steps which have already found `subsampling_spikes` "
            "spikes are only allowed to end early if they additionally cover this "
            "fraction of the recording, to make sure there's good coverage of "
            "conditions for template estimation."
        ),
    ] = 0.1
    matching_iterations: Annotated[
        int,
        Doc(
            "By default, 1 template matching step is carried out using templates "
            "estimated from the initial detection round."
        ),
    ] = 1

    # -- computer options
    n_jobs_cpu: Annotated[
        int,
        Doc(
            "Number of parallel workers to use when running on CPU. "
            "0 means everything runs on the main thread; negative means "
            "#cpu - (val+1) so that -1 is all cores, -2 is all less 1, etc."
        ),
    ] = 0
    n_jobs_gpu: Annotated[
        int, Doc("Number of parallel workers to use when running on GPU.")
    ] = 0
    n_jobs_small: Annotated[int, Doc("Max workers to use for small jobs.")] = -2
    n_jobs_small_gpu: Annotated[
        int, Doc("Max workers to use for small jobs running on GPU.")
    ] = 4
    device: Annotated[
        str | None,
        Doc(
            "The name of the PyTorch device to use. For example, 'cpu' "
            "or 'cuda' or 'cuda:1'. If unset, uses n_jobs_gpu of your CUDA "
            "GPUs if you have multiple, or else just the one, or your CPU."
        ),
    ] = None
    executor: str = "threading_unless_multigpu"
    chunk_length_samples: int = 30_000

    # -- storage behavior
    # TODO: document
    work_in_tmpdir: bool = False
    copy_recording_to_tmpdir: bool = False
    workdir_copier: Literal["shutil", "rsync"] = "shutil"
    workdir_follow_symlinks: bool = False
    tmpdir_parent: str | None = None
    save_intermediates: bool = False
    save_final_features: bool = True

    # -- waveform snippet length parameters
    ms_before: Annotated[
        float,
        Field(gt=0),
        Doc(
            "Length of time (ms) before trough (or peak) in waveform snippets. "
            "Default value corresponds to 42 samples at 30kHz."
        ),
    ] = 1.4
    ms_after: Annotated[
        float,
        Field(gt=0),
        Doc(
            "Length of time (ms) after trough (or peak) in waveform snippets. "
            "Default value corresponds to 79 samples at 30kHz."
        ),
    ] = 2.6 + 0.1 / 3
    alignment_ms: Annotated[
        float, Field(gt=0), Doc("Largest time shift allowed when re-aligning events.")
    ] = 1.5

    # -- thresholds
    peak_sign: Annotated[
        Literal["neg", "both", "pos"],
        Doc(
            "Allow only troughs or events of both signs when detecting threshold "
            "crossings during initialization. Or positive only, if that's your thing."
        ),
    ] = "both"
    voltage_threshold: Annotated[
        float,
        Field(gt=0),
        Doc(
            "Threshold in standardized (SNR) voltage units for initial detection; "
            "peaks or troughs larger than this value will be grabbed."
        ),
    ] = 3.0
    matching_threshold: Annotated[
        float,
        Field(gt=0),
        Doc(
            "Template matching threshold. If subtracting a template leads "
            "to at least this great of a decrease in the norm of the residual, "
            "that match will be used."
        ),
    ] = 8.0
    initial_threshold: Annotated[
        float,
        Field(gt=0),
        Doc(
            "Initial detection's neural net matching threshold. Same as "
            "matching_threshold, except that a neural net is trying to guess "
            "the true waveforms here, rather than using cluster templates."
        ),
    ] = 10.0
    motion_voltage_threshold: Annotated[
        float,
        Field(gt=0),
        Doc(
            "If subsampling, a quick thresholding will be run at this voltage "
            "threshold to grab spikes for motion estimation purposes."
        ),
    ] = 4.0

    # -- featurization length, radius, rank parameters
    temporal_pca_rank: Annotated[
        int,
        Field(gt=0),
        Doc("Rank of temporal PCAs used in denoising and featurization."),
    ] = 8
    feature_ms_before: Annotated[
        float,
        Field(gt=0),
        Doc("As ms_before, but used only when computing PCA features in clustering."),
    ] = 0.75
    feature_ms_after: Annotated[
        float,
        Field(gt=0),
        Doc("As ms_after, but used only when computing PCA features in clustering."),
    ] = 1.25
    subtraction_radius_um: Annotated[
        float,
        Field(gt=0),
        Doc(
            "Radius of neighborhoods around spike events extracted "
            "when denoising and subtracting NN-denoised events."
        ),
    ] = 200.0
    deduplication_radius_um: Annotated[
        float,
        Field(gt=0),
        Doc(
            "During initial detection, if two spike events occur at the "
            "same time within this radius, then the smaller of the two is "
            "ignored. But also all of the secondary channels of the big one, "
            "which is important."
        ),
    ] = 50.0
    featurization_radius_um: Annotated[
        float,
        Field(gt=0),
        Doc(
            "Radius around detection channel or template peak channel used "
            "to extract spike features for clustering."
        ),
    ] = 100.0
    fit_radius_um: Annotated[
        float,
        Field(gt=0),
        Doc(
            "Extraction radius when fitting features like PCA; "
            "smaller than other radii to include less noise."
        ),
    ] = 75.0
    localization_radius_um: Annotated[
        float,
        Field(gt=0),
        Doc("Radius around main channel used when localizing spikes."),
    ] = 100.0

    # -- subtraction neural net
    nn_denoiser_class_name: Annotated[
        Literal["SingleChannelWaveformDenoiser", "Decollider"],
        Doc(
            "Which neural net to use in initial detection? Set to Decollider (and set the pretrained "
            "path to None to train a  brand-new unsupervised denoiser."
        ),
    ] = "Decollider"
    nn_denoiser_pretrained_path: Annotated[
        str | None,
        Doc(
            "Path to a pytorch saved model (.pt file as dumped by torch.save()). If this is None, the "
            "model will be fit."
        ),
    ] = None

    # -- matching parameters
    # TODO: document
    amplitude_scaling_stddev: Annotated[float, Field(ge=0)] = 0.01
    amplitude_scaling_boundary: Annotated[float, Field(ge=0)] = 1.0 / 3.0
    temporal_upsamples: Annotated[int, Field(ge=1)] = 4

    # -- motion estimation parameters
    rigid: Annotated[
        bool, Doc("Use rigid registration and ignore the window parameters.")
    ] = False
    probe_boundary_padding_um: float = 100.0
    spatial_bin_length_um: Annotated[float, Field(gt=0)] = 1.0
    temporal_bin_length_s: Annotated[float, Field(gt=0)] = 1.0
    smoothing_um: Annotated[float, Field(gt=0)] | None = 3.0
    smoothing_s: Annotated[float, Field(gt=0)] | None = None
    window_step_um: Annotated[float, Field(gt=0)] = 400.0
    window_scale_um: Annotated[float, Field(gt=0)] = 600.0
    window_margin_um: Annotated[float, Field(gt=0)] | None = None
    max_dt_s: Annotated[float, Field(gt=0)] = 500.0
    max_disp_um: Annotated[float, Field(gt=0)] | None = None
    correlation_threshold: Annotated[float, Field(gt=0, lt=1)] = 0.1
    min_amplitude: float | None = None
    speed_limit_um_per_s: Annotated[
        float,
        Doc("Motion bins exceeding this speed will be replaced by interpolation."),
    ] = 500.0
    max_dist_from_median_um: Annotated[
        float,
        Doc(
            "Motion bins farther than this from the local median will be replaced by interpolation."
        ),
    ] = 250.0
    median_neighborhood_bins: int = 51
    dredge_only: Annotated[
        bool, Doc("Whether to stop after initial localization and motion tracking.")
    ] = False


@cfg_dataclass
class DeveloperConfig(DARTsortUserConfig):
    """Additional parameters for experiments. This API will never be stable."""

    # high level behavior
    initial_steps: Sequence[MixtureStep] = ("split", "demolish")
    later_steps: Sequence[MixtureStep] = ("split", "merge", "demolish")
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
    channel_selection_radius: float | None = None
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
    extrap_method: InterpMethod | None = None
    extrap_kernel: InterpKernel | None = None
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
    n_neighbors_search: int | None = 50

    # gaussian mixture high level
    initial_rank: int | None = None
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
    gmm_n_search: int | None = 3
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
    precomputed_templates_npz: str | None = None
    save_everything_on_error: bool = False
    link_from: str | None = None
    link_step: Literal["denoising", "detection", "refined0", "matching1"] = "refined0"
