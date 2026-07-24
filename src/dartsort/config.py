from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import Field

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
    """User-facing configuration options

    To change dartsort's behavior, set parameters here and pass the object
    to the `dartsort()` function.
    """

    # -- high level behavior

    do_motion_estimation: bool = True
    """Set this to false if your data is super stable or already motion-corrected."""

    preprocessing: PreprocessingStrategy = "none"
    """If other than `'none'`, dartsort will apply some preprocessing to the
    recording. Leave as `'none'` if you are passing in an already-preprocesed
    recording. If so, be aware that dartsort expects its input to be standardized on
    each channel in addition to the usual highpass filtering, but that
    whitening is handled internally. See util/preprocess_util.py if you're
    curious about the details of the methods.

    Options: `'ibllikecmr', 'ibllike', 'standardize', 'none'`
    """

    preprocessing_dtype: Literal["float16", "float32"] = "float32"
    """If you have a lot of data and you're using a workflow where it is important
    to save a preprocessed copy of the recording, float16 is a good option. Only
    relevant if `preprocessing != 'none'`. If the recording isn't getting saved,
    stick to float32."""

    subsampling_spikes_per_channel: int | None = 5000
    """Detection steps before the final matching round will run until at least
    this many spikes are found or the whole recording is covered, to make sure
    that there is enough data for clustering. See also subsampling_presence.
    Set to None to disable subsampling."""

    subsampling_presence: Annotated[float, Field(gt=0.0, le=1.0)] = 0.025
    """Early detection steps which have already found `subsampling_spikes`
    spikes are only allowed to end early if they additionally cover this
    fraction of the recording, to make sure there's some coverage of
    conditions for template estimation."""

    matching_iterations: int = 1
    """By default, 1 template matching step is carried out using templates
    estimated from the initial detection round."""

    dredge_only: bool = False
    """Whether to stop after initial localization and motion tracking."""

    # -- computer options
    n_jobs_cpu: int = 0
    """Number of parallel workers to use when running on CPU. 0 means
    everything runs on the main thread; negative means `#cpu - (val+1)`
    so that -1 is all cores, -2 is all less 1, etc."""

    n_jobs_gpu: int = 0
    """Number of parallel workers to use when running on GPU."""

    n_jobs_small: int = -2
    """Max workers to use for small jobs."""

    n_jobs_small_gpu: int = 4
    """Max workers to use for small jobs running on GPU."""

    device: str | None = None
    """The name of the PyTorch device to use. For example, 'cpu'
    or 'cuda' or 'cuda:1'. If unset, uses n_jobs_gpu of your CUDA
    GPUs if you have multiple, or else just the one, or your CPU."""

    executor: str = "threading_unless_multigpu"
    """Choose: 'threading_unless_multigpu', 'ThreadPoolExecutor', 'ProcessPoolExecutor', or some others."""

    chunk_length_samples: int = 30_000
    """Batch size for data processing."""

    # -- storage behavior
    work_in_tmpdir: bool = False
    """If True, dartsort will store all temporary data in a scratch directory in tmpdir_parent or TMPDIR."""

    copy_recording_to_tmpdir: bool = False
    """Save a copy of the preprocessed recording to a tmpdir?"""

    workdir_copier: Literal["shutil", "rsync"] = "shutil"
    """'shutil' or 'rsync'"""

    workdir_follow_symlinks: bool = False
    tmpdir_parent: str | None = None
    """Control where tmpdirs are created."""
    save_intermediates: bool = False
    """Store all spike features from intermediate steps (for debugging)"""

    save_final_features: bool = True
    """Store the spike features from the final step (instead of just basic spike train outputs)."""

    # -- waveform snippet length parameters
    ms_before: Annotated[float, Field(gt=0)] = 1.4
    """Length of time (ms) before trough (or peak) in waveform snippets.
    Default value corresponds to 42 samples at 30kHz."""

    ms_after: Annotated[float, Field(gt=0)] = 2.6 + 0.1 / 3
    """Length of time (ms) after trough (or peak) in waveform snippets.
    Default value corresponds to 79 samples at 30kHz."""

    alignment_ms: Annotated[float, Field(gt=0)] = 1.5
    """Largest time shift allowed when re-aligning events."""

    deduplication_ms: Annotated[float, Field(gt=0)] = 0.5
    """As a final postprocessing step, only the higher-scoring of any spikes within
    this time radius of each other are kept.
    If this is negative, it does nothing. If it's 0, exact duplicates are dropped.
    """

    # -- thresholds
    peak_sign: Literal["neg", "both", "pos"] = "both"
    """Allow only troughs or events of both signs when detecting threshold
    crossings during initialization. Or positive only, if that's your thing."""

    voltage_threshold: Annotated[float, Field(gt=0)] = 3.0
    """Threshold in standardized (SNR) voltage units for initial detection;
    peaks or troughs larger than this value will be grabbed."""

    matching_threshold: Annotated[float, Field(gt=0)] = 6.0
    """Template matching threshold. If subtracting a template leads
    to at least this great of a decrease in the norm of the residual,
    that match will be used. This is in the same units as the corresponding
    threshold in Kilosort and other sorters, and it represents reduction in
    Euclidean norm of standardized data due to matching a new event."""

    initial_threshold: Annotated[float, Field(gt=0)] = 7.0
    """Initial detection's neural net matching threshold. Same as
    matching_threshold, except that a neural net is trying to guess
    the true waveforms here, rather than using cluster templates."""

    motion_voltage_threshold: Annotated[float, Field(gt=0)] = 4.0
    """If subsampling, a quick thresholding will be run at this voltage
    threshold to grab spikes for motion estimation purposes."""

    # -- featurization length, radius, rank parameters
    temporal_pca_rank: Annotated[int, Field(gt=0)] = 8
    """Rank of temporal PCAs used in denoising and featurization."""

    feature_ms_before: Annotated[float, Field(gt=0)] = 0.75
    """As ms_before, but used only when computing PCA features in clustering."""

    feature_ms_after: Annotated[float, Field(gt=0)] = 1.25
    """As ms_after, but used only when computing PCA features in clustering."""

    subtraction_radius_um: Annotated[float, Field(gt=0)] = 200.0
    """Radius of neighborhoods around spike events extracted
    when denoising and subtracting NN-denoised events."""

    deduplication_radius_um: Annotated[float, Field(gt=0)] = 50.0
    """During initial detection, if two spike events occur at the
    same time within this radius, then the smaller of the two is
    ignored. But also all of the secondary channels of the big one,
    which is important."""

    featurization_radius_um: Annotated[float, Field(gt=0)] = 100.0
    """Radius around detection channel or template peak channel used
    to extract spike features for clustering."""

    fit_radius_um: Annotated[float, Field(gt=0)] = 75.0
    """Extraction radius when fitting features like PCA; smaller than other radii to include less noise."""

    localization_radius_um: Annotated[float, Field(gt=0)] = 100.0
    """Radius around main channel used when localizing spikes."""

    # -- subtraction neural net
    nn_denoiser_class_name: Literal["SingleChannelWaveformDenoiser", "Decollider"] = (
        "Decollider"
    )
    """Which neural net to use in initial detection? Set to Decollider (and set the pretrained
    path to None to train a  brand-new unsupervised denoiser."""

    nn_denoiser_pretrained_path: str | None = None
    """Path to a pytorch saved model (.pt file as dumped by torch.save()). If this is None, a new model will be fit."""

    # -- matching parameters
    amplitude_scaling_stddev: Annotated[float, Field(ge=0)] = 0.01
    """Standard deviation of amplitude scaling regularization prior in template matching."""

    amplitude_scaling_boundary: Annotated[float, Field(ge=0)] = 1.0 / 3.0
    """Boundaries on the amount of scaling allowed."""

    temporal_upsamples: Annotated[int, Field(ge=1)] = 4
    """Upsampling of templates during matching to allow for temporal aliasing of waveforms."""

    # -- final merge step
    agg_kind: Literal["none", "template_distance", "qda"] = "qda"
    """Final distance or GMM-based merge type."""

    spikeinterface_merge_preset: str | Literal["none"] = "none"
    """Call out to SpikeInterface's auto_merge() for a final merge using timing / RP information.
    Setting this is slightly different' from calling auto_merge() externally, since the internal
    version will make use of dartsort's templates and template distances.
    dartsort extends auto_merge() with some additional presets: dartsort_slay_xc_ccg,
    dartsort_slay_xc, dartsort_slay_ccg. These are conservative presets; see and cite
    Koukuntla et al., 2025 for the SLAY score criterion."""

    # -- motion estimation parameters
    rigid: bool = False
    """Use rigid registration and ignore the window parameters."""
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
    speed_limit_um_per_s: float = 500.0
    """Motion bins exceeding this speed will be replaced by interpolation."""
    max_dist_from_median_um: float = 250.0
    """Motion bins farther than this from the local median will be replaced by interpolation."""
    median_neighborhood_bins: int = 51


@cfg_dataclass
class DeveloperConfig(DARTsortUserConfig):
    """Additional parameters for experiments. This API will never be stable."""

    # high level behavior
    initial_steps: Sequence[MixtureStep] = ("split", "demolish", "demolish")
    later_steps: Sequence[MixtureStep] = ("split", "merge", "demolish")
    detection_type: Literal["subtract", "match", "threshold"] = "subtract"
    cluster_strategy: str = "dpc"
    refinement_strategy: str = "tmm"
    recluster_after_first_matching: bool = False

    # general peeling
    n_waveforms_fit: int = 40_000
    max_waveforms_fit: int = 50_000
    fit_sampling: Literal["random", "amp_reweighted"] = "amp_reweighted"
    n_residual_snips: int = 2 * 4096

    # initial detection
    nn_denoiser_max_waveforms_fit: int = 512_000
    nn_denoiser_noise_waveforms: int = 100 * 256
    nn_denoiser_extra_kwargs: dict | None = None
    do_tpca_denoise: bool = True
    first_denoiser_thinning: float = 0.0
    first_denoiser_spatial_dedup_radius: float = 100.0
    realign_to_denoiser: bool = True
    use_nn_in_subtraction: bool = True
    whiten_in_subtraction: bool = True
    threshold_before_whitening: float = 10.0
    temporal_dedup_radius_samples: int = 7
    positive_temporal_dedup_radius_samples: int = 41
    spikeinterface_merge_max_distance: float = 0.8

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
    min_template_count: int = 10
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
    whiten_temporal_length: int | None = 3
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
    clustering_max_spikes: Annotated[int, Field(gt=0)] = 1024 * 1000
    pre_refinement_merge: bool = True
    post_refinement_merge: bool = False
    pre_refinement_merge_metric: str = "normeuc"
    pre_refinement_merge_threshold: float = 0.1
    use_hellinger: bool = True
    density_bandwidth: Annotated[float, Field(gt=0)] = 5.0
    component_overlap: float = 0.95
    hellinger_strong: float = 0.0
    hellinger_weak: float = 0.0
    dpc_mop: bool = True
    n_neighbors_search: int | None = 50

    # filters
    gmm_isolation_threshold: float | None = None
    collision_cleaning_error_threshold: float | None = 0.3

    # gaussian mixture high level
    initial_rank: int | None = None
    initialize_at_rank_0: bool = False
    signal_rank: Annotated[int, Field(ge=0)] = 3
    gmm_max_spikes: Annotated[int, Field(gt=0)] = 2_048_000
    kmeansk: int = 4
    min_cluster_size: int = 5

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
    em_after_demolish: bool = True
    tpca_from_templates: bool = True

    # agglomeration
    agg_qda_max_template_distance: float = 0.6
    agg_no_qda_template_distance: float = 0.3
    agg_qda_linkage: Literal["single", "complete"] = "single"
    agg_template_linkage: Literal["single", "complete"] = "complete"
    agg_template_whiten_strategy: WhiteningStrategy = "none"

    # store extra intermediates@
    save_subtracted_waveforms: bool = False
    save_collisioncleaned_waveforms: bool = False
    always_save_detailed_features: bool = False
    precomputed_templates_npz: str | None = None
    save_everything_on_error: bool = False
    link_from: str | None = None
    link_step: Literal[
        "denoising", "detection", "refined0", "matching1_models", "matching1"
    ] = "refined0"
