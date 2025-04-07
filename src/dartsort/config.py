"""Knobs"""

from typing import Annotated

from pydantic import Field
from pydantic.dataclasses import dataclass

from .util.internal_config import *


@dataclass(frozen=True, kw_only=True, slots=True)
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

    # -- waveform snippet length parameters
    ms_before: Annotated[float, Field(gt=0)] = argfield(
        default=1.4,
        doc="Length of time (ms) before trough (or peak) in waveform snippets. "
        "Default value corresponds to 42 samples at 30kHz.",
    )
    ms_after: Annotated[float, Field(gt=0)] = argfield(
        default=2.6,
        doc="Length of time (ms) after trough (or peak) in waveform snippets. "
        "Default value corresponds to 79 samples at 30kHz.",
    )
    alignment_ms: Annotated[float, Field(gt=0)] = argfield(
        default=1.0,
        doc="Time shift allowed when aligning events.",
    )

    # -- thresholds
    initial_threshold: Annotated[float, Field(gt=0)] = argfield(
        default=4.0,
        doc="Threshold in standardized voltage units for initial detection; "
        "peaks or troughs larger than this value will be grabbed.",
    )
    matching_threshold: Annotated[float, Field(gt=0)] | Literal["fp_control"] = (
        argfield(
            default=15.0,
            doc="Template matching threshold. If subtracting a template leads "
            "to at least this great of a decrease in the norm of the residual, "
            "that match will be used.",
        )
    )
    denoiser_badness_factor: Annotated[float, Field(gt=0, lt=1)] = argfield(
        default=0.1,
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
        default=150.0,
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
    fit_radius_um: Annotated[int, Field(gt=0)] = argfield(
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
    interpolation_bandwidth: Annotated[float, Field(gt=0)] = 20.0

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


@dataclass(frozen=True, kw_only=True, slots=True)
class DeveloperConfig(DARTsortUserConfig):
    """Additional parameters for experiments. This API will never be stable."""

    use_nn_in_subtraction: bool = True
    use_singlechan_templates: bool = False
    use_universal_templates: bool = False
    signal_rank: Annotated[int, Field(ge=0)] = 0
    truncated: bool = True
    overwrite_matching: bool = False

    merge_criterion_threshold: float = 0.0
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
    n_refinement_iters: int = 3
    n_em_iters: int = 50
    channels_strategy: str = "count"
    hard_noise: bool = False

    gmm_max_spikes: Annotated[int, Field(gt=0)] = 4_000_000
    gmm_val_proportion: Annotated[float, Field(gt=0)] = 0.25
    gmm_split_decision_algorithm: str = "tree"
    gmm_merge_decision_algorithm: str = "brute"

    save_intermediate_labels: bool = False
