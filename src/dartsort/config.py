"""Configuration classes

Users should not edit this file!

Rather, make your own custom configs by instantiating new
config objects, for example, to turn off neural net denoising
in the featurization pipeline you can make:

```
featurization_config = FeaturizationConfig(do_nn_denoise=False)
```

This will use all the other parameters' default values. This
object can then be passed into the high level functions like
`subtract(...)`.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    from importlib.resources import files
except ImportError:
    try:
        from importlib_resources import files
    except ImportError:
        raise ValueError("Need python>=3.10 or pip install importlib_resources.")

default_pretrained_path = files("dartsort.pretrained")
default_pretrained_path = default_pretrained_path.joinpath("single_chan_denoiser.pt")


@dataclass(frozen=True)
class WaveformConfig:
    """Defaults yield 42 sample trough offset and 121 total at 30kHz."""

    ms_before: float = 1.4
    ms_after: float = 2.6

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


@dataclass(frozen=True)
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

    # -- denoising configuration
    do_nn_denoise: bool = True
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
    save_all_amplitudes: bool = True
    # localization runs on output waveforms
    do_localization: bool = True
    localization_radius: float = 100.0
    # these are saved always if do_localization
    localization_amplitude_type: str = "peak"
    localization_model: str = "pointsource"
    nn_localization: bool = True

    # -- further info about denoising
    # in the future we may add multi-channel or other nns
    nn_denoiser_class_name: str = "SingleChannelWaveformDenoiser"
    nn_denoiser_pretrained_path: Optional[str] = default_pretrained_path
    nn_denoiser_train_epochs: int = 50
    nn_denoiser_extra_kwargs: Optional[dict] = None

    # optionally restrict how many channels TPCA are fit on
    tpca_fit_radius: Optional[float] = None
    tpca_rank: int = 8
    tpca_centered: bool = True
    input_tpca_projs_temporal_slice: Optional[slice] = None

    # used when naming datasets saved to h5 files
    input_waveforms_name: str = "collisioncleaned"
    output_waveforms_name: str = "denoised"


@dataclass(frozen=True)
class SubtractionConfig:
    detection_thresholds: List[int] = (10, 8, 6, 5, 4)
    chunk_length_samples: int = 30_000
    peak_sign: str = "both"
    spatial_dedup_radius: float = 150.0
    subtract_radius: float = 200.0
    extract_radius: float = 100.0
    n_chunks_fit: int = 100
    max_waveforms_fit: int = 50_000
    n_waveforms_fit: int = 20_000
    fit_subsampling_random_state: int = 0
    fit_sampling: str = "random"
    residnorm_decrease_threshold: float = 3.162  # sqrt(10)

    # how will waveforms be denoised before subtraction?
    # users can also save waveforms/features during subtraction
    subtraction_denoising_config: FeaturizationConfig = FeaturizationConfig(
        denoise_only=True,
        input_waveforms_name="raw",
        output_waveforms_name="subtracted",
    )


@dataclass(frozen=True)
class MotionEstimationConfig:
    """Configure motion estimation.

    You can also make your own and pass it to dartsort() to bypass this
    """

    do_motion_estimation: bool = True

    # sometimes spikes can be localized far away from the probe, causing
    # issues with motion estimation, we will ignore such spikes
    probe_boundary_padding_um: float = 100.0

    # DREDge parameters
    spatial_bin_length_um: float = 1.0
    temporal_bin_length_s: float = 1.0
    window_step_um: float = 400.0
    window_scale_um: float = 450.0
    window_margin_um: Optional[float] = None
    max_dt_s: float = 1000.0
    max_disp_um: Optional[float] = None
    correlation_threshold: float = 0.1
    min_amplitude: Optional[float] = None
    rigid: bool = False


@dataclass(frozen=True)
class TemplateConfig:
    spikes_per_unit: int = 500

    # -- template construction parameters
    # registered templates?
    registered_templates: bool = True
    registered_template_localization_radius_um: float = 100.0

    # superresolved templates
    superres_templates: bool = True
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
    # TODO: maybe this should be done in clustering?
    realign_peaks: bool = True
    realign_max_sample_shift: int = 20

    # track template over time
    time_tracking: bool = False
    chunk_size_s: int = 300


@dataclass(frozen=True)
class MatchingConfig:
    chunk_length_samples: int = 30_000
    extract_radius: float = 100.0
    n_chunks_fit: int = 100
    max_waveforms_fit: int = 50_000
    n_waveforms_fit: int = 20_000
    fit_subsampling_random_state: int = 0
    fit_sampling: str = "random"

    # template matching parameters
    threshold: float = 150.0
    template_svd_compression_rank: int = 10
    template_temporal_upsampling_factor: int = 8
    template_min_channel_amplitude: float = 1.0
    refractory_radius_frames: int = 10
    amplitude_scaling_variance: float = 0.0
    amplitude_scaling_boundary: float = 0.5
    max_iter: int = 1000
    conv_ignore_threshold: float = 5.0
    coarse_approx_error_threshold: float = 0.0
    coarse_objective: bool = True


@dataclass(frozen=True)
class SplitMergeConfig:
    # -- split
    split_strategy: str = "FeatureSplit"
    recursive_split: bool = True
    split_strategy_kwargs: Optional[dict] = field(
        default_factory=lambda: dict(max_spikes=20_000)
    )

    # -- merge
    merge_template_config: TemplateConfig = TemplateConfig(superres_templates=False)
    linkage: str = "complete"
    merge_distance_threshold: float = 0.25
    cross_merge_distance_threshold: float = 0.5
    min_spatial_cosine: float = 0.0


@dataclass(frozen=True)
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
    sigma_regional: Optional[float] = 25.0
    workers: Optional[int] = -1
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
    sigma_local_low: Optional[float] = None
    sigma_regional_low: Optional[float] = None
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
    ensemble_strategy: Optional[str] = None
    chunk_size_s: float = 300.0
    split_merge_ensemble_config: SplitMergeConfig = SplitMergeConfig()


@dataclass(frozen=True)
class ComputationConfig:
    n_jobs_cpu: int = 0
    n_jobs_gpu: int = 0
    device: Optional[torch.device] = None

    @property
    def actual_device(self):
        if self.device is None:
            have_cuda = torch.cuda.is_available()
            return torch.device("cuda" if have_cuda else "cpu")
        return torch.device(self.device)

    @property
    def actual_n_jobs_gpu(self):
        if self.actual_device.type == "cuda":
            return self.n_jobs_gpu
        return self.n_jobs_cpu


@dataclass(frozen=True)
class DARTsortConfig:
    waveform_config: WaveformConfig = WaveformConfig()
    featurization_config: FeaturizationConfig = FeaturizationConfig()
    subtraction_config: SubtractionConfig = SubtractionConfig()
    template_config: TemplateConfig = TemplateConfig()
    clustering_config: ClusteringConfig = ClusteringConfig()
    split_merge_config: SplitMergeConfig = SplitMergeConfig()
    matching_config: MatchingConfig = MatchingConfig()
    motion_estimation_config: MotionEstimationConfig = MotionEstimationConfig()
    computation_config: ComputationConfig = ComputationConfig()

    # high level behavior
    subtract_only: bool = False
    do_initial_split_merge: bool = True
    do_final_split_merge: bool = False
    matching_iterations: int = 1
    intermediate_matching_subsampling: float = 1.0


default_waveform_config = WaveformConfig()
default_featurization_config = FeaturizationConfig()
default_subtraction_config = SubtractionConfig()
default_template_config = TemplateConfig()
default_clustering_config = ClusteringConfig()
default_split_merge_config = SplitMergeConfig()
coarse_template_config = TemplateConfig(superres_templates=False)
raw_template_config = TemplateConfig(
    realign_peaks=False, low_rank_denoising=False, superres_templates=False
)
unshifted_raw_template_config = TemplateConfig(
    registered_templates=False, realign_peaks=False, low_rank_denoising=False, superres_templates=False
)
unaligned_coarse_denoised_template_config = TemplateConfig(
    realign_peaks=False, low_rank_denoising=True, superres_templates=False
)
default_matching_config = MatchingConfig()
default_motion_estimation_config = MotionEstimationConfig()
default_computation_config = ComputationConfig()
default_dartsort_config = DARTsortConfig()
