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

TODO: Add a CommonConfig for parameters which show up in multiple
      of the below classes, so that users don't forget to change
      them in multiple places. Then the rest of the configs eat
      a commonconfig.
"""
from dataclasses import dataclass
from typing import List, Optional

try:
    from importlib.resources import files
except ImportError:
    try:
        from importlib_resources import files
    except ImportError:
        raise ValueError("Need python>=3.10 or pip install importlib_resources.")

default_pretrained_path = files("dartsort.pretrained")
default_pretrained_path = default_pretrained_path.joinpath("single_chan_denoiser.pt")



# TODO: integrate this in the other configs
@dataclass(frozen=True)
class WaveformConfig:
    """Defaults yield 42 sample trough offset and 121 total at 30kHz."""

    ms_before: float = 1.4
    ms_after: float = 2.6

    def trough_offset_samples(self, sampling_frequency=30_000):
        return int(self.ms_before * (sampling_frequency / 1000))

    def spike_length_samples(self, sampling_frequency=30_000):
        spike_len_ms = self.ms_before + self.ms_after
        return int(spike_len_ms * (sampling_frequency / 1000))


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
    """

    # -- denoising configuration
    do_nn_denoise: bool = True
    do_tpca_denoise: bool = True
    do_enforce_decrease: bool = True
    # turn off features below
    denoise_only: bool = False

    # -- featurization configuration
    save_input_waveforms: bool = False
    save_input_tpca_projs: bool = True
    save_output_waveforms: bool = False
    save_output_tpca_projs: bool = False
    save_amplitudes: bool = True
    # localization runs on output waveforms
    do_localization: bool = True
    localization_radius: float = 100.0
    # these are saved always if do_localization
    save_amplitude_vectors: bool = True
    localization_model = "dipole"

    # -- further info about denoising
    # in the future we may add multi-channel or other nns
    nn_denoiser_class_name: str = "SingleChannelWaveformDenoiser"
    nn_denoiser_pretrained_path: str = default_pretrained_path
    # optionally restrict how many channels TPCA are fit on
    tpca_fit_radius: Optional[float] = None
    tpca_rank: int = 8

    # used when naming datasets saved to h5 files
    input_waveforms_name: str = "collisioncleaned"
    output_waveforms_name: str = "denoised"


@dataclass(frozen=True)
class SubtractionConfig:
    trough_offset_samples: int = 42
    spike_length_samples: int = 121
    detection_thresholds: List[int] = (10, 8, 6, 5, 4)
    chunk_length_samples: int = 30_000
    peak_sign: str = "neg"
    spatial_dedup_radius: float = 150.0
    extract_radius: float = 200.0
    n_chunks_fit: int = 40
    fit_subsampling_random_state: int = 0
    residnorm_decrease_threshold: float = 3.162  # sqrt(10)

    # how will waveforms be denoised before subtraction?
    # users can also save waveforms/features during subtraction
    subtraction_denoising_config: FeaturizationConfig = FeaturizationConfig(
        denoise_only=True,
        input_waveforms_name="raw",
        output_waveforms_name="subtracted",
    )


@dataclass(frozen=True)
class TemplateConfig:
    trough_offset_samples: int = 42
    spike_length_samples: int = 121
    spikes_per_unit = 500

    # -- template construction parameters
    # registered templates?
    registered_templates: bool = True
    registered_template_localization_radius_um: float = 100.0

    # superresolved templates
    superres_templates: bool = True
    superres_bin_size_um: float = 10.0
    superres_bin_min_spikes: int = 5
    superres_strategy: str = "drift_pitch_loc_bin"

    # low rank denoising?
    low_rank_denoising: bool = True
    denoising_rank: int = 5
    denoising_snr_threshold: float = 50.0
    denoising_fit_radius: float = 75.0

    # realignment
    # TODO: maybe this should be done in clustering?
    realign_peaks: bool = True
    realign_max_sample_shift: int = 20


@dataclass(frozen=True)
class MatchingConfig:
    trough_offset_samples: int = 42
    spike_length_samples: int = 121
    chunk_length_samples: int = 30_000
    extract_radius: float = 100.0
    n_chunks_fit: int = 40
    fit_subsampling_random_state: int = 0

    # template matching parameters
    threshold: float = 50.0
    template_svd_compression_rank: int = 10
    template_temporal_upsampling_factor: int = 8
    template_min_channel_amplitude: float = 1.0
    refractory_radius_frames: int = 10
    amplitude_scaling_variance: float = 0.0
    amplitude_scaling_boundary: float = 0.5
    max_iter: int = 1000
    conv_ignore_threshold: float = 5.0
    coarse_approx_error_threshold: float = 5.0


@dataclass(frozen=True)
class ClusteringConfig:
    # -- initial clustering
    feature_scales = (1, 1, 50)
    log_c: int = 5
    # hdbscan parameters
    cluster_strategy: str = "hdbscan"
    min_cluster_size: int = 25
    min_samples: int = 25
    cluster_selection_epsilon: int = 1
    # -- ensembling
    ensemble_strategy: Optional[str] = "forward_backward"
    chunk_size_s: int = 150
    # forward-backward


default_featurization_config = FeaturizationConfig()
default_subtraction_config = SubtractionConfig()
default_template_config = TemplateConfig()
default_clustering_config = ClusteringConfig()
coarse_template_config = TemplateConfig(superres_templates=False)
default_matching_config = MatchingConfig()
