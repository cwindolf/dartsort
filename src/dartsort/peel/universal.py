import torch

from ..util import universal_util
from .matching import ObjectiveUpdateTemplateMatchingPeeler


class UniversalTemplatesMatchingPeeler(ObjectiveUpdateTemplateMatchingPeeler):
    """KS-style universal-templates-from-data detection

    This tries to rephrase their algorithm as faithfully as possible
    using dartsort tools, for comparison purposes with our algorithms.

    The idea is to estimate some (they use 6, it turns out) single-channel
    shapes via K means applied to single-channel waveforms. These
    are then expanded out into a full template library by spatial
    convs with various Gaussians. Then, throw them into the matcher.
    Since KS' matcher has scale_std --> infty, we can put a large
    scale prior variance to match the spirit of the thing.

    TODO maybe I should implement scale_prior->infty in our matcher?
    """

    def __init__(
        self,
        recording,
        channel_index,
        featurization_pipeline,
        threshold=50.0,
        trough_offset_samples=42,
        spike_length_samples=121,
        singlechan_detection_threshold=6.0,
        singlechan_alignment_padding=20,
        svd_compression_rank=10,
        amplitude_scaling_variance=100.0,
        amplitude_scaling_boundary=500.0,
        chunk_length_samples=30_000,
        n_chunks_fit=40,
        max_waveforms_fit=50_000,
        n_waveforms_fit=20_000,
        fit_subsampling_random_state=0,
        fit_sampling="random",
        dtype=torch.float,
    ):
        template_data = universal_util.universal_templates_from_data(...)
        super().__init__(
            recording,
            template_data,
            channel_index,
            featurization_pipeline,
            threshold=threshold,
            amplitude_scaling_variance=amplitude_scaling_variance,
            amplitude_scaling_boundary=amplitude_scaling_boundary,
            svd_compression_rank=svd_compression_rank,
            # usual gizmos
            trough_offset_samples=trough_offset_samples,
            chunk_length_samples=chunk_length_samples,
            n_chunks_fit=n_chunks_fit,
            max_waveforms_fit=max_waveforms_fit,
            n_waveforms_fit=n_waveforms_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
            fit_sampling=fit_sampling,
            dtype=dtype,
            # matching params which don't need setting
            min_channel_amplitude=1.0,
            motion_est=None,
            coarse_approx_error_threshold=0.0,
            conv_ignore_threshold=5.0,
            coarse_objective=True,
            temporal_upsampling_factor=1,
            refractory_radius_frames=10,
            max_iter=1000,
        )
