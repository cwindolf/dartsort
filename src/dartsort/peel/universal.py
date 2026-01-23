from typing import Literal
import numpy as np
import torch

from ..templates.template_util import LowRankTemplates, compressed_upsampled_templates
from ..transform import WaveformPipeline
from ..util import universal_util, waveform_util
from ..util.internal_config import (
    UniversalMatchingConfig,
    FitSamplingConfig,
    FeaturizationConfig,
    WaveformConfig,
)
from .matching import ObjectiveUpdateTemplateMatchingPeeler
from .matching_util.compressed_upsampled import CompressedUpsampledMatchingTemplates
from .matching_util.pairwise import SeparablePairwiseConv


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
        threshold=100.0,
        trough_offset_samples=42,
        spike_length_samples=121,
        amplitude_scaling_variance=100.0,
        amplitude_scaling_boundary=500.0,
        detection_threshold=6.0,
        alignment_padding=20,
        n_centroids=6,
        pca_rank=8,
        taper=True,
        n_sigmas=5,
        min_template_size=10.0,
        max_distance=32.0,
        dx=32.0,
        chunk_length_samples=1000,
        n_seconds_fit=40,
        max_waveforms_fit=50_000,
        n_waveforms_fit=20_000,
        fit_subsampling_random_state=0,
        fit_sampling: Literal["random", "amp_reweighted"] = "random",
        fit_max_reweighting=4.0,
        refractory_radius_frames=10,
        max_iter=1000,
        dtype=torch.float,
    ):
        shapes, footprints, template_data = (
            universal_util.universal_templates_from_data(
                rec=recording,
                detection_threshold=detection_threshold,
                trough_offset_samples=trough_offset_samples,
                spike_length_samples=spike_length_samples,
                alignment_padding=alignment_padding,
                n_centroids=n_centroids,
                pca_rank=pca_rank,
                n_waveforms_fit=n_waveforms_fit,
                taper=taper,
                taper_start=alignment_padding // 2,
                taper_end=alignment_padding // 2,
                random_seed=fit_subsampling_random_state,
                n_sigmas=n_sigmas,
                min_template_size=min_template_size,
                max_distance=max_distance,
                dx=dx,
                # let's not worry about exposing these
                deduplication_radius=150.0,
                kmeanspp_initial="random",
            )
        )

        # TODO: this should all be put in a matchingtemplate+builder
        # so that it is lazy and resuming works better.
        Nf = len(footprints)
        Ns = len(shapes)
        shapes_ixd = torch.asarray(shapes)[None]
        shapes_ixd = shapes_ixd.broadcast_to((Nf, Ns, *shapes.shape[1:]))
        shapes_ixd = shapes_ixd.reshape(Nf * Ns, *shapes.shape[1:], 1)
        footprints_ixd = torch.asarray(footprints)[:, None]
        footprints_ixd = footprints_ixd.broadcast_to((Nf, Ns, *footprints.shape[1:]))
        footprints_ixd = footprints_ixd.reshape(Nf * Ns, 1, *footprints.shape[1:])
        low_rank_templates = LowRankTemplates(
            unit_ids=np.arange(Nf * Ns),
            temporal_components=shapes_ixd.numpy(),
            singular_values=shapes_ixd.new_ones(Nf * Ns, 1).numpy(),
            spatial_components=footprints_ixd.numpy(),
            spike_counts_by_channel=np.broadcast_to(
                np.atleast_2d([100]), (Nf * Ns, footprints.shape[1])
            ),
        )
        pairwise_conv_db = SeparablePairwiseConv(footprints, shapes)

        cupt = compressed_upsampled_templates(
            low_rank_templates.temporal_components, max_upsample=1
        )

        # TODO: re-using the individually compressed basis algorithm here
        # which is general enough that it works, but for this separable
        # basis there is a faster algorithm.
        matching_templates = CompressedUpsampledMatchingTemplates(
            trough_offset_samples=trough_offset_samples,
            low_rank_templates=low_rank_templates,
            pconv_db=pairwise_conv_db,
            compressed_upsampled_temporal=cupt,
        )

        super().__init__(
            recording=recording,
            matching_templates=matching_templates,
            channel_index=channel_index,
            featurization_pipeline=featurization_pipeline,
            threshold=threshold,
            amplitude_scaling_variance=amplitude_scaling_variance,
            amplitude_scaling_boundary=amplitude_scaling_boundary,
            trough_offset_samples=trough_offset_samples,
            chunk_length_samples=chunk_length_samples,
            n_seconds_fit=n_seconds_fit,
            max_waveforms_fit=max_waveforms_fit,
            n_waveforms_fit=n_waveforms_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
            fit_sampling=fit_sampling,
            fit_max_reweighting=fit_max_reweighting,
            dtype=dtype,
            refractory_radius_frames=refractory_radius_frames,
            max_iter=max_iter,
        )

    @classmethod
    def from_config(  # type: ignore[reportIncompatibleOverride]
        cls,
        recording,
        *,
        universal_cfg: UniversalMatchingConfig,
        featurization_cfg: FeaturizationConfig,
        waveform_cfg: WaveformConfig,
        sampling_cfg: FitSamplingConfig,
    ):
        geom = torch.tensor(recording.get_channel_locations())
        channel_index = waveform_util.make_channel_index(
            geom, featurization_cfg.extract_radius, to_torch=True
        )
        featurization_pipeline = WaveformPipeline.from_config(
            geom=geom,
            channel_index=channel_index,
            featurization_cfg=featurization_cfg,
            waveform_cfg=waveform_cfg,
            sampling_frequency=recording.sampling_frequency,
        )
        trough_offset_samples = waveform_cfg.trough_offset_samples(
            recording.sampling_frequency
        )
        spike_length_samples = waveform_cfg.spike_length_samples(
            recording.sampling_frequency
        )
        fs_ms = recording.sampling_frequency / 1000
        alignment_padding = int(universal_cfg.alignment_padding_ms * fs_ms)
        return cls(
            recording,
            chunk_length_samples=universal_cfg.chunk_length_samples,
            n_seconds_fit=universal_cfg.n_seconds_fit,
            n_waveforms_fit=sampling_cfg.n_waveforms_fit,
            max_waveforms_fit=sampling_cfg.max_waveforms_fit,
            fit_subsampling_random_state=sampling_cfg.fit_subsampling_random_state,
            fit_sampling=sampling_cfg.fit_sampling,
            fit_max_reweighting=sampling_cfg.fit_max_reweighting,
            threshold=universal_cfg.threshold,
            detection_threshold=universal_cfg.detection_threshold,
            n_sigmas=universal_cfg.n_sigmas,
            channel_index=channel_index,
            alignment_padding=alignment_padding,
            featurization_pipeline=featurization_pipeline,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
        )
