"""A peeler implementing mean or median reduction for estimating template waveforms."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import numpy as np
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from spikeinterface.core import BaseRecording

from ..templates.get_templates import denoising_weights, fit_tsvd
from ..templates.templates import TemplateData
from ..transform.interp import WaveformInterpolator
from ..transform.pipeline import WaveformPipeline
from ..transform.reduction import TemplateWaveformReducer
from ..transform.temporal_pca import FullProbeTemporalPCAEmbedder
from ..transform.whiten import WaveformWhitener
from ..util.data_util import (
    DARTsortSorting,
    get_top_assignment_weights,
    subsample_by_count_and_valid_time,
)
from ..util.internal_config import (
    ComputationConfig,
    TemplateConfig,
    WaveformConfig,
    default_waveform_cfg,
)
from ..util.job_util import ensure_computation_config
from ..util.logging_util import get_logger
from ..util.motion import MotionInfo
from ..util.noise_util import SpatialWhitener
from ..util.py_util import ensure_path
from ..util.waveform_util import full_channel_index
from .grab import GrabAndFeaturize

logger = get_logger(__name__)


# -- TemplateData plugin


class ReductionTemplateData(TemplateData):
    _algorithm: ClassVar = "peelreduce"

    @classmethod
    def _from_config(
        cls,
        *,
        recording: BaseRecording,
        sorting: DARTsortSorting,
        template_cfg: TemplateConfig,
        waveform_cfg: WaveformConfig = default_waveform_cfg,
        motion: MotionInfo,
        tsvd=None,
        whitener: SpatialWhitener | None = None,
        computation_cfg: ComputationConfig | None = None,
        show_progress: bool = True,
    ) -> TemplateData:
        computation_cfg = ensure_computation_config(computation_cfg)
        # subsample sorting
        sorting = subsample_by_count_and_valid_time(
            sorting,
            max_spikes=template_cfg.spikes_per_unit,
            recording=recording,
            waveform_cfg=_handle_internal_pad(recording, waveform_cfg, template_cfg)[1],
        )
        sorting = sorting.ensure_no_missing()

        # build engine object
        sorting_flat = sorting.flatten()
        p = TemplateReduction.from_config(
            recording=recording,
            sorting=sorting_flat,
            motion=motion,
            waveform_cfg=waveform_cfg,
            template_cfg=template_cfg,
            computation_cfg=computation_cfg,
            whitener=whitener,
            tsvd=tsvd,
        )

        # TODO: file not always needed
        if template_cfg.reduction == "mean":
            # TODO: reducer doesn't work in parallel when gathering means, go to single job
            computation_cfg = ComputationConfig(
                device=computation_cfg.actual_device().type,
                n_jobs_small=computation_cfg.n_jobs_small,
                tmpdir_parent=computation_cfg.tmpdir_parent,
            )
        with TemporaryDirectory(
            prefix="dartsorttemplates",
            ignore_cleanup_errors=True,
            dir=computation_cfg.tmpdir_parent,
        ) as tdir:
            tdir = ensure_path(tdir)
            h5p = tdir / "tmp.h5"
            p.load_or_fit_and_save_models(tdir / "models")
            if template_cfg.denoising_method == "none" and not template_cfg.use_svd:
                task_name = "Raw templates"
            else:
                task_name = "Templates"
            p.peel(
                output_hdf5_filename=h5p,
                show_progress=show_progress,
                task_name=task_name,
                computation_cfg=computation_cfg,
                ignore_resuming=True,
                known_spike_count=len(sorting_flat),
            )

            # extract outputs and handle denoising method
            count, raw_mean, raw_std, svd_mean = p.reduction_results(
                h5p, computation_cfg=computation_cfg, show_progress=show_progress
            )

        trough = waveform_cfg.trough_offset_samples(recording.sampling_frequency)
        unit_ids = sorting.unit_ids
        if template_cfg.denoising_method == "svd":
            # svd-only templates
            assert svd_mean is not None
            assert raw_std is None
            templates = svd_mean
        elif template_cfg.denoising_method == "none":
            assert raw_mean is not None
            templates = raw_mean
        elif template_cfg.denoising_method == "exp_weighted":
            assert raw_mean is not None
            assert svd_mean is not None
            snrs_by_channel = np.ptp(raw_mean, 1) * np.sqrt(count)
            assert np.isfinite(snrs_by_channel).all()
            assert snrs_by_channel.max() > 0
            weights = denoising_weights(
                snrs=snrs_by_channel,
                spike_length_samples=raw_mean.shape[1],
                trough_offset=trough,
                snr_threshold=template_cfg.exp_weight_snr_threshold,
            )
            weights = weights.astype(raw_mean.dtype)
            logger.dartsortdebug(
                f"exp_weighted: weight mean/max={weights.mean().item()},{weights.max().item()}"
            )
            templates = weights * raw_mean + (1 - weights) * svd_mean
        else:
            assert False

        spike_counts = count.max(axis=1)
        if motion.drifting:
            msk = np.logical_or(
                count >= template_cfg.min_count_at_shift,
                count >= template_cfg.min_fraction_at_shift * spike_counts[:, None],
            )
            msk = msk[:, None, :].astype(templates.dtype)
            templates *= msk

        if whitener is None:
            whitener_np = covariance_np = None
        else:
            whitener_np, covariance_np = whitener.to_numpy()

        return TemplateData(
            unit_ids=unit_ids,
            templates=templates,
            raw_std_dev=raw_std,
            spike_counts=spike_counts,
            spike_counts_by_channel=count,
            registered_geom=motion.rgeom,
            trough_offset_samples=trough,
            tsvd=p.temporal_svd(),
            whitener=whitener_np,
            covariance=covariance_np,
            sampling_frequency=recording.sampling_frequency,
            whiten_strategy=template_cfg.whitening.strategy,
        )


# -- reduction engine


class TemplateReduction(GrabAndFeaturize):
    @classmethod
    def from_config(  # type: ignore
        cls,
        recording: BaseRecording,
        *,
        motion: MotionInfo,
        tsvd: TruncatedSVD | PCA | FullProbeTemporalPCAEmbedder | None,
        sorting: DARTsortSorting,
        waveform_cfg: WaveformConfig,
        template_cfg: TemplateConfig,
        computation_cfg: ComputationConfig,
        whitener: SpatialWhitener | None = None,
    ):
        # geom processing
        rgeom = torch.asarray(motion.rgeom)
        geom = torch.asarray(motion.geom)
        channel_index = full_channel_index(len(geom), to_torch=True)

        # internal realignment
        do_align, padded_waveform_cfg, trough, tslice, align_pad = _handle_internal_pad(
            recording, waveform_cfg, template_cfg
        )

        # handle tsvd fit preferences
        pad_spike_len = padded_waveform_cfg.spike_length_samples(
            recording.sampling_frequency
        )
        rank = template_cfg.denoising_rank
        if template_cfg.use_svd and tsvd is not None:
            if isinstance(tsvd, FullProbeTemporalPCAEmbedder):
                if do_align:
                    raise ValueError("Haven't handled svd alignment in this case.")
            else:
                assert tsvd.components_.shape[0] <= rank
                rank = tsvd.components_.shape[0]
                tsvd = FullProbeTemporalPCAEmbedder.from_sklearn(
                    channel_index=channel_index,
                    pca=tsvd,
                    alignment_iterations=template_cfg.svd_alignment_iterations,
                    temporal_slice=tslice,
                    trough=trough,
                    spike_length_samples=pad_spike_len,
                    align_pad=align_pad,
                )
        elif template_cfg.use_svd and template_cfg.svd_method != "peeler":
            tsvd = fit_tsvd(
                recording=recording,
                sorting=sorting,
                motion=motion,
                template_cfg=template_cfg,
                waveform_cfg=waveform_cfg,
                computation_cfg=computation_cfg,
            )
            assert tsvd.components_.shape[0] <= rank
            rank = tsvd.components_.shape[0]
            tsvd = FullProbeTemporalPCAEmbedder.from_sklearn(
                channel_index=channel_index,
                pca=tsvd,
                alignment_iterations=template_cfg.svd_alignment_iterations,
                temporal_slice=tslice,
                trough=trough,
                spike_length_samples=pad_spike_len,
                align_pad=align_pad,
            )
        elif template_cfg.use_svd:
            tsvd = FullProbeTemporalPCAEmbedder(
                channel_index=channel_index,
                rank=rank,
                geom=geom,
                fit_radius=template_cfg.denoising_fit_radius,
                max_waveforms=template_cfg.denoising_fit_sampling_cfg.n_waveforms_fit,
                alignment_iterations=template_cfg.svd_alignment_iterations,
                temporal_slice=tslice,
                trough=trough,
                align_pad=align_pad,
            )
        else:
            tsvd = None

        # build a featurization pipeline which handles interpolation and
        # raw/svd waveform statistics
        transformers = []
        if whitener is None:
            assert template_cfg.whitening.strategy == "none"
        else:
            assert template_cfg.whitening.strategy in (
                "prewhiten",
                "prewhiten_postapply",
                "postwhiten",
            )
        if template_cfg.whitening.strategy == "prewhiten":
            assert whitener is not None
            transformers.append(
                WaveformWhitener(
                    geom=geom, channel_index=channel_index, whitener=whitener
                )
            )
        if motion.drifting:
            interp = WaveformInterpolator(
                geom=geom,
                channel_index=channel_index,
                params=template_cfg.template_interp_params,
            )
        else:
            interp = None
        # if raw is included, interp at beginning, else after SVD (cheaper)
        interp_early = motion.drifting and template_cfg.denoising_method == "none"
        interp_late = motion.drifting and not interp_early
        if interp_early:
            assert interp is not None
            transformers.append(interp)
        if template_cfg.denoising_method in ("none", "exp_weighted"):
            raw_reduce = TemplateWaveformReducer(
                geom=geom,
                channel_index=channel_index,
                name_prefix="raw",
                with_raw_std_dev=template_cfg.with_raw_std_dev,
                n_units=sorting.n_units,
                feature_dim=waveform_cfg.spike_length_samples(
                    recording.sampling_frequency
                ),
                output_channels=len(rgeom),
                reduction=template_cfg.reduction,
            )
            transformers.append(raw_reduce)
        if template_cfg.use_svd:
            assert tsvd is not None
            transformers.append(tsvd)
        if interp_late:
            assert interp is not None
            transformers.append(interp)
        if template_cfg.whitening.strategy == "postwhiten":
            assert whitener is not None
            transformers.append(
                WaveformWhitener(
                    geom=geom, channel_index=channel_index, whitener=whitener
                )
            )
        if template_cfg.use_svd:
            svd_reduce = TemplateWaveformReducer(
                geom=geom,
                channel_index=channel_index,
                name_prefix="svd",
                with_raw_std_dev=False,
                n_units=sorting.n_units,
                feature_dim=rank,
                output_channels=len(rgeom),
                reduction=template_cfg.reduction,
            )
            transformers.append(svd_reduce)

        # assemble pipeline
        fp = WaveformPipeline(transformers=transformers)
        fp.attach_motion(motion)
        fp.precompute()
        logger.dartsortverbose("Template pipeline: %s", fp)

        # grab weights and labels for fixed_properties
        assert sorting.labels is not None
        fixed_properties = {"labels": sorting.labels, "channels": sorting.channels}
        if template_cfg.weighted:
            weights = get_top_assignment_weights(sorting)
            fixed_properties["template_weights"] = weights
        if (c := getattr(sorting, "alignment_signs", None)) is not None:
            fixed_properties["alignment_signs"] = c
        else:
            fixed_properties["alignment_signs"] = np.ones(
                sorting.channels.shape, dtype=np.float32
            )

        return cls(
            channel_index=channel_index,
            recording=recording,
            times_samples=sorting.times_samples,
            featurization_pipeline=fp,
            fixed_properties=fixed_properties,
            chunk_length_samples=template_cfg.grab_chunk_length_samples,
            fit_sampling_cfg=template_cfg.denoising_fit_sampling_cfg,
            waveform_cfg=padded_waveform_cfg,
        )

    def temporal_svd(self) -> PCA | None:
        assert self.featurization_pipeline is not None
        tsvd = None
        for ft in self.featurization_pipeline.transformers:
            if isinstance(ft, FullProbeTemporalPCAEmbedder):
                assert tsvd is None  # should find only one of these
                tsvd = ft.to_sklearn()
        return tsvd

    def reduction_results(
        self,
        hdf5_path: Path,
        show_progress: bool = False,
        computation_cfg: ComputationConfig | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        # get raw and svd transformers
        assert self.featurization_pipeline is not None
        transformers = self.featurization_pipeline.transformers
        reducers = [f for f in transformers if isinstance(f, TemplateWaveformReducer)]

        assert 0 < len(reducers) <= 2
        if reducers[0].name_prefix == "raw":
            raw_f = reducers[0]
            if len(reducers) == 2:
                assert reducers[1].name_prefix == "svd"
                svd_f = reducers[1]
            elif len(reducers) == 1:
                svd_f = None
            else:
                assert False
        elif reducers[0].name_prefix == "svd":
            assert len(reducers) == 1
            raw_f = None
            svd_f = reducers[0]
        else:
            assert False

        if raw_f is not None:
            counts, raw_mean, raw_std = raw_f.reduction_results(
                hdf5_path=hdf5_path,
                labels=self.b.labels,
                show_progress=show_progress,
                computation_cfg=computation_cfg,
            )
        else:
            counts = raw_mean = raw_std = None
        if svd_f is not None:
            svd_counts, svd_mean, svd_std = svd_f.reduction_results(
                hdf5_path=hdf5_path,
                labels=self.b.labels,
                show_progress=show_progress,
                computation_cfg=computation_cfg,
            )
        else:
            svd_counts = svd_mean = svd_std = None
        assert svd_std is None
        if counts is None:
            assert svd_counts is not None
            counts = svd_counts

        # reconstruct SVD mean
        if svd_mean is not None:
            (tsvd,) = [
                f for f in transformers if isinstance(f, FullProbeTemporalPCAEmbedder)
            ]
            svd_mean = torch.asarray(svd_mean)
            svd_mean = tsvd.force_reconstruct(svd_mean)
            assert svd_mean.isfinite().all()
            svd_mean = svd_mean.numpy(force=True)

        return counts, raw_mean, raw_std, svd_mean


def _handle_internal_pad(
    recording: BaseRecording, waveform_cfg: WaveformConfig, template_cfg: TemplateConfig
):
    trough = waveform_cfg.trough_offset_samples(recording.sampling_frequency)
    do_align = (
        template_cfg.use_svd
        and template_cfg.svd_alignment_iterations
        and template_cfg.svd_alignment_ms
    )
    if do_align:
        padded_waveform_cfg = waveform_cfg.pad(template_cfg.svd_alignment_ms)
        tslice = waveform_cfg.relative_slice(
            padded_waveform_cfg, sampling_frequency=recording.sampling_frequency
        )
        align_pad = tslice.start
    else:
        padded_waveform_cfg = waveform_cfg
        tslice = None
        align_pad = 0

    return do_align, padded_waveform_cfg, trough, tslice, align_pad
