"""Fast template extractor

Algorithm: Each unit / pitch shift combo has a raw and low-rank template
computed. These are shift-averaged and then weighted combined.

The raw and low-rank templates are computed using Welford.
"""
from dataclasses import dataclass, field
from logging import getLogger
from typing import Literal

import numpy as np
import torch
from torch.distributions import StudentT, Normal
import torch.nn.functional as F


from ..templates.superres_util import superres_sorting
from ..transform.transform_base import BaseWaveformModule
from ..transform import WaveformPipeline, Waveform, TemporalPCAFeaturizer
from ..util import drift_util
from ..util.data_util import load_stored_tsvd, subsample_to_max_count
from ..util.logging_util import DARTsortLogger
from ..util.spiketorch import ptp
from ..util.waveform_util import full_channel_index

from .grab import GrabAndFeaturize


logger: DARTsortLogger = getLogger(__name__)


class RunningTemplates(GrabAndFeaturize):
    """Compute templates online by Welford summation.

    No medians :(.

    TODO: Denoising and weights.
    """

    peel_kind = "Templates"

    def __init__(
        self,
        recording,
        channel_index,
        times_samples,
        channels,
        labels,
        n_pitches_shift=None,
        denoising_method=None,
        tsvd=None,
        tsvd_rank=5,
        tsvd_fit_radius=75.0,
        denoising_snr_threshold=50.0,
        group_ids=None,
        motion_est=None,
        time_shifts=None,
        gamma_df: float | None = None,
        initial_df: float = 1.0,
        t_iters: int = 1,
        with_raw_std_dev=False,
        trough_offset_samples=42,
        spike_length_samples=121,
        chunk_length_samples=30_000,
        n_seconds_fit=100,
        fit_subsampling_random_state: int | np.random.Generator = 0,
    ):
        n_channels = recording.get_num_channels()
        channel_index = torch.tile(torch.arange(n_channels), (n_channels, 1))
        assert labels.min() >= 0

        if denoising_method is None:
            denoising_method = "none"
        self.denoising_method = denoising_method
        self.denoising_snr_threshold = denoising_snr_threshold
        self.use_svd = False  # updated below
        self.denoising_rank = None
        self.gamma_df = gamma_df
        self.initial_df = initial_df

        # these are not used internally, just stored to record what happened in
        # .from_config() so that users can realign their sortings to match
        self.time_shifts = time_shifts

        waveform_feature = Waveform(channel_index=channel_index)
        feats: list[BaseWaveformModule] = [waveform_feature]
        if denoising_method in ("exp_weighted_svd", "t_svd"):
            if tsvd is not None:
                tpca = TemporalPCAFeaturizer.from_sklearn(
                    channel_index, tsvd, getattr(tsvd, "temporal_slice", None)
                )
            else:
                tpca = TemporalPCAFeaturizer(
                    channel_index=channel_index,
                    geom=torch.asarray(recording.get_channel_locations()),
                    rank=tsvd_rank,
                    centered=False,
                    fit_radius=tsvd_fit_radius,
                )
            feats.append(tpca)
            self.denoising_rank = tpca.rank
            use_svd = True
        elif denoising_method in ("none", "t", None):
            tpca = None
        else:
            raise ValueError(f"Unknown {denoising_method=}.")

        featurization_pipeline = WaveformPipeline(feats)

        super().__init__(
            recording=recording,
            featurization_pipeline=featurization_pipeline,
            times_samples=times_samples,
            channels=channels,
            labels=labels,
            channel_index=channel_index,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            chunk_length_samples=chunk_length_samples,
            n_seconds_fit=n_seconds_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
        )

        self.motion_aware = motion_est is not None
        self.motion_est = motion_est
        self.with_raw_std_dev = with_raw_std_dev
        self.n_units = labels.max() + 1
        self.group_ids = group_ids
        self.n_pitches_shift = None
        self.full_featurization_pipeline = featurization_pipeline
        self.short_featurization_pipeline = WaveformPipeline([waveform_feature])
        self.tpca = tpca

        self.tasks = RunningTemplatesTasks(
            denoising_method=denoising_method,
            has_gamma_df=self.gamma_df is not None,
            with_raw_std_dev=with_raw_std_dev,
            t_iters=t_iters,
        )

    @classmethod
    def from_config(
        cls,
        sorting,
        recording,
        waveform_cfg,
        template_cfg,
        tsvd=None,
        motion_est=None,
        show_progress=True,
        computation_cfg=None,
        random_state=0,
    ):
        random_state = np.random.default_rng(random_state)

        # superres sorting if requested
        if template_cfg.superres_templates:
            group_ids, sorting = superres_sorting(
                sorting,
                recording.get_channel_locations(),
                motion_est=motion_est,
                strategy=template_cfg.superres_strategy,
                superres_bin_size_um=template_cfg.superres_bin_size_um,
                min_spikes_per_bin=template_cfg.superres_bin_min_spikes,
            )
        else:
            group_ids = None

        # restrict to max spikes/template
        sorting = subsample_to_max_count(
            sorting, max_spikes=template_cfg.spikes_per_unit, seed=random_state
        )

        # realign to empirical trough if necessary
        fs = recording.sampling_frequency
        realign_samples = waveform_cfg.ms_to_samples(
            template_cfg.realign_shift_ms, sampling_frequency=fs
        )
        n_pitches_shift = time_shifts = None
        if template_cfg.realign_peaks and realign_samples:
            sorting, n_pitches_shift, time_shifts = realign_by_running_templates(
                sorting,
                recording,
                motion_est=motion_est,
                realign_samples=realign_samples,
                show_progress=show_progress,
                computation_cfg=computation_cfg,
            )

        # remove discarded spikes so they don't get loaded
        sorting = sorting.drop_missing()

        # try to load denoiser if denoising is happening
        if tsvd is None and template_cfg.denoising_method in ("t_svd", "exp_weighted_svd"):
            if not template_cfg.recompute_tsvd:
                tsvd = load_stored_tsvd(sorting)

        return cls(
            recording=recording,
            channel_index=full_channel_index(recording.get_num_channels()),
            times_samples=sorting.times_samples,
            channels=sorting.channels,
            labels=sorting.labels,
            n_pitches_shift=n_pitches_shift,
            denoising_method=template_cfg.denoising_method,
            tsvd=tsvd,
            tsvd_rank=template_cfg.denoising_rank,
            tsvd_fit_radius=template_cfg.denoising_fit_radius,
            denoising_snr_threshold=template_cfg.denoising_snr_threshold,
            group_ids=group_ids,
            motion_est=motion_est,
            with_raw_std_dev=template_cfg.with_raw_std_dev,
            trough_offset_samples=waveform_cfg.trough_offset_samples(fs),
            spike_length_samples=waveform_cfg.spike_length_samples(fs),
            fit_subsampling_random_state=random_state,
            time_shifts=time_shifts,
            gamma_df=template_cfg.fixed_t_df,
            initial_df=template_cfg.initial_t_df,
            t_iters=template_cfg.t_iters,
        )

    def compute_template_data(
        self, show_progress=True, computation_cfg=None, task_name=None
    ):
        super().fit_featurization_pipeline(computation_cfg=computation_cfg)
        self.setup()  # initialize all buffers
        while self.tasks.plan_step():
            self.setup_step()
            self.peel(
                output_hdf5_filename=None,
                ignore_resuming=True,
                show_progress=show_progress,
                computation_cfg=computation_cfg,
                task_name=task_name or f"{self.peel_kind}:{self.denoising_method}<{self.tasks.active_step}>",
            )
            self.finalize_step()
            self.tasks.finalize_step()

        return self.to_template_data()

    def finalize_step(self):
        if self.tasks.needs_raw_pass:
            # sets self.raw_stds
            self.finalize_raw()
        elif self.tasks.needs_svd_pass:
            # sets nu (gamma concentration) and svd_stds if is_t
            self.finalize_svd()
        elif self.tasks.needs_t_pass:
            # nothing to do
            self.finalize_t()
        else:
            assert False

    def setup(self):
        # this sets motion-related buffers (pitch shifts, geom)
        self.setup_global()
        # sets up buffers for raw means + meansq
        self.setup_raw()
        # sets up pcmeans, gamma mean buffers
        self.setup_svd()
        # sets up raw_t_mean, svd_t_mean, raw_t_counts, svd_t_counts
        self.setup_t()

    def setup_step(self):
        self.n_spikes = 0
        self.counts.fill_(0.0)
        if hasattr(self, 'raw_t_counts'):
            self.raw_t_counts.fill_(0.0)
        if hasattr(self, 'svd_t_counts'):
            self.svd_t_counts.fill_(0.0)

        if self.tasks.using_tsvd_projection:
            self.featurization_pipeline = self.full_featurization_pipeline
        else:
            self.featurization_pipeline = self.short_featurization_pipeline

    def to_template_data(self):
        from ..templates.templates import TemplateData

        counts_by_channel = self.counts.numpy(force=True)
        if self.group_ids is None:
            unit_ids = np.arange(self.n_units)
        else:
            unit_ids = self.group_ids

        templates = self.templates()
        raw_stds = getattr(self, "raw_stds", None)
        if raw_stds is not None:
            raw_stds = raw_stds.numpy(force=True)

        return TemplateData(
            templates=templates,
            unit_ids=unit_ids,
            spike_counts_by_channel=counts_by_channel,
            spike_counts=counts_by_channel.max(1),
            raw_std_dev=raw_stds,
            registered_geom=self.reg_geom,
            trough_offset_samples=self.trough_offset_samples,
        )

    def templates(self):
        assert self.tasks.done

        if self.denoising_method in (None, "none"):
            return self.raw_means.nan_to_num_().numpy(force=True)

        if self.denoising_method == "exp_weighted_svd":
            from ..templates.get_templates import denoising_weights

            raw_means = self.raw_means.nan_to_num_().numpy(force=True)
            snrs = ptp(self.raw_means.nan_to_num(nan=-torch.inf)).mul_(
                self.counts.to(self.raw_means).sqrt()
            )
            weights = denoising_weights(
                snrs.numpy(force=True),
                spike_length_samples=self.spike_length_samples,
                trough_offset=self.trough_offset_samples,
                snr_threshold=self.denoising_snr_threshold,
            )
            w = weights.astype(raw_means.dtype)
            svd_means = self.svd_means.nan_to_num_().numpy(force=True)
            logger.dartsortdebug(
                f"exp_weighted_svd: Raw weight mean/max={w.mean().item()}/{w.max().item()}"
            )
            return w * raw_means + (1 - w) * svd_means

        if self.denoising_method == "t":
            return self.raw_t_means.nan_to_num_().numpy(force=True)

        if self.denoising_method == "t_svd":
            rm = self.raw_t_means.nan_to_num_()
            sm = self.svd_t_means.nan_to_num_()
            n, t, c = sm.shape
            sm = self.tpca._project_in_probe(sm.mT.reshape(n * c, t))
            sm = sm.reshape(n, c, t).mT
            denom = self.raw_t_counts + self.svd_t_counts
            denom = torch.where(denom > 0, denom, 1.0)
            w = self.raw_t_counts / denom
            logger.dartsortdebug(
                f"t_svd: Raw weight mean/max={w.mean().item()}/{w.max().item()}"
            )
            return rm.mul_(w).add_(sm.mul_(1.0 - w)).numpy(force=True)

        assert False

    def process_chunk(
        self,
        chunk_start_samples,
        chunk_end_samples=None,
        return_residual=False,
        skip_features=False,
        n_resid_snips=None,
        to_numpy=False,
    ):
        res = super().process_chunk(
            chunk_start_samples=chunk_start_samples,
            n_resid_snips=n_resid_snips,
            chunk_end_samples=chunk_end_samples,
            return_residual=return_residual,
            skip_features=skip_features,
            to_numpy=to_numpy,
        )
        if to_numpy or not res["n_spikes"]:
            return res

        ires = self._process_chunk_main(
            res["waveforms"], res["indices"], res["labels"], res.get("tpca_features")
        )
        res.update(ires)

        return res

    def gather_chunk_result(
        self,
        cur_n_spikes,
        chunk_start_samples,
        chunk_result,
        h5_spike_datasets,
        output_h5,
        residual_file,
        residual_to_h5,
        ignore_resuming,
        skip_features,
    ):
        if output_h5 is not None:
            # this path is active when fitting models (i.e., temporal svd).
            # otherwise, peeling is run with no output file, and everything
            # is done in the arrays created during setup().
            # in this branch, to_numpy was True in process_chunk().
            return super().gather_chunk_result(
                cur_n_spikes,
                chunk_start_samples,
                chunk_result,
                h5_spike_datasets,
                output_h5,
                residual_file,
                residual_to_h5,
                ignore_resuming,
                skip_features,
            )

        assert residual_file is None
        assert not residual_to_h5
        assert ignore_resuming

        n_spikes = chunk_result["n_spikes"]
        if not n_spikes:
            return n_spikes

        self.n_spikes += n_spikes
        self._gather_chunk_main(chunk_result)

        return n_spikes

    def _process_chunk_main(
        self,
        waveforms: torch.Tensor,
        spike_ix: torch.LongTensor,
        labels: torch.LongTensor,
        pcfeats: torch.Tensor | None,
    ):
        """Handles batch statistics for raw or exp weighted templates

        Also is the initialization stage for the t flavored versions.
        """
        # scatter waveforms to reg channels, if necessary
        if self.motion_aware:
            chanix = self.target_channels[self.pitch_shift_ixs[spike_ix]]
            waveforms = registerize(waveforms, self.n_channels_full, chanix)
        else:
            chanix = None

        if self.tasks.active_step in ("raw", "svd"):
            return self._process_chunk_raw_svd(
                waveforms, spike_ix, labels, chanix, pcfeats
            )
        elif self.tasks.active_step.startswith("t"):
            return self._process_chunk_t(waveforms, spike_ix, labels)

        assert False

    def _process_chunk_raw_svd(
        self,
        waveforms: torch.Tensor,
        spike_ix: torch.LongTensor,
        labels: torch.LongTensor,
        chanix: torch.LongTensor | None,
        pcfeats: torch.Tensor | None,
    ):
        res = {}

        # determine each unit's per-channel counts, for weighting purposes
        # TODO: sum responsibility weights here
        counts = waveforms.new_zeros((self.n_units, self.n_channels_full))
        weights = waveforms[:, 0].isfinite().to(waveforms)
        ix = labels[:, None].broadcast_to(weights.shape)
        counts.scatter_add_(dim=0, index=ix, src=weights)
        res["counts"] = counts

        # normalized weights in each unit
        denom = torch.where(counts > 0, counts, 1.0)
        weights.div_(denom[labels])

        if self.tasks.updating_gamma_mean:
            # self.raw_means already done with. easiest to get w hats
            # while waveforms still have nans that can be deleted.
            raw_what = get_gamma_latent(
                x=waveforms,
                labels=labels,
                mu=self.raw_means,
                nu=self.initial_df,
                sigma=1.0,
            )
            assert raw_what.shape == waveforms.shape
            res["wbar"] = torch.nanmean(raw_what.mean(dim=1))
            res["sqwbar"] = torch.nanmean(raw_what.square().mean(dim=1))
            res["logwbar"] = torch.nanmean(raw_what.log().mean(dim=1))

        waveforms.nan_to_num_()

        # weighted sum
        x = None
        if self.tasks.updating_raw_means:
            raw_means = waveforms.new_zeros(self.raw_means.shape)
            x = waveforms.mul(weights[:, None])
            ix = labels[:, None, None].broadcast_to(x.shape)
            raw_means.scatter_add_(dim=0, index=ix, src=x)
            res["raw_means"] = raw_means

        # "" std, optionally ""
        if self.tasks.needs_raw_stds:
            meansq = waveforms.new_zeros(self.raw_means.shape)
            if x is None:
                x = waveforms.mul(weights[:, None])
            x.mul_(waveforms)
            meansq.scatter_add_(dim=0, index=ix, src=x)
            res["meansq"] = meansq

        if self.tasks.updating_svd_means:
            assert self.pcmeans is not None
            assert pcfeats is not None
            pcmeans = waveforms.new_zeros(self.pcmeans.shape)
            pcfeats = registerize(pcfeats, self.n_channels_full, chanix, pad_value=0.0)
            x = pcfeats.mul(weights[:, None])
            ix = labels[:, None, None].broadcast_to(x.shape)
            pcmeans.scatter_add_(dim=0, index=ix, src=x)
            res["pcmeans"] = pcmeans

        return res

    def _process_chunk_t(
        self,
        waveforms: torch.Tensor,
        spike_ix: torch.LongTensor,
        labels: torch.LongTensor,
    ):
        pshape = self.raw_means.shape
        res = {}

        raw_what = get_gamma_latent(
            x=waveforms,
            labels=labels,
            mu=self.raw_means,
            nu=self.nu,
            sigma=self.raw_stds,
        )

        if self.tasks.updating_gamma_mean:
            res["wbar"] = torch.nanmean(raw_what.mean(dim=1))
            res["sqwbar"] = torch.nanmean(raw_what.square().mean(dim=1))
            res["logwbar"] = torch.nanmean(raw_what.log().mean(dim=1))

        if self.tasks.is_svd:
            svd_what = get_gamma_latent(
                x=waveforms,
                labels=labels,
                mu=self.svd_means,
                nu=self.svd_nu,
                sigma=self.svd_stds,
            )

            if self.tasks.updating_gamma_mean:
                res["svd_wbar"] = torch.nanmean(svd_what.mean(dim=1))
                res["svd_sqwbar"] = torch.nanmean(svd_what.square().mean(dim=1))
                res["svd_logwbar"] = torch.nanmean(svd_what.log().mean(dim=1))

            resps = get_t_responsibilities(
                x=waveforms,
                nu0=self.nu,
                nu1=self.svd_nu,
                labels=labels,
                lp0=self.raw_t_log_prop,
                mu0=self.raw_means,
                sigma0=self.raw_stds,
                lp1=self.svd_t_log_prop,
                mu1=self.svd_means,
                sigma1=self.svd_stds,
            )
            raw_weight = resps[0] * raw_what
            svd_weight = resps[1] * raw_what
            svd_t_counts, svd_t_means, svd_t_meansq = count_and_mean(
                waveforms, labels, pshape, svd_weight, with_sq=self.tasks.updating_t_stds
            )
            assert svd_t_counts.isfinite().all()
            assert svd_t_means.isfinite().all()
            res["svd_t_counts"] = svd_t_counts
            res["svd_t_means"] = svd_t_means
            if svd_t_meansq is not None:
                res["svd_t_meansq"] = svd_t_meansq
        else:
            raw_weight = raw_what
        raw_t_counts, raw_t_means, raw_t_meansq = count_and_mean(
            waveforms, labels, pshape, raw_weight, with_sq=self.tasks.updating_t_stds
        )
        res["raw_t_counts"] = raw_t_counts
        res["raw_t_means"] = raw_t_means
        assert raw_t_counts.isfinite().all()
        assert raw_t_means.isfinite().all()
        if raw_t_meansq is not None:
            assert raw_t_meansq.isfinite().all()
            res["raw_t_meansq"] = raw_t_meansq
        return res

    def _gather_chunk_main(self, chunk_result):
        # update running count and do Welford sums

        # this is the raw/svd step update
        if "counts" in chunk_result:
            self.counts += chunk_result["counts"]
            denom = torch.where(self.counts > 0, self.counts, 1.0)
            w = chunk_result["counts"].div_(denom).unsqueeze(1)
        if "raw_means" in chunk_result:
            self.raw_means += chunk_result["raw_means"].sub_(self.raw_means).mul_(w)
        if "meansq" in chunk_result:
            self.meansq += chunk_result["meansq"].sub_(self.meansq).mul_(w)
        if "pcmeans" in chunk_result:
            self.pcmeans += chunk_result["pcmeans"].sub_(self.pcmeans).mul_(w)
        if "wbar" in chunk_result:
            ww = chunk_result["n_spikes"] / max(self.n_spikes, 1)
            self.wbar += ww * (chunk_result["wbar"] - self.wbar)
            self.logwbar += ww * (chunk_result["logwbar"] - self.logwbar)
            self.sqwbar += ww * (chunk_result["sqwbar"] - self.sqwbar)
        if "svd_wbar" in chunk_result:
            ww = chunk_result["n_spikes"] / max(self.n_spikes, 1)
            self.svd_wbar += ww * (chunk_result["svd_wbar"] - self.svd_wbar)
            self.svd_logwbar += ww * (chunk_result["svd_logwbar"] - self.svd_logwbar)
            self.svd_sqwbar += ww * (chunk_result["svd_sqwbar"] - self.svd_sqwbar)

        # and, the t step update
        if "raw_t_counts" in chunk_result:
            assert "raw_t_means" in chunk_result
            self.raw_t_counts += chunk_result["raw_t_counts"]
            d = torch.where(self.raw_t_counts > 0, self.raw_t_counts, 1.0)
            w = chunk_result["raw_t_counts"].div_(d)
            self.raw_t_means += (
                chunk_result["raw_t_means"].sub_(self.raw_t_means).mul_(w)
            )
        if "raw_t_counts" in chunk_result and "raw_t_meansq" in chunk_result:
            self.raw_t_stds += (
                chunk_result["raw_t_meansq"].sub_(self.raw_t_stds).mul_(w)
            )
        if "svd_t_counts" in chunk_result:
            assert "svd_t_means" in chunk_result
            self.svd_t_counts += chunk_result["svd_t_counts"]
            d = torch.where(self.svd_t_counts > 0, self.svd_t_counts, 1.0)
            w = chunk_result["svd_t_counts"].div_(d)
            self.svd_t_means += (
                chunk_result["svd_t_means"].sub_(self.svd_t_means).mul_(w)
            )
        if "svd_t_counts" in chunk_result and "svd_t_meansq" in chunk_result:
            self.svd_t_stds += (
                chunk_result["svd_t_meansq"].sub_(self.svd_t_stds).mul_(w)
            )

    def setup_global(self):
        geom = self.recording.get_channel_locations()
        if self.motion_aware:
            self.reg_geom = drift_util.registered_geometry(geom, self.motion_est)
            self.n_channels_full = len(self.reg_geom)

            n_pitches_shift = self.n_pitches_shift
            if n_pitches_shift is None:
                depths = np.atleast_1d(geom[self.channels, 1])
                times = self.recording.sample_index_to_time(self.times_samples)
                n_pitches_shift = drift_util.get_spike_pitch_shifts(
                    depths, geom, times_s=times, motion_est=self.motion_est
                )
            unique_shifts, pitch_shift_ixs = np.unique(
                n_pitches_shift, return_inverse=True
            )
            res = drift_util.get_stable_channels(
                geom=geom,
                channels=np.zeros_like(unique_shifts),
                channel_index=self.channel_index.numpy(force=True),
                registered_geom=self.reg_geom,
                n_pitches_shift=unique_shifts,
            )
            target_channels = torch.asarray(res[0], device=self.channel_index.device)
            pitch_shift_ixs = torch.asarray(
                pitch_shift_ixs, device=self.channel_index.device
            )
            self.register_buffer("pitch_shift_ixs", pitch_shift_ixs)
            self.register_buffer("target_channels", target_channels)
        else:
            self.reg_geom = geom
            self.n_channels_full = self.recording.get_num_channels()

    def setup_raw(self):
        n = self.n_units
        t = self.spike_length_samples
        c = self.n_channels_full
        self.mean_shape = n, t, c
        self.register_buffer("raw_means", torch.zeros(self.mean_shape))
        self.register_buffer("counts", torch.zeros((n, c)))
        if self.tasks.needs_raw_stds:
            self.register_buffer("meansq", torch.zeros(self.mean_shape))
        else:
            self.meansq = None

    def setup_svd(self):
        n = self.n_units
        c = self.n_channels_full
        if self.tasks.needs_svd_means:
            assert self.denoising_rank is not None
            self.register_buffer("pcmeans", torch.zeros((n, self.denoising_rank, c)))
        else:
            self.pcmeans = None
        if self.tasks.needs_gamma_mean:
            self.register_buffer("wbar", torch.zeros(()))
            self.register_buffer("logwbar", torch.zeros(()))
            self.register_buffer("sqwbar", torch.zeros(()))
        if self.tasks.needs_gamma_mean and self.tasks.is_svd and self.tasks.remaining_t_iters > 1:
            self.register_buffer("svd_wbar", torch.zeros(()))
            self.register_buffer("svd_logwbar", torch.zeros(()))
            self.register_buffer("svd_sqwbar", torch.zeros(()))

    def setup_t(self):
        if not self.tasks.is_t:
            return
        self.register_buffer("raw_t_counts", torch.zeros(self.mean_shape))
        self.register_buffer("raw_t_log_prop", torch.zeros(self.mean_shape))
        self.register_buffer("raw_t_means", torch.zeros(self.mean_shape))
        if self.tasks.needs_t_stds:
            self.register_buffer("raw_t_stds", torch.zeros(self.mean_shape))
        if self.tasks.is_svd:
            self.register_buffer("svd_t_counts", torch.zeros(self.mean_shape))
            self.register_buffer("svd_t_log_prop", torch.zeros(self.mean_shape))
            self.register_buffer("svd_t_means", torch.zeros(self.mean_shape))
        if self.tasks.is_svd and self.tasks.needs_t_stds:
            self.register_buffer("svd_t_stds", torch.zeros(self.mean_shape))
        if self.gamma_df is not None:
            self.register_buffer("nu", torch.asarray(self.gamma_df))
            self.register_buffer("svd_nu", torch.asarray(self.gamma_df))

        self.register_buffer("unit_spike_counts", torch.zeros(self.mean_shape[0]))
        # todo need to count by channel! this is temporary.
        assert (self.labels >= 0).all()
        _1 = torch.tensor(1.0).to(self.unit_spike_counts).broadcast_to(self.labels.shape)
        self.unit_spike_counts.scatter_add_(dim=0, index=self.labels, src=_1)

    def finalize_raw(self):
        if self.tasks.needs_raw_stds:
            self.register_buffer("raw_stds", self.raw_means.square())
            torch.subtract(self.meansq, self.raw_stds, out=self.raw_stds)
            self.raw_stds.abs_().nan_to_num_().sqrt_()
            assert self.raw_stds.isfinite().all()

    def finalize_svd(self):
        self.finalize_raw()
        if self.tasks.needs_svd_means:
            assert self.tpca is not None
            assert self.pcmeans is not None
            svd_means = self.pcmeans.nan_to_num()
            svd_means = self.tpca.force_reconstruct(svd_means)
            self.register_buffer("svd_means", svd_means.to(self.raw_means))
            assert self.svd_means.isfinite().all()

        if self.tasks.needs_svd_stds:
            assert getattr(self, "raw_stds", None) is not None
            raw_var = self.raw_stds.square()
            dmeansq = (self.svd_means - self.raw_means).square_()
            self.register_buffer("svd_stds", (raw_var.add_(dmeansq)).sqrt_())
            assert self.svd_stds.isfinite().all()

        if self.tasks.needs_gamma_mean:
            assert self.gamma_df is None
            self.register_buffer("nu", estimate_gamma_df(self.wbar, self.logwbar, self.sqwbar))
            self.register_buffer("svd_nu", self.nu.clone().detach())
            self.wbar.fill_(0.0)
            self.logwbar.fill_(0.0)
            self.sqwbar.fill_(0.0)

    def finalize_t(self):
        if not self.tasks.is_t:
            return

        if self.tasks.is_svd:
            assert self.tpca is not None
            assert self.svd_t_means is not None
            svd_means = self.svd_t_means.nan_to_num()
            svd_means = self.tpca.force_project(svd_means)
            self.svd_t_means.copy_(svd_means)
            assert self.svd_t_means.isfinite().all()

        if self.tasks.remaining_t_iters > 1:
            #todo overwrite raw_means and svd_means with t versions and zero t buffers
            self.raw_means.copy_(self.raw_t_means)
            assert self.raw_means.isfinite().all()
            self.raw_t_means.fill_(0.0)

        if self.tasks.is_svd and self.tasks.remaining_t_iters > 1:
            self.svd_means.copy_(self.svd_t_means)
            assert self.svd_means.isfinite().all()
            self.svd_t_means.fill_(0.0)

        if self.tasks.is_svd and self.tasks.remaining_t_iters > 1:
            # props
            self.raw_t_log_prop = self.raw_t_counts.log() - (self.raw_t_counts + self.svd_t_counts).log()
            self.svd_t_log_prop = 1.0 - self.raw_t_log_prop

        if self.tasks.updating_t_stds:
            #todo overwrite raw_stds and svd_stds with t versions and zero t buffers
            assert self.raw_t_stds.isfinite().all()
            assert self.raw_t_means.isfinite().all()
            self.raw_t_stds.sub_(self.raw_t_means.square())
            # rescale by count factor...
            rescale = self.raw_t_counts / self.unit_spike_counts[:, None, None]
            self.raw_t_stds.mul_(rescale)
            self.raw_t_stds.sqrt_()
            assert self.raw_t_stds.isfinite().all()
            self.raw_stds.copy_(self.raw_t_stds)
            assert self.raw_stds.isfinite().all()
            self.raw_t_stds.fill_(0.0)

        if self.tasks.is_svd and self.tasks.updating_t_stds:
            self.svd_t_stds.sub_(self.svd_t_means.square())
            rescale = self.svd_t_counts / self.unit_spike_counts[:, None, None]
            self.svd_t_stds.mul_(rescale)
            self.svd_t_stds.sqrt_()
            self.svd_stds.copy_(self.svd_t_stds)
            assert self.svd_stds.isfinite().all()
            self.svd_t_stds.fill_(0.0)

        if self.tasks.updating_gamma_mean:
            self.nu.fill_(estimate_gamma_df(self.wbar, self.logwbar, self.sqwbar))
            self.wbar.fill_(0.0)
            self.logwbar.fill_(0.0)
            self.sqwbar.fill_(0.0)

        if self.tasks.is_svd and self.tasks.updating_gamma_mean:
            self.svd_nu.fill_(estimate_gamma_df(self.svd_wbar, self.svd_logwbar, self.svd_sqwbar))
            self.svd_wbar.fill_(0.0)
            self.svd_logwbar.fill_(0.0)
            self.svd_sqwbar.fill_(0.0)


@dataclass(kw_only=True)
class RunningTemplatesTasks:
    """Helper state machine for the RunningTemplates peeler.

    Cooperates with peeler's .compute_template_data() and .finalize_step().
    Sets updating_* flags read by peeler's chunk process and gather methods
    so it knows what it's doing.
    """

    denoising_method: Literal["none", "exp_weighted_svd", "t", "t_svd"]
    has_gamma_df: bool
    with_raw_std_dev: bool
    t_iters: int

    # tasks that are requested but have not been done
    # these will be updated as we go through the steps and are determined
    # based on denoising_method
    needs_raw_means: bool = field(init=False)
    needs_raw_stds: bool = field(init=False)
    needs_svd_means: bool = field(init=False)
    needs_svd_stds: bool = field(init=False)
    needs_gamma_mean: bool = field(init=False)
    needs_t_params: bool = field(init=False)
    needs_t_stds: bool = field(init=False)
    remaining_t_iters: int = field(init=False)

    # these hold the tasks that are currently being processed
    active_step: Literal["", "raw", "svd", "t"] = ""
    updating_raw_means: bool = False
    updating_svd_means: bool = False
    updating_gamma_mean: bool = False
    updating_t_params: bool = False
    updating_t_stds: bool = False
    using_tsvd_projection: bool = False

    def __post_init__(self):
        self.is_t = self.denoising_method in ("t", "t_svd")
        self.is_none = self.denoising_method in ("none", None)
        self.is_weighted = self.denoising_method == "exp_weighted_svd"
        self.is_svd = self.denoising_method in ("t_svd", "exp_weighted_svd")

        self.needs_raw_means = True
        self.needs_raw_stds = self.with_raw_std_dev or self.is_t
        self.needs_svd_means = self.is_weighted or self.is_svd
        self.needs_svd_stds = self.is_svd and self.is_t
        self.needs_gamma_mean = self.is_t and not self.has_gamma_df
        self.needs_t_params = self.is_t
        self.remaining_t_iters = int(self.is_t) * self.t_iters
        self.needs_t_stds = self.is_t and self.remaining_t_iters > 1

    def plan_step(self):
        if self.done:
            self.active_step = ""
            logger.dartsortdebug("Template tasks done.")
            return False
        elif self.needs_raw_pass:
            self.updating_raw_means = self.needs_raw_means
            self.updating_raw_stds = self.needs_raw_stds
            self.active_step = "raw"
        elif self.needs_svd_pass:
            self.updating_raw_means = self.needs_raw_means
            self.updating_raw_stds = self.needs_raw_stds
            self.updating_svd_means = self.needs_svd_means
            self.using_tsvd_projection = self.updating_svd_means
            self.updating_gamma_mean = self.needs_gamma_mean
            self.active_step = "svd"
        elif self.needs_t_pass:
            self.updating_t_params = self.needs_t_params
            self.updating_t_stds = self.remaining_t_iters > 1
            self.updating_gamma_mean = self.remaining_t_iters > 1
            self.active_step = f"t{1 + self.t_iters - self.remaining_t_iters}/{self.t_iters}"
        else:
            assert False
        logger.dartsortdebug(f"Template task: {self.active_step}.")
        return True

    def finalize_step(self):
        if self.done:
            assert False
        elif self.needs_raw_pass:
            self.updating_raw_means = self.needs_raw_means = False
            self.updating_raw_stds = self.needs_raw_stds = False
        elif self.needs_svd_pass:
            self.updating_raw_means = self.needs_raw_means = False
            self.updating_svd_means = self.needs_svd_means = False
            self.updating_gamma_mean = self.needs_gamma_mean = False
            self.using_tsvd_projection = False
            self.needs_svd_stds = False
        elif self.needs_t_pass:
            self.updating_t_params = False
            self.updating_t_stds = False
            self.updating_gamma_mean = False
            self.remaining_t_iters -= 1
            self.needs_t_params = self.remaining_t_iters > 0
        else:
            assert False

    @property
    def done(self):
        return not (self.needs_raw_pass or self.needs_svd_pass or self.needs_t_pass)

    @property
    def needs_raw_pass(self):
        """Does the peeler need to do a pre-initialization pass?

        This is only used to set up the t weights, or if there's no
        denoising of any kind.
        """
        raw_done = not (self.needs_raw_means or self.needs_raw_stds)
        none_and_not_done = self.is_none and not raw_done
        t_and_not_done = self.is_t and not raw_done
        return none_and_not_done or t_and_not_done

    @property
    def needs_svd_pass(self):
        """This pass computes pc means and the gamma latent mean (if nec)."""
        svd_done = not (self.needs_svd_means or self.needs_gamma_mean)
        would_need_svd = self.is_t or self.is_svd
        return would_need_svd and not svd_done

    @property
    def needs_t_pass(self):
        t_done = not self.needs_t_params
        return self.is_t and not t_done


def realign_by_running_templates(
    sorting,
    recording,
    motion_est=None,
    realign_samples=0,
    realign_to="trough_factor",
    trough_factor=3.0,
    show_progress=True,
    computation_cfg=None,
):
    """Realign spike times by label according to the empirical template's trough/peak"""
    from ..templates.get_templates import realign_sorting

    if not realign_samples:
        return sorting, None, None

    # compute "templates". actually need only a narrow window here, and
    # will not do any denoising in this case. configs to match.
    dropped_sorting = sorting.drop_missing()
    peeler = RunningTemplates(
        recording=recording,
        channel_index=full_channel_index(recording.get_num_channels()),
        times_samples=dropped_sorting.times_samples,
        channels=dropped_sorting.channels,
        labels=dropped_sorting.labels,
        trough_offset_samples=realign_samples,
        spike_length_samples=2 * realign_samples + 1,
        motion_est=motion_est,
    )
    template_data = peeler.compute_template_data(
        show_progress=show_progress,
        computation_cfg=computation_cfg,
        task_name="Realign",
    )

    sorting, templates, time_shifts = realign_sorting(
        sorting,
        templates=template_data.templates,
        snrs_by_channel=template_data.snrs_by_channel(),
        unit_ids=template_data.unit_ids,
        max_shift=realign_samples,
        trough_factor=trough_factor,
        realign_to=realign_to,
        trough_offset_samples=0,
        recording_length_samples=recording.get_total_samples(),
    )
    return sorting, peeler.n_pitches_shift, time_shifts


def registerize(
    x: torch.Tensor,
    nc: int,
    chanix: torch.LongTensor | None = None,
    pad_value=torch.nan,
) -> torch.Tensor:
    if chanix is None:
        return x
    reg_x = x.new_full((*x.shape[:2], nc), pad_value)
    ix = chanix[:, None, :].broadcast_to(x.shape)
    reg_x.scatter_(src=x, dim=2, index=ix)
    return reg_x


def get_gamma_latent(x, labels, mu, nu=1.0, sigma=1.0):
    nu = torch.asarray(nu)
    if torch.isinf(nu):
        return x.isfinite().to(x)

    if isinstance(sigma, float):
        sigma = torch.asarray(sigma).to(x)
    else:
        sigma = sigma[labels]
    z = mu[labels].sub_(x).div_(sigma)
    denom = z.square_().add_(nu)
    # w = denom.reciprocal_().mul_(nu + 1.0)
    w = torch.divide(nu + 1.0, denom, out=denom)
    assert (w >= 0).all()
    return w


def estimate_gamma_df(wbar, logwbar, sqwbar, cutoff=100, min_df=0.01):
    """Solves digamma(nu/2) - log(nu) = 1 + logwbar - wbar."""
    from scipy.special import psi, polygamma
    from scipy.optimize import root_scalar

    dtype = wbar.dtype
    dev = wbar.device

    wbar = wbar.numpy(force=True).astype(np.float64)
    logwbar = logwbar.numpy(force=True).astype(np.float64)
    sqwbar = sqwbar.numpy(force=True).astype(np.float64)

    # start with a guess based on the gamma MoM
    vw = sqwbar - wbar**2
    if not np.isfinite(vw) and vw >= 0:
        raise ValueError(
            f"Strange gamma statistics {wbar=} {logwbar=} {sqwbar=}."
        )
    if vw == 0:
        logger.info("Estimated infinite gamma df.")
        return torch.asarray(torch.inf, device=dev, dtype=dtype)
    x0 = -0.16 / (1 + logwbar - wbar)  # tay
    x1 = 1.0 / vw  # mom

    rhs = 1 + logwbar - wbar
    min_rhs = psi(min_df / 2) - np.log(min_df / 2)
    if min_rhs > rhs:
        logger.dartsortdebug(f"df from {rhs=} too small, setting to {min_df}.")
        return torch.asarray(min_df, device=dev, dtype=dtype)

    def f(x):
        p = psi(x)
        # p1 = polygamma(1, x)
        # p2 = polygamma(2, x)
        # invx = np.reciprocal(x)
        f = p - np.log(x) - rhs
        # df = p1 - invx
        # ddf = p2 + invx * invx
        return f #, df, ddf

    init = x0 if x0 > 0 else x1
    assert init > 0
    left = x0 
    while f(left) > 0:
        left /= 2
    assert left > 0
    right = x0
    while f(right) < 0:
        right *= 2
    assert right > left > 0
    if x1 > right or x1 < left:
        x1 = (right + left) / 2
    assert left <= x0 <= right
    assert left <= x1 <= right

    sol = root_scalar(f, bracket=(left, right), x0=x0, x1=x1)# fprime2=True)
    nu = sol.root / 2.0
    assert sol.converged

    logger.dartsortdebug(
        "Gamma nu estimate: final=%s x0=%s for wbar=%s logwbar=%s sqwbar=%s",
        nu,
        x0,
        wbar,
        logwbar,
        sqwbar,
    )
    if nu > cutoff:
        nu = torch.inf
    return torch.asarray(nu, device=dev, dtype=dtype)


def get_t_responsibilities(x, nu0, nu1, labels, lp0, mu0, sigma0, lp1, mu1, sigma1):
    finite = x.isfinite().nonzero(as_tuple=True)
    xf = x[finite]
    lp0 = lp0[labels][finite]
    mu0 = mu0[labels][finite]
    sigma0 = sigma0[labels][finite].clamp_(min=1e-5)
    lp1 = lp1[labels][finite]
    mu1 = mu1[labels][finite]
    sigma1 = sigma1[labels][finite].clamp_(min=1e-5)
    if nu0 < torch.inf:
        l0 = StudentT(nu0, loc=mu0, scale=sigma0, validate_args=False).log_prob(xf)
    else:
        rt2 = torch.sqrt(torch.tensor(2.0)).to(xf)
        l2pi = torch.log(torch.tensor(2.0 * torch.pi)).to(xf) * 0.5
        l0 = sigma0.log().add_(l2pi).add_(
            torch.subtract(mu0, xf, out=mu0)
            .div_(sigma0.mul(rt2))
            .square_()
        ).neg_()
    if nu1 < torch.inf:
        l1 = StudentT(nu1, loc=mu1, scale=sigma1, validate_args=False).log_prob(xf)
    else:
        rt2 = torch.sqrt(torch.tensor(2.0)).to(xf)
        l2pi = torch.log(torch.tensor(2.0 * torch.pi)).to(xf) * 0.5
        l1 = sigma1.log().add_(l2pi).add_(
            torch.subtract(mu1, xf, out=mu1)
            .div_(sigma1.mul(rt2))
            .square_()
        ).neg_()
    ll = x.new_full((2, *x.shape), -torch.inf)
    ll[0, *finite] = l0.add_(lp0)
    ll[1, *finite] = l1.add_(lp1)
    resp = F.softmax(ll, dim=0)
    return resp


def count_and_mean(waveforms, labels, pshape, weights, with_sq=False):
    assert weights.shape == waveforms.shape

    counts = waveforms.new_zeros(pshape)
    ix = labels[:, None, None].broadcast_to(weights.shape)
    counts.scatter_add_(dim=0, index=ix, src=weights)

    denom = torch.where(counts > 0, counts, 1.0)[labels]
    weights = torch.divide(weights, denom, out=denom)

    means = waveforms.new_zeros(pshape)
    x = weights.mul_(waveforms)
    del weights
    ix = labels[:, None, None].broadcast_to(waveforms.shape)
    means.scatter_add_(dim=0, index=ix, src=x)

    if with_sq:
        meansq = waveforms.new_zeros(pshape)
        x.mul_(waveforms)
        ix = labels[:, None, None].broadcast_to(x.shape)
        meansq.scatter_add_(dim=0, index=ix, src=x)
    else:
        meansq = None

    return counts, means, meansq
