from dataclasses import dataclass, field
from logging import getLogger
from typing import Literal

import numpy as np
from spikeinterface.core import BaseRecording
from sympy.utilities.misc import rawlines
import torch
from torch.distributions import StudentT, Normal
import torch.nn.functional as F

from dartsort.util.internal_config import TemplateConfig, WaveformConfig


from ..templates.superres_util import superres_sorting
from ..transform.transform_base import BaseWaveformModule
from ..transform import WaveformPipeline, Waveform, TemporalPCAFeaturizer
from ..util import drift_util
from ..util.data_util import DARTsortSorting, load_stored_tsvd, subsample_to_max_count
from ..util.logging_util import DARTsortLogger
from ..util.spiketorch import ptp
from ..util.waveform_util import full_channel_index

from .grab import GrabAndFeaturize


logger: DARTsortLogger = getLogger(__name__)  # pyright:ignore


class RunningTemplates(GrabAndFeaturize):
    """Compute templates online by Welford summation.

    No medians :(. But has some other denoising methods.
    """

    peel_kind = "Templates"

    def __init__(
        self,
        recording,
        channel_index,
        times_samples,
        channels,
        labels,
        motion_est=None,
        n_pitches_shift=None,
        use_zero=False,
        use_raw=True,
        use_svd=False,
        with_raw_std_dev=False,
        denoising_method: Literal["none", "exp_weighted", "loot", "t"] = "none",
        tsvd=None,
        tsvd_rank=5,
        tsvd_fit_radius=75.0,
        exp_weight_snr_threshold=50.0,
        gamma_df: float | None = None,
        initial_df: float = 3.0,
        t_iters: int = 1,
        group_ids=None,
        properties=None,
        trough_offset_samples=42,
        spike_length_samples=121,
        chunk_length_samples=30_000,
        n_seconds_fit=600,
        fit_subsampling_random_state: int | np.random.Generator = 0,
    ):
        n_channels = recording.get_num_channels()
        channel_index = torch.tile(torch.arange(n_channels), (n_channels, 1))
        assert labels.min() >= 0
        assert denoising_method in ("none", "exp_weighted", "loot", "t")

        self.use_zero = use_zero
        self.use_raw = use_raw
        self.use_svd = use_svd
        self.n_subspaces = use_zero + use_raw + use_svd
        if self.n_subspaces == 1 and denoising_method == "exp_weighted":
            # this would do nothing.
            denoising_method = "none"
        if denoising_method == "exp_weighted":
            assert self.n_subspaces == 2
        self.denoising_method = denoising_method

        self.exp_weight_snr_threshold = exp_weight_snr_threshold
        self.gamma_df = gamma_df
        self.initial_df = initial_df

        waveform_feature = Waveform(channel_index=channel_index)
        feats: list[BaseWaveformModule] = [waveform_feature]
        if self.use_svd:
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
        else:
            self.denoising_rank = None
            tpca = None

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
        self.n_pitches_shift = n_pitches_shift
        self.short_pipeline = WaveformPipeline([waveform_feature])
        self.tpca = tpca

        self.subspace_projectors = []
        self.raw_subspace_ix = int(use_zero)
        self.svd_subspace_ix = int(use_zero) + 1
        if use_zero:
            self.subspace_projectors.append(torch.zeros_like)
        if use_raw:
            self.subspace_projectors.append(identity)
        if use_svd:
            assert self.tpca is not None
            self.subspace_projectors.append(self.tpca.force_project)

        # these are not used internally, just stored to record what happened in
        # .from_config() and passed through to the TemplateData's properties later
        self.properties = properties or {}

        self.tasks = RunningTemplatesTasks(
            denoising_method=denoising_method,
            fixed_df=self.gamma_df is not None,
            with_raw_std_dev=with_raw_std_dev,
            t_iters=t_iters,
            use_zero=use_zero,
            use_raw=use_raw,
            use_svd=use_svd,
        )

    @classmethod
    def from_config(  # pyright: ignore
        cls,
        *,
        sorting: DARTsortSorting,
        recording: BaseRecording,
        waveform_cfg: WaveformConfig,
        template_cfg: TemplateConfig,
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
                sorting=sorting,
                geom=recording.get_channel_locations(),
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
        if tsvd is None and template_cfg.denoising_method in (
            "t_svd",
            "exp_weighted_svd",
        ):
            if not template_cfg.recompute_tsvd:
                tsvd = load_stored_tsvd(sorting)

        properties = {}
        if time_shifts is not None:
            properties["time_shifts"] = time_shifts

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
            exp_weight_snr_threshold=template_cfg.exp_weight_snr_threshold,
            group_ids=group_ids,
            motion_est=motion_est,
            use_zero=template_cfg.use_zero,
            use_svd=template_cfg.use_svd,
            with_raw_std_dev=template_cfg.with_raw_std_dev,
            trough_offset_samples=waveform_cfg.trough_offset_samples(fs),
            spike_length_samples=waveform_cfg.spike_length_samples(fs),
            fit_subsampling_random_state=random_state,
            properties=properties,
            gamma_df=template_cfg.fixed_t_df,
            initial_df=template_cfg.initial_t_df,
            t_iters=template_cfg.t_iters,
        )

    def compute_template_data(
        self, show_progress=True, computation_cfg=None, task_name=None
    ):
        # fit my TSVD object if necessary, and then turn it off
        # we don't need the TSVD projections in the processing, just need
        # to project at the end. (used to need it inside for the median.)
        super().fit_featurization_pipeline(computation_cfg=computation_cfg)
        self.featurization_pipeline = self.short_pipeline

        self.setup()
        while self.tasks.plan_step():
            self.setup_step()
            self.peel(
                output_hdf5_filename=None,
                ignore_resuming=True,
                show_progress=show_progress,
                computation_cfg=computation_cfg,
                task_name=task_name
                or f"{self.peel_kind}:{self.denoising_method}<{self.tasks.active_step}>",
            )
            self.finalize_step()
            self.tasks.finalize_step()

        return self.to_template_data()

    def finalize_step(self):
        if self.tasks.needs_basic_pass:
            self.finalize_basic()
        elif self.tasks.needs_mixture_pass:
            self.finalize_mixture()
        else:
            assert False

    def setup(self):
        # this sets motion-related buffers (pitch shifts, geom)
        self.setup_global()
        # sets up buffers for raw means + meansq
        self.setup_basic()
        # sets up subspace stuff and gamma stats
        self.setup_mixture()

    def setup_step(self):
        self.n_spikes = 0
        if self.tasks.active_step == "mix":
            self.b.total_resp.zero_()
            self.b.total_weight.zero_()
            self.b.subspace_resp.zero_()

    def to_template_data(self):
        from ..templates.templates import TemplateData

        counts_by_channel = self.counts.numpy(force=True)
        if self.group_ids is None:
            unit_ids = np.arange(self.n_units)
        else:
            unit_ids = self.group_ids

        templates = self.templates()
        assert templates.shape == (
            self.n_units,
            self.spike_length_samples,
            self.n_channels_full,
        )

        if not self.with_raw_std_dev:
            raw_stds = None
        elif self.denoising_method in ("none", "exp_weighted"):
            raw_stds = self.b.meansq.sub(self.raw_means.square())
            raw_stds = raw_stds.clamp_(min=0.0).sqrt_()
        else:
            raw_stds = self.b.meansq[None].sub(self.b.means.square())
            pi = self.b.log_props.exp()
            raw_stds = (raw_stds * pi).sum(dim=0)
            raw_stds = raw_stds.clamp_(min=0.0).sqrt_()
        if raw_stds is not None:
            assert raw_stds.isfinite().all()
            assert raw_stds.shape == templates.shape
            raw_stds = raw_stds.numpy(force=True)

        return TemplateData(
            templates=templates,
            unit_ids=unit_ids,
            spike_counts_by_channel=counts_by_channel,
            spike_counts=counts_by_channel.max(1),
            raw_std_dev=raw_stds,
            registered_geom=self.reg_geom,
            trough_offset_samples=self.trough_offset_samples,
            properties=self.properties or None,
        )

    def templates(self):
        assert self.tasks.done

        if self.denoising_method in (None, "none"):
            t = self.raw_means
            assert t.isfinite().all()
            return t.numpy(force=True)

        if self.denoising_method == "exp_weighted":
            from ..templates.get_templates import denoising_weights

            raw_means = self.raw_means.nan_to_num_().numpy(force=True)
            snrs = ptp(self.raw_means.nan_to_num(nan=-torch.inf)).mul_(
                self.b.counts.to(self.raw_means).sqrt()
            )
            weights = denoising_weights(
                snrs.numpy(force=True),
                spike_length_samples=self.spike_length_samples,
                trough_offset=self.trough_offset_samples,
                snr_threshold=self.exp_weight_snr_threshold,
            )
            w = weights.astype(raw_means.dtype)
            svd_means = self.b.means[self.svd_subspace_ix]
            svd_means = svd_means.nan_to_num_().numpy(force=True)
            logger.dartsortdebug(
                f"exp_weighted_svd: Raw weight mean/max={w.mean().item()}/{w.max().item()}"
            )
            m = w * raw_means + (1 - w) * svd_means
            assert np.isfinite(m).all()
            return m

        if self.denoising_method in ("loot", "t"):
            pi = self.b.log_props.exp()
            raw_pi = pi[self.raw_subspace_ix]
            logger.dartsortdebug(
                f"t_svd: Raw weight mean/max={raw_pi.mean().item():.4f}"
                f"/{raw_pi.max().item():.4f}"
            )
            mu = (pi * self.b.means).sum(dim=0)
            assert mu.isfinite().all()
            return mu.numpy(force=True)

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

        ires = self._process_chunk_main(res["waveforms"], res["indices"], res["labels"])
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
    ):
        """Handles batch statistics for raw or exp weighted templates

        Also is the initialization stage for the t flavored versions.
        """
        # scatter waveforms to reg channels, if necessary
        if self.motion_aware:
            chanix = self.b.target_channels[self.b.pitch_shift_ixs[spike_ix]]
            waveforms = registerize(waveforms, self.n_channels_full, chanix)
        else:
            chanix = None

        if self.tasks.active_step == "basic":
            return self._process_chunk_basic(waveforms, labels)
        elif self.tasks.active_step == "mixture":
            return self._process_chunk_mixture(waveforms, labels)

        assert False

    def _process_chunk_basic(self, waveforms: torch.Tensor, labels: torch.LongTensor):
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

        waveforms.nan_to_num_()

        # weighted avg
        raw_means = waveforms.new_zeros(self.raw_means.shape)
        x = waveforms.mul(weights[:, None])
        ix = labels[:, None, None].broadcast_to(x.shape)
        raw_means.scatter_add_(dim=0, index=ix, src=x)
        res["raw_means"] = raw_means

        if self.tasks.needs_meansq:
            meansq = waveforms.new_zeros(self.raw_means.shape)
            x.mul_(waveforms)
            meansq.scatter_add_(dim=0, index=ix, src=x)
            res["meansq"] = meansq

        return res

    def _process_chunk_mixture(self, waveforms: torch.Tensor, labels: torch.LongTensor):
        resp, what = smsm_resps_and_latents(
            x=waveforms,
            labels=labels,
            lps=self.log_props,
            nus=self.nu,
            mus=self.means,
            sigmas=self.scale,
            meansqs=self.meansq,
            loo=self.tasks.current_step_is_loot,
            loo_N=self.counts,
        )
        wsum, xbar, xsqbar = count_and_mean(
            resp=resp,
            w=what,
            x=waveforms,
            labels=labels,
            s=self.n_subspaces,
            k=self.n_units,
            with_sq=self.tasks.updating_subspace_stds,
        )
        rsum, rtotal, wbar, sqwbar, logwbar = gamma_count_and_mean(
            resp=resp,
            w=what,
            labels=labels,
            s=self.n_subspaces,
            k=self.n_units,
            count_only=not self.tasks.updating_df,
        )

        res = {}
        res["weights"] = wsum
        res["means"] = xbar
        if xsqbar is not None:
            res["meansq"] = xsqbar
        res["resps"] = rsum
        if wbar is not None:
            res["rtotal"] = rtotal
            res["wbar"] = wbar
            res["sqwbar"] = sqwbar
            res["logwbar"] = logwbar

        return res

    def _gather_chunk_main(self, chunk_result):
        """Update running stats with Welford formula."""
        if self.tasks.active_step == "basic":
            self._gather_chunk_basic(chunk_result)
        elif self.tasks.active_step == "mix":
            self._gather_chunk_mixture(chunk_result)
        else:
            assert False

    def _gather_chunk_basic(self, ch_res):
        self.counts += ch_res["counts"]
        denom = torch.where(self.counts > 0, self.counts, 1.0)
        w = ch_res["counts"].div_(denom).unsqueeze(1).to(self.raw_means)
        self.raw_means += ch_res["raw_means"].sub_(self.raw_means).mul_(w)
        if "meansq" in ch_res:
            self.meansq += ch_res["meansq"].sub_(self.meansq).mul_(w)

    def _gather_chunk_mixture(self, ch_res):
        # track responsibility sum for proportions / scale update
        self.total_resp += ch_res["resps"]

        # gamma weight
        self.total_weight += ch_res["weights"]
        d = torch.where(self.total_weight > 0, self.total_weight, 1.0)
        gw = ch_res["weights"].div_(d)

        self.means_new += ch_res["means"].sub_(self.means_new).mul_(gw)
        if self.tasks.updating_subspace_stds:
            self.scale_new += ch_res["meansq"].sub_(self.scale_new).mul_(gw)
        if self.tasks.updating_df:
            # resp weight only needed for gamma stats
            self.subspace_resp += ch_res["rtotal"]
            d = torch.where(self.subspace_resp > 0, self.subspace_resp, 1.0)
            rw = ch_res["rtotal"].div_(d)
            self.wbar += rw * (ch_res["wbar"] - self.wbar)
            self.logwbar += rw * (ch_res["logwbar"] - self.logwbar)
            self.sqwbar += rw * (ch_res["sqwbar"] - self.sqwbar)

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
                channel_index=self.b.channel_index.numpy(force=True),
                registered_geom=self.reg_geom,
                n_pitches_shift=unique_shifts,
            )
            target_channels = torch.asarray(res[0], device=self.b.channel_index.device)
            pitch_shift_ixs = torch.asarray(
                pitch_shift_ixs, device=self.b.channel_index.device
            )
            self.register_buffer("pitch_shift_ixs", pitch_shift_ixs)
            self.register_buffer("target_channels", target_channels)
        else:
            self.reg_geom = geom
            self.n_channels_full = self.recording.get_num_channels()

    def setup_basic(self):
        k = self.n_units
        t = self.spike_length_samples
        c = self.n_channels_full
        s = self.n_subspaces

        self.register_buffer("counts", torch.zeros((k, c)))

        self.register_buffer("means", torch.zeros((s, k, t, c)))
        # alias for raw part
        # this is the only bit that's written in the basic pass
        self.raw_means = self.b.means[self.raw_subspace_ix]

        if self.tasks.needs_meansq:
            self.register_buffer("meansq", torch.zeros((k, t, c)))
        else:
            self.meansq = None

        if self.tasks.needs_double_buffer:
            self.register_buffer("means_new", torch.zeros_like(self.b.means))
        else:
            self.means_new = self.means

    def setup_mixture(self):
        if not self.tasks.n_mixture_passes:
            return

        k = self.n_units
        t = self.spike_length_samples
        c = self.n_channels_full
        s = self.n_subspaces

        # -- gamma df
        # initial value for subspace dfs will be used in first mixture pass
        self.register_buffer("nu", torch.full((s,), self.initial_df))
        # ^ will be updated based on ...
        self.register_buffer("subspace_resp", torch.zeros((s,)))
        self.register_buffer("wbar", torch.zeros((s,)))
        self.register_buffer("sqwbar", torch.zeros((s,)))
        self.register_buffer("logwbar", torch.zeros((s,)))

        # -- pixelwise weights and responsibilities
        # log proportion is zero in first pass
        self.register_buffer("log_props", torch.zeros_like(self.b.means))
        # ^ is updated based on ...
        self.register_buffer("total_resp", torch.zeros_like(self.b.means))
        # total weight doesn't contribute to total resp, but it needs to be
        # known for intermediate averaging and also to rescale sigma ests in finalize
        # they're (sum weight moment)/Nk, not (sum weight moment)/(sum weight)
        self.register_buffer("total_weight", torch.zeros_like(self.b.means))

        # -- subspace scale params
        if self.tasks.needs_subspace_stds:
            self.register_buffer("scale", torch.zeros_like(self.b.means))
        else:
            self.scale = None
        if self.tasks.std_needs_double_buffer:
            self.register_buffer("scale_new", torch.zeros_like(self.b.means))
        elif self.tasks.needs_subspace_stds:
            self.scale_new = self.scale

    def finalize_basic(self):
        # save projected means
        print(f"fbasic {self.n_subspaces=}")
        for j in range(self.n_subspaces):
            if j == self.raw_subspace_ix:
                continue
            print(f"{j=} {self.subspace_projectors[j]=}")
            self.b.means[j].copy_(self.subspace_projectors[j](self.raw_means))
        assert self.b.means.isfinite().all()

        # we wrote into means here, not means_new, and i am acknowledging that
        if self.tasks.needs_double_buffer:
            pass

    def finalize_mixture(self):
        # -- means
        if self.needs_double_buffer:
            self.b.means.copy_(self.b.means_new)
            self.b.means_new.zero_()
            assert self.b.means.isfinite().all()

        # -- update stds
        if self.tasks.updating_subspace_stds:
            # to update sigma, formula as follows. let R be sum of resp,
            # W be sum of weight (ie total weight and total resp).
            # we have currently 1/W sum w y^2. our subspace means are
            # 1/W sum w y, so thankfully
            # 1/R sum w (y-mu)^2 = W/R [1/W (sum w y^2) + mu^2].
            self.b.scale_new.add_(self.b.means.square())
            self.b.scale_new.mul_(self.b.total_weight.div_(self.total_resp))
        if self.tasks.updating_subspace_stds and self.tasks.std_needs_double_buffer:
            self.b.scale.copy_(self.b.scale_new)
            self.b.scale_new.zero_()
        assert self.b.scale.isfinite().all()
        self.b.total_weight.zero_()

        # -- proportions
        self.b.total_resp.log_()
        self.b.log_props.copy_(F.log_softmax(self.total_resp, dim=0))
        assert self.b.log_props.isfinite().all()
        self.b.total_resp.zero_()

        # -- dof
        if self.tasks.updating_df:
            dfs = [
                estimate_gamma_df(wb, logwb, sqwb)
                for wb, logwb, sqwb in zip(self.wbar, self.logwbar, self.sqwbar)
            ]
            dfs = torch.tensor(dfs).to(self.b.nu)
            self.b.nu.copy_(dfs)
            self.b.wbar.zero_()
            self.b.logwbar.zero_()
            self.b.sqwbar.zero_()
            assert self.b.nu.isfinite().all()


@dataclass(kw_only=True)
class RunningTemplatesTasks:
    """Helper state machine for the RunningTemplates peeler.

    Cooperates with peeler's .compute_template_data() and .finalize_step().
    Sets updating_* flags read by peeler's chunk process and gather methods
    so it knows what it's doing.
    """

    denoising_method: Literal["none", "exp_weighted", "loot", "t"]
    fixed_df: bool
    with_raw_std_dev: bool
    t_iters: int
    use_zero: bool
    use_raw: bool
    use_svd: bool

    # things that are requested, so that peeler knows what to setup()
    needs_meansq: bool = field(init=False)
    needs_subspace_stds: bool = field(init=False)
    needs_double_buffer: bool = field(init=False)
    std_needs_double_buffer: bool = field(init=False)

    # progress tracker
    n_mixture_passes: int = field(init=False)
    n_steps: int = field(init=False)
    active_step: Literal["", "basic", "mix"] = ""

    # currently being processed
    updating_subspace_stds: bool = False
    current_step_is_loot: bool = False
    basic_pass_done: int = False
    mixture_passes_done: int = 0

    def __post_init__(self):
        assert self.denoising_method in ("none", "exp_weighted", "loot", "t")
        self.is_none = self.denoising_method == "none"
        self.is_loot = self.denoising_method == "loot"
        self.is_t = self.denoising_method == "t"
        self.is_weighted = self.denoising_method == "exp_weighted"
        self.is_t_or_loot = self.is_t or self.is_loot

        self.n_mixture_passes = (self.t_iters * self.is_t) + self.is_t_or_loot
        self.n_steps = 1 + self.n_mixture_passes

        self.needs_meansq = self.with_raw_std_dev or self.is_t or self.is_loot
        # only t needs to actually instantiate subspace std buffers
        # for loot, can read it off from meansq and subspace means
        self.needs_subspace_stds = self.is_t
        self.needs_double_buffer = bool(self.n_mixture_passes)
        self.std_needs_double_buffer = self.needs_subspace_stds and self.t_iters > 1

    def step_description(self):
        s = self.active_step
        if self.active_step == "basic" and self.needs_meansq:
            s = f"{s}+meansq"
        if self.n_steps > 1:
            istep = self.basic_pass_done + self.mixture_passes_done
            s = f"{s} [{istep}/{self.n_steps}]"
        return s

    def plan_step(self):
        if self.done:
            self.active_step = ""
            logger.dartsortdebug("Template tasks done.")
            return False
        elif self.needs_basic_pass:
            self.active_step = "basic"
        elif self.needs_mixture_pass:
            self.active_step = "mix"
            self.updating_subspace_stds = (
                self.needs_subspace_stds and not self.is_last_pass
            )
            self.updating_df = not self.is_last_pass
            self.current_step_is_loot = self.n_mixture_passes == 0
        else:
            assert False
        logger.dartsortdebug(f"Template task: {self.step_description()}.")
        return True

    def finalize_step(self):
        if self.done:
            assert False
        elif self.needs_basic_pass:
            self.basic_pass_done = True
        elif self.needs_mixture_pass:
            self.mixture_passes_done += 1
            self.updating_subspace_stds = False
            self.updating_df = False
            self.current_step_is_loot = False
        else:
            assert False

    @property
    def done(self):
        return not self.needs_basic_pass or self.needs_mixture_pass

    @property
    def needs_basic_pass(self):
        return not self.basic_pass_done

    @property
    def needs_mixture_pass(self):
        return self.mixture_passes_done < self.n_mixture_passes

    @property
    def is_last_pass(self):
        if not self.n_mixture_passes:
            return True
        else:
            return self.mixture_passes_done == self.n_mixture_passes - 1


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
    chanix: torch.Tensor | None = None,
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
    if isinstance(sigma, (float, int)):
        sigma = torch.asarray(float(sigma)).to(x)
    else:
        sigma = sigma[labels]
    z = mu[labels].sub_(x).div_(sigma)
    denom = z.square_().add_(nu)
    w = torch.divide(nu + 1.0, denom, out=denom)
    return w


def estimate_gamma_df(wbar, logwbar, sqwbar, cutoff=100, min_df=0.01):
    """Solves digamma(nu/2) - log(nu) = 1 + logwbar - wbar."""
    from scipy.special import psi  # , polygamma
    from scipy.optimize import root_scalar

    dtype = wbar.dtype
    dev = wbar.device

    wbar = wbar.numpy(force=True).astype(np.float64)
    logwbar = logwbar.numpy(force=True).astype(np.float64)
    sqwbar = sqwbar.numpy(force=True).astype(np.float64)

    # start with a guess based on the gamma MoM
    vw = sqwbar - wbar**2
    if not np.isfinite(vw) and vw >= 0:
        raise ValueError(f"Strange gamma statistics {wbar=} {logwbar=} {sqwbar=}.")
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

    # brent turned out to be better than deriv aware methods
    def f(x):
        p = psi(x)
        # p1 = polygamma(1, x)
        # p2 = polygamma(2, x)
        # invx = np.reciprocal(x)
        f = p - np.log(x) - rhs
        # df = p1 - invx
        # ddf = p2 + invx * invx
        return f  # , df, ddf

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

    sol = root_scalar(f, bracket=(left, right), x0=x0, x1=x1)  # fprime2=True)
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


def smsm_resps_and_latents(
    x, labels, lps, nus, mus, sigmas=None, meansqs=None, loo=False, loo_N=None
):
    s = len(lps)

    # TODO: can speed this up. all times same.
    finite = x.isfinite().nonzero(as_tuple=True)
    xf = x[finite]

    # log liks for computing resps
    ll = x.new_full((s, *x.shape), -torch.inf)
    # gamma latent weight
    w = x.new_zeros((s, *x.shape))

    if loo:
        assert loo_N is not None
        loo_N = loo_N[labels]
        assert loo_N.shape == x.shape
        Nf = loo_N[finite]
        N_N1 = Nf / (Nf - 1)
        xf_N = xf / Nf
        xfsq_N = xf.square().div_(Nf)
    else:
        N_N1 = xf_N = xfsq_N = None

    if sigmas is None:
        sigmas = [None] * s
    if meansqs is None:
        meansqs = [None] * s

    params = zip(lps, nus, mus, sigmas, meansqs)
    for j, (lp, nu, mu, sigma, meansq) in enumerate(params):
        lp = lp[labels][finite]
        loc = mu[labels][finite]

        if loo:
            assert meansq is not None
            loc.sub_(xf_N).mul_(N_N1)
            scale = meansq[labels][finite].sub_(xfsq_N).mul_(N_N1)
            scale.sub_(loc.square())
        else:
            assert sigma is not None
            scale = sigma[labels][finite]

        if s == 1:
            l = 0.0  # doesn't matter at all
        elif nu < torch.inf:
            l = StudentT(nu, loc=loc, scale=scale, validate_args=False).log_prob(xf)
            l.add_(lp)
        else:
            l = Normal(loc=loc, scale=scale, validate_args=False).log_prob(xf)
            l.add_(lp)

        ll[j, *finite] = l

        if nu < torch.inf:
            z = loc.sub(xf).div_(scale)
            denom = z.square_().add_(nu)
            w[j, *finite] = torch.divide(nu + 1.0, denom, out=denom)
        else:
            w[j, *finite] = 1.0

    resp = F.softmax(ll, dim=0)
    return resp, w


def count_and_mean(
    resp,
    w,
    x,
    labels,
    s,
    k,
    with_sq=False,
):
    n, t, c = x.shape
    assert w.shape == (s, n, t, c)

    wr = w * resp

    wsum = w.new_zeros((s, k, t, c))
    ix = labels[None, :, None, None].broadcast_to(w.shape)
    wsum.scatter_add_(dim=1, index=ix, src=wr)

    denom = torch.where(wsum > 0, wsum, 1.0)[:, labels]
    assert denom.shape == (n, s, t, c)
    ww = wr / denom.permute(1, 0, 2, 3)

    xbar = x.new_zeros((s, k, t, c))
    wx = ww.mul_(x[None])
    del ww
    xbar.scatter_add_(dim=1, index=ix, src=wx)

    if with_sq:
        xsqbar = x.new_zeros((s, k, t, c))
        wx.mul_(x[None])
        xsqbar.scatter_add_(dim=1, index=ix, src=wx)
    else:
        xsqbar = None

    return wsum, xbar, xsqbar


def gamma_count_and_mean(resp, w, labels, s, k, count_only=False):
    n, t, c = w.shape

    rsum = resp.new_zeros((s, k, t, c))
    ix = labels[None, :, None, None].broadcast_to(w.shape)
    rsum.scatter_add_(dim=1, index=ix, src=resp)

    if count_only:
        return rsum, None, None, None, None

    rtotal = rsum.sum(dim=(1, 2, 3))
    weight = resp / rtotal
    w = w.nan_to_num()
    wbar = (w * weight).sum(dim=(1, 2, 3))
    sqwbar = w.square().mul_(weight).sum(dim=(1, 2, 3))
    logwbar = w.log().mul_(weight).sum(dim=(1, 2, 3))

    return rsum, rtotal, wbar, sqwbar, logwbar


def wsum(shp, w, ix, src, dim=0):
    out = src.new_zeros(shp)
    out.scatter_add_(dim=dim, ix=ix, src=src.mul(w))


def identity(x):
    return x
