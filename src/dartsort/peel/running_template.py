"""Fast template extractor

Algorithm: Each unit / pitch shift combo has a raw and low-rank template
computed. These are shift-averaged and then weighted combined.

The raw and low-rank templates are computed using Welford.
"""

import numpy as np
import torch


from ..templates.superres_util import superres_sorting
from ..transform.transform_base import BaseWaveformModule
from ..transform import WaveformPipeline, Waveform, TemporalPCAFeaturizer
from ..util import drift_util
from ..util.data_util import load_stored_tsvd, subsample_to_max_count
from ..util.spiketorch import ptp
from ..util.waveform_util import full_channel_index

from .grab import GrabAndFeaturize


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

        self.denoising_method = denoising_method
        self.denoising_rank = None
        self.denoising_snr_threshold = denoising_snr_threshold
        self.compute_low_rank_mean = False

        # these are not used internally, just stored to record what happened in
        # .from_config() so that users can realign their sortings to match
        self.time_shifts = time_shifts

        featurization_pipeline: List[BaseWaveformModule] = [
            Waveform(channel_index=channel_index)
        ]
        if denoising_method not in ("none", None):
            if denoising_method == "exp_weighted_svd":
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
                featurization_pipeline.append(tpca)
                self.denoising_rank = tpca.rank
                self.compute_low_rank_mean = True
            else:
                raise ValueError(f"Unknown {denoising_method=}.")
        else:
            tpca = None
        featurization_pipeline = WaveformPipeline(featurization_pipeline)

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
        self.tpca = tpca

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
        if tsvd is None and template_cfg.denoising_method not in ("none", None):
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
        )

    def compute_template_data(
        self, show_progress=True, computation_cfg=None, task_name=None
    ):
        super().fit_featurization_pipeline(computation_cfg=computation_cfg)
        self.setup()
        self.peel(
            output_hdf5_filename=None,
            ignore_resuming=True,
            show_progress=show_progress,
            computation_cfg=computation_cfg,
            task_name=task_name,
        )
        return self.to_template_data()

    def setup(self):
        geom = self.recording.get_channel_locations()
        if self.motion_aware:
            self.reg_geom = drift_util.registered_geometry(geom, self.motion_est)
            self.n_channels_full = len(self.reg_geom)

            n_pitches_shift = self.n_pitches_shift
            if n_pitches_shift is None:
                depths = np.atleast_1d(geom[self.channels, 1])
                times = self.recording.sample_index_to_time(self.times_samples)
                print(f"{depths.shape=} {times.shape=}")
                print(f"{depths=} {times=}")
                print(f"{self.channels=} {self.times_samples=}")
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
            pitch_shift_ixs = torch.asarray(pitch_shift_ixs, device=self.channel_index.device)
            self.register_buffer("pitch_shift_ixs", pitch_shift_ixs)
            self.register_buffer("target_channels", target_channels)
        else:
            self.reg_geom = geom
            self.n_channels_full = self.recording.get_num_channels()

        # storage for templates, stds, counts
        n = self.n_units
        t = self.spike_length_samples
        c = self.n_channels_full
        self.register_buffer("means", torch.zeros((n, t, c)))
        self.register_buffer("counts", torch.zeros((n, c)))
        if self.with_raw_std_dev:
            self.register_buffer("meansq", torch.zeros((n, t, c)))
        else:
            self.meansq = None
        if self.compute_low_rank_mean:
            assert self.denoising_rank is not None
            self.register_buffer("pcmeans", torch.zeros((n, self.denoising_rank, c)))
        else:
            self.pcmeans = None

    def to_template_data(self):
        from ..templates.templates import TemplateData

        counts_by_channel = self.counts.numpy(force=True)
        if self.group_ids is None:
            unit_ids = np.arange(self.n_units)
        else:
            unit_ids = self.group_ids

        templates = self.templates()

        return TemplateData(
            templates=templates,
            unit_ids=unit_ids,
            spike_counts_by_channel=counts_by_channel,
            spike_counts=counts_by_channel.max(1),
            raw_std_dev=self.stds(numpy=True),
            registered_geom=self.reg_geom,
            trough_offset_samples=self.trough_offset_samples,
        )

    def templates(self):
        raw_templates = self.means.numpy(force=True)
        raw_templates = np.nan_to_num(raw_templates)

        if self.denoising_method in (None, "none"):
            return raw_templates

        if self.denoising_method == "exp_weighted_svd":
            from ..templates.get_templates import denoising_weights

            assert self.tpca is not None
            assert self.pcmeans is not None

            weights = denoising_weights(
                self.snrs_by_channel().numpy(force=True),
                spike_length_samples=self.spike_length_samples,
                trough_offset=self.trough_offset_samples,
                snr_threshold=self.denoising_snr_threshold,
            )
            weights = weights.astype(raw_templates.dtype)
            low_rank_templates = self.pcmeans.nan_to_num()
            low_rank_templates = self.tpca.force_reconstruct(low_rank_templates)
            low_rank_templates = low_rank_templates.numpy(force=True)
            templates = weights * raw_templates + (1 - weights) * low_rank_templates
            return templates

        assert False

    def snrs_by_channel(self):
        return ptp(self.means.nan_to_num(nan=-torch.inf)).mul_(self.counts.to(self.means).sqrt())

    def stds(self, numpy=False):
        if self.meansq is None:
            return None
        means_sq = self.means.square()
        stds = torch.subtract(self.meansq, means_sq, out=means_sq)
        stds = stds.abs_().nan_to_num_().sqrt_()
        if numpy:
            stds = stds.numpy(force=True)
        return stds

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

        # scatter waveforms to reg channels, if necessary
        waveforms: torch.Tensor = res["waveforms"]
        spike_ix: torch.LongTensor = res["indices"]
        labels: torch.LongTensor = res["labels"]
        chanix = None
        if self.motion_aware:
            chanix = self.target_channels[self.pitch_shift_ixs[spike_ix]]
            chanix = chanix[:, None, :]
            ix = chanix.broadcast_to(waveforms.shape)
            reg_waveforms = waveforms.new_full(
                (*waveforms.shape[:2], self.n_channels_full), torch.nan
            )
            waveforms = reg_waveforms.scatter_(src=waveforms, dim=2, index=ix)

        # determine each unit's per-channel counts, for weighting purposes
        # TODO: sum responsibility weights here
        counts = waveforms.new_zeros((self.n_units, self.n_channels_full))
        weights = waveforms[:, 0].isfinite().to(waveforms)
        ix = labels[:, None].broadcast_to(weights.shape)
        counts.scatter_add_(dim=0, index=ix, src=weights)

        # normalized weights in each unit
        denom = torch.where(counts > 0, counts, 1.0)
        weights.div_(denom[labels])

        # extract squares into a copy if nec
        waveforms.nan_to_num_()
        waveformsq = None
        if self.with_raw_std_dev:
            waveformsq = waveforms.square()

        # weighted sum
        means = waveforms.new_zeros(self.means.shape)
        waveforms.mul_(weights[:, None])
        ix = labels[:, None, None].broadcast_to(waveforms.shape)
        means.scatter_add_(dim=0, index=ix, src=waveforms)

        # "" std, optionally ""
        if self.with_raw_std_dev:
            assert waveformsq is not None
            meansq = waveforms.new_zeros(means.shape)
            waveformsq.mul_(weights[:, None])
            meansq.scatter_add_(dim=0, index=ix, src=waveformsq)
            res["meansq"] = meansq

        if self.compute_low_rank_mean:
            assert self.pcmeans is not None
            pcmeans = waveforms.new_zeros(self.pcmeans.shape)
            pcfeats: torch.Tensor = res["tpca_features"]
            if self.motion_aware:
                assert chanix is not None
                reg_pcfeats = pcfeats.new_full(
                    (*pcfeats.shape[:2], self.n_channels_full), torch.nan
                )
                pcfeats = reg_pcfeats.scatter_(
                    src=pcfeats, dim=2, index=chanix.broadcast_to(pcfeats.shape)
                )
            pcfeats.mul_(weights[:, None])
            ix = labels[:, None, None].broadcast_to(pcfeats.shape)
            pcmeans.scatter_add_(dim=0, index=ix, src=pcfeats)
            res["pcmeans"] = pcmeans

        res["counts"] = counts
        res["means"] = means

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

        # update running count and do Welford sums
        self.counts += chunk_result["counts"]
        denom = torch.where(self.counts > 0, self.counts, 1.0)
        w = chunk_result["counts"].div_(denom).unsqueeze(1)
        self.means += chunk_result["means"].sub_(self.means).mul_(w)
        if self.with_raw_std_dev:
            self.meansq += chunk_result["meansq"].sub_(self.meansq).mul_(w)
        if self.compute_low_rank_mean:
            self.pcmeans += chunk_result["pcmeans"].sub_(self.pcmeans).mul_(w)

        return n_spikes


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

    sorting, _, time_shifts = realign_sorting(
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
