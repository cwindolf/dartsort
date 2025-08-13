"""Fast template extractor

Algorithm: Each unit / pitch shift combo has a raw and low-rank template
computed. These are shift-averaged and then weighted combined.

The raw and low-rank templates are computed using Welford.
"""

import numpy as np
import torch

from ..util import drift_util
from ..transform import WaveformPipeline, Waveform, TemporalPCADenoiser

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
        denoising_method=None,
        tsvd=None,
        tsvd_rank=5,
        tsvd_fit_radius=75.0,
        group_ids=None,
        motion_est=None,
        with_raw_std_dev=False,
        trough_offset_samples=42,
        spike_length_samples=121,
        chunk_length_samples=30_000,
        n_seconds_fit=40,
        fit_subsampling_random_state=0,
    ):
        n_channels = recording.get_num_channels()
        channel_index = torch.tile(torch.arange(n_channels), (n_channels, 1))
        assert labels.min() >= 0

        self.denoising_method = denoising_method
        featurization_pipeline = None
        if denoising_method not in ("none", None):
            if denoising_method == "exp_weighted_svd":
                if tsvd is not None:
                    tpca = TemporalPCADenoiser.from_sklearn(
                        channel_index, tsvd, getattr(tsvd, "temporal_slice", None)
                    )
                else:
                    tpca = TemporalPCADenoiser(
                        channel_index, rank=tsvd_rank, centered=False
                    )
                featurization_pipeline = WaveformPipeline([tpca, Waveform(channel_index)])
            else:
                raise ValueError(f"Unknown {denoising_method=}.")

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
        if group_ids is not None:
            assert group_ids.shape == (self.n_units,)

    def get_templates(self, show_progress=True, computation_cfg=None) -> TemplateData:
        self.setup()
        self.peel(
            output_hdf5_filename=None,
            skip_features=True,
            ignore_resuming=True,
            show_progress=show_progress,
            computation_cfg=computation_cfg,
        )
        return self.to_template_data()

    def setup(self):
        if self.motion_aware:
            # TODO: user can pass in pitch shifts and reg geom
            geom = self.recording.get_channel_locations()
            self.reg_geom = drift_util.registered_geometry(geom, self.motion_est)
            self.n_channels_full = len(self.reg_geom)

            depths = geom[self.channels, 1]
            n_pitches_shift = drift_util.get_spike_pitch_shifts(
                depths, geom, motion_est=self.motion_est
            )
            unique_shifts, self.pitch_shift_ixs = n_pitches_shift.unique(
                return_inverse=True
            )
            res = drift_util.get_stable_channels(
                geom,
                torch.zeros_like(unique_shifts),
                self.channel_index,
                self.reg_geom,
                unique_shifts,
            )
            target_channels = torch.asarray(res[0], device=self.channel_index.device)
            self.register_buffer("target_channels", target_channels)
        else:
            self.reg_geom = None
            self.n_channels_full = self.recording.get_num_channels()

        # storage for templates, stds, counts
        n = self.n_units
        t = self.spike_length_samples
        c = self.n_channels_full
        self.register_buffer("means", torch.zeros((n, t, c)))
        self.register_buffer("counts", torch.zeros((n, c)))
        if self.with_raw_std_dev:
            self.register_buffer("stds", torch.zeros((n, t, c)))
        else:
            self.stds = None

    def to_template_data(self):
        from ..templates.templates import TemplateData

        counts_by_channel = self.counts.numpy(force=True)
        if self.group_ids is None:
            unit_ids = np.arange(self.n_units)
        else:
            unit_ids = self.group_ids
        return TemplateData(
            templates=self.means.numpy(force=True),
            unit_ids=unit_ids,
            spike_counts_by_channel=counts_by_channel,
            spike_counts=counts_by_channel.max(1),
            raw_std_dev=self.stds.numpy(force=True),
            registered_geom=self.reg_geom,
            trough_offset_samples=self.trough_offset_samples,
        )

    def process_chunk(
        self,
        chunk_start_samples,
        n_resid_snips=None,
        chunk_end_samples=None,
        return_residual=False,
        skip_features=False,
    ):
        res = super().process_chunk(
            chunk_start_samples,
            n_resid_snips,
            chunk_end_samples,
            return_residual,
            skip_features,
        )
        if not res["n_spikes"]:
            return res

        # scatter waveforms to reg channels, if necessary
        waveforms: torch.Tensor = res["collisioncleaned_waveforms"]
        spike_ix: torch.LongTensor = res["indices"]
        labels: torch.LongTensor = res["labels"]
        if self.motion_aware:
            ix = self.target_channels[self.pitch_shift_ixs[spike_ix]]
            ix = ix[:, None, :].broadcast_to(waveforms.shape)
            reg_waveforms = waveforms.new_full(
                (*waveforms.shape[:2], self.n_reg_channels), torch.nan
            )
            waveforms = reg_waveforms.scatter_(waveforms, dim=2, index=ix)

        # determine each unit's per-channel counts, for weighting purposes
        # TODO: sum responsibility weights here
        counts = waveforms.new_zeros((self.n_units, self.n_channels_full))
        weights = waveforms[:, 0].isfinite().to(waveforms)
        counts.scatter_add_(dim=0, index=labels[:, None], src=weights)

        # normalized weights in each unit
        weights.div_(counts[labels])

        # extract squares into a copy if nec
        waveforms.nan_to_num_()
        waveformsq = None
        if self.with_raw_std_dev:
            waveformsq = waveforms.square()

        # weighted sum
        means = waveforms.new_zeros(self.means.shape)
        waveforms.mul_(weights[:, None])
        means.scatter_add_(dim=0, index=labels[:, None, None], src=waveforms)

        # "" std, optionally ""
        if self.with_raw_std_dev:
            assert waveformsq is not None
            stds = waveforms.new_zeros(means.shape)
            waveformsq.mul_(weights[:, None])
            stds.scatter_add_(dim=0, index=labels[:, None, None], src=waveformsq)
            stds.sub_(means.square())
            res["stds"] = stds

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
        assert residual_file is None
        assert not residual_to_h5
        assert skip_features
        assert ignore_resuming

        if not chunk_result["n_spikes"]:
            return

        # update running count and do Welford sums
        self.counts += chunk_result["counts"]
        w = chunk_result["counts"].div_(self.counts)
        self.means += chunk_result["means"].sub_(self.means).mul_(w)
        if self.with_raw_std_dev:
            self.stds += chunk_result["stds"].sub_(self.stds).mul_(w)
