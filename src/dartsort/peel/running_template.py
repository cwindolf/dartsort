"""Fast template extractor

Algorithm: Each unit / pitch shift combo has a raw and low-rank template
computed. These are shift-averaged and then weighted combined.

The raw and low-rank templates are computed using Welford.
"""

import torch

from ..templates.templates import TemplateData
from ..util import drift_util

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
        motion_est=None,
        with_raw_std_dev=False,
        trough_offset_samples=42,
        spike_length_samples=121,
        chunk_length_samples=30_000,
        n_seconds_fit=40,
        fit_subsampling_random_state=0,
    ):
        n_channels = recording.get_num_channels()
        full_channel_index = torch.tile(torch.arange(n_channels), (n_channels, 1))
        assert labels.min() >= 0

        super().__init__(
            recording=recording,
            featurization_pipeline=None,
            times_samples=times_samples,
            channels=channels,
            labels=labels,
            channel_index=full_channel_index,
            trough_offset_samples=trough_offset_samples,
            spike_length_samples=spike_length_samples,
            chunk_length_samples=chunk_length_samples,
            n_seconds_fit=n_seconds_fit,
            fit_subsampling_random_state=fit_subsampling_random_state,
        )

        self.motion_aware = motion_est is not None
        self.motion_est = motion_est
        self.with_raw_std_dev = with_raw_std_dev

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
            geom = self.recording.get_channel_locations()
            self.reg_geom = drift_util.registered_geometry(geom, self.motion_est)
            self.n_channels_full = len(self.reg_geom)

            depths = geom[self.channels, 1]
            n_pitches_shift = drift_util.get_spike_pitch_shifts(depths, geom, motion_est=self.motion_est)
            unique_shifts, self.pitch_shift_ixs = n_pitches_shift.unique(return_inverse=True)
            res = drift_util.get_stable_channels(
                geom, torch.zeros_like(unique_shifts), self.channel_index, self.reg_geom, unique_shifts,
            )
            target_channels = torch.asarray(res[0], device=self.channel_index.device)
            self.register_buffer("target_channels", target_channels)
        else:
            self.n_channels_full = self.recording.get_num_channels()

        # storage for templates, stds, counts

        pass

    def to_template_data(self):
        pass

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
        if not res['n_spikes']:
            return res

        # -- chunk-local average/count/std on registered channels
        # scatter to reg channels, if necessary
        waveforms = res['collisioncleaned_waveforms']
        spike_ix = res['indices']
        labels = res['labels']
        if self.motion_aware:
            ix = self.target_channels[self.pitch_shift_ixs[spike_ix]]
            ix = ix[:, None, :].broadcast_to(waveforms.shape)
            reg_waveforms = waveforms.new_full((*waveforms.shape[:2], self.n_reg_channels), torch.nan)
            waveforms = reg_waveforms.scatter_(waveforms, dim=2, index=ix)

        # weighted sum. TODO, responsibility weights.
        unique_labels, unique_inds, counts = labels.unique(return_inverse=True, return_counts=True)
        weights = (1.0 / counts)[unique_inds]

        means = waveforms.new_zeros((self.n_units, self.n_channels_full))

        # "" count ""
        # count need not be per-chan if not motion aware.
        # "" std, optionally ""

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
        # update running count and do welford sums
        pass
