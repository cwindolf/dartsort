"""
TODO:
 - uses np1 probe in full_denoising
 - SubtractionExtractor
"""
import numpy as np
import torch
import h5py

from . import spikeio, subtract, denoise, localize_index, waveform_utils


class BaseExtractor:
    """
    This class is never used, it's just a stencil to show (or help
    us remember) the API that the classes below support.
    """

    def __init__(self, seed):
        self.rg = np.random.default_rng(seed)

    def get_unit_waveforms(
        self, unit, channel_index=None, n_samples=250, kind="cleaned"
    ):
        assert kind in ("raw", "subtracted", "cleaned", "denoised")
        in_unit = np.flatnonzero(self.spike_train[:, 1] == unit)
        choices = self.rg.choice(
            in_unit, size=min(n_samples, in_unit.size), replace=False
        )
        waveforms, skipped_idx = self.get_waveforms(choices)
        return waveforms

    def get_localizations(self, indices):
        x, y, z, ptp = None, None, None, None
        return x, y, z, ptp

    def get_waveforms(self, indices, channel_index=None, kind="cleaned"):
        assert kind in ("raw", "subtracted", "cleaned", "denoised")
        waveforms = None
        return waveforms


class DeconvExtractor(BaseExtractor):
    """Requires a residual, but figures out waveforms on the fly."""

    def __init__(
        self,
        spike_train,
        templates,
        spike_train_up,
        templates_up,
        raw_bin,
        residual_bin,
        geom,
        loc_radius_um=100,
        device=None,
        tpca=None,
        trough_offset=42,
        spike_length_samples=121,
        seed=0,
        n_loc_workers=1,
    ):
        assert self.spike_train.shape == self.spike_train_up.shape
        super().__init__(seed)

        self.spike_train = spike_train
        self.templates = templates
        self.spike_train_up = spike_train_up
        self.templates_up = templates_up
        self.raw_bin = raw_bin
        self.residual_bin = residual_bin
        self.geom = geom

        self.trough_offset = trough_offset
        self.spike_length_samples = spike_length_samples

        self.template_maxchans = self.templates.ptp(1).argmax(1)
        self.templates_up_maxchans = self.templates_up.ptp(1).argmax(1)
        self.n_spikes = self.spike_train.shape[0]
        self.n_channels = self.geom.shape[0]
        self.max_channels = self.templates_up_maxchans[self.spike_train[:, 1]]

        self.localized = np.full(self.n_spikes, False)
        self.x = np.full(self.n_spikes, np.nan)
        self.y = np.full(self.n_spikes, np.nan)
        self.z = np.full(self.n_spikes, np.nan)
        self.ptp = np.full(self.n_spikes, np.nan)
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        self.denoiser = None
        self.tpca = tpca
        self.loc_channel_index = subtract.make_channel_index(
            geom, loc_radius_um, distance_order=False
        )
        self.n_loc_workers = n_loc_workers

    def get_localizations(self, indices):
        indices_to_localize = indices[~self.localized[indices]]
        if indices_to_localize.size:
            waveforms = self.get_waveforms(
                indices_to_localize, self.loc_channel_index, kind="denoised"
            )
            ptps = waveforms.ptp(1)
            max_ptps = ptps.max(1)
            x, y, z_rel, z_abs, alpha = localize_index.localize_ptps_index(
                ptps,
                self.geom,
                self.max_channels[indices_to_localize],
                self.loc_channel_index,
                n_workers=self.n_loc_workers,
                pbar=True,
                logbarrier=True,
            )
            self.x[indices_to_localize] = x
            self.y[indices_to_localize] = y
            self.z[indices_to_localize] = z_abs
            self.ptp[indices_to_localize] = max_ptps
        self.localized[indices_to_localize] = True
        return (
            self.x[indices],
            self.y[indices],
            self.z[indices],
            self.ptp[indices],
        )

    def get_waveforms(self, indices, channel_index=None, kind="cleaned", channels=None):
        assert kind in ("raw", "subtracted", "cleaned", "denoised")
        trough_times = self.spike_train[indices, 0]
        max_channels = self.max_channels[indices]

        # user can supply their own set of channels
        if channels is not None:
            assert channel_index is None
            channels = np.atleast_1d(channels)
            assert channels.ndim == 1
            channel_index = np.tile(channels, (self.channel_index.size, 1))

        if kind == "raw":
            waveforms, skipped_idx = spikeio.read_waveforms(
                trough_times,
                self.raw_bin,
                self.n_channels,
                channel_index=channel_index,
                max_channels=max_channels,
                spike_length_samples=self.spike_length_samples,
                trough_offset=self.trough_offset,
                dtype=np.float32,
                fill_value=np.nan,
            )

            return waveforms

        subtracted_wfs = self._get_subtracted_waveforms(indices, channel_index)

        if kind == "subtracted":
            return subtracted_wfs

        # for either collision-cleaned or denoised waveforms,
        # start with the residual
        waveforms, skipped_idx = spikeio.read_waveforms(
            trough_times,
            self.residual_bin,
            self.n_channels,
            channel_index=channel_index,
            max_channels=max_channels,
            spike_length_samples=self.spike_length_samples,
            trough_offset=self.trough_offset,
            dtype=np.float32,
            fill_value=np.nan,
        )

        if skipped_idx.size:
            raise ValueError("Invalid trough times when loading non-raw wfs.")

        waveforms += subtracted_wfs
        del subtracted_wfs

        if kind == "cleaned":
            return waveforms

        # now, kind == "denoised"
        # for denoised waveforms, all that's left is to run the denoiser
        # we lazily initialize it
        if self.denoiser is None:
            self.denoiser = denoise.SingleChanDenoiser()
            self.denoiser.load()
            self.denoiser.to(self.device)

        waveforms = subtract.full_denoising(
            waveforms,
            max_channels,
            channel_index,
            probe="np1",
            tpca=self.tpca,
            device=self.device,
            denoiser=self.denoiser,
        )

        return waveforms

    def _get_subtracted_waveforms(self, indices, channel_index):
        templates_up_loc = np.empty(
            (*self.templates_up.shape[:2], channel_index.shape[1]),
            dtype=self.templates_up.dtype,
        )
        for i in range(self.templates.shape[0]):
            templates_up_loc[i] = self.templates_up[i][
                :, channel_index[self.templates_up_maxchans[i]]
            ]
        return templates_up_loc[self.spike_train_up[indices, 1]]


class DeconvH5Extractor(BaseExtractor):
    """In case you already ran the full `extract_deconv`."""

    def __init__(
        self, deconv_results_h5, raw_bin, spike_train=None, templates=None, residual_bin=None, tpca=None, device=None, seed=0,
    ):
        super().__init__(seed)
        self.deconv_results_h5 = deconv_results_h5
        self.residual_bin = residual_bin
        self.raw_bin = raw_bin
        
        # these are optional, since they aren't necessary for just loading waveforms
        self.spike_train = spike_train
        self.templates = templates

        # extract some of the things that fit in memory, but not
        # the waveforms
        with h5py.File(deconv_results_h5, "r") as h5:
            self.channel_index = h5["channel_index"][:]
            self.localizations = h5["localizations"][:]
            self.ptp = h5["maxptps"][:]
            self.templates_up = h5["templates_up"][:]
            self.templates_up_loc = h5["templates_loc"][:]
            self.templates_up_maxchans = h5["templates_up_maxchans"][:]
            self.spike_train_up = h5["spike_train_up"][:]
            self.max_channels = h5["spike_index_up"][:, 1]

            assert "cleaned_waveforms" in h5
            self.has_denoised = "denoised_waveforms" in h5

        self.spike_length_samples = self.templates_up.shape[1]
        self.denoiser = None
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        self.tpca = tpca

    def _get_subtracted_waveforms(self, indices, channel_index):
        temp_wfs_loc = self.templates_up_loc[self.spike_train_up[indices, 1]]

        if channel_index.shape == self.channel_index.shape:
            if (channel_index == self.channel_index).all():
                return temp_wfs_loc

        return waveform_utils.channel_subset_by_index(
            temp_wfs_loc,
            self.max_channels[indices],
            self.channel_index,
            channel_index,
        )

    def _waveforms_from_h5(
        self, indices, channel_index, kind="cleaned", batch_size=1000
    ):
        assert kind in ("cleaned", "denoised")  # the others are not in the h5
        key = f"{kind}_waveforms"

        # are we loading fewer chans than stored in h5?
        load_subset = True
        if channel_index.shape == self.channel_index.shape:
            if (channel_index == self.channel_index).all():
                return False

        with h5py.File(self.deconv_results_h5, "r") as h5:
            wfs_dset = h5[key]
            out = np.empty(
                (
                    indices.size,
                    self.spike_length_samples,
                    channel_index.shape[1],
                ),
                dtype=wfs_dset.dtype,
            )

            for start in range(0, indices.size, batch_size):
                end = min(indices.size, start + batch_size)
                ixs = indices[start:end]
                wfs = wfs_dset[ixs]

                if load_subset:
                    wfs = waveform_utils.channel_subset_by_index(
                        wfs,
                        self.max_channels[ixs],
                        self.channel_index,
                        channel_index,
                    )

                out[start:end] = wfs

        return out

    def get_waveforms(self, indices, channel_index=None, kind="cleaned", channels=None):
        assert kind in ("raw", "subtracted", "cleaned", "denoised")
        trough_times = self.spike_train_up[indices, 0]
        max_channels = self.max_channels[indices]

        # user can supply their own set of channels
        if channels is not None:
            assert channel_index is None
            channels = np.atleast_1d(channels)
            assert channels.ndim == 1
            channel_index = np.tile(channels, (self.channel_index.size, 1))

        if kind == "raw":
            waveforms, skipped_idx = spikeio.read_waveforms(
                trough_times,
                self.raw_bin,
                self.n_channels,
                channel_index=channel_index,
                max_channels=max_channels,
                spike_length_samples=self.spike_length_samples,
                trough_offset=self.trough_offset,
                dtype=np.float32,
                fill_value=np.nan,
            )
            return waveforms

        if kind == "subtracted":
            waveforms = self._get_subtracted_waveforms(indices, channel_index)
            return waveforms

        if kind == "cleaned" or not self.has_denoised:
            waveforms = self._waveforms_from_h5(
                indices, channel_index, kind="cleaned"
            )
            if kind == "cleaned":
                return waveforms

        if self.has_denoised:
            waveforms = self._waveforms_from_h5(
                indices, channel_index, kind="denoised"
            )
            return waveforms

        # now, kind == "denoised" and they are not in h5
        # for denoised waveforms, all that's left is to run the denoiser
        # we lazily initialize it
        if self.denoiser is None:
            self.denoiser = denoise.SingleChanDenoiser()
            self.denoiser.load()
            self.denoiser.to(self.device)

        waveforms = subtract.full_denoising(
            waveforms,
            max_channels,
            channel_index,
            probe="np1",
            tpca=self.tpca,
            device=self.device,
            denoiser=self.denoiser,
        )

        return waveforms
