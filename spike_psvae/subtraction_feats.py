import numpy as np
from sklearn.decomposition import PCA

from spike_psvae import localize_index


class ExtraFeat:
    # subclasses should have these as class properties,
    # or assigned during __init__
    name = NotImplemented
    # this can be assigned during fit if necessary
    out_shape = NotImplemented  # shape per waveform
    # default dtype is set to our default wf dtype
    # but we could change this...
    dtype = np.float32  # dtype of this feature
    # helper property, set it if fit is implemented
    needs_fit = False
    # featurizers should set this to one of "subtracted", "cleaned", "denoised".
    # featurizers will return None when this kind of wf is not passed to their
    # transform step.
    which_waveforms = NotImplemented

    def fit(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
    ):
        """Some ExtraFeats don't fit, so useful to inherit this."""
        pass

    def to_h5(self, h5):
        """
        If this model computes things during fit, save them here.
        They will be used to reconstruct the object in from_h5.
        Please create a group in the h5 if you can.
        """
        pass

    def from_h5(self, h5):
        pass

    def transform(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
    ):
        """ExtraFeats should implement this."""
        raise NotImplementedError

    def handle_which_wfs(self, subtracted_wfs, cleaned_wfs, denoised_wfs):
        if self.which_waveforms == "subtracted":
            wfs = subtracted_wfs
        elif self.which_waveforms == "cleaned":
            wfs = cleaned_wfs
        elif self.which_waveforms == "denoised":
            wfs = denoised_wfs
        return wfs


# -- a couple of very basic extra features


class MaxPTP(ExtraFeat):

    name = "maxptps"
    # scalar
    out_shape = ()

    def __init__(self, which_waveforms="denoised"):
        self.which_waveforms = which_waveforms

    def transform(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
    ):
        wfs = self.handle_which_wfs(subtracted_wfs, cleaned_wfs, denoised_wfs)
        if wfs is None:
            return None
        maxptps = np.nanmax(wfs.ptp(1), axis=1)
        return maxptps


class TroughDepth(ExtraFeat):

    name = "trough_depths"
    # scalar
    out_shape = ()

    def __init__(self, which_waveforms="denoised"):
        self.which_waveforms = which_waveforms

    def transform(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
    ):
        wfs = self.handle_which_wfs(subtracted_wfs, cleaned_wfs, denoised_wfs)
        if wfs is None:
            return None
        mcs = np.nanargmax(wfs.ptp(1), axis=1)
        maxchan_traces = wfs[np.arange(len(wfs)), :, mcs]
        trough_depths = maxchan_traces.min(1)
        return trough_depths


class PeakHeight(ExtraFeat):

    name = "peak_heights"
    # scalar
    out_shape = ()

    def __init__(self, which_waveforms="denoised"):
        self.which_waveforms = which_waveforms

    def transform(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
    ):
        wfs = self.handle_which_wfs(subtracted_wfs, cleaned_wfs, denoised_wfs)
        if wfs is None:
            return None
        mcs = np.nanargmax(wfs.ptp(1), axis=1)
        maxchan_traces = wfs[np.arange(len(wfs)), :, mcs]
        peak_heights = maxchan_traces.max(1)
        return peak_heights


class Waveform(ExtraFeat):
    def __init__(self, which_waveforms):
        super().__init__()
        assert which_waveforms in ("subtracted", "cleaned", "denoised")
        self.which_waveforms = which_waveforms
        self.name = f"{which_waveforms}_waveforms"

    def fit(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
    ):
        wfs = self.handle_which_wfs(subtracted_wfs, cleaned_wfs, denoised_wfs)
        if wfs is None:
            return None

        N, T, C = wfs.shape
        self.out_shape = (T, C)
        self.dtype = wfs.dtype
        self.needs_fit = False

    def to_h5(self, h5):
        group = h5.create_group(f"{self.which_waveforms}_waveforms_info")
        group.create_dataset("T", data=self.out_shape[0])
        group.create_dataset("C", data=self.out_shape[1])

    def from_h5(self, h5):
        try:
            group = h5[f"{self.which_waveforms}_waveforms_info"]
            self.out_shape = (group["T"][()], group["C"][()])
            self.needs_fit = False
        except KeyError:
            pass

    def transform(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
    ):
        return self.handle_which_wfs(subtracted_wfs, cleaned_wfs, denoised_wfs)


# -- localization


class Localization(ExtraFeat):

    name = "localizations"
    out_shape = (5,)

    def __init__(
        self,
        geom,
        channel_index,
        loc_n_chans=None,
        loc_radius=None,
        n_workers=1,
        localization_kind="logbarrier",
    ):
        super().__init__()
        assert channel_index.shape[0] == geom.shape[0]
        self.geom = geom
        self.channel_index = channel_index
        self.localization_kind = localization_kind
        self.loc_n_chans = loc_n_chans
        self.loc_radius = loc_radius
        self.n_workers = n_workers

    def transform(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
    ):
        wfs = self.handle_which_wfs(subtracted_wfs, cleaned_wfs, denoised_wfs)
        if wfs is None:
            return None
        ptps = wfs.ptp(1)
        xs, ys, z_rels, z_abss, alphas = localize_index.localize_ptps_index(
            ptps,
            self.geom,
            max_channels,
            self.channel_index,
            n_channels=self.loc_n_chans,
            radius=self.loc_radius,
            n_workers=self.n_workers,
            pbar=False,
            logbarrier=self.localization_kind == "logbarrier",
        )
        # NOTE the reordering, same as it used to be...
        return np.c_[xs, ys, z_abss, alphas, z_rels]


# -- a more involved example


class TPCA(ExtraFeat):
    def __init__(self, rank, channel_index, which_waveforms, random_state=0):
        super().__init__()
        assert which_waveforms in ("subtracted", "cleaned", "denoised")
        self.which_waveforms = which_waveforms
        self.rank = rank
        self.name = f"{which_waveforms}_tpca_projs"
        self.needs_fit = True
        self.channel_index = channel_index
        self.random_state = random_state
        self.C = channel_index.shape[1]
        self.out_shape = (self.rank, self.C)
        self.tpca = PCA(self.rank, random_state=self.random_state)

    def fit(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
    ):
        wfs = self.handle_which_wfs(subtracted_wfs, cleaned_wfs, denoised_wfs)
        assert wfs is not None
        N, T, C = wfs.shape
        self.T = T

        wfs_in_probe = wfs.transpose(0, 2, 1)
        in_probe_index = self.channel_index < self.channel_index.shape[0]
        wfs_in_probe = wfs_in_probe[in_probe_index[max_channels]]

        self.tpca.fit(wfs_in_probe)
        self.needs_fit = False
        self.dtype = self.tpca.components_.dtype

    def to_h5(self, h5):
        group = h5.create_group(f"{self.which_waveforms}_tpca")
        group.create_dataset("T", data=self.T)
        group.create_dataset("tpca_mean", data=self.tpca.mean_)
        group.create_dataset("tpca_components", data=self.tpca.components_)

    def from_h5(self, h5):
        try:
            group = h5[f"{self.which_waveforms}_tpca"]
            self.T = group["T"][()]
            self.tpca = PCA(self.rank)
            self.tpca.mean_ = group["tpca_mean"][:]
            self.tpca.components_ = group["tpca_components"][:]
            self.needs_fit = False
        except KeyError:
            pass

    def transform(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
    ):
        wfs = self.handle_which_wfs(subtracted_wfs, cleaned_wfs, denoised_wfs)
        if wfs is None:
            return None

        features = np.full(
            (wfs.shape[0], *self.out_shape), np.nan, dtype=self.dtype
        )
        features = features.transpose(0, 2, 1)

        wfs_in_probe = wfs.transpose(0, 2, 1)
        in_probe_index = self.channel_index < self.channel_index.shape[0]
        chans_in_probe = in_probe_index[max_channels]
        wfs_in_probe = wfs_in_probe[chans_in_probe]

        features[chans_in_probe] = self.tpca.transform(wfs_in_probe)
        features = features.transpose(0, 2, 1)
        return features

    def denoise(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
    ):
        wfs = self.handle_which_wfs(subtracted_wfs, cleaned_wfs, denoised_wfs)
        if wfs is None:
            return None
        out = wfs.copy()

        out = out.transpose(0, 2, 1)
        in_probe_index = self.channel_index < self.channel_index.shape[0]
        chans_in_probe = in_probe_index[max_channels]
        wfs_in_probe = out[chans_in_probe]

        out[chans_in_probe] = self.tpca.inverse_transform(
            self.tpca.transform(wfs_in_probe)
        )
        out = out.transpose(0, 2, 1)
        return out
