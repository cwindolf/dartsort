import numpy as np
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from spike_psvae import localize_index, localize_torch, waveform_utils


# %%
class ChunkFeature:
    """Feature computers for chunk pipelines (subtract and extract_deconv)

    Subclasses ompute features from subtracted/cleaned/denoised waveforms,
    fit featurizers, save and load things from hdf5, ...
    """

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
    tensor_ok = False

    def fit(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
        geom=None,
        training_radius=None,
    ):
        """Some ChunkFeatures don't fit, so useful to inherit this."""
        pass

    def to_h5(self, h5):
        """
        If this model computes things during fit, save them here.
        They will be used to reconstruct the object in from_h5.
        Please create a group in the h5 if you can.
        """
        pass

    def to(self, device):
        return self

    def from_h5(self, h5):
        pass

    def transform(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
    ):
        """ChunkFeatures should implement this."""
        raise NotImplementedError

    def handle_which_wfs(self, subtracted_wfs, cleaned_wfs, denoised_wfs):
        if self.which_waveforms == "subtracted":
            wfs = subtracted_wfs
        elif self.which_waveforms == "cleaned":
            wfs = cleaned_wfs
        elif self.which_waveforms == "denoised":
            wfs = denoised_wfs
        else:
            raise ValueError(
                f"which_waveforms={self.which_waveforms} not in ('subtracted', 'cleaned', denoised)"
            )
        if not self.tensor_ok and torch.is_tensor(wfs):
            wfs = wfs.cpu().numpy()
        return wfs


# %% [markdown]
# -- a couple of very basic extra features


# %%
class MaxPTP(ChunkFeature):
    name = "maxptps"
    # scalar
    out_shape = ()
    tensor_ok = True

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
        if torch.is_tensor(wfs):
            ptps = wfs.max(1).values - wfs.min(1).values
            ptps[torch.isnan(ptps)] = -1
            maxptps = ptps.max(1).values
        else:
            maxptps = np.nanmax(wfs.ptp(1), axis=1)
        return maxptps


class TroughDepth(ChunkFeature):
    name = "trough_depths"
    # scalar
    out_shape = ()
    tensor_ok = True

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
        if torch.is_tensor(wfs):
            ptps = wfs.max(dim=1).values - wfs.min(dim=1).values
            ptps[torch.isnan(ptps)] = -1
            mcs = torch.argmax(ptps, dim=1)
            maxchan_traces = wfs[torch.arange(len(wfs)), :, mcs]
            trough_depths = maxchan_traces.min(1).values
        else:
            ptps = wfs.ptp(1)
            maxchan_traces = wfs[np.arange(len(wfs)), :, mcs]
            trough_depths = maxchan_traces.min(1)
        return trough_depths


class PeakHeight(ChunkFeature):
    name = "peak_heights"
    # scalar
    out_shape = ()
    tensor_ok = True

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
        if torch.is_tensor(wfs):
            ptps = wfs.max(dim=1).values - wfs.min(dim=1).values
            ptps[torch.isnan(ptps)] = -1
            mcs = torch.argmax(ptps, dim=1)
            maxchan_traces = wfs[torch.arange(len(wfs)), :, mcs]
            peak_heights = maxchan_traces.max(1).values
        else:
            mcs = np.nanargmax(wfs.ptp(1), axis=1)
            maxchan_traces = wfs[np.arange(len(wfs)), :, mcs]
            peak_heights = maxchan_traces.max(1)
        return peak_heights


class PTPVector(ChunkFeature):
    name = "ptp_vectors"
    needs_fit = True

    def __init__(
        self, which_waveforms="denoised", channel_index=None, dtype=np.float32
    ):
        self.which_waveforms = which_waveforms
        if channel_index is not None:
            self.C = channel_index.shape[1]
            self.dtype = dtype
            self.needs_fit = False

    def fit(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
        geom=None,
        training_radius=None,
    ):
        wfs = self.handle_which_wfs(subtracted_wfs, cleaned_wfs, denoised_wfs)
        if wfs is None:
            return None

        N, T, C = wfs.shape
        self.out_shape = (C,)
        self.dtype = wfs.dtype
        self.needs_fit = False

    def to_h5(self, h5):
        if f"{self.which_waveforms}_ptpvector_info" in h5:
            return
        group = h5.create_group(f"{self.which_waveforms}_ptpvector_info")
        group.create_dataset("C", data=self.out_shape[0])

    def from_h5(self, h5):
        try:
            group = h5[f"{self.which_waveforms}_ptpvector_info"]
            self.out_shape = (group["C"][()],)
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
        return wfs.ptp(1)


class Waveform(ChunkFeature):
    needs_fit = True

    def __init__(
        self,
        which_waveforms,
        spike_length_samples=None,
        channel_index=None,
        dtype=np.float32,
    ):
        super().__init__()
        assert which_waveforms in ("subtracted", "cleaned", "denoised")
        self.which_waveforms = which_waveforms
        self.name = f"{which_waveforms}_waveforms"
        if channel_index is not None and spike_length_samples is not None:
            self.out_shape = (spike_length_samples, channel_index.shape[1])
            self.dtype = dtype
            self.needs_fit = False

    def fit(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
        geom=None,
        training_radius=None,
    ):
        wfs = self.handle_which_wfs(subtracted_wfs, cleaned_wfs, denoised_wfs)
        if wfs is None:
            return None

        N, T, C = wfs.shape
        self.out_shape = (T, C)
        self.dtype = wfs.dtype
        self.needs_fit = False

    def to_h5(self, h5):
        if f"{self.which_waveforms}_waveforms_info" in h5:
            return
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


class Localization(ChunkFeature):
    out_shape = (5,)
    tensor_ok = True

    def __init__(
        self,
        geom,
        channel_index,
        loc_n_chans=None,
        loc_radius=None,
        n_workers=1,
        localization_model="pointsource",
        localization_kind="logbarrier",
        which_waveforms="denoised",
        feature="ptp",
        name_extra="",
        ptp_precision_decimals=None,
    ):
        super().__init__()
        assert channel_index.shape[0] == geom.shape[0]
        self.geom = geom
        self.channel_index = channel_index
        self.localization_kind = localization_kind
        self.localizaton_model = localization_model
        self.loc_n_chans = loc_n_chans
        self.loc_radius = loc_radius
        self.n_workers = n_workers
        self.which_waveforms = which_waveforms
        self.feature = feature
        if not name_extra and feature != "ptp":
            name_extra = feature
        self.name = f"localizations{name_extra}"
        self.dogpu = "gpu" in feature
        self.opt = "lbfgs"
        self.ptp_precision_decimals = ptp_precision_decimals
        if "adam" in feature:
            self.opt = "adam"

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

        if "ptp" in self.feature:
            if torch.is_tensor(wfs):
                ptps = wfs.max(dim=1).values - wfs.min(dim=1).values
                if not self.dogpu:
                    ptps = ptps.cpu().numpy()
            else:
                ptps = wfs.ptp(1)
        elif "peak" in self.feature:
            if torch.is_tensor(wfs):
                abswfs = torch.abs(wfs)
                peaks, argpeaks = abswfs.max(dim=1)
                peaks[torch.isnan(peaks)] = -1
                mcs = torch.argmax(peaks, dim=1)
                argpeaks = argpeaks[torch.arange(len(argpeaks)), mcs]
                ptps = abswfs[torch.arange(len(mcs)), argpeaks, :]
                if not self.dogpu:
                    ptps = ptps.cpu().numpy()
            else:
                peaks = np.max(np.absolute(wfs), axis=1)
                argpeaks = np.argmax(np.absolute(wfs), axis=1)
                mcs = np.nanargmax(peaks, axis=1)
                argpeaks = argpeaks[np.arange(len(argpeaks)), mcs]
                ptps = wfs[np.arange(len(mcs)), argpeaks, :]
                ptps = np.abs(ptps)
        else:
            raise NameError("Use ptp or peak value for localization.")

        if torch.is_tensor(ptps):
            if self.ptp_precision_decimals is not None:
                ptps = torch.round(ptps, decimals=self.ptp_precision_decimals)
            x, y, z_rel, z_abs, alpha = localize_torch.localize_ptps_index_lm(
                ptps,
                self.geom,
                max_channels,
                self.channel_index,
                n_channels=self.loc_n_chans,
                radius=self.loc_radius,
                logbarrier=self.localization_kind == "logbarrier",
            )
            return torch.column_stack((x, y, z_abs, alpha, z_rel))
        else:
            if self.ptp_precision_decimals is not None:
                ptps = np.round(ptps, decimals=self.ptp_precision_decimals)
            (
                xs,
                ys,
                z_rels,
                z_abss,
                alphas,
            ) = localize_index.localize_ptps_index(
                ptps,
                self.geom,
                max_channels,
                self.channel_index,
                n_channels=self.loc_n_chans,
                radius=self.loc_radius,
                n_workers=self.n_workers,
                pbar=False,
                logbarrier=self.localization_kind == "logbarrier",
                model=self.localizaton_model,
            )
            return np.c_[xs, ys, z_abss, alphas, z_rels]


# -- a more involved example


class TPCA(ChunkFeature):
    needs_fit = True
    tensor_ok = True

    def __init__(
        self,
        rank,
        channel_index,
        which_waveforms,
        centered=True,
        random_state=0,
    ):
        super().__init__()
        assert which_waveforms in ("subtracted", "cleaned", "denoised")
        self.which_waveforms = which_waveforms
        self.rank = rank
        self.name = f"{which_waveforms}_tpca_projs"
        self.channel_index = channel_index
        self.C = channel_index.shape[1]
        self.out_shape = (self.rank, self.C)
        self.centered = centered
        self.random_state = random_state
        self.tpca = PCA(self.rank, random_state=random_state, copy=False)

    @classmethod
    def load_from_h5(cls, h5, which_waveforms, random_state=0):
        group = h5[f"{which_waveforms}_tpca"]
        T = group["T"][()]
        mean_ = group["tpca_mean"][:]
        components_ = group["tpca_components"][:]
        rank = components_.shape[0]
        channel_index = h5["channel_index"][:]

        self = cls(
            rank, channel_index, which_waveforms, random_state=random_state
        )

        self.T = T
        self.tpca.mean_ = mean_
        self.tpca.components_ = components_
        self.dtype = components_.dtype
        self.needs_fit = False

        return self

    def to(self, device):
        self.mean_ = torch.as_tensor(self.tpca.mean_, device=device)
        self.components_ = torch.as_tensor(
            self.tpca.components_, device=device
        )
        self.whiten = self.tpca.whiten
        self.whitener = torch.as_tensor(self.whitener, device=device)
        self.channel_index = torch.as_tensor(self.channel_index, device=device)
        return self

    def raw_fit(self, wfs, max_channels):
        # For fitting a tpca object with given wfs and max chans
        N, T, C = wfs.shape
        wfs = wfs.transpose(0, 2, 1)
        in_probe_index = self.channel_index < self.channel_index.shape[0]
        wfs = wfs[in_probe_index[max_channels]]

        if self.centered:
            self.tpca.fit(wfs)
        else:
            tsvd = TruncatedSVD(self.rank, random_state=self.random_state).fit(wfs)
            self.tpca.mean_ = np.zeros_like(wfs[0])
            self.tpca.components_ = tsvd.components_

        self.needs_fit = False
        self.dtype = self.tpca.components_.dtype
        self.n_components = self.tpca.n_components
        self.components_ = self.tpca.components_
        self.mean_ = self.tpca.mean_
        if self.centered:  # otherwise SVD
            self.whiten = self.tpca.whiten
            self.whitener = np.sqrt(self.tpca.explained_variance_)

    def raw_transform(self, X):
        X = X - self.mean_
        Xt = X @ self.components_.T
        if self.centered:
            if self.whiten:
                Xt /= self.whitener
        return Xt

    def raw_inverse_transform(self, X):
        if self.centered and self.whiten:
            return (X @ (self.whitener * self.components_)) + self.mean_
        else:
            return (X @ self.components_) + self.mean_

    def to_h5(self, h5):
        if f"{self.which_waveforms}_tpca" in h5:
            return
        group = h5.create_group(f"{self.which_waveforms}_tpca")
        group.create_dataset("T", data=self.T)
        group.create_dataset("tpca_mean", data=self.tpca.mean_)
        group.create_dataset("tpca_components", data=self.tpca.components_)
        group.create_dataset("tpca_centered", data=self.centered)
        if self.centered:
            group.create_dataset("tpca_whiten", data=self.whiten)
            group.create_dataset("tpca_whitener", data=self.whitener)

    def from_h5(self, h5, random_state=0):
        try:
            group = h5[f"{self.which_waveforms}_tpca"]
            self.T = group["T"][()]
            self.tpca = PCA(self.rank, random_state=random_state)
            self.tpca.mean_ = group["tpca_mean"][:]
            self.tpca.components_ = group["tpca_components"][:]
            self.n_components = self.tpca.n_components
            self.centered = group["tpca_centered"][()]
            if self.centered:
                self.whiten = group["tpca_whiten"][()]
                self.whitener = group["tpca_whitener"][()]
            self.needs_fit = False
        except KeyError:
            print("Failed to load", f"{self.which_waveforms}_tpca")
            pass

    def from_sklearn(self, sklearn_pca):
        self.T = sklearn_pca.components_.shape[1]
        self.tpca = sklearn_pca
        self.dtype = sklearn_pca.components_.dtype
        self.needs_fit = False
        self.components_ = sklearn_pca.components_
        self.n_components = sklearn_pca.n_components
        self.mean_ = sklearn_pca.mean_
        self.whiten = sklearn_pca.whiten
        self.whitener = np.sqrt(sklearn_pca.explained_variance_)
        return self

    def __str__(self):
        if self.needs_fit:
            return f"TPCA(needs_fit=True, C={self.C}, PCA={self.tpca})"
        else:
            return f"TPCA(needs_fit=False, T={self.T}, C={self.C}, PCA={self.tpca})"

    def fit(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
        geom=None,
        training_radius=None,
    ):
        wfs = self.handle_which_wfs(subtracted_wfs, cleaned_wfs, denoised_wfs)
        if wfs is None:
            return

        if torch.is_tensor(wfs):
            wfs = wfs.cpu().numpy()

        N, T, C = wfs.shape
        self.T = T

        wfs_in_probe = wfs.transpose(0, 2, 1)
        if geom is None:
            in_probe_index = self.channel_index < self.channel_index.shape[0]
            wfs_in_probe = wfs_in_probe[in_probe_index[max_channels]]
        else:
            distances_channels = np.sqrt(
                (geom[:, None] - geom[None, :]) ** 2
            ).sum(2)
            distances_channels = np.pad(
                distances_channels,
                ((0, 0), (0, 1)),
                mode="constant",
                constant_values=training_radius * 2,
            )
            distances_channels = distances_channels[
                np.repeat(np.arange(geom.shape[0]), C),
                self.channel_index.flatten(),
            ].reshape((geom.shape[0], C))
            in_probe_index = distances_channels < training_radius
            wfs_in_probe = wfs_in_probe[in_probe_index[max_channels]]
            del distances_channels

        if self.centered:
            self.tpca.fit(wfs_in_probe)
        else:
            tsvd = TruncatedSVD(self.rank).fit(wfs_in_probe)
            self.tpca.mean_ = np.zeros_like(wfs_in_probe[0])
            self.tpca.components_ = tsvd.components_

        self.needs_fit = False
        self.dtype = self.tpca.components_.dtype
        self.n_components = self.tpca.n_components

        self.components_ = self.tpca.components_
        self.mean_ = self.tpca.mean_
        if self.centered:
            self.whiten = self.tpca.whiten
            self.whitener = np.sqrt(self.tpca.explained_variance_)

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

        if torch.is_tensor(wfs):
            wfs_in_probe = wfs.permute(0, 2, 1)
            features = torch.full(
                (wfs.shape[0], *self.out_shape),
                torch.nan,
                dtype=wfs.dtype,
                device=wfs.device,
            )
            features_ = features.permute(0, 2, 1)
        else:
            wfs_in_probe = wfs.transpose(0, 2, 1)
            features = np.full(
                (wfs.shape[0], *self.out_shape), np.nan, dtype=wfs.dtype
            )
            features_ = features.transpose(0, 2, 1)

        in_probe_index = self.channel_index < self.channel_index.shape[0]
        chans_in_probe = in_probe_index[max_channels]
        wfs_in_probe = wfs_in_probe[chans_in_probe]
        features_[chans_in_probe] = self.raw_transform(wfs_in_probe)

        return features

    def inverse_transform(self, features, max_channels, channel_index):
        in_probe_index = self.channel_index < self.channel_index.shape[0]
        chans_in_probe = in_probe_index[max_channels]
        if torch.is_tensor(features):
            wfs = torch.full(
                (features.shape[0], self.C, self.T),
                torch.nan,
                dtype=self.dtype,
            )
            wfs[chans_in_probe] = self.raw_inverse_transform(
                features.permute(0, 2, 1)[chans_in_probe]
            )
            wfs = wfs.permute(0, 2, 1)
        else:
            wfs = np.full(
                (features.shape[0], self.C, self.T), np.nan, dtype=self.dtype
            )
            wfs[chans_in_probe] = self.tpca.inverse_transform(
                features.transpose(0, 2, 1)[chans_in_probe]
            )
            wfs = wfs.transpose(0, 2, 1)
        return wfs

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


class STPCA(ChunkFeature):
    needs_fit = True
    tensor_ok = True

    def __init__(
        self,
        rank,
        channel_index,
        geom,
        n_channels,
        which_waveforms,
        random_state=0,
    ):
        """
        Fit a PCA to waveforms extracted on the n_channels closest
        channels to the detection channel.
        """
        super().__init__()
        assert which_waveforms in ("subtracted", "cleaned", "denoised")
        self.which_waveforms = which_waveforms
        self.rank = rank
        self.name = f"{which_waveforms}_pca_projs"
        self.channel_index = channel_index
        self.C = channel_index.shape[1]
        self.out_shape = (self.rank,)
        self.pca = PCA(self.rank, random_state=random_state)
        self.sub_channel_index = waveform_utils.closest_chans_channel_index(
            geom, n_channels
        )

    @classmethod
    def load_from_h5(cls, h5, which_waveforms, random_state=0):
        group = h5[f"{which_waveforms}_pca"]
        T = group["T"][()]
        mean_ = group["pca_mean"][:]
        components_ = group["pca_components"][:]
        rank = components_.shape[0]
        channel_index = h5["channel_index"][:]

        self = cls(
            rank, channel_index, which_waveforms, random_state=random_state
        )

        self.T = T
        self.pca.mean_ = mean_
        self.pca.components_ = components_
        self.dtype = components_.dtype
        self.needs_fit = False

        return self

    def to(self, device):
        self.mean_ = torch.as_tensor(self.pca.mean_, device=device)
        self.components_ = torch.as_tensor(self.pca.components_, device=device)
        self.whiten = self.pca.whiten
        if self.whiten:
            self.whitener = torch.as_tensor(self.whitener, device=device)
        return self

    def raw_transform(self, X):
        X = X - self.mean_
        Xt = X @ self.components_.T
        if self.whiten:
            Xt /= self.whitener
        return Xt

    def to_h5(self, h5):
        if f"{self.which_waveforms}_pca" in h5:
            return
        group = h5.create_group(f"{self.which_waveforms}_pca")
        group.create_dataset("T", data=self.T)
        group.create_dataset("pca_mean", data=self.pca.mean_)
        group.create_dataset("pca_components", data=self.pca.components_)
        if self.whiten:
            group.create_dataset("pca_whitener", data=self.pca.components_)

    def from_h5(self, h5):
        try:
            group = h5[f"{self.which_waveforms}_pca"]
            self.T = group["T"][()]
            self.pca = PCA(self.rank)
            self.pca.mean_ = group["pca_mean"][:]
            self.pca.components_ = group["pca_components"][:]
            self.needs_fit = False
        except KeyError:
            pass

    def from_sklearn(self, sklearn_pca):
        self.T = sklearn_pca.components_.shape[1]
        self.pca = sklearn_pca
        self.dtype = sklearn_pca.components_.dtype
        self.needs_fit = False
        self.components_ = sklearn_pca.components_
        self.n_components = sklearn_pca.n_components
        self.mean_ = sklearn_pca.mean_
        self.whiten = sklearn_pca.whiten
        self.whitener = np.sqrt(sklearn_pca.explained_variance_)

        return self

    def fit(
        self,
        max_channels=None,
        subtracted_wfs=None,
        cleaned_wfs=None,
        denoised_wfs=None,
        geom=None,
        training_radius=None,
    ):
        wfs = self.handle_which_wfs(subtracted_wfs, cleaned_wfs, denoised_wfs)
        if wfs is None:
            return

        if torch.is_tensor(wfs):
            wfs = wfs.cpu().numpy()

        N, T, C = wfs.shape
        self.T = T

        sub_wfs = waveform_utils.channel_subset_by_index(
            wfs,
            max_channels,
            self.channel_index,
            self.sub_channel_index,
            fill_value=np.nan,
        )
        assert not np.isnan(sub_wfs).any()
        assert sub_wfs.shape == (N, T, self.sub_channel_index.shape[1])

        self.pca.fit(sub_wfs.reshape(N, -1))
        self.needs_fit = False
        self.dtype = self.pca.components_.dtype
        self.n_components = self.pca.n_components

        self.components_ = self.pca.components_
        self.mean_ = self.pca.mean_
        self.whiten = self.pca.whiten
        self.whitener = np.sqrt(self.pca.explained_variance_)

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

        sub_wfs = waveform_utils.channel_subset_by_index(
            wfs,
            max_channels,
            self.channel_index,
            self.sub_channel_index,
            fill_value=np.nan,
        )

        return self.raw_transform(sub_wfs.reshape(len(wfs), -1))
