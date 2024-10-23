import numpy as np
import torch
from dartsort.util.waveform_util import (channel_subset_by_radius,
                                         get_channels_in_probe,
                                         set_channels_in_probe)
from sklearn.decomposition import PCA, TruncatedSVD

from .transform_base import (BaseWaveformAutoencoder, BaseWaveformDenoiser,
                             BaseWaveformFeaturizer, BaseWaveformModule)


class BaseTemporalPCA(BaseWaveformModule):
    def __init__(
        self,
        channel_index,
        geom=None,
        rank=8,
        whiten=False,
        centered=True,
        fit_radius=None,
        random_state=0,
        name=None,
        name_prefix="",
        temporal_slice=None,
        n_oversamples=10,
    ):
        if fit_radius is not None:
            if geom is None or channel_index is None:
                raise ValueError(
                    "TemporalPCA with fit_radius!=None requires geom."
                )
        super().__init__(
            channel_index=channel_index,
            geom=geom,
            name=name,
            name_prefix=name_prefix,
        )
        self.rank = rank
        self._needs_fit = True
        self.random_state = random_state
        self.shape = (rank, channel_index.shape[1])
        self.fit_radius = fit_radius
        self.centered = centered
        self.whiten = whiten
        self.temporal_slice = temporal_slice
        self.n_oversamples = n_oversamples

    def fit(self, waveforms, max_channels, recording=None):
        waveforms = self._temporal_slice(waveforms)
        self.dtype = waveforms.dtype
        train_channel_index = self.channel_index
        if self.fit_radius is not None:
            waveforms, train_channel_index = channel_subset_by_radius(
                waveforms,
                max_channels,
                self.channel_index,
                self.geom,
                self.fit_radius,
            )
        _, waveforms_fit = get_channels_in_probe(
            waveforms, max_channels, train_channel_index
        )

        if self.centered:
            mean = waveforms_fit.mean(0)
        else:
            mean = torch.zeros_like(waveforms_fit[0])

        q = self.rank + self.n_oversamples
        n_samples, n_times = waveforms_fit.shape
        assert q < min(n_samples, n_times)
        M = None
        if self.centered:
            # torch does not seem always to want to broadcast M as advertised?
            M = mean[None].broadcast_to(waveforms_fit.shape)

        # 7 is based on sklearn's auto choice
        U, S, V = torch.svd_lowrank(waveforms_fit, q=q, M=M, niter=7)
        U = U[..., :self.rank]
        S = S[..., :self.rank]
        V = V[..., :self.rank]

        # loadings = U * S[..., None, :]
        components = V.T.contiguous()
        explained_variance = (S**2) / (n_samples - 1)
        whitener = torch.sqrt(explained_variance)

        self.register_buffer("mean", mean)
        self.register_buffer("components", components)
        self.register_buffer("whitener", whitener)
        self._needs_fit = False

    def needs_fit(self):
        return self._needs_fit

    def _temporal_slice(self, waveforms):
        if self.temporal_slice is None:
            return waveforms

        return waveforms[:, self.temporal_slice]

    def _transform_in_probe(self, waveforms_in_probe):
        x = waveforms_in_probe
        if self.centered:
            x = x - self.mean
        W = self.components
        if self.whiten:
            W = self.components / self.whitener
        x = x @ W.T
        return x

    def _inverse_transform_in_probe(self, features):
        W = self.components
        if self.whiten:
            W = W * self.whitener
        if self.centered:
            return torch.addmm(self.mean, features, W)
        return torch.mm(features, W)

    def _project_in_probe(self, waveforms_in_probe):
        if self.centered:
            return torch.addmm(
                self.mean,
                waveforms_in_probe - self.mean,
                self.components.T @ self.components,
            )
        return torch.mm(
            waveforms_in_probe,
            self.components.T @ self.components,
        )

    def force_reconstruct(self, features):
        ndim = features.ndim
        if ndim == 2:
            features = features.unsqueeze(0)
        n, r, c = features.shape
        waveforms = features.permute(0, 2, 1).reshape(n * c, r)
        waveforms = self._inverse_transform_in_probe(waveforms)
        waveforms = waveforms.reshape(n, c, -1).permute(0, 2, 1)
        if ndim == 2:
            waveforms = waveforms[0]
        return waveforms

    def to_sklearn(self):
        pca = PCA(
            self.rank,
            random_state=self.random_state,
            whiten=self.whiten,
        )
        pca.mean_ = self.mean.numpy(force=True)
        pca.components_ = self.components.numpy(force=True)
        pca.explained_variance_ = np.square(self.whitener.numpy(force=True))
        pca.temporal_slice = self.temporal_slice  # this is not standard
        return pca


class TemporalPCADenoiser(BaseWaveformDenoiser, BaseTemporalPCA):
    default_name = "temporal_pca"

    def forward(self, waveforms, max_channels):
        waveforms = self._temporal_slice(waveforms)
        (
            channels_in_probe,
            waveforms_in_probe,
        ) = get_channels_in_probe(waveforms, max_channels, self.channel_index)
        waveforms_in_probe = self._project_in_probe(waveforms_in_probe)
        return set_channels_in_probe(
            waveforms_in_probe,
            waveforms,
            channels_in_probe,
            in_place=True,
        )


class TemporalPCAFeaturizer(BaseWaveformFeaturizer, BaseTemporalPCA):
    default_name = "tpca_features"

    def transform(self, waveforms, max_channels, channel_index=None, return_in_probe=False):
        waveforms = self._temporal_slice(waveforms)
        if channel_index is None:
            channel_index = self.channel_index
        (
            channels_in_probe,
            waveforms_in_probe,
        ) = get_channels_in_probe(waveforms, max_channels, channel_index)
        features_in_probe = self._transform_in_probe(waveforms_in_probe)
        features = torch.full(
            (waveforms.shape[0], self.rank, channel_index.shape[1]),
            torch.nan,
            dtype=features_in_probe.dtype,
            device=features_in_probe.device,
        )
        features = set_channels_in_probe(
            features_in_probe,
            features,
            channels_in_probe,
        )
        if return_in_probe:
            return features_in_probe, channels_in_probe, {self.name: features}

        return {self.name: features}

    def inverse_transform(self, features, max_channels, channel_index=None):
        if channel_index is None:
            channel_index = self.channel_index
        (
            channels_in_probe,
            features_in_probe,
        ) = get_channels_in_probe(features, max_channels, channel_index)
        reconstructions_in_probe = self._inverse_transform_in_probe(
            features_in_probe
        )
        reconstructions = torch.full(
            (
                features.shape[0],
                self.components.shape[1],
                channel_index.shape[1],
            ),
            torch.nan,
            dtype=reconstructions_in_probe.dtype,
            device=reconstructions_in_probe.device,
        )
        return set_channels_in_probe(
            reconstructions_in_probe,
            reconstructions,
            channels_in_probe,
        )


class TemporalPCA(BaseWaveformAutoencoder, TemporalPCAFeaturizer):

    def forward(self, waveforms, max_channels):
        waveforms = self._temporal_slice(waveforms)
        features_in_probe, channels_in_probe, features = self.transform(
            waveforms, max_channels, return_in_probe=True,
        )
        reconstructions_in_probe = self._inverse_transform_in_probe(
            features_in_probe
        )
        reconstructions = set_channels_in_probe(
            reconstructions_in_probe,
            waveforms,
            channels_in_probe,
        )
        return reconstructions, features


# could also have one which is both by subclassing both of the above,
# but we don't use features from PCAs which act as denoisers right now
