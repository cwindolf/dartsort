import numpy as np
import torch
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils.extmath import svd_flip

from ..util.waveform_util import (
    channel_subset_by_radius,
    get_channels_in_probe,
    set_channels_in_probe,
)
from .transform_base import (
    BaseWaveformAutoencoder,
    BaseWaveformDenoiser,
    BaseWaveformFeaturizer,
    BaseWaveformModule,
)


class BaseTemporalPCA(BaseWaveformModule):
    default_name = "basis"

    def __init__(
        self,
        channel_index,
        geom=None,
        rank=8,
        whiten=False,
        centered=False,
        fit_radius=None,
        random_state=0,
        name=None,
        name_prefix="",
        temporal_slice: slice | None = None,
        n_oversamples=10,
        niter=21,
        max_waveforms=20_000,
        fit_dtype=torch.double,
    ):
        if fit_radius is not None:
            if geom is None or channel_index is None:
                raise ValueError("TemporalPCA with fit_radius!=None requires geom.")

        super().__init__(
            channel_index=channel_index, geom=geom, name=name, name_prefix=name_prefix
        )

        # behavior
        self.rank = rank
        self.centered = centered
        self.whiten = whiten
        self.temporal_slice = temporal_slice

        # fit control
        self.fit_radius = fit_radius
        self.random_state = random_state
        self.n_oversamples = n_oversamples
        self.niter = niter
        self.fit_dtype = fit_dtype
        self.max_waveforms = max_waveforms

        # gizmo
        self._needs_fit = True
        self.shape = (rank, channel_index.shape[1])

    def initialize_spike_length_dependent_params(self):
        nt = self.spike_length_samples
        assert nt is not None
        if self.temporal_slice is None:
            self._temporal_ix = None
        else:
            self.register_buffer("_temporal_ix", torch.arange(nt)[self.temporal_slice])
            nt = self.b._temporal_ix.numel()
        self.register_buffer("mean", torch.zeros(nt))
        self.register_buffer("components", torch.zeros(self.rank, nt))
        self.register_buffer("whitener", torch.zeros(self.rank))
        self.to(self.b.channel_index.device)

    def fit(
        self,
        recording,
        waveforms,
        *,
        channels,
        weights=None,
        time_shifts=None,
        **unused,
    ):
        super().fit(recording, waveforms, channels=channels, weights=weights)
        if weights is not None and waveforms.shape[0] > self.max_waveforms:
            self.random_state = np.random.default_rng(self.random_state)
            weights = weights.numpy(force=True) if torch.is_tensor(weights) else weights
            weights = weights.astype(np.float64)
            weights = weights / weights.sum()
            choices = self.random_state.choice(
                len(weights), p=weights, size=self.max_waveforms
            )
            choices.sort()
            choices = torch.from_numpy(choices)
            waveforms = waveforms[choices]
            channels = channels[choices]
        waveforms = self._temporal_slice(waveforms, time_shifts=time_shifts)
        self.dtype = waveforms.dtype
        train_channel_index = self.b.channel_index
        if waveforms.device != train_channel_index.device:
            waveforms = waveforms.to(train_channel_index.device)
            channels = channels.to(train_channel_index.device)
        if self.fit_radius is not None:
            waveforms, train_channel_index = channel_subset_by_radius(
                waveforms,
                channels,
                self.channel_index,
                self.geom,
                self.fit_radius,
            )
        _, waveforms_fit = get_channels_in_probe(
            waveforms, channels, train_channel_index
        )

        if self.centered:
            mean = waveforms_fit.mean(0)
        else:
            mean = torch.zeros_like(waveforms_fit[0])

        n_samples, n_times = waveforms_fit.shape
        q = min(self.rank + self.n_oversamples, n_samples, n_times)
        M = None
        if self.centered:
            # torch does not seem always to want to broadcast M as advertised?
            M = mean[None].broadcast_to(waveforms_fit.shape)

        # niter=7 is sklearn's auto choice. but that's usually double...
        orig_dtype = waveforms_fit.dtype
        waveforms_fit = waveforms_fit.to(dtype=self.fit_dtype)
        U, S, V = torch.svd_lowrank(waveforms_fit, q=q, M=M, niter=self.niter)
        U = U[..., : self.rank]
        S = S[..., : self.rank]
        V = V[..., : self.rank]
        Vt = V.T

        # fix sign ambiguity for better reproducibility in unit tests
        U, Vt = svd_flip(U.numpy(force=True), Vt.numpy(force=True))
        U = torch.asarray(U, dtype=orig_dtype, device=S.device).contiguous()
        Vt = torch.asarray(Vt, dtype=orig_dtype, device=S.device).contiguous()

        # loadings = U * S[..., None, :]
        components = Vt
        explained_variance = S.square() / (n_samples - 1)
        whitener = torch.sqrt(explained_variance)

        self.b.mean.copy_(mean)
        self.b.components.copy_(components)
        self.b.whitener.copy_(whitener)
        self._needs_fit = False

    def needs_fit(self):
        return self._needs_fit

    def _temporal_slice(self, waveforms: torch.Tensor, time_shifts=None) -> torch.Tensor:
        if self.temporal_slice is None:
            return waveforms

        if time_shifts is None:
            return waveforms[:, self.temporal_slice]

        assert self._temporal_ix is not None
        n, t, c = waveforms.shape
        t_ = self._temporal_ix.numel()
        assert t_ <= t
        assert time_shifts.shape == (n,)
        time_ix = self._temporal_ix[None, :, None] + time_shifts[:, None, None]
        waveforms = waveforms.take_along_dim(dim=1, indices=time_ix)
        assert waveforms.shape == (n, t_, c)
        return waveforms

    def _transform_in_probe(self, waveforms_in_probe):
        x = waveforms_in_probe
        if self.centered:
            x = x - self.mean
        W = self.components
        if self.whiten:
            W = self.b.components / self.b.whitener
        x = x @ W.T
        return x

    def _inverse_transform_in_probe(self, features):
        W = self.b.components
        if self.whiten:
            W = W * self.b.whitener
        if self.centered:
            return torch.addmm(self.b.mean, features, W)
        return torch.mm(features, W)

    def _project_in_probe(self, waveforms_in_probe):
        if self.centered:
            return torch.addmm(
                self.b.mean,
                waveforms_in_probe - self.mean,
                self.b.components.T @ self.b.components,
            )
        return torch.mm(waveforms_in_probe, self.b.components.T @ self.b.components)

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

    def force_project(self, features):
        ndim = features.ndim
        if ndim == 2:
            features = features.unsqueeze(0)
        n, t, c = features.shape
        waveforms = features.mT.reshape(n * c, t)
        waveforms = self._project_in_probe(waveforms)
        waveforms = waveforms.reshape(n, c, t).mT
        if ndim == 2:
            waveforms = waveforms[0]
        return waveforms

    def to_sklearn(self) -> PCA:
        pca = PCA(
            n_components=self.rank,
            random_state=self.random_state,  # type: ignore
            whiten=self.whiten,
        )
        pca.mean_ = self.b.mean.numpy(force=True)
        pca.components_ = self.b.components.numpy(force=True)
        pca.explained_variance_ = np.square(self.b.whitener.numpy(force=True))
        pca.temporal_slice = self.temporal_slice  # this is not standard  # type: ignore
        return pca

    @classmethod
    def from_sklearn(cls, channel_index, pca: PCA | TruncatedSVD, temporal_slice=None):
        if isinstance(pca, PCA):
            whiten = pca.whiten  # type: ignore
        elif isinstance(pca, TruncatedSVD):
            whiten = False
        else:
            assert False
        self = cls(
            channel_index,
            rank=pca.n_components,  # type: ignore
            whiten=whiten,
            temporal_slice=temporal_slice,
        )
        self.initialize_from_sklearn(pca)
        return self

    @classmethod
    def convert(cls, other):
        assert not other._needs_fit  # doesn't handle fit_radius right.
        self = cls(
            channel_index=other.channel_index,
            rank=other.rank,
            centered=other.centered,
            whiten=other.whiten,
            temporal_slice=other.temporal_slice,
        )
        self._needs_fit = other._needs_fit
        self.spike_length_samples = other.spike_length_samples
        self.initialize_spike_length_dependent_params()
        self.to(other.channel_index.device)
        self.b.mean.copy_(other.mean)
        self.b.components.copy_(other.components)
        self.b.whitener.copy_(other.whitener)
        return self

    def initialize_from_sklearn(self, pca):
        if self.temporal_slice is None:
            self.spike_length_samples = pca.components_.shape[1]
        else:
            # not really -- this is a hack.
            self.spike_length_samples = (
                self.temporal_slice.stop - self.temporal_slice.start
            )
        self.initialize_spike_length_dependent_params()
        if hasattr(pca, "mean_"):
            self.b.mean.copy_(torch.from_numpy(pca.mean_))
            self.center = not (self.b.mean == 0.0).all()
        else:
            self.b.mean.zero_()
            self.center = False
        self.b.components.copy_(torch.from_numpy(pca.components_))
        self.b.whitener.copy_(torch.from_numpy(pca.explained_variance_)).sqrt_()
        self._needs_fit = False


class TemporalPCADenoiser(BaseWaveformDenoiser, BaseTemporalPCA):
    default_name = "temporal_pca"

    def forward(self, waveforms, *, channels, time_shifts=None, **unused):
        waveforms = self._temporal_slice(waveforms, time_shifts=time_shifts)
        channels_in_probe, waveforms_in_probe = get_channels_in_probe(
            waveforms, channels, self.channel_index
        )
        waveforms_in_probe = self._project_in_probe(waveforms_in_probe)
        return set_channels_in_probe(waveforms_in_probe, waveforms, channels_in_probe)


class TemporalPCAFeaturizer(BaseWaveformFeaturizer, BaseTemporalPCA):
    default_name = "tpca_features"

    def transform(
        self,
        waveforms,
        *,
        channels,
        channel_index=None,
        _return_for_combined=False,
        time_shifts=None,
        **unused,
    ):
        waveforms = self._temporal_slice(waveforms, time_shifts=time_shifts)

        if channel_index is None:
            channel_index = self.b.channel_index
        channels_in_probe, waveforms_in_probe = get_channels_in_probe(
            waveforms, channels, channel_index
        )
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
        if _return_for_combined:
            return (
                waveforms,
                features_in_probe,
                channels_in_probe,
                {self.name: features},
            )

        return {self.name: features}

    def inverse_transform(self, features, channels, channel_index=None):
        if channel_index is None:
            channel_index = self.b.channel_index
        channels_in_probe, features_in_probe = get_channels_in_probe(
            features, channels, channel_index
        )
        reconstructions_in_probe = self._inverse_transform_in_probe(features_in_probe)
        recshp = (features.shape[0], self.b.components.shape[1], channel_index.shape[1])
        reconstructions = torch.full(
            recshp,
            torch.nan,
            dtype=reconstructions_in_probe.dtype,
            device=reconstructions_in_probe.device,
        )
        return set_channels_in_probe(
            reconstructions_in_probe, reconstructions, channels_in_probe
        )


class TemporalPCA(BaseWaveformAutoencoder, TemporalPCAFeaturizer):
    default_name = "tpca_features"

    def forward(self, waveforms, *, channels, time_shifts=None, **unused):
        waveforms, features_in_probe, channels_in_probe, features = self.transform(
            waveforms,
            channels=channels,
            time_shifts=time_shifts,
            _return_for_combined=True,
        )
        reconstructions_in_probe = self._inverse_transform_in_probe(features_in_probe)
        reconstructions = set_channels_in_probe(
            reconstructions_in_probe, waveforms, channels_in_probe
        )
        return reconstructions, features
