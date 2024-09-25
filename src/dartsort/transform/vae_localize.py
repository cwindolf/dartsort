import torch
import torch.nn as nn
import torch.nn.functional as F
from dartsort.util import nn_util
from dartsort.util.spiketorch import get_relative_index, ptp, reindex
from dartsort.util.waveform_util import make_regular_channel_index
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange

from .transform_base import BaseWaveformFeaturizer


class VAELocalization(BaseWaveformFeaturizer):
    """Order of output columns: x, y, z_abs, alpha"""

    default_name = "vae_point_source_localizations"
    shape = (4,)
    dtype = torch.double

    def __init__(
        self,
        channel_index,
        geom,
        radius=100.0,
        amplitude_kind="peak",
        localization_model="pointsource",
        hidden_dims=(256, 128),
        name=None,
        name_prefix="",
        epochs=100,
        learning_rate=1e-3,
        batch_size=32,
        use_batchnorm=True,
        alpha_closed_form=True,
        amplitudes_only=False,
        prior_variance=80.0,
        convergence_eps=0.01,
        min_epochs=2,
        scale_loss_by_mean=True,
        reference='main_channel',
        channelwise_dropout_p=0.0,
    ):
        assert localization_model in ("pointsource", "dipole")
        assert amplitude_kind in ("peak", "ptp")
        assert reference in ('main_channel', 'com')
        super().__init__(
            geom=geom, channel_index=channel_index, name=name, name_prefix=name_prefix
        )

        self.amplitude_kind = amplitude_kind
        self.radius = radius
        self.localization_model = localization_model
        alpha_dim = 1 + 2 * (localization_model == "dipole")
        self.latent_dim = 3 + (not alpha_closed_form) * alpha_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.encoder = None
        self.amplitudes_only = amplitudes_only
        self.use_batchnorm = use_batchnorm
        self.hidden_dims = hidden_dims
        self.alpha_closed_form = alpha_closed_form
        self.variational = prior_variance is not None
        self.prior_variance = torch.tensor(prior_variance) if prior_variance is not None else None
        self.convergence_eps = convergence_eps
        self.min_epochs = min_epochs
        self.scale_loss_by_mean = scale_loss_by_mean
        self.channelwise_dropout_p = channelwise_dropout_p
        self.reference = reference

        self.register_buffer(
            "padded_geom", F.pad(self.geom.to(torch.float), (0, 0, 0, 1))
        )
        self.register_buffer(
            "model_channel_index",
            make_regular_channel_index(geom=self.geom, radius=radius, to_torch=True),
        )
        self.register_buffer(
            "relative_index",
            get_relative_index(self.channel_index, self.model_channel_index),
        )
        self.nc = len(self.geom)
        self._needs_fit = True

    def needs_fit(self):
        return self._needs_fit

    def initialize_net(self, spike_length_samples):
        if self.encoder is not None:
            return

        n_latent = self.latent_dim
        if self.variational:
            n_latent *= 2

        self.encoder = nn_util.get_waveform_mlp(
            spike_length_samples,
            self.model_channel_index.shape[1],
            self.hidden_dims,
            n_latent,
            use_batchnorm=self.use_batchnorm,
            channelwise_dropout_p=self.channelwise_dropout_p,
        )
        self.encoder.to(self.padded_geom.device)

    def reparameterize(self, mu, var):
        if var is None:
            return mu
        std = var.sqrt()
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_reference_points(self, channels, obs_amps=None, neighborhoods=None):
        if self.reference == 'main_channel':
            return self.padded_geom[channels]
        elif self.reference == 'com':
            if neighborhoods is None:
                neighborhoods = self.padded_geom[self.model_channel_index[channels]]
            w = obs_amps / obs_amps.sum(1, keepdims=True)
            centers = torch.sum(w.unsqueeze(-1) * neighborhoods, dim=1)
            return centers
        else:
            assert False

    def local_distances(self, z, channels, obs_amps=None):
        """Return distances from each z to its local geom centered at channels."""
        neighbors = self.padded_geom[self.model_channel_index[channels]]
        centers = self.get_reference_points(channels, obs_amps=obs_amps, neighborhoods=neighbors)
        local_geom = neighbors - centers.unsqueeze(1)
        dx = z[:, 0, None] - local_geom[:, :, 0]
        dz = z[:, 2, None] - local_geom[:, :, 1]
        y = F.softplus(z[:, 1]).unsqueeze(1)
        dists = torch.sqrt(dx**2 + dz**2 + y**2)
        return dists

    def get_alphas(self, obs_amps, dists, masks, return_pred=False):
        pred_amps_alpha1 = 1.0 / dists
        # least squares with no intercept
        alphas = (masks * obs_amps * pred_amps_alpha1).sum(1) / (
            masks * pred_amps_alpha1
        ).square().sum(1)
        if return_pred:
            return alphas, alphas.unsqueeze(1) * pred_amps_alpha1
        return alphas

    def point_source_model(self, z, obs_amps, masks, channels):
        dists = self.local_distances(z, channels, obs_amps=obs_amps)
        if self.alpha_closed_form:
            alphas, pred_amps = self.get_alphas(
                obs_amps, dists, masks, return_pred=True
            )
        else:
            alphas = F.softplus(z[:, 3])
            pred_amps = alphas.unsqueeze(1) / dists
        return alphas, pred_amps

    def dipole_model(self, z, obs_amps, masks, channels):
        neighbors = self.padded_geom[self.model_channel_index[channels]]
        centers = self.get_reference_points(channels, obs_amps=obs_amps, neighborhoods=neighbors)
        local_geom = neighbors - centers.unsqueeze(1)

        # displacements from probe
        dx = z[:, 0, None] - local_geom[:, :, 0]
        dz = z[:, 2, None] - local_geom[:, :, 1]
        y = F.softplus(z[:, 1]).unsqueeze(1)
        duv = torch.stack((dx, y.broadcast_to(dx.shape), dz), dim=2)

        # displacment over distance cubed. (n_spikes, n_chans, 3)
        X = duv * duv.square().sum(2, keepdim=True).pow(-1.5)
        if self.alpha_closed_form:
            # beta = torch.linalg.pinv(X.mT @ X) @ (X.mT @ obs_amps.unsqueeze(2))
            # beta = torch.linalg.lstsq(X.mT @ X, X.mT @ obs_amps.unsqueeze(2)).solution
            beta = torch.linalg.lstsq(X, obs_amps.unsqueeze(2)).solution
            pred_amps = (X @ beta)[:, :, 0]
            beta = beta[:, :, 0]
        else:
            beta = z[:, 3:]
            pred_amps = (X @ beta.unsqueeze(2))[:, :, 0]

        return beta, pred_amps

    def decode(self, z, channels, obs_amps, masks):
        if self.localization_model in ("pointsource", "monopole"):
            alphas, pred_amps = self.point_source_model(z, obs_amps, masks, channels)
        elif self.localization_model == "dipole":
            alphas, pred_amps = self.dipole_model(z, obs_amps, masks, channels)
        else:
            assert False
        return alphas, pred_amps

    def forward(self, x, mask, obs_amps, channels):
        x_mask = torch.cat((x, mask.unsqueeze(1)), dim=1)
        mu = self.encoder(x_mask)
        var = None
        if self.variational:
            mu, var = mu.chunk(2, dim=-1)
            var = F.softplus(var)
        z = self.reparameterize(mu, var)
        alphas, pred_amps = self.decode(z, channels, obs_amps, mask)
        return pred_amps, mu, var

    def loss_function(self, recon_x, x, mask, mu, var):
        recon_x_masked = recon_x * mask
        x_masked = x * mask
        if self.scale_loss_by_mean:
            # 1/mean amplitude
            rescale = mask.sum(1, keepdim=True) / x_masked.sum(1, keepdim=True)
            x_masked *= rescale
            recon_x_masked *= rescale
        MSE = F.mse_loss(recon_x_masked, x_masked, reduction="sum") / self.batch_size
        KLD = 0.0
        if self.variational:
            KLD = 0.5 * (
                + torch.log(self.prior_variance / var)
                + mu.pow(2) / self.prior_variance
                - 1
            ).sum() / self.batch_size
        return MSE, KLD

    def _fit(self, waveforms, channels):
        # apply channel reindexing before any fitting...
        if waveforms.ndim == 2:
            assert self.amplitudes_only
            waveforms = waveforms.unsqueeze(1)
            waveforms = reindex(channels, waveforms, self.relative_index, pad_value=0.0)
            amps = waveforms[:, 0]
        elif self.amplitudes_only:
            if self.amplitude_kind == "ptp":
                waveforms = ptp(waveforms)
            elif self.amplitude_kind == "peak":
                waveforms = waveforms.abs().max(dim=1).values
            waveforms = waveforms[:, None]
            waveforms = reindex(channels, waveforms, self.relative_index, pad_value=0.0)
            amps = waveforms[:, 0]
        else:
            waveforms = reindex(channels, waveforms, self.relative_index, pad_value=0.0)
            if self.amplitude_kind == "ptp":
                amps = ptp(waveforms)
            elif self.amplitude_kind == "peak":
                amps = waveforms.abs().max(dim=1).values

        self.initialize_net(waveforms.shape[1])

        dataset = TensorDataset(waveforms, amps, channels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.train()
        mse_history = []
        with trange(self.epochs, desc="Epochs", unit="epoch") as pbar:
            for epoch in pbar:
                total_loss = 0
                total_mse = 0
                total_kld = 0
                for waveform_batch, amps_batch, channels_batch in dataloader:
                    optimizer.zero_grad()
                    channels_mask = self.model_channel_index[channels_batch] < len(
                        self.geom
                    )
                    channels_mask = channels_mask.to(waveform_batch)
                    reconstructed_amps, mu, var = self.forward(
                        waveform_batch, channels_mask, amps_batch, channels_batch
                    )
                    mse, kld = self.loss_function(
                        reconstructed_amps, amps_batch, channels_mask, mu, var
                    )
                    loss = mse + kld
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    total_mse += mse.item()
                    total_kld += kld.item() if self.variational else kld

                loss = total_loss / len(dataloader)
                mse = total_mse / len(dataloader)
                mse_history.append(mse)
                desc = f"[loss={loss:0.2f},mse={total_mse/len(dataloader):0.2f},kld={total_kld/len(dataloader):0.2f}]"
                pbar.set_description(f"Epochs {desc}")

                # check convergence
                if epoch < self.min_epochs:
                    continue

                diff = min(mse_history[:-1]) - mse
                if diff / min(mse_history[:-1]) < self.convergence_eps:
                    pbar.set_description(f"Converged epoch={epoch} {desc}")
                    break

    def fit(self, waveforms, max_channels):
        with torch.enable_grad():
            self._fit(waveforms, max_channels)
        self._needs_fit = False

    def transform(self, waveforms, max_channels, return_extra=False):
        """
        waveforms : torch.tensor, shape (num_waveforms, n_timesteps, n_channels_subset)
        max_channels : torch.tensor, shape (num_waveforms,)

        waveform[n] lives on channels self.channel_index[max_channels[n]]
        """
        # handle getting amplitudes, reindexing channels, and amplitudes_only logic
        if waveforms.ndim == 2:
            assert self.amplitudes_only
            waveforms = waveforms.unsqueeze(1)
        elif self.amplitudes_only:
            if self.amplitude_kind == "ptp":
                obs_amps = ptp(waveforms)
            elif self.amplitude_kind == "peak":
                obs_amps = waveforms.abs().max(dim=1).values
                waveforms = obs_amps[:, None]
        waveforms = reindex(max_channels, waveforms, self.relative_index, pad_value=0.0)
        if self.amplitudes_only:
            obs_amps = waveforms[:, 0]
        elif return_extra or self.reference == 'com':
            if self.amplitude_kind == "ptp":
                obs_amps = ptp(waveforms)
            elif self.amplitude_kind == "peak":
                obs_amps = waveforms.abs().max(dim=1).values
        else:
            # in this condition, we don't need the amp vecs
            obs_amps = None

        # nn inputs
        mask = self.model_channel_index[max_channels] < self.nc
        mask = mask.to(waveforms)
        x_mask = torch.cat((waveforms, mask.unsqueeze(1)), dim=1)

        # encode
        mu = self.encoder(x_mask)
        var = None
        if self.variational:
            mu, var = mu.chunk(2, dim=-1)
        x, y, z = mu[:, :3].T
        y = F.softplus(y)
        mx, mz = self.get_reference_points(max_channels, obs_amps=obs_amps).T
        x = x + mx
        z = z + mz

        if return_extra:
            alphas, pred_amps = self.decode(mu, max_channels, obs_amps, mask)
            return x, y, z, obs_amps, pred_amps, mx, mz

        return x, y, z
