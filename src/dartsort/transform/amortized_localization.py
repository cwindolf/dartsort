import numpy as np
import torch
import torch.nn.functional as F
from dartsort.util import nn_util
from dartsort.util.spiketorch import get_relative_index, ptp, reindex, spawn_torch_rg
from dartsort.util.waveform_util import make_regular_channel_index
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    RandomSampler,
    BatchSampler,
    WeightedRandomSampler,
)
from tqdm.auto import trange

from .transform_base import BaseWaveformFeaturizer


class AmortizedLocalization(BaseWaveformFeaturizer):
    """Order of output columns: x, y, z_abs."""

    default_name = "point_source_localizations"
    shape = (3,)
    dtype = torch.float

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
        n_epochs=100,
        learning_rate=3e-3,
        batch_size=32,
        inference_batch_size=2**14,
        norm_kind="layernorm",
        alpha_closed_form=True,
        amplitudes_only=True,
        prior_variance=None,
        convergence_rtol=0.01,
        convergence_atol=1e-4,
        min_epochs=10,
        scale_loss_by_mean=True,
        reference="main_channel",
        channelwise_dropout_p=0.00,
        epoch_size=50_000,
        val_split_p=0.3,
        random_seed=0,
    ):
        assert localization_model in ("pointsource", "dipole")
        assert amplitude_kind in ("peak", "ptp")
        assert reference in ("main_channel", "com")
        super().__init__(
            geom=geom, channel_index=channel_index, name=name, name_prefix=name_prefix
        )

        self.amplitude_kind = amplitude_kind
        self.radius = radius
        self.localization_model = localization_model
        alpha_dim = 1 + 2 * (localization_model == "dipole")
        self.latent_dim = 3 + (not alpha_closed_form) * alpha_dim
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.encoder = None
        self.amplitudes_only = amplitudes_only
        self.norm_kind = norm_kind
        self.hidden_dims = hidden_dims
        self.alpha_closed_form = alpha_closed_form
        self.variational = prior_variance is not None
        self.prior_variance = (
            torch.tensor(prior_variance) if prior_variance is not None else None
        )
        self.convergence_atol = convergence_atol
        self.convergence_rtol = convergence_rtol
        self.min_epochs = min_epochs
        self.scale_loss_by_mean = scale_loss_by_mean
        self.channelwise_dropout_p = channelwise_dropout_p
        self.reference = reference
        self.epoch_size = epoch_size
        self.val_split_p = val_split_p
        self.random_seed = random_seed
        self.inference_batch_size = inference_batch_size

        self.register_buffer(
            "padded_geom", F.pad(self.geom.to(torch.float), (0, 0, 0, 1))
        )
        mci = make_regular_channel_index(geom=self.geom, radius=radius, to_torch=True)
        self.register_buffer("model_channel_index", mci)
        ri = get_relative_index(self.channel_index, self.model_channel_index)
        self.register_buffer("relative_index", ri)
        self._needs_fit = True

    def needs_fit(self):
        return self._needs_fit

    def fit(self, waveforms, max_channels, recording=None, weights=None):
        super().fit(waveforms, max_channels, recording, weights)
        with torch.enable_grad():
            self._fit(waveforms, max_channels, weights=weights)
        self.eval()
        self._needs_fit = False

    def initialize_spike_length_dependent_params(self):
        if self.encoder is not None:
            return

        n_latent = self.latent_dim
        if self.variational:
            n_latent *= 2

        self.encoder = nn_util.get_waveform_mlp(
            1 if self.amplitudes_only else self.spike_length_samples,
            self.model_channel_index.shape[1],
            self.hidden_dims,
            n_latent,
            norm_kind=self.norm_kind,
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
        if self.reference == "main_channel":
            return self.padded_geom[channels]
        elif self.reference == "com":
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
        centers = self.get_reference_points(
            channels, obs_amps=obs_amps, neighborhoods=neighbors
        )
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
        centers = self.get_reference_points(
            channels, obs_amps=obs_amps, neighborhoods=neighbors
        )
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
            # 1/(n_chans_retained*mean amplitude)
            rescale = 1.0 / x_masked.sum(1, keepdim=True)
        else:
            rescale = 1.0 / mask.sum(1, keepdim=True)
        x_masked *= rescale
        recon_x_masked *= rescale
        mse = F.mse_loss(recon_x_masked, x_masked, reduction="sum") / self.batch_size
        kld = 0.0
        if self.variational:
            kld = (
                0.5
                * (
                    +torch.log(self.prior_variance / var)
                    + mu.pow(2) / self.prior_variance
                    - 1
                ).sum()
                / self.batch_size
            )
        return mse, kld

    def _fit(self, waveforms, channels, weights=None):
        # apply channel reindexing before any fitting...
        amps = None
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
        assert amps is not None

        rg = np.random.default_rng(self.random_seed)

        # make a validation set for early stopping
        if self.val_split_p:
            istrain = rg.binomial(1, p=self.val_split_p, size=len(waveforms))
            istrain = istrain.astype(bool)
            isval = np.logical_not(istrain)
            val_waveforms = waveforms[isval]
            val_amps = amps[isval]
            val_channels = channels[isval]
            waveforms = waveforms[istrain]
            amps = amps[istrain]
            channels = channels[istrain]
            weights = weights[istrain] if weights is not None else None
        else:
            # early stopping will just be done on the train wfs
            val_waveforms = waveforms
            val_amps = amps
            val_channels = channels

        dataset = TensorDataset(waveforms, amps, channels)
        if weights is None:
            sampler = RandomSampler(dataset, generator=spawn_torch_rg(rg))
        else:
            assert len(weights) == len(dataset)
            sampler = WeightedRandomSampler(
                weights,
                num_samples=len(dataset),
                generator=spawn_torch_rg(rg),
            )
        sampler = BatchSampler(sampler, batch_size=self.batch_size, drop_last=True)
        dataloader = DataLoader(dataset, sampler=sampler)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        val_dataset = TensorDataset(val_waveforms, val_amps, val_channels)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        self.train()
        mse_history = []
        with trange(self.n_epochs, desc="Train localizer", unit="epoch") as pbar:
            for epoch in pbar:
                total_loss = 0
                total_mse = 0
                total_kld = 0
                n_examples = 0
                for waveform_batch, amps_batch, chans_batch in dataloader:
                    # for whatever reason, batch sampler adds an empty dim
                    waveform_batch = waveform_batch[0]
                    amps_batch = amps_batch[0]
                    chans_batch = chans_batch[0]

                    optimizer.zero_grad()
                    channels_mask = self.model_channel_index[chans_batch] < len(
                        self.geom
                    )
                    channels_mask = channels_mask.to(waveform_batch)
                    reconstructed_amps, mu, var = self.forward(
                        waveform_batch, channels_mask, amps_batch, chans_batch
                    )
                    mse, kld = self.loss_function(
                        reconstructed_amps, amps_batch, channels_mask, mu, var
                    )
                    loss = mse
                    if self.variational:
                        loss = loss + kld
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    total_mse += mse.item()
                    if self.variational:
                        total_kld += kld.item()

                    n_examples += self.batch_size
                    if n_examples >= self.epoch_size:
                        break

                valbatch = 0
                val_loss = 0.0
                with torch.no_grad():
                    self.eval()
                    for waveform_batch, amps_batch, chans_batch in val_loader:
                        ci_batch = self.model_channel_index[chans_batch]
                        channels_mask = ci_batch < len(self.geom)
                        channels_mask = channels_mask.to(waveform_batch)
                        reconstructed_amps, mu, var = self.forward(
                            waveform_batch, channels_mask, amps_batch, chans_batch
                        )
                        mse, kld = self.loss_function(
                            reconstructed_amps, amps_batch, channels_mask, mu, var
                        )
                        loss = mse
                        if self.variational:
                            loss = loss + kld
                        val_loss += loss.item()
                        valbatch += 1
                    self.train()

                nbatch = n_examples / self.batch_size
                loss = total_loss / nbatch
                mse = total_mse / nbatch
                val_loss = val_loss / valbatch
                mse_history.append(val_loss)
                desc = f"[loss={loss:0.4f},val={val_loss:0.4f}"
                if self.variational:
                    kld = total_kld / nbatch
                    desc += f",mse={mse:0.2f},kld={kld:0.2f}"
                desc += "]"
                pbar.set_description(f"Train localizer {desc}")

                # check convergence
                if epoch < self.min_epochs:
                    continue

                # positive if cur is smaller than prev
                adiff = min(mse_history[:-1]) - mse
                rdiff = adiff / min(mse_history[:-1])
                if rdiff < self.convergence_rtol or adiff < self.convergence_atol:
                    pbar.set_description(f"Localizer converged at epoch={epoch} {desc}")
                    break

    def transform_unbatched(self, waveforms, max_channels, return_extra=False):
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
        elif return_extra or self.reference == "com":
            if self.amplitude_kind == "ptp":
                obs_amps = ptp(waveforms)
            elif self.amplitude_kind == "peak":
                obs_amps = waveforms.abs().max(dim=1).values
        else:
            # in this condition, we don't need the amp vecs
            obs_amps = None

        # nn inputs
        mask = self.model_channel_index[max_channels] < self.geom.shape[0]
        mask = mask.to(waveforms)
        x_mask = torch.cat((waveforms, mask.unsqueeze(1)), dim=1)

        # encode
        # this is where we need to batch
        mu = self.encoder(x_mask)
        var = None
        if self.variational:
            mu, var = mu.chunk(2, dim=-1)
        x, y, z = mu[:, :3].T
        y = F.softplus(y)
        mx, mz = self.get_reference_points(max_channels, obs_amps=obs_amps).T
        x = x + mx
        z = z + mz
        locs = torch.column_stack((x, y, z))

        if return_extra:
            alphas, pred_amps = self.decode(mu, max_channels, obs_amps, mask)
            return dict(locs=locs, obs_amps=obs_amps, pred_amps=pred_amps, mx=mx, mz=mz)

        return locs

    def transform(self, waveforms, max_channels, show_progress=False):
        n = len(waveforms)
        with torch.no_grad():
            if n > self.inference_batch_size:
                locs = waveforms.new_empty((n, 3))
                rg = trange if show_progress else range
                my_device = self.padded_geom.device
                device_in = locs.device
                for bs in rg(0, n, self.inference_batch_size):
                    be = bs + self.inference_batch_size
                    batch = waveforms[bs:be].to(my_device)
                    batch_chans = max_channels[bs:be].to(my_device)
                    res = self.transform_unbatched(batch, batch_chans)
                    locs[bs:be] = res.to(device_in)
                    del batch, batch_chans, res
            else:
                locs = self.transform_unbatched(waveforms, max_channels)
        return {self.name: locs}
