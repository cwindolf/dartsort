from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    TensorDataset,
    WeightedRandomSampler,
)

from ..util import nn_util
from ..util.logging_util import progrange
from ..util.py_util import panic
from ..util.spiketorch import get_relative_index, ptp, reindex, spawn_torch_rg
from ..util.waveform_util import make_regular_channel_index
from .transform_base import BaseWaveformFeaturizer


class AmortizedLocalization(BaseWaveformFeaturizer):
    """Localize spike waveform sources in space with a neural network.

    Order of output columns: x, y, z_abs."""

    default_name = "point_source_localizations"
    shape = (3,)
    dtype = torch.float

    def __init__(
        self,
        channel_index,
        geom,
        *,
        waveform_cfg,
        sampling_frequency=30_000.0,
        radius=100.0,
        amplitude_kind="peak",
        localization_model="pointsource",
        encoder_kind="mlp",
        hidden_dims=(256, 128),
        channel_hidden_dims=(64, 64),
        set_embed_dim=128,
        name=None,
        name_prefix="",
        n_epochs=100,
        learning_rate=3e-3,
        batch_size=256,
        fused_opt=True,
        inference_batch_size=2**14,
        norm_kind="layernorm",
        alpha_closed_form=True,
        bandwidth_scale=10.0,
        prior_std=32.0,
        amp_noise_var=0.25,
        log_amp_input=True,
        convergence_rtol=0.0,
        convergence_atol=1e-4,
        convergence_patience=10,
        min_epochs=10,
        reference="com",
        channelwise_dropout_p=0.00,
        decay_power=1,
        epoch_size=50_000,
        val_split_p=0.3,
        random_seed=0,
    ):
        assert localization_model in ("pointsource", "dipole", "gaussian")
        assert amplitude_kind in ("peak", "ptp")
        assert reference in ("main_channel", "com")
        assert encoder_kind in ("mlp", "deepsets")
        if localization_model == "gaussian":
            assert decay_power == 2
        super().__init__(
            geom=geom,
            channel_index=channel_index,
            name=name,
            name_prefix=name_prefix,
            waveform_cfg=waveform_cfg,
            sampling_frequency=sampling_frequency,
        )

        if amplitude_kind == "atpeak":
            assert self.trough_offset_samples is not None
        self.amplitude_kind = amplitude_kind
        self.radius = radius
        self.decay_power = decay_power
        self.localization_model = localization_model
        alpha_dim = 1 + 2 * (localization_model == "dipole")
        # the bandwidth has no closed form, so it is always a latent
        gaussian = localization_model == "gaussian"
        self.latent_dim = 3 + gaussian + (not alpha_closed_form) * alpha_dim
        self.bandwidth_scale = bandwidth_scale
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.encoder = None
        self.norm_kind = norm_kind
        self.encoder_kind = encoder_kind
        self.hidden_dims = hidden_dims
        self.channel_hidden_dims = channel_hidden_dims
        self.set_embed_dim = set_embed_dim
        self.alpha_closed_form = alpha_closed_form
        self.variational = prior_std is not None
        self.register_buffer_or_none(
            "prior_std", None if prior_std is None else torch.tensor(prior_std)
        )
        self.amp_noise_var = amp_noise_var
        self.log_amp_input = log_amp_input
        self.convergence_atol = convergence_atol
        self.convergence_rtol = convergence_rtol
        self.convergence_patience = convergence_patience
        self.min_epochs = min_epochs
        self.channelwise_dropout_p = channelwise_dropout_p
        self.reference = reference
        self.epoch_size = epoch_size
        self.val_split_p = val_split_p
        self.random_seed = random_seed
        self.inference_batch_size = inference_batch_size
        self.nc = self.b.geom.shape[0]
        self.fused_opt = fused_opt

        if self.nc > 1:
            self.register_buffer(
                "padded_geom", F.pad(self.b.geom.to(torch.float), (0, 0, 0, 1))
            )
            mci = make_regular_channel_index(
                geom=self.b.geom, radius=radius, to_torch=True
            )
            self.register_buffer("model_channel_index", mci)
            ri = get_relative_index(self.channel_index, self.b.model_channel_index)
            self.register_buffer("relative_index", ri)
            self.register_buffer(
                "model_channel_mask", (mci < self.nc).to(self.b.padded_geom)
            )
            self._needs_fit = True
        else:
            self._needs_fit = False

    def needs_fit(self):
        return self._needs_fit

    def fit(
        self, recording, waveforms, *, computation_cfg, channels, **fixed_properties
    ):
        weights = fixed_properties.get("weights")
        super().fit(
            recording, waveforms, computation_cfg=computation_cfg, channels=channels
        )  # just for spike len stuff
        with torch.enable_grad():
            self._fit(waveforms, channels, weights=weights)
        self.eval()
        self._needs_fit = False

    def initialize_spike_length_dependent_params(self):
        if self.encoder is not None:
            return
        if self.nc == 1:
            return

        n_latent = self.latent_dim
        if self.variational:
            n_latent *= 2

        # torch's initializers use the global rg, so fork it to stay reproducible
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_seed)
            if self.encoder_kind == "mlp":
                self.encoder = nn_util.get_waveform_mlp(
                    1,  # amplitudes only, time dim is just one amplitude feature
                    self.b.model_channel_index.shape[1],
                    self.hidden_dims,
                    n_latent,
                    norm_kind=self.norm_kind,
                    channelwise_dropout_p=self.channelwise_dropout_p,
                )
            elif self.encoder_kind == "deepsets":
                n_channel_features = self.b.geom.shape[1] + 2
                self.encoder = nn_util.get_waveform_deepsets_model(
                    n_channel_features,
                    self.channel_hidden_dims,
                    self.hidden_dims,
                    n_latent,
                    embed_dim=self.set_embed_dim,
                    norm_kind=self.norm_kind,
                )
            else:
                panic(self.encoder_kind)
        self.encoder.to(self.b.padded_geom.device)

    def reparameterize(self, mu, var, generator: torch.Generator | None = None):
        if var is None or not self.training:
            return mu
        std = var.relu().sqrt()
        eps = torch.randn(
            std.shape, device=std.device, dtype=std.dtype, generator=generator
        )
        return mu + eps * std

    def get_reference_points(self, channels, obs_amps=None, neighborhoods=None):
        if self.reference == "main_channel":
            return self.b.padded_geom[channels]
        elif self.reference == "com":
            assert obs_amps is not None
            if neighborhoods is None:
                neighborhoods = self.b.padded_geom[self.b.model_channel_index[channels]]
            w = obs_amps / obs_amps.sum(1, keepdims=True).clamp(min=1e-6)
            centers = torch.sum(w.unsqueeze(-1) * neighborhoods, dim=1)
            return centers
        else:
            panic(self.reference)

    def local_geometry(
        self, channels: torch.Tensor, obs_amps: torch.Tensor | None = None
    ) -> torch.Tensor:
        neighbors = self.b.padded_geom[self.b.model_channel_index[channels]]
        centers = self.get_reference_points(
            channels, obs_amps=obs_amps, neighborhoods=neighbors
        )
        return neighbors - centers.unsqueeze(1)

    def local_distances(self, z, channels, obs_amps=None):
        """Return distances from each z to its local geom centered at channels."""
        local_geom = self.local_geometry(channels, obs_amps=obs_amps)
        dx = z[:, 0, None] - local_geom[:, :, 0]
        dz = z[:, 2, None] - local_geom[:, :, 1]
        y = F.softplus(z[:, 1]).unsqueeze(1)
        dists = dx**2 + dz**2 + y**2
        if self.decay_power == 1:
            dists = dists.relu().sqrt()
        elif self.decay_power == 2:
            pass
        else:
            panic(self.decay_power)
        return dists

    def get_alphas(self, obs_amps, pred_amps_alpha1, masks, return_pred=False):
        # least squares with no intercept
        a0 = masks * pred_amps_alpha1
        numer = a0.mul(obs_amps).sum(dim=1)
        denom = a0.square().sum(dim=1)
        alphas = numer.div(denom.clamp(min=1e-6))
        if return_pred:
            return alphas, alphas.unsqueeze(1) * pred_amps_alpha1
        return alphas

    def get_bandwidths(self, z):
        return self.bandwidth_scale * F.softplus(z[:, 3])

    def point_source_model(self, z, obs_amps, masks, channels):
        dists = self.local_distances(z, channels, obs_amps=obs_amps)
        if self.alpha_closed_form:
            pred_amps_alpha1 = dists.clamp(min=1e-6).reciprocal()
            alphas, pred_amps = self.get_alphas(
                obs_amps, pred_amps_alpha1, masks, return_pred=True
            )
        else:
            alphas = F.softplus(z[:, 3])
            pred_amps = alphas.unsqueeze(1) / (dists + 1e-6)

        return alphas, pred_amps

    def gaussian_model(self, z, obs_amps, masks, channels):
        # decay_power==2, so these are squared distances
        sq_dists = self.local_distances(z, channels, obs_amps=obs_amps)
        sigmas = self.get_bandwidths(z)
        twosigmasq = sigmas.square().mul(2).clamp(min=1e-6).unsqueeze(1)
        pred_amps_alpha1 = sq_dists.div(twosigmasq).neg().exp()
        if self.alpha_closed_form:
            alphas, pred_amps = self.get_alphas(
                obs_amps, pred_amps_alpha1, masks, return_pred=True
            )
        else:
            alphas = F.softplus(z[:, 4])
            pred_amps = alphas.unsqueeze(1) * pred_amps_alpha1

        return alphas, pred_amps

    def dipole_model(self, z, obs_amps, masks, channels):
        local_geom = self.local_geometry(channels, obs_amps=obs_amps)

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
        elif self.localization_model == "gaussian":
            alphas, pred_amps = self.gaussian_model(z, obs_amps, masks, channels)
        elif self.localization_model == "dipole":
            alphas, pred_amps = self.dipole_model(z, obs_amps, masks, channels)
        else:
            panic(self.localization_model)
        return alphas, pred_amps

    def encoder_features(self, amps, mask, channels, obs_amps):
        if self.log_amp_input:
            x = amps.clamp(min=1e-3).log()
            denom = mask.sum(1, keepdim=True).clamp(min=1.0)
            x = x - x.mul(mask).sum(1, keepdim=True).div(denom)
            x = x * mask
        else:
            x = amps

        if self.encoder_kind == "mlp":
            return torch.stack((x, mask), dim=1)
        elif self.encoder_kind == "deepsets":
            local_geom = self.local_geometry(channels, obs_amps=obs_amps)
            amp_mask = torch.stack((x, mask), dim=2)
            return torch.cat((local_geom.div(self.radius), amp_mask), dim=2), mask
        else:
            panic(self.encoder_kind)

    def forward(self, amps, mask, obs_amps, channels, generator=None):
        x_mask = self.encoder_features(amps, mask, channels, obs_amps)
        assert self.encoder is not None
        mu = self.encoder(x_mask)
        var = None
        if self.variational:
            mu, var = mu.chunk(2, dim=-1)
            var = F.softplus(var)
        z = self.reparameterize(mu, var, generator=generator)
        _alphas, pred_amps = self.decode(z, channels, obs_amps, mask)
        return pred_amps, mu, var

    def latents_to_locs(
        self, mu: torch.Tensor, channels: torch.Tensor, obs_amps: torch.Tensor
    ) -> torch.Tensor:
        x, y, z = mu[:, :3].T
        y = F.softplus(y)
        mx, mz = self.get_reference_points(channels, obs_amps=obs_amps).T
        return torch.column_stack((x + mx, y, z + mz))

    def loss_function(self, recon_x, x, mask, mu, var):
        resid = (recon_x - x).mul(mask)
        nll = resid.square().sum(dim=1).div(2 * self.amp_noise_var).mean()
        kld = 0.0
        if self.variational:
            prior_var = self.b.prior_std.square()
            ratio = (var + mu.pow(2)) / prior_var - 1
            kld = torch.log(prior_var / var).add(ratio)
            kld = kld.sum(dim=1).mul(0.5).mean()
        return nll, kld

    def _fit(self, waveforms, channels, weights=None):
        # apply channel reindexing before any fitting...
        wf_dev = waveforms.device
        my_dev = self.b.padded_geom.device
        if waveforms.ndim == 2:
            amps = waveforms.unsqueeze(1)
        else:
            if self.amplitude_kind == "ptp":
                amps = ptp(waveforms)
            elif self.amplitude_kind == "peak":
                amin, amax = waveforms.aminmax(dim=1)
                amps = torch.maximum(amin.abs_(), amax.abs_())
            else:
                panic(self.amplitude_kind)
            amps = amps[:, None]
        amps = reindex(
            channels.to(device=wf_dev),
            amps,
            self.relative_index.to(device=wf_dev),
            pad_value=0.0,
        )[:, 0]
        del waveforms
        assert amps is not None
        amps = amps.to(device=my_dev)
        channels = channels.to(device=my_dev)

        rg = np.random.default_rng(self.random_seed)
        gen = spawn_torch_rg(rg, device=my_dev)

        # make a validation set for early stopping
        if self.val_split_p:
            n_val = int(np.ceil(self.val_split_p * len(amps)))
            istrain = np.ones(len(amps), dtype=bool)
            val_ix = rg.choice(len(amps), size=n_val, replace=False)
            val_ix.sort()
            istrain[val_ix] = False
            val_amps = amps[val_ix]
            val_channels = channels[val_ix]
            train_ix = np.flatnonzero(istrain)
            amps = amps[train_ix]
            channels = channels[train_ix]
            weights = weights[train_ix] if weights is not None else None
        else:
            # early stopping will just be done on the train wfs
            val_amps = amps
            val_channels = channels

        dataset = TensorDataset(amps, channels)
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
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, fused=self.fused_opt
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs
        )

        val_dataset = TensorDataset(val_amps, val_channels)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        self.train()
        best_val = None
        best_state = None
        n_no_improve = 0
        with progrange(self.n_epochs, desc="Train localizer", unit="epoch") as pbar:
            for epoch in pbar:
                total_loss = 0
                total_nll = 0
                total_kld = 0
                nbatch = 0
                n_examples = 0
                for amps_batch, chans_batch in dataloader:
                    # for whatever reason, batch sampler adds an empty dim
                    amps_batch = amps_batch[0].to(device=my_dev)
                    chans_batch = chans_batch[0].to(device=my_dev)

                    optimizer.zero_grad()
                    channels_mask = self.b.model_channel_mask[chans_batch]
                    reconstructed_amps, mu, var = self.forward(
                        amps_batch,
                        channels_mask,
                        amps_batch,
                        chans_batch,
                        generator=gen,
                    )
                    nll, kld = self.loss_function(
                        reconstructed_amps, amps_batch, channels_mask, mu, var
                    )
                    loss = nll
                    if self.variational:
                        loss = loss + kld
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    total_nll += nll.item()
                    if self.variational:
                        total_kld += kld.item()

                    nbatch += 1
                    n_examples += chans_batch.numel()
                    if n_examples >= self.epoch_size:
                        break

                valbatch = 0
                val_loss = 0.0
                with torch.no_grad():
                    self.eval()
                    n_examples = 0
                    for amps_batch, chans_batch in val_loader:
                        amps_batch = amps_batch.to(device=my_dev)
                        chans_batch = chans_batch.to(device=my_dev)
                        channels_mask = self.b.model_channel_mask[chans_batch]
                        reconstructed_amps, mu, var = self.forward(
                            amps_batch, channels_mask, amps_batch, chans_batch
                        )
                        nll, kld = self.loss_function(
                            reconstructed_amps, amps_batch, channels_mask, mu, var
                        )
                        loss = nll
                        if self.variational:
                            loss = loss + kld
                        val_loss += loss.item()
                        valbatch += 1
                        n_examples += chans_batch.numel()
                        if n_examples >= self.epoch_size:
                            break
                    self.train()

                scheduler.step()
                nbatch = max(1, nbatch)
                loss = total_loss / nbatch
                nll = total_nll / nbatch
                val_loss = val_loss / valbatch
                desc = f"[loss={loss:0.4f},val={val_loss:0.4f}"
                if self.variational:
                    kld = total_kld / nbatch
                    desc += f",nll={nll:0.2f},kld={kld:0.2f}"
                desc += "]"
                pbar.set_description(f"Train localizer {desc}")

                if best_val is None:
                    improved = True
                else:
                    adiff = best_val - val_loss
                    rtol = self.convergence_rtol * abs(best_val)
                    improved = adiff > max(rtol, self.convergence_atol)
                n_no_improve = 0 if improved else n_no_improve + 1
                if best_val is None or val_loss < best_val:
                    best_val = val_loss
                    assert self.encoder is not None
                    best_state = {
                        k: v.detach().clone()
                        for k, v in self.encoder.state_dict().items()
                    }

                if epoch < self.min_epochs:
                    continue
                if n_no_improve >= self.convergence_patience:
                    pbar.set_description(f"Localizer converged at epoch={epoch} {desc}")
                    break

        if best_state is not None:
            assert self.encoder is not None
            self.encoder.load_state_dict(best_state)

    def transform_unbatched(self, waveforms, channels, return_extra=False):
        # handle getting amplitudes, reindexing channels
        obs_amps = None
        if waveforms.ndim == 2:
            waveforms = waveforms.unsqueeze(1)
        else:
            if self.amplitude_kind == "ptp":
                obs_amps = ptp(waveforms)
            elif self.amplitude_kind == "peak":
                obs_amps = waveforms.abs().max(dim=1).values
            else:
                panic(self.amplitude_kind)
            waveforms = obs_amps[:, None]

        waveforms = waveforms.to(device=self.relative_index.device)
        waveforms = reindex(channels, waveforms, self.relative_index, pad_value=0.0)
        obs_amps = waveforms[:, 0]

        # nn inputs
        mask = self.b.model_channel_mask[channels]
        x_mask = self.encoder_features(obs_amps, mask, channels, obs_amps)

        # encode
        # this is where we need to batch
        assert self.encoder is not None
        mu = self.encoder(x_mask)
        if self.variational:
            mu, _var = mu.chunk(2, dim=-1)
        locs = self.latents_to_locs(mu, channels, obs_amps)

        if return_extra:
            mx, mz = self.get_reference_points(channels, obs_amps=obs_amps).T
            _alphas, pred_amps = self.decode(mu, channels, obs_amps, mask)
            return dict(locs=locs, obs_amps=obs_amps, pred_amps=pred_amps, mx=mx, mz=mz)

        return locs

    def transform(self, waveforms, *, channels, **fixed_properties):
        n = len(waveforms)
        if self.nc == 1:
            return {self.name: waveforms.new_zeros((n, 3))}
        with torch.no_grad():
            if n > self.inference_batch_size:
                locs = waveforms.new_empty((n, 3))
                my_device = self.b.padded_geom.device
                device_in = locs.device
                for bs in range(0, n, self.inference_batch_size):
                    be = bs + self.inference_batch_size
                    batch = waveforms[bs:be].to(my_device)
                    batch_chans = channels[bs:be].to(my_device)
                    res = self.transform_unbatched(batch, batch_chans)
                    locs[bs:be] = cast(torch.Tensor, res).to(device_in)
                    del batch, batch_chans, res
            else:
                locs = self.transform_unbatched(waveforms, channels)
        return {self.name: locs}
