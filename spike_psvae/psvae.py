# helpful code references:
# github.com/AntixK/PyTorch-VAE/
# github.com/themattinthehatt/behavenet/blob/master/behavenet/models/vaes.py
# github.com/pytorch/examples/blob/master/vae/main.py

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

from . import layers


class PSVAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        supervised_latent_dim,
        unsupervised_latent_dim,
        alpha=1.0,
    ):
        """
        Encoder
        """
        super(PSVAE, self).__init__()

        latent_dim = unsupervised_latent_dim + supervised_latent_dim
        self.latent_dim = latent_dim
        self.unsupervised_latent_dim = unsupervised_latent_dim
        self.supervised_latent_dim = supervised_latent_dim

        self.alpha = alpha

        self.encoder = encoder
        self.decoder = decoder

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
        self.diag_y_hat = layers.DiagLinear(supervised_latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def split(self, z):
        # TODO diagonal matrix multiply?
        zs = z[: self.supervised_latent_dim]
        zu = z[self.supervised_latent_dim :]
        return zs, zu

    def decode(self, z):
        # TODO might need to add an activation?
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        zs, zu = self.split(z)
        y_hat = self.diag_y_hat(zs)
        recon_x = self.decode(z)
        return recon_x, y_hat, mu, logvar

    def loss(self, x, y, recon_x, y_hat, mu, logvar):
        # mean over batches, sum over data dims

        # TODO is the distribution of latents different in PSVAE?
        #      like, supervised have their own mean?
        # -KL divergence to iid standard normal
        # 1312.6114 appendix B
        # note, -DKL is in ELBO which we want to maximize
        # here we are minimizing, so take just DKL
        # we omit the factor of 1/2 here and in the reconstruction
        # error
        dkl = torch.mean((mu.pow(2) + logvar.exp() - 1 - logvar).sum(axis=1))

        # reconstruction error -- conditioned gaussian log likelihood
        # we make the "variational assumption" that p(x | z) has std=1
        # so that the only relevant term is the mse (after omitting the
        # half as above)
        mse_recon_full = F.mse_loss(x, recon_x, reduction="none")
        mse_recon = torch.mean(
            torch.sum(
                mse_recon_full,
                axis=tuple(range(1, mse_recon_full.ndim)),
            )
        )

        # supervised loss
        mse_labels_full = F.mse_loss(y, y_hat, reduction="none")
        mse_labels = torch.mean(
            torch.sum(
                mse_labels_full,
                axis=tuple(range(1, mse_labels_full.ndim)),
            )
        )

        # TODO total correlation? beta annealing?
        #      unsupervised latents index-code mutual information?

        loss = dkl + mse_recon + self.alpha * mse_labels
        return loss
