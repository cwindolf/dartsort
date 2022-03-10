# helpful code references:
# github.com/AntixK/PyTorch-VAE/
# github.com/themattinthehatt/behavenet/blob/master/behavenet/models/vaes.py
# github.com/pytorch/examples/blob/master/vae/main.py

import torch
from torch import nn
from torch.nn import functional as F


class PTPVAE(nn.Module):
    def __init__(
        self,
        encoder,
        local_geom,
        analytical_alpha=True,
        variational=False,
    ):
        super(PTPVAE, self).__init__()
        self.variational = variational
        self.analytical_alpha = analytical_alpha
        self.latent_dim = 3 + analytical_alpha
        self.local_geom = local_geom

        self.encoder = encoder

        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        if variational:
            self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

    def dists(self, x, log_y, z):
        # B x C
        return torch.sqrt(
            # B x 1
            torch.exp(2 * log_y)[:, None]
            # 1 x C - B x 1 = B x C
            + torch.square(self.local_geom[None, :, 0] - x[:, None])
            + torch.square(self.local_geom[None, :, 1] - z[:, None])
        )

    def encode(self, x):
        h = self.encoder(x)
        if self.variational:
            return self.fc_mu(h), self.fc_logvar(h)
        else:
            return self.fc_mu(h), None

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, x, z):
        x, log_y, z = z.T
        q = self.dists(x, log_y, z)
        alpha = (x * q).sum(1) / torch.square(q).sum(1)
        return alpha[:, None] * q

    def forward(self, x):
        # print("forward x.shape", x.shape)
        mu, logvar = self.encode(x)
        # print("forward mu.shape", mu.shape, "logvar.shape", logvar.shape)
        if logvar is not None:
            z = self.reparametrize(mu, logvar)
        else:
            z = mu
        # print("forward z.shape", z.shape)
        recon_x = self.decode(x, z)
        # print("forward recon_x.shape", recon_x.shape)
        return recon_x, mu, logvar

    def loss(self, x, y, recon_x, y_hat, mu, logvar):
        # mean over batches, sum over data dims

        # reconstruction error -- conditioned gaussian log likelihood
        # we make the "variational assumption" that p(x | z) has std=1
        # so that the only relevant term is the mse (after omitting the
        # half as above)
        mse_recon = F.mse_loss(x, recon_x)
        loss = mse_recon
        loss_dict = {"mse_recon": mse_recon}

        # -KL divergence to iid standard normal
        # 1312.6114 appendix B
        # note, -DKL is in ELBO, which we want to maximize.
        # here, we are minimizing, so take just DKL.
        # we omit the factor of 1/2 here and in errors below
        if self.variational:
            dkl = torch.mean(
                (mu.pow(2) + logvar.exp() - 1 - logvar).sum(axis=1)
            )
            loss = loss + dkl
            loss_dict["dkl"] = dkl

        return loss, loss_dict
