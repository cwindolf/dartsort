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
        variational=False,
        dipole=False,
    ):
        super(PTPVAE, self).__init__()
        self.variational = variational
        self.dipole = dipole

        self.latent_dim = 3
        self.local_geom = nn.Parameter(
            data=torch.tensor(local_geom, requires_grad=False),
            requires_grad=False,
        )

        self.encoder = encoder

        if variational:
            self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
            self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

    def dists(self, x, log_y, z):
        # B x C
        return torch.sqrt(
            # B x 1
            torch.exp(2 * log_y)[:, None]
            + 0.01
            # 1 x C - B x 1 = B x C
            + torch.square(self.local_geom[None, :, 0] - x[:, None])
            + torch.square(self.local_geom[None, :, 1] - z[:, None])
        )

    def dipole_x_beta(self, input_ptp, x, log_y, z):
        # B x C x 3
        duv = torch.stack(
            [
                x[:, None] - self.local_geom[None, :, 0],
                torch.broadcast_to(
                    torch.exp(2 * log_y)[:, None], input_ptp.shape
                ),
                z[:, None] - self.local_geom[None, :, 1],
            ],
            axis=2,
        )
        # B x C x 3
        X = duv / (torch.square(duv).sum(axis=2, keepdim=True) + 0.01)
        beta = torch.linalg.solve(
            torch.einsum("bci,bcj->bij", X, X),
            torch.einsum("bck,bc->bk", X, input_ptp),
        )
        return X, beta

    def localize(self, x):
        mu, logvar = self.encode(x)
        x, log_y, z = mu.T
        return x, log_y, z

    def encode(self, x):
        h = self.encoder(x)
        if self.variational:
            return self.fc_mu(h), self.fc_logvar(h)
        else:
            return h, None

    def encode_with_alpha(self, input_ptp):
        h = self.encoder(input_ptp)
        mu = self.fc_mu(h) if self.variational else h
        x, log_y, z = mu.T

        if self.dipole:
            X, beta = self.dipole_x_beta(input_ptp, x, log_y, z)
            alpha = torch.square(beta).sum(axis=1) ** 0.5
        else:
            q = 1 / self.dists(x, log_y, z)
            alpha = (input_ptp * q).sum(1) / torch.square(q).sum(1)

        return x, torch.exp(log_y), z, alpha

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, input_ptp, z):
        x, log_y, z = z.T

        if self.dipole:
            X, beta = self.dipole_x_beta(input_ptp, x, log_y, z)
            return torch.einsum("bck,bk->bc", X, beta)
        else:
            q = 1 / self.dists(x, log_y, z)
            alpha = (input_ptp * q).sum(1) / torch.square(q).sum(1)
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

    def loss(self, x, recon_x, mu, logvar):
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
