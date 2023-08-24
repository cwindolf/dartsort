"""Deep version of linear_ae[_index].py
"""
import torch
from torch import nn
from torch.nn import functional as F


class RelocAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        local_geom,
        latent_dim=1,
        analytical_alpha=True,
        variational=False,
        dipole=False,
    ):
        super(RelocAE, self).__init__()
        self.variational = variational
        self.dipole = dipole

        self.latent_dim = latent_dim
        self.local_geom = nn.Parameter(
            data=torch.tensor(local_geom, requires_grad=False),
            requires_grad=False,
        )

        self.encoder = encoder
        self.decoder = decoder

        if variational:
            self.fc_mu = nn.Linear(latent_dim, latent_dim)
            self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    # -- structured decoder

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

    def decode(self, features, loc):
        x, log_y, z = loc.T

        # BK -> BTC
        unrelocated_output_wf = self.decoder(features)
        # BTC -> BC
        output_ptp = (
            torch.max(unrelocated_output_wf, dim=1)[0]
            - torch.min(unrelocated_output_wf, dim=1)[0]
        )

        if self.dipole:
            X, beta = self.dipole_x_beta(output_ptp, x, log_y, z)
            target_ptp = torch.einsum("bck,bk->bc", X, beta)
        else:
            q = 1 / self.dists(x, log_y, z)
            alpha = (output_ptp * q).sum(1) / torch.square(q).sum(1)
            target_ptp = alpha[:, None] * q

        return unrelocated_output_wf * (target_ptp / output_ptp)[:, None, :]

    # -- forward pass

    def encode(self, x):
        h = self.encoder(x)
        if self.variational:
            return self.fc_mu(h), self.fc_logvar(h)
        else:
            return h, None

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, loc):
        mu, logvar = self.encode(x)
        if logvar is not None:
            z = self.reparametrize(mu, logvar)
        else:
            z = mu
        recon_x = self.decode(z, loc)
        return recon_x, mu, logvar

    # -- training

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
