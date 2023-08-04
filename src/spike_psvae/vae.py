# helpful code references:
# github.com/AntixK/PyTorch-VAE/
# github.com/themattinthehatt/behavenet/blob/master/behavenet/models/vaes.py
# github.com/pytorch/examples/blob/master/vae/main.py

import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        beta=1.0,
        variational=True,
    ):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = encoder
        self.decoder = decoder
        self.variational = variational
        self.beta = beta

        if self.variational:
            self.fc_mu = nn.Linear(latent_dim, latent_dim)
            self.fc_logvar = nn.Linear(latent_dim, latent_dim)

    def encode(self, x):
        h = self.encoder(x)
        if self.variational:
            return self.fc_mu(h), self.fc_logvar(h)
        else:
            return h, None

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        # TODO might need to add an activation?
        return self.decoder(z)

    def forward(self, x):
        # print("forward x.shape", x.shape)
        mu, logvar = self.encode(x)
        # print("forward mu.shape", mu.shape, "logvar.shape", logvar.shape)
        if self.variational:
            z = self.reparametrize(mu, logvar)
        else:
            z = mu
        # print("forward z.shape", z.shape)
        recon_x = self.decode(z)
        # print("forward recon_x.shape", recon_x.shape)
        return recon_x, mu, logvar

    def loss(self, x, recon_x, mu, logvar):
        # print(
        #    "loss \n\t- x.shape", x.shape,
        #    "\n\t- y.shape", y.shape,
        #     "\n\t- recon_x.shape",
        #     recon_x.shape,
        #     "\n\t- y_hat.shape", y_hat.shape,
        #     "\n\t- mu.shape", mu.shape,
        #     "\n\t- logvar.shape", logvar.shape,
        # )
        # mean over batches, sum over data dims

        # -KL divergence to iid standard normal
        # 1312.6114 appendix B
        # note, -DKL is in ELBO, which we want to maximize.
        # here, we are minimizing, so take just DKL.
        # we omit the factor of 1/2 here and in errors below
        dkl = 0
        if self.variational:
            dkl = torch.mean((mu.pow(2) + logvar.exp() - 1 - logvar).sum(axis=1))

        # reconstruction error -- conditioned gaussian log likelihood
        # we make the "variational assumption" that p(x | z) has std=1
        # so that the only relevant term is the mse (after omitting the
        # half as above)
        mse_recon = F.mse_loss(x, recon_x)

        loss = self.beta * dkl + mse_recon
        loss_dict = {
            "dkl": dkl,
            "mse_recon": mse_recon,
        }
        return loss, loss_dict
