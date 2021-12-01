# helpful code references:
# github.com/AntixK/PyTorch-VAE/
# github.com/themattinthehatt/behavenet/blob/master/behavenet/models/vaes.py
# github.com/pytorch/examples/blob/master/vae/main.py

import torch
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
        super(PSVAE, self).__init__()

        latent_dim = unsupervised_latent_dim + supervised_latent_dim
        self.latent_dim = latent_dim
        self.unsupervised_latent_dim = unsupervised_latent_dim
        self.supervised_latent_dim = supervised_latent_dim

        self.alpha = alpha

        self.encoder = encoder
        self.decoder = decoder
        self.final_hidden_dim = self.encoder[-1].output_dim

        self.fc_mu = nn.Linear(self.final_hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.final_hidden_dim, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, self.final_hidden_dim)
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
        zs = z[:, : self.supervised_latent_dim]
        zu = z[:, self.supervised_latent_dim :]
        return zs, zu

    def decode(self, z):
        # TODO might need to add an activation?
        decoder_input = self.fc_decoder(z)
        return self.decoder(decoder_input)

    def forward(self, x):
        # print("forward x.shape", x.shape)
        mu, logvar = self.encode(x)
        # print("forward mu.shape", mu.shape, "logvar.shape", logvar.shape)
        z = self.reparametrize(mu, logvar)
        # print("forward z.shape", z.shape)
        zs, zu = self.split(z)
        # print("forward zs.shape", zs.shape, "zu.shape", zu.shape)
        y_hat = self.diag_y_hat(zs)
        # print("forward y_hat.shape", y_hat.shape)
        recon_x = self.decode(z)
        # print("forward recon_x.shape", recon_x.shape)
        return recon_x, y_hat, mu, logvar

    def loss(self, x, y, recon_x, y_hat, mu, logvar):
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
        # we omit the factor of 1/2 here and in errors below,
        # and we omit the 1.
        dkl = torch.mean(mu.pow(2) + logvar.exp() - logvar)

        # reconstruction error -- conditioned gaussian log likelihood
        # we make the "variational assumption" that p(x | z) has std=1
        # so that the only relevant term is the mse (after omitting the
        # half as above)
        mse_recon = F.mse_loss(x, recon_x)

        # supervised loss
        mse_labels = F.mse_loss(y, y_hat)

        # TODO total correlation? beta annealing?
        #      unsupervised latents index-code mutual information?

        loss = dkl + mse_recon + self.alpha * mse_labels
        loss_dict = {
            "dkl": dkl,
            "mse_recon": mse_recon,
            "mse_labels": mse_labels,
        }
        return loss, loss_dict
