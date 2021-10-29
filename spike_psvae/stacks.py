import numpy as np
from torch import nn


def linear_module(in_dim, out_dim, batchnorm=True):
    if batchnorm:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(),
        )
    else:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(),
        )


def linear_encoder(in_dim, hidden_dims, n_latents, batchnorm=True):
    adims = [in_dim, *hidden_dims]
    bdims = [*hidden_dims, n_latents]
    return nn.Sequential(
        nn.Flatten(),
        *[
            linear_module(a, b, batchnorm=batchnorm)
            for a, b in zip(adims, bdims)
        ]
    )


def linear_decoder(n_latents, hidden_dims, out_shape, batchnorm=True):
    out_dim = np.prod(out_shape)
    adims = [n_latents, *hidden_dims]
    bdims = [*hidden_dims, out_dim]
    return nn.Sequential(
        *[
            linear_module(a, b, batchnorm=batchnorm)
            for a, b in zip(adims, bdims)
        ],
        nn.Unflatten(1, out_shape)
    )
