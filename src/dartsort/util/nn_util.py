from torch import nn


def get_mlp(input_dim, hidden_dims, output_dim, use_batchnorm=True):
    input_dims = [input_dim, *hidden_dims[:-1]]
    layers = []

    for ind, outd in zip(input_dims, hidden_dims):
        layers.append(nn.Linear(ind, outd))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(outd))
        layers.append(nn.ReLU())

    final_dim = hidden_dims[-1] if hidden_dims else input_dim
    layers.append(nn.Linear(final_dim, output_dim))

    return nn.Sequential(*layers)
