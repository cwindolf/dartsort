from torch import nn
import torch.nn.functional as F


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


def get_waveform_mlp(
    spike_length_samples,
    n_input_channels,
    hidden_dims,
    output_dim,
    input_includes_mask=True,
    use_batchnorm=True,
    channelwise_dropout_p=0.0,
):
    input_dim = n_input_channels * (spike_length_samples + input_includes_mask)

    layers = []
    if channelwise_dropout_p:
        layers.append(ChannelwiseDropout(channelwise_dropout_p))
    layers.append(nn.Flatten())
    layers.append(get_mlp(input_dim, hidden_dims, output_dim, use_batchnorm=use_batchnorm))
    return nn.Sequential(*layers)


class ChannelwiseDropout(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, waveforms):
        return F.dropout1d(
            waveforms.permute(0, 2, 1),
            p=self.p,
            training=self.training,
        ).permute(0, 2, 1)