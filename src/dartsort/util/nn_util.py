import torch
import torch.nn.functional as F
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


def get_waveform_mlp(
    spike_length_samples,
    n_input_channels,
    hidden_dims,
    output_dim,
    input_includes_mask=True,
    use_batchnorm=True,
    channelwise_dropout_p=0.0,
    separated_mask_input=False,
    initial_conv_fullheight=False,
    final_conv_fullheight=False,
    return_initial_shape=False,
    residual=False,
):
    input_dim = n_input_channels * (spike_length_samples + input_includes_mask)

    layers = []
    if initial_conv_fullheight:
        # what Conv1d considers channels is actually time (conv1d is ncl).
        # so this is matmul over time, and kernel size is 1 to be separate over chans
        layers.append(
            WaveformOnly(
                nn.Conv1d(spike_length_samples, spike_length_samples, kernel_size=1)
            )
        )
        if use_batchnorm:
            layers.append(
                WaveformOnly(
                    nn.Sequential(
                        Permute(0, 2, 1),
                        nn.BatchNorm1d(n_input_channels),
                        Permute(0, 2, 1),
                    )
                )
            )
        layers.append(WaveformOnly(nn.ReLU()))
    if separated_mask_input:
        layers.append(Cat(dim=1))
    if channelwise_dropout_p:
        layers.append(ChannelwiseDropout(channelwise_dropout_p))
    layers.append(nn.Flatten())
    layers.append(
        get_mlp(input_dim, hidden_dims, output_dim, use_batchnorm=use_batchnorm)
    )
    if return_initial_shape:
        layers.append(nn.Unflatten(-1, (spike_length_samples, n_input_channels)))
    if final_conv_fullheight:
        if use_batchnorm:
            layers.append(
                nn.Sequential(
                    Permute(0, 2, 1), nn.BatchNorm1d(n_input_channels), Permute(0, 2, 1)
                )
            )
        layers.append(nn.ReLU())
        layers.append(
            nn.Conv1d(spike_length_samples, spike_length_samples, kernel_size=1)
        )

    net = nn.Sequential(*layers)
    if residual:
        net = WaveformOnlyResidualForm(net)

    return net


class ResidualForm(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        output = self.module(input)
        return input + output


class WaveformOnlyResidualForm(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        waveforms, masks = inputs
        output = self.module(inputs)
        return waveforms + output


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


class Cat(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, inputs):
        return inputs.permute(*self.dims)


class WaveformOnly(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        waveforms, masks = inputs
        return self.module(waveforms), masks


# is this what they want us to do??
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(
        [
            ResidualForm,
            WaveformOnlyResidualForm,
            ChannelwiseDropout,
            Cat,
            Permute,
            WaveformOnly,
            nn.Flatten,
            nn.Linear,
            nn.Conv1d,
            nn.ReLU,
            nn.BatchNorm1d,
        ]
    )
