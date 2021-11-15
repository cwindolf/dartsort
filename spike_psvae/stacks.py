import numpy as np
from torch import nn

from .layers import Permute


# -- linear / mlp with batch normalization and leaky relu


def linear_module(in_dim, out_dim, batchnorm=True):
    if batchnorm:
        seq = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(),
        )
    else:
        seq = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(),
        )
    seq.output_dim = out_dim

    return seq


def linear_encoder(in_dim, hidden_dims, final_hidden_dim, batchnorm=True):
    adims = [in_dim, *hidden_dims]
    bdims = [*hidden_dims, final_hidden_dim]
    return nn.Sequential(
        nn.Flatten(),
        *[
            linear_module(a, b, batchnorm=batchnorm)
            for a, b in zip(adims, bdims)
        ],
    )


def linear_decoder(final_hidden_dim, hidden_dims, out_shape, batchnorm=True):
    out_dim = np.prod(out_shape)
    adims = [final_hidden_dim, *hidden_dims]
    bdims = [*hidden_dims, out_dim]
    return nn.Sequential(
        *[
            linear_module(a, b, batchnorm=batchnorm)
            for a, b in zip(adims, bdims)
        ],
        nn.Unflatten(1, out_shape),
    )


# -- a similar convolutional idea
# we'll have several convolution layers, followed by a convolution
# which is "full-height" along the time dimension (with several output
# channels). this will go into a linear layer to get to the latent shape.


def convolutional_module(
    in_channels, out_channels, kernel_size, batchnorm=True
):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding="valid")

    if batchnorm:
        return nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )
    else:
        return nn.Sequential(conv, nn.LeakyReLU())


def convtranspose_module(
    in_channels, out_channels, kernel_size, batchnorm=True
):
    # this padding corresponds to valid convs on the way in
    deconv = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, padding=kernel_size // 2
    )

    if batchnorm:
        return nn.Sequential(
            deconv,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )
    else:
        return nn.Sequential(deconv, nn.LeakyReLU())


def convolutional_encoder(
    in_shape, channels, kernel_sizes, final_hidden_dim, batchnorm=True
):
    # -- input shape logic
    # data should come in as T x channels. But, our probes have
    # two columns of electrodes, and we will treat this "2" as the
    # color/channel dimension on the input for convolutions.
    assert len(in_shape) == 2
    T, C = in_shape
    assert C % 2 == 0
    channel_radius = C // 2

    # -- more shape logic for the hidden layers
    in_channels = [2, *channels]
    out_channels = channels[:-1]
    # output shape of last layer under valid padding and unit stride
    last_h = T - sum(k - 1 for k in kernel_sizes)
    last_w = channel_radius - sum(k - 1 for k in kernel_sizes)
    last_c = out_channels[-1]
    assert last_w > 0  # you have too many layers for your kernel size

    return nn.Sequential(
        # BTC -> BTchannel_radius2
        nn.Unflatten(2, (channel_radius, 2)),
        # oh **** torch is NCHW not NHWC
        Permute(0, 3, 1, 2),
        # conv modules
        *[
            convolutional_module(inc, outc, ks, batchnorm=batchnorm)
            for inc, outc, ks in zip(in_channels, out_channels, kernel_sizes)
        ],
        # time collapse conv?
        # flatten and linear module for latents
        nn.Flatten(),
        linear_module(
            last_h * last_w * last_c, final_hidden_dim, batchnorm=batchnorm
        ),
    )


def convolutional_decoder(
    final_hidden_dim, channels, kernel_sizes, out_shape, batchnorm=True
):
    # -- "transposed" shape logic to the above
    assert len(out_shape) == 2
    T, C = out_shape
    assert C % 2 == 0
    channel_radius = C // 2

    first_h = T - sum(k - 1 for k in kernel_sizes)
    first_w = channel_radius - sum(k - 1 for k in kernel_sizes)
    first_c = channels[0]
    assert first_w > 0  # you have too many layers for your kernel size

    in_channels = channels
    out_channels = [*channels[:-1], 2]

    return nn.Sequential(
        linear_module(
            final_hidden_dim, first_h * first_w * first_c, batchnorm=batchnorm
        ),
        nn.Unflatten(1, (first_c, first_h, first_w)),
        # deconv modules
        *[
            convtranspose_module(inc, outc, ks, batchnorm=batchnorm)
            for inc, outc, ks in zip(in_channels, out_channels, kernel_sizes)
        ],
        Permute(0, 2, 3, 1),
        nn.Flatten(2),
    )


# -- command line arg helper


def netspec(spec, in_shape, batchnorm):
    in_dim = np.prod(in_shape)

    if spec.startswith("linear"):
        hidden_dims = list(map(int, spec.split(":")[1].split(",")))
        final_hidden_dim = int(spec.split(":")[2].split(","))

        encoder = linear_encoder(
            in_dim, hidden_dims, final_hidden_dim, batchnorm=batchnorm
        )
        decoder = linear_decoder(
            final_hidden_dim, hidden_dims[::-1], in_shape, batchnorm=batchnorm
        )
    elif spec.startswith("conv"):
        channels = list(map(int, spec.split(":")[1].split(",")))
        kernel_sizes = list(map(int, spec.split(":")[2].split(",")))
        final_hidden_dim = int(spec.split(":")[3])

        encoder = convolutional_encoder(
            in_shape,
            channels,
            kernel_sizes,
            final_hidden_dim,
            batchnorm=batchnorm,
        )
        decoder = convolutional_decoder(
            final_hidden_dim,
            channels[::-1],
            kernel_sizes[::-1],
            in_shape,
            batchnorm=batchnorm,
        )

    return encoder, decoder
