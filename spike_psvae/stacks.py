import numpy as np
from torch import nn

from .layers import Permute, Squeeze, Unsqueeze


# -- linear / mlp with batch normalization and leaky relu


def linear_module(in_dim, out_dim, batchnorm=True, activation=True):
    if batchnorm:
        seq = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            *((nn.LeakyReLU(),) if activation else ()),
        )
    else:
        seq = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            *((nn.LeakyReLU(),) if activation else ()),
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
    in_channels, out_channels, kernel_size, *, stride=1, batchnorm=True
):
    conv = nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, padding="valid"
    )

    if batchnorm:
        return nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )
    else:
        return nn.Sequential(conv, nn.LeakyReLU())


def convtranspose_module(
    in_channels, out_channels, kernel_size, *, stride=1, batchnorm=True, activation=True
):
    # this padding corresponds to valid convs on the way in
    deconv = nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride=stride
    )

    if batchnorm:
        return nn.Sequential(
            deconv,
            nn.BatchNorm2d(out_channels),
            *((nn.LeakyReLU(),) if activation else ()),
        )
    else:
        return nn.Sequential(
            deconv,
            *((nn.LeakyReLU(),) if activation else ()),
        )


def convolutional_encoder(
    in_shape, channels, kernel_sizes, final_hidden_dims, batchnorm=True
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
    in_channels = [2, *channels[:-1]]
    out_channels = channels
    # output shape of last layer under valid padding and unit stride
    last_h = T - sum(k - 1 for k in kernel_sizes)
    last_w = channel_radius - sum(k - 1 for k in kernel_sizes)
    last_c = out_channels[-1]
    assert last_w > 0  # you have too many layers for your kernel size
    print("enc", last_h, last_w, last_c, last_h * last_w * last_c)
    # final mlp shapes
    in_dims = [last_h * last_w * last_c, *final_hidden_dims[:-1]]
    out_dims = final_hidden_dims

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
        *[
            linear_module(
                ind, outd, batchnorm=batchnorm
            )
            for ind, outd in zip(in_dims, out_dims)
        ],
    )


def convolutional_decoder(
    final_hidden_dims, channels, kernel_sizes, out_shape, batchnorm=True
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
    print("dec", first_h, first_w, first_c, first_h * first_w * first_c)

    in_dims = final_hidden_dims
    out_dims = [*final_hidden_dims[1:], first_h * first_w * first_c]
    in_channels = channels
    out_channels = [*channels[1:], 2]

    return nn.Sequential(
        *[
            linear_module(
                ind, outd, batchnorm=batchnorm
            )
            for ind, outd in zip(in_dims, out_dims)
        ],
        nn.Unflatten(1, (first_c, first_h, first_w)),
        # deconv modules
        *[
            convtranspose_module(inc, outc, ks, batchnorm=batchnorm)
            for inc, outc, ks in zip(in_channels, out_channels, kernel_sizes)
        ],
        Permute(0, 2, 3, 1),
        nn.Flatten(2),
    )


# -- another convolutional idea
# let's not put x on the channels.
# instead, stride 2 on the first conv's width dim. use even ks.


def convb_encoder(
    in_shape,
    channels,
    kernel_sizes,
    final_hidden_dims,
    batchnorm=True,
):
    # -- input shape logic
    # data should come in as T x channels. But, our probes have
    # two columns of electrodes, and we will treat this "2" as the
    # color/channel dimension on the input for convolutions.
    assert len(in_shape) == 2
    T, C = in_shape

    # -- more shape logic for the hidden layers
    in_channels = [1, *channels[:-1]]
    out_channels = channels
    # output shape of last layer under valid padding and unit stride
    last_h = T - sum(k - 1 for k in kernel_sizes)
    last_w = ((C - kernel_sizes[0]) // 2 + 1) - sum(
        k - 1 for k in kernel_sizes[1:]
    )
    last_c = out_channels[-1]
    assert last_w > 0  # you have too many layers for your kernel size
    print("enc", last_h, last_w, last_c, last_h * last_w * last_c)
    strides = [(1, 2), *([1] * (len(channels) - 1))]
    # final mlp shapes
    in_dims = [last_h * last_w * last_c, *final_hidden_dims[:-1]]
    out_dims = final_hidden_dims

    return nn.Sequential(
        # BTC -> B1TC
        Unsqueeze(1),
        # conv modules
        *[
            convolutional_module(
                inc, outc, ks, stride=stride, batchnorm=batchnorm
            )
            for inc, outc, ks, stride in zip(
                in_channels, out_channels, kernel_sizes, strides
            )
        ],
        # time collapse conv?
        # flatten and linear module for latents
        nn.Flatten(),
        *[
            linear_module(
                ind, outd, batchnorm=batchnorm
            )
            for ind, outd in zip(in_dims, out_dims)
        ],
    )


def convb_decoder(
    final_hidden_dims,
    channels,
    kernel_sizes,
    out_shape,
    batchnorm=True,
):
    # -- "transposed" shape logic to the above
    assert len(out_shape) == 2
    T, C = out_shape

    first_h = T - sum(k - 1 for k in kernel_sizes)
    first_w = ((C - kernel_sizes[-1]) // 2 + 1) - sum(
        k - 1 for k in kernel_sizes[:-1]
    )
    first_c = channels[0]
    assert first_w > 0  # you have too many layers for your kernel size
    print("dec", first_h, first_w, first_c, first_h * first_w * first_c)

    in_dims = final_hidden_dims
    out_dims = [*final_hidden_dims[1:], first_h * first_w * first_c]
    in_channels = channels
    out_channels = [*channels[1:], 1]
    strides = [*([1] * (len(channels) - 1)), (1, 2)]

    return nn.Sequential(
        *[
            linear_module(
                ind, outd, batchnorm=batchnorm
            )
            for ind, outd in zip(in_dims, out_dims)
        ],
        nn.Unflatten(1, (first_c, first_h, first_w)),
        # deconv modules
        *[
            convtranspose_module(
                inc, outc, ks, stride=stride, batchnorm=batchnorm
            )
            for inc, outc, ks, stride in zip(
                in_channels, out_channels, kernel_sizes, strides
            )
        ],
        Squeeze(),
    )


def convc_encoder(
    in_shape,
    channels,
    kernel_sizes,
    final_hidden_dims,
    batchnorm=True,
):
    # -- input shape logic
    # data should come in as T x channels. But, our probes have
    # two columns of electrodes, and we will treat this "2" as the
    # color/channel dimension on the input for convolutions.
    assert len(in_shape) == 2
    T, C = in_shape

    # -- more shape logic for the hidden layers
    in_channels = [1, *channels[:-1]]
    out_channels = channels
    # output shape of last layer under valid padding and unit stride
    last_h = T - sum(k[0] - 1 for k in kernel_sizes)
    last_w = C - sum(k[1] - 1 for k in kernel_sizes)
    last_c = out_channels[-1]
    assert last_w > 0  # you have too many layers for your kernel size
    print("enc", last_h, last_w, last_c, last_h * last_w * last_c)
    # final mlp shapes
    in_dims = [last_h * last_w * last_c, *final_hidden_dims[:-1]]
    out_dims = final_hidden_dims

    return nn.Sequential(
        # BTC -> B1TC
        Unsqueeze(1),
        # conv modules
        *[
            convolutional_module(
                inc, outc, ks, batchnorm=batchnorm
            )
            for inc, outc, ks in zip(
                in_channels, out_channels, kernel_sizes
            )
        ],
        # time collapse conv?
        # flatten and linear module for latents
        nn.Flatten(),
        *[
            linear_module(
                ind, outd, batchnorm=batchnorm, activation=i < len(in_dims) - 1
            )
            for i, (ind, outd) in enumerate(zip(in_dims, out_dims))
        ],
    )


def convc_decoder(
    final_hidden_dims,
    channels,
    kernel_sizes,
    out_shape,
    batchnorm=True,
):
    # -- "transposed" shape logic to the above
    assert len(out_shape) == 2
    T, C = out_shape

    first_h = T - sum(k[0] - 1 for k in kernel_sizes)
    first_w = C - sum(k[1] - 1 for k in kernel_sizes)
    first_c = channels[0]
    assert first_w > 0  # you have too many layers for your kernel size
    print("dec", first_h, first_w, first_c, first_h * first_w * first_c)

    in_dims = final_hidden_dims
    out_dims = [*final_hidden_dims[1:], first_h * first_w * first_c]
    in_channels = channels
    out_channels = [*channels[1:], 1]

    return nn.Sequential(
        *[
            linear_module(
                ind, outd, batchnorm=batchnorm
            )
            for ind, outd in zip(in_dims, out_dims)
        ],
        nn.Unflatten(1, (first_c, first_h, first_w)),
        # deconv modules
        *[
            convtranspose_module(
                inc, outc, ks, batchnorm=batchnorm, activation=i < len(in_channels) - 1
            )
            for i, (inc, outc, ks) in enumerate(zip(
                in_channels, out_channels, kernel_sizes
            ))
        ],
        Squeeze(),
    )


# -- command line arg helper


def netspec(spec, in_shape, batchnorm):
    in_dim = np.prod(in_shape)

    if spec.startswith("linear:"):
        hidden_dims = list(map(int, spec.split(":")[1].split(",")))
        final_hidden_dim = int(spec.split(":")[2])

        encoder = linear_encoder(
            in_dim, hidden_dims, final_hidden_dim, batchnorm=batchnorm
        )
        decoder = linear_decoder(
            final_hidden_dim, hidden_dims[::-1], in_shape, batchnorm=batchnorm
        )
    elif spec.startswith("conv:"):
        channels = list(map(int, spec.split(":")[1].split(",")))
        kernel_sizes = list(map(int, spec.split(":")[2].split(",")))
        final_hidden_dims = list(map(int, spec.split(":")[3].split(",")))

        encoder = convolutional_encoder(
            in_shape,
            channels,
            kernel_sizes,
            final_hidden_dims,
            batchnorm=batchnorm,
        )
        decoder = convolutional_decoder(
            final_hidden_dims[::-1],
            channels[::-1],
            kernel_sizes[::-1],
            in_shape,
            batchnorm=batchnorm,
        )
    elif spec.startswith("convb:"):
        channels = list(map(int, spec.split(":")[1].split(",")))
        kernel_sizes = list(map(int, spec.split(":")[2].split(",")))
        final_hidden_dims = list(map(int, spec.split(":")[3].split(",")))

        encoder = convb_encoder(
            in_shape,
            channels,
            kernel_sizes,
            final_hidden_dims,
            batchnorm=batchnorm,
        )
        decoder = convb_decoder(
            final_hidden_dims[::-1],
            channels[::-1],
            kernel_sizes[::-1],
            in_shape,
            batchnorm=batchnorm,
        )

    return encoder, decoder
