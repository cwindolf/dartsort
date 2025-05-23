import torch
import torch.nn.functional as F
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scaling = embed_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_length, embed_dim]
        q = self.q_proj(x)  # [batch_size, seq_length, embed_dim]
        k = self.k_proj(x)  # [batch_size, seq_length, embed_dim]
        v = self.v_proj(x)  # [batch_size, seq_length, embed_dim]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        # [batch_size, seq_length, seq_length]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights
        # [batch_size, seq_length, embed_dim]
        attn_output = torch.matmul(attn_weights, v)
        return self.out_proj(attn_output)


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm_kind="batchnorm"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm_kind = norm_kind

        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = get_norm(output_dim, norm_kind)
        self.relu = nn.ReLU()
        self.projection = None
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        residual = x
        out = self.linear(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.relu(out)
        if self.projection is not None:
            residual = self.projection(residual)
        return out + residual


class ConcatResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm_kind="batchnorm"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm_kind = norm_kind

        self.linear = nn.Linear(input_dim, output_dim)
        self.project_back = nn.Linear(input_dim + output_dim, output_dim)
        self.norm = get_norm(output_dim, norm_kind)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.relu(out)

        concatenated = torch.cat([x, out], dim=-1)
        projected = self.project_back(concatenated)
        return projected


class LinearResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, norm_kind=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.norm_kind = norm_kind

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.norm = get_norm(hidden_dim, norm_kind)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out + x


class UNetLinear(nn.Module):
    def __init__(
        self,
        input_dim,
        half_hidden_dims,
        nonlinearity,
        norm_kind=None,
    ):
        self.blocks_in = nn.ModuleList()
        self.blocks_out = nn.ModuleList()

        inner_gated = nonlinearity = torch.nn.GLU

        *in_dims, middle_dim = half_hidden_dims
        out_dims = in_dims[::-1]

        current_dim = input_dim
        out_dims_in = []
        for out_dim in half_hiddden_dims:
            block_layers = []
            block_layers.append(nn.Linear(current_dim, (1 + inner_gated) * out_dim))
            norm = get_norm((1 + inner_gated) * out_dim, norm_kind)
            if norm is not None:
                block_layers.append(norm)
            block_layers.append(nonlinearity())
            self.blocks_in.append(nn.Sequential(*block_layers))
            current_dim = out_dim
            out_dims_in.append(out_dim)

        block_layers = []
        block_layers.append(nn.Linear(current_dim, (1 + inner_gated) * middle_dim))
        norm = get_norm((1 + inner_gated) * middle_dim, norm_kind)
        if norm is not None:
            block_layers.append(norm)
        block_layers.append(nonlinearity())
        self.middle_block = nn.Sequential(*block_layers)
        current_dim = middle_dim

        for out_dim in out_dims:
            cat_dim = out_dims_in.pop()
            current_dim = current_dim + cat_dim

            block_layers = [Cat()]
            block_layers.append(nn.Linear(current_dim, (1 + inner_gated) * out_dim))
            norm = get_norm((1 + inner_gated) * out_dim, norm_kind)
            if norm is not None:
                block_layers.append(norm)
            block_layers.append(nonlinearity())
            self.blocks_out.append(nn.Sequential(*block_layers))
            current_dim = out_dim
        self.final_dim = current_dim
        assert not out_dims_in

    def forward(self, x):
        stack = []
        for b in self.blocks_in:
            x = b(x)
            stack.append(x)
        x = self.middle_block(x)
        for b in self.blocks_out:
            x = b((x, stack.pop()))
        return x


def get_mlp(
    input_dim,
    hidden_dims,
    output_dim,
    norm_kind="batchnorm",
    res_type="none",
    output_layer="linear",
    nonlinearity="ReLU",
    attention_layer=False,
    num_heads=4,
):
    layers = []
    if isinstance(nonlinearity, str):
        nonlinearity = getattr(torch.nn, nonlinearity)
    gated = nonlinearity == torch.nn.GLU

    if res_type == "blocks_concat":
        current_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(ConcatResidualBlock(current_dim, out_dim, norm_kind))
            current_dim = out_dim
            if attention_layer:
                layers.append(AttentionBlock(current_dim, num_heads=num_heads))
        final_dim = current_dim

    elif res_type == "blocks_add":
        input_dims = [input_dim, *hidden_dims[:-1]]
        for ind, outd in zip(input_dims, hidden_dims):
            layers.append(ResidualBlock(ind, outd, norm_kind))
            if attention_layer:
                layers.append(AttentionBlock(outd, num_heads=num_heads))
        final_dim = hidden_dims[-1] if hidden_dims else input_dim

    elif res_type == "blocks_linear":
        assert len(hidden_dims) > 1
        compression_dim = hidden_dims[0]
        layers.append(nn.Linear(input_dim, compression_dim))
        for hd in hidden_dims[1:]:
            layers.append(LinearResidualBlock(compression_dim, hd, norm_kind=norm_kind))
        final_dim = compression_dim

    elif res_type == "unet_linear":
        layers.append(
            UNetLinear(input_dim, hidden_dims, out_dim, nonlinearity)
        )
        final_dim = layers[-1].final_dim

    elif res_type in ("none", "outer"):
        # res_type == "none" is handled in get_waveform_mlp
        # res_type in get_mlp is the *inner* residual type in each block
        current_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, (1 + gated) * out_dim))
            norm = get_norm((1 + gated) * out_dim, norm_kind)
            if norm is not None:
                layers.append(norm)
            layers.append(nonlinearity())
            current_dim = out_dim
            if attention_layer:
                layers.append(AttentionBlock(current_dim, num_heads=num_heads))
        final_dim = current_dim

    else:
        raise ValueError(f"Unsupported res_type: {res_type}")

    if output_layer == "linear":
        layers.append(nn.Linear(final_dim, output_dim))
    elif output_layer == "gated_linear":
        layers.append(nn.Linear(final_dim, 2 * output_dim))
        layers.append(nn.GLU())
    else:
        assert False

    return nn.Sequential(*layers)


def get_waveform_mlp(
    spike_length_samples,
    n_input_channels,
    hidden_dims,
    output_dim,
    input_includes_mask=True,
    norm_kind="batchnorm",
    scaling=None,
    log_transform=False,
    channelwise_dropout_p=0.0,
    separated_mask_input=False,
    initial_conv_fullheight=False,
    final_conv_fullheight=False,
    return_initial_shape=False,
    res_type="none",
    output_layer="linear",
    nonlinearity="ReLU",
    attention_layer=False,
    num_heads=4,
):
    input_dim = n_input_channels * (spike_length_samples + input_includes_mask)
    layers = []
    if log_transform:
        layers.append(WaveformOnly(LogTransform()))
    
    if initial_conv_fullheight:
        # what Conv1d considers channels is actually time (conv1d is ncl).
        # so this is matmul over time, and kernel size is 1 to be separate over chans
        conv = nn.Conv1d(spike_length_samples, spike_length_samples, kernel_size=1)
        layers.append(WaveformOnly(conv))
        norm = get_norm(n_input_channels, norm_kind)
        if norm is not None:
            layers.append(
                WaveformOnly(nn.Sequential(Permute(0, 2, 1), norm, Permute(0, 2, 1)))
            )
        layers.append(WaveformOnly(nn.ReLU()))

    if separated_mask_input:
        layers.append(Cat(dim=1))

    if channelwise_dropout_p:
        layers.append(ChannelwiseDropout(channelwise_dropout_p))

    layers.append(nn.Flatten())
    mlp = get_mlp(
        input_dim,
        hidden_dims,
        output_dim,
        norm_kind=norm_kind,
        res_type=res_type,
        nonlinearity=nonlinearity,
        attention_layer=attention_layer,
        output_layer=output_layer,
        num_heads=num_heads,
    )
    layers.append(mlp)

    if return_initial_shape:
        layers.append(nn.Unflatten(-1, (spike_length_samples, n_input_channels)))
    if final_conv_fullheight:
        norm = get_norm(n_input_channels, norm_kind)
        if norm is not None:
            layers.append(nn.Sequential(Permute(0, 2, 1), norm, Permute(0, 2, 1)))
        layers.append(nn.ReLU())
        conv = nn.Conv1d(spike_length_samples, spike_length_samples, kernel_size=1)
        layers.append(conv)

    net = nn.Sequential(*layers)

    if scaling not in (None, "none"):
        net = Rescaling(net, scaling, waveform_only=True)

    if res_type == "outer":
        net = WaveformOnlyResidualForm(net)

    if log_transform:
        layers.append(WaveformOnly(ExpTransform()))

    return net


def get_norm(n_features, norm_kind=None):
    if norm_kind == "batchnorm":
        return nn.BatchNorm1d(n_features)
    if norm_kind == "layernorm":
        return nn.LayerNorm(n_features)
    assert norm_kind in ("none", None)
    return None


class ResidualForm(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        output = self.module(input)
        return input + output


class Rescaling(nn.Module):
    def __init__(self, module, scale_kind="std", waveform_only=False):
        super().__init__()
        self.scale_kind = scale_kind
        self.module = module
        self.waveform_only = waveform_only

    def forward(self, input):
        if self.waveform_only:
            input, masks = input
        dim = tuple(range(1, input.ndim))
        if self.scale_kind == "max":
            scales = input.abs().amax(dim=dim, keepdim=True)
        elif self.scale_kind == "std":
            scales = input.std(dim=dim, keepdim=True)
        else:
            assert False
        if self.waveform_only:
            output = self.module((input / scales, masks))
        else:
            output = self.module(input / scales)
        return output * scales


class LogTransform(nn.Module):
    def forward(self, input):
        sgn = torch.sign(input)
        log = torch.log1p(input.abs())
        return log * sgn


class ExpTransform(nn.Module):
    def forward(self, input):
        sgn = torch.sign(input)
        exp = torch.expm1(input.abs())
        return exp * sgn


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
        res = F.dropout1d(waveforms.permute(0, 2, 1), p=self.p, training=self.training)
        res = res.permute(0, 2, 1)
        return res


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
            Rescaling,
            LogTransform,
            ExpTransform,
            LinearResidualBlock,
            UNetLinear,
            nn.Flatten,
            nn.Linear,
            nn.Conv1d,
            nn.ReLU,
            nn.BatchNorm1d,
            nn.LayerNorm,
            nn.ELU,
            nn.PReLU,
            nn.GLU,
            nn.Unflatten,
        ]
    )
