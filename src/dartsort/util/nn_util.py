import torch
import torch.nn.functional as F
from torch import nn

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=1, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scaling = embed_dim ** -0.5

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
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch_size, seq_length, seq_length]
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, seq_length, embed_dim]
        return self.out_proj(attn_output)















class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm_kind="batchnorm"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm_kind = norm_kind

        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = None
        if norm_kind == "batchnorm":
            self.norm = nn.BatchNorm1d(output_dim)
        elif norm_kind == "layernorm":
            self.norm = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU()

        self.projection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x):
        residual = x
        out = self.linear(x)
        if self.norm:
            out = self.norm(out)
        out = self.relu(out)
        if self.projection:
            residual = self.projection(residual)
        return out + residual



class ResidualBlock2(nn.Module):
    def __init__(self, input_dim, output_dim, norm_kind="batchnorm"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm_kind = norm_kind

        self.linear = nn.Linear(input_dim, output_dim)
        self.project_back = nn.Linear(input_dim + output_dim, output_dim)

        if norm_kind == "batchnorm":
            self.norm = nn.BatchNorm1d(output_dim)
        elif norm_kind == "layernorm":
            self.norm = nn.LayerNorm(output_dim)
        else:
            self.norm = None

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear(x) 
        if self.norm:
            out = self.norm(out)  
        out = self.relu(out)

        concatenated = torch.cat([x, out], dim=-1) 
        projected = self.project_back(concatenated) 
        return projected



def get_mlp(input_dim, hidden_dims, output_dim, norm_kind="batchnorm", 
            residual = False, residual_blocks = False, 
            res_type = "none", attention_layer=False, num_heads=4):
    layers = []

    if res_type == "blocks_concat":
        current_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(ResidualBlock2(current_dim, out_dim, norm_kind))
            current_dim = out_dim  
            if attention_layer:
                layers.append(AttentionBlock(current_dim, num_heads=num_heads))
        layers.append(nn.Linear(current_dim, output_dim))

    elif res_type == "blocks_add":
        input_dims = [input_dim, *hidden_dims[:-1]]
        for ind, outd in zip(input_dims, hidden_dims):
            layers.append(ResidualBlock(ind, outd, norm_kind))
            if attention_layer:
                layers.append(AttentionBlock(outd, num_heads=num_heads))
        final_dim = hidden_dims[-1] if hidden_dims else input_dim
        layers.append(nn.Linear(final_dim, output_dim))

    elif res_type == "none" or res_type == "outer":  
        current_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, out_dim))
            if norm_kind == "batchnorm":
                layers.append(nn.BatchNorm1d(out_dim))
            elif norm_kind == "layernorm":
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU())
            current_dim = out_dim
            if attention_layer:
                layers.append(AttentionBlock(current_dim, num_heads=num_heads))
        layers.append(nn.Linear(current_dim, output_dim))

    else:
        raise ValueError(f"Unsupported res_type: {res_type}")

    return nn.Sequential(*layers)




# def get_mlp(input_dim, hidden_dims, output_dim, norm_kind="batchnorm", residual=False, attention_layer=False, num_heads=1):
#     layers = []
#     current_dim = input_dim

#     for out_dim in hidden_dims:
#         if residual:
#             layers.append(ResidualBlock2(current_dim, out_dim, norm_kind))
#             current_dim = out_dim  
#         else:
#             layers.append(nn.Linear(current_dim, out_dim))
#             if norm_kind == "batchnorm":
#                 layers.append(nn.BatchNorm1d(out_dim))
#             elif norm_kind == "layernorm":
#                 layers.append(nn.LayerNorm(out_dim))
#             layers.append(nn.ReLU())
#             current_dim = out_dim

#         if attention_layer:
#             layers.append(AttentionBlock(current_dim, num_heads=num_heads))

#     layers.append(nn.Linear(current_dim, output_dim))

#     return nn.Sequential(*layers)






# def get_mlp(input_dim, hidden_dims, output_dim, norm_kind="batchnorm", residual=False, attention_layer=False, num_heads=1):
#     input_dims = [input_dim, *hidden_dims[:-1]]
#     layers = []

#     for ind, outd in zip(input_dims, hidden_dims):
#         if residual:
#             layers.append(ResidualBlock(ind, outd, norm_kind))
#         else:
#             layers.append(nn.Linear(ind, outd))
#             if norm_kind == "batchnorm":
#                 layers.append(nn.BatchNorm1d(outd))
#             elif norm_kind == "layernorm":
#                 layers.append(nn.LayerNorm(outd))
#             layers.append(nn.ReLU())

#         if attention_layer:
#             layers.append(AttentionBlock(outd, num_heads=num_heads))

#     final_dim = hidden_dims[-1] if hidden_dims else input_dim
#     layers.append(nn.Linear(final_dim, output_dim))

#     return nn.Sequential(*layers)


# V1 
def get_waveform_mlp(
    spike_length_samples,
    n_input_channels,
    hidden_dims,
    output_dim,
    input_includes_mask=True,
    norm_kind="batchnorm",
    channelwise_dropout_p=0.0,
    separated_mask_input=False,
    initial_conv_fullheight=False,
    final_conv_fullheight=False,
    return_initial_shape=False,
    residual=False,
    residual_blocks=False,
    res_type = "none", 
    attention_layer = False
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
        if norm_kind:
            if norm_kind == "batchnorm":
                norm = nn.BatchNorm1d(n_input_channels)
            elif norm_kind == "layernorm":
                norm = nn.LayerNorm(n_input_channels)
            layers.append(
                WaveformOnly(
                    nn.Sequential(
                        Permute(0, 2, 1),
                        norm,
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
    # layers.append(
    #     get_mlp(input_dim, hidden_dims, output_dim, norm_kind=norm_kind)
    # )
    layers.append(get_mlp(input_dim, hidden_dims, output_dim, norm_kind=norm_kind, res_type = res_type, attention_layer=False,num_heads=4))

    
    if return_initial_shape:
        layers.append(nn.Unflatten(-1, (spike_length_samples, n_input_channels)))
    if final_conv_fullheight:
        if norm_kind:
            if norm_kind == "batchnorm":
                norm = nn.BatchNorm1d(n_input_channels)
            elif norm_kind == "layernorm":
                norm = nn.LayerNorm(n_input_channels)
            layers.append(
                nn.Sequential(
                    Permute(0, 2, 1), norm, Permute(0, 2, 1)
                )
            )
        layers.append(nn.ReLU())
        layers.append(
            nn.Conv1d(spike_length_samples, spike_length_samples, kernel_size=1)
        )

    net = nn.Sequential(*layers)
    # if residual:
    #     net = WaveformOnlyResidualForm(net)

    if res_type == "outer":
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
            nn.LayerNorm,
        ]
    )
