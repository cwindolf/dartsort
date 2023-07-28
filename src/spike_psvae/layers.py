import math
import torch
from torch import nn
from typing import Tuple


class Permute(nn.Module):
    __constants__ = ["dims"]
    dims: Tuple[int]

    def __init__(self, *dims: int) -> None:
        super(Permute, self).__init__()
        self.dims = tuple(dims)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.permute(input, self.dims)

    def extra_repr(self) -> str:
        return ", ".join(map(str, self.dims))


class Squeeze(nn.Module):
    def __init__(self) -> None:
        super(Squeeze, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(input)


class Unsqueeze(nn.Module):
    __constants__ = ["dim"]
    dim: int

    def __init__(self, dim: int) -> None:
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.unsqueeze(input, self.dim)

    def extra_repr(self) -> str:
        return str(self.dim)


# modified from github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L44-L48  # noqa
# see also https://github.com/themattinthehatt/behavenet/blob/1cb36a655b78307efecca18d6cd9632566bc9308/behavenet/models/base.py#L70  # noqa


class DiagLinear(nn.Module):
    __constants__ = ["features"]
    features: int
    weight: torch.Tensor

    def __init__(
        self, features: int, bias: bool = True, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(DiagLinear, self).__init__()
        self.features = features
        self.weight = nn.Parameter(torch.empty(features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # for diagonal matrix, fan_in = 1. so, this is Kaiming.
        nn.init.uniform_(self.weight, -1.0, 1.0)
        if self.bias is not None:
            # maintaining this init as it is in nn.Linear
            bound = 1 / math.sqrt(self.features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # recall broadcasting operates on trailing axis, so if input
        # is B x N, this is diag mult on the N axis
        return input * self.weight + self.bias

    def extra_repr(self) -> str:
        return "features={}, bias={}".format(
            self.features, self.bias is not None
        )
