import torch
import torch.nn.functional as F
from torch import Tensor

from ..util.logging_util import get_logger
from ..util.py_util import panic

logger = get_logger(__name__)


class ScoreNet(torch.nn.Module):
    def __init__(
        self,
        n_before: int,
        n_after: int,
        n_score_channels: int,
        n_temporal: int = 8,
        n_moments: int = 4,
        spatial_mode: str = "full",
        hidden_dims: tuple[int, ...] = (32, 32),
        half_square: bool = False,
        signed_moments: bool = True,
        mask_features: bool = True,
        energy_powers: tuple[str, ...] = (),
        random_seed: int = 0,
    ):
        super().__init__()
        if spatial_mode not in ("full", "diag"):
            panic(f"{spatial_mode=}")
        for p in energy_powers:
            if p not in ("abs", "log1p"):
                panic(f"unknown energy power {p}")

        self.n_before = n_before
        self.n_after = n_after
        self.window = n_before + n_after
        self.n_temporal = n_temporal
        self.n_moments = n_moments
        self.n_score_channels = n_score_channels
        self.spatial_mode = spatial_mode
        self.half_square = half_square
        self.signed_moments = signed_moments
        self.mask_features = mask_features
        self.energy_powers = tuple(energy_powers)

        self.n_energies = n_temporal * (1 + half_square)
        self.n_quad_features = n_moments * self.n_energies
        self.n_head_features = (
            self.n_quad_features
            + len(self.energy_powers) * self.n_quad_features
            + signed_moments * (n_moments * n_temporal)
            + mask_features * n_moments
        )

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(random_seed)
            temporal = torch.randn(n_temporal, self.window).div_(self.window**0.5)
            # all pos so /n; later and above /sqrt(n) for mixed sign
            pool = torch.randn(n_moments, n_score_channels).div_(n_score_channels)
            pool[0] = 1.0 / n_score_channels
            pool_signed = torch.randn(n_moments, n_score_channels).div_(
                n_score_channels**0.5
            )
            # shape stats for decay estimation
            pool_powers = torch.randn(
                len(self.energy_powers), n_moments, n_score_channels
            ).div_(n_score_channels)

        self.temporal_weight = torch.nn.Parameter(temporal)
        self.pool = torch.nn.Parameter(pool)
        self.pool_signed = torch.nn.Parameter(pool_signed)
        self.pool_powers = torch.nn.Parameter(pool_powers)
        if spatial_mode == "full":
            self.spatial_weight = torch.nn.Parameter(torch.eye(n_score_channels))
        else:
            self.spatial_weight = torch.nn.Parameter(torch.ones(n_score_channels))

        self.quad_weights = torch.nn.Parameter(
            torch.full((self.n_quad_features,), 1.0 / self.n_quad_features)
        )
        self.bias = torch.nn.Parameter(torch.zeros(()))

        if hidden_dims:
            layers: list[torch.nn.Module] = []
            dim = self.n_head_features
            for h in hidden_dims:
                layers += [torch.nn.Conv1d(dim, h, 1), torch.nn.PReLU()]
                dim = h
            layers.append(torch.nn.Conv1d(dim, 1, 1))
            torch.nn.init.zeros_(layers[-1].weight)  # ty: ignore[invalid-argument-type]
            torch.nn.init.zeros_(layers[-1].bias)  # ty: ignore[invalid-argument-type]
            self.head = torch.nn.Sequential(*layers)
        else:
            self.head = None

        self.register_buffer("whitening_kernel", None)
        self.register_buffer("local_whiteners", None)
        self.register_buffer("baked_temporal", None)
        self.register_buffer("baked_spatial", None)

    def set_whiteners(
        self, local_whiteners: Tensor, whitening_kernel: Tensor | None
    ) -> None:
        assert local_whiteners.shape[1:] == (
            self.n_score_channels,
            self.n_score_channels,
        )
        self.whitening_kernel = whitening_kernel
        self.local_whiteners = local_whiteners
        self.baked_temporal = self.baked_spatial = None

    def bake(self) -> None:
        assert not self.training
        with torch.no_grad():
            self.baked_temporal = self.effective_temporal()
            self.baked_spatial = self.effective_spatial()

    def attach_persistent_buffers(self, state_dict: dict, prefix: str) -> None:
        """Helps with None-initialized stuff when loading from state dict. Probably a bad pattern."""
        device = self.temporal_weight.device
        for name in list(self._buffers):
            if name in self._non_persistent_buffers_set:
                continue
            key = f"{prefix}{name}"
            if key in state_dict and self._buffers[name] is None:
                setattr(self, name, torch.empty_like(state_dict[key], device=device))

    def train(self, mode: bool = True) -> "ScoreNet":
        if mode:
            self.baked_temporal = self.baked_spatial = None
        return super().train(mode)

    @property
    def temporal_pad(self) -> int:
        if self.whitening_kernel is None:
            return 0
        return (self.whitening_kernel.shape[0] - 1) // 2

    @property
    def trough_offset(self) -> int:
        """Output time t aligned to input trough at index t + trough_offset"""
        return self.n_before + self.temporal_pad

    @property
    def receptive_field(self) -> int:
        return self.window + 2 * self.temporal_pad

    def effective_temporal(self) -> Tensor:
        if self.baked_temporal is not None:
            return self.baked_temporal
        k = self.whitening_kernel
        if k is None:
            return self.temporal_weight
        k = k.flip(0)[None, None].to(self.temporal_weight)
        f = self.temporal_weight[:, None]
        return F.conv1d(f, k, padding=k.shape[-1] - 1).squeeze(1)

    def effective_spatial(self) -> Tensor:
        if self.baked_spatial is not None:
            return self.baked_spatial
        w = self.local_whiteners
        if w is None:
            panic()
        w = w.to(self.spatial_weight)
        if self.spatial_mode == "diag":
            return self.spatial_weight[None] * torch.diagonal(w, dim1=1, dim2=2)
        return torch.einsum("ab,cbd->cad", self.spatial_weight, w)

    # -- stages

    def apply_temporal(self, waveforms: Tensor) -> Tensor:
        """(n, c, t) -> (n, c, n_temporal, t - receptive_field + 1)"""
        n, c, t = waveforms.shape
        weight = self.effective_temporal()
        out = F.conv1d(waveforms.reshape(n * c, 1, t), weight[:, None])
        return out.view(n, c, self.n_temporal, -1)

    def apply_spatial(self, features: Tensor, spatial_op: Tensor) -> Tensor:
        if self.spatial_mode == "diag":
            return features * spatial_op[:, :, None, None]
        return torch.einsum("nab,nbkt->nakt", spatial_op, features)

    def apply_moments(self, features: Tensor, mask: Tensor | None) -> Tensor:
        """(n, c, k, t) -> (n, n_head_features, t)"""
        if self.half_square:
            rectified = torch.cat((features.relu(), (-features).relu()), dim=2)
        else:
            rectified = features
        energies = torch.einsum("mc,nckt->nmkt", self.pool, rectified.square())
        out = energies.flatten(1, 2)
        for i, power in enumerate(self.energy_powers):
            if power == "abs":
                g = rectified.abs()
            else:
                g = torch.log1p(rectified.square())
            moments = torch.einsum("mc,nckt->nmkt", self.pool_powers[i], g)
            out = torch.cat((out, moments.flatten(1, 2)), dim=1)
        if self.signed_moments:
            signed = torch.einsum("mc,nckt->nmkt", self.pool_signed, features)
            out = torch.cat((out, signed.flatten(1, 2)), dim=1)
        if self.mask_features:
            if mask is None:
                mask = features.new_ones(len(features), self.n_score_channels)
            pooled = torch.einsum("mc,nc->nm", self.pool, mask.to(features))
            out = torch.cat((out, pooled[:, :, None].expand(-1, -1, out.shape[2])), 1)
        return out

    def apply_head(self, features: Tensor) -> Tensor:
        """(n, n_head_features, t) -> (n, t)"""
        energies = features[:, : self.n_quad_features]
        quadratic = energies.mul(self.quad_weights[None, :, None]).sum(dim=1)
        score = signed_sqrt(quadratic) + self.bias
        if self.head is not None:
            scaled = torch.cat(
                (signed_sqrt(energies), features[:, self.n_quad_features :]), dim=1
            )
            score = score + self.head(scaled).squeeze(1)
        return score

    def forward_dense(
        self,
        traces: Tensor,
        score_channel_index: Tensor,
        mask: Tensor | None = None,
        time_chunk: int | None = None,
    ) -> Tensor:
        """(t, n_channels) -> (n_channels, t - receptive_field + 1)

        out[:,i] aligned to input trough at i + n_before + temporal_pad
        """
        t = traces.shape[0]
        n_out = t - self.receptive_field + 1
        if time_chunk is None or time_chunk >= n_out:
            return self._dense_block(traces, score_channel_index, mask)

        out = []
        for i0 in range(0, n_out, time_chunk):
            i1 = min(n_out, i0 + time_chunk)
            block = traces[i0 : i1 + self.receptive_field - 1]
            out.append(self._dense_block(block, score_channel_index, mask))
        return torch.cat(out, dim=1)

    def _dense_block(
        self, traces: Tensor, score_channel_index: Tensor, mask: Tensor | None
    ) -> Tensor:
        features = self.gather_temporal(traces, score_channel_index)
        return self.score_from_features(features, mask=mask)

    def gather_temporal(self, traces: Tensor, score_channel_index: Tensor) -> Tensor:
        """(t, n) -> (n, c, n_temporal, t - receptive_field + 1)"""
        weight = self.effective_temporal()
        features = F.conv1d(traces.T[:, None], weight[:, None])
        features = torch.cat((features, features.new_zeros(1, *features.shape[1:])))
        return features[score_channel_index]

    def score_from_features(
        self,
        features: Tensor,
        channels: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """(n, c, n_temporal, t) temporal features on main `channels`"""
        spatial_op = self.effective_spatial()
        if channels is not None:
            spatial_op = spatial_op[channels]
        features = self.apply_spatial(features, spatial_op)
        return self.apply_head(self.apply_moments(features, mask))

    def forward(
        self, waveforms: Tensor, channels: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        """(n, c, t) neighborhoods on main channels -> (n, t - receptive_field + 1)"""
        return self.score_from_features(self.apply_temporal(waveforms), channels, mask)


def signed_sqrt(x: Tensor, eps: float = 1e-3) -> Tensor:
    return x.sign() * ((x.abs() + eps).sqrt() - eps**0.5)
