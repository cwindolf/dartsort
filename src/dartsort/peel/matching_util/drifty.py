"""Template data for matching with on-the-fly drift handling

Features
 - Shared or individual template SVD compression
 - Amplitude scaling
 - Temporal upsampling
    - No YASS-style adaptive temporal compression, which adds overhead
 - No mutually-refractory group support yet (TODO if needed)
 - Pick your favorite interpolation strategy for drift handling. If it's not
   thin plate splines, then maybe you haven't heard of 'em?

Strategy
 - Pre-compute the convolutions of all pairs of temporal basis elements
   across all pairs of units
   - This is shared if the template basis is shared
 - Read off the pairwise convolutions by multiplication with the spatial basis
   as needed
   - This can be precomputed if there's no drift

Coarse-to-fine approach
 - Objective is amplitude-scaled but not temporally upsampled
 - Separate notions of "coarse templates" (not upsampled) and "fine templates"
   (upsampled)
 - For pairwise convolutions, we need the convolution of coarse library with
   the fine library
 - At matching time, we propose peaks coarsely and then refine them
   - TODO how to do the fine step most efficiently? Can probably do the inner
     product more efficiently than I did before...

TODO: MatchingPeaks may need to be tweaked or abstracted?
"""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from ...util.interpolation_util import (
    InterpolationParams,
    FullProbeInterpolator,
    bake_interpolation_1d,
    default_interpolation_params,
)
from ...util.py_util import databag
from ...util.spiketorch import add_at_
from ...util.waveform_util import upsample_singlechan_torch
from .matching_base import (
    ChunkTemplateData,
    MatchingPeaks,
    MatchingTemplates,
)

default_upsampling_params = InterpolationParams(sigma=1.0)


class DriftyMatchingTemplates(MatchingTemplates):
    template_type = "drifty"

    def __init__(
        self,
        *,
        temporal_comps: Tensor,
        spatial_sing: Tensor,
        motion_est,
        geom: Tensor,
        unit_ids: Tensor | None = None,
        rgeom: Tensor | None = None,
        up_factor: int = 1,
        up_method: Literal["interpolation", "keys3", "keys4", "direct"] = "keys4",
        interp_up_radius: int = 8,
        up_interp_params=default_upsampling_params,
        drift_interp_params: InterpolationParams = default_interpolation_params,
        interp_neighborhood_radius: float = 150.0,
        refractory_radius_frames: int = 10,
        device: torch.device,
    ):
        """
        Arguments
        ---------
        up_method:
            How to pick the upsampling index? If we have a coarse match at time t,
        """
        self.upsampling = up_factor > 1
        self.up_method = up_method
        self.interpolating = motion_est is not None

        # validation / shape documentation
        assert temporal_comps.ndim == 2
        assert self.interpolating == motion_est is not None
        self.n_units, rank, n_channels = spatial_sing.shape
        assert rank == temporal_comps.shape[0]
        self.spike_length_samples = temporal_comps.shape[0]

        if self.interpolating:
            assert rgeom is not None
            self.erp = FullProbeInterpolator(
                geom=geom,
                rgeom=rgeom,
                neighborhood_radius=interp_neighborhood_radius,
                motion_est=motion_est,
                params=drift_interp_params,
            )
        else:
            self.erp = None

        if self.upsampling and up_method != "direct":
            self.up_data = get_interp_upsampling_data(
                up_factor=up_factor,
                up_method=up_method,
                interp_up_radius=interp_up_radius,
                interp_params=up_interp_params,
                device=device,
            )
        else:
            self.up_data = None

        up_temporal_comps = upsample_singlechan_torch(
            temporal_comps, temporal_jitter=up_factor
        )
        temporal_pconv = shared_temporal_pconv(temporal_comps, up_temporal_comps)

        self.register_buffer("temporal_comps", temporal_comps)
        self.register_buffer_or_none("up_temporal_comps", up_temporal_comps)
        if up_temporal_comps is not None:
            up_major_temporal_comps = up_temporal_comps.permute(1, 0, 2).contiguous()
        else:
            up_major_temporal_comps = None
        self.register_buffer_or_none("up_major_temporal_comps", up_major_temporal_comps)
        self.register_buffer("spatial_sing", spatial_sing)
        self.register_buffer("temporal_pconv", temporal_pconv)
        if unit_ids is None:
            unit_ids = torch.arange(self.n_units, device=device)
        self.register_buffer("unit_ids", unit_ids)

        if not self.interpolating:
            # can precompute the full pconv in this case.
            pconv = full_shared_pconv(self.b.temporal_pconv, self.b.spatial_sing)
        else:
            pconv = None
        self.register_buffer_or_none("pconv", pconv)

        # indexing helpers
        t = self.spike_length_samples
        rr = refractory_radius_frames
        self.register_buffer("refrac_ix", torch.arange(-rr, rr + 1, device=device))
        self.register_buffer("time_ix", torch.arange(t, device=device))
        self.register_buffer("chan_ix", torch.arange(n_channels, device=device))
        self.register_buffer("conv_lags", torch.arange(-t + 1, t, device=device))

    def spatial_sing_at_time(self, t_s: float) -> Tensor:
        if not self.interpolating:
            return self.b.spatial_sing
        assert self.erp is not None
        return self.erp.interp_at_time(t_s=t_s, waveforms=self.b.spatial_sing)

    def data_at_time(
        self,
        t_s: float,
        scaling: bool,
        inv_lambda: float,
        scale_min: float,
        scale_max: float,
    ) -> ChunkTemplateData:
        spatial_sing = self.spatial_sing_at_time(t_s=t_s)
        normsq_by_chan = spatial_sing.square().sum(dim=1)
        main_channels = normsq_by_chan.argmax(dim=1)
        normsq = normsq_by_chan.sum(dim=1)
        if self.b.pconv is None:
            pconv = full_shared_pconv(self.b.temporal_pconv, spatial_sing)
        else:
            pconv = self.b.pconv
        padded_spatial_sing = F.pad(spatial_sing, (0, 1))
        return DriftyChunkTemplateData(
            spike_length_samples=self.spike_length_samples,
            unit_ids=self.b.unit_ids,
            main_channels=main_channels,
            obj_normsq=normsq,
            obj_n_templates=self.n_units,
            scaling=scaling,
            upsampling=self.upsampling,
            needs_fine_pass=self.upsampling,
            inv_lambda=torch.asarray(inv_lambda).to(normsq, non_blocking=True),
            scale_min=torch.asarray(scale_min).to(normsq, non_blocking=True),
            scale_max=torch.asarray(scale_max).to(normsq, non_blocking=True),
            temporal_comps=self.b.temporal_comps,
            temporal_comps_up=self.b.up_major_temporal_comps,
            spatial_sing=spatial_sing,
            pconv=pconv,
            time_ix=self.b.time_ix,
            chan_ix=self.b.chan_ix,
            rank_ix=self.b.rank_ix,
            conv_lags=self.b.conv_lags,
            refrac_ix=self.b.refrac_ix,
            padded_spatial_sing=padded_spatial_sing,
        )


@databag
class DriftyChunkTemplateData(ChunkTemplateData):
    spike_length_samples: int
    # for the full templates
    unit_ids: Tensor
    main_channels: Tensor
    # for obj templates
    obj_normsq: Tensor
    obj_n_templates: int
    upsampling: bool
    scaling: bool
    needs_fine_pass: bool
    inv_lambda: Tensor
    scale_min: Tensor
    scale_max: Tensor

    temporal_comps: Tensor
    temporal_comps_up: Tensor
    spatial_sing: Tensor
    padded_spatial_sing: Tensor
    pconv: Tensor

    time_ix: Tensor
    chan_ix: Tensor
    rank_ix: Tensor
    refrac_ix: Tensor
    conv_lags: Tensor

    # template grouping is not implemented, so this is always false.
    coarse_objective: bool = False

    def convolve(self, traces: Tensor, padding: int = 0, out: Tensor | None = None):
        out_len = traces.shape[1] + 2 * padding - self.spike_length_samples + 1
        if out is not None:
            assert out.shape == (self.obj_n_templates, out_len)
        return convolve_lowrank_shared(
            traces=traces,
            spatial_singular=self.spatial_sing,
            temporal_components=self.temporal_comps,
            padding=padding,
            out=out,
        )

    def subtract(self, traces: Tensor, peaks: "MatchingPeaks", sign: int = -1):
        batch_templates = torch.einsum(
            "n,nrc,nrt->ntc",
            peaks.scalings,
            self.spatial_sing[peaks.template_indices],
            self.temporal_comps_up[peaks.upsampling_indices],
        )
        time_ix = peaks.times[:, None, None] + self.time_ix[None, :, None]
        add_at_(
            traces, (time_ix, self.chan_ix[None, None, :]), batch_templates, sign=sign
        )

    def subtract_conv(
        self, conv: Tensor, peaks: "MatchingPeaks", padding=0, batch_size=256, sign=-1
    ):
        assert conv.shape[0] == self.pconv.shape[0]
        assert sign in (-1, 1)
        padded_lags = padding + self.conv_lags
        subtract_precomputed_pconv(
            conv=conv,
            pconv=self.pconv,
            template_indices=peaks.template_indices,
            upsampling_indices=peaks.upsampling_indices,
            scalings=peaks.scalings,
            times=peaks.times,
            padded_conv_lags=padded_lags,
            neg=sign == -1,
            batch_size=batch_size,
        )

    def get_clean_waveforms(
        self,
        peaks: "MatchingPeaks",
        channels: Tensor,
        channel_index: Tensor,
        add_into: Tensor | None = None,
    ):
        spatial = self.padded_spatial_sing[
            peaks.template_indices[:, None, None],
            self.rank_ix[None, :, None],
            channel_index[channels, None, :],
        ]
        temporal = self.temporal_comps_up[peaks.upsampling_indices]
        temporal.mul_(peaks.scalings[:, None, None])
        if add_into is None:
            return temporal.bmm(spatial)
        else:
            return add_into.baddbmm_(temporal, spatial)

    def _enforce_refractory(self, mask, peaks, offset=0, value=-torch.inf):
        time_ix = peaks.times[:, None] + (self.refrac_ix[None, :] + offset)
        row_ix = peaks.objective_template_indices[:, None]
        mask[row_ix, time_ix] = value

    def fine_match(
        self, *, peaks: "MatchingPeaks", residual: Tensor
    ) -> "MatchingPeaks":
        if not self.needs_fine_pass:
            return peaks


# -- helpers


def get_interp_upsampling_indices(
    *, up_factor: int, up_radius: int, device: torch.device
):
    """Time domain and indexing helper for upsampled objective methods (not direct)."""
    assert not (up_factor % 2)

    # window around the coarse peak in the objective which will be the input to upsampling
    objective_window = torch.arange(-up_radius, up_radius + 1, device=device)

    # above considered as the domain of a function
    objective_tt = objective_window.to(torch.float)

    # we want to get interpolated objective at the following domain
    # the idea is that the peak would have been at a different time point if the optimal
    # upsampled value weren't within 0.5 (inclusive) of the peak time
    up_half = up_factor // 2
    up_tt = torch.linspace(-0.5, 0.5, steps=2 * up_half + 1, device=device)

    # which upsampled template would correspond to a match at each upsampled time?
    uarange = torch.arange(up_factor, device=device)
    up_template_ix = torch.concatenate([uarange[-up_half:], uarange[: up_half + 1]])

    # tricky part: at which time should we then subtract the upsampled template?
    # if matched up_tt < 0, it's still the current time.
    # at up_tt == 0, same (actually it's just the coarse template!)
    # at up_tt > 0, you actually want to shift the time by + 1.
    up_time_shift = (up_tt > 0).to(dtype=torch.long)
    return objective_window, objective_tt, up_tt, up_template_ix, up_time_shift


@databag
class UpsamplingData:
    objective_window: Tensor
    objective_tt: Tensor
    up_tt: Tensor
    up_template_ix: Tensor
    up_time_shift: Tensor
    interpolator: Tensor
    zpad: int


def get_interp_upsampling_data(
    *,
    up_factor: int,
    up_method: Literal["interpolation", "keys3", "keys4"],
    interp_up_radius: int,
    interp_params: InterpolationParams = default_upsampling_params,
    device: torch.device,
):
    if up_method == "interpolation":
        up_radius = interp_up_radius
    elif up_method == "keys3":
        # keys3 only ever needs to look at 2 neighbors
        up_radius = 2
    elif up_method == "keys4":
        up_radius = 3
    else:
        assert False
    ixs = get_interp_upsampling_indices(
        up_factor=up_factor, up_radius=up_radius, device=device
    )
    objective_window, objective_tt, up_tt, up_template_ix, up_time_shift = ixs
    if up_method == "interpolation":
        kernel, zpad = bake_interpolation_1d(objective_tt, up_tt, interp_params)
    elif up_method == "keys3":
        kernel = _this_keys_kernel(3, up_tt)
        zpad = 0
    elif up_method == "keys4":
        kernel = _this_keys_kernel(4, up_tt)
        zpad = 0
    else:
        assert False
    return UpsamplingData(
        objective_window=objective_window,
        objective_tt=objective_tt,
        up_tt=up_tt,
        up_template_ix=up_template_ix,
        up_time_shift=up_time_shift,
        interpolator=kernel,
        zpad=zpad,
    )


# -- computing pairwise convolutions


def shared_temporal_pconv(temporal_comps: Tensor, up_temporal_comps: Tensor) -> Tensor:
    rank, t = temporal_comps.shape
    assert t >= rank
    rank_, up, t_ = up_temporal_comps.shape
    assert t == t_
    assert rank == rank_

    # NIL = rank, 1, t
    inp = temporal_comps[:, None, :]
    # OIL = rank * up, 1, t. rank major, not up major.
    fil = up_temporal_comps.reshape(rank * up, 1, t)
    # NOL = rank, rank * up, 2 * t - 1
    pconv = F.conv1d(input=inp, weight=fil, padding=t - 1)
    assert pconv.shape == (rank, rank * up, 2 * t - 1)
    pconv = pconv.view(rank, rank, up, 2 * t - 1)

    return pconv


def full_shared_pconv(temporal_pconv: Tensor, spatial_sing: Tensor):
    rank, rank_, up, conv_len = temporal_pconv.shape
    n_units, rank__, chans = spatial_sing.shape
    assert rank == rank_ == rank__
    return torch.einsum(
        "ipc,pqul,jqc->ijul", spatial_sing, temporal_pconv, spatial_sing
    )


# -- convolution


@torch.jit.script
def convolve_lowrank_shared(
    traces: Tensor,
    spatial_singular: Tensor,
    temporal_components: Tensor,
    padding: int = 0,
    out: Tensor | None = None,
):
    for q in range(temporal_components.shape[0]):
        # convolve recording with this rank's basis element
        tconv = F.conv1d(
            traces[:, None, :], temporal_components[q, None, None], padding=padding
        )
        assert tconv.shape[1] == 1
        tconv = tconv[:, 0]

        # multiply spatially and add into output
        if out is None:
            out = torch.mm(spatial_singular[:, q, :], tconv)
        else:
            out.addmm_(spatial_singular[:, q, :], tconv)

    return out


@torch.jit.script
def subtract_precomputed_pconv(
    conv: Tensor,
    pconv: Tensor,
    template_indices: Tensor,
    upsampling_indices: Tensor,
    scalings: Tensor,
    times: Tensor,
    padded_conv_lags: Tensor,
    neg: bool,
    batch_size: int = 128,
):
    ix_time = times[:, None] + padded_conv_lags[None, :]
    if neg:
        scalings = -scalings
    for i0 in range(0, conv.shape[0], batch_size):
        i1 = min(conv.shape[0], i0 + batch_size)
        batch = pconv[i0:i1, template_indices, upsampling_indices]
        batch.mul_(scalings[None, :, None])
        ix = ix_time.broadcast_to(batch.shape)
        conv[i0:i1].scatter_add_(dim=1, src=batch, index=ix)


# -- Keys' piecewise cubic interpolation impl (order 3 and 4)


def _this_keys_kernel(deg: Literal[3, 4], xx_: Tensor) -> Tensor:
    """Dense Keys' cubic convolution kernel for this setting (specific length, no boundary)."""
    # this function assumes that...
    assert xx_.ndim == 1
    assert xx_.shape[0] % 2
    assert torch.equal(xx_, -xx_.flip((0,)))
    assert (xx_.abs() < 1).all()

    # t     -2  -1   0   1   2
    # x      0   1   2   3   4
    #        |   |   |   |   |
    # left:       x_
    # |s|<   | 2 | 1 | 2 |   |
    # right:           x_
    # |s|<   |   | 2 | 1 | 2 |

    # the argument to u is s + j - k, where j is the left endpoint
    if deg == 3:
        jj = torch.tensor([1.0, 1, 2, 2, 2]).to(xx_) + (deg - 3)
    else:
        jj = torch.tensor([1.0, 1, 1, 2, 2, 2, 2]).to(xx_) + (deg - 3)
    kk = torch.arange(2 * (deg - 1) + 1, device=xx_.device, dtype=xx_.dtype)

    # the output kernel has input on rows, output on columns (multiplies on the right side)
    if deg == 3:
        xjj = torch.tensor([-1.0, -1, 0, 0, 0]).to(xx_)
    else:
        xjj = torch.tensor([-1.0, -1, -1, 0, 0, 0, 0]).to(xx_)
    u_arg = (jj - kk)[:, None] + (xx_ - xjj[:, None])
    if deg == 3:
        kernel = _keys3_u(u_arg)
    elif deg == 4:
        kernel = _keys4_u(u_arg)
    else:
        assert False

    # interp at |s| = 0 is exact, and I didn't handle that case in _keys_u
    kernel[:, xx_.shape[0] // 2] = 0.0
    kernel[deg - 1, xx_.shape[0] // 2] = 1.0

    return kernel


def _keys3_u(s: Tensor):
    s_abs = s.abs()
    case0 = s_abs < 1
    case1 = (s_abs < 2).logical_and_(case0.logical_not())
    case0_val = 1.5 * s_abs**3 - 2.5 * s_abs**2 + 1.0
    case1_val = -0.5 * s_abs**3 + 2.5 * s_abs**2 - 4.0 * s_abs + 2.0
    out = torch.zeros_like(s)
    out = torch.where(case0, case0_val, out)
    out = torch.where(case1, case1_val, out)
    return out


def _keys4_u(s: Tensor):
    s_abs = s.abs()
    case0 = s_abs < 1
    case1 = (s_abs < 2).logical_and_(case0.logical_not())
    case2 = (s_abs < 3).logical_and_(case0.logical_or(case1).logical_not())
    case0_val = (4 / 3) * s_abs**3 - (7 / 3) * s_abs**2 + 1.0
    case1_val = (-7 / 12) * s_abs**3 + 3 * s_abs**2 - (59 / 12) * s_abs + (15 / 6)
    case2_val = (1 / 12) * s_abs**3 - (2 / 3) * s_abs**2 + (21 / 12) * s_abs - (3 / 2)
    out = torch.zeros_like(s)
    out = torch.where(case0, case0_val, out)
    out = torch.where(case1, case1_val, out)
    out = torch.where(case2, case2_val, out)
    return out
