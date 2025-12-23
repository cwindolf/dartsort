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

from pathlib import Path
from typing import Literal, Self

import numpy as np
import torch
import torch.nn.functional as F
from spikeinterface.core import BaseRecording
from torch import Tensor

from ...templates import TemplateData
from ...templates.template_util import (
    shared_basis_compress_templates,
    singlechan_alignments,
)
from ...util.internal_config import ComputationConfig, MatchingConfig
from ...util.interpolation_util import (
    FullProbeInterpolator,
    InterpolationParams,
    bake_interpolation_1d,
    default_interpolation_params,
)
from ...util.logging_util import get_logger
from ...util.job_util import ensure_computation_config
from ...util.py_util import databag
from ...util.waveform_util import upsample_singlechan_torch
from .matching_base import (
    ChunkTemplateData,
    MatchingPeaks,
    MatchingTemplates,
    subtract_precomputed_pconv,
)


logger = get_logger(__name__)

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
        trough_offset_samples: int,
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
        super().__init__()
        self.upsampling = up_factor > 1
        self.up_method = up_method
        self.interpolating = motion_est is not None

        # validation / shape documentation
        assert temporal_comps.ndim == 2
        assert self.interpolating == (motion_est is not None)
        self.n_units, rank, n_channels = spatial_sing.shape
        assert rank == temporal_comps.shape[0]
        self.spike_length_samples = temporal_comps.shape[1]
        assert trough_offset_samples <= self.spike_length_samples

        if self.interpolating:
            assert rgeom is not None
            logger.dartsortdebug("Drifty matching will interpolate.")
            self.erp = FullProbeInterpolator(
                geom=geom,
                rgeom=rgeom,
                neighborhood_radius=interp_neighborhood_radius,
                motion_est=motion_est,
                params=drift_interp_params,
            )
        else:
            logger.dartsortdebug("No interpolation in matching.")
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

        assert temporal_comps.shape == (rank, self.spike_length_samples)
        self.register_buffer("temporal_comps", temporal_comps.contiguous())
        assert up_temporal_comps.shape == (rank, up_factor, self.spike_length_samples)
        self.register_buffer_or_none(
            "up_temporal_comps", up_temporal_comps.contiguous()
        )
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
        self.register_buffer("rank_ix", torch.arange(rank, device=device))
        self.register_buffer("conv_lags", torch.arange(-t + 1, t, device=device))
        offset = torch.asarray(trough_offset_samples, device=device)
        self.register_buffer("trough_offset_samples", offset)

    @classmethod
    def _from_config(
        cls,
        save_folder: Path,
        recording: BaseRecording,
        template_data: TemplateData,
        matching_cfg: MatchingConfig,
        computation_cfg: ComputationConfig | None = None,
        motion_est=None,
        overwrite: bool = False,
        dtype=torch.float,
    ) -> Self:
        del overwrite, save_folder  # I don't save anything.

        computation_cfg = ensure_computation_config(computation_cfg)
        device = computation_cfg.actual_device()

        unit_ids = torch.asarray(template_data.unit_ids, device=device)
        geom = torch.asarray(recording.get_channel_locations())
        geom = geom.to(device=device, dtype=dtype)
        rgeom = torch.asarray(template_data.registered_geom)
        rgeom = rgeom.to(device=device, dtype=dtype)

        shared_basis_temps = shared_basis_compress_templates(
            template_data,
            min_channel_amplitude=matching_cfg.template_min_channel_amplitude,
            rank=matching_cfg.template_svd_compression_rank,
            computation_cfg=computation_cfg,
        )
        temporal_comps = torch.asarray(shared_basis_temps.temporal_components)
        spatial_sing = torch.asarray(shared_basis_temps.spatial_singular)

        return cls(
            temporal_comps=temporal_comps.to(device=device, dtype=dtype),
            spatial_sing=spatial_sing.to(device=device, dtype=dtype),
            motion_est=motion_est,
            geom=geom,
            trough_offset_samples=template_data.trough_offset_samples,
            unit_ids=unit_ids,
            rgeom=rgeom,
            up_factor=matching_cfg.template_temporal_upsampling_factor,
            up_method=matching_cfg.up_method,
            interp_up_radius=matching_cfg.upsampling_radius,
            drift_interp_params=matching_cfg.drift_interp_params,
            interp_neighborhood_radius=matching_cfg.drift_interp_neighborhood_radius,
            refractory_radius_frames=matching_cfg.refractory_radius_frames,
            device=device,
        )

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
        trough_shifts = _calc_trough_shifts(
            spatial_sing=spatial_sing,
            main_channels=main_channels,
            up_temporal_comps=self.b.up_temporal_comps,
            trough_offset_samples=self.trough_offset_samples,
        )
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
            up_major_temporal_comps=self.b.up_major_temporal_comps,
            spatial_sing=spatial_sing,
            pconv=pconv,
            time_ix=self.b.time_ix,
            chan_ix=self.b.chan_ix,
            rank_ix=self.b.rank_ix,
            conv_lags=self.b.conv_lags,
            refrac_ix=self.b.refrac_ix,
            padded_spatial_sing=padded_spatial_sing,
            up_data=self.up_data,
            up_trough_shifts=trough_shifts,
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
    up_major_temporal_comps: Tensor
    spatial_sing: Tensor
    padded_spatial_sing: Tensor
    pconv: Tensor

    time_ix: Tensor
    chan_ix: Tensor
    rank_ix: Tensor
    refrac_ix: Tensor
    conv_lags: Tensor
    up_data: "UpsamplingData | None"
    up_trough_shifts: Tensor

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
        if not peaks.n_spikes:
            return
        assert peaks.times is not None
        assert peaks.template_inds is not None
        if peaks.up_inds is not None:
            tempc = self.up_major_temporal_comps[peaks.up_inds]
        else:
            tempc = self.temporal_comps
            tempc = tempc[None].broadcast_to(peaks.n_spikes, *tempc.shape)
        if peaks.scalings is None:
            batch_templates = torch.bmm(
                tempc.mT, self.spatial_sing[peaks.template_inds]
            )
        else:
            tempc = tempc * peaks.scalings[:, None, None]
            batch_templates = torch.bmm(
                tempc.mT, self.spatial_sing[peaks.template_inds]
            )
        n, t, c = batch_templates.shape
        time_ix = peaks.times[:, None] + self.time_ix[None, :]
        assert traces.shape[1] in (c, c + 1)
        assert time_ix.shape == (n, t)
        batch_templates = batch_templates.view(n * t, c)
        time_ix = time_ix.view(n * t)[:, None].broadcast_to(batch_templates.shape)
        if sign == -1:
            traces[:, :c].scatter_add_(dim=0, src=batch_templates._neg_view(), index=time_ix)
        elif sign == 1:
            traces[:, :c].scatter_add_(dim=0, src=batch_templates, index=time_ix)
        else:
            assert False

    def subtract_conv(
        self, conv: Tensor, peaks: "MatchingPeaks", padding=0, batch_size=256, sign=-1
    ):
        subtract_precomputed_pconv(
            conv=conv,
            pconv=self.pconv,
            peaks=peaks,
            padding=padding,
            conv_lags=self.conv_lags,
            sign=sign,
            batch_size=batch_size,
        )

    def get_clean_waveforms(
        self,
        peaks: "MatchingPeaks",
        channels: Tensor,
        channel_index: Tensor,
        add_into: Tensor | None = None,
    ):
        if not peaks.n_spikes:
            return add_into
        assert peaks.times is not None
        assert peaks.template_inds is not None
        spatial = self.padded_spatial_sing[
            peaks.template_inds[:, None, None],
            self.rank_ix[None, :, None],
            channel_index[channels, None, :],
        ]
        if peaks.scalings is not None:
            spatial.mul_(peaks.scalings[:, None, None])
        if peaks.up_inds is not None:
            tempc = self.up_major_temporal_comps[peaks.up_inds]
        else:
            tempc = self.temporal_comps
            tempc = tempc[None].broadcast_to(peaks.n_spikes, *tempc.shape)
        if add_into is None:
            return tempc.mT.bmm(spatial)
        else:
            return add_into.baddbmm_(tempc.mT, spatial)

    def _enforce_refractory(self, mask, peaks, offset=0, value=-torch.inf):
        if not peaks.n_spikes:
            return
        assert peaks.times is not None
        assert peaks.obj_template_inds is not None
        time_ix = peaks.times[:, None] + (self.refrac_ix[None, :] + offset)
        row_ix = peaks.obj_template_inds[:, None]
        mask[row_ix, time_ix] = value

    def trough_shifts(self, peaks: "MatchingPeaks") -> Tensor:
        if peaks.up_inds is None:
            assert self.up_trough_shifts.shape[1] == 1
            return self.up_trough_shifts[peaks.template_inds][:, 0]
        else:
            return self.up_trough_shifts[peaks.template_inds, peaks.up_inds]

    def fine_match(
        self,
        *,
        peaks: "MatchingPeaks",
        residual: Tensor,
        conv: Tensor,
        padding: int = 0,
    ) -> "MatchingPeaks":
        if not self.needs_fine_pass:
            return peaks
        if not peaks.n_spikes:
            return peaks
        if self.up_data is not None:
            scalings, up_inds, time_shifts, objs = _upsampling_fine_match(
                conv=conv,
                template_inds=peaks.template_inds,
                times=peaks.times,
                padding=padding,
                normsq=self.obj_normsq,
                scaling=self.scaling,
                inv_lambda=self.inv_lambda,
                scale_min=self.scale_min,
                scale_max=self.scale_max,
                up_zpad=self.up_data.zpad,
                objective_window=self.up_data.objective_window,
                up_ix=self.up_data.up_ix,
                interpolator=self.up_data.interpolator,
                up_time_shift=self.up_data.up_time_shift,
            )
        else:
            del residual
            raise NotImplementedError
            # scalings, up_inds, time_shifts, objs = _direct_fine_match()
        assert self.scaling == (scalings is not None)
        return MatchingPeaks(
            times=peaks.times + time_shifts,
            obj_template_inds=peaks.obj_template_inds,
            template_inds=peaks.template_inds,
            up_inds=up_inds,
            scalings=scalings,
            scores=objs,
        )

    def reconstruct_up_templates(self):
        return torch.einsum(
            "nrc,urt->nutc", self.spatial_sing.cpu(), self.up_major_temporal_comps.cpu()
        )


# -- helpers


@torch.jit.script
def _calc_trough_shifts(
    spatial_sing: Tensor,
    main_channels: Tensor,
    up_temporal_comps: Tensor,
    trough_offset_samples: Tensor,
) -> Tensor:
    main_sing = spatial_sing.take_along_dim(
        dim=2, indices=main_channels[:, None, None]
    )[:, :, 0]
    rank, up, t = up_temporal_comps.shape
    tcomps = up_temporal_comps.view(rank, up * t)
    main_traces_up = main_sing @ tcomps
    main_traces_up = main_traces_up.view(main_channels.shape[0], up, t)
    trough_shifts = singlechan_alignments(main_traces_up, dim=2)
    trough_shifts = trough_shifts.sub(trough_offset_samples)
    return trough_shifts


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
    uarange = np.arange(up_factor)
    up_ix = np.concatenate([uarange[: up_half + 1][::-1], uarange[-up_half:][::-1]])
    up_ix = torch.asarray(up_ix, device=device)

    # tricky part: at which time should we then subtract the upsampled template?
    up_time_shift = (up_tt > 0).to(dtype=torch.long)
    return objective_window, objective_tt, up_tt, up_ix, up_time_shift


@databag
class UpsamplingData:
    objective_window: Tensor
    objective_tt: Tensor
    up_tt: Tensor
    up_ix: Tensor
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
        up_ix=up_template_ix,
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
    pconv = torch.flip(pconv, dims=(3,))

    return pconv


@torch.jit.script
def full_shared_pconv(
    temporal_pconv: Tensor, spatial_sing: Tensor, batch_size: int = 64
) -> Tensor:
    rank, rank_, up, conv_len = temporal_pconv.shape
    n_units, rank__, chans = spatial_sing.shape
    assert rank == rank_ == rank__
    out = spatial_sing.new_empty((n_units, n_units, up, conv_len))
    spatial_sing_flat = spatial_sing.view(n_units * rank, chans)
    temporal_pconv_flat = temporal_pconv.view(rank * rank, up * conv_len)
    for i0 in range(0, n_units, batch_size):
        i1 = min(n_units, i0 + batch_size)
        chunksz = (i1 - i0) * n_units
        spatial_left = spatial_sing[i0:i1]
        spatial_outer = spatial_left.view((i1 - i0) * rank, chans) @ spatial_sing_flat.T
        spatial_outer = spatial_outer.view(i1 - i0, rank, n_units, rank)
        spatial_outer = spatial_outer.permute(0, 2, 1, 3).reshape(chunksz, rank * rank)
        torch.mm(
            spatial_outer,
            temporal_pconv_flat,
            out=out[i0:i1].view(chunksz, up * conv_len),
        )
    return out


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


# -- fine matching


@torch.jit.script
def _upsampling_fine_match(
    *,
    conv: Tensor,
    template_inds: Tensor,
    times: Tensor,
    padding: int,
    normsq: Tensor,
    scaling: bool,
    inv_lambda: Tensor,
    scale_min: Tensor,
    scale_max: Tensor,
    up_zpad: int,
    objective_window: Tensor,
    up_ix: Tensor,
    interpolator: Tensor,
    up_time_shift: Tensor,
) -> tuple[Tensor | None, Tensor, Tensor, Tensor]:
    # extract conv snippets
    conv_snips = conv[
        template_inds[:, None], times[:, None] + objective_window + padding
    ]

    # upsample conv
    if up_zpad:
        conv_snips = F.pad(conv_snips, (0, up_zpad))
    conv_up = conv_snips @ interpolator

    # compute objective
    if scaling:
        b = conv_up + inv_lambda
        a = normsq[template_inds] + inv_lambda
        scalings = (b / a[:, None]).clamp_(scale_min, scale_max)
        obj_up = 2.0 * scalings * b - scalings.square() * a[:, None] - inv_lambda
    else:
        scalings = None
        obj_up = 2.0 * conv_up - normsq[template_inds, None]

    # get best match and figure out what to do
    objs, up_best = obj_up.max(dim=1)
    upsampling_ixs = up_ix[up_best]
    time_shifts = up_time_shift[up_best]

    if scalings is not None:
        scalings = scalings.take_along_dim(dim=1, indices=up_best[:, None])[:, 0]

    return scalings, upsampling_ixs, time_shifts, objs


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
