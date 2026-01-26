from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, Sequence, cast

import torch
import torch.nn.functional as F
from dredge.motion_util import MotionEstimate
from spikeinterface.core import BaseRecording
from torch import Tensor

from ...templates import TemplateData
from ...util.internal_config import ComputationConfig, MatchingConfig
from ...util.logging_util import DARTSORTVERBOSE, get_logger
from ...util.py_util import databag
from ...util.spiketorch import argrelmax, grab_spikes, ptp
from ...util.torch_util import BModule

logger = get_logger(__name__)
_extra_checks = logger.isEnabledFor(DARTSORTVERBOSE)


class MatchingTemplates(BModule):
    # subclasses should have their own template_type
    template_type = "base"
    # shared subclass registry for from_config()
    _registry = {}

    # these should be assigned in __init__. adding here for typing.
    spike_length_samples: int

    def __init_subclass__(cls):
        logger.dartsortverbose("Register templates type: %s", cls.template_type)
        cls._registry[cls.template_type] = cls

    @classmethod
    def from_config(
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
        global _extra_checks
        _extra_checks = logger.isEnabledFor(DARTSORTVERBOSE)
        if _extra_checks:
            logger.dartsortverbose(f"Extra checks enabled in matching.")
        return cls._registry[matching_cfg.template_type]._from_config(
            save_folder=save_folder,
            recording=recording,
            template_data=template_data,
            matching_cfg=matching_cfg,
            computation_cfg=computation_cfg,
            motion_est=motion_est,
            overwrite=overwrite,
            dtype=dtype,
        )

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
        raise NotImplementedError

    def data_at_time(
        self,
        t_s: float,
        scaling: bool,
        inv_lambda: float,
        scale_min: float,
        scale_max: float,
    ) -> "ChunkTemplateData":
        raise NotImplementedError


@dataclass(kw_only=True, frozen=True)
class MatchingTemplatesBuilder:
    """Helper so that the matching peeler can be a little lazy

    The matcher's from_config passes the builder, and build() is called in
    its precompute_peeling_data(). This exists so that resuming the sorter
    using run_peeler() logic after matching doesn't require doing a million
    SVDs and whatnot -- the peeler has to be constructed, but we can keep
    this stuff lazy.
    """

    recording: BaseRecording
    template_data: TemplateData
    matching_cfg: MatchingConfig
    motion_est: MotionEstimate | None = None
    dtype: torch.dtype = torch.float

    def build(
        self,
        save_folder: Path,
        computation_cfg: ComputationConfig | None,
        overwrite: bool = False,
    ) -> MatchingTemplates:
        return MatchingTemplates.from_config(
            save_folder=save_folder,
            recording=self.recording,
            template_data=self.template_data,
            matching_cfg=self.matching_cfg,
            computation_cfg=computation_cfg,
            motion_est=self.motion_est,
            dtype=self.dtype,
            overwrite=overwrite,
        )

    @property
    def spike_length_samples(self) -> int:
        return self.template_data.spike_length_samples


class ChunkTemplateData:
    # -- subclasses must assign the following properties that the matcher uses.
    spike_length_samples: int
    # for the full templates
    unit_ids: Tensor
    main_channels: Tensor
    # for obj templates
    obj_normsq: Tensor
    obj_n_templates: int
    coarse_objective: bool
    upsampling: bool
    scaling: bool
    needs_fine_pass: bool
    up_factor: int
    inv_lambda: Tensor
    scale_min: Tensor
    scale_max: Tensor

    # -- subclasses must implement

    def convolve(self, traces: Tensor, padding: int = 0, out: Tensor | None = None):
        raise NotImplementedError

    def subtract(self, traces: Tensor, peaks: "MatchingPeaks", sign: int = -1):
        raise NotImplementedError

    def subtract_conv(
        self, conv: Tensor, peaks: "MatchingPeaks", padding=0, batch_size=256, sign=-1
    ):
        raise NotImplementedError

    def get_clean_waveforms(
        self,
        peaks: "MatchingPeaks",
        channels: Tensor,
        channel_index: Tensor,
        add_into: Tensor | None = None,
    ):
        raise NotImplementedError

    def _enforce_refractory(
        self, mask: Tensor, peaks: "MatchingPeaks", offset: int = 0, value=-torch.inf
    ):
        raise NotImplementedError

    def fine_match(
        self,
        *,
        peaks: "MatchingPeaks",
        residual: Tensor,
        conv: Tensor,
        padding: int = 0,
    ) -> "MatchingPeaks":
        raise NotImplementedError

    # this one is just for debugging / unit testing
    def reconstruct_up_templates(self):
        raise NotImplementedError

    # -- super handles below

    def enforce_refractory(self, mask, peaks, offset=0):
        if not peaks.n_spikes:
            return
        self._enforce_refractory(mask, peaks, offset=offset, value=-torch.inf)

    def forget_refractory(self, mask, peaks, offset=0):
        if not peaks.n_spikes:
            return
        self._enforce_refractory(mask, peaks, offset=offset, value=0.0)

    def unsubtract(self, traces: Tensor, peaks: "MatchingPeaks"):
        return self.subtract(traces, peaks, sign=1)

    def unsubtract_conv(
        self, conv: Tensor, peaks: "MatchingPeaks", padding=0, batch_size=256
    ):
        return self.subtract_conv(
            conv=conv, peaks=peaks, padding=padding, batch_size=batch_size, sign=1
        )

    def obj_from_conv(
        self,
        *,
        conv: Tensor,
        out: Tensor,
        scalings_out: Tensor | None = None,
    ) -> Tensor:
        if self.scaling and self.inv_lambda == 0.0:
            assert scalings_out is not None
            return _free_coarse_objective(
                conv=conv, normsq=self.obj_normsq, out=out, scalings=scalings_out
            )
        elif self.scaling:
            assert scalings_out is not None
            return _scaled_coarse_objective(
                conv=conv,
                normsq=self.obj_normsq,
                out=out,
                scalings=scalings_out,
                inv_lambda=self.inv_lambda,
                scale_min=self.scale_min,
                scale_max=self.scale_max,
            )
        else:
            return torch.add(
                self.obj_normsq[:, None]._neg_view(), conv, alpha=2.0, out=out
            )

    def coarse_match(
        self,
        objective: Tensor,
        scalings: Tensor | None,
        thresholdsq: float,
        obj_arange: Tensor,
    ) -> "MatchingPeaks":
        objective_max, max_obj_template = objective.max(dim=0)
        times = argrelmax(
            x=objective_max,
            radius=self.spike_length_samples,
            threshold=thresholdsq,
            arange=obj_arange[: objective_max.numel()],
        )
        n_spikes = times.numel()
        if not n_spikes:
            return MatchingPeaks()

        objs = objective_max[times]
        template_inds = max_obj_template[times]
        if _extra_checks:
            assert (objective_max[times] >= thresholdsq).all()
        if self.scaling:
            assert scalings is not None
            scalings = scalings[template_inds, times]
        else:
            scalings = None
        if _extra_checks:
            assert (objs >= thresholdsq).all()
        if _extra_checks and scalings is not None:
            assert (scalings >= self.scale_min).all()
            assert (scalings <= self.scale_max).all()
        return MatchingPeaks(
            times=times,
            obj_template_inds=template_inds,
            template_inds=template_inds,
            scalings=scalings,
            scores=objs,
        )

    def get_collisioncleaned_waveforms(
        self,
        residual_padded: Tensor,
        peaks: "MatchingPeaks",
        channels: Tensor | Literal["template", "amplitude"],
        channel_index: Tensor,
        channel_selection_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not peaks.n_spikes:
            empty_channels = residual_padded.new_zeros(size=(0,), dtype=torch.long)
            empty_waveforms = residual_padded.new_zeros(size=())
            return empty_channels, empty_waveforms

        if channels == "template":
            channels = self.main_channels[peaks.template_inds]
            selecting_channels = False
            active_channel_index = channel_index
        elif torch.is_tensor(channels):
            selecting_channels = False
            active_channel_index = channel_index
        elif channels == "amplitude":
            selecting_channels = True
            assert channel_selection_index is not None
            active_channel_index = channel_selection_index
            channels = self.main_channels[peaks.template_inds]
        else:
            assert False

        times = peaks.times
        assert times is not None

        # get noise
        waveforms = grab_spikes(
            residual_padded,
            times,
            channels,
            active_channel_index,
            trough_offset=0,
            spike_length_samples=self.spike_length_samples,
            buffer=0,
            already_padded=True,
        )
        waveforms = self.get_clean_waveforms(
            peaks=peaks,
            channel_index=active_channel_index,
            channels=channels,
            add_into=waveforms,
        )

        if not selecting_channels:
            return channels, waveforms

        cix = ptp(waveforms).nan_to_num_(nan=-torch.inf).argmax(dim=1)
        assert channel_selection_index is not None
        sel = channel_selection_index[channels]
        channels = sel.take_along_dim(cix[:, None], dim=1)[:, 0]
        return self.get_collisioncleaned_waveforms(
            residual_padded=residual_padded,
            peaks=peaks,
            channel_index=channel_index,
            channels=channels,
        )


class PconvBase(BModule):
    def query(
        self,
        template_indices_a,
        template_indices_b,
        upsampling_indices_b=None,
        shifts_a=None,
        shifts_b=None,
        scalings_b=None,
    ):
        raise NotImplementedError


@databag
class MatchingPeaks:
    times: Tensor | None = None
    obj_template_inds: Tensor | None = None
    template_inds: Tensor | None = None
    up_inds: Tensor | None = None
    scalings: Tensor | None = None
    scores: Tensor | None = None
    time_shifts: Tensor | None = None

    if _extra_checks:

        def __post_init__(self):
            if self.times is None:
                assert self.obj_template_inds is None
                assert self.template_inds is None
                assert self.up_inds is None
                assert self.scalings is None
                assert self.scores is None
                assert self.time_shifts is None
            else:
                assert self.times.ndim == 1
                assert self.obj_template_inds is not None
                assert self.times.shape == self.obj_template_inds.shape
                assert self.template_inds is not None
                assert self.times.shape == self.template_inds.shape
                assert (self.up_inds is None) or (
                    self.times.shape == self.up_inds.shape
                )
                assert (self.scalings is None) or (
                    self.times.shape == self.scalings.shape
                )
                assert self.scores is not None
                assert self.times.shape == self.scores.shape
                assert (self.time_shifts is None) or (
                    self.times.shape == self.time_shifts.shape
                )

    @property
    def n_spikes(self):
        if self.times is None:
            return 0
        else:
            return self.times.numel()

    def subset_by_time(
        self, min_time: int, max_time: int, offset: int, sort: bool = True
    ) -> Self:
        if not self.n_spikes:
            return self
        assert self.times is not None
        t = self.times + offset
        mask = t == t.clamp(min_time, max_time)
        return self.subset(mask=mask, sort=sort)

    def subset(self, mask: Tensor, sort: bool = True) -> Self:
        if not self.n_spikes:
            assert not mask.numel()
            return self
        if mask.dtype == torch.bool:
            (mask,) = mask.nonzero(as_tuple=True)
        if sort:
            assert self.times is not None
            times = self.times[mask]
            times, order = torch.sort(times)
            mask = mask[order]
        else:
            times = _mask_or_none(self.times, mask)
        return self.__class__(
            times=times,
            obj_template_inds=_mask_or_none(self.obj_template_inds, mask),
            template_inds=_mask_or_none(self.template_inds, mask),
            up_inds=_mask_or_none(self.up_inds, mask),
            scalings=_mask_or_none(self.scalings, mask),
            scores=_mask_or_none(self.scores, mask),
            time_shifts=_mask_or_none(self.time_shifts, mask),
        )

    @classmethod
    def concatenate(cls, peaks: list[Self]) -> Self:
        if len(peaks) == 0:
            return cls()
        elif len(peaks) == 1:
            return peaks[0]
        return cls(
            times=_cat_or_none([p.times for p in peaks]),
            obj_template_inds=_cat_or_none([p.obj_template_inds for p in peaks]),
            template_inds=_cat_or_none([p.template_inds for p in peaks]),
            up_inds=_cat_or_none([p.up_inds for p in peaks]),
            scalings=_cat_or_none([p.scalings for p in peaks]),
            scores=_cat_or_none([p.scores for p in peaks]),
            time_shifts=_cat_or_none([p.time_shifts for p in peaks]),
        )


def _mask_or_none(x: Tensor | None, mask: Tensor) -> Tensor | None:
    if x is None:
        return None
    else:
        return x[mask]


def _cat_or_none(xs: list[Tensor | None]) -> Tensor | None:
    if xs[0] is None:
        return None
    else:
        return torch.concatenate(cast(list[Tensor], xs))


# -- matching helper fn library


def subtract_precomputed_pconv(
    *,
    conv: Tensor,
    pconv: Tensor,
    peaks: MatchingPeaks,
    conv_lags: Tensor,
    sign: int,
    padding: int,
    batch_size: int = 128,
):
    assert conv.shape[0] == pconv.shape[0]
    assert sign in (-1, 1)
    if not peaks.n_spikes:
        return
    padded_lags = padding + conv_lags
    times = peaks.times
    assert times is not None
    assert peaks.template_inds is not None
    up_inds = peaks.up_inds
    if up_inds is None:
        up_inds = torch.zeros_like(peaks.template_inds)
    if peaks.scalings is None:
        _subtract_precomputed_pconv_unscaled(
            conv=conv,
            pconv=pconv,
            template_indices=peaks.template_inds,
            upsampling_indices=up_inds,
            times=times,
            padded_conv_lags=padded_lags,
            neg=sign == -1,
            batch_size=batch_size,
        )
    else:
        _subtract_precomputed_pconv_scaled(
            conv=conv,
            pconv=pconv,
            template_indices=peaks.template_inds,
            upsampling_indices=up_inds,
            scalings=peaks.scalings,
            times=times,
            padded_conv_lags=padded_lags,
            neg=sign == -1,
            batch_size=batch_size,
        )


@torch.jit.script
def _subtract_precomputed_pconv_unscaled(
    conv: Tensor,
    pconv: Tensor,
    template_indices: Tensor,
    upsampling_indices: Tensor,
    times: Tensor,
    padded_conv_lags: Tensor,
    neg: bool,
    batch_size: int = 128,
):
    ix_time = times[:, None] + padded_conv_lags[None, :]
    for i0 in range(0, conv.shape[0], batch_size):
        i1 = min(conv.shape[0], i0 + batch_size)
        batch = pconv[i0:i1, template_indices, upsampling_indices]
        ix = ix_time.broadcast_to(batch.shape)
        batch = batch.reshape(i1 - i0, -1)
        ix = ix.reshape(i1 - i0, batch.shape[1])
        if neg:
            batch = batch._neg_view()
        conv[i0:i1].scatter_add_(dim=1, src=batch, index=ix)


@torch.jit.script
def _subtract_precomputed_pconv_scaled(
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
    scalings = scalings[None, :, None]
    for i0 in range(0, conv.shape[0], batch_size):
        i1 = min(conv.shape[0], i0 + batch_size)
        batch = pconv[i0:i1, template_indices, upsampling_indices]
        batch.mul_(scalings)
        ix = ix_time.broadcast_to(batch.shape)
        batch = batch.reshape(i1 - i0, -1)
        ix = ix.reshape(i1 - i0, batch.shape[1])
        if neg:
            batch = batch._neg_view()
        conv[i0:i1].scatter_add_(dim=1, src=batch, index=ix)


@torch.jit.script
def _free_coarse_objective(
    conv: Tensor,
    normsq: Tensor,
    out: Tensor,
    scalings: Tensor,
) -> Tensor:
    out.copy_(conv)
    F.relu(out, inplace=True)
    torch.divide(out, normsq[:, None], out=scalings)
    obj = out.mul_(scalings)
    return obj


@torch.jit.script
def _scaled_coarse_objective(
    conv: Tensor,
    normsq: Tensor,
    out: Tensor,
    scalings: Tensor,
    inv_lambda: Tensor,
    scale_min: Tensor,
    scale_max: Tensor,
) -> Tensor:
    b = conv + inv_lambda
    a = normsq[:, None] + inv_lambda
    torch.divide(b, a, out=scalings)
    scalings.clamp_(min=scale_min, max=scale_max)
    # this is 2 * sc * b - sc**2 * a - inv_lambda
    torch.square(scalings, out=out)
    torch.addcmul(-inv_lambda, -a, out, out=out)
    out.addcmul_(scalings, b, value=2.0)
    return out


@torch.jit.script
def _coarse_match_scaled(
    conv: Tensor,
    template_inds: Tensor,
    times: Tensor,
    obj_normsq: Tensor,
    inv_lambda: Tensor,
    scale_min: Tensor,
    scale_max: Tensor,
):
    b = conv[template_inds, times] + inv_lambda
    a = obj_normsq[template_inds] + inv_lambda
    scalings = b.div(a).clamp_(min=scale_min, max=scale_max)
    objs = scalings.square().mul_(-a)
    objs.addcmul_(scalings, b, value=2.0).sub_(inv_lambda)
    return scalings, objs
