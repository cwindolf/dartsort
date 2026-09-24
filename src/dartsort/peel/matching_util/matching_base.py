from math import ceil
from pathlib import Path
from typing import Literal, Self, cast

import torch
import torch.nn.functional as F
from spikeinterface.core import BaseRecording
from torch import Tensor

from ...templates import TemplateData
from ...util.internal_config import ComputationConfig, MatchingConfig
from ...util.logging_util import DARTSORTVERBOSE, get_logger
from ...util.motion import MotionInfo
from ...util.py_util import databag, panic
from ...util.spiketorch import (
    _nonzero_static,
    argrelmax_dedup,
    argrelmax_dedup_mask_no_thresh,
    grab_spikes,
    ptp,
)
from ...util.torch_util import BModule, torch_compiler

logger = get_logger(__name__)
_extra_checks = logger.isEnabledFor(DARTSORTVERBOSE)


class MatchingTemplates(BModule):
    # subclasses should have their own template_type
    template_type = "base"
    # shared subclass registry for from_config()
    _registry = {}  # noqa: RUF012

    # these should be assigned in __init__. adding here for typing.
    spike_length_samples: int

    def __init_subclass__(cls):
        logger.dartsortverbose(
            "Register matching templates type: %s", cls.template_type
        )
        cls._registry[cls.template_type] = cls

    @classmethod
    def from_config(
        cls,
        *,
        save_folder: Path | None,
        recording: BaseRecording,
        template_data: TemplateData,
        matching_cfg: MatchingConfig,
        computation_cfg: ComputationConfig | None = None,
        motion: MotionInfo,
        overwrite: bool = False,
        dtype=torch.float,
    ) -> Self:
        global _extra_checks
        _extra_checks = logger.isEnabledFor(DARTSORTVERBOSE)
        if _extra_checks:
            logger.dartsortverbose("Extra checks enabled in matching.")
        return cls._registry[matching_cfg.template_type]._from_config(
            save_folder=save_folder,
            recording=recording,
            template_data=template_data,
            matching_cfg=matching_cfg,
            computation_cfg=computation_cfg,
            motion=motion,
            overwrite=overwrite,
            dtype=dtype,
        )

    @classmethod
    def _from_config(
        cls,
        *,
        save_folder: Path | None,
        recording: BaseRecording,
        template_data: TemplateData,
        matching_cfg: MatchingConfig,
        computation_cfg: ComputationConfig | None,
        motion: MotionInfo,
        overwrite: bool,
        dtype: torch.dtype,
    ) -> Self:
        raise NotImplementedError

    def data_at_time(
        self,
        t_s: float,
        *,
        scaling: bool,
        free_scaling: bool,
        inv_lambda: float,
        scale_min: float,
        scale_max: float,
        resid_offset: int,
    ) -> "ChunkTemplateData":
        raise NotImplementedError


@databag
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
    motion: MotionInfo
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
            motion=self.motion,
            dtype=self.dtype,
            overwrite=overwrite,
        )

    @property
    def spike_length_samples(self) -> int:
        return self.template_data.spike_length_samples


class ChunkTemplateData:
    # -- subclasses must assign the following properties that the matcher uses.
    spike_length_samples: int
    filter_length_samples: int
    resid_offset: int

    # for the full templates
    unit_ids: Tensor
    main_channels: Tensor
    # for obj templates
    obj_normsq: Tensor
    obj_normsq_plus_inv_lambda: Tensor
    inv_obj_normsq_plus_inv_lambda: Tensor
    obj_n_templates: int
    upsampling: bool
    scaling: bool
    free_scaling: bool
    needs_fine_pass: bool
    needs_residual: bool
    up_factor: int
    prewhiten: bool
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

    def fine_match(
        self,
        *,
        peaks: "MatchingPeaks",
        residual: Tensor | None,
        conv: Tensor,
        padding: int = 0,
    ) -> "MatchingPeaks":
        raise NotImplementedError

    def whiten_traces(self, traces: Tensor, out: Tensor | None = None):
        assert not self.prewhiten
        if out is not None:
            return out.copy_(traces.T)
        else:
            return traces.T.contiguous()

    # this one is just for debugging / unit testing
    def reconstruct_up_templates(self):
        raise NotImplementedError

    # -- super handles below

    def unsubtract(self, traces: Tensor, peaks: "MatchingPeaks"):
        return self.subtract(traces, peaks, sign=1)

    def unsubtract_conv(self, conv: Tensor, peaks: "MatchingPeaks", padding=0):
        return self.subtract_conv(conv=conv, peaks=peaks, padding=padding, sign=1)

    def obj_from_conv(
        self,
        *,
        conv: Tensor,
        out: Tensor,
        scalings_out: Tensor | None = None,
    ) -> Tensor:
        assert conv.shape == out.shape
        if scalings_out is not None:
            assert scalings_out.shape == out.shape
        elif self.scaling:
            scalings_out = torch.empty_like(out)
        if self.free_scaling:
            assert scalings_out is not None
            return _free_coarse_objective(
                conv=conv, normsq=self.obj_normsq, out=out, scalings=scalings_out
            )
        elif self.scaling:
            assert scalings_out is not None
            return _scaled_coarse_objective(
                conv=conv,
                term_a=self.obj_normsq_plus_inv_lambda,
                inv_term_a=self.inv_obj_normsq_plus_inv_lambda,
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

    def obj_from_conv_known_pos(
        self,
        *,
        conv: Tensor,
        out: Tensor,
        scalings_out: Tensor | None = None,
    ) -> Tensor:
        assert conv.shape == out.shape
        if scalings_out is not None:
            assert scalings_out.shape == out.shape
        if self.free_scaling:
            assert scalings_out is not None
            return _free_coarse_objective(
                conv=conv, normsq=self.obj_normsq, out=out, scalings=scalings_out
            )
        elif self.scaling:
            assert scalings_out is not None
            return _scaled_coarse_objective_known_pos(
                conv=conv,
                # normsq=self.obj_normsq,
                term_a=self.obj_normsq_plus_inv_lambda,
                inv_term_a=self.inv_obj_normsq_plus_inv_lambda,
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

    def quick_match(
        self,
        *,
        padded_conv: Tensor,
        padded_objective_buf: Tensor,
        thresholdsq: float,
        obj_arange: Tensor,
        padding: int,
        exclude_extra_padding: int = 0,
        peak_dt: int = 3,
        return_scalings=False,
    ) -> "MatchingPeaks":
        obj = torch.add(
            self.obj_normsq[:, None]._neg_view(),
            padded_conv,
            alpha=2.0,
            out=padded_objective_buf[:-1],
        )
        if not self.scaling:
            obj_max, obj_temp = obj.max(dim=0)
            nt = obj_max.shape[0]
            times = argrelmax_dedup(
                x=obj_max,
                dedup_radius=self.filter_length_samples,
                threshold=thresholdsq,
                arange=obj_arange[:nt],
                padding=padding + exclude_extra_padding,
            )
            template_inds = obj_temp[times]
            scores = obj_max[times]
            scalings = torch.ones_like(scores) if return_scalings else None
        else:
            times, template_inds, scores, scalings = self._quick_propose_scaled_peaks(
                padded_conv=padded_conv,
                obj=obj,
                obj_arange=obj_arange,
                padding=padding,
                exclude_extra_padding=exclude_extra_padding,
                thresholdsq=thresholdsq,
                peak_dt=peak_dt,
                return_scalings=return_scalings,
            )

        if not times.numel():
            return MatchingPeaks()

        return MatchingPeaks(
            times=times - padding,
            template_inds=template_inds,
            scalings=scalings,
            scores=scores,
        )

    def get_collisioncleaned_waveforms(
        self,
        residual_padded: Tensor,
        peaks: "MatchingPeaks",
        channels: Tensor | Literal["template", "amplitude"],
        channel_index: Tensor,
        channel_selection_index: Tensor | None = None,
        with_coll: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        if not peaks.n_spikes:
            empty_channels = residual_padded.new_zeros(size=(0,), dtype=torch.long)
            empty_waveforms = residual_padded.new_zeros(size=())
            empty_coll = residual_padded.new_zeros(size=()) if with_coll else None
            return empty_channels, empty_waveforms, empty_coll

        if channels == "template":
            channels = self.main_channels[peaks.template_inds]
            selecting_channels = False
            sel_ci = channel_index
        elif torch.is_tensor(channels):
            selecting_channels = False
            sel_ci = channel_index
        elif channels == "amplitude":
            selecting_channels = True
            assert channel_selection_index is not None
            sel_ci = channel_selection_index
            channels = self.main_channels[peaks.template_inds]
        else:
            panic(channels)
        assert isinstance(channels, Tensor)

        times = peaks.times
        assert times is not None

        # get noise
        # TODO check the offset is correct
        waveforms = grab_spikes(
            residual_padded,
            times,
            channels,
            sel_ci,
            trough_offset=self.resid_offset,
            spike_length_samples=self.spike_length_samples,
            buffer=0,
            already_padded=True,
        )
        if with_coll:
            coll = waveforms.square().nanmean(dim=(1, 2)).sqrt_()
        else:
            coll = None
        waveforms = self.get_clean_waveforms(
            peaks=peaks, channel_index=sel_ci, channels=channels, add_into=waveforms
        )

        if not selecting_channels:
            return channels, waveforms, coll

        cix = ptp(waveforms).nan_to_num_(nan=-torch.inf).argmax(dim=1)
        assert channel_selection_index is not None
        sel = channel_selection_index[channels]
        channels = sel.take_along_dim(cix[:, None], dim=1)[:, 0]
        return self.get_collisioncleaned_waveforms(
            residual_padded=residual_padded,
            peaks=peaks,
            channel_index=channel_index,
            channels=channels,
            with_coll=with_coll,
        )

    def _quick_propose_scaled_peaks(
        self,
        *,
        padded_conv: Tensor,
        obj: Tensor,
        obj_arange: Tensor,
        padding: int,
        exclude_extra_padding: int,
        thresholdsq: float,
        peak_dt: int,
        return_scalings: bool,
    ):
        nu, nt = padded_conv.shape

        # propose peaks
        obj_max = obj.amax(dim=0)
        mask = argrelmax_dedup_mask_no_thresh(
            x=obj_max,
            dedup_radius=self.filter_length_samples + peak_dt,
            arange=obj_arange[:nt],
            padding=padding + exclude_extra_padding,
        )
        max_peaks = ceil(nt / self.filter_length_samples)
        peaks = _nonzero_static(mask, size=max_peaks, le_ok=True)
        valid = peaks >= 0

        # scaled refinement
        windows = peaks + torch.arange(-peak_dt, peak_dt + 1, device=peaks.device)
        windows.masked_fill_(valid.logical_not(), 0)
        windows_flat = windows.view(-1)
        window_conv = padded_conv[:, windows_flat]
        window_objs = torch.empty_like(window_conv)
        if self.scaling:
            window_scs = torch.empty_like(window_conv)
        else:
            window_scs = None
        self.obj_from_conv_known_pos(
            conv=window_conv, out=window_objs, scalings_out=window_scs
        )
        window_objs = window_objs.T.reshape(windows.shape[0], windows.shape[1] * nu)
        best_val, best_ix = window_objs.max(dim=1)
        best_shift = best_ix // nu
        best_unit = best_ix % nu

        peaks = peaks.squeeze()
        times = peaks.add_(best_shift).sub_(peak_dt)
        valid = valid.squeeze()
        valid.logical_and_(best_val >= thresholdsq)
        valid.logical_and_(
            times
            == times.clamp(
                padding + exclude_extra_padding,
                nt - padding - exclude_extra_padding - 1,
            )
        )

        # grab result
        (keep,) = valid.nonzero(as_tuple=True)

        # best_ix = best_ix[keep]
        best_shift = best_shift[keep]

        scores = best_val[keep]
        times = times[keep]
        template_inds = best_unit[keep]
        if not return_scalings or window_scs is None:
            scalings = None
        else:
            scalings = window_scs.view(nu, *windows.shape)[
                template_inds, keep, best_shift
            ]

        return times, template_inds, scores, scalings


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
    template_inds: Tensor | None = None
    up_inds: Tensor | None = None
    scalings: Tensor | None = None
    scores: Tensor | None = None
    time_shifts: Tensor | None = None

    if _extra_checks:

        def __post_init__(self):
            if self.times is None:
                assert self.template_inds is None
                assert self.up_inds is None
                assert self.scalings is None
                assert self.scores is None
                assert self.time_shifts is None
            else:
                assert self.times.ndim == 1
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
            template_inds=_cat_or_none([p.template_inds for p in peaks]),
            up_inds=_cat_or_none([p.up_inds for p in peaks]),
            scalings=_cat_or_none([p.scalings for p in peaks]),
            scores=_cat_or_none([p.scores for p in peaks]),
            time_shifts=_cat_or_none([p.time_shifts for p in peaks]),
        )

    def __str__(self):
        dsets = ""
        if self.times is not None:
            dsets += "times,"
        if self.up_inds is not None:
            dsets += "up_inds,"
        if self.scalings is not None:
            dsets += "scalings,"
        if self.scores is not None:
            dsets += "scores,"
        dsets = dsets.removesuffix(",")
        return f"{self.__class__.__name__}(n_spikes={self.n_spikes},{dsets})"


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


@torch_compiler(fullgraph=False)
def _subtract_precomputed_pconv_unscaled(
    conv: Tensor,
    pconv: Tensor,
    template_indices: Tensor,
    upsampling_indices: Tensor,
    times: Tensor,
    padded_conv_lags: Tensor,
    neg: bool,
    batch_size: int = 256,
):
    ix_time = times[:, None] + padded_conv_lags[None, :]
    ix_time = ix_time.view(-1)
    nixt = ix_time.shape[0]
    alpha = -1 if neg else 1
    for i0 in range(0, conv.shape[0], batch_size):
        i1 = min(conv.shape[0], i0 + batch_size)
        batch = pconv[i0:i1, template_indices, upsampling_indices]
        batch = batch.reshape(i1 - i0, nixt)
        conv[i0:i1].index_add_(dim=1, source=batch, index=ix_time, alpha=alpha)


@torch_compiler(fullgraph=False)
def _subtract_precomputed_pconv_scaled(
    conv: Tensor,
    pconv: Tensor,
    template_indices: Tensor,
    upsampling_indices: Tensor,
    scalings: Tensor,
    times: Tensor,
    padded_conv_lags: Tensor,
    neg: bool,
    batch_size: int = 256,
):
    ix_time = times[:, None] + padded_conv_lags[None, :]
    ix_time = ix_time.view(-1)
    nixt = ix_time.shape[0]
    scalings = scalings[None, :, None]
    alpha = -1 if neg else 1
    for i0 in range(0, conv.shape[0], batch_size):
        i1 = min(conv.shape[0], i0 + batch_size)
        batch = pconv[i0:i1, template_indices, upsampling_indices]
        batch.mul_(scalings)
        batch = batch.reshape(i1 - i0, nixt)
        conv[i0:i1].index_add_(dim=1, source=batch, index=ix_time, alpha=alpha)


@torch_compiler()
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


@torch_compiler()
def _scaled_coarse_objective(
    conv: Tensor,
    term_a: Tensor,
    inv_term_a: Tensor,
    out: Tensor,
    scalings: Tensor,
    inv_lambda: Tensor,
    scale_min: Tensor,
    scale_max: Tensor,
) -> Tensor:
    neg = conv < 0
    b = conv + inv_lambda
    # a = normsq[:, None] + inv_lambda
    torch.mul(b, inv_term_a, out=scalings)
    scalings.clamp_(min=scale_min, max=scale_max)
    scalings.masked_fill_(neg, 0.0)
    # this is 2 * sc * b - sc**2 * a - inv_lambda
    torch.square(scalings, out=out)
    torch.addcmul(-inv_lambda, term_a, out, value=-1, out=out)
    out.addcmul_(scalings, b, value=2.0)
    return out


@torch_compiler()
def _scaled_coarse_objective_known_pos(
    conv: Tensor,
    term_a: Tensor,
    inv_term_a: Tensor,
    out: Tensor,
    scalings: Tensor,
    inv_lambda: Tensor,
    scale_min: Tensor,
    scale_max: Tensor,
) -> Tensor:
    b = conv + inv_lambda
    # a = normsq[:, None] + inv_lambda
    torch.mul(b, inv_term_a, out=scalings)
    scalings.clamp_(min=scale_min, max=scale_max)
    # this is 2 * sc * b - sc**2 * a - inv_lambda
    torch.square(scalings, out=out)
    torch.addcmul(inv_lambda._neg_view(), term_a, out, value=-1, out=out)
    out.addcmul_(scalings, b, value=2.0)
    return out
