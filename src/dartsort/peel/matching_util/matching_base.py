from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, Self

from dredge.motion_util import MotionEstimate
from spikeinterface.core import BaseRecording
import torch
from torch import Tensor

from ...templates import TemplateData
from ...util.internal_config import ComputationConfig, MatchingConfig
from ...util.spiketorch import grab_spikes, ptp, argrelmax
from ...util.torch_util import BModule


class MatchingTemplates(BModule):
    # subclasses should have their own template_type
    template_type = "base"
    # shared subclass registry for from_config()
    _registry = {}

    # these should be assigned in __init__. adding here for typing.
    spike_length_samples: int
    pconv_db: "PconvBase"

    def __init_subclass__(cls):
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
    motion_est: MotionEstimate | None = None  # type: ignore
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

    def _enforce_refractory(self, mask, peaks, offset=0, value=-torch.inf):
        raise NotImplementedError

    def fine_match(
        self, *, peaks: "MatchingPeaks", residual: Tensor
    ) -> "MatchingPeaks":
        raise NotImplementedError

    # -- super handles below

    def enforce_refractory(self, mask, peaks, offset=0, value=-torch.inf):
        if not peaks.n_spikes:
            return
        self._enforce_refractory(mask, peaks, offset=offset, value=value)

    def forget_refractory(self, *args, **kwargs):
        self.enforce_refractory(*args, **kwargs, value=0.0)

    def unsubtract(self, traces: Tensor, peaks: "MatchingPeaks"):
        return self.subtract(traces, peaks, sign=1)

    def unsubtract_conv(self, *args, **kwargs):
        return self.subtract_conv(*args, **kwargs, sign=1)

    def obj_from_conv(self, conv: Tensor, out=None) -> Tensor:
        return torch.add(self.obj_normsq[:, None]._neg_view(), conv, alpha=2.0, out=out)

    def coarse_match(
        self,
        conv: Tensor,
        objective: Tensor,
        thresholdsq: float | Tensor,
        skip_scaling: bool = False,
    ) -> "MatchingPeaks":
        objective_max, max_obj_template = objective.max(dim=0)
        times = argrelmax(objective_max, self.spike_length_samples, thresholdsq)
        n_spikes = times.numel()
        if not n_spikes:
            return empty_matching_peaks

        template_indices = max_obj_template[times]
        if skip_scaling or not self.scaling:
            objs = objective_max[times]
            return MatchingPeaks(
                n_spikes=n_spikes,
                times=times,
                objective_template_indices=template_indices,
                template_indices=template_indices,
                scores=objs,
            )
        scalings, objs = _coarse_match_scaled(
            conv=conv,
            template_indices=template_indices,
            times=times,
            obj_normsq=self.obj_normsq,
            inv_lambda=self.inv_lambda,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
        )
        return MatchingPeaks(
            n_spikes=n_spikes,
            times=times,
            objective_template_indices=template_indices,
            template_indices=template_indices,
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
        if channels == "template":
            channels = self.main_channels[peaks.template_indices]
            selecting_channels = False
            active_channel_index = channel_index
        elif torch.is_tensor(channels):
            selecting_channels = False
            active_channel_index = channel_index
        elif channels == "amplitude":
            selecting_channels = True
            assert channel_selection_index is not None
            active_channel_index = channel_selection_index
            channels = self.main_channels[peaks.template_indices]
        else:
            assert False

        # get noise
        waveforms = grab_spikes(
            residual_padded,
            peaks.times,
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
            residual_padded,
            peaks,
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


class MatchingPeaks:
    BUFFER_INIT: int = 1000
    BUFFER_GROWTH: float = 13.0 / 8.0

    def __init__(
        self,
        n_spikes: int = 0,
        times: Tensor | None = None,
        objective_template_indices: Tensor | None = None,
        template_indices: Tensor | None = None,
        upsampling_indices: Tensor | None = None,
        scalings: Tensor | None = None,
        scores: Tensor | None = None,
        device: torch.device | None = None,
        buf_size: int | None = None,
    ):
        self.n_spikes: int = n_spikes

        if times is not None:
            self.cur_buf_size = len(times)
        elif buf_size is None:
            self.cur_buf_size = self.BUFFER_INIT
        else:
            self.cur_buf_size = buf_size

        if device is None and times is not None:
            device = times.device

        if times is None:
            self._times = torch.zeros(
                self.cur_buf_size, dtype=torch.long, device=device
            )
        else:
            assert self.cur_buf_size == n_spikes
            self._times = times

        if template_indices is None:
            self._template_indices = torch.zeros(
                self.cur_buf_size, dtype=torch.long, device=device
            )
        else:
            self._template_indices = template_indices

        if objective_template_indices is None:
            self._objective_template_indices = torch.zeros(
                self.cur_buf_size, dtype=torch.long, device=device
            )
        else:
            self._objective_template_indices = objective_template_indices

        if scalings is None:
            self._scalings = torch.ones(self.cur_buf_size, device=device)
        else:
            self._scalings = scalings

        if upsampling_indices is None:
            self._upsampling_indices = torch.zeros(
                self.cur_buf_size, dtype=torch.long, device=device
            )
        else:
            self._upsampling_indices = upsampling_indices

        if scores is None:
            self._scores = torch.zeros(self.cur_buf_size, device=device)
        else:
            self._scores = scores

    @classmethod
    def concatenate(cls, peaks: Sequence[Self]) -> Self:
        peaks = list(peaks)
        times = objective_template_indices = template_indices = None
        upsampling_indices = scalings = scores = None

        n_spikes = sum(p.n_spikes for p in peaks)

        if n_spikes and peaks[0].times is not None:
            times = torch.concatenate([p.times for p in peaks])
        if n_spikes and peaks[0].objective_template_indices is not None:
            objective_template_indices = torch.concatenate(
                [p.objective_template_indices for p in peaks]
            )
        if n_spikes and peaks[0].template_indices is not None:
            template_indices = torch.concatenate([p.template_indices for p in peaks])
        if n_spikes and peaks[0].upsampling_indices is not None:
            upsampling_indices = torch.concatenate(
                [p.upsampling_indices for p in peaks]
            )
        if n_spikes and peaks[0].scalings is not None:
            scalings = torch.concatenate([p.scalings for p in peaks])
        if n_spikes and peaks[0].scores is not None:
            scores = torch.concatenate([p.scores for p in peaks])

        return cls(
            n_spikes=n_spikes,
            times=times,
            objective_template_indices=objective_template_indices,
            template_indices=template_indices,
            upsampling_indices=upsampling_indices,
            scalings=scalings,
            scores=scores,
            buf_size=n_spikes,
        )

    def __str__(self):
        return f"{self.__class__.__name__}(n_spikes={self.n_spikes})"

    @property
    def times(self):
        return self._times[: self.n_spikes]

    @property
    def template_indices(self):
        return self._template_indices[: self.n_spikes]

    @property
    def objective_template_indices(self):
        return self._objective_template_indices[: self.n_spikes]

    @property
    def upsampling_indices(self):
        return self._upsampling_indices[: self.n_spikes]

    @property
    def scalings(self):
        return self._scalings[: self.n_spikes]

    @property
    def scores(self):
        return self._scores[: self.n_spikes]

    def subset(self, which, sort=False):
        self._times = self.times[which]
        if sort:
            self._times, order = torch.sort(self._times, stable=True)
            which = which[order]
        self._template_indices = self.template_indices[which]
        self._objective_template_indices = self.objective_template_indices[which]
        self._upsampling_indices = self.upsampling_indices[which]
        self._scalings = self.scalings[which]
        self._scores = self.scores[which]
        self.n_spikes = self._times.numel()

    def grow_buffers(self, min_size=0):
        sz = max(min_size, int(self.cur_buf_size * self.BUFFER_GROWTH))
        k = self.n_spikes
        self._times = _grow_buffer(self._times, k, sz)
        self._template_indices = _grow_buffer(self._template_indices, k, sz)
        self._objective_template_indices = _grow_buffer(
            self._objective_template_indices, k, sz
        )
        self._upsampling_indices = _grow_buffer(self._upsampling_indices, k, sz)
        self._scalings = _grow_buffer(self._scalings, k, sz)
        self._scores = _grow_buffer(self._scores, k, sz)
        self.cur_buf_size = sz

    def sort(self):
        sl = slice(0, self.n_spikes)

        times = self._times[sl]
        order = torch.argsort(times, stable=True)

        self._times[sl] = times[order]
        self._template_indices[sl] = self.template_indices[order]
        self._objective_template_indices[sl] = self.objective_template_indices[order]
        self._upsampling_indices[sl] = self.upsampling_indices[order]
        self._scalings[sl] = self.scalings[order]
        self._scores[sl] = self.scores[order]

    def extend(self, other):
        new_n_spikes = other.n_spikes + self.n_spikes
        sl_new = slice(self.n_spikes, new_n_spikes)

        if new_n_spikes > self.cur_buf_size:
            self.grow_buffers(min_size=new_n_spikes)

        self._times[sl_new] = other.times
        self._template_indices[sl_new] = other.template_indices
        self._objective_template_indices[sl_new] = other.objective_template_indices
        self._upsampling_indices[sl_new] = other.upsampling_indices
        self._scalings[sl_new] = other.scalings
        self._scores[sl_new] = other.scores

        self.n_spikes = new_n_spikes


empty_matching_peaks = MatchingPeaks()


def _grow_buffer(x, old_length, new_size):
    new = torch.empty(new_size, dtype=x.dtype, device=x.device)
    new[:old_length] = x[:old_length]
    return new


@torch.jit.script
def _coarse_match_scaled(
    conv: Tensor,
    template_indices: Tensor,
    times: Tensor,
    obj_normsq: Tensor,
    inv_lambda: Tensor,
    scale_min: Tensor,
    scale_max: Tensor,
):
    b = conv[template_indices, times].add_(inv_lambda)
    a = obj_normsq[template_indices].add_(inv_lambda)
    scalings = b.div(a).clamp_(min=scale_min, max=scale_max)
    objs = scalings.square().mul_(-a)
    objs.addcmul_(scalings, b, value=2.0).sub_(inv_lambda)
    return scalings, objs
