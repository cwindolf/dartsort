from pathlib import Path
from typing import TYPE_CHECKING

import torch
from spikeinterface.core import BaseRecording

from ..util.data_util import DARTsortSorting
from ..util.internal_config import (
    ComputationConfig,
    WaveformConfig,
    WhiteningConfig,
    default_waveform_cfg,
)
from .transform_base import BaseWaveformDenoiser

if TYPE_CHECKING:
    from ..util.noise_util import SpatialWhitener
    from .pipeline import WaveformPipeline


class WaveformWhitener(BaseWaveformDenoiser):
    default_name = "whiten"
    needs_residual = True

    def __init__(
        self,
        *,
        geom,
        channel_index,
        name=None,
        name_prefix=None,
        waveform_cfg: WaveformConfig | None = default_waveform_cfg,
        whitener: "SpatialWhitener | None" = None,
        disabled: bool = True,
        whiten_cfg: WhiteningConfig = WhiteningConfig(),
        sampling_frequency: float = 30_000.0,
    ):
        super().__init__(
            name=name, name_prefix=name_prefix, geom=geom, channel_index=channel_index
        )
        self.whitener = whitener
        self.disabled = disabled
        self.whiten_cfg = whiten_cfg
        self.motion = None

    def needs_fit(self):
        return self.whitener is None

    def forward(self, waveforms, **unused):
        del unused
        if self.disabled or self.whitener is None:
            return waveforms
        else:
            return self.whitener.whiten(x=waveforms)

    def attach_motion(self, motion):
        self.motion = motion

    def _other_pre_load_state(self, state_dict, prefix):
        if self.whitener is not None:
            return
        from ..util.noise_util import SpatialWhitener

        self.whitener = SpatialWhitener.blank(len(self.b.geom), self.b.geom.device)

    def fit(
        self,
        recording: BaseRecording,
        waveforms: torch.Tensor,
        *,
        hdf5_filename: Path | None = None,
        computation_cfg: ComputationConfig,
        pipeline: "WaveformPipeline | None" = None,
        **spike_data: torch.Tensor,
    ):
        super().fit(
            recording,
            waveforms,
            hdf5_filename=hdf5_filename,
            computation_cfg=computation_cfg,
            pipeline=pipeline,
            **spike_data,
        )
        del recording, spike_data, waveforms, pipeline
        from ..util.noise_util import SpatialWhitener

        assert hdf5_filename is not None

        if self.motion is None:
            assert self.whiten_cfg.strategy != "postwhiten"

        sorting = DARTsortSorting.from_peeling_hdf5(
            hdf5_filename, load_simple_features=False
        )
        self.whitener = SpatialWhitener.from_config(
            sorting=sorting,
            motion=self.motion,
            whiten_cfg=self.whiten_cfg,
            computation_cfg=computation_cfg,
        )
