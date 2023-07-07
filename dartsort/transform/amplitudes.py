import torch
from dartsort.util.spiketorch import ptp

from .base import BaseWaveformFeaturizer


class AmplitudeVector(BaseWaveformFeaturizer):
    default_name = "amplitude_vectors"

    def __init__(
        self, channel_index, kind="peak", dtype=torch.float, name=None
    ):
        assert kind in ("peak", "ptp")
        super().__init__(name)
        self.kind = kind
        self.shape = (channel_index.shape[1],)
        self.dtype = dtype

    def forward(self, waveforms, max_channels=None):
        if self.kind == "peak":
            return waveforms.abs().max(dim=2).values
        elif self.kind == "ptp":
            return ptp(waveforms, dim=1)


class MaxAmplitude(BaseWaveformFeaturizer):
    default_name = "amplitudes"

    def __init__(self, name, kind="peak", dtype=torch.float):
        assert kind in ("peak", "ptp")
        super().__init__(name)
        self.kind = kind
        self.shape = ()
        self.dtype = dtype

    def forward(self, waveforms, max_channels=None):
        if self.kind == "peak":
            return waveforms.abs().max(dim=(1, 2)).values
        elif self.kind == "ptp":
            return ptp(waveforms, dim=1).max(dim=1).values
