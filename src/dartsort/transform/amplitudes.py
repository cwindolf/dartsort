import torch
from dartsort.util.spiketorch import ptp

from .transform_base import BaseWaveformFeaturizer


class AmplitudeVector(BaseWaveformFeaturizer):
    default_name = "amplitude_vectors"

    def __init__(
        self,
        channel_index,
        geom=None,
        kind="peak",
        dtype=torch.float,
        name=None,
        name_prefix="",
    ):
        assert kind in ("peak", "ptp")
        super().__init__(name=name, name_prefix=name_prefix)
        self.kind = kind
        self.shape = (channel_index.shape[1],)
        self.dtype = dtype

    def transform(self, waveforms, max_channels=None):
        if self.kind == "peak":
            return waveforms.abs().max(dim=1).values
        elif self.kind == "ptp":
            return ptp(waveforms, dim=1)


class MaxAmplitude(BaseWaveformFeaturizer):
    default_name = "amplitudes"
    shape = ()

    def __init__(
        self,
        channel_index=None,
        geom=None,
        kind="ptp",
        dtype=torch.float,
        name=None,
        name_prefix="",
    ):
        assert kind in ("peak", "ptp")
        super().__init__(name=name, name_prefix=name_prefix)
        self.kind = kind
        self.dtype = dtype

    def transform(self, waveforms, max_channels=None):
        if self.kind == "peak":
            return torch.nan_to_num(waveforms.abs()).max(dim=(1, 2)).values
        elif self.kind == "ptp":
            return torch.nan_to_num(ptp(waveforms, dim=1)).max(dim=1).values
