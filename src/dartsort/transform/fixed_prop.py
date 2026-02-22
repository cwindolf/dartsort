import torch
from dartsort.util.spiketorch import ptp

from .transform_base import BaseWaveformFeaturizer

class FixedProperty(BaseWaveformFeaturizer):
    default_name = ""

    def __init__(
        self,
        channel_index,
        geom=None,
        dtype=torch.float,
        name=None,
        name_prefix=None,
    ):
        assert name is not None
        super().__init__(
            geom=geom,
            channel_index=channel_index,
            name=name,
            name_prefix=name_prefix,
        )
        self.shape = ()
        self.dtype = dtype

    def transform(self, waveforms, **rest):
        return {self.name: rest[self.name]}