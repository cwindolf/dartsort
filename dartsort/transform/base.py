import torch
from dartsort.util.data_util import SpikeDataset


class BaseWaveformModule(torch.nn.Module):
    # if this is True, then model fitting+saving+loading will happen
    needs_fit = False
    is_denoiser = False
    is_featurizer = False
    default_name = ""

    def __init__(self, name):
        self.name = name
        if name is None:
            name = self.default_name

    def fit(self, waveforms, max_channels=None):
        pass


class BaseWaveformDenoiser(BaseWaveformModule):
    is_denoiser = True

    def forward(self, waveforms, max_channels=None):
        raise NotImplementedError


class BaseWaveformFeaturizer(BaseWaveformModule):
    is_featurizer = True
    # output shape per waveform
    shape = ()
    # output dtye
    dtype = torch.float

    def transform(self, waveforms, max_channels=None):
        raise NotImplementedError

    @property
    def spike_dataset(self):
        torch_dtype_as_str = str(self.dtype).split(".")[1]
        return SpikeDataset(
            name=self.name,
            shape_per_spike=self.shape,
            dtype=torch_dtype_as_str,
        )


class IdentityWaveformDenoiser(BaseWaveformDenoiser):
    def forward(self, waveforms, max_channels=None):
        return waveforms


class WaveformFeaturizer(BaseWaveformFeaturizer):
    def __init__(
        self, channel_index, spike_length_samples=121, dtype=torch.float
    ):
        self.shape = (spike_length_samples, channel_index.shape[1])
        self.dtype = dtype

    def forward(self, waveforms, max_channels=None):
        return waveforms


class ZerosWaveformFeaturizer(BaseWaveformModule):
    shape = ()
    dtype = torch.float

    def transform(self, waveforms, max_channels=None):
        return torch.zeros(
            waveforms.shape[0], device=waveforms.device, dtype=torch.float
        )
