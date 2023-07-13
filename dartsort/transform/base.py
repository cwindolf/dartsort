import torch
from dartsort.util.data_util import SpikeDataset


class BaseWaveformModule(torch.nn.Module):
    # if this is True, then model fitting+saving+loading will happen
    needs_fit = False
    is_denoiser = False
    is_featurizer = False
    default_name = ""

    def __init__(
        self, channel_index=None, geom=None, name=None, name_prefix=""
    ):
        super().__init__()
        if name is None:
            name = self.default_name
            if name_prefix:
                name = name_prefix + "_" + name
        self.name = name

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
        print("Hi from spike_dataset", self)
        print(self.dtype)
        torch_dtype_as_str = str(self.dtype).split(".")[1]
        return SpikeDataset(
            name=self.name,
            shape_per_spike=self.shape,
            dtype=torch_dtype_as_str,
        )


class IdentityWaveformDenoiser(BaseWaveformDenoiser):
    def forward(self, waveforms, max_channels=None):
        return waveforms


class Waveform(BaseWaveformFeaturizer):
    default_name = "waveforms"

    def __init__(
        self,
        channel_index,
        geom=None,
        spike_length_samples=121,
        dtype=torch.float,
        name=None,
        name_prefix="",
    ):
        super().__init__(
            geom=geom,
            channel_index=channel_index,
            name=name,
            name_prefix=name_prefix,
        )
        self.shape = (spike_length_samples, channel_index.shape[1])
        self.dtype = dtype

    def forward(self, waveforms, max_channels=None):
        return waveforms


class ZerosWaveformFeaturizer(BaseWaveformModule):
    shape = ()
    dtype = torch.float
    default_name = "zeros_like_waveforms"

    def transform(self, waveforms, max_channels=None):
        return torch.zeros(
            waveforms.shape[0], device=waveforms.device, dtype=torch.float
        )
