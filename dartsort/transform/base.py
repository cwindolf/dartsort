import torch


class BaseWaveformModule(torch.nn.Module):
    # if this is True, then model fitting+saving+loading will happen
    needs_fit = False
    is_denoiser = False
    is_featurizer = False

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


# these classes below are just examples


class IdentityWaveformDenoiser(BaseWaveformDenoiser):
    def forward(self, waveforms, max_channels=None):
        return waveforms


class ZerosWaveformFeaturizer(BaseWaveformModule):
    shape = ()
    dtype = torch.float

    def transform(self, waveforms, max_channels=None):
        return torch.zeros(
            waveforms.shape[0], device=waveforms.device, dtype=torch.float
        )
