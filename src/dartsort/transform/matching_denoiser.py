from ..util.waveform_util import get_channels_in_probe, set_channels_in_probe
from .transform_base import BaseWaveformDenoiser


class MatchingPursuitDenoiser(BaseWaveformDenoiser):
    default_name = "pursuit_basis"

    def __init__(self, channel_index, geom=None, basis=None, name=None, name_prefix=""):
        assert basis is not None
        super().__init__(
            channel_index=channel_index, name=name, name_prefix=name_prefix
        )
        assert basis.ndim == 2
        self.register_buffer("basis", basis)

    def _project_in_probe(self, waveforms_in_probe):
        dots = waveforms_in_probe @ self.basis.T
        best = dots.abs().argmax(dim=1)
        dots = dots.take_along_dim(best[:, None], dim=1)[:, 0]
        denoised = self.b.basis.take_along_dim(best[:, None], dim=0)
        denoised = denoised.mul_(dots.sign()[:, None])
        return denoised

    def forward(self, waveforms, *, channels, time_shifts=None, **unused):
        channels_in_probe, waveforms_in_probe = get_channels_in_probe(
            waveforms, channels, self.channel_index
        )
        waveforms_in_probe = self._project_in_probe(waveforms_in_probe)
        return set_channels_in_probe(waveforms_in_probe, waveforms, channels_in_probe)
