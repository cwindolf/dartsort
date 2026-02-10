import torch

from ..util.waveform_util import get_channels_in_probe, set_channels_in_probe
from .transform_base import BaseWaveformDenoiser


class DebugMatchingPursuitDenoiser(BaseWaveformDenoiser):
    """This denoiser is used for testing purposes only."""

    default_name = "pursuit_basis"

    def __init__(self, channel_index, geom=None, basis=None, name=None, name_prefix=""):
        super().__init__(
            channel_index=channel_index, name=name, name_prefix=name_prefix
        )
        if basis is not None:
            assert basis.ndim == 2
            self.register_buffer("basis", basis)

    def initialize_spike_length_dependent_params(self):
        if not hasattr(self, "basis"):
            # this path is only hit by test_transform
            zbuf = torch.zeros((1, self.spike_length_samples))  # type: ignore
            self.register_buffer("basis", zbuf)

    def needs_fit(self) -> bool:
        return not hasattr(self, "basis")

    def _project_in_probe(self, waveforms_in_probe):
        dev = waveforms_in_probe.device
        assert self.basis is not None
        dots = waveforms_in_probe.to(device=self.b.basis.device) @ self.basis.T
        best = dots.abs().argmax(dim=1)
        dots = dots.take_along_dim(best[:, None], dim=1)[:, 0]
        denoised = self.b.basis.take_along_dim(best[:, None], dim=0)
        denoised = denoised.mul_(dots.sign()[:, None])
        return denoised.to(device=dev)

    def forward(self, waveforms, *, channels, **unused):
        channels_in_probe, waveforms_in_probe = get_channels_in_probe(
            waveforms, channels, self.channel_index
        )
        waveforms_in_probe = self._project_in_probe(waveforms_in_probe)
        return set_channels_in_probe(waveforms_in_probe, waveforms, channels_in_probe)
