from typing import TYPE_CHECKING

from .transform_base import BaseWaveformDenoiser

if TYPE_CHECKING:
    from ..util.noise_util import SpatialWhitener


class WaveformWhitener(BaseWaveformDenoiser):
    default_name = "whiten"

    def __init__(
        self,
        *,
        geom,
        channel_index,
        name=None,
        name_prefix=None,
        whitener: "SpatialWhitener | None" = None,
    ):
        super().__init__(
            name=name, name_prefix=name_prefix, geom=geom, channel_index=channel_index
        )
        assert channel_index.shape[1] == geom.shape[0], (
            "Meant to be used with full-probe data."
        )
        self.whitener = whitener

    def forward(self, waveforms, **unused):
        del unused
        if self.whitener is None:
            return waveforms
        else:
            return self.whitener.whiten(x=waveforms)
