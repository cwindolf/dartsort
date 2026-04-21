import torch

from ..util.internal_config import InterpolationParams
from ..util.interpolation_util import ToFullProbeInterpolator
from ..util.logging_util import get_logger
from ..util.motion import MotionInfo
from .transform_base import BaseWaveformDenoiser

logger = get_logger(__name__)


class WaveformInterpolator(BaseWaveformDenoiser):
    default_name = "interpolated"

    def __init__(
        self,
        *,
        geom,
        channel_index,
        name=None,
        name_prefix=None,
        motion: MotionInfo,
        params: InterpolationParams,
    ):
        super().__init__(
            name=name, name_prefix=name_prefix, geom=geom, channel_index=channel_index
        )
        assert channel_index.shape[1] == geom.shape[0], (
            "Meant to be used with full-probe data."
        )
        geom = torch.asarray(geom, dtype=torch.float)
        params = params.normalize()
        logger.dartsortverbose(
            "Make WaveformInterpolator with method=%s, kernel=%s, and %s extrap.",
            params.method,
            params.kernel,
            "different" if params.extrap_diff() else "same",
        )
        self.erp = ToFullProbeInterpolator(
            motion=motion, params=params, device=geom.device
        )

    def forward(self, waveforms, **fixed_properties):
        chunk_center_s = fixed_properties["chunk_center_s"]
        waveforms = self.erp.interp_at_time(
            t_s=chunk_center_s.item(), waveforms=waveforms
        )
        return waveforms
