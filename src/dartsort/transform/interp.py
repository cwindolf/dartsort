import torch

from .transform_base import BaseWaveformDenoiser
from ..util.internal_config import InterpolationParams
from ..util.interpolation_util import ToFullProbeInterpolator
from ..util.drift_util import registered_geometry
from ..util.logging_util import get_logger

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
        motion_est=None,
        rgeom=None,
        params: InterpolationParams,
    ):
        super().__init__(
            name=name, name_prefix=name_prefix, geom=geom, channel_index=channel_index
        )
        assert channel_index.shape[1] == geom.shape[0], (
            "Meant to be used with full-probe data."
        )
        if rgeom is None:
            rgeom = registered_geometry(geom=geom, motion_est=motion_est)
        geom = torch.asarray(geom, dtype=torch.float)
        rgeom = torch.asarray(rgeom, dtype=torch.float)
        params = params.normalize()
        logger.dartsortdebug(
            "Make WaveformInterpolator with method=%s, kernel=%s, and %s extrap.",
            params.method,
            params.kernel,
            "different" if params.extrap_diff() else "same",
        )
        self.erp = ToFullProbeInterpolator(
            geom=geom, rgeom=rgeom, motion_est=motion_est, params=params
        )

    def forward(self, waveforms, *, chunk_center_s, **unused):
        del unused
        waveforms = self.erp.interp_at_time(
            t_s=chunk_center_s.item(), waveforms=waveforms
        )
        return waveforms
