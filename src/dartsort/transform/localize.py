import torch
from dartsort.localize.localize_torch import localize_amplitude_vectors
from dartsort.util.spiketorch import ptp

from .transform_base import BaseWaveformFeaturizer


class PointSourceLocalization(BaseWaveformFeaturizer):
    """Order of output columns: x, y, z_abs, alpha"""

    default_name = "point_source_localizations"
    shape = (4,)
    dtype = torch.double

    def __init__(
        self,
        channel_index,
        geom,
        radius=None,
        n_channels_subset=None,
        logbarrier=True,
        amplitude_kind="peak",
        name=None,
        name_prefix="",
    ):
        assert amplitude_kind in ("peak", "ptp")
        super().__init__(
            geom=geom,
            channel_index=channel_index,
            name=name,
            name_prefix=name_prefix,
        )
        self.amplitude_kind = amplitude_kind
        self.radius = radius
        self.n_channels_subset = n_channels_subset
        self.logbarrier = logbarrier

    def transform(self, waveforms, max_channels=None):
        # get amplitude vectors
        if self.amplitude_kind == "peak":
            ampvecs = waveforms.abs().max(dim=1).values
        elif self.amplitude_kind == "ptp":
            ampvecs = ptp(waveforms, dim=1)

        with torch.enable_grad():
            loc_result = localize_amplitude_vectors(
                ampvecs,
                self.geom,
                max_channels,
                channel_index=self.channel_index,
                radius=self.radius,
                n_channels_subset=self.n_channels_subset,
                logbarrier=self.logbarrier,
                dtype=self.dtype,
            )

        localizations = torch.column_stack(
            [
                loc_result["x"],
                loc_result["y"],
                loc_result["z_abs"],
                loc_result["alpha"],
            ]
        )
        return localizations
