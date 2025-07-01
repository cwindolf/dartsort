import torch
from dartsort.localize.localize_torch import localize_amplitude_vectors
from dartsort.util.spiketorch import ptp

from .transform_base import BaseWaveformFeaturizer


class Localization(BaseWaveformFeaturizer):
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
        localization_model="pointsource",
        name=None,
        name_prefix="",
        batch_size=1024,
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
        self.localization_model = localization_model
        self.batch_size = batch_size

    def transform(self, waveforms, max_channels=None):
        # get amplitude vectors
        if self.amplitude_kind == "peak":
            ampvecs = waveforms.abs().max(dim=1).values
        elif self.amplitude_kind == "ptp":
            ampvecs = ptp(waveforms, dim=1)
        else:
            assert False

        n = len(ampvecs)
        localizations = ampvecs.new_empty((n, 4), dtype=self.dtype)

        with torch.enable_grad():
            for batch_start in range(0, n, self.batch_size):
                batch_end = min(n, batch_start + self.batch_size)
                sl = slice(batch_start, batch_end)
                loc_result = localize_amplitude_vectors(
                    ampvecs[sl],
                    self.geom,
                    max_channels[sl],
                    channel_index=self.channel_index,
                    radius=self.radius,
                    n_channels_subset=self.n_channels_subset,
                    logbarrier=self.logbarrier,
                    dtype=self.dtype,
                    model=self.localization_model,
                )
                localizations[sl, 0] = loc_result["x"]
                if "y" in loc_result:
                    localizations[sl, 1] = loc_result["y"]
                else:
                    localizations[sl, 1] = 0.0
                localizations[sl, 2] = loc_result["z_abs"]
                if "alpha" in loc_result:
                    localizations[sl, 3] = loc_result["alpha"]
                else:
                    localizations[sl, 3] = 0.0

        return {self.name: localizations}


PointSourceLocalization = Localization
