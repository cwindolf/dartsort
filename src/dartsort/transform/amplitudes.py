import torch
from dartsort.util.spiketorch import ptp

from .transform_base import BaseWaveformFeaturizer


class AmplitudeFeatures(BaseWaveformFeaturizer):
    is_multi = True

    def __init__(
        self,
        channel_index,
        geom=None,
        dtype=torch.float,
        name_prefix="",
        ptp_max_amplitude=True,
        peak_amplitude_vectors=True,
        ptp_amplitude_vectors=True,
        log_peak_to_trough=True,
    ):
        self.shape = []
        names = []
        self.dtype = []

        self.ptp_max_amplitude = ptp_max_amplitude
        if ptp_max_amplitude:
            name = "ptp_amplitudes"
            if name_prefix:
                name = f"{name_prefix}_{name}"
            self.ptp_max_amplitude_name = name
            self.shape.append(())
            names.append(name)
            self.dtype.append(dtype)

        self.peak_amplitude_vectors = peak_amplitude_vectors
        if peak_amplitude_vectors:
            name = "peak_amplitude_vectors"
            if name_prefix:
                name = f"{name_prefix}_{name}"
            self.peak_amplitude_vectors_name = name
            self.shape.append((channel_index.shape[1],))
            names.append(name)
            self.dtype.append(dtype)

        self.ptp_amplitude_vectors = ptp_amplitude_vectors
        if ptp_amplitude_vectors:
            name = "ptp_amplitude_vectors"
            if name_prefix:
                name = f"{name_prefix}_{name}"
            self.ptp_amplitude_vectors_name = name
            self.shape.append((channel_index.shape[1],))
            names.append(name)
            self.dtype.append(dtype)

        self.log_peak_to_trough = log_peak_to_trough
        if log_peak_to_trough:
            name = "logpeaktotrough"
            if name_prefix:
                name = f"{name_prefix}_{name}"
            self.log_peak_to_trough_name = name
            self.shape.append(())
            names.append(name)
            self.dtype.append(dtype)

        self.compute_minmax_vectors = (
            ptp_max_amplitude or ptp_amplitude_vectors or log_peak_to_trough
        )
        self.compute_maxchan = ptp_max_amplitude or log_peak_to_trough

        super().__init__(name=names)

    def transform(self, waveforms, max_channels=None):
        features = {}
        if self.peak_amplitude_vectors:
            features[self.peak_amplitude_vectors_name] = (
                waveforms.abs().max(dim=1).values
            )

        if self.compute_minmax_vectors:
            max_vectors = torch.nan_to_num(waveforms.max(dim=1).values)
            min_vectors = torch.nan_to_num(waveforms.min(dim=1).values)
            ptp_vectors = max_vectors - min_vectors
        if self.compute_maxchan:
            maxchans = torch.argmax(torch.nan_to_num(ptp_vectors), dim=1, keepdim=True)
            if self.log_peak_to_trough:
                maxs = torch.take_along_dim(max_vectors, maxchans, dim=1)[:, 0]
                mins = torch.take_along_dim(min_vectors, maxchans, dim=1)[:, 0]
                if self.ptp_max_amplitude:
                    ptps = maxs - mins
            elif self.ptp_max_amplitude:
                ptps = torch.take_along_dim(ptp_vectors, maxchans, dim=1)[:, 0]

        if self.ptp_amplitude_vectors:
            features[self.ptp_amplitude_vectors_name] = ptp_vectors
        if self.ptp_max_amplitude:
            features[self.ptp_max_amplitude_name] = ptps
        if self.log_peak_to_trough:
            features[self.log_peak_to_trough_name] = torch.log(maxs) - torch.log(
                mins.neg()
            )

        return features


class AmplitudeVector(BaseWaveformFeaturizer):
    default_name = "amplitude_vectors"

    def __init__(
        self,
        channel_index,
        geom=None,
        kind="peak",
        dtype=torch.float,
        name=None,
        name_prefix="",
    ):
        assert kind in ("peak", "ptp")
        if name is None:
            name = f"{kind}_{self.default_name}"
            if name_prefix:
                name = f"{name_prefix}_{name}"
        super().__init__(name=name, name_prefix=name_prefix)
        self.kind = kind
        self.shape = (channel_index.shape[1],)
        self.dtype = dtype

    def transform(self, waveforms, max_channels=None):
        if self.kind == "peak":
            return {self.name: waveforms.abs().max(dim=1).values}
        elif self.kind == "ptp":
            return {self.name: ptp(waveforms, dim=1)}


class MaxAmplitude(BaseWaveformFeaturizer):
    default_name = "amplitudes"
    shape = ()

    def __init__(
        self,
        channel_index=None,
        geom=None,
        kind="ptp",
        dtype=torch.float,
        name=None,
        name_prefix="",
    ):
        assert kind in ("peak", "ptp")
        if name is None:
            name = f"{kind}_{self.default_name}"
            if name_prefix:
                name = f"{name_prefix}_{name}"
        super().__init__(name=name, name_prefix=name_prefix)
        self.kind = kind
        self.dtype = dtype

    def transform(self, waveforms, max_channels=None):
        if self.kind == "peak":
            return {self.name: torch.nan_to_num(waveforms.abs()).max(dim=(1, 2)).values}
        elif self.kind == "ptp":
            return {
                self.name: torch.nan_to_num(ptp(waveforms, dim=1)).max(dim=1).values
            }
