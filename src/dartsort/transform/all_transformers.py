from .amplitudes import AmplitudeVector, MaxAmplitude
from .enforce_decrease import EnforceDecrease
from .localize import Localization, PointSourceLocalization
from .single_channel_denoiser import SingleChannelWaveformDenoiser
from .temporal_pca import TemporalPCADenoiser, TemporalPCAFeaturizer
from .transform_base import Waveform

all_transformers = [
    Waveform,
    AmplitudeVector,
    MaxAmplitude,
    EnforceDecrease,
    SingleChannelWaveformDenoiser,
    TemporalPCADenoiser,
    TemporalPCAFeaturizer,
    Localization,
    PointSourceLocalization,
]

transformers_by_class_name = {cls.__name__: cls for cls in all_transformers}
