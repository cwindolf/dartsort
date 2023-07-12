from .amplitudes import AmplitudeVector, MaxAmplitude
from .base import Waveform
from .enforce_decrease import EnforceDecrease
from .single_channel_denoiser import SingleChannelWaveformDenoiser
from .temporal_pca import TemporalPCADenoiser, TemporalPCAFeaturizer

all_transformers = [
    Waveform,
    AmplitudeVector,
    MaxAmplitude,
    EnforceDecrease,
    SingleChannelWaveformDenoiser,
    TemporalPCADenoiser,
    TemporalPCAFeaturizer,
]

transformers_by_class_name = {cls.__name__: cls for cls in all_transformers}
