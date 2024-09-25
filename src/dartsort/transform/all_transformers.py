from .amplitudes import AmplitudeFeatures, AmplitudeVector, MaxAmplitude, Voltage
from .enforce_decrease import EnforceDecrease
from .localize import Localization, PointSourceLocalization
from .vae_localize import VAELocalization
from .single_channel_denoiser import SingleChannelWaveformDenoiser
from .temporal_pca import TemporalPCADenoiser, TemporalPCAFeaturizer, TemporalPCA
from .transform_base import Waveform, Passthrough
from .decollider import Decollider

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
    VAELocalization,
    AmplitudeFeatures,
    TemporalPCA,
    Voltage,
    Decollider,
    Passthrough,
]

transformers_by_class_name = {cls.__name__: cls for cls in all_transformers}
