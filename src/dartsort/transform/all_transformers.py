import torch
from .amplitudes import AmplitudeFeatures, AmplitudeVector, MaxAmplitude, Voltage
from .enforce_decrease import EnforceDecrease
from .localize import Localization, PointSourceLocalization
from .amortized_localization import AmortizedLocalization
from .single_channel_denoiser import SingleChannelWaveformDenoiser
from .temporal_pca import TemporalPCADenoiser, TemporalPCAFeaturizer, TemporalPCA
from .transform_base import Waveform, Passthrough
from .decollider import Decollider
from .pipeline import WaveformPipeline

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
    AmortizedLocalization,
    AmplitudeFeatures,
    TemporalPCA,
    Voltage,
    Decollider,
    Passthrough,
]

transformers_by_class_name = {cls.__name__: cls for cls in all_transformers}

# serialization
others = [WaveformPipeline, set, torch.nn.ModuleList, slice, torch.nn.Sequential]
torch.serialization.add_safe_globals(all_transformers + others)
