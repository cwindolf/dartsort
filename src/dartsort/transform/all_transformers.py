import torch

from .amortized_localization import AmortizedLocalization
from .amplitudes import (AmplitudeFeatures, AmplitudeVector, MaxAmplitude,
                         Voltage)
from .decollider import Decollider
from .enforce_decrease import EnforceDecrease
from .localize import Localization, PointSourceLocalization
from .pipeline import WaveformPipeline
from .single_channel_denoiser import SingleChannelWaveformDenoiser, SingleChannelDenoiser
from .temporal_pca import (TemporalPCA, TemporalPCADenoiser,
                           TemporalPCAFeaturizer)
from .transform_base import Passthrough, Waveform

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
]

transformers_by_class_name = {cls.__name__: cls for cls in all_transformers}

# serialization
if hasattr(torch.serialization, 'add_safe_globals'):
    others = [WaveformPipeline, set, torch.nn.ModuleList, SingleChannelDenoiser, slice, torch.nn.Sequential]
    torch.serialization.add_safe_globals(all_transformers + others)
