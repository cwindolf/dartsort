import torch

from .amortized_localization import AmortizedLocalization
from .amplitudes import AmplitudeFeatures, AmplitudeVector, MaxAmplitude, Voltage
from .decollider import Decollider
from .enforce_decrease import EnforceDecrease
from .localize import Localization, PointSourceLocalization
from .matching_denoiser import DebugMatchingPursuitDenoiser
from .pipeline import WaveformPipeline
from .single_channel_denoiser import (
    SingleChannelDenoiser,
    SingleChannelWaveformDenoiser,
)
from .single_channel_templates import SingleChannelTemplates
from .supervised_denoiser import SupervisedDenoiser
from .temporal_pca import (
    BaseTemporalPCA,
    TemporalPCA,
    TemporalPCADenoiser,
    TemporalPCAFeaturizer,
)
from .transform_base import Passthrough, Waveform

all_transformers = [
    Waveform,
    AmplitudeVector,
    MaxAmplitude,
    EnforceDecrease,
    SingleChannelWaveformDenoiser,
    BaseTemporalPCA,
    TemporalPCADenoiser,
    TemporalPCAFeaturizer,
    TemporalPCA,
    Localization,
    PointSourceLocalization,
    AmortizedLocalization,
    AmplitudeFeatures,
    Voltage,
    Decollider,
    SupervisedDenoiser,
    Passthrough,
    SingleChannelTemplates,
    DebugMatchingPursuitDenoiser,
]

transformers_by_class_name = {cls.__name__: cls for cls in all_transformers}

# serialization
if hasattr(torch.serialization, "add_safe_globals"):
    others = [
        set,
        slice,
        torch.nn.ModuleList,
        torch.nn.Sequential,
        WaveformPipeline,
        SingleChannelDenoiser,
    ]
    torch.serialization.add_safe_globals(all_transformers + others)
