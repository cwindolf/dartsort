import torch

from .amortized_localization import AmortizedLocalization
from .amplitudes import AmplitudeFeatures, AmplitudeVector, MaxAmplitude, Voltage
from .decollider import Decollider
from .enforce_decrease import EnforceDecrease
from .fixed_prop import FixedProperty
from .interp import WaveformInterpolator
from .localize import Localization, PointSourceLocalization
from .matching_denoiser import DebugMatchingPursuitDenoiser
from .pipeline import WaveformPipeline
from .single_channel_denoiser import (
    SingleChannelDenoiser,
    SingleChannelWaveformDenoiser,
)
from .supervised_denoiser import SupervisedDenoiser
from .temporal_pca import (
    BaseTemporalPCA,
    FullProbeTemporalPCAEmbedder,
    TemporalPCA,
    TemporalPCADenoiser,
    TemporalPCAFeaturizer,
)
from .transform_base import Passthrough, Waveform, BaseWaveformModule
from .whiten import WaveformWhitener

all_transformers: list[type[BaseWaveformModule]] = [
    Waveform,
    AmplitudeVector,
    MaxAmplitude,
    EnforceDecrease,
    FixedProperty,
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
    DebugMatchingPursuitDenoiser,
    WaveformInterpolator,
    FullProbeTemporalPCAEmbedder,
    WaveformWhitener,
]

transformers_by_class_name = {cls.__name__: cls for cls in all_transformers}

# serialization
if hasattr(torch.serialization, "add_safe_globals"):
    from ..util.internal_config import WaveformConfig

    others = [
        set,
        slice,
        torch.nn.ModuleList,
        torch.nn.Sequential,
        WaveformPipeline,
        SingleChannelDenoiser,
        WaveformConfig,
    ]
    torch.serialization.add_safe_globals(all_transformers + others)  # type: ignore
