import torch.serialization

from .amortized_localization import AmortizedLocalization
from .amplitudes import AmplitudeFeatures, AmplitudeVector, MaxAmplitude, Voltage
from .decollider import Decollider
from .denoising_scorer import DenoisingScorer
from .enforce_decrease import EnforceDecrease
from .fixed_prop import FixedProperty
from .interp import WaveformInterpolator
from .localize import Localization, PointSourceLocalization
from .matching_denoiser import DebugMatchingPursuitDenoiser
from .mixture_classifier import TruncatedMixtureModelTransformer
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
from .transform_base import BaseWaveformModule, Passthrough, Waveform
from .whiten import WaveformWhitener

all_transformers: list[type[BaseWaveformModule]] = [
    AmortizedLocalization,
    AmplitudeVector,
    AmplitudeFeatures,
    BaseTemporalPCA,
    DebugMatchingPursuitDenoiser,
    Decollider,
    DenoisingScorer,
    EnforceDecrease,
    FixedProperty,
    FullProbeTemporalPCAEmbedder,
    Localization,
    MaxAmplitude,
    Passthrough,
    PointSourceLocalization,
    SingleChannelWaveformDenoiser,
    SupervisedDenoiser,
    TemporalPCA,
    TemporalPCADenoiser,
    TemporalPCAFeaturizer,
    TruncatedMixtureModelTransformer,
    Voltage,
    WaveformInterpolator,
    WaveformWhitener,
    Waveform,
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
    torch.serialization.add_safe_globals(all_transformers + others)  # type: ignore  # ty: ignore[x]
