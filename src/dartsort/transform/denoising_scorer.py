from typing import TYPE_CHECKING

import torch

from .transform_base import BaseWaveformAutoencoder

if TYPE_CHECKING:
    from .pipeline import WaveformPipeline


class DenoisingScorer(BaseWaveformAutoencoder):
    default_name = "denoising_scores"
    shape = ()
    dtype = torch.float32

    def __init__(self, denoising_pipeline: "WaveformPipeline"):
        super().__init__(channel_index=denoising_pipeline.channel_index)
        self.denoising_pipeline = denoising_pipeline

    def forward(self, waveforms, **spike_data):
        denoised_waveforms, features = self.denoising_pipeline(waveforms, **spike_data)
        _wf = features.pop("waveforms")
        assert _wf is denoised_waveforms
        del _wf

        buf = (waveforms * denoised_waveforms).nan_to_num_()
        conv = buf.sum(dim=(1, 2))
        torch.square(denoised_waveforms, out=buf).nan_to_num_()
        norm = buf.sum(dim=(1, 2))
        reduction = conv.mul_(2.0).sub_(norm).abs_().sqrt_()

        features[self.name] = reduction
        return denoised_waveforms, features
