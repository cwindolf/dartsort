"""A class which manages pipelines of denoisers and featurizers
"""
from collections import namedtuple

import torch

NamedTransformer = namedtuple("NamedTransformer", ["name", "transformer"])


class WaveformPipeline(torch.nn.Module):
    def __init__(self, named_transformers):
        super().__init__()
        self.named_transformers = named_transformers
        # register the modules so .to() etc work
        for name, transformer in self.named_transformers:
            self.add_module(name, transformer)

    @property
    def needs_fit(self):
        return any(t.transformer.needs_fit for t in self.named_transformers)

    def forward(self, waveforms, max_channels=None):
        features = {}
        for name, transformer in self.named_transformers:
            # didn't figure out both at once yet bc we don't use it
            assert not (transformer.is_denoiser and transformer.is_featurizer)
            if transformer.is_denoiser:
                waveforms = transformer(waveforms, max_channels=max_channels)
            if transformer.is_featurizer:
                features[name] = transformer.transform(
                    waveforms, max_channels=max_channels
                )
        return waveforms, features

    def fit(self, waveforms, max_channels=None):
        if not self.needs_fit:
            return

        for name, transformer in self.named_transformers:
            # didn't figure out both at once yet bc we don't use it
            transformer.fit(waveforms, max_channels=max_channels)
            if transformer.is_denoiser:
                waveforms = transformer(waveforms, max_channels=max_channels)
