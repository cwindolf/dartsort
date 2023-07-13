"""A class which manages pipelines of denoisers and featurizers
"""
import torch

from .all_transformers import transformers_by_class_name


class WaveformPipeline(torch.nn.Module):
    def __init__(self, transformers):
        super().__init__()
        self.transformers = transformers
        # register the modules so torch's .to() etc work
        for transformer in self.transformers:
            self.add_module(transformer.name, transformer)

    @classmethod
    def from_class_names_and_kwargs(
        cls, geom, channel_index, class_names_and_kwargs
    ):
        return cls(
            [
                transformers_by_class_name[name](
                    geom=geom, channel_index=channel_index, **kwargs
                )
                for name, kwargs in class_names_and_kwargs
            ]
        )

    @classmethod
    def from_config(cls, geom, channel_index, featurization_config):
        return cls.from_class_names_and_kwargs(
            geom,
            channel_index,
            featurization_config.to_class_names_and_kwargs(),
        )

    @property
    def needs_fit(self):
        return any(t.transformer.needs_fit for t in self.named_transformers)

    def forward(self, waveforms, max_channels=None):
        if max_channels is not None:
            assert (waveforms.shape[0],) == max_channels.shape

        features = {}

        if not waveforms.shape[0]:
            return waveforms, features

        for transformer in self.transformers:
            # didn't figure out both at once yet bc we don't use it
            assert not (transformer.is_denoiser and transformer.is_featurizer)
            if transformer.is_featurizer:
                features[transformer.name] = transformer.transform(
                    waveforms, max_channels=max_channels
                )
            if transformer.is_denoiser:
                waveforms = transformer(waveforms, max_channels=max_channels)

        return waveforms, features

    def fit(self, waveforms, max_channels=None):
        if not self.needs_fit:
            return

        for name, transformer in self.named_transformers:
            transformer.fit(waveforms, max_channels=max_channels)
            if transformer.is_denoiser:
                waveforms = transformer(waveforms, max_channels=max_channels)
