"""A class which manages pipelines of denoisers and featurizers
"""
import torch

from .all_transformers import transformers_by_class_name


class WaveformPipeline(torch.nn.Module):
    def __init__(self, transformers):
        super().__init__()
        check_unique_feature_names(transformers)
        self.transformers = torch.nn.ModuleList(transformers)

    @classmethod
    def from_class_names_and_kwargs(
        cls, geom, channel_index, class_names_and_kwargs
    ):
        return cls(
            [
                transformers_by_class_name[name](
                    channel_index=torch.as_tensor(channel_index),
                    geom=torch.as_tensor(geom),
                    **kwargs
                )
                for name, kwargs in class_names_and_kwargs
            ]
        )

    @classmethod
    def from_config(cls, geom, channel_index, featurization_config):
        return cls.from_class_names_and_kwargs(
            geom,
            channel_index,
            featurization_config_to_class_names_and_kwargs(featurization_config),
        )

    def needs_fit(self):
        return any(t.needs_fit() for t in self.transformers)

    def forward(self, waveforms, max_channels):
        assert waveforms.ndim == 3
        assert max_channels.shape[0] == waveforms.shape[0]

        features = {}

        if not waveforms.shape[0]:
            return waveforms, features

        for transformer in self.transformers:
            if transformer.is_featurizer:
                features[transformer.name] = transformer.transform(
                    waveforms, max_channels=max_channels
                )
            if transformer.is_denoiser:
                waveforms = transformer(waveforms, max_channels=max_channels)

        return waveforms, features

    def fit(self, waveforms, max_channels):
        assert waveforms.ndim == 3
        assert max_channels.shape[0] == waveforms.shape[0]

        if not self.needs_fit():
            return

        for transformer in self.transformers:
            transformer.fit(waveforms, max_channels=max_channels)
            if transformer.is_denoiser:
                waveforms = transformer(waveforms, max_channels=max_channels)

    def __iter__(self):
        return iter(self.transformers)


def check_unique_feature_names(transformers):
    fnames = [f.name for f in transformers if f.is_featurizer]
    if not len(fnames) == len(set(fnames)):
        raise ValueError("Featurizer name collision in a WaveformPipeline")


def featurization_config_to_class_names_and_kwargs(fconf):
    """Convert this config into a list of waveform transformer classes and arguments

    Used by WaveformPipeline.from_config(...) to construct WaveformPipelines
    from FeaturizationConfig objects.
    """
    class_names_and_kwargs = []

    do_feats = not fconf.denoise_only

    if do_feats and fconf.save_input_waveforms:
        class_names_and_kwargs.append(
            ("Waveform", {"name_prefix": fconf.input_waveforms_name})
        )
    if do_feats and fconf.save_input_tpca_projs:
        class_names_and_kwargs.append(
            (
                "TemporalPCAFeaturizer",
                {
                    "rank": fconf.tpca_rank,
                    "name_prefix": fconf.input_waveforms_name,
                },
            )
        )
    if fconf.do_nn_denoise:
        class_names_and_kwargs.append(
            (
                fconf.nn_denoiser_class_name,
                {"pretrained_path": fconf.nn_denoiser_pretrained_path},
            )
        )
    if fconf.do_tpca_denoise:
        class_names_and_kwargs.append(
            (
                "TemporalPCADenoiser",
                {
                    "rank": fconf.tpca_rank,
                    "fit_radius": fconf.tpca_fit_radius,
                },
            )
        )
    if fconf.do_enforce_decrease:
        class_names_and_kwargs.append(("EnforceDecrease", {}))
    if do_feats and fconf.save_output_waveforms:
        class_names_and_kwargs.append(
            (
                "Waveform",
                {"name_prefix": fconf.output_waveforms_name},
            )
        )
    if do_feats and fconf.save_output_tpca_projs:
        class_names_and_kwargs.append(
            (
                "TemporalPCAFeaturizer",
                {
                    "rank": fconf.tpca_rank,
                    "name_prefix": fconf.output_waveforms_name,
                },
            )
        )
    if do_feats and fconf.do_localization and fconf.localization_amplitude_type == "peak":
        class_names_and_kwargs.append(
            (
                "AmplitudeVector",
                {"name_prefix": fconf.output_waveforms_name},
            )
        )
    if do_feats:
        # we'll also need a peak-to-peak amplitude vector in other places
        class_names_and_kwargs.append(
            (
                "AmplitudeVector",
                {"name_prefix": fconf.output_waveforms_name, "kind": "ptp"},
            )
        )
    if do_feats and fconf.save_amplitudes:
        class_names_and_kwargs.append(
            (
                "MaxAmplitude",
                {"name_prefix": fconf.output_waveforms_name},
            )
        )

    return class_names_and_kwargs
