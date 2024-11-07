"""A class which manages pipelines of denoisers and featurizers
"""
import torch

class WaveformPipeline(torch.nn.Module):
    def __init__(self, transformers):
        super().__init__()
        check_unique_feature_names(transformers)
        self.transformers = torch.nn.ModuleList(transformers)

    @classmethod
    def from_class_names_and_kwargs(
        cls, geom, channel_index, class_names_and_kwargs
    ):
        from .all_transformers import transformers_by_class_name
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

    def needs_precompute(self):
        return any(t.needs_precompute() for t in self.transformers)

    def needs_fit(self):
        return any(t.needs_fit() for t in self.transformers)

    def forward(self, waveforms, max_channels):
        assert waveforms.ndim == 3
        assert max_channels.shape[0] == waveforms.shape[0]

        features = {}

        if not waveforms.shape[0]:
            return waveforms, features

        for transformer in self.transformers:
            if transformer.is_featurizer and transformer.is_denoiser:
                waveforms, new_features = transformer(waveforms, max_channels=max_channels)
                features.update(new_features)

            elif transformer.is_featurizer:
                features.update(
                    transformer.transform(
                        waveforms, max_channels=max_channels
                    )
                )

            elif transformer.is_denoiser:
                waveforms = transformer(waveforms, max_channels=max_channels)

        return waveforms, features

    def fit(self, waveforms, max_channels, recording):
        assert waveforms.ndim == 3
        assert max_channels.shape[0] == waveforms.shape[0]

        if not self.needs_fit():
            return

        for transformer in self.transformers:
            if transformer.needs_fit():
                transformer.train()
                transformer.fit(waveforms, max_channels=max_channels, recording=recording)
            transformer.eval()

            # if we're done already, stop before denoising
            if not self.needs_fit():
                break

            if transformer.is_denoiser:
                waveforms = transformer(waveforms, max_channels=max_channels)
                if transformer.is_featurizer:
                    # result is tuple wfs, feats
                    waveforms = waveforms[0]

    def precompute(self):
        for transformer in self.transformers:
            transformer.precompute()

    def __iter__(self):
        return iter(self.transformers)


def check_unique_feature_names(transformers):
    fnames = []
    for f in transformers:
        if f.is_featurizer:
            if f.is_multi:
                fnames.extend(f.name)
            else:
                fnames.append(f.name)
    if not len(fnames) == len(set(fnames)):
        raise ValueError("Featurizer name collision in a WaveformPipeline")


def featurization_config_to_class_names_and_kwargs(fconf):
    """Convert this config into a list of waveform transformer classes and arguments

    Used by WaveformPipeline.from_config(...) to construct WaveformPipelines
    from FeaturizationConfig objects.
    """
    class_names_and_kwargs = []

    do_feats = not fconf.denoise_only

    if do_feats and fconf.save_input_voltages:
        class_names_and_kwargs.append(
            ("Voltage", {"name_prefix": fconf.input_waveforms_name})
        )

    if do_feats and fconf.save_input_waveforms:
        class_names_and_kwargs.append(
            ("Waveform", {"name_prefix": fconf.input_waveforms_name})
        )
    # combined_tpca = (
    #     do_feats
    #     and fconf.save_input_tpca_projs
    #     and fconf.do_tpca_denoise
    #     and not fconf.do_nn_denoise
    # )
    # if combined_tpca:
    #     class_names_and_kwargs.append(
    #         (
    #             "TemporalPCA",
    #             {
    #                 "rank": fconf.tpca_rank,
    #                 "name_prefix": fconf.input_waveforms_name,
    #                 "centered": fconf.tpca_centered,
    #                 "temporal_slice": fconf.input_tpca_projs_temporal_slice,
    #             },
    #         )
    #     )
    # else:
    if do_feats and fconf.save_input_tpca_projs:
        class_names_and_kwargs.append(
            (
                "TemporalPCAFeaturizer",
                {
                    "rank": fconf.tpca_rank,
                    "name_prefix": fconf.input_waveforms_name,
                    "centered": fconf.tpca_centered,
                    "temporal_slice": fconf.input_tpca_projs_temporal_slice,
                },
            )
        )
    if fconf.do_nn_denoise:
        class_names_and_kwargs.append(
            (
                fconf.nn_denoiser_class_name,
                {
                    "pretrained_path": fconf.nn_denoiser_pretrained_path,
                    "n_epochs": fconf.nn_denoiser_train_epochs,
                    **(fconf.nn_denoiser_extra_kwargs or {}),
                },
            )
        )
    if fconf.do_tpca_denoise:
        class_names_and_kwargs.append(
            (
                "TemporalPCADenoiser",
                {
                    "rank": fconf.tpca_rank,
                    "fit_radius": fconf.tpca_fit_radius,
                    "centered": fconf.tpca_centered,
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
                    "centered": fconf.tpca_centered,
                },
            )
        )
    if do_feats and fconf.do_localization and fconf.nn_localization:
        class_names_and_kwargs.append(
            (
                "AmortizedLocalization",
                {
                    "amplitude_kind": fconf.localization_amplitude_type,
                    "localization_model": fconf.localization_model,
                },
            )
        )

    do_ptp_amp = do_feats and fconf.save_amplitudes
    do_peak_vec = do_feats and (
        fconf.do_localization
        and fconf.localization_amplitude_type == "peak"
        and not fconf.nn_localization
    )
    do_ptp_vec = do_feats and fconf.save_amplitudes
    do_logptt = do_feats and fconf.save_amplitudes
    do_any_amp = do_peak_vec or do_ptp_vec or do_ptp_amp or do_logptt
    if do_any_amp:
        class_names_and_kwargs.append(
            (
                "AmplitudeFeatures",
                {
                    "name_prefix": fconf.output_waveforms_name,
                    "ptp_max_amplitude": do_ptp_amp or fconf.save_all_amplitudes,
                    "peak_amplitude_vectors": do_peak_vec or fconf.save_all_amplitudes,
                    "ptp_amplitude_vectors": do_ptp_vec or fconf.save_all_amplitudes,
                    "log_peak_to_trough": do_logptt or fconf.save_all_amplitudes,
                },
            )
        )

    return class_names_and_kwargs
