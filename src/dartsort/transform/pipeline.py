"""A class which manages pipelines of denoisers and featurizers"""

import torch


class WaveformPipeline(torch.nn.Module):
    def __init__(self, transformers, kwargs_to_store=None):
        super().__init__()
        check_unique_feature_names(transformers)
        self.transformers = torch.nn.ModuleList(transformers)
        self.kwargs_to_store = kwargs_to_store

    def __len__(self):
        return len(self.transformers)

    def __bool__(self):
        # if I am False, I return empty dicts always.
        return bool(len(self.transformers))

    def get_extra_state(self):
        if self.kwargs_to_store is None:
            return {}
        return dict(class_names_and_kwargs=self.kwargs_to_store)

    def set_extra_state(self, state):
        # needed so that load_state_dict doesn't complain if there's
        # extra state in there.
        pass

    @classmethod
    def from_state_dict_pt(cls, geom, channel_index, state_dict_pt):
        state_dict = torch.load(state_dict_pt)
        extra_state = state_dict.get("_extra_state", {})
        class_names_and_kwargs = extra_state.get("class_names_and_kwargs")
        if class_names_and_kwargs is None:
            raise ValueError(
                "Can't load a featurization pipeline from state dict if it "
                "wasn't initially created with from_config() or "
                "from_class_names_and_kwargs(). Instead, you can instantiate "
                "it the same way you originally did and use .load_state_dict() "
                "directly. (You may want to just use torch.save() and load()!)"
            )
        self = cls.from_class_names_and_kwargs(
            geom, channel_index, class_names_and_kwargs
        )
        self.precompute()
        # strict=False is needed here, because some transformers don't have
        # parameters initialized until their pre-load hooks are called, and
        # the strict check happens before that...
        self.load_state_dict(state_dict, strict=False)
        return self

    @classmethod
    def from_class_names_and_kwargs(cls, geom, channel_index, class_names_and_kwargs):
        from .all_transformers import transformers_by_class_name

        channel_index = torch.as_tensor(channel_index)
        geom = torch.as_tensor(geom)
        probe_kw = dict(channel_index=channel_index, geom=geom)

        transformers = []
        for name, kwargs in class_names_and_kwargs:
            transformer_cls = transformers_by_class_name[name]
            if kwargs.get("pretrained_path") is not None:
                transformer = transformer_cls.load_from_pt(**probe_kw, **kwargs)
            else:
                assert kwargs.pop("pretrained_path", None) is None
                transformer = transformer_cls(**probe_kw, **kwargs)
            transformers.append(transformer)

        return cls(transformers, kwargs_to_store=class_names_and_kwargs)

    @classmethod
    def from_config(
        cls,
        featurization_config,
        waveform_config,
        recording=None,
        geom=None,
        channel_index=None,
        sampling_frequency=30_000,
    ):
        if geom is None:
            from dartsort.util.waveform_util import make_channel_index

            assert recording is not None
            sampling_frequency = recording.sampling_frequency
            geom = torch.tensor(recording.get_channel_locations())
            channel_index = make_channel_index(
                geom, featurization_config.extract_radius, to_torch=True
            )
        else:
            assert recording is None
            assert channel_index is not None
        args = featurization_config_to_class_names_and_kwargs(
            featurization_config, waveform_config, sampling_frequency=sampling_frequency
        )
        return cls.from_class_names_and_kwargs(geom, channel_index, args)

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
                waveforms, new_features = transformer(
                    waveforms, max_channels=max_channels
                )
                features.update(new_features)

            elif transformer.is_featurizer:
                features.update(
                    transformer.transform(waveforms, max_channels=max_channels)
                )

            elif transformer.is_denoiser:
                waveforms = transformer(waveforms, max_channels=max_channels)

        return waveforms, features

    def fit(self, waveforms, max_channels, recording, weights=None):
        assert waveforms.ndim == 3
        assert max_channels.shape[0] == waveforms.shape[0]

        if not self.needs_fit():
            return

        for transformer in self.transformers:
            if transformer.needs_fit():
                transformer.train()
                transformer.fit(
                    waveforms,
                    max_channels=max_channels,
                    recording=recording,
                    weights=weights,
                )
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


def featurization_config_to_class_names_and_kwargs(
    featurization_config,
    waveform_config,
    sampling_frequency=30_000,
):
    """Convert this config into a list of waveform transformer classes and arguments

    Used by WaveformPipeline.from_config(...) to construct WaveformPipelines
    from FeaturizationConfig objects.
    """
    fc = featurization_config
    if fc.skip:
        return []

    class_names_and_kwargs = []
    do_feats = not fc.denoise_only
    sls_kw = dict(
        spike_length_samples=waveform_config.spike_length_samples(sampling_frequency)
    )

    if do_feats and fc.save_input_voltages:
        class_names_and_kwargs.append(
            ("Voltage", {"name_prefix": fc.input_waveforms_name})
        )
    if do_feats and fc.save_input_waveforms:
        class_names_and_kwargs.append(
            ("Waveform", {"name_prefix": fc.input_waveforms_name, **sls_kw})
        )
    if fc.learn_cleaned_tpca_basis:
        class_names_and_kwargs.append(
            ("BaseTemporalPCA", {"rank": fc.tpca_rank, "centered": False})
        )
    if do_feats and fc.save_input_tpca_projs:
        tslice = fc.input_tpca_waveform_config.relative_slice(waveform_config)
        class_names_and_kwargs.append(
            (
                "TemporalPCAFeaturizer",
                {
                    "rank": fc.tpca_rank,
                    "name_prefix": fc.input_waveforms_name,
                    "centered": fc.tpca_centered,
                    "temporal_slice": tslice,
                    "max_waveforms": fc.tpca_max_waveforms,
                },
            )
        )
    if fc.do_nn_denoise:
        class_names_and_kwargs.append(
            (
                fc.nn_denoiser_class_name,
                {
                    "pretrained_path": fc.nn_denoiser_pretrained_path,
                    "n_epochs": fc.nn_denoiser_train_epochs,
                    "epoch_size": fc.nn_denoiser_epoch_size,
                    **(fc.nn_denoiser_extra_kwargs or {}),
                },
            )
        )
    if fc.do_tpca_denoise:
        class_names_and_kwargs.append(
            (
                "TemporalPCADenoiser",
                {
                    "rank": fc.tpca_rank,
                    "fit_radius": fc.tpca_fit_radius,
                    "centered": fc.tpca_centered,
                },
            )
        )
    if fc.do_enforce_decrease:
        class_names_and_kwargs.append(("EnforceDecrease", {}))
    if do_feats and fc.save_output_waveforms:
        class_names_and_kwargs.append(
            ("Waveform", {"name_prefix": fc.output_waveforms_name, **sls_kw})
        )

    if do_feats and fc.save_output_tpca_projs:
        class_names_and_kwargs.append(
            (
                "TemporalPCAFeaturizer",
                {
                    "rank": fc.tpca_rank,
                    "name_prefix": fc.output_waveforms_name,
                    "centered": fc.tpca_centered,
                },
            )
        )
    if do_feats and fc.do_localization and fc.nn_localization:
        class_names_and_kwargs.append(
            (
                "AmortizedLocalization",
                {
                    "amplitude_kind": fc.localization_amplitude_type,
                    "localization_model": fc.localization_model,
                },
            )
        )

    do_ptp_amp = do_feats and fc.save_amplitudes
    do_peak_vec = do_feats and (
        fc.do_localization
        and fc.localization_amplitude_type == "peak"
        and not fc.nn_localization
    )
    do_ptp_vec = do_feats and fc.save_amplitudes
    do_logptt = do_feats and fc.save_amplitudes
    do_any_amp = do_peak_vec or do_ptp_vec or do_ptp_amp or do_logptt
    if do_any_amp:
        class_names_and_kwargs.append(
            (
                "AmplitudeFeatures",
                {
                    "name_prefix": fc.output_waveforms_name,
                    "ptp_max_amplitude": do_ptp_amp or fc.save_all_amplitudes,
                    "peak_amplitude_vectors": do_peak_vec or fc.save_all_amplitudes,
                    "ptp_amplitude_vectors": do_ptp_vec or fc.save_all_amplitudes,
                    "log_peak_to_trough": do_logptt or fc.save_all_amplitudes,
                },
            )
        )

    return class_names_and_kwargs
