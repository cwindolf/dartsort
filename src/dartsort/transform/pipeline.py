"""A class which manages pipelines of denoisers and featurizers"""

import torch

from .transform_base import BaseWaveformModule, BaseWaveformFeaturizer
from ..util.data_util import SpikeDataset


class WaveformPipeline(torch.nn.Module):
    def __init__(self, transformers: list[BaseWaveformModule], kwargs_to_store=None):
        super().__init__()
        check_unique_feature_names(transformers)
        self.transformers: list[BaseWaveformModule] = torch.nn.ModuleList(transformers)  # type: ignore
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

    def spike_datasets(self) -> list[SpikeDataset]:
        datasets = []
        for transformer in self.transformers:
            if transformer.is_featurizer:
                assert isinstance(transformer, BaseWaveformFeaturizer)
                datasets.extend(transformer.spike_datasets)
        return datasets

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
        featurization_cfg,
        waveform_cfg,
        recording=None,
        geom=None,
        channel_index=None,
        sampling_frequency: int | float=30_000,
    ):
        if geom is None:
            from dartsort.util.waveform_util import make_channel_index

            assert recording is not None
            sampling_frequency = recording.sampling_frequency
            geom = torch.tensor(recording.get_channel_locations())
            channel_index = make_channel_index(
                geom, featurization_cfg.extract_radius, to_torch=True
            )
        else:
            assert recording is None
            assert channel_index is not None
        args = featurization_config_to_class_names_and_kwargs(
            featurization_cfg, waveform_cfg, sampling_frequency=sampling_frequency
        )
        return cls.from_class_names_and_kwargs(geom, channel_index, args)

    def needs_precompute(self):
        return any(t.needs_precompute() for t in self.transformers)

    def needs_fit(self):
        return any(t.needs_fit() for t in self.transformers)

    def forward(self, waveforms, **fixed_properties):
        """
        fixed_properties usually contains max_channels, and may contain other relevant
        unchanging aspects of spikes like weights (used in fit()) or temporal shifts
        (used in pca slice logic).
        """
        waveforms = torch.asarray(waveforms)
        fixed_properties = {k: torch.asarray(v) for k, v in fixed_properties.items()}
        assert waveforms.ndim == 3
        for v in fixed_properties.values():
            assert v.shape[0] == waveforms.shape[0]

        features = {}

        if not waveforms.shape[0]:
            return waveforms, features

        for transformer in self.transformers:
            if transformer.is_featurizer and transformer.is_denoiser:
                waveforms, new_features = transformer(waveforms, **fixed_properties)
                features.update(new_features)
            elif transformer.is_featurizer:
                assert isinstance(transformer, BaseWaveformFeaturizer)
                features.update(transformer.transform(waveforms, **fixed_properties))
            elif transformer.is_denoiser:
                waveforms = transformer(waveforms, **fixed_properties)

        return waveforms, features

    def fit(self, recording, waveforms, **fixed_properties):
        waveforms = torch.asarray(waveforms)
        fixed_properties = {k: torch.asarray(v) for k, v in fixed_properties.items()}
        assert waveforms.ndim == 3
        for v in fixed_properties.values():
            assert v.shape[0] == waveforms.shape[0]

        if not self.needs_fit():
            return

        for transformer in self.transformers:
            if transformer.needs_fit():
                transformer.train()
                transformer.fit(
                    recording=recording, waveforms=waveforms, **fixed_properties
                )
            transformer.eval()
            transformer.requires_grad_(False)

            # if we're done already, stop before denoising
            if not self.needs_fit():
                break

            if transformer.is_denoiser:
                waveforms = transformer(waveforms, **fixed_properties)
                if transformer.is_featurizer:
                    # result is tuple wfs, feats
                    waveforms = waveforms[0]
        assert not waveforms.requires_grad

    def precompute(self):
        for transformer in self.transformers:
            transformer.precompute()

    def __iter__(self):
        return iter(self.transformers)

    def get_transformer(self, transformer_name):
        for t in self.transformers:
            if t.name == transformer_name:
                return t
        return None


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
    featurization_cfg,
    waveform_cfg,
    sampling_frequency: int | float=30_000,
):
    """Convert this config into a list of waveform transformer classes and arguments

    Used by WaveformPipeline.from_config(...) to construct WaveformPipelines
    from FeaturizationConfig objects.
    """
    fc = featurization_cfg
    if fc.skip:
        return []

    class_names_and_kwargs = []
    do_feats = not fc.denoise_only
    sls_kw = dict(
        spike_length_samples=waveform_cfg.spike_length_samples(sampling_frequency)
    )

    if do_feats and fc.save_input_voltages:
        class_names_and_kwargs.append(
            ("Voltage", {"name_prefix": fc.input_waveforms_name})
        )
    if fc.save_input_waveforms:
        class_names_and_kwargs.append(
            ("Waveform", {"name_prefix": fc.input_waveforms_name, **sls_kw})
        )
    if do_feats and fc.learn_cleaned_tpca_basis:
        class_names_and_kwargs.append(
            (
                "BaseTemporalPCA",
                {
                    "rank": fc.tpca_rank,
                    "name_prefix": fc.input_waveforms_name,
                    "centered": False,
                    "max_waveforms": fc.tpca_max_waveforms,
                    "fit_radius": fc.tpca_fit_radius,
                },
            )
        )

    # logic for picking an efficient combo of tpcas and nn denoisers
    class_names_and_kwargs.extend(_add_tpca_and_nn(featurization_cfg, waveform_cfg))

    if fc.do_enforce_decrease:
        class_names_and_kwargs.append(("EnforceDecrease", {}))
    if fc.save_output_waveforms:
        class_names_and_kwargs.append(
            ("Waveform", {"name_prefix": fc.output_waveforms_name, **sls_kw})
        )

    if fc.save_output_tpca_projs:
        class_names_and_kwargs.append(
            (
                "TemporalPCAFeaturizer",
                {
                    "rank": fc.tpca_rank,
                    "name_prefix": fc.output_waveforms_name,
                    "centered": fc.tpca_centered,
                    "fit_radius": fc.tpca_fit_radius,
                },
            )
        )

    # logic for grabbing localizations and amplitude vectors
    class_names_and_kwargs.extend(_add_localization_and_ampvec(featurization_cfg))

    return class_names_and_kwargs


def _add_tpca_and_nn(fc, wc):
    do_feats = not fc.denoise_only
    more = []

    # now, if there's no downstream waveform feature or output tpca feature,
    # we can combine the tpca featurization and denoising into one step
    # it's not exactly equivalent, since the denoiser would have run on the full
    # time length, but it's close enough
    combine = (
        do_feats
        and fc.save_input_tpca_projs
        and not (
            fc.do_nn_denoise or fc.save_output_waveforms or fc.save_output_tpca_projs
        )
    )
    if combine or (do_feats and fc.save_input_tpca_projs):
        tslice = fc.input_tpca_waveform_cfg.relative_slice(wc)
        more.append(
            (
                "TemporalPCA",
                {
                    "rank": fc.tpca_rank,
                    "name_prefix": fc.input_waveforms_name,
                    "centered": fc.tpca_centered,
                    "temporal_slice": tslice,
                    "max_waveforms": fc.tpca_max_waveforms,
                    "fit_radius": fc.tpca_fit_radius,
                },
            )
        )

    if combine:
        # that was it, all in one as discussed above.
        return more

    if fc.do_nn_denoise:
        more.append(
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
        more.append(
            (
                "TemporalPCADenoiser",
                {
                    "rank": fc.tpca_rank,
                    "fit_radius": fc.tpca_fit_radius,
                    "centered": fc.tpca_centered,
                    "max_waveforms": fc.tpca_max_waveforms,
                },
            )
        )

    return more


def _add_localization_and_ampvec(fc):
    do_feats = not fc.denoise_only
    more = []

    if do_feats and fc.do_localization and fc.nn_localization:
        more.append(
            (
                "AmortizedLocalization",
                {
                    "amplitude_kind": fc.localization_amplitude_type,
                    "localization_model": fc.localization_model,
                    "radius": fc.localization_radius,
                    "softmax_noise_floor": fc.localization_noise_floor,
                },
            )
        )
    if do_feats and fc.additional_com_localization:
        more.append(
            (
                "Localization",
                {
                    "amplitude_kind": fc.localization_amplitude_type,
                    "localization_model": "com",
                    "radius": fc.localization_radius,
                    "name": "com_localizations",
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
    if do_any_amp or (do_feats and fc.save_all_amplitudes):
        more.append(
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

    return more
