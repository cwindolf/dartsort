"""A class which manages pipelines of denoisers and featurizers"""

from copy import deepcopy
from pathlib import Path
from typing import Sequence

import torch
from spikeinterface.core import BaseRecording

from ..util.data_util import SpikeDataset, yield_chunks
from ..util.internal_config import (
    ComputationConfig,
    FeaturizationConfig,
    WaveformConfig,
)
from ..util.logging_util import get_logger
from ..util.py_util import ensure_path
from ..util.waveform_util import assert_all_finite_in_probe
from .transform_base import BaseWaveformFeaturizer, BaseWaveformModule

logger = get_logger(__name__)


class WaveformPipeline(torch.nn.Module):
    """Pipelines of featurization nodes."""

    def __init__(
        self,
        transformers: Sequence[BaseWaveformModule],
        channel_index: torch.Tensor | None = None,
        kwargs_to_store=None,
        waveform_cfg=None,
        motion=None,
        sampling_frequency=30_000.0,
    ):
        super().__init__()
        check_unique_feature_names(transformers)
        self.transformers: list[BaseWaveformModule] = torch.nn.ModuleList(transformers)  # type: ignore
        self.kwargs_to_store = kwargs_to_store
        self.sampling_frequency = sampling_frequency
        self.waveform_cfg = waveform_cfg
        self.register_buffer("_device_tracker", torch.zeros((0,)))
        if channel_index is not None:
            self.register_buffer("channel_index", channel_index.clone())
        else:
            self.channel_index = None
        self.safe = False
        self.motion = motion

    @property
    def device(self):
        return self._device_tracker.device

    def __len__(self):
        return len(self.transformers)

    def __bool__(self):
        # if I am False, I return empty dicts always.
        return bool(len(self.transformers))

    def get_extra_state(self):
        extra_state = {}
        if self.kwargs_to_store is not None:
            extra_state["class_names_and_kwargs"] = self.kwargs_to_store
        if self.waveform_cfg is not None:
            extra_state["waveform_cfg"] = self.waveform_cfg
        if self.sampling_frequency is not None:
            extra_state["sampling_frequency"] = self.sampling_frequency
        if self.motion is not None:
            extra_state["motion"] = self.motion
        return extra_state

    def set_extra_state(self, state):
        # needed so that load_state_dict doesn't complain if there's
        # extra state in there.
        pass

    def spike_datasets(
        self,
        start_index: int | None = None,
        up_to_index: int | None = None,
        force_save: bool = False,
    ) -> list[SpikeDataset]:
        datasets = []
        for tix, transformer in enumerate(self.transformers):
            if start_index is not None and tix < start_index:
                continue
            if tix == up_to_index:
                break
            if transformer.is_featurizer:
                assert isinstance(transformer, BaseWaveformFeaturizer)
                datasets.extend(transformer.spike_datasets(force_save=force_save))
        return datasets

    @classmethod
    def from_state_dict_pt(cls, geom, state_dict_pt, motion=None):
        """Load a pipeline from file."""
        state_dict = torch.load(state_dict_pt, weights_only=True)
        extra_state = state_dict.get("_extra_state", {})
        channel_index = state_dict["channel_index"]
        class_names_and_kwargs = extra_state.get("class_names_and_kwargs")
        waveform_cfg = extra_state.get("waveform_cfg")
        sampling_frequency = extra_state.get("sampling_frequency")
        motion = extra_state.get("motion", motion)
        if class_names_and_kwargs is None:
            raise ValueError(
                "Can't load a featurization pipeline from state dict if it "
                "wasn't initially created with from_config() or "
                "from_class_names_and_kwargs(). Instead, you can instantiate "
                "it the same way you originally did and use .load_state_dict() "
                "directly. (You may want to just use torch.save() and load()!)"
            )
        self = cls.from_class_names_and_kwargs(
            geom,
            channel_index,
            class_names_and_kwargs,
            waveform_cfg=waveform_cfg,
            sampling_frequency=sampling_frequency,
        )
        if motion is not None:
            self.attach_motion(motion)
        self.precompute()
        # strict=False is needed here, because some transformers don't have
        # parameters initialized until their pre-load hooks are called, and
        # the strict check happens before that...
        self.load_state_dict(state_dict, strict=False)
        return self

    @classmethod
    def from_class_names_and_kwargs(
        cls,
        geom,
        channel_index,
        class_names_and_kwargs,
        waveform_cfg: WaveformConfig | None,
        sampling_frequency: float = 30_000.0,
    ):
        """Construct a pipeline from a sequence of BaseWaveformModule class names and constructor arguments."""
        from .all_transformers import transformers_by_class_name

        # need to modify this dict, so don't mess with the original
        class_names_and_kwargs = deepcopy(class_names_and_kwargs)

        channel_index = torch.as_tensor(channel_index)
        geom = torch.as_tensor(geom)

        transformers = []
        for name, kwargs in class_names_and_kwargs:
            transformer_cls = transformers_by_class_name[name]
            pretrained_path = kwargs.pop("pretrained_path", None)
            if pretrained_path is not None:
                transformer = transformer_cls.load_from_pt(
                    pretrained_path=pretrained_path,
                    channel_index=channel_index,
                    geom=geom,
                    **kwargs,
                )
            else:
                transformer = transformer_cls(
                    channel_index=channel_index,
                    geom=geom,
                    waveform_cfg=waveform_cfg,
                    sampling_frequency=sampling_frequency,
                    **kwargs,
                )
            transformers.append(transformer)

        return cls(
            transformers,
            channel_index=channel_index,
            kwargs_to_store=class_names_and_kwargs,
            waveform_cfg=waveform_cfg,
            sampling_frequency=sampling_frequency,
        )

    @classmethod
    def from_config(
        cls,
        *,
        featurization_cfg: FeaturizationConfig,
        waveform_cfg: WaveformConfig,
        recording=None,
        geom=None,
        channel_index=None,
        sampling_frequency: float,
    ):
        """Construct a pipeline based on configuration options."""
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
            featurization_cfg,
            waveform_cfg,
            sampling_frequency=sampling_frequency,
        )
        return cls.from_class_names_and_kwargs(
            geom,
            channel_index,
            args,
            waveform_cfg=waveform_cfg,
            sampling_frequency=sampling_frequency,
        )

    def needs_precompute(self):
        return any(t.needs_precompute() for t in self.transformers)

    def needs_fit(self):
        return any(t.needs_fit() for t in self.transformers)

    def attach_motion(self, motion):
        for t in self.transformers:
            t.attach_motion(motion)

    def register_cpu_workers(self, workers: int):
        for t in self.transformers:
            t.register_cpu_workers(workers)

    def needs_residual(self):
        return any(t.needs_residual for t in self.transformers)

    def needs_more_features(self):
        return any(t.needs_more_features for t in self.transformers)

    def forward(
        self,
        waveforms,
        *,
        up_to_index: int | None = None,
        start_index: int | None = None,
        **fixed_properties,
    ):
        """Run waveforms and fixed properties through pipeline, extracting features and denoising."""
        waveforms = torch.asarray(waveforms)
        fixed_properties = {k: torch.asarray(v) for k, v in fixed_properties.items()}
        assert waveforms.ndim == 3
        for v in fixed_properties.values():
            assert v.shape == () or v.shape[0] == waveforms.shape[0]

        features = fixed_properties.copy()
        features["waveforms"] = waveforms

        if not waveforms.shape[0]:
            return waveforms, features

        for tix, transformer in enumerate(self.transformers):
            if start_index is not None and tix < start_index:
                continue
            if tix == up_to_index:
                break
            if transformer.is_featurizer and transformer.is_denoiser:
                waveforms, new_features = transformer(**features)
                features.update(waveforms=waveforms, **new_features)
            elif transformer.is_featurizer:
                assert isinstance(transformer, BaseWaveformFeaturizer)
                features.update(transformer.transform(**features))
            elif transformer.is_denoiser:
                features["waveforms"] = waveforms = transformer(**features)

            if self.safe:
                assert_all_finite_in_probe(
                    waveforms,
                    features["channels"],
                    self.channel_index,
                    str(transformer),
                )

        return waveforms, features

    def fit(
        self,
        recording: BaseRecording,
        waveforms: torch.Tensor,
        computation_cfg: ComputationConfig,
        *,
        hdf5_filename: str | Path | None = None,
        waveforms_dataset_name: str = "waveforms",
        **fixed_properties: torch.Tensor,
    ):
        """Fit my transformers in sequence, giving each the outputs of its predecessors."""
        waveforms = torch.asarray(waveforms)
        fixed_properties = {k: torch.asarray(v) for k, v in fixed_properties.items()}
        assert waveforms.ndim == 3
        for v in fixed_properties.values():
            if torch.is_tensor(v):
                assert v.shape[0] == waveforms.shape[0]

        if not self.needs_fit():
            return

        features = fixed_properties.copy()
        features["waveforms"] = waveforms
        del waveforms

        if hdf5_filename is not None:
            hdf5_filename = ensure_path(hdf5_filename, strict=True)

        if self.safe:
            assert torch.is_tensor(features["waveforms"])
            assert_all_finite_in_probe(
                features["waveforms"], features["channels"], self.channel_index, "Init"
            )

        for tix, transformer in enumerate(self.transformers):
            if transformer.fits_from_disk:
                assert hdf5_filename is not None
                self.transform_to_disk(
                    hdf5_filename=hdf5_filename,
                    waveforms_dataset_name=waveforms_dataset_name,
                    up_to_index=tix,
                )

            if transformer.needs_fit():
                transformer.train()
                transformer.fit(
                    recording=recording,
                    computation_cfg=computation_cfg,
                    hdf5_filename=hdf5_filename,
                    pipeline=self,
                    **features,
                )
            transformer.eval()
            transformer.requires_grad_(False)

            # if we're done already, stop before denoising
            if not self.needs_fit():
                break

            if transformer.is_featurizer and transformer.is_denoiser:
                waveforms, new_features = transformer(**features)
                features.update(waveforms=waveforms, **new_features)
            elif transformer.is_featurizer:
                assert isinstance(transformer, BaseWaveformFeaturizer)
                features.update(transformer.transform(**features))
            elif transformer.is_denoiser:
                features["waveforms"] = transformer(**features)

            if self.safe:
                assert torch.is_tensor(features["waveforms"])
                this = str(type(transformer).__name__)
                prev = ">".join(str(type(t).__name__) for t in self.transformers[:tix])
                assert_all_finite_in_probe(
                    features["waveforms"],
                    features["channels"],
                    self.channel_index,
                    f"({tix}): {this} <- [{prev}]",
                )

        assert not features["waveforms"].requires_grad
        assert not self.needs_fit()

    def precompute(self):
        """Give my transformers the chance to precompute stuff."""
        for transformer in self.transformers:
            transformer.precompute()
        for tf in self.transformers:
            if tf.needs_precompute():
                raise ValueError(f"Precompute didn't stick in transformer {tf}.")

    def transform_to_disk(
        self,
        hdf5_filename: str | Path,
        waveforms_dataset_name: str | None = "waveforms",
        other_dset_names: Sequence[str] | None = None,
        start_index: int | None = None,
        up_to_index: int | None = None,
    ):
        """Save my features to new h5 datasets by running in batches through waveforms saved in h5."""
        from h5py import File

        from ..util.data_util import DARTsortSorting

        if up_to_index == 0:
            return

        hdf5_filename = ensure_path(hdf5_filename, strict=True)
        dev = self.device

        # use sorting as a way to load all 1d features, which transformers
        # may want to depend on. but this won't load waveforms (which is good).
        sorting = DARTsortSorting.from_peeling_hdf5(hdf5_filename)
        n = len(sorting)
        fixed_properties = {
            k: torch.asarray(v) for k, v in sorting.spike_feature_dict.items()
        }
        other_dset_names = other_dset_names or []

        # what will we save?
        datasets = self.spike_datasets(
            start_index=start_index, up_to_index=up_to_index, force_save=True
        )
        logger.dartsortdebug(
            f"transform_to_disk will create {[d.name for d in datasets]}"
        )

        # open h5, create new datasets, loop over batches and save feats
        with File(hdf5_filename, mode="r+", libver="latest", locking=False) as h5:
            if all(ds.name in h5 for ds in datasets):
                return
            wfs = h5[waveforms_dataset_name] if waveforms_dataset_name else None
            other_dsets = {od: h5[od] for od in other_dset_names}
            outs = {
                ds.name: h5.create_dataset(
                    ds.name, dtype=ds.dtype, shape=(n, *ds.shape_per_spike)
                )
                for ds in datasets
            }
            for sli, chk in yield_chunks(
                wfs, desc_prefix="Transform to disk", show_progress=False
            ):
                chk_fp = {k: v[sli].to(device=dev) for k, v in fixed_properties.items()}
                other_fp = {
                    k: torch.asarray(ds[sli], device=dev)
                    for k, ds in other_dsets.items()
                }
                _, chk_feat = self(
                    waveforms=torch.asarray(chk, device=dev),
                    **chk_fp,
                    **other_fp,
                    start_index=start_index,
                    up_to_index=up_to_index,
                )
                for ds in datasets:
                    outs[ds.name][sli] = chk_feat[ds.name].numpy(force=True)

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
    featurization_cfg: FeaturizationConfig,
    waveform_cfg: WaveformConfig,
    sampling_frequency: float,
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

    if do_feats and fc.save_input_voltages:
        class_names_and_kwargs.append(
            ("Voltage", {"name_prefix": fc.input_waveforms_name})
        )
    if do_feats and fc.save_collidedness:
        class_names_and_kwargs.append(("FixedProperty", {"name": "collidedness"}))
    if fc.save_input_waveforms:
        class_names_and_kwargs.append(
            ("Waveform", {"name_prefix": fc.input_waveforms_name})
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
    class_names_and_kwargs.extend(
        _add_tpca_and_nn(featurization_cfg, waveform_cfg, sampling_frequency)
    )

    if fc.do_enforce_decrease is True:
        class_names_and_kwargs.append(("EnforceDecrease", {}))
    if fc.save_output_waveforms:
        class_names_and_kwargs.append(
            ("Waveform", {"name_prefix": fc.output_waveforms_name})
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

    if fc.use_gmm_classifier:
        class_names_and_kwargs.append(
            (
                "TruncatedMixtureModelTransformer",
                {
                    "clustering_cfg": fc.pre_gmm_clustering_cfg,
                    "clustering_features_cfg": fc.gmm_clustering_features_cfg,
                    "pre_gmm_refinement_cfgs": fc.pre_gmm_refinement_cfgs,
                    "gmm_refinement_cfg": fc.gmm_refinement_cfg,
                },
            )
        )

    return class_names_and_kwargs


def _add_tpca_and_nn(fc, wc, fs):
    do_feats = not fc.denoise_only
    more = []

    # now, if there's no downstream waveform feature or output tpca feature,
    # we can combine the tpca featurization and denoising into one step
    # it's not exactly equivalent, since the denoiser would have run on the full
    # time length, but it's close enough
    need_input_tpca_projs = (
        fc.compute_input_tpca_projs_regardless or fc.save_input_tpca_projs
    )
    combine = (
        do_feats
        and need_input_tpca_projs
        and not (
            fc.do_nn_denoise or fc.save_output_waveforms or fc.save_output_tpca_projs
        )
    )
    if combine or (do_feats and need_input_tpca_projs):
        tslice = fc.input_tpca_waveform_cfg.relative_slice(wc, fs)
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
                    "save_feature": fc.save_input_tpca_projs,
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

    if fc.do_enforce_decrease == "loc_only" and fc.do_localization:
        more.append(("EnforceDecrease", {}))

    if do_feats and fc.do_localization and fc.nn_localization:
        more.append(
            (
                "AmortizedLocalization",
                {
                    "amplitude_kind": fc.localization_amplitude_type,
                    "localization_model": fc.localization_model,
                    "radius": fc.localization_radius,
                    "decay_power": fc.localization_decay_power,
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

    return more
