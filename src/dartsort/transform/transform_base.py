from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Self

import torch
from spikeinterface.core import BaseRecording

from ..util.data_util import SpikeDataset
from ..util.internal_config import (
    ComputationConfig,
    WaveformConfig,
    default_waveform_cfg,
)
from ..util.torch_util import BModule

if TYPE_CHECKING:
    from .pipeline import WaveformPipeline


class BaseWaveformModule(BModule):
    is_denoiser = False
    is_featurizer = False
    default_name = ""
    needs_residual = False
    fits_from_disk = False
    needs_more_features = False

    def __init__(
        self,
        channel_index=None,
        geom=None,
        name=None,
        name_prefix=None,
        waveform_cfg: WaveformConfig | None = default_waveform_cfg,
        sampling_frequency: float = 30_000.0,
    ):
        super().__init__()
        if name is None:
            name = self.default_name
            if name_prefix:
                name = f"{name_prefix}_{name}"
        self.name = name
        # these buffers below need to be copied, else they share references
        # across all the transformers which seems to cause problems!
        if channel_index is not None:
            channel_index = torch.asarray(channel_index, copy=True)
            self.register_buffer("channel_index", channel_index)
        if geom is not None:
            geom = torch.asarray(geom, dtype=torch.float, copy=True)
            self.register_buffer("geom", geom)

        self.waveform_cfg = waveform_cfg
        if waveform_cfg is None:
            self.spike_length_samples = None
            self.trough_offset_samples = None
        else:
            self.trough_offset_samples = waveform_cfg.trough_offset_samples(
                sampling_frequency
            )
            self.spike_length_samples = waveform_cfg.spike_length_samples(
                sampling_frequency
            )
        try:
            self._hook = self.register_load_state_dict_pre_hook(
                self.__class__._pre_load_state
            )
        except AttributeError:
            # ...? seems to happen in 2.4.1...
            self._hook = self._register_load_state_dict_pre_hook(
                self.__class__._pre_load_state
            )

    def __getstate__(self):
        self._hook.remove()
        state = super().__getstate__()
        del state["_hook"]
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        try:
            self._hook = self.register_load_state_dict_pre_hook(
                self.__class__._pre_load_state
            )
        except AttributeError:
            self._hook = self._register_load_state_dict_pre_hook(
                self.__class__._pre_load_state
            )

    def fit(
        self,
        recording: BaseRecording,
        waveforms: torch.Tensor,
        *,
        hdf5_filename: Path | None = None,
        computation_cfg: ComputationConfig,
        channels: torch.Tensor,
        pipeline: "WaveformPipeline | None" = None,
        **spike_data: torch.Tensor,
    ) -> Any:
        del recording, spike_data
        self.spike_length_samples = waveforms.shape[1]
        self.initialize_spike_length_dependent_params()

    def get_extra_state(self):
        return dict(
            spike_length_samples=self.spike_length_samples, needs_fit=self.needs_fit()
        )

    @classmethod
    def load_from_pt(
        cls,
        *,
        pretrained_path: str | Path,
        channel_index: torch.Tensor,
        geom: torch.Tensor,
        **kwargs: dict[str, Any],
    ) -> Self:
        raise NotImplementedError

    def set_extra_state(self, state):
        self.spike_length_samples = state["spike_length_samples"]
        if hasattr(self, "_needs_fit"):
            self._needs_fit = state["needs_fit"]
        self._init_bgetter()

    def _other_pre_load_state(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        pass

    def _pre_load_state(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # wish torch would strip the prefix for us?
        extra_state_keys = [k for k in state_dict.keys() if k.endswith("_extra_state")]
        assert len(extra_state_keys) <= 1
        if extra_state_keys:
            extra_state = state_dict[extra_state_keys[0]]

            # some modules want to know the spike length before loading the state dict
            # and unfortunately set_extra_state usually runs after. doesn't hurt to run now.
            self.spike_length_samples = extra_state["spike_length_samples"]

        # and this is how subclasses use that info
        # if state dict was dumped before fit, then sls was never known and we
        # don't want to call that initializer, so don't if sls is None.
        if self.spike_length_samples is not None:
            self.initialize_spike_length_dependent_params()

        self._other_pre_load_state(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def initialize_spike_length_dependent_params(self):
        pass

    def needs_fit(self) -> bool:
        return False

    def needs_precompute(self) -> bool:
        return False

    def attach_motion(self, motion):
        pass

    def register_cpu_workers(self, workers: int):
        pass

    def precompute(self):
        pass

    def extra_repr(self):
        return f"name={self.name},needs_fit={self.needs_fit()}"


class BaseWaveformDenoiser(BaseWaveformModule):
    is_denoiser = True

    def forward(self, waveforms, *, channels: torch.Tensor, **spike_data):
        del waveforms, spike_data
        raise NotImplementedError


class BaseWaveformFeaturizer(BaseWaveformModule):
    is_featurizer = True
    is_multi = False
    saving = True
    # output shape per waveform
    shape: tuple | list[tuple] = ()
    # output dtye
    dtype: torch.dtype | list[torch.dtype] = torch.float

    def transform(
        self,
        waveforms: torch.Tensor,
        *,
        channels: torch.Tensor,
        **spike_data: torch.Tensor,
    ):
        del waveforms, spike_data, channels
        # returns dict {feat name: feature, ...}
        raise NotImplementedError

    def spike_datasets(self, force_save: bool = False) -> Iterable[SpikeDataset]:
        if not self.saving and not force_save:
            return ()
        if self.is_multi:
            assert isinstance(self.dtype, (list, tuple))
            datasets = [
                SpikeDataset(
                    name=n,
                    shape_per_spike=s,
                    dtype=str(d).split(".")[1],
                )
                for n, s, d in zip(self.name, self.shape, self.dtype)
            ]
            return datasets
        else:
            torch_dtype_as_str = str(self.dtype).split(".")[1]
            dataset = SpikeDataset(
                name=self.name,
                shape_per_spike=self.shape,
                dtype=torch_dtype_as_str,
            )
            return (dataset,)


class BaseWaveformAutoencoder(BaseWaveformDenoiser, BaseWaveformFeaturizer):
    pass


class Passthrough(BaseWaveformDenoiser, BaseWaveformFeaturizer):
    def __init__(
        self,
        pipeline=None,
        geom=None,
        channel_index=None,
        name=None,
        name_prefix=None,
        waveform_cfg=None,
        sampling_frequency=30_000.0,
    ):
        del geom, channel_index
        t = []
        if pipeline is not None:
            t = [t for t in pipeline if t.is_featurizer]
            if not len(t):
                t = pipeline.transformers
            if name is None:
                name = f"passthrough_{t[0].name}"
        super().__init__(name=name, name_prefix=name_prefix)
        self.pipeline = pipeline

    def needs_precompute(self):
        if self.pipeline is None:
            return False
        return self.pipeline.needs_precompute()

    def precompute(self):
        if self.pipeline is None:
            return
        return self.pipeline.precompute()

    def needs_fit(self):
        if self.pipeline is None:
            return False
        return self.pipeline.needs_fit()

    def fit(self, recording, waveforms, **spike_data):
        if self.pipeline is None:
            return
        self.pipeline.fit(recording, waveforms, **spike_data)

    def forward(self, waveforms, **spike_data):
        if self.pipeline is None:
            return waveforms, {}
        pipeline_waveforms, pipeline_features = self.pipeline(waveforms, **spike_data)
        del pipeline_waveforms  # passthrough!
        return waveforms, pipeline_features

    def spike_datasets(self, force_save=False):
        datasets = []
        if self.pipeline is not None:
            for t in self.pipeline.transformers:
                if t.is_featurizer:
                    datasets.extend(t.spike_datasets())
        return datasets

    def transform(self, waveforms, **spike_data):
        if self.pipeline is None:
            return {}
        pipeline_waveforms, pipeline_features = self.pipeline(waveforms, **spike_data)
        del pipeline_waveforms
        return pipeline_features


class IdentityWaveformDenoiser(BaseWaveformDenoiser):
    def forward(self, waveforms, **spike_data):
        del spike_data
        return waveforms


class Waveform(BaseWaveformFeaturizer):
    default_name = "waveforms"

    def __init__(
        self,
        channel_index,
        geom=None,
        dtype=torch.float,
        name=None,
        name_prefix=None,
        waveform_cfg: WaveformConfig = default_waveform_cfg,
        sampling_frequency=30_000.0,
    ):
        super().__init__(
            geom=geom,
            channel_index=channel_index,
            name=name,
            name_prefix=name_prefix,
            waveform_cfg=waveform_cfg,
            sampling_frequency=sampling_frequency,
        )
        assert self.spike_length_samples is not None
        self.shape = (self.spike_length_samples, channel_index.shape[1])
        self.dtype = dtype

    def transform(self, waveforms, **spike_data):
        del spike_data
        return {self.name: waveforms.clone()}
