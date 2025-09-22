from typing import Any

import torch

from ..util.data_util import SpikeDataset


class BaseWaveformModule(torch.nn.Module):
    is_denoiser = False
    is_featurizer = False
    default_name = ""

    def __init__(self, channel_index=None, geom=None, name=None, name_prefix=None):
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

        self.spike_length_samples = None
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
        state = self.__dict__.copy()
        del state["_hook"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            self._hook = self.register_load_state_dict_pre_hook(
                self.__class__._pre_load_state
            )
        except AttributeError:
            self._hook = self._register_load_state_dict_pre_hook(
                self.__class__._pre_load_state
            )

    def fit(self, recording, waveforms, **fixed_properties) -> Any:
        del recording
        self.spike_length_samples = waveforms.shape[1]
        self.initialize_spike_length_dependent_params()

    def get_extra_state(self):
        return dict(
            spike_length_samples=self.spike_length_samples, needs_fit=self.needs_fit()
        )

    def set_extra_state(self, extra_state):
        self.spike_length_samples = extra_state["spike_length_samples"]
        if hasattr(self, "_needs_fit"):
            self._needs_fit = extra_state["needs_fit"]

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
        assert len(extra_state_keys) == 1
        extra_state = state_dict[extra_state_keys[0]]

        # some modules want to know the spike length before loading the state dict
        # and unfortunately set_extra_state usually runs after. doesn't hurt to run now.
        self.spike_length_samples = extra_state["spike_length_samples"]

        # and this is how subclasses use that info
        # if state dict was dumped before fit, then sls was never known and we
        # don't want to call that initializer, so don't if sls is None.
        if self.spike_length_samples is not None:
            self.initialize_spike_length_dependent_params()

    def initialize_spike_length_dependent_params(self):
        pass

    def needs_fit(self) -> bool:
        return False

    def needs_precompute(self) -> bool:
        return False

    def precompute(self):
        pass

    def extra_repr(self):
        return f"name={self.name},needs_fit={self.needs_fit()}"


class BaseWaveformDenoiser(BaseWaveformModule):
    is_denoiser = True

    def forward(self, waveforms, **unused):
        raise NotImplementedError


class BaseWaveformFeaturizer(BaseWaveformModule):
    is_featurizer = True
    is_multi = False
    # output shape per waveform
    shape = ()
    # output dtye
    dtype = torch.float

    def transform(self, waveforms, **unused):
        # returns dict {key=feat name, value=feature}
        raise NotImplementedError

    @property
    def spike_datasets(self):
        if self.is_multi:
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
        self, pipeline=None, geom=None, channel_index=None, name=None, name_prefix=None
    ):
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

    def fit(self, recording, waveforms, **fixed_properties):
        if self.pipeline is None:
            return
        self.pipeline.fit(recording, waveforms, **fixed_properties)

    def forward(self, waveforms, **fixed_properties):
        if self.pipeline is None:
            return waveforms, {}
        pipeline_waveforms, pipeline_features = self.pipeline(waveforms, **fixed_properties)
        return waveforms, pipeline_features

    @property
    def spike_datasets(self):
        datasets = []
        if self.pipeline is not None:
            for t in self.pipeline.transformers:
                if t.is_featurizer:
                    datasets.extend(t.spike_datasets)
        return datasets

    def transform(self, waveforms, **fixed_properties):
        if self.pipeline is None:
            return {}
        pipeline_waveforms, pipeline_features = self.pipeline(waveforms, **fixed_properties)
        return pipeline_features


class IdentityWaveformDenoiser(BaseWaveformDenoiser):
    def forward(self, waveforms, **unused):
        return waveforms


class Waveform(BaseWaveformFeaturizer):
    default_name = "waveforms"

    def __init__(
        self,
        channel_index,
        geom=None,
        spike_length_samples=121,
        dtype=torch.float,
        name=None,
        name_prefix=None,
    ):
        super().__init__(
            geom=geom,
            channel_index=channel_index,
            name=name,
            name_prefix=name_prefix,
        )
        self.shape = (spike_length_samples, channel_index.shape[1])
        self.dtype = dtype

    def transform(self, waveforms, **unused):
        return {self.name: waveforms.clone()}
