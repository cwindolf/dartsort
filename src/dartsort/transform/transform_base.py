import torch
from dartsort.util.data_util import SpikeDataset


class BaseWaveformModule(torch.nn.Module):
    is_denoiser = False
    is_featurizer = False
    default_name = ""

    def __init__(
        self, channel_index=None, geom=None, name=None, name_prefix=""
    ):
        super().__init__()
        if name is None:
            name = self.default_name
            if name_prefix:
                name = f"{name_prefix}_{name}"
        self.name = name
        if channel_index is not None:
            self.register_buffer(
                "channel_index", torch.asarray(channel_index, copy=True)
            )
        if geom is not None:
            self.register_buffer("geom", torch.asarray(geom, copy=True))

    def fit(self, waveforms, max_channels=None, recording=None):
        pass

    def needs_fit(self):
        return False

    def needs_precompute(self):
        return False

    def precompute(self):
        pass

    def extra_repr(self):
        return f"name={self.name},needs_fit={self.needs_fit()}"


class BaseWaveformDenoiser(BaseWaveformModule):
    is_denoiser = True

    def forward(self, waveforms, max_channels=None):
        raise NotImplementedError


class BaseWaveformFeaturizer(BaseWaveformModule):
    is_featurizer = True
    is_multi = False
    # output shape per waveform
    shape = ()
    # output dtye
    dtype = torch.float

    def transform(self, waveforms, max_channels=None):
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

    def __init__(self, pipeline, geom=None, channel_index=None):
        t = [t for t in pipeline if t.is_featurizer]
        if not len(t):
            t = pipeline.transformers
        name = f"passthrough_{t[0].name}"
        super().__init__(name=name)
        self.pipeline = pipeline

    def needs_precompute(self):
        return self.pipeline.needs_precompute()

    def precompute(self):
        return self.pipeline.precompute()

    def needs_fit(self):
        return self.pipeline.needs_fit()

    def fit(self, waveforms, max_channels, recording=None):
        self.pipeline.fit(waveforms, max_channels)

    def forward(self, waveforms, max_channels=None):
        pipeline_waveforms, pipeline_features = self.pipeline(waveforms, max_channels)
        return waveforms, pipeline_features

    @property
    def spike_datasets(self):
        datasets = []
        for t in self.pipeline.transformers:
            if t.is_featurizer:
                datasets.extend(t.spike_datasets)
        return datasets

    def transform(self, waveforms, max_channels=None):
        pipeline_waveforms, pipeline_features = self.pipeline(waveforms, max_channels)
        return pipeline_features


class IdentityWaveformDenoiser(BaseWaveformDenoiser):
    def forward(self, waveforms, max_channels=None):
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

    def transform(self, waveforms, max_channels=None):
        return {self.name: waveforms.clone()}
