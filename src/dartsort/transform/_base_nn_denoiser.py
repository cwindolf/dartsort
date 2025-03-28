import torch
import numpy as np

from .transform_base import BaseWaveformDenoiser
from ..util.waveform_util import regularize_channel_index
from ..util.spiketorch import get_relative_index, reindex, spawn_torch_rg
from ..util import nn_util


class BaseMultichannelDenoiser(BaseWaveformDenoiser):

    def __init__(
        self,
        channel_index,
        geom,
        hidden_dims=(256, 256, 256),
        norm_kind="layernorm",
        name=None,
        name_prefix="",
        batch_size=32,
        learning_rate=1e-3,
        n_epochs=50,
        channelwise_dropout_p=0.0,
        with_conv_fullheight=False,
        pretrained_path=None,
        val_split_p=0.2,
        min_epochs=10,
        earlystop_eps=None,
        random_seed=0,
        res_type="none",
        lr_schedule=None,
        lr_schedule_kwargs=None,
    ):
        assert pretrained_path is None, "Need to implement loading."
        super().__init__(
            geom=geom, channel_index=channel_index, name=name, name_prefix=name_prefix
        )

        self.norm_kind = norm_kind
        self.hidden_dims = hidden_dims
        self.n_channels = len(geom)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.channelwise_dropout_p = channelwise_dropout_p
        self.with_conv_fullheight = with_conv_fullheight
        self.lr_schedule = lr_schedule
        self.lr_schedule_kwargs = lr_schedule_kwargs

        self.model_channel_index_np = regularize_channel_index(
            geom=self.geom, channel_index=channel_index
        )
        self.register_buffer(
            "model_channel_index", torch.from_numpy(self.model_channel_index_np)
        )
        self.register_buffer(
            "relative_index",
            get_relative_index(self.channel_index, self.model_channel_index),
        )
        # suburban lawns -- janitor
        self.register_buffer(
            "irrelative_index",
            get_relative_index(self.model_channel_index, self.channel_index),
        )
        self._needs_fit = True
        self.val_split_p = val_split_p
        self.min_epochs = min_epochs
        self.earlystop_eps = earlystop_eps
        self.rg = np.random.default_rng(random_seed)
        self.generator = spawn_torch_rg(self.rg)
        self.res_type = res_type

    def initialize_shapes(self, spike_length_samples):
        # we don't know these dimensions til we see a spike
        self.spike_length_samples = spike_length_samples
        self.wf_dim = spike_length_samples * self.model_channel_index.shape[1]
        self.output_dim = self.wf_dim

    @property
    def device(self):
        return self.channel_index.device

    def needs_fit(self):
        return self._needs_fit

    def get_mlp(self, res_type="none"):
        return nn_util.get_waveform_mlp(
            self.spike_length_samples,
            self.model_channel_index.shape[1],
            self.hidden_dims,
            self.output_dim,
            norm_kind=self.norm_kind,
            channelwise_dropout_p=self.channelwise_dropout_p,
            separated_mask_input=True,
            return_initial_shape=True,
            initial_conv_fullheight=self.with_conv_fullheight,
            final_conv_fullheight=self.with_conv_fullheight,
            res_type=res_type,
        )

    def get_scheduler(self, optimizer):
        if self.lr_schedule in (None, "none"):
            return None

        sched_kw = self.lr_schedule_kwargs or {}
        if issubclass(self.lr_schedule, torch.optim.lr_scheduler.LRScheduler):
            return self.lr_schedule(optimizer, **sched_kw)

        assert False

    def to_nn_channels(self, waveforms, max_channels):
        waveforms = reindex(max_channels, waveforms, self.relative_index, pad_value=0.0)
        masks = self.get_masks(max_channels).to(waveforms)
        return waveforms, masks

    def to_orig_channels(self, waveforms, max_channels):
        return reindex(max_channels, waveforms, self.irrelative_index)

    def get_masks(self, max_channels):
        return self.model_channel_index[max_channels] < self.n_channels
