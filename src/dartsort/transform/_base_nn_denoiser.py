from logging import getLogger

import torch
import numpy as np

from .transform_base import BaseWaveformDenoiser
from ..util.waveform_util import regularize_channel_index
from ..util.spiketorch import get_relative_index, reindex, spawn_torch_rg
from ..util import nn_util


logger = getLogger(__name__)


class BaseMultichannelDenoiser(BaseWaveformDenoiser):

    def __init__(
        self,
        channel_index,
        geom,
        hidden_dims=(1024, 1024),
        norm_kind="none",
        name=None,
        name_prefix="",
        batch_size=256,
        learning_rate=2.5e-4,
        weight_decay=1e-5,
        n_epochs=10,
        channelwise_dropout_p=0.0,
        with_conv_fullheight=False,
        pretrained_path=None,
        val_split_p=0.0,
        min_epochs=10,
        earlystop_eps=None,
        random_seed=0,
        res_type="none",
        lr_schedule=None,
        lr_schedule_kwargs=None,
        inference_batch_size=1024,
        optimizer=None,
        optimizer_kwargs=None,
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
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

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
        self.inference_batch_size = inference_batch_size

    def forward(self, waveforms, max_channels):
        out = torch.empty_like(waveforms)
        odev = waveforms.device
        idev = self.device

        for bs in range(0, len(waveforms), self.inference_batch_size):
            be = min(bs + self.inference_batch_size, len(waveforms))
            pred = self.forward_unbatched(
                waveforms[bs:be].to(idev), max_channels[bs:be].to(idev)
            )
            out[bs:be] = pred.to(odev)

        return out

    def initialize_shapes(self, spike_length_samples):
        logger.dartsortdebug(
            f"Initialize {self.__class__.__name__} with {spike_length_samples=}."
        )
        # we don't know these dimensions til we see a spike
        self.spike_length_samples = spike_length_samples
        self.wf_dim = spike_length_samples * self.model_channel_index.shape[1]
        self.output_dim = self.wf_dim

    def get_optimizer(self):
        opt = self.optimizer
        okw = self.optimizer_kwargs
        if isinstance(opt, str):
            opt = getattr(torch.optim, opt)
        if opt is None:
            opt = torch.optim.AdamW
        assert issubclass(opt, torch.optim.Optimizer)
        if okw is None:
            okw = {}
        return opt(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            **okw,
        )

    def get_scheduler(self, optimizer):
        if self.lr_schedule in (None, "none"):
            return None

        lr_schedule = self.lr_schedule
        if isinstance(lr_schedule, str):
            lr_schedule = getattr(torch.optim.lr_scheduler, lr_schedule)

        sched_kw = self.lr_schedule_kwargs or {}
        assert issubclass(lr_schedule, torch.optim.lr_scheduler.LRScheduler)
        return lr_schedule(optimizer, T_max=self.n_epochs, **sched_kw)

    @property
    def device(self):
        return self.channel_index.device

    def needs_fit(self):
        return self._needs_fit

    def get_mlp(self, res_type="none", hidden_dims=None, output_layer="linear"):
        if hidden_dims is None:
            hidden_dims = self.hidden_dims
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
            output_layer=output_layer,
            res_type=res_type,
        )

    def to_nn_channels(self, waveforms, max_channels):
        waveforms = reindex(max_channels, waveforms, self.relative_index, pad_value=0.0)
        masks = self.get_masks(max_channels).to(waveforms)
        return waveforms, masks

    def to_orig_channels(self, waveforms, max_channels):
        return reindex(max_channels, waveforms, self.irrelative_index)

    def get_masks(self, max_channels):
        return self.model_channel_index[max_channels] < self.n_channels
