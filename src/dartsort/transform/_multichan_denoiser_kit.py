from logging import getLogger
from threading import Thread
from queue import Queue

import h5py
import torch
import torch.nn.functional as F
import numpy as np

from dartsort.vis import waveforms

from .transform_base import BaseWaveformDenoiser
from ..util.waveform_util import regularize_channel_index
from ..util.spiketorch import get_relative_index, reindex, spawn_torch_rg
from ..util import nn_util
from ..util import spikeio


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
        nonlinearity="ReLU",
        scaling=None,
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
        self.nonlinearity = nonlinearity
        self.scaling = scaling

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

        sched_kw = self.lr_schedule_kwargs or dict(T_max=self.n_epochs)
        assert issubclass(lr_schedule, torch.optim.lr_scheduler.LRScheduler)
        return lr_schedule(optimizer, **sched_kw)

    @property
    def device(self):
        return self.channel_index.device

    def needs_fit(self):
        return self._needs_fit

    def get_mlp(
        self,
        res_type="none",
        hidden_dims=None,
        output_layer="linear",
        log_transform=False,
    ):
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
            nonlinearity=self.nonlinearity,
            log_transform=log_transform,
            scaling=self.scaling,
        )

    def to_nn_channels(self, waveforms, max_channels):
        waveforms = reindex(max_channels, waveforms, self.relative_index, pad_value=0.0)
        masks = self.get_masks(max_channels).to(waveforms)
        return waveforms, masks

    def to_orig_channels(self, waveforms, max_channels):
        return reindex(max_channels, waveforms, self.irrelative_index)

    def get_masks(self, max_channels):
        return self.model_channel_index[max_channels] < self.n_channels


# -- torch data / noise utilities


def get_noise(
    recording,
    channels,
    channel_index,
    spike_length_samples=121,
    rg: int | None | np.random.Generator = 0,
    generator: torch.Generator | None = None,
):
    if rg is not None:
        rg = np.random.default_rng(rg)
        # pick random times
        times_samples = rg.integers(
            recording.get_num_samples() - spike_length_samples,
            size=len(channels),
        )
    else:
        times_samples = torch.randint(
            low=0,
            high=recording.get_num_samples() - spike_length_samples,
            size=(len(channels),),
            device="cpu",
            generator=generator,
        ).numpy()

    order = np.argsort(times_samples)
    inv_order = np.argsort(order)

    # load waveforms on the same channels and channel neighborhoods
    noise_waveforms = spikeio.read_waveforms_channel_index(
        recording,
        times_samples[order],
        channel_index,
        channels,
        trough_offset_samples=0,
        spike_length_samples=spike_length_samples,
        fill_value=0.0,
    )

    # back to random order
    noise_waveforms = noise_waveforms[inv_order]

    return torch.from_numpy(noise_waveforms)


class NoneDataset(Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return None


class SameChannelNoiseDataset(Dataset):
    def __init__(
        self,
        recording,
        channels,
        channel_index,
        spike_length_samples=121,
        generator=None,
    ):
        super().__init__()
        self.recording = recording
        self.channels = channels
        self.spike_length_samples = spike_length_samples
        self.channel_index = channel_index
        self.generator = generator

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, index):
        noise = get_noise(
            self.recording,
            self.channels[index],
            self.channel_index,
            spike_length_samples=self.spike_length_samples,
            rg=None,
            generator=self.generator,
        )
        return noise


class AOTIndicesWeightedRandomBatchSampler(Sampler):
    def __init__(
        self,
        n_examples=None,
        weights=None,
        replacement=True,
        batch_size=None,
        generator=None,
        epoch_size=None,
    ):
        super().__init__()

        if weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.double)
        if n_examples is None:
            n_examples = len(weights)

        self.n_examples = n_examples
        self.weights = weights
        self.replacement = replacement
        self.generator = generator
        self.epoch_size = epoch_size

        self.indices = None
        self.batch_size = batch_size

    def __len__(self):
        n = self.epoch_size or self.n_examples
        if self.batch_size is None:
            return n
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.batch_size is None:
            yield from self.indices
        else:
            n = self.epoch_size or self.n_examples
            for bs in range(0, n, self.batch_size):
                be = min(n, bs + self.batch_size)
                yield self.indices[bs:be]

    def refresh(self):
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        n_draws = self.epoch_size or self.n_examples

        if self.weights is None:
            if self.replacement:
                self.indices = torch.randint(
                    high=self.n_examples,
                    size=(n_draws,),
                    dtype=torch.int64,
                    generator=generator,
                )
            else:
                self.indices = torch.randperm(self.n_examples, generator=generator)[
                    :n_draws
                ]
        else:
            self.indices = torch.multinomial(
                self.weights, n_draws, self.replacement, generator=generator
            )


class AsyncSameChannelNoiseDataset(Dataset):
    def __init__(
        self,
        channels,
        channel_index,
        spike_length_samples=121,
        generator=None,
        chunk_size=2048,
        queue_chunks=8,
    ):
        super().__init__()
        self.channels = channels
        self.spike_length_samples = spike_length_samples
        self.channel_index = channel_index
        # note: although there's threading here, this is only used in one
        # thread, so that's okay.
        self.generator = generator

        self.chunk_size = chunk_size
        self._queue = Queue(maxsize=queue_chunks)
        self._thread = None
        self._indices = None
        self._cur_data_ix = None
        self._cur_chunk = None
        self._cur_chunk_ix = None

    def __len__(self):
        return len(self.channels)

    def cleanup(self):
        if self._thread is not None:
            self._thread.join()
            assert not self._thread.is_alive()
        self._thread = None

    def refresh(self, indices):
        assert self._thread is None, "Run cleanup()!"
        self._cur_data_ix = 0
        self._indices = indices
        self._run_thread()

    def __getitem__(self, index):
        assert self._indices is not None
        assert self._cur_data_ix is not None
        assert self._cur_chunk_ix is not None

        bs = len(index)
        my_indices = self._indices[self._cur_data_ix : self._cur_data_ix + bs]
        assert torch.equal(torch.asarray(index), my_indices)

        # need to batch up the chunks...
        if self._cur_chunk is None:
            self._cur_chunk = self._queue.get()
            self._cur_chunk_ix = 0

        chunk_end_ix = min(self._cur_chunk_ix + bs, len(self._cur_chunk))
        noise_batch = self._cur_chunk[self._cur_chunk_ix : chunk_end_ix]

        # check if we are done with this chunk
        if chunk_end_ix == len(self._cur_chunk):
            self._cur_chunk = None
            self._cur_chunk_ix = None
        else:
            self._cur_chunk_ix = chunk_end_ix

        self._cur_data_ix += bs
        return noise_batch

    def _run_thread(self):
        self._thread = Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def _thread_main(self):
        assert self._indices is not None
        for chunk_start in range(0, len(self._indices), self.chunk_size):
            chunk_end = min(len(self._indices), chunk_start + self.chunk_size)

            index = self._indices[chunk_start:chunk_end]
            noise = self.load_noise(index)

            self._queue.put(noise)

    def load_noise(self, index):
        # subclasses must implement
        raise NotImplementedError


class AsyncSameChannelRecordingNoiseDataset(AsyncSameChannelNoiseDataset):
    def __init__(
        self,
        recording,
        channels,
        channel_index,
        spike_length_samples=121,
        generator=None,
        chunk_size=2048,
        queue_chunks=8,
    ):
        super().__init__(
            channels=channels,
            channel_index=channel_index,
            spike_length_samples=spike_length_samples,
            generator=generator,
            chunk_size=chunk_size,
            queue_chunks=queue_chunks,
        )
        self.recording = recording

    def load_noise(self, index):
        return get_noise(
            self.recording,
            self.channels[index],
            self.channel_index,
            spike_length_samples=self.spike_length_samples,
            rg=None,
            generator=self.generator,
        )


class AsyncSameChannelHDF5NoiseDataset(AsyncSameChannelNoiseDataset):
    def __init__(
        self,
        hdf5_path,
        channels,
        channel_index,
        noise_dataset_name="noise",
        spike_length_samples=121,
        generator=None,
        chunk_size=2048,
        queue_chunks=8,
    ):
        super().__init__(
            channels=channels,
            channel_index=channel_index,
            spike_length_samples=spike_length_samples,
            generator=generator,
            chunk_size=chunk_size,
            queue_chunks=queue_chunks,
        )

        self.hdf5_path = hdf5_path
        self._h5 = h5py.File(hdf5_path, "r", locking=False)
        self._dataset: h5py.Dataset = self._h5[noise_dataset_name]
        assert self._dataset.ndim == 3
        assert self._dataset.shape[2] == len(channel_index)
        self.noise_snip_len = self._dataset.shape[1]
        self.nsnips = len(self._datasset)
        self._tix_rel = torch.arange(spike_length_samples)
        self._tix_max = self.noise_snip_len - spike_length_samples

    def __del__(self):
        del self._dataset
        self._h5.close()

    def load_noise(self, index):
        nix = index.numel()
        dixs = torch.randint(
            low=0, high=self.nsnips, generator=self.generator, size=nix
        )

        data = self._dataset[dixs.numpy()]
        data = torch.from_numpy(data)

        chans = self.channels[index]
        chan_inds = self.channel_index[chans]
        data = F.pad(data, (0, 1), value=torch.nan)
        data = data.take_along_dim(chan_inds[:, None, :], dim=2)

        tixs = torch.randint(
            low=0, high=self._tix_max, generator=self.generator, size=nix
        )
        time_inds = tixs[:, None] + self._tix_rel
        data = data.take_along_dim(time_inds[, :, None], dim=1)

        return data
