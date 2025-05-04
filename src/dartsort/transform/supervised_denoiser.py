import math

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import trange


from ._multichan_denoiser_kit import (
    BaseMultichannelDenoiser,
    RefreshableDataset,
    RefreshableDataLoader,
    AsyncBatchDataset,
)
from ..util import waveform_util


class SupervisedDenoiser(BaseMultichannelDenoiser):
    default_name = "superviseddenoiser"

    def initialize_spike_length_dependent_params(self):
        self.initialize_shapes()
        self.exy = self.get_mlp(res_type=self.res_type)
        self.to(self.device)

    def forward_unbatched(self, waveforms, max_channels, to_orig_channels=True):
        """Called only at inference time."""
        waveforms, masks = self.to_nn_channels(waveforms, max_channels)
        net_input = waveforms, masks.unsqueeze(1)
        pred = self.exy(net_input)
        if to_orig_channels:
            pred = self.to_orig_channels(pred, max_channels)
        return pred

    def fit(self, waveforms, gt_waveforms, max_channels):
        super().fit(waveforms, max_channels, None, None)
        train_loader, val_loader = self._waveforms_to_loaders(
            waveforms, gt_waveforms, max_channels
        )
        with torch.enable_grad():
            res = self._fit(train_loader, val_loader)
        self._needs_fit = False
        return res

    def loss(self, mask, gt_waveforms, pred):
        mask = mask.unsqueeze(1)
        loss_dict = dict(mse=F.mse_loss(mask * gt_waveforms, mask * pred))
        return loss_dict

    def fit_with_loaders(
        self,
        train_loader: RefreshableDataLoader,
        val_loader: DataLoader | None = None,
    ):
        _wf, *_ = next(iter(train_loader))

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        train_losses_per_epoch = []
        val_losses_per_epoch = []
        last_val_loss = None

        with trange(self.n_epochs, desc="Epochs", unit="epoch") as pbar:
            for epoch in pbar:
                self.train()
                train_loss_sum = 0.0
                train_loader.refresh()
                for waveform_batch, gt_waveform_batch, channels_batch in train_loader:
                    optimizer.zero_grad()

                    waveform_batch = waveform_batch.to(self.device)
                    gt_waveform_batch = gt_waveform_batch.to(self.device)
                    channels_batch = channels_batch.to(self.device)

                    gt_waveform_batch, mask = self.to_nn_channels(
                        gt_waveform_batch, channels_batch
                    )
                    pred = self.forward_unbatched(
                        waveform_batch, channels_batch, to_orig_channels=False
                    )

                    loss_dict = self.loss(mask, gt_waveform_batch, pred)
                    loss = sum(loss_dict.values())
                    loss.backward()
                    optimizer.step()

                    train_loss_sum += loss.item()

                avg_train_loss = train_loss_sum / len(train_loader)
                train_losses_per_epoch.append(avg_train_loss)
                train_loader.cleanup()

                avg_val_loss = None
                if val_loader is not None:
                    self.eval()
                    val_loss_sum = 0.0
                    with torch.no_grad():
                        for (
                            waveform_batch,
                            gt_waveform_batch,
                            channels_batch,
                        ) in val_loader:
                            waveform_batch = waveform_batch.to(self.device)
                            gt_waveform_batch = gt_waveform_batch.to(self.device)
                            channels_batch = channels_batch.to(self.device)

                            gt_waveform_batch, mask = self.to_nn_channels(
                                gt_waveform_batch, channels_batch
                            )
                            pred = self.forward_unbatched(
                                waveform_batch, channels_batch, to_orig_channels=False
                            )

                            loss_dict = self.loss(mask, gt_waveform_batch, pred)
                            loss = sum(loss_dict.values())
                            val_loss_sum += loss.item()

                    avg_val_loss = val_loss_sum / len(val_loader)
                    val_losses_per_epoch.append(avg_val_loss)

                    if (
                        self.earlystop_eps is not None
                        and last_val_loss is not None
                        and avg_val_loss - last_val_loss > self.earlystop_eps
                    ):
                        if epoch >= self.min_epochs:
                            print(f"Early stopping after {epoch} epochs.")
                            break
                    last_val_loss = avg_val_loss

                if self.step_callback is not None:
                    self.step_callback(self, epoch, avg_val_loss)

                loss_str = f"Train loss: {avg_train_loss:3g}"
                if val_loader is not None:
                    loss_str += f" | Val loss: {avg_val_loss:3g}"
                pbar.set_description(f"Epochs [{loss_str}]")

                self.step_scheduler(scheduler, avg_train_loss, avg_val_loss)

        return dict(
            train_losses=train_losses_per_epoch,
            val_losses=val_losses_per_epoch,
        )

    def _waveforms_to_loaders(self, waveforms, gt_waveforms, channels):
        num_samples = len(waveforms)
        train_size = math.ceil(num_samples * (1 - self.val_split_p))
        val_size = num_samples - train_size
        full_dataset = TensorDataset(waveforms, gt_waveforms, channels)
        if val_size:
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            train_dataset = full_dataset
            val_loader = None

        train_loader = RefreshableDataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        return train_loader, val_loader


class PairedHDF5WaveformsDataset(AsyncBatchDataset):
    def __init__(
        self,
        hdf5_path,
        channel_index,
        waveforms_dataset_name="waveforms",
        gt_waveforms_dataset_name="gt_waveforms",
        channels_dataset_name="channels",
        gt_waveform_index_dataset_name=None,
        geom=None,
        trough_offset_samples=42,
        spike_length_samples=121,
        time_jitter_samples=3,
        spatial_jitter_um=35.0,
        generator=None,
        chunk_size=2048,
        queue_chunks=8,
    ):
        """This is meant to be used with the AOTIndicesRefreshableDataLoader

        Class balancing etc can be achieved with its weights argument. They would
        'hit' the waveforms here (not gt_waveforms, if those are indexed via
        labels in gt_waveform_index_dataset_name).
        """
        self.hdf5_path = hdf5_path
        self._h5 = h5py.File(hdf5_path, "r", locking=False)
        self._waveforms_dataset: h5py.Dataset = self._h5[waveforms_dataset_name]
        self._gt_waveforms_dataset: h5py.Dataset = self._h5[gt_waveforms_dataset_name]
        self._channels = torch.from_numpy(self._h5[channels_dataset_name][:])
        if gt_waveform_index_dataset_name:
            self._gt_indices = self._h5[gt_waveform_index_dataset_name][:]
        else:
            self._gt_indices = None

        super().__init__(
            n_examples=len(self._channels),
            spike_length_samples=spike_length_samples,
            generator=generator,
            chunk_size=chunk_size,
            queue_chunks=queue_chunks,
        )
        self.trough_offset_samples = trough_offset_samples
        self.time_jitter_samples = time_jitter_samples
        self.spatial_jitter_um = spatial_jitter_um

        # spatial information
        self.channel_index = channel_index
        self.n_channels = len(channel_index)
        if spatial_jitter_um:
            if geom is None:
                geom = self._h5["geom"][:]
            self.spatial_jitter_index = waveform_util.make_channel_index(
                geom, spatial_jitter_um
            )
        else:
            self.spatial_jitter_index = None

        # temporal info
        min_length = spike_length_samples + 2 * time_jitter_samples
        self._wf_length = self._waveforms_dataset.shape[1]
        assert self._wf_length >= min_length
        offset = (self._wf_length - spike_length_samples) // 2
        assert self._wf_length == 2 * offset + spike_length_samples
        self._wf_offset = offset
        self._tix_rel = torch.arange(spike_length_samples)

    def __del__(self):
        del self._gt_waveforms_dataset
        del self._waveforms_dataset
        self._h5.close()

    def load_batch(self, index):
        n = index.numel()
        waveforms = self._waveforms_dataset[index.numpy()]
        gt_index = index.numpy()
        if self._gt_indices is not None:
            gt_index = self._gt_indices[gt_index]
        gt_waveforms = self._gt_waveforms_dataset[gt_index]
        channels = self._channels[index.numpy()]

        # jitter the channels
        if self.spatial_jitter_index:
            choices = self.spatial_jitter_index[channels]
            n_valid = (choices < self.n_channels).sum(1)
            # wish they had tensor bounds.
            which = torch.randint(2**63 - 1, size=n, generator=self.generator) % n_valid
            channels = choices[torch.arange(n), which]

        # extract the resulting channel subsets
        chan_inds = self.channel_index[channels]
        waveforms = F.pad(waveforms, (0, 1), value=torch.nan)
        waveforms = waveforms.take_along_dim(chan_inds[:, None, :], dim=2)
        gt_waveforms = F.pad(gt_waveforms, (0, 1), value=torch.nan)
        gt_waveforms = gt_waveforms.take_along_dim(chan_inds[:, None, :], dim=2)

        # jitter time
        time_ixs = torch.full((n,), self._wf_offset)
        if self.time_jitter_samples:
            time_ixs += torch.randint(
                -self.time_jitter_samples,
                self.time_jitter_samples + 1,
                generator=self.generator,
                size=n,
            )
        time_ixs = time_ixs[:, None] + self._tix_rel
        waveforms = waveforms.take_along_dim(time_ixs[:, :, None], dim=1)
        gt_waveforms = gt_waveforms.take_along_dim(time_ixs[:, :, None], dim=1)

        return waveforms, gt_waveforms, channels


class NoisedTemplatesDataset(AsyncBatchDataset):
    def __init__(
        self,
        channel_index,
        templates,
        noise,
        geom=None,
        collision_prob=0.5,
        max_collisions=18,
        collision_sample_weights=None,
        trough_offset_samples=42,
        spike_length_samples=121,
        time_jitter_samples=3,
        spatial_jitter_um=35.0,
        amplitude_jitter_std=0.1,
        amplitude_jitter_max=1.5,
        temporal_upsampling_factor=8,
        generator=None,
        chunk_size=2048,
        queue_chunks=8,
        dtype=torch.float,
    ):
        """This is meant to be used with the AOTIndicesRefreshableDataLoader

        Class balancing etc can be achieved with its weights argument. This
        would assign a weight to each template.

        NOTE: collisions are on the full channel set. So, many collisions will
        do nothing! Figure that the probe is 4000um and that a "significant"
        collision occurs when units are spaced by <100um. If the unit positions
        are random, a collision has ~200/4000=0.05 chance of overlapping significantly.
        Twice that for a minor overlap?

        Collisions are added binomially (# ~ Bin(max,p)), up to max_collisions,
        so the mean # is like max_collisions*collision_prob.
        So, the real mean is something like 0.05*max_collisions*p but censored.
        For p=0.5 max=4, that would be like .5*4/20=0.1ish? But then the completely
        random shifting reduces the probability by another factor of 3!

        So the defaults are set to 18 max collisions for this reason:
        18 max * 0.5 p * 1/3 time * 1/20 space = 0.15.


        `noise` should probably be a noise_util.FactorizedNoise object, or
        one of the other obects there which have simulate() methods.
        """
        super().__init__(
            n_examples=len(templates),
            spike_length_samples=spike_length_samples,
            generator=generator,
            chunk_size=chunk_size,
            queue_chunks=queue_chunks,
        )
        self.trough_offset_samples = trough_offset_samples
        self.time_jitter_samples = time_jitter_samples
        self.spatial_jitter_um = spatial_jitter_um
        self.amplitude_jitter_std = amplitude_jitter_std
        self.amplitude_jitter_max = amplitude_jitter_max
        self.amplitude_jitter_min = 1.0 / amplitude_jitter_max
        self.temporal_upsampling_factor = temporal_upsampling_factor

        self.noise = noise

        # collision control
        self.collision_prob = collision_prob
        self.max_collisions = max_collisions
        self.collision_sample_weights = collision_sample_weights

        # upsample and store templates as n_examples, time_up, time, chans
        if torch.is_tensor(templates):
            templates = templates.numpy(force=True)
        n, t, c = templates.shape
        templates = templates.transpose(0, 2, 1).reshape(n * c, t)
        templates_up = waveform_util.upsample_singlechan(
            templates, temporal_jitter=temporal_upsampling_factor
        )
        templates_up = templates_up.reshape(n, c, temporal_upsampling_factor, t)
        templates_up = templates_up.transpose(0, 2, 3, 1)
        self.templates_up = torch.asarray(templates_up, dtype=dtype).contiguous()
        self.channels_up = self.templates_up.abs().amax(dim=2).argmax(dim=2)

        # spatial information
        self.channel_index = channel_index
        self.n_channels = len(channel_index)
        if spatial_jitter_um:
            assert geom is not None
            self.spatial_jitter_index = waveform_util.make_channel_index(
                geom, spatial_jitter_um
            )
        else:
            self.spatial_jitter_index = None

        # temporal info
        min_length = spike_length_samples + 2 * time_jitter_samples
        self._wf_length = t
        assert self._wf_length >= min_length, "Waveforms need to be padded."
        offset = (self._wf_length - spike_length_samples) // 2
        assert self._wf_length == 2 * offset + spike_length_samples
        self._wf_offset = offset
        self._tix_rel = torch.arange(spike_length_samples)

    def load_batch(self, index):
        n = index.numel()

        # get gt waveforms
        up_ix = torch.randint(
            0, self.temporal_upsampling_factor, size=n, generator=self.generator
        )
        gt_waveforms = self.templates_up[index, up_ix]
        channels = self.channels_up[index, up_ix]

        # jitter amplitude
        if self.amplitude_jitter_std:
            scaling = torch.normal(
                1.0, self.amplitude_jitter_std, size=n, generator=self.generator
            )
            scaling.clamp_(self.amplitude_jitter_min, self.amplitude_jitter_max)
            gt_waveforms = gt_waveforms * scaling[..., None, None]

        # get noise waveforms
        noise = self.noise.simulate(size=n, generator=self.generator)

        # add in the collisions
        p = torch.full((self.max_collisions, n), self.collision_prob)
        mask = torch.bernoulli(p)
        collisions = self.get_batch_of_collisions(mask.sum())
        s = 0
        for m in mask:
            (m,) = m.nonzero(as_tuple=True)
            noise[m] += collisions[s : s + m.numel()]
            s += m.numel()

        # noisy waveforms...
        waveforms = gt_waveforms + noise

        # jitter the channels
        if self.spatial_jitter_index:
            choices = self.spatial_jitter_index[channels]
            n_valid = (choices < self.n_channels).sum(1)
            # wish they had tensor bounds.
            which = torch.randint(2**63 - 1, size=n, generator=self.generator) % n_valid
            channels = choices[torch.arange(n), which]

        # extract the resulting channel subsets
        chan_inds = self.channel_index[channels]
        waveforms = F.pad(waveforms, (0, 1), value=torch.nan)
        waveforms = waveforms.take_along_dim(chan_inds[:, None, :], dim=2)
        gt_waveforms = F.pad(gt_waveforms, (0, 1), value=torch.nan)
        gt_waveforms = gt_waveforms.take_along_dim(chan_inds[:, None, :], dim=2)

        # jitter time
        time_ixs = torch.full((n,), self._wf_offset)
        if self.time_jitter_samples:
            time_ixs += torch.randint(
                -self.time_jitter_samples,
                self.time_jitter_samples + 1,
                generator=self.generator,
                size=n,
            )
        time_ixs = time_ixs[:, None] + self._tix_rel
        waveforms = waveforms.take_along_dim(time_ixs[:, :, None], dim=1)
        gt_waveforms = gt_waveforms.take_along_dim(time_ixs[:, :, None], dim=1)

        return waveforms, gt_waveforms, channels

    def get_batch_of_collisions(self, n):
        # draw weighted samples
        if self.collision_sample_weights is None:
            index = torch.randint(
                high=self.n_examples,
                size=n,
                dtype=torch.int64,
                generator=self.generator,
            )
        else:
            index = torch.multinomial(
                self.collision_sample_weights,
                n,
                replacement=True,
                generator=self.generator,
            )
        up_ix = torch.randint(
            0, self.temporal_upsampling_factor, size=n, generator=self.generator
        )
        waveforms = self.templates_up[index, up_ix]

        # shift to a completely random offset, in the valid range
        # TODO should this be some kind of taper rather than 0pad?
        full_length = 3 * self.spike_length_samples
        pad = (full_length - waveforms.shape[1]) // 2
        waveforms = F.pad(waveforms, (0, 0, pad, pad))
        time_ixs = torch.randint(
            0, 2 * self.spike_length_samples, generator=self.generator, size=n
        )
        time_ixs = time_ixs[:, None] + self._tix_rel
        waveforms = waveforms.take_along_dim(time_ixs[:, :, None], dim=1)

        # jitter amplitude
        if self.amplitude_jitter_std:
            scaling = torch.normal(
                1.0, self.amplitude_jitter_std, size=n, generator=self.generator
            )
            scaling.clamp_(self.amplitude_jitter_min, self.amplitude_jitter_max)
            waveforms.mul_(scaling[..., None, None])

        return waveforms
