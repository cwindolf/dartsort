from logging import getLogger

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    StackDataset,
    TensorDataset,
    WeightedRandomSampler,
)
from tqdm.auto import trange

from ..util import spikeio
from ..util.spiketorch import reindex, spawn_torch_rg
from ._base_nn_denoiser import BaseMultichannelDenoiser


logger = getLogger(__name__)


class Decollider(BaseMultichannelDenoiser):
    default_name = "decollider"

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
        weight_decay=0.0,
        n_epochs=10,
        channelwise_dropout_p=0.0,
        with_conv_fullheight=False,
        pretrained_path=None,
        val_split_p=0.0,
        min_epochs=10,
        earlystop_eps=None,
        random_seed=0,
        res_type="none",
        lr_schedule='CosineAnnealingLR',
        lr_schedule_kwargs=None,
        optimizer=None,
        optimizer_kwargs=None,
        # my args. todo: port over common ones.
        examples_per_epoch=50_000,
        inference_z_samples=10,
        detach_amortizer=True,
        exz_estimator="n3n",
        inference_kind="amortized",
        eyz_res_type="none",
        e_exz_y_res_type="none",
        emz_res_type="none",
        n_data_workers=4,
        l4_alpha=0,
        output_l1_alpha=5e-4,
        cycle_loss_alpha=1.0,
        separate_cycle_net=False,
        detach_cycle_loss=False,
        val_noise_random_seed=0,
        inf_net_hidden_dims=None,
        eyz_net_hidden_dims=None,
        step_callback=None,
    ):
        assert exz_estimator in ("n2n", "2n2", "n3n", "3n3")
        assert inference_kind in ("raw", "exz", "exz_fromz", "amortized", "exy_fake")

        super().__init__(
            geom=geom,
            channel_index=channel_index,
            name=name,
            name_prefix=name_prefix,
            hidden_dims=hidden_dims,
            norm_kind=norm_kind,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            channelwise_dropout_p=channelwise_dropout_p,
            with_conv_fullheight=with_conv_fullheight,
            pretrained_path=pretrained_path,
            val_split_p=val_split_p,
            min_epochs=min_epochs,
            earlystop_eps=earlystop_eps,
            random_seed=random_seed,
            res_type=res_type,
            lr_schedule=lr_schedule,
            lr_schedule_kwargs=lr_schedule_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.examples_per_epoch = examples_per_epoch
        self.inference_z_samples = inference_z_samples
        self.detach_amortizer = detach_amortizer
        self.exz_estimator = exz_estimator
        self.inference_kind = inference_kind
        self.eyz_res_type = eyz_res_type
        self.e_exz_y_res_type = e_exz_y_res_type
        self.emz_res_type = emz_res_type
        self.n_data_workers = n_data_workers
        self.val_noise_random_seed = val_noise_random_seed
        self.step_callback = step_callback
        self.inf_net_hidden_dims = inf_net_hidden_dims
        self.eyz_net_hidden_dims = eyz_net_hidden_dims
        self.l4_alpha = l4_alpha
        self.output_l1_alpha = output_l1_alpha
        self.cycle_loss_alpha = cycle_loss_alpha
        self.separate_cycle_net = separate_cycle_net
        self.detach_cycle_loss = detach_cycle_loss
        if separate_cycle_net:
            assert cycle_loss_alpha > 0

    def initialize_nets(self, spike_length_samples):
        if hasattr(self, 'inf_net'):
            logger.dartsortdebug('Already initialized.')
            return
        self.initialize_shapes(spike_length_samples)
        if self.exz_estimator in ("n2n", "n3n"):
            self.eyz = self.get_mlp(res_type=self.eyz_res_type, hidden_dims=self.eyz_net_hidden_dims)
        if self.exz_estimator in ("n3n", "2n2", "3n3"):
            self.emz = self.get_mlp(res_type=self.emz_res_type)
        if self.inference_kind == "amortized":
            self.inf_net = self.get_mlp(res_type=self.e_exz_y_res_type, hidden_dims=self.inf_net_hidden_dims)
        if self.separate_cycle_net:
            self.den_net = self.get_mlp(res_type=self.e_exz_y_res_type, hidden_dims=self.inf_net_hidden_dims)
        else:
            self.den_net = self.inf_net
        self.to(self.device)

    def fit(self, waveforms, max_channels, recording, weights=None):
        with torch.enable_grad():
            res = self._fit(waveforms, max_channels, recording, weights=weights)
        self._needs_fit = False
        return res

    def forward_unbatched(self, waveforms, max_channels):
        """Called only at inference time."""
        # TODO: batch all of this.
        waveforms, masks = self.to_nn_channels(waveforms, max_channels)
        net_input = waveforms, masks.unsqueeze(1)

        if self.inference_kind == "amortized":
            pred = self.den_net(net_input)
        elif self.inference_kind == "raw":
            if hasattr(self, "emz"):
                emz = self.emz(net_input)
                pred = waveforms - emz
            elif hasattr(self, "eyz"):
                pred = self.eyz(net_input)
            else:
                assert False
        elif self.inference_kind == "exz_fromz":
            pred = torch.zeros_like(waveforms)
            for j in range(self.inference_z_samples):
                m = get_noise(
                    self.recording,
                    max_channels.numpy(force=True),
                    self.model_channel_index_np,
                    spike_length_samples=self.spike_length_samples,
                    rg=None,
                )
                m = m.to(waveforms)
                z = waveforms + m
                net_input = z, masks.unsqueeze(1)
                if self.exz_estimator == "n2n":
                    eyz = self.eyz(net_input)
                    pred += 2 * eyz - z
                elif self.exz_estimator == "2n2":
                    emz = self.emz(net_input)
                    pred += z - 2 * emz
                elif self.exz_estimator == "n3n":
                    eyz = self.eyz(net_input)
                    emz = self.emz(net_input)
                    pred += eyz - emz
                elif self.exz_estimator == "3n3":
                    emz = self.emz(net_input)
                    pred += z - emz
                else:
                    assert False
            pred /= self.inference_z_samples
        elif self.inference_kind == "exz":
            if self.exz_estimator == "n2n":
                eyz = self.eyz(net_input)
                pred = 2 * eyz - waveforms
            elif self.exz_estimator == "2n2":
                emz = self.emz(net_input)
                pred = waveforms - 2 * emz
            elif self.exz_estimator == "n3n":
                eyz = self.eyz(net_input)
                emz = self.emz(net_input)
                pred = eyz - emz
            elif self.exz_estimator == "3n3":
                emz = self.emz(net_input)
                pred = waveforms - emz
            else:
                assert False
        else:
            assert False

        pred = self.to_orig_channels(pred, max_channels)

        return pred

    def train_forward(self, y, m, ell, mask):
        z = y + m

        # predictions given z
        # TODO: variance given z and put it in the loss
        exz = eyz = emz = e_exz_y = cycle_output = None
        net_input = z, mask.unsqueeze(1)
        if self.exz_estimator == "n2n":
            eyz = self.eyz(net_input)
            exz = 2 * eyz - z
        elif self.exz_estimator == "2n2":
            emz = self.emz(net_input)
            exz = z - 2 * emz
        elif self.exz_estimator == "n3n":
            eyz = self.eyz(net_input)
            emz = self.emz(net_input)
            exz = eyz - emz
        elif self.exz_estimator == "3n3":
            emz = self.emz(net_input)
            exz = y - emz
        else:
            assert False

        # predictions given y, if relevant
        if self.inference_kind == "amortized":
            e_exz_y = self.inf_net((y, mask.unsqueeze(1)))

        if self.cycle_loss_alpha:
            cycle_targ = e_exz_y.detach() if self.detach_cycle_loss else e_exz_y
            cycle_input = cycle_targ + ell
            cycle_output = self.den_net((cycle_input, mask.unsqueeze(1)))

        return dict(
            exz=exz,
            eyz=eyz,
            emz=emz,
            e_exz_y=e_exz_y,
            cycle_output=cycle_output,
        )

    def loss(self, mask, waveforms, m, net_outputs, l4_alpha=None, output_l1_alpha=None):
        loss_dict = {}
        mask = mask.unsqueeze(1)

        exz = net_outputs['exz']
        eyz = net_outputs['eyz']
        emz = net_outputs['emz']
        e_exz_y = net_outputs['e_exz_y']
        cycle_output = net_outputs['cycle_output']

        if eyz is not None:
            eyz_mask = mask * eyz
            loss_dict["eyz"] = F.mse_loss(eyz_mask, mask * waveforms)
            if l4_alpha:
                loss_dict["eyz_l4"] = l4_alpha * ((eyz - waveforms).mul_(mask) ** 4).mean()
            if output_l1_alpha:
                loss_dict["eyz_l1"] = output_l1_alpha * eyz_mask.abs().mean()
        if emz is not None:
            loss_dict["emz"] = F.mse_loss(mask * emz, mask * m)
            if l4_alpha:
                loss_dict["emz_l4"] = l4_alpha * ((emz - m).mul_(mask) ** 4).mean()
        if e_exz_y is not None:
            to_amortize = exz
            if self.detach_amortizer:
                # should amortize-ability affect the learning of eyz, emz?
                to_amortize = to_amortize.detach()
            am_mask = mask * to_amortize
            loss_dict["e_exz_y"] = F.mse_loss(am_mask, mask * e_exz_y)
            if l4_alpha:
                loss_dict["e_exz_y_l4"] = l4_alpha * ((to_amortize - e_exz_y).mul_(mask) ** 4).mean()
            if output_l1_alpha:
                loss_dict["e_exz_y_l1"] = output_l1_alpha * am_mask.abs().mean()
        if cycle_output is not None:
            coef = 1 if self.separate_cycle_net else self.cycle_loss_alpha
            cycle_targ = e_exz_y.detach() if self.detach_cycle_loss else e_exz_y
            loss_dict["cycle"] = coef * F.mse_loss(mask * cycle_targ, mask * cycle_output)
        return loss_dict

    def get_losses(self, waveforms, channels, recording):
        n = len(waveforms)
        noise = get_noise(
            recording,
            channels.numpy(force=True),
            self.model_channel_index_np,
            spike_length_samples=self.spike_length_samples,
            rg=self.rg,
        )
        dataset = TensorDataset(waveforms, channels, noise)
        loader = DataLoader(dataset, batch_size=self.inference_batch_size)

        losses = {
            'eyz': np.zeros(n, dtype=np.float32),
            'emz': np.zeros(n, dtype=np.float32),
            'e_exz_y': np.zeros(n, dtype=np.float32),
        }

        bs = 0
        self.eval()
        with torch.no_grad():
            for waveform_batch, channels_batch, noise_batch in loader:
                waveform_batch = waveform_batch.to(self.device)
                channels_batch = channels_batch.to(self.device)
                noise_batch = noise_batch.to(self.device)
                be = bs + len(waveform_batch)

                waveform_batch = reindex(
                    channels_batch,
                    waveform_batch,
                    self.relative_index,
                    pad_value=0.0,
                )
                m = noise_batch.to(waveform_batch)
                mask = self.get_masks(channels_batch).to(waveform_batch)
                fres = self.train_forward(waveform_batch, m, mask)

                eyz = fres['eyz']
                emz = fres['emz']
                e_exz_y = fres['e_exz_y']

                mask = mask.unsqueeze(1)
                if eyz is not None:
                    ll = F.mse_loss(mask * eyz, mask * waveform_batch, reduction='none')
                    losses["eyz"][bs:be] = ll.mean(dim=(1, 2)).numpy(force=True)
                if emz is not None:
                    ll = F.mse_loss(mask * emz, mask * m, reduction='none')
                    losses["emz"][bs:be] = ll.mean(dim=(1, 2)).numpy(force=True)
                if e_exz_y is not None:
                    ll = F.mse_loss(mask * exz, mask * e_exz_y, reduction='none')
                    losses["e_exz_y"][bs:be] = ll.mean(dim=(1, 2)).numpy(force=True)
                bs = be
        return losses

    def _fit(self, waveforms, channels, recording, weights=None):
        self.initialize_nets(waveforms.shape[1])
        waveforms = waveforms.cpu()
        channels = channels.cpu()

        val_size = 0
        train_indices = slice(None)
        val_indices = None
        if self.val_split_p:
            num_samples = len(waveforms)
            val_size = int(self.val_split_p * num_samples)
            train_size = num_samples - val_size
            train_indices = self.rg.choice(num_samples, size=train_size, replace=False)
            val_indices = np.setdiff1d(np.arange(num_samples), train_indices)

        # training dataset
        train_waveforms = waveforms[train_indices]
        train_channels = channels[train_indices]
        train_dataset = TensorDataset(train_waveforms, train_channels)
        noise_train_dataset = SameChannelNoiseDataset(
            recording,
            train_channels.numpy(force=True),
            self.model_channel_index_np,
            spike_length_samples=self.spike_length_samples,
            generator=spawn_torch_rg(self.rg),
        )
        if self.cycle_loss_alpha:
            cycle_noise_dataset = SameChannelNoiseDataset(
                recording,
                train_channels.numpy(force=True),
                self.model_channel_index_np,
                spike_length_samples=self.spike_length_samples,
                generator=spawn_torch_rg(self.rg),
            )
        else:
            cycle_noise_dataset = NoneDataset(len(train_channels))
        train_stack_dataset = StackDataset(train_dataset, noise_train_dataset, cycle_noise_dataset)
        if weights is None:
            train_sampler = RandomSampler(
                train_stack_dataset, generator=spawn_torch_rg(self.rg)
            )
        else:
            weights = weights[train_indices]
            train_sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_stack_dataset),
                generator=spawn_torch_rg(self.rg),
            )
        train_loader = DataLoader(
            train_stack_dataset,
            sampler=BatchSampler(
                train_sampler, batch_size=self.batch_size, drop_last=True
            ),
            num_workers=self.n_data_workers,
            persistent_workers=bool(self.n_data_workers),
            batch_size=None,
        )

        # initialize validation datasets only if val_split_p > 0
        if val_size > 0:
            val_waveforms = waveforms[val_indices]
            val_channels = channels[val_indices]
            val_noise = get_noise(
                recording,
                val_channels.numpy(force=True),
                self.model_channel_index_np,
                spike_length_samples=self.spike_length_samples,
                rg=self.rg,
            )
            val_dataset = TensorDataset(val_waveforms, val_channels, val_noise)

            # val set does not need shuffling
            val_loader = DataLoader(
                val_dataset,
                num_workers=self.n_data_workers,
                persistent_workers=bool(self.n_data_workers),
                batch_size=self.batch_size,
            )
        else:
            print("Skipping validation as val_split_p=0")
            val_loader = None

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        last_val_loss = None
        train_records = []

        with trange(self.n_epochs, desc="Epochs", unit="epoch") as pbar:
            for epoch in pbar:
                # Training phase
                self.train()
                train_losses = {}
                examples_this_epoch = 0
                for (waveform_batch, channels_batch), noise_batch, cnoise_batch in train_loader:
                    waveform_batch = waveform_batch.to(self.device)
                    channels_batch = channels_batch.to(self.device)
                    waveform_batch = reindex(
                        channels_batch,
                        waveform_batch,
                        self.relative_index,
                        pad_value=0.0,
                    )

                    optimizer.zero_grad()
                    m = noise_batch.to(waveform_batch)
                    ell = None
                    if cnoise_batch is not None:
                        ell = cnoise_batch.to(waveform_batch)
                    mask = self.get_masks(channels_batch).to(waveform_batch)
                    fres = self.train_forward(waveform_batch, m, ell, mask)
                    loss_dict = self.loss(
                        mask,
                        waveform_batch,
                        m,
                        fres,
                        l4_alpha=self.l4_alpha,
                        output_l1_alpha=self.output_l1_alpha,
                    )
                    loss = sum(loss_dict.values())
                    loss.backward()
                    optimizer.step()

                    for k, v in loss_dict.items():
                        train_losses[k] = v.item() + train_losses.get(k, 0.0)

                    examples_this_epoch += len(channels_batch)
                    if examples_this_epoch > self.examples_per_epoch:
                        break

                train_losses = {
                    k: v / len(train_loader) for k, v in train_losses.items()
                }
                train_records.append({**train_losses})

                if self.step_callback is not None:
                    self.step_callback(self, epoch)

                # Validation phase (only if val_loader is not None)
                val_losses = {}
                if val_loader:
                    self.eval()
                    val_losses = {}
                    with torch.no_grad():
                        for waveform_batch, channels_batch, noise_batch in val_loader:
                            waveform_batch = waveform_batch.to(self.device)
                            channels_batch = channels_batch.to(self.device)
                            noise_batch = noise_batch.to(self.device)

                            waveform_batch, mask = self.to_nn_channels(
                                waveform_batch, channels_batch
                            )
                            m = noise_batch.to(waveform_batch)
                            exz, eyz, emz, e_exz_y = self.train_forward(
                                waveform_batch, m, mask
                            )
                            loss_dict = self.loss(
                                mask, waveform_batch, m, exz, eyz, emz, e_exz_y
                            )
                            for k, v in loss_dict.items():
                                val_losses[k] = v.item() + val_losses.get(k, 0.0)

                    val_losses = {k: v / len(val_loader) for k, v in val_losses.items()}
                    val_loss = sum(val_losses.values())
                    train_records[-1]["val_loss"] = val_loss

                    if (
                        self.earlystop_eps is not None
                        and last_val_loss is not None
                        and val_loss - last_val_loss > self.earlystop_eps
                    ):
                        if epoch >= self.min_epochs:
                            print(f"Early stopping after {epoch} epochs.")
                            break
                    last_val_loss = val_loss

                # Print loss summary
                loss_str = " | ".join(
                    [f"Train {k}: {v:.3f}" for k, v in train_losses.items()]
                    + (
                        [f"Val {k}: {v:.3f}" for k, v in val_losses.items()]
                        if val_loader
                        else []
                    )
                )
                pbar.set_description(f"Epochs [{loss_str}]")

                if scheduler is not None:
                    scheduler.step()

        train_df = pd.DataFrame.from_records(train_records)
        return train_df


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
        with_indices=False,
        generator=None,
    ):
        super().__init__()
        self.recording = recording
        self.channels = channels
        self.spike_length_samples = spike_length_samples
        self.channel_index = channel_index
        self.with_indices = with_indices
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
        if self.with_indices:
            return index, noise
        return noise
