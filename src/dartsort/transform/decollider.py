import numpy as np
import torch
import torch.nn.functional as F
from dartsort.util import spikeio
from dartsort.util.spiketorch import reindex, spawn_torch_rg
import pandas as pd
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    StackDataset,
    TensorDataset,
)
from tqdm.auto import trange

from ._base_nn_denoiser import BaseMultichannelDenoiser


class Decollider(BaseMultichannelDenoiser):
    default_name = "decollider"

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
        weight_decay=0,
        n_epochs=50,
        channelwise_dropout_p=0.0,
        with_conv_fullheight=False,
        pretrained_path=None,
        val_split_p=0.0,
        min_epochs=10,
        earlystop_eps=None,
        random_seed=0,
        res_type="none",
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
        val_noise_random_seed=0,
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

    def initialize_nets(self, spike_length_samples):
        self.initialize_shapes(spike_length_samples)
        if self.exz_estimator in ("n2n", "n3n"):
            self.eyz = self.get_mlp(res_type=self.eyz_res_type)
        if self.exz_estimator in ("n3n", "2n2", "3n3"):
            self.emz = self.get_mlp(res_type=self.emz_res_type)
        if self.inference_kind == "amortized":
            self.inf_net = self.get_mlp(res_type=self.e_exz_y_res_type)
        self.to(self.device)

    def fit(self, waveforms, max_channels, recording):
        with torch.enable_grad():
            res = self._fit(waveforms, max_channels, recording)
        self._needs_fit = False
        return res

    def forward(self, waveforms, max_channels):
        """Called only at inference time."""
        # TODO: batch all of this.
        waveforms, masks = self.to_nn_channels(waveforms, max_channels)
        net_input = waveforms, masks.unsqueeze(1)

        if self.inference_kind == "amortized":
            pred = self.inf_net(net_input)
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

        pred = self.to_orig_channels(waveforms, max_channels)

        return pred

    def train_forward(self, y, m, mask):
        z = y + m

        # predictions given z
        # TODO: variance given z and put it in the loss
        exz = eyz = emz = e_exz_y = None
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

        return exz, eyz, emz, e_exz_y

    def loss(self, mask, waveforms, m, exz, eyz=None, emz=None, e_exz_y=None):
        loss_dict = {}
        mask = mask.unsqueeze(1)
        if eyz is not None:
            loss_dict["eyz"] = F.mse_loss(mask * eyz, mask * waveforms)
        if emz is not None:
            loss_dict["emz"] = F.mse_loss(mask * emz, mask * m)
        if e_exz_y is not None:
            to_amortize = exz
            if self.detach_amortizer:
                # should amortize-ability affect the learning of eyz, emz?
                to_amortize = to_amortize.detach()
            loss_dict["e_exz_y"] = F.mse_loss(mask * to_amortize, mask * e_exz_y)
        return loss_dict

    def _fit(self, waveforms, channels, recording):
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
        train_stack_dataset = StackDataset(train_dataset, noise_train_dataset)
        train_sampler = RandomSampler(
            train_stack_dataset, generator=spawn_torch_rg(self.rg)
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
        noise_val_dataset = None  # for pyright
        if val_size > 0:
            val_waveforms, val_channels = waveforms[val_indices], channels[val_indices]
            val_dataset = TensorDataset(val_waveforms, val_channels)
            noise_val_dataset = SameChannelNoiseDataset(
                recording,
                val_channels.numpy(force=True),
                self.model_channel_index_np,
                spike_length_samples=self.spike_length_samples,
                # NB: we re-seed this guy's generator before every validation below
                # so that the noise is always the same!
                generator=spawn_torch_rg(self.val_noise_random_seed),
            )
            val_stack_dataset = StackDataset(val_dataset, noise_val_dataset)

            # val set does not need shuffling
            val_sampler = SequentialSampler(val_stack_dataset)
            val_loader = DataLoader(
                val_stack_dataset,
                sampler=BatchSampler(
                    val_sampler, batch_size=self.batch_size, drop_last=False
                ),
                num_workers=self.n_data_workers,
                persistent_workers=bool(self.n_data_workers),
                batch_size=None,
            )
        else:
            print("Skipping validation as val_split_p=0")
            val_loader = None

        optimizer = self.get_optimizer()

        last_val_loss = None
        train_records = []

        with trange(self.n_epochs, desc="Epochs", unit="epoch") as pbar:
            for epoch in pbar:
                # Training phase
                self.train()
                train_losses = {}
                examples_this_epoch = 0
                for (waveform_batch, channels_batch), noise_batch in train_loader:
                    waveform_batch = waveform_batch.to(self.device)
                    channels_batch = channels_batch.to(self.device)
                    noise_batch = noise_batch.to(self.device)
                    waveform_batch = reindex(
                        channels_batch,
                        waveform_batch,
                        self.relative_index,
                        pad_value=0.0,
                    )

                    optimizer.zero_grad()
                    m = noise_batch.to(waveform_batch)
                    mask = self.get_masks(channels_batch).to(waveform_batch)
                    exz, eyz, emz, e_exz_y = self.train_forward(waveform_batch, m, mask)
                    loss_dict = self.loss(
                        mask, waveform_batch, m, exz, eyz, emz, e_exz_y
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

                # Validation phase (only if val_loader is not None)
                val_losses = {}
                if val_loader:
                    assert noise_val_dataset is not None
                    noise_val_dataset.generator.manual_seed(self.val_noise_random_seed)
                    self.eval()
                    val_losses = {}
                    with torch.no_grad():
                        for (waveform_batch, channels_batch), noise_batch in val_loader:
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
