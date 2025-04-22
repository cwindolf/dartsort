import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange


from ._multichan_denoiser_kit import BaseMultichannelDenoiser
from torch.utils.data import random_split


class SupervisedDenoiser(BaseMultichannelDenoiser):
    default_name = "superviseddenoiser"

    def initialize_nets(self, spike_length_samples):
        self.initialize_shapes(spike_length_samples)
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
        with torch.enable_grad():
            res = self._fit(waveforms, gt_waveforms, max_channels)
        self._needs_fit = False
        return res

    def loss(self, mask, gt_waveforms, pred):
        mask = mask.unsqueeze(1)
        loss_dict = dict(mse=F.mse_loss(mask * gt_waveforms, mask * pred))
        return loss_dict

    def _fit(self, waveforms, gt_waveforms, channels):
        self.initialize_nets(waveforms.shape[1])
        waveforms = waveforms.cpu()
        gt_waveforms = gt_waveforms.cpu()
        channels = channels.cpu()
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
            val_dataset = None
            val_loader = None

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        train_losses_per_epoch = []
        val_losses_per_epoch = []
        last_val_loss = None

        with trange(self.n_epochs, desc="Epochs", unit="epoch") as pbar:
            for epoch in pbar:
                self.train()
                train_loss_sum = 0.0
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

                avg_val_loss = None
                if val_size:
                    assert val_loader is not None
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

                loss_str = f"Train loss: {avg_train_loss:3g}"
                if val_size:
                    loss_str += f" | Val loss: {avg_val_loss:3g}"
                pbar.set_description(f"Epochs [{loss_str}]")

                if scheduler is not None:
                    scheduler.step()

        return dict(
            train_losses=train_losses_per_epoch,
            val_losses=val_losses_per_epoch,
        )
