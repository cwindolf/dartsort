import numpy as np
import torch
import torch.nn.functional as F
from dartsort.util import nn_util, spikeio
from dartsort.util.spiketorch import get_relative_index, reindex
from dartsort.util.waveform_util import (grab_main_channels,
                                         regularize_channel_index)
from torch.utils.data import (BatchSampler, DataLoader, Dataset, RandomSampler,
                              StackDataset, TensorDataset,
                              WeightedRandomSampler)
from tqdm.auto import trange


from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from .transform_base import BaseWaveformDenoiser
from torch.utils.data import random_split

class SupervisedDenoiser(BaseWaveformDenoiser):
    default_name = "supervisedenoiser"

    def __init__(
        self,
        channel_index,
        geom,
        hidden_dims=(256, 256, 256),
        norm_kind="layernorm",
        name=None,
        name_prefix="",
        batch_size=32, #####
        learning_rate=1e-3,
        n_epochs=50,
        channelwise_dropout_p=0.2,
        n_data_workers=4,
        with_conv_fullheight=False,
        pretrained_path=None,
        val_split_p=0.1,    
        min_epochs=5,    
        convergence_eps=0.01,
        random_seed=0,
        res_type="none", # added
    ):
        assert pretrained_path is None
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
        self.n_data_workers = n_data_workers
        self.with_conv_fullheight = with_conv_fullheight
        self.emz_res_type = "none"

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
        self._needs_fit = True # 
        self.val_split_p = val_split_p
        self.min_epochs = min_epochs
        self.convergence_eps = convergence_eps
        self.rg = np.random.default_rng(random_seed) # 
        self.res_type = res_type


    @property
    def device(self):
        return self.channel_index.device

    def needs_fit(self):
        return self._needs_fit

    def get_mlp(self, residual=False, res_type="none"):
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
            residual=residual,
            residual_blocks=False,
            res_type=res_type
        )


    def initialize_nets(self, spike_length_samples):
        self.spike_length_samples = spike_length_samples
        self.output_dim = self.wf_dim = (
            spike_length_samples * self.model_channel_index.shape[1]
        )
        self.exy = self.get_mlp(res_type = self.res_type)
        self.to(self.relative_index.device)

    def fit(self, waveforms, gt_waveforms, max_channels, recording):
        with torch.enable_grad():
            self._fit(waveforms, gt_waveforms, max_channels, recording)
        self._needs_fit = False

    def forward(self, waveforms, max_channels):
        """Called only at inference time."""
        # batching? 
        waveforms = reindex(max_channels, waveforms, self.relative_index, pad_value=0.0)
        masks = self.get_masks(max_channels).to(waveforms)
        net_input = waveforms, masks.unsqueeze(1)
        pred = self.exy(net_input)
        pred = reindex(max_channels, pred, self.irrelative_index)
        return pred

    def get_masks(self, max_channels):
        return self.model_channel_index[max_channels] < self.n_channels

    def loss(self, mask, gt_waveforms, pred):
        mask = mask.unsqueeze(1)
        loss_dict = dict(mse=F.mse_loss(mask * gt_waveforms, mask * pred))
        return loss_dict

            
    def _fit(self, waveforms, gt_waveforms, channels, recording):
        self.initialize_nets(waveforms.shape[1])
        waveforms = waveforms.cpu()
        gt_waveforms = gt_waveforms.cpu()
        channels = channels.cpu()
        print(self.res_type)
        num_samples = len(waveforms)
        train_size = num_samples 
    
        train_dataset = TensorDataset(waveforms, gt_waveforms, channels)
    
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_data_workers,
            persistent_workers=bool(self.n_data_workers),
        )
    
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        
        with trange(self.n_epochs, desc="Epochs", unit="epoch") as pbar:
            for epoch in pbar:
                self.train()
                train_losses = {}
    
                for waveform_batch, gt_waveform_batch, channels_batch in train_loader:
                    waveform_batch = waveform_batch.to(self.device)
                    gt_waveform_batch = gt_waveform_batch.to(self.device)
                    channels_batch = channels_batch.to(self.device)
               
       


                    waveform_batch = reindex(channels_batch, waveform_batch, self.relative_index, pad_value=0.0)
                    gt_waveform_batch = reindex(channels_batch, gt_waveform_batch, self.relative_index, pad_value=0.0)
                                       

    
                    optimizer.zero_grad()
    
                    mask = self.get_masks(channels_batch).to(waveform_batch)
                    pred = self.forward(waveform_batch, channels_batch)

                    pred = reindex(channels_batch, pred, self.relative_index, pad_value=0.0)


                    # print("waveform_batch shape:", waveform_batch.shape) 
                    # print("gt_waveform_batch shape", gt_waveform_batch.shape)
                    # print("pred shape", pred.shape)

                    
                    loss_dict = self.loss(mask, gt_waveform_batch, pred)
                    loss = sum(loss_dict.values())
                    loss.backward()
                    optimizer.step()
    
                    for k, v in loss_dict.items():
                        train_losses[k] = v.item() + train_losses.get(k, 0.0)
    
                train_losses = {k: v / len(train_loader) for k, v in train_losses.items()}
    
                loss_str = " | ".join([f"Train {k}: {v:.3f}" for k, v in train_losses.items()])
                pbar.set_description(f"Epochs [{loss_str}]")
                scheduler.step(epoch + 1)


        

        
    # def _fit(self, waveforms, gt_waveforms, channels, recording):
    #     self.initialize_nets(waveforms.shape[1])
    #     waveforms = waveforms.cpu()
    #     gt_waveforms = gt_waveforms.cpu()
    #     channels = channels.cpu()
    
    #     num_samples = len(waveforms)
    #     val_size = int(self.val_split_p * num_samples)
    #     train_size = num_samples - val_size
    #     train_indices = self.rg.choice(num_samples, size=train_size, replace=False)
    #     val_indices = np.setdiff1d(np.arange(num_samples), train_indices)
    
    #     train_waveforms, val_waveforms = waveforms[train_indices], waveforms[val_indices]
    #     train_targets, val_targets = gt_waveforms[train_indices], gt_waveforms[val_indices]
    #     train_channels, val_channels = channels[train_indices], channels[val_indices]
    
    #     train_dataset = TensorDataset(train_waveforms, train_targets, train_channels)
    #     val_dataset = TensorDataset(val_waveforms, val_targets, val_channels)
    #     # TODO left off here

    #     noise_train_dataset = TensorDataset(train_waveforms)  # Noisy version
    #     noise_val_dataset = TensorDataset(val_waveforms)  # Noisy version

    #     # Wrap datasets for training
    #     train_stack_dataset = StackDataset(train_dataset, noise_train_dataset)
    #     val_stack_dataset = StackDataset(val_dataset, noise_val_dataset)

    #     # Define data samplers
    #     train_sampler = RandomSampler(train_stack_dataset)
    #     val_sampler = RandomSampler(val_stack_dataset)

    #     # Define data loaders
    #     train_loader = DataLoader(
    #         train_stack_dataset,
    #         sampler=BatchSampler(train_sampler, batch_size=self.batch_size, drop_last=True),
    #         num_workers=self.n_data_workers,
    #         persistent_workers=bool(self.n_data_workers),
    #     )
    #     val_loader = DataLoader(
    #         val_stack_dataset,
    #         sampler=BatchSampler(val_sampler, batch_size=self.batch_size, drop_last=False),
    #         num_workers=self.n_data_workers,
    #         persistent_workers=bool(self.n_data_workers),
    #     )

    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    #     best_val_loss = float('inf')
    #     last_val_loss = None

    #     with trange(self.n_epochs, desc="Epochs", unit="epoch") as pbar:
    #         for epoch in pbar:
    #             self.train()
    #             train_losses = {}
    #             for (waveform_batch, gt_waveform_batch, channels_batch), noise_batch in train_loader:
    #                 waveform_batch = waveform_batch.to(self.device)  
    #                 gt_waveform_batch = gt_waveform_batch.to(self.device)  
    #                 channels_batch = channels_batch.to(self.device)

    #                 waveform_batch = reindex(channels_batch, waveform_batch, self.relative_index, pad_value=0.0)

    #                 optimizer.zero_grad()

    #                 mask = self.get_masks(channels_batch).to(waveform_batch)
    #                 pred = self.forward(waveform_batch, channels_batch)

    #                 loss_dict = self.loss(mask, gt_waveform_batch, pred)
    #                 loss = sum(loss_dict.values())
    #                 loss.backward()
    #                 optimizer.step()

    #                 for k, v in loss_dict.items():
    #                     train_losses[k] = v.item() + train_losses.get(k, 0.0)

    #             train_losses = {k: v / len(train_loader) for k, v in train_losses.items()}

    #             self.eval()
    #             val_losses = {}
    #             with torch.no_grad():
    #                 for (waveform_batch, gt_waveform_batch, channels_batch), noise_batch in val_loader:
    #                     waveform_batch = waveform_batch.to(self.device)
    #                     gt_waveform_batch = gt_waveform_batch.to(self.device)
    #                     channels_batch = channels_batch.to(self.device)

    #                     waveform_batch = reindex(channels_batch, waveform_batch, self.relative_index, pad_value=0.0)

    #                     pred = self.forward(waveform_batch, channels_batch)
    #                     loss_dict = self.loss(mask, gt_waveform_batch, pred)
    #                     for k, v in loss_dict.items():
    #                         val_losses[k] = v.item() + val_losses.get(k, 0.0)

    #             val_losses = {k: v / len(val_loader) for k, v in val_losses.items()}
    #             val_loss = sum(val_losses.values())

    #             # Early stopping condition
    #             if last_val_loss is not None and abs(last_val_loss - val_loss) < self.convergence_eps:
    #                 if epoch >= self.min_epochs:
    #                     print(f"Early stopping after {epoch} epochs.")
    #                     break

    #             last_val_loss = val_loss
    #             if val_loss < best_val_loss:
    #                 best_val_loss = val_loss

    #             # Update progress bar
    #             loss_str = " | ".join(
    #                 [f"Train {k}: {v:.3f}" for k, v in train_losses.items()] +
    #                 [f"Val {k}: {v:.3f}" for k, v in val_losses.items()]
    #             )
    #             pbar.set_description(f"Epochs [{loss_str}]")


# unsupervised code    
        # train_stack_dataset = StackDataset(train_dataset, noise_train_dataset)
        # val_stack_dataset = StackDataset(val_dataset, noise_val_dataset)
    
        # train_sampler = RandomSampler(train_stack_dataset)
        # val_sampler = RandomSampler(val_stack_dataset)
    
        # train_loader = DataLoader(
        #     train_stack_dataset,
        #     sampler=BatchSampler(train_sampler, batch_size=self.batch_size, drop_last=True),
        #     num_workers=self.n_data_workers,
        #     persistent_workers=bool(self.n_data_workers),
        # )
        # val_loader = DataLoader(
        #     val_stack_dataset,
        #     sampler=BatchSampler(val_sampler, batch_size=self.batch_size, drop_last=False),
        #     num_workers=self.n_data_workers,
        #     persistent_workers=bool(self.n_data_workers),
        # )
    
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
        # best_val_loss = float('inf')
        # last_val_loss = None
        # with trange(self.n_epochs, desc="Epochs", unit="epoch") as pbar:
        #     for epoch in pbar:
        #         # Training phase
        #         self.train()
        #         train_losses = {}
        #         for (waveform_batch, channels_batch), noise_batch in train_loader:
        #             waveform_batch = waveform_batch[0].to(self.device)
        #             channels_batch = channels_batch[0].to(self.device)
        #             noise_batch = noise_batch[0].to(self.device)
        #             waveform_batch = reindex(channels_batch, waveform_batch, self.relative_index, pad_value=0.0)
    
        #             optimizer.zero_grad()
    
        #             m = noise_batch.to(waveform_batch)
        #             mask = self.get_masks(channels_batch).to(waveform_batch)
        #             exz, eyz, emz, e_exz_y = self.train_forward(waveform_batch, m, mask)
        #             loss_dict = self.loss(mask, waveform_batch, m, exz, eyz, emz, e_exz_y)
        #             loss = sum(loss_dict.values())
        #             loss.backward()
        #             optimizer.step()
    
        #             for k, v in loss_dict.items():
        #                 train_losses[k] = v.item() + train_losses.get(k, 0.0)
    
        #         train_losses = {k: v / len(train_loader) for k, v in train_losses.items()}
    
        #         self.eval()
        #         val_losses = {}
        #         with torch.no_grad():
        #             for (waveform_batch, channels_batch), noise_batch in val_loader:
        #                 waveform_batch = waveform_batch[0].to(self.device)
        #                 channels_batch = channels_batch[0].to(self.device)
        #                 noise_batch = noise_batch[0].to(self.device)
        #                 waveform_batch = reindex(channels_batch, waveform_batch, self.relative_index, pad_value=0.0)
    
        #                 m = noise_batch.to(waveform_batch)
        #                 mask = self.get_masks(channels_batch).to(waveform_batch)
        #                 exz, eyz, emz, e_exz_y = self.train_forward(waveform_batch, m, mask)
        #                 loss_dict = self.loss(mask, waveform_batch, m, exz, eyz, emz, e_exz_y)
        #                 for k, v in loss_dict.items():
        #                     val_losses[k] = v.item() + val_losses.get(k, 0.0)
    
        #         val_losses = {k: v / len(val_loader) for k, v in val_losses.items()}
        #         val_loss = sum(val_losses.values())
    
        #         if last_val_loss is not None and abs(last_val_loss - val_loss) < self.convergence_eps:
        #             if epoch >= self.min_epochs:
        #                 print(f"Early stopping after {epoch} epochs.")
        #                 break
    
        #         last_val_loss = val_loss
        #         if val_loss < best_val_loss:
        #             best_val_loss = val_loss
    
        #         # loss_str = ", ".join(f"{k}: {v:.3f}" for k, v in {**train_losses, **val_losses}.items())
        #         loss_str = " | ".join([f"Train {k}: {v:.3f}" for k, v in train_losses.items()] +[f"Val {k}:{v:.3f}" for k, v in val_losses.items()])
        #         pbar.set_description(f"Epochs [{loss_str}]")