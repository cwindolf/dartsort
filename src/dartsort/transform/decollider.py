import numpy as np
import torch
import torch.nn.functional as F
from dartsort.util import nn_util, spikeio
from dartsort.util.spiketorch import get_relative_index, reindex
from dartsort.util.waveform_util import regularize_channel_index
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import trange

from .transform_base import BaseWaveformDenoiser


class Decollider(BaseWaveformDenoiser):
    default_name = "decollider"

    def __init__(
        self,
        channel_index,
        geom,
        recording,
        hidden_dims=(256, 256),
        use_batchnorm=True,
        name=None,
        name_prefix="",
        noisier3noise=False,
        inference_kind="raw",
        seed=0,
        batch_size=32,
        learning_rate=1e-3,
        epochs=25,
    ):
        assert inference_kind in ("raw", "amortized")

        self.use_batchnorm = use_batchnorm
        self.noisier3noise = noisier3noise
        self.inference_kind = inference_kind
        self.hidden_dims = hidden_dims
        self.n_channels = len(geom)
        self.recording = recording
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.rg = np.random.default_rng(seed)

        super().__init__(
            geom=geom, channel_index=channel_index, name=name, name_prefix=name_prefix
        )

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

    def needs_fit(self):
        return self._needs_fit

    def initialize_nets(self, spike_length_samples):
        self.spike_length_samples = spike_length_samples
        self.input_dim = (spike_length_samples + 1) * self.model_channel_index.shape[1]
        # i'm giving this two names... input_dim is the actual wf dim plus mask...
        self.output_dim = self.wf_dim = (
            spike_length_samples * self.model_channel_index.shape[1]
        )
        self.eyz = nn_util.get_mlp(
            self.input_dim,
            self.hidden_dims,
            self.output_dim,
            use_batchnorm=self.use_batchnorm,
        )
        if self.noisier3noise:
            self.emz = nn_util.get_mlp(
                self.input_dim,
                self.hidden_dims,
                self.output_dim,
                use_batchnorm=self.use_batchnorm,
            )
        if self.inference_kind == "amortized":
            self.inf_net = nn_util.get_mlp(
                self.input_dim,
                self.hidden_dims,
                self.output_dim,
                use_batchnorm=self.use_batchnorm,
            )

    def fit(self, waveforms, max_channels):
        waveforms = reindex(max_channels, waveforms, self.relative_index, pad_value=0.0)
        with torch.enable_grad():
            self._fit(waveforms, max_channels)
        self._needs_fit = False

    def forward(self, waveforms, max_channels):
        """Called only at inference time."""
        n = len(waveforms)
        waveforms = reindex(max_channels, waveforms, self.relative_index, pad_value=0.0)
        masks = self.get_masks(max_channels).to(waveforms)

        net_input = torch.cat((waveforms.view(n, self.wf_dim), masks), dim=1)
        if self.inference_kind == "amortized":
            pred = self.inf_net(net_input).view(waveforms.shape)
        elif self.inference_kind == "raw":
            pred = self.eyz(net_input).view(waveforms.shape)
        else:
            assert False

        pred = reindex(max_channels, pred, self.irrelative_index)

        return pred

    def get_masks(self, max_channels):
        return self.model_channel_index[max_channels] < self.n_channels

    def train_forward(self, y, m, mask):
        n = len(y)
        z = y + m
        z_flat = z.view(n, self.wf_dim)
        z_masked = torch.cat((z_flat, mask), dim=1)

        # things we may be computing
        exz = None
        eyz = None
        emz = None
        exy = None

        # predictions given z
        if self.noisier3noise:
            eyz = self.eyz(z_masked).view(y.shape)
            emz = self.emz(z_masked).view(y.shape)
            exz = eyz - emz
        else:
            eyz = self.eyz(z_masked).view(y.shape)
            exz = 2 * eyz - z

        # predictions given y, if relevant
        if self.inference_kind == "amortized":
            y_flat = y.view(n, self.wf_dim)
            y_masked = torch.cat((y_flat, mask), dim=1)
            exy = self.inf_net(y_masked).view(y.shape)

        return exz, eyz, emz, exy

    def get_noise(self, channels):
        # pick random times
        times_samples = self.rg.integers(
            self.recording.get_num_samples() - self.spike_length_samples,
            size=len(channels),
        )
        order = np.argsort(times_samples)
        inv_order = np.argsort(order)

        # load waveforms on the same channels and channel neighborhoods
        noise_waveforms = spikeio.read_waveforms_channel_index(
            self.recording,
            times_samples[order],
            self.model_channel_index_np,
            channels,
            trough_offset_samples=0,
            spike_length_samples=self.spike_length_samples,
            fill_value=0.0,
        )

        # back to random order
        noise_waveforms = noise_waveforms[inv_order]

        return torch.from_numpy(noise_waveforms)

    def loss(self, mask, waveforms, m, exz, eyz, emz=None, exy=None):
        loss_dict = {}
        mask = mask.unsqueeze(1)
        loss_dict["eyz"] = F.mse_loss(mask * eyz, mask * waveforms)
        if emz is not None:
            loss_dict["emz"] = F.mse_loss(mask * emz, mask * m)
        if exy is not None:
            loss_dict["exy"] = F.mse_loss(mask * exz, mask * exy)
        return loss_dict

    def _fit(self, waveforms, channels):
        self.initialize_nets(waveforms.shape[1])
        dataset = TensorDataset(waveforms, channels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(waveforms.device)

        with trange(self.epochs, desc="Epochs", unit="epoch") as pbar:
            for epoch in pbar:
                epoch_losses = {}
                for waveform_batch, channels_batch in dataloader:
                    optimizer.zero_grad()

                    # get a batch of noise samples
                    m = self.get_noise(channels_batch).to(waveform_batch)
                    mask = self.get_masks(channels_batch).to(waveform_batch)
                    exz, eyz, emz, exy = self.train_forward(waveform_batch, m, mask)
                    loss_dict = self.loss(mask, waveform_batch, m, exz, eyz, emz, exy)
                    loss = sum(loss_dict.values())
                    loss.backward()
                    optimizer.step()

                    for k, v in loss_dict.items():
                        epoch_losses[k] = v.item() + epoch_losses.get(k, 0.0)

                epoch_losses = {k: v / len(dataloader) for k, v in epoch_losses.items()}
                loss_str = ", ".join(f"{k}: {v:.3f}" for k, v in epoch_losses.items())
                pbar.set_description(f"Epochs [{loss_str}]")
