from pathlib import Path

import torch
from dartsort.util.torch_waveform_util import (get_channels_in_probe,
                                               set_channels_in_probe)
from torch import nn

from .base import BaseWaveformDenoiser

pretrained_path = (
    Path(__file__).parent.parent.parent / "pretrained/single_chan_denoiser.pt"
)


class SingleChannelWaveformDenoiser(BaseWaveformDenoiser):
    default_name = "single_chan_denoiser"

    def __init__(
        self,
        denoiser,
        channel_index,
        batch_size=128,
        in_place=False,
        name=None,
    ):
        super().__init__(name)
        self.denoiser = denoiser
        self.channel_index = channel_index
        self.batch_size = batch_size
        self.in_place = in_place

    @classmethod
    def load_pretrained(
        cls,
        channel_index,
        pretrained_path=pretrained_path,
        batch_size=128,
        in_place=False,
    ):
        denoiser = SingleChannelDenoiser().load(pretrained_path)
        denoiser.eval()
        return cls(
            channel_index, denoiser, batch_size=batch_size, in_place=in_place
        )

    def forward(self, waveforms, max_channels=None):
        (
            channels_in_probe,
            waveforms_in_probe,
        ) = get_channels_in_probe(waveforms, max_channels, self.channel_index)

        n_in_probe = len(waveforms_in_probe)
        for batch_start in range(0, n_in_probe, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_in_probe)
            waveforms_in_probe[batch_start:batch_end] = self.denoiser(
                waveforms_in_probe[batch_start:batch_end]
            )

        waveforms = set_channels_in_probe(
            waveforms_in_probe,
            waveforms,
            channels_in_probe,
            in_place=self.in_place,
        )

        return waveforms


class SingleChannelDenoiser(nn.Module):
    def __init__(
        self,
        n_filters=[16, 8],
        filter_sizes=[5, 11],
        spike_size=121,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, n_filters[0], filter_sizes[0]), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_filters[0], n_filters[1], filter_sizes[1]), nn.ReLU()
        )
        n_input_feat = n_filters[1] * (
            spike_size - filter_sizes[0] - filter_sizes[1] + 2
        )
        self.out = nn.Linear(n_input_feat, spike_size)

    def forward(self, x):
        x = x[:, None]
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        return self.out(x)

    def load(self, fname_model=pretrained_path):
        checkpoint = torch.load(fname_model, map_location="cpu")
        self.load_state_dict(checkpoint)
        return self
