import torch
from dartsort.util.waveform_util import (get_channels_in_probe,
                                         set_channels_in_probe)
from torch import nn

from .transform_base import BaseWaveformDenoiser

try:
    from importlib.resources import files
except ImportError:
    try:
        from importlib_resources import files
    except ImportError:
        raise ValueError("Need python>=3.10 or pip install importlib_resources.")

default_pretrained_path = files("dartsort.pretrained")
default_pretrained_path = default_pretrained_path.joinpath("single_chan_denoiser.pt")


class SingleChannelWaveformDenoiser(BaseWaveformDenoiser):
    default_name = "single_chan_denoiser"

    def __init__(
        self,
        channel_index,
        geom=None,
        denoiser=None,
        batch_size=32,
        in_place=True,
        pretrained_path=default_pretrained_path,
        name=None,
        name_prefix="",
        clsname="SingleChannelDenoiser",
        n_epochs=None,
    ):
        super().__init__(
            channel_index=channel_index, name=name, name_prefix=name_prefix
        )
        self.batch_size = batch_size
        self.in_place = in_place

        if denoiser is None:
            denoiser = dnclss[clsname]().load(pretrained_path)
            denoiser.eval()
            denoiser.requires_grad_(False)
        self.denoiser = denoiser

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

    def load(self, pretrained_path=default_pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        self.load_state_dict(checkpoint)
        return self


class FlexibleSingleChanDenoiser(nn.Module):
    def __init__(self, pretrained_path=None, n_filters=[32, 32, 32], filter_sizes=[11, 11, 11], spike_size=121):
        super().__init__()
        nets = []
        for inf, outf, s in zip([1, *n_filters], n_filters, filter_sizes):
            nets.append(nn.Sequential(nn.Conv1d(inf, outf, s), nn.ReLU()))
        # self.conv1 = nn.Sequential(nn.Conv1d(1, feat1, size1), nn.ReLU())
        # self.conv2 = nn.Sequential(nn.Conv1d(feat1, feat2, size2), nn.ReLU())
        self.conv = nn.Sequential(*nets)
        n_input_feat = n_filters[-1] * (spike_size - sum(filter_sizes) + len(filter_sizes))
        self.out = nn.Linear(n_input_feat, spike_size)

    def forward(self, x):
        x = x[:, None]
        x = self.conv(x)
        # x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        return self.out(x)

    def load(self, pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        self.load_state_dict(checkpoint)
        return self

dnclss = {'FlexibleSingleChanDenoiser': FlexibleSingleChanDenoiser, 'SingleChannelDenoiser': SingleChannelDenoiser}