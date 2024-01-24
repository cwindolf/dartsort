import torch
from torch import nn

# TODO implement WaveformDenoiser versions


class Decollider(nn.Module):
    """Implements save/load logic for subclasses."""

    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    def save(self, pt_path):
        data = self.state_dict()
        data["decollider_subclass"] = self.__class__.name
        torch.save(data, pt_path)

    @classmethod
    def load(cls, pt_path):
        data = torch.load(pt_path)
        cls_name = data.pop("decollider_subclass")
        subcls = cls.subclasses[cls_name]
        self = subcls()
        self.load_state_dict(data, map_location="cpu")
        return self

    def n2n_predict(self, noisier_waveforms, channel_masks=None, alpha=1.0):
        """See Noisier2Noise paper. This is their Eq. 6.

        If you plan to use this at inference time, then multiply your noise2 during
        training by alpha.
        """
        expected_noisy_waveforms = self.forward(noisier_waveforms, channel_masks=channel_masks)
        if alpha == 1.0:
            return 2.0 * expected_noisy_waveforms - noisier_waveforms
        alpha2 = alpha * alpha
        return ((1.0 + alpha2) * expected_noisy_waveforms - noisier_waveforms) / alpha2


# -- single channel decolliders


class SingleChannelDecollider(Decollider):
    """N1T -> N1T"""

    def forward(self, waveforms, channel_masks=None):
        return self.net(waveforms)


class ConvToLinearSingleChannelDecollider(SingleChannelDecollider):
    def __init__(
        self,
        out_channels=(16, 32, 64),
        kernel_lengths=(5, 5, 11),
        spike_length_samples=121,
    ):
        super().__init__()
        in_channels = (1,) + out_channels[:-1]
        self.net = nn.Sequential()
        for ic, oc, k in zip(in_channels, out_channels, kernel_lengths):
            self.net.append(nn.Conv1d(ic, oc, k))
            self.net.append(nn.ReLU())
        self.net.append(nn.Flatten())
        flat_dim = out_channels[-1] * (
            spike_length_samples - sum(kernel_lengths) + len(kernel_lengths)
        )
        self.net.append(nn.Linear(flat_dim, spike_length_samples))
        # add the empty channel dim back in
        self.net.append(nn.Unflatten(1, (1, spike_length_samples)))


class MLPSingleChannelDecollider(SingleChannelDecollider):
    def __init__(self, hidden_sizes=(512, 256, 256), spike_length_samples=121):
        super().__init__()
        self.net = nn.Sequential()
        self.net.append(nn.Flatten())
        input_sizes = (spike_length_samples,) + hidden_sizes[:-1]
        output_sizes = hidden_sizes
        for fin, fout in zip(input_sizes, output_sizes):
            self.net.append(nn.Linear(fin, fout))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_sizes[-1], spike_length_samples))
        # add the empty channel dim back in
        self.net.append(nn.Unflatten(1, (1, spike_length_samples)))


# -- multi channel decolliders


class MultiChannelDecollider(Decollider):
    """NCT -> NCT

    self.net must map NC2T -> NCT

    Mask is added like so:
    waveforms  NCT  ->  N1CT \
    masks      NC   ->  N1C1 -> N2CT (broadcast and concat)
    """

    def forward(self, waveforms, channel_masks=None):
        # add the masks as an input channel
        # I somehow feel that receiving a "badness indicator" is more useful,
        # and the masks indicate good channels, so hence the flip below
        masks = torch.logical_not(channel_masks).to(waveforms)
        # NCT -> N1CT (channels are height in Conv2D NCHW convention)
        waveforms = waveforms[:, None, :, :]
        # NC -> N1CT
        masks = torch.broadcast_to(masks[:, None, :, None], waveforms.shape)
        # -> N2CT, concatenate on channel dimension (NCHW)
        combined = torch.concatenate((waveforms, masks), dim=1)
        return self.net(combined)


class ConvToLinearMultiChannelDecollider(SingleChannelDecollider):
    def __init__(
        self,
        out_channels=(16, 32, 64),
        kernel_height=4,
        kernel_lengths=(5, 5, 11),
        n_channels=1,
        spike_length_samples=121,
    ):
        super().__init__()
        in_channels = (2,) + out_channels[:-1]
        self.net = nn.Sequential()
        for ic, oc, k in zip(in_channels, out_channels, kernel_lengths):
            self.net.append(nn.Conv2d(ic, oc, (kernel_height, k)))
            self.net.append(nn.ReLU())
        self.net.append(nn.Flatten())
        out_w = spike_length_samples - sum(kernel_lengths) + len(kernel_lengths)
        out_h = n_channels - len(kernel_lengths) * (1 + kernel_height)
        flat_dim = out_channels[-1] * out_w * out_h
        self.net.append(nn.Linear(flat_dim, spike_length_samples))
        self.net.append(nn.Unflatten(1, (n_channels, spike_length_samples)))


class MLPMultiChannelDecollider(MultiChannelDecollider):
    def __init__(
        self, hidden_sizes=(1024, 512, 512), n_channels=1, spike_length_samples=121
    ):
        super().__init__()
        self.net = nn.Sequential()
        self.net.append(nn.Flatten())
        input_sizes = (n_channels * spike_length_samples,) + hidden_sizes[:-1]
        output_sizes = hidden_sizes
        for fin, fout in zip(input_sizes, output_sizes):
            self.net.append(nn.Linear(fin, fout))
            self.net.append(nn.ReLU())
        self.net.append(nn.Linear(hidden_sizes[-1], n_channels * spike_length_samples))
        self.net.append(nn.Unflatten(1, (n_channels, spike_length_samples)))