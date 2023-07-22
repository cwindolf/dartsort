import numpy as np
import torch
import torch.nn.functional as F
from dartsort.util import waveform_util
from neuropixel import dense_layout


def test_make_channel_index():
    h = dense_layout()
    geom = np.c_[h["x"], h["y"]]
    assert waveform_util.make_channel_index(geom, 200).shape == (384, 40)


def test_channels_in_probe():
    h = dense_layout()
    geom = np.c_[h["x"], h["y"]]
    channel_index = waveform_util.make_channel_index(geom, 200)
    n_neighbors = channel_index.shape[1]

    # test channel in probe get/set
    n_samples = 3
    max_channels = torch.concatenate(
        [torch.arange(len(geom)), torch.flip(torch.arange(len(geom)), (0,))]
    )
    recording = F.pad(
        torch.zeros((n_samples, len(geom))), (0, 1), value=torch.nan
    )
    assert recording.shape == (n_samples, len(geom) + 1)
    waveforms = recording[
        torch.arange(n_samples)[None, :, None],
        channel_index[max_channels][:, None, :],
    ]
    assert waveforms.shape == (len(max_channels), n_samples, n_neighbors)
    assert np.array_equal(
        torch.isnan(waveforms[: len(geom), 0, :]).numpy(),
        channel_index == len(geom),
    )
    assert np.array_equal(
        torch.isnan(waveforms[len(geom) :, 0, :]).numpy(),
        (channel_index == len(geom))[::-1],
    )

    (
        channels_in_probe,
        waveforms_in_probe,
    ) = waveform_util.get_channels_in_probe(
        waveforms, max_channels, channel_index
    )
    assert not torch.isnan(waveforms_in_probe).any()
    assert 0 < len(waveforms_in_probe) < len(max_channels) * n_neighbors
    assert (waveforms_in_probe == 0).all()

    waveform_util.set_channels_in_probe(
        waveforms_in_probe + 1, waveforms, channels_in_probe, in_place=True
    )
    assert np.array_equal(
        torch.isnan(waveforms[: len(geom), 0, :]).numpy(),
        channel_index == len(geom),
    )
    assert np.array_equal(
        torch.isnan(waveforms[len(geom) :, 0, :]).numpy(),
        (channel_index == len(geom))[::-1],
    )
    assert np.array_equal(
        (waveforms[: len(geom), 0, :] == 1).numpy(),
        channel_index < len(geom),
    )
    assert np.array_equal(
        (waveforms[len(geom) :, 0, :] == 1).numpy(),
        (channel_index < len(geom))[::-1],
    )

    (
        channels_in_probe,
        waveforms_in_probe,
    ) = waveform_util.get_channels_in_probe(
        waveforms, max_channels, channel_index
    )
    assert not torch.isnan(waveforms_in_probe).any()
    assert 0 < len(waveforms_in_probe) < len(max_channels) * n_neighbors
    assert (waveforms_in_probe == 1).all()


def test_channel_subsetting():
    h = dense_layout()
    geom = np.c_[h["x"], h["y"]]
    channel_index = waveform_util.make_channel_index(geom, 200)
    n_neighbors = channel_index.shape[1]

    # test channel in probe get/set
    n_samples = 3
    max_channels = torch.arange(len(geom))
    recording = F.pad(
        torch.zeros((n_samples, len(geom))), (0, 1), value=torch.nan
    )
    waveforms = recording[
        torch.arange(n_samples)[None, :, None],
        channel_index[max_channels][:, None, :],
    ]

    (
        waveforms_small,
        small_channel_index,
    ) = waveform_util.channel_subset_by_radius(
        waveforms,
        max_channels,
        channel_index,
        geom,
        100,
    )
    assert waveforms_small.shape == (
        *waveforms.shape[:2],
        small_channel_index.shape[1],
    )

    (
        channels_in_probe_small,
        waveforms_in_probe_small,
    ) = waveform_util.get_channels_in_probe(
        waveforms_small,
        max_channels,
        small_channel_index,
    )

    assert not torch.isnan(waveforms_in_probe_small).any()


if __name__ == "__main__":
    test_channels_in_probe()
    test_channel_subsetting()
