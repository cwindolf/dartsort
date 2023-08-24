import numpy as np
import torch
import torch.nn.functional as F
from dartsort.util import spiketorch


def test_ptp():
    rg = np.random.default_rng(0)

    x = rg.normal(size=100)
    assert np.array_equal(spiketorch.ptp(torch.tensor(x), 0).numpy(), x.ptp())
    x = rg.normal(size=(100, 100))
    assert np.array_equal(spiketorch.ptp(torch.tensor(x), 0).numpy(), x.ptp(0))
    assert np.array_equal(spiketorch.ptp(torch.tensor(x), 1).numpy(), x.ptp(1))


def test_ravel_multi_index():
    # no broadcasting case
    x = torch.zeros((40, 41))
    x2 = x.clone()
    inds = (torch.LongTensor([1, 20, 30]), torch.LongTensor([2, 31, 30]))
    raveled = spiketorch.ravel_multi_index(inds, x.shape)
    x.view(-1)[raveled] = 1
    x2[inds] = 1
    assert np.array_equal(x.numpy(), x2.numpy())
    assert x.sum() == 3

    # case with broadcasting
    x = torch.zeros((40, 41))
    x2 = x.clone()
    inds = (
        torch.LongTensor([1, 20, 30])[:, None],
        torch.LongTensor([2, 31, 30])[:, None]
        + torch.tensor([-1, 0, 1])[None, :],
    )
    raveled = spiketorch.ravel_multi_index(inds, x.shape)
    x.view(-1)[raveled] = 1
    x2[inds] = 1
    assert np.array_equal(x.numpy(), x2.numpy())
    assert x.sum() == 9


def test_add_at_():
    # basic tests
    # 1d
    x = torch.zeros(10)
    inds = torch.LongTensor([1, 1, 1, 1, 2, 2, 2])
    add = torch.ones(7)
    spiketorch.add_at_(x, (inds,), add)
    assert x.sum() == 7
    assert x[0] == 0
    assert x[1] == 4
    assert x[2] == 3
    # 2d no broadcasting
    x = torch.zeros((10, 11))
    ii = torch.LongTensor([1, 1, 1, 1, 2, 2, 2, 0, 0, 0])
    jj = torch.LongTensor([1, 0, 0, 0, 2, 0, 0, 0, 0, 0])
    add = torch.ones(ii.shape)
    spiketorch.add_at_(x, (ii, jj), add)
    assert x.sum() == ii.numel()
    assert x[0, 0] == 3
    assert x[1, 0] == 3
    assert x[1, 1] == 1
    assert x[2, 2] == 1
    assert x[2, 0] == 2

    # closer to what we usually do with broadcasting
    recording = torch.zeros((32, 5))
    channel_index = torch.LongTensor(
        [[0, 1, 5], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
    )
    waveforms = torch.ones((3, 11, 3))
    max_channels = torch.LongTensor([0, 3, 3])
    times = torch.LongTensor([0, 10, 18])
    dts = torch.arange(11)
    time_ix = times[:, None, None] + dts[None, :, None]
    chan_ix = channel_index[max_channels][:, None, :]
    recording = F.pad(recording, (0, 1))
    spiketorch.add_at_(
        recording,
        (time_ix, chan_ix),
        waveforms,
    )
    assert recording.sum() == waveforms.sum()
    assert recording[2, 0] == 1
    assert recording[2, 4] == 0
    assert recording[10, 1] == 1
    assert recording[10, 2] == 1
    assert recording[10, 3] == 1
    assert recording[18, 3] == 2
    assert recording[19, 3] == 2
    assert recording[19, 1] == 0
    assert recording[21, 3] == 1


def test_add_spikes_():
    # dupe of add_at_ test
    recording = torch.zeros((32, 5))
    channel_index = torch.LongTensor(
        [[0, 1, 5], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
    )
    waveforms = torch.ones((3, 11, 3))
    max_channels = torch.LongTensor([0, 3, 3])
    times = torch.LongTensor([0, 10, 18])
    recording = spiketorch.add_spikes_(
        recording,
        times,
        max_channels,
        channel_index,
        waveforms,
        trough_offset=0,
        already_padded=False,
        pad_value=0.0,
        sign=-1,
    )
    assert -recording.sum() == waveforms.sum()
    assert -recording[2, 0] == 1
    assert -recording[2, 4] == 0
    assert -recording[10, 1] == 1
    assert -recording[10, 2] == 1
    assert -recording[10, 3] == 1
    assert -recording[18, 3] == 2
    assert -recording[19, 3] == 2
    assert -recording[19, 1] == 0
    assert -recording[21, 3] == 1
