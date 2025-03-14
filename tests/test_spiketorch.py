import numpy as np
import torch
import torch.nn.functional as F
from dartsort.util import spiketorch
from scipy.signal import resample


def test_ptp():
    rg = np.random.default_rng(0)

    x = rg.normal(size=100)
    assert np.array_equal(spiketorch.ptp(torch.tensor(x), 0).numpy(), np.ptp(x))
    x = rg.normal(size=(100, 100))
    assert np.array_equal(spiketorch.ptp(torch.tensor(x), 0).numpy(), np.ptp(x, 0))
    assert np.array_equal(spiketorch.ptp(torch.tensor(x), 1).numpy(), np.ptp(x, 1))


def test_isin_sorted():
    x = torch.arange(5)
    y = torch.arange(10)
    for i in range(-30, 30):
        for j in range(-30, 30):
            assert torch.equal(
                torch.isin(x + i, y + j),
                spiketorch.isin_sorted(x + i, y + j),
            )
    x = x[[1, 2, 4]]
    y = y[::2]
    for i in range(-30, 30):
        for j in range(-30, 30):
            assert torch.equal(
                torch.isin(x + i, y + j),
                spiketorch.isin_sorted(x + i, y + j),
            )
    rg = np.random.default_rng()
    x = rg.integers(low=0, high=100, size=20)
    x = np.concatenate((np.zeros(5, dtype=int), x))
    x.sort()
    x = torch.from_numpy(x)
    y = rg.integers(low=0, high=20, size=40)
    y = np.concatenate((np.zeros(5, dtype=int), y))
    y.sort()
    y = torch.from_numpy(y)
    for i in range(-30, 30):
        for j in range(-30, 30):
            assert torch.equal(
                torch.isin(x + i, y + j),
                spiketorch.isin_sorted(x + i, y + j),
            )

    # edge cases with empties
    empty = torch.arange(0)
    full = torch.arange(5)
    empty_b = torch.zeros((0,), dtype=bool)
    full_b = torch.zeros((5,), dtype=bool)
    assert torch.equal(spiketorch.isin_sorted(empty, full), empty_b)
    assert torch.equal(spiketorch.isin_sorted(full, empty), full_b)


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
        torch.LongTensor([2, 31, 30])[:, None] + torch.tensor([-1, 0, 1])[None, :],
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


def test_resample():
    rg = np.random.default_rng(0)
    x = rg.normal(size=(10, 101, 3))
    xup_scipy = resample(x, 80)
    xup_torch = spiketorch.real_resample(torch.as_tensor(x), 80)
    assert np.isclose(xup_scipy, xup_torch).all()


def test_depthwise_oaconv1d():
    rg = np.random.default_rng(0)
    # ns=500 tests fallback, vanilla FFT correlation in torch
    # ns=5000 tests overlap-add implementation
    for ns in [500, 5000]:
        spike_length, trough_offset = 121, 42

        # fake templates in white noise
        t0 = np.exp(-(((np.arange(spike_length) - trough_offset) / 10) ** 2))
        t1 = np.exp(-(((np.arange(spike_length) - trough_offset) / 30) ** 2))

        templates = torch.Tensor(np.stack([t0, t1]))
        traces = 0.1 * torch.Tensor(rg.normal(size=(2, ns)))

        traces[0, 100 : 100 + spike_length] += templates[0]
        traces[1, 300 : 300 + spike_length] += templates[1]

        torch_conv = F.conv1d(
            traces[None, :, :],
            templates[:, None, :],
            groups=2,
        )

        oa_conv = spiketorch.depthwise_oaconv1d(traces, templates)

        assert np.allclose(torch_conv, oa_conv, atol=1e-5)
