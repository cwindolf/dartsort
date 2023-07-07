import numpy as np
import torch
from dartsort.detect.detect import detect_and_deduplicate
from dartsort.util.waveform_util import make_channel_index
from neuropixel import dense_layout


def test_detect_and_deduplicate():
    h = dense_layout()
    g = np.c_[h["x"], h["y"]]
    ci = make_channel_index(g, 75)

    threshold = 10
    dedup_t = 7

    x = np.random.default_rng(0).normal(size=(30000, 384)).astype(np.float32)
    assert (x > threshold).sum() == 0
    times = np.array([100, 200, 300, 400, 500, 600, 700])
    chans = np.array([100, 200, 300, 100, 200, 300, 100])
    times2 = times + np.arange(4, 4 + len(times))
    x[times, chans] = 3 * threshold
    x[times2, chans] = 2 * threshold
    x[times + 10000, chans] = 3 * threshold
    x[times2 + 10000, chans + 1] = 2 * threshold

    desired_times = np.concatenate([times, times2[times2 - times > dedup_t]])
    desired_times = np.concatenate([desired_times, desired_times + 10000])
    desired_times.sort()

    times, chans = detect_and_deduplicate(
        torch.tensor(x),
        threshold=threshold,
        peak_sign="both",
        dedup_channel_index=ci,
        dedup_temporal_radius=dedup_t,
    )

    assert np.array_equal(times.numpy(), desired_times)
    assert times.shape == chans.shape
    assert len(times.shape) == 1

    times2, chans2 = detect_and_deduplicate(
        torch.tensor(x),
        threshold=threshold,
        peak_sign="pos",
        dedup_channel_index=ci,
        dedup_temporal_radius=dedup_t,
    )
    assert np.array_equal(times.numpy(), times2.numpy())
    assert np.array_equal(chans.numpy(), chans2.numpy())

    times, chans = detect_and_deduplicate(
        torch.tensor(x),
        threshold=threshold,
        peak_sign="neg",
        dedup_channel_index=ci,
        dedup_temporal_radius=dedup_t,
    )
    assert times.numel() == chans.numel() == 0
