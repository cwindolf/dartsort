import numpy as np
import pytest
import torch
from dartsort.detect.detect import detect_and_deduplicate
from dartsort.detect.detect_filters import convexity_filter
from dartsort.util.waveform_util import make_channel_index
from test_util import dense_layout


@pytest.mark.parametrize("dedup_exact", [False, True])
def test_detect_and_deduplicate(dedup_exact):
    h = dense_layout()
    g = np.c_[h["x"], h["y"]]
    ci = make_channel_index(g, 75, to_torch=True)

    if dedup_exact:
        carange = torch.arange(len(ci))
        jj, dedup_index_inds = (ci == carange[:, None]).nonzero(as_tuple=True)
        assert torch.equal(jj, carange)
        cvals = ci.take_along_dim(dedup_index_inds[:, None], dim=1)[:, 0]
        assert torch.equal(carange, cvals)
    else:
        dedup_index_inds = None

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
        remove_exact_duplicates=dedup_exact,
        dedup_index_inds=dedup_index_inds,
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
        remove_exact_duplicates=dedup_exact,
        dedup_index_inds=dedup_index_inds,
    )
    assert np.array_equal(times.numpy(), times2.numpy())
    assert np.array_equal(chans.numpy(), chans2.numpy())

    times, chans = detect_and_deduplicate(
        torch.tensor(x),
        threshold=threshold,
        peak_sign="neg",
        dedup_channel_index=ci,
        dedup_temporal_radius=dedup_t,
        remove_exact_duplicates=dedup_exact,
        dedup_index_inds=dedup_index_inds,
    )
    assert times.numel() == chans.numel() == 0


T = 100
C = 5
t0 = T // 2
eps = 2**-12

# these edge cases will be injected into a zeros recording with
# geometry such that consecutive channels are neighbors for dedup
# the threshold will be 1.0.

dedup_channel_index = [[0, 1, 4], [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0]]
dedup_cinds = [0, 1, 1, 1, 1]
threshold = 1.0
peak_dt = 3
dedup_dt = 7

dt0 = dedup_dt // 2
dt1 = dedup_dt
empty = [[], [], []]
v0 = threshold + eps
v1 = 2 * v0

# fmt: off
detect_edgecases = {
    # we don't detect nothing
    "blank": 2 * empty,
    # we don't detect an event at the threshold
    "single": [
        [T // 2], [1], [threshold],
        *empty,
    ],
    # we do detect an event above the threshold
    "single": 2 * [[T // 2], [1], [v0]],
    # we don't detect on the boundary
    "left": [[0], [1], [1.0], *empty],
    "right": [[T - 1], [1], [v0], *empty],
    "both": [[0, T - 1], [C - 1, C - 1], [v0, v0], *empty],
    "multileft": [[0, 0, 0], [0, 1, 2], [v0, v0, v0], *empty],
    # we detect the higher peak on the same channel
    "firsttime": [
        [t0, t0 + dt0], [1, 1], [v1, v0],
        [t0], [1], [v1],
    ],
    # we detect the higher peak at the same time
    "firstchan": [
        [t0, t0], [1, 2], [v1, v0],
        [t0], [1], [v1],
    ],
    # we detect the higher peak nearby
    "firstboth": [
        [t0, t0 + dt0], [1, 2], [v1, v0],
        [t0], [1], [v1],
    ],
    # same but bigger offset
    "firsttime1": [
        [t0, t0 + dt1], [1, 1], [v1, v0],
        [t0], [1], [v1],
    ],
    "firstchan1": [
        [t0, t0], [1, 2], [v1, v0],
        [t0], [1], [v1],
    ],
    "firstboth1": [
        [t0, t0 + dt1], [1, 2], [v1, v0],
        [t0], [1], [v1],
    ],
    # same as above, but the second was larger
    "secondtime": [
        [t0, t0 + dt0], [1, 1], [v0, v1],
        [t0 + dt0], [1], [v1],
    ],
    "secondchan": [
        [t0, t0], [1, 2], [v0, v1],
        [t0], [2], [v1],
    ],
    "secondboth": [
        [t0, t0 + dt1], [1, 2], [v0, v1],
        [t0 + dt1], [2], [v1],
    ],
    "secondtime1": [
        [t0, t0 + dt1], [1, 1], [v0, v1],
        [t0 + dt1], [1], [v1],
    ],
    "secondchan1": [
        [t0, t0], [1, 2], [v0, v1],
        [t0], [2], [v1],
    ],
    "secondboth1": [
        [t0, t0 + dt1], [1, 2], [v0, v1],
        [t0 + dt1], [2], [v1],
    ],
    # if the values are identical, we detect only the first (in time and space)
    "samevaltime": [
        [t0, t0 + dt0], [1, 1], [v0, v0],
        [t0], [1], [v0],
    ],
    "samevalchan": [
        [t0, t0], [1, 2], [v0, v0],
        [t0], [1], [v0],
    ],
    "samevalboth": [
        [t0, t0 + dt0], [1, 2], [v0, v0],
        [t0], [1], [v0],
    ],
    # space is prioritized over time as far as first-ness goes in dedup
    # maybe this is not ideal, but it's an edge case.
    "samevalrevchan0": [
        [t0, t0 + dt0], [2, 1], [v0, v0],
        [t0 + dt0], [1], [v0],
    ],
    "samevalrevchan1": [
        [t0, t0 + dt1], [2, 1], [v0, v0],
        [t0 + dt1], [1], [v0],
    ],
    "samevaltime1": [
        [t0, t0 + dt1], [1, 1], [v0, v0],
        [t0], [1], [v0],
    ],
    "samevalchan1": [
        [t0, t0], [1, 2], [v0, v0],
        [t0], [1], [v0],
    ],
    "samevalboth1": [
        [t0, t0 + dt1], [1, 2], [v0, v0],
        [t0], [1], [v0],
    ],
    "samevalrevchan2": [
        [t0, t0 + dt1], [2, 1], [v1, v1],
        [t0 + dt1], [1], [v1],
    ],
    # channels too far apart for dedup
    "samevaltime": 2 * [[t0, t0], [1, 3], [v0, v0]],
    "difvaltime": 2 * [[t0, t0], [1, 3], [v0, v1]],
    "samevaltime0": 2 * [[t0, t0 + dt0], [1, 3], [v0, v0]],
    "difvaltime0": 2 * [[t0, t0 + dt0], [1, 3], [v0, v1]],
    "samevaltime1": 2 * [[t0, t0 + dt1], [1, 3], [v0, v0]],
    "difvaltime1": 2 * [[t0, t0 + dt1], [1, 3], [v0, v1]],
    # some cases with 3 spikes
    # all same time, center biggest
    "sametime3center": [
        [t0, t0, t0], [1, 2, 3], [v0, v1, v0],
        [t0], [2], [v1],
    ],
    # all same time, first biggest
    "sametime3first": [
        [t0, t0, t0], [1, 2, 3], [v1, v0, v0],
        [t0], [1], [v1],
    ],
    # all same time, last biggest
    # the one above dedups because 2 gets chosen over 3 and
    # then beaten by 1. here, 1 gets chosen over 2, and 3 was
    # a winner.
    "sametime3last": [
        [t0, t0, t0], [1, 2, 3], [v0, v0, v1],
        [t0, t0], [1, 3], [v0, v1],
    ],
}
# fmt: on
emptyi = torch.tensor([], dtype=torch.long)
emptyf = torch.tensor([], dtype=torch.float)
dtps = 2 * [torch.long, torch.long, torch.float]


@pytest.mark.parametrize("convexity_threshold", [None, 0.0])
@pytest.mark.parametrize("peak_sign", ["pos", "neg", "both"])
@pytest.mark.parametrize("case", detect_edgecases.keys())
def test_detect_edgecases(case, peak_sign, convexity_threshold):
    case_tcv = [
        torch.tensor(x, dtype=dtp) for x, dtp in zip(detect_edgecases[case], dtps)
    ]
    # run on gpu if pos
    if torch.cuda.is_available():
        case_tcv = [x.cuda() for x in case_tcv]
        dev = "cuda"
    else:
        dev = "cpu"

    in_t, in_c, in_v = case_tcv[:3]
    targ_t, targ_c, targ_v = case_tcv[3:]

    traces = torch.zeros((T, C), dtype=torch.float, device=dev)
    traces[in_t, in_c] = in_v
    res = detect_and_deduplicate(
        traces,
        threshold=threshold,
        peak_sign=peak_sign,
        dedup_channel_index=torch.tensor(dedup_channel_index, device=dev),
        dedup_index_inds=torch.tensor(dedup_cinds, device=dev),
        dedup_temporal_radius=dedup_dt,
        relative_peak_radius=peak_dt,
        return_energies=True,
        remove_exact_duplicates=True,
    )
    assert len(res) == 3
    out_t, out_c, out_v = res
    keep = convexity_filter(traces, out_t, out_c, threshold=convexity_threshold)
    out_t = out_t[keep]
    out_c = out_c[keep]
    out_v = out_v[keep]
    if peak_sign == "neg":
        assert torch.equal(out_t, emptyi)
        assert torch.equal(out_c, emptyi)
        assert torch.equal(out_v, emptyf)
    else:
        assert out_t.numel() == targ_t.numel()
        assert out_t.numel() == out_c.numel() == out_v.numel()

        assert torch.equal(out_t, targ_t)
        assert torch.equal(out_c, targ_c)
        assert torch.equal(out_v, targ_v)
