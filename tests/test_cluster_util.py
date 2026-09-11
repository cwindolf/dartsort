import numpy as np
import pytest

from dartsort.clustering.cluster_util import (
    _jitter_spread_violation_weights,
    decrumb_labels,
    recluster,
    reorder_by_depth,
    violation_statistics,
)
from dartsort.util.data_util import DARTsortSorting


@pytest.mark.parametrize("n_units", [0, 1, 2, 8])
def test_reorder_by_depth(n_units):
    # generate a sorting with n_units units where each unit has a specific
    # depth coordinate
    n_spikes_per_unit = 10
    n_spikes = n_units * n_spikes_per_unit

    times_samples = np.arange(n_spikes)
    channels = np.zeros(n_spikes, dtype=np.int64)
    labels = np.repeat(np.arange(n_units), n_spikes_per_unit)

    # centroids backwards
    centroids = np.arange(n_units, dtype=np.float64)[::-1]

    sorting = DARTsortSorting(
        times_samples=times_samples, channels=channels, labels=labels
    )

    reordered_sorting, reorder = reorder_by_depth(sorting, centroids=centroids)
    assert reordered_sorting.labels is not None

    # test that reorder[j] is the new label of unit j
    for j in range(n_units):
        in_j = labels == j
        assert np.all(reordered_sorting.labels[in_j] == reorder[j])

    # test that centroids are increasing in the reordered sorting
    new_centroids = np.empty(n_units)
    new_centroids[reorder] = centroids
    assert (np.diff(new_centroids) >= 0).all()


@pytest.mark.parametrize("in_place", [False, True])
def test_reorder_by_depth_remaps_candidates(in_place):
    # decreasing depths, so reverse
    labels = np.array([0, 0, 1, 1, 2, 2])
    centroids = np.array([2.0, 1.0, 0.0])
    candidates = np.array(
        [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, -1]], dtype=np.int64
    )
    responsibilities = np.full((6, 3), 1.0 / 3.0)
    log_liks = np.zeros((6, 3))
    log_liks[5, 1] = -np.inf

    sorting = DARTsortSorting(
        times_samples=np.arange(6, dtype=np.int64),
        channels=np.zeros(6, dtype=np.int64),
        labels=labels.copy(),
        ephemeral_features={
            "gmm_candidates": candidates.copy(),
            "gmm_responsibilities": responsibilities.copy(),
            "gmm_log_liks": log_liks.copy(),
        },
    )
    assert sorting.labels is not None

    out, reorder = reorder_by_depth(sorting, centroids=centroids, in_place=in_place)
    assert out.labels is not None
    assert np.array_equal(reorder, [2, 1, 0])

    expected = np.where(candidates >= 0, reorder[candidates], -1)
    assert np.array_equal(out.gmm_candidates, expected)
    assert np.array_equal(out.labels, reorder[labels])

    assert out.gmm_responsibilities[5, 1] == 0.0
    np.testing.assert_allclose(out.gmm_responsibilities[5, -1], 2.0 / 3.0)

    if not in_place:
        assert np.array_equal(sorting.gmm_candidates, candidates)
        assert np.array_equal(sorting.labels, labels)


def test_recluster():
    # basic check of recluster
    coords = np.array([0.0, 0.1, 1.0, 1.1, 2.0])
    n_units = len(coords)
    dists = np.abs(coords[:, None] - coords[None, :])

    sorting = DARTsortSorting(
        times_samples=np.arange(n_units),
        channels=np.zeros(n_units, dtype=np.int64),
        labels=np.arange(n_units),
    )

    new_sorting, new_ids = recluster(sorting=sorting, dists=dists, threshold=0.15)
    assert new_sorting.labels is not None

    # check that new_ids[j] is the new label of unit j
    for j in range(n_units):
        assert np.all(new_sorting.labels[sorting.labels == j] == new_ids[j])

    # merges correspond to the 1d coordinate:
    # close pairs share a new label; distant units do not
    assert new_ids[0] == new_ids[1]  # distance 0.1 < threshold
    assert new_ids[2] == new_ids[3]  # distance 0.1 < threshold
    assert new_ids[0] != new_ids[2]
    assert new_ids[0] != new_ids[4]
    assert new_ids[2] != new_ids[4]


@pytest.mark.parametrize("n_units", [0, 1, 8])
def test_decrumb(n_units):
    min_size = 5
    # even-indexed units get 10 spikes (big enough); odd-indexed get 2 (crumbs)
    counts = np.where(np.arange(n_units) % 2 == 0, 10, 2)
    labels = np.repeat(np.arange(n_units), counts)

    new_labels = decrumb_labels(labels, min_size=min_size)

    # big units are relabeled 0, 1, 2, ... in original order
    for new_id, old_id in enumerate(range(0, n_units, 2)):
        assert np.all(new_labels[labels == old_id] == new_id)

    # small units are set to -1
    for old_id in range(1, n_units, 2):
        assert np.all(new_labels[labels == old_id] == -1)


def _viol_sorting(times_samples, labels):
    return DARTsortSorting(
        times_samples=np.asarray(times_samples, dtype=np.int64),
        channels=np.zeros(len(times_samples), dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        sampling_frequency=10_000.0,
    )


def _brute_force_viols(times, labels, n_units, censor_samples=0):
    viols = np.zeros((n_units, n_units), dtype=np.int64)
    viol_samples = 10  # using 10kHz above
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if labels[i] < 0 or labels[j] < 0:
                continue
            dt = abs(times[i] - times[j])
            # censorship is right-exclusive, violations are inclusive
            if dt < censor_samples or dt > viol_samples:
                continue
            viols[labels[i], labels[j]] += 1
            if labels[i] != labels[j]:
                viols[labels[j], labels[i]] += 1
    return viols


@pytest.mark.parametrize(
    "censor_ms, expected",
    [
        (0.0, [[1, 2], [2, 1]]),
        (0.8, [[1, 1], [1, 1]]),
        (0.9, [[0, 1], [1, 1]]),
        (1.0, [[0, 0], [0, 1]]),
        (1.1, [[0, 0], [0, 0]]),
        (2, [[0, 0], [0, 0]]),
    ],
)
def test_violation_counts_explicit(censor_ms, expected):
    times = [0, 0, 100, 108, 200, 209, 300, 310, 400, 405, 411]
    labels = [0, 1, 0, 0, 0, 1, 1, 1, 0, -1, 1]
    res = violation_statistics(
        _viol_sorting(times, labels),
        censor_ms=censor_ms,
        viol_ms=1.0,
        chance_method="none",
    )

    assert np.array_equal(res.unit_ids, [0, 1])
    assert np.array_equal(res.spike_counts, [5, 5])
    assert np.array_equal(res.viol_counts, expected)


@pytest.mark.parametrize("censor_ms", [0.0, 0.2])
def test_violation_counts_sametime(censor_ms):
    times = [0] * 7
    labels = [0, 0, 1, 1, 1, 2, -1]
    res = violation_statistics(
        _viol_sorting(times, labels),
        censor_ms=censor_ms,
        viol_ms=1.0,
        chance_method="none",
    )

    assert np.array_equal(res.unit_ids, [0, 1, 2])
    assert np.array_equal(res.spike_counts, [2, 3, 1])
    if censor_ms:
        assert np.array_equal(res.viol_counts, [[0] * 3] * 3)
    else:
        assert np.array_equal(res.viol_counts, [[1, 6, 2], [6, 3, 3], [2, 3, 0]])


def _brute_force_weights(censor, viol, jitter, max_gap=None):
    if max_gap is None:
        max_gap = 2 * jitter + viol
    draws = np.arange(-jitter, jitter + 1)
    shifts = draws[:, None] - draws[None, :]
    gaps = np.arange(max_gap + 1)
    jittered = np.abs(gaps[:, None, None] + shifts[None, :, :])
    violates = (jittered >= censor) & (jittered <= viol)
    return violates.mean(axis=(1, 2))


@pytest.mark.parametrize("censor, viol, jitter", [(0, 1, 3), (1, 2, 4), (0, 0, 2)])
def test_jitter_weights(censor, viol, jitter):
    weights = _jitter_spread_violation_weights(censor, viol, jitter)
    np.testing.assert_allclose(weights, _brute_force_weights(censor, viol, jitter))

    # no pair past the end of the table can reach the window at all
    assert weights.size == 2 * jitter + viol + 1
    wider = _brute_force_weights(censor, viol, jitter, max_gap=4 * jitter + viol)
    assert not wider[weights.size :].any()


@pytest.mark.parametrize("censor_ms", [0.0, 0.3])
def test_violation_chance_matches_observed(censor_ms):
    # the observed half of the result has to agree with chance_method="none"
    rg = np.random.default_rng(0)
    times = np.sort(rg.integers(0, 100_000, size=400))
    labels = rg.integers(-1, 3, size=400)
    st = _viol_sorting(times, labels)

    res = violation_statistics(
        st, censor_ms=censor_ms, viol_ms=1.0, jitter_ms=20.0, n_resamples=2
    )
    expected = violation_statistics(
        st, censor_ms=censor_ms, viol_ms=1.0, chance_method="none"
    )

    assert np.array_equal(res.viol_counts, expected.viol_counts)
    assert np.array_equal(res.unit_ids, expected.unit_ids)
    assert np.array_equal(res.spike_counts, expected.spike_counts)


@pytest.mark.parametrize("censor_ms", [0.0, 0.3])
def test_violation_chance_montecarlo(censor_ms):
    n_resamples = 200
    rg = np.random.default_rng(1)
    times = np.sort(rg.integers(0, 10_000, size=120))
    labels = rg.integers(0, 3, size=120)
    st = _viol_sorting(times, labels)

    res = violation_statistics(
        st,
        censor_ms=censor_ms,
        viol_ms=1.0,
        jitter_ms=20.0,
        chance_method="resample",
        n_resamples=n_resamples,
        rg=7,
    )
    assert res.jitter_counts is not None
    assert res.jitter_counts.max() > 1.0

    exact = violation_statistics(
        st, censor_ms=censor_ms, viol_ms=1.0, jitter_ms=20.0, chance_method="weighted"
    )
    assert exact.n_resamples == 0
    assert exact.jitter_counts is not None
    assert np.array_equal(exact.viol_counts, res.viol_counts)
    atol = 6 * np.sqrt(exact.jitter_counts.max() / n_resamples)
    assert np.allclose(res.jitter_counts, exact.jitter_counts, rtol=0, atol=atol)


def test_violation_chance_deterministic():
    rg = np.random.default_rng(2)
    times = np.sort(rg.integers(0, 50_000, size=200))
    labels = rg.integers(0, 2, size=200)
    st = _viol_sorting(times, labels)

    kw = dict(
        viol_ms=1.0, jitter_ms=20.0, chance_method="resample", n_resamples=8, rg=3
    )
    a = violation_statistics(st, **kw)  # ty: ignore[invalid-argument-type]
    b = violation_statistics(st, **kw)  # ty: ignore[invalid-argument-type]
    assert a.jitter_counts is not None
    assert b.jitter_counts is not None
    assert np.array_equal(a.jitter_counts, b.jitter_counts)

    c = violation_statistics(st, **(kw | dict(rg=4)))  # ty: ignore[invalid-argument-type]
    assert c.jitter_counts is not None
    assert not np.array_equal(a.jitter_counts, c.jitter_counts)


def _refractory_train(rg, mean_gap, duration, refractory):
    n = int(2 * duration / mean_gap)
    gaps = refractory + rg.exponential(mean_gap - refractory, size=n)
    t = np.cumsum(gaps).astype(np.int64)
    return t[t < duration]


def test_jitter_viol_ratio_blanks():
    st = _viol_sorting([0, 5, 100, 105], [0, 0, 1, 1])
    res = violation_statistics(st, censor_ms=0.0, viol_ms=1.0, jitter_ms=20.0)
    # to feew to reach 4.6 expected
    assert np.isnan(res.jitter_viol_ratio()).all()
    assert not np.isnan(res.jitter_viol_ratio(min_jitter=0.0)).all()


@pytest.mark.parametrize("censor_ms, viol_ms", [(0.25, 1.0), (1.0, 0.5)])
def test_violation_chance_degenerate(censor_ms, viol_ms):
    empty = violation_statistics(
        _viol_sorting([], []), censor_ms=censor_ms, viol_ms=viol_ms, jitter_ms=20.0
    )
    assert empty.viol_counts.shape == (0, 0)
    assert (empty.jitter_counts is None) or (empty.jitter_counts.shape == (0, 0))

    st = _viol_sorting([0, 5, 10, 15], [0, 1, 0, 1])
    res = violation_statistics(
        st, censor_ms=censor_ms, viol_ms=viol_ms, jitter_ms=20.0, n_resamples=4
    )
    assert res.jitter_counts is not None
    assert res.viol_counts.shape == res.jitter_counts.shape == (2, 2)
    if viol_ms < censor_ms:
        assert not res.jitter_counts.any()


@pytest.mark.parametrize(
    "n_units, n_spikes, seed",
    [(0, 0, 0), (1, 1, 0), (1, 50, 1), (3, 200, 2), (6, 500, 3)],
)
@pytest.mark.parametrize("censor_ms", [0.0, 0.3, 1.0])
def test_violation_counts(n_units, n_spikes, seed, censor_ms):
    censor_samples = int(censor_ms * 10)  # 10 kHz
    rg = np.random.default_rng(seed)

    # should be lots of viols here
    times = rg.integers(0, 5 * n_spikes + 1, size=n_spikes)
    times.sort()

    # uniform shuffle labels
    labels = np.concatenate(
        (np.arange(n_units), rg.integers(-1, n_units, size=max(0, n_spikes - n_units)))
    )
    rg.shuffle(labels)

    sorting = _viol_sorting(times, labels)
    res = violation_statistics(
        sorting, censor_ms=censor_ms, viol_ms=1.0, chance_method="none"
    )

    assert res.viol_counts.shape == (n_units, n_units)
    assert np.array_equal(res.unit_ids, np.arange(n_units))
    assert np.array_equal(
        res.spike_counts, np.bincount(labels[labels >= 0], minlength=n_units)
    )
    assert np.array_equal(res.viol_counts, res.viol_counts.T)
    assert np.array_equal(
        res.viol_counts,
        _brute_force_viols(times, labels, n_units, censor_samples=censor_samples),
    )

    # unsorted spike times don't change result
    order = rg.permutation(n_spikes)
    shuffled = violation_statistics(
        _viol_sorting(times[order], labels[order]),
        censor_ms=censor_ms,
        viol_ms=1.0,
        chance_method="none",
    )
    assert np.array_equal(shuffled.viol_counts, res.viol_counts)

    # a window of 0ms only counts simultaneous spikes
    zero = violation_statistics(
        sorting, censor_ms=0.0, viol_ms=0.0, chance_method="none"
    )
    same_time = np.zeros((n_units, n_units), dtype=np.int64)
    for t in np.unique(times):
        in_t = labels[times == t]
        in_t = in_t[in_t >= 0]
        for a in range(len(in_t)):
            for b in range(a + 1, len(in_t)):
                same_time[in_t[a], in_t[b]] += 1
                if in_t[a] != in_t[b]:
                    same_time[in_t[b], in_t[a]] += 1
    assert np.array_equal(zero.viol_counts, same_time)
