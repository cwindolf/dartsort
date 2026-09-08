import numpy as np
import pytest

from dartsort.clustering.agglomerate import (
    clean_final_sorting,
    combine_gmm_scores,
    deduplicate_spikes,
)
from dartsort.util.data_util import DARTsortSorting
from dartsort.util.motion import MotionInfo

STATIC_MOTION = MotionInfo.static(np.c_[np.zeros(8), np.arange(8) * 10.0])


@pytest.fixture
def sorting():
    TIMES = np.array([1000, 1100, 1200, 2000, 2000, 2100, 3000, 3009, 3100])
    LABELS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    SCORES = np.array([2, 2, 2, 2, 0, 2, 2, 1, 2])
    return DARTsortSorting(
        times_samples=TIMES,
        channels=np.zeros_like(TIMES),
        labels=LABELS,
        sampling_frequency=30_000.0,
        ephemeral_features={"scores": SCORES},
    )


@pytest.mark.parametrize("radius_ms", [-1.0, 0.0, 0.3])
def test_deduplicate_spikes(sorting, radius_ms):
    scores = sorting.scores
    original_labels = sorting.labels.copy()
    out = deduplicate_spikes(sorting, radius_ms=radius_ms).labels
    assert out is not None
    if radius_ms < 0:
        np.testing.assert_array_equal(out, original_labels)
        return
    assert np.all(out[scores == 0] == -1)
    assert np.all(out[scores == 2] != -1)
    if radius_ms >= 0.3:
        assert np.all(out[scores == 1] == -1)
    else:
        assert np.all(out[scores == 1] != -1)


@pytest.mark.parametrize("radius_ms", [0.0, 0.3])
@pytest.mark.parametrize("shuffle_times", [False, True])
def test_deduplicate_spikes_in_place(sorting, radius_ms, shuffle_times):
    if shuffle_times:
        order = np.array([4, 0, 8, 2, 6, 1, 7, 3, 5])
        sorting = DARTsortSorting(
            times_samples=sorting.times_samples[order],
            channels=sorting.channels[order],
            labels=sorting.labels[order],
            sampling_frequency=sorting.sampling_frequency,
            ephemeral_features={"scores": sorting.scores[order]},
        )
    assert sorting.labels is not None
    original_labels = sorting.labels.copy()

    copied = deduplicate_spikes(sorting, radius_ms=radius_ms, in_place=False)
    np.testing.assert_array_equal(sorting.labels, original_labels)

    mutated = deduplicate_spikes(sorting, radius_ms=radius_ms, in_place=True)
    np.testing.assert_array_equal(mutated.labels, copied.labels)
    np.testing.assert_array_equal(sorting.labels, copied.labels)
    assert mutated.labels is sorting.labels


# -- combine_gmm_scores


def _random_merge_mapping(rg, n_units, n_groups):
    groups = np.concatenate(
        (np.arange(n_groups), rg.integers(0, n_groups, size=n_units - n_groups))
    )
    rg.shuffle(groups)
    # relabel groups in order of first appearance so ids are contiguous
    _, new_ids = np.unique(groups, return_inverse=True)
    return new_ids.astype(np.int64)


def _random_gmm_sorting(rg, n_spikes, n_units, n_cand, new_ids, tie_grid=None):
    cand = np.full((n_spikes, n_cand), -1, dtype=np.int64)
    logliks = np.full((n_spikes, n_cand + 1), -np.inf)
    for s in range(n_spikes):
        k = rg.integers(1, n_cand + 1)
        cand[s, :k] = rg.choice(n_units, size=k, replace=False)
        logliks[s, :k] = rg.normal(size=k)
    logliks[:, -1] = rg.normal(size=n_spikes)

    if tie_grid is not None:
        # produce ties by binning
        finite = np.isfinite(logliks)
        logliks[finite] = np.round(logliks[finite] / tie_grid) * tie_grid

    # likelihoods need to decrease
    order = np.argsort(-logliks[:, :n_cand], axis=1, kind="stable")
    cand = np.take_along_axis(cand, order, axis=1)
    logliks[:, :n_cand] = np.take_along_axis(logliks[:, :n_cand], order, axis=1)

    resp = np.exp(logliks - logliks.max(axis=1, keepdims=True))
    resp /= resp.sum(axis=1, keepdims=True)

    # labels already merged
    labels = np.where(logliks[:, 0] >= logliks[:, -1], new_ids[cand[:, 0]], -1)

    return DARTsortSorting(
        times_samples=np.arange(n_spikes, dtype=np.int64),
        channels=np.zeros(n_spikes, dtype=np.int64),
        labels=labels,
        sampling_frequency=30_000.0,
        ephemeral_features={
            "gmm_candidates": cand,
            "gmm_responsibilities": resp,
            "gmm_log_liks": logliks,
        },
    )


def _reference_combine(cand, resp, logliks, new_ids):
    n_spikes, n_cand = cand.shape
    n_new = new_ids.max() + 1
    cand = cand.copy()
    resp = resp.copy()
    logliks = logliks.copy()

    for s in range(n_spikes):
        # remap into the merged unit ID space
        for j in range(n_cand):
            cand[s, j] = new_ids[cand[s, j]] if cand[s, j] >= 0 else -1

        # deduplicate candidates
        for j in range(n_cand):
            if cand[s, j] < 0:
                continue
            for k in range(j + 1, n_cand):
                if cand[s, k] != cand[s, j]:
                    continue
                cand[s, k] = -1
                if logliks[s, k] == -np.inf:
                    continue
                resp[s, j] += resp[s, k]
                logliks[s, j] = np.logaddexp(logliks[s, j], logliks[s, k])
                resp[s, k] = 0.0
                logliks[s, k] = -np.inf

        # vacuum ghost prob
        for j in range(n_cand):
            if 0 <= cand[s, j] < n_new:
                continue
            resp[s, -1] += resp[s, j]
            resp[s, j] = 0.0

    order = np.argsort(-logliks[:, :n_cand], axis=1, kind="stable")
    cand = np.take_along_axis(cand, order, axis=1)
    resp[:, :n_cand] = np.take_along_axis(resp[:, :n_cand], order, axis=1)
    logliks[:, :n_cand] = np.take_along_axis(logliks[:, :n_cand], order, axis=1)

    labels = np.where(logliks[:, 0] >= logliks[:, -1], cand[:, 0], -1)
    return labels, cand, resp, logliks


@pytest.mark.parametrize(
    "n_spikes, n_units, n_groups, n_cand, seed",
    [
        (1, 2, 1, 1, 0),
        (50, 4, 2, 3, 1),
        (200, 8, 3, 5, 2),
        (200, 8, 8, 5, 3),  # identity mapping: nothing to merge
        (317, 11, 4, 2, 4),
    ],
)
@pytest.mark.parametrize("tie_grid", [None, 0.5])
@pytest.mark.parametrize("in_place", [False, True])
def test_combine_gmm_scores(
    n_spikes, n_units, n_groups, n_cand, seed, in_place, tie_grid
):
    rg = np.random.default_rng(seed)
    new_ids = _random_merge_mapping(rg, n_units, n_groups)
    sorting = _random_gmm_sorting(rg, n_spikes, n_units, n_cand, new_ids, tie_grid)

    orig = {
        k: getattr(sorting, k).copy()
        for k in ("labels", "gmm_candidates", "gmm_responsibilities", "gmm_log_liks")
    }
    exp_labels, exp_cand, exp_resp, exp_logliks = _reference_combine(
        orig["gmm_candidates"], orig["gmm_responsibilities"], orig["gmm_log_liks"],
        new_ids,
    )

    out = combine_gmm_scores(sorting, new_ids=new_ids, in_place=in_place)

    np.testing.assert_array_equal(out.labels, exp_labels)
    np.testing.assert_array_equal(out.gmm_candidates, exp_cand)
    np.testing.assert_allclose(out.gmm_responsibilities, exp_resp)
    np.testing.assert_allclose(out.gmm_log_liks, exp_logliks)

    np.testing.assert_allclose(out.gmm_responsibilities.sum(axis=1), 1.0)
    if n_cand > 1:
        assert np.diff(out.gmm_responsibilities[:, :n_cand], axis=1).max() <= 1e-3

    if in_place:
        assert out.labels is sorting.labels
        assert out.gmm_candidates is sorting.gmm_candidates
    else:
        for k, v in orig.items():
            np.testing.assert_array_equal(getattr(sorting, k), v)


def test_combine_gmm_scores_ties_and_neg_infs():
    # units 0 and 1 merge into 0; unit 2 becomes 1
    new_ids = np.array([0, 0, 1])
    cand = np.array(
        [
            [0, 1, 2, -1],  # 01 merge
            [2, 0, -1, -1],  # no merge, tie
            [2, 0, 1, -1],  # merge into a tie edge case
        ]
    )
    log2 = np.log(2.0)
    logliks = np.array(
        [
            [1.0, 1.0, 0.0, -np.inf, -5.0],
            [3.0, 3.0, -np.inf, -np.inf, -5.0],
            [log2, 0.0, 0.0, -np.inf, -5.0],
        ]
    )
    resp = np.exp(logliks - logliks.max(axis=1, keepdims=True))
    resp /= resp.sum(axis=1, keepdims=True)

    sorting = DARTsortSorting(
        times_samples=np.arange(3, dtype=np.int64),
        channels=np.zeros(3, dtype=np.int64),
        labels=np.array([0, 1, 1]),
        ephemeral_features={
            "gmm_candidates": cand,
            "gmm_responsibilities": resp,
            "gmm_log_liks": logliks,
        },
    )
    out = combine_gmm_scores(sorting, new_ids=new_ids, in_place=False)

    assert np.array_equal(
        out.gmm_candidates,
        [
            [0, 1, -1, -1],
            [1, 0, -1, -1],
            [1, 0, -1, -1],
        ],
    )
    np.testing.assert_allclose(
        out.gmm_log_liks,
        [
            [1.0 + log2, 0.0, -np.inf, -np.inf, -5.0],
            [3.0, 3.0, -np.inf, -np.inf, -5.0],
            [log2, log2, -np.inf, -np.inf, -5.0],
        ],
    )
    np.testing.assert_allclose(out.gmm_responsibilities.sum(axis=1), 1.0)
    assert np.diff(out.gmm_responsibilities[:, :4], axis=1).max() <= 1e-3
    assert out.labels is not None
    assert np.array_equal(out.labels, [0, 1, 1])


def test_combine_gmm_scores_no_scores():
    sorting = DARTsortSorting(
        times_samples=np.arange(4, dtype=np.int64),
        channels=np.zeros(4, dtype=np.int64),
        labels=np.array([0, 0, 1, 1]),
    )
    out = combine_gmm_scores(sorting, new_ids=np.array([0, 0]))
    assert out is sorting


# -- clean_final_sorting


def _clean_sorting(times, labels, scores, z):
    times = np.asarray(times, dtype=np.int64)
    xyz = np.zeros((len(times), 3))
    xyz[:, 2] = z
    return DARTsortSorting(
        times_samples=times,
        channels=np.zeros(len(times), dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        sampling_frequency=30_000.0,
        ephemeral_features={
            "scores": np.asarray(scores, dtype=np.float64),
            "point_source_localizations": xyz,
            "times_seconds": times / 30_000.0,
        },
    )


def test_clean_final_sorting():
    times = [0, 10, 1000, 1010, 2000, 2010, 3000, 3001]
    labels = [0, 0, 2, 2, 4, 4, 4, 4]
    scores = [1, 1, 1, 1, 1, 1, 1, 1]
    depths = [30.0, 30.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0]
    sorting = _clean_sorting(times, labels, scores, depths)

    out, mapping = clean_final_sorting(sorting, motion=STATIC_MOTION, dedup_ms=-1.0)
    assert out.labels is not None

    # gaps closed (0, 2, 4 -> three units) and ordered by increasing depth
    assert np.array_equal(np.unique(out.labels), [0, 1, 2])
    assert np.array_equal(out.labels, [2, 2, 0, 0, 1, 1, 1, 1])
    assert np.array_equal(mapping, [2, 0, 1])


def test_clean_final_sorting_dedup():
    # similar but with dedup applied
    times = [0, 1000, 2000, 3000, 3001]
    labels = [0, 0, 4, 4, 4]
    scores = [1.0, 1.0, 1.0, 5.0, 0.0]
    depths = [30.0, 30.0, 10.0, 10.0, 10.0]
    sorting = _clean_sorting(times, labels, scores, depths)

    out, mapping = clean_final_sorting(sorting, motion=STATIC_MOTION, dedup_ms=0.1)
    assert out.labels is not None

    assert np.array_equal(out.labels, [1, 1, 0, 0, -1])
    assert np.array_equal(mapping, [1, 0])


def test_clean_final_sorting_composes_merge_mapping():
    times = [0, 1000, 2000, 3000]
    labels = [0, 0, 1, 1]
    depths = [30.0, 30.0, 10.0, 10.0]
    sorting = _clean_sorting(times, labels, [1.0] * 4, depths)

    merge_mapping = np.array([0, 1, 1, 0])
    out, mapping = clean_final_sorting(
        sorting, motion=STATIC_MOTION, dedup_ms=-1.0, merge_mapping=merge_mapping
    )
    assert out.labels is not None

    assert np.array_equal(out.labels, [1, 1, 0, 0])
    assert np.array_equal(mapping, [1, 0, 0, 1])
