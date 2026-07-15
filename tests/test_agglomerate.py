import numpy as np
import pytest

from dartsort.clustering.agglomerate import deduplicate_spikes
from dartsort.util.data_util import DARTsortSorting


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
    out = deduplicate_spikes(sorting, radius_ms=radius_ms).labels
    assert out is not None
    if radius_ms < 0:
        np.testing.assert_array_equal(out, sorting.labels)
        return
    assert np.all(out[scores == 0] == -1)
    assert np.all(out[scores == 2] != -1)
    if radius_ms >= 0.3:
        assert np.all(out[scores == 1] == -1)
    else:
        assert np.all(out[scores == 1] != -1)
