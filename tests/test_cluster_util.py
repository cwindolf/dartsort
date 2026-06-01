import numpy as np
import pytest

from dartsort.clustering.cluster_util import reorder_by_depth
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
