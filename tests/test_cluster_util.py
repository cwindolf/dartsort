import numpy as np
import pytest

from dartsort.clustering.cluster_util import recluster, reorder_by_depth
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
