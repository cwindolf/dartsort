import numpy as np
import pytest
import torch

from dartsort.cluster import kmeans


@pytest.fixture
def blobs():
    """Any kmeans should cluster this perfectly."""
    snr = 10
    K = 3
    n = 128

    rg = np.random.default_rng(0)
    centroids = rg.uniform(-snr, snr, size=(K, 2))
    order = np.lexsort(centroids.T)
    centroids = centroids[order]
    X = np.repeat(centroids, n, axis=0)
    labels = np.repeat(np.arange(K), n)
    X += rg.normal(size=X.shape)

    X = torch.asarray(X, dtype=torch.float)
    centroids = torch.asarray(centroids, dtype=torch.float)

    return dict(K=K, X=X, centroids=centroids, labels=labels)


@pytest.mark.parametrize(
    "algorithm",
    [kmeans.kmeans, kmeans.truncated_kmeans],
)
def test_kmeans(blobs, algorithm):
    res = algorithm(blobs["X"], n_components=blobs["K"])
    order = np.lexsort(np.asarray(res["centroids"]).T)
    labels = np.argsort(order)[np.asarray(res["labels"])]

    assert np.allclose(res["centroids"][order], blobs["centroids"], atol=0.25)
    assert np.array_equal(labels, blobs["labels"])
