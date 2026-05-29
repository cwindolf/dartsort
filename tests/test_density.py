import numpy as np
import pytest
from scipy.spatial import KDTree
from sklearn.datasets import make_blobs
from sklearn.neighbors import KernelDensity

from dartsort.clustering.density import kdt_density, sort_density

_bw1 = 1.0  # np.sqrt(2.0)
_bw5 = 5.0


@pytest.fixture(scope="module")
def blobs():
    X, y = make_blobs(random_state=0, n_samples=10)
    kde1 = KernelDensity(bandwidth=_bw1).fit(X)
    kde5 = KernelDensity(bandwidth=_bw5).fit(X)
    logdens1 = kde1.score_samples(X)
    logdens5 = kde5.score_samples(X)
    dens1 = np.exp(logdens1)
    dens5 = np.exp(logdens5)
    return X, dens1, dens5, np.exp(logdens1 - logdens5)


@pytest.fixture(scope="module")
def blobs5k():
    X, y = make_blobs(random_state=0, n_samples=1000)
    kde1 = KernelDensity(bandwidth=_bw1).fit(X)
    kde5 = KernelDensity(bandwidth=_bw5).fit(X)
    logdens1 = kde1.score_samples(X)
    logdens5 = kde5.score_samples(X)
    dens1 = np.exp(logdens1)
    dens5 = np.exp(logdens5)
    return X, dens1, dens5, np.exp(logdens1 - logdens5)


@pytest.mark.parametrize("big", [False, True])
def test_kdt_density(blobs, blobs5k, big):
    if big:
        X, dens1, dens5, ratio = blobs5k
    else:
        X, dens1, dens5, ratio = blobs
    kdt = KDTree(X)
    kratio = kdt_density(
        kdt, X, sigma=1.0, sigma_regional=5.0, max_sigma=1e10, n_threads=0
    )
    np.testing.assert_allclose(kratio, ratio)


@pytest.mark.parametrize("big", [False, True])
def test_sort_density(blobs, blobs5k, big):
    if big:
        X, dens1, dens5, ratio = blobs5k
    else:
        X, dens1, dens5, ratio = blobs
    sratio = sort_density(X, sigma0=1.0, sigma1=5.0, max_sigma=1e10)
    np.testing.assert_allclose(sratio, ratio)


@pytest.mark.parametrize("big", [False, True])
def test_truncated_sort_ratio(blobs, blobs5k, big):
    if big:
        X, dens1, dens5, ratio = blobs5k
    else:
        X, dens1, dens5, ratio = blobs
    kdt = KDTree(X)
    kratio = kdt_density(
        kdt, X, sigma=0.2, sigma_regional=1.0, max_sigma=1.0, n_threads=0
    )
    sratio = sort_density(X, sigma0=0.2, sigma1=1.0, max_sigma=1.0)
    np.testing.assert_allclose(sratio, kratio)
