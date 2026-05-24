import numpy as np
import pytest
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs
from sklearn.neighbors import KernelDensity

from dartsort.clustering.density import KmeansppBallTree, kdt_density

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


@pytest.mark.parametrize("branching", [2, 16])
@pytest.mark.parametrize("sort_0", [False, True])
@pytest.mark.parametrize("big", [False, True])
def test_ball_tree(blobs, blobs5k, big, sort_0, branching):
    if big:
        X, dens1, dens5, _ = blobs5k
    else:
        X, dens1, dens5, _ = blobs

    max_roots = 128
    if big:
        target_leafsize = 2 if branching == 2 else 128
        if branching == 2:
            max_roots = 4
    else:
        target_leafsize = 2 if branching == 2 else 128
        if branching == 2:
            max_roots = 2

    bt = KmeansppBallTree(
        X,
        sort_0=sort_0,
        device=None,
        max_roots=max_roots,
        target_branching=branching,
        target_leafsize=target_leafsize,
    )

    d1 = bt.sparse_distance_matrix(1e10).to_dense().sqrt_().numpy(force=True)

    bdens1, bdens5 = bt.gaussian_kdes(1.0, 5.0, max_sigma=100.0)

    d = X.shape[1]
    d0 = cdist(X, X)
    dens0 = np.exp(-(d0**2) / 2).mean(1) / (np.sqrt(2 * np.pi) ** d)
    np.testing.assert_allclose(dens0, dens1)
    np.testing.assert_allclose(np.diagonal(d1), 0.0, atol=1e-6)
    np.testing.assert_allclose(d1, d0, atol=1e-6)
    d11 = bt.sparse_distance_matrix(1e10).to_dense().sqrt_().numpy(force=True)
    np.testing.assert_allclose(d11, d0, atol=1e-6)

    np.testing.assert_allclose(bdens1.cpu(), dens1)
    np.testing.assert_allclose(bdens5.cpu(), dens5)
    if big and branching == 2:
        assert len(bt.layers) > 2  # make sure we test a deeper tree


@pytest.mark.parametrize("branching", [16, 2])
@pytest.mark.parametrize("sort_0", [False, True])
@pytest.mark.parametrize("big", [False, True])
def test_truncated_ratio(blobs, blobs5k, big, sort_0, branching):
    if big:
        X, dens1, dens5, ratio = blobs5k
    else:
        X, dens1, dens5, ratio = blobs
    kdt = KDTree(X)
    kratio = kdt_density(
        kdt, X, sigma=0.2, sigma_regional=1.0, max_sigma=1.0, n_threads=0
    )
    max_roots = 128
    if big:
        target_leafsize = 2 if branching == 2 else 128
    else:
        target_leafsize = 2 if branching == 2 else 128
        if branching == 2:
            max_roots = 2

    bt = KmeansppBallTree(
        X,
        sort_0=sort_0,
        device=None,
        max_roots=max_roots,
        target_branching=branching,
        target_leafsize=target_leafsize,
    )
    bdens1, bdens5 = bt.gaussian_kdes(0.2, 1.0, max_sigma=1.0)
    np.testing.assert_allclose((bdens1 / bdens5).cpu(), kratio)
