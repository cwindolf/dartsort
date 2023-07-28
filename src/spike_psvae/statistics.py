import numpy as np
from scipy.stats import spearmanr  # noqa
from scipy.spatial import KDTree


def gcs(x, y):
    # no ties version
    assert x.ndim == y.ndim == 1
    assert x.size == y.size
    x = np.asarray(x)
    y = np.asarray(y)
    x_ranks = np.argsort(x)
    ri = np.argsort(y[x_ranks])
    return 1 - 3 * np.abs(ri[1:] - ri[:-1]).sum() / (x.size ** 2 - 1)


def gcsorig(Y, Z):
    if Z.shape[0] < 3:
        return 0
    if len(Z.shape) == 1:
        Z = Z[:, np.newaxis]
    n = Y.shape[0]
    R = np.argsort(Y)
    tree = KDTree(Z)
    _, ind = tree.query(Z, k=3)
    Ri = np.zeros(Y.shape[0])
    Ri[R] = np.arange(Y.shape[0]) + 1
    Li = n - Ri + 1
    return (n * np.minimum(Ri, Ri[ind[:, 1]]) - Li ** 2).sum() / (
        Li * (n - Li)
    ).sum()
