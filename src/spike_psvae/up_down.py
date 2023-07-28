import numpy as np
from .jisotonic5 import jisotonic5


def up_down_isotonic_regression(x, weights=None):
    x = x.astype(np.float64)
    if weights is None:
        weights = np.ones_like(x)

    # determine switch point
    y1_, mse1 = jisotonic5(x, weights)
    y2_, mse2r = jisotonic5(x[::-1].copy(), weights[::-1].copy())
    mse0 = mse1 + mse2r[::-1]
    best_ind = mse0.argmin()

    if best_ind == 0:
        return y2_[::-1]
    if best_ind == x.shape[0]:
        return y1_

    # regressions. note the negatives for decreasing.
    y1, _ = jisotonic5(x[:best_ind], weights[:best_ind])
    y2, _ = jisotonic5(-x[best_ind:], weights[best_ind:])
    y2 = -y2

    return np.hstack([y1, y2])


def central_up_down_isotonic_regression(x, weights=None):
    x = x.astype(np.float64)
    if weights is None:
        weights = np.ones_like(x)

    ind = x.shape[0] // 2 + 1

    # regressions. note the negatives for decreasing.
    y1, _ = jisotonic5(x[:ind], weights[:ind])
    y2, _ = jisotonic5(-x[ind:], weights[ind:])
    y2 = -y2

    return np.hstack([y1, y2])


def down_up_isotonic_regression(x, weights=None):
    return -up_down_isotonic_regression(-x, weights=weights)
