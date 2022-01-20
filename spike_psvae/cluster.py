import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize, Bounds, nnls
import scipy.linalg as la


def pairdists(waveforms, log=False, square=True):
    pd = pdist(waveforms.reshape(waveforms.shape[0], -1))
    if log:
        pd = np.log(pd)

    if square:
        return squareform(pd)
    else:
        return pd


def dim_scales(waveforms, features):
    n, t, c = waveforms.shape
    print("a")
    orig_pd = pdist(waveforms.reshape(waveforms.shape[0], -1))
    print("b")

    def obj(x):
        return np.sqrt(np.square(orig_pd - pdist(features * x)).mean())

    # tinds = np.triu_indices(n, k=1)
    # diffs = np.array(
    #     [
    #         np.square(features[:, i, None] - features.T[None, i, :])[:, tinds]
    #         for i in range(features.shape[1])
    #     ]
    # )
    # print(diffs.shape)

    # def obj(x):
    #     return np.sqrt(np.square(orig_pd - x @ diffs).mean())

    bounds = Bounds(
        np.full(features.shape[1], 0.01), np.full(features.shape[1], np.inf)
    )
    x0 = np.ones(features.shape[1])
    res = minimize(
        obj,
        x0,
        bounds=bounds,
        method="Nelder-Mead",
        options=dict(maxiter=10000),
    )
    # return res.x

    rg = np.random.default_rng(0)
    ress = [
        minimize(
            obj,
            rg.uniform(0.5, 5, size=features.shape[1]),
            bounds=bounds,
            method="Nelder-Mead",
            options=dict(maxiter=10000),
        )
        for _ in range(5)
    ]
    ress = [res] + ress
    objs = [res.fun for res in ress]
    print("objectives", objs)
    return ress[np.argmin(objs)].x


def dim_scales_lsq(waveforms, features):
    n, t, c = waveforms.shape
    orig_pd = pdist(
        waveforms.reshape(waveforms.shape[0], -1), metric="sqeuclidean"
    )
    f_pd = np.array(
        [pdist(f[:, None], metric="sqeuclidean") for f in features.T]
    )
    # return np.sqrt(np.abs(la.inv(f_pd @ f_pd.T) @ (f_pd @ orig_pd)))

    x, rnorm = nnls(f_pd.T, orig_pd)
    return np.sqrt(x)
