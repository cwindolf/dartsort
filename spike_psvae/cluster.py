import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize, Bounds


def pairdists(waveforms, log=False, square=True):
    pd = pdist(waveforms.reshape(waveforms.shape[0], -1))
    if log:
        pd = np.log(pd)

    if square:
        return squareform(pd)
    else:
        return pd


def dim_scales(waveforms, features):
    orig_pd = pdist(waveforms.reshape(waveforms.shape[0], -1))

    def obj(x):
        return np.sqrt(np.square(orig_pd - pdist(features * x)).mean())

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
    ress = [minimize(obj, rg.uniform(0.5, 5, size=features.shape[1]), bounds=bounds, method="Nelder-Mead", options=dict(maxiter=10000)) for _ in range(5)]
    ress = [res] + ress
    objs = [res.fun for res in ress]
    print("objectives", objs)
    return ress[np.argmin(objs)].x
