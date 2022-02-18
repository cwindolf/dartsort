import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import nnls


def pairdists(waveforms, log=False, square=True):
    pd = pdist(waveforms.reshape(waveforms.shape[0], -1))
    if log:
        pd = np.log(pd)

    if square:
        return squareform(pd)
    else:
        return pd


def dim_scales_lsq(waveforms, features):
    n, t, c = waveforms.shape
    n_, k = features.shape
    assert n == n_
    orig_pd = pdist(
        waveforms.reshape(waveforms.shape[0], -1), metric="sqeuclidean"
    )
    f_pd = np.array(
        [pdist(f[:, None], metric="sqeuclidean") for f in features.T]
    )
    # return np.sqrt(np.abs(la.inv(f_pd @ f_pd.T) @ (f_pd @ orig_pd)))

    x, rnorm = nnls(f_pd.T, orig_pd)
    return np.sqrt(x)
