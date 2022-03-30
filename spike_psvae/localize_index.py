"""Localization with channel subsetting based on channel index
"""
from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm

from .waveform_utils import channel_index_subset

# box constraint on optimization x, y, z (z relative to max chan)
BOUNDS = [(-100, 170), (1e-4, 250), (-100, 100)]

# how to initialize y?
Y0 = 20.0


def ptp_at(x, y, z, alpha, local_geom):
    return alpha / np.sqrt(
        np.square(x - local_geom[:, 0])
        + np.square(z - local_geom[:, 1])
        + np.square(y)
    )


def localize_ptp_index(ptp, local_geom):
    """Find the localization result for a single ptp vector

    Arguments
    ---------
    ptp : np.array (2 * channel_radius,)
    maxchan : int
    geom : np.array (total_channels, 2)

    Returns
    -------
    x, y, z_rel, z_abs, alpha
    """
    # initialize x, z with CoM
    good = np.flatnonzero(~np.isnan(ptp))
    ptp = ptp[good].astype(float)
    local_geom = local_geom[good].astype(float)
    ptp_p = ptp / ptp.sum()
    xcom, zcom = (ptp_p[:, None] * local_geom).sum(axis=0)
    maxptp = ptp.max()

    def ptp_at(x, y, z, alpha):
        return alpha / np.sqrt(
            np.square(x - local_geom[:, 0])
            + np.square(z - local_geom[:, 1])
            + np.square(y)
        )

    def mse(loc):
        x, y, z = loc
        q = ptp_at(x, y, z, 1.0)
        alpha = (q * ptp / maxptp).sum() / (q * q).sum()
        return (
            np.square(ptp / maxptp - ptp_at(x, y, z, alpha)).mean()
            - np.log1p(10.0 * y) / 10000.0
        )

    result = minimize(
        mse,
        x0=[xcom, Y0, zcom],
        bounds=BOUNDS,
    )

    # print(result)
    bx, by, bz_rel = result.x
    q = ptp_at(bx, by, bz_rel, 1.0)
    balpha = (ptp * q).sum() / np.square(q).sum()
    return bx, by, bz_rel, balpha


def localize_ptps_index(
    ptps,
    geom,
    maxchans,
    channel_index,
    n_channels=None,
    radius=100,
    n_workers=None,
    pbar=True,
):
    """Localize a bunch of waveforms

    waveforms is N x T x C, where C can be either 2 * channel_radius, or
    C can be 384 (or geom.shape[0]). If the latter, the 2 * channel_radius
    bits will be extracted.

    Returns
    -------
    xs, ys, z_rels, z_abss, alphas
    """
    N, C = ptps.shape
    maxchans = maxchans.astype(int)

    local_geoms = np.pad(geom, [(0, 1), (0, 0)])[channel_index[maxchans]]
    local_geoms[:, :, 1] -= geom[maxchans, 1][:, None]
    subset = channel_index_subset(
        geom, channel_index, n_channels=n_channels, radius=radius
    )

    # handle pbars
    xqdm = tqdm if pbar else lambda a, total, desc: a

    # -- run the least squares
    xs = np.empty(N)
    ys = np.empty(N)
    z_rels = np.empty(N)
    alphas = np.empty(N)
    with Parallel(n_workers) as pool:
        for n, (x, y, z_rel, alpha) in enumerate(
            pool(
                delayed(localize_ptp_index)(
                    ptp[subset[mc]],
                    local_geom[subset[mc]],
                )
                for ptp, mc, local_geom in xqdm(
                    zip(ptps, maxchans, local_geoms), total=N, desc="lsq"
                )
            )
        ):
            xs[n] = x
            ys[n] = y
            z_rels[n] = z_rel
            alphas[n] = alpha

    z_abss = z_rels + geom[maxchans, 1]
    return xs, ys, z_rels, z_abss, alphas
