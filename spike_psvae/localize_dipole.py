import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from tqdm.auto import tqdm
from .waveform_utils import get_local_geom

BOUNDS_NP1 = [(-100, 132), (1e-4, 250), (-100, 100)]
BOUNDS = {20: BOUNDS_NP1, 15: BOUNDS_NP1}

# how to initialize y, alpha?
Y0, ALPHA0 = 20.0, 1000.0


def localize_ptp(
    ptp,
    firstchan,
    maxchan,
    geom,
):
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
    n_channels = ptp.size
    local_geom, z_maxchan = get_local_geom(
        geom,
        firstchan,
        maxchan,
        n_channels,
        return_z_maxchan=True,
    )

    # initialize x, z with CoM
    ptp = ptp.astype(float)
    local_geom = local_geom.astype(float)
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
        # q = ptp_at(x, y, z, 1.0)
        # alpha = (q * (ptp / maxptp - delta)).sum() / (q * q).sum()
        duv = np.c_[
            x - local_geom[:, 0],
            np.broadcast_to(y, ptp.shape),
            z - local_geom[:, 1],
        ]
        X = duv / np.square(duv).sum(axis=1, keepdims=True)
        beta = np.linalg.solve(X.T @ X, X.T @ (ptp / maxptp))
        qtq = X @ beta
        return (
            np.square(ptp / maxptp - qtq).mean()
            # np.square(ptp / maxptp - delta - ptp_at(x, y, z, alpha)).mean()
            # np.square(np.maximum(0, ptp / maxptp - ptp_at(x, y, z, alpha))).mean()
            - np.log1p(10.0 * y) / 10000.0
        )

    result = minimize(
        mse,
        x0=[xcom, Y0, zcom],
        bounds=[(-150, 209), (1e-4, 500), (-100, 100)],
    )

    # print(result)
    bx, by, bz_rel = result.x
    # q = ptp_at(bx, by, bz_rel, 1.0)
    # balpha = (ptp * q).sum() / np.square(q).sum()
    duv = np.c_[
        bx - local_geom[:, 0],
        np.broadcast_to(by, ptp.shape),
        bz_rel - local_geom[:, 1],
    ]
    X = duv / np.square(duv).sum(axis=1, keepdims=True)
    beta = np.linalg.solve(X.T @ X, X.T @ ptp)
    # print(X)
    # print(beta)
    # print(X @ beta)
    return bx, by, bz_rel, geom[maxchan, 1] + bz_rel, beta, X @ beta


def localize_ptps(
    ptps,
    geom,
    firstchans,
    maxchans,
    n_workers=None,
    pbar=True,
    dipole=False,
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
    firstchans = firstchans.astype(int)

    # handle pbars
    xqdm = tqdm if pbar else lambda a, total, desc: a

    # -- run the least squares
    xs = np.empty(N)
    ys = np.empty(N)
    z_rels = np.empty(N)
    z_abss = np.empty(N)
    betas = np.empty((N, 3))
    ptp_fits = np.empty(ptps.shape)
    with Parallel(n_workers) as pool:
        for n, (x, y, z_rel, z_abs, beta, pred) in enumerate(
            pool(
                delayed(localize_ptp)(
                    ptp,
                    firstchan,
                    maxchan,
                    geom,
                )
                for ptp, maxchan, firstchan in xqdm(
                    zip(ptps, maxchans, firstchans), total=N, desc="lsq"
                )
            )
        ):
            xs[n] = x
            ys[n] = y
            z_rels[n] = z_rel
            z_abss[n] = z_abs
            betas[n] = beta
            ptp_fits[n] = pred

    return xs, ys, z_rels, z_abss, betas, ptp_fits
