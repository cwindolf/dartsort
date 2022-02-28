from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm
from .waveform_utils import get_local_geom, relativize_waveforms

# (x_low, y_low, z_low, alpha_low), (x_high, y_high, z_high, alpha_high)
# BOUNDS_NP1 = (-100, 1e-4, -100, 0), (132, 250, 100, 20000)
# BOUNDS_NP2 = (-100, 1e-4, -100, 0), (132, 250, 100, 20000)
BOUNDS_NP1 = [(-100, 132), (1e-4, 250), (-100, 100)]
BOUNDS = {20: BOUNDS_NP1, 15: BOUNDS_NP1}

# how to initialize y, alpha?
Y0, ALPHA0 = 20.0, 1000.0


def check_shapes(waveforms, maxchans, geom, firstchans):
    N, T, C = waveforms.shape
    C_, d = geom.shape
    assert d == 2
    assert C <= C_
    assert (firstchans + C <= C_).all()
    assert (firstchans <= maxchans).all()
    assert (maxchans < firstchans + C).all()
    return N, T, C


def ptp_at(x, y, z, alpha, local_geom):
    return alpha / np.sqrt(
        np.square(x - local_geom[:, 0])
        + np.square(z - local_geom[:, 1])
        + np.square(y)
    )


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
        q = ptp_at(x, y, z, 1.0)
        alpha = (q * ptp / maxptp).sum() / (q * q).sum()
        return (
            np.square(ptp / maxptp - ptp_at(x, y, z, alpha)).mean()
            - np.log1p(10.0 * y) / 10000.0
        )

    result = minimize(
        mse,
        x0=[xcom, Y0, zcom],
        bounds=[(-100, 132), (1e-4, 250), (-100, 100)],
    )

    # print(result)
    bx, by, bz_rel = result.x
    q = ptp_at(bx, by, bz_rel, 1.0)
    balpha = (ptp * q).sum() / np.square(q).sum()
    return bx, by, bz_rel, geom[maxchan, 1] + bz_rel, balpha


def localize_ptps(
    ptps,
    geom,
    firstchans,
    maxchans,
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
    if pbar:
        N, _, C = check_shapes(
            ptps[:, None, :],
            maxchans,
            geom,
            firstchans,
        )
    else:
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
    alphas = np.empty(N)
    with Parallel(n_workers) as pool:
        for n, (x, y, z_rel, z_abs, alpha) in enumerate(
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
            alphas[n] = alpha

    return xs, ys, z_rels, z_abss, alphas


def localize_waveforms(
    waveforms,
    geom,
    firstchans,
    maxchans,
    n_workers=1,
    n_channels=None,
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
    # if not running with a progress bar, probably being called by
    # `localize_waveforms_batched`, so let's skip the shape checks
    # which have already been done.
    if pbar:
        N, T, C = check_shapes(waveforms, maxchans, geom, firstchans)
    else:
        N, T, C = waveforms.shape
    maxchans = maxchans.astype(int)

    if n_channels is not None and n_channels < C:
        waveforms, firstchans, maxchans, _ = relativize_waveforms(
            waveforms,
            firstchans,
            None,
            geom,
            maxchans_orig=None,
            feat_chans=n_channels,
        )

    ptps = waveforms.ptp(1)
    return localize_ptps(
        ptps,
        geom,
        firstchans,
        maxchans,
        n_workers=n_workers,
        pbar=pbar,
    )


def localize_waveforms_batched(
    waveforms,
    geom,
    firstchans,
    maxchans,
    n_workers=1,
    batch_size=128,
    n_channels=None,
):
    """A helper for running the above on hdf5 datasets or similar"""
    N, T, C = check_shapes(waveforms, maxchans, geom, firstchans)
    xs = np.empty(N)
    ys = np.empty(N)
    z_rels = np.empty(N)
    z_abss = np.empty(N)
    alphas = np.empty(N)

    starts = list(range(0, N, batch_size))
    ends = [min(start + batch_size, N) for start in starts]

    with Parallel(n_workers) as pool:
        for batch_idx, (x, y, z_rel, z_abs, alpha) in enumerate(
            pool(
                delayed(localize_waveforms)(
                    waveforms[start:end],
                    geom,
                    firstchans[start:end],
                    maxchans[start:end],
                    pbar=False,
                    n_channels=n_channels,
                )
                for start, end in tqdm(
                    zip(starts, ends), total=len(starts), desc="loc batches"
                )
            )
        ):
            start = starts[batch_idx]
            end = ends[batch_idx]
            xs[start:end] = x
            ys[start:end] = y
            z_rels[start:end] = z_rel
            z_abss[start:end] = z_abs
            alphas[start:end] = alpha

    return xs, ys, z_rels, z_abss, alphas
