import itertools
from joblib import Parallel, delayed
import numpy as np
import multiprocessing
import h5py
from scipy.optimize import minimize
from tqdm.auto import tqdm
# from .waveform_utils import get_local_geom, relativize_waveforms, channel_index_subset

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


def localize_ptp_index(
    ptp,
    local_geom,
    logbarrier=True,
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
            - (np.log1p(10.0 * y) / 10000.0 if logbarrier else 0)
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
    return bx, by, bz_rel, balpha, balpha * q


def localize_ptps_index(
    ptps,
    geom,
    maxchans,
    channel_index,
    n_channels=None,
    radius=100,
    n_workers=None,
    pbar=True,
    logbarrier=True,
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
    ptp_fits = np.empty_like(ptps)
    with Parallel(n_workers) as pool:
        for n, (x, y, z_rel, alpha, pred) in enumerate(
            pool(
                delayed(localize_ptp_index)(
                    ptp[subset[mc]],
                    local_geom[subset[mc]],
                    logbarrier=logbarrier,
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
            ptp_fits[n] = pred

    z_abss = z_rels + geom[maxchans, 1]
    return xs, ys, z_rels, z_abss, alphas, ptp_fits


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
    return bx, by, bz_rel, geom[maxchan, 1] + bz_rel, balpha, balpha * q


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
    ptp_fits = np.empty_like(ptps)
    with Parallel(n_workers) as pool:
        for n, (x, y, z_rel, z_abs, alpha, pred) in enumerate(
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
            ptp_fits[n] = pred

    return xs, ys, z_rels, z_abss, alphas, ptp_fits


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
    jobs = (
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

    with Parallel(n_workers) as pool:
        for batch in grouper(10 * n_workers, jobs):
            for batch_idx, (x, y, zr, za, alpha) in enumerate(pool(jobs)):
                start = starts[batch_idx]
                end = ends[batch_idx]
                xs[start:end] = x
                ys[start:end] = y
                z_rels[start:end] = zr
                z_abss[start:end] = za
                alphas[start:end] = alpha

    return xs, ys, z_rels, z_abss, alphas


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def _loc_worker(start_end):
    start, end = start_end
    wfs = _loc_worker.wfs[start:end]
    fcs = _loc_worker.firstchans[start:end]
    mcs = _loc_worker.maxchans[start:end]

    ptps = wfs.ptp(1)
    maxptps = ptps.ptp(1)

    x, y, zr, za, alpha = localize_ptps(
        ptps,
        _loc_worker.geom,
        fcs,
        mcs,
        n_workers=1,
        pbar=False,
    )

    return start, end, maxptps, x, y, zr, za, alpha


def _proc_init(
    h5_path,
    geom_key,
    wfs_key,
    firstchans_key,
    spike_index_key,
):
    h5 = h5py.File(h5_path, "r")
    _loc_worker.geom = h5[geom_key]
    _loc_worker.wfs = h5[wfs_key]
    _loc_worker.firstchans = h5[firstchans_key][:]
    _loc_worker.maxchans = h5[spike_index_key][:, 1]


def localize_h5(
    h5_path,
    geom_key="geom",
    wfs_key="cleaned_waveforms",
    firstchans_key="first_channels",
    spike_index_key="spike_index",
    n_workers=1,
    batch_size=4096,
):
    """Localize and compute max ptp in parallel"""
    with h5py.File(h5_path, "r") as f:
        N = len(f[spike_index_key])

    maxptps = np.empty(N)
    xs = np.empty(N)
    ys = np.empty(N)
    z_rels = np.empty(N)
    z_abss = np.empty(N)
    alphas = np.empty(N)

    starts = list(range(0, N, batch_size))
    ends = [min(start + batch_size, N) for start in starts]

    with multiprocessing.Pool(
        n_workers,
        initializer=_proc_init,
        initargs=(
            h5_path,
            geom_key,
            wfs_key,
            firstchans_key,
            spike_index_key,
        ),
    ) as pool:
        for bs, be, maxptp, x, y, zr, za, alpha in tqdm(
            pool.imap(_loc_worker, zip(starts, ends)),
            desc="Localize batches",
            total=len(starts),
            smoothing=0,
        ):
            maxptps[bs:be] = maxptp
            xs[bs:be] = x
            ys[bs:be] = y
            z_rels[bs:be] = zr
            z_abss[bs:be] = za
            alphas[bs:be] = alpha

    return maxptps, xs, ys, z_rels, z_abss, alphas
