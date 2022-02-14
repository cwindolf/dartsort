from itertools import repeat
from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import least_squares
from tqdm.auto import trange, tqdm

from .waveform_utils import get_local_geom, get_local_chans

# (x_low, y_low, z_low, alpha_low), (x_high, y_high, z_high, alpha_high)
BOUNDS_NP1 = (-150, 0, -150, 0), (209, 250, 150, 10000)
BOUNDS_NP2 = (-100, 0, -100, 0), (132, 250, 100, 10000)
BOUNDS = {20: BOUNDS_NP1, 15: BOUNDS_NP2}

# how to initialize y, alpha?
Y0, ALPHA0 = 21.0, 1000.0


def check_shapes(
    waveforms, maxchans, channel_radius, geom, firstchans, geomkind
):
    N, T, C = waveforms.shape
    C_, d = geom.shape
    assert d == 2

    if C == 2 * channel_radius:
        assert geomkind in ("updown", "firstchan")
        print(f"Waveforms are already trimmed to {C} channels.")
        if geomkind == "updown" and maxchans is None:
            # we will need maxchans later to determine local geometries
            raise ValueError(
                "maxchans can't be None when geomkind==updown and "
                "waveform channels < geom channels"
            )
        if geomkind == "firstchan" and firstchans is None:
            raise ValueError(
                "firstchans can't be None when geomkind==firstchan"
            )
    elif C == 2 * channel_radius + 2:
        assert geomkind in ("standard", "firstchanstandard")
        print(f"Waveforms are already trimmed to {C} channels.")
        if maxchans is None:
            # we will need maxchans later to determine local geometries
            raise ValueError(
                "maxchans can't be None when waveform channels < geom channels"
            )
        if geomkind == "firstchanstandard" and firstchans is None:
            raise ValueError(
                "firstchans can't be None when geomkind==firstchanstandard"
            )
    elif C == C_:
        print(
            f"Waveforms are on all {C} channels. "
            f"Trimming with radius {channel_radius}."
        )
    else:
        raise ValueError(
            f"Not sure what to do with waveforms.shape={waveforms.shape} "
            f"and channel_radius={channel_radius} in geomkind {geomkind}"
        )

    return N, T, C


def ptp_at(x, y, z, alpha, local_geom):
    return alpha / np.sqrt(
        np.square(x - local_geom[:, 0])
        + np.square(z - local_geom[:, 1])
        + np.square(y)
    )


def localize_ptp(
    ptp,
    maxchan,
    geom,
    jac=False,
    logbarrier=True,
    firstchan=None,
    geomkind="updown",
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
    if logbarrier:
        assert not jac

    channel_radius = ptp.shape[0] // 2 - ("standard" in geomkind)
    # local_geom is 2*channel_radius, 2
    local_geom, z_maxchan = get_local_geom(
        geom,
        maxchan,
        channel_radius,
        ptp,
        return_z_maxchan=True,
        firstchan=firstchan,
        geomkind=geomkind,
    )

    # initialize x, z with CoM
    ptp_p = ptp / ptp.sum()
    xcom, zcom = (ptp_p[:, None] * local_geom).sum(axis=0)
    maxptp = ptp.max()

    if logbarrier:
        # TODO: can we break this up over components like before?
        def residual(loc):
            phat = ptp_at(*loc, local_geom)
            logpenalty = np.log1p(np.log1p(maxptp * loc[1]) / 50.0)
            return (ptp - phat).sum() - logpenalty
    else:
        def residual(loc):
            return ptp - ptp_at(*loc, local_geom)

    jacobian = "2-point"
    if jac:

        def jacobian(loc):
            x, y, z, alpha = loc
            dxz = np.array(((x, z),)) - local_geom
            sq_dxz = np.square(dxz)
            sqdists = (y ** 2 + sq_dxz).sum(axis=1)
            d12 = np.sqrt(sqdists)
            inv_d32 = 1.0 / (sqdists * d12)
            ddx = alpha * dxz[:, 0] * inv_d32
            ddy = alpha * y * inv_d32
            ddz = alpha * dxz[:, 1] * inv_d32
            dda = -1.0 / d12
            return np.stack((ddx, ddy, ddz, dda), axis=1)

    result = least_squares(
        residual,
        jac=jacobian,
        x0=[xcom, Y0, zcom, ALPHA0],
        bounds=BOUNDS[int(geom[0, 2] - geom[0, 0])],
    )

    # convert to absolute positions
    x, y, z_rel, alpha = result.x
    z_abs = z_rel + z_maxchan

    return x, y, z_rel, z_abs, alpha


def localize_ptps(
    ptps,
    geom,
    maxchans,
    channel_radius=10,
    n_workers=None,
    jac=False,
    logbarrier=True,
    firstchans=None,
    geomkind="updown",
    _not_helper=True,
):
    """Localize a bunch of waveforms

    waveforms is N x T x C, where C can be either 2 * channel_radius, or
    C can be 384 (or geom.shape[0]). If the latter, the 2 * channel_radius
    bits will be extracted.

    Returns
    -------
    xs, ys, z_rels, z_abss, alphas
    """
    if logbarrier:
        assert not jac

    if _not_helper:
        N, _, C = check_shapes(
            ptps[:, None, :],
            maxchans,
            channel_radius,
            geom,
            firstchans,
            geomkind,
        )
    else:
        N, C = ptps.shape

    # I have them stored as floats and keep forgetting to int them.
    maxchans = maxchans.astype(int)
    if firstchans is None:
        firstchans = repeat(None)

    # handle pbars
    xqdm = tqdm if _not_helper else lambda a, total, desc: a

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
                    maxchan,
                    geom,
                    jac=jac,
                    firstchan=firstchan,
                    geomkind=geomkind,
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
    maxchans=None,
    channel_radius=10,
    n_workers=1,
    jac=False,
    logbarrier=True,
    firstchans=None,
    geomkind="updown",
    _not_helper=True,
):
    """Localize a bunch of waveforms

    waveforms is N x T x C, where C can be either 2 * channel_radius, or
    C can be 384 (or geom.shape[0]). If the latter, the 2 * channel_radius
    bits will be extracted.

    Returns
    -------
    xs, ys, z_rels, z_abss, alphas
    """
    if logbarrier:
        assert not jac

    if _not_helper:
        N, T, C = check_shapes(
            waveforms, maxchans, channel_radius, geom, firstchans, geomkind
        )
    else:
        N, T, C = waveforms.shape

    # I have them stored as floats and keep forgetting to int them.
    if maxchans is not None:
        maxchans = maxchans.astype(int)

    # handle pbars
    xrange = trange if _not_helper else lambda a, desc: range(a)
    xqdm = tqdm if _not_helper else lambda a, total, desc: a

    # -- get N x local_neighborhood_size array of PTPs
    if C in (2 * channel_radius, 2 * channel_radius + 2):
        ptps = waveforms.ptp(axis=1)
    else:
        # we need maxchans to extract the local waveform
        ptps_full = waveforms.ptp(axis=1)
        if maxchans is None:
            maxchans = np.argmax(ptps_full, axis=1)
            bad = np.flatnonzero(ptps_full.ptp(1) == 0)
            if bad:
                raise ValueError(f"Some waveforms were all zero: {bad}.")

        ptps = np.empty(
            (N, 2 * channel_radius + 2 * ("standard" in geomkind)),
            dtype=waveforms.dtype,
        )
        for n in xrange(N, desc="extracting channels"):
            low, high = get_local_chans(
                geom,
                maxchans[n],
                channel_radius,
                ptps_full[n],
                firstchan=firstchans[n],
                geomkind=geomkind,
            )
            ptps[n] = ptps_full[n, low:high]
        del ptps_full

    if firstchans is None:
        firstchans = repeat(None)

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
                    maxchan,
                    geom,
                    jac=jac,
                    firstchan=firstchan,
                    geomkind=geomkind,
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


def localize_waveforms_batched(
    waveforms,
    geom,
    maxchans=None,
    channel_radius=10,
    n_workers=1,
    jac=False,
    logbarrier=True,
    firstchans=None,
    geomkind="updown",
    batch_size=128,
):
    """A helper for running the above on hdf5 datasets or similar"""
    if logbarrier:
        assert not jac

    N, T, C = check_shapes(
        waveforms, maxchans, channel_radius, geom, firstchans, geomkind
    )
    xs = np.empty(N)
    ys = np.empty(N)
    z_rels = np.empty(N)
    z_abss = np.empty(N)
    alphas = np.empty(N)

    starts = list(range(0, N, batch_size))
    ends = [min(start + batch_size, N) for start in starts]

    def maxchan_batch(start, end):
        if maxchans is None:
            return None
        else:
            return maxchans[start:end]

    def firstchan_batch(start, end):
        if firstchans is None:
            return None
        else:
            return firstchans[start:end]

    with Parallel(n_workers) as pool:
        for batch_idx, (x, y, z_rel, z_abs, alpha) in enumerate(
            pool(
                delayed(localize_waveforms)(
                    waveforms[start:end],
                    geom,
                    maxchans=maxchan_batch(start, end),
                    channel_radius=channel_radius,
                    jac=jac,
                    geomkind=geomkind,
                    firstchans=firstchan_batch(start, end),
                    _not_helper=False,
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
