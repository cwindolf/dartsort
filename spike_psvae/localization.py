from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import least_squares
from tqdm.auto import trange, tqdm

from .waveform_utils import get_local_geom, get_local_chans

# (x_low, y_low, z_low, alpha_low), (x_high, y_high, z_high, alpha_high)
BOUNDS = (-100, 0, -100, 0), (132, 250, 100, 10000)
# how to initialize y, alpha?
Y0, ALPHA0 = 21.0, 1000.0


def check_shapes(waveforms, maxchans, channel_radius, geom, geomkind):
    N, T, C = waveforms.shape
    C_, d = geom.shape
    assert d == 2

    if C == 2 * channel_radius:
        assert geomkind == "updown"
        print(f"Waveforms are already trimmed to {C} channels.")
        if maxchans is None:
            # we will need maxchans later to determine local geometries
            raise ValueError(
                "maxchans can't be None when waveform channels < geom channels"
            )
    elif C == 2 * channel_radius + 2:
        assert geomkind == "standard"
        print(f"Waveforms are already trimmed to {C} channels.")
        if maxchans is None:
            # we will need maxchans later to determine local geometries
            raise ValueError(
                "maxchans can't be None when waveform channels < geom channels"
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


def localize_ptp(
    ptp, maxchan, geom, jac=False, return_z_rel=False, geomkind="updown"
):
    """Find the localization result for a single ptp vector

    ptp : np.array (2 * channel_radius,)
    maxchan : int
    geom : np.array (total_channels, 2)
    """
    channel_radius = ptp.shape[0] // 2 - (geomkind == "standard")
    # local_geom is 2*channel_radius, 2
    local_geom, z_maxchan = get_local_geom(
        geom,
        maxchan,
        channel_radius,
        ptp,
        return_z_maxchan=True,
        geomkind=geomkind,
    )

    # initialize x, z with CoM
    ptp_p = ptp / ptp.sum()
    xcom, zcom = (ptp_p[:, None] * local_geom).sum(axis=0)

    def residual(loc):
        x, y, z, alpha = loc
        sq_dxz = np.square(local_geom - np.array(((x, z),)))
        dists = np.sqrt((y ** 2 + sq_dxz).sum(axis=1))
        return ptp - alpha / dists

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
        bounds=BOUNDS,
    )

    # convert to absolute positions
    x, y, z_rel, alpha = result.x
    z_abs = z_rel + z_maxchan

    return x, y, z_rel, z_abs, alpha


def localize_waveforms(
    waveforms,
    geom,
    maxchans=None,
    channel_radius=10,
    n_workers=1,
    jac=False,
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
    if _not_helper:
        N, T, C = check_shapes(waveforms, maxchans, channel_radius, geom, geomkind)
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
            (N, 2 * channel_radius + 2 * (geomkind == "standard")),
            dtype=waveforms.dtype,
        )
        for n in xrange(N, desc="extracting channels"):
            low, high = get_local_chans(
                geom,
                maxchans[n],
                channel_radius,
                ptps_full[n],
                geomkind=geomkind,
            )
            ptps[n] = ptps_full[n, low:high]
        del ptps_full

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
                    ptp, maxchan, geom, jac=jac, geomkind=geomkind
                )
                for ptp, maxchan in xqdm(
                    zip(ptps, maxchans), total=N, desc="lsq"
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
    geomkind="updown",
    batch_size=128,
):
    """A helper for running the above on hdf5 datasets or similar"""
    N, T, C = check_shapes(waveforms, maxchans, channel_radius, geom, geomkind)
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
