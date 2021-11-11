from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import least_squares
from tqdm.auto import trange, tqdm

from .waveform_utils import get_local_geom

# (x_low, y_low, z_low, alpha_low), (x_high, y_high, z_high, alpha_high)
BOUNDS = (-100, 0, -100, 0), (132, 250, 100, 10000)
Y0, ALPHA0 = 21.0, 1000.0


def check_shapes(waveforms, maxchans, channel_radius, geom):
    N, T, C = waveforms.shape
    C_, d = geom.shape
    assert d == 2

    if C == 2 * channel_radius:
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
            f"Not sure what to do with waveforms.shape={waveforms.shape}."
        )

    return N, T, C


def localize_ptp(ptp, maxchan, geom):
    """Find the localization result for a single ptp vector

    ptp : np.array (2 * channel_radius,)
    maxchan : int
    geom : np.array (total_channels, 2)
    """
    channel_radius = ptp.shape[0] // 2
    # local_geom is 2*channel_radius, 2
    local_geom, z_maxchan = get_local_geom(
        geom, maxchan, channel_radius, return_z_maxchan=True
    )

    # initialize x, z with CoM
    ptp_p = ptp / ptp.sum()
    xcom, zcom = (ptp_p[:, None] * local_geom).sum(axis=0)

    def residual(loc):
        x, y, z, alpha = loc
        sq_dxz = np.square(local_geom - np.array(((x, z),)))
        dists = np.sqrt((y ** 2 + sq_dxz).sum(axis=1))
        return ptp - alpha / dists

    # def jacobian(loc):

    result = least_squares(
        residual,
        x0=[xcom, Y0, zcom, ALPHA0],
        bounds=BOUNDS,
    )

    # convert to absolute positions
    x, y, z, alpha = result.x
    z_abs = z + z_maxchan
    return x, y, z_abs, alpha


def localize_waveforms(
    waveforms,
    geom,
    maxchans=None,
    channel_radius=10,
    n_workers=1,
    _not_helper=True,
):
    """Localize a bunch of waveforms

    waveforms is N x T x C, where C can be either 2 * channel_radius, or
    C can be 384 (or geom.shape[0]). If the latter, the 2 * channel_radius
    bits will be extracted.
    """
    if _not_helper:
        N, T, C = check_shapes(waveforms, maxchans, channel_radius, geom)
    else:
        N, T, C = waveforms.shape

    # I have them stored as floats and keep forgetting to int them.
    maxchans = maxchans.astype(int)

    # handle pbars
    xrange = trange if _not_helper else lambda a, desc: range(a)
    xqdm = tqdm if _not_helper else lambda a, total, desc: a

    # -- get N x (2 * channel_radius) array of PTPs
    if C == 2 * channel_radius:
        ptps = waveforms.ptp(axis=1)
    else:
        # we need maxchans to extract the local waveform
        ptps_full = waveforms.ptp(axis=1)
        if maxchans is None:
            maxchans = np.argmax(ptps_full, axis=1)
            bad = np.flatnonzero(ptps_full.ptp(1) == 0)
            if bad:
                raise ValueError(f"Some waveforms were all zero: {bad}.")

        ptps = np.empty((N, 2 * channel_radius), dtype=waveforms.dtype)
        for n in xrange(N, desc="extracting channels"):
            low = maxchans[n] - channel_radius
            high = maxchans[n] + channel_radius
            if low < 0:
                low = 0
                high = 2 * channel_radius
            if high > C:
                high = C
                low = C - 2 * channel_radius
            ptps[n] = ptps_full[n, low:high]
        del ptps_full

    # -- run the least squares
    xs = np.empty(N)
    ys = np.empty(N)
    zs = np.empty(N)
    alphas = np.empty(N)
    with Parallel(n_workers) as pool:
        for n, (x, y, z, alpha) in enumerate(
            pool(
                delayed(localize_ptp)(ptp, maxchan, geom)
                for ptp, maxchan in xqdm(
                    zip(ptps, maxchans), total=N, desc="lsq"
                )
            )
        ):
            xs[n] = x
            ys[n] = y
            zs[n] = z
            alphas[n] = alpha

    return xs, ys, zs, alphas


def localize_waveforms_batched(
    waveforms,
    geom,
    maxchans=None,
    channel_radius=10,
    n_workers=1,
    batch_size=128,
):
    """A helper for running the above on hdf5 datasets or similar"""
    N, T, C = check_shapes(waveforms, maxchans, channel_radius, geom)
    xs = np.empty(N)
    ys = np.empty(N)
    zs = np.empty(N)
    alphas = np.empty(N)

    starts = list(range(0, N, batch_size))
    ends = [min(start + batch_size, N) for start in starts]

    def maxchan_batch(start, end):
        if maxchans is None:
            return None
        else:
            return maxchans[start:end]

    with Parallel(n_workers) as pool:
        for batch_idx, (x, y, z, alpha) in enumerate(
            pool(
                delayed(localize_waveforms)(
                    waveforms[start:end],
                    geom,
                    maxchans=maxchan_batch(start, end),
                    channel_radius=channel_radius,
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
            zs[start:end] = z
            alphas[start:end] = alpha

    return xs, ys, zs, alphas
