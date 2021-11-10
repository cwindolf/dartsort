import numpy as np
from scipy.optimize import least_squares
from tqdm.auto import trange, tqdm
from joblib import Parallel, delayed

# (x_low, y_low, z_low, alpha_low), (x_high, y_high, z_high, alpha_high)
BOUNDS = (-100, 0, -100, 0), (132, 250, 100, 10000)
Y0, ALPHA0 = 21.0, 1000.0


def localize_ptp(ptp, maxchan, geom):
    """Find the localization result for a single ptp vector

    ptp : np.array (2 * channel_radius,)
    maxchan : int
    geom : np.array (total_channels, 2)
    """
    channel_radius = ptp.shape[0] // 2
    # local_geom is 2*channel_radius, 2
    low = maxchan - channel_radius
    high = maxchan + channel_radius
    if low < 0:
        low = 0
        high = 2 * channel_radius
    if high > geom.shape[0]:
        high = geom.shape[0]
        low = geom.shape[0] - 2 * channel_radius
    local_geom = geom[low:high].copy()
    z_maxchan = geom[maxchan, 1]
    local_geom[:, 1] -= z_maxchan
    # print(local_geom.shape)
    # print(maxchan, local_geom)

    # initialize x, z with CoM
    ptp_p = ptp / ptp.sum()
    # print("ptpsummin", ptp.sum(), ptp.min())
    xcom, zcom = (ptp_p[:, None] * local_geom).sum(axis=0)
    # print(local_geom.min(axis=0), local_geom.max(axis=0), [xcom, zcom])

    def residual(loc):
        x, y, z, alpha = loc
        sq_dxz = np.square(local_geom - np.array(((x, z),)))
        dists = np.sqrt((y ** 2 + sq_dxz).sum(axis=1))
        return ptp - alpha / dists

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
):
    """Localize a bunch of waveforms

    waveforms is N x T x C, where C can be either 2 * channel_radius, or
    C can be 384 (or geom.shape[0]). If the latter, the 2 * channel_radius
    bits will be extracted.
    """
    N, T, C = waveforms.shape
    C_, d = geom.shape
    assert d == 2

    # -- get N x (2 * channel_radius) array of PTPs
    if C == 2 * channel_radius:
        print(f"Waveforms are already trimmed to {C} channels.")
        ptps = waveforms.ptp(axis=1)
        if maxchans is None:
            raise ValueError(
                "maxchans can't be None when waveform channels < geom channels"
            )
    elif C == C_:
        print(
            f"Waveforms are on all {C} channels. "
            f"Trimming with radius {channel_radius}."
        )
        if maxchans is None:
            ptps_full = waveforms.ptp(axis=1)
            maxchans = np.argmax(ptps_full, axis=1)
            bad = np.flatnonzero(ptps_full.ptp(1) == 0)
            if bad:
                raise ValueError(f"Some waveforms were all zero: {bad}.")
        ptps = np.empty((N, 2 * channel_radius), dtype=waveforms.dtype)
        for n in trange(N, desc="extracting channels"):
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
    else:
        raise ValueError(
            f"Not sure what to do with waveforms.shape={waveforms.shape}."
        )

    # -- run the least squares
    xs = np.empty(N)
    ys = np.empty(N)
    zs = np.empty(N)
    alphas = np.empty(N)
    with Parallel(n_workers) as pool:
        for n, (x, y, z, alpha) in enumerate(
            pool(
                delayed(localize_ptp)(ptp, maxchan, geom)
                for ptp, maxchan in tqdm(
                    zip(ptps, maxchans), total=N, desc="lsq"
                )
            )
        ):
            xs[n] = x
            ys[n] = y
            zs[n] = z
            alphas[n] = alpha

    return xs, ys, zs, alphas
