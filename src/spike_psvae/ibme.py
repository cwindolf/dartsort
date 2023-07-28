import numpy as np
import torch
from scipy.stats import norm
from scipy.interpolate import interp1d, RectBivariateSpline
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter1d
from scipy import signal


from .ibme_corr import (
    register_raster_rigid,
    calc_corr_decent,
    calc_corr_decent_pair,
    psolvecorr,
    psolvecorr_spatial,
)
from .motion_utils import (
    RigidMotionEstimate,
    NonrigidMotionEstimate,
    IdentityMotionEstimate,
    ComposeMotionEstimates,
    fast_raster,
    get_windows,
)


# -- main functions: rigid and nonrigid registration


def register_rigid(
    amps,
    depths,
    times,
    disp=None,
    robust_sigma=0.0,
    corr_threshold=0.0,
    adaptive_mincorr_percentile=None,
    normalized=True,
    max_dt=None,
    batch_size=1,
    prior_lambda=0,
    return_extra=False,
    bin_um=1,
    bin_s=1,
    amp_scale_fn=np.log1p,
    gaussian_smoothing_sigma_um=0,
):
    """1D rigid registration

    Arguments
    ---------
    amps, depths, times : np.arrays of shape (N_spikes,)
        Spike amplitudes, depths (locations along probe),
        and times (in seconds)
    disp : int, optional
        Maximum displacement to search over. By default,
        will use half of the length of the depth domain.
    robust_sigma : float
        If >0, a robust least squares procedure similar to
        least trimmed squares will be used. A smaller choice
        means that more points are considered outliers.
    denoise_sigma : float
        Strength of Poisson denoising applied to the raster
    corr_threshold : float
        Exclude pairs of timebins from the optimization if
        their maximum correlation does not exceed this value
    destripe : bool
        Apply a wavelet-based destriping to the raster

    Returns
    -------
    depths_reg : np.array of shape (N_spikes,)
        The registered depths
    p : np.array
        The estimated displacement for each time bin.
    """
    # -- compute time/depth binned amplitude averages
    raster, spatial_bin_edges_um, time_bin_edges_s = fast_raster(
        amps,
        depths,
        times,
        bin_um=bin_um,
        bin_s=bin_s,
        amp_scale_fn=amp_scale_fn,
        gaussian_smoothing_sigma_um=gaussian_smoothing_sigma_um,
    )

    # -- main routine from other file `ibme_corr`
    p, D, C = register_raster_rigid(
        raster,
        mincorr=corr_threshold,
        robust_sigma=robust_sigma,
        disp=disp,
        batch_size=batch_size,
        normalized=normalized,
        max_dt=max_dt,
        adaptive_mincorr_percentile=adaptive_mincorr_percentile,
        prior_lambda=prior_lambda,
    )
    p = p * bin_um

    rme = RigidMotionEstimate(p, time_bin_edges_s)
    extra = dict(
        D=D,
        C=C,
        spatial_bin_edges_um=spatial_bin_edges_um,
        time_bin_edges_s=time_bin_edges_s,
    )

    return rme, extra
    # depths_reg = rme.correct_s(times, depths)


@torch.no_grad()
def register_nonrigid(
    amps,
    depths,
    times,
    geom,
    robust_sigma=0.0,
    corr_threshold=0.0,
    soft_weights=True,
    window_shape="gaussian",
    adaptive_mincorr_percentile=None,
    prior_lambda=0,
    spatial_prior=False,
    normalized=True,
    rigid_disp=400,
    disp=100,
    rigid_init=False,
    win_step_um=100,
    win_sigma_um=100,
    widthmul=1.0,
    max_dt=None,
    device=None,
    batch_size=1,
    bin_um=1,
    bin_s=1,
    avg_in_bin=False,
    amp_scale_fn=np.log1p,
    post_transform=np.log1p,
    gaussian_smoothing_sigma_um=3,
    upsample_to_histogram_bin=False,
    reference=None,
    pbar=True,
    return_CD=False,
):
    """1D nonrigid registration

    This is optionally an iterative procedure if rigid_init=True
    or n_windows is iterable. If rigid_init is True, first a rigid
    registration is applied to depths. Then for each number of
    windows, a nonrigid registration with so many windows is applied.
    These are done cumulatively, making it possible to take a
    coarse-to-fine approach.

    Arguments
    ---------
    amps, depths, times : np.arrays of shape (N_spikes,)
        Spike amplitudes, depths (locations along probe),
        and times (in seconds)
    rigid_disp : int, optional
        Maximum displacement to search over during rigid
        initialization when rigid_init=True.
    disp : int
        Base displacement during nonrigid registration.
        The actual value used is disp / n_windows
    rigid_init : bool
        Should we initialize with a rigid registration?
    robust_sigma : float
        If >0, a robust least squares procedure similar to
        least trimmed squares will be used. A smaller choice
        means that more points are considered outliers.
    denoise_sigma : float
        Strength of Poisson denoising applied to the raster
    corr_threshold : float
        Exclude pairs of timebins from the optimization if
        their maximum correlation does not exceed this value
    destripe : bool
        Apply a wavelet-based destriping to the raster

    Returns
    -------
    depths : np.array of shape (N_spikes,)
        The registered depths
    total_shift : DxT array
        The displacement map
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # first pass of rigid registration
    if rigid_init:
        first_me, extra_rigid = register_rigid(
            amps,
            depths,
            times,
            robust_sigma=robust_sigma,
            corr_threshold=corr_threshold,
            adaptive_mincorr_percentile=adaptive_mincorr_percentile,
            prior_lambda=prior_lambda,
            normalized=normalized,
            batch_size=batch_size,
            disp=rigid_disp,
            max_dt=max_dt,
            bin_s=bin_s,
            bin_um=bin_um,
            amp_scale_fn=amp_scale_fn,
            gaussian_smoothing_sigma_um=gaussian_smoothing_sigma_um,
        )
        extra = dict(extra_rigid=extra_rigid)
    else:
        first_me = IdentityMotionEstimate()
        extra = {}
    depths1 = first_me.correct_s(times, depths)

    raster, spatial_bin_edges_um, time_bin_edges_s = fast_raster(
        amps,
        depths1,
        times,
        bin_um=bin_um,
        bin_s=bin_s,
        amp_scale_fn=amp_scale_fn,
        avg_in_bin=avg_in_bin,
        post_transform=post_transform,
        gaussian_smoothing_sigma_um=gaussian_smoothing_sigma_um,
    )
    extra["spatial_bin_edges_um"] = spatial_bin_edges_um
    extra["time_bin_edges_s"] = time_bin_edges_s
    T = time_bin_edges_s.size - 1

    windows, slices, window_centers = get_windows(
        bin_um,
        spatial_bin_edges_um,
        geom,
        win_step_um,
        win_sigma_um,
        margin_um=-win_step_um / 2,
        win_shape=window_shape,
        return_locs=False,
        zero_threshold=1e-5,
    )
    extra["windows"] = windows
    extra["window_centers"] = window_centers
    if return_CD:
        extra["C"] = []
        extra["D"] = []
    n_windows = windows.shape[0]

    # torch versions on device
    windows_ = torch.as_tensor(windows, dtype=torch.float, device=device)
    raster_ = torch.as_tensor(raster, dtype=torch.float, device=device)

    # estimate each window's displacement
    if spatial_prior:
        block_Ds = np.empty((n_windows, T, T))
        block_Cs = np.empty((n_windows, T, T))
    else:
        ps = np.empty((n_windows, T))

    it = windows_
    if pbar:
        it = tqdm(windows_, desc="windows")
    for k, (window, sl) in enumerate(zip(it, slices)):
        # we search for the template (windowed part of raster a)
        # within a larger-than-the-window neighborhood in raster b
        b_low = max(0, sl.start - disp)
        b_high = min(raster_.shape[0], sl.stop + disp)

        # arithmetic to compute the lags in um corresponding to
        # corr argmaxes
        padding = 0
        n_left = padding + sl.start - b_low
        n_right = padding + b_high - sl.stop
        poss_disp = -np.arange(-n_left, n_right + 1) * bin_um

        D, C = calc_corr_decent_pair(
            raster_[sl],
            raster_[b_low:b_high],
            weights=window[sl],
            disp=padding,
            batch_size=batch_size,
            normalized=normalized,
            possible_displacement=poss_disp,
            device=device,
        )

        if spatial_prior:
            block_Ds[k] = D
            block_Cs[k] = C
        else:
            ps[k] = psolvecorr(
                D,
                C,
                mincorr=corr_threshold,
                robust_sigma=robust_sigma,
                max_dt=max_dt * bin_s if max_dt is not None else None,
                prior_lambda=prior_lambda,
                soft_weights=soft_weights,
            )
        
        if return_CD:
            extra["C"].append(C)
            extra["D"].append(D)

    if spatial_prior:
        ps = psolvecorr_spatial(
            block_Ds,
            block_Cs,
            mincorr=corr_threshold,
            max_dt=max_dt,
            temporal_prior=prior_lambda > 0,
            spatial_prior=True,
            reference_displacement=None,
        )
        ps = ps * bin_um
    
    if reference is not None:
        if isinstance(reference, int):
            ps -= ps[:, 0, None]

    extra["ps"] = ps

    # warp depths
    if upsample_to_histogram_bin:
        windows = windows / windows.sum(axis=0, keepdims=True)
        extra["upsample"] = windows
        dispmap = windows.T @ ps
        nrme = NonrigidMotionEstimate(
            dispmap,
            time_bin_edges_s=time_bin_edges_s,
            spatial_bin_edges_um=spatial_bin_edges_um,
        )
    else:
        nrme = NonrigidMotionEstimate(
            ps,
            time_bin_edges_s=time_bin_edges_s,
            spatial_bin_centers_um=window_centers,
        )

    total_me = ComposeMotionEstimates(first_me, nrme)

    return total_me, extra


# -- nonrigid reg helpers


def compose_shifts_in_orig_domain(orig_shift, new_shift):
    """
    Context: we are in the middle of an iterative algorithm which
    computes displacements. Previously, we had computed some DxT
    displacement matrix `orig_shift`, giving a displacement at all
    depths and times. Then, we used that to compute a new registered
    raster, and we got a new displacement estimate from that new
    raster, `new_shift`. The problem is, that new raster may have a
    different depth domain than the original -- so, how do we combine
    these?

    Like so: if (x, t) is a point in the original domain, then think of
    `orig_shift` as a function f(x, t) giving the displacement at depth
    x and time t. Similarly, write `new_shift` as g(x', t), where x' is
    a point in the domain after it has been displaced by f, i.e.,
    x' = f(x, t) + x. Then we want to know, how much is a point (x,t)
    displaced by applying both f and g?

    The answer:
        (x, t) -> (f(x, t) + g(f(x, t) + x, t), t).
    So, this function returns h(x, t) = f(x, t) + g(f(x, t) + x, t)
    """
    D, T = orig_shift.shape
    D_, T_ = new_shift.shape
    assert T == T_

    orig_depth_domain = np.arange(D, dtype=float)
    g_domain = np.arange(D_, dtype=float)
    time_domain = np.arange(T, dtype=float)

    x_plus_f = orig_depth_domain[:, None] + orig_shift

    g_lerp = RectBivariateSpline(
        g_domain,
        time_domain,
        new_shift,
        kx=1,
        ky=1,
    )
    h = g_lerp(x_plus_f.ravel(), np.tile(time_domain, D), grid=False)

    return orig_shift + h.reshape(orig_shift.shape)
