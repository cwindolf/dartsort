import numpy as np
import scipy.linalg as la
import torch
from scipy.linalg import solve, lstsq
from tqdm.auto import trange

from .ibme_corr import calc_corr_decent_pair
from .motion_utils import (
    fast_raster,
    get_bins,
    get_motion_estimate,
    get_window_domains,
    get_windows,
)

default_raster_kw = dict(
    amp_scale_fn=None,
    post_transform=np.log1p,
    gaussian_smoothing_sigma_um=1,
)


def laplacian(n, wink=True, eps=1e-10, lambd=1.0):
    lap = (lambd + eps) * np.eye(n)
    if wink:
        lap[0, 0] -= 0.5 * lambd
        lap[-1, -1] -= 0.5 * lambd
    lap -= np.diag(0.5 * lambd * np.ones(n - 1), k=1)
    lap -= np.diag(0.5 * lambd * np.ones(n - 1), k=-1)
    return lap


def neg_hessian_likelihood_term(Ub, Ub_prevcur=None, Ub_curprev=None):
    # negative Hessian of p(D | p) inside a block
    negHUb = -Ub.copy()
    negHUb -= Ub.T
    diagonal_terms = np.diagonal(negHUb) + Ub.sum(1) + Ub.sum(0)
    if Ub_prevcur is None:
        np.fill_diagonal(negHUb, diagonal_terms)
    else:
        diagonal_terms += Ub_prevcur.sum(0) + Ub_curprev.sum(1)
        np.fill_diagonal(negHUb, diagonal_terms)
    return negHUb


def newton_rhs(
    Db,
    Ub,
    Pb_prev=None,
    Db_prevcur=None,
    Ub_prevcur=None,
    Db_curprev=None,
    Ub_curprev=None,
):
    UDb = Ub * Db
    grad_at_0 = UDb.sum(1) - UDb.sum(0)
    if Pb_prev is None:
        return grad_at_0

    # online case
    # the math is written without assuming symmetry, so it has a sum
    # of the two off-diagonals instead of 2x the upper. symmetry only
    # approximately holds due to the way we cross-correlate in the
    # nonrigid case (it holds absolutely in the rigid case), but it's
    # approximately fine and it would be a waste to compute both off-diagonal
    # xcorrs.
    # same goes for the UDb_prev stuff below
    align_term = (Ub_prevcur.T + Ub_curprev) @ Pb_prev
    rhs = (
        align_term
        + grad_at_0
        + (Ub_curprev * Db_curprev).sum(1)
        - (Ub_prevcur * Db_prevcur).sum(0)
    )

    return rhs


def newton_solve_rigid(
    D,
    U,
    Sigma0inv,
    Pb_prev=None,
    Db_prevcur=None,
    Ub_prevcur=None,
    Db_curprev=None,
    Ub_curprev=None,
):
    """D is TxT displacement, U is TxT subsampling or soft weights matrix"""
    negHU = neg_hessian_likelihood_term(
        U,
        Ub_prevcur=Ub_prevcur,
        Ub_curprev=Ub_curprev,
    )
    targ = newton_rhs(
        D,
        U,
        Pb_prev=Pb_prev,
        Db_prevcur=Db_prevcur,
        Ub_prevcur=Ub_prevcur,
        Db_curprev=Db_curprev,
        Ub_curprev=Ub_curprev,
    )
    # p, *_ = lstsq(Sigma0inv + negHU, targ)#, assume_a="pos")
    p = solve(Sigma0inv + negHU, targ, assume_a="pos")
    return p, negHU


default_thomas_kw = dict(
    lambda_s=1.0,
    lambda_t=1.0,
    eps=1e-10,
)


def thomas_solve(
    Ds,
    Us,
    lambda_t=1.0,
    lambda_s=1.0,
    eps=1e-10,
    P_prev=None,
    Ds_prevcur=None,
    Us_prevcur=None,
    Ds_curprev=None,
    Us_curprev=None,
):
    """Block tridiagonal algorithm, special cased to our setting

    This code solves for the displacement estimates across the nonrigid windows,
    given blockwise, pairwise (BxTxT) displacement and weights arrays `Ds` and `Us`.

    If `lambda_t>0`, a temporal prior is applied to "fill the gaps", effectively
    interpolating through time to avoid artifacts in low-signal areas. Setting this
    to 0 can lead to numerical warnings and should be done with care.

    If `lambda_s>0`, a spatial prior is applied. This can help fill gaps more
    meaningfully in the nonrigid case, using information from the neighboring nonrigid
    windows to inform the estimate in an untrusted region of a given window.

    If arguments `P_prev,Ds_prevcur,Us_prevcur` are supplied, this code handles the
    online case. The return value will be the new chunk's displacement estimate,
    solving the online registration problem.
    """
    Ds = np.asarray(Ds, dtype=np.float64)
    Us = np.asarray(Us, dtype=np.float64)
    online = P_prev is not None
    online_kw_rhs = online_kw_hess = lambda b: {}
    if online:
        assert Ds_prevcur is not None
        assert Us_prevcur is not None
        online_kw_rhs = lambda b: dict(
            Pb_prev=P_prev[b].astype(np.float64),
            Db_prevcur=Ds_prevcur[b].astype(np.float64),
            Ub_prevcur=Us_prevcur[b].astype(np.float64),
            Db_curprev=Ds_curprev[b].astype(np.float64),
            Ub_curprev=Us_curprev[b].astype(np.float64),
        )
        online_kw_hess = lambda b: dict(
            Ub_prevcur=Us_prevcur[b].astype(np.float64),
            Ub_curprev=Us_curprev[b].astype(np.float64),
        )

    B, T, T_ = Ds.shape
    assert T == T_
    assert Us.shape == Ds.shape
    # temporal prior matrix
    L_t = laplacian(T, eps=eps, lambd=lambda_t)
    extra = dict(L_t=L_t)

    # just solve independent problems when there's no spatial regularization
    # not that there's much overhead to the backward pass etc but might as well
    if B == 1 or lambda_s == 0:
        P = np.zeros((B, T))
        extra["HU"] = np.zeros((B, T, T))
        for b in range(B):
            P[b], extra["HU"][b] = newton_solve_rigid(
                Ds[b], Us[b], L_t, **online_kw_rhs(b)
            )
        return P, extra

    # spatial prior is a sparse, block tridiagonal kronecker product
    # the first and last diagonal blocks are
    # Lambda_s_diag0 = (lambda_s / 2) * (L_t + eps * np.eye(T))
    # the other diagonal blocks are
    Lambda_s_diag1 = (lambda_s + eps) * laplacian(T, eps=eps, lambd=1.0)
    # and the off-diagonal blocks are
    Lambda_s_offdiag = (-lambda_s / 2) * laplacian(T, eps=eps, lambd=1.0)

    # initialize block-LU stuff and forward variable
    alpha_hat_b = (
        L_t
        + Lambda_s_diag1 / 2
        + neg_hessian_likelihood_term(Us[0], **online_kw_hess(0))
    )
    targets = np.c_[Lambda_s_offdiag, newton_rhs(Us[0], Ds[0], **online_kw_rhs(0))]
    res = solve(alpha_hat_b, targets, assume_a="pos")
    # res = solve(alpha_hat_b, targets, assume_a="pos")
    assert res.shape == (T, T + 1)
    gamma_hats = [res[:, :T]]
    ys = [res[:, T]]

    # forward pass
    for b in range(1, B):
        s_factor = 1 if b < B - 1 else 0.5
        Ab = (
            L_t
            + Lambda_s_diag1 * s_factor
            + neg_hessian_likelihood_term(Us[b], **online_kw_hess(b))
        )
        alpha_hat_b = Ab - Lambda_s_offdiag @ gamma_hats[b - 1]
        targets[:, T] = newton_rhs(Us[b], Ds[b], **online_kw_rhs(b))
        targets[:, T] -= Lambda_s_offdiag @ ys[b - 1]
        res = solve(alpha_hat_b, targets)
        assert res.shape == (T, T + 1)
        gamma_hats.append(res[:, :T])
        ys.append(res[:, T])

    # back substitution
    xs = [None] * B
    xs[-1] = ys[-1]
    for b in range(B - 2, -1, -1):
        xs[b] = ys[b] - gamma_hats[b] @ xs[b + 1]

    # un-vectorize
    P = np.concatenate(xs).reshape(B, T)

    return P, extra


def get_weights_in_window(window, db_unreg, db_reg, raster_reg):
    ilow, ihigh = np.flatnonzero(window)[[0, -1]]
    window_sliced = window[ilow : ihigh + 1]
    tbcu = 0.5 * (db_unreg[1:] + db_unreg[:-1])
    dlow, dhigh = tbcu[[ilow, ihigh]]
    tbcr = 0.5 * (db_reg[1:] + db_reg[:-1])
    rilow, rihigh = np.flatnonzero((tbcr >= dlow) & (tbcr <= dhigh))[[0, -1]]
    return window_sliced @ raster_reg[rilow : rihigh + 1]


def get_weights(
    Ds,
    Ss,
    Sigma0inv_t,
    windows,
    amps,
    depths,
    times,
    weights_threshold_low=0.0,
    weights_threshold_high=np.inf,
    raster_kw=default_raster_kw,
    pbar=False,
):
    r, dbe, tbe = fast_raster(amps, depths, times, **raster_kw)
    assert windows.shape[1] == dbe.size - 1
    weights = []
    p_inds = []
    # start with rigid registration with weights=inf independently in each window
    for b in trange((len(Ds)), desc="Weights") if pbar else range(len(Ds)):
        # rigid motion estimate in this window
        p = newton_solve_rigid(Ds[b], Ss[b], Sigma0inv_t)[0]
        p_inds.append(p)
        me = get_motion_estimate(p, time_bin_edges_s=tbe)
        depths_reg = me.correct_s(times, depths)

        # raster just our window's bins
        # take care when tracking start/end indices of bin centers v bin edges
        ilow, ihigh = np.flatnonzero(windows[b])[[0, -1]]
        ihigh = ihigh + 1
        window_sliced = windows[b, ilow:ihigh]
        rr, dbr, tbr = fast_raster(
            amps,
            depths_reg,
            times,
            spatial_bin_edges_um=dbe[ilow : ihigh + 1],
            time_bin_edges_s=tbe,
            **raster_kw,
        )
        assert (rr.shape[0],) == window_sliced.shape
        if rr.sum() <= 0:
            raise ValueError("Convergence issue when getting weights.")
        weights.append(window_sliced @ rr)

    weights_orig = np.array(weights)

    scale_fn = raster_kw["post_transform"] or raster_kw["amp_scale_fn"]
    if isinstance(weights_threshold_low, tuple):
        nspikes_threshold_low, amp_threshold_low = weights_threshold_low
        unif = np.full_like(windows[0], 1 / len(windows[0]))
        weights_threshold_low = (
            scale_fn(amp_threshold_low) * windows @ (nspikes_threshold_low * unif)
        )
        weights_threshold_low = weights_threshold_low[:, None]
    if isinstance(weights_threshold_high, tuple):
        nspikes_threshold_high, amp_threshold_high = weights_threshold_high
        unif = np.full_like(windows[0], 1 / len(windows[0]))
        weights_threshold_high = (
            scale_fn(amp_threshold_high) * windows @ (nspikes_threshold_high * unif)
        )
        weights_threshold_high = weights_threshold_high[:, None]
    weights_thresh = weights_orig.copy()
    weights_thresh[weights_orig < weights_threshold_low] = 0
    weights_thresh[weights_orig > weights_threshold_high] = np.inf

    return weights, weights_thresh, p_inds


def threshold_correlation_matrix(
    Cs,
    mincorr=0.0,
    mincorr_percentile=None,
    mincorr_percentile_nneighbs=20,
    max_dt_s=0,
    in_place=False,
    bin_s=1,
    T=None,
    soft=True,
):
    if mincorr_percentile is not None:
        diags = [
            np.diagonal(Cs, offset=j, axis1=1, axis2=2).ravel()
            for j in range(1, mincorr_percentile_nneighbs)
        ]
        mincorr = np.percentile(
            np.concatenate(diags),
            mincorr_percentile,
        )

    # need abs to avoid -0.0s which cause numerical issues
    if in_place:
        Ss = Cs
        if soft:
            Ss[Ss < mincorr] = 0
        else:
            Ss = (Ss >= mincorr).astype(Cs.dtype)
        np.square(Ss, out=Ss)
    else:
        if soft:
            Ss = np.square((Cs >= mincorr) * Cs)
        else:
            Ss = (Cs >= mincorr).astype(Cs.dtype)
    if max_dt_s is not None and max_dt_s > 0 and T is not None and max_dt_s < T:
        mask = la.toeplitz(
            np.r_[
                np.ones(int(max_dt_s // bin_s), dtype=Ss.dtype),
                np.zeros(T - int(max_dt_s // bin_s), dtype=Ss.dtype),
            ]
        )
        Ss *= mask[None]
    return Ss, mincorr


def weight_correlation_matrix(
    Ds,
    Cs,
    amps,
    depths_um,
    times_s,
    windows,
    mincorr=0.0,
    mincorr_percentile=None,
    mincorr_percentile_nneighbs=20,
    max_dt_s=None,
    lambda_t=1,
    eps=1e-10,
    do_window_weights=True,
    weights_threshold_low=0.0,
    weights_threshold_high=np.inf,
    raster_kw=default_raster_kw,
    pbar=True,
):
    extra = {}
    bin_s = raster_kw["bin_s"]
    bin_um = raster_kw["bin_um"]

    # TODO: upsample_to_histogram_bin
    # handle shapes and the rigid case
    Ds = np.asarray(Ds)
    Cs = np.asarray(Cs)
    if Ds.ndim == 2:
        Ds = Ds[None]
        Cs = Cs[None]
    B, T, T_ = Ds.shape
    assert T == T_
    assert Ds.shape == Cs.shape
    spatial_bin_edges_um, time_bin_edges_s = get_bins(depths_um, times_s, bin_um, bin_s)
    assert (T + 1,) == time_bin_edges_s.shape
    extra = {}

    Ss, mincorr = threshold_correlation_matrix(
        Cs,
        mincorr=mincorr,
        mincorr_percentile=mincorr_percentile,
        mincorr_percentile_nneighbs=mincorr_percentile_nneighbs,
        max_dt_s=max_dt_s,
        bin_s=bin_s,
        T=T,
    )
    extra["S"] = Ss
    extra["mincorr"] = mincorr

    if not do_window_weights:
        return Ss, extra

    # get weights
    L_t = lambda_t * laplacian(T, eps=max(1e-5, eps))
    weights_orig, weights_thresh, Pind = get_weights(
        Ds,
        Ss,
        L_t,
        windows,
        amps,
        depths_um,
        times_s,
        weights_threshold_low=weights_threshold_low,
        weights_threshold_high=weights_threshold_high,
        raster_kw=raster_kw,
        pbar=pbar,
    )
    extra["weights_orig"] = weights_orig
    extra["weights_thresh"] = weights_thresh
    extra["Pind"] = Pind

    # update noise model. we deliberately divide by zero and inf here.
    with np.errstate(divide="ignore"):
        invW = 1.0 / weights_thresh
        invWbtt = invW[:, :, None] + invW[:, None, :]
        Us = np.abs(1.0 / (invWbtt + 1.0 / Ss))
    extra["U"] = Us

    return Us, extra


default_xcorr_kw = dict(
    centered=True,
    normalized=True,
)


def xcorr_windows(
    raster_a,
    windows,
    spatial_bin_edges_um,
    win_scale_um,
    raster_b=None,
    rigid=False,
    bin_um=1,
    max_disp_um=None,
    pbar=True,
    xcorr_kw=default_xcorr_kw,
    device=None,
):
    xcorr_kw = default_xcorr_kw | xcorr_kw

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if max_disp_um is None:
        if rigid:
            max_disp_um = int(spatial_bin_edges_um.ptp() // 4)
        else:
            max_disp_um = int(win_scale_um // 4)

    max_disp_bins = int(max_disp_um // bin_um)
    slices = get_window_domains(windows)
    B, D = windows.shape
    D_, T0 = raster_a.shape
    assert D == D_

    # torch versions on device
    windows_ = torch.as_tensor(windows, dtype=torch.float, device=device)
    raster_a_ = torch.as_tensor(raster_a, dtype=torch.float, device=device)
    if raster_b is not None:
        assert raster_b.shape[0] == D
        T1 = raster_b.shape[1]
        raster_b_ = torch.as_tensor(raster_b, dtype=torch.float, device=device)
    else:
        T1 = T0
        raster_b_ = raster_a_

    # estimate each window's displacement
    Ds = np.empty((B, T0, T1), dtype=np.float32)
    Cs = np.empty((B, T0, T1), dtype=np.float32)
    block_iter = trange(B, desc="Cross correlation") if pbar else range(B)
    for b in block_iter:
        window = windows_[b]

        # we search for the template (windowed part of raster a)
        # within a larger-than-the-window neighborhood in raster b
        targ_low = slices[b].start - max_disp_bins
        b_low = max(0, targ_low)
        targ_high = slices[b].stop + max_disp_bins
        b_high = min(D, targ_high)
        padding = max(b_low - targ_low, targ_high - b_high)

        # arithmetic to compute the lags in um corresponding to
        # corr argmaxes
        n_left = padding + slices[b].start - b_low
        n_right = padding + b_high - slices[b].stop
        poss_disp = -np.arange(-n_left, n_right + 1) * bin_um

        Ds[b], Cs[b] = calc_corr_decent_pair(
            raster_a_[slices[b]],
            raster_b_[b_low:b_high],
            weights=window[slices[b]],
            disp=padding,
            possible_displacement=poss_disp,
            device=device,
            **xcorr_kw,
        )

    return Ds, Cs, max_disp_um


default_weights_kw = dict(
    mincorr=0.0,
    max_dt_s=None,
    do_window_weights=True,
    weights_threshold_low=0.0,
    weights_threshold_high=np.inf,
)


def register(
    amps,
    depths_um,
    times_s,
    rigid=False,
    bin_um=1.0,
    bin_s=1.0,
    win_shape="gaussian",
    win_step_um=400,
    win_scale_um=450,
    win_margin_um=None,
    max_disp_um=None,
    thomas_kw=default_thomas_kw,
    xcorr_kw=default_xcorr_kw,
    raster_kw=default_raster_kw,
    weights_kw=default_weights_kw,
    upsample_to_histogram_bin=False,
    device=None,
    pbar=True,
    save_full=False,
    precomputed_D_C_maxdisp=None,
):
    # TODO: upsample_to_histogram_bin
    assert not upsample_to_histogram_bin

    thomas_kw = default_thomas_kw | thomas_kw
    raster_kw = default_raster_kw | raster_kw
    weights_kw = default_weights_kw | weights_kw
    raster_kw["bin_s"] = bin_s
    raster_kw["bin_um"] = bin_um

    extra = {}

    raster, spatial_bin_edges_um, time_bin_edges_s = fast_raster(
        amps,
        depths_um,
        times_s,
        **raster_kw,
    )
    windows, window_centers = get_windows(
        # pseudo geom to fool spikeinterface
        np.c_[np.zeros_like(spatial_bin_edges_um), spatial_bin_edges_um],
        win_step_um,
        win_scale_um,
        spatial_bin_edges=spatial_bin_edges_um,
        margin_um=-win_scale_um / 2 if win_margin_um is None else win_margin_um,
        win_shape=win_shape,
        zero_threshold=1e-5,
        rigid=rigid,
    )

    # cross-correlate to get D and C
    if precomputed_D_C_maxdisp is None:
        Ds, Cs, max_disp_um = xcorr_windows(
            raster,
            windows,
            spatial_bin_edges_um,
            win_scale_um,
            rigid=rigid,
            bin_um=bin_um,
            max_disp_um=max_disp_um,
            pbar=pbar,
            xcorr_kw=xcorr_kw,
            device=device,
        )
    else:
        Ds, Cs, max_disp_um = precomputed_D_C_maxdisp

    # turn Cs into weights
    Us, wextra = weight_correlation_matrix(
        Ds,
        Cs,
        amps,
        depths_um,
        times_s,
        windows,
        lambda_t=thomas_kw["lambda_t"],
        eps=thomas_kw["eps"],
        raster_kw=raster_kw,
        pbar=pbar,
        **weights_kw,
    )
    extra.update({k: wextra[k] for k in wextra if k not in ("S", "U")})
    if save_full:
        extra.update({k: wextra[k] for k in wextra if k in ("S", "U")})

    # solve for P
    # now we can do our tridiag solve
    P, textra = thomas_solve(Ds, Us, **thomas_kw)
    me = get_motion_estimate(
        P,
        spatial_bin_centers_um=window_centers,
        time_bin_edges_s=time_bin_edges_s,
    )
    if save_full:
        extra.update(textra)

    extra["windows"] = windows
    extra["window_centers"] = window_centers
    extra["max_disp_um"] = max_disp_um
    if save_full:
        extra["D"] = Ds
        extra["C"] = Cs

    return me, extra


default_weights_kw_lfp = dict(
    mincorr=0.8,
    max_dt_s=None,
    do_window_weights=False,
    mincorr_percentile_nneighbs=20,
    soft=False,
)


def register_online_lfp(
    lfp_recording,
    rigid=True,
    chunk_len_s=10.0,
    win_shape="gaussian",
    win_step_um=400,
    win_scale_um=450,
    win_margin_um=None,
    max_disp_um=None,
    thomas_kw=default_thomas_kw,
    xcorr_kw=default_xcorr_kw,
    weights_kw=default_weights_kw_lfp,
    upsample_to_histogram_bin=False,
    save_full=False,
    device=None,
    pbar=True,
):
    """Online registration of a preprocessed lfp recording"""
    # TODO: upsample_to_histogram_bin
    assert not upsample_to_histogram_bin

    geom = lfp_recording.get_channel_locations()
    fs = lfp_recording.get_sampling_frequency()
    T_total = lfp_recording.get_num_samples()
    T_chunk = min(int(np.floor(fs * chunk_len_s)), T_total)

    # kwarg defaults and handling
    # need lfp-specific defaults
    weights_kw = default_weights_kw_lfp | weights_kw
    xcorr_kw = default_xcorr_kw | xcorr_kw
    thomas_kw = default_thomas_kw | thomas_kw
    full_xcorr_kw = dict(
        rigid=rigid,
        bin_um=np.median(np.diff(geom[:, 1])),
        max_disp_um=max_disp_um,
        pbar=False,
        xcorr_kw=xcorr_kw,
        device=device,
    )
    mincorr_percentile = None
    mincorr = weights_kw["mincorr"]
    threshold_kw = dict(
        mincorr_percentile_nneighbs=weights_kw["mincorr_percentile_nneighbs"],
        max_dt_s=weights_kw["max_dt_s"],
        bin_s=1 / fs,
        in_place=True,
        soft=weights_kw["soft"],
    )
    if "mincorr_percentile" in weights_kw:
        mincorr_percentile = weights_kw["mincorr_percentile"]

    # get windows
    windows, window_centers = get_windows(
        geom,
        win_step_um,
        win_scale_um,
        spatial_bin_centers=geom[:, 1],
        margin_um=-win_scale_um / 2 if win_margin_um is None else win_margin_um,
        win_shape=win_shape,
        zero_threshold=1e-5,
        rigid=rigid,
    )
    B = len(windows)
    extra = dict(window_centers=window_centers, windows=windows)

    # -- allocate output and initialize first chunk
    P_online = np.empty((B, T_total), dtype=np.float32)
    # below, t0 is start of prev chunk, t1 start of cur chunk, t2 end of cur
    t0, t1 = 0, T_chunk
    traces0 = lfp_recording.get_traces(start_frame=t0, end_frame=t1)
    Ds0, Cs0, max_disp_um = xcorr_windows(
        traces0.T, windows, geom[:, 1], win_scale_um, **full_xcorr_kw
    )
    full_xcorr_kw["max_disp_um"] = max_disp_um
    Ss0, mincorr0 = threshold_correlation_matrix(
        Cs0, mincorr_percentile=mincorr_percentile, mincorr=mincorr, **threshold_kw
    )
    if save_full:
        extra["D"] = [Ds0]
        extra["C"] = [Cs0]
        extra["S"] = [Ss0]
        extra["D01"] = []
        extra["C01"] = []
        extra["S01"] = []
    extra["mincorrs"] = [mincorr0]
    extra["max_disp_um"] = max_disp_um
    P_online[:, t0:t1], _ = thomas_solve(Ds0, Ss0, **thomas_kw)

    # -- loop through chunks
    if pbar:
        chunk_starts = trange(T_chunk, T_total, T_chunk, desc="chunks")
    else:
        chunk_starts = range(T_chunk, T_total, T_chunk)
    for t1 in chunk_starts:
        t2 = min(T_total, t1 + T_chunk)
        traces1 = lfp_recording.get_traces(start_frame=t1, end_frame=t2)

        # cross-correlations between prev/cur chunks
        Ds10, Cs10, _ = xcorr_windows(
            traces1.T,
            windows,
            geom[:, 1],
            win_scale_um,
            raster_b=traces0.T,
            **full_xcorr_kw,
        )
        # Ds01, Cs01, _ = xcorr_windows(
        #     traces0.T,
        #     windows,
        #     geom[:, 1],
        #     win_scale_um,
        #     raster_b=traces1.T,
        #     **full_xcorr_kw,
        # )
        # Ss01 = threshold_correlation_matrix(Cs01, **threshold_kw)

        # cross-correlation in current chunk
        Ds1, Cs1, _ = xcorr_windows(
            traces1.T, windows, geom[:, 1], win_scale_um, **full_xcorr_kw
        )
        Ss1, mincorr1 = threshold_correlation_matrix(
            Cs1, mincorr_percentile=mincorr_percentile, mincorr=mincorr, **threshold_kw
        )
        Ss10, _ = threshold_correlation_matrix(Cs10, mincorr=mincorr1, **threshold_kw)
        extra["mincorrs"].append(mincorr1)

        if save_full:
            extra["D"].append(Ds1)
            extra["C"].append(Cs1)
            extra["S"].append(Ss1)
            extra["D01"].append(Ds10)
            extra["C01"].append(Cs10)
            extra["S01"].append(Ss10)

        # solve online problem
        P_online[:, t1:t2], _ = thomas_solve(
            Ds1,
            Ss1,
            P_prev=P_online[:, t0:t1],
            Ds_curprev=Ds10,
            Us_curprev=Ss10,
            Ds_prevcur=-Ds10.transpose(0, 2, 1),
            Us_prevcur=Ss10.transpose(0, 2, 1),
            # Ds_prevcur=Ds01,
            # Us_prevcur=Ss01,
            **thomas_kw,
        )

        # update loop vars
        t0, t1 = t1, t2
        traces0 = traces1

    # -- convert to motion estimate and return
    # should use get_times or something
    me = get_motion_estimate(
        P_online,
        time_bin_centers_s=lfp_recording.get_times(0),
        spatial_bin_centers_um=window_centers,
    )
    return me, extra
