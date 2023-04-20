import numpy as np
import scipy.linalg as la
from scipy.linalg import solve
import torch
from tqdm.auto import trange
from .motion_utils import (
    get_motion_estimate,
    get_bins,
    get_windows,
    get_window_domains,
    fast_raster,
)
from .ibme_corr import calc_corr_decent_pair


default_raster_kw = dict(
    amp_scale_fn=None,
    post_transform=np.log1p,
    gaussian_smoothing_sigma_um=3,
)


def laplacian(T):
    lap = np.eye(T)
    lap -= np.diag(0.5 * np.ones(T - 1), k=1)
    lap -= np.diag(0.5 * np.ones(T - 1), k=-1)
    return lap


def neg_hessian_likelihood_term(S):
    # the likelihood tearms
    negHS = -S.copy()
    negHS -= S.T
    np.fill_diagonal(negHS, np.diagonal(negHS) + S.sum(1) + S.sum(0))
    return negHS


def newton_rhs(D, S):
    SD = S * D
    grad_at_0 = SD.sum(1) - SD.sum(0)
    return grad_at_0


def newton_solve_rigid(D, S, Sigma0inv):
    """D is TxT displacement, S is TxT subsampling or soft weights matrix"""
    D = D.astype(np.float64)
    S = S.astype(np.float64)
    Sigma0inv = Sigma0inv.astype(np.float64)
    negHS = neg_hessian_likelihood_term(S)
    targ = newton_rhs(D, S)
    p = solve(Sigma0inv + negHS, targ, assume_a="pos")
    return p, negHS


def thomas_solve(Ds, Us, lambda_t=1.0, lambda_s=1.0):
    """Block tridiagonal algorithm, special cased to our setting
    """
    Ds = np.asarray(Ds, dtype=np.float64)
    Us = np.asarray(Us, dtype=np.float64)

    B, T, T_ = Ds.shape
    assert T == T_
    assert Us.shape == Ds.shape
    L_t = lambda_t * laplacian(T)
    eye = np.eye(T)
    eye_s = lambda_s * eye
    # diag_prior_terms = L_t + lambda_s * eye
    offdiag_const = -(lambda_s / 2)
    offdiag_prior_terms = offdiag_const * eye
    del eye

    # just solve independent problems when there's no spatial regularization
    # not that there's much overhead to the backward pass etc but might as well
    if lambda_s == 0:
        return np.array([newton_solve_rigid(D, U, L_t)[0] for D, U in zip(Ds, Us)])

    # initialize block-LU stuff and forward variable
    alpha_hat_b = L_t + eye_s / 2 + neg_hessian_likelihood_term(Us[0])
    targets = np.c_[offdiag_prior_terms, newton_rhs(Us[0], Ds[0])]
    res = solve(alpha_hat_b, targets, assume_a="pos")
    assert res.shape == (T, T + 1)
    gamma_hats = [res[:, :T]]
    ys = [res[:, T]]

    # forward pass
    for b in range(1, B):
        s_factor = 1 if b < B - 1 else 0.5
        Ab = L_t + eye_s * s_factor + neg_hessian_likelihood_term(Us[b])
        # we can * instead of @ since our offdiag is diag
        alpha_hat_b = Ab - offdiag_const * gamma_hats[b - 1]
        targets[:, T] = newton_rhs(Us[b], Ds[b]) - offdiag_const * ys[b - 1]
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

    return P


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



def weight_correlation_matrix(
    Ds,
    Cs,
    amps,
    depths_um,
    times_s,
    windows,
    mincorr=0.0,
    max_dt_s=None,
    bin_s=1,
    bin_um=1,
    lambda_t=1,
    do_window_weights=True,
    weights_threshold_low=0.0,
    weights_threshold_high=np.inf,
    raster_kw=default_raster_kw,
    pbar=True,
):
    extra = {}

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

    # need abs to avoid -0.0s which cause numerical issues below
    Ss = np.square((Cs >= mincorr) * Cs)
    if max_dt_s is not None and max_dt_s > 0:
        mask = la.toeplitz(
            np.r_[
                np.ones(int(max_dt_s // bin_s), dtype=Ss.dtype),
                np.zeros(T - int(max_dt_s // bin_s), dtype=Ss.dtype),
            ]
        )
        Ss *= mask[None]
    extra["S"] = Ss

    if not do_window_weights:
        return Ss, extra

    # get weights
    L_t = lambda_t * laplacian(T)
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

default_thomas_kw = dict(
    lambda_s=1.0,
    lambda_t=1.0,
)


def full_thomas(
    Ds,
    Us,
    time_bin_edges_s,
    lambda_s=1.0,
    lambda_t=1.0,
    window_centers=None,
):
    assert Ds.ndim == Us.ndim == 3
    assert Ds.shape == Us.shape
    B, T, T_ = Ds.shape
    assert T == T_
    if B == 1:
        lambda_s = 0  # no space to have a prior on

    # now we can do our tridiag solve
    P = thomas_solve(Ds, Us, lambda_t=lambda_t, lambda_s=lambda_s)
    me1 = get_motion_estimate(
        P,
        spatial_bin_centers_um=window_centers,
        time_bin_edges_s=time_bin_edges_s,
    )

    return me1


def solve_spatial(wt, pt, lambd=1):
    k = wt.size
    assert wt.shape == pt.shape == (k,)
    finite = np.isfinite(wt)
    finite_inds = np.flatnonzero(finite)
    if not finite_inds.size:
        return pt

    coefts = np.diag(wt[finite_inds] + lambd + 1e-9)
    coefts[0, 0] -= lambd / 2
    coefts[-1, -1] -= lambd / 2
    target = 2 * wt[finite_inds] * pt[finite_inds]
    for i, j in enumerate(finite_inds):
        if j > 0:
            if finite[j - 1]:
                coefts[i, i - 1] = coefts[i - 1, i] = -lambd / 2
            else:
                target[i] += lambd * pt[j - 1]
        if j < k - 1:
            if finite[j + 1]:
                coefts[i, i + 1] = coefts[i + 1, i] = -lambd / 2
            else:
                target[i] += lambd * pt[j + 1]
    try:
        r_finite = solve(coefts, target)
    except np.linalg.LinAlgError:
        print(f"{np.array2string(coefts, precision=2, max_line_width=100)} {target=} {coefts.shape=} {target.shape=}")
        raise
    r = pt.copy()
    r[finite_inds] = r_finite
    return r


default_xcorr_kw = dict(
    centered=True,
    normalized=True,
)


def xcorr_windows(
    raster,
    windows,
    spatial_bin_edges_um,
    win_scale_um,
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

    # torch versions on device
    windows_ = torch.as_tensor(windows, dtype=torch.float, device=device)
    raster_ = torch.as_tensor(raster, dtype=torch.float, device=device)
    B = windows.shape[0]
    T = raster.shape[1]

    # estimate each window's displacement
    Ds = np.empty((B, T, T), dtype=np.float32)
    Cs = np.empty((B, T, T), dtype=np.float32)
    block_iter = trange(B, desc="Cross correlation") if pbar else range(B)
    for b in block_iter:
        window = windows_[b]

        # we search for the template (windowed part of raster a)
        # within a larger-than-the-window neighborhood in raster b
        targ_low = slices[b].start - max_disp_bins
        b_low = max(0, targ_low)
        targ_high = slices[b].stop + max_disp_bins
        b_high = min(raster_.shape[0], targ_high)
        padding = max(b_low - targ_low, targ_high - b_high)

        # arithmetic to compute the lags in um corresponding to
        # corr argmaxes
        n_left = padding + slices[b].start - b_low
        n_right = padding + b_high - slices[b].stop
        poss_disp = -np.arange(-n_left, n_right + 1) * bin_um

        Ds[b], Cs[b] = calc_corr_decent_pair(
            raster_[slices[b]],
            raster_[b_low:b_high],
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
    max_disp_um=None,
    thomas_kw=default_thomas_kw,
    xcorr_kw=default_xcorr_kw,
    raster_kw=default_raster_kw,
    weights_kw=default_weights_kw,
    upsample_to_histogram_bin=False,
    do_local_templates=False,
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

    extra = {}

    raster, spatial_bin_edges_um, time_bin_edges_s = fast_raster(
        amps,
        depths_um,
        times_s,
        bin_um=bin_um,
        bin_s=bin_s,
        **raster_kw,
    )
    windows, window_centers = get_windows(
        bin_um,
        spatial_bin_edges_um,
        # pseudo geom
        np.c_[np.zeros_like(spatial_bin_edges_um), spatial_bin_edges_um],
        win_step_um,
        win_scale_um,
        margin_um=-win_scale_um / 2,
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
    B = Ds.shape[0]

    # turn Cs into weights
    weights_kw = weights_kw | dict()
    Us, wextra = weight_correlation_matrix(
        Ds,
        Cs,
        amps,
        depths_um,
        times_s,
        windows,
        bin_s=bin_s,
        bin_um=bin_um,
        lambda_t=thomas_kw["lambda_t"],
        raster_kw=raster_kw,
        pbar=pbar,
        **weights_kw,
    )
    extra.update({k: wextra[k] for k in wextra if k not in ("S", "U")})
    if save_full:
        extra.update({k: wextra[k] for k in wextra if k in ("S", "U")})

    if do_local_templates:
        local_templates = np.array(
            [
                print(f"st{b}") or shifted_templates(raster, Ds[b], Us[b])
                for b in range(B)
            ]
        )
        for b in trange(B, desc="Corr for local templates") if pbar else range(B):
            Ds[b], _, _ = xcorr_windows(
                local_templates[b],
                windows[b, None],
                spatial_bin_edges_um,
                win_scale_um,
                rigid=rigid,
                bin_um=bin_um,
                max_disp_um=max_disp_um,
                pbar=False,
                xcorr_kw=xcorr_kw,
                device=device,
            )
        extra["local_templates"] = local_templates

    # solve for P
    me = full_thomas(
        Ds,
        Us,
        time_bin_edges_s,
        window_centers=window_centers,
        **thomas_kw,
    )

    extra["windows"] = windows
    extra["window_centers"] = window_centers
    extra["max_disp_um"] = max_disp_um
    if save_full:
        extra["D"] = Ds
        extra["C"] = Cs

    return me, extra




def shifted_template(R, Di, Si, bin_um=1):
    assert (R.shape[1],) == Di.shape == Si.shape
    pad = int(np.abs(Di // bin_um).max())
    Rshifted = np.empty_like(R)
    for i, d in enumerate((Di // bin_um).astype(int)):
        Rshifted[i, :] = np.pad(R[i, :], [(pad, pad)])[pad + d : R.shape[1] + pad + d]
    return (Rshifted * Si[None, :]).sum(1) / Si.sum()


def shifted_templates(R, D, S):
    sts = np.empty_like(R)
    for i, (Di, Si) in enumerate(zip(D.T, S.T)):
        sts[:, i] = shifted_template(R, Di, Si)
    return sts