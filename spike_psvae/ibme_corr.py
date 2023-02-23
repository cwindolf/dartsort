import gc
import numpy as np
import scipy.linalg as la
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.optimize import minimize
from scipy.stats import zscore
from tqdm.auto import trange


def register_raster_rigid(
    raster,
    weights=None,
    mincorr=0.0,
    disp=None,
    robust_sigma=0.0,
    adaptive_mincorr_percentile=None,
    normalized=True,
    max_dt=None,
    batch_size=8,
    device=None,
    pbar=True,
    prior_lambda=0,
):
    """Rigid correlation subsampled registration

    Arguments
    ---------
    raster : array (D, T)
    mincorr : float
        Correlation threshold
    disp : int, optional
        Maximum displacement during pairwise displacement estimates.
        If `None`, half of the depth domain's length will be used.
    batch_size : int
        See `calc_corr_decent`
    Returns: p, array (T,)
    """
    D, C = calc_corr_decent(
        raster,
        weights=weights,
        disp=disp,
        normalized=normalized,
        batch_size=batch_size,
        device=device,
        pbar=pbar,
    )
    if adaptive_mincorr_percentile is not None:
        mincorr = np.percentile(np.diagonal(C, 1), adaptive_mincorr_percentile)
    p = psolvecorr(
        D,
        C,
        mincorr=mincorr,
        robust_sigma=robust_sigma,
        max_dt=max_dt,
        prior_lambda=prior_lambda,
    )
    return p, D, C


def weighted_lsqr(Wij, Dij, I, J, T, p0):
    W = sparse.csr_matrix((Wij, (I, J)), shape=(T, T))
    WD = sparse.csr_matrix((Wij * Dij, (I, J)), shape=(T, T))
    fixed_terms = (W @ WD).diagonal() - (WD @ W).diagonal()
    diag_WW = (W @ W).diagonal()
    Wsq = W.power(2)

    def obj(p):
        return 0.5 * np.square(Wij * (Dij - (p[I] - p[J]))).sum()

    def jac(p):
        return fixed_terms - 2 * (Wsq @ p) + 2 * p * diag_WW

    res = minimize(fun=obj, jac=jac, x0=p0, method="L-BFGS-B")
    if not res.success:
        print("Global displacement gradient descent had an error")
    p = res.x

    return p


def psolvecorr(
    D,
    C,
    mincorr=0.0,
    robust_sigma=0,
    robust_iter=5,
    max_dt=None,
    prior_lambda=0,
):
    """Solve for rigid displacement given pairwise disps + corrs"""
    T = D.shape[0]
    assert (T, T) == D.shape == C.shape

    # subsample where corr > mincorr
    S = C >= mincorr
    if max_dt is not None and max_dt > 0:
        S &= la.toeplitz(
            np.r_[
                np.ones(max_dt, dtype=bool), np.zeros(T - max_dt, dtype=bool)
            ]
        )
    I, J = np.where(S == 1)
    n_sampled = I.shape[0]

    # construct Kroneckers
    ones = np.ones(n_sampled)
    M = sparse.csr_matrix((ones, (range(n_sampled), I)), shape=(n_sampled, T))
    N = sparse.csr_matrix((ones, (range(n_sampled), J)), shape=(n_sampled, T))
    A = M - N
    V = D[I, J]

    # add in our prior p_{i+1} - p_i ~ N(0, lambda) by extending the problem
    if prior_lambda > 0:
        diff = sparse.diags(
            (
                np.full(T - 1, -prior_lambda, dtype=A.dtype),
                np.full(T - 1, prior_lambda, dtype=A.dtype),
            ),
            offsets=(0, 1),
            shape=(T - 1, T),
        )
        A = sparse.vstack((A, diff), format="csr")
        V = np.concatenate(
            (V, np.zeros(T - 1)),
        )

    # solve sparse least squares problem
    if robust_sigma is not None and robust_sigma > 0:
        idx = slice(None)
        for _ in trange(robust_iter, desc="robust lsqr"):
            p, *_ = sparse.linalg.lsmr(A[idx], V[idx])
            idx = np.flatnonzero(np.abs(zscore(A @ p - V)) <= robust_sigma)
    else:
        p, *_ = sparse.linalg.lsmr(A, V)

    return p


def psolvecorr_spatial(
    D,
    C,
    mincorr=0.0,
    robust_sigma=0,
    robust_iter=5,
    max_dt=None,
    temporal_prior=True,
    spatial_prior=True,
    reference_displacement="mode_search",
):
    # D = pairwise_displacement
    W = C > mincorr

    # weighted problem
    # if pairwise_displacement_weight is None:
    #     pairwise_displacement_weight = np.ones_like(D)
    # if sparse_mask is None:
    #     sparse_mask = np.ones_like(D)
    # W = pairwise_displacement_weight * sparse_mask

    assert D.shape == W.shape

    # first dimension is the windows dim, which could be empty in rigid case
    # we expand dims so that below we can consider only the nonrigid case
    if D.ndim == 2:
        W = W[None]
        D = D[None]
    assert D.ndim == W.ndim == 3
    B, T, T_ = D.shape
    assert T == T_

    if max_dt is not None and max_dt > 0:
        horiz = la.toeplitz(
            np.r_[
                np.ones(max_dt, dtype=bool), np.zeros(T - max_dt, dtype=bool)
            ]
        )
        for k in range(W.shape[0]):
            W[k] &= horiz

    # sparsify the problem
    # we will make a list of temporal problems and then
    # stack over the windows axis to finish.
    # each matrix in coefficients will be (sparse_dim, T)
    coefficients = []
    # each vector in targets will be (T,)
    targets = []
    # we want to solve for a vector of shape BT, which we will reshape
    # into a (B, T) matrix.
    # after the loop below, we will stack a coefts matrix (sparse_dim, B, T)
    # and a target vector of shape (B, T), both to be vectorized on last two axes,
    # so that the target p is indexed by i = bT + t (block/window major).

    # calculate coefficients matrices and target vector
    for Wb, Db in zip(W, D):
        # indices of active temporal pairs in this window
        I, J = np.nonzero(Wb > 0)
        n_sampled = I.size

        # construct Kroneckers and sparse objective in this window
        ones = np.ones(n_sampled)
        Mb = sparse.csr_matrix(
            (ones, (range(n_sampled), I)), shape=(n_sampled, T)
        )
        Nb = sparse.csr_matrix(
            (ones, (range(n_sampled), J)), shape=(n_sampled, T)
        )
        block_sparse_kron = Mb - Nb
        block_disp_pairs = Db[I, J]
        
        # add the temporal smoothness prior in this window
        if temporal_prior:
            temporal_diff_operator = sparse.diags(
                (
                    np.full(T - 1, -1, dtype=block_sparse_kron.dtype),
                    np.full(T - 1, 1, dtype=block_sparse_kron.dtype),
                ),
                offsets=(0, 1),
                shape=(T - 1, T),
            )
            block_sparse_kron = sparse.vstack(
                (block_sparse_kron, temporal_diff_operator),
                format="csr",
            )
            block_disp_pairs = np.concatenate(
                (block_disp_pairs, np.zeros(T - 1)),
            )

        coefficients.append(block_sparse_kron)
        targets.append(block_disp_pairs)
    coefficients = sparse.block_diag(coefficients)
    targets = np.concatenate(targets, axis=0)

    # spatial smoothness prior: penalize difference of each block's
    # displacement with the next.
    # only if B > 1, and not in the last window.
    # this is a (BT, BT) sparse matrix D such that:
    # entry at (i, j) is:
    #  {   1 if i = j, i.e., i = j = bT + t for b = 0,...,B-2
    #  {  -1 if i = bT + t and j = (b+1)T + t for b = 0,...,B-2
    #  {   0 otherwise.
    # put more simply, the first (B-1)T diagonal entries are 1,
    # and entries (i, j) such that i = j - T are -1.
    if B > 1 and spatial_prior:
        spatial_diff_operator = sparse.diags(
            (
                np.ones((B - 1) * T, dtype=block_sparse_kron.dtype),
                np.full((B - 1) * T, -1, dtype=block_sparse_kron.dtype),
            ),
            offsets=(0, T),
            shape=((B - 1) * T, B * T),
        )
        coefficients = sparse.vstack((coefficients, spatial_diff_operator))
        targets = np.concatenate(
            (targets, np.zeros((B - 1) * T, dtype=targets.dtype))
        )

    # initialize at the column mean of pairwise displacements (in each window)
    p0 = D.mean(axis=2).reshape(B * T)

    # use LSMR to solve the whole problem
    displacement, *_ = sparse.linalg.lsmr(coefficients, targets, x0=p0)

    # TODO: do we need to weight the upsampling somehow?

    # try to avoid spurious constant offsets
    # let the user choose how to do this. here are some ideas.
    # (one can also -= their own number on the result of this function.)
    if reference_displacement == "mean":
        displacement -= displacement.mean()
    elif reference_displacement == "median":
        displacement -= np.median(displacement)
    elif reference_displacement == "mode_search":
        # just a sketch of an idea -- things might want to change.
        step_size = 0.1
        round_mode = np.round  # floor?
        best_ref = np.median(displacement)
        max_zeros = np.sum(round_mode(displacement - best_ref) == 0)
        for ref in np.arange(
            np.floor(displacement.min()),
            np.ceil(displacement.max()),
            step_size,
        ):
            n_zeros = np.sum(round_mode(displacement - ref) == 0)
            if n_zeros > max_zeros:
                max_zeros = n_zeros
                best_ref = ref
        displacement -= best_ref
    displacement = displacement.reshape(B, T)

    return np.squeeze(displacement)


@torch.no_grad()
def calc_corr_decent(
    raster,
    weights=None,
    disp=None,
    normalized=True,
    batch_size=32,
    step_size=1,
    device=None,
    pbar=True,
):
    """Calculate TxT normalized xcorr and best displacement matrices

    Given a DxT raster, this computes normalized cross correlations for
    all pairs of time bins at offsets in the range [-disp, disp]. Then it
    finds the best one and its corresponding displacement, resulting in
    two TxT matrices: one for the normxcorrs at the best displacement,
    and the matrix of the best displacements.

    Arguments
    ---------
    raster : DxT array
        Depth by time. We want to find the best spatial (depth) displacement
        for all pairs of time bins.
    disp : int
        Maximum displacement (translates to padding for xcorr)
    normalized : bool
        If True, use normalized and centered cross correlations. Otherwise,
        no normalization or centering is used.
    batch_size : int
        How many raster columns to xcorr against the whole raster at once.
    step_size : int
        Displacement increment for coarse-grained search.
        Not implemented yet but easy to do.
    device : torch.device (optional)
    pbar : bool
        Display a progress bar.

    Returns
    -------
    D : TxT array of best displacments
    C : TxT array of best xcorrs or normxcorrs
    """
    # this is not implemented but could be done easily via stride
    if step_size > 1:
        raise NotImplementedError(
            "Have not implemented step_size > 1 yet, reach out if wanted"
        )

    disp_total, T = raster.shape
    if weights is not None:
        assert weights.shape == (disp_total,)

    # sensible default: at most half the domain.
    disp = disp or disp_total // 2
    assert disp > 0

    # pick torch device if unset
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # range of displacements
    possible_displacement = np.arange(-disp, disp + step_size, step_size)

    # process raster into the tensors we need for conv below
    # note the transpose from DxT to TxD (batches on first dimension)
    raster = torch.as_tensor(raster, dtype=torch.float32, device=device).T
    if not normalized:
        # if we're not doing full normxcorr, we still want to keep
        # the outputs between 0 and 1
        if weights is not None:
            raster *= weights
        raster /= torch.sqrt((raster**2).sum(dim=1, keepdim=True))

    D = np.empty((T, T), dtype=np.float32)
    C = np.empty((T, T), dtype=np.float32)
    xrange = trange if pbar else range
    for i in xrange(0, T, batch_size):
        if normalized:
            corr = normxcorr1d(
                raster,
                raster[i : i + batch_size],
                weights=weights,
                padding=possible_displacement.size // 2,
            )
        else:
            corr = F.conv1d(
                raster[i : i + batch_size, None, :],
                raster[:, None, :],
                padding=possible_displacement.size // 2,
            )
        max_corr, best_disp_inds = torch.max(corr, dim=2)
        best_disp = possible_displacement[best_disp_inds.cpu()]
        D[i : i + batch_size] = best_disp
        C[i : i + batch_size] = max_corr.cpu()
        gc.collect()
        torch.cuda.empty_cache()

    return D, C


def normxcorr1d(template, x,  weights=None, padding=None):
    """normxcorr: Normalized cross-correlation, optionally weighted

    Returns the cross-correlation of `template` and `x` at spatial lags
    determined by `mode`. Useful for estimating the location of `template`
    within `x`.

    This might not be the most efficient implementation -- ideas welcome.
    It uses a direct convolutional translation of the formula
        corr = (E[XY] - EX EY) / sqrt(var X * var Y)

    This also supports weights! In that case, the usual adaptation of
    the above formula is made to the weighted case -- and all of the
    normalizations are done per block in the same way.

    Arguments
    ---------
    template : tensor, shape (num_templates, length)
        The reference template signal
    x : tensor, 1d shape (length,) or 2d shape (num_inputs, length)
        The signal in which to find `template`
    weights : tensor, shape (length,)
        Will use weighted means + variances in this case.
    padding : int, optional
        How far to look? if unset, we'll use half the length
    assume_centered : bool
        Avoid a copy if your data is centered already.

    Returns
    -------
    corr : tensor
    """
    template = torch.as_tensor(template)
    x = torch.atleast_2d(torch.as_tensor(x))
    assert x.device == template.device
    num_templates, length = template.shape
    num_inputs, length_ = template.shape
    assert length == length_

    if padding is None:
        padding = length // 2

    # generalize over weighted / unweighted case
    ones = torch.ones((1, 1, length), dtype=x.dtype, device=x.device)
    no_weights = weights is None
    if no_weights:
        weights = ones
        wt = template[:, None, :]
    else:
        assert weights.shape == (length,)
        weights = weights[None, None]
        wt = template[:, None, :] * weights

    # conv1d valid rule:
    # (B,1,L),(O,1,L)->(B,O,L)

    # compute expectations
    # how many points in each window? seems necessary to normalize
    # for numerical stability.
    N = F.conv1d(ones, weights, padding=padding)
    Et = F.conv1d(ones, wt, padding=padding) / N
    Ex = F.conv1d(x[:, None, :], weights, padding=padding) / N

    # compute (weighted) covariance
    # important: the formula E[XY] - EX EY is well-suited here,
    # because the means are naturally subtracted correctly
    # patch-wise. you couldn't pre-subtract them!
    cov = F.conv1d(x[:, None, :], wt, padding=padding) / N
    cov -= Ex * Et

    # compute variances for denominator, using var X = E[X^2] - (EX)^2
    var_template = F.conv1d(
        ones, wt * template[:, None, :], padding=padding
    )
    del wt
    var_template /= N
    var_template -= torch.square(Et)
    var_x = F.conv1d(
        torch.square(x)[:, None, :], weights, padding=padding
    )
    var_x /= N
    var_x -= torch.square(Ex)
    del Ex, Et, N

    # now find the final normxcorr
    corr = cov  # renaming for clarity
    corr /= torch.sqrt(var_x)
    corr /= torch.sqrt(var_template)
    # get rid of NaNs in zero-variance areas
    corr[~torch.isfinite(corr)] = 0

    return corr


# -- online methods: not exposed in `ibme` API yet


def calc_corr_decent_pair(
    raster_a, raster_b, disp=None, batch_size=32, step_size=1, device=None
):
    """Calculate TxT normalized xcorr and best displacement matrices
    Given a DxT raster, this computes normalized cross correlations for
    all pairs of time bins at offsets in the range [-disp, disp], by
    increments of step_size. Then it finds the best one and its
    corresponding displacement, resulting in two TxT matrices: one for
    the normxcorrs at the best displacement, and the matrix of the best
    displacements.
    Note the correlations are normalized but not centered (no mean is
    subtracted).
    Arguments
    ---------
    raster : DxT array
    batch_size : int
        How many raster rows to xcorr against the whole raster
        at once.
    step_size : int
        Displacement increment. Not implemented yet but easy to do.
    disp : int
        Maximum displacement
    device : torch device
    Returns: D, C: TxT arrays
    """
    # this is not implemented but could be done easily via stride
    if step_size > 1:
        raise NotImplementedError(
            "Have not implemented step_size > 1 yet, reach out if wanted"
        )

    D, Ta = raster_a.shape
    D_, Tb = raster_b.shape
    assert D == D_

    # sensible default: at most half the domain.
    disp = disp or D // 2
    assert disp > 0

    # pick torch device if unset
    if device is None:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    # range of displacements
    possible_displacement = np.arange(-disp, disp + step_size, step_size)

    # process rasters into the tensors we need for conv2ds below
    raster_a = torch.tensor(
        raster_a.T, dtype=torch.float32, device=device, requires_grad=False
    )
    # normalize over depth for normalized (uncentered) xcorrs
    raster_a /= torch.sqrt((raster_a**2).sum(dim=1, keepdim=True))
    image = raster_a[:, None, None, :]  # T11D - NCHW
    raster_b = torch.tensor(
        raster_b.T, dtype=torch.float32, device=device, requires_grad=False
    )
    # normalize over depth for normalized (uncentered) xcorrs
    raster_b /= torch.sqrt((raster_b**2).sum(dim=1, keepdim=True))
    weights = raster_b[:, None, None, :]  # T11D - OIHW

    D = np.empty((Ta, Tb), dtype=np.float32)
    C = np.empty((Ta, Tb), dtype=np.float32)
    for i in range(0, Ta, batch_size):
        batch = image[i : i + batch_size]
        corr = F.conv2d(  # BT1P
            batch,  # B11D
            weights,
            padding=[0, possible_displacement.size // 2],
        )
        max_corr, best_disp_inds = torch.max(corr[:, :, 0, :], dim=2)
        best_disp = possible_displacement[best_disp_inds.cpu()]
        D[i : i + batch_size] = best_disp
        C[i : i + batch_size] = max_corr.cpu()

    # free GPU memory (except torch drivers... happens when process ends)
    del (
        raster_a,
        raster_b,
        corr,
        batch,
        max_corr,
        best_disp_inds,
        image,
        weights,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return D, C


def psolveonline(D01, C01, D11, C11, p0, mincorr=0, prior_lambda=0):
    """Solves for the displacement of the new block in the online setting"""
    # subsample where corr > mincorr
    i0, j0 = np.nonzero(C01 >= mincorr)
    n0 = i0.shape[0]
    t1 = D01.shape[1]
    i1, j1 = np.nonzero(C11 >= mincorr)
    n1 = i1.shape[0]
    assert t1 == D11.shape[0]

    # construct Kroneckers
    ones0 = np.ones(n0)
    ones1 = np.ones(n1)
    U = sparse.coo_matrix((ones1, (range(n1), i1)), shape=(n1, t1))
    V = sparse.coo_matrix((ones1, (range(n1), j1)), shape=(n1, t1))
    W = sparse.coo_matrix(
        (np.sqrt(2) * ones0, (range(n0), j0)), shape=(n0, t1)
    )

    # build basic lsqr problem
    A = sparse.vstack([U - V, W]).tocsc()
    b = np.concatenate(
        [D11[i1, j1], -np.sqrt(2) * (D01 - p0[:, None])[i0, j0]]
    )

    # add in prior if requested
    if prior_lambda > 0:
        diff = sparse.diags(
            (
                np.full(t1 - 1, -prior_lambda, dtype=A.dtype),
                np.full(t1 - 1, prior_lambda, dtype=A.dtype),
            ),
            offsets=(0, 1),
            shape=(t1 - 1, t1),
        )
        A = sparse.vstack((A, diff), format="csr")
        b = np.concatenate(
            (b, np.zeros(t1 - 1)),
        )

    # solve
    p1, *_ = sparse.linalg.lsmr(A, b)

    return p1


def online_register_rigid(
    raster,
    batch_length=10000,
    time_downsample_factor=1,
    mincorr=0.7,
    disp=None,
    batch_size=32,
    device=None,
    adaptive_mincorr_percentile=None,
    prior_lambda=0,
):
    """Online rigid registration

    Lower memory and faster for large recordings.

    Returns:
    p : the vector of estimated displacements
    """
    T = raster.shape[1]

    # -- initialize
    raster0 = raster[:, 0:batch_length:time_downsample_factor]
    D00, C00 = calc_corr_decent(
        raster0,
        disp=disp,
        # pbar=False,
        batch_size=batch_size,
        device=device,
        pbar=True,
    )
    if adaptive_mincorr_percentile:
        mincorr = np.percentile(
            np.diagonal(C00, 1), adaptive_mincorr_percentile
        )
    p0 = psolvecorr(D00, C00, mincorr=mincorr, prior_lambda=prior_lambda)

    # -- loop
    ps = [p0]
    for bs in trange(batch_length, T, batch_length, desc="batches"):
        be = min(T, bs + batch_length)
        raster1 = raster[:, bs:be:time_downsample_factor]
        D01, C01 = calc_corr_decent_pair(
            raster0,
            raster1,
            disp=disp,
            batch_size=batch_size,
            device=device,
        )
        D11, C11 = calc_corr_decent(
            raster1,
            disp=disp,
            pbar=False,
            batch_size=batch_size,
            device=device,
        )
        if adaptive_mincorr_percentile:
            mincorr = np.percentile(
                np.diagonal(C11, 1), adaptive_mincorr_percentile
            )
        p1 = psolveonline(
            D01, C01, D11, C11, p0, mincorr=mincorr, prior_lambda=prior_lambda
        )
        ps.append(p1)

        # update loop variables
        raster0 = raster1
        D00 = D11
        C00 = C11
        p0 = p1

    p = np.concatenate(ps)
    return p
