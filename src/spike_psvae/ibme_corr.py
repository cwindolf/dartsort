import gc
import numpy as np
import scipy.linalg as la
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.optimize import minimize
from scipy.stats import zscore
from tqdm.auto import trange

from .motion_utils import get_motion_estimate, get_bins


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
    soft_weights=True,
):
    """Solve for rigid displacement given pairwise disps + corrs"""
    T = D.shape[0]
    assert (T, T) == D.shape == C.shape

    # subsample where corr > mincorr
    S = C * (C >= mincorr)
    if max_dt is not None and max_dt > 0:
        S *= la.toeplitz(
            np.r_[
                np.ones(max_dt, dtype=S.dtype),
                np.zeros(T - max_dt, dtype=S.dtype),
            ]
        )
    I, J = np.where(S > 0)
    n_sampled = I.shape[0]

    # construct Kroneckers
    if soft_weights:
        pair_weights = S[I, J]
    else:
        pair_weights = np.ones(n_sampled)
    M = sparse.csr_matrix(
        (pair_weights, (range(n_sampled), I)), shape=(n_sampled, T)
    )
    N = sparse.csr_matrix(
        (pair_weights, (range(n_sampled), J)), shape=(n_sampled, T)
    )
    A = M - N
    V = pair_weights * D[I, J]

    # add in our prior p_{i+1} - p_i ~ N(0, lambda) by extending the problem
    if prior_lambda > 0:
        diff = sparse.diags(
            (
                np.full(T - 1, -prior_lambda/2, dtype=A.dtype),
                np.full(T, prior_lambda, dtype=A.dtype),
                np.full(T - 1, -prior_lambda/2, dtype=A.dtype),
            ),
            offsets=(-1, 0, 1),
            shape=(T, T),
        )
        A = sparse.vstack((A, diff), format="csr")
        V = np.concatenate(
            (V, np.zeros(T)),
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
    soft_weights=True,
):
    # D = pairwise_displacement
    W = (C > mincorr) * C

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
                np.ones(max_dt, dtype=W.dtype),
                np.zeros(T - max_dt, dtype=W.dtype),
            ]
        )
        for k in range(W.shape[0]):
            W[k] *= horiz

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
        if soft_weights:
            pair_weights = Wb[I, J]
        else:
            pair_weights = np.ones(n_sampled)
        Mb = sparse.csr_matrix(
            (pair_weights, (range(n_sampled), I)), shape=(n_sampled, T)
        )
        Nb = sparse.csr_matrix(
            (pair_weights, (range(n_sampled), J)), shape=(n_sampled, T)
        )
        block_sparse_kron = Mb - Nb
        block_disp_pairs = pair_weights * Db[I, J]

        # add the temporal smoothness prior in this window
        if temporal_prior:
            print(block_sparse_kron.dtype)
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
            print(f"{block_sparse_kron.shape=} {block_disp_pairs.shape=}")

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
    
    return coefficients, targets, p0

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
    centered=True,
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

    D = np.empty((T, T), dtype=np.float32)
    C = np.empty((T, T), dtype=np.float32)
    xrange = trange if pbar else range
    for i in xrange(0, T, batch_size):
        corr = normxcorr1d(
            raster,
            raster[i : i + batch_size],
            weights=weights,
            padding=possible_displacement.size // 2,
            normalized=normalized,
            centered=centered,
        )
        max_corr, best_disp_inds = torch.max(corr, dim=2)
        best_disp = possible_displacement[best_disp_inds.cpu()]
        D[i : i + batch_size] = best_disp
        C[i : i + batch_size] = max_corr.cpu()

    return D, C


def normxcorr1d(
    template,
    x,
    weights=None,
    centered=True,
    normalized=True,
    padding="same",
    conv_engine="torch",
):
    """normxcorr1d: Normalized cross-correlation, optionally weighted

    The API is like torch's F.conv1d, except I have accidentally
    changed the position of input/weights -- template acts like weights,
    and x acts like input.

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
        Will use weighted means, variances, covariances if supplied.
    centered : bool
        If true, means will be subtracted (per weighted patch).
    normalized : bool
        If true, normalize by the variance (per weighted patch).
    padding : int, optional
        How far to look? if unset, we'll use half the length
    conv_engine : string, one of "torch", "numpy"
        What library to use for computing cross-correlations.
        If numpy, falls back to the scipy correlate function.

    Returns
    -------
    corr : tensor
    """
    if conv_engine == "torch":
        conv1d = F.conv1d
        npx = torch
    elif conv_engine == "numpy":
        conv1d = scipy_conv1d
        npx = np
    else:
        raise ValueError(f"Unknown conv_engine {conv_engine}")

    x = npx.atleast_2d(x)
    num_templates, lengtht = template.shape
    num_inputs, lengthx = x.shape

    # generalize over weighted / unweighted case
    device_kw = {} if conv_engine == "numpy" else dict(device=x.device)
    onesx = npx.ones((1, 1, lengthx), dtype=x.dtype, **device_kw)
    no_weights = weights is None
    if no_weights:
        weights = npx.ones((1, 1, lengtht), dtype=x.dtype, **device_kw)
        wt = template[:, None, :]
    else:
        assert weights.shape == (lengtht,)
        weights = weights[None, None]
        wt = template[:, None, :] * weights

    # conv1d valid rule:
    # (B,1,L),(O,1,L)->(B,O,L)

    # compute expectations
    # how many points in each window? seems necessary to normalize
    # for numerical stability.
    Nx = conv1d(onesx, weights, padding=padding)
    if centered:
        Et = conv1d(onesx, wt, padding=padding)
        Et /= Nx
        Ex = conv1d(x[:, None, :], weights, padding=padding)
        Ex /= Nx

    # compute (weighted) covariance
    # important: the formula E[XY] - EX EY is well-suited here,
    # because the means are naturally subtracted correctly
    # patch-wise. you couldn't pre-subtract them!
    cov = conv1d(x[:, None, :], wt, padding=padding)
    cov /= Nx
    if centered:
        cov -= Ex * Et

    # compute variances for denominator, using var X = E[X^2] - (EX)^2
    if normalized:
        var_template = conv1d(
            onesx, wt * template[:, None, :], padding=padding
        )
        var_template /= Nx
        var_x = conv1d(npx.square(x)[:, None, :], weights, padding=padding)
        var_x /= Nx
        if centered:
            var_template -= npx.square(Et)
            var_x -= npx.square(Ex)

        # fill in zeros to avoid problems when dividing
        var_template[var_template == 0] = 1
        var_x[var_x == 0] = 1

    # now find the final normxcorr
    corr = cov  # renaming for clarity
    if normalized:
        corr /= npx.sqrt(var_x)
        corr /= npx.sqrt(var_template)

    return corr


def normxcorr2d(
    template,
    x,
    weights=None,
    centered=True,
    normalized=True,
    padding=0,
):
    """normxcorr1d: Normalized cross-correlation, optionally weighted

    The API is like torch's F.conv1d, except I have accidentally
    changed the position of input/weights -- template acts like weights,
    and x acts like input.

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
        Will use weighted means, variances, covariances if supplied.
    centered : bool
        If true, means will be subtracted (per weighted patch).
    normalized : bool
        If true, normalize by the variance (per weighted patch).
    padding : int, optional
        How far to look? if unset, we'll use half the length
    conv_engine : string, one of "torch", "numpy"
        What library to use for computing cross-correlations.
        If numpy, falls back to the scipy correlate function.

    Returns
    -------
    corr : tensor
    """
    x = torch.atleast_2d(x)
    num_templates, lengths, lengtht = template.shape
    num_inputs, lengths_, lengthx = x.shape
    assert lengths == lengths_
    padding = (0, padding)

    # generalize over weighted / unweighted case
    device_kw = dict(device=x.device)
    onesx = torch.ones((1, 1, lengths, lengthx), dtype=x.dtype, **device_kw)
    no_weights = weights is None
    if no_weights:
        weights = torch.ones((1, 1, lengths, lengtht), dtype=x.dtype, **device_kw)
        wt = template[:, None, :, :]
    else:
        assert weights.shape == (lengtht,)
        weights = weights[None, None, None] * torch.ones((1, 1, lengthx, 1))
        wt = template[:, None, :, :] * weights

    # conv1d valid rule:
    # (B,1,L),(O,1,L)->(B,O,L)

    # compute expectations
    # how many points in each window? seems necessary to normalize
    # for numerical stability.
    print(f"{onesx.shape=} {weights.shape=}")
    Nx = F.conv2d(onesx, weights, padding=padding)
    print(f"{Nx.shape=}")
    if centered:
        Et = F.conv2d(onesx, wt, padding=padding)
        print(f"{Et.shape=}")
        Et /= Nx
        Ex = F.conv2d(x[:, None, :], weights, padding=padding)
        print(f"{Ex.shape=}")
        Ex /= Nx

    # compute (weighted) covariance
    # important: the formula E[XY] - EX EY is well-suited here,
    # because the means are naturally subtracted correctly
    # patch-wise. you couldn't pre-subtract them!
    cov = F.conv2d(x[:, None, :], wt, padding=padding)
    cov /= Nx
    if centered:
        cov -= Ex * Et

    # compute variances for denominator, using var X = E[X^2] - (EX)^2
    if normalized:
        var_template = F.conv2d(
            onesx, wt * template[:, None, :], padding=padding
        )
        var_template /= Nx
        var_x = F.conv2d(torch.square(x)[:, None, :], weights, padding=padding)
        var_x /= Nx
        if centered:
            var_template -= torch.square(Et)
            var_x -= torch.square(Ex)

        # fill in zeros to avoid problems when dividing
        var_template[var_template == 0] = 1
        var_x[var_x == 0] = 1

    # now find the final normxcorr
    corr = cov  # renaming for clarity
    if normalized:
        corr /= torch.sqrt(var_x)
        corr /= torch.sqrt(var_template)

    assert corr.shape[2] == 1
    corr = corr[:, :, 0, :]

    return corr


def scipy_conv1d(input, weights, padding="valid"):
    """SciPy translation of torch F.conv1d"""
    from scipy.signal import correlate

    n, c_in, length = input.shape
    c_out, in_by_groups, kernel_size = weights.shape
    assert in_by_groups == c_in == 1

    if padding == "same":
        mode = "same"
        length_out = length
    elif padding == "valid":
        mode = "valid"
        length_out = length - 2 * (kernel_size // 2)
    elif isinstance(padding, int):
        mode = "valid"
        input = np.pad(
            input, [*[(0, 0)] * (input.ndim - 1), (padding, padding)]
        )
        length_out = length - (kernel_size - 1) + 2 * padding
    else:
        raise ValueError(f"Unknown padding {padding}")

    output = np.zeros((n, c_out, length_out), dtype=input.dtype)
    for m in range(n):
        for c in range(c_out):
            output[m, c] = correlate(input[m, 0], weights[c, 0], mode=mode)

    return output


# -- online methods: not exposed in `ibme` API yet


def calc_corr_decent_pair(
    raster_a,
    raster_b,
    weights=None,
    disp=None,
    batch_size=32,
    normalized=True,
    centered=True,
    possible_displacement=None,
    device=None,
):
    """Calculate TxT normalized xcorr and best displacement matrices
    Given a DxT raster, this computes normalized cross correlations for
    all pairs of time bins at offsets in the range [-disp, disp], by
    increments of step_size. Then it finds the best one and its
    corresponding displacement, resulting in two TxT matrices: one for
    the normxcorrs at the best displacement, and the matrix of the best
    displacements.

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
    D, Ta = raster_a.shape
    D_, Tb = raster_b.shape

    # sensible default: at most half the domain.
    if disp is None:
        disp == D // 2

    # range of displacements
    if D == D_:
        if possible_displacement is None:
            possible_displacement = np.arange(-disp, disp + 1)
    else:
        assert possible_displacement is not None
        assert disp is not None

    # pick torch device if unset
    if device is None:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    # process rasters into the tensors we need for conv2ds below
    raster_a = torch.as_tensor(raster_a, dtype=torch.float32, device=device).T
    # normalize over depth for normalized (uncentered) xcorrs
    raster_b = torch.as_tensor(raster_b, dtype=torch.float32, device=device).T

    D = np.empty((Ta, Tb), dtype=np.float32)
    C = np.empty((Ta, Tb), dtype=np.float32)
    for i in range(0, Tb, batch_size):
        corr = normxcorr1d(
            raster_a,
            raster_b[i : i + batch_size],
            weights=weights,
            padding=disp,
            normalized=normalized,
            centered=centered,
        )
        max_corr, best_disp_inds = torch.max(corr, dim=2)
        best_disp = possible_displacement[best_disp_inds.cpu()]
        D[:, i : i + batch_size] = best_disp.T
        C[:, i : i + batch_size] = max_corr.cpu().T

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


# -- newton stuff

