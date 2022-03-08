import gc
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from tqdm.auto import trange


def register_rigid(
    raster,
    mincorr=0.7,
    disp=None,
    batch_size=32,
    step_size=1,
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
    batch_size, step_size : int
        See `calc_corr_decent`
    Returns: p, array (T,)
    """
    D, C = calc_corr_decent(
        raster,
        disp=disp,
        batch_size=batch_size,
        step_size=step_size,
    )
    p = psolvecorr(D, C, mincorr=mincorr)
    return p


def online_register_rigid(
    raster,
    batch_length=10000,
    time_downsample_factor=1,
    mincorr=0.7,
    disp=None,
    csd=False,
    channels=slice(None),
):
    T = raster.shape[1]

    # -- initialize
    raster0 = raster[0:batch_length:time_downsample_factor]
    D00, C00 = calc_corr_decent(
        raster0,
        disp=disp,
        pbar=False,
    )
    p0 = psolvecorr(D00, C00, mincorr=mincorr)

    # -- loop
    ps = [p0]
    for bs in trange(batch_length, T, batch_length, desc="batches"):
        be = min(T, bs + batch_length)
        raster1 = raster[bs:be:time_downsample_factor]
        D01, C01 = calc_corr_decent_pair(raster0, raster1, disp=disp)
        D11, C11 = calc_corr_decent(
            raster1,
            disp=disp,
            pbar=False,
        )
        p1 = psolveonline(D01, C01, D11, C11, p0, mincorr)
        ps.append(p1)

        # update loop variables
        raster0 = raster1
        D00 = D11
        C00 = C11
        p0 = p1

    p = np.concatenate(ps)
    return p


def psolveonline(D01, C01, D11, C11, p0, mincorr):
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
    W = sparse.coo_matrix((ones0, (range(n0), j0)), shape=(n0, t1))

    # build lsqr problem
    A = sparse.vstack([U - V, W]).tocsc()
    b = np.concatenate([D11[i1, j1], -(D01 - p0[:, None])[i0, j0]])
    p1, *_ = sparse.linalg.lsmr(A, b)
    return p1


def psolvecorr(D, C, mincorr=0.7):
    """Solve for rigid displacement given pairwise disps + corrs"""
    T = D.shape[0]
    assert (T, T) == D.shape == C.shape

    # subsample where corr > mincorr
    S = C >= mincorr
    I, J = np.where(S == 1)
    n_sampled = I.shape[0]

    # construct Kroneckers
    ones = np.ones(n_sampled)
    M = sparse.csr_matrix((ones, (range(n_sampled), I)), shape=(n_sampled, T))
    N = sparse.csr_matrix((ones, (range(n_sampled), J)), shape=(n_sampled, T))

    # solve sparse least squares problem
    p, *_ = sparse.linalg.lsqr(M - N, D[I, J])
    return p


def calc_corr_decent(
    raster, disp=None, batch_size=32, step_size=1, device=None, pbar=True
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

    D, T = raster.shape

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

    # process raster into the tensors we need for conv2ds below
    raster = torch.tensor(
        raster.T, dtype=torch.float32, device=device, requires_grad=False
    )
    # normalize over depth for normalized (uncentered) xcorrs
    raster /= torch.sqrt((raster ** 2).sum(dim=1, keepdim=True))
    # conv weights
    image = raster[:, None, None, :]  # T11D - NCHW
    weights = image  # T11D - OIHW

    D = np.empty((T, T), dtype=np.float32)
    C = np.empty((T, T), dtype=np.float32)
    xrange = trange if pbar else range
    for i in xrange(0, T, batch_size):
        batch = image[i:i + batch_size]
        corr = F.conv2d(  # BT1P
            batch,  # B11D
            weights,
            padding=[0, possible_displacement.size // 2],
        )
        max_corr, best_disp_inds = torch.max(corr[:, :, 0, :], dim=2)
        best_disp = possible_displacement[best_disp_inds.cpu()]
        D[i:i + batch_size] = best_disp
        C[i:i + batch_size] = max_corr.cpu()

    # free GPU memory (except torch drivers... happens when process ends)
    del raster, corr, batch, max_corr, best_disp_inds, image, weights
    gc.collect()
    torch.cuda.empty_cache()

    return D, C


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
    raster_a /= torch.sqrt((raster_a ** 2).sum(dim=1, keepdim=True))
    image = raster_a[:, None, None, :]  # T11D - NCHW
    raster_b = torch.tensor(
        raster_b.T, dtype=torch.float32, device=device, requires_grad=False
    )
    # normalize over depth for normalized (uncentered) xcorrs
    raster_b /= torch.sqrt((raster_b ** 2).sum(dim=1, keepdim=True))
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
