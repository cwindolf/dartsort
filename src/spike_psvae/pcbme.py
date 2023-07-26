import numpy as np
from scipy import sparse
from scipy.stats import zscore
from scipy.spatial.distance import pdist, squareform


def kronsolve(D, robust_sigma=0):
    T, T_ = D.shape
    assert T == T_

    if robust_sigma > 0:
        Dz = zscore(D, axis=1)
        S = (np.abs(Dz) < robust_sigma).astype(D.dtype)

    _1 = np.ones((T, 1))
    Id = sparse.identity(T)
    kron = sparse.kron(Id, _1).tocsr() - sparse.kron(_1, Id).tocsr()

    if robust_sigma <= 0:
        p, *_ = sparse.linalg.lsqr(kron, D.ravel())
    else:
        dvS = sparse.diags(S.ravel())
        p, *_ = sparse.linalg.lsqr(dvS @ kron, dvS @ D.ravel())

    return p


def psolve(D, S, error, robust_sigma=0, time_sigma=1.0, error_sigma=0.1):

    error_S = error[np.where(S != 0)]
    W1 = np.exp(
        -(
            (error_S - error_S.min())
            / (error_S.max() - error_S.min())
        )
        / error_sigma
    )

    W2 = np.exp(
        -squareform(pdist(np.arange(error.shape[0])[:, None]))
        / time_sigma
    )
    W2 = W2[np.where(S != 0)]

    W = (W2 * W1)[:, None]

    I, J = np.where(S != 0)
    V = displacement_matrix[np.where(S != 0)]
    M = csr_matrix((np.ones(I.shape[0]), (np.arange(I.shape[0]), I)))
    N = csr_matrix((np.ones(I.shape[0]), (np.arange(I.shape[0]), J)))
    A = M - N
    idx = np.ones(A.shape[0]).astype(bool)
    for i in tqdm(range(robust_regression_n_iters)):
        p = lsqr(A[idx].multiply(W[idx]), V[idx] * W[idx][:, 0])[0]
        idx = np.where(np.abs(zscore(A @ p - V)) <= robust_regression_sigma)
    ps = p
