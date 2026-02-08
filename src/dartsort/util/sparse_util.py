import torch
from scipy.sparse import coo_array

try:
    import cupy as cp  # type: ignore
    from cupyx.scipy.sparse import coo_matrix as cupy_coo_matrix  # type: ignore

except ImportError:
    cp = None
    cupy_coo_matrix = lambda a, shape: NotImplemented


def coo_to_scipy(coo_tensor):
    data = coo_tensor.values().numpy(force=True)
    coords = coo_tensor.indices().numpy(force=True)
    return coo_array((data, coords), shape=coo_tensor.shape)


def coo_to_cupy(coo_tensor):
    assert cp is not None
    data = cp.asarray(coo_tensor.values())
    iijj = cp.asarray(coo_tensor.indices())
    return cupy_coo_matrix((data, iijj), shape=coo_tensor.shape)


# sparse kmeans helpers


def sparse_centroid_distsq(X, centroids, labels, centroid_mask, dbufs):
    neighbors = centroid_mask[labels]
    coo = neighbors.nonzero()
    ii, cc = coo.T
    nn = len(ii)

    dbufx, dbufc = dbufs
    dbufx = dbufx.resize_(nn, *dbufx.shape[1:])
    dbufc = dbufc.resize_(nn, *dbufc.shape[1:])

    torch.index_select(X, dim=0, index=ii, out=dbufx)
    torch.index_select(centroids, dim=0, index=cc, out=dbufc)
    dsq = dbufx.sub_(dbufc).square_().sum(dim=1)

    distsq_coo = torch.sparse_coo_tensor(
        coo.T, dsq, size=(len(X), len(centroids)), is_coalesced=True
    )

    return distsq_coo, (dbufx, dbufc)


def distsq_to_lik_coo(distsq_coo, sigmasq, log_proportions, in_place=False):
    liks = distsq_coo
    if not in_place:
        liks = liks.clone()
    liks.values().mul_(-0.5 / sigmasq).add_(log_proportions[liks.indices()[1]])
    return liks


def logsumexp_coo(coo):
    """Like torch.sparse.softmax, assumes non-explicit 0s are -inf

    out should be initialized to -inf. Uses stable logsumexp trick.
    """
    v = coo.values()
    i = coo.indices()[0]

    # first, put the max values into out
    max_values = v.new_full((coo.shape[0],), -torch.inf)
    logsumexps = v.new_zeros((coo.shape[0],))
    max_values.scatter_reduce_(dim=0, index=i, src=v, reduce="amax")

    # now stable exponential
    sv = max_values[i]
    torch.subtract(v, sv, out=sv)
    sv.exp_()

    # add stable exps, take log, add back max vals
    logsumexps.scatter_add_(dim=0, index=i, src=sv)
    logsumexps.log_()
    logsumexps += max_values

    return logsumexps
