import numpy as np
from scipy.sparse import csc_array, coo_array
from dartsort.util import sparse_util


shape = (512, 1024)
nnz = 30 * 1024


def test_csc_insert():
    rg = np.random.default_rng(10)
    ij = rg.integers(low=((0, 0),), high=(shape,), size=(nnz, 2))
    ij = np.unique(ij, axis=0)
    assert (np.diff(ij[:, 0]) >= 0).all()
    assert not (np.diff(ij[:, 1]) >= 0).all()
    vals = rg.normal(size=len(ij)).astype(np.float32)

    x0 = coo_array((vals, ij.T), shape).tocsc()
    assert x0.has_canonical_format

    ii, jj = ij.T
    jj_order = np.argsort(jj, kind="stable")
    ii_sorted = ii[jj_order]
    jj_sorted = jj[jj_order]
    vals_sorted = vals[jj_order]

    jj_unique, jj_inv, jj_count = np.unique(
        jj_sorted, return_inverse=True, return_counts=True
    )
    bincounts = np.zeros(jj_unique.max() + 1, dtype=jj.dtype)
    np.add.at(bincounts, jj_sorted, 1)

    indptr = np.concatenate([[0], np.cumsum(bincounts)])
    write_offsets = indptr[:-1].copy()

    row_inds, data = sparse_util.get_csc_storage(len(ij), None, False)

    for i in range(shape[0]):
        in_row = np.flatnonzero(ii_sorted == i)
        inds = jj_sorted[in_row]
        sparse_util.csc_insert(
            i, write_offsets, inds, row_inds, data, vals_sorted[in_row]
        )

    x1 = csc_array((data, row_inds, indptr), shape)
    assert x1.nnz == x0.nnz
    assert np.array_equal(x1.indptr, x0.indptr)
    assert np.array_equal(x1.indices, x0.indices)
    assert np.array_equal(x1.data, x0.data)


def test_csc_mask():
    rg = np.random.default_rng(10)
    ij = rg.integers(low=((0, 0),), high=(shape,), size=(nnz, 2))
    ij = np.unique(ij, axis=0)
    assert (np.diff(ij[:, 0]) >= 0).all()
    assert not (np.diff(ij[:, 1]) >= 0).all()
    vals = rg.normal(size=len(ij)).astype(np.float32)

    x = coo_array((vals, ij.T), shape).tocsc()

    masks = [np.zeros(shape[0], dtype=bool), np.ones(shape[0], dtype=bool)]
    masks.append(masks[-1].copy())
    masks.append(masks[-1].copy())
    masks[-2][0] = 0
    masks[-1][-1] = 0

    for _ in range(10):
        p = rg.beta(1, 1)
        masks.append(rg.binomial(1, p, size=shape[0]).astype(bool))

    for mask in masks:
        kept_rows = np.flatnonzero(mask)
        x0 = x[kept_rows]
        x1 = sparse_util.csc_sparse_mask_rows(x, mask)

        assert x1.nnz == x0.nnz
        assert np.array_equal(x1.indptr, x0.indptr)
        assert np.array_equal(x1.indices, x0.indices)
        assert np.array_equal(x1.data, x0.data)


def test_csc_getrow():
    rg = np.random.default_rng(10)
    ij = rg.integers(low=((0, 0),), high=(shape,), size=(nnz, 2))
    ij = np.unique(ij, axis=0)
    assert (np.diff(ij[:, 0]) >= 0).all()
    assert not (np.diff(ij[:, 1]) >= 0).all()
    vals = rg.normal(size=len(ij)).astype(np.float32)

    x = coo_array((vals, ij.T), shape).tocsc()

    for row in range(x.shape[0]):
        x0 = x[[row]]
        columns, data = sparse_util.csc_sparse_getrow(x, row, x0.nnz)

        assert len(columns) == len(data) == x0.nnz
        x0coo = x0.tocoo()
        assert np.array_equal(columns, x0coo.coords[1])
        assert np.array_equal(data, x0coo.data)


if __name__ == "__main__":
    test_csc_getrow()
    test_csc_insert()
    test_csc_mask()
