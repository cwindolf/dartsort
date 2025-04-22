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


def test_topk_sparse():
    rg = np.random.default_rng(0)
    ncols = 100
    nrows = 10
    k = nrows // 2

    # we'll only ever insert on half the columns just for fun
    cols_valid = np.arange(0, ncols, 2)

    # allocate topk sparse array
    topk = sparse_util.allocate_topk(ncols, k)

    csc = sparse_util.topk_sparse_tocsc(topk, nrows)
    assert (csc.todense() == 0.0).all()

    # insert 5 big rows
    for j in range(nrows // 2 + 1):
        rowdata = 100 + rg.normal(size=cols_valid.size).astype("float32")
        sparse_util.topk_sparse_insert(j, cols_valid, rowdata, topk)

        # check it now...
        csc = sparse_util.topk_sparse_tocsc(topk, nrows)
        assert csc.shape == (nrows, ncols)
        assert csc.nnz == min(k, (j + 1)) * cols_valid.size
        # this one below is stochastic but should succeed whp
        if ncols > 80:
            assert np.array_equal(np.unique(csc.indices), np.arange(j + 1))
        counts = np.zeros(ncols, dtype=int)
        counts[cols_valid] = min(k, j + 1)
        assert np.array_equal(csc.indptr, np.concatenate([[0], np.cumsum(counts)]))
    csc_last = csc

    # insert a small row and check it did nothing
    rowdata = -100 + rg.normal(size=cols_valid.size).astype("float32")
    sparse_util.topk_sparse_insert(nrows - 1, cols_valid, rowdata, topk)
    csc = sparse_util.topk_sparse_tocsc(topk, nrows)

    assert np.array_equal(csc_last.data, csc.data)
    assert np.array_equal(csc_last.indptr, csc.indptr)
    assert csc_last.nnz == csc.nnz

    assert csc.shape == (nrows, ncols)
    assert csc.nnz == k * cols_valid.size
    # this one below is stochastic but should succeed whp
    if ncols > 80:
        assert np.array_equal(np.unique(csc.indices), np.arange(j + 1))
    counts = np.zeros(ncols, dtype=int)
    counts[cols_valid] = k
    assert np.array_equal(csc.indptr, np.concatenate([[0], np.cumsum(counts)]))

    # insert the extra row
    extra_row = np.full((ncols,), 1000.0, dtype="float32")
    csc = sparse_util.topk_sparse_tocsc(topk, nrows, extra_row)
    assert csc.shape == (nrows + 1, ncols)
    assert csc.nnz == k * cols_valid.size + ncols
    # this one below is stochastic but should succeed whp
    if ncols > 80:
        assert np.array_equal(
            np.unique(csc.indices), np.concatenate([np.arange(j + 1), [nrows]])
        )
    counts = np.ones(ncols, dtype=int)
    counts[cols_valid] += k
    assert np.array_equal(csc.indptr, np.concatenate([[0], np.cumsum(counts)]))
    dns = csc.todense()
    assert np.all(dns.max(axis=0) == 1000.0)
    assert dns.max(1)[-1] == 1000.0
    assert np.all(dns.max(1)[:-1] < 150.0)
    assert dns.min(1)[-1] == 1000.0


def test_topk_sparse_sparse():
    rg = np.random.default_rng(0)
    ncols = 100
    nrows = 10
    k = nrows // 2

    # we'll only ever insert on half the columns just for fun
    cols_valid = np.arange(0, ncols, 2)
    ins_inds = np.arange(cols_valid.size)

    # allocate topk sparse array
    topk = sparse_util.allocate_topk(cols_valid.size, k)

    csc = sparse_util.topk_sparse_tocsc(topk, nrows)
    assert (csc.todense() == 0.0).all()

    # insert 5 big rows
    for j in range(nrows // 2 + 1):
        rowdata = 100 + rg.normal(size=cols_valid.size).astype("float32")
        sparse_util.topk_sparse_insert(j, ins_inds, rowdata, topk)

        # check it now...
        csc = sparse_util.topk_sparse_tocsc(
            topk, nrows, column_support=cols_valid, n_columns_full=ncols
        )
        assert csc.shape == (nrows, ncols)
        assert csc.nnz == min(k, (j + 1)) * cols_valid.size
        # this one below is stochastic but should succeed whp
        if ncols > 80:
            assert np.array_equal(np.unique(csc.indices), np.arange(j + 1))
        counts = np.zeros(ncols, dtype=int)
        counts[cols_valid] = min(k, j + 1)
        assert np.array_equal(csc.indptr, np.concatenate([[0], np.cumsum(counts)]))
    csc_last = csc

    # insert a small row and check it did nothing
    rowdata = -100 + rg.normal(size=cols_valid.size).astype("float32")
    sparse_util.topk_sparse_insert(nrows - 1, ins_inds, rowdata, topk)
    csc = sparse_util.topk_sparse_tocsc(
        topk, nrows, column_support=cols_valid, n_columns_full=ncols
    )

    assert np.array_equal(csc_last.data, csc.data)
    assert np.array_equal(csc_last.indptr, csc.indptr)
    assert csc_last.nnz == csc.nnz

    assert csc.shape == (nrows, ncols)
    assert csc.nnz == k * cols_valid.size
    # this one below is stochastic but should succeed whp
    if ncols > 80:
        assert np.array_equal(np.unique(csc.indices), np.arange(j + 1))
    counts = np.zeros(ncols, dtype=int)
    counts[cols_valid] = k
    assert np.array_equal(csc.indptr, np.concatenate([[0], np.cumsum(counts)]))

    # insert the extra row
    extra_row = np.full((cols_valid.size,), 1000.0, dtype="float32")
    csc = sparse_util.topk_sparse_tocsc(
        topk, nrows, extra_row, column_support=cols_valid, n_columns_full=ncols
    )
    assert csc.shape == (nrows + 1, ncols)
    assert csc.nnz == k * cols_valid.size + cols_valid.size
    # this one below is stochastic but should succeed whp
    if ncols > 80:
        assert np.array_equal(
            np.unique(csc.indices), np.concatenate([np.arange(j + 1), [nrows]])
        )
    counts = np.zeros(ncols, dtype=int)
    counts[cols_valid] += k + 1
    assert np.array_equal(csc.indptr, np.concatenate([[0], np.cumsum(counts)]))
    dns = csc.todense()
    assert np.array_equal(
        np.unique(dns.max(axis=0)), np.array([0.0, 1000], dtype="float32")
    )
    assert dns.max(1)[-1] == 1000.0
    assert np.all(dns.max(1)[:-1] < 150.0)
    assert dns.min(1)[-1] == 0.0


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


def test_double_searchsorted():
    rg = np.random.default_rng(0)

    a = np.arange(10)
    v = np.array([0, 1, 3])
    i = sparse_util.double_searchsorted(a, v)
    assert np.array_equal(a[i], v)
    assert np.array_equal(i, v)

    a = np.arange(100)
    v = rg.choice(100, size=50, replace=False)
    v.sort()
    i = sparse_util.double_searchsorted(a, v)
    assert np.array_equal(i, v)
    assert np.array_equal(a[i], v)

    a = rg.choice(1000, size=100, replace=False)
    a.sort()
    v = rg.choice(a, size=50, replace=False)
    v.sort()
    i = sparse_util.double_searchsorted(a, v)
    assert np.array_equal(a[i], v)


if __name__ == "__main__":
    test_topk_sparse()
    # test_csc_getrow()
    # test_csc_insert()
    # test_csc_mask()
