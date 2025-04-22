import numpy as np
import torch
import numba
from scipy.sparse import coo_array, csc_array

from scipy.special import logsumexp


def get_coo_storage(ns_total, storage, use_storage):
    if not use_storage:
        # coo_uix = np.empty(ns_total, dtype=int)
        coo_six = np.empty(ns_total, dtype=int)
        coo_data = np.empty(ns_total, dtype=np.float32)
        return coo_six, coo_data

    if hasattr(storage, "coo_data"):
        if storage.coo_data.size < ns_total:
            # del storage.coo_uix
            del storage.coo_six
            del storage.coo_data
        # storage.coo_uix = np.empty(ns_total, dtype=int)
        storage.coo_six = np.empty(ns_total, dtype=int)
        storage.coo_data = np.empty(ns_total, dtype=np.float32)
    else:
        # storage.coo_uix = np.empty(ns_total, dtype=int)
        storage.coo_six = np.empty(ns_total, dtype=int)
        storage.coo_data = np.empty(ns_total, dtype=np.float32)

    # return storage.coo_uix, storage.coo_six, storage.coo_data
    return storage.coo_six, storage.coo_data


def torch_coo_to_dense(coo_array, fill_value):
    data = coo_array.data()
    out = data.new_full(coo_array.shape, fill_value)
    out[coo_array.indices()] = data
    return out


def coo_to_torch(
    coo_array, dtype, transpose=False, is_coalesced=False, copy_data=False
):
    coo = (
        torch.from_numpy(coo_array.coords[int(transpose)]),
        torch.from_numpy(coo_array.coords[1 - int(transpose)]),
    )
    s0, s1 = coo_array.shape
    if transpose:
        s0, s1 = s1, s0
    res = torch.sparse_coo_tensor(
        torch.row_stack(coo),
        torch.asarray(coo_array.data, dtype=torch.float, copy=copy_data),
        size=(s0, s1),
        is_coalesced=is_coalesced,
    )
    if not is_coalesced:
        res = res.coalesce()
    return res


def coo_to_scipy(coo_tensor):
    data = coo_tensor.values().numpy(force=True)
    coords = coo_tensor.indices().numpy(force=True)
    return coo_array((data, coords), shape=coo_tensor.shape)


def get_csc_storage(ns_total, storage, use_storage):
    if not use_storage:
        csc_row_indices = np.empty(ns_total, dtype=int)
        csc_data = np.empty(ns_total, dtype=np.float32)
        return csc_row_indices, csc_data

    if hasattr(storage, "csc_data"):
        if storage.csc_data.size < ns_total:
            del storage.csc_row_indices
            del storage.csc_data
        storage.csc_row_indices = np.empty(ns_total, dtype=int)
        storage.csc_data = np.empty(ns_total, dtype=np.float32)
    else:
        storage.csc_row_indices = np.empty(ns_total, dtype=int)
        storage.csc_data = np.empty(ns_total, dtype=np.float32)

    return storage.csc_row_indices, storage.csc_data


sig = "void(i8, i8[::1], i8[::1], i8[::1], f4[::1], f4[::1])"


@numba.njit(sig, error_model="numpy", nogil=True, parallel=True)
def csc_insert(row, write_offsets, inds, csc_indices, csc_data, liks):
    """Insert elements into a CSC sparse array

    To use this, you need to know the indptr in advance. Then this function
    can help you to insert a row into the array. You have to insert all nz
    elements for that row at once in a single call to this function, and
    rows must be written in order.

    (However, the columns within each row can be unordered.)

    write_offsets should be initialized at the indptr -- so, they are
    each column's "write head", indicating how many rows have been written
    for that column so far.

    Then, this updates the row indices (csc_indices) array with the correct
    row for each column (inds), and adds the data in the right place. The
    "write heads" are incremented, so that when this fn is called for the next
    row things are in the right place.

    This would be equivalent to:
        data_ixs = write_offsets[inds]
        csc_indices[data_ixs] = row
        csc_data[data_ixs] = liks
        write_offsets[inds] += 1
    """
    for j in numba.prange(inds.shape[0]):
        ind = inds[j]
        data_ix = write_offsets[ind]
        csc_indices[data_ix] = row
        csc_data[data_ix] = liks[j]
        write_offsets[ind] += 1


def allocate_topk(n_columns, k):
    data = np.full((n_columns, k), -np.inf, dtype="float32")
    row_indices = np.full((n_columns, k), -1, dtype=int)
    return row_indices, data


def topk_sparse_insert(row, row_column_indices, row_data, topk):
    """topk sparse array builder

    A topk sparse array is something close to a CSC array. It is a tuple
    (data, row_indices), where data is n_columns x k and row_indices is
    n_columns x k, and these should be contiguous on the inner axis.

    It has the following invariants:
     - data is sorted in decreasing order on the inner (1th) axis
       (the k dimension)
     - row_indices can be <0, in which case the corresponding entry of
       data is -inf
     - -infs don't count, they always have row_indices[.,.] == -1

    This function inserts "a new row" into the array, maintaining those invariants.
    So, if one of the elements of the row you're inserting is -inf or smaller than
    all of the entries in the corresponding column of data, that's a no-op.
    """
    assert isinstance(topk, tuple)
    topk_row_indices, topk_data = topk
    assert row_column_indices.shape == row_data.shape
    assert topk_row_indices.shape == topk_data.shape
    assert len(row_column_indices) <= len(topk_data)
    _topk_sparse_insert(row, row_column_indices, row_data, topk_row_indices, topk_data)


@numba.njit(
    "i8,i8[::1],f4[::1],i8[:,::1],f4[:,::1]",
    error_model="numpy",
    nogil=True,
    parallel=True,
)
def _topk_sparse_insert(row, row_column_indices, row_data, topk_row_indices, topk_data):
    k = topk_data.shape[1]
    for j in numba.prange(row_column_indices.shape[0]):
        col_ix = row_column_indices[j]
        datum = row_data[j]

        # optimize for case where datum won't be inserted
        if not datum > topk_data[col_ix, -1]:
            continue

        # otherwise everyone shimmies down
        row_ins = row
        for i in range(k):
            entry = topk_data[col_ix, i]
            entry_row = topk_row_indices[col_ix, i]
            if datum > entry:
                topk_data[col_ix, i] = datum
                topk_row_indices[col_ix, i] = row_ins
                datum = entry
                row_ins = entry_row


def topk_sparse_tocsc(
    topk, n_rows, extra_row=None, column_support=None, n_columns_full=None
):
    """Convert a topk sparse array to CSC.

    For noise unit purposes, allows inserting an extra row at the last minute.
    n_rows should NOT include the extra row.
    """
    assert isinstance(topk, tuple)
    topk_row_indices, topk_data = topk
    n, k = topk_data.shape
    assert topk_row_indices.shape == (n, k)

    order = topk_row_indices.argsort(axis=1)
    data = np.take_along_axis(topk_data, order, axis=1)
    row_inds = np.take_along_axis(topk_row_indices, order, axis=1)

    start = searchsorted_along_columns(row_inds, 0)
    nnz = row_inds.size - start.sum()
    if extra_row is not None:
        assert extra_row.shape == (n,)
        nnz += n

    ncols = n
    if column_support is not None:
        assert n_columns_full
        ncols = n_columns_full

    shape = (n_rows + (extra_row is not None), ncols)
    dtype = topk_data.dtype
    data_storage = np.empty((nnz,), dtype=dtype)
    index_storage = np.empty((nnz,), dtype=int)

    if (
        column_support is None
        or isinstance(column_support, slice)
        and column_support == slice(None)
    ):
        indptr = np.full((n + 1,), k + (extra_row is not None), dtype=int)
        indptr[0] = 0
        indptr[1:] -= start
        np.cumsum(indptr[1:], out=indptr[1:])
    else:
        indptr = np.zeros((ncols + 1,), dtype=int)
        indptr[1 + column_support] = k + (extra_row is not None)
        indptr[1 + column_support] -= start
        np.cumsum(indptr, out=indptr)

    if extra_row is None:
        _topk_pack(index_storage, data_storage, start, data, row_inds)
    else:
        _topk_pack_extra(
            index_storage,
            data_storage,
            start,
            data,
            row_inds,
            extra_row,
            n_rows,
        )

    a = csc_array((data_storage, index_storage, indptr), shape=shape)
    a._has_sorted_indices = True
    a._has_canonical_format = True
    return a


@numba.njit(
    "i8[::1],f4[::1],i8[::1],f4[:,::1],i8[:,::1]",
    error_model="numpy",
    nogil=True,
)
def _topk_pack(istorage, dstorage, start, topk_data, topk_inds):
    nzix = 0
    k = topk_data.shape[1]
    for j in range(start.shape[0]):
        for i in range(start[j], k):
            dstorage[nzix] = topk_data[j, i]
            istorage[nzix] = topk_inds[j, i]
            nzix += 1


@numba.njit(
    "i8[::1],f4[::1],i8[::1],f4[:,::1],i8[:,::1],f4[::1],i8",
    error_model="numpy",
    nogil=True,
)
def _topk_pack_extra(
    istorage, dstorage, start, topk_data, topk_inds, extra_row, extra_row_ind
):
    nzix = 0
    k = topk_data.shape[1]
    for j in range(start.shape[0]):
        s = start[j]
        for i in range(s, k):
            dstorage[nzix] = topk_data[j, i]
            istorage[nzix] = topk_inds[j, i]
            nzix += 1
        dstorage[nzix] = extra_row[j]
        istorage[nzix] = extra_row_ind
        nzix += 1


def double_searchsorted(a, v):
    i0 = np.searchsorted(a, v[0])
    assert a[i0] == v[0]
    out = np.empty(v.shape, dtype=int)
    out[0] = i0
    _double_searchsorted(a, v, out)
    return out


@numba.njit("i8[::1],i8[::1],i8[::1]", error_model="numpy", nogil=True)
def _double_searchsorted(a, v, out):
    i = out[0]
    for vix in range(1, v.shape[0]):
        val = v[vix]
        while val > a[i]:
            i += 1
        out[vix] = i


def coo_sparse_mask_rows(coo, keep_mask):
    """Row indexing with a boolean mask."""
    if keep_mask.all():
        return coo

    kept_label_indices = np.flatnonzero(keep_mask)
    ii, jj = coo.coords
    ixs = np.searchsorted(kept_label_indices, ii)
    ixs.clip(0, kept_label_indices.size - 1, out=ixs)
    valid = np.flatnonzero(kept_label_indices[ixs] == ii)
    coo = coo_array(
        (coo.data[valid], (ixs[valid], jj[valid])),
        shape=(kept_label_indices.size, coo.shape[1]),
    )
    return coo


def csc_sparse_mask_rows(csc, keep_mask, in_place=False):
    if keep_mask.all():
        return csc

    if not in_place:
        csc = csc.copy()

    rowix_dtype = csc.indices.dtype
    kept_row_inds = np.flatnonzero(keep_mask).astype(rowix_dtype)
    oldrow_to_newrow = np.zeros(len(keep_mask), dtype=rowix_dtype)
    oldrow_to_newrow[kept_row_inds] = np.arange(len(kept_row_inds))
    nnz = _csc_sparse_mask_rows(
        csc.indices, csc.indptr, csc.data, oldrow_to_newrow, keep_mask
    )

    return csc_array(
        (csc.data[:nnz], csc.indices[:nnz], csc.indptr),
        shape=(len(kept_row_inds), csc.shape[1]),
    )


sigs = [
    "i8(i8[::1], i8[::1], f4[::1], i8[::1], bool_[::1])",
    "i8(i4[::1], i4[::1], f4[::1], i4[::1], bool_[::1])",
]


@numba.njit(sigs, error_model="numpy", nogil=True)
def _csc_sparse_mask_rows(indices, indptr, data, oldrow_to_newrow, keep_mask):
    write_ix = 0

    column = 0
    column_kept_count = 0
    column_end = indptr[1]

    for read_ix in range(len(indices)):
        row = indices[read_ix]
        if not keep_mask[row]:
            continue

        # write data for this sample
        indices[write_ix] = oldrow_to_newrow[row]
        data[write_ix] = data[read_ix]
        write_ix += 1

        while read_ix >= column_end:
            indptr[column + 1] = write_ix - 1
            column += 1
            column_end = indptr[column + 1]

    while column < len(indptr) - 1:
        indptr[column + 1] = write_ix
        column += 1
        column_end = indptr[column + 1]

    return write_ix


def csc_sparse_getrow(csc, row, rowcount):
    rowix_dtype = csc.indices.dtype
    columns_out = np.empty(rowcount, dtype=rowix_dtype)
    data_out = np.empty(rowcount, dtype=csc.data.dtype)
    _csc_sparse_getrow(
        csc.indices,
        csc.indptr,
        csc.data,
        columns_out,
        data_out,
        rowix_dtype.type(row),
        rowcount,
    )

    return columns_out, data_out


sigs = [
    "void(i8[::1], i8[::1], f4[::1], i8[::1], f4[::1], i8, i8)",
    "void(i4[::1], i4[::1], f4[::1], i4[::1], f4[::1], i4, i8)",
]


@numba.njit(sigs, error_model="numpy", nogil=True)
def _csc_sparse_getrow(indices, indptr, data, columns_out, data_out, the_row, count):
    write_ix = 0

    column = 0
    column_end = indptr[1]

    for read_ix in range(len(indices)):
        row = indices[read_ix]
        if row != the_row:
            continue

        # write data for this sample
        data_out[write_ix] = data[read_ix]
        while read_ix >= column_end:
            column += 1
            column_end = indptr[column + 1]
        columns_out[write_ix] = column
        write_ix += 1
        if write_ix >= count:
            return


def sparse_topk(liks, log_proportions=None, k=3):
    """Top k units for each spike"""
    # csc is needed here for this to be fast
    liks = liks.tocsc()
    nz_lines = np.flatnonzero(np.diff(liks.indptr))
    nnz = len(nz_lines)

    # see scipy csc argmin/argmax for reference here. this is just numba-ing
    # a special case of that code which has a python hot loop.
    topk = np.full((nnz, k), -1)
    if log_proportions is None:
        log_proportions = np.zeros(liks.shape[0], dtype=np.float32)
    else:
        assert liks.shape[0] == len(log_proportions)
        log_proportions = log_proportions.astype(np.float32)

    # this loop ignores sparse zeros. so, no sweat for negative inputs.
    topk_loop(
        topk,
        nz_lines,
        liks.indptr,
        liks.data,
        liks.indices,
        log_proportions,
    )

    return nz_lines, topk


sigs = [
    "void(i8[:, ::1], i8[::1], i4[::1], f4[::1], i4[::1], f4[::1])",
    "void(i8[:, ::1], i8[::1], i8[::1], f4[::1], i8[::1], f4[::1])",
]


@numba.njit(
    sigs,
    error_model="numpy",
    nogil=True,
    parallel=True,
)
def topk_loop(topk, nz_lines, indptr, data, indices, log_proportions):
    # for i in nz_lines:
    k = topk.shape[1]
    for j in numba.prange(nz_lines.shape[0]):
        i = nz_lines[j]
        p = indptr[i]
        q = indptr[i + 1]
        ix = indices[p:q]
        dx = data[p:q] + log_proportions[ix]
        kj = min(k, q - p)
        top = dx.argsort()[-kj:]
        topk[j, :kj] = ix[top[::-1]]


def sparse_reassign(liks, proportions=None, log_proportions=None, hard_noise=False):
    """Reassign spikes to units with largest likelihood

    liks is (n_units, n_spikes). This computes the argmax for each column,
    treating sparse 0s as -infs rather than as 0s.

    Turns out that scipy's sparse argmin/max have a slow python inner loop,
    this uses a numba replacement, but I'd like to upstream a cython version.
    """
    if not liks.nnz:
        return (
            np.arange(0),
            liks,
            np.full(liks.shape[1], -1),
            np.full(liks.shape[1], -np.inf),
        )

    # csc is needed here for this to be fast
    liks = liks.tocsc()
    nz_lines = np.flatnonzero(np.diff(liks.indptr))
    nnz = len(nz_lines)

    # see scipy csc argmin/argmax for reference here. this is just numba-ing
    # a special case of that code which has a python hot loop.
    assignments = np.full(nnz, -1)
    # these will be filled with logsumexps
    likelihoods = np.full(nnz, -np.inf, dtype=np.float32)

    # get log proportions, either given logs or otherwise...
    if log_proportions is None:
        if proportions is None:
            log_proportions = np.full(nnz, -np.log(liks.shape[0]), dtype=np.float32)
        elif torch.is_tensor(proportions):
            log_proportions = proportions.log().numpy(force=True)
        else:
            log_proportions = np.log(proportions)
    else:
        if torch.is_tensor(log_proportions):
            log_proportions = log_proportions.numpy(force=True)
    log_proportions = log_proportions.astype(np.float32)

    # this loop ignores sparse zeros. so, no sweat for negative inputs.
    if hard_noise:
        log_proportions = log_proportions[:-1] - logsumexp(log_proportions[:-1])
        hard_noise_argmax_loop(
            assignments,
            likelihoods,
            nz_lines,
            liks.indptr,
            liks.data,
            liks.indices,
            log_proportions,
        )
    else:
        hot_argmax_loop(
            assignments,
            likelihoods,
            nz_lines,
            liks.indptr,
            liks.data,
            liks.indices,
            log_proportions,
        )

    return nz_lines, liks, assignments, likelihoods


# csc can have int32 or 64 coos on dif platforms? is this an intp? :P
sigs = [
    "void(i8[::1], f4[::1], i8[::1], i4[::1], f4[::1], i4[::1], f4[::1])",
    "void(i8[::1], f4[::1], i8[::1], i8[::1], f4[::1], i8[::1], f4[::1])",
]


@numba.njit(
    sigs,
    error_model="numpy",
    nogil=True,
    parallel=True,
)
def hot_argmax_loop(
    assignments, scores, nz_lines, indptr, data, indices, log_proportions
):
    # for i in nz_lines:
    for j in numba.prange(nz_lines.shape[0]):
        i = nz_lines[j]
        p = indptr[i]
        q = indptr[i + 1]
        ix = indices[p:q]
        dx = data[p:q] + log_proportions[ix]
        best = dx.argmax()
        assignments[j] = ix[best]
        mx = dx.max()
        scores[j] = mx + np.log(np.exp(dx - mx).sum())


@numba.njit(
    sigs,
    error_model="numpy",
    nogil=True,
    parallel=True,
)
def hard_noise_argmax_loop(
    assignments, scores, nz_lines, indptr, data, indices, log_proportions
):
    # noise_ix = log_proportions.shape[0]
    # for i in nz_lines:
    for j in numba.prange(nz_lines.shape[0]):
        i = nz_lines[j]
        p = indptr[i]
        q = indptr[i + 1] - 1  # skip the noise
        ix = indices[p:q]
        dx = data[p:q] + log_proportions[ix]
        mx = dx.max()
        score = mx + np.log(np.exp(dx - mx).sum())
        noise_score = data[q]  # indptr[i+1]-1 is the noise ix
        if score > noise_score:
            scores[j] = score
            best = dx.argmax()
            assignments[j] = ix[best]
        else:
            scores[j] = noise_score
            # best = noise_ix


def searchsorted_along_columns(arr, value):
    out = np.empty((arr.shape[0],), dtype=int)
    _searchsorted_along_columns(out, arr, value)
    return out


@numba.njit(
    "i8[::1],i8[:,::1],i8",
    error_model="numpy",
    nogil=True,
    parallel=True,
)
def _searchsorted_along_columns(out, arr, value):
    k = arr.shape[1]
    for j in numba.prange(out.shape[0]):
        i = 0
        while i < k and value > arr[j, i]:
            i += 1
        out[j] = i


def integers_without_inner_replacement(rg, high, size):
    assert len(size) == 2
    out = np.empty(size, dtype=np.int64)
    out_write = out.reshape((-1, size[-1]))

    if isinstance(high, np.ndarray):
        assert high.shape == size[:-1]
        _fisher_yates_loop_vec(rg, high.ravel(), out_write)
    else:
        assert isinstance(high, int) or high.shape == ()
        assert high >= size[-1]
        _fisher_yates_loop_scalar(rg, high, out_write)
    return out


@numba.njit(error_model="numpy")
def _fisher_yates_loop_scalar(rg, high, out):
    """I think this is FY? At least it is uniform..."""
    k = out.shape[1]
    for i in range(out.shape[0]):
        for j in range(k):
            out[i, j] = rg.integers(0, high - j)
        for j in range(k - 1, -1, -1):
            for jj in range(j - 1, -1, -1):
                if out[i, j] == out[i, jj]:
                    out[i, j] = high - jj - 1


@numba.njit(error_model="numpy")
def _fisher_yates_loop_vec(rg, high, out):
    """In rows where high[i] < output shape, we put everything and negatives."""
    k = out.shape[1]
    for i in range(out.shape[0]):
        h = high[i]
        if h <= k:
            for j in range(k):
                out[i, j] = max(-1, h - 1 - j)
            continue
        for j in range(k):
            out[i, j] = rg.integers(0, h - j)
        for j in range(k - 1, -1, -1):
            for jj in range(j - 1, -1, -1):
                if out[i, j] == out[i, jj]:
                    out[i, j] = h - jj - 1


def fisher_yates_replace(rg, high, data):
    """Replace -1s with FY shuffle

    NEEDS descending sorted input along last axis. (FY sampling.)
    Maintains already existing guys in the set.
    """
    _fisher_yates_replace(rg, high, data.reshape((-1, data.shape[-1])))


@numba.njit(error_model="numpy")
def _fisher_yates_replace(rg, h, out):
    k = out.shape[1]
    for i in range(out.shape[0]):
        for j in range(k):
            if out[i, j] < 0:
                out[i, j] = rg.integers(0, h - j)
        for j in range(k - 1, -1, -1):
            for jj in range(j - 1, -1, -1):
                if out[i, j] == out[i, jj]:
                    out[i, j] = h - jj - 1


@numba.njit(parallel=True, error_model="numpy")
def erase_dups(arr):
    for i in numba.prange(arr.shape[0]):
        x = arr[i]
        for j, xx in enumerate(x):
            if xx == -1:
                continue
            for k in range(j + 1, x.shape[0]):
                if x[k] == xx:
                    x[k] = -1
