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
        log_proportions = np.zeros(len(liks), dtype=np.float32)
    else:
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
