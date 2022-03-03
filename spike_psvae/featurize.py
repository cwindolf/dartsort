import itertools
import numpy as np

from sklearn.decomposition import PCA
from tqdm.auto import trange, tqdm
from joblib import Parallel, delayed

from .waveform_utils import relativize_waveforms  # noqa
from .point_source_centering import relocate_simple


def pca_reload(
    original_waveforms,
    relocated_waveforms,
    orig_ptps,
    standard_ptps,
    rank=10,
    B_updates=2,
    pbar=True,
):
    N, T, C = original_waveforms.shape
    assert relocated_waveforms.shape == (N, T, C)
    assert orig_ptps.shape == standard_ptps.shape == (N, C)
    xrange = trange if pbar else lambda *args, **kwargs: range(*args)

    destandardization = (orig_ptps / standard_ptps)[:, None, :]

    # fit PCA in relocated space
    reloc_pca = PCA(rank).fit(relocated_waveforms.reshape(N, T * C))
    pca_basis = reloc_pca.components_.reshape(rank, T, C)

    # rank 0 model
    relocated_mean = reloc_pca.mean_.reshape(T, C)
    unrelocated_means = relocated_mean[None, :, :] * destandardization
    decentered_original_waveforms = original_waveforms - unrelocated_means

    # re-compute the loadings to minimize loss in original space
    reloadings = np.zeros((N, rank))
    err = 0.0
    for n in xrange(N):
        A = (
            (destandardization[n, None, :, :] * pca_basis)
            .reshape(rank, T * C)
            .T
        )
        b = decentered_original_waveforms[n].reshape(T * C)
        x, resid, *_ = np.linalg.lstsq(A, b, rcond=None)
        reloadings[n] = x
        err += resid
        errors[n] = resid / (T * C)
    err = err / (N * T * C)

    for _ in xrange(B_updates, desc="B updates"):
        # update B
        # flat view
        B = pca_basis.reshape(rank, T * C)
        W = decentered_original_waveforms.reshape(N, T * C)
        for c in range(C):
            A = destandardization[:, 0, c, None] * reloadings
            for t in xrange(T):
                i = t * C + c
                res, *_ = np.linalg.lstsq(A, W[:, i], rcond=None)
                B[:, i] = res

        # re-update reloadings
        reloadings = np.zeros((N, rank))
        err = 0.0
        for n in xrange(N):
            A = (
                (destandardization[n, None, :, :] * pca_basis)
                .reshape(rank, T * C)
                .T
            )
            b = decentered_original_waveforms[n].reshape(T * C)
            x, resid, *_ = np.linalg.lstsq(A, b, rcond=None)
            reloadings[n] = x
            err += resid
            errors[n] = resid / (T * C)
        err = err / (N * T * C)

    return reloadings, err


def relocated_ae(
    waveforms,
    firstchans,
    maxchans,
    geom,
    x,
    y,
    z_rel,
    alpha,
    relocate_dims="xyza",
    rank=10,
    B_updates=2,
    pbar=True,
):
    # -- compute the relocation
    waveforms_reloc, std_ptp, pred_ptp = relocate_simple(
        waveforms,
        geom,
        firstchans,
        maxchans,
        x,
        y,
        z_rel,
        alpha,
        relocate_dims=relocate_dims,
    )
    # torch -> numpy
    waveforms_reloc = waveforms_reloc.cpu().numpy()
    std_ptp = std_ptp.cpu().numpy()
    pred_ptp = pred_ptp.cpu().numpy()

    # -- get the features
    feats, err = pca_reload(
        waveforms,
        waveforms_reloc,
        pred_ptp,
        std_ptp,
        rank=rank,
        B_updates=B_updates,
        pbar=pbar,
    )

    return feats, err


def relocated_ae_batched(
    waveforms,
    firstchans,
    maxchans,
    geom,
    x,
    y,
    z_rel,
    alpha,
    relocate_dims="xyza",
    rank=10,
    B_updates=2,
    batch_size=50000,
    n_jobs=1,
):
    N, T, C = waveforms.shape

    # we should be able to store features in memory:
    # 5 million spikes x 10 features x 4 bytes is like .2 gig
    features = np.empty((N, rank))
    errors = np.empty(N // batch_size + 1)

    @delayed
    def job(bs, be, wfs, fcs, mcs):
        feats, err = relocated_ae(
            wfs,
            fcs,
            mcs,
            geom,
            x[bs:be],
            y[bs:be],
            z_rel[bs:be],
            alpha[bs:be],
            relocate_dims="xyza",
            rank=rank,
            B_updates=B_updates,
            pbar=False,
        )
        return bs, be, feats, err

    i = 0
    for batch in grouper(
        n_jobs, trange(0, N, batch_size, desc="Feature batches")
    ):
        for bs, be, feats, err in Parallel(n_jobs, mmap_mode="r+")(
            job(
                bs,
                min(bs + batch_size, N),
                waveforms[bs : min(bs + batch_size, N)],
                firstchans[bs : min(bs + batch_size, N)],
                maxchans[bs : min(bs + batch_size, N)],
            )
            for bs in batch
        ):
            features[bs:be] = feats
            errors[i] = err
            i += 1

    return features, errors


def maxptp_batched(
    waveforms,
    firstchans,
    maxchans,
    n_workers=1,
    batch_size=1024,
):
    """A helper for running the above on hdf5 datasets or similar"""
    N = len(firstchans)
    starts = list(range(0, N, batch_size))
    ends = [min(start + batch_size, N) for start in starts]

    @delayed
    def getmaxptp(start, end):
        mcrels = maxchans[start:end] - firstchans[start:end]
        return waveforms[start:end][np.arange(end - start), :, mcrels].ptp(1)

    jobs = (
        getmaxptp(start, end)
        for start, end in tqdm(
            zip(starts, ends), total=len(starts), desc="ptp batches"
        )
    )

    maxptp = np.empty(N)
    with Parallel(n_workers, require="sharedmem") as pool:
        for batch in grouper(10 * n_workers, jobs):
            for batch_idx, res in enumerate(pool(jobs)):
                start = starts[batch_idx]
                end = ends[batch_idx]
                maxptp[start:end] = res

    return maxptp


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
