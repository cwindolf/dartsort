import numpy as np

from sklearn.decomposition import PCA
from tqdm.auto import trange

from .waveform_utils import relativize_waveforms  # noqa


def pca_reload(
    original_waveforms,
    relocated_waveforms,
    orig_ptps,
    standard_ptps,
    rank=10,
    B_updates=0,
):
    N, T, C = original_waveforms.shape
    assert relocated_waveforms.shape == (N, T, C)
    assert orig_ptps.shape == standard_ptps.shape == (N, C)

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
    for n in trange(N):
        A = (
            (destandardization[n, None, :, :] * pca_basis)
            .reshape(rank, T * C)
            .T
        )
        b = decentered_original_waveforms[n].reshape(T * C)
        x, resid, *_ = np.linalg.lstsq(A, b, rcond=None)
        reloadings[n] = x
        err += resid
    err = err / (N * T * C)

    for _ in trange(B_updates, desc="B updates"):
        # update B
        # flat view
        B = pca_basis.reshape(rank, T * C)
        W = decentered_original_waveforms.reshape(N, T * C)
        for c in range(C):
            A = destandardization[:, 0, c, None] * reloadings
            for t in range(T):
                i = t * C + c
                res, *_ = np.linalg.lstsq(A, W[:, i], rcond=None)
                B[:, i] = res

        # re-update reloadings
        reloadings = np.zeros((N, rank))
        err = 0.0
        for n in trange(N):
            A = (
                (destandardization[n, None, :, :] * pca_basis)
                .reshape(rank, T * C)
                .T
            )
            b = decentered_original_waveforms[n].reshape(T * C)
            x, resid, *_ = np.linalg.lstsq(A, b, rcond=None)
            reloadings[n] = x
            err += resid
        err = err / (N * T * C)

    return reloadings, err
