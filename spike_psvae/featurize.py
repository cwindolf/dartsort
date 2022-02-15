import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from tqdm.auto import trange

from . import localization, point_source_centering


# from . import up_down


# def isotonic_ptp(ptps, central=False):
#     if central:
#         return np.array(
#             [
#                 np.c_[
#                     up_down.central_up_down_isotonic_regression(ptp[::2]),
#                     up_down.central_up_down_isotonic_regression(ptp[1::2]),
#                 ].ravel()
#                 for ptp in ptps
#             ],
#             dtype=ptps.dtype,
#         )
#     else:
#         return np.array(
#             [
#                 np.c_[
#                     up_down.up_down_isotonic_regression(ptp[::2]),
#                     up_down.up_down_isotonic_regression(ptp[1::2]),
#                 ].ravel()
#                 for ptp in ptps
#             ],
#             dtype=ptps.dtype,
#         )


def relativize_waveforms(wfs, firstchans, z, geom, feat_chans=18):
    chans_down = feat_chans // 2
    chans_down -= chans_down % 2

    stdwfs = np.zeros(
        (wfs.shape[0], wfs.shape[1], feat_chans), dtype=wfs.dtype
    )

    firstchans_std = firstchans.copy().astype(int)
    maxchans = np.zeros(firstchans.shape, dtype=int)
    if z is not None:
        z_rel = np.zeros_like(z)

    for i in range(wfs.shape[0]):
        wf = wfs[i]
        mcrel = wf.ptp(0).argmax()
        mcrix = mcrel - mcrel % 2
        if z is not None:
            z_rel[i] = z[i] - geom[firstchans[i] + mcrel, 1]

        low, high = mcrix - chans_down, mcrix + feat_chans - chans_down
        if low < 0:
            low, high = 0, feat_chans
        if high > wfs.shape[2]:
            low, high = wfs.shape[2] - feat_chans, wfs.shape[2]

        firstchans_std[i] += low
        stdwfs[i] = wf[:, low:high]
        maxchans[i] = firstchans_std[i] + stdwfs[i].ptp(0).argmax()

    if z is not None:
        return stdwfs, firstchans_std, maxchans, z_rel, chans_down
    else:
        return stdwfs, firstchans_std, maxchans, chans_down


def featurize(
    waveforms,
    maxchans,
    geom,
    k=3,
    iso_ptp=False,
    relocate_dims="yza",
    return_recons=False,
    return_rel=False,
):
    ptps = waveforms.ptp(1)

    if iso_ptp:
        print("iso!")
        ptps = isotonic_ptp(ptps, central=False)

    xs, ys, z_rels, z_abss, alphas = localization.localize_ptps(
        ptps, geom, maxchans, channel_radius=8, geomkind="standard"
    )
    relocs, q, p = point_source_centering.relocate_simple(
        waveforms,
        geom,
        maxchans,
        xs,
        ys,
        z_rels,
        alphas,
        channel_radius=8,
        geomkind="standard",
        relocate_dims=relocate_dims,
    )
    relocs = relocs.numpy()
    means = relocs.mean(axis=0, keepdims=True)
    relocs -= means
    n, t, c = relocs.shape
    u, s, vh = sparse.linalg.svds(
        relocs.reshape(relocs.shape[0], -1),
        k=k,
    )
    # standardized loadings
    loadings = np.sqrt(n - 1) * u
    features = np.c_[xs, ys, z_abss, alphas, loadings]

    if return_rel:
        return features, z_rels

    if not return_recons:
        return features

    recons = (u @ np.diag(s) @ vh).reshape((n, t, c)) + means
    recons_inv = recons * (p / q).numpy()[:, None, :]
    return features, recons_inv


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
