import numpy as np
from scipy import sparse

from . import localization, point_source_centering
from . import up_down


def isotonic_ptp(ptps, central=False):
    if central:
        return np.array(
            [
                np.c_[
                    up_down.central_up_down_isotonic_regression(ptp[::2]),
                    up_down.central_up_down_isotonic_regression(ptp[1::2]),
                ].ravel()
                for ptp in ptps
            ],
            dtype=ptps.dtype
        )
    else:
        return np.array(
            [
                np.c_[
                    up_down.up_down_isotonic_regression(ptp[::2]),
                    up_down.up_down_isotonic_regression(ptp[1::2]),
                ].ravel()
                for ptp in ptps
            ],
            dtype=ptps.dtype
        )


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
