import numpy as np
from scipy import sparse

from . import localization, point_source_centering


def featurize(waveforms, maxchans, geom, k=3, relocate_dims="yza", return_recons=False):
    xs, ys, z_rels, z_abss, alphas = localization.localize_waveforms(
        waveforms, geom, maxchans, channel_radius=8, geomkind="standard"
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
    print(q.shape, p.shape)
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
    features = np.c_[xs, ys, z_rels, alphas, loadings]

    if not return_recons:
        return features

    recons = (u @ np.diag(s) @ vh).reshape((n, t, c)) + means
    recons_inv = recons * (p / q).numpy()[:, None, :]
    return features, recons_inv
