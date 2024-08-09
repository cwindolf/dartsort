import numpy as np
from . import density
# todo: replace all isosplit stuff with things based on scipy's isotonic regression.


def fit_unimodal_right(x, f, weights=None, cut=0, hard=False):
    """Unimodal to the right of cut, increasing to the left. Continuity at the border."""
    from isosplit import up_down_isotonic_regression, jisotonic5
    if weights is None:
        weights = np.ones_like(f)

    # figure out where cut lands and what f(cut) should be
    cuti_left = np.searchsorted(x, cut)
    if cuti_left >= len(x) - 1:
        # everything is to the left... fit increasing!
        return jisotonic5.jisotonic5(f, weights=weights)[0]
    cuti_right = np.searchsorted(x, cut, side="right") - 1
    if cuti_right <= 0:
        # everything is to the right... fit unimodal!
        return up_down_isotonic_regression(f, weights=weights)
    # else, in the middle somewhere...
    assert cuti_left in (cuti_right, cuti_right + 1)

    out = np.empty_like(f)
    out[cuti_right:] = up_down_isotonic_regression(
        np.ascontiguousarray(f[cuti_right:]),
        weights=np.ascontiguousarray(weights[cuti_right:]),
    )

    # impose continuity with a large penalty
    f_left = np.concatenate((f[:cuti_left], out[cuti_right:cuti_right+1]))
    w_left = np.concatenate((weights[:cuti_left], 1e6 + 10 * weights.sum(keepdims=True)))
    out[:cuti_left] = jisotonic5.jisotonic5(f_left, w_left)[0][:-1]

    return out


def fit_bimodal_at(x, f, weights=None, cut=0):
    from isosplit import up_down_isotonic_regression
    if weights is None:
        weights = np.ones_like(f)
    which_right = x > cut
    which_left = np.logical_not(which_right)
    out = np.empty_like(f)
    if which_left.any():
        out[which_left] = up_down_isotonic_regression(f[which_left], weights=weights[which_left])
    if which_right.any():
        out[which_right] = up_down_isotonic_regression(f[which_right], weights=weights[which_right])
    return out


def smoothed_dipscore_at(cut, samples, sample_weights, alternative="smoothed", dipscore_only=False, score_kind="tv"):
    if sample_weights is None:
        sample_weights = np.ones_like(samples)
    densities = density.get_smoothed_densities(
        samples[:, None],
        sigmas=0.5 * 1.06 * samples.std() * (samples.size ** -0.2),
        bin_size_ratio=20.0,
        min_bin_size=0.025,
        max_n_bins=512,
        min_n_bins=10,
        weights=sample_weights,
    )
    spacings = np.diff(samples)
    spacings = np.concatenate((spacings[:1], 0.5 * (spacings[1:] + spacings[:-1]), spacings[-1:]))
    if alternative == "bimodal":
        densities = fit_bimodal_at(samples, densities, weights=sample_weights, cut=cut)
    densities /= (densities * spacings).sum()

    score = 0
    best_dens_err = np.inf
    best_uni = None
    # change to density which is best overall fit
    for hard in (False, True):
        mask = samples < cut if hard else samples <= cut
        if score_kind == "ks":
            empirical = (densities[mask] * spacings[mask]).sum()

        for order, sign in zip(
            (slice(None), slice(None, None, -1)),
            (1, -1),
        ):
            s = np.ascontiguousarray(samples[order])
            d = np.ascontiguousarray(densities[order])
            w = np.ascontiguousarray(sample_weights[order])
            dens = fit_unimodal_right(sign * s, d, weights=w, hard=hard)
            dens = dens[order]
            dens /= (dens * spacings).sum()
            dens_err = (np.abs(dens - densities) * spacings).sum()
            if score_kind == "ks":
                my_score = abs(empirical - np.sum(dens[mask] * spacings[mask])) * np.sqrt(sample_weights.sum())
            elif score_kind == "tv":
                my_score = 0.5 * np.sum(np.abs(densities - dens) * spacings)
            # if my_score > score:
            if dens_err < best_dens_err:
                score = my_score
                best_dens_err = dens_err
                best_uni = dens

    if dipscore_only:
        return score

    return score, samples, densities, best_uni
