import numpy as np
from scipy.signal import find_peaks
from scipy.stats import norm

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


def fit_truncnorm_right(x, f, weights=None, cut=0, hard=False, n_iter=10):
    """Like above, but fits truncated normal to xs > cut with MoM."""
    if weights is None:
        weights = np.ones_like(f)

    # figure out where cut lands and what f(cut) should be
    cuti_left = np.searchsorted(x, cut)
    if cuti_left >= len(x) - 2:
        # everything is to the left... return uniform
        return np.full_like(f, 1/len(f))
    cuti_right = np.searchsorted(x, cut, side="right") - 1
    if cuti_right <= 1:
        # everything is to the right... fit normal!
        mean = np.average(x, weights=weights)
        var = np.average((x - mean) ** 2, weights=weights)
        return norm.pdf(x, loc=mean, scale=np.sqrt(var))
    # else, in the middle somewhere...
    assert cuti_left in (cuti_right, cuti_right + 1)

    xcut = x[cuti_right:]
    mean = np.average(xcut, weights=weights[cuti_right:])
    var = np.average((xcut - mean) ** 2, weights=weights[cuti_right:])
    mu = mean
    sigma = std = np.sqrt(var)
    for i in range(n_iter):
        alpha = (cut - mu) / sigma
        phi_alpha = norm.cdf(alpha)
        Z = 1.0 - phi_alpha
        # we have: mean ~ mu + sigma*phi(alpha)/Z
        # and      var ~ sigma^2(1+alpha phi(alpha) /Z - (phi(alpha)/Z)^2)
        # implies mu = mean - sigma*phi(alpha)/Z
        #         sigma^2 = var/(1+alpha phi(alpha) /Z - (phi(alpha)/Z)^2)
        sigma = np.sqrt(var / (1 + alpha * phi_alpha / Z - (phi_alpha / Z) ** 2))
        mu = mean - sigma * phi_alpha / Z

    return norm.pdf(x, loc=mu, scale=sigma)


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


def smoothed_dipscore_at(
    cut,
    samples,
    sample_weights,
    alternative="smoothed",
    dipscore_only=False,
    score_kind="tv",
    cut_relmax_order=3,
    kind="isotonic",
    debug_info=None,
):
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
    densities /= (densities * spacings).sum()

    if cut is None:
        # closest maxes left + right of 0
        maxers, _ = find_peaks(densities, distance=cut_relmax_order)
        coords = samples[maxers]
        left_cut = right_cut = 0
        if (coords < 0).any():
            left_cut = coords[coords < 0].max()
        if (coords > 0).any():
            right_cut = coords[coords > 0].min()
        candidates = np.logical_and(samples >= left_cut, samples <= right_cut)
        cut = 0
        if candidates.any():
            cut = samples[candidates][np.argmin(densities[candidates])]

    if alternative == "bimodal":
        densities = fit_bimodal_at(samples, densities, weights=sample_weights, cut=cut)
        densities /= (densities * spacings).sum()
    else:
        assert alternative == "smoothed"

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
            if kind == "isotonic":
                dens = fit_unimodal_right(sign * s, d, weights=w, cut=sign * cut, hard=hard)
            elif kind == "truncnorm":
                dens = fit_truncnorm_right(sign * s, d, weights=w, cut=sign * cut, hard=hard)
            else:
                assert False
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

    if debug_info is not None:
        debug_info["domain"] = samples
        debug_info["alternative_density"] = densities
        debug_info["cut"] = cut
        debug_info["score"] = score
        debug_info["score_kind"] = score_kind
        debug_info["uni_density"] = dens
        debug_info["sample_weights"] = sample_weights
        debug_info["samples"] = samples

    if dipscore_only:
        return score

    return score, samples, densities, best_uni, cut
