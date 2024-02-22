from collections.abc import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from sklearn.neighbors import KDTree as SKDTree


def kdtree_inliers(
    X, kdtree=None, n_neighbors=10, distance_upper_bound=25.0, workers=1
):
    """Mark outlying points by a neighbors distance criterion

    Returns
    -------
    inliers : bool array
        inliers[i] = X[i] has at least n_neighbors neighbors within distance
        distance_upper_bound
    kdtree : KDTree
    """
    if kdtree is None:
        kdtree = KDTree(X)

    distances, indices = kdtree.query(
        X, k=1 + n_neighbors, distance_upper_bound=distance_upper_bound, workers=workers
    )
    inliers = (indices[:, 1:] < len(X)).sum(1) >= n_neighbors

    return inliers, kdtree

def get_smoothed_densities(
    X,
    inliers=slice(None),
    sigmas=None,
    return_hist=False,
    sigma_lows=None,
    sigma_ramp_ax=-1,
    bin_sizes=None,
    bin_size_ratio=5.0,
    min_bin_size=1.0,
    ramp_min_bin_size=5.0,
    revert=False,
):
    """Get RBF density estimates for each X[i] and bandwidth in sigmas

    Outliers will be marked with NaN KDEs. Please pass inliers, or else your
    histogram is liable to be way too big.
    """
    # figure out what bandwidths we'll be working on
    seq = isinstance(sigmas, Sequence)
    if not seq:
        sigmas = (sigmas,)
    do_ramp = sigma_lows is not None
    if not do_ramp:
        sigma_lows = [None for _ in sigmas]
    elif not seq:
        sigma_lows = (sigma_lows,)
    min_sigma = min(sigmas)
    if do_ramp:
        min_sigma = min(min_sigma, min(sigma_lows))

    # histogram bin sizes -- not too big, not too small
    infeats = X[inliers]
    bin_sizes = np.full(X.shape[1], min_sigma / bin_size_ratio)
    bin_sizes = np.maximum(min_bin_size, bin_sizes)
    if do_ramp:
        bin_sizes[sigma_ramp_ax] = max(bin_sizes[sigma_ramp_ax], ramp_min_bin_size)

    # select bin edges
    extents = np.c_[np.floor(infeats.min(0)), np.ceil(infeats.max(0))]
    nbins = np.ceil(extents.ptp(1) / bin_sizes).astype(int)
    bin_edges = [np.linspace(e[0], e[1], num=nb) for e, nb in zip(extents, nbins)]

    # compute histogram and figure out how big the bins actually were
    raw_histogram, bin_edges = np.histogramdd(infeats, bins=bin_edges)
    for be in bin_edges:
        if len(be)<2:
            return None
    bin_sizes = np.array([(be[1] - be[0]) for be in bin_edges])
    bin_centers = [0.5 * (be[1:] + be[:-1]) for be in bin_edges]

    # normalize histogram to samples / volume
    raw_histogram = raw_histogram / bin_sizes.prod()

    kdes = []
    if return_hist:
        hists = []
    for sigma, sigma_low in zip(list(sigmas), list(sigma_lows)):
        # fix up sigma, sigma_low based on bin size
        if sigma is not None:
            sigma = sigma / bin_sizes
        if sigma_low is not None:
            sigma_low = sigma_low / bin_sizes

        # figure out how histogram should be filtered
        hist = raw_histogram
        if sigma is not None and sigma_low is None:
            hist = gaussian_filter(raw_histogram, sigma)
        elif sigma is not None and sigma_low is not None:
            # filter by a sequence of bandwidths
            ramp = np.linspace(sigma_low, sigma, num=hist.shape[sigma_ramp_ax])
            if revert:
                ramp[:, sigma_ramp_ax] = np.linspace(
                    sigma[sigma_ramp_ax],
                    sigma_low[sigma_ramp_ax],
                    num=hist.shape[sigma_ramp_ax],
                )
                # ramp[:, 0] = sigma[0]
                # ramp[:, 1] = sigma[1]

            # operate along the ramp axis
            hist_move = np.moveaxis(hist, sigma_ramp_ax, 0)
            hist_smoothed = hist_move.copy()
            for j, sig in enumerate(ramp):
                sig_move = sig.copy()
                sig_move[0] = sig[sigma_ramp_ax]
                sig_move[sigma_ramp_ax] = sig[0]
                hist_smoothed[j] = gaussian_filter(hist_move, sig_move)[j]  # sig_move
            hist = np.moveaxis(hist_smoothed, 0, sigma_ramp_ax)
        if return_hist:
            hists.append(hist)

        lerp = RegularGridInterpolator(bin_centers, hist, bounds_error=False)
        kdes.append(lerp(X))

    kdes = kdes if seq else kdes[0]
    if return_hist:
        hists = hists if seq else hists[0]
        return kdes, hists
    return kdes


def nearest_higher_density_neighbor(
    kdtree, density, n_neighbors_search=10, distance_upper_bound=5.0, workers=1
):
    distances, indices = kdtree.query(
        kdtree.data,
        k=1 + n_neighbors_search,
        distance_upper_bound=distance_upper_bound,
        workers=workers,
    )
    # exclude self
    distances, indices = distances[:, 1:].copy(), indices[:, 1:].copy()

    # find lowest distance higher density neighbor
    density_padded = np.pad(density, (0, 1), constant_values=np.inf)
    is_lower_density = density_padded[indices] <= density[:, None]
    distances[is_lower_density] = np.inf
    indices[is_lower_density] = kdtree.n
    nhdn = indices[np.arange(kdtree.n), distances.argmin(1)]

    return nhdn, distances, indices


def remove_border_points(
    labels, density, kdtree, search_radius=1.0, n_neighbors_search=5, workers=1
):
    distances, indices = kdtree.query(
        kdtree.data,
        k=1 + n_neighbors_search,
        distance_upper_bound=search_radius,
        workers=workers,
    )

    labels_padded = np.pad(labels, (0, 1), constant_values=-2)
    in_border = np.any(
        labels_padded[indices] != labels_padded[indices[:, 0, None]], axis=1
    )
    in_border = np.flatnonzero(in_border)
    labels_in_border = labels[in_border]

    units = np.unique(labels)
    new_labels = labels.copy()
    for i, u in enumerate(units[units > 0]):
        in_unit = np.flatnonzero(labels == u)
        in_unit_border = in_border[labels_in_border == u]
        if not in_unit_border.size:
            continue

        to_remove = density[in_unit] < density[in_unit_border].max()
        new_labels[in_unit[to_remove]] = -1

    return new_labels


def decrumb(labels, min_size=5):
    units, counts = np.unique(labels, return_counts=True)
    big_enough = counts >= min_size
    units[~big_enough] = -1
    units[big_enough] = np.arange(big_enough.sum())
    return units[labels]


def density_peaks_clustering(
    X,
    l2_norm=None,
    kdtree=None,
    sigma_local=5.0,
    sigma_local_low=None,
    sigma_regional=None,
    sigma_regional_low=None,
    outlier_neighbor_count=10,
    outlier_radius=25.0,
    n_neighbors_search=10,
    radius_search=5.0,
    noise_density=0.0,
    remove_clusters_smaller_than=10,
    remove_borders=False,
    border_search_radius=10.0,
    border_search_neighbors=3,
    workers=1,
    return_extra=False,
    triage_quantile_per_cluster=0,
    amp_no_triaging=12,
    revert=False,
):

    """
    if l2_norm is passed as argument, it will be used to compute density and nhdn
    """
    n = len(X)

    inliers, kdtree = kdtree_inliers(
        X,
        kdtree=kdtree,
        n_neighbors=outlier_neighbor_count,
        distance_upper_bound=outlier_radius,
        workers=workers,
    )

    if l2_norm is None:
        do_ratio = int(sigma_regional is not None)
        density = get_smoothed_densities(
            X,
            inliers=inliers,
            sigmas=sigma_local,
            sigma_lows=sigma_local_low,
            revert=revert,
        )
        if density is None:
            return np.full(X.shape[0], -1)
        if do_ratio:
            reg_density = get_smoothed_densities(
                X, inliers=inliers, sigmas=sigma_regional, sigma_lows=sigma_regional_low
            )
            density = np.nan_to_num(density / reg_density)
            
        nhdn, distances, indices = nearest_higher_density_neighbor(
            kdtree,
            density,
            n_neighbors_search=n_neighbors_search,
            distance_upper_bound=radius_search,
            workers=workers,
        )
        
    else:
        # inliers don't matter here? Or should we still remove them?...
        # indices = np.full(l2_norm.shape[0], n)
        # l2_norm_inliers = l2_norm[inliers] #?? NO -> keep everything but only compute for inliers
        n = l2_norm.shape[0]
        indices = l2_norm.argsort()[:, :1 + n_neighbors_search]
        distances = l2_norm[np.arange(n)[:, None], indices]
        assert distances.shape == (l2_norm.shape[0], 1 + n_neighbors_search)
        density = np.median(distances, axis=1)
        density_padded = np.pad(density, (0, 1), constant_values=np.inf)
        is_higher_density = density_padded[indices] >= density[:, None]
        distances[is_higher_density] = np.inf
        indices[is_higher_density] = n
        nhdn = indices[np.arange(n), distances.argmin(1)]

    if noise_density and l2_norm is None:
        nhdn[density <= noise_density] = n
    nhdn = nhdn.astype(np.intc)
    has_nhdn = np.flatnonzero(nhdn < n).astype(np.intc)

    graph = coo_array(
        (np.ones(has_nhdn.size), (nhdn[has_nhdn], has_nhdn)), shape=(n, n)
    )
    ncc, labels = connected_components(graph)

    if remove_borders:
        labels = remove_border_points(
            labels,
            density,
            kdtree,
            search_radius=border_search_radius,
            n_neighbors_search=border_search_neighbors,
            workers=workers,
        )

    if remove_clusters_smaller_than:
        labels = decrumb(labels, min_size=remove_clusters_smaller_than)

    if triage_quantile_per_cluster > 0:
        for k in np.unique(labels[labels > -1]):
            idx_label = np.flatnonzero(labels == k)
            amp_vec = X[idx_label, 2]
            # triage_quantile_unit = triage_quantile_per_cluster
            if l2_norm is None:
                q = np.quantile(density[idx_label], triage_quantile_per_cluster)
                spikes_to_remove = np.flatnonzero(
                    np.logical_and(
                        density[idx_label] < q,
                        amp_vec < amp_no_triaging,
                    )
                )
            else:
                q = np.quantile(density[idx_label], 1-triage_quantile_per_cluster)
                spikes_to_remove = np.flatnonzero(
                    np.logical_and(
                        density[idx_label] > q,
                        amp_vec < amp_no_triaging,
                    )
                )

            labels[idx_label[spikes_to_remove]] = -1

    if not return_extra:
        return labels

    return dict(
        density=density,
        nhdn=nhdn,
        labels=labels,
    )
