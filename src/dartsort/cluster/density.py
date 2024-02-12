import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree


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


def get_smoothed_densities(X, inliers=slice(None), sigmas=None):
    """Get RBF density estimates for each X[i] and bandwidth in sigmas

    Outliers will be marked with NaN KDEs. Please pass inliers, or else your
    histogram is liable to be way too big.
    """
    infeats = X[inliers]
    extents = np.c_[np.floor(infeats.min(0)), np.ceil(infeats.max(0))]
    bin_edges = [np.arange(e[0], e[1] + 1) for e in extents]
    raw_histogram, bin_edges = np.histogramdd(infeats, bins=bin_edges)
    bin_centers = [0.5 * (be[1:] + be[:-1]) for be in bin_edges]

    kdes = []
    for sigma in list(sigmas):
        hist = raw_histogram
        if sigma is not None:
            hist = gaussian_filter(raw_histogram, sigma)

        lerp = RegularGridInterpolator(bin_centers, hist, bounds_error=False)
        kdes.append(lerp(X))

    kdes = kdes[0] if len(kdes) == 1 else kdes
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
    kdtree=None,
    sigma_local=5.0,
    sigma_regional=None,
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
):
    n = len(X)

    inliers, kdtree = kdtree_inliers(
        X,
        kdtree=kdtree,
        n_neighbors=outlier_neighbor_count,
        distance_upper_bound=outlier_radius,
        workers=workers,
    )

    sigmas = [sigma_local] + ([sigma_regional] * int(sigma_regional is not None))
    density = get_smoothed_densities(X, inliers=inliers, sigmas=sigmas)
    if sigma_regional is not None:
        density = density[0] / density[1]

    nhdn, distances, indices = nearest_higher_density_neighbor(
        kdtree,
        density,
        n_neighbors_search=n_neighbors_search,
        distance_upper_bound=radius_search,
        workers=workers,
    )
    if noise_density:
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

    if not return_extra:
        return labels

    return dict(
        density=density,
        nhdn=nhdn,
        labels=labels,
    )
