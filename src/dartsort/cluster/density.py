from collections.abc import Sequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from scipy.stats import bernoulli
import multiprocessing


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
        X,
        k=1 + n_neighbors,
        distance_upper_bound=distance_upper_bound,
        workers=workers,
    )
    inliers = (indices[:, 1:] < len(X)).sum(1) >= n_neighbors

    return inliers, kdtree


def get_smoothed_densities(
    X,
    inliers=slice(None),
    sigmas=None,
    weights=None,
    return_hist=False,
    sigma_lows=None,
    sigma_ramp_ax=-1,
    bin_sizes=None,
    bin_size_ratio=10.0,
    min_bin_size=0.1,
    ramp_min_bin_size=5.0,
    max_n_bins=128,
    min_n_bins=5,
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
    dextents = np.ptp(extents, 1)
    if not (dextents > 0).all():
        raise ValueError(f"Issue in KDE. {dextents=} {infeats.shape=} {sigmas=} {bin_sizes=}.")
    nbins = np.ceil(dextents / bin_sizes).astype(int)
    nbins = nbins.clip(min_n_bins, max_n_bins)
    bin_edges = [
        np.linspace(e[0], e[1], num=nb + 1)
        for e, nb in zip(extents, nbins)
    ]

    # compute histogram and figure out how big the bins actually were
    raw_histogram, bin_edges = np.histogramdd(infeats, bins=bin_edges, weights=weights)
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
        low = np.array([np.min(bc) for bc in bin_centers])
        high = np.array([np.max(bc) for bc in bin_centers])
        dens = lerp(X.clip(low, high))
        kdes.append(dens)

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
    units[np.logical_not(big_enough)] = -1
    units[big_enough] = np.arange(big_enough.sum())
    return units[labels]


def density_peaks(
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
    workers=-1,
    return_extra=False,
):
    """Density peaks clustering as described by Rodriguez and Laio, but...

    This implementation has three tweaks from original DPC
     - KDE density estimation, implemented by smoothing histograms
     - "Sharpening": density can be a ratio of KDEs, like a sharpening kernel
     - Noise: you can throw away points with too low of a density (ratio)
    """
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
    else:
        density = density[0]

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
    data = np.ones(has_nhdn.size)
    graph = coo_array((data, (nhdn[has_nhdn], has_nhdn)), shape=(n, n))
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

    return dict(
        density=density,
        nhdn=nhdn,
        labels=labels,
    )


# -- version used in UHD project


def density_peaks_fancy(
    xyza,
    amps,
    to_cluster,
    sorting,
    motion_est,
    clustering_config,
    ramp_num_spikes=[10, 60],
    ramp_ptp=[2, 6],
):
    z = xyza[to_cluster, 2]
    if motion_est is not None:
        z = motion_est.correct_s(sorting.times_seconds[to_cluster], z)
    z_not_reg = xyza[to_cluster, 2]
    ampfeat = clustering_config.amp_scale * np.log(
        clustering_config.amp_log_c + amps[to_cluster]
    )
    res = density.density_peaks_clustering(
        np.c_[scales[0] * xyza[to_cluster, 0], scales[1] * z, ampfeat],
        geom=geom,
        y=xyza[to_cluster, 1],
        z_not_reg=z_not_reg,
        use_y_triaging=clustering_config.use_y_triaging,
        sigma_local=clustering_config.sigma_local,
        sigma_local_low=clustering_config.sigma_local_low,
        sigma_regional=clustering_config.sigma_regional,
        sigma_regional_low=clustering_config.sigma_regional_low,
        n_neighbors_search=clustering_config.n_neighbors_search,
        radius_search=clustering_config.radius_search,
        remove_clusters_smaller_than=clustering_config.remove_clusters_smaller_than,
        noise_density=clustering_config.noise_density,
        triage_quantile_per_cluster=clustering_config.triage_quantile_per_cluster,
        ramp_triage_per_cluster=clustering_config.ramp_triage_per_cluster,
        revert=clustering_config.revert,
        triage_quantile_before_clustering=clustering_config.triage_quantile_before_clustering,
        amp_no_triaging_before_clustering=clustering_config.amp_no_triaging_before_clustering,
        amp_no_triaging_after_clustering=clustering_config.amp_no_triaging_after_clustering,
        distance_dependent_noise_density=clustering_config.distance_dependent_noise_density,
        outlier_radius=clustering_config.outlier_radius,
        outlier_neighbor_count=clustering_config.outlier_neighbor_count,
        scales=scales,
        log_c=clustering_config.log_c,
        workers=clustering_config.workers,
        return_extra=clustering_config.attach_density_feature,
    )

    if clustering_config.remove_small_far_clusters:
        if clustering_config.attach_density_feature:
            labels_sort = res["labels"]
        else:
            labels_sort = res
        z = xyza[to_cluster, 2]
        if motion_est is not None:
            z = motion_est.correct_s(times_s[to_cluster], z)
        all_med_ptp = []
        all_med_z_spread = []
        all_med_x_spread = []
        num_spikes = []
        for k in np.unique(labels_sort)[np.unique(labels_sort)>-1]:
            all_med_ptp.append(np.median(amps[to_cluster[labels_sort == k]]))
            all_med_x_spread.append(xyza[to_cluster[labels_sort == k], 0].std())
            all_med_z_spread.append(z[labels_sort == k].std())
            num_spikes.append((labels_sort == k).sum())

        all_med_ptp = np.array(all_med_ptp)
        all_med_x_spread = np.array(all_med_x_spread)
        all_med_z_spread = np.array(all_med_z_spread)
        num_spikes = np.array(num_spikes)

        # ramp from ptp 2 to 6 with n spikes from 60 to 10 per minute!
        idx_low = np.flatnonzero(np.logical_and(
            np.isin(labels_sort, np.flatnonzero(num_spikes<=(chunk_time_range_s[1]-chunk_time_range_s[0])/60*(ramp_num_spikes[1] - (all_med_ptp - ramp_ptp[0])/(ramp_ptp[1]-ramp_ptp[0])*(ramp_num_spikes[1]-ramp_num_spikes[0])))),
            np.isin(labels_sort, np.flatnonzero(all_med_ptp<=ramp_ptp[1]))
        ))
        if clustering_config.attach_density_feature:
            res["labels"][idx_low] = -1
        else:
            res[idx_low] = -1
    return res

def density_peaks_clustering(
    X,
    geom=None,
    y=None,
    z_not_reg=None,
    use_y_triaging=False,
    l2_norm=None,
    kdtree=None,
    sigma_local=5.0,
    sigma_local_low=None,
    sigma_regional=None,
    sigma_regional_low=None,
    min_bin_size=1.0,
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
    triage_quantile_before_clustering=0,
    amp_no_triaging_before_clustering=6,
    ramp_triage_before_clustering=False,
    radius_triage_before_clustering=75,
    triage_quantile_per_cluster=0,
    amp_no_triaging_after_clustering=12,
    ramp_triage_per_cluster=False,
    revert=False,
    distance_dependent_noise_density=False,
    amp_lowest_noise_density=8,
    min_distance_noise_density=0,
    min_distance_noise_density_10=200,
    max_noise_density=10,
    max_n_bins=128,
    scales=None,
    log_c=None,
):
    """
    if l2_norm is passed as argument, it will be used to compute density and nhdn
    """
    # n = len(X)
    if workers < 0:
        workers = multiprocessing.cpu_count() + workers + 1

    if ramp_triage_before_clustering and geom is not None:
        inliers_first = np.ones(len(X)).astype("bool")
        idx_low_ptp = np.flatnonzero(
            X[:, 2] < scales[2] * np.log(log_c + amp_no_triaging_before_clustering)
        )
        distances_to_geom = (
            np.sqrt((X[idx_low_ptp, :2][None] - geom[:, None]) ** 2).sum(2)
        ).min(0)
        probabilities = (
            np.minimum(distances_to_geom, radius_triage_before_clustering)
            / radius_triage_before_clustering
        )
        which_to_discard = bernoulli.rvs(probabilities).astype("bool")
        inliers_first[idx_low_ptp[which_to_discard]] = False
        inliers_first = np.arange(len(X))[inliers_first]
    else:
        inliers_first = np.arange(len(X))

    n = len(inliers_first)
    if n <= 1:
        if return_extra:
            return dict(labels=np.full(X.shape[0], -1))
        return np.full(X.shape[0], -1)

    if use_y_triaging and y is not None:
        inliers, _ = kdtree_inliers(
            np.c_[X[inliers_first], y[inliers_first]],
            kdtree=kdtree,
            n_neighbors=outlier_neighbor_count,
            distance_upper_bound=outlier_radius * np.sqrt(4 / 3),
            workers=workers,
        )
        _, kdtree = kdtree_inliers(
            X[inliers_first],
            kdtree=kdtree,
            n_neighbors=outlier_neighbor_count,
            distance_upper_bound=outlier_radius,
            workers=workers,
        )
    else:
        inliers, kdtree = kdtree_inliers(
            X[inliers_first],
            kdtree=kdtree,
            n_neighbors=outlier_neighbor_count,
            distance_upper_bound=outlier_radius,
            workers=workers,
        )

    if not inliers.sum() > 1:
        if return_extra:
            return dict(labels=np.full(X.shape[0], -1))
        return np.full(X.shape[0], -1)

    # inliers = inliers_first[inliers]

    if isinstance(sigma_local, str) and sigma_local.startswith("rule_of_thumb"):
        factor = 1
        if "*" in sigma_local:
            factor = float(sigma_local.split("*")[1].strip())
        sigma_local = (
            1.06
            * factor
            * np.linalg.norm(np.std(X[inliers_first][inliers], axis=0))
            * np.power(inliers.sum(), -0.2)
        )
        if sigma_local <= 0:
            raise ValueError(
                f"rule of thumb problems. {sigma_local=} "
                f"{np.std(X[inliers_first][inliers], axis=0)=} "
                f"{len(X)=} {np.power(len(X), -0.2)=} "
                f"{X.shape=} {inliers_first.shape=} {inliers.sum()=} "
                f"{X[inliers_first][inliers].shape=}"
            )
        if sigma_regional == "rule_of_thumb":
            radius_search = radius_search * sigma_local
            sigma_regional = 10 * sigma_local
        else:
            radius_search = radius_search * sigma_local

    if l2_norm is None:
        do_ratio = sigma_regional is not None
        density = get_smoothed_densities(
            X[inliers_first],
            inliers=inliers,
            sigmas=sigma_local,
            sigma_lows=sigma_local_low,
            revert=revert,
            min_bin_size=min_bin_size,
            max_n_bins=max_n_bins,
        )
        if density is None:
            if return_extra:
                return dict(labels=np.full(X.shape[0], -1))
            return np.full(X.shape[0], -1)
        if do_ratio:
            reg_density = get_smoothed_densities(
                X[inliers_first],
                inliers=inliers,
                sigmas=sigma_regional,
                sigma_lows=sigma_regional_low,
                min_bin_size=min_bin_size,
            )
            density = density / reg_density
        density = np.nan_to_num(density)

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
        l2_norm = l2_norm[inliers_first][:, inliers_first]
        n = l2_norm.shape[0]
        indices = l2_norm.argsort()[:, : 1 + n_neighbors_search]
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

    if distance_dependent_noise_density and noise_density is not None:
        dist = np.sqrt(
            (
                (
                    np.c_[X[inliers_first, 0], z_not_reg[inliers_first]][:, None]
                    - geom[None]
                )
                ** 2
            ).sum(2)
        ).min(1)
        noise_density_dist = (
            np.minimum(
                np.maximum(dist - min_distance_noise_density, 0),
                min_distance_noise_density_10 - min_distance_noise_density,
            )
            * (max_noise_density - noise_density)
            / (min_distance_noise_density_10 - min_distance_noise_density)
            + noise_density
        )
        noise_density_dist[
            X[inliers_first, 2] > scales[2] * np.log(log_c + amp_lowest_noise_density)
        ] = 2
        nhdn[density <= noise_density_dist] = n
    if triage_quantile_before_clustering and l2_norm is None:
        q = np.quantile(density, triage_quantile_before_clustering)
        idx_triaging = np.flatnonzero(
            np.logical_and(
                X[inliers_first, 2]
                < scales[2] * np.log(log_c + amp_no_triaging_before_clustering),
                density < q,
            )
        )
        nhdn[idx_triaging] = n

    nhdn = nhdn.astype(np.intc)
    has_nhdn = np.flatnonzero(nhdn < n).astype(np.intc)
    rc = (
        np.concatenate([nhdn[has_nhdn], has_nhdn]),
        np.concatenate([has_nhdn, nhdn[has_nhdn]]),
    )

    graph = coo_array((np.ones(2 * has_nhdn.size), rc), shape=(n, n))
    ncc, labels = connected_components(graph, directed=False)

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

    if triage_quantile_per_cluster:
        amp_no_triaging_after_clustering = scales[2] * np.log(
            log_c + amp_no_triaging_after_clustering
        )
        for k in np.unique(labels[labels > -1]):
            idx_label = np.flatnonzero(labels == k)
            amp_vec = X[inliers_first][idx_label, 2]
            med_amp = np.median(amp_vec)
            if med_amp < amp_no_triaging_after_clustering:
                if ramp_triage_per_cluster:
                    triage_quantile_per_cluster = (
                        amp_no_triaging_after_clustering - med_amp
                    ) / amp_no_triaging_after_clustering
                # triage_quantile_unit = triage_quantile_per_cluster
                if l2_norm is None:
                    q = np.quantile(density[idx_label], triage_quantile_per_cluster)
                    spikes_to_remove = np.flatnonzero(
                        np.logical_and(
                            density[idx_label] < q,
                            amp_vec < amp_no_triaging_after_clustering,
                        )
                    )
                else:
                    q = np.quantile(density[idx_label], 1 - triage_quantile_per_cluster)
                    spikes_to_remove = np.flatnonzero(
                        np.logical_and(
                            density[idx_label] > q,
                            amp_vec < amp_no_triaging_after_clustering,
                        )
                    )

                labels[idx_label[spikes_to_remove]] = -1

    labels_all = np.full(len(X), -1)
    labels_all[inliers_first] = labels

    if not return_extra:
        return labels_all

    density_all = np.zeros(len(X))
    nhdn_all = np.full(len(X), n)
    density_all[inliers_first] = density
    nhdn_all[inliers_first] = nhdn

    return dict(
        density=density_all,
        nhdn=nhdn_all,
        labels=labels_all,
    )

def mad(x, axis=0):
    x = x - np.median(x, axis=axis, keepdims=True)
    np.abs(x, out=x)
    return np.median(x, axis=axis)
