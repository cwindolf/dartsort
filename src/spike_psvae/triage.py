import numpy as np
from sklearn.neighbors import KernelDensity
import networkx as nx
from scipy.spatial import KDTree
import scipy


def weighted_knn_triage(
    xyzap, feats, scales=[1, 10, 1, 15, 30, 10, 10, 10], percentile=85
):
    vecs = np.c_[xyzap, feats] @ np.diag(scales)
    log_ptp = xyzap[:, 4]
    tree = KDTree(vecs)
    dist, ind = tree.query(vecs, k=6)
    dist = np.sum(dist / log_ptp[:, None], 1)

    idx_keep1 = dist <= np.percentile(dist, percentile)

    return idx_keep1


def run_weighted_triage_low_ptp(
    x,
    z,
    maxptps,
    scales=(1, 1, 30),
    threshold=80,
    ptp_low_threshold=3,
    ptp_high_threshold=6,
    c=1,
):
    # only triage high ptp spikes
    high_ptp_filter = maxptps < ptp_high_threshold
    high_ptp_filter = np.flatnonzero(high_ptp_filter)
    x = x[high_ptp_filter]
    z = z[high_ptp_filter]
    maxptps = maxptps[high_ptp_filter]

    # filter by low ptp
    low_ptp_filter = maxptps > ptp_low_threshold
    low_ptp_filter = np.flatnonzero(low_ptp_filter)
    x = x[low_ptp_filter]
    z = z[low_ptp_filter]
    maxptps = maxptps[low_ptp_filter]

    feats = np.c_[scales[0] * x, scales[1] * z, scales[2] * np.log(maxptps)]

    tree = KDTree(feats)
    dist, ind = tree.query(feats, k=6)
    dist = dist[:, 1:]
    dist = np.sum(
        c * np.log(dist) + np.log(1 / (scales[2] * np.log(maxptps)))[:, None],
        1,
    )
    idx_keep = dist <= np.percentile(dist, threshold)

    triaged_x = x[idx_keep]
    triaged_z = z[idx_keep]
    triaged_maxptps = maxptps[idx_keep]

    return idx_keep, high_ptp_filter, low_ptp_filter


def run_weighted_triage_adaptive(
    x,
    z,
    maxptps,
    scales=(1, 1, 50),
    log_c=5,
    threshold=80,
    ptp_low_threshold=3,
    bin_size=5,
    region_size=25,
):

    # filter by low ptp
    low_ptp_filter = maxptps > ptp_low_threshold
    low_ptp_filter = np.flatnonzero(low_ptp_filter)
    x = x[low_ptp_filter]
    z = z[low_ptp_filter]
    maxptps = maxptps[low_ptp_filter]

    # get local distances
    feats = np.c_[
        scales[0] * x, scales[1] * z, scales[2] * np.log(log_c + maxptps)
    ]

    tree = KDTree(feats)
    dist, ind = tree.query(feats, k=6)
    dist = dist[:, 1:]
    dist = np.mean(dist, 1)

    spike_region_dist = 1
    if region_size is not None:
        # bin spikes
        min_z, max_z = np.min(z), np.max(z)
        bins = np.arange(min_z, max_z, bin_size)
        binned_z = np.digitize(z, bins, right=False)

        # get mean distance in bin
        binned_dist = []
        spikes_in_bin_list = []
        for bin_id in range(len(bins)):
            spikes_in_bin = np.where(binned_z == bin_id)[0]
            spikes_in_bin_list.append(len(spikes_in_bin))
            if len(spikes_in_bin) == 0:
                binned_mean_dist = np.max(dist)
            else:
                binned_mean_dist = np.mean(dist[np.where(binned_z == bin_id)])
            binned_dist.append(binned_mean_dist)

        # gaussian filter sigma=region_size/bin_size
        region_dist = scipy.ndimage.gaussian_filter(
            binned_dist, sigma=region_size / bin_size, mode="constant"
        )
        spike_region_dist_func = scipy.interpolate.interp1d(
            bins, region_dist, fill_value="extrapolate"
        )
        spike_region_dist = spike_region_dist_func(z)

    true_dist = (
        np.log(dist)
        - np.log(spike_region_dist)
        + np.log(1 / (scales[2] * np.log(maxptps)))
    )

    idx_keep = true_dist <= np.percentile(true_dist, threshold)

    triaged_x = x[idx_keep]
    triaged_z = z[idx_keep]
    triaged_maxptps = maxptps[idx_keep]

    return triaged_x, triaged_z, triaged_maxptps, idx_keep, low_ptp_filter


def run_weighted_triage(
    x,
    y,
    z,
    alpha,
    maxptps,
    pcs=None,
    scales=(1, 10, 1, 15, 30, 10),
    threshold=80,
    ptp_threshold=3,
    c=1,
    mask=None,
):
    ptp_filter = maxptps > ptp_threshold
    ptp_filter = np.flatnonzero(ptp_filter)
    x = x[ptp_filter]
    y = y[ptp_filter]
    z = z[ptp_filter]
    alpha = alpha[ptp_filter]
    maxptps = maxptps[ptp_filter]
    if pcs is not None:
        pcs = pcs[ptp_filter]
        feats = np.c_[
            scales[0] * x,
            scales[1] * np.log(y),
            scales[2] * z,
            scales[3] * np.log(alpha),
            scales[4] * np.log(maxptps),
            scales[5] * pcs[:, :3],
        ]
    else:
        feats = np.c_[
            scales[0] * x,
            # scales[1]*np.log(y),
            scales[2] * z,
            # scales[3]*np.log(alpha),
            scales[4] * np.log(maxptps),
        ]

    tree = KDTree(feats)
    dist, ind = tree.query(feats, k=6)
    dist = dist[:, 1:]
    dist = np.sum(
        c * np.log(dist) + np.log(1 / (scales[4] * np.log(maxptps)))[:, None],
        1,
    )
    idx_keep = dist <= np.percentile(dist, threshold)

    triaged_x = x[idx_keep]
    triaged_y = y[idx_keep]
    triaged_z = z[idx_keep]
    triaged_alpha = alpha[idx_keep]
    triaged_maxptps = maxptps[idx_keep]
    triaged_pcs = None
    if pcs is not None:
        triaged_pcs = pcs[idx_keep]

    return (
        triaged_x,
        triaged_y,
        triaged_z,
        triaged_alpha,
        triaged_maxptps,
        triaged_pcs,
        ptp_filter,
        idx_keep,
    )


def weighted_triage_ix(
    x,
    y,
    z,
    alpha,
    maxptps,
    pcs=None,
    scales=(1, 10, 1, 15, 30, 10),
    threshold=100,
    ptp_threshold=3,
    c=1,
):
    ptp_filter = maxptps > ptp_threshold
    x = x[ptp_filter]
    y = y[ptp_filter]
    z = z[ptp_filter]
    alpha = alpha[ptp_filter]
    maxptps = maxptps[ptp_filter]
    if pcs is not None:
        # pcs = pcs[ptp_filter]
        feats = np.c_[
            scales[0] * x,
            scales[1] * np.log(y),
            scales[2] * z,
            scales[3] * np.log(alpha),
            scales[4] * np.log(maxptps),
            scales[5] * pcs[:, :3],
        ]
    else:
        feats = np.c_[
            scales[0] * x,
            # scales[1]*np.log(y),
            scales[2] * z,
            # scales[3]*np.log(alpha),
            scales[4] * np.log(maxptps),
        ]

    tree = KDTree(feats)
    dist, ind = tree.query(feats, k=6)
    print(dist.shape)
    dist = dist[:, 1:]
    print(np.isfinite(np.log(dist)).all(axis=1).mean())
    dist = np.sum(
        c * np.log(dist) + np.log(1 / (scales[4] * np.log(maxptps)))[:, None],
        1,
    )
    idx_keep = dist <= np.percentile(dist, threshold)
    idx_keep_full = ptp_filter
    idx_keep_full[np.flatnonzero(idx_keep_full)[~idx_keep]] = 0
    return idx_keep_full


def coarse_split(
    xyzp,
    feats,
    scales=[1, 10, 1, 15, 30, 10, 10, 10],
):
    vecs = np.c_[xyzp, feats] @ np.diag(scales)
    tree = KDTree(vecs)
    dist, ind = tree.query(vecs, k=6)
    dist = np.mean(dist, 1)
    print(dist.mean() + 6 * dist.std(), dist.mean(), dist.std())
    W = tree.sparse_distance_matrix(
        tree, max_distance=dist.mean() + 6 * dist.std()
    )
    G = nx.convert_matrix.from_scipy_sparse_matrix(W)
    components = list(nx.algorithms.components.connected_components(G))
    print(len(components), flush=True)
    labels = np.zeros(xyzp.shape[0], dtype=np.int)
    for k in range(len(components)):
        mask = np.isin(np.arange(xyzp.shape[0]), list(components[k]))
        labels[mask] = k
    return labels
