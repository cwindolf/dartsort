import dataclasses
from logging import getLogger

import h5py
try:
    from hdbscan import HDBSCAN
except ImportError:
    from sklearn.cluster import HDBSCAN

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import KDTree
from sklearn.neighbors import KNeighborsClassifier

from dartsort.util import data_util, drift_util, waveform_util
from dredge.motion_util import IdentityMotionEstimate


logger = getLogger(__name__)


def agglomerate(labels, distances, linkage_method="complete", threshold=1.0):
    """"""
    n = distances.shape[0]
    if n <= 1:
        return labels, np.arange(n)
    pdist = distances[np.triu_indices(n, k=1)]
    if pdist.min() > threshold:
        if labels is None:
            return None, np.arange(n)
        else:
            ids = np.unique(labels)
            return labels, ids[ids >= 0]

    finite = np.isfinite(pdist)
    if not finite.all():
        inf = max(0, pdist[finite].max()) + threshold + 1.0
        pdist[np.logical_not(finite)] = inf

    Z = linkage(pdist, method=linkage_method)
    new_ids = fcluster(Z, threshold, criterion="distance")
    # offset by 1, I think always, but I don't want to be wrong?
    new_ids -= new_ids.min()

    if labels is None:
        new_labels = None
    else:
        kept = np.flatnonzero(labels >= 0)
        new_labels = np.full_like(labels, -1)
        new_labels[kept] = new_ids[labels[kept]]

    return new_labels, new_ids


def leafsets(Z, max_distance=np.inf):
    """For a linkage Z, get the leaves in each non-leaf cluster."""
    n = len(Z) + 1
    leaves = {}
    for i, row in enumerate(Z):
        pa, pb, dist, nab = row
        if dist > max_distance:
            break
        leavesa = leaves.get(pa, [int(pa)])
        leavesb = leaves.get(pb, [int(pb)])
        leaves[n + i] = leavesa + leavesb
        leaves[n + i].sort()
    return leaves


def is_largest_set_smaller_than(Z, leaf_descendants, max_size=5):
    n_branches = len(Z)
    n_units = n_branches + 1
    indicator = np.zeros(n_branches, dtype=bool)
    # this walks up from the leaves
    # at each branch, if #descendents<size, branch gets true and
    # its parents are set to false. that way only one ancestor
    # is true for each leaf.
    for i, (pa, pb, dist, nab) in enumerate(Z):
        sz = len(leaf_descendants[n_units + i])
        if sz > max_size:
            continue
        pa = int(pa)
        pb = int(pb)
        indicator[i] = True
        if pa >= n_units:
            indicator[pa - n_units] = False
        if pb >= n_units:
            indicator[pb - n_units] = False
    # assert that at most one ancestor is true for each leaf
    counts = np.zeros(n_units, dtype=int)
    for i, ind in enumerate(indicator):
        counts[leaf_descendants[n_units + i]] += ind
    assert counts.max() <= 1
    return indicator


def combine_distances(
    distances,
    thresholds,
    agg_function=np.maximum,
    sym_function=np.maximum,
):
    """Combine several distance matrices and symmetrize them

    They have different reference thresholds, but the result of this function
    has threshold 1.
    """
    dists = distances[0] / thresholds[0]
    for dist, thresh in zip(distances[1:], thresholds[1:]):
        dists = agg_function(dists, dist / thresh)
    return sym_function(dists, dists.T)


def combine_disjoint(inds_a, labels_a, inds_b, labels_b):
    labels = np.full(labels_a.size + labels_b.size, -1, dtype=labels_a.dtype)
    labels[inds_a] = labels_a
    labels[inds_b] = labels_b
    return labels


def reorder_by_depth(sorting, motion_est=None):
    kept = np.flatnonzero(sorting.labels >= 0)
    kept_labels = sorting.labels[kept]

    units, kept_labels = np.unique(kept_labels, return_inverse=True)

    depths = sorting.point_source_localizations[kept, 2]
    if motion_est is not None:
        depths = motion_est.correct_s(sorting.times_seconds[kept], depths)

    centroids = np.zeros(units.size)
    for u in range(units.size):
        inu = np.flatnonzero(kept_labels == u)
        centroids[u] = np.median(depths[inu])

    labels = sorting.labels.copy()
    # this one is some food for thought, lol.
    labels[kept] = np.argsort(np.argsort(centroids))[kept_labels]

    return dataclasses.replace(sorting, labels=labels)


def closest_registered_channels(
    times_seconds, x, z_abs, geom, z_reg=None, motion_est=None
):
    """Assign spikes to the drift-extended channel closest to their registered position"""
    if motion_est is None:
        motion_est = IdentityMotionEstimate()
    registered_geom = drift_util.registered_geometry(geom, motion_est)
    if z_reg is None:
        z_reg = motion_est.correct_s(times_seconds, z_abs)
    reg_pos = np.c_[x, z_reg]

    registered_kdt = KDTree(registered_geom)
    _, reg_channels = registered_kdt.query(reg_pos)

    return reg_channels


def grid_snap(
    times_seconds, x, z_abs, geom, grid_dx=15., grid_dz=15., z_reg=None, motion_est=None
):
    if motion_est is None:
        motion_est = IdentityMotionEstimate()
    if z_reg is None:
        z_reg = motion_est.correct_s(times_seconds, z_abs)
    reg_pos = np.c_[x, z_reg]

    # make a grid inside the registered geom bounding box
    registered_geom = drift_util.registered_geometry(geom, motion_est)
    min_x, max_x = registered_geom[:, 0].min(), registered_geom[:, 0].max()
    min_z, max_z = registered_geom[:, 1].min(), registered_geom[:, 1].max()
    grid_x = np.arange(min_x, max_x, grid_dx)
    grid_x += (min_x + max_x) / 2 - grid_x.mean()
    grid_z = np.arange(min_z, max_z, grid_dz)
    grid_z += (min_z + max_z) / 2 - grid_z.mean()
    grid_xx, grid_zz = np.meshgrid(grid_x, grid_z, indexing="ij")
    grid = np.c_[grid_xx.ravel(), grid_zz.ravel()]

    # snap to closest grid point
    registered_kdt = KDTree(grid)
    _, reg_channels = registered_kdt.query(reg_pos)

    return reg_channels


def recursive_hdbscan_clustering(
    features,
    min_cluster_size=25,
    min_samples=25,
    cluster_selection_epsilon=1,
    recursive=True,
):
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        min_samples=min_samples,
    )
    clusterer.fit(features)

    if not recursive:
        return clusterer.labels_

    # recursively split clusters as long as HDBSCAN keeps finding more than 1
    units = np.unique(clusterer.labels_)
    if units[units >= 0].size <= 1:
        return np.zeros_like(clusterer.labels_)

    # else, recursively enter all labels and split them
    labels = clusterer.labels_.copy()
    next_label = units.max() + 1
    for unit in units[units >= 0]:
        in_unit = np.flatnonzero(clusterer.labels_ == unit)
        split_labels = recursive_hdbscan_clustering(
            features[in_unit],
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            recursive=recursive,
        )
        kept = split_labels >= 0
        dropped = np.logical_not(kept)
        labels[in_unit[dropped]] = split_labels[dropped]
        labels[in_unit[kept]] = split_labels[kept] + next_label
        next_label += split_labels.max() + 1

    # reindex
    _, labels[labels >= 0] = np.unique(labels[labels >= 0], return_inverse=True)
    return labels


def knn_reassign_outliers(labels, features):
    outliers = labels < 0
    outliers_idx = np.flatnonzero(outliers)
    if not outliers_idx.size:
        return labels
    knn = KNeighborsClassifier()
    knn.fit(features[~outliers], labels[~outliers])
    new_labels = labels.copy()
    new_labels[outliers_idx] = knn.predict(features[outliers_idx])
    return new_labels


def get_main_channel_pcs(
    sorting,
    which=slice(None),
    rank=1,
    show_progress=False,
    dataset_name="collisioncleaned_tpca_features",
):
    mask = np.zeros(len(sorting), dtype=bool)
    mask[which] = True
    channels = sorting.channels[which]

    features = getattr(sorting, "collisioncleaned_tpca_features", None)
    channel_index = getattr(sorting, "channel_index", None)
    if features is not None and channel_index is not None:
        features = features[which][:, :rank]
        return waveform_util.grab_main_channels(features, channels, channel_index)

    features = np.empty((mask.sum(), rank), dtype=np.float32)
    with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
        feats_dset = h5[dataset_name]
        channel_index = h5["channel_index"][:]
        for ixs, feats in data_util.yield_masked_chunks(
            mask, feats_dset, show_progress=show_progress, desc_prefix="Main channel"
        ):
            feats = feats[:, :rank]
            feats = waveform_util.grab_main_channels(
                feats, channels[ixs], channel_index
            )
            features[ixs] = feats
    return features


def decrumb(labels, min_size=5, in_place=False, flatten=True):
    kept = np.flatnonzero(labels >= 0)
    labels_kept = labels[kept]
    labels = labels if in_place else labels.copy()

    units_sparse, counts_sparse = np.unique(labels_kept, return_counts=True)
    if not units_sparse.size:
        return labels

    k = units_sparse.max() + 1
    counts = np.zeros(k, dtype=counts_sparse.dtype)
    counts[units_sparse] = counts_sparse
    units = np.arange(k)

    big_enough = counts >= min_size
    k1 = big_enough.sum()
    units[np.logical_not(big_enough)] = -1
    if flatten:
        units[big_enough] = np.arange(k1)

    logger.dartsortdebug(f"decrumb: {k}->{k1}.")

    labels[kept] = units[labels_kept]

    return labels
