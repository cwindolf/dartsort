from typing import cast

import h5py
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import KDTree

from ..util import data_util, waveform_util
from ..util.data_util import DARTsortSorting
from ..util.logging_util import get_logger
from ..util.motion import MotionInfo

logger = get_logger(__name__)


def recluster(
    *,
    sorting: DARTsortSorting,
    dists: np.ndarray,
    unit_ids: np.ndarray | None = None,
    shifts: np.ndarray | None = None,
    unit_snrs: np.ndarray | None = None,
    threshold=0.25,
    link="complete",
):
    """Distance-based hierarchical clustering of units

    Parameters
    ----------
    sorting : DARTsortSorting
    dists: np.ndarray
    unit_ids: np.ndarray | None, default None
    shifts: np.ndarray | None, default None
        shifts[i, j] is how far ahead unit i is from unit j, so, it's
        like trough[i] - trough[j]
    unit_snrs: np.ndarray | None, default None,
    threshold=0.25,
    link="complete",
    """
    new_labels, new_ids = hierarchical_cluster(
        sorting.labels, dists, linkage_method=link, threshold=threshold
    )
    if unit_ids is not None:
        assert np.array_equal(unit_ids, np.arange(dists.shape[0]))
    assert new_labels is not None
    new_sorting = apply_reclustering(
        sorting=sorting, merge_mapping=new_ids, shifts=shifts, unit_snrs=unit_snrs
    )
    return new_sorting, new_ids


def apply_reclustering(
    sorting: DARTsortSorting,
    merge_mapping: np.ndarray,
    new_labels: np.ndarray | None = None,
    shifts: np.ndarray | None = None,
    unit_snrs: np.ndarray | None = None,
) -> DARTsortSorting:
    assert sorting.labels is not None

    if new_labels is None:
        new_labels = np.full_like(sorting.labels, -1)
        kept = np.flatnonzero(sorting.labels >= 0)
        new_labels[kept] = merge_mapping[sorting.labels[kept]]

    if shifts is None:
        return sorting.ephemeral_replace(labels=new_labels)
    assert unit_snrs is not None

    # find original labels in each cluster
    clust_inverse = {i: [] for i in merge_mapping}
    for orig_label, new_label in enumerate(merge_mapping):
        clust_inverse[new_label].append(orig_label)

    # align to best snr unit
    times_updated = sorting.times_samples.copy()
    for new_label, orig_labels in clust_inverse.items():
        # we don't need to realign clusters which didn't change
        if len(orig_labels) <= 1:
            continue

        orig_snrs = unit_snrs[orig_labels]
        best_orig = orig_labels[orig_snrs.argmax()]
        for ogl in np.setdiff1d(orig_labels, [best_orig]):
            in_orig_unit = np.flatnonzero(sorting.labels == ogl)
            # this is like trough[best] - trough[ogl]
            shift_og_best = shifts[best_orig, ogl]
            # if >0, trough of og is behind trough of best.
            # subtracting will move trough of og to the right.
            times_updated[in_orig_unit] -= shift_og_best

    return sorting.ephemeral_replace(times_samples=times_updated, labels=new_labels)


def hierarchical_cluster(
    labels: np.ndarray | None,
    distances: np.ndarray,
    linkage_method="complete",
    threshold=1.0,
    eps=1e-5,
):
    """"""
    n = distances.shape[0]
    assert eps < threshold  # that would be confusing.
    if n <= 1:
        return labels, np.arange(n)
    pdist = distances[np.triu_indices(n, k=1)]
    assert not np.isnan(pdist).any()
    assert not np.isneginf(pdist).any()
    finite = np.isfinite(pdist)
    if not finite.any():
        return labels, np.arange(n)
    # tolearate some numerical zeros.
    pdist[np.logical_and(pdist > -eps, pdist < 0)] = 0.0

    if pdist.min() > threshold:
        if labels is None:
            return None, np.arange(n)
        else:
            ids = np.unique(labels)
            return labels, ids[ids >= 0]

    if not finite.all():
        inf = max(0, pdist[finite].max()) + threshold + 1.0
        pdist[np.logical_not(finite)] = inf

    Z = linkage(pdist, method=linkage_method)
    try:
        new_ids = fcluster(Z, threshold, criterion="distance")
    except ValueError as e:
        raise ValueError(
            f"fcluster failed with {threshold=} and smallest pdist {pdist.min()}."
        ) from e

    new_uniq = np.unique(new_ids)
    n_new = new_uniq.shape[0]
    assert np.array_equal(new_uniq, 1 + np.arange(n_new))
    n_old = new_ids.shape[0]
    n_merged = n_old - n_new
    merge_pct = 100 * n_merged / n_old
    logger.info(
        f"{linkage_method} link merged {n_merged} units "
        f"({n_old} -> {n_new}, {merge_pct:.1f}% reduction)."
    )

    # offset by 1
    new_ids -= 1

    if labels is None:
        new_labels = None
    else:
        kept = np.flatnonzero(labels >= 0)
        new_labels = np.full_like(labels, -1)
        new_labels[kept] = new_ids[labels[kept]]

    return new_labels, new_ids


def linkage_mask(
    distances: np.ndarray, linkage_method="complete", threshold=1.0
) -> np.ndarray:
    _, ids = hierarchical_cluster(
        labels=None,
        distances=distances,
        linkage_method=linkage_method,
        threshold=threshold,
    )
    mask = ids[:, None] == ids[None, :]
    assert mask.any(1).all()
    return mask


def sparsify_labels(labels: np.ndarray) -> dict[int, np.ndarray]:
    assert labels.ndim == 1
    ids = np.unique(labels)
    ids = ids[ids >= 0]
    inj = {}
    for j in ids:
        inj[j] = np.flatnonzero(labels == j)
    return inj


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


def maximal_leaf_groups(
    Z, distances: np.ndarray, max_distance=np.inf, max_group_size: int = 100
):
    """Get largest groups in linkage Z within some max complete dist and group size."""
    n = len(Z) + 1
    covered = set()
    leaves = leafsets(Z, max_distance=max_distance)
    leaves = {k: set(v) for k, v in leaves.items()}
    group_parents = []
    for i, row in reversed(list(enumerate(Z))):
        pa, pb, dist, nab = row
        if nab > max_group_size:
            continue
        if n + i not in leaves:
            # this is the distance check, since leafsets covers that.
            continue
        if leaves[n + i].issubset(covered):
            continue
        group_parents.append(n + i)
        covered.update(leaves[n + i])

    groups = [tuple(leaves[p]) for p in group_parents]

    # at this point, some nodes are not covered, but it's possible that they may
    # still be close enough to a group. this is a greedy algorithm to find all
    # of those nodes and add them to the best groups starting with the closest
    # matches first. we'll enforce complete linkage here (the most strict).
    if len(groups):
        too_big_penalty = np.array([len(g) >= max_group_size for g in groups])
        too_big_penalty = np.where(too_big_penalty, np.inf, 0.0)
        uncovered = [i for i in range(n) if i not in covered]
        group_distances = [
            np.array([distances[ui, g].max() for g in groups]) + too_big_penalty
            for ui in uncovered
        ]
        while len(uncovered):
            min_distances = [gd.min() for gd in group_distances]
            argmin_leaf = int(np.argmin(min_distances))
            if min_distances[argmin_leaf] > max_distance:
                break

            argmin_group = group_distances[argmin_leaf].argmin()

            # add leaf to group and coverage
            new_leaf = uncovered[argmin_leaf]
            groups[argmin_group] = (*groups[argmin_group], new_leaf)
            covered.add(new_leaf)

            # remove leaf from uncovered
            del uncovered[argmin_leaf]
            del group_distances[argmin_leaf]

            # update other leaves' dists to this group
            g = groups[argmin_group]
            for j, ui in enumerate(uncovered):
                if len(g) >= max_group_size:
                    group_distances[j][argmin_group] = np.inf
                else:
                    group_distances[j][argmin_group] = distances[ui, g].max()

    # now the true singletons are added
    for i in range(n):
        if i not in covered:
            groups.append((i,))

    assert max(map(len, groups)) <= max_group_size
    assert sum(map(len, groups)) == n
    assert set(gv for g in groups for gv in g) == set(range(n))

    groups = [tuple(sorted(g)) for g in groups]

    return groups


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


def reorder_by_depth(
    sorting: DARTsortSorting,
    motion: MotionInfo | None = None,
    spatial_footprints: np.ndarray | None = None,
    geom: np.ndarray | None = None,
    centroids: np.ndarray | None = None,
) -> tuple[DARTsortSorting, np.ndarray]:
    """Reorder cluster labels so that centroid depth is increasing

    Parameters
    ----------
    sorting : DARTsortSorting
    motion : MotionInfo | None, optional
    spatial_footprints : np.ndarray | None, optional
    geom : np.ndarray | None, optional
    centroids : np.ndarray | None, optional

    Returns
    -------
    reordered_sorting: DARTsortSorting
    reorder: np.ndarray
        reorder[j] is the new label of original unit j.
    """
    assert sorting.labels is not None
    kept = np.flatnonzero(sorting.labels >= 0)
    kept_labels = sorting.labels[kept]

    units, kept_labels = np.unique(kept_labels, return_inverse=True)

    if geom is None and motion is not None:
        geom = motion.rgeom

    if spatial_footprints is not None:
        assert centroids is None
        assert geom is not None
        assert spatial_footprints.shape[1] == geom.shape[0]
        assert spatial_footprints.shape[0] == units.shape[0]
        w = spatial_footprints / spatial_footprints.sum(1, keepdims=True)
        assert np.isfinite(w).all()
        centroids = w @ geom[:, 1]

    if centroids is None:
        depths = sorting.point_source_localizations[kept, 2]
        if motion is not None:
            depths = motion.correct_s(sorting.times_seconds[kept], depths)

        centroids = np.zeros(units.size)
        for u in range(units.size):
            inu = np.flatnonzero(kept_labels == u)
            centroids[u] = np.median(depths[inu])
    assert centroids.shape[0] == units.shape[0]

    labels = sorting.labels.copy()
    # this one is some food for thought, lol.
    reorder = np.argsort(np.argsort(centroids, kind="stable"), kind="stable")
    labels[kept] = reorder[kept_labels]
    reordered_sorting = sorting.ephemeral_replace(labels=labels)

    return reordered_sorting, reorder


def closest_registered_channels(
    *, times_seconds, x, z_abs, z_reg=None, motion: MotionInfo
) -> np.ndarray:
    """Assign spikes to the drift-extended channel closest to their registered position"""
    if z_reg is None:
        assert motion is not None
        z_reg = motion.correct_s(times_seconds, z_abs)
    reg_pos = np.c_[x, z_reg]

    _, reg_channels = motion.rgeom_kdt.query(reg_pos)
    reg_channels = np.atleast_1d(reg_channels)

    return reg_channels


def grid_snap(
    *,
    times_seconds,
    x,
    z_abs,
    grid_dx=15.0,
    grid_dz=15.0,
    z_reg=None,
    motion: MotionInfo,
) -> np.ndarray:
    if z_reg is None:
        z_reg = motion.correct_s(times_seconds, z_abs)
    reg_pos = np.c_[x, z_reg]

    # make a grid inside the registered geom bounding box
    min_x, max_x = motion.rgeom[:, 0].min(), motion.rgeom[:, 0].max()
    min_z, max_z = motion.rgeom[:, 1].min(), motion.rgeom[:, 1].max()
    grid_x = np.arange(min_x, max_x, grid_dx)
    grid_x += (min_x + max_x) / 2 - grid_x.mean()
    grid_z = np.arange(min_z, max_z, grid_dz)
    grid_z += (min_z + max_z) / 2 - grid_z.mean()
    grid_xx, grid_zz = np.meshgrid(grid_x, grid_z, indexing="ij")
    grid = np.c_[grid_xx.ravel(), grid_zz.ravel()]

    # snap to closest grid point
    registered_kdt = KDTree(grid)
    _, reg_channels = registered_kdt.query(reg_pos)
    reg_channels = np.atleast_1d(reg_channels)

    return reg_channels


def recursive_hdbscan_clustering(
    features,
    min_cluster_size=25,
    min_samples=25,
    cluster_selection_epsilon=1,
    recursive=True,
):
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        from sklearn.cluster import HDBSCAN

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
    from sklearn.neighbors import KNeighborsClassifier

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

    features = getattr(sorting, dataset_name, None)
    channel_index = getattr(sorting, "channel_index", None)
    if features is not None and channel_index is not None:
        features = features[which][:, :rank]
        return waveform_util.grab_main_channels(features, channels, channel_index)

    features = np.empty((mask.sum(), rank), dtype=np.float32)
    with h5py.File(sorting.parent_h5_path, "r", locking=False) as h5:
        feats_dset = h5[dataset_name]
        channel_index = cast(h5py.Dataset, h5["channel_index"])[:]
        for ixs, feats in data_util.yield_masked_chunks(
            mask, feats_dset, show_progress=show_progress, desc_prefix="Main channel"
        ):
            feats = feats[:, :rank]
            feats = waveform_util.grab_main_channels(
                feats, channels[ixs], channel_index
            )
            features[ixs] = feats
    return features


def decrumb(labels: np.ndarray, min_size: int=5, in_place=False, flatten=True):
    """Remove small units

    Parameters
    ----------
    labels : np.ndarray
    min_size : int
    in_place : bool
    flatten : bool
        Flatten the output label space to be contiguous.

    Returns
    -------
    labels
        The (flattened) decrumbed labels.
    """
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

    logger.dartsortdebug(f"decrumb ({min_size}): {k}->{k1}.")

    labels[kept] = units[labels_kept]

    return labels
