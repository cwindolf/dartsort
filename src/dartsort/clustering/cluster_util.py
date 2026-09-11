from typing import Literal, cast

import h5py
import numba
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import KDTree

from ..util import waveform_util
from ..util.data_util import (
    DARTsortSorting,
    apply_label_remapping_in_place,
    count_not_sorted,
    mean_by_label_1d,
    pos_int_unique_and_counts,
    yield_masked_chunks,
)
from ..util.logging_util import get_logger
from ..util.motion import MotionInfo
from ..util.py_util import databag, panic

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
    in_place: bool = False,
) -> DARTsortSorting:
    assert sorting.labels is not None

    # shifts are indexed by the pre-merge labels. apply them before remapping.
    if shifts is None:
        times_updated = None
    else:
        assert unit_snrs is not None
        times_samples = sorting.times_samples
        times_updated = times_samples if in_place else times_samples.copy()
        apply_time_shifts(
            times_updated,
            sorting.labels,
            merge_group_shifts(merge_mapping, shifts, unit_snrs),
        )

    if new_labels is None:
        new_labels = sorting.labels if in_place else sorting.labels.copy()
        apply_label_remapping_in_place(new_labels, merge_mapping)

    if times_updated is None:
        return sorting.ephemeral_replace(labels=new_labels)
    return sorting.ephemeral_replace(times_samples=times_updated, labels=new_labels)


def merge_group_shifts(
    merge_mapping: np.ndarray, shifts: np.ndarray, unit_snrs: np.ndarray
) -> np.ndarray:
    """Each unit's shift rel to the highest SNR unit in its group

    shifts[i, j] is like trough[i] - trough[j]. Subtracting the result from
    a unit's times aligns it to the group's best unit. And the return would
    be unit_shifts[j] = best one - trough[j]

    Use me with apply_time_shifts. We have the same sign convention.
    """
    assert np.abs(np.diagonal(shifts)).max() == 0

    # find each unit's group-best-unit
    best_units = np.full(merge_mapping.max(initial=-1) + 1, -1)
    for unit, group in enumerate(merge_mapping):
        best = best_units[group]
        if best < 0 or unit_snrs[unit] > unit_snrs[best]:
            best_units[group] = unit

    # pick out the shift entry
    arange = np.arange(merge_mapping.shape[0])
    target = best_units[merge_mapping]
    unit_shifts = shifts[target, arange]
    assert np.all(unit_shifts[target == arange] == 0)

    return unit_shifts


@numba.njit(nogil=True, parallel=True)
def apply_time_shifts(times: np.ndarray, labels: np.ndarray, unit_shifts: np.ndarray):
    for i in numba.prange(times.shape[0]):  # ty: ignore[not-iterable]
        label = labels[i]
        if label >= 0:
            times[i] -= unit_shifts[label]


def hierarchical_cluster(
    labels: np.ndarray | None,
    distances: np.ndarray,
    linkage_method="complete",
    threshold=1.0,
    eps=1e-5,
):
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
            ids, _, _ = pos_int_unique_and_counts(labels)
            return labels, ids

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


def sparsify_labels(
    labels: np.ndarray, ids: np.ndarray | None = None
) -> dict[int, np.ndarray]:
    assert labels.ndim == 1
    if ids is None:
        ids, _, _ = pos_int_unique_and_counts(labels)
    inj = {}
    for j in ids:
        inj[j] = np.flatnonzero(labels == j)
    return inj


def leafsets(Z, max_distance=np.inf):
    """For a linkage Z, get the leaves in each non-leaf cluster."""
    n = len(Z) + 1
    leaves = {}
    for i, row in enumerate(Z):
        pa, pb, dist, _nab = row
        if dist > max_distance:
            break
        leavesa = leaves.get(pa, [int(pa)])
        leavesb = leaves.get(pb, [int(pb)])
        leaves[n + i] = leavesa + leavesb
        leaves[n + i].sort()
    return leaves


def meet(id_mapping_a: np.ndarray, id_mapping_b: np.ndarray) -> np.ndarray:
    """The meet opration on the lattice of partitions."""
    # this doesn't support destroying ids
    assert id_mapping_a.min() >= 0
    assert id_mapping_b.min() >= 0

    _, meet_mapping = np.unique(
        np.column_stack((id_mapping_a, id_mapping_b)), axis=0, return_inverse=True
    )
    return meet_mapping.ravel()


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
        _pa, _pb, _dist, nab = row
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
    assert {gv for g in groups for gv in g} == set(range(n))

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
    for i, (pa, pb, _dist, _nab) in enumerate(Z):
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
    for dist, thresh in zip(distances[1:], thresholds[1:], strict=True):
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
    in_place: bool = False,
) -> tuple[DARTsortSorting, np.ndarray]:
    """Reorder cluster labels so that centroid depth is increasing

    Also deals with soft assign candidates, if present.

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
    sorting = sorting.flatten(include_gmm_properties=True, in_place=in_place)
    assert sorting.labels is not None

    if geom is None and motion is not None:
        geom = motion.rgeom

    if spatial_footprints is not None:
        assert centroids is None
        assert geom is not None
        assert spatial_footprints.shape[1] == geom.shape[0]
        w = spatial_footprints / spatial_footprints.sum(1, keepdims=True)
        assert np.isfinite(w).all()
        centroids = w @ geom[:, 1]

    if centroids is None:
        z = sorting.load_dataset("point_source_localizations", sl=(slice(None), 2))
        if motion is not None:
            z = motion.correct_s(sorting.times_seconds, z)
        centroids = mean_by_label_1d(sorting, z)

    labels = sorting.labels if in_place else sorting.labels.copy()
    reorder = np.argsort(np.argsort(centroids, kind="stable"), kind="stable")
    apply_label_remapping_in_place(labels, reorder)

    new_props: dict[str, np.ndarray] = dict(labels=labels)
    new_props.update(
        sorting.remap_gmm_properties(reorder, new_K=reorder.size, in_place=in_place)
    )
    reordered_sorting = sorting.ephemeral_replace(**new_props)

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
        for ixs, feats in yield_masked_chunks(
            mask, feats_dset, show_progress=show_progress, desc_prefix="Main channel"
        ):
            feats = feats[:, :rank]
            feats = waveform_util.grab_main_channels(
                feats, channels[ixs], channel_index
            )
            features[ixs] = feats
    return features


def decrumb_labels(labels: np.ndarray, min_size: int = 5, in_place=False, flatten=True):
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
    units, counts, _ = pos_int_unique_and_counts(labels)
    if not units.size:
        return labels
    all_big = counts.min() >= min_size
    flat_ok = (not flatten) or np.array_equal(units, np.arange(len(units)))
    if all_big and flat_ok:
        return labels
    remapping = np.full((units.max() + 1,), -1, dtype=labels.dtype)
    kept_units = units[counts >= min_size]
    if flatten:
        remapping[kept_units] = np.arange(len(kept_units))
    else:
        remapping[kept_units] = kept_units
    new_labels = labels if in_place else labels.copy()
    apply_label_remapping_in_place(new_labels, remapping)
    logger.dartsortdebug(f"decrumb ({min_size}): {units.size}->{kept_units.size}.")
    return new_labels


def decrumb(
    sorting: DARTsortSorting, min_size: int = 5, in_place=False, flatten=True
) -> DARTsortSorting:
    assert sorting.labels is not None
    units, counts, _ = pos_int_unique_and_counts(sorting.labels)
    if not units.size:
        return sorting
    all_big = counts.min() >= min_size
    flat_ok = (not flatten) or np.array_equal(units, np.arange(len(units)))
    if all_big and flat_ok:
        return sorting

    remapping = np.full((units.max() + 1,), -1, dtype=sorting.labels.dtype)
    kept_units = units[counts >= min_size]
    remapping[kept_units] = kept_units
    new_labels = sorting.labels if in_place else sorting.labels.copy()
    apply_label_remapping_in_place(new_labels, remapping)
    sorting = sorting.ephemeral_replace(labels=new_labels)
    if flatten:
        sorting = sorting.flatten(in_place=in_place)
    return sorting


@databag
class ViolationInfo:
    unit_ids: np.ndarray
    spike_counts: np.ndarray
    """Same shape as unit_ids (flat)"""
    viol_counts: np.ndarray
    """Observed violated spike pair count. Indexed by pair of ids (not flat)."""
    jitter_counts: np.ndarray | None
    """Expected counts under a jitter resampling null model. (Also not flat.)"""

    n_resamples: int
    """Sample count if Monte Carlo was used, else 0."""
    jitter_ms: float
    censor_ms: float
    viol_ms: float

    def jitter_viol_ratio(
        self, min_jitter: float = 4.6, fill_value=np.nan
    ) -> np.ndarray:
        """observed / expected, but masked out with fill_value where expected is small."""
        assert self.jitter_counts is not None
        ratio = np.full(self.viol_counts.shape, fill_value, dtype=np.float64)
        np.divide(
            self.viol_counts,
            self.jitter_counts,
            out=ratio,
            where=self.jitter_counts >= min_jitter,
        )
        return ratio


def violation_statistics(
    st: DARTsortSorting,
    *,
    censor_ms: float = 0.25,
    viol_ms: float = 1.0,
    jitter_ms: float = 20.0,
    chance_method: Literal["weighted", "resample", "none"] = "weighted",
    n_resamples: int = 32,
    rg: int | np.random.Generator = 0,
) -> ViolationInfo:
    """Observed ACG/CCG violations together with a jittered chance level

    Count ACG and CCG violations within viol_ms.

    Times within censor_ms of each other are ignored in the violation
    count. The censorship is right-exclusive, so that if censor_ms is 0,
    exact duplicates are counted; if censor_ms corresponds to 10 samples,
    9-sample viols are excluded and 10-sample viols are counted.

    This also supports jitter-based null models. The default (weighted)
    gets the expected violation count under a null model without drawing
    jitter samples. "none" is well hey go figure. "resample" uses Monte
    Carlo and is just there for unit tests pretty much.
    """
    assert st.labels is not None
    rg = np.random.default_rng(rg)
    censor_samples = max(0, int(censor_ms * (st.sampling_frequency / 1000.0)))
    viol_samples = int(viol_ms * (st.sampling_frequency / 1000.0))
    jitter_samples = int(jitter_ms * (st.sampling_frequency / 1000.0))
    if chance_method != "none":
        # well you should probably do a good amount more but it's a start
        assert jitter_samples > viol_samples

    unit_ids, spike_counts, _ = pos_int_unique_and_counts(st.labels)
    nu = (unit_ids.max() + 1).item() if unit_ids.size else 0
    if not nu or viol_samples < censor_samples:
        blank = np.zeros((nu, nu))
        return ViolationInfo(
            unit_ids=unit_ids,
            spike_counts=spike_counts,
            viol_counts=blank.astype(np.int64),
            jitter_counts=None if chance_method == "none" else blank,
            n_resamples=n_resamples if chance_method == "resample" else 0,
            jitter_ms=jitter_ms,
            censor_ms=censor_ms,
            viol_ms=viol_ms,
        )

    labels = st.labels
    times = st.times_samples
    if count_not_sorted(times) > 0:
        tsort = np.argsort(times, kind="stable")
        labels = labels[tsort]
        times = times[tsort]

    viol_counts, buffer = count_violations(
        times, labels, censor_samples, viol_samples, nu
    )

    if chance_method == "none":
        jitter_counts = None
        n_resamples = 0
    elif chance_method == "weighted":
        jitter_counts, _ = count_violations(
            times,
            labels,
            censor_samples,
            viol_samples,
            nu,
            jitter_samples=jitter_samples,
            buffer=buffer.view(np.float64),
        )
        n_resamples = 0
    elif chance_method == "resample":
        jittered_times = np.empty_like(times)
        jittered_labels = np.empty_like(labels)
        jitter_counts = np.zeros((nu, nu))
        for _ in range(n_resamples):
            offsets = rg.integers(-jitter_samples, jitter_samples + 1, size=times.size)
            np.add(times, offsets, out=jittered_times)
            order = np.argsort(jittered_times, kind="stable")
            np.take(labels, order, out=jittered_labels)
            jittered_times.sort()
            sample, buffer = count_violations(
                jittered_times,
                jittered_labels,
                censor_samples,
                viol_samples,
                nu,
                buffer=buffer,
            )
            jitter_counts += sample
        jitter_counts /= n_resamples
    else:
        panic(chance_method)

    return ViolationInfo(
        unit_ids=unit_ids,
        spike_counts=spike_counts,
        viol_counts=viol_counts,
        jitter_counts=jitter_counts,
        n_resamples=n_resamples,
        jitter_ms=jitter_ms,
        censor_ms=censor_ms,
        viol_ms=viol_ms,
    )


def count_violations(
    times: np.ndarray,
    labels: np.ndarray,
    censor_samples: int,
    viol_samples: int,
    n_units: int,
    jitter_samples: int | None = None,
    buffer=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a matrix counting spike pair violations

    Helper function for violation_statistics

    Each violating pair is counted exactly once!

    weights are used if jitter_samples is supplied. It is maybe a bit
    confusing to do it that way, because jitter is not simulated. It's
    a closed form for the expectation.
    """
    n = times.shape[0]
    nchunks = max(1, min(numba.get_num_threads(), max(n, 1)))
    dtype = np.float64 if jitter_samples is not None else np.int64
    if buffer is None:
        buffer = np.zeros((nchunks, n_units, n_units), dtype=dtype)
    else:
        assert buffer.shape == (nchunks, n_units, n_units)
        assert buffer.dtype == dtype
        buffer.fill(0)
    chunk_starts = (np.arange(buffer.shape[0] + 1) * times.size) // buffer.shape[0]
    if jitter_samples is None:
        _violation_count_matrix(
            times,
            labels,
            censor_samples,
            viol_samples,
            chunk_starts,
            buffer,
        )
    else:
        weights = _jitter_spread_violation_weights(censor_samples, viol_samples, jitter_samples)
        _violation_weight_matrix(
            times,
            labels,
            weights,
            chunk_starts,
            buffer,
        )

    # sum chunks, symmetrize, handle diag
    totals = buffer.sum(axis=0)
    diag = np.diagonal(totals).copy()
    totals += totals.T
    np.fill_diagonal(totals, diag)
    return totals, buffer


def _jitter_spread_violation_weights(
    censor_samples: int, viol_samples: int, jitter_samples: int
) -> np.ndarray:
    r"""Helper for getting the jitter-expected counts

    The violation region gets spread out according to the joint distribution
    of two Uniform{-J,...,J} variables. The jitter shift that results
    has a triangular law (jitter_shifts and its probs below). So the final
    answer is the convolution of the violation region with the triangle.

    That expression is the expectation of violation over jitter outcomes.
    Returned length goes out to 2 * jitter + viol, which is the support of
    that expectation as f(observed dt). We return the right half including
    0 lag.
    """
    assert 0 <= censor_samples <= viol_samples < jitter_samples

    m = 2 * jitter_samples + 1
    jitter_shifts = np.arange(-2 * jitter_samples, 2 * jitter_samples + 1)
    jitter_shift_probs = (m - np.abs(jitter_shifts)) / (m * m)

    offsets = np.arange(-viol_samples, viol_samples + 1)
    violates = (np.abs(offsets) >= censor_samples).astype(float)

    center = 2 * jitter_samples + viol_samples
    conv = np.convolve(jitter_shift_probs, violates)
    assert conv.shape == (2 * center + 1,)
    # want the piece from zero lag on because we index with positive lags
    return conv[center:]


@numba.njit(nogil=True, parallel=True)
def _violation_count_matrix(
    times: np.ndarray,
    labels: np.ndarray,
    censor_samples: int,
    viol_samples: int,
    starts: np.ndarray,
    counts: np.ndarray,
):
    """numba loop: count violations: dts in [censor_samples, viol_samples)"""
    n = times.shape[0]

    # parallelize over chunks
    for c in numba.prange(starts.shape[0] - 1):  # ty: ignore[not-iterable]
        out = counts[c]  # my thread's output buffer

        for i in range(starts[c], starts[c + 1]):
            li = labels[i]
            if li < 0:
                continue

            ti = times[i]
            first = ti + censor_samples
            last = ti + viol_samples

            # make sure to read js past the chunk end!
            for j in range(i + 1, n):
                if times[j] < first:
                    continue
                if times[j] > last:  # be inclusive here i suppose
                    break
                lj = labels[j]
                if lj < 0:
                    continue
                out[li, lj] += 1


@numba.njit(nogil=True, parallel=True)
def _violation_weight_matrix(
    times: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    starts: np.ndarray,
    sums: np.ndarray,
):
    """numba loop: for pairs of spikes, sum weights[dt]"""
    n = times.shape[0]
    window_length = weights.shape[0] - 1

    # parallelize over chunks
    for c in numba.prange(starts.shape[0] - 1):  # ty: ignore[not-iterable]
        out = sums[c]  # my thread's output buffer

        for i in range(starts[c], starts[c + 1]):
            li = labels[i]
            if li < 0:
                continue
            ti = times[i]

            # make sure to read js past the chunk end!
            for j in range(i + 1, n):
                dt = times[j] - ti
                if dt > window_length:
                    break
                lj = labels[j]
                if lj < 0:
                    continue
                out[li, lj] += weights[dt]
