import numpy as np
from scipy.spatial import KDTree
import networkx as nx


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
