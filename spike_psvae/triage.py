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


def run_weighted_triage(x, y, z, alpha, maxptps, pcs=None, 
                        scales=(1,10,1,15,30,10),
                        threshold=100, ptp_threshold=3, c=1):
    ptp_filter = np.flatnonzero(maxptps>ptp_threshold)
    x = x[ptp_filter]
    y = y[ptp_filter]
    z = z[ptp_filter]
    alpha = alpha[ptp_filter]
    maxptps = maxptps[ptp_filter]
    if pcs is not None:
        pcs = pcs[ptp_filter]
        feats = np.c_[scales[0]*x,
                      scales[1]*np.log(y),
                      scales[2]*z,
                      scales[3]*np.log(alpha),
                      scales[4]*np.log(maxptps),
                      scales[5]*pcs[:,:3]]
    else:
        feats = np.c_[scales[0]*x,
                      # scales[1]*np.log(y),
                      scales[2]*z,
                      # scales[3]*np.log(alpha),
                      scales[4]*np.log(maxptps)]
    
    tree = KDTree(feats)
    dist, ind = tree.query(feats, k=6)
    dist = dist[:,1:]
    dist = np.sum(c*np.log(dist) + np.log(1/(scales[4]*np.log(maxptps)))[:,None], 1)
    idx_keep = dist <= np.percentile(dist, threshold)
    
    triaged_x = x[idx_keep]
    triaged_y = y[idx_keep]
    triaged_z = z[idx_keep]
    triaged_alpha = alpha[idx_keep]
    triaged_maxptps = maxptps[idx_keep]
    triaged_pcs = None
    if pcs is not None:
        triaged_pcs = pcs[idx_keep]
        
    
    return triaged_x, triaged_y, triaged_z, triaged_alpha, triaged_maxptps, triaged_pcs, ptp_filter, idx_keep

def weighted_triage_ix(x, y, z, alpha, maxptps, pcs=None, 
                        scales=(1,10,1,15,30,10),
                        threshold=100, ptp_threshold=3, c=1):
    ptp_filter = maxptps > ptp_threshold
    x = x[ptp_filter]
    y = y[ptp_filter]
    z = z[ptp_filter]
    alpha = alpha[ptp_filter]
    maxptps = maxptps[ptp_filter]
    if pcs is not None:
        # pcs = pcs[ptp_filter]
        feats = np.c_[scales[0]*x,
                      scales[1]*np.log(y),
                      scales[2]*z,
                      scales[3]*np.log(alpha),
                      scales[4]*np.log(maxptps),
                      scales[5]*pcs[:,:3]]
    else:
        feats = np.c_[scales[0]*x,
                      # scales[1]*np.log(y),
                      scales[2]*z,
                      # scales[3]*np.log(alpha),
                      scales[4]*np.log(maxptps)]
    
    tree = KDTree(feats)
    dist, ind = tree.query(feats, k=6)
    print(dist.shape)
    dist = dist[:,1:]
    print(np.isfinite(np.log(dist)).all(axis=1).mean())
    dist = np.sum(c*np.log(dist) + np.log(1/(scales[4]*np.log(maxptps)))[:,None], 1)
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
