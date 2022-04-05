import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import nnls
import hdbscan


def pairdists(waveforms, log=False, square=True):
    pd = pdist(waveforms.reshape(waveforms.shape[0], -1))
    if log:
        pd = np.log(pd)

    if square:
        return squareform(pd)
    else:
        return pd


def dim_scales_lsq(waveforms, features):
    n, t, c = waveforms.shape
    n_, k = features.shape
    assert n == n_
    orig_pd = pdist(
        waveforms.reshape(waveforms.shape[0], -1), metric="sqeuclidean"
    )
    f_pd = np.array(
        [pdist(f[:, None], metric="sqeuclidean") for f in features.T]
    )
    # return np.sqrt(np.abs(la.inv(f_pd @ f_pd.T) @ (f_pd @ orig_pd)))

    x, rnorm = nnls(f_pd.T, orig_pd)
    return np.sqrt(x)


def cluster_hdbscan(features, min_cluster_size=25, min_samples=25, z_ind=1):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusterer.fit(features)
    print(clusterer)

    cluster_centers = []
    for label in np.unique(clusterer.labels_):
        if label != -1:
            cluster_centers.append(clusterer.weighted_cluster_centroid(label))
    cluster_centers = np.asarray(cluster_centers)
    
    # re-label each cluster by z-depth
    labels_depth = np.argsort(-cluster_centers[:, z_ind])
    
    
    label_to_id = {}
    for i, label in enumerate(labels_depth):
        label_to_id[label] = i
    label_to_id[-1] = -1
    new_labels = np.vectorize(label_to_id.get)(clusterer.labels_) 
    clusterer.labels_ = new_labels
    
    return clusterer