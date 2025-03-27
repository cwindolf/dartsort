import numpy as np
import torch
import torch.nn.functional as F
from .density import guess_mode


def kmeanspp(X, n_components=10, random_state=0, kmeanspp_initial="mean", mode_dim=2):
    """K-means++ initialization

    Start at a random point (kmeanspp_initial=='random') or at the point
    farthest from the mean (kmeanspp_initial=='mean').
    """
    n, p = X.shape

    rg = np.random.default_rng(random_state)
    if kmeanspp_initial == "random":
        centroid_ixs = [rg.integers(n)]
    elif kmeanspp_initial == "mean":
        closest = torch.cdist(X, X.mean(0, keepdim=True)).argmax()
        centroid_ixs = [closest.item()]
    elif kmeanspp_initial == "mode":
        Xm = X
        if Xm.shape[1] > mode_dim:
            q = min(mode_dim + 10, *Xm.shape)
            u, s, v = torch.pca_lowrank(Xm, q=q, niter=7)
            Xm = u[:, :mode_dim].mul_(s[:mode_dim])
        centroid_ixs = [guess_mode(Xm.numpy(force=True))]
    else:
        assert False

    dists = (X - X[centroid_ixs[-1]]).square_().sum(1)
    assignments = torch.zeros((n,), dtype=int, device=X.device)

    for j in range(1, n_components):
        p = (dists / dists.sum()).numpy(force=True)
        centroid_ixs.append(rg.choice(n, p=p))
        newdists = (X - X[centroid_ixs[-1]]).square_().sum(1)
        closer = newdists < dists
        assignments[closer] = j
        dists[closer] = newdists[closer]

    centroid_ixs = torch.tensor(centroid_ixs).to(assignments)
    phi = dists.sum()

    return centroid_ixs, assignments, dists, phi


def kmeans(
    X,
    n_kmeanspp_tries=5,
    n_iter=100,
    n_components=10,
    random_state=0,
    kmeanspp_initial="mean",
    with_proportions=False,
    drop_prop=0.025,
    drop_sum=5.0,
    return_centroids=False,
):
    """A bit more than K-means

    Supports a proportion vector, as well as automatically dropping tiny clusters.
    """
    best_phi = torch.inf
    centroid_ixs = labels = None
    for _ in range(n_kmeanspp_tries):
        _centroid_ixs, _labels, _, phi = kmeanspp(
            X,
            n_components=n_components,
            random_state=random_state,
            kmeanspp_initial=kmeanspp_initial,
        )
        if phi < best_phi:
            centroid_ixs = _centroid_ixs
            labels = _labels

    centroids = X[centroid_ixs]
    dists = torch.cdist(X, centroids).square_()
    # responsibilities, sum to 1 over centroids
    e = F.softmax(-0.5 * dists, dim=1)
    if not n_iter:
        if return_centroids:
            return labels, e, centroids
        return labels, e

    proportions = e.mean(0)

    for j in range(n_iter):
        keep = None
        if drop_prop:
            keep = proportions > drop_prop
            e = e[:, keep]
            proportions = proportions[keep]
        if drop_sum:
            totals = e.sum(0)
            keep = totals > drop_sum
            e = e[:, keep]
            proportions = proportions[keep]
        if keep is not None and (not keep.numel() or not keep.any()):
            return torch.full_like(labels, 0), None

        # update centroids
        w = e / e.sum(0)
        centroids = w.T @ X

        # e step
        dists = torch.cdist(X, centroids).square_()
        if with_proportions:
            e = F.softmax(-0.5 * dists + proportions.log(), dim=1)
        else:
            e = F.softmax(-0.5 * dists, dim=1)
        proportions = e.mean(0)

    assignments = torch.argmin(dists, 1)
    if return_centroids:
        return assignments, e, centroids
    return assignments, e
