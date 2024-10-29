import numpy as np
import torch
import torch.nn.functional as F


def kmeanspp(X, n_components=10, random_state=0, kmeanspp_initial="mean"):
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

    return centroid_ixs, assignments, dists


def kmeans(
    X,
    n_iter=100,
    n_components=10,
    random_state=0,
    kmeanspp_initial="mean",
    with_proportions=False,
    drop_prop=0.025,
):
    """A bit more than K-means

    Supports a proportion vector, as well as automatically dropping tiny clusters.
    """
    centroid_ixs, labels, dists = kmeanspp(
        X,
        n_components=n_components,
        random_state=random_state,
        kmeanspp_initial=kmeanspp_initial,
    )
    # responsibilities, sum to 1 over centroids
    if not n_iter:
        return labels, e

    centroids = X[centroid_ixs]
    dists = torch.cdist(X, centroids).square_()
    e = F.softmax(-0.5 * dists, dim=1)
    proportions = e.mean(0)

    for j in range(n_iter):
        if drop_prop:
            keep = proportions > drop_prop
            e = e[:, keep]
            proportions = proportions[keep]

        # update centroids
        w = e / e.sum(0)
        centroids = w.T @ X

        # e step
        dists = torch.cdist(X, centroids).square_()
        if with_proportions:
            e = F.softmax(-0.5 * dists + proportions.log(), dim=1)
        else:
            e = F.softmax(-0.5 * dists, dim=1)

    assignments = torch.argmin(dists, 1)
    return assignments, e
