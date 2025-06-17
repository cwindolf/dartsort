import numpy as np
import torch
import torch.nn.functional as F
from .density import guess_mode


def kmeanspp(
    X,
    n_components=10,
    random_state: np.random.Generator | int = 0,
    kmeanspp_initial="random",
    mode_dim=2,
    skip_assignment=False,
):
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
    assignments = None
    if not skip_assignment:
        assignments = torch.zeros((n,), dtype=torch.long, device=X.device)

    for j in range(1, n_components):
        p = (dists / dists.sum()).numpy(force=True)
        centroid_ixs.append(rg.choice(n, p=p))
        newdists = (X - X[centroid_ixs[-1]]).square_().sum(1)
        closer = newdists < dists
        if not skip_assignment:
            assignments[closer] = j
        dists[closer] = newdists[closer]

    centroid_ixs = torch.tensor(centroid_ixs)
    if not skip_assignment:
        centroid_ixs = centroid_ixs.to(assignments)
    phi = dists.mean()

    return centroid_ixs, assignments, dists, phi


def kdtree_kmeans(
    X,
    max_distance,
    n_components=10,
    n_initializations=10,
    random_state=0,
    n_iter=100,
    batch_size=512,
    min_log_prop=-25.0,
    dirichlet_alpha=1.0
):
    from scipy.spatial import KDTree
    from scipy.sparse import coo_array
    import torch.nn.functional as F

    best_phi = torch.inf
    centroid_ixs = None
    random_state = np.random.default_rng(random_state)
    for j in range(n_initializations):
        print("kmeanspp", j)
        _centroid_ixs, _labels, _, phi = kmeanspp(
            torch.from_numpy(X),
            n_components=n_components,
            random_state=random_state,
            skip_assignment=True,
        )
        if phi < best_phi:
            centroid_ixs = _centroid_ixs
            best_phi = phi
        print(phi)
    assert centroid_ixs is not None

    # initialize
    X = X.astype("float32")
    X_kdt = KDTree(X)
    centroids = X[centroid_ixs]
    log_proportions = np.full(n_components, -np.log(n_components))
    lik_buf = np.empty((batch_size, n_components), dtype="float32")
    sigmasq = float(best_phi)
    print(f"{best_phi=} {sigmasq=}")
    Xbuf = None

    # infer gmm with isotropic covs
    for j in range(n_iter):
        print("kdtkm", j)
        C_kdt = KDTree(centroids)
        dists = X_kdt.sparse_distance_matrix(
            C_kdt, max_distance=max_distance, output_type="ndarray"
        )
        # this is nxk csc
        dists = coo_array(
            (dists["v"], (dists["j"], dists["i"])), shape=(C_kdt.n, X_kdt.n)
        ).tocsc()

        new_centroids = np.zeros_like(centroids)
        new_sigmasq = 0.0
        batch_centroids = np.zeros_like(centroids)
        N = np.zeros(n_components)
        n_so_far = 0
        print('a')

        for bs in range(0, X_kdt.n, batch_size):
            be = min(X_kdt.n, bs + batch_size)

            ll = lik_buf[: be - bs]
            ll.fill(-torch.inf)
            dsl = dists[:, bs:be]
            dsl = dsl.tocoo()
            cc, ii = dsl.coords
            liks = dsl.data.ravel()
            liks *= liks
            dsq = liks.copy()
            liks *= -0.5 / sigmasq
            liks += log_proportions[cc]
            ll[ii, cc] = liks

            ll = torch.from_numpy(ll)
            ll = F.softmax(ll, dim=1)
            ll = ll.nan_to_num_(nan=0.0)
            ll = ll.numpy()
            N1 = ll.sum(0)
            if Xbuf is None or len(ii) > len(Xbuf):
                Xbuf = X[ii + bs]
                Xbatch = Xbuf
            else:
                Xbatch = np.take(X, ii + bs, axis=0, out=Xbuf[: len(ii)])
            llnorm = ll / N1.clip(min=1e-5)
            weights = llnorm[ii, cc]
            Xbatch *= weights[:, None]
            np.add.at(batch_centroids, (cc[:, None], np.arange(X.shape[1])[None]), Xbatch)

            N += N1
            n_so_far += be - bs
            n1_n01 = N1 / N.clip(min=1e-5)

            batch_centroids -= new_centroids
            batch_centroids *= n1_n01[:, None]
            new_centroids += batch_centroids

            weights = ll[ii, cc]
            weights /= weights.sum() * X.shape[1]
            batch_sigmasq = np.sum(weights * dsq)
            batch_sigmasq -= new_sigmasq
            batch_sigmasq *= (be - bs) / n_so_far
            new_sigmasq += batch_sigmasq

        centroids = new_centroids
        sigmasq = new_sigmasq
        print(f"{sigmasq=}")
        log_proportions = F.log_softmax(
            torch.log(torch.tensor(N, dtype=torch.double) + dirichlet_alpha), dim=0).numpy()
        log_proportions = np.maximum(log_proportions, min_log_prop)

    return dict(
        centroids=centroids,
        sigmasq=sigmasq,
        log_proportions=log_proportions,
    )


def kmeans_inner(
    X,
    n_kmeanspp_tries=5,
    n_iter=100,
    n_components=10,
    random_state: np.random.Generator | int = 0,
    kmeanspp_initial="random",
    with_proportions=False,
    drop_prop=0.025,
    drop_sum=5.0,
):
    """A bit more than K-means

    Supports a proportion vector, as well as automatically dropping tiny clusters.
    """
    best_phi = torch.inf
    centroid_ixs = labels = None
    random_state = np.random.default_rng(random_state)
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
            best_phi = phi
    assert labels is not None

    centroids = X[centroid_ixs]
    dists = torch.cdist(X, centroids).square_()
    # responsibilities, sum to 1 over centroids
    e = F.softmax(-0.5 * dists, dim=1)
    if not n_iter:
        return labels, e, centroids, dists

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
            return torch.full_like(labels, 0), None, None, None

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
    return assignments, e, centroids, dists


def kmeans(
    X,
    n_kmeans_tries=5,
    n_kmeanspp_tries=5,
    n_iter=100,
    n_components=10,
    random_state: np.random.Generator | int = 0,
    kmeanspp_initial="random",
    with_proportions=False,
    drop_prop=0.025,
    drop_sum=5.0,
):
    best_phi = np.inf
    random_state = np.random.default_rng(random_state)
    assignments = torch.zeros(len(X), dtype=torch.long)
    e = centroids = dists = None
    for j in range(n_kmeans_tries):
        aa, ee, cc, dists = kmeans_inner(
            X,
            n_kmeanspp_tries=n_kmeanspp_tries,
            n_iter=n_iter,
            n_components=n_components,
            random_state=random_state,
            kmeanspp_initial=kmeanspp_initial,
            with_proportions=with_proportions,
            drop_prop=drop_prop,
            drop_sum=drop_sum,
        )
        if dists is None:
            continue
        assert ee is not None
        phi = (ee * dists).sum(1).mean().numpy(force=True)
        if phi < best_phi:
            best_phi = phi
            assignments = aa
            e = ee
            centroids = cc
    return dict(
        labels=assignments, responsibilities=e, centroids=centroids, dists=dists
    )
