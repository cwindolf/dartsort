from logging import getLogger

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_array
import torch
import torch.nn.functional as F
from tqdm.auto import trange, tqdm


from .density import guess_mode
from ..util.multiprocessing_util import get_pool
from ..util.sparse_util import coo_to_torch


logger = getLogger(__name__)


def kmeanspp(
    X,
    n_components=10,
    random_state: np.random.Generator | int = 0,
    kmeanspp_initial="random",
    mode_dim=2,
    skip_assignment=False,
    min_distance=None,
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

    diff_buffer = X.clone()

    dists = torch.subtract(X, X[centroid_ixs[-1]], out=diff_buffer).square_().sum(1)
    assignments = None
    if not skip_assignment:
        assignments = torch.zeros((n,), dtype=torch.long, device=X.device)

    p = dists.clone()
    for j in range(1, n_components):
        p.copy_(dists)
        if min_distance:
            invalid = dists < min_distance
            if invalid.all():
                logger.dartsortdebug(f"kmeanspp: All close, stop at iteration {j}.")
                break
            p[invalid] = 0.0
        psum = p.sum()
        assert torch.isfinite(psum) and psum > 0
        p /= psum
        centroid_ixs.append(rg.choice(n, p=p.cpu().numpy()))
        newdists = torch.subtract(X, X[centroid_ixs[-1]], out=diff_buffer).square_()
        newdists = torch.sum(newdists, dim=1, out=p)
        if not skip_assignment:
            closer = newdists < dists
            assert assignments is not None
            assignments[closer] = j
            dists[closer] = newdists[closer]
        else:
            torch.minimum(dists, newdists, out=dists)

    centroid_ixs = torch.tensor(centroid_ixs)
    if not skip_assignment:
        centroid_ixs = centroid_ixs.to(assignments)
    phi = dists.mean()

    return centroid_ixs, assignments, dists, phi


def kdtree_kmeans(
    X,
    max_sigma=5.0,
    n_components=10,
    n_initializations=10,
    random_state: int | np.random.Generator = 0,
    n_iter=100,
    min_log_prop=-25.0,
    dirichlet_alpha=1.0,
    kmeanspp_min_dist=0.0,
    show_progress=False,
    sigma_atol=1e-3,
    batch_size=50_000,
    workers=-1,
    device=None,
):
    import torch.nn.functional as F

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    xrange = trange if show_progress else range
    n = len(X)
    batches = [slice(bs, min(n, bs + batch_size)) for bs in range(0, n, batch_size)]
    workers = min(len(batches), workers)
    X_torch = torch.asarray(X, device=device)

    n_jobs, Executor, context = get_pool(workers, cls="ThreadPoolExecutor")
    with Executor(n_jobs, context) as pool:
        random_state = np.random.default_rng(random_state)
        kmeanspp_rgs = random_state.spawn(n_initializations)
        kmpkw = dict(n_components=n_components, skip_assignment=True, min_distance=kmeanspp_min_dist)
        kmeanspp_jobs = (((X_torch,), kmpkw | dict(random_state=rg)) for rg in kmeanspp_rgs)

        best_phi = torch.inf
        centroid_ixs = None
        results = pool.map(_kmeanspp_job, kmeanspp_jobs)
        if show_progress:
            results = tqdm(results, total=n_initializations)
        for _centroid_ixs, _labels, _, phi in results:
            if phi < best_phi:
                centroid_ixs = _centroid_ixs
                best_phi = phi
        assert centroid_ixs is not None
        n_components = len(centroid_ixs)

        # state vars
        centroids = X[centroid_ixs]
        log_proportions = np.full(n_components, -np.log(n_components), dtype=X.dtype)
        new_centroids = np.zeros_like(centroids)
        sigmasq = float(best_phi)
        prev_sigma = np.sqrt(sigmasq)
        N = np.zeros(n_components)

        # batched k-d trees
        X_kdts = list(pool.map(KDTree, (X[sl] for sl in batches)))

        # kmeans iters
        for j in xrange(n_iter):
            new_centroids.fill(0.0)
            new_sigmasq = 0.0
            N.fill(0.0)
            n_so_far = 0

            C_kdt = KDTree(centroids)
            max_distance = max_sigma * np.sqrt(sigmasq) * np.sqrt(X.shape[1])

            fixed_args = C_kdt, max_distance, sigmasq, log_proportions, device, X.dtype
            jobs = (
                (X_kdt, X_torch[sl], *fixed_args) for X_kdt, sl in zip(X_kdts, batches)
            )
            for res in pool.map(_kdtree_kmeans_job, jobs):
                N1, batch_centroids, batch_sigmasq, wsum = res

                N += N1
                n1_n01 = N1 / N.clip(min=1e-5)

                batch_centroids -= new_centroids
                batch_centroids *= n1_n01[:, None]
                new_centroids += batch_centroids

                n_so_far += wsum
                batch_sigmasq -= new_sigmasq
                batch_sigmasq *= wsum / n_so_far
                new_sigmasq += batch_sigmasq

            centroids, new_centroids = new_centroids, centroids
            sigmasq = new_sigmasq
            log_proportions = F.log_softmax(
                torch.log(torch.tensor(N, dtype=torch.double) + dirichlet_alpha), dim=0
            ).numpy()
            log_proportions = np.maximum(log_proportions, min_log_prop)

            new_sigma = np.sqrt(sigmasq)
            if np.isclose(new_sigma, prev_sigma, atol=sigma_atol):
                break
            prev_sigma = new_sigma

    return dict(centroids=centroids, sigmasq=sigmasq, log_proportions=log_proportions)


def _kmeanspp_job(args_kwargs):
    a, k = args_kwargs
    return kmeanspp(*a, **k)


def _kdtree_kmeans_job(args):
    X_kdt, X_torch, C_kdt, max_distance, sigmasq, log_proportions, device, dtype = args

    dists = X_kdt.sparse_distance_matrix(
        C_kdt, max_distance=max_distance, output_type="ndarray"
    )
    dists = coo_array(
        (dists["v"].astype(dtype), (dists["i"], dists["j"])),
        shape=(X_kdt.n, C_kdt.n),
    )
    dtype = torch.double if dists.data.dtype == np.float64 else torch.float
    dists = coo_to_torch(dists, dtype).to(device)

    # get explicit sq dists, then set entries of dists to log likelihoods
    dists.values().square_()
    dsq = torch.asarray(dists.values(), dtype=torch.double, copy=True)
    log_proportions = torch.asarray(log_proportions).to(dsq)
    dists.values().mul_(-0.5 / sigmasq).add_(log_proportions[dists.indices()[1]])
    # --> dists is now sparse coo log likelihoods. dsq corresponds.

    # softmax over centroids (dim=1)
    dists = torch.sparse.softmax(dists, dim=1)

    # these are the responsibility weights used for sigma update below
    ww = torch.asarray(dists.values(), dtype=torch.double, copy=True)
    wsum = ww.sum()
    ww = ww.div_(wsum)

    # now divide by the sum in each unit
    N1 = dists.sum(dim=0).to_dense()
    dists.values().div_(N1[dists.indices()[1]])

    # take weighted mean of X
    batch_centroids = dists.T @ X_torch

    # take weighted mean of dsq
    batch_sigmasq = (dsq * ww).sum() / X_torch.shape[1]
    return (
        N1.numpy(force=True),
        batch_centroids.numpy(force=True),
        batch_sigmasq.numpy(force=True),
        wsum.numpy(force=True),
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
