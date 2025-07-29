from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import trange

try:
    import cupy

    del cupy

    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False


from .density import guess_mode
from ..util.sparse_util import (
    coo_to_scipy,
    coo_to_cupy,
    distsq_to_lik_coo,
    logsumexp_coo,
    sparse_centroid_distsq,
)
from ..util.spiketorch import spawn_torch_rg
from ..util.logging_util import DARTSORTDEBUG


logger = getLogger(__name__)


def kmeanspp(
    X,
    n_components=10,
    random_state: np.random.Generator | torch.Generator | int = 0,
    kmeanspp_initial="random",
    mode_dim=2,
    skip_assignment=False,
    min_distance=None,
    show_progress=False,
    initial_distances=None,
):
    """K-means++ initialization

    Start at a random point (kmeanspp_initial=='random') or at the point
    farthest from the mean (kmeanspp_initial=='mean').
    """
    X = torch.asarray(X)
    n, p = X.shape
    n_components = min(n, n_components)

    if isinstance(random_state, torch.Generator):
        gen = random_state
    else:
        rg = np.random.default_rng(random_state)
        gen = spawn_torch_rg(rg, device=X.device)

    has_initial_dists = initial_distances is not None

    centroid_ixs = torch.full((n_components,), n, dtype=torch.long, device=X.device)
    dists = None
    if has_initial_dists:
        idists = torch.asarray(initial_distances, dtype=X.dtype, device=X.device)

    if kmeanspp_initial == "random":
        if dists is None:
            centroid_ixs[0] = torch.randint(n, size=(), device=X.device, generator=gen)
        else:
            centroid_ixs[0] = torch.multinomial(idists, 1, generator=gen)
    elif kmeanspp_initial == "mean":
        closest = torch.cdist(X, X.mean(0, keepdim=True)).argmax()
        centroid_ixs[0] = closest.item()
    elif kmeanspp_initial == "mode":
        Xm = X
        if Xm.shape[1] > mode_dim:
            q = min(mode_dim + 10, *Xm.shape)
            u, s, v = torch.pca_lowrank(Xm, q=q, niter=7)
            Xm = u[:, :mode_dim].mul_(s[:mode_dim])
        centroid_ixs = guess_mode(Xm.numpy(force=True))
    else:
        assert False

    diff_buffer = X.clone()

    dists = torch.subtract(X, X[centroid_ixs[0]], out=diff_buffer).square_().sum(1)
    assignments = None
    if not skip_assignment:
        assignments = torch.zeros((n,), dtype=torch.long, device=X.device)

    p = dists.clone()
    xrange = trange if show_progress else range
    for j in xrange(1, n_components):
        if has_initial_dists:
            torch.minimum(dists, idists, out=p)
        else:
            p.copy_(dists)
        if min_distance:
            invalid = dists < min_distance
            if invalid.all():
                logger.dartsortdebug(f"kmeanspp: All close, stop at iteration {j}.")
                break
            p[invalid] = 0.0
        centroid_ixs[j] = torch.multinomial(p, 1, generator=gen)

        newdists = torch.subtract(X, X[centroid_ixs[j]], out=diff_buffer).square_()
        newdists = torch.sum(newdists, dim=1, out=p)
        if not skip_assignment:
            closer = newdists < dists
            assert assignments is not None
            assignments[closer] = j
            dists[closer] = newdists[closer]
        else:
            torch.minimum(dists, newdists, out=dists)
    else:
        j += 1

    centroid_ixs = centroid_ixs[:j]
    if not skip_assignment:
        assignments = assignments
        centroid_ixs = centroid_ixs.to(assignments)
    phi = dists.mean()

    return centroid_ixs, assignments, dists, phi


def truncated_kmeans(
    X,
    max_sigma=6.0,
    n_components=10,
    n_initializations=10,
    random_state: int | np.random.Generator = 0,
    n_iter=100,
    min_log_prop=-25.0,
    dirichlet_alpha=1.0,
    kmeanspp_min_dist=0.0,
    sigma_atol=1e-3,
    batch_size=2048,
    dcc_batch_size=64,
    noise_const_dims=None,
    with_log_likelihoods=False,
    device=None,
    show_progress=False,
):
    rg = np.random.default_rng(random_state)

    if torch.is_tensor(X):
        device = X.device
    elif device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    is_gpu = device.type == "cuda"
    X = torch.asarray(X, device=device)
    n, p = X.shape

    # initialize...
    sigmasq = torch.inf
    nearest_distsq = centroid_ixs = labels = None
    if show_progress:
        it = trange(n_initializations, desc="kmeans++")
    else:
        it = range(n_initializations)
    initial_distances = None
    if noise_const_dims is not None:
        noise_dims = np.setdiff1d(np.arange(X.shape[1]), noise_const_dims)
        initial_distances = X[:, noise_dims].square().sum(1)
    for _ in it:
        _c, _l, _d, _p = kmeanspp(
            X,
            n_components=n_components,
            random_state=rg,
            kmeanspp_initial="random",
            min_distance=kmeanspp_min_dist,
            initial_distances=initial_distances,
        )
        if _p < sigmasq:
            sigmasq = _p
            centroid_ixs = _c
            labels = _l
            nearest_distsq = _d
    assert torch.isfinite(sigmasq)
    assert centroid_ixs is not None
    assert labels is not None
    assert nearest_distsq is not None

    if logger.isEnabledFor(DARTSORTDEBUG):
        logger.dartsortdebug(
            f"truncated_kmeans: Max dist {nearest_distsq.max().sqrt().item()} for "
            f"{len(centroid_ixs)} centroids. phi={sigmasq.sqrt().item()}."
        )
    del nearest_distsq, _d

    # initialize parameters
    n_components = len(centroid_ixs)
    neg_nc_log = -torch.log(torch.tensor(float(n_components)))
    log_proportions = X.new_full((n_components,), neg_nc_log)
    sigmasq = sigmasq / p
    centroids = X[centroid_ixs]

    # scratch buffers
    new_centroids = torch.zeros_like(centroids)
    N = X.new_zeros(log_proportions.shape, dtype=torch.double)
    prev_sigma = torch.inf
    dcc = X.new_zeros((n_components, n_components))
    dccbuf = X.new_zeros((dcc_batch_size, n_components, p))
    distsq_buf = torch.zeros_like(X[: 20 * batch_size])
    distsq_buf = (distsq_buf, distsq_buf.clone())
    sigma = sigmasq.sqrt().numpy(force=True).item()
    if with_log_likelihoods:
        log_likelihoods = X.new_full((n,), -torch.inf)
    else:
        log_likelihoods = None

    if show_progress:
        it = trange(n_iter, desc=f"kmeans σ={sigma:0.4f}")
    else:
        it = range(n_iter)

    done = False
    for j in it:
        done = done or j == n_iter - 1
        max_distance_sq = max_sigma * sigmasq * p

        new_centroids.fill_(0.0)
        N.fill_(0.0)
        new_sigmasq = 0.0
        weight = 0.0

        # update centroid dists
        for i0 in range(0, n_components, dcc_batch_size):
            i1 = min(n_components, i0 + dcc_batch_size)
            torch.subtract(centroids[None], centroids[i0:i1, None], out=dccbuf[i0:i1]).square_()
            torch.sum(dccbuf[i0:i1], dim=2, out=dcc[i0:i1])
        dccmask = dcc < max_distance_sq

        for i0 in range(0, n, batch_size):
            i1 = min(n, i0 + batch_size)
            distsq_coo, distsq_buf = sparse_centroid_distsq(
                X[i0:i1],
                centroids,
                labels=labels[i0:i1],
                centroid_mask=dccmask,
                dbufs=distsq_buf,
            )
            assert distsq_coo.shape == (i1 - i0, n_components)
            distsq_values = distsq_coo.values().clone()
            liks = distsq_to_lik_coo(
                distsq_coo, sigmasq, log_proportions, in_place=True
            )
            del distsq_coo
            if done and with_log_likelihoods:
                assert log_likelihoods is not None
                log_likelihoods[i0:i1] = logsumexp_coo(liks)
                continue

            resps = torch.sparse.softmax(liks, dim=1)

            # update labels... torch sparse has no argmax(), so need scipy
            # or cupy. scipy is a big slowdown here, so cupy if possible.
            if is_gpu and HAVE_CUPY:
                resps_cupy = coo_to_cupy(resps).tocsc()
                batch_labels = resps_cupy.argmax(axis=1).squeeze()
            else:
                resps_scipy = coo_to_scipy(resps)
                batch_labels = resps_scipy.argmax(axis=1, explicit=True)
            labels[i0:i1] = torch.asarray(batch_labels).to(labels)

            # get sigmasq
            w = resps.values().clone()
            batch_w = w.sum()
            w /= batch_w
            batch_sigmasq = torch.sum(distsq_values.mul_(w)) / p

            # get N and centroids
            batch_N = resps.sum(dim=0).to_dense()
            resps.values().div_(batch_N[resps.indices()[1]])
            batch_centroids = resps.T @ X[i0:i1]

            # update counts
            N += batch_N
            weight += batch_w

            # update Welford running means
            n1_n01 = batch_N.div_(N.clip(min=1e-5))[:, None]
            w1_w01 = batch_w / weight
            new_centroids += batch_centroids.sub_(new_centroids).mul_(n1_n01)
            new_sigmasq += batch_sigmasq.sub_(new_sigmasq).mul_(w1_w01)

        if done:
            break

        # update state
        logN = N.log_() + dirichlet_alpha
        log_proportions = F.log_softmax(logN, dim=0).to(X)
        log_proportions = log_proportions.clamp_(min=min_log_prop)
        centroids, new_centroids = new_centroids, centroids
        sigmasq = new_sigmasq

        # check convergence
        sigma = torch.sqrt(sigmasq).numpy(force=True).item()
        if abs(sigma - prev_sigma) < sigma_atol:
            done = True
            if not with_log_likelihoods:
                break
        prev_sigma = sigma
        if show_progress:
            it.set_description(f"kmeans σ={sigma:0.4f}")

    return dict(
        centroid_ixs=centroid_ixs,
        centroids=centroids,
        sigmasq=sigmasq,
        sigma=sigma,
        log_proportions=log_proportions,
        labels=labels,
        log_likelihoods=log_likelihoods,
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
    test_convergence=True,
    atol=1e-5,
):
    """A bit more than K-means

    Supports a proportion vector, as well as automatically dropping tiny clusters.
    """
    best_phi = torch.inf
    centroid_ixs = labels = None
    if isinstance(random_state, int):
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
    phi = None

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

        if test_convergence:
            phi_ = (e @ dists.T).mean()
            if phi is not None and torch.isclose(phi, phi_, atol=atol):
                break
            phi = phi_

    assignments = torch.argmin(dists, 1)
    return assignments, e, centroids, dists


def kmeans(
    X,
    n_kmeans_tries=5,
    n_kmeanspp_tries=5,
    n_iter=100,
    n_components=10,
    random_state: np.random.Generator | torch.Generator | int = 0,
    kmeanspp_initial="random",
    with_proportions=False,
    drop_prop=0.025,
    drop_sum=5.0,
    test_convergence=True,
):
    best_phi = np.inf
    if isinstance(random_state, int):
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
            test_convergence=test_convergence,
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
