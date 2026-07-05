import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from dartsort.util.py_util import databag

try:
    import cupy  # type: ignore # ty: ignore[x]

    del cupy

    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False


from ..util.logging_util import DARTSORTDEBUG, get_logger, progrange
from ..util.sparse_util import (
    coo_to_cupy,
    coo_to_scipy,
    distsq_to_lik_coo,
    logsumexp_coo,
    sparse_centroid_distsq,
)
from ..util.spiketorch import spawn_torch_rg, sqeuc_cdist_known_norm

logger = get_logger(__name__)


def kmeanspp(
    X: Tensor,
    weights: Tensor | None = None,
    n_components=10,
    random_state: np.random.Generator | torch.Generator | int = 0,
    kmeanspp_initial="random",
    skip_assignment=False,
    min_distance=None,
    Xnormsq: Tensor | None = None,
):
    """K-means++ initialization

    Start at a random point (kmeanspp_initial=='random') or at the point
    farthest from the mean (kmeanspp_initial=='mean').
    """
    n, p = X.shape
    n_components = min(n, n_components)
    gen = spawn_torch_rg(random_state, device=X.device)

    centroid_ixs = torch.full((n_components,), n, dtype=torch.long, device=X.device)

    if kmeanspp_initial == "random":
        if weights is not None:
            centroid_ixs[0] = torch.multinomial(weights, 1, generator=gen)
        else:
            centroid_ixs[0] = torch.randint(n, size=(), device=X.device, generator=gen)
    elif kmeanspp_initial == "mean":
        closest = torch.cdist(X, X.mean(0, keepdim=True)).argmax()
        centroid_ixs[0] = closest.item()
    else:
        assert False

    if Xnormsq is None:
        Xnormsq = torch.linalg.vector_norm(X, dim=1).square_()

    dists_ = X.new_zeros(len(X))[:, None]
    dists = _sqeuc(X, Xnormsq[:, None], Y=X[centroid_ixs[0]][None], out=dists_)[:, 0]
    if skip_assignment:
        assignments = None
    else:
        assignments = X.new_zeros((n,), dtype=torch.long)

    if weights is None:
        p = dists.clone()
    else:
        p = dists * weights

    simple_case = (not min_distance) and (not skip_assignment)
    if simple_case:
        assert assignments is not None
        _kmeanspp_simple_loop(
            X=X,
            Xnormsq=Xnormsq,
            dists=dists,
            p=p,
            weights=weights,
            centroid_ixs=centroid_ixs,
            gen=gen,
            assignments=assignments,
        )
        j = n_components
    elif not min_distance:
        _kmeanspp_noassign_loop(
            X=X,
            Xnormsq=Xnormsq,
            dists=dists,
            p=p,
            weights=weights,
            centroid_ixs=centroid_ixs,
            gen=gen,
        )
        j = n_components
    else:
        j = 0
        for j in range(1, n_components):
            if weights is not None:
                torch.mul(dists, weights, out=p)
            else:
                p.copy_(dists)
            if min_distance:
                invalid = dists < min_distance
                if invalid.all():
                    break
                p.masked_fill_(invalid, 0.0)
            ci = torch.multinomial(p, 1, generator=gen)
            centroid_ixs[j] = ci

            newdists = sqeuc_cdist_known_norm(
                X,
                Xnormsq,
                X[ci],
                Xnormsq[ci],
                out=p[:, None],
            )[:, 0]
            if not skip_assignment:
                closer = newdists < dists
                assert assignments is not None
                assignments.masked_fill_(closer, j)
            torch.minimum(dists, newdists, out=dists)
        else:
            j += 1

    centroid_ixs = centroid_ixs[:j]
    if not skip_assignment:
        centroid_ixs = centroid_ixs.to(assignments)
    if weights is None:
        phi = dists.mean()
    else:
        phi = dists.mul_(weights).mean() / weights.mean()

    dists.relu_()

    return centroid_ixs, assignments, dists, phi


def truncated_kmeans(
    X,
    max_sigma=5.0,
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
    gen = spawn_torch_rg(rg, device=X.device)
    n, p = X.shape

    # initialize...
    sigmasq = torch.tensor(torch.inf)
    nearest_distsq = centroid_ixs = labels = None
    if show_progress:
        it = progrange(n_initializations, desc="kmeans++")
    else:
        it = range(n_initializations)
    _d = None
    for _ in it:
        _c, _l, _d, _p = kmeanspp(
            X,
            n_components=n_components,
            random_state=gen,
            kmeanspp_initial="random",
            min_distance=kmeanspp_min_dist,
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
    log_proportions = X.new_full((n_components,), neg_nc_log)  # type: ignore
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
        it = progrange(n_iter, desc=f"kmeans σ={sigma:0.4f}")
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
            torch.subtract(
                centroids[None], centroids[i0:i1, None], out=dccbuf[: i1 - i0]
            ).square_()
            torch.sum(dccbuf[: i1 - i0], dim=2, out=dcc[i0:i1])
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
                batch_liks = logsumexp_coo(liks)
                log_likelihoods[i0:i1] = batch_liks
                if done:
                    continue

            resps = torch.sparse.softmax(liks, dim=1)
            # update labels... torch sparse has no argmax(), so need scipy
            # or cupy. scipy is a big slowdown here, so cupy if possible.
            if is_gpu and HAVE_CUPY:
                resps_cupy = coo_to_cupy(resps).tocsc()
                batch_labels = resps_cupy.argmax(axis=1)
            else:
                resps_scipy = coo_to_scipy(resps)
                batch_labels = resps_scipy.argmax(axis=1, explicit=True)
            labels[i0:i1] = torch.as_tensor(batch_labels).to(labels).squeeze()

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
        sigma = torch.sqrt(sigmasq).numpy(force=True).item()  # type: ignore
        if abs(sigma - prev_sigma) < sigma_atol:
            done = True
            if not with_log_likelihoods:
                break
        prev_sigma = sigma
        if show_progress:
            it.set_description(f"kmeans σ={sigma:0.4f}")  # type: ignore

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
    *,
    n_kmeanspp_tries=5,
    n_iter=100,
    n_components=10,
    random_state: np.random.Generator | torch.Generator | int = 0,
    weights: Tensor | None = None,
    kmeanspp_initial="random",
    with_proportions=False,
    drop_prop: Tensor,
    test_convergence_every=10,
    atol=1e-5,
    X_normsq: Tensor | None = None,
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
            weights=weights,
            n_components=n_components,
            random_state=random_state,
            kmeanspp_initial=kmeanspp_initial,
            Xnormsq=X_normsq,
            skip_assignment=bool(n_iter),
        )
        if phi < best_phi:
            centroid_ixs = _centroid_ixs
            labels = _labels
            best_phi = phi

    centroids = X[centroid_ixs]

    if X_normsq is None:
        X_normsq = torch.linalg.vector_norm(X, dim=1).square_()
    assert X_normsq is not None
    X_normsq = X_normsq[:, None]
    dists_out = X.new_empty((X.shape[0], centroids.shape[0]))
    dists = _sqeuc(X, X_normsq, centroids, dists_out)
    # responsibilities, sum to 1 over centroids
    e = F.softmax(-0.5 * dists, dim=1)
    if not n_iter:
        assert labels is not None
        return labels, e, centroids, dists

    e, centroids, dists, proportions = _kmeans_main_loop(
        n_iter=n_iter,
        e=e,
        drop_prop=drop_prop,
        test_convergence_every=test_convergence_every,
        atol=atol,
        X=X,
        Xnormsq=X_normsq,
        dists=dists,
        weights=None if weights is None else weights[:, None],
        with_proportions=with_proportions,
        centroids=centroids,
    )
    (keep,) = proportions.nonzero(as_tuple=True)
    if not keep.numel():
        return X.new_full(X.shape[0], 0, dtype=torch.long), None, None, None
    e = e[:, keep]
    e.div_(e.sum(1, keepdim=True))
    centroids = centroids[keep]
    dists = dists[:, keep]
    proportions = e.mean(0)
    assignments = torch.argmin(dists, 1)
    return assignments, e, centroids, dists


@databag
class KMeansResult:
    labels: Tensor | None
    responsibilities: Tensor | None
    centroids: Tensor | None
    dists: Tensor | None


def kmeans(
    X: Tensor,
    n_kmeans_tries=5,
    n_kmeanspp_tries=5,
    n_iter=100,
    n_components=10,
    random_state: np.random.Generator | torch.Generator | int = 0,
    kmeanspp_initial="random",
    with_proportions=False,
    drop_prop=0.0,
    drop_sum=0.0,
    weights: Tensor | None = None,
    test_convergence_every=10,
) -> KMeansResult:
    best_phi = np.inf
    if isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)
    random_state = spawn_torch_rg(random_state, device=X.device)
    assignments = X.new_zeros(X.shape[:1], dtype=torch.long)
    e = centroids = dists = None
    X_normsq = torch.linalg.norm(X, dim=1).square_()
    drop_prop = max(drop_prop, drop_sum / len(X))
    drop_prop = X.new_full((), drop_prop)
    if weights is not None:
        weights = weights.to(device=X.device)
    with torch.jit.optimized_execution(True):
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
                test_convergence_every=test_convergence_every,
                weights=weights,
                X_normsq=X_normsq,
            )
            if dists is None:
                continue
            assert ee is not None
            phi = (ee * dists).sum(1).mean().numpy(force=True)
            if phi < best_phi:
                best_phi = phi
                assignments = aa.clone()
                e = ee
                centroids = cc
    return KMeansResult(
        labels=assignments, responsibilities=e, centroids=centroids, dists=dists
    )


def _uniform(gen: torch.Generator, buf: Tensor):
    return buf.uniform_(generator=gen)


def _one_gumbel_nolog(p: Tensor, gen: torch.Generator, buf: Tensor):
    z = _uniform(gen, buf)
    z = z.log_()
    return torch.divide(p, z, out=z).argmin(dim=0)


def _sqeuc(X: Tensor, Xnormsq: Tensor, Y: Tensor, out: Tensor):
    torch.addmm(Xnormsq, X, Y.transpose(0, 1), alpha=-2.0, out=out)
    Ynormsq = (Y**2).sum(dim=1)
    out.add_(Ynormsq)
    return out


def _kmeanspp_simple_loop(
    *,
    X: Tensor,
    Xnormsq: Tensor,
    dists: Tensor,
    p: Tensor,
    weights: Tensor | None,
    centroid_ixs: Tensor,
    gen: torch.Generator,
    assignments: Tensor,
):
    buf = torch.empty_like(p)
    buf_ = buf[:, None]
    closer = X.new_zeros(X.shape[:1], dtype=torch.bool)
    cent = torch.zeros_like(X[:1])
    cnormsq = torch.zeros_like(Xnormsq[:1])
    for j in range(1, centroid_ixs.shape[0]):
        if weights is None:
            p = dists
        else:
            torch.mul(dists, weights, out=p)

        # centroid_ixs[j : j + 1] = torch.multinomial(p, 1, generator=gen)
        cj = _one_gumbel_nolog(p, gen, buf)
        centroid_ixs[j] = cj

        torch.index_select(X, dim=0, index=cj, out=cent)
        torch.index_select(Xnormsq, dim=0, index=cj, out=cnormsq)
        newdists = sqeuc_cdist_known_norm(X, Xnormsq, cent, cnormsq, out=buf_).view(-1)

        torch.lt(newdists, dists, out=closer)
        assignments.masked_fill_(closer, j)
        torch.minimum(dists, newdists, out=dists)


def _kmeanspp_noassign_loop(
    *,
    X: Tensor,
    Xnormsq: Tensor,
    dists: Tensor,
    p: Tensor,
    weights: Tensor | None,
    centroid_ixs: Tensor,
    gen: torch.Generator,
):
    buf = torch.empty_like(p)
    buf_ = buf[:, None]
    cent = torch.zeros_like(X[:1])
    cnormsq = torch.zeros_like(Xnormsq[:1])
    for j in range(1, centroid_ixs.shape[0]):
        if weights is None:
            p = dists
        else:
            torch.mul(dists, weights, out=p)

        # centroid_ixs[j : j + 1] = torch.multinomial(p, 1, generator=gen)
        cj = _one_gumbel_nolog(p, gen, buf)
        centroid_ixs[j] = cj

        torch.index_select(X, dim=0, index=cj, out=cent)
        torch.index_select(Xnormsq, dim=0, index=cj, out=cnormsq)
        newdists = sqeuc_cdist_known_norm(X, Xnormsq, cent, cnormsq, out=buf_).view(-1)

        torch.minimum(dists, newdists, out=dists)


def _kmeans_main_loop(
    n_iter: int,
    e: Tensor,
    drop_prop: Tensor,
    test_convergence_every: int,
    atol: float,
    X: Tensor,
    Xnormsq: Tensor,
    dists: Tensor,
    weights: Tensor | None,
    with_proportions: bool,
    centroids: Tensor,
):
    check_prop = bool(drop_prop)
    phi = e.new_full((), torch.nan)
    N = e.sum(dim=0)
    Ntot = e.new_full((), float(e.shape[0]))
    proportions = N / Ntot
    maskb = torch.zeros(proportions.shape, dtype=torch.bool, device=X.device)
    mask = proportions.clone()
    _1 = torch.ones_like(mask)
    _0 = torch.zeros_like(mask)
    for j in range(n_iter):
        # update centroids
        w = e.div_(N)
        centroids = torch.mm(w.transpose(0, 1), X, out=centroids)

        # e step
        dists = _sqeuc(X, Xnormsq, centroids, dists)
        if weights is not None:
            dists.mul_(weights)

        if with_proportions:
            torch.add(proportions.log_(), dists, alpha=-0.5, out=e)
            e = F.softmax(e, dim=1)
        else:
            e = F.softmax(dists.mul_(-0.5), dim=1)
        N = torch.sum(e, dim=0, out=N)
        torch.div(N, Ntot, out=proportions)

        if check_prop:
            torch.gt(proportions, drop_prop, out=maskb)
            torch.where(maskb, _1, _0, out=mask)
            e.mul_(mask)
            proportions.mul_(mask)

        if not test_convergence_every:
            continue
        elif j == 0:
            phi = dists.mul_(e).sum(1).mean()
            continue
        elif j % test_convergence_every in (0, test_convergence_every - 1):
            phi_ = dists.mul_(e).sum(1).mean()
            done = torch.isclose(phi, phi_, atol=atol) or phi_.abs() < atol
            phi = phi_
            if done:
                break
    dists = _sqeuc(X, Xnormsq, centroids, dists)
    return e, centroids, dists, proportions


def batched_kmeans(
    X: Tensor,
    n_components: int,
    seed: torch.Generator | np.random.Generator | int = 0,
    n_iter: int = 100,
    kmeanspp_seeds_per_try: int = 5,
    n_tries: int = 10,
    test_convergence_every=10,
    atol=1e-5,
    with_labels=True,
    with_proportions=True,
    beta: float = 1.0,
) -> KMeansResult:
    """
    Compared to above:
     - with_proportions = True
     - drop_prop = 0
     - no weights allowed for now
     - kmeanspp always random initial
     - n_iter > 0
    """
    k = n_components
    del n_components
    assert n_iter > 0
    assert n_tries > 0
    assert k > 1
    assert kmeanspp_seeds_per_try >= 1

    dev = X.device
    gen = spawn_torch_rg(seed, device=dev)
    Xnormsq = torch.linalg.norm(X, dim=1).square_()
    n, dim = X.shape
    k = min(n, k)
    assert n >= k > 1
    ntries_k = n_tries * k
    n_kmeanspps = n_tries * kmeanspp_seeds_per_try

    # -- kmeanspp stage: initialization
    centroid_ixs = torch.full((k, n_kmeanspps), n, dtype=torch.long, device=dev)
    centroid_ixs[0] = torch.randint(n, size=(n_kmeanspps,), device=dev, generator=gen)
    dists = X.new_empty((n_kmeanspps, n))
    Y = X[centroid_ixs[0]]
    Ynormsq = Xnormsq[centroid_ixs[0]]
    dists = sqeuc_cdist_known_norm(Y, Ynormsq, X, Xnormsq, dists)

    # -- kmeanspp stage: loop
    # buf for random sampling with Gumbel trick and new distance storage
    _buf = torch.empty_like(dists)
    for j in range(1, k):
        # sample new centroid indices wppt dists (which is squared)
        # gumbel max: argmax [log(d) + -log(-log(u))]
        #  = argmax d / (-log u) = argmin d / (log u)
        u = _buf.uniform_(generator=gen).log_()
        u = torch.div(dists, u, out=u)
        cix_j = torch.argmin(u, dim=1, out=centroid_ixs[j])

        # grab jth centroid data
        torch.index_select(X, dim=0, index=cix_j, out=Y)
        torch.index_select(Xnormsq, dim=0, index=cix_j, out=Ynormsq)

        # update distances
        newdists = sqeuc_cdist_known_norm(Y, Ynormsq, X, Xnormsq, _buf)
        torch.minimum(dists, newdists, out=dists)

    # -- kmeanspp finish: pick best by phi
    if kmeanspp_seeds_per_try > 1:
        phi = dists.mean(1).view(n_tries, kmeanspp_seeds_per_try)
        best_kmpp = phi.argmin(1, keepdim=True)
        centroid_ixs = centroid_ixs.view(k, n_tries, kmeanspp_seeds_per_try)
        centroid_ixs = centroid_ixs.take_along_dim(dim=2, indices=best_kmpp[None, :, :])
        assert centroid_ixs.shape == (k, n_tries, 1)
        centroid_ixs = centroid_ixs[:, :, 0]
    assert centroid_ixs.shape == (k, n_tries)
    centroid_ixs = centroid_ixs.T.contiguous()

    # -- kmeans stage: initialization
    Y = X[centroid_ixs].view(ntries_k, dim)
    Ynormsq = Xnormsq[centroid_ixs].view(ntries_k)
    dists = dists.resize_(n, ntries_k)
    e = X.new_empty((n, n_tries, k))
    N = X.new_ones((n_tries, k))
    Ntot = X.new_full((), float(n))
    log_props = X.new_zeros((n_tries, k))
    phi = phi_ = e.new_full((n_tries,), torch.nan)

    # -- kmeans stage: loop
    check = False
    for j in range(n_iter):
        jmod = j % test_convergence_every
        check = (j in (0, n_iter - 1)) or (jmod in (0, test_convergence_every - 1))

        # e step
        dists = sqeuc_cdist_known_norm(X, Xnormsq, Y, Ynormsq, dists.view(n, ntries_k))
        dists = dists.view(n, n_tries, k)
        e = torch.add(log_props, dists, alpha=-0.5 * beta, out=e)
        e = F.softmax(e, dim=2)
        if check:
            phi_ = dists.mul_(e).mean(0).sum(1)
            assert phi_.shape == phi.shape

        # m step
        N = torch.sum(e, dim=0, out=N)
        if with_proportions:
            torch.div(N, Ntot, out=log_props).log_()
        w = e.div_(N)
        Y = torch.mm(w.view(n, ntries_k).t(), X, out=Y)
        Ynormsq = torch.linalg.vector_norm(Y, dim=1, out=Ynormsq)
        Ynormsq.square_()

        # check convergence
        if check:
            done = torch.allclose(phi, phi_, atol=atol) or phi_.max() < atol
            phi = phi_
            if done:
                break
    assert check  # => phi is up to date

    # -- kmeans finish: pick best kmeans by phi, update responsibilities
    best = phi.argmin()
    Y = Y.view(n_tries, k, dim)[best]
    Ynormsq = Ynormsq.view(n_tries, k)[best]
    log_props = log_props[best]
    dists = dists.resize_(n, k)
    e = e.resize_(n, k)
    dists = sqeuc_cdist_known_norm(X, Xnormsq, Y, Ynormsq, dists)
    e = torch.add(log_props, dists, alpha=-0.5 * beta, out=e)
    e = F.softmax(e, dim=1)
    if with_labels:
        labels = e.argmax(dim=1)
    else:
        labels = None

    return KMeansResult(centroids=Y, responsibilities=e, labels=labels, dists=dists)


def truncated_kmeans_from_labels(
    X: Tensor | np.ndarray,
    labels: Tensor | np.ndarray,
    device=None,
    atol=1e-3,
    max_sigma=5.0,
    dirichlet_alpha=1.0,
    n_iter=100,
    show_progress: bool = True,
    batch_size: int = 4096,
    centroid_dist_batch_size: int = 128,
    min_log_prop=-25.0,
    trunc_guess=20,
    initial_undershoot=2.0,
) -> KMeansResult:
    X = torch.asarray(X, device=device)
    labels = torch.asarray(labels, device=device)
    n, dim = X.shape
    assert labels.shape == (n,)
    is_gpu = X.device.type == "cuda"

    # flatten and count labels
    ulabels, labels = labels.unique(return_inverse=True)
    k = ulabels.shape[0]
    del ulabels

    # initialize parameters
    e = F.one_hot(labels, k).to(X)
    log_props = e.mean(0).log_()
    N = e.sum(dim=0)
    w = e.div_(N)
    Y = torch.mm(w.view(n, k).t(), X)
    Ynormsq = torch.linalg.vector_norm(Y, dim=1).square_()

    # initialize sigma
    sigmasq = X.new_zeros(())
    bY = X.new_empty((batch_size, dim))
    for i0 in range(0, n, batch_size):
        i1 = min(n, i0 + batch_size)
        bY = torch.index_select(Y, 0, labels[i0:i1], out=bY[: i1 - i0])
        bsigsq = bY.sub_(X[i0:i1]).square_().mean()
        sigmasq += (bsigsq - sigmasq) * ((i1 - i0) / i1)
    del bY
    assert sigmasq > 0
    sigma = initial_undershoot * sigmasq.sqrt_()
    assert sigma.isfinite().item()
    prev_sigma = sigma.clone()

    # storage
    dYY = X.new_zeros((k, k))
    dYYmask = X.new_zeros((k, k), dtype=torch.bool)
    new_Y = torch.empty_like(Y)
    distsq_buf = torch.zeros_like(X[: min(trunc_guess, k) * batch_size])
    distsq_buf = (distsq_buf, torch.zeros_like(distsq_buf))

    if show_progress:
        it = progrange(n_iter, desc=f"kmeans σ={sigma:0.4f}")
    else:
        it = range(n_iter)

    done = False
    for j in it:
        done = done or j == n_iter - 1
        max_distance_sq = max_sigma * sigmasq * dim

        new_Y.fill_(0.0)
        N.fill_(0.0)
        new_sigmasq = 0.0
        weight = 0.0

        # update centroid dists
        for i0 in range(0, k, centroid_dist_batch_size):
            i1 = min(k, i0 + centroid_dist_batch_size)
            sqeuc_cdist_known_norm(Y[i0:i1], Ynormsq[i0:i1], Y, Ynormsq, out=dYY[i0:i1])
        torch.lt(dYY, max_distance_sq, out=dYYmask)

        for i0 in range(0, n, batch_size):
            i1 = min(n, i0 + batch_size)
            distsq_coo, distsq_buf = sparse_centroid_distsq(
                X[i0:i1],
                Y,
                labels=labels[i0:i1],
                centroid_mask=dYYmask,
                dbufs=distsq_buf,
            )
            assert distsq_coo.shape == (i1 - i0, k)
            distsq_values = distsq_coo.values().clone()
            liks = distsq_to_lik_coo(distsq_coo, sigmasq, log_props, in_place=True)
            del distsq_coo

            resps = torch.sparse.softmax(liks, dim=1)
            # update labels... torch sparse has no argmax(), so need scipy
            # or cupy. scipy is a big slowdown here, so cupy if possible.
            if is_gpu and HAVE_CUPY:
                resps_cupy = coo_to_cupy(resps).tocsc()
                batch_labels = resps_cupy.argmax(axis=1)
            else:
                resps_scipy = coo_to_scipy(resps)
                batch_labels = resps_scipy.argmax(axis=1, explicit=True)
            labels[i0:i1] = torch.as_tensor(batch_labels).to(labels).squeeze()

            # get sigmasq
            w = resps.values().clone()
            batch_w = w.sum()
            w /= batch_w
            batch_sigmasq = torch.sum(distsq_values.mul_(w)) / dim

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
            new_Y += batch_centroids.sub_(new_Y).mul_(n1_n01)
            new_sigmasq += batch_sigmasq.sub_(new_sigmasq).mul_(w1_w01)

        # update state
        logN = N.log_() + dirichlet_alpha
        log_props = F.log_softmax(logN, dim=0).to(X)
        log_props = log_props.clamp_(min=min_log_prop)
        Y, new_Y = new_Y, Y
        Ynormsq = torch.linalg.vector_norm(Y, dim=1).square_()
        sigmasq = new_sigmasq

        # check convergence
        sigma = torch.sqrt(sigmasq).numpy(force=True).item()  # type: ignore
        if abs(sigma - prev_sigma) < atol:
            break

        prev_sigma = sigma
        if show_progress:
            it.set_description(f"kmeans σ={sigma:0.4f}")  # type: ignore

    return KMeansResult(
        labels=labels,
        responsibilities=None,
        centroids=Y,
        dists=None,
    )
