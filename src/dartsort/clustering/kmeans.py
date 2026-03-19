import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm.auto import trange

try:
    import cupy  # type: ignore[reportMissingImports]

    del cupy

    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False


from ..util.logging_util import DARTSORTDEBUG, get_logger
from ..util.sparse_util import (
    coo_to_cupy,
    coo_to_scipy,
    distsq_to_lik_coo,
    logsumexp_coo,
    sparse_centroid_distsq,
)
from ..util.spiketorch import spawn_torch_rg
from .density import guess_mode

logger = get_logger(__name__)


def kmeanspp(
    X,
    weights: Tensor | None = None,
    n_components=10,
    random_state: np.random.Generator | torch.Generator | int = 0,
    kmeanspp_initial="random",
    mode_dim=2,
    skip_assignment=False,
    min_distance=None,
    initial_distances=None,
    Xnormsq: Tensor | None = None,
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
        assert weights is None, "Not implemented."
        idists = torch.asarray(initial_distances, dtype=X.dtype, device=X.device)
    else:
        idists = None

    if kmeanspp_initial == "random":
        if weights is not None:
            centroid_ixs[0] = torch.multinomial(weights, 1, generator=gen)
        elif dists is None:
            centroid_ixs[0] = torch.randint(n, size=(), device=X.device, generator=gen)
        else:
            assert idists is not None
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
        cixs = guess_mode(Xm.numpy(force=True))
        cixs = torch.asarray(cixs, dtype=centroid_ixs.dtype, device=centroid_ixs.device)
        centroid_ixs.copy_(cixs)
    else:
        assert False

    if Xnormsq is None:
        Xnormsq = torch.square(X).sum(1)

    if dists is None:
        dists = X.new_zeros(len(X))
    dists = _sqeuc(X, Xnormsq[:, None], Y=X[centroid_ixs[0]][None], out=dists[:, None])[
        :, 0
    ]
    assignments = None
    if not skip_assignment:
        assignments = torch.zeros((n,), dtype=torch.long, device=X.device)

    if weights is None:
        p = dists.clone()
    else:
        p = dists * weights

    simple_case = (
        (not has_initial_dists) and (not min_distance) and (not skip_assignment)
    )
    if simple_case:
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
    else:
        for j in range(1, n_components):
            if has_initial_dists:
                assert idists is not None
                assert weights is None
                torch.minimum(dists, idists, out=p)
            elif weights is not None:
                torch.mul(dists, weights, out=p)
            else:
                p.copy_(dists)
            if min_distance:
                invalid = dists < min_distance
                if invalid.all():
                    break
                p.masked_fill_(invalid, 0.0)
            centroid_ixs[j] = torch.multinomial(p, 1, generator=gen)

            # newdists = torch.subtract(X, X[centroid_ixs[j]], out=diff_buffer).square_()
            # newdists = torch.sum(newdists, dim=1, out=p)
            newdists = _sqeuc(
                X, Xnormsq[:, None], X[centroid_ixs[j]][None], out=p[:, None]
            )[:, 0]
            if not skip_assignment:
                closer = newdists < dists
                assert assignments is not None
                assignments.masked_fill_(closer, j)
            torch.minimum(dists, newdists, out=dists)
        else:
            j += 1  # type: ignore

    centroid_ixs = centroid_ixs[:j]
    if not skip_assignment:
        assignments = assignments
        centroid_ixs = centroid_ixs.to(assignments)
    if weights is None:
        phi = dists.mean()
    else:
        phi = dists.mul_(weights).mean() / weights.mean()

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
    gen = spawn_torch_rg(rg, device=X.device)
    n, p = X.shape

    # initialize...
    sigmasq = torch.tensor(torch.inf)
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
            random_state=gen,
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
    del nearest_distsq, _d  # type: ignore

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
    n_kmeanspp_tries=5,
    n_iter=100,
    n_components=10,
    random_state: np.random.Generator | torch.Generator | int = 0,
    weights: Tensor | None = None,
    kmeanspp_initial="random",
    with_proportions=False,
    drop_prop=0.025,
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
        )
        if phi < best_phi:
            centroid_ixs = _centroid_ixs
            labels = _labels
            best_phi = phi
    assert labels is not None

    centroids = X[centroid_ixs]

    if X_normsq is None:
        X_normsq = torch.linalg.norm(X, dim=1).square_()
    assert X_normsq is not None
    X_normsq = X_normsq[:, None]
    dists_out = X.new_empty((X.shape[0], centroids.shape[0]))
    dists = _sqeuc(X, X_normsq, centroids, dists_out)
    # responsibilities, sum to 1 over centroids
    e = F.softmax(-0.5 * dists, dim=1)
    if not n_iter:
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
        return torch.full_like(labels, 0), None, None, None
    e = e[:, keep]
    e.div_(e.sum(1, keepdim=True))
    centroids = centroids[keep]
    dists = dists[:, keep]
    proportions = e.mean(0)
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
    weights: Tensor | None = None,
    test_convergence_every=10,
):
    best_phi = np.inf
    if isinstance(random_state, int):
        random_state = np.random.default_rng(random_state)
    assignments = torch.zeros(len(X), dtype=torch.long)
    e = centroids = dists = None
    X_normsq = torch.linalg.norm(X, dim=1).square_()
    drop_prop = max(drop_prop, drop_sum / len(X))
    with torch.jit.optimized_execution(True):  # type: ignore
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
                assignments = aa
                e = ee
                centroids = cc
    return dict(
        labels=assignments, responsibilities=e, centroids=centroids, dists=dists
    )


@torch.jit.script
def _one_gumbel_nolog(p: Tensor, gen: torch.Generator, buf: Tensor):
    x = p.log_()
    z = torch.rand(size=x.shape, generator=gen, out=buf)
    return x.sub_(z.log_().neg_().log_()).argmax()


@torch.jit.script
def _sqeuc(X: Tensor, Xnormsq: Tensor, Y: Tensor, out: Tensor):
    out = torch.addmm(Xnormsq, X, Y.t(), alpha=-2.0, out=out)
    Ynormsq = Y.square().sum(1)
    out.add_(Ynormsq)
    return out


@torch.jit.script
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
    Xnormsq = Xnormsq[:, None]
    for j in range(1, centroid_ixs.shape[0]):
        if weights is None:
            p.copy_(dists)
        else:
            torch.mul(dists, weights, out=p)

        # centroid_ixs[j : j + 1] = torch.multinomial(p, 1, generator=gen)
        cj = _one_gumbel_nolog(p, gen, buf)
        centroid_ixs[j] = cj

        cent = X[cj : cj + 1]
        newdists = _sqeuc(X, Xnormsq, cent, out=buf_)[:, 0]

        closer = newdists < dists
        assignments.masked_fill_(closer, j)
        torch.minimum(dists, newdists, out=dists)


@torch.jit.script
def _kmeans_main_loop(
    n_iter: int,
    e: Tensor,
    drop_prop: float,
    test_convergence_every: int,
    atol: float,
    X: Tensor,
    Xnormsq: Tensor,
    dists: Tensor,
    weights: Tensor | None,
    with_proportions: bool,
    centroids: Tensor,
):
    K = e.shape[1]
    check_prop = bool(drop_prop)
    proportions = e.mean(0)
    phi = e.new_full((), torch.inf)
    for j in range(n_iter):
        # update centroids
        w = e / e.sum(0)
        centroids = torch.mm(w.T, X, out=centroids)

        # e step
        dists = _sqeuc(X, Xnormsq, centroids, dists)
        if weights is not None:
            dists.mul_(weights)

        if with_proportions:
            e = F.softmax(-0.5 * dists + proportions.log(), dim=1)
        else:
            e = F.softmax(-0.5 * dists, dim=1)
        proportions = e.mean(0)

        if check_prop:
            maskb = proportions > drop_prop
            mask = maskb.to(e)
            e.mul_(mask)
            proportions.mul_(mask)

        if test_convergence_every and j:
            test = (not j % test_convergence_every) or (
                not (j + 1) % test_convergence_every
            )
            if test:
                phi_ = (e @ dists.T).mean()
                done = torch.isclose(phi, phi_, atol=atol) or phi_.abs() < atol
                phi = phi_
                if done:
                    break
    return e, centroids, dists, proportions
