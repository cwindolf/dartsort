from threading import local
from typing import Literal, cast

import numba
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from torch import Tensor

from ..util.logging_util import get_logger, progbar, progrange
from ..util.multiprocessing_util import get_pool
from .cluster_util import decrumb

logger = get_logger(__name__)


def kdtree_inliers(
    X,
    kdtree=None,
    n_neighbors: int | None = 10,
    distance_upper_bound: float | None = 25.0,
    workers=1,
    batch_size=2**16,
    leafsize=10,
    show_progress=False,
) -> tuple[np.ndarray, KDTree]:
    """Mark outlying points by a neighbors distance criterion

    Returns
    -------
    inliers : bool array
        inliers[i] = X[i] has at least n_neighbors neighbors within distance
        distance_upper_bound
    kdtree : KDTree
    """
    if kdtree is None:
        kdtree = KDTree(X, leafsize=leafsize)

    if distance_upper_bound is None or not n_neighbors:
        inliers = np.ones(kdtree.n, dtype=bool)
        return inliers, kdtree

    inliers = np.zeros(kdtree.n, dtype=bool)
    if show_progress:
        iters = progrange(0, kdtree.n, batch_size, desc="KDTin")
    else:
        iters = range(0, kdtree.n, batch_size)
    for i0 in iters:
        i1 = min(kdtree.n, i0 + batch_size)

        _, indices = kdtree.query(
            X[i0:i1],
            k=1 + n_neighbors,
            distance_upper_bound=distance_upper_bound,
            workers=workers,
        )
        indices = cast(np.ndarray, indices)
        inliers[i0:i1] = indices[:, -1] < kdtree.n

    return inliers, kdtree


def get_smoothed_density(
    X,
    inliers: np.ndarray | slice = slice(None),
    sigma: float = 1.0,
    sigma_low: float | None = None,
    sigma_ramp_ax=-1,
    bin_sizes=None,
    bin_size_ratio=10.0,
    min_bin_size=0.1,
    ramp_min_bin_size=5.0,
    max_n_bins=128,
    min_n_bins=5,
) -> np.ndarray:
    """Get RBF density estimates for each X[i] and bandwidth in sigmas

    Outliers will be marked with NaN KDEs. Please pass inliers, or else your
    histogram is liable to be way too big.
    """
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter
    # figure out what bandwidths we'll be working on

    if do_ramp := bool(sigma_low):
        min_sigma = min(sigma, sigma_low)  # type: ignore
    else:
        min_sigma = sigma

    # histogram bin sizes -- not too big, not too small
    infeats = X[inliers]
    bin_sizes = np.full(X.shape[1], min_sigma / bin_size_ratio)
    bin_sizes = np.maximum(min_bin_size, bin_sizes)
    if do_ramp:
        bin_sizes[sigma_ramp_ax] = max(bin_sizes[sigma_ramp_ax], ramp_min_bin_size)

    # select bin edges
    extents = np.c_[np.floor(infeats.min(0)), np.ceil(infeats.max(0))]
    dextents = np.ptp(extents, 1)
    if not (dextents > 0).all():
        raise ValueError(
            f"Issue in KDE. {dextents=} {infeats.shape=} {sigma=} {bin_sizes=}."
        )
    nbins = np.ceil(dextents / bin_sizes).astype(int)
    nbins = nbins.clip(min_n_bins, max_n_bins)
    bin_edges = [np.linspace(e[0], e[1], num=nb + 1) for e, nb in zip(extents, nbins)]

    # compute histogram and figure out how big the bins actually were
    raw_histogram, bin_edges = np.histogramdd(infeats, bins=bin_edges)
    bin_sizes = np.array([(be[1] - be[0]) for be in bin_edges])
    bin_centers = [0.5 * (be[1:] + be[:-1]) for be in bin_edges]

    # normalize histogram to samples / volume
    raw_histogram = raw_histogram / bin_sizes.prod()

    # figure out how histogram should be filtered
    hist = raw_histogram
    if sigma is not None and sigma_low is None:
        sigma_hist = sigma * np.reciprocal(bin_sizes)
        hist = gaussian_filter(raw_histogram, sigma_hist)
    elif sigma is not None and sigma_low is not None:
        # filter by a sequence of bandwidths
        sigma_hist = sigma * np.reciprocal(bin_sizes)
        sigma_low_hist = sigma_low * np.reciprocal(bin_sizes)
        ramp = np.linspace(
            sigma_low_hist[sigma_ramp_ax],
            sigma_hist[sigma_ramp_ax],
            num=hist.shape[sigma_ramp_ax],
        )

        # operate along the ramp axis
        hist_move = np.moveaxis(hist, sigma_ramp_ax, 0)
        hist_smoothed = hist_move.copy()
        for j, sig in enumerate(ramp):
            sig_move = sig.copy()
            sig_move[0] = sig[sigma_ramp_ax]
            sig_move[sigma_ramp_ax] = sig[0]
            hist_smoothed[j] = gaussian_filter(hist_move, sig_move)[j]  # sig_move
        hist = np.moveaxis(hist_smoothed, 0, sigma_ramp_ax)

    lerp = RegularGridInterpolator(bin_centers, hist, bounds_error=False)
    low = np.array([np.min(bc) for bc in bin_centers])
    high = np.array([np.max(bc) for bc in bin_centers])
    dens = lerp(X.clip(low, high))

    return dens


def get_smoothed_density_ratio(
    X,
    *,
    inliers: np.ndarray | slice = slice(None),
    sigmas: list[float],
    sigma_lows: list[float] | list[None] | None = None,
    sigma_ramp_ax=-1,
    bin_sizes=None,
    bin_size_ratio=10.0,
    min_bin_size=0.1,
    ramp_min_bin_size=5.0,
    max_n_bins=128,
    min_n_bins=5,
) -> np.ndarray:
    assert 0 < len(sigmas) <= 2
    sigma_lows = sigma_lows or [None] * len(sigmas)
    dens = [
        get_smoothed_density(
            X=X,
            inliers=inliers,
            sigma=sigma,
            sigma_low=sigma_low,
            sigma_ramp_ax=sigma_ramp_ax,
            bin_sizes=bin_sizes,
            bin_size_ratio=bin_size_ratio,
            min_bin_size=min_bin_size,
            ramp_min_bin_size=ramp_min_bin_size,
            max_n_bins=max_n_bins,
            min_n_bins=min_n_bins,
        )
        for sigma, sigma_low in zip(sigmas, sigma_lows)
    ]
    if len(sigmas) == 1:
        return dens[0]
    else:
        dens_ = dens[0]
        dens_ /= dens[1]
        np.nan_to_num(dens_, out=dens_)  # type: ignore
        return dens_


def nearest_higher_density_neighbor(
    kdtree,
    density,
    n_neighbors_search=20,
    distance_upper_bound=5.0,
    workers=1,
    batch_size=2**16,
) -> np.ndarray:
    nhdn = np.full(kdtree.n, kdtree.n, dtype=np.intp)
    density_padded = np.pad(density, (0, 1), constant_values=np.inf)

    for i0 in range(0, kdtree.n, batch_size):
        i1 = min(kdtree.n, i0 + batch_size)

        distances, indices = kdtree.query(
            kdtree.data[i0:i1],
            k=1 + n_neighbors_search,
            distance_upper_bound=distance_upper_bound,
            workers=workers,
        )
        # exclude self
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        missing = indices == kdtree.n
        distances[missing] = np.inf

        # find lowest distance higher density neighbor
        bdens = density[i0:i1]
        bneighb_dens = density_padded[indices]
        is_lower_density = np.logical_or(
            bneighb_dens <= bdens[:, None], np.isneginf(bneighb_dens)
        )
        distances[is_lower_density] = np.inf
        indices[is_lower_density] = kdtree.n

        nearest = distances.argmin(1, keepdims=True)
        nhdn[i0:i1] = np.take_along_axis(indices, nearest, axis=1)[:, 0]

    return nhdn


def remove_border_points(
    labels, density, kdtree, search_radius=1.0, n_neighbors_search=5, workers=1
):
    distances, indices = kdtree.query(
        kdtree.data,
        k=1 + n_neighbors_search,
        distance_upper_bound=search_radius,
        workers=workers,
    )

    labels_padded = np.pad(labels, (0, 1), constant_values=-2)
    in_border = np.any(
        labels_padded[indices] != labels_padded[indices[:, 0, None]], axis=1
    )
    in_border = np.flatnonzero(in_border)
    labels_in_border = labels[in_border]

    units = np.unique(labels)
    new_labels = labels.copy()
    for i, u in enumerate(units[units > 0]):
        in_unit = np.flatnonzero(labels == u)
        in_unit_border = in_border[labels_in_border == u]
        if not in_unit_border.size:
            continue

        to_remove = density[in_unit] < density[in_unit_border].max()
        new_labels[in_unit[to_remove]] = -1

    return new_labels


def guess_mode(
    X,
    sigma: float | Literal["rule_of_thumb"] = "rule_of_thumb",
    outlier_neighbor_count=10,
    outlier_sigma=3.0,
    kdtree=None,
    workers=1,
):
    """Use a KDE to guess the highest density point."""
    n = len(X)
    sigma0 = np.sqrt(X.var(axis=0).sum())
    inliers, kdtree = kdtree_inliers(
        X,
        kdtree=kdtree,
        n_neighbors=outlier_neighbor_count,
        distance_upper_bound=outlier_sigma * sigma0,
        workers=workers,
    )

    if sigma == "rule_of_thumb":
        sigma_dens = (
            1.06
            * np.linalg.norm(np.std(X[inliers], axis=0))
            * np.power(inliers.sum(), -0.2)
        )
    else:
        sigma_dens = sigma

    density = get_smoothed_density(X, inliers=inliers, sigma=sigma_dens)
    assert isinstance(density, np.ndarray)
    assert density.shape == (n,)

    return np.argmax(density)


class KmeansppBallTree:
    def __init__(
        self,
        X: np.ndarray | torch.Tensor,
        max_roots=128,
        max_layer_size=38400,
        target_leafsize=128,
        target_branching=16,
        batch_size=2048,
        sort_0: bool = True,
        device: torch.device | None = torch.device("cpu"),
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target_leafsize = target_leafsize
        self.target_branching = target_branching
        self.max_layer_size = max_layer_size
        self.batch_size = batch_size

        # -- lists of tensors. for each layer, the tensor is the same length
        # leaves are datapoints; higher layers are kmeanspp centroids
        self.layers = []
        # each datapoint or centroid's sqeucnorm
        self.layer_normsqs = []
        # for non-leaf layers, the centroid's index in the layer below
        self.layer_descendant_inds = [None]
        # each leaf or centroid's parent (closest centroid)
        self.layer_parent_inds = []
        # all inds of descendents for each centroid
        self.layer_children = [None]
        self.layer_nchildren = [None]
        # this centroid's max leaf descendant dist
        self.layer_radii = [None]

        self.add_leaf_layer(X, device, sort_0)

        while self.layers[-1].shape[0] > max_roots:
            self.add_layer()

        nlayers = len(self.layers)
        layerlens = [len(ll) for ll in self.layers]
        logger.info(f"tree: {nlayers=} {layerlens=}")

    def add_leaf_layer(self, X, device, sort_0):
        X = torch.asarray(X, device=device)
        if sort_0:
            self.leaf_order = torch.argsort(X[:, 0])
            X = X[self.leaf_order]
        else:
            self.leaf_order = None
        assert X.ndim == 2
        Xnormsq = torch.linalg.vector_norm(X, dim=1).square_()

        self.layers.append(X)
        self.layer_normsqs.append(Xnormsq)

    def add_layer(self):
        from .kmeans import kmeanspp

        X = self.layers[-1]
        Xnormsq = self.layer_normsqs[-1]

        n = len(X)
        logn = int(np.ceil(np.log2(n)))

        if len(self.layers) == 1:
            # descendents are leaves
            branching = max(self.target_leafsize, logn)
        else:
            # ancestor branching
            branching = max(self.target_branching, logn)
        ny = min(self.max_layer_size, n, n // branching)

        Y_Xixs, lx2y, dsqxy, _ = kmeanspp(X, n_components=ny, Xnormsq=Xnormsq)
        assert Y_Xixs.shape[0] == ny
        assert lx2y.shape[0] == dsqxy.shape[0] == n

        # reorder descendents by label in first layer
        if len(self.layers) == 1:
            lx2y, order = lx2y.sort(stable=True)
            reorder = torch.argsort(order)
            if self.leaf_order is not None:
                self.leaf_order = self.leaf_order[order]
            else:
                self.leaf_order = order
            Y_Xixs = reorder[Y_Xixs]
            X = X[order]
            Xnormsq = Xnormsq[order]
            dsqxy = dsqxy[order]
            self.layers[-1] = X
            self.layer_normsqs[-1] = Xnormsq

        # sparsify labels: "indptr" style in first layer, coo later
        if len(self.layers) == 1:
            diff = lx2y.diff().cpu()
            assert (diff >= 0).all()
            indptr = diff.nonzero()[:, 0].add_(1)
            _0 = indptr.new_zeros((1,))
            indptr = torch.concatenate([_0, indptr, _0 + n])
            assert len(indptr) == ny + 1
            children = [slice(i0, i1) for i0, i1 in zip(indptr[:-1], indptr[1:])]
            nchildren = [c.stop - c.start for c in children]
        else:
            children = [(lx2y == ll).nonzero()[:, 0] for ll in range(ny)]
            nchildren = [c.shape[0] for c in children]

        # get centroid radii
        radii = dsqxy.new_zeros(ny)
        if len(self.layers) == 1:
            child_radii = dsqxy.sqrt_()
        else:
            child_radii = dsqxy.sqrt_().add_(self.layer_radii[-1])
        assert child_radii is not None
        for i, ixs in enumerate(children):
            if isinstance(ixs, Tensor):
                assert ixs.numel() > 0
            else:
                assert ixs.stop > ixs.start
            radii[i] = child_radii[ixs].amax()
        assert radii.isfinite().all()

        # store new layer
        self.layers.append(X[Y_Xixs])
        self.layer_normsqs.append(Xnormsq[Y_Xixs])
        self.layer_descendant_inds.append(Y_Xixs)
        self.layer_parent_inds.append(lx2y)
        self.layer_children.append(children)
        self.layer_radii.append(radii)
        self.layer_nchildren.append(lx2y.new_tensor(nchildren))

    def candidate_root_neighbors(self, radius: float, include_diag=True):
        """
        Two root nodes are neighbors if they could have a pair of
        leaf descendents within radius.

        That can be figured out using their radii: let xi be in the descendent
        set of root ri, xj for rj. Then
          d(xi, xj) >= d(ri, rj) - rootradi - rootradj

        Then if
          d(ri, rj) >= radius + rootradi + rootradj
        it follows that
          d(xi, xj) >= max_sigma
        """
        nr = self.layers[-1].shape[0]
        triu = torch.triu_indices(nr, nr, offset=1, device=self.layers[-1].device)
        ii, jj = triu
        root_dists_triu = _sqeuc_pdist_known_norm(
            X=self.layers[-1], Xnormsq=self.layer_normsqs[-1], ii=ii, jj=jj
        )
        root_radii = self.layer_radii[-1]
        if root_radii is None:
            # everyone is a leaf, no radius
            lb = radius**2
        else:
            rad_ii = root_radii[ii]
            rad_jj = root_radii[jj]
            lb = (rad_ii + radius).add_(rad_jj).square_()

        cand = root_dists_triu <= lb
        pairs = triu.T[cand]
        dist = root_dists_triu[cand]

        if include_diag:
            idc = torch.arange(nr, device=cand.device)
            idc = torch.stack([idc, idc], dim=1)
            pairs = torch.concatenate([pairs, idc])

            # lex sorting with unique
            pairs, inv = pairs.unique(dim=0, return_inverse=True)
            inv = torch.argsort(inv)

            # reorder distances with diagonal terms
            _0 = root_dists_triu.new_zeros(nr)
            dist = torch.concatenate([dist, _0])
            dist = dist[inv]

        return pairs, dist

    def candidate_node_neighbor_inds(self, child_layer: int, radius: float):
        if child_layer == len(self.layers) - 1:
            yield self.candidate_root_neighbors(radius, include_diag=True)
            return

        parent_iter = self.candidate_node_neighbor_inds(child_layer + 1, radius)

        pairs_batch = []
        dist_batch = []
        npairs = 0
        in_parent = self.layer_children[child_layer + 1]
        layer_radii = self.layer_radii[child_layer]
        assert in_parent is not None

        for iijj, dist in parent_iter:
            dev = iijj.device
            for ci, cj in iijj:
                # get batch of triu pairs in current layer
                in_ci = in_parent[ci]
                in_cj = in_parent[cj]

                if isinstance(in_ci, slice):
                    in_ci = torch.arange(in_ci.start, in_ci.stop, device=dev)
                    in_cj = torch.arange(in_cj.start, in_cj.stop, device=dev)

                # print
                assert (in_ci.diff() > 0).all()
                assert (in_cj.diff() > 0).all()
                if ci == cj:
                    ii, jj = torch.triu_indices(len(in_ci), len(in_ci), device=dev)
                    in_ci, in_cj = in_ci[ii], in_cj[jj]
                else:
                    in_ci, in_cj = torch.cartesian_prod(in_ci, in_cj).T

                # their pdist
                d_ci_cj = _sqeuc_pdist_known_norm(
                    X=self.layers[child_layer],
                    Xnormsq=self.layer_normsqs[child_layer],
                    ii=in_ci,
                    jj=in_cj,
                )

                # ones who COULD have leaf descendents within radius
                if layer_radii is None:
                    lb = radius**2
                else:
                    rad_ci = layer_radii[in_ci]
                    rad_cj = layer_radii[in_cj]
                    lb = (rad_ci + radius).add_(rad_cj).square_()

                (cand,) = (d_ci_cj <= lb).nonzero(as_tuple=True)
                if not cand.numel():
                    continue

                new_pairs = torch.column_stack([in_ci[cand], in_cj[cand]])
                new_npairs = new_pairs.shape[0]
                new_sqdist = d_ci_cj[cand]

                if npairs and (npairs + new_npairs >= self.batch_size):
                    yield torch.concatenate(pairs_batch), torch.concatenate(dist_batch)
                    pairs_batch = []
                    dist_batch = []
                    npairs = 0

                if new_npairs >= self.batch_size:
                    assert not npairs
                    assert len(pairs_batch) == 0
                    yield new_pairs, new_sqdist
                else:
                    pairs_batch.append(new_pairs)
                    dist_batch.append(new_sqdist)
                    npairs += new_npairs

            if npairs:
                yield torch.concatenate(pairs_batch), torch.concatenate(dist_batch)
                pairs_batch = []
                dist_batch = []
                npairs = 0

    def candidate_leaf_neighbor_inds(self, radius: float):
        yield from self.candidate_node_neighbor_inds(0, radius)

    def sparse_distance_matrix(self, radius: float):
        coos = []
        sqdists = []
        for iijj, sqdist in self.candidate_leaf_neighbor_inds(radius):
            coos.append(iijj)
            sqdists.append(sqdist)
        coos = torch.concatenate(coos).T
        if self.leaf_order is not None:
            coos = self.leaf_order[coos]
        coo = torch.sparse_coo_tensor(
            indices=coos,
            values=torch.concatenate(sqdists),
            size=(self.layers[0].shape[0], self.layers[0].shape[0]),
        )
        return coo + coo.T

    def gaussian_kdes(
        self,
        *sigmas: float,
        max_sigma: float = 3.0,
        show_progress: bool = False,
        normalized=True,
    ):
        # truncate dists larger than...
        radius = max_sigma * max(sigmas)

        iter = self.candidate_leaf_neighbor_inds(radius)
        if show_progress:
            iter = progbar(iter, desc="kmppKDE")

        X = self.layers[0]
        n, d = X.shape
        dens = [X.new_zeros(n) for _ in sigmas]
        gk = [-0.5 * sig**-2 for sig in sigmas]
        gk = [X.new_tensor(_k, device=X.device) for _k in gk]
        gn = [-(d / 2.0) * np.log(2.0 * np.pi * sig * sig) for sig in sigmas]
        gn = [X.new_tensor(_k, device=X.device) for _k in gn]
        for iijj, sqdist in iter:
            ii, jj = iijj.T
            diag = ii == jj
            for d, sgk, sgn in zip(dens, gk, gn):
                gauss = torch.addcmul(sgn, sgk, sqdist).exp_()
                d.scatter_add_(dim=0, index=ii, src=gauss)
                d.scatter_add_(dim=0, index=jj, src=gauss.masked_fill(diag, 0.0))

        if normalized:
            dens = [d.div_(n) for d in dens]

        # handle order pls
        if self.leaf_order is not None:
            deorder = torch.argsort(self.leaf_order)
            dens = [d[deorder] for d in dens]

        return dens


@torch.jit.script
def _sqeuc_pdist_known_norm(
    X: Tensor,
    Xnormsq: Tensor,
    ii: Tensor,
    jj: Tensor,
    batch_size: int = 1024,
):
    ntriu = ii.shape[0]
    out = X.new_zeros((ntriu,))
    Xl = X[:, None, :]
    Xr = X[:, :, None]
    for i0 in range(0, ntriu, batch_size):
        i1 = min(i0 + batch_size, ntriu)

        bii = ii[i0:i1]
        bjj = jj[i0:i1]

        Xlii = Xl[bii]
        Xrjj = Xr[bjj]

        normsq_sum = torch.add(Xnormsq[bii], Xnormsq[bjj], out=out[i0:i1])
        normsq_sum[:, None, None].baddbmm_(Xlii, Xrjj, alpha=-2.0)
    out.relu_()
    return out


def kdt_density(
    kdtree: KDTree,
    X: np.ndarray,
    sigma: float,
    sigma_regional: float,
    batch_size=2048,
    max_sigma: float = 3.0,
    n_threads=8,
):
    n, d = X.shape
    max_dist = max_sigma * sigma_regional

    jobs = range(0, n, batch_size)
    n_jobs, Executor, context = get_pool(n_jobs=n_threads, cls="ThreadPoolExecutor")
    with Executor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=_kdtdens_init,
        initargs=(kdtree, X, batch_size, sigma, sigma_regional, max_dist),
    ) as pool:
        density = np.full((n,), -np.inf)
        for i0, i1, dens in progbar(
            pool.map(_kdtdens_job, jobs),
            total=len(jobs),
            smoothing=0.0,
            desc=f"KDTdens[{n_jobs}]",
        ):
            density[i0:i1] = dens
    np.nan_to_num(density, copy=False)
    return density


_kdtdens_ctx = local()
_kdtdens_ctx.sargs = None


def _kdtdens_init(kdtree, X, batch_size, sigma, sigma_regional, max_dist):
    global _kdtdens_ctx
    _kdtdens_ctx.sargs = (kdtree, X, batch_size, sigma, sigma_regional, max_dist)


def _kdtdens_job(i0):
    global _kdtdens_ctx
    kdtree, X, batch_size, sigma, sigma_regional, max_dist = _kdtdens_ctx.sargs
    i1 = min(kdtree.n, i0 + batch_size)
    batch_kdt = KDTree(X[i0:i1])
    sdm = batch_kdt.sparse_distance_matrix(kdtree, max_dist, output_type="ndarray")
    ii = sdm["i"]
    jj = sdm["j"]
    v_local = sdm["v"]
    v_regional = sdm["v"].copy()
    dim = float(X.shape[1])
    _gausskernel(v_local, sigma, dim)
    _gausskernel(v_regional, sigma_regional, dim)
    coo_local = coo_array(
        (v_local, (ii, jj)), shape=(batch_kdt.n, kdtree.n), dtype=v_local.dtype
    )
    coo_regional = coo_array(
        (v_regional, (ii, jj)), shape=(batch_kdt.n, kdtree.n), dtype=v_local.dtype
    )
    dens = coo_local.sum(axis=1)
    dens_regional = coo_regional.sum(axis=1)
    dens /= dens_regional

    return i0, i1, dens


_log_2pi = np.log(2.0 * np.pi)


@numba.jit(
    parallel=False, nopython=True, boundscheck=False, nogil=True, error_model="numpy"
)
def _gausskernel(v: np.ndarray, sigma: float, dim: float):
    h = np.sqrt(2.0) * sigma
    const = 1.0 / h
    addconst = (dim / 2) * np.log(2.0 * np.pi * sigma * sigma)
    for i in range(v.size):
        v[i] = np.exp(-np.square(v[i] * const) - addconst)


def knn_density(
    kdtree, X, k, distance_upper_bound, batch_size=2**12, workers=-1, sigma=None
):
    n, d = X.shape
    density = np.full(n, -np.inf, dtype="float32")

    for i0 in range(0, n, batch_size):
        i1 = min(n, i0 + batch_size)
        distances, indices = kdtree.query(
            X[i0:i1],
            k=k,
            distance_upper_bound=distance_upper_bound,
            workers=workers,
        )
        if sigma is None:
            np.nan_to_num(distances, copy=False, posinf=-np.inf)
            n_neighbs = (indices < kdtree.n).sum(axis=1)
            max_dist = distances[:, 1:].max(1)
            np.nan_to_num(max_dist, copy=False, posinf=-np.inf)
            # forget the sphere factor and the factor of n.
            density[i0:i1] = n_neighbs / (max_dist**d)
        else:
            kernel = np.square(distances, out=distances)
            kernel *= -(sigma**-2)
            np.exp(kernel, out=kernel)
            density[i0:i1] = kernel.sum(1)

    return density


def density_peaks(
    X,
    kdtree=None,
    density=None,
    knn_k=None,
    use_knn=False,
    use_histograms=False,
    sigma_local=5.0,
    sigma_regional=None,
    outlier_neighbor_count: int | None = 10,
    outlier_radius: float | None = 25.0,
    n_neighbors_search=20,
    radius_search=25.0,
    noise_density=0.0,
    remove_clusters_smaller_than=10,
    remove_borders=False,
    border_search_radius=10.0,
    border_search_neighbors=3,
    inlier_dims=slice(0, 2),
    leafsize=24,
    workers=-1,
    device: torch.device = torch.device("cpu"),
):
    """Density peaks clustering as described by Rodriguez and Laio, but...

    This implementation has three tweaks from original DPC
     - KDE density estimation, implemented by smoothing histograms
     - "Sharpening": density can be a ratio of KDEs, like a sharpening kernel
     - Noise: you can throw away points with too low of a density (ratio)
    """
    n = len(X)

    radius_search = radius_search * np.sqrt(X.shape[1])
    dim_arange = np.arange(X.shape[1])
    inlier_dim_ixs = dim_arange[inlier_dims]
    if outlier_radius is not None:
        outlier_radius = outlier_radius * np.sqrt(inlier_dim_ixs.size)
    inliers, kdtree = kdtree_inliers(
        X[:, inlier_dims],
        kdtree=kdtree,
        n_neighbors=outlier_neighbor_count,
        distance_upper_bound=outlier_radius,
        workers=workers,
        leafsize=leafsize,
    )
    if not np.array_equal(inlier_dim_ixs, dim_arange):
        kdtree = KDTree(X)

    if density is None:
        if sigma_regional and not use_histograms:
            density = kdt_density(
                kdtree,
                X,
                sigma=sigma_local,
                sigma_regional=sigma_regional,
                n_threads=workers,
            )
        elif use_knn:
            density = knn_density(
                kdtree, X, k=knn_k, distance_upper_bound=radius_search, workers=workers
            )
        elif use_histograms:
            sigmas = [sigma_local]
            if sigma_regional is not None:
                sigmas.append(sigma_regional)
            density = get_smoothed_density_ratio(X, inliers=inliers, sigmas=sigmas)
        else:
            density = knn_density(
                kdtree,
                X,
                k=n_neighbors_search,
                distance_upper_bound=radius_search,
                workers=workers,
                sigma=sigma_local,
            )

    nhdn = nearest_higher_density_neighbor(
        kdtree,
        density,
        n_neighbors_search=n_neighbors_search,
        distance_upper_bound=radius_search,
        workers=workers,
    )
    if noise_density:
        nhdn[density < noise_density] = n

    nhdn = nhdn.astype(np.intc)
    has_nhdn = np.flatnonzero(nhdn < n).astype(np.intc)
    data = np.ones(has_nhdn.size)
    graph = coo_array((data, (nhdn[has_nhdn], has_nhdn)), shape=(n, n))
    ncc, labels = connected_components(graph)

    if remove_borders:
        labels = remove_border_points(
            labels,
            density,
            kdtree,
            search_radius=border_search_radius,
            n_neighbors_search=border_search_neighbors,
            workers=workers,
        )

    if remove_clusters_smaller_than:
        labels = decrumb(labels, min_size=remove_clusters_smaller_than, in_place=True)

    return dict(
        density=density,
        nhdn=nhdn,
        labels=labels,
        kdtree=kdtree,
    )


def nearest_neighbor_assign(
    kdtree, tree_labels, X_other, radius_search=5.0, workers=-1
):
    _, inds = kdtree.query(
        X_other, k=1, distance_upper_bound=radius_search, workers=workers
    )
    found = np.flatnonzero(inds < kdtree.n)
    other_labels = np.full(len(X_other), -1, dtype=tree_labels.dtype)
    other_labels[found] = tree_labels[inds[found]]
    return other_labels


def sparse_iso_hellinger(centroids, sigma, hellinger_threshold=0.25, centroids_b=None):
    """Sparse Hellinger distance between isotropic Gaussian components"""
    c = centroids * (1.0 / (sigma * np.sqrt(8.0)))
    kdt = KDTree(c)
    if centroids_b is None:
        kdt_b = kdt
    else:
        c_b = centroids_b * (1.0 / (sigma * np.sqrt(8.0)))
        kdt_b = KDTree(c_b)
    max_distance = np.sqrt(-np.log(1 - hellinger_threshold))
    dists = kdt.sparse_distance_matrix(
        kdt_b, max_distance=max_distance, output_type="ndarray"
    )
    vals = dists["v"]
    np.square(vals, out=vals)
    vals *= -1
    np.exp(vals, out=vals)  # now BC
    np.subtract(1, vals, out=vals)  # and now hell^2.
    np.abs(vals, out=vals)  # numerical -0
    np.sqrt(vals, out=vals)  # and now hell
    dists = coo_array((vals, (dists["j"], dists["i"])), shape=(kdt_b.n, kdt.n))
    return dists


def coo_nhdn(coo, densities) -> np.ndarray:
    ii, jj = coo.coords
    dists = coo.data.copy()
    M = dists.max()
    dens_ii = densities[ii]
    dens_jj = densities[jj]
    not_higher = dens_jj <= dens_ii

    # only higher density
    dists[not_higher] = np.inf
    dists[ii == jj] = 2 * M + 1  # i am my last resort
    coo2 = coo_array((dists, (ii, jj)), shape=coo.shape)
    # argmin == nearest
    nhdn = coo2.tocsc().argmin(axis=1, explicit=True)
    nhdn = np.atleast_1d(nhdn)
    assert nhdn.shape == densities.shape[:1]
    return nhdn


def gmm_density_peaks(
    X,
    channels,
    outlier_neighbor_count=10,
    outlier_radius=25.0,
    remove_clusters_smaller_than=50,
    workers=-1,
    n_initializations=10,
    n_iter=50,
    max_components_per_channel=20,
    min_spikes_per_component=10,
    random_state=0,
    kmeanspp_min_dist=0.0,
    hellinger_cutoff=0.95,
    hellinger_strong=0.0,
    hellinger_weak=0.999,
    max_sigma=5.0,
    sigma_atol=1e-4,
    max_samples=2_000_000,
    show_progress=True,
    use_hellinger=False,
    mop=True,
    n_neighbors_search=20,
    device=None,
):
    """Density peaks clustering via an isotropic GMM density estimate

    Idea: use an overfitted isotropic GMM (i.e. lots of components) to capture the
    density in a point cloud. Then, each GMM component is grouped together with its
    nearest higher density (i.e., higher mixing proportion, since isotropic) neighbor,
    much like the usual density peaks clustering algorithm.

    Neighbor-ness is determined by a Hellinger overlap criterion. Contrasted with the
    smoothed histogram DPC algorithm elsewhere in this file, this algorithm is a bit
    more adaptive, and that neighbor criterion is easier to tune than the k-d tree query
    parameters which define the point-to-point neighbor criterion for the other algorithm.

    TODO: parametrize as max_components_per_square_micron or something instead of per
    channel so that this has more hope of transferring from probe to probe.

    TODO: see parallelism todo in gmm_kmeans. This could be a lot faster.

    Parameters
    ----------
    X : (N_spikes, n_features) array
        For instance, the features property of a SimpleMatrixFeatures
    channels: (N_spikes,) array
        The channels to which each spike belongs.
        TODO: currently used in controlling the number of components, but it may
        change when the TODO above is implemented.
    """
    from .kmeans import truncated_kmeans

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    n = len(X)
    if n > max_samples:
        random_state = np.random.default_rng(random_state)
        choices = random_state.choice(n, size=max_samples, replace=False)
        choices.sort()
    else:
        choices = slice(None)

    Xi = X[choices]
    nc = len(Xi)
    inliers, kdtree = kdtree_inliers(
        Xi,
        n_neighbors=outlier_neighbor_count,
        distance_upper_bound=outlier_radius * np.sqrt(X.shape[1]),
        workers=workers or 1,
    )
    if not isinstance(choices, slice):
        inliers = choices[inliers]

    Xi = X[inliers]
    ni = len(Xi)
    uchans, cchans = np.unique(channels[inliers], return_counts=True)
    comps_per_chan = np.minimum(
        max_components_per_channel,
        np.ceil(cchans / min_spikes_per_component).astype(int),
    )
    n_components = min(
        comps_per_chan.sum(), int(np.ceil(ni / min_spikes_per_component))
    )
    logger.dartsortdebug(
        f"gmmdpc: {n_components} components for {cchans.size} channels"
        f"over {ni}/{nc} inliers"
    )
    res = truncated_kmeans(
        Xi,
        max_sigma=max_sigma,
        n_components=n_components,
        n_initializations=n_initializations,
        random_state=random_state,
        n_iter=n_iter,
        dirichlet_alpha=1.0 / n_components,
        kmeanspp_min_dist=kmeanspp_min_dist,
        device=device,
        show_progress=show_progress,
        with_log_likelihoods=mop or not use_hellinger,
        sigma_atol=sigma_atol,
    )
    res["n_components"] = n_components
    res["kmeans_labels"] = res["labels"]
    n_components = len(cast(torch.Tensor, res["centroids"]))
    res["n_components_kept"] = n_components
    maxdist = max_sigma * float(res["sigma"])  # * np.sqrt(X.shape[1])
    if use_hellinger:
        centroids = cast(torch.Tensor, res["centroids"]).numpy(force=True)
        log_proportions = cast(torch.Tensor, res["log_proportions"]).numpy(force=True)
        labels = gmmdpc_hellinger(
            X=X,
            centroids=centroids,
            sigma=res["sigma"],
            log_proportions=log_proportions,
            maxdist=maxdist,
            workers=workers,
            hellinger_cutoff=hellinger_cutoff,
            hellinger_strong=hellinger_strong,
            hellinger_weak=hellinger_weak,
        )
    else:
        log_likelihoods = cast(torch.Tensor, res["log_likelihoods"]).numpy(force=True)
        labels = gmmdpc_classic(
            X=X,
            inliers=inliers,
            log_likelihoods=log_likelihoods,
            maxdist=maxdist,
            workers=workers,
            n_neighbors_search=n_neighbors_search,
        )

    if remove_clusters_smaller_than:
        if show_progress:
            logger.info("Clean...")
        labels = decrumb(labels, min_size=remove_clusters_smaller_than, in_place=True)
    res["labels"] = labels

    if mop:
        # use k-dtree dpc to mop up any remaining clusters...
        inliers = np.flatnonzero(inliers)
        discarded = np.flatnonzero(labels[inliers] < 0)
        log_likelihoods = cast(torch.Tensor, res["log_likelihoods"]).numpy(force=True)
        mop_res = density_peaks(
            Xi[discarded],
            density=log_likelihoods[discarded],
            outlier_radius=outlier_radius,
            outlier_neighbor_count=outlier_neighbor_count,
            radius_search=maxdist,
            workers=workers,
            remove_clusters_smaller_than=remove_clusters_smaller_than,
            n_neighbors_search=n_neighbors_search,
        )
        mopped = np.flatnonzero(mop_res["labels"] >= 0)
        mopped_labels = mop_res["labels"][mopped]
        mopped_labels += labels.max() + 1
        labels[inliers[discarded[mopped]]] = mopped_labels

    return res


def gmmdpc_classic(
    X,
    inliers,
    log_likelihoods,
    maxdist,
    workers=-1,
    n_neighbors_search=20,
):
    kdtree_res = density_peaks(
        X[inliers],
        density=log_likelihoods,
        outlier_radius=None,
        outlier_neighbor_count=None,
        radius_search=maxdist,
        workers=workers,
        remove_clusters_smaller_than=0,
        n_neighbors_search=n_neighbors_search,
    )
    labels = nearest_neighbor_assign(
        kdtree=kdtree_res["kdtree"],
        tree_labels=kdtree_res["labels"],
        X_other=X,
        radius_search=maxdist,
        workers=workers,
    )
    return labels


def gmmdpc_hellinger(
    X,
    centroids,
    sigma,
    log_proportions,
    maxdist: float,
    workers=-1,
    hellinger_cutoff=0.95,
    hellinger_strong=0.0,
    hellinger_weak=0.999,
) -> np.ndarray:
    """Density peaks clustering via isotropic GMM (Hellinger version)

    Spikes are assigned to isotropic Gaussian clusters by k-means. Then
    density peaks is carried out on centroids, where the notion of density
    for a centroid is its log proportion in the mixture.

    As in any DPC implementation, the key is the neighbor criterion. Here,
    components are neighbors if they meet certain criteria about Hellinger
    distance, which is like cosine distance for probability distributions.
    """
    logger.dartsortdebug("GMMDPC: Hellinger...")
    coo = sparse_iso_hellinger(
        centroids,
        sigma,
        hellinger_threshold=hellinger_cutoff,
    )
    nhdn = coo_nhdn(coo, log_proportions)
    assert nhdn.shape == (len(centroids),)
    n_components = len(centroids)
    ii = np.arange(len(nhdn))
    jj = nhdn

    if hellinger_strong:
        strong = np.flatnonzero(coo.data < hellinger_strong)
        ii = np.concatenate([ii, coo.coords[0][strong]])
        jj = np.concatenate([jj, coo.coords[1][strong]])
        order = np.argsort(ii, kind="stable")
        ii = ii[order]
        jj = jj[order]

    if hellinger_weak:
        disconnected = nhdn == ii
        disconnected = np.logical_and(
            disconnected,
            np.logical_not(np.isin(ii, nhdn[np.logical_not(disconnected)])),
        )
        disconnected = np.flatnonzero(disconnected)
    else:
        disconnected = np.array([], dtype=int)

    if hellinger_weak and disconnected.size:
        coo_weak = sparse_iso_hellinger(
            centroids[disconnected],
            sigma,
            hellinger_threshold=hellinger_weak,
        )
        coo_weak = coo_array(
            (coo_weak.data, (coo_weak.coords[1], coo_weak.coords[0])),
            shape=(disconnected.size, disconnected.size),
        )
        weak_nhdn = coo_nhdn(coo_weak, log_proportions[disconnected])
        jj[disconnected] = disconnected[weak_nhdn]

    _1 = np.ones((1,), dtype="float32")
    _1 = np.broadcast_to(_1, jj.shape)
    nhdn_coo = coo_array((_1, (ii, jj)), shape=(n_components, n_components))
    _, labels = connected_components(nhdn_coo)
    assert labels.shape == (n_components,)
    labels_padded = np.pad(labels, [(0, 1)], constant_values=-1)

    ckdt = KDTree(centroids)
    _, q = ckdt.query(X, workers=workers or 1, distance_upper_bound=maxdist)
    labels = labels_padded[q]
    return labels


# -- versions used in UHD projectss


def density_peaks_fancy(
    xyza,
    amps,
    sorting,
    motion,
    geom,
    sigma_local=5.0,
    sigma_regional=None,
    outlier_neighbor_count=10,
    outlier_radius=25.0,
    n_neighbors_search=20,
    radius_search=5.0,
    noise_density=0.0,
    remove_clusters_smaller_than=10,
    workers=-1,
    scales=(1.0, 1.0, 50.0),
    amp_log_c=5.0,
    sigma_local_low: float | None = None,
    sigma_regional_low: float | None = None,
    distance_dependent_noise_density=False,
    attach_density_feature=False,
    triage_quantile_per_cluster=0.0,
    ramp_triage_per_cluster=False,
    triage_quantile_before_clustering=0.0,
    amp_no_triaging_before_clustering=6.0,
    amp_no_triaging_after_clustering=8.0,
    use_y_triaging=False,
):
    z = xyza[:, 2]
    if motion is not None:
        z = motion.correct_s(sorting.times_seconds, z)
    z_not_reg = xyza[:, 2]
    ampfeat = scales[2] * np.log(amp_log_c + amps[:])
    res = _density_peaks_clustering_uhd_implementation(
        np.c_[scales[0] * xyza[:, 0], scales[1] * z, ampfeat],
        geom=geom,
        y=xyza[:, 1],
        z_not_reg=z_not_reg,
        use_y_triaging=use_y_triaging,
        sigma_local=sigma_local,
        sigma_local_low=sigma_local_low,
        sigma_regional=sigma_regional,
        sigma_regional_low=sigma_regional_low,
        n_neighbors_search=n_neighbors_search,
        radius_search=radius_search,
        remove_clusters_smaller_than=remove_clusters_smaller_than,
        noise_density=noise_density,
        triage_quantile_per_cluster=triage_quantile_per_cluster,
        ramp_triage_per_cluster=ramp_triage_per_cluster,
        triage_quantile_before_clustering=triage_quantile_before_clustering,
        amp_no_triaging_before_clustering=amp_no_triaging_before_clustering,
        amp_no_triaging_after_clustering=amp_no_triaging_after_clustering,
        distance_dependent_noise_density=distance_dependent_noise_density,
        outlier_radius=outlier_radius,
        outlier_neighbor_count=outlier_neighbor_count,
        scales=scales,
        log_c=amp_log_c,
        workers=workers,
        return_extra=attach_density_feature,
    )
    return res


def _density_peaks_clustering_uhd_implementation(
    X,
    geom=None,
    y=None,
    z_not_reg=None,
    use_y_triaging=False,
    l2_norm=None,
    kdtree=None,
    sigma_local=5.0,
    sigma_local_low=None,
    sigma_regional=None,
    sigma_regional_low=None,
    min_bin_size=1.0,
    outlier_neighbor_count=10,
    outlier_radius=25.0,
    n_neighbors_search=20,
    radius_search=5.0,
    noise_density=0.0,
    remove_clusters_smaller_than=10,
    remove_borders=False,
    border_search_radius=10.0,
    border_search_neighbors=3,
    workers=1,
    return_extra=False,
    triage_quantile_before_clustering=0.0,
    amp_no_triaging_before_clustering=6.0,
    ramp_triage_before_clustering=False,
    radius_triage_before_clustering=75.0,
    triage_quantile_per_cluster=0.0,
    amp_no_triaging_after_clustering=12.0,
    ramp_triage_per_cluster=False,
    distance_dependent_noise_density=False,
    amp_lowest_noise_density=8.0,
    min_distance_noise_density=0.0,
    min_distance_noise_density_10=200,
    max_noise_density=10.0,
    max_n_bins=128,
    scales=None,
    log_c=None,
):
    """
    if l2_norm is passed as argument, it will be used to compute density and nhdn
    """
    from scipy.stats import bernoulli
    # n = len(X)

    if ramp_triage_before_clustering and geom is not None:
        assert scales is not None
        assert log_c is not None
        inliers_first = np.ones(len(X)).astype("bool")
        idx_low_ptp = np.flatnonzero(
            X[:, 2] < scales[2] * np.log(log_c + amp_no_triaging_before_clustering)
        )
        distances_to_geom = (
            np.sqrt((X[idx_low_ptp, :2][None] - geom[:, None]) ** 2).sum(2)
        ).min(0)
        probabilities = (
            np.minimum(distances_to_geom, radius_triage_before_clustering)
            / radius_triage_before_clustering
        )
        which_to_discard = bernoulli.rvs(probabilities).astype("bool")
        inliers_first[idx_low_ptp[which_to_discard]] = False
        inliers_first = np.arange(len(X))[inliers_first]
    else:
        inliers_first = np.arange(len(X))

    n = len(inliers_first)
    if n <= 1:
        return dict(labels=np.full(X.shape[0], -1))

    if use_y_triaging and y is not None:
        inliers, _ = kdtree_inliers(
            np.c_[X[inliers_first], y[inliers_first]],
            kdtree=kdtree,
            n_neighbors=outlier_neighbor_count,
            distance_upper_bound=outlier_radius * np.sqrt(4 / 3),
            workers=workers,
        )
        _, kdtree = kdtree_inliers(
            X[inliers_first],
            kdtree=kdtree,
            n_neighbors=outlier_neighbor_count,
            distance_upper_bound=outlier_radius,
            workers=workers,
        )
    else:
        inliers, kdtree = kdtree_inliers(
            X[inliers_first],
            kdtree=kdtree,
            n_neighbors=outlier_neighbor_count,
            distance_upper_bound=outlier_radius,
            workers=workers,
        )

    if not inliers.sum() > 1:
        return dict(labels=np.full(X.shape[0], -1))

    # inliers = inliers_first[inliers]

    if isinstance(sigma_local, str) and sigma_local.startswith("rule_of_thumb"):
        factor = 1
        if "*" in sigma_local:
            factor = float(sigma_local.split("*")[1].strip())
        sigma_local = (
            1.06
            * factor
            * np.linalg.norm(np.std(X[inliers_first][inliers], axis=0))
            * np.power(inliers.sum(), -0.2)
        )
        if sigma_local <= 0:
            raise ValueError(
                f"rule of thumb problems. {sigma_local=} "
                f"{np.std(X[inliers_first][inliers], axis=0)=} "
                f"{len(X)=} {np.power(len(X), -0.2)=} "
                f"{X.shape=} {inliers_first.shape=} {inliers.sum()=} "
                f"{X[inliers_first][inliers].shape=}"
            )
        if sigma_regional == "rule_of_thumb":
            radius_search = radius_search * sigma_local
            sigma_regional = 10 * sigma_local
        else:
            radius_search = radius_search * sigma_local

    if l2_norm is None:
        do_ratio = sigma_regional is not None
        density = get_smoothed_density(
            X[inliers_first],
            inliers=inliers,
            sigma=sigma_local,
            sigma_low=sigma_local_low,
            min_bin_size=min_bin_size,
            max_n_bins=max_n_bins,
        )
        if density is None:
            return dict(labels=np.full(X.shape[0], -1))
        if do_ratio:
            reg_density = get_smoothed_density(
                X[inliers_first],
                inliers=inliers,
                sigma=sigma_regional,  # type: ignore # ty: ignore[x]
                sigma_low=sigma_regional_low,
                min_bin_size=min_bin_size,
            )
            density = density / reg_density
        density = np.nan_to_num(density)

        nhdn = nearest_higher_density_neighbor(
            kdtree,
            density,
            n_neighbors_search=n_neighbors_search,
            distance_upper_bound=radius_search,
            workers=workers,
        )
    else:
        # inliers don't matter here? Or should we still remove them?...
        # indices = np.full(l2_norm.shape[0], n)
        # l2_norm_inliers = l2_norm[inliers] #?? NO -> keep everything but only compute for inliers
        l2_norm = l2_norm[inliers_first][:, inliers_first]
        n = l2_norm.shape[0]
        indices = l2_norm.argsort()[:, : 1 + n_neighbors_search]
        distances = l2_norm[np.arange(n)[:, None], indices]
        assert distances.shape == (l2_norm.shape[0], 1 + n_neighbors_search)
        density = np.median(distances, axis=1)
        density_padded = np.pad(density, (0, 1), constant_values=np.inf)
        is_higher_density = density_padded[indices] >= density[:, None]
        distances[is_higher_density] = np.inf
        indices[is_higher_density] = n
        nhdn = indices[np.arange(n), distances.argmin(1)]
        del distances, indices

    if noise_density and l2_norm is None:
        nhdn[density <= noise_density] = n

    if distance_dependent_noise_density and noise_density is not None:
        assert scales is not None
        assert log_c is not None
        assert geom is not None
        dist = np.sqrt(
            (
                (
                    np.c_[X[inliers_first, 0], z_not_reg[inliers_first]][:, None]  # type: ignore
                    - geom[None]
                )
                ** 2
            ).sum(2)
        ).min(1)
        noise_density_dist = (
            np.minimum(
                np.maximum(dist - min_distance_noise_density, 0),
                min_distance_noise_density_10 - min_distance_noise_density,
            )
            * (max_noise_density - noise_density)
            / (min_distance_noise_density_10 - min_distance_noise_density)
            + noise_density
        )
        noise_density_dist[
            X[inliers_first, 2] > scales[2] * np.log(log_c + amp_lowest_noise_density)
        ] = 2
        nhdn[density <= noise_density_dist] = n
    if triage_quantile_before_clustering and l2_norm is None:
        assert scales is not None
        assert log_c is not None
        q = np.quantile(density, triage_quantile_before_clustering)
        idx_triaging = np.flatnonzero(
            np.logical_and(
                X[inliers_first, 2]
                < scales[2] * np.log(log_c + amp_no_triaging_before_clustering),
                density < q,
            )
        )
        nhdn[idx_triaging] = n

    nhdn = nhdn.astype(np.intc)
    has_nhdn = np.flatnonzero(nhdn < n).astype(np.intc)
    rc = (
        np.concatenate([nhdn[has_nhdn], has_nhdn]),
        np.concatenate([has_nhdn, nhdn[has_nhdn]]),
    )

    graph = coo_array((np.ones(2 * has_nhdn.size), rc), shape=(n, n))
    ncc, labels = connected_components(graph, directed=False)

    if remove_borders:
        labels = remove_border_points(
            labels,
            density,
            kdtree,
            search_radius=border_search_radius,
            n_neighbors_search=border_search_neighbors,
            workers=workers,
        )

    if remove_clusters_smaller_than:
        labels = decrumb(labels, min_size=remove_clusters_smaller_than)

    if triage_quantile_per_cluster:
        assert scales is not None
        assert log_c is not None
        amp_no_triaging_after_clustering = scales[2] * np.log(
            log_c + amp_no_triaging_after_clustering
        )
        for k in np.unique(labels[labels > -1]):
            idx_label = np.flatnonzero(labels == k)
            amp_vec = X[inliers_first][idx_label, 2]
            med_amp = np.median(amp_vec)
            if med_amp < amp_no_triaging_after_clustering:
                if ramp_triage_per_cluster:
                    triage_quantile_per_cluster = (
                        amp_no_triaging_after_clustering - med_amp
                    ) / amp_no_triaging_after_clustering
                # triage_quantile_unit = triage_quantile_per_cluster
                if l2_norm is None:
                    q = np.quantile(density[idx_label], triage_quantile_per_cluster)
                    spikes_to_remove = np.flatnonzero(
                        np.logical_and(
                            density[idx_label] < q,
                            amp_vec < amp_no_triaging_after_clustering,
                        )
                    )
                else:
                    q = np.quantile(density[idx_label], 1 - triage_quantile_per_cluster)
                    spikes_to_remove = np.flatnonzero(
                        np.logical_and(
                            density[idx_label] > q,
                            amp_vec < amp_no_triaging_after_clustering,
                        )
                    )

                labels[idx_label[spikes_to_remove]] = -1

    labels_all = np.full(len(X), -1)
    labels_all[inliers_first] = labels

    density_all = np.zeros(len(X))
    nhdn_all = np.full(len(X), n)
    density_all[inliers_first] = density
    nhdn_all[inliers_first] = nhdn

    return dict(density=density_all, nhdn=nhdn_all, labels=labels_all, kdtree=kdtree)
