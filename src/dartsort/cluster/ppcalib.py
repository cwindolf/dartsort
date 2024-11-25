import linear_operator
import numpy as np
import torch
import torch.nn.functional as F
from linear_operator import operators
from tqdm.auto import trange

from ..util.noise_util import EmbeddedNoise
from .stable_features import SpikeFeatures, SpikeNeighborhoods

vecdot = torch.linalg.vecdot


def ppca_em(
    sp: SpikeFeatures,
    noise: EmbeddedNoise,
    neighborhoods: SpikeNeighborhoods,
    active_channels,
    active_mean=None,
    active_W=None,
    weights=None,
    cache_prefix=None,
    M=1,
    n_iter=1,
    mean_prior_pseudocount=10.0,
    show_progress=False,
    W_initialization="zeros",
):
    new_zeros = sp.features.new_zeros
    if active_W is not None:
        assert active_W.shape[2] == M

    if active_mean is None:
        active_mean, nobs = initialize_mean(
            sp,
            active_channels,
            neighborhoods,
            weights=weights,
            mean_prior_pseudocount=mean_prior_pseudocount,
        )
    else:
        nobs = get_nobs(
            sp,
            active_channels,
            neighborhoods,
            weights=weights,
        )

    W_needs_initialization = active_W is None and M > 0
    if W_needs_initialization:
        if W_initialization == "zeros":
            active_W = new_zeros((*active_mean.shape, M))
        else:
            assert False

    iters = trange(n_iter, desc="PPCA") if show_progress else range(n_iter)
    state = dict(mu=active_mean, W=active_W)
    for i in iters:
        e = ppca_e_step(
            sp=sp,
            noise=noise,
            neighborhoods=neighborhoods,
            active_channels=active_channels,
            active_mean=state["mu"],
            active_W=state["W"],
            weights=weights,
            cache_prefix=cache_prefix,
            return_yc=W_needs_initialization and not i,
        )
        state = ppca_m_step(
            **e,
            M=M,
            mean_prior_pseudocount=mean_prior_pseudocount,
            noise=noise,
            active_channels=active_channels,
        )

    state["nobs"] = nobs

    return state


def ppca_e_step(
    sp: SpikeFeatures,
    noise: EmbeddedNoise,
    neighborhoods: SpikeNeighborhoods,
    active_channels,
    active_mean,
    return_yc=False,
    active_W=None,
    weights=None,
    center_u=True,
    cache_prefix=None,
):
    """E step in lightweight PPCA

    This is not the full mixture of PPCAs E step, just the inner PPCA
    E step. The only difference between the plain PPCA E step and the
    one inside a mixture is the use of responsibility weights, so this
    fn handles both cases.

    This computes:
        - weighted mean of expected complete data
            (Called e_y)
        - weighted mean of expected embeddings
            (Called e_u)
        - weighted mean of R_n, where R_n is the expected outer product
          of centered (complete) data and embeddings
            (Called e_ycu)
        - weighted mean of embeddings' second moment matrix
            (Called e_uu)

    This also handles the "rank 0 case" where we're just kinda imputing
    according to the noise covariance.
    """
    n = len(sp)
    nc = len(active_channels)
    rank, nc_ = active_mean.shape
    assert nc_ == nc
    no_pca = active_W is None
    yes_pca = not no_pca
    if yes_pca:
        rank_, nc_, M = active_W.shape
        assert (rank_, nc_) == (rank, nc)
    D = rank * nc

    # get normalized weights
    y = sp.features
    if weights is None:
        ess = n
        weights = y.new_full((n,), 1.0 / n)
    else:
        assert weights.shape == (n,)
        ess = weights.sum()
        weights = weights / ess

    # we will build our outputs by iterating over the unique
    # neighborhoods and adding weighted sums of moments in each
    e_y = y.new_zeros((rank, nc))
    yc = e_u = e_ycu = e_uu = None
    if return_yc:
        yc = y.new_zeros((n, rank, nc))
    if yes_pca:
        e_u = y.new_zeros((M,))
        e_ycu = y.new_zeros((rank, nc, M))
        e_uu = y.new_zeros((M, M))

    # helpful tensors to keep around

    neighb_data = get_neighborhood_data(
        sp,
        neighborhoods,
        noise,
        active_W,
        active_mean,
        active_channels,
        rank,
        weights,
        cache_prefix,
        yes_pca,
        M,
        D,
    )
    if yes_pca:
        full_ubar, full_uubar, full_T, active_W = embed(
            sp, neighb_data, M, weights, active_W, normalize=False
        )
    for ndata in neighb_data:
        in_neighborhood = ndata["in_neighborhood"]
        neighb_subset = ndata["neighb_subset"]
        active_subset = ndata["active_subset"]
        missing_subset = ndata["missing_subset"]
        missing_chans = ndata["missing_chans"]
        n_neighb = ndata["n_neighb"]
        neighb_nc = ndata["neighb_nc"]
        D_neighb = ndata["D_neighb"]
        C_oo = ndata["C_oo"]
        C_mo = ndata["C_mo"]
        nu = ndata["nu"]
        tnu = ndata["tnu"]
        w = ndata["w0"]
        w_ = ndata["w1"]
        w__ = ndata["w2"]
        have_missing = ndata["have_missing"]
        W_o = active_W[:, active_subset].reshape(D_neighb, M)
        if have_missing and yes_pca:
            W_m = active_W[:, missing_subset].reshape(D - D_neighb, M)

        if yes_pca:
            ubar = full_ubar[in_neighborhood]
            uubar = full_uubar[in_neighborhood]
            # T = full_T[in_neighborhood]

        # actual data in neighborhood
        x = sp.features[in_neighborhood][:, :, neighb_subset]
        x = x.reshape(n_neighb, D_neighb)
        xc = x - nu

        # we need these ones everywhere
        Cooinvxc = C_oo.solve(xc.T).T

        # pca-centered data
        Cooinvxcc = Cooinvxc
        if yes_pca:
            xcc = torch.addmm(xc, ubar, W_o.T, alpha=-1)
            Cooinvxcc = C_oo.solve(xcc.T).T

        # first data moment
        if have_missing:
            xbar_m = torch.addmm(tnu, Cooinvxcc, C_mo.T)
            if yes_pca:
                xbar_m.addmm_(ubar, W_m.T)

        # cross moment
        if yes_pca:
            e_xcu = (x - nu)[:, :, None] * ubar[:, None, :]
        if yes_pca and have_missing:
            e_mxcu = (Cooinvxc @ C_mo.T)[:, :, None] * ubar[:, None, :]
            CmoCooinvWo = linear_operator.solve(lhs=C_mo, input=C_oo, rhs=W_o)
            e_mxcu += (uubar @ (W_m - CmoCooinvWo).T).mT

        # take weighted averages
        if yes_pca:
            mean_ubar = torch.linalg.vecdot(w_, ubar, dim=0)
            mean_uubar = torch.linalg.vecdot(w__, uubar, dim=0)

        if have_missing:
            wx = torch.linalg.vecdot(w_, x, dim=0)
            wxbar_m = torch.linalg.vecdot(w_, xbar_m, dim=0)
            ybar = y.new_zeros((rank, nc))
            ybar[:, active_subset] = wx.view(rank, neighb_nc)
            ybar[:, missing_subset] = wxbar_m.view(rank, nc - neighb_nc)
        else:
            ybar = torch.linalg.vecdot(w_, x, dim=0).view(rank, nc)

        if have_missing and yes_pca:
            wxcu = torch.linalg.vecdot(w__, e_xcu, dim=0)
            wmxcu = torch.linalg.vecdot(w__, e_mxcu, dim=0)
            ycubar = y.new_zeros((rank, nc, M))
            ycubar[:, active_subset] = wxcu.view(rank, neighb_nc, M)
            ycubar[:, missing_subset] = wmxcu.view(rank, nc - neighb_nc, M)
        elif yes_pca:
            ycubar = torch.linalg.vecdot(w__, e_xcu, dim=0).view(rank, nc, M)

        # residual imputed
        if return_yc:
            if have_missing:
                xc = xc.view(n_neighb, rank, neighb_nc).mT
                yc[in_neighborhood[:, None], :, active_subset[None, :]] = xc
                xbar_m -= tnu
                txc = xbar_m.view(n_neighb, rank, nc - neighb_nc).mT
                yc[in_neighborhood[:, None], :, missing_subset[None, :]] = txc
            else:
                yc[in_neighborhood] = xc.view(n_neighb, rank, neighb_nc)

        # accumulate results
        e_y += ybar
        if yes_pca:
            e_u += mean_ubar
            e_uu += mean_uubar
            e_ycu += ycubar

    return dict(
        e_y=e_y,
        e_u=e_u,
        e_ycu=e_ycu,
        e_uu=e_uu,
        ess=ess,
        yc=yc,
        W_old=active_W,
    )


def embed(sp, neighb_data, M, weights, W, normalize=True):
    N = len(sp)
    _ubar = sp.features.new_zeros((N, M))
    if not normalize:
        _uubar = sp.features.new_zeros(N, M, M)
    _T = sp.features.new_zeros((N, M, M))
    eye_M = torch.eye(M, device=sp.features.device, dtype=sp.features.dtype)

    for ndata in neighb_data:
        in_neighborhood = ndata["in_neighborhood"]
        neighb_subset = ndata["neighb_subset"]
        active_subset = ndata["active_subset"]
        n_neighb = ndata["n_neighb"]
        D_neighb = ndata["D_neighb"]
        C_oo = ndata["C_oo"]
        nu = ndata["nu"]
        W_o = W[:, active_subset].reshape(D_neighb, M)

        x = sp.features[in_neighborhood][:, :, neighb_subset]
        x = x.reshape(n_neighb, D_neighb)
        xc = x - nu

        # we need these ones everywhere
        Cooinvxc = C_oo.solve(xc.T).T

        # moments of embeddings
        # T is MxM and we cache C_oo's Cholesky, so this is the quick one.
        T_inv = eye_M + linear_operator.solve(lhs=W_o.T, input=C_oo, rhs=W_o)
        T = torch.linalg.inv(T_inv)
        ubar = Cooinvxc @ (W_o @ T)
        uubar = ubar[:, :, None] * ubar[:, None, :]
        uubar.add_(T)

        _ubar[in_neighborhood] = ubar
        if not normalize:
            _uubar[in_neighborhood] = uubar
        _T[in_neighborhood] = T

    if normalize:
        um = vecdot(weights[:, None], _ubar, dim=0)
        _ubar -= um
        # _uubar = -= um[:, None] * um
        us = vecdot(weights[:, None], _T.diagonal(dim1=-2, dim2=-1).sqrt(), dim=0)
        _T.div_(us[:, None] * us[None, :])
        _uubar = torch.baddbmm(_T, _ubar[:, :, None], _ubar[:, None])
        W = W / us

    return _ubar, _uubar, _T, W


def get_neighborhood_data(
    sp,
    neighborhoods,
    noise,
    active_W,
    active_mean,
    active_channels,
    rank,
    weights,
    cache_prefix,
    yes_pca,
    M,
    D,
):
    neighborhood_data = []

    unique_nids = torch.unique(sp.neighborhood_ids)
    for nid in unique_nids:
        # neighborhood channels logic
        (in_neighborhood,) = (sp.neighborhood_ids == nid).nonzero(as_tuple=True)
        n_neighb = in_neighborhood.numel()
        neighb_chans = neighborhoods.neighborhoods[nid]
        active_subset = torch.isin(active_channels, neighb_chans)
        neighb_subset = torch.isin(neighb_chans, active_channels)
        assert torch.equal(neighb_chans[neighb_subset], active_channels[active_subset])
        neighb_chans = neighb_chans[neighb_chans < neighborhoods.n_channels]
        have_missing = not active_subset.all()
        missing_subset = missing_chans = None
        if have_missing:
            (missing_subset,) = torch.logical_not(active_subset).nonzero(as_tuple=True)
            missing_chans = active_channels[missing_subset]
        (active_subset,) = active_subset.nonzero(as_tuple=True)
        neighb_nc = active_subset.numel()
        D_neighb = rank * neighb_nc
        if not have_missing:
            assert D_neighb == D

        # neighborhood's known quantities
        w = weights[in_neighborhood]
        w_ = w[:, None]
        w__ = w_[:, None]
        C_oo = noise.marginal_covariance(
            channels=neighb_chans, cache_prefix=cache_prefix, cache_key=nid
        )
        nu = active_mean[:, active_subset].reshape(D_neighb)
        tnu = C_mo = W_m = None
        if have_missing:
            C_mo = noise.offdiag_covariance(
                channels_left=missing_chans, channels_right=neighb_chans
            )
            # ZeroLinearOperators sometims behave weirdly?
            C_mo = C_mo.to_dense()
            tnu = active_mean[:, missing_subset].reshape(D - D_neighb)
        assert C_oo.shape == (D_neighb, D_neighb)

        neighborhood_data.append(
            dict(
                nid=nid,
                have_missing=have_missing,
                in_neighborhood=in_neighborhood,
                n_neighb=n_neighb,
                neighb_chans=neighb_chans,
                active_subset=active_subset,
                neighb_subset=neighb_subset,
                missing_subset=missing_subset,
                missing_chans=missing_chans,
                neighb_nc=neighb_nc,
                D_neighb=D_neighb,
                w0=w,
                w1=w_,
                w2=w__,
                nu=nu,
                tnu=tnu,
                C_oo=C_oo,
                C_mo=C_mo,
            )
        )

    return neighborhood_data


def ppca_m_step(
    e_y,
    e_u,
    e_ycu,
    e_uu,
    ess,
    W_old,
    yc=None,
    M=0,
    noise=None,
    active_cov=None,
    active_channels=None,
    mean_prior_pseudocount=10.0,
    rescale=False,
):
    """Lightweight PPCA M step"""
    rank, nc = e_y.shape

    # update mean
    mu = e_y
    if e_u is not None:
        mu -= W_old @ e_u
    if mean_prior_pseudocount:
        mu *= ess / (ess + mean_prior_pseudocount)

    # initialize W via SVD of whitened residual
    if yc is not None and M:
        n = len(yc)
        if active_cov is None:
            active_cov = noise.marginal_covariance(active_channels)
        L = torch.linalg.cholesky(active_cov)
        yc = yc.view(n, rank * nc)
        yc = L.solve(yc.T).T
        assert yc.shape == (n, rank * nc)

        u, s, v = torch.pca_lowrank(yc, q=min(*yc.shape, M + 10), center=False, niter=7)
        s = s[:M].square_().div(n - 1.0)
        s = F.relu(s - 1)
        s[s <= 0] = 1e-5
        W = v[:, :M].mul_(s.sqrt_())
        assert W.shape == (rank * nc, M)

        W = L @ W
        W = W.view(rank, nc, M)

        # # update mean
        # mu = e_y
        # if e_u is not None:
        #     mu -= W @ e_u
        # if mean_prior_pseudocount:
        #     mu *= ess / (ess + mean_prior_pseudocount)

        return dict(mu=mu, W=W)

    if e_u is None:
        return dict(mu=mu, W=None)

    if rescale:
        # sigma_u = e_uu - e_u[:, None] * e_u[None]
        scales = e_uu.diagonal().sqrt()
        e_uu = e_uu / (scales[:, None] * scales[None, :])
        e_ycu = e_ycu / scales

    W = torch.linalg.solve(e_uu, e_ycu.view(rank * nc, M), left=False)

    if rescale:
        W.mul_(scales)

    W = W.view(rank, nc, M)

    # # update mean
    # mu = e_y
    # if e_u is not None:
    #     mu -= W @ e_u
    # if mean_prior_pseudocount:
    #     mu *= ess / (ess + mean_prior_pseudocount)

    return dict(mu=mu, W=W)


def initialize_mean(
    sp, active_channels, neighborhoods, weights=None, mean_prior_pseudocount=10.0
):
    nc = active_channels.numel()
    ns, rank = sp.features.shape[:2]
    weighted_sum = sp.features.new_zeros((rank, nc))
    nobs = sp.features.new_zeros((nc,))
    if weights is None:
        weights = sp.features.new_ones((ns,))

    unique_nids = torch.unique(sp.neighborhood_ids)
    for nid in unique_nids:
        # neighborhood channels logic
        (in_neighborhood,) = (sp.neighborhood_ids == nid).nonzero(as_tuple=True)
        neighb_chans = neighborhoods.neighborhoods[nid]
        active_subset = torch.isin(active_channels, neighb_chans)
        neighb_subset = torch.isin(neighb_chans, active_channels)
        assert torch.equal(neighb_chans[neighb_subset], active_channels[active_subset])
        (active_subset,) = active_subset.nonzero(as_tuple=True)

        w = weights[in_neighborhood, None, None]
        x = sp.features[in_neighborhood][:, :, neighb_subset]
        weighted_sum[:, active_subset] += torch.linalg.vecdot(w, x, dim=0)
        nobs[active_subset] += w.sum()

    mean = weighted_sum / (nobs + mean_prior_pseudocount)
    return mean, nobs


def get_nobs(sp, active_channels, neighborhoods, weights=None):
    nc = active_channels.numel()
    ns, rank = sp.features.shape[:2]
    nobs = sp.features.new_zeros((nc,))
    if weights is None:
        weights = sp.features.new_ones((ns,))

    unique_nids = torch.unique(sp.neighborhood_ids)
    for nid in unique_nids:
        # neighborhood channels logic
        (in_neighborhood,) = (sp.neighborhood_ids == nid).nonzero(as_tuple=True)
        neighb_chans = neighborhoods.neighborhood_channels(nid)
        active_subset = torch.isin(active_channels, neighb_chans)
        (active_subset,) = active_subset.nonzero(as_tuple=True)

        w = weights[in_neighborhood]
        nobs[active_subset] += w.sum()

    return nobs
