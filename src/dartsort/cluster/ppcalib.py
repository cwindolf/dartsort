import linear_operator
import torch
from tqdm.auto import trange

from ..util.noise_util import EmbeddedNoise
from .stable_features import SpikeFeatures, SpikeNeighborhoods


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

    if active_W is None and M > 0:
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
        )
        state = ppca_m_step(
            **e, W_old=state["W"], mean_prior_pseudocount=mean_prior_pseudocount
        )

    state["nobs"] = nobs

    return state


def ppca_e_step(
    sp: SpikeFeatures,
    noise: EmbeddedNoise,
    neighborhoods: SpikeNeighborhoods,
    active_channels,
    active_mean,
    active_W=None,
    weights=None,
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
    e_u = e_ycu = e_uu = None
    if yes_pca:
        e_u = y.new_zeros((M,))
        e_ycu = y.new_zeros((rank, nc, M))
        e_uu = y.new_zeros((M, M))

    # helpful tensors to keep around
    if yes_pca:
        eye_M = torch.eye(M, device=y.device, dtype=y.dtype)

    unique_nids = torch.unique(sp.neighborhood_ids)
    for nid in unique_nids:
        # neighborhood channels logic
        (in_neighborhood,) = (sp.neighborhood_ids == nid).nonzero(as_tuple=True)
        n_neighb = in_neighborhood.numel()
        neighb_chans = neighborhoods.neighborhood_channels(nid)
        active_subset = torch.isin(active_channels, neighb_chans)
        have_missing = not active_subset.all()
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
        C_oo = noise.marginal_covariance(
            channels=neighb_chans, cache_prefix=cache_prefix, cache_key=nid
        )
        if yes_pca:
            W_o = active_W[:, active_subset].reshape(D_neighb, M)
        nu = active_mean[:, active_subset].reshape(D_neighb)
        if have_missing:
            C_mo = noise.offdiag_covariance(
                channels_left=missing_chans, channels_right=neighb_chans
            )
            tnu = active_mean[:, missing_chans].reshape(D - D_neighb)
            if yes_pca:
                W_m = active_W[:, missing_subset].reshape(D - D_neighb, M)
        assert C_oo.shape == (D_neighb, D_neighb)

        # actual data in neighborhood
        x = sp.features[in_neighborhood][:, active_subset].reshape(n_neighb, D_neighb)
        Cinvxc = torch.linalg.solve(C_oo, x - nu)

        # moments of embeddings
        # T is MxM, so don't woodbury it. it's the good one already.
        if yes_pca:
            T_inv = eye_M + linear_operator.solve(lhs=W_o.T, input=C_oo, rhs=W_o)
            T = torch.linalg.inv(T_inv)
            ubar = Cinvxc @ (W_o @ T)
            uubar = T + ubar[:, :, None] * ubar[:, None, :]

        # first data moment
        if have_missing:
            x_m = tnu + C_mo @ Cinvxc
            if yes_pca:
                x_m.add_(W_m @ ubar)

        # cross moment
        if yes_pca:
            e_xcu = (x - nu)[:, :, None] * ubar[:, None, :]
            if have_missing:
                CinvxcTCm = Cinvxc @ C_mo.T
                e_mxcu = CinvxcTCm[:, :, None] * ubar[:, None, :]
                e_mxcu.add_(uubar @ W_m.T)

        # take weighted averages
        if yes_pca:
            mean_ubar = torch.linalg.vecdot(w, ubar, dim=0)
            mean_uubar = torch.linalg.vecdot(w, uubar, dim=0)
        if have_missing:
            wx = torch.linalg.vecdot(w, x, dim=0)
            wx_m = torch.linalg.vecdot(w, x_m, dim=0)
            ybar = y.new_zeros((rank, nc))
            ybar[:, active_subset] = wx.view(rank, neighb_nc)
            ybar[:, missing_subset] = wx_m.view(rank, nc - neighb_nc)
        else:
            ybar = torch.linalg.vecdot(w, x, dim=0)
        if have_missing and yes_pca:
            wxcu = torch.linalg.vecdot(w, e_xcu, dim=0)
            wmxcu = torch.linalg.vecdot(w, e_mxcu)
            ycubar = y.new_zeros((rank, nc, M))
            ycubar[:, active_subset] = wxcu.view(rank, neighb_nc, M)
            ycubar[:, missing_subset] = wmxcu.view(rank, nc - neighb_nc, M)
        elif yes_pca:
            ycubar = torch.linalg.vecdot(w, e_xcu, dim=0)

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
    )


def ppca_m_step(
    e_y, e_u, e_ycu, e_uu, ess, W_old, mean_prior_pseudocount=10.0, rescale=True
):
    """Lightweight PPCA M step"""
    rank, nc, M = e_ycu.shape
    mu = e_y - W_old @ e_u
    if mean_prior_pseudocount:
        mu *= ess / (ess + mean_prior_pseudocount)
    if e_u is None:
        return dict(mu=mu, W=None)
    if rescale:
        sigma_u = e_uu - e_u[:, None] * e_u[None]
        scales = sigma_u.diagonal().sqrt()
        e_uu = e_uu / scales
        e_ycu = e_ycu / scales
    W = torch.linalg.solve(e_uu, e_ycu.view(rank * nc, M), left=False)
    if rescale:
        W.mul_(scales)
    W = W.view(rank, nc, M)
    return dict(mu=mu, W=W)


def initialize_mean(
    sp, active_channels, neighborhoods, weights=None, mean_prior_pseudocount=10.0
):
    ns, rank, nc = sp.features.shape
    weighted_sum = sp.features.new_zeros((rank, nc))
    nobs = sp.features.new_zeros((nc,))

    unique_nids = torch.unique(sp.neighborhood_ids)
    for nid in unique_nids:
        # neighborhood channels logic
        (in_neighborhood,) = (sp.neighborhood_ids == nid).nonzero(as_tuple=True)
        neighb_chans = neighborhoods.neighborhood_channels(nid)
        active_subset = torch.isin(active_channels, neighb_chans)
        (active_subset,) = active_subset.nonzero(as_tuple=True)

        w = weights[in_neighborhood]
        x = sp.features[:, :, active_subset]
        weighted_sum[:, active_subset] += torch.linalg.vecdot(w, x, dim=0)
        nobs[active_subset] += w.sum()

    mean = weighted_sum / (nobs + mean_prior_pseudocount)
    return mean, nobs


def get_nobs(sp, active_channels, neighborhoods, weights=None):
    ns, rank, nc = sp.features.shape
    nobs = sp.features.new_zeros((nc,))

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
