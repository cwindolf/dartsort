import warnings
from typing import Optional

import linear_operator
from linear_operator.operators import CholLinearOperator
import torch
import torch.nn.functional as F
from tqdm.auto import trange
from dataclasses import dataclass

from ..util.noise_util import EmbeddedNoise
from .stable_features import SpikeFeatures, SpikeNeighborhoods
from ..util import spiketorch

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
    mean_prior_pseudocount=0.0,
    show_progress=False,
    W_initialization="svd",
    normalize=False,
    em_converged_atol=0.1,
    prior_var=1.0,
    cache_direct=True,
):
    new_zeros = sp.features.new_zeros
    if active_W is not None:
        assert active_W.shape[2] == M
    n = len(sp)
    if n < M:
        raise ValueError(f"Too few samples {n=} for rank {M=}.")

    rank = noise.rank
    do_pca = M > 0
    D = rank * active_channels.numel()
    if weights is None:
        weights = sp.features.new_ones(len(sp))
    else:
        assert (weights > 0).all()
        assert torch.isfinite(weights).all()
    ess = weights.sum()
    assert torch.isfinite(ess)
    neighb_data = get_neighborhood_data(
        sp, neighborhoods, active_channels, rank, weights, D, noise, cache_prefix
    )
    any_missing = any(nd.have_missing for nd in neighb_data)
    cache_kw = {}
    if cache_direct:
        cache_kw = dict(
            cache_prefix="direct", cache_key=tuple(active_channels.tolist())
        )
    active_cov = noise.marginal_covariance(channels=active_channels, **cache_kw)
    active_cov_chol_factor = active_cov.cholesky().to_dense()

    scratch = None
    if do_pca:
        scratch_NM = sp.features.new_zeros((n, M))
        scratch_NMM = sp.features.new_zeros(n, M, M)
        scratch = (scratch_NM, scratch_NMM)

    if active_mean is None:
        active_mean, nobs = initialize_mean(
            sp,
            neighb_data,
            active_channels,
            weights=weights,
            mean_prior_pseudocount=mean_prior_pseudocount,
        )
    else:
        nobs = initialize_mean(
            sp,
            neighb_data,
            active_channels,
            weights=weights,
            mean_prior_pseudocount=mean_prior_pseudocount,
            nobs_only=True,
        )

    W_needs_initialization = active_W is None and M > 0
    if W_needs_initialization:
        if W_initialization in ("svd", "zeros"):
            active_W = new_zeros((*active_mean.shape, M))
        elif W_initialization == "random":
            active_W = new_zeros((*active_mean.shape, M))
            torch.randn(out=active_W, size=active_W.shape)
            W_needs_initialization = False
        else:
            assert False

    iters = trange(n_iter, desc="PPCA") if show_progress else range(n_iter)
    state = dict(mu=active_mean, W=active_W)
    for i in iters:
        e = ppca_e_step(
            sp=sp,
            noise=noise,
            ess=ess,
            neighb_data=neighb_data,
            active_channels=active_channels,
            active_mean=state["mu"],
            active_W=state["W"],
            weights=weights,
            active_cov_chol_factor=active_cov_chol_factor,
            normalize=normalize and not (W_needs_initialization and not i),
            return_yc=W_needs_initialization and not i,
            prior_var=prior_var,
            scratch=scratch,
        )
        old_state = state
        state = ppca_m_step(
            **e,
            M=M,
            ess=ess,
            active_cov_chol_factor=active_cov_chol_factor,
            mean_prior_pseudocount=mean_prior_pseudocount,
            noise=noise,
            active_channels=active_channels,
        )
        dmu = torch.abs(state["mu"] - old_state["mu"]).abs().max()
        dW = 0
        if state["W"] is not None:
            dW = torch.abs(state["W"] - old_state["W"]).abs().max()
        if not any_missing:
            break
        if max(dmu, dW) < em_converged_atol:
            break
        if show_progress:
            iters.set_description(f"PPCA[{dmu=:.2g}, {dW=:.2g}]")

    if normalize and any_missing and state["W"] is not None:
        _, _, state["W"], state["mu"] = embed(
            sp,
            noise,
            neighb_data,
            M,
            weights,
            state["W"],
            state["mu"],
            active_channels=active_channels,
            active_cov_chol_factor=active_cov_chol_factor,
            prior_var=prior_var,
            normalize=normalize,
            scratch=scratch,
        )

    state["nobs"] = nobs

    return state


def ppca_e_step(
    sp: SpikeFeatures,
    noise: EmbeddedNoise,
    neighb_data,
    active_channels,
    active_mean,
    ess,
    active_cov_chol_factor=None,
    return_yc=False,
    active_W=None,
    weights=None,
    normalize=True,
    prior_var=1.0,
    scratch=None,
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
    M = 0
    if yes_pca:
        rank_, nc_, M = active_W.shape
        assert (rank_, nc_) == (rank, nc)
    D = rank * nc

    # get normalized weights
    y = sp.features

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
    if yes_pca:
        full_ubar, full_uubar, active_W, active_mean = embed(
            sp,
            noise,
            neighb_data,
            M,
            weights,
            active_W,
            active_mean,
            ess=ess,
            active_cov_chol_factor=active_cov_chol_factor,
            active_channels=active_channels,
            prior_var=prior_var,
            normalize=normalize,
            scratch=scratch,
        )
    for nd in neighb_data:
        nu = active_mean[:, nd.active_subset].reshape(nd.D_neighb)
        if nd.have_missing:
            tnu = active_mean[:, nd.missing_subset].reshape(D - nd.D_neighb)

        if yes_pca:
            W_o = active_W[:, nd.active_subset].reshape(nd.D_neighb, M)
            if nd.have_missing and yes_pca:
                W_m = active_W[:, nd.missing_subset].reshape(D - nd.D_neighb, M)

        if yes_pca:
            ubar = full_ubar[nd.neighb_members]
            uubar = full_uubar[nd.neighb_members]

        # actual data in neighborhood
        xc = nd.x - nu

        # we need these ones everywhere
        Cooinvxc = nd.C_oo_chol.solve(xc.T).T

        # pca-centered data
        if yes_pca and nd.have_missing:
            CooinvWo = nd.C_oo_chol.solve(W_o)
            # xcc = torch.addmm(xc, ubar, W_o.T, alpha=-1)
            # Cooinvxcc = C_oochol.solve(xcc.T).T
            Cooinvxcc = Cooinvxc.addmm(ubar, CooinvWo.T, alpha=-1)
        else:
            Cooinvxcc = Cooinvxc

        # first data moment
        if nd.have_missing:
            xbar_m = torch.addmm(tnu, Cooinvxcc, nd.C_mo.T)
            if yes_pca:
                xbar_m.addmm_(ubar, W_m.T)

        # cross moment
        if yes_pca:
            e_xcu = xc[:, :, None] * ubar[:, None, :]
        if yes_pca and nd.have_missing:
            e_mxcu = (Cooinvxc @ nd.C_mo.T)[:, :, None] * ubar[:, None, :]
            # CmoCooinvWo = C_mo @ CooinvWo
            Wm_less_CmoCooinvWo = W_m.addmm(nd.C_mo, CooinvWo, beta=-1)
            shp = Wm_less_CmoCooinvWo.shape
            # e_mxcu += (uubar @ (W_m - CmoCooinvWo).T).mT
            Wm_less_CmoCooinvWo = Wm_less_CmoCooinvWo.unsqueeze(0)
            Wm_less_CmoCooinvWo = Wm_less_CmoCooinvWo.broadcast_to((len(uubar), *shp))
            # e_mxcu += uubar.mT @ Wm_less_CmoCooinvWo
            e_mxcu.baddbmm_(Wm_less_CmoCooinvWo, uubar)

        # take weighted averages
        if yes_pca:
            mean_ubar = nd.w_norm @ ubar
            mean_uubar = nd.w_norm @ uubar.view(nd.neighb_n_spikes, -1)
            mean_uubar = mean_uubar.view(uubar.shape[1:])

        wx = nd.w_norm @ nd.x
        if nd.have_missing:
            wxbar_m = nd.w_norm @ xbar_m
            ybar = y.new_zeros((rank, nc))
            ybar[:, nd.active_subset] = wx.view(rank, nd.neighb_nc)
            ybar[:, nd.missing_subset] = wxbar_m.view(rank, nc - nd.neighb_nc)
        else:
            ybar = wx.view(rank, nc)

        if yes_pca:
            wxcu = nd.w_norm @ e_xcu.view(nd.neighb_n_spikes, -1)
            wxcu = wxcu.view(e_xcu.shape[1:])
        if nd.have_missing and yes_pca:
            wmxcu = nd.w_norm @ e_mxcu.view(nd.neighb_n_spikes, -1)
            wmxcu = wmxcu.view(e_mxcu.shape[1:])
            ycubar = y.new_zeros((rank, nc, M))
            ycubar[:, nd.active_subset] = wxcu.view(rank, nd.neighb_nc, M)
            ycubar[:, nd.missing_subset] = wmxcu.view(rank, nc - nd.neighb_nc, M)
        elif yes_pca:
            ycubar = wxcu.view(rank, nc, M)

        # residual imputed
        if return_yc:
            if nd.have_missing:
                xc = xc.view(nd.neighb_n_spikes, rank, nd.neighb_nc).mT
                yc[nd.neighb_members[:, None], :, nd.active_subset[None, :]] = xc
                xbar_m -= tnu
                txc = xbar_m.view(nd.neighb_n_spikes, rank, nc - nd.neighb_nc).mT
                yc[nd.neighb_members[:, None], :, nd.missing_subset[None, :]] = txc
            else:
                yc[nd.neighb_members] = xc.view(nd.neighb_n_spikes, rank, nd.neighb_nc)

        # accumulate results
        e_y += ybar
        if yes_pca:
            e_u += mean_ubar
            e_uu += mean_uubar
            e_ycu += ycubar

    return dict(e_y=e_y, e_u=e_u, e_ycu=e_ycu, e_uu=e_uu, yc=yc, W_old=active_W)


def embed(
    sp,
    noise,
    neighb_data,
    M,
    weights,
    W,
    active_mean,
    active_channels,
    ess,
    active_cov_chol_factor=None,
    prior_var=1.0,
    normalize=True,
    scratch=None,
):
    N = len(sp)
    if scratch is not None:
        _ubar, _uubar = scratch
    else:
        _ubar = sp.features.new_zeros((N, M))
        # if not normalize:
        _uubar = sp.features.new_zeros(N, M, M)
    # _T = sp.features.new_zeros((N, M, M))
    eye_M = prior_var * torch.eye(M, device=sp.features.device, dtype=sp.features.dtype)

    for nd in neighb_data:
        nu = active_mean[:, nd.active_subset].reshape(nd.D_neighb)
        W_o = W[:, nd.active_subset].reshape(nd.D_neighb, M)
        xc = nd.x - nu

        # we need these ones everywhere
        Cooinvxc = nd.C_oo_chol.solve(xc.T).T

        # moments of embeddings
        T_inv = eye_M + W_o.T @ nd.C_oo_chol.solve(W_o)
        T = torch.linalg.inv(T_inv)
        ubar = Cooinvxc @ (W_o @ T)
        uubar = torch.baddbmm(T, ubar[:, :, None], ubar[:, None, :])

        _ubar[nd.neighb_members] = ubar
        _uubar[nd.neighb_members] = uubar

    if normalize:
        if active_cov_chol_factor is None:
            active_cov = noise.marginal_covariance(channels=active_channels)
            active_cov_chol_factor = torch.linalg.cholesky(active_cov).to_dense()
        Wflat = W.view(-1, M)

        # centering
        ess = weights.sum()
        weights = weights / ess
        um = weights @ _ubar
        _ubar -= um
        _uubar -= um[:, None] * um
        # active_mean = active_mean + W @ um

        # whitening. need to do a GEVP to start...
        S = (weights @ _uubar.view(N, M * M)).view(N, M, M)
        Dx, U = torch.linalg.eigh(S)
        Dx = Dx.flip(dims=(0,))
        U = U.flip(dims=(1,))
        U.mul_(sgn(U[0]))
        UDxrt = U * Dx.sqrt()
        rhs = Wflat @ UDxrt.T
        gevp_W_right = torch.linalg.solve_triangular(active_cov_chol_factor, rhs)
        gevp_W = gevp_W_right.T @ gevp_W_right
        # gevp_W = linear_operator.solve(lhs=rhs.T, input=active_cov, rhs=rhs)
        Dw, V = torch.linalg.eigh(gevp_W)
        Dw = Dw.flip(dims=(0,))
        V = V.flip(dims=(1,))
        V.mul_(sgn(V[0]))

        # this gives us transforms...
        W_tf = UDxrt @ V
        u_tf = (U / Dx.sqrt()) @ V

        # which we apply.
        W @= W_tf
        _ubar @= u_tf
        _uubar = torch.einsum("nij,ip,jq->npq", _uubar, u_tf, u_tf)
        active_mean.addmm_(W, um)

    return _ubar, _uubar, W, active_mean


@dataclass(kw_only=True, frozen=True, slots=True)
class NeighborhoodPPCAData:
    neighb_id: int
    neighb_nc: int
    neighb_n_spikes: int
    D_neighb: int
    have_missing: bool

    C_oo: linear_operator.LinearOperator
    C_oo_chol: CholLinearOperator
    w: torch.Tensor
    w_norm: torch.Tensor
    x: torch.Tensor
    neighb_members: torch.Tensor

    C_mo: Optional[torch.Tensor]
    active_subset: Optional[torch.Tensor]
    missing_subset: Optional[torch.Tensor]


def get_neighborhood_data(
    sp, neighborhoods, active_channels, rank, weights, D, noise, cache_prefix
):
    neighborhood_info, ns = neighborhoods.spike_neighborhoods(
        channels=active_channels,
        neighborhood_ids=sp.neighborhood_ids,
        min_coverage=0,
    )
    neighborhood_data = []
    ess = weights.sum()
    for nid, neighb_chans, neighb_members, _ in neighborhood_info:
        n_neighb = neighb_members.numel()

        # -- neighborhood channels
        neighb_valid = neighborhoods.valid_mask(nid)
        # subset of neighborhood's chans which are active
        # needs to be subset of full nchans set, not just the valid ones
        # neighb_subset = spiketorch.isin_sorted(neighb_chans, active_channels)
        neighb_subset = neighb_valid  # those are the same. tested by assert blo.
        # ok to restrict to valid below
        neighb_chans = neighb_chans[neighb_valid]
        assert spiketorch.isin_sorted(neighb_chans, active_channels).all()
        # subset of active chans which are in the neighborhood
        active_subset = spiketorch.isin_sorted(active_channels, neighb_chans)

        # -- missing channels
        have_missing = not active_subset.all()
        missing_subset = missing_chans = None
        if have_missing:
            (missing_subset,) = torch.logical_not(active_subset).nonzero(as_tuple=True)
            missing_chans = active_channels[missing_subset]
        (active_subset,) = active_subset.nonzero(as_tuple=True)
        neighb_nc = active_subset.numel()
        D_neighb = rank * neighb_nc

        # -- neighborhood data
        device = sp.features.device
        C_oo = noise.marginal_covariance(
            channels=neighb_chans,
            cache_prefix=cache_prefix,
            cache_key=nid,
            device=device,
        )
        assert C_oo.shape == (D_neighb, D_neighb)
        C_oo_chol = CholLinearOperator(C_oo.cholesky())
        w = weights[neighb_members]
        C_mo = None
        if have_missing:
            C_mo = noise.offdiag_covariance(
                channels_left=missing_chans,
                channels_right=neighb_chans,
            )
            C_mo = C_mo.to_dense().to(device)
        x = sp.features[neighb_members][:, :, neighb_subset]
        x = x.view(n_neighb, D_neighb)

        nd = NeighborhoodPPCAData(
            neighb_id=nid,
            neighb_nc=neighb_nc,
            neighb_n_spikes=n_neighb,
            D_neighb=D_neighb,
            have_missing=have_missing,
            C_oo=C_oo,
            C_oo_chol=C_oo_chol,
            w=w,
            w_norm=w / ess,
            x=x,
            neighb_members=neighb_members,
            C_mo=C_mo,
            active_subset=active_subset,
            missing_subset=missing_subset,
        )
        neighborhood_data.append(nd)

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
    active_cov_chol_factor=None,
    active_channels=None,
    mean_prior_pseudocount=0.0,
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
        if active_cov_chol_factor is None:
            active_cov = noise.marginal_covariance(active_channels)
            L = torch.linalg.cholesky(active_cov).to_dense()
        else:
            L = active_cov_chol_factor
        yc = yc.view(n, rank * nc)
        ycw = torch.linalg.solve_triangular(L, yc.T, upper=False).T
        assert ycw.shape == (n, rank * nc)

        try:
            u, s, v = torch.pca_lowrank(
                ycw, q=min(*ycw.shape, M + 10), center=False, niter=7
            )
        except Exception as e:
            err = ValueError(
                f"{torch.isfinite(yc).all()=} {yc.shape=}"
                f"{torch.isfinite(ycw).all()=} {ycw.shape=}"
            )
            raise err from e
        s = s[:M].square_().div(n - 1.0)
        s = F.relu(s - 1)
        s[s <= 0] = 1e-5
        W = v[:, :M].mul_(s.sqrt_() * sgn(v[0, :M]))
        # svd sign ambiguity
        assert W.shape == (rank * nc, M), f"{W.shape=} {(rank * nc, M)=} {yc.shape=}"

        W = L @ W
        W = W.view(rank, nc, M)

        return dict(mu=mu, W=W)

    if e_u is None:
        return dict(mu=mu, W=None)

    if rescale:
        scales = e_uu.diagonal().sqrt()
        e_uu = e_uu / (scales[:, None] * scales[None, :])
        e_ycu = e_ycu / scales

    W = torch.linalg.solve(e_uu, e_ycu.view(rank * nc, M), left=False)
    if rescale:
        W.mul_(scales)
    W = W.view(rank, nc, M)

    return dict(mu=mu, W=W)


def initialize_mean(
    sp,
    neighborhood_data,
    active_channels,
    weights=None,
    mean_prior_pseudocount=10.0,
    nobs_only=False,
):
    nc = active_channels.numel()
    ns, rank = sp.features.shape[:2]
    if not nobs_only:
        weighted_sum = sp.features.new_zeros((rank, nc))
    nobs = sp.features.new_zeros((nc,))
    if weights is None:
        weights = sp.features.new_ones((ns,))

    for nd in neighborhood_data:
        if not nobs_only:
            ws = nd.w @ nd.x.view(nd.neighb_n_spikes, -1)
            weighted_sum[:, nd.active_subset] += ws.view(rank, nd.neighb_nc)
        nobs[nd.active_subset] += nd.w.sum()

    if not nobs_only:
        mean = weighted_sum / (nobs + mean_prior_pseudocount)
        return mean, nobs
    return nobs


def sgn(x):
    s = torch.sign(x)
    s.mul_(2.0).add_(1.0).clamp_(-1.0, 1.0)
    return s
