from dataclasses import dataclass, fields

import numpy as np
import torch
from linear_operator.operators import DenseLinearOperator

from ...util import spiketorch
from ...util.logging_util import DARTSORTDEBUG, DARTSORTVERBOSE, get_logger

log2pi = torch.log(torch.tensor(2 * np.pi))
_1 = torch.tensor(1.0)
logger = get_logger(__name__)
noise_eps = torch.tensor(1e-3)


tem_add_at_ = spiketorch.torch_add_at_
# tem_add_at_ = spiketorch.cupy_add_at_
# tem_add_at_ = spiketorch.try_cupy_add_at_


ds_verbose = logger.isEnabledFor(DARTSORTVERBOSE)
FRZ = not DARTSORTDEBUG
_debug = logger.isEnabledFor(DARTSORTVERBOSE)


def __debug_init__(self):
    msg = "-" * 40 + " " + self.__class__.__name__ + "\n"
    res = {}
    for f in fields(self):
        v = getattr(self, f.name)
        if torch.is_tensor(v):
            res[f.name] = v.isnan().any().item()
            msg += f" - {f.name}: {v.shape} min={v.min().item():g} mean={v.to(torch.float).mean().item():g} max={v.max().item():g} sum={v.sum().item():g}\n"
    if any(res.values()):
        raise ValueError(f"NaNs in {self.__class__.__name__}: {res}")
    msg += "//" + "-" * 38
    logger.dartsortverbose("->\n" + msg)


@dataclass(slots=FRZ, kw_only=True, frozen=FRZ, eq=False, repr=False)
class TEStepResult:
    obs_elbo: torch.Tensor
    noise_N: torch.Tensor
    N: torch.Tensor
    m: torch.Tensor

    R: torch.Tensor | None = None
    U: torch.Tensor | None = None

    kl: torch.Tensor | None = None

    hard_labels: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    count: int | None = None

    # if ds_verbose:
    #     __post_init__ = __debug_init__


@dataclass(slots=FRZ, kw_only=True, frozen=FRZ, eq=False, repr=False)
class TEBatchResult:
    indices: slice | torch.Tensor
    candidates: torch.Tensor
    obs_elbo: torch.Tensor
    noise_N: torch.Tensor
    N: torch.Tensor
    Nlut: torch.Tensor
    m: torch.Tensor

    R: torch.Tensor | None = None
    Ulut: torch.Tensor | None = None

    ddlogpi: torch.Tensor | None = None
    ddlognoisep: torch.Tensor | None = None
    ddm: torch.Tensor | None = None
    ddW: torch.Tensor | None = None

    ncc: torch.Tensor | None = None
    dkl: torch.Tensor | None = None

    noise_lls: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    hard_labels: torch.Tensor | None = None
    invquad: torch.Tensor | None = None
    edata: dict | None = None
    origcandidates: torch.Tensor | None = None

    # if ds_verbose:
    #     __post_init__ = __debug_init__


def _te_batch_e(
    n_units,
    n_candidates,
    noise_log_prop,
    candidates,
    vcand_ii,
    vcand_jj,
    whitenedx,
    whitenednu,
    nobs,
    vnobs,
    obs_logdets,
    Coo_logdet,
    log_proportions,
    noise_lls=None,
    noise_log_priors=None,
    wburyroot=None,
    with_kl=False,
    with_probs=False,
    with_invquad=False,
):
    """This is the "E step" within the E step."""
    pinobs = log2pi * nobs
    pivnobs = log2pi * vnobs

    # marginal noise log likelihoods, if needed
    nlls = noise_lls
    if noise_lls is None:
        inv_quad = whitenedx.square().sum(dim=1)
        nlls = inv_quad.add_(Coo_logdet).add_(pinobs).mul_(-0.5)
        if noise_log_priors is not None:
            nlls -= noise_log_priors
    # noise_log_prop changes, so can't be baked in
    noise_lls = nlls + (noise_log_prop + noise_eps)

    # observed log likelihoods
    whitenedx = whitenedx[vcand_ii]
    inv_quad = woodbury_inv_quad(
        whitenedx, whitenednu, wburyroot=wburyroot, flat=True, ow_wnu=True
    )
    invquad = inv_quad.clone() if with_invquad else None
    del whitenednu
    lls_unnorm = inv_quad.add_(obs_logdets).add_(pivnobs).mul_(-0.5)
    if with_kl:
        lls = lls_unnorm + log_proportions
    else:
        lls = lls_unnorm.add_(log_proportions)

    # cvalid = candidates >= 0
    # lls = torch.where(cvalid, lls, -torch.inf)
    # lls[candidates < 0] = -torch.inf
    _lls_dense = lls.new_full(candidates.shape, -torch.inf)
    _lls_dense[vcand_ii, vcand_jj] = lls
    lls = _lls_dense

    # -- update K_ns
    # toplls, topinds = lls.sort(dim=1, descending=True)
    # toplls = toplls[:, :n_candidates]
    # topinds = topinds[:, :n_candidates]
    all_lls = lls.new_empty((lls.shape[0], n_candidates + 1))
    all_inds = torch.empty(
        (lls.shape[0], n_candidates), dtype=torch.long, device=lls.device
    )
    topk_out = (all_lls[:, :n_candidates], all_inds)
    toplls, topinds = torch.topk(lls, n_candidates, dim=1, out=topk_out)

    # -- compute Q
    # all_lls = torch.concatenate((toplls, noise_lls.unsqueeze(1)), dim=1)
    all_lls[:, -1] = noise_lls
    Q = torch.softmax(all_lls, dim=1)
    new_candidates = candidates.take_along_dim(topinds, 1)
    if with_invquad:
        invquad_dense = lls.new_full(candidates.shape, torch.nan)
        invquad_dense[vcand_ii, vcand_jj] = invquad
        invquad = invquad_dense.take_along_dim(topinds, 1)
    if _debug and not ((new_candidates >= 0).sum(1) >= lls.isfinite().sum(1).clamp_(max=n_candidates)).all():
        (bad_ix,) = torch.nonzero((new_candidates < 0).any(dim=1).cpu(), as_tuple=True)
        raise ValueError(
            f"Bad candidates {lls=} {lls[bad_ix]=} {toplls[bad_ix]=} {topinds[bad_ix]=}"
        )

    ncc = dkl = None
    if with_kl:
        # todo: this is probably broken after the candidate masking was
        # implemented. leaving it for now because unused.
        cvalid = cvalid.to(torch.float)
        ncc = Q.new_zeros((n_units, n_units))
        tem_add_at_(
            ncc,
            (new_candidates[:, None, :1], candidates[:, :, None]),
            cvalid[:, :, None],
        )
        dkl = Q.new_zeros((n_units, n_units))
        top_lls_unnorm = lls_unnorm.take_along_dim(topinds[:, :1], dim=1)
        tem_add_at_(
            dkl,
            (new_candidates[:, :1], candidates),
            (top_lls_unnorm - lls_unnorm) * cvalid,
        )

    return dict(
        candidates=new_candidates,
        Q=Q,
        log_liks=all_lls,
        ncc=ncc,
        dkl=dkl,
        noise_lls=nlls,
        probs=toplls if with_probs else None,
        invquad=invquad,
    )


def _te_batch_m_counts(n_units, nlut, candidates, lut_ixs, Q):
    """Part 1/2 of the M step within the E step"""
    Q_ = Q[:, :-1]
    assert Q_.shape == candidates.shape
    Q_ = Q_.reshape(-1)
    candidates_pos = torch.where(candidates < 0, n_units, candidates)

    N = Q.new_zeros(n_units + 1)
    N.scatter_add_(dim=0, index=candidates_pos.view(-1), src=Q_)

    Nlut = Q.new_zeros(nlut + 1)
    Nlut.scatter_add_(dim=0, index=lut_ixs.view(-1), src=Q_)

    noise_N = Q[:, -1].sum()

    return noise_N, N, Nlut, candidates_pos


def _te_batch_m_rank0(
    rank,
    n_units,
    nlut,
    nc,
    nc_obs,
    nc_miss,
    # common args
    lut_ixs,
    candidates,
    candidates_pos,
    obs_ix,
    miss_ix,
    Q,
    N,
    x,
    Cmo_Cooinv_x,
    Cmo_Cooinv_nu,
):
    """Rank (M) 0 case of part 2/2 of the M step within the E step"""
    del nlut, lut_ixs
    N_denom = torch.where(N == 0, 1.0, N)

    Qn = Q[:, :-1] / N_denom[candidates_pos]
    n, C = Qn.shape

    m_full = Q.new_zeros((n, C, rank, nc + 1))
    m = Qn.new_zeros((n_units + 1, rank, nc))

    mm = torch.subtract(Cmo_Cooinv_nu, Cmo_Cooinv_x[:, None], out=Cmo_Cooinv_nu)

    src = x.view(n, rank, nc_obs)[:, None].broadcast_to(n, C, rank, nc_obs)
    ix = obs_ix[:, None, None, :].broadcast_to(src.shape)
    m_full.scatter_(dim=3, index=ix, src=src)
    src = mm.view(n, C, rank, nc_miss)
    ix = miss_ix[:, None, None, :].broadcast_to(src.shape)
    m_full.scatter_add_(dim=3, index=ix, src=src)

    Qmf = m_full[..., :nc].mul_(Qn[:, :, None, None]).view(-1, *m.shape[1:])
    ix = candidates_pos.view(-1)[:, None, None].broadcast_to(Qmf.shape)
    m.scatter_add_(dim=0, index=ix, src=Qmf)
    m = m[:n_units]

    return dict(m=m)


def _te_batch_m_ppca(
    rank,
    n_units,
    nlut,
    nc,
    nc_obs,
    nc_miss,
    # common args
    lut_ixs,
    candidates,
    candidates_pos,
    obs_ix,
    miss_ix,
    Q,
    N,
    Nlut,
    x,
    nu,
    Cmo_Cooinv_x,
    Cmo_Cooinv_nu,
    # M>0 only args
    inv_cap,
    inv_cap_Wobs_Cooinv,
    Cmo_Cooinv_WobsT,
    Wobs,
):
    """Rank (M) >0 case of part 2/2 of the M step within the E step"""
    N_denom = torch.where(N == 0, 1.0, N)
    Nlut_denom = torch.where(Nlut == 0, 1.0, Nlut)

    Qn = Q[:, :-1] / N_denom[candidates_pos]
    Qnlut = Q[:, :-1] / Nlut_denom[lut_ixs]
    n, C = Qn.shape
    M = inv_cap.shape[-1]

    # U = Q.new_zeros((n_units, M, M))
    R = Q.new_zeros((n_units + 1, M, rank, nc))
    m = Q.new_zeros((n_units + 1, rank, nc))
    Ulut = Q.new_zeros((nlut + 1, M, M))

    R_full = Q.new_zeros((n, C, M, rank, nc + 1))
    m_full = Q.new_zeros((n, C, rank, nc + 1))

    xc = torch.sub(x[:, None], nu, out=nu)
    del nu
    Cmo_Cooinv_xc = torch.sub(Cmo_Cooinv_x[:, None], Cmo_Cooinv_nu, out=Cmo_Cooinv_nu)
    del Cmo_Cooinv_nu

    ubar = torch.einsum("ncpj,ncj->ncp", inv_cap_Wobs_Cooinv, xc)
    Euu = (ubar.view(n, C, M, 1) * ubar.view(n, C, 1, M)).add_(inv_cap)

    WobsT_ubar = torch.einsum("ncpk,ncp->nck", Wobs, ubar)
    Cmo_Cooinv_WobsT_ubar = torch.einsum("nckp,ncp->nck", Cmo_Cooinv_WobsT, ubar)

    m_missing = torch.sub(
        Cmo_Cooinv_xc, Cmo_Cooinv_WobsT_ubar, out=Cmo_Cooinv_WobsT_ubar
    )
    del Cmo_Cooinv_WobsT_ubar
    m_observed = torch.sub(x[:, None], WobsT_ubar, out=WobsT_ubar)
    del WobsT_ubar

    src = m_observed.view(n, C, rank, nc_obs)
    ix = obs_ix[:, None, None, :].broadcast_to(src.shape)
    m_full.scatter_(dim=3, index=ix, src=src)
    src = m_missing.view(n, C, rank, nc_miss)
    ix = miss_ix[:, None, None, :].broadcast_to(src.shape)
    m_full.scatter_add_(dim=3, index=ix, src=src)

    R_observed = ubar[:, :, :, None] * xc[:, :, None, :]
    R_missing = ubar[:, :, :, None] * Cmo_Cooinv_xc[:, :, None, :]

    src = R_observed.view(n, C, M, rank, nc_obs)
    ix = obs_ix[:, None, None, None, :].broadcast_to(src.shape)
    R_full.scatter_(dim=4, index=ix, src=src)
    src = R_missing.view(n, C, M, rank, nc_miss)
    ix = miss_ix[:, None, None, None, :].broadcast_to(src.shape)
    R_full.scatter_add_(dim=4, index=ix, src=src)

    QRf = R_full[..., :nc].mul_(Qn[:, :, None, None, None]).view(-1, *R.shape[1:])
    del R_full
    Qmf = m_full[..., :nc].mul_(Qn[:, :, None, None]).view(-1, *m.shape[1:])
    del m_full

    ix = candidates_pos.view(-1)[:, None, None, None].broadcast_to(QRf.shape)
    R.scatter_add_(dim=0, index=ix, src=QRf)
    R = R[:n_units]

    ix = candidates_pos.view(-1)[:, None, None].broadcast_to(Qmf.shape)
    m.scatter_add_(dim=0, index=ix, src=Qmf)
    m = m[:n_units]

    assert lut_ixs.shape == (n, C)
    QUlut = Euu.mul_(Qnlut[:, :, None, None]).view(n * C, M, M)
    ix = lut_ixs.view(-1)[:, None, None].broadcast_to(QUlut.shape)
    Ulut.scatter_add_(dim=0, index=ix, src=QUlut)
    Ulut = Ulut[:nlut]

    return dict(m=m, R=R, Ulut=Ulut)


def _grad_counts(noise_N, N, log_pi, log_noise_prop):
    """Gradient of the ELBO with respect to the log proportions

    In EM, the update is derived with Lagrange multipliers. Here
    we work by reparameterizing -- backprop thru softmax.

    I'm dividing by batch size here (Q sums to batch size).
    """
    Ntot = N.sum() + noise_N
    delbo_dlogpi = N / Ntot - log_pi.exp()
    delbo_dlognoiseprop = noise_N / Ntot - log_noise_prop.exp()
    return Ntot, delbo_dlogpi, delbo_dlognoiseprop


#
# in next 2 fns, note that suff stats m, R, U are already /= Ntot.
#


def _grad_mean(Ntot, N, m, mu, active=slice(None), Cinv=None):
    N = N / Ntot
    d = m.reshape(mu.shape) - mu
    d.mul_(N[:, None])
    if Cinv is not None and (active == slice(None) or active.numel()):
        d[active] = d[active] @ Cinv.T
    return d


def _grad_basis(Ntot, N, R, W, U, active=slice(None), Cinv=None):
    N = N / Ntot
    d = torch.baddbmm(R.reshape(W.shape), U, W, alpha=-1)
    d.mul_(N[:, None, None])
    if Cinv is not None and (active == slice(None) or active.numel()):
        d[active] = torch.einsum("ij,npj->npi", Cinv, d[active])
    return d.view(R.shape)


def _elbo_prior_correction(
    alpha0, total_count, nc, mu, W, Cinv, alpha=None, mean_prior=False
):
    mu_term = 0.0
    if mean_prior:
        mu_term = torch.einsum("ki,ij,kj->", mu, Cinv, mu).double()
        mu_term *= -0.5 * (alpha0 / total_count)
    if W is None:
        return mu_term
    if alpha is None or nc is None:
        W_term = torch.einsum("kli,ij,klj->", W, Cinv.to(W), W).double()
        return mu_term - 0.5 * (alpha0 / total_count) * W_term

    W_term = torch.einsum("kli,ij,klj,kl->", W, Cinv.to(W), W, alpha.to(W)).double()
    alpha_term = alpha.log().mul_(nc).sum() / (2 * total_count)
    return mu_term - 0.5 * W_term / total_count + alpha_term


def woodbury_inv_quad(whitenedx, whitenednu, wburyroot=None, flat=False, ow_wnu=False):
    """Faster inv quad term in log likelihoods

    We want to compute
        (x - nu)' [Co + Wo Wo']^-1 (x - nu)
          = (x-nu)' [Co^-1 - Co^-1 Wo(I_m+Wo'Co^-1Wo)^-1Wo'Co^-1] (x-nu)

    Let's say we already have computed...
        %  name in code: whitenednu
        z  = Co^{-1/2}nu
        %  name in code: whitenedx
        x' = Co^{-1/2} x
        %  name in code: wburyroot
        A  = (I_m+Wo'Co^-1Wo)^{-1/2} Wo'Co^{-1/2}

    Break into terms. First,
        (A+)  (x-nu)'Co^-1(x-nu) = |x' - z|^2
    It doesn't seem helpful to break that down any further.

    Next up, while we have x' - z computed, notice that
        (B-) (x-nu)' Co^-1 Wo(I_m+Wo'Co^-1Wo)^-1Wo'Co^-1 (x-nu)
                = | A (x'-z') |^2.
    """
    if not flat:
        whitenedx = whitenedx.unsqueeze(1)
    wdxz = torch.subtract(whitenedx, whitenednu, out=whitenednu if ow_wnu else None)
    dim = 1 + (not flat)
    if wburyroot is None:
        return wdxz.square_().sum(dim=dim)
    if flat:
        term_b = torch.einsum("nj,njp->np", wdxz, wburyroot)
    else:
        term_b = torch.einsum("ncj,ncjp->ncp", wdxz, wburyroot)
    term_a = wdxz.square_().sum(dim=dim)
    term_b = term_b.square_().sum(dim=dim)
    return term_a.sub_(term_b)


def neighb_lut(unit_neighborhood_counts):
    """Unit-neighborhood ID lookup table

    Given the set of neighborhoods overlapping each unit, return
    a unique ID for each present unit/neighb combination.

    This will be returned as an array which can be indexed by
    unit_id, neighborhood_id to get a combined ID. If the combined
    ID is n_units * n_neighbs, it indicates a missing value.

    Arguments
    ---------
    unit_neighborhood_counts : array (n_units, n_neighborhoods)

    Returns
    -------

    """
    n_units, n_neighborhoods = unit_neighborhood_counts.shape
    unit_ids, neighborhood_ids = np.nonzero(unit_neighborhood_counts)

    # this has n_units + 1 because sometimes unit ix is -1
    # when unit ix is -1, we use nlut as the fill value. this works together
    # with the lut buffers in the model: those at index nlut are 0.
    lut = np.full(
        (n_units + 1, n_neighborhoods), (n_units + 1) * n_neighborhoods, dtype=np.int64
    )
    lut[unit_ids, neighborhood_ids] = np.arange(len(unit_ids))
    lut[-1] = len(unit_ids)

    return lut, unit_ids, neighborhood_ids


def units_overlapping_neighborhoods(unit_neighborhood_counts):
    """
    Arguments
    ---------
    unit_neighborhood_counts : array (n_units, n_neighborhoods)

    Returns
    -------
    neighb_units: LongTensor (n_neighborhoods, <n_units)
        Filled with n_units where irrelevant.
    """
    n_units, n_neighborhoods = unit_neighborhood_counts.shape
    unit_ids, neighborhood_ids = np.nonzero(unit_neighborhood_counts)
    max_overlap = (unit_neighborhood_counts > 0).sum(0).max()

    neighb_units = np.full((n_neighborhoods, max_overlap), n_units)
    for j in range(n_neighborhoods):
        inj = np.flatnonzero(neighborhood_ids == j)
        neighb_units[j, : inj.size] = unit_ids[inj]

    return neighb_units


def observed_and_missing_marginals(noise, neighborhoods, missing_chans):
    # Get Coos
    # Have to do everything as a list. That's what the
    # noise object supports, but also, we don't want to
    # pad with 0s since that would make logdets 0, Chols
    # confusing etc.
    # We will use the neighborhood valid ixs to pad later.
    # Since we cache everything we need, update can avoid
    # having to do lists stuff.
    Coo = []
    Com = []
    for ni in range(neighborhoods.n_neighborhoods):
        nci = neighborhoods.neighborhood_channels(ni)
        mci = missing_chans[ni]
        mci = mci[mci < noise.n_channels]
        Cooi = noise.marginal_covariance(
            channels=nci,
            cache_prefix=neighborhoods.name,
            cache_key=ni,
        )
        Comi = (
            noise.offdiag_covariance(
                channels_left=nci,
                channels_right=mci,
                device=Cooi.device,
            )
            .to_dense()
            .to(Cooi.device)
        )
        Coo.append(Cooi)
        Com.append(Comi)

    return Coo, Com


def missing_indices(
    neighborhoods,
    zero_radius: float | None = None,
    pgeom=None,
    device: str | torch.device = "cpu",
):
    # determine missing channels
    missing_chans = []
    full_missing_chans = []
    truncate = zero_radius and np.isfinite(zero_radius)
    for ni in range(neighborhoods.n_neighborhoods):
        mix = neighborhoods.missing_channels(ni)
        full_missing_chans.append(mix)

        if truncate:
            assert pgeom is not None
            assert zero_radius is not None
            oix = neighborhoods.neighborhood_channels(ni)
            d = torch.cdist(pgeom[mix], pgeom[oix]).min(dim=1).values
            (kept,) = (d < zero_radius).cpu().nonzero(as_tuple=True)
            mix = mix[kept]
        missing_chans.append(mix)

    nc_miss = max(map(len, missing_chans))
    nc_miss_full = max(map(len, full_missing_chans))

    miss_ix = torch.full(
        (neighborhoods.n_neighborhoods, nc_miss),
        fill_value=neighborhoods.n_channels,
        dtype=torch.long,
        device=device,
    )
    miss_ix_full = torch.full(
        (neighborhoods.n_neighborhoods, nc_miss_full),
        fill_value=neighborhoods.n_channels,
        dtype=torch.long,
        device=device,
    )

    for ni in range(neighborhoods.n_neighborhoods):
        miss_ix[ni, : missing_chans[ni].numel()] = missing_chans[ni]
        neighb_fmc = full_missing_chans[ni]
        miss_ix_full[ni, : neighb_fmc.numel()] = neighb_fmc

    miss_full_masks = torch.zeros(
        (neighborhoods.n_neighborhoods, neighborhoods.n_channels + 1),
        device=miss_ix_full.device,
    )
    src = _1[None, None].broadcast_to(miss_ix_full.shape).to(miss_ix_full.device)
    miss_full_masks.scatter_(dim=1, index=miss_ix_full, src=src)
    miss_full_masks[:, -1] = 0.0

    return miss_ix, miss_ix_full, miss_full_masks


def _processor_update_mean_batch(proc, sl):
    n = sl.stop - sl.start
    neighb_ix = proc.lut_neighbs[sl]

    Coo_invsqrt = proc.Coo_invsqrt[neighb_ix]
    Cooinv_Com = proc.Cooinv_Com[neighb_ix]
    nu_ = proc.nu[sl].reshape(n, proc.rank * proc.nc_obs, 1)

    torch.bmm(Coo_invsqrt, nu_, out=proc.whitenednu[sl].view(nu_.shape))
    torch.bmm(
        Cooinv_Com.mT,
        nu_,
        out=proc.Cmo_Cooinv_nu[sl].view(n, proc.rank * proc.nc_miss, 1),
    )


def _processor_update_pca_batch(proc, sl_W):
    sl, W = sl_W
    n = sl.stop - sl.start
    neighb_ix = proc.lut_neighbs[sl]
    unit_ix = proc.lut_units[sl]
    unit_ix_ = unit_ix[:, None, None, None]
    nobs_ix = proc.nobs_ix[sl]
    nobs_ix_ = nobs_ix[:, None, None, :]

    Coo_inv = proc.Coo_inv[neighb_ix]
    Coo_invsqrt = proc.Coo_invsqrt[neighb_ix]
    Cooinv_Com = proc.Cooinv_Com[neighb_ix]
    Coo_logdet = proc.Coo_logdet[neighb_ix]

    Wobs = W[unit_ix_, proc.M_ix, proc.r_ix, nobs_ix_]
    assert Wobs.shape == (n, proc.M, proc.rank, proc.nc_obs)
    Wobs = Wobs.view(n, proc.M, -1)
    proc.Wobs[sl] = Wobs

    torch.bmm(Cooinv_Com.mT, Wobs.mT, out=proc.Cmo_Cooinv_WobsT[sl])

    Cooinv_WobsT = Coo_inv.bmm(Wobs.mT)
    cap = Wobs.bmm(Cooinv_WobsT)
    cap.diagonal(dim1=-2, dim2=-1).add_(1.0)
    cap_chol = DenseLinearOperator(cap).cholesky()
    # cap^{-1} = L^-T L^-1, this is L-1
    cap_invsqrt = cap_chol.inverse().to_dense()
    cap_logdets = cap_chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=1).mul_(2.0)
    inv_cap = torch.bmm(cap_invsqrt.mT, cap_invsqrt, out=proc.inv_cap[sl])
    proc.obs_logdets[sl] = cap_logdets
    proc.obs_logdets[sl].add_(Coo_logdet)

    root_left = Wobs.mT.bmm(cap_invsqrt.mT)
    torch.bmm(Coo_invsqrt, root_left, out=proc.wburyroot[sl])
    torch.bmm(inv_cap, Cooinv_WobsT.mT, out=proc.inv_cap_Wobs_Cooinv[sl])


def _finalize_missing_full_m(proc, Nlut_N, m):
    """missing part aka tnu needs to be added with the right weights to m."""
    nlut = len(proc.lut_units)
    bs = proc.update_batch_size
    bargs = [
        (slice(i0, min(i0 + bs, nlut)), proc, Nlut_N, m) for i0 in range(0, nlut, bs)
    ]
    for _ in map(_finalize_missing_full_m_batch, bargs):
        pass


def _finalize_missing_full_m_batch(args):
    sl, proc, Nlut_N, m = args

    lut_units = proc.lut_units[sl]
    lut_neighbs = proc.lut_neighbs[sl]
    means = proc.means[lut_units]
    masks = proc.miss_full_masks[lut_neighbs]

    tnu = means.mul_(masks[:, None])[..., : proc.n_channels]
    tnu = tnu.mul_(Nlut_N[sl, None, None])
    ix = lut_units[:, None, None].broadcast_to(tnu.shape)
    m.scatter_add_(dim=0, index=ix, src=tnu)


def _finalize_missing_full_R(proc, Nlut_N, R, Ulut):
    nlut = len(proc.lut_units)
    bs = proc.update_batch_size
    bargs = [
        (slice(i0, min(i0 + bs, nlut)), proc, Nlut_N, R, Ulut)
        for i0 in range(0, nlut, bs)
    ]
    for _ in map(_finalize_missing_full_R_batch, bargs):
        pass


def _finalize_missing_full_R_batch(args):
    sl, proc, Nlut_N, R, Ulut = args
    n = sl.stop - sl.start

    neighb_ix = proc.lut_neighbs[sl]
    unit_ix = proc.lut_units[sl]
    miss_ix = proc.miss_ix[neighb_ix]
    masks = proc.miss_full_masks[neighb_ix]
    W = proc.bases[unit_ix]
    Nlut_N = Nlut_N[sl]

    # start with wmiss
    Wmiss = W.mul_(masks[:, None, None])
    del W
    assert Wmiss.shape == (n, proc.M, proc.rank, proc.n_channels + 1)
    WCC = proc.Cmo_Cooinv_WobsT[sl]
    assert WCC.shape == (n, proc.rank * proc.nc_miss, proc.M)
    src = WCC.mT.reshape(n, proc.M, proc.rank, proc.nc_miss)
    ix = miss_ix[:, None, None, :].broadcast_to(src.shape)
    W_WCC = Wmiss.scatter_add_(dim=3, index=ix, src=src.neg_())
    del Wmiss

    Euu_W_WCC = Ulut[sl].bmm(W_WCC.view(n, proc.M, -1))
    Euu_W_WCC = Euu_W_WCC.view(W_WCC.shape)[..., : proc.n_channels]

    src = Euu_W_WCC.mul_(Nlut_N[:, None, None, None])
    ix = unit_ix[:, None, None, None].broadcast_to(src.shape)
    R.scatter_add_(dim=0, index=ix, src=src)
