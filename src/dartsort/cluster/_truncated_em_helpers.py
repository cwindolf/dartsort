from logging import getLogger
from dataclasses import dataclass, fields

import numpy as np
import torch

from ..util import spiketorch
from ..util.logging_util import DARTSORTVERBOSE

log2pi = torch.log(torch.tensor(2 * np.pi))
logger = getLogger(__name__)


ds_verbose = logger.isEnabledFor(DARTSORTVERBOSE)
FRZ = not ds_verbose


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


@dataclass(slots=FRZ, kw_only=FRZ, frozen=FRZ)
class TEStepResult:
    elbo: torch.Tensor | None = None
    obs_elbo: torch.Tensor | None = None

    noise_N: torch.Tensor | None = None
    N: torch.Tensor | None = None
    R: torch.Tensor | None = None
    U: torch.Tensor | None = None
    m: torch.Tensor | None = None

    kl: torch.Tensor | None = None

    hard_labels: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    count: int | None = None

    if ds_verbose:
        __post_init__ = __debug_init__


@dataclass(slots=FRZ, kw_only=FRZ, frozen=FRZ)
class TEBatchResult:
    indices: slice | torch.Tensor
    candidates: torch.Tensor

    elbo: torch.Tensor | None = None
    obs_elbo: torch.Tensor | None = None

    noise_N: torch.Tensor | None = None
    N: torch.Tensor | None = None
    R: torch.Tensor | None = None
    U: torch.Tensor | None = None
    m: torch.Tensor | None = None

    ddlogpi: torch.Tensor | None = None
    ddlognoisep: torch.Tensor | None = None
    ddm: torch.Tensor | None = None
    ddW: torch.Tensor | None = None

    ncc: torch.Tensor | None = None
    dkl: torch.Tensor | None = None

    noise_lls: torch.Tensor | None = None
    probs: torch.Tensor | None = None
    hard_labels: torch.Tensor | None = None

    if ds_verbose:
        __post_init__ = __debug_init__


@dataclass(slots=FRZ, kw_only=FRZ, frozen=FRZ)
class TEBatchEData:
    n: int
    indices: slice | torch.Tensor
    whitenedx: torch.Tensor
    whitenednu: torch.Tensor
    Coo_logdet: torch.Tensor
    nobs: torch.Tensor
    wburyroot: torch.Tensor | None
    noise_lls: torch.Tensor | None


def _te_batch_e(
    n_units,
    n_candidates,
    noise_log_prop,
    candidates,
    whitenedx,
    whitenednu,
    nobs,
    obs_logdets,
    Coo_logdet,
    log_proportions,
    noise_lls=None,
    noise_trunc_factors=None,
    wburyroot=None,
    with_kl=False,
    with_probs=False,
):
    """This is the "E step" within the E step."""
    pinobs = log2pi * nobs

    # marginal noise log likelihoods, if needed
    nlls = noise_lls
    if noise_lls is None:
        inv_quad = whitenedx.square().sum(dim=1)
        nlls = inv_quad.add_(Coo_logdet).add_(pinobs).mul_(-0.5)
        if noise_trunc_factors is not None:
            nlls += noise_trunc_factors
    # noise_log_prop changes, so can't be baked in
    noise_lls = nlls + noise_log_prop

    # observed log likelihoods
    if ds_verbose:
        logger.dartsortverbose(f"{whitenednu.isfinite().all()=}")
    inv_quad = woodbury_inv_quad(
        whitenedx, whitenednu, wburyroot=wburyroot, overwrite_nu=True
    )
    del whitenednu
    lls_unnorm = inv_quad.add_(obs_logdets).add_(pinobs[:, None]).mul_(-0.5)
    lls = lls_unnorm + log_proportions

    if ds_verbose:
        llsfinite = lls.isfinite()
        if not llsfinite.all():
            llsinf = torch.logical_not(llsfinite)
            infspk = llsinf.any(1)
            infcands = candidates[torch.logical_not(llsfinite)].unique()
            logger.dartsortverbose(
                f"_te_batch_e: {lls.shape=} {llsfinite.sum()=} {lls[llsinf]=} {inv_quad[llsinf]=}"
                f"{infcands=} {obs_logdets[infcands]=} {log_proportions[infcands]=} "
                f"{pinobs[infcands]=} {whitenedx[infspk]} {whitenedx.isfinite().all()=}"
            )
            if wburyroot is not None:
                logger.dartsortverbose(f"{wburyroot[infcands]=}")

    cvalid = candidates >= 0
    lls = torch.where(cvalid, lls, -torch.inf)

    # -- update K_ns
    toplls, topinds = lls.sort(dim=1, descending=True)
    toplls = toplls[:, :n_candidates]
    topinds = topinds[:, :n_candidates]

    # -- compute Q
    all_lls = torch.concatenate((toplls, noise_lls.unsqueeze(1)), dim=1)
    Q = torch.softmax(all_lls, dim=1)
    new_candidates = candidates.take_along_dim(topinds, 1)
    assert (new_candidates >= 0).all()

    ncc = dkl = None
    if with_kl:
        # todo: this is probably broken after the candidate masking was
        # implemented. leaving it for now because unused.
        cvalid = cvalid.to(torch.float)
        ncc = Q.new_zeros((n_units, n_units))
        spiketorch.add_at_(
            ncc,
            (new_candidates[:, None, :1], candidates[:, :, None]),
            cvalid[:, :, None],
        )
        dkl = Q.new_zeros((n_units, n_units))
        top_lls_unnorm = lls_unnorm.take_along_dim(topinds[:, :1], dim=1)
        spiketorch.add_at_(
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
    )


def _te_batch_m_counts(n_units, candidates, Q):
    """Part 1/2 of the M step within the E step"""
    Q_ = Q[:, :-1]
    N = Q.new_zeros(n_units)
    spiketorch.add_at_(N, candidates.view(-1), Q_.reshape(-1))
    noise_N = Q[:, -1].sum()
    return noise_N, N


def _te_batch_m_rank0(
    rank,
    n_units,
    nc,
    nc_obs,
    nc_miss,
    # common args
    candidates,
    obs_ix,
    miss_ix,
    Q,
    N,
    x,
    nu,
    tnu,
    Cmo_Cooinv_x,
    Cmo_Cooinv_nu,
):
    """Rank (M) 0 case of part 2/2 of the M step within the E step"""

    Qn = Q[:, :-1] / N.clamp(min=1.0)[candidates]
    n, C = Qn.shape
    arange_rank = torch.arange(rank, device=Q.device)

    xc = torch.sub(x[:, None], nu, out=nu)
    del nu

    mm = tnu
    del tnu
    mm += Cmo_Cooinv_x[:, None]
    mm -= Cmo_Cooinv_nu
    mm.mul_(Qn.unsqueeze(2))
    out = xc.view(n, C, rank, nc_obs)
    del xc
    mo = torch.mul(Qn[:, :, None, None], x.view(n, 1, rank, nc_obs), out=out)

    m = Qn.new_zeros((n_units, rank, nc + 1))
    spiketorch.add_at_(
        m,
        (
            candidates[:, :, None, None],
            arange_rank[None, None, :, None],
            obs_ix[:, None, None, :],
        ),
        mo,
    )
    spiketorch.add_at_(
        m,
        (
            candidates[:, :, None, None],
            arange_rank[None, None, :, None],
            miss_ix[:, None, None, :],
        ),
        mm.view(n, C, rank, nc_miss),
    )

    return dict(m=m)


def _te_batch_m_ppca(
    rank,
    n_units,
    nc,
    nc_obs,
    nc_miss,
    # common args
    candidates,
    obs_ix,
    miss_ix,
    Q,
    N,
    x,
    nu,
    tnu,
    Cmo_Cooinv_x,
    Cmo_Cooinv_nu,
    # M>0 only args
    inv_cap,
    inv_cap_Wobs_Cooinv,
    Cmo_Cooinv_WobsT,
    inv_cap_W_WCC,
    W_WCC,
    Wobs,
):
    """Rank (M) >0 case of part 2/2 of the M step within the E step"""
    Qn = Q[:, :-1] / N.clamp(min=1.0)[candidates]
    n, C = Qn.shape
    arange_rank = torch.arange(rank, device=Q.device)
    M = inv_cap.shape[-1]
    arange_M = torch.arange(M, device=Q.device)

    xc = torch.sub(x[:, None], nu, out=nu)
    del nu
    Cmo_Cooinv_xc = torch.sub(Cmo_Cooinv_x[:, None], Cmo_Cooinv_nu, out=Cmo_Cooinv_nu)
    del Cmo_Cooinv_nu

    ubar = torch.einsum("ncpj,ncj->ncp", inv_cap_Wobs_Cooinv, xc)
    EuuT = inv_cap
    del inv_cap
    EuuT += ubar.unsqueeze(2) * ubar.unsqueeze(3)

    WobsT_ubar = torch.einsum("ncpk,ncp->nck", Wobs, ubar)
    Cmo_Cooinv_WobsT_ubar = torch.einsum("nckp,ncp->nck", Cmo_Cooinv_WobsT, ubar)

    R_observed = ubar[:, :, :, None] * xc[:, :, None, :]
    R_missing = inv_cap_W_WCC
    del inv_cap_W_WCC
    R_missing += torch.einsum("ncpk,ncp,ncq->ncqk", W_WCC, ubar, ubar)
    R_missing += ubar.unsqueeze(3) * Cmo_Cooinv_xc.unsqueeze(2)

    m_missing = tnu.add_(Cmo_Cooinv_xc).sub_(Cmo_Cooinv_WobsT_ubar)
    del tnu
    m_observed = torch.sub(x[:, None], WobsT_ubar, out=WobsT_ubar)
    del WobsT_ubar

    QU = EuuT.mul_(Qn[:, :, None, None])
    QRo = R_observed.mul_(Qn[:, :, None, None])
    QRm = R_missing.mul_(Qn[:, :, None, None])
    Qmo = m_observed.mul_(Qn[:, :, None])
    Qmm = m_missing.mul_(Qn[:, :, None])

    U = Q.new_zeros((n_units, M, M))
    spiketorch.add_at_(
        U,
        (
            candidates[:, :, None, None],
            arange_M[None, None, :, None],
            arange_M[None, None, None, :],
        ),
        QU,
    )

    R = Q.new_zeros((n_units, M, rank, nc + 1))
    QRo = QRo.view(n, C, *R.shape[1:-1], nc_obs)
    QRm = QRm.view(n, C, *R.shape[1:-1], nc_miss)
    spiketorch.add_at_(
        R,
        (
            candidates[:, :, None, None, None],
            arange_M[None, None, :, None, None],
            arange_rank[None, None, None, :, None],
            obs_ix[:, None, None, None, :],
        ),
        QRo,
    )
    spiketorch.add_at_(
        R,
        (
            candidates[:, :, None, None, None],
            arange_M[None, None, :, None, None],
            arange_rank[None, None, None, :, None],
            miss_ix[:, None, None, None, :],
        ),
        QRm,
    )

    m = Qn.new_zeros((n_units, rank, nc + 1))
    Qmo = Qmo.view(n, C, *m.shape[1:-1], nc_obs)
    Qmm = Qmm.view(n, C, *m.shape[1:-1], nc_miss)
    spiketorch.add_at_(
        m,
        (
            candidates[:, :, None, None],
            arange_rank[None, None, :, None],
            obs_ix[:, None, None, :],
        ),
        Qmo,
    )
    spiketorch.add_at_(
        m,
        (
            candidates[:, :, None, None],
            arange_rank[None, None, :, None],
            miss_ix[:, None, None, :],
        ),
        Qmm,
    )
    m = m.view(n_units, rank, nc + 1)

    return dict(m=m, R=R, U=U)


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


def _elbo_prior_correction(prior_pseudocount, total_count, mu, W, Cinv, alpha=None):
    mu_term = torch.einsum("ki,ij,kj->", mu, Cinv, mu).double()
    mu_term = -0.5 * (prior_pseudocount / total_count) * mu_term
    if alpha is None:
        W_term = 0
        if W is not None:
            W_term = torch.einsum("kli,ij,klj->", W, Cinv, W).double()
        return mu_term - 0.5 * (prior_pseudocount / total_count) * W_term

    W_term = torch.einsum("kli,ij,klj,kl->", W, Cinv, W, alpha.to(W)).double()
    alpha_term = (W.shape[2] / 2) * alpha.log().sum() / total_count
    return mu_term - 0.5 * W_term / total_count + alpha_term


def woodbury_inv_quad(whitenedx, whitenednu, wburyroot=None, overwrite_nu=False):
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
    out = whitenednu if overwrite_nu else None
    wdxz = torch.sub(
        whitenedx.unsqueeze(1),
        whitenednu,
        # out=out,
    )
    if ds_verbose:
        logger.dartsortverbose(f"{wdxz.isfinite().all()=}")
    if wburyroot is None:
        return wdxz.square_().sum(dim=2)
    term_b = torch.einsum("ncj,ncjp->ncp", wdxz, wburyroot)
    term_a = wdxz.square_().sum(dim=2)
    term_b = term_b.square_().sum(dim=2)
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

    lut = np.full_like(unit_neighborhood_counts, n_units * n_neighborhoods)
    lut[unit_ids, neighborhood_ids] = np.arange(len(unit_ids))

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
