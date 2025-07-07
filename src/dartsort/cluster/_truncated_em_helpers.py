from logging import getLogger
from dataclasses import dataclass, fields

import numpy as np
import torch
from linear_operator.operators import DenseLinearOperator

from ..util import spiketorch
from ..util.logging_util import DARTSORTVERBOSE, DARTSORTDEBUG

log2pi = torch.log(torch.tensor(2 * np.pi))
logger = getLogger(__name__)
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
    # noise_log_prop changes, so can't be baked in
    noise_lls = nlls + (noise_log_prop + noise_eps)

    # observed log likelihoods
    if ds_verbose:
        logger.dartsortverbose(f"{whitenednu.isfinite().all()=}")
    inv_quad = woodbury_inv_quad(whitenedx, whitenednu, wburyroot=wburyroot)
    del whitenednu
    lls_unnorm = inv_quad.add_(obs_logdets).add_(pinobs[:, None]).mul_(-0.5)
    if with_kl:
        lls = lls_unnorm + log_proportions
    else:
        lls = lls_unnorm.add_(log_proportions)

    if _debug:
        llsfinite = lls.isfinite()
        if not torch.logical_or(candidates < 0, llsfinite).all():
            bad = torch.logical_and(candidates >= 0, torch.logical_not(llsfinite))
            infspk = bad.any(1)
            infcands = candidates[bad].unique()
            msg = (
                f"_te_batch_e: {lls.shape=} {llsfinite.sum()=} {lls[bad]=} {inv_quad[bad]=}"
                f"{infcands=} {obs_logdets[infcands]=} {log_proportions[infcands]=} "
                f"{pinobs[infcands]=} {whitenedx[infspk]} {whitenedx.isfinite().all()=}"
            )
            if wburyroot is not None:
                msg += f" {wburyroot[infcands]=}"
            raise ValueError(f"{bad.sum()=} {msg}")

    # cvalid = candidates >= 0
    # lls = torch.where(cvalid, lls, -torch.inf)
    lls[candidates < 0] = -torch.inf

    # -- update K_ns
    # toplls, topinds = lls.sort(dim=1, descending=True)
    # toplls = toplls[:, :n_candidates]
    # topinds = topinds[:, :n_candidates]
    all_lls = lls.new_empty((lls.shape[0], n_candidates + 1))
    all_inds = torch.empty((lls.shape[0], n_candidates), dtype=torch.long, device=lls.device)
    topk_out = (all_lls[:, :n_candidates], all_inds)
    toplls, topinds = torch.topk(lls, n_candidates, dim=1, sorted=False, out=topk_out)

    # -- compute Q
    # all_lls = torch.concatenate((toplls, noise_lls.unsqueeze(1)), dim=1)
    all_lls[:, -1] = noise_lls
    Q = torch.softmax(all_lls, dim=1)
    new_candidates = candidates.take_along_dim(topinds, 1)
    if _debug and not (new_candidates >= 0).all():
        (bad_ix,) = torch.nonzero((new_candidates < 0).any(dim=1).cpu(), as_tuple=True)
        raise ValueError(
            f"Bad candidates {lls=} {lls[bad_ix]=} {toplls[bad_ix]=} {topinds[bad_ix]=} {cvalid[bad_ix]=} {cvalid[topinds][bad_ix]=}"
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
    )


def _te_batch_m_counts(n_units, candidates, Q):
    """Part 1/2 of the M step within the E step"""
    Q_ = Q[:, :-1]
    assert Q_.shape == candidates.shape
    N = Q.new_zeros(n_units)
    tem_add_at_(N, candidates.view(-1), Q_.reshape(-1))
    noise_N = Q[:, -1].sum()
    return noise_N, N


def _te_batch_m_rank0(
    rank,
    n_units,
    nc,
    nc_obs,
    nc_miss,
    nc_miss_full,
    miss_to_full,
    # common args
    candidates,
    obs_ix,
    miss_ix,
    miss_ix_full,
    Q,
    N,
    x,
    nu,
    tnu,
    Cmo_Cooinv_x,
    Cmo_Cooinv_nu,
):
    """Rank (M) 0 case of part 2/2 of the M step within the E step"""

    Qn = Q[:, :-1] / N.clamp(min=1e-5)[candidates]
    n, C = Qn.shape
    arange_rank = torch.arange(rank, device=Q.device)

    mm = Cmo_Cooinv_nu
    del Cmo_Cooinv_nu
    mm -= Cmo_Cooinv_x[:, None]
    mm.mul_(Qn.unsqueeze(2))
    mmf = tnu
    del tnu
    mmf.mul_(Qn.unsqueeze(2))
    mo = Qn[:, :, None, None] * x.view(n, 1, rank, nc_obs)

    m = Qn.new_zeros((n_units, rank, nc + 1))
    tem_add_at_(
        m,
        (
            candidates[:, :, None, None],
            arange_rank[None, None, :, None],
            obs_ix[:, None, None, :],
        ),
        mo,
    )
    tem_add_at_(
        m,
        (
            candidates[:, :, None, None],
            arange_rank[None, None, :, None],
            miss_ix[:, None, None, :],
        ),
        mm.view(n, C, rank, nc_miss),
    )
    tem_add_at_(
        m,
        (
            candidates[:, :, None, None],
            arange_rank[None, None, :, None],
            miss_ix_full[:, None, None, :],
        ),
        mmf.view(n, C, rank, nc_miss_full),
    )

    return dict(m=m)


def _te_batch_m_ppca(
    rank,
    n_units,
    nc,
    nc_obs,
    nc_miss,
    nc_miss_full,
    # common args
    candidates,
    obs_ix,
    miss_ix,
    miss_ix_full,
    miss_to_full,
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
    Qn = Q[:, :-1] / N.clamp(min=1e-5)[candidates]
    n, C = Qn.shape
    arange_rank = torch.arange(rank, device=Q.device)
    M = inv_cap.shape[-1]
    arange_M = torch.arange(M, device=Q.device)

    U = Q.new_zeros((n_units, M, M))
    R = Q.new_zeros((n_units, M, rank, nc))
    m = Qn.new_zeros((n_units, rank, nc))

    R_full = Q.new_zeros((n, C, M, rank, nc + 1))
    m_full = Q.new_zeros((n, C, rank, nc + 1))

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
    R_missing_full = inv_cap_W_WCC
    del inv_cap_W_WCC
    R_missing_full += torch.einsum("ncpk,ncp,ncq->ncqk", W_WCC, ubar, ubar)
    R_missing = ubar.unsqueeze(3) * Cmo_Cooinv_xc.unsqueeze(2)

    src = R_observed.view(n, C, M, rank, nc_obs)
    ix = obs_ix[:, None, None, None, :].broadcast_to(src.shape)
    R_full.scatter_add_(dim=4, index=ix, src=src)
    src = R_missing.view(n, C, M, rank, nc_miss)
    ix = miss_ix[:, None, None, None, :].broadcast_to(src.shape)
    R_full.scatter_add_(dim=4, index=ix, src=src)
    src = R_missing_full.view(n, C, M, rank, nc_miss_full)
    ix = miss_ix_full[:, None, None, None, :].broadcast_to(src.shape)
    R_full.scatter_add_(dim=4, index=ix, src=src)

    m_missing_full = tnu
    del tnu
    m_missing = torch.sub(
        Cmo_Cooinv_xc, Cmo_Cooinv_WobsT_ubar, out=Cmo_Cooinv_WobsT_ubar
    )
    del Cmo_Cooinv_WobsT_ubar
    m_observed = torch.sub(x[:, None], WobsT_ubar, out=WobsT_ubar)
    del WobsT_ubar

    src = m_observed.view(n, C, rank, nc_obs)
    ix = obs_ix[:, None, None, :].broadcast_to(src.shape)
    m_full.scatter_add_(dim=3, index=ix, src=src)
    src = m_missing.view(n, C, rank, nc_miss)
    ix = miss_ix[:, None, None, :].broadcast_to(src.shape)
    m_full.scatter_add_(dim=3, index=ix, src=src)
    src = m_missing_full.view(n, C, rank, nc_miss_full)
    ix = miss_ix_full[:, None, None, :].broadcast_to(src.shape)
    m_full.scatter_add_(dim=3, index=ix, src=src)

    QU = EuuT.mul_(Qn[:, :, None, None]).view(-1, *U.shape[1:])
    QRf = R_full[..., :nc].mul_(Qn[:, :, None, None, None]).view(-1, *R.shape[1:])
    Qmf = m_full[..., :nc].mul_(Qn[:, :, None, None]).view(-1, *m.shape[1:])

    ix = candidates.view(-1)[:, None, None].broadcast_to(QU.shape)
    U.scatter_add_(dim=0, index=ix, src=QU)

    ix = candidates.view(-1)[:, None, None, None].broadcast_to(QRf.shape)
    R.scatter_add_(dim=0, index=ix, src=QRf)
    
    ix = candidates.view(-1)[:, None, None].broadcast_to(Qmf.shape)
    m.scatter_add_(dim=0, index=ix, src=Qmf)

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


def woodbury_inv_quad(whitenedx, whitenednu, wburyroot=None):
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
    wdxz = whitenedx.unsqueeze(1) - whitenednu
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


def observed_and_missing_marginals(
    noise, neighborhoods, missing_chans
):
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
        Comi = noise.offdiag_covariance(
            channels_left=nci,
            channels_right=mci,
            device=Cooi.device,
        ).to_dense().to(Cooi.device)
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
    missing_to_full = [] if truncate else None
    for ni in range(neighborhoods.n_neighborhoods):
        mix = neighborhoods.missing_channels(ni)
        full_missing_chans.append(mix)

        if truncate:
            assert pgeom is not None
            assert missing_to_full is not None
            assert zero_radius is not None
            oix = neighborhoods.neighborhood_channels(ni)
            d = torch.cdist(pgeom[mix], pgeom[oix]).min(dim=1).values
            (kept,) = (d < zero_radius).cpu().nonzero(as_tuple=True)
            mix = mix[kept]
            missing_to_full.append(kept)
        missing_chans.append(mix)

    nc_miss = max(map(len, missing_chans))
    nc_miss_full = max(map(len, full_missing_chans))

    # pack the lists into more usable buffers
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
    miss_to_full = torch.zeros_like(miss_ix) if truncate else None

    for ni in range(neighborhoods.n_neighborhoods):
        miss_ix[ni, : missing_chans[ni].numel()] = missing_chans[ni]
        neighb_fmc = full_missing_chans[ni]
        miss_ix_full[ni, : neighb_fmc.numel()] = neighb_fmc
        if truncate:
            assert miss_to_full is not None
            assert missing_to_full is not None
            neighb_m2f = missing_to_full[ni]
            miss_to_full[ni, : neighb_m2f.numel()] = neighb_m2f

    return miss_ix, miss_ix_full, miss_to_full


def _processor_update_mean_batch(proc, sl):
    n = sl.stop - sl.start
    neighb_ix = proc.lut_neighbs[sl]

    Coo_invsqrt = proc.Coo_invsqrt[neighb_ix]
    Cooinv_Com = proc.Cooinv_Com[neighb_ix]
    nu_ = proc.nu[sl].reshape(n, proc.rank * proc.nc_obs, 1)

    torch.bmm(Coo_invsqrt, nu_, out=proc.whitenednu[sl].view(nu_.shape))
    torch.bmm(
        Cooinv_Com.mT, nu_, out=proc.Cmo_Cooinv_nu[sl].view(n, proc.rank * proc.nc_miss, 1)
    )


def _processor_update_pca_batch(proc, sl_W):
    sl, W = sl_W
    n = sl.stop - sl.start
    neighb_ix = proc.lut_neighbs[sl]
    unit_ix = proc.lut_units[sl]
    unit_ix_ = unit_ix[:, None, None, None]
    nobs_ix = proc.nobs_ix[neighb_ix]
    nobs_ix_ = nobs_ix[:, None, None, :]
    nmissfull_ix = proc.miss_ix_full[neighb_ix]
    nmissfull_ix_ = nmissfull_ix[:, None, None, :]

    Coo_inv = proc.Coo_inv[neighb_ix]
    Coo_invsqrt = proc.Coo_invsqrt[neighb_ix]
    Cooinv_Com = proc.Cooinv_Com[neighb_ix]
    Coo_logdet = proc.Coo_logdet[neighb_ix]

    Wobs = W[unit_ix_, proc.M_ix, proc.r_ix, nobs_ix_]
    assert Wobs.shape == (n, proc.M, proc.rank, proc.nc_obs)
    Wobs = Wobs.view(n, proc.M, -1)
    proc.Wobs[sl] = Wobs


    Wmiss = W[unit_ix_, proc.M_ix, proc.r_ix, nmissfull_ix_]
    assert Wmiss.shape == (n, proc.M, proc.rank, proc.nc_miss_full)
    Wmiss = Wmiss.view(n, proc.M, -1)

    Cmo_Cooinv_WobsT = torch.bmm(Cooinv_Com.mT, Wobs.mT, out=proc.Cmo_Cooinv_WobsT[sl])

    Cooinv_WobsT = Coo_inv.bmm(Wobs.mT)
    cap = Wobs.bmm(Cooinv_WobsT)
    cap.diagonal(dim1=-2, dim2=-1).add_(1.0)
    # 
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

    if proc.miss_to_full is None:
        W_WCC = torch.subtract(Wmiss, Cmo_Cooinv_WobsT.mT, out=Wmiss)
    else:
        to_sub = Cmo_Cooinv_WobsT.mT.reshape(n, proc.M, proc.rank, proc.nc_miss)
        miss_to_full = proc.miss_to_full[neighb_ix]
        miss_to_full = miss_to_full[:, None, None].broadcast_to(to_sub.shape)
        W_WCC = Wmiss.view(n, proc.M, proc.rank, proc.nc_miss_full)
        W_WCC.scatter_add_(index=miss_to_full, dim=3, src=to_sub.neg_())
        W_WCC = W_WCC.view(n, proc.M, -1)

    inv_cap_W_WCC = inv_cap.bmm(W_WCC)
    proc.W_WCC[sl] = W_WCC.to(proc.W_WCC)
    proc.inv_cap_W_WCC[sl] = inv_cap_W_WCC.to(proc.inv_cap_W_WCC)

