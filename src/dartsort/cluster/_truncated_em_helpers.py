from dataclasses import dataclass, field, replace, fields
from typing import Union, Optional
import numpy as np
import torch
from linear_operator.utils.cholesky import psd_safe_cholesky

from ..util import spiketorch

log2pi = torch.log(torch.tensor(2 * np.pi))
_1 = torch.tensor(1.0)


DEBUG = False


def __debug_init__(self):
    print("-" * 40, self.__class__.__name__)
    res = {}
    for f in fields(self):
        v = getattr(self, f.name)
        if torch.is_tensor(v):
            res[f.name] = v.isnan().any().item()
            print(
                f" - {f.name}: {v.shape} min={v.min().item():g} mean={v.to(torch.float).mean().item():g} max={v.max().item():g} sum={v.sum().item():g}"
            )
    if any(res.values()):
        msg = f"NaNs in {self.__class__.__name__}: {res}"
        raise ValueError(msg)
    print("//" + "-" * 38)


@dataclass  # (slots=True, kw_only=True, frozen=True)
class TEStepResult:
    elbo: Optional[torch.Tensor] = None
    obs_elbo: Optional[torch.Tensor] = None

    noise_N: Optional[torch.Tensor] = None
    N: Optional[torch.Tensor] = None
    R: Optional[torch.Tensor] = None
    U: Optional[torch.Tensor] = None
    m: Optional[torch.Tensor] = None

    kl: Optional[torch.Tensor] = None

    if DEBUG:
        __post_init__ = __debug_init__


@dataclass  # (slots=True, kw_only=True, frozen=True)
class TEBatchResult:
    indices: Union[slice, torch.Tensor]
    candidates: torch.Tensor

    elbo: Optional[torch.Tensor] = None
    obs_elbo: Optional[torch.Tensor] = None

    noise_N: Optional[torch.Tensor] = None
    N: Optional[torch.Tensor] = None
    R: Optional[torch.Tensor] = None
    U: Optional[torch.Tensor] = None
    m: Optional[torch.Tensor] = None

    ddW: Optional[torch.Tensor] = None
    ddm: Optional[torch.Tensor] = None
    noise_lls: Optional[torch.Tensor] = None

    ncc: Optional[torch.Tensor] = None
    dkl: Optional[torch.Tensor] = None

    if DEBUG:
        __post_init__ = __debug_init__


_hc = dict(has_candidates=True)


@dataclass  # (slots=True, kw_only=True, frozen=True)
class TEBatchData:
    indices: Union[slice, torch.Tensor]
    n: int

    # -- data
    # (n, D_obs_max)
    x: torch.Tensor
    # (n, C_)
    candidates: torch.Tensor = field(metadata=_hc)

    # -- neighborhood dependent
    # (n,)
    Coo_logdet: torch.Tensor
    # (n, D_obs_max, D_obs_max)
    Coo_inv: torch.Tensor
    # (n, D_obs_max, D_miss_max)
    Cooinv_Com: torch.Tensor
    # (n, D_obs_max)
    obs_ix: torch.Tensor
    # (n, D_miss_max)
    miss_ix: torch.Tensor
    # (n,)
    nobs: torch.Tensor

    # -- parameters
    # (n, n_candidate_units)
    log_proportions: torch.Tensor = field(metadata=_hc)
    # (n, n_candidate_units, D_obs_max)
    nu: torch.Tensor = field(metadata=_hc)
    tnu: torch.Tensor = field(metadata=_hc)
    # (n, n_candidate_units, D_obs_max)
    Cooinv_nu: torch.Tensor = field(metadata=_hc)

    # -- parameters with pca
    Cooinv_WobsT: Optional[torch.Tensor] = field(metadata=_hc)
    obs_logdets: torch.Tensor = field(metadata=_hc)
    W_WCC: Optional[torch.Tensor] = field(metadata=_hc)
    inv_cap: Optional[torch.Tensor] = field(metadata=_hc)
    Wobs: Optional[torch.Tensor] = field(metadata=_hc)

    # (n,)
    noise_lls: Optional[torch.Tensor] = None

    # possible precomputed stuff / buffers
    # (n, D_obs_max)
    Cooinv_x: Optional[torch.Tensor] = None

    if DEBUG:
        __post_init__ = __debug_init__


def take_along_candidates(inds, data, out=None):
    if out is None:
        out = [None] * len(data)
    K = inds.shape[1]
    res = []
    for j, d in enumerate(data):
        o = out[j]
        if o is not None:
            o = o[:, :K]
        if d is not None:
            d = torch.take_along_dim(
                d, dim=1, indices=inds[..., *([None] * (d.ndim - 2))], out=o
            )
        res.append(d)
    return res


def _te_batch(
    self,
    bd: TEBatchData,
    with_grads=False,
    with_stats=False,
    with_kl=False,
    with_elbo=False,
    with_obs_elbo=False,
) -> TEBatchResult:
    """This is bound as a method of TruncatedExpectationProcessor"""
    C_ = bd.candidates.shape[1]
    Do = bd.x.shape[1]

    Cooinv_x = bd.Cooinv_x
    if Cooinv_x is None:
        Cooinv_x = torch.bmm(bd.Coo_inv, bd.x.unsqueeze(2))[..., 0]
    Cooinv_x = Cooinv_x.unsqueeze(2)
    pinobs = log2pi * bd.nobs

    # marginal noise log likelihoods, if needed
    nlls = bd.noise_lls
    if bd.noise_lls is None:
        # inv_quad = torch.bmm(bd.Coo_invsqrt, bd.x.unsqueeze(1))
        # inv_quad = inv_quad.square_().sum(dim=(1, 2))
        inv_quad = torch.bmm(bd.x.unsqueeze(1), Cooinv_x).view(bd.n)
        nlls = inv_quad.add_(bd.Coo_logdet).add_(pinobs).mul_(-0.5)
    # this may change, so can't be baked in
    noise_lls = nlls + self.noise_log_prop

    # -- observed log likelihoods
    # Woodbury solve of inv(Coo + Wobs Wobs') (x - mu)
    # inv(Coo + Wobs Wobs') = Cooinv - Cooinv W inv(I + W'CooinvW) W' Cooinv
    # parenthesize this as:
    # %% avoids multiplying Cooinv with Nu*n vecs. it's Nu+n vecs.
    # a = Cooinvx - Cooinvmu
    # %% break into two DxMs
    # B = inv(I + W'CooinvW) W' is MxD
    # b = Ba
    # C = Cooinv W is DxM
    # c = a - Cb  %% (TODO overwriting a with baddbmm?)
    # TODO: after implementing, re-evaluate if it would be better to multiply by
    # the inv sqrt and square here. in particular... maybe we want to avoid having
    # to build xc?
    xc = bd.x[:, None] - bd.nu
    # Cooinv_xc = Cooinv_x.mT[:, None] - bd.Cooinv_nu
    Cooinv_xc = torch.add(
        Cooinv_x.mT,
        bd.Cooinv_nu,
        alpha=-1,
        out=bd.Cooinv_nu,
    )
    WobsT_Cooinv_xc = None
    if self.M:
        assert bd.Wobs is not None
        assert bd.inv_cap is not None
        assert bd.Cooinv_WobsT is not None
        WobsT_Cooinv_xc = bd.Wobs.view(-1, self.M, Do).bmm(
            Cooinv_xc.view(-1, Do).unsqueeze(2)
        )
        inner = bd.inv_cap.view(-1, self.M, self.M).bmm(WobsT_Cooinv_xc)
        inv_quad = Cooinv_xc.view(-1, Do, 1).baddbmm_(
            # Cooinv_xc.view(-1, Do, 1),
            bd.Cooinv_WobsT.view(-1, Do, self.M),
            inner,
            alpha=-1,
        )
        inv_quad = inv_quad.view(bd.n, C_, Do)
        WobsT_Cooinv_xc = WobsT_Cooinv_xc.view(bd.n, C_, self.M)
    else:
        inv_quad = Cooinv_xc
    del Cooinv_xc  # overwritten in both branches
    inv_quad = inv_quad.mul_(xc).sum(dim=2)
    lls_unnorm = inv_quad.add_(bd.obs_logdets).add_(pinobs[:, None]).mul_(-0.5)
    # the proportions...
    lls = lls_unnorm + bd.log_proportions

    # -- update K_ns
    toplls, topinds = lls.sort(dim=1, descending=True)
    toplls = toplls[:, : self.n_candidates]
    topinds = topinds[:, : self.n_candidates]

    # -- compute Q
    all_lls = torch.concatenate((toplls, noise_lls.unsqueeze(1)), dim=1)
    Q = torch.softmax(all_lls, dim=1)
    Q_ = Q[:, :-1]

    # extract top candidate inds
    orig_candidates = bd.candidates
    new_candidates, xc, nu, tnu, WobsT_Cooinv_xc, inv_cap, W_WCC, Wobs = (
        take_along_candidates(
            topinds,
            data=(
                bd.candidates,
                xc,
                bd.nu,
                bd.tnu,
                WobsT_Cooinv_xc,
                bd.inv_cap,
                bd.W_WCC,
                bd.Wobs,
            ),
            out=(
                None,
                xc,
                bd.nu,
                bd.tnu,
                None,
                None,
                None,
                None,
            ),
        )
    )
    assert nu is not None
    assert tnu is not None
    assert xc is not None
    assert new_candidates is not None
    C = new_candidates.shape[1]

    # -- KL part comes before throwing away extra units
    ncc = dkl = None
    if with_kl:
        ncc = Q.new_zeros((self.n_units, self.n_units))
        spiketorch.add_at_(
            ncc,
            (new_candidates[:, None, :1], orig_candidates[:, :, None]),
            # _1.broadcast_to(orig_candidates.shape),
            1.0,
        )
        dkl = Q.new_zeros((self.n_units, self.n_units))
        top_lls_unnorm = lls_unnorm.take_along_dim(topinds[:, :1], dim=1)
        spiketorch.add_at_(
            dkl,
            (new_candidates[:, :1], orig_candidates),
            top_lls_unnorm - lls_unnorm,
        )

    # -- sufficient statistics
    Qn = N = noise_N = None
    if with_stats:
        N = Q.new_zeros(self.n_units)
        spiketorch.add_at_(N, new_candidates.view(-1), Q_.reshape(-1))
        noise_N = Q[:, -1].sum()
        # used below for weighted averaging
        Qn = Q_ / N.clamp(min=1.0)[new_candidates]
        # Qn = Q_

    R = U = m = None
    arange_rank = torch.arange(self.rank, device=Q.device)
    if with_stats and self.M:
        assert Qn is not None
        assert WobsT_Cooinv_xc is not None  # for pyright
        assert bd.W_WCC is not None
        assert Wobs is not None
        assert inv_cap is not None
        arange_M = torch.arange(self.M, device=Q.device)

        # embedding moments
        # T low key is inv_cap. overwriting since this is the last use.
        ubar = torch.bmm(
            inv_cap.view(-1, self.M, self.M),
            WobsT_Cooinv_xc.view(-1, self.M, 1),
            # out=bd.ubar.view(-1, self.M, 1),
        )[..., 0]
        Euu = inv_cap.view(-1, self.M, self.M).baddbmm_(
            ubar.unsqueeze(2), ubar.unsqueeze(1)
        )
        ubar = ubar.view(bd.n, C, self.M)
        Euu = Euu.view(bd.n, C, self.M, self.M)
        Qu = Qn[:, :, None, None] * ubar[:, :, :, None]
        QU = Qn[:, :, None, None] * Euu

        U = Q.new_zeros((self.n_units, self.M, self.M))
        spiketorch.add_at_(
            U,
            (
                new_candidates[:, :, None, None],
                arange_M[None, None, :, None],
                arange_M[None, None, None, :],
            ),
            QU,
        )

        Ro = Qu * xc[:, :, None, :]
        # this does not have an n dim. we're reducing.
        # Ro is (n, C, M, Do). R is (Ctotal, M, Do)
        R = Q.new_zeros((self.n_units, self.M, self.rank, self.nc + 1))
        spiketorch.add_at_(
            R,
            (
                new_candidates[:, :, None, None, None],
                arange_M[None, None, :, None, None],
                arange_rank[None, None, None, :, None],
                bd.obs_ix[:, None, None, None, :],
            ),
            Ro.view(*Ro.shape[:3], self.rank, self.nc_obs),
        )
        Rm = torch.bmm(
            QU.view(-1, self.M, self.M),
            W_WCC.view(-1, *W_WCC.shape[-2:]),
            out=W_WCC.view(-1, *W_WCC.shape[-2:]),
        )
        del QU, W_WCC
        Rm = Rm.view(bd.n, C, self.M, self.rank * self.nc_miss)
        # don't want to materialize a big array of Cooinv_Coms.
        for j in range(C):
            Rm[:, j].baddbmm_(Ro[:, j], bd.Cooinv_Com)
        spiketorch.add_at_(
            R,
            (
                new_candidates[:, :, None, None, None],
                arange_M[None, None, :, None, None],
                arange_rank[None, None, None, :, None],
                bd.miss_ix[:, None, None, None, :],
            ),
            Rm.view(*Rm.shape[:3], self.rank, self.nc_miss),
        )
        # E[y-Wu]
        m = Q.new_zeros((self.n_units, self.rank, self.nc + 1))
        Wobs_ubar = torch.bmm(
            Wobs.view(-1, self.M, self.rank * self.nc_obs).mT,
            ubar.view(-1, self.M, 1),
            # out=bd.Wobs_ubar.view(-1, self.rank * self.nc_obs, 1),
        )

        # add xs to m's observed part
        xc = xc.view(bd.n, C, self.rank * self.nc_obs)
        xc.sub_(Wobs_ubar.view(xc.shape))

        mm = tnu + torch.matmul(xc, bd.Cooinv_Com)
        mm.mul_(Qn.unsqueeze(2))
        # mm = tnu.baddbmm_(
        #     bd.Cooinv_Com.mT, xc.view(bd.n * C, self.rank * self.nc_obs)
        # ).mul_(Qn[:, :, None])
        del tnu
        mo = xc.add_(nu)
        del xc  # overwritten
        mo.sub_(Wobs_ubar.view(mo.shape))
        mo.mul_(Qn.unsqueeze(2))
        spiketorch.add_at_(
            m,
            (
                new_candidates[:, :, None, None],
                arange_rank[None, None, :, None],
                bd.obs_ix[:, None, None, :],
            ),
            mo.view(bd.n, C, self.rank, self.nc_obs),
        )
        spiketorch.add_at_(
            m,
            (
                new_candidates[:, :, None, None],
                arange_rank[None, None, :, None],
                bd.miss_ix[:, None, None, :],
            ),
            mm.view(bd.n, C, self.rank, self.nc_miss),
        )
    elif with_stats:
        assert Qn is not None

        m = Qn.new_zeros((self.n_units, self.rank, self.nc + 1))

        mm = tnu.baddbmm_(xc, bd.Cooinv_Com).mul_(Qn.unsqueeze(2))
        out = xc.view(bd.n, C, self.rank, self.nc_obs)
        del xc  # overwritten
        mo = torch.mul(
            Qn[:, :, None, None], bd.x.view(bd.n, 1, self.rank, self.nc_obs), out=out
        )

        spiketorch.add_at_(
            m,
            (
                new_candidates[:, :, None, None],
                arange_rank[None, None, :, None],
                bd.obs_ix[:, None, None, :],
            ),
            mo,
        )
        spiketorch.add_at_(
            m,
            (
                new_candidates[:, :, None, None],
                arange_rank[None, None, :, None],
                bd.miss_ix[:, None, None, :],
            ),
            mm.view(bd.n, C, self.rank, self.nc_miss),
        )

    # -- gradients
    # not implemented
    assert not with_grads

    # -- elbo/n
    # not implemented
    assert not with_elbo

    # -- obs elbo/n
    obs_elbo = None
    if with_obs_elbo:
        logQ = torch.where(Q > 0, Q, _1).log()
        all_lls = torch.where(Q > 0, all_lls, torch.tensor(0.0))
        obs_elbo = torch.sum(Q * (all_lls + logQ), dim=1).mean()

    return TEBatchResult(
        indices=bd.indices,
        candidates=new_candidates,
        elbo=None,
        obs_elbo=obs_elbo,
        noise_N=noise_N,
        N=N,
        R=R,
        U=U,
        m=m,
        ddW=None,
        ddm=None,
        ncc=ncc,
        dkl=dkl,
        noise_lls=nlls if bd.noise_lls is None else None,
    )


def woodbury_kl_divergence(C, mu, W=None, out=None, batch_size=32, affine_ok=True):
    """KL divergence with the lemmas, up to affine constant with respect to mu and W

    Here's the logic between the lines below. Variable names follow this notation.

    If W is None, then
        2 DKL_0(other || self) = (mu_self - mu_other)' C^-1 (mu_self - mu_other)
    where d is the dimension.

    To do n triangular matmuls rather than n^2 full ones, compute this via:
        mu' = C^{-1/2} mu
        (mu_self - mu_other)' C^-1 (mu_self - mu_other) = |mu'_self - mu'_other|^2

    Otherwise,
        2 DKL(other || self) = tr(S_self^-1 S_other)
                               + log(|S_self| / |S_other|)
                               + (mu_self - mu_other)' S_self^-1 (mu_self - mu_other)
                               - d

    For the log dets,
        log|S| = log|C + WW'| = log|C| + log|I_M + W'C^-1W|
    Forget about log|C|, even though it's precomputed. For the other part, to save
    some matmuls, let
        U = C^{-1/2}W
        cap = I_M + W'C^{-1}W = I_m + U'U
    Note that log|C| will cancel.

    For the trace part, we have
        S_self^-1 S_other = (C + Ws Ws')^-1 (C + Wo Wo')
            = [C^-1 - C^-1 Ws caps^-1 Ws' C^-1] (C + Wo Wo')
            = C^-1 C
              + C^-1 Wo Wo'
              - C^-1 Ws caps^-1 Ws'
              - C^-1 Ws caps^-1 Ws' C^-1 Wo Wo'

    tr(C^-1 C) = d, so that cancels the constant.

    To save matmuls, introduce
        Vs = Us caps^{-1/2}

    Then by cyclic property, save more operations by rewriting
        (A+) tr(C^-1 Wo Wo') = tr(Uo' Uo)
    and
        (B-) tr(C^-1 Ws caps^-1 Ws') = tr(Us caps^-1 Us')
                                     = tr(Vs Vs')
    and
        (C-) tr(C^-1 Ws caps^-1 Ws' C^-1 Wo Wo')
                = tr(Wo' C^{-1/2} C^{-1/2} Ws caps^{-1/2} caps^{-1/2} Ws' C^{-1/2} C^{-1/2} Wo)
                = tr(Uo' Vs Vs' Uo)

    The Mahalanobis term remains. Woodbury says
        (mus - muo)' (C + Ws Ws')^-1 (mus - muo)
            = (mus - muo)' [C^-1 - C^-1 Ws cap^-1 Ws' C^-1] (mus - muo)
            = |mu's - mu'o|^2
              - (mu's - mu'o)' C^{-1/2} Ws cap^-1 Ws' C^{-1/2} (mu's - mu'o)
            = |mu's - mu'o|^2 - |caps^{-1/2} Ws' C^{-1/2} (mu's - mu'o)|^2
            = |mu's - mu'o|^2 - |Vs' (mu's - mu'o)|^2

    TODO: There is something wrong here.

    Returns
    -------
    out : Tensor
        out[i, j] = const(wrt mu,W) * (D(component i || component j) + const(wrt mu,W))
        i = other, j = self
    """
    n, d = mu.shape
    assert C.shape == (d, d)
    if out is None:
        out = mu.new_empty((n, n))
    out.fill_(0.0)

    Cchol = C.cholesky().to_dense()
    mu_ = torch.linalg.solve_triangular(Cchol, mu.unsqueeze(2), upper=False)

    if W is None:
        # else, better to do this later
        for bs in range(0, n, batch_size):
            osl = slice(bs, min(bs + batch_size, n))
            out[osl] = (mu_[osl, None] - mu_[None]).square_().sum(dim=(2, 3))
        return out

    M = W.shape[2]
    assert W.shape == (n, d, M)
    U = torch.linalg.solve_triangular(Cchol, W, upper=False)

    # first part of trace
    UTU = U.mT.bmm(U)
    # term A: it's an other-vec
    trA = UTU.diagonal(dim1=-2, dim2=-1).sum(dim=1)
    # assert torch.isclose(trA, U.square().sum(dim=(1, 2))).all()

    # make cap matrix and get its Cholesky
    cap = UTU
    del UTU
    cap.diagonal(dim1=-2, dim2=-1).add_(1.0)
    capchol = psd_safe_cholesky(cap)
    # V is (n, d, M)
    V = torch.linalg.solve_triangular(capchol, U, upper=False, left=False)

    # log dets via Cholesky
    logdet = capchol.diagonal(dim1=-2, dim2=-1).log().sum(dim=1).mul_(2.0)
    # assert torch.isclose(logdet, cap.logdet()).all()

    # trace term C: it's a matrix of Frob inner products
    tmp = None
    for bs in range(0, n, batch_size):
        osl = slice(bs, min(bs + batch_size, n))
        if tmp is not None:
            tmp = tmp[: osl.stop - bs]
        tmp = torch.matmul(V.mT[None], U[osl, None], out=tmp)
        out[osl] = -tmp.square_().sum(dim=(2, 3))

    # mahalanobis term
    dmu = None
    for bs in range(0, n, batch_size):
        osl = slice(bs, min(bs + batch_size, n))
        if dmu is not None:
            dmu = dmu[: osl.stop - bs]
        dmu = torch.sub(mu_[None], mu_[osl, None], out=dmu)
        corr = dmu.mT @ V[None]
        corr = corr.square_().sum(dim=(2, 3))
        out[osl] -= corr
        mah = dmu.square_().sum(dim=(2, 3))
        out[osl] += mah

    # trace term B: it's a self-vec
    trB = V.square_().sum(dim=(1, 2))

    # put back together
    # remember: self dim is dim 1, other dim is dim 0
    out += logdet[None, :]
    out -= logdet[:, None]
    out += trA[:, None]
    out -= trB[None, :]

    return out


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
