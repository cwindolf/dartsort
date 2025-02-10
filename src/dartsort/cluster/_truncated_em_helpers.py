from dataclasses import dataclass, field, replace, fields
from typing import Union, Optional
import numpy as np
import torch

from ..util import spiketorch

log2pi = torch.log(torch.tensor(2 * np.pi))
_1 = torch.tensor(1.0)


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

    def take_along_candidates(self, inds, extras):
        cand_fields = [
            f.name
            for f in fields(self)
            if f.metadata.get("has_candidates", False)
            and getattr(self, f.name) is not None
        ]
        replacements = {
            k: getattr(self, k).take_along_dim(
                dim=1, indices=inds[..., *([None] * (getattr(self, k).ndim - 2))]
            )
            for k in cand_fields
        }
        new_tebd = replace(self, **replacements)
        if extras:
            extras = [
                (
                    e.take_along_dim(dim=1, indices=inds[..., *([None] * (e.ndim - 2))])
                    if e is not None
                    else None
                )
                for e in extras
            ]
        return new_tebd, extras


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

    Cooinv_x = torch.bmm(bd.Coo_inv, bd.x.unsqueeze(2))
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
        WobsT_Cooinv_xc = bd.Wobs.mT.bmm(Cooinv_xc)
        inner = bd.inv_cap.bmm(WobsT_Cooinv_xc)
        inv_quad = torch.baddbmm(
            Cooinv_xc.view(-1, Do, 1),
            bd.Cooinv_WobsT.view(-1, Do, self.M),
            inner,
            alpha=-1,
        )
    else:
        inv_quad = Cooinv_xc
    del Cooinv_xc  # overwritten
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
    bd_new, (xc, WobsT_Cooinv_xc) = bd.take_along_candidates(
        topinds, (xc, WobsT_Cooinv_xc)
    )
    assert xc is not None

    # -- KL part comes before throwing away extra units
    ncc = dkl = None
    if with_kl:
        ncc = Q.new_zeros((self.n_units, self.n_units))
        spiketorch.add_at_(
            ncc,
            (bd_new.candidates[:, None, :1], bd.candidates[:, :, None]),
            _1.broadcast_to(bd.candidates.shape),
        )
        dkl = Q.new_zeros((self.n_units, self.n_units))
        spiketorch.add_at_(
            dkl,
            (bd_new.candidates[:, None, :1], bd.candidates[:, :, None]),
            lls_unnorm[:, 0, None] - lls_unnorm,
        )

    # -- replace bd name...
    bd = bd_new
    C = bd.candidates.shape[1]

    # -- sufficient statistics
    Qn = N = noise_N = None
    if with_stats:
        N = Q.new_zeros(self.n_units)
        spiketorch.add_at_(N, bd.candidates.view(-1), Q_.reshape(-1))
        noise_N = Q[:, -1].sum()
        # used below for weighted averaging
        Qn = Q_ / N[bd.candidates]

    R = U = m = None
    if with_stats and self.M:
        assert Qn is not None
        assert WobsT_Cooinv_xc is not None  # for pyright
        assert bd.W_WCC is not None
        assert bd.Wobs is not None
        assert bd.inv_cap is not None

        # embedding moments
        # T low key is inv_cap. overwriting since this is the last use.
        ubar = bd.inv_cap.bmm(WobsT_Cooinv_xc)
        U = bd.inv_cap.baddbmm_(ubar.unsqueeze(2), ubar.unsqueeze(1))
        Qu = Qn[:, :, None, None] * ubar[:, :, :, None]
        QU = Qn[:, :, None, None] * U
        Ro = Qu * xc[:, :, None, :]
        # this does not have an n dim. we're reducing.
        # Ro is (n, C, M, Do). R is (Ctotal, M, Do)
        R = Q.new_zeros((self.n_units, self.M, self.rank, self.nc + 1))
        spiketorch.add_at_(
            R,
            (
                bd.candidates[:, :, None, None, None],
                torch.arange(self.M)[:, None, :, None, None],
                torch.arange(self.rank)[:, None, None, :, None],
                bd.obs_ix[:, None, None, None, :],
            ),
            Ro.view(*Ro.shape[:3], self.rank, self.nc_obs),
        )
        Rm = QU.bmm(bd.W_WCC).baddbmm(Ro, bd.Cooinv_Com)
        spiketorch.add_at_(
            R,
            (
                bd.candidates[:, :, None, None, None],
                torch.arange(self.M)[:, None, :, None, None],
                torch.arange(self.rank)[:, None, None, :, None],
                bd.miss_ix[:, None, None, None, :],
            ),
            Rm.view(*Rm.shape[:3], self.rank, self.nc_miss),
        )
        # E[y-Wu]
        m = Q.new_zeros((self.n_units, self.rank, self.nc + 1))
        Wobs_ubar = torch.bmm(bd.Wobs, ubar)
        mo = Qn[:, :, None] * (Wobs_ubar + bd.x[:, None])
        mo.baddbmm(bd.Wobs, Qu, alpha=-1)
        spiketorch.add_at_(
            m,
            (
                bd.candidates[:, :, None, None],
                torch.arange(self.rank)[None, None, :, None],
                bd.obs_ix[:, None, None, :],
            ),
            mo.view(bd.n, C, self.rank, self.nc_obs),
        )

        mm = Qn[:, :, None] * (bd.tnu.baddbmm_(bd.Cooinv_Com.mT, xc - Wobs_ubar))
        spiketorch.add_at_(
            m,
            (
                bd.candidates[:, :, None, None],
                torch.arange(self.rank)[None, None, :, None],
                bd.miss_ix[:, None, None, :],
            ),
            mm.view(C, self.rank, self.nc_miss),
        )
    elif with_stats:
        assert Qn is not None

        m = Qn.new_zeros((self.n_units, self.rank, self.nc + 1))
        spiketorch.add_at_(
            m,
            (
                bd.candidates[:, :, None, None],
                torch.arange(self.rank)[None, None, :, None],
                bd.obs_ix[:, None, None, :],
            ),
            Qn[:, :, None, None] * bd.x.view(bd.n, 1, self.rank, self.nc_obs),
        )

        mm = Qn[:, :, None] * (bd.tnu.baddbmm_(xc, bd.Cooinv_Com))
        spiketorch.add_at_(
            m,
            (
                bd.candidates[:, :, None, None],
                torch.arange(self.rank)[None, None, :, None],
                bd.miss_ix[:, None, None, :],
            ),
            mm.view(bd.n, C, self.rank, self.nc_miss),
        )

    # -- gradients
    # not implemented
    assert not with_grads

    # -- elbo
    # not implemented
    assert not with_elbo

    # -- obs elbo
    obs_elbo = None
    if with_obs_elbo:
        obs_elbo = torch.sum(Q * (all_lls + Q.log()))

    return TEBatchResult(
        indices=bd.indices,
        candidates=bd.candidates,
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
