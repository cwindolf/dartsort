"""Truncated mixture modeling (see papers by Sebastian, Lücke, et al.)

Inference in the model
  l_n ~ pi
  observed part of spike_n | l_n ~ N(mu[l_n], W[l_n]@W[l_n].T + noise)
with truncated expectation maximization.

Also has some parallel cross validation ideas for merging and splitting units.

Notes for readers on this implementation.
 - The main annoying thing is missing channels. That's why there's so much going on.
 - ^ More on this: each spike `i` has observed (well, stored) data living on channels called
   `obs_ix[neighborhood_ids[i]]` below. In other words, there are some unique possible observed
   channel neighborhoods, and the structure is shared via this `neighborhood_ids` array. The
   neighborhood ID depends on the spike's main channel and on the drift at that time (if relevant).
 - The covariance is restricted to be zero outside noise.zero_radius. The "near missing ixs"
   below are within that radius. But, some things need to happen on the full probe regardless;
   that's miss_full_masks.
 - There's scatter_ and scatter_add_ everywhere! Usually over channels, sometimes units. This
   function is great but also footgun: the index argument doesn't broadcast and can silently
   do only part of what you want if you don't manually broadcast_to(src.shape). So every single
   scatter here has index argument with same shape as src.
 - There are these "LUT"s here. This is what makes this possible to do fast at all. The idea
   is that there are "neigbhorhoods" (unique subsets of channels that spikes are observed on)
   such that many covariance-related things can be cached and re-used per neighborhood (that's
   the NeighborhoodCovariance) object. But some things depend on the unit and the neighborhood,
   like for instance the part of a unit's mean inside the neighborhood. But units and neighborhoods
   coincide very sparsely, so we don't want to compute every subset for all neighborhoods for all
   units. The LUT sparsely stores which ones actually happen / are allowed to happen in different
   situations. Keep it small!

TODO items:
 - Use the usual Kronecker structure of the noise covariance better. We can save a lot of flops
   here by not making CmoCooinv et al dense. A refactor would probably want to maintain the
   dense functionality though, since it's useful for testing and who knows. So, maybe the way
   to go is to add a method to the NeighborhoodCovariance or LUTParams objects to case over
   this. (I think probably best not to store linear_operator.LinearOperators, right?)
"""

import gc
import math
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Literal, NamedTuple, Optional, Self

import numpy as np
import torch
import torch.nn.functional as F
from sympy.utilities.iterables import multiset_partitions
from torch import Tensor
from tqdm.auto import tqdm, trange

from ...util.data_util import DARTsortSorting, subset_sorting_by_spike_count
from ...util.internal_config import (
    ComputationConfig,
    DARTsortInternalConfig,
    RefinementConfig,
)
from ...util.interpolation_util import NeighborhoodInterpolator
from ...util.job_util import ensure_computation_config
from ...util.logging_util import DARTSORTDEBUG, DARTSORTVERBOSE, get_logger
from ...util.main_util import ds_save_intermediate_labels
from ...util.noise_util import EmbeddedNoise
from ...util.py_util import databag
from ...util.spiketorch import (
    cosine_distance,
    ecl,
    elbo,
    entropy,
    mean_elbo_dim1,
    sign,
    spawn_torch_rg,
)
from ...util.torch_util import BModule
from ..cluster_util import linkage, maximal_leaf_groups
from ..kmeans import kmeans
from .stable_features import (
    SpikeNeighborhoods,
    StableSpikeDataset,
)

logger = get_logger(__name__)
pnoid = logger.isEnabledFor(DARTSORTVERBOSE)


# -- main


def tmm_demix(
    *,
    sorting: DARTsortSorting,
    motion_est,
    refinement_cfg: RefinementConfig,
    computation_cfg: ComputationConfig | None,
    save_step_labels_format: str | None = None,
    save_step_labels_dir: Path | None = None,
    save_cfg: DARTsortInternalConfig | None = None,
    seed: int | np.random.Generator = 0,
) -> DARTsortSorting:
    """GMM-based spike clustering using truncated expectation maximization

    Infers #units using a cross-validation criterion evaluated over proposed
    splits and merges.

    TODO output the soft assignments as extra features of the returned sorting.
    """
    global pnoid
    pnoid = logger.isEnabledFor(DARTSORTVERBOSE)
    if pnoid:
        logger.dartsortverbose("Extra TMM asserts are on.")
    prog_level = 1 + logger.isEnabledFor(DARTSORTVERBOSE)

    tmm, train_data, val_data, full_data = instantiate_and_bootstrap_tmm(
        sorting=sorting,
        motion_est=motion_est,
        refinement_cfg=refinement_cfg,
        seed=seed,
        computation_cfg=computation_cfg,
    )

    saving = save_cfg is not None and save_cfg.save_intermediate_labels
    save_kw = dict(
        save_step_labels_format=save_step_labels_format,
        full_data=full_data,
        original_sorting=sorting,
        save_step_labels_dir=save_step_labels_dir,
        save_cfg=save_cfg,
    )
    if saving:
        assert save_step_labels_format is not None

    # start with one round of em. below flow is like split-em-merge-em-repeat.
    tmm.em(train_data)

    for outer_it in range(refinement_cfg.n_total_iters):
        do_split = bool(outer_it) or not refinement_cfg.skip_first_split
        break_after_split = refinement_cfg.one_split_only

        if do_split:
            run_split(tmm, train_data, val_data, prog_level - 1)
            tmm.em(train_data, show_progress=prog_level)
        if saving:
            save_tmm_labels(tmm=tmm, stepname=f"tmm{outer_it}asplit", **save_kw)  # type: ignore
        if break_after_split:
            break

        run_merge(tmm, train_data, val_data, prog_level - 1)
        tmm.em(train_data, show_progress=prog_level)
        if saving:
            save_tmm_labels(tmm=tmm, stepname=f"tmm{outer_it}bmerge", **save_kw)  # type: ignore

    # final assignments
    # TODO output the soft probs somehow
    full_scores = tmm.soft_assign(
        data=full_data, full_proposal_view=True, needs_bootstrap=False
    )
    labels = labels_from_scores(full_scores)
    neighb_ids = full_data.neighborhood_ids.numpy(force=True).copy()
    del tmm, train_data, val_data, full_data, full_scores
    sorting = sorting.ephemeral_replace(labels=labels, neighborhood_ids=neighb_ids)
    return sorting


# -- shared objects holding precomputed neighborhood-related data


class MixtureModelAndDatasets(NamedTuple):
    tmm: "TruncatedMixtureModel"
    train_data: "TruncatedSpikeData"
    val_data: "TruncatedSpikeData | None"
    full_data: "StreamingSpikeData"


@databag
class NeighborhoodCovariance:
    """Holds precomputed terms which depend only on the neighborhood."""

    feat_rank: int
    # most observed channels in any neighborhood
    max_nc_obs: int
    # most missing channels (inside the cov zero radius)
    max_nc_miss_near: int
    # total n channels
    n_channels: int

    # -- indexing
    # number of observed features (rank*chans) by neighborhood
    # TODO: if we only use this for liks, maybe pre-bake the log2pi?
    nobs: Tensor
    # observed channels (just the neighborhoods array)
    obs_ix: Tensor
    # channels within zero rad of observed
    miss_near_ix: Tensor
    # indicator of missingness (not restricted by zero rad)
    miss_full_mask: Tensor
    # miss_ix_full not used

    # -- matrices
    # log det of observed covariance
    logdet: Tensor
    Cooinv: Tensor
    # Com is observed to miss-near
    CooinvCom: Tensor
    # Coo = LL', and this is Linv. Cinv=Linv'Linv, so this multiplies from the left.
    Linv: Tensor
    full_Linv: Tensor

    def __post_init__(self):
        """Shape validation / documentation."""
        nneighb = self.nobs.shape[0]
        obsdim = self.feat_rank * self.max_nc_obs
        neardim = self.feat_rank * self.max_nc_miss_near
        assert self.nobs.shape == (nneighb,)
        assert self.obs_ix.shape == (nneighb, self.max_nc_obs)
        assert self.miss_near_ix.shape == (nneighb, self.max_nc_miss_near)
        assert self.miss_full_mask.shape == (nneighb, self.n_channels)
        assert self.logdet.shape == (nneighb,)
        assert self.Cooinv.shape == (nneighb, obsdim, obsdim)
        assert self.CooinvCom.shape == (nneighb, obsdim, neardim)

    @classmethod
    def from_noise_and_neighborhoods(
        cls, prgeom: Tensor, noise: EmbeddedNoise, neighborhoods: SpikeNeighborhoods
    ) -> Self:
        dev: torch.device = noise.device  # type: ignore
        neighborhoods = neighborhoods.to(device=dev)
        nc_obs, obs_ix, miss_near_ix, miss_full_mask = _neighborhood_indices(
            neighborhoods=neighborhoods,
            zero_radius=noise.zero_radius,
            prgeom=prgeom.to(device=dev),
        )
        logdet, Cooinv, CooinvCom, Linv = _noise_factors(
            noise=noise,
            obs_ix=obs_ix,
            miss_near_ix=miss_near_ix,
            cache_prefix=neighborhoods.name,
        )
        assert obs_ix.shape[0] == neighborhoods.n_neighborhoods
        assert obs_ix.shape[0] == miss_near_ix.shape[0] == miss_full_mask.shape[0]
        assert obs_ix.shape[0] == logdet.shape[0] == Cooinv.shape[0]
        assert obs_ix.shape[0] == CooinvCom.shape[0] == Linv.shape[0]
        return cls(
            feat_rank=noise.rank,
            max_nc_obs=obs_ix.shape[1],
            max_nc_miss_near=miss_near_ix.shape[1],
            n_channels=neighborhoods.n_channels,
            nobs=noise.rank * nc_obs,
            obs_ix=obs_ix,
            miss_near_ix=miss_near_ix,
            miss_full_mask=miss_full_mask,
            logdet=logdet,
            Cooinv=Cooinv,
            CooinvCom=CooinvCom,
            Linv=Linv,
            full_Linv=noise.whitener(),
        )


@databag
class NeighborhoodLUT:
    """Which labels coincided with which spike neighborhoods? Need this all over the place."""

    unit_ids: Tensor
    neighb_ids: Tensor
    # lut[unit_ids, neighb_ids] = arange(n_lut)
    # everything else is n_lut (not -1)
    lut: Tensor

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NeighborhoodLUT):
            return False
        if not torch.equal(self.unit_ids, other.unit_ids):
            return False
        if not torch.equal(self.neighb_ids, other.neighb_ids):
            return False
        return torch.equal(self.lut, other.lut)


@databag
class LUTParams:
    """Holds precomputed terms which depend on neighborhood and GMM parameters."""

    signal_rank: int
    n_lut: int

    # unit- and neighborhood-dependent terms
    muo: Tensor
    Linvmuo: Tensor
    CmoCooinvmuo: Tensor
    # low-rank model observed-data log determinant plus other constant lik terms
    constplogdet: Tensor

    # signal_rank > 0 only
    TWoCooinvsqrt: Tensor | None
    TWoCooinvmuo: Tensor | None
    # T, zero-padded on inner dim with an extra column
    Tpad: Tensor | None
    # main factor for using inverse lemma to compute likelihoods
    wburyroot: Tensor | None

    @classmethod
    def from_lut(
        cls,
        means: Tensor,
        bases: Tensor | None,
        neighb_cov: NeighborhoodCovariance,
        lut: NeighborhoodLUT,
        puff=1.0,
    ):
        signal_rank = 0 if bases is None else bases.shape[1]
        self = cls._new_empty(
            lut.unit_ids.shape[0],
            signal_rank,
            dim_obs=neighb_cov.max_nc_obs * neighb_cov.feat_rank,
            dim_miss_near=neighb_cov.max_nc_miss_near * neighb_cov.feat_rank,
            device=means.device,
            puff=puff,
        )
        self.update(means, bases, neighb_cov, lut)
        return self

    def update(
        self,
        means: Tensor,
        bases: Tensor | None,
        neighb_cov: NeighborhoodCovariance,
        lut: NeighborhoodLUT,
        batch_size=64,
    ):
        self._resize_(lut.unit_ids.shape[0])
        for i0 in range(0, self.n_lut, batch_size):
            i1 = min(self.n_lut, i0 + batch_size)
            # nb, this initializes constplogdet with the noise part
            # the signal part is added (det lemma) below if nec
            _update_lut_mean_batch(self, i0, i1, means, lut, neighb_cov)
            if self.signal_rank == 0:
                continue
            assert bases is not None
            _update_lut_ppca_batch(self, i0, i1, bases, lut, neighb_cov)
        if pnoid:
            self.check()

    @classmethod
    def _new_empty(
        cls,
        n_lut: int,
        signal_rank: int,
        dim_obs: int,
        dim_miss_near: int,
        device: torch.device,
        puff=1.0,
    ) -> Self:
        n0 = n_lut if puff == 1.0 else int(puff * n_lut)
        o = torch.zeros((n0, dim_obs), device=device)
        m = torch.zeros((n0, dim_miss_near), device=device)
        sc = torch.zeros((n0,), device=device)
        if signal_rank:
            ro = torch.zeros((n0, signal_rank, dim_obs), device=device)
            r = torch.zeros((n0, signal_rank), device=device)
            rh = torch.zeros((n0, signal_rank, signal_rank + 1), device=device)
            romt = torch.zeros_like(ro.mT)
        else:
            ro = r = rh = romt = None
        self = cls(
            signal_rank=signal_rank,
            n_lut=n0,
            muo=o,
            Linvmuo=torch.zeros_like(o),
            CmoCooinvmuo=m,
            constplogdet=sc,
            TWoCooinvsqrt=ro,
            TWoCooinvmuo=r,
            Tpad=rh,
            wburyroot=romt,
        )
        if n0 != n_lut:
            self._resize_(n_lut)
        return self

    def _resize_(self, n_lut_new):
        if n_lut_new == self.n_lut:
            return

        self.n_lut = n_lut_new

        self.muo.resize_(n_lut_new, *self.muo.shape[1:])
        self.muo.fill_(0.0)
        self.Linvmuo.resize_(n_lut_new, *self.Linvmuo.shape[1:])
        self.Linvmuo.fill_(0.0)
        self.CmoCooinvmuo.resize_(n_lut_new, *self.CmoCooinvmuo.shape[1:])
        self.CmoCooinvmuo.fill_(0.0)
        self.constplogdet.resize_(n_lut_new, *self.constplogdet.shape[1:])
        self.constplogdet.fill_(0.0)
        if not self.signal_rank:
            return
        assert self.TWoCooinvsqrt is not None
        assert self.TWoCooinvmuo is not None
        assert self.Tpad is not None
        assert self.wburyroot is not None
        self.TWoCooinvsqrt.resize_(n_lut_new, *self.TWoCooinvsqrt.shape[1:])
        self.TWoCooinvsqrt.fill_(0.0)
        self.TWoCooinvmuo.resize_(n_lut_new, *self.TWoCooinvmuo.shape[1:])
        self.TWoCooinvmuo.fill_(0.0)
        self.Tpad.resize_(n_lut_new, *self.Tpad.shape[1:])
        self.Tpad.fill_(0.0)
        self.wburyroot.resize_(n_lut_new, *self.wburyroot.shape[1:])
        self.wburyroot.fill_(0.0)

    def check(self):
        assert self.muo.isfinite().all()
        assert self.Linvmuo.isfinite().all()
        assert self.CmoCooinvmuo.isfinite().all()
        assert self.constplogdet.isfinite().all()
        if not self.signal_rank:
            return
        assert self.TWoCooinvsqrt is not None
        assert self.TWoCooinvmuo is not None
        assert self.Tpad is not None
        assert self.wburyroot is not None
        assert self.TWoCooinvsqrt.isfinite().all()
        assert self.TWoCooinvmuo.isfinite().all()
        assert self.Tpad.isfinite().all()
        assert self.wburyroot.isfinite().all()


# -- messenger classes


@databag
class Scores:
    log_liks: Tensor
    responsibilities: Tensor | None
    candidates: Tensor

    def slice(self, indices: Tensor | slice) -> "Scores":
        if self.responsibilities is None:
            r = None
        else:
            r = self.responsibilities[indices]
        return Scores(
            log_liks=self.log_liks[indices],
            responsibilities=r,
            candidates=self.candidates[indices],
        )


@databag
class SufficientStatistics:
    """Represents a batch's statistics or the whole dataset's statistics in E step."""

    count: float
    noise_N: Tensor | None
    N: Tensor
    Nlut: Tensor
    R: Tensor
    Ulut: Tensor | None
    elbo: Tensor

    @classmethod
    def zeros(
        cls,
        n_units: int,
        n_lut: int,
        n_channels: int,
        feature_rank: int,
        signal_rank: int,
        skip_noise: bool,
        device: torch.device,
        count_dtype=torch.double,
    ) -> Self:
        hat_dim = signal_rank + 1

        if signal_rank:
            Ulut = torch.zeros((n_lut, signal_rank, hat_dim), device=device)
        else:
            Ulut = None
        if skip_noise:
            noise_N = None
        else:
            noise_N = torch.zeros((), device=device, dtype=count_dtype)
        return cls(
            count=0.0,
            noise_N=noise_N,
            N=torch.zeros((n_units,), device=device, dtype=count_dtype),
            Nlut=torch.zeros((n_lut,), device=device, dtype=count_dtype),
            R=torch.zeros((n_units, n_channels, hat_dim, feature_rank), device=device),
            Ulut=Ulut,
            elbo=torch.zeros((), device=device, dtype=count_dtype),
        )

    def reset_(self, n_units: int, n_channels: int, signal_rank: int, feature_rank: int):
        self.count = 0.0
        if self.noise_N is not None:
            self.noise_N.zero_()
        self.N.zero_()
        self.Nlut.zero_()
        self.R.zero_()
        hat_dim = signal_rank + 1
        assert self.R.numel() == n_units * n_channels * hat_dim * feature_rank
        self.R.resize_(n_units, n_channels, hat_dim, feature_rank)
        if self.Ulut is not None:
            self.Ulut.zero_()
        self.elbo.zero_()

    def combine(self, other: "SufficientStatistics", eps: Tensor):
        """Welford running means."""
        self.count += other.count
        w_count = other.count / max(1, self.count)
        self.N += other.N
        w_N = other.N.div_(self.N.clamp(min=eps))
        self.Nlut += other.Nlut

        self.elbo += (other.elbo - self.elbo) * w_count
        self.R += other.R.sub_(self.R).mul_(w_N[:, None, None, None])
        if pnoid:
            assert self.elbo.isfinite()

        if self.noise_N is not None:
            assert other.noise_N is not None
            self.noise_N += other.noise_N

        if self.Ulut is not None:
            assert other.Ulut is not None
            w_Nlut = other.Nlut.div_(self.Nlut.clamp(min=eps))
            self.Ulut += other.Ulut.sub_(self.Ulut).mul_(w_Nlut[:, None, None])


@databag
class TruncatedEStepResult:
    candidates: Tensor
    stats: SufficientStatistics


@databag
class SuccessfulUnitSplitResult:
    """Result of splitting an individual unit (or none, see next def)."""

    unit_id: int
    n_split: int
    train_indices: Tensor
    train_assignments: Tensor
    means: Tensor
    sub_proportions: Tensor
    bases: Tensor | None


UnitSplitResult = Optional[SuccessfulUnitSplitResult]


@databag
class SplitResult:
    """Final result of .split()"""

    any_split: bool
    n_new_units: int
    train_unit_mask: Tensor
    train_split_spike_mask: Tensor
    train_split_spike_labels: Tensor
    eval_split_spike_mask: Tensor | None
    eval_split_spike_labels: Tensor | None


@databag
class GroupPartition:
    """Represents a candidate grouping of some unit_ids into subsets."""

    unit_ids: Tensor
    group_ids: Tensor
    n_groups: int
    single_ixs: list[int]
    subset_ids: list[int]


@databag
class SuccessfulGroupMergeResult:
    """Object returned when merging a single group of units."""

    grouping: GroupPartition
    improvement: float
    train_assignments: Tensor
    train_indices: Tensor
    means: Tensor
    sub_proportions: Tensor
    bases: Tensor | None


# none means "accept the full model"
GroupMergeResult = Optional[SuccessfulGroupMergeResult]


@databag
class UnitRemapping:
    """Returned by merging and .cleanup()"""

    mapping: Tensor

    @classmethod
    def identity(cls, n_units) -> Self:
        return cls(mapping=torch.arange(n_units))

    def is_identity(self):
        return torch.equal(self.mapping, torch.arange(len(self.mapping)))

    def nuniq(self):
        return self.mapping[self.mapping >= 0].unique().numel()


@databag
class EMResult:
    elbos: list[float]


@databag
class SpikeDataBatch:
    """Yielded by data object .batches()

    # x, CmoCooinvx are not needed for soft_assign/score, so they can be None.
    """

    batch: Tensor | slice
    neighborhood_ids: Tensor
    xt: Tensor | None
    candidates: Tensor
    whitenedx: Tensor
    noise_logliks: Tensor
    CmoCooinvx: Tensor | None
    candidate_count: int | None


@databag
class DenseSpikeData:
    """Used in split/merge, like a batch but no candidates + extra logic useful for fitting."""

    indices: Tensor
    neighborhoods: SpikeNeighborhoods
    neighb_supset: Tensor
    neighborhood_ids: Tensor
    x: Tensor
    xt: Tensor
    whitenedx: Tensor
    CmoCooinvx: Tensor
    noise_logliks: Tensor
    batch_size: int

    def to_batches(
        self, unit_ids: Tensor, lut: NeighborhoodLUT
    ) -> list[SpikeDataBatch]:
        """Convert dense data to a batch, which means picking candidates.

        The candidates are all of the unit ids, so that each col of the candidates
        array is just unit_ids[col] repeated. The thing is that not all units overlap
        all neighborhoods, so some of these will be -1s per the lut. But we can still
        think of each col as corresponding to a unit.
        """
        unit_ids = torch.as_tensor(unit_ids, device=self.neighborhood_ids.device)
        covered = self.lut_coverage(unit_ids, lut)
        candidates = torch.where(covered, unit_ids[None, :], -1)
        candidate_counts = covered.sum(dim=1).cpu()

        batches = []
        for i0 in range(0, self.xt.shape[0], self.batch_size):
            sl = slice(i0, min(self.xt.shape[0], i0 + self.batch_size))
            b = SpikeDataBatch(
                batch=sl,
                neighborhood_ids=self.neighborhood_ids[sl],
                xt=self.xt[sl],
                whitenedx=self.whitenedx[sl],
                noise_logliks=self.noise_logliks[sl],
                CmoCooinvx=self.CmoCooinvx[sl],
                candidates=candidates[sl],
                candidate_count=int(candidate_counts[sl].sum()),
            )
            batches.append(b)
        return batches

    def lut_coverage(self, unit_ids: Tensor, lut: NeighborhoodLUT):
        lut_ixs = lut.lut[unit_ids[None, :], self.neighborhood_ids[:, None]]
        covered = lut_ixs < lut.unit_ids.shape[0]
        return covered

    def slice_by_coverage(self, unit_ids: Tensor, lut: NeighborhoodLUT):
        unit_ids = torch.as_tensor(unit_ids, device=self.neighborhood_ids.device)
        covered = self.lut_coverage(unit_ids, lut)
        (covered_inds,) = covered.any(dim=1).nonzero(as_tuple=True)
        if covered_inds.numel() == covered.shape[0]:
            return self, slice(None)
        return self.slice(covered_inds), covered_inds

    def slice(self, indices: Tensor):
        return DenseSpikeData(
            indices=self.indices[indices],
            neighborhoods=self.neighborhoods,
            neighb_supset=self.neighb_supset,
            neighborhood_ids=self.neighborhood_ids[indices],
            x=self.x[indices],
            xt=self.xt[indices],
            whitenedx=self.whitenedx[indices],
            CmoCooinvx=self.CmoCooinvx[indices],
            noise_logliks=self.noise_logliks[indices],
            batch_size=self.batch_size,
        )

    def covered_channels(self, min_count=0):
        counts = self.neighborhoods.b.indicators[:, self.neighborhood_ids].sum(1)
        if min_count:
            (channels,) = (counts >= min_count).nonzero(as_tuple=True)
        else:
            (channels,) = counts.nonzero(as_tuple=True)
        return channels

    def weighted_covered_channels(
        self, weights: Tensor, within: Tensor | None = None, min_count: int = 0
    ) -> tuple[list[Tensor], Tensor]:
        assert weights.shape[0] == self.xt.shape[0]
        assert weights.ndim == 2

        # spike_chan_inds: n_channels x n_spikes
        if within is None:
            spike_chan_inds = self.neighborhoods.b.indicators[:, self.neighborhood_ids]
        else:
            spike_chan_inds = self.neighborhoods.b.indicators[within][
                :, self.neighborhood_ids
            ]

        # counts: count on each channel for each unit, K x n_channels
        counts = weights.T @ spike_chan_inds.T
        unit_masks = counts >= min_count

        # subsets: covered channel sets for each unit, list of length K
        subsets = [m.nonzero(as_tuple=True)[0] for m in unit_masks]

        # covered spikes are which spikes live inside each subset
        spike_chan_counts = spike_chan_inds.sum(dim=0)
        chan_coverage = spike_chan_inds.T @ unit_masks.float().T
        # spikes x units
        full_coverage = chan_coverage >= spike_chan_counts[:, None]

        return subsets, full_coverage


class BatchedSpikeData:
    def __init__(
        self,
        *,
        N: int,
        n_candidates: int,
        n_search: int,
        n_explore: int,
        max_n_candidates: int,
        max_n_search: int,
        max_n_explore: int | None,
        neighborhoods: SpikeNeighborhoods,
        device: torch.device,
        candidates: Tensor | None = None,
        neighb_adj: Tensor | None = None,
        neighb_supset: Tensor | None = None,
        seed: int | np.random.Generator | None = 0,
        batch_size: int = 128,
        explore_neighb_steps: int = 1,
        neighb_overlap: float = 0.75,
        proposal_is_complete: bool = False,
    ):
        assert n_candidates <= max_n_candidates
        assert n_search <= max_n_search
        assert max_n_explore is None or n_explore <= max_n_explore
        n_total = n_candidates * (n_search + 1) + n_explore
        if max_n_explore is None:
            max_n_total = None
        else:
            max_n_total = max_n_candidates * (max_n_search + 1) + max_n_explore
            assert max_n_total >= n_total

        self.N = N
        self.batch_size = batch_size
        self.proposal_is_complete = proposal_is_complete

        self.candidates = candidates

        self.n_candidates = n_candidates
        self.n_search = n_search
        self.n_explore = n_explore
        self.max_n_candidates = max_n_candidates
        self.max_n_search = max_n_search
        self.max_n_explore = max_n_explore
        self.max_n_total = max_n_total
        self._update_slices()

        # adjacencies...
        self.neighborhoods = neighborhoods
        self.neighborhood_ids = neighborhoods.b.neighborhood_ids
        self.explore_neighb_steps = explore_neighb_steps
        if neighb_adj is None:
            self.neighb_adj = neighborhoods.adjacency(neighb_overlap)
        else:
            self.neighb_adj = neighb_adj
        if neighb_supset is None:
            self.neighb_supset = neighborhoods.partial_order().float()
        else:
            self.neighb_supset = neighb_supset
        nneighb = self.neighborhoods.n_neighborhoods
        assert self.neighb_supset.shape == (nneighb, nneighb)

        self.device = device
        if seed is None:
            self.rg = None
        else:
            self.rg = spawn_torch_rg(seed, device=device)
            assert self.rg.device == self.device

    def _update_sizes(self, n_candidates: int, n_search: int, n_explore: int):
        self.n_candidates = n_candidates
        self.n_search = n_search
        self.n_explore = n_explore
        self._update_slices()

    def _update_slices(self):
        self.n_search_total = self.n_candidates * self.n_search
        self.n_total = self.n_candidates + self.n_search_total + self.n_explore
        self.search_slice = slice(
            self.n_candidates, self.n_candidates + self.n_search_total
        )
        self.explore_slice = slice(
            self.n_candidates + self.n_search_total, self.n_total
        )

    def _update_sizes_from_n_units(self, n_units: int):
        n_candidates, n_search, n_explore, *_ = _pick_search_size(
            n_units=n_units,
            max_n_candidates=self.max_n_candidates,
            max_n_search=self.max_n_search,
            max_n_explore=self.max_n_explore,
        )
        self._update_sizes(n_candidates, n_search, n_explore)

    def batches(
        self, show_progress: bool = False, desc: str = "Batches"
    ) -> Iterable[SpikeDataBatch]:
        if show_progress:
            batch_starts = trange(0, self.N, self.batch_size, desc=desc)
        else:
            batch_starts = range(0, self.N, self.batch_size)
        for b, i0 in enumerate(batch_starts):
            batch = slice(i0, min(self.N, i0 + self.batch_size))
            yield self.batch(batch, batch_index=b)

    def batch(
        self, spike_indices: Tensor | slice, batch_index: int | None = None
    ) -> SpikeDataBatch:
        raise NotImplementedError

    def update_adjacency(
        self,
        n_units: int,
        un_adj_lut: NeighborhoodLUT | None = None,
        expand_lut: bool = False,
    ):
        """Update my adjacency either using my labels or just a fixed LUT."""
        self._update_sizes_from_n_units(n_units)
        if expand_lut:
            assert self.un_adj_lut is not None
            assert un_adj_lut is None
            expand_from_lut = self.un_adj_lut
        else:
            expand_from_lut = None
        if un_adj_lut is not None:
            assert un_adj_lut.lut.shape[0] == n_units
        if self.candidates is None:
            assert un_adj_lut is not None
            labels = None
        else:
            labels = self.candidates[:, 0]
        self.un_adj_lut, self.un_adj, self.explore_adj = candidate_adjacencies(
            labels=labels,
            neighb_supset=self.neighb_supset,
            neighb_adj=self.neighb_adj,
            n_units=n_units,
            explore_steps=self.explore_neighb_steps,
            neighborhoods=self.neighborhoods,
            neighborhood_ids=self.neighborhood_ids,
            un_adj_lut=un_adj_lut,
            expand_from_lut=expand_from_lut,
            device=self.device,
        )
        if pnoid:
            assert self.un_adj.max().item() == 1.0

    def erase_candidates(self):
        if self.candidates is not None:
            self.candidates.fill_(-1)

    def bootstrap_candidates(
        self, distances: Tensor, un_adj_lut: NeighborhoodLUT | None = None
    ) -> NeighborhoodLUT:
        self.update_adjacency(n_units=distances.shape[0], un_adj_lut=un_adj_lut)

        # fill in missing labels randomly, obeying un_adj
        if self.candidates is not None:
            assert self.rg is not None
            same_adj = _fill_blank_labels(
                labels=self.candidates[:, 0],
                un_adj=self.un_adj,
                explore_adj=self.explore_adj,
                neighb_adj=self.neighb_adj,
                neighborhood_ids=self.neighborhood_ids,
                gen=self.rg,
            )
            if not same_adj:
                logger.dartsortverbose(
                    "_fill_blank_labels needed to use explore adjacency."
                )
                self.update_adjacency(n_units=distances.shape[0])
            if pnoid:
                assert (self.candidates[:, 0] >= 0).all()
        if pnoid:
            assert torch.equal(
                self.un_adj,
                (self.un_adj_lut.lut < self.un_adj_lut.unit_ids.shape[0]).float(),
            )

        # fill in candidates[:, 1:n_candidates] at random obeying un_adj
        # choosing not to use distances here, since they get used in search sets
        if self.candidates is not None:
            assert self.rg is not None
            _bootstrap_top(
                candidates=self.candidates,
                neighborhood_ids=self.neighborhood_ids,
                n_candidates=self.n_candidates,
                un_adj=self.un_adj,
                un_adj_lut=self.un_adj_lut,
                gen=self.rg,
            )
            if pnoid:
                assert (self.candidates[:, 0] >= 0).all()

        # call update to fill in search + explore and build lut
        _, lut = self.update(new_top_candidates=None, distances=distances)
        return lut

    def update(
        self,
        new_top_candidates: Tensor | None,
        distances: Tensor,
        expand_lut: bool = False,
    ) -> tuple[bool, NeighborhoodLUT]:
        """Subclasses do what they need to do to update their candidates.

        Returns
        -------
        lut_changed : bool
        new_lut : NeighborhoodLUT
        """
        raise NotImplementedError

    def full_proposal_view(self, un_adj_lut: NeighborhoodLUT) -> Self:
        raise NotImplementedError


class StreamingSpikeData(BatchedSpikeData):
    """Like a TruncatedSpikeData, but only stores x, and transfers to device on the fly

    This also implements a different search strategy than the evolutionary search
    used in the TruncatedSpikeData. In that setting, we're evolving the posterior
    in the context of a truncated EM algorithm while we're still learning the parameters,
    so it needs to be adaptive. This class is used as a target for .soft_assign() below,
    implementing a likelihood computation while parameters are held fixed.

    In this setting, we don't want to use the usual "search sets" and explore sets.
    Actually, we want to propose every possible match (as determined by the LUT) and
    save out the best n_candidates matches. This is totally deterministic, there are
    no multinomials here.

    There's no need to save any state (persistent candidates array) here, then. They
    are always just all of the possible candidates, padded out with -1s for raggedness.
    Then score_batch() logic deals with grabbing the best n_candidates ones.
    """

    def __init__(
        self,
        *,
        n_candidates: int,
        max_n_candidates: int,
        x: Tensor,
        neighborhoods: SpikeNeighborhoods,
        device: torch.device,
        neighb_cov: NeighborhoodCovariance,
        batch_size: int = 512,
    ):
        # I must have explore adj == un_adj. Otherwise, update() should call
        # update_adjacency(). But this object is just designed to do soft
        # assignment given a fixed adjacency, so it doesn't.
        super().__init__(
            N=x.shape[0],
            n_candidates=n_candidates,
            n_search=0,
            n_explore=0,
            max_n_candidates=max_n_candidates,
            max_n_search=0,
            max_n_explore=None,
            batch_size=batch_size,
            neighborhoods=neighborhoods,
            explore_neighb_steps=0,
            neighb_overlap=0.0,
            seed=None,
            device=device,
            proposal_is_complete=True,
        )
        self.neighb_cov = neighb_cov
        self.x = x
        self.proposals = None

    def update(
        self,
        new_top_candidates: Tensor | None,
        distances: Tensor,
        expand_lut: bool = False,
    ) -> tuple[bool, NeighborhoodLUT]:
        # I only do candidates on the fly, so this does nothing.
        # it is an error to assign candidates which would lead to the LUT
        # changing, but that error cannot occur, because I would never
        # propose such candidates!
        assert distances.shape[0] == self.un_adj_lut.lut.shape[0]
        return False, self.un_adj_lut

    def update_adjacency(
        self,
        n_units: int,
        un_adj_lut: NeighborhoodLUT | None = None,
        expand_lut: bool = False,
    ):
        super().update_adjacency(
            n_units=n_units, un_adj_lut=un_adj_lut, expand_lut=expand_lut
        )
        # my invariant, see __init__
        assert torch.equal(self.un_adj, self.explore_adj)
        self.max_n_total = max_units_per_neighb(self.un_adj_lut)
        self._update_sizes_from_n_units(n_units)

        # which units can be proposed for each neighborhood? all of the ones in the LUT
        assert self.max_n_total is not None
        assert self.n_total == self.max_n_total
        proposals = full_proposal_by_neighb(self.un_adj_lut, self.n_total)
        self.proposals = proposals.to(self.device)

    def _update_sizes_from_n_units(self, n_units: int):
        if self.max_n_total is None:
            self.max_n_total = n_units
        assert self.max_n_total <= n_units
        n_candidates = min(self.max_n_candidates, self.max_n_total)
        n_explore = self.max_n_total - n_candidates
        self._update_sizes(n_candidates, 0, n_explore)

    def batch(
        self, spike_indices: Tensor | slice, batch_index: int | None = None
    ) -> SpikeDataBatch:
        neighb_ids = self.neighborhood_ids[spike_indices]
        assert self.proposals is not None
        candidates = self.proposals[neighb_ids]
        x = self.x[spike_indices].to(self.device)
        wx, noise_loglik = _whiten_and_noise_score_batch(
            x=x, neighb_ids=neighb_ids, neighb_cov=self.neighb_cov
        )
        # my x is not channels-major as the TruncatedSpikeData's is, and that's
        # what's assumed in the statistics pass.
        del x
        return SpikeDataBatch(
            batch=spike_indices,
            xt=None,
            candidates=candidates,
            neighborhood_ids=neighb_ids,
            whitenedx=wx,
            noise_logliks=noise_loglik,
            CmoCooinvx=None,
            candidate_count=None,
        )

    def full_proposal_view(self, un_adj_lut: NeighborhoodLUT):
        self.update_adjacency(n_units=un_adj_lut.lut.shape[0], un_adj_lut=un_adj_lut)
        return self


class TruncatedSpikeData(BatchedSpikeData):
    def __init__(
        self,
        *,
        n_candidates: int,
        n_search: int,
        n_explore: int,
        max_n_candidates: int,
        max_n_search: int,
        max_n_explore: int,
        dense_slice_size_per_unit: int,
        x: Tensor,
        xt: Tensor,
        whitenedx: Tensor,
        CmoCooinvx: Tensor,
        noise_logliks: Tensor,
        neighborhoods: SpikeNeighborhoods,
        seed: int | np.random.Generator = 0,
        batch_size: int = 128,
        explore_neighb_steps: int = 1,
        neighb_overlap: float = 0.75,
    ):
        """Data holder and candidate component manager for truncated EM

        This class collaborates with the TruncatedMixtureModel in its em() method.

        INVARIANT: candidates[:, n_candidates] is ordered by descending topness.
         - labels = candidates[:, 0]
         - top = candidates[:, n_candidates]
         - search = candidates[:, n_candidates:n_candidates + n_search*n_candidates]
         - explore = candidates[:, n_candidates + n_search*n_candidates:]
                   = candidates[:, -n_explore:]
        """
        device = noise_logliks.device
        super().__init__(
            N=x.shape[0],
            n_candidates=n_candidates,
            n_search=n_search,
            n_explore=n_explore,
            max_n_candidates=max_n_candidates,
            max_n_search=max_n_search,
            max_n_explore=max_n_explore,
            batch_size=batch_size,
            neighborhoods=neighborhoods,
            explore_neighb_steps=explore_neighb_steps,
            neighb_overlap=neighb_overlap,
            seed=seed,
            device=device,
        )
        assert self.max_n_explore is not None
        assert self.max_n_total is not None
        self._candidates_full = torch.full(
            (x.shape[0], self.max_n_total), -1, device=device
        )
        self.candidates = self._candidates_full[:, : self.n_total]
        assert self.candidates.shape[1] >= n_candidates

        assert noise_logliks.shape == (self.N,)
        assert x.device == neighborhoods.neighborhood_ids.device == device
        assert whitenedx.shape == x.shape
        assert neighborhoods.neighborhood_ids.shape == (self.N,)

        self.x = x
        self.xt = xt
        self.whitenedx = whitenedx
        self.CmoCooinvx = CmoCooinvx
        self.noise_logliks = noise_logliks
        self.dense_slice_size_per_unit = dense_slice_size_per_unit

        n_batches = len(range(0, self.N, self.batch_size))
        self.batch_candidate_counts = torch.zeros(n_batches, dtype=torch.long)

    def _update_sizes(self, n_candidates: int, n_search: int, n_explore: int):
        super()._update_sizes(n_candidates, n_search, n_explore)
        assert self.n_candidates <= self.max_n_candidates
        assert self.n_search <= self.max_n_search
        assert self.max_n_explore is not None
        assert self.n_explore <= self.max_n_explore
        assert self.max_n_total is not None
        assert self.n_total <= self.max_n_total
        assert self.n_total <= self._candidates_full.shape[1]
        self.candidates = self._candidates_full[:, : self.n_total]

    @classmethod
    def initialize_from_labels(
        cls,
        n_candidates: int,
        n_search: int,
        n_explore: int,
        max_n_candidates: int,
        max_n_search: int,
        max_n_explore: int,
        dense_slice_size_per_unit: int,
        labels: Tensor,
        x: Tensor,
        neighborhoods: SpikeNeighborhoods,
        neighb_cov: NeighborhoodCovariance,
        seed: int | np.random.Generator = 0,
        explore_neighb_steps: int = 1,
        neighb_overlap: float = 0.75,
        batch_size: int = 128,
    ) -> Self:
        assert len(x) == len(neighborhoods.b.neighborhood_ids)
        xt, whitenedx, CmoCooinvx, noise_logliks = _whiten_impute_and_noise_score(
            x=x, neighborhoods=neighborhoods, neighb_cov=neighb_cov
        )
        self = cls(
            n_candidates=n_candidates,
            n_search=n_search,
            n_explore=n_explore,
            max_n_candidates=max_n_candidates,
            max_n_search=max_n_search,
            max_n_explore=max_n_explore,
            dense_slice_size_per_unit=dense_slice_size_per_unit,
            x=x,
            xt=xt,
            whitenedx=whitenedx,
            CmoCooinvx=CmoCooinvx,
            noise_logliks=noise_logliks,
            neighborhoods=neighborhoods,
            seed=seed,
            batch_size=batch_size,
            explore_neighb_steps=explore_neighb_steps,
            neighb_overlap=neighb_overlap,
        )
        self.candidates[:, 0] = labels
        return self

    def update(
        self,
        new_top_candidates: Tensor | None,
        distances: Tensor,
        expand_lut: bool = False,
    ) -> tuple[bool, NeighborhoodLUT]:
        self._update_sizes_from_n_units(distances.shape[0])
        # fill in top spots
        if new_top_candidates is not None:
            self.candidates[:, : self.n_candidates] = new_top_candidates
            self.update_adjacency(distances.shape[0], expand_lut=expand_lut)

        lut_padded = F.pad(
            self.un_adj_lut.lut, (0, 0, 0, 1), value=self.un_adj_lut.unit_ids.shape[0]
        )
        top_lut_ixs = lut_padded[
            self.candidates[:, : self.n_candidates], self.neighborhood_ids[:, None]
        ]

        # fill in search neighbors with topk of distances, constrained by adj
        search_sets = candidate_search_sets(
            distances, self.un_adj_lut, self.un_adj, self.n_search
        )
        torch.take_along_dim(
            search_sets[:, None, :],
            dim=0,
            indices=top_lut_ixs[:, :, None],
            out=self.candidates[:, self.search_slice].view(
                self.N, self.n_candidates, self.n_search
            ),
        )

        # fill in explore set randomly using explore adjacency
        if self.n_explore:
            assert self.rg is not None
            p, inds = _get_explore_sampling_data(
                un_adj_lut=self.un_adj_lut,
                explore_adj=self.explore_adj,
                search_sets=search_sets,
                n_explore=self.n_explore,
            )
            _sample_explore_candidates(
                p=p,
                inds=inds,
                candidates=self.candidates,
                lut_ixs=top_lut_ixs[:, 0],
                n_explore=self.n_explore,
                gen=self.rg,
            )

        # finally replace any duplicates with -1
        _dedup_candidates(self.candidates)

        # update counts
        _count_candidates(self.candidates, self.batch_candidate_counts, self.batch_size)

        # update lut for caller
        lut = lut_from_candidates_and_neighborhoods(
            candidates=self.candidates,
            neighborhoods=self.neighborhoods,
            neighborhood_ids=self.neighborhood_ids,
            neighb_supset=self.neighb_supset,
            n_units=distances.shape[0],
        )
        if expand_lut:
            lut = combine_luts(self.un_adj_lut, lut)
        return True, lut

    def erase_candidates(self):
        self.candidates.fill_(-1)
        self.batch_candidate_counts.zero_()

    def batch(
        self, spike_indices: Tensor | slice, batch_index: int | None = None
    ) -> SpikeDataBatch:
        candidates = self.candidates[spike_indices]
        return SpikeDataBatch(
            batch=spike_indices,
            neighborhood_ids=self.neighborhood_ids[spike_indices],
            xt=self.xt[spike_indices],
            candidates=candidates,
            CmoCooinvx=self.CmoCooinvx[spike_indices],
            whitenedx=self.whitenedx[spike_indices],
            noise_logliks=self.noise_logliks[spike_indices],
            candidate_count=int(self.batch_candidate_counts[batch_index]),
        )

    def dense_slice(self, spike_indices: Tensor) -> DenseSpikeData:
        return DenseSpikeData(
            indices=spike_indices,
            neighborhoods=self.neighborhoods,
            neighborhood_ids=self.neighborhood_ids[spike_indices],
            neighb_supset=self.neighb_supset,
            x=self.x[spike_indices],
            xt=self.xt[spike_indices],
            whitenedx=self.whitenedx[spike_indices],
            CmoCooinvx=self.CmoCooinvx[spike_indices],
            noise_logliks=self.noise_logliks[spike_indices],
            batch_size=self.batch_size,
        )

    def dense_slice_by_unit(
        self, unit_ids: Tensor | int | None, gen: torch.Generator, min_count: int = 0
    ):
        assert unit_ids is not None
        unit_ids = torch.as_tensor(unit_ids, device=self.candidates.device)
        if unit_ids.ndim == 0:
            mask = self.candidates[:, 0] == unit_ids
        else:
            mask = torch.isin(self.candidates[:, 0], unit_ids)

        (ixs,) = mask.nonzero(as_tuple=True)
        if min_count and ixs.numel() < min_count:
            return None
        if (nixs := ixs.numel()) > self.dense_slice_size_per_unit:
            perm = torch.randperm(nixs, generator=gen, device=ixs.device)
            ixs = ixs[perm[: self.dense_slice_size_per_unit]]
            ixs = torch.msort(ixs)

        return self.dense_slice(ixs)

    def remap(self, remapping: UnitRemapping, distances: Tensor) -> NeighborhoodLUT:
        """Re-map my top candidate labels and re-do LUTs, search, explore."""
        n_units_orig = remapping.mapping.shape[0]
        assert distances.shape[0] <= n_units_orig
        assert remapping.mapping.max() + 1 == distances.shape[0]
        self._update_sizes_from_n_units(distances.shape[0])

        # extra -1 for the -1 candidates to index into
        mapping = F.pad(remapping.mapping, (0, 1), value=-1)
        mapping = mapping.to(device=self.candidates.device)
        mapping = mapping[None].broadcast_to((self.N, mapping.shape[0]))

        # replace my -1s with n_units_orig in place bc take_along_dim doesn't like -1s
        F.threshold(
            self.candidates[:, : self.n_candidates], -1, n_units_orig, inplace=True
        )

        # remap in place
        torch.take_along_dim(
            mapping,
            self.candidates[:, : self.n_candidates],
            dim=1,
            out=self.candidates[:, : self.n_candidates],
        )
        self.candidates[:, self.n_candidates :].fill_(-1)
        self.update_adjacency(distances.shape[0])
        _, lut = self.update(new_top_candidates=None, distances=distances)
        return lut

    def update_from_split(
        self,
        unit_mask: Tensor,
        split_mask: Tensor,
        split_labels: Tensor,
        distances: Tensor,
    ) -> NeighborhoodLUT:
        """Erase candidates for spikes in the split, replace labels, re-bootstrap."""
        self.candidates[:, 0].masked_fill_(unit_mask, -1)
        (split_ix,) = split_mask.nonzero(as_tuple=True)
        self.candidates[split_ix, 0] = split_labels[split_ix]

        # have to do a full bootstrap, bc it's hard to figure out what to do with
        # spikes whose candidates contain the units that were split. this way, the
        # lut invariants are maintained, and at least the top labels are the same.
        return self.bootstrap_candidates(distances)

    def full_proposal_view(self, un_adj_lut: NeighborhoodLUT):
        return FullProposalDataView.from_truncated_spike_data(self, un_adj_lut)


class FullProposalDataView(BatchedSpikeData):
    def __init__(self, n_explore: int, data: TruncatedSpikeData, proposals: Tensor):
        super().__init__(
            N=data.N,
            n_candidates=data.n_candidates,
            n_search=0,
            n_explore=n_explore,
            max_n_candidates=data.n_candidates,
            max_n_search=0,
            max_n_explore=n_explore,
            neighborhoods=data.neighborhoods,
            device=data.device,
            candidates=None,
            neighb_adj=data.neighb_adj,
            neighb_supset=data.neighb_supset,
            seed=None,
            batch_size=data.batch_size,
            explore_neighb_steps=0,
            neighb_overlap=0.0,
            proposal_is_complete=True,
        )
        self.proposals = proposals.to(self.device)
        self.data = data

    @classmethod
    def from_truncated_spike_data(
        cls, data: TruncatedSpikeData, un_adj_lut: NeighborhoodLUT
    ) -> Self:
        n_proposed = max_units_per_neighb(un_adj_lut)
        n_proposed = max(data.n_candidates, n_proposed)
        n_explore = n_proposed - data.n_candidates
        proposals = full_proposal_by_neighb(un_adj_lut, n_proposed)
        self = cls(n_explore=n_explore, data=data, proposals=proposals)
        return self

    def update(
        self,
        new_top_candidates: Tensor | None,
        distances: Tensor,
        expand_lut: bool = False,
    ) -> tuple[bool, NeighborhoodLUT]:
        raise ValueError("View doesn't update.")

    def update_adjacency(
        self,
        n_units: int,
        un_adj_lut: NeighborhoodLUT | None = None,
        expand_lut: bool = False,
    ):
        raise ValueError("View doesn't update.")

    def batch(
        self, spike_indices: Tensor | slice, batch_index: int | None = None
    ) -> SpikeDataBatch:
        neighb_ids = self.neighborhood_ids[spike_indices]
        candidates = self.proposals[neighb_ids]
        return SpikeDataBatch(
            batch=spike_indices,
            neighborhood_ids=self.neighborhood_ids[spike_indices],
            xt=self.data.xt[spike_indices],
            candidates=candidates,
            CmoCooinvx=self.data.CmoCooinvx[spike_indices],
            whitenedx=self.data.whitenedx[spike_indices],
            noise_logliks=self.data.noise_logliks[spike_indices],
            candidate_count=None,
        )


class BaseMixtureModel(BModule):
    noise_log_prop: Tensor

    def __init__(
        self,
        *,
        max_group_size: int,
        max_distance: float,
        unit_ids: Tensor,
        distance_kind: Literal["cosine"] = "cosine",
        signal_rank: int,
        neighb_cov: NeighborhoodCovariance,
        noise: EmbeddedNoise,
        erp: NeighborhoodInterpolator,
        em_iters: int,
        prior_pseudocount: float,
        criterion_em_iters: int,
        cl_alpha: float,
        min_channel_count: int = 1,
        elbo_atol: float,
    ):
        super().__init__()
        self.distance_kind = distance_kind
        self.max_group_size = max_group_size
        self.min_channel_count = min_channel_count
        self.max_distance = max_distance
        self.unit_ids = unit_ids
        self.n_units = unit_ids.shape[0]
        self.signal_rank = signal_rank
        self.erp = erp
        self.neighb_cov = neighb_cov
        self.noise = noise
        self.em_iters = em_iters
        self.criterion_em_iters = criterion_em_iters
        self.prior_pseudocount = prior_pseudocount
        self.cl_alpha = cl_alpha
        self.elbo_atol = elbo_atol

    # -- subclasses implement

    def unit_slice(self, unit_ids: Tensor) -> "TMMView":
        raise NotImplementedError

    def get_params_at(self, indices: Tensor) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError

    def get_lut(self) -> NeighborhoodLUT:
        raise NotImplementedError

    @property
    def centroids(self) -> Tensor:
        """Interface used by unit_distance_matrix."""
        raise NotImplementedError

    def non_noise_log_proportion(self):
        raise NotImplementedError

    def score(self, data: DenseSpikeData, *, skip_noise: bool = False) -> Scores:
        raise NotImplementedError

    # -- shared logic

    def group_units_for_merge(self) -> Iterable[Tensor]:
        distances = self.unit_distance_matrix()
        return tree_groups(
            distances,
            max_group_size=self.max_group_size,
            max_distance=self.max_distance,
        )

    def merge_as_group(
        self,
        train_data: DenseSpikeData,
        eval_data: DenseSpikeData | None,
        pair_mask: Tensor | None,
        cur_scores: Scores,
        cur_unit_ids: Tensor,
        responsibilities: Tensor | None,
        skip_full: bool,
        skip_single: bool,
    ) -> GroupMergeResult:
        """Find an optimal partition of this model's units as a whole."""
        return brute_merge(
            self,
            train_data=train_data,
            eval_data=eval_data,
            pair_mask=pair_mask,
            cur_scores=cur_scores,
            cur_unit_ids=cur_unit_ids,
            responsibilities=responsibilities,
            skip_full=skip_full,
            skip_single=skip_single,
        )

    def unit_distance_matrix(self) -> Tensor:
        if self.distance_kind == "cosine":
            return cosine_distance(self.centroids)
        else:
            assert False


class TruncatedMixtureModel(BaseMixtureModel):
    LP_MIN = -1000.0

    def __init__(
        self,
        *,
        max_group_size: int,
        max_distance: float,
        split_k: int,
        log_proportions: Tensor,
        means: Tensor,
        bases: Tensor | None,
        noise_log_prop: Tensor | None,
        neighb_cov: NeighborhoodCovariance,
        noise: EmbeddedNoise,
        erp: NeighborhoodInterpolator,
        em_iters: int,
        criterion_em_iters: int,
        min_count: int,
        prior_pseudocount: float,
        cl_alpha: float,
        full_proposal_every: int = 10,
        distance_kind: Literal["cosine"] = "cosine",
        lut_puff: float = 1.5,
        seed: int | np.random.Generator | torch.Generator = 0,
        min_channel_count: int = 1,
        elbo_atol: float,
        min_em_iters: int = 1,
    ):
        super().__init__(
            distance_kind=distance_kind,
            max_group_size=max_group_size,
            max_distance=max_distance,
            min_channel_count=min_channel_count,
            unit_ids=torch.arange(means.shape[0], device=log_proportions.device),
            signal_rank=bases.shape[2] if bases is not None else 0,
            erp=erp,
            neighb_cov=neighb_cov,
            noise=noise,
            em_iters=em_iters,
            criterion_em_iters=criterion_em_iters,
            prior_pseudocount=prior_pseudocount,
            cl_alpha=cl_alpha,
            elbo_atol=elbo_atol,
        )
        self.min_count = min_count
        self.split_k = split_k
        self.lut_puff = lut_puff
        self.full_proposal_every = full_proposal_every
        if noise_log_prop is not None:
            assert (
                log_proportions.untyped_storage().data_ptr
                != noise_log_prop.untyped_storage().data_ptr
            )
        self.register_buffer("log_proportions", log_proportions)
        self.register_buffer_or_none("noise_log_prop", noise_log_prop)
        self.register_buffer("means", means)
        self.register_buffer_or_none("bases", bases)
        assert self.b.means.is_contiguous()
        if bases is not None:
            assert self.b.bases.is_contiguous()
        self.eps = torch.tensor(torch.finfo(means.dtype).tiny, device=means.device)
        self.min_em_iters = min_em_iters

        # needs to be initialized before doing anything serious
        # see bootstrapping stage in main function below
        self.lut = NeighborhoodLUT(
            unit_ids=torch.arange(0),
            neighb_ids=torch.arange(0),
            lut=torch.arange(0)[None],
        )

        if bases is not None:
            assert bases.shape[0] == self.n_units
            assert bases.shape[2] == means.shape[1]
            self.signal_rank = bases.shape[1]
        else:
            self.signal_rank = 0

        self.rg = spawn_torch_rg(seed, device=means.device)
        self.lut_params = None

    def get_lut(self):
        return self.lut

    @classmethod
    def initialize_from_data_with_labels(
        cls,
        *,
        neighb_cov: NeighborhoodCovariance,
        max_group_size: int,
        max_distance: float,
        signal_rank: int,
        split_k: int,
        data: TruncatedSpikeData,
        noise: EmbeddedNoise,
        erp: NeighborhoodInterpolator,
        em_iters: int,
        criterion_em_iters: int,
        min_count: int,
        seed: int | np.random.Generator,
        prior_pseudocount: float,
        cl_alpha: float,
        full_proposal_every: int = 10,
        distance_kind: Literal["cosine"] = "cosine",
        min_channel_count: int = 1,
        elbo_atol: float = 1e-4,
    ) -> Self:
        """Constructor for the full mixture model, called by from_config()"""
        rg = spawn_torch_rg(seed, device=neighb_cov.obs_ix.device)
        log_props, noise_log_prop, means, bases = initialize_parameters_by_unit(
            data=data,
            signal_rank=signal_rank,
            noise=noise,
            erp=erp,
            gen=rg,
            min_channel_count=min_channel_count,
            prior_pseudocount=prior_pseudocount,
            puff=split_k,
        )
        return cls(
            neighb_cov=neighb_cov,
            erp=erp,
            max_group_size=max_group_size,
            max_distance=max_distance,
            distance_kind=distance_kind,
            log_proportions=log_props,
            split_k=split_k,
            means=means,
            bases=bases,
            noise_log_prop=noise_log_prop,
            min_count=min_count,
            min_channel_count=min_channel_count,
            seed=rg,
            noise=noise,
            em_iters=em_iters,
            criterion_em_iters=criterion_em_iters,
            prior_pseudocount=prior_pseudocount,
            full_proposal_every=full_proposal_every,
            elbo_atol=elbo_atol,
            cl_alpha=cl_alpha,
        )

    @classmethod
    def from_config(
        cls,
        *,
        noise: EmbeddedNoise,
        erp: NeighborhoodInterpolator,
        neighb_cov: NeighborhoodCovariance,
        train_data: TruncatedSpikeData,
        seed: int | np.random.Generator,
        refinement_cfg: RefinementConfig,
    ) -> Self:
        assert refinement_cfg.search_type == "topk"
        assert refinement_cfg.merge_decision_algorithm == "brute"
        assert refinement_cfg.split_decision_algorithm == "brute"
        # TODO: remove all unused refinement_cfg parameters
        return cls.initialize_from_data_with_labels(
            noise=noise,
            erp=erp,
            neighb_cov=neighb_cov,
            seed=seed,
            max_group_size=refinement_cfg.merge_group_size,
            data=train_data,
            max_distance=refinement_cfg.merge_distance_threshold,
            signal_rank=refinement_cfg.signal_rank,
            min_count=refinement_cfg.min_count,
            split_k=refinement_cfg.kmeansk,
            min_channel_count=refinement_cfg.channels_count_min,
            em_iters=refinement_cfg.n_em_iters,
            criterion_em_iters=refinement_cfg.criterion_em_iters,
            elbo_atol=refinement_cfg.em_converged_atol,
            prior_pseudocount=refinement_cfg.prior_pseudocount,
            full_proposal_every=refinement_cfg.full_proposal_every,
            cl_alpha=refinement_cfg.cl_alpha,
        )

    @classmethod
    def initialize_from_dense_data_with_fixed_responsibilities(
        cls,
        *,
        signal_rank: int,
        erp: NeighborhoodInterpolator,
        min_count: int,
        min_channel_count: int,
        data: DenseSpikeData,
        responsibilities: Tensor,
        noise: EmbeddedNoise,
        max_group_size: int,
        max_distance: float,
        neighb_cov: NeighborhoodCovariance,
        min_iter: int,
        max_iter: int,
        total_log_proportion: float,
        prior_pseudocount: float,
        cl_alpha: float,
        elbo_atol: float,
        noise_log_prop: Tensor | float = -torch.inf,
    ) -> tuple[Self, Tensor, DenseSpikeData, bool, Tensor, Tensor]:
        """Fit units with fixed label posterior

        Used to construct hypothetical models:
         - Fitting split units based on k-means weights
         - Fitting combinations of split or main model units to evaluate
           the brute force merge criterion
        """
        initialization, chan_coverage = initialize_params_from_dense_data(
            data=data,
            rank=signal_rank,
            erp=erp,
            min_channel_count=min_channel_count,
            noise=noise,
            weights=responsibilities,
            prior_pseudocount=prior_pseudocount,
        )
        assert chan_coverage is not None
        valid = torch.tensor([r is not None for r in initialization])
        initialization = [r for r in initialization if r is not None]
        responsibilities = responsibilities[:, valid]
        K = responsibilities.shape[1]

        fdim = noise.rank * noise.n_channels
        means = data.xt.new_zeros((K, fdim))
        if signal_rank:
            bases = data.xt.new_zeros((K, signal_rank, fdim))
        else:
            bases = None
        basis_reshape = signal_rank, noise.rank, noise.n_channels
        for k, (schans, mean, basis) in enumerate(initialization):
            means[k].view(noise.rank, noise.n_channels)[:, schans] = mean
            if signal_rank:
                assert bases is not None
                assert basis is not None
                bases[k].view(basis_reshape)[:, :, schans] = basis

        # deliberately not normalizing the proportions here
        # these were historically always fixed from the get
        # normalizing would be bad when we have overlapping subsets here.
        log_proportions = responsibilities.mean(0).log_()
        log_proportions += total_log_proportion

        self = cls(
            max_group_size=max_group_size,
            max_distance=max_distance,
            min_count=min_count,
            min_channel_count=min_channel_count,
            split_k=0,
            log_proportions=log_proportions,
            means=means,
            bases=bases,
            noise_log_prop=torch.as_tensor(noise_log_prop),
            neighb_cov=neighb_cov,
            noise=noise,
            erp=erp,
            em_iters=max_iter,
            criterion_em_iters=min_iter,
            elbo_atol=elbo_atol,
            prior_pseudocount=prior_pseudocount,
            cl_alpha=cl_alpha,
        )

        # subset data to coverage
        keep_mask = (responsibilities > 0).logical_and_(chan_coverage)
        keep_mask = keep_mask.any(dim=1)
        (keep_spikes,) = keep_mask.nonzero(as_tuple=True)
        any_spikes_discarded = keep_spikes.numel() < responsibilities.shape[0]
        if any_spikes_discarded:
            responsibilities = responsibilities[keep_spikes]
            data = data.slice(keep_spikes)

        # get lut, which will never change
        candidates = torch.where(responsibilities > 0, self.unit_ids[None, :], -1)
        if pnoid:
            assert (responsibilities > 0).any(dim=1).all()
            assert (candidates >= 0).any(dim=1).all()
        lut = lut_from_candidates_and_neighborhoods(
            candidates=candidates,
            neighborhoods=data.neighborhoods,
            neighborhood_ids=data.neighborhood_ids,
            neighb_supset=data.neighb_supset,
            n_units=self.n_units,
        )
        self.update_lut(lut)

        # run some em steps
        self.fixed_weight_em(data=data, responsibilities=responsibilities)
        return self, valid, data, any_spikes_discarded, keep_mask, keep_spikes

    def em(self, data: TruncatedSpikeData, show_progress: int = 1):
        assert self.lut_params is not None
        if show_progress:
            iters = trange(self.em_iters, desc="EM")
        else:
            iters = range(self.em_iters)
        elbos = []
        for j in iters:
            if pnoid:
                self.lut_params.check()
            if j and self.full_proposal_every and not j % self.full_proposal_every:
                step_data = data.full_proposal_view(self.lut)
            else:
                step_data = data
            eres = self.e_step(step_data, show_progress=show_progress > 2)
            elb = eres.stats.elbo.cpu().item()
            assert math.isfinite(elb)
            elbos.append(elb)
            if show_progress:
                iters.set_description(f"EM(elbo={elb:.3f})")  # type: ignore
            self.m_step(eres.stats)
            lut_changed, lut = data.update(eres.candidates, self.unit_distance_matrix())
            del lut_changed  # doesn't matter if it did or not, my parameters changed
            self.update_lut(lut)
            if j > self.min_em_iters and elbos[-1] - elbos[-2] < self.elbo_atol:
                break

        if logger.isEnabledFor(DARTSORTDEBUG) and len(elbos) > 1:
            if len(elbos) > 8:
                a = ", ".join(f"{x:0.4f}" for x in elbos[:4])
                b = ", ".join(f"{x:0.4f}" for x in elbos[-4:])
                elbstr = a + " ... " + b
            else:
                elbstr = ", ".join(f"{x:0.4f}" for x in elbos)
            begend = elbos[-1] - elbos[0]
            smalldif = np.diff(elbos).min()
            bigdiff = np.diff(elbos).max()
            logger.info(
                f"EM elbos={elbstr}, with end-start={begend:0.3f} and biggest/smallest "
                f"diffs {bigdiff:0.3f} and {smalldif:0.3f}. Ran for {len(elbos)} / "
                f"{self.em_iters} iterations."
            )
        assert not self.em_iters or elbos[-1] > elbos[0] - 1e-3
        return EMResult(elbos=elbos)

    def e_step(
        self,
        data: TruncatedSpikeData | FullProposalDataView,
        show_progress: bool = False,
    ) -> TruncatedEStepResult:
        assert self.lut_params is not None
        candidates = torch.empty(
            (data.N, data.n_candidates), dtype=torch.long, device=data.device
        )
        stats = SufficientStatistics.zeros(
            n_units=self.n_units,
            n_lut=self.lut_params.n_lut,
            n_channels=self.neighb_cov.n_channels,
            feature_rank=self.neighb_cov.feat_rank,
            signal_rank=self.signal_rank,
            device=self.b.means.device,
            skip_noise=False,
        )
        for batch in data.batches(show_progress=show_progress, desc="E"):
            batch_scores = self.score_batch(batch, data.n_candidates)
            stats.combine(self.estep_stats_batch(batch, batch_scores), eps=self.eps)
            candidates[batch.batch] = batch_scores.candidates
        _finalize_e_stats(
            means=self.b.means,
            bases=self.b.bases,
            stats=stats,
            lut=self.lut,
            lut_params=self.lut_params,
            neighb_cov=self.neighb_cov,
            prior_pseudocount=self.prior_pseudocount,
        )
        return TruncatedEStepResult(candidates=candidates, stats=stats)

    def m_step(self, stats: SufficientStatistics, skip_proportions: bool = False):
        # proportions
        if not skip_proportions:
            assert stats.noise_N is not None
            lp = torch.concatenate([stats.N, stats.noise_N[None]])
            lp = F.log_softmax(lp.log_(), dim=0).clamp_(min=self.LP_MIN)
            self.b.log_proportions.copy_(lp[:-1])
            self.b.noise_log_prop.fill_(lp[-1])
            if pnoid:
                assert self.b.log_proportions.isfinite().all()
                assert self.b.noise_log_prop.isfinite().all()
        else:
            assert stats.noise_N is None

        # rank 0 case: mean only
        if self.signal_rank == 0:
            assert stats.R.shape[1] == 1
            self.b.means.copy_(stats.R[:, 0])
            if pnoid:
                assert self.b.means.isfinite().all()
            return

        # solve mean and basis together
        U = _get_u_from_ulut(self.lut, stats)
        if self.prior_pseudocount:
            denom = stats.N.clamp_(min=self.eps)
            tikh = self.prior_pseudocount / denom
            U.diagonal(dim1=-2, dim2=-1).add_(tikh[:, None])
        soln = torch.linalg.solve(U, stats.R)
        self.b.means.copy_(soln[:, -1])
        self.b.bases.copy_(soln[:, :-1])

        if pnoid:
            assert soln.isfinite().all()
        if not skip_proportions:
            assert lp.isfinite().all()  # type: ignore

    def fixed_weight_em(self, data: DenseSpikeData, responsibilities: Tensor):
        assert self.lut_params is not None
        batches = data.to_batches(self.unit_ids, self.lut)
        elbos = []
        j = -1
        assert responsibilities.shape[1] == batches[0].candidates.shape[1]
        stats = SufficientStatistics.zeros(
            n_units=self.n_units,
            n_lut=self.lut_params.n_lut,
            n_channels=self.neighb_cov.n_channels,
            feature_rank=self.neighb_cov.feat_rank,
            signal_rank=self.signal_rank,
            device=self.b.means.device,
            skip_noise=True,
        )
        # make use of the fixed sparsity structure, avoiding implicit nonzero()s below
        batch_sparse_ixs = []
        for batch in batches:
            spike_ixs, candidate_ixs, unit_ixs, neighb_ixs = _sparsify_candidates(
                batch.candidates,
                batch.neighborhood_ids,
                static_size=batch.candidate_count,
            )
            lut_ixs = self.lut.lut[unit_ixs, neighb_ixs]
            batch_sparse_ixs.append(
                (spike_ixs, candidate_ixs, unit_ixs, neighb_ixs, lut_ixs)
            )
        for j in range(self.em_iters):
            for batch, spixs in zip(batches, batch_sparse_ixs):
                spike_ixs, candidate_ixs, unit_ixs, neighb_ixs, lut_ixs = spixs
                bresp = responsibilities[batch.batch]
                if j >= self.criterion_em_iters:
                    # compute log liks for convergence testing below
                    batch_scores = self.score_batch(
                        batch=batch,
                        n_candidates=batch.candidates.shape[1],
                        fixed_responsibilities=bresp,
                        skip_responsibility=True,
                        skip_noise=True,
                        spike_ixs=spike_ixs,
                        candidate_ixs=candidate_ixs,
                        unit_ixs=unit_ixs,
                        neighb_ixs=neighb_ixs,
                        lut_ixs=lut_ixs,
                    )
                else:
                    # too soon to test convergence, no need for log liks
                    batch_scores = Scores(
                        log_liks=bresp,  # unused, just to fill the field.
                        responsibilities=bresp,
                        candidates=batch.candidates,
                    )
                batch_stats = self.estep_stats_batch(
                    batch=batch,
                    scores=batch_scores,
                    spike_ixs=spike_ixs,
                    candidate_ixs=candidate_ixs,
                    unit_ixs=unit_ixs,
                    neighb_ixs=neighb_ixs,
                    lut_ixs=lut_ixs,
                )
                stats.combine(batch_stats, eps=self.eps)
            _finalize_e_stats(
                means=self.b.means,
                bases=self.b.bases,
                stats=stats,
                lut=self.lut,
                lut_params=self.lut_params,
                neighb_cov=self.neighb_cov,
            )
            self.m_step(stats, skip_proportions=True)
            self.update_lut(self.lut)

            if j >= self.criterion_em_iters:
                elb = stats.elbo
                assert elb.isfinite()
                elbos.append(elb.cpu().item())
            if j >= self.criterion_em_iters + 1:
                if elbos[-1] - elbos[-2] < self.elbo_atol:
                    break
            stats.reset_(
                n_units=self.n_units,
                n_channels=self.neighb_cov.n_channels,
                feature_rank=self.neighb_cov.feat_rank,
                signal_rank=self.signal_rank,
            )

        if logger.isEnabledFor(DARTSORTVERBOSE) and len(elbos):
            begend = elbos[-1] - elbos[0]
            delbos = np.diff(elbos)
            smalldif = delbos.min()
            bigdiff = delbos.max()
            logger.dartsortdebug(
                f"Fixed fit elbo end-start={begend:0.4f} over {j + 1} iterations, "
                f"biggest and smallest diffs {bigdiff:0.4f} and {smalldif:0.4f}."
            )

        # Since responsibilities are fixed, it's possible (I think) for the
        # support of the likelihoods to drift away slightly from the weights',
        # and this can lead to slightly negative elbo changes. giving a little
        # more leeway than in em() here for when to panic about it.
        assert (len(elbos) < 2) or (elbos[-1] > elbos[0] - 5e-2)

    def score(self, data: DenseSpikeData, *, skip_noise: bool = False) -> Scores:
        scores = []
        for batch in data.to_batches(self.unit_ids, self.lut):
            batch_scores = self.score_batch(
                batch=batch,
                n_candidates=batch.candidates.shape[1],
                skip_responsibility=True,
                skip_noise=skip_noise,
            )
            scores.append(batch_scores)
        return concatenate_scores(scores)

    def score_batch(
        self,
        batch: SpikeDataBatch,
        n_candidates: int,
        fixed_responsibilities: Tensor | None = None,
        skip_responsibility: bool = False,
        skip_noise: bool = False,
        *,
        spike_ixs: Tensor | None = None,
        candidate_ixs: Tensor | None = None,
        unit_ixs: Tensor | None = None,
        neighb_ixs: Tensor | None = None,
        lut_ixs: Tensor | None = None,
    ) -> Scores:
        assert self.lut_params is not None
        return _score_batch(
            candidates=batch.candidates,
            neighborhood_ids=batch.neighborhood_ids,
            n_candidates=n_candidates,
            whitenedx=batch.whitenedx,
            noise_logliks=batch.noise_logliks,
            log_proportions=self.b.log_proportions,
            noise_log_prop=self.b.noise_log_prop,
            lut_params=self.lut_params,
            lut=self.lut,
            fixed_responsibilities=fixed_responsibilities,
            skip_responsibility=skip_responsibility,
            spike_ixs=spike_ixs,
            candidate_ixs=candidate_ixs,
            unit_ixs=unit_ixs,
            neighb_ixs=neighb_ixs,
            lut_ixs=lut_ixs,
            static_size=batch.candidate_count,
            skip_noise=skip_noise,
        )

    def estep_stats_batch(
        self,
        batch: SpikeDataBatch,
        scores: Scores,
        *,
        spike_ixs: Tensor | None = None,
        candidate_ixs: Tensor | None = None,
        unit_ixs: Tensor | None = None,
        neighb_ixs: Tensor | None = None,
        lut_ixs: Tensor | None = None,
        static_size: int | None = None,
    ):
        assert self.lut_params is not None
        assert batch.CmoCooinvx is not None
        assert batch.xt is not None
        assert scores.responsibilities is not None
        if spike_ixs is None:
            spike_ixs, candidate_ixs, unit_ixs, neighb_ixs = _sparsify_candidates(
                scores.candidates, batch.neighborhood_ids, static_size=static_size
            )
            lut_ixs = self.lut.lut[unit_ixs, neighb_ixs]
        else:
            assert candidate_ixs is not None
            assert unit_ixs is not None
            assert neighb_ixs is not None
            assert lut_ixs is not None
        return _stat_pass_batch(
            xt=batch.xt,
            whitenedx=batch.whitenedx,
            CmoCooinvx=batch.CmoCooinvx,
            responsibilities=scores.responsibilities,
            log_liks=scores.log_liks,
            spike_ixs=spike_ixs,
            candidate_ixs=candidate_ixs,
            unit_ixs=unit_ixs,
            neighb_ixs=neighb_ixs,
            lut_ixs=lut_ixs,
            n_candidates=scores.candidates.shape[1],
            n_units=self.n_units,
            neighb_cov=self.neighb_cov,
            lut_params=self.lut_params,
            eps=self.eps,
        )

    def soft_assign(
        self,
        *,
        data: BatchedSpikeData,
        needs_bootstrap: bool,
        full_proposal_view: bool,
        max_iter: int = 10,
        show_progress: int = 0,
    ) -> Scores:
        """Run E steps until candidates converge, holding my LUT fixed."""
        # start by telling the data how to search
        distances = self.unit_distance_matrix()
        if full_proposal_view:
            data.update_adjacency(n_units=self.n_units, un_adj_lut=self.lut)
            target_data = data.full_proposal_view(self.lut)
        elif needs_bootstrap:
            # TODO cut this?
            data.erase_candidates()
            lut = data.bootstrap_candidates(distances=distances, un_adj_lut=self.lut)
            assert lut == self.lut
            # self.update_lut(lut, no_parameter_changes=True)
            target_data = data
        else:
            target_data = data

        # initialize storage for output on data's device
        if data.candidates is None:
            dev = torch.device("cpu")
        else:
            dev = data.candidates.device
        scores = Scores(
            log_liks=torch.empty((data.N, data.n_candidates + 1), device=dev),
            responsibilities=torch.empty((data.N, data.n_candidates + 1), device=dev),
            candidates=torch.full((data.N, data.n_candidates), -1, device=dev),
        )
        assert scores.responsibilities is not None
        assert max_iter >= 1
        if show_progress > 1 and not data.proposal_is_complete:
            iters = trange(max_iter, desc="SoftAssign")
            show_batch_progress = show_progress > 2 or data.proposal_is_complete
        else:
            iters = range(max_iter)
            show_batch_progress = False
        for it in iters:
            for batch in target_data.batches(show_progress=show_batch_progress):
                batch_scores = self.score_batch(batch, data.n_candidates)
                assert batch_scores.responsibilities is not None
                bix = batch.batch
                scores.candidates[bix] = batch_scores.candidates.to(dev)
                scores.log_liks[bix] = batch_scores.log_liks.to(dev)
                scores.responsibilities[bix] = batch_scores.responsibilities.to(dev)

            lut_changed, lut = data.update(
                scores.candidates, distances, expand_lut=True
            )
            if lut_changed:
                # it can't actually change.
                assert lut == self.lut
            if target_data.proposal_is_complete:
                break
            assert data.candidates is not None
            is_final = it == max_iter - 1
            if not is_final and torch.equal(
                scores.candidates[:, : data.n_candidates],
                data.candidates[:, : data.n_candidates],
            ):
                logger.dartsortdebug(f"Soft assign converged at iteration {it}.")
                break

        return scores

    def update_lut(self, lut: NeighborhoodLUT, no_parameter_changes: bool = False):
        if no_parameter_changes and self.lut == lut:
            return
        self.lut = lut
        if self.lut_params is None:
            self.lut_params = LUTParams.from_lut(
                self.b.means, self.b.bases, self.neighb_cov, lut, puff=self.lut_puff
            )
        else:
            self.lut_params.update(self.b.means, self.b.bases, self.neighb_cov, lut)

    def split_unit(
        self,
        unit_id: int | Tensor,
        train_data: TruncatedSpikeData,
        eval_data: TruncatedSpikeData | None,
        scores: Scores,
    ) -> UnitSplitResult:
        # get dense train set slice in unit_id
        min_count_split = 2 * self.min_count
        split_data = train_data.dense_slice_by_unit(
            unit_id, gen=self.rg, min_count=min_count_split
        )
        if split_data is None:
            return None

        # kmeans on interp whitened feats
        assert self.erp is not None
        kmeans_responsibliities = try_kmeans(
            split_data,
            k=self.split_k,
            erp=self.erp,
            gen=self.rg,
            feature_rank=self.noise.rank,
            min_count=self.min_count,
        )
        if kmeans_responsibliities is None:
            logger.dartsortverbose(f"Split {unit_id}: kmeans bailed.")
            return None
        assert kmeans_responsibliities.shape[1] >= 2

        # initialize dense model with fixed resps
        split_model, _, split_data, any_spikes_discarded, keep_mask, keep_spikes = (
            TruncatedMixtureModel.initialize_from_dense_data_with_fixed_responsibilities(
                data=split_data,
                responsibilities=kmeans_responsibliities,
                signal_rank=self.signal_rank,
                erp=self.erp,
                min_count=self.min_count,
                min_channel_count=self.min_channel_count,
                noise=self.noise,
                max_group_size=self.max_group_size,
                max_distance=self.max_distance,
                neighb_cov=self.neighb_cov,
                min_iter=self.criterion_em_iters,
                max_iter=self.em_iters,
                elbo_atol=self.elbo_atol,
                prior_pseudocount=self.prior_pseudocount,
                cl_alpha=self.cl_alpha,
                total_log_proportion=self.b.log_proportions[unit_id].item(),
                noise_log_prop=self.b.noise_log_prop,
            )
        )
        if any_spikes_discarded:
            kmeans_responsibliities = kmeans_responsibliities[keep_spikes]
        if split_model.n_units <= 1:
            logger.dartsortverbose(
                f"Split {unit_id}: only {split_model.n_units} of {kmeans_responsibliities.shape[1]} sub-units."
            )
            return None

        # see if that dog hunts
        if eval_data is None:
            split_eval_data = None
            assert scores.log_liks.shape[0] == train_data.N
            cur_scores_batch = scores.slice(split_data.indices)
        else:
            split_eval_data = eval_data.dense_slice_by_unit(unit_id, gen=self.rg)
            assert split_eval_data is not None
            assert scores.log_liks.shape[0] == eval_data.N
            cur_scores_batch = scores.slice(split_eval_data.indices)

        pair_mask = split_model.unit_distance_matrix() <= self.max_distance
        merge_res = split_model.merge_as_group(
            train_data=split_data,
            eval_data=split_eval_data,
            pair_mask=pair_mask,
            cur_scores=cur_scores_batch,
            cur_unit_ids=torch.as_tensor(unit_id, device=train_data.x.device),
            responsibilities=kmeans_responsibliities,
            skip_full=False,
            skip_single=False,
        )
        if merge_res is not None and merge_res.grouping.n_groups <= 1:
            logger.dartsortverbose(
                f"Split {unit_id}: merged all {split_model.n_units} split sub-units."
            )
            return None
        if merge_res is not None and merge_res.improvement <= 0.0:
            logger.dartsortverbose(
                f"Split {unit_id}: mini-merge improvement {merge_res.improvement} too small."
            )
            return None

        no_allowed_partitions = not pair_mask.fill_diagonal_(False).any()
        if no_allowed_partitions:
            assert merge_res is None
            # units are too separated to merge. just assign by scoring.
            train_scores = split_model.score(split_data)
            n_candidates = train_scores.candidates.shape[1]
            assert n_candidates > 0
            cix = train_scores.log_liks[:, :n_candidates].argmax(1, keepdim=True)
            train_labels = train_scores.candidates.take_along_dim(cix, dim=1)[:, 0]
            train_indices = split_data.indices
            n_groups = split_model.n_units
            means = split_model.b.means
            bases = split_model.b.bases
            sub_proportions = F.softmax(split_model.b.log_proportions, dim=0)
        elif merge_res is None:
            logger.dartsortverbose(
                f"Split {unit_id}: no split, merge failed with allowed partitions."
            )
            return None
        else:
            train_labels = merge_res.train_assignments
            train_indices = merge_res.train_indices
            n_groups = merge_res.grouping.n_groups
            means = merge_res.means
            bases = merge_res.bases
            sub_proportions = merge_res.sub_proportions

        if logger.isEnabledFor(DARTSORTVERBOSE):
            _l, _c = train_labels.unique(return_counts=True)
            imp = None if merge_res is None else merge_res.improvement
            logger.dartsortverbose(
                f"Split {unit_id}: {n_groups} parts with improvement {imp}, "
                f"assigned to {_l.tolist()} with counts {_c.tolist()}."
            )

        return SuccessfulUnitSplitResult(
            unit_id=int(unit_id),
            n_split=n_groups,
            train_indices=train_indices,
            train_assignments=train_labels,
            means=means,
            bases=bases,
            sub_proportions=sub_proportions,
        )

    def split(
        self,
        train_data: TruncatedSpikeData,
        eval_data: TruncatedSpikeData | None,
        scores: Scores,
        show_progress: bool = True,
    ) -> SplitResult:
        if show_progress:
            unit_ids = tqdm(self.unit_ids, desc="Split", smoothing=0.0)
        else:
            unit_ids = self.unit_ids
        split_results = (
            self.split_unit(unit_id, train_data, eval_data, scores)
            for unit_id in unit_ids
        )

        split_result = self._apply_unit_splits(
            split_results, train_data=train_data, eval_data=eval_data
        )

        # split generates a lot of weird small buffers
        gc.collect()
        torch.cuda.empty_cache()

        return split_result

    def merge(
        self,
        train_data: TruncatedSpikeData,
        eval_data: TruncatedSpikeData | None,
        scores: Scores,
        show_progress: bool = True,
    ) -> UnitRemapping:
        """Break into subgroups and merge those. This will also modify data objects."""
        result_map = UnitRemapping.identity(self.n_units)
        any_merged = False

        groups = self.group_units_for_merge()
        logger.dartsortverbose("Merge groups: %s.", groups)
        if show_progress:
            groups = tqdm(groups, desc="Merge", smoothing=0.0)

        for group in groups:
            if group.numel() == 1:
                continue
            assert group.numel() <= self.max_group_size

            view = self.unit_slice(group)
            group_train_data = train_data.dense_slice_by_unit(group, gen=self.rg)
            assert group_train_data is not None
            if pnoid:
                cov = group_train_data.lut_coverage(view.unit_ids, self.lut)
                assert cov.any(dim=1).all()
            if eval_data is None:
                group_eval_data = None
                assert scores.log_liks.shape[0] == train_data.N
                group_scores = scores.slice(group_train_data.indices)
            else:
                group_eval_data = eval_data.dense_slice_by_unit(group, gen=self.rg)
                assert group_eval_data is not None
                assert scores.log_liks.shape[0] == eval_data.N
                group_scores = scores.slice(group_eval_data.indices)
            group_res = view.merge_as_group(
                train_data=group_train_data,
                eval_data=group_eval_data,
                pair_mask=None,
                cur_scores=group_scores,
                responsibilities=None,
                cur_unit_ids=torch.as_tensor(group, device=train_data.x.device),
                skip_full=False,
                skip_single=False,
            )
            del group_train_data, group_eval_data

            # no-merge cases
            if group_res is None:
                continue
            if logger.isEnabledFor(DARTSORTVERBOSE):
                logger.dartsortverbose(
                    "Group %s best partition %s had improvement %s.",
                    group.tolist(),
                    group_res.grouping.group_ids.tolist(),
                    group_res.improvement,
                )
            if group_res.improvement <= 0:
                continue
            if group_res.grouping.n_groups == group.numel():
                continue
            any_merged = True

            groups = group_res.grouping.group_ids.unique()
            groups = groups[groups >= 0]
            assert groups.numel() == group_res.grouping.n_groups

            # log(new_props = props[group].sum() * sub_props)
            group_log_prop = self.b.log_proportions[group].logsumexp(dim=0)
            new_log_props = group_res.sub_proportions.log().add_(group_log_prop)
            assert group_res.sub_proportions.shape == (group_res.grouping.n_groups,)
            assert group_res.means.shape[0] == group_res.grouping.n_groups

            for gix, g in enumerate(groups):
                (in_group,) = (group_res.grouping.group_ids == g).nonzero(as_tuple=True)
                ids_in_group = group[in_group.to(device=group.device)]
                first = ids_in_group[0]
                rest = ids_in_group[1:]

                result_map.mapping[ids_in_group] = ids_in_group[0]

                # first is the group, rest are discarded. first retains id for now.
                # first gets the parameter update
                self.b.log_proportions[first] = new_log_props[gix]
                self.b.means[first] = group_res.means[gix]
                if self.signal_rank:
                    assert group_res.bases is not None
                    self.b.bases[first] = group_res.bases[gix]

                # rest are poisoned
                self.b.log_proportions[rest] = -torch.inf
                self.b.means[rest].fill_(torch.nan)
                if self.signal_rank:
                    self.b.bases[rest].fill_(torch.nan)

        assert any_merged != result_map.is_identity()
        if not any_merged:
            return result_map

        # fix up my parameters, update the datasets, and update the lut
        flat_map = self.cleanup(result_map)
        distances = self.unit_distance_matrix()
        if pnoid:
            assert distances.isfinite().all()
        lut = train_data.remap(remapping=flat_map, distances=distances)
        self.update_lut(lut)
        assert self.lut_params is not None
        if pnoid:
            self.lut_params.check()

        # merge generates a lot of weird small buffers
        gc.collect()
        torch.cuda.empty_cache()

        return flat_map

    def get_params_at(self, indices: Tensor):
        m = self.b.means[indices]
        if self.signal_rank:
            b = self.b.bases[indices]
        else:
            b = None
        return m, b

    def reset_noise_log_prop(self):
        tmp = self.b.log_proportions.new_empty((self.n_units + 1,))
        tmp[: self.n_units] = self.b.log_proportions
        tmp[-1] = tmp[: self.n_units].logsumexp(dim=0) - torch.log(
            torch.tensor(float(self.n_units))
        )
        tmp = F.log_softmax(tmp, dim=0)
        self.b.log_proportions.copy_(tmp[: self.n_units])
        self.b.noise_log_prop.copy_(tmp[-1])

    def non_noise_log_proportion(self):
        return self.b.log_proportions.logsumexp(dim=0)

    @property
    def centroids(self) -> Tensor:
        """Interface used by unit_distance_matrix."""
        return self.b.means

    def unit_slice(self, unit_ids: Tensor) -> "TMMView":
        return TMMView(self, unit_ids)

    def cleanup(self, remapping: UnitRemapping) -> UnitRemapping:
        """
        The returned mapping still has the same number of elements as the input
        (which is the original number of units), but this function flattens the
        space of positive ids.

        So, all entries of remapping which map to the same id will now map to the
        same flattened ID.

        This function also cleans up parameters. The new unit count will be
        the number of unique positive mapping targets of remapping. The parameters
        in each resulting unit will be taken from the first appearance of each
        remapped ID.
        """
        # torch doesn't have return_index
        uniq_remapped_ids, uniq_first_inds = np.unique(
            remapping.mapping.cpu(), return_index=True
        )
        uniq_remapped_ids = torch.from_numpy(uniq_remapped_ids)
        uniq_first_inds = torch.from_numpy(uniq_first_inds)
        uniq_first_inds = uniq_first_inds[uniq_remapped_ids >= 0]
        uniq_remapped_ids = uniq_remapped_ids[uniq_remapped_ids >= 0]

        # figure out flat mapping this is filled out in the for loop
        # (could probably use the unique inverse if you had your thinking cap on)
        new_n_units = uniq_remapped_ids.numel()
        new_ids = torch.arange(new_n_units)
        new_remapping = UnitRemapping(mapping=torch.full_like(remapping.mapping, -1))

        # rearrange parameters and delete parameters for destroyed units
        if (uniq_first_inds >= new_ids).all():
            # since we read later indices than are being written, in place is fine
            assert torch.equal(uniq_first_inds.sort().values, uniq_first_inds)
            assert torch.equal(new_ids, torch.arange(new_n_units))
            for old_id, new_id in zip(uniq_first_inds, new_ids):
                assert old_id >= new_id
                new_remapping.mapping[remapping.mapping == old_id] = new_id
                if old_id == new_id:
                    continue

                self.b.means[new_id] = self.b.means[old_id]
                self.b.log_proportions[new_id] = self.b.log_proportions[old_id]
                if self.signal_rank:
                    self.b.bases[new_id] = self.b.bases[old_id]

                self.b.means[old_id].fill_(torch.nan)
                self.b.log_proportions[old_id] = -torch.inf
                if self.signal_rank:
                    self.b.bases[old_id].fill_(torch.nan)

            # double check to fix up log props. should exp-sum to 1-noiseprop
            logsum = self.b.log_proportions.logsumexp(dim=0)
            targsum = torch.log(1.0 - self.b.noise_log_prop.exp())
            assert torch.isclose(logsum, targsum, atol=1e-6)
            self.b.log_proportions.add_(targsum - logsum)
        else:
            # this case is really only hit in testing. i'm asserting that this is
            # a permutation, because that's all i've got implemented
            # new_ids is sorted, uniq_first_inds not
            assert torch.equal(uniq_first_inds.sort().values, new_ids)
            assert torch.equal(new_ids, uniq_remapped_ids)
            new_remapping.mapping.copy_(remapping.mapping)
            self.b.means[new_ids] = self.b.means[uniq_first_inds]
            self.b.log_proportions[new_ids] = self.b.log_proportions[uniq_first_inds]
            if self.signal_rank:
                self.b.bases[new_ids] = self.b.bases[uniq_first_inds]
            self.b.log_proportions[new_n_units:] = -torch.inf

        assert self.b.log_proportions[new_n_units:].isneginf().all()

        logger.dartsortdebug("Cleanup %s -> %s.", self.n_units, new_n_units)

        self.n_units = new_n_units
        self.unit_ids = new_ids
        self.b.means.resize_(new_n_units, *self.b.means.shape[1:])
        self.b.log_proportions.resize_(new_n_units, *self.b.log_proportions.shape[1:])
        if self.signal_rank:
            self.b.bases.resize_(new_n_units, *self.b.bases.shape[1:])

        # final checks (historically easy for me to get this remapping stuff wrong)
        logsum = self.b.log_proportions.logsumexp(dim=0)
        targsum = torch.log(1.0 - self.b.noise_log_prop.exp())
        assert torch.isclose(logsum, targsum, atol=1e-6)
        assert self.b.means[:, 0].isfinite().all()
        if self.signal_rank:
            assert self.b.bases[:, 0, 0].isfinite().all()

        return new_remapping

    def _apply_unit_splits(
        self,
        unit_splits: Iterable[UnitSplitResult],
        train_data: TruncatedSpikeData,
        eval_data: TruncatedSpikeData | None,
    ):
        any_split = False
        train_unit_mask = torch.zeros_like(
            train_data.candidates[:, 0], dtype=torch.bool
        )
        train_split_mask = torch.zeros_like(
            train_data.candidates[:, 0], dtype=torch.bool
        )
        train_labels = torch.full_like(train_data.candidates[:, 0], -1)
        if eval_data is not None:
            eval_unit_mask = torch.zeros_like(
                eval_data.candidates[:, 0], dtype=torch.bool
            )
            eval_split_mask = torch.zeros_like(
                eval_data.candidates[:, 0], dtype=torch.bool
            )
            eval_labels = torch.full_like(eval_data.candidates[:, 0], -1)
        else:
            eval_mask = eval_labels = None

        n_new_units = 0
        new_means = []
        new_log_props = []
        new_bases = []
        cur_max_label = self.n_units - 1

        for res in unit_splits:
            if res is None:
                logger.dartsortverbose("Split early exit.")
                continue
            if res.n_split <= 1:
                logger.dartsortverbose("Didn't split %s.", res.unit_id)
                continue

            logger.dartsortverbose("Applying %s split.", res.unit_id)
            unit_id = res.unit_id
            any_split = True

            # invalidate this unit
            train_unit_mask.logical_or_(train_data.candidates[:, 0] == unit_id)

            # spikes which will get new labels
            train_split_mask[res.train_indices] = True

            # assign labels within that subset
            # # split id 0 gets unit_id
            for_current = res.train_assignments == 0
            train_labels[res.train_indices[for_current]] = unit_id

            # rest are new ids. these split ids start at 1, so add K-1 rather than K
            # for the first new id to be K (which is the next label).
            for_other = res.train_assignments > 0
            other_labels = res.train_assignments[for_other].add_(cur_max_label)
            train_labels[res.train_indices[for_other]] = other_labels

            n_new_units += res.n_split - 1
            cur_max_label += res.n_split - 1

            # divvy up my log proportion
            if pnoid:
                _lp = res.sub_proportions.sum()
                assert torch.isclose(_lp, torch.ones_like(_lp))
            split_log_props = (
                res.sub_proportions.log() + self.b.log_proportions[unit_id]
            )

            # assign split result first params to unit_id's spot
            assert res.means.shape[0] == res.n_split
            assert res.sub_proportions.shape == (res.n_split,)
            self.b.log_proportions[unit_id] = split_log_props[0]
            self.b.means[unit_id] = res.means[0]
            if self.signal_rank:
                assert res.bases is not None
                self.b.bases[unit_id] = res.bases[0]

            # append rest to new_* lists
            new_means.append(res.means[1:])
            new_log_props.append(split_log_props[1:])
            if self.signal_rank:
                new_bases.append(res.bases[1:])  # type: ignore
            else:
                new_bases.append(None)

        assert cur_max_label + 1 == self.n_units + n_new_units
        assert train_labels.max() <= cur_max_label
        assert n_new_units == sum(m.shape[0] for m in new_means)
        assert n_new_units == sum(lp.numel() for lp in new_log_props)
        if self.signal_rank:
            assert n_new_units == sum(b.shape[0] for b in new_bases)  # type: ignore

        # resize params to allow space for the new guys
        Korig = self.n_units
        Knew = Korig + n_new_units
        logger.dartsortdebug(
            f"Split created {n_new_units} new units ({Korig} -> {Knew})."
        )
        self.b.means.resize_(Knew, *self.b.means.shape[1:])
        self.b.log_proportions.resize_(Knew, *self.b.log_proportions.shape[1:])
        if self.signal_rank:
            self.b.bases.resize_(Knew, *self.b.bases.shape[1:])

        # update other important things
        self.n_units = Knew
        self.unit_ids = torch.arange(Knew)

        # assign them (can just loop)
        k0 = Korig
        for nm, nlp, nb in zip(new_means, new_log_props, new_bases):
            newk = nlp.numel()
            self.b.means[k0 : k0 + newk] = nm
            self.b.log_proportions[k0 : k0 + newk] = nlp
            if nb is not None:
                self.b.bases[k0 : k0 + newk] = nb
            k0 += newk
        assert k0 == Knew
        if pnoid:
            _lp = torch.logaddexp(self.noise_log_prop, self.non_noise_log_proportion())
            assert torch.isclose(_lp, torch.zeros(()), atol=1e-6)
            assert self.b.means.isfinite().all()
            assert self.b.log_proportions.isfinite().all()
            if self.signal_rank:
                assert self.b.bases.isfinite().all()

        # call data methods for processing splits
        distances = self.unit_distance_matrix()
        if pnoid:
            assert distances.isfinite().all()
        lut = train_data.update_from_split(
            train_unit_mask, train_split_mask, train_labels, distances
        )
        self.update_lut(lut)
        assert self.lut_params is not None
        if pnoid:
            self.lut_params.check()

        return SplitResult(
            any_split=any_split,
            n_new_units=n_new_units,
            train_unit_mask=train_unit_mask,
            train_split_spike_mask=train_split_mask,
            train_split_spike_labels=train_labels,
            eval_split_spike_mask=None,  # TODO: use it or lose it.
            eval_split_spike_labels=None,
        )


class TMMView(BaseMixtureModel):
    def __init__(self, tmm: TruncatedMixtureModel, unit_ids: Tensor):
        super().__init__(
            max_distance=tmm.max_distance,
            max_group_size=tmm.max_group_size,
            unit_ids=torch.as_tensor(unit_ids, device=tmm.unit_ids.device),
            distance_kind=tmm.distance_kind,  # type: ignore
            min_channel_count=tmm.min_channel_count,
            signal_rank=tmm.signal_rank,
            erp=tmm.erp,
            neighb_cov=tmm.neighb_cov,
            noise=tmm.noise,
            em_iters=tmm.em_iters,
            criterion_em_iters=tmm.criterion_em_iters,
            prior_pseudocount=tmm.prior_pseudocount,
            cl_alpha=tmm.cl_alpha,
            elbo_atol=tmm.elbo_atol,
        )
        self.tmm = tmm

    def get_params_at(self, indices: Tensor):
        # need this one!
        m = self.tmm.b.means[self.unit_ids[indices]]
        if self.signal_rank:
            b = self.tmm.b.bases[self.unit_ids[indices]]
        else:
            b = None
        return m, b

    def get_lut(self) -> NeighborhoodLUT:
        return self.tmm.lut

    @property
    def centroids(self):
        return self.tmm.b.means[self.unit_ids]

    def non_noise_log_proportion(self):
        return self.tmm.b.log_proportions[self.unit_ids].logsumexp(dim=0)

    @property
    def noise_log_prop(self) -> Tensor:  # type: ignore
        return self.tmm.b.noise_log_prop

    def score(self, data: DenseSpikeData, *, skip_noise: bool = False) -> Scores:
        scores = []
        for batch in data.to_batches(self.unit_ids, self.tmm.lut):
            batch_scores = self.tmm.score_batch(
                batch=batch,
                n_candidates=batch.candidates.shape[1],
                skip_responsibility=True,
                skip_noise=skip_noise,
            )
            scores.append(batch_scores)
        return concatenate_scores(scores)


class TMMStack(BaseMixtureModel):
    def __init__(self, tmms: list[TruncatedMixtureModel]):
        assert len(tmms) > 0

        # build a unit id lookup table
        # it is assumed that input models ids are linear, and this one
        # makes its own linear space too
        sub_model_unit_ids = []
        sub_model_inds = []
        self.start_ixs = []
        n_so_far = 0
        for j, tmm in enumerate(tmms):
            if pnoid:
                assert torch.equal(tmm.unit_ids.cpu(), torch.arange(len(tmm.unit_ids)))
            sub_model_unit_ids.append(tmm.unit_ids)
            sub_model_inds.append(torch.full_like(tmm.unit_ids, j))
            self.start_ixs.append(n_so_far)
            n_so_far += tmm.n_units
        self.sub_model_unit_ids = torch.concatenate(sub_model_unit_ids)
        self.sub_model_inds = torch.concatenate(sub_model_inds)
        self.unit_ids = torch.arange(
            self.sub_model_unit_ids.shape[0], device=self.sub_model_unit_ids.device
        )

        super().__init__(
            max_distance=tmms[0].max_distance,
            max_group_size=tmms[0].max_group_size,
            unit_ids=self.unit_ids,
            distance_kind=tmm.distance_kind,  # type: ignore
            min_channel_count=tmms[0].min_channel_count,
            signal_rank=tmms[0].signal_rank,
            erp=tmms[0].erp,
            neighb_cov=tmms[0].neighb_cov,
            noise=tmms[0].noise,
            em_iters=tmms[0].em_iters,
            criterion_em_iters=tmms[0].criterion_em_iters,
            prior_pseudocount=tmms[0].prior_pseudocount,
            cl_alpha=tmms[0].cl_alpha,
            elbo_atol=tmms[0].elbo_atol,
        )
        self.tmms = tmms

    # this is used intermediately only intermediately in brute_merge,
    # and only these methods are called

    def get_lut(self):
        unit_ids = []
        neighb_ids = []
        for tmm, start_ix in zip(self.tmms, self.start_ixs):
            unit_ids.append(tmm.lut.unit_ids + start_ix)
            neighb_ids.append(tmm.lut.neighb_ids)
        unit_ids = torch.concatenate(unit_ids)
        neighb_ids = torch.concatenate(neighb_ids)
        lut = torch.full(
            (self.n_units, self.tmms[0].lut.lut.shape[1]),
            unit_ids.shape[0],
            device=self.tmms[0].lut.lut.device,
        )
        lut[unit_ids, neighb_ids] = torch.arange(unit_ids.shape[0], device=lut.device)
        return NeighborhoodLUT(unit_ids=unit_ids, neighb_ids=neighb_ids, lut=lut)

    def get_params_at(self, indices: Tensor):
        if not len(indices):
            m = torch.empty_like(self.tmms[0].b.means[:0])
            if self.signal_rank:
                b = torch.empty_like(self.tmms[0].b.bases[:0])
            else:
                b = None
            return m, b

        sub_model_inds = self.sub_model_inds[indices]
        model_rel_inds = self.sub_model_unit_ids[indices]
        means = []
        bases = []
        for j in sub_model_inds.unique():
            (in_j,) = (sub_model_inds == j).nonzero(as_tuple=True)
            inds_j = model_rel_inds[in_j]
            means.append(self.tmms[j].b.means[inds_j])
            if self.signal_rank:
                bases.append(self.tmms[j].b.bases[inds_j])
        m = torch.concatenate(means)
        if self.signal_rank:
            b = torch.concatenate(bases)
        else:
            b = None
        return m, b

    def score(self, data: DenseSpikeData, *, skip_noise: bool = False) -> Scores:
        assert skip_noise
        scores = [tmm.score(data, skip_noise=skip_noise) for tmm in self.tmms]

        # stack the scores with unit ids remapped
        candidates = self.unit_ids[None].broadcast_to(
            scores[0].candidates.shape[0], self.n_units
        )
        candidates = candidates.contiguous()
        i0 = 0
        for score in scores:
            i1 = i0 + score.candidates.shape[1]
            assert score.candidates.shape[1] == score.log_liks.shape[1]
            assert score.responsibilities is None
            candidates[:, i0:i1].masked_fill_(score.candidates < 0, -1)
            i0 = i1
        log_liks = torch.concatenate([score.log_liks for score in scores], dim=1)
        return Scores(log_liks=log_liks, candidates=candidates, responsibilities=None)


# -- helpers


def get_truncated_datasets(
    *, sorting, motion_est, refinement_cfg, device, rg, noise=None, stable_data=None
):
    assert sorting.labels is not None
    labels = torch.tensor(sorting.labels, device=device)

    # we assume that the core neighborhoods are exactly the same as extract ones
    assert refinement_cfg.core_radius == "extract"
    data = get_full_neighborhood_data(
        sorting=sorting,
        motion_est=motion_est,
        rg=rg,
        refinement_cfg=refinement_cfg,
        stable_data=stable_data,
        device=device,
    )
    full_features, full_neighbs, train_ixs, train_neighbs, val_ixs, val_neighbs, prgeom = data  # fmt: skip
    del stable_data

    if noise is None:
        noise = EmbeddedNoise.estimate_from_hdf5(
            sorting.parent_h5_path,
            motion_est=motion_est,
            zero_radius=refinement_cfg.cov_radius,
            cov_kind=refinement_cfg.cov_kind,
            glasso_alpha=refinement_cfg.glasso_alpha,
            interp_params=refinement_cfg.noise_interp_params,
            device=device,
            rgeom=prgeom[:-1],
        )
    assert isinstance(noise, EmbeddedNoise)
    noise.to(device=device)
    neighb_cov = NeighborhoodCovariance.from_noise_and_neighborhoods(
        prgeom=prgeom,
        noise=noise,
        neighborhoods=full_neighbs,
    )
    n_candidates, n_search, n_explore, max_candidates, max_search, max_explore = (
        _pick_search_size(n_units=sorting.n_units, refinement_cfg=refinement_cfg)
    )
    train_data = TruncatedSpikeData.initialize_from_labels(
        n_candidates=n_candidates,
        n_search=n_search,
        n_explore=n_explore,
        max_n_candidates=max_candidates,
        max_n_search=max_search,
        max_n_explore=max_explore,
        dense_slice_size_per_unit=refinement_cfg.n_spikes_fit,
        labels=labels[train_ixs].to(device=device),
        x=full_features[train_ixs].to(device=device),
        neighborhoods=train_neighbs,
        neighb_cov=neighb_cov,
        neighb_overlap=refinement_cfg.neighb_overlap,
        explore_neighb_steps=refinement_cfg.explore_neighb_steps,
        seed=rg,
        batch_size=refinement_cfg.train_batch_size,
    )
    val_n_candidates = min(
        sorting.n_units, max(refinement_cfg.merge_group_size, n_candidates)
    )
    max_val_n_candidates = max(val_n_candidates, max_candidates)
    if refinement_cfg.criterion.startswith("heldout"):
        assert val_ixs is not None
        assert val_neighbs is not None
        val_data = TruncatedSpikeData.initialize_from_labels(
            n_candidates=val_n_candidates,
            n_search=n_search,
            n_explore=0,
            max_n_candidates=max_val_n_candidates,
            max_n_search=max_search,
            max_n_explore=0,
            dense_slice_size_per_unit=refinement_cfg.n_spikes_fit,
            labels=labels[val_ixs].to(device=device),
            explore_neighb_steps=0,
            x=full_features[val_ixs].to(device=device),
            neighborhoods=val_neighbs,
            neighb_cov=neighb_cov,
            seed=rg,
            batch_size=refinement_cfg.eval_batch_size,
        )
        assert torch.equal(
            val_data.neighborhoods.b.neighborhoods,
            train_data.neighborhoods.b.neighborhoods,
        )
    else:
        val_data = None

    assert torch.equal(full_neighbs.b.neighborhoods, train_neighbs.b.neighborhoods)
    assert full_neighbs.neighborhood_ids.shape == (full_features.shape[0],)
    full_data = StreamingSpikeData(
        n_candidates=n_candidates,
        max_n_candidates=max_candidates,
        x=full_features,
        neighborhoods=full_neighbs,
        device=device,
        batch_size=refinement_cfg.eval_batch_size,
        neighb_cov=neighb_cov,
    )
    assert torch.equal(
        full_data.neighborhoods.b.neighborhoods,
        train_data.neighborhoods.b.neighborhoods,
    )

    erp = NeighborhoodInterpolator(
        prgeom=prgeom,
        neighborhoods=full_neighbs,
        params=refinement_cfg.interp_params,
    )

    return neighb_cov, erp, train_data, val_data, full_data, noise


def get_full_neighborhood_data(
    *,
    sorting: DARTsortSorting,
    motion_est,
    refinement_cfg: RefinementConfig,
    device: torch.device | None,
    rg: np.random.Generator | int,
    stable_data: StableSpikeDataset | None = None,
) -> tuple[
    Tensor,
    SpikeNeighborhoods,
    Tensor | slice,
    SpikeNeighborhoods,
    Tensor | slice | None,
    SpikeNeighborhoods | None,
    Tensor,
]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO clean up stable data constructor, just keep what's needed
    # don't store full core features...? just extract feats by split?
    # or, just get rid of the stable dataset and grab only what's needed.
    if stable_data is None:
        vp = refinement_cfg.val_proportion
        stable_data = StableSpikeDataset.from_sorting(
            sorting,
            motion_est=motion_est,
            _core_feature_splits=(),  # turn off feat cache
            core_radius="extract",
            max_n_spikes=refinement_cfg.max_n_spikes,
            split_proportions=(1.0 - vp, vp),
            interp_params=refinement_cfg.interp_params.normalize(),
            min_count=refinement_cfg.min_count,
            random_seed=rg,
            device=device,
        )

    full_neighborhoods = stable_data._core_neighborhoods["key_full"]
    train_indices = stable_data.split_indices["train"]
    val_indices = stable_data.split_indices.get("val")
    assert isinstance(full_neighborhoods, SpikeNeighborhoods)
    assert stable_data.core_features is not None
    xfull = stable_data.core_features
    prgeom = stable_data.prgeom.to(device=device)
    assert isinstance(prgeom, Tensor)
    del stable_data  # TODO: just compute above stuff directly.

    assert xfull.shape[2] == full_neighborhoods.b.neighborhoods.shape[1]
    assert xfull.shape[0] == full_neighborhoods.b.neighborhood_ids.shape[0]
    xfull = xfull.view(len(xfull), -1)
    xfull = xfull.nan_to_num_()

    full_neighborhoods = full_neighborhoods.to(device=device)
    train_neighborhoods = full_neighborhoods.slice(train_indices)
    if val_indices is None:
        val_neighborhoods = None
    else:
        val_neighborhoods = full_neighborhoods.slice(val_indices)

    return (
        xfull,
        full_neighborhoods,
        train_indices,
        train_neighborhoods,
        val_indices,
        val_neighborhoods,
        prgeom,
    )


def instantiate_and_bootstrap_tmm(
    *,
    sorting: DARTsortSorting,
    motion_est,
    refinement_cfg: RefinementConfig,
    seed: np.random.Generator | int = 0,
    computation_cfg: ComputationConfig | None = None,
) -> MixtureModelAndDatasets:
    rg = np.random.default_rng(seed)
    computation_cfg = ensure_computation_config(computation_cfg)
    device = computation_cfg.actual_device()

    sorting = subset_sorting_by_spike_count(
        sorting, min_spikes=refinement_cfg.min_count
    )
    sorting = sorting.flatten()

    neighb_cov, erp, train_data, val_data, full_data, noise = get_truncated_datasets(
        sorting=sorting,
        motion_est=motion_est,
        refinement_cfg=refinement_cfg,
        device=device,
        rg=rg,
    )

    # bootstrapping phase: tmm needs lut, lut needs distances, distances need tmm...
    tmm = TruncatedMixtureModel.from_config(
        noise=noise,
        erp=erp,
        neighb_cov=neighb_cov,
        train_data=train_data,
        refinement_cfg=refinement_cfg,
        seed=rg,
    )
    logger.dartsortdebug(f"Initialize TMM with signal_rank={tmm.signal_rank}")
    D = tmm.unit_distance_matrix()
    lut = train_data.bootstrap_candidates(D)
    tmm.update_lut(lut)

    return MixtureModelAndDatasets(tmm, train_data, val_data, full_data)


def run_split(tmm, train_data, val_data, prog_level):
    if val_data is not None:
        eval_scores = tmm.soft_assign(
            data=val_data,
            full_proposal_view=True,
            needs_bootstrap=False,
            show_progress=prog_level,
        )
    else:
        eval_scores = tmm.soft_assign(
            data=train_data,
            full_proposal_view=False,
            needs_bootstrap=False,
            max_iter=1,
            show_progress=prog_level,
        )
    split_res = tmm.split(
        train_data, val_data, scores=eval_scores, show_progress=prog_level > 0
    )
    logger.info(f"Split created {split_res.n_new_units} new units.")


def run_merge(tmm, train_data, val_data, prog_level):
    if val_data is not None:
        eval_scores = tmm.soft_assign(
            data=val_data,
            full_proposal_view=True,
            needs_bootstrap=False,
            show_progress=prog_level,
        )
    else:
        eval_scores = tmm.soft_assign(
            data=train_data,
            full_proposal_view=False,
            needs_bootstrap=False,
            max_iter=1,
            show_progress=prog_level,
        )
    merge_map = tmm.merge(
        train_data, val_data, scores=eval_scores, show_progress=prog_level > 0
    )
    logger.info(f"Merge {merge_map.mapping.shape[0]} -> {merge_map.nuniq()} units.")


def save_tmm_labels(
    *,
    tmm: TruncatedMixtureModel,
    stepname: str,
    save_step_labels_format: str | None,
    full_data: BatchedSpikeData,
    original_sorting: DARTsortSorting,
    save_step_labels_dir: Path | None,
    save_cfg: DARTsortInternalConfig | None,
    full_proposal: bool = True,
):
    assert save_step_labels_format is not None
    full_scores = tmm.soft_assign(
        data=full_data,
        full_proposal_view=full_proposal,
        needs_bootstrap=not full_proposal,
    )
    sorting = original_sorting.ephemeral_replace(labels=labels_from_scores(full_scores))
    ds_save_intermediate_labels(
        step_name=save_step_labels_format.format(stepname=stepname),
        step_sorting=sorting,
        output_dir=save_step_labels_dir,
        cfg=save_cfg,
    )


def initialize_parameters_by_unit(
    *,
    data: TruncatedSpikeData,
    signal_rank: int,
    noise: EmbeddedNoise,
    erp: NeighborhoodInterpolator,
    prior_pseudocount: float,
    gen: torch.Generator,
    min_channel_count: int = 1,
    puff=1.0,
):
    units, counts = data.candidates[:, 0].unique(return_counts=True)
    K = int(units.amax().item()) + 1
    counts = counts[units >= 0]
    units = units[units >= 0]
    assert units.shape == (K,)
    feat_rank = noise.rank
    nc = data.neighborhoods.n_channels

    puff_K = max(K, int(puff * K))

    # do log_softmax in separate tensor (tmp) so that noise_log_prop is not
    # a view of an element of log_proportions buffer
    log_proportions = torch.zeros((puff_K + 1,), device=counts.device)
    log_proportions = log_proportions.resize_(K)
    tmp = log_proportions.new_empty((K + 1,))
    tmp[:K] = counts.float()
    tmp[-1] = tmp[:K].mean()
    tmp = F.log_softmax(tmp.log_(), dim=0)
    noise_log_prop = tmp[-1].clone()
    log_proportions.copy_(tmp[:K])

    dev = data.noise_logliks.device
    means = torch.zeros((puff_K, feat_rank, nc), device=dev)
    means = means.resize_(K, *means.shape[1:])
    if signal_rank:
        bases = torch.zeros((puff_K, signal_rank, feat_rank, nc), device=dev)
        bases = bases.resize_(K, *bases.shape[1:])
    else:
        bases = None
    for k in range(K):
        kdata = data.dense_slice_by_unit(k, gen=gen)
        assert kdata is not None
        ((kc, km, ks),), _ = initialize_params_from_dense_data(
            kdata,
            erp=erp,
            rank=signal_rank,
            noise=noise,
            min_channel_count=min_channel_count,
            prior_pseudocount=prior_pseudocount,
        )
        means[k, :, kc] = km
        if bases is not None:
            assert ks is not None
            bases[k, :, :, kc] = ks

    # flatten
    means = means.view(K, -1)
    if bases is not None:
        bases = bases.view(K, signal_rank, -1)

    return log_proportions, noise_log_prop, means, bases


def _desparsifiers(unit_ids, candidates):
    unit_ids = torch.as_tensor(unit_ids, device=candidates.device)
    ii, jj = (candidates >= 0).nonzero(as_tuple=True)
    uu = candidates[ii, jj]
    unit_ids, order = unit_ids.sort()
    ix_in_unit_ids = torch.searchsorted(unit_ids, uu)
    if pnoid:
        assert torch.equal(order, torch.arange(len(order), device=unit_ids.device))
        assert torch.equal(unit_ids[ix_in_unit_ids], uu)
    return ii, jj, ix_in_unit_ids


def get_log_liks_matrix_from_scores(unit_ids: Tensor, scores: Scores) -> Tensor:
    ii, jj, ix_in_unit_ids = _desparsifiers(unit_ids, scores.candidates)
    n = scores.log_liks.shape[0]
    log_liks = scores.log_liks.new_full((n, unit_ids.shape[0]), -torch.inf)
    log_liks[ii, ix_in_unit_ids] = scores.log_liks[ii, jj]
    return log_liks


def get_responsibilities_matrix_from_scores(unit_ids: Tensor, scores: Scores) -> Tensor:
    assert scores.responsibilities is not None
    ii, jj, ix_in_unit_ids = _desparsifiers(unit_ids, scores.candidates)
    n = scores.responsibilities.shape[0]
    responsibilities = scores.responsibilities.new_zeros((n, unit_ids.shape[0]))
    responsibilities[ii, ix_in_unit_ids] = scores.responsibilities[ii, jj]
    return responsibilities


def initialize_params_from_dense_data(
    data: DenseSpikeData,
    rank: int,
    erp: NeighborhoodInterpolator,
    noise: EmbeddedNoise,
    prior_pseudocount: float,
    min_channel_count: int = 1,
    weights: Tensor | None = None,
) -> tuple[list[tuple[Tensor, Tensor, Tensor | None]], Tensor | None]:
    """Weighted mean and basis, possibly by multiple weight vectors at once.

    The chosen channel neighborhood is the union of my spikes' neighborhoods. If there
    are weights, the neighborhood will be refined for each weight vector.

    TODO: choosing to be bold for now and always interpolate densely to the channels.
    That's different from what was done before (per-channel missing weighting.) Can
    change this by supporting erp=None later here.
    """
    covered_chans = data.covered_channels(min_channel_count)

    x_erp = data.x.view(data.x.shape[0], noise.rank, -1)
    x = erp.interp_to_chans(x_erp, data.neighborhood_ids, target_channels=covered_chans)

    if weights is None:
        mean, W = _initialize_single(x, covered_chans, noise, rank, prior_pseudocount)
        return [(covered_chans, mean, W)], None

    # weighted case. re-choose channel neighborhoods.
    subsets, full_coverage = data.weighted_covered_channels(
        weights=weights, within=covered_chans, min_count=min_channel_count
    )
    res = []
    for subset, weight in zip(subsets, weights.T):
        schans = covered_chans[subset]
        if not schans.numel():
            res.append(None)
            continue
        m, w = _initialize_single(
            x=x[:, :, subset],
            chans=schans,
            noise=noise,
            rank=rank,
            weight=weight,
            prior_pseudocount=prior_pseudocount,
        )
        res.append((schans, m, w))
    return res, full_coverage


def tree_groups(
    distances: Tensor, max_group_size: int, max_distance: float, link="complete"
) -> list[Tensor]:
    device = distances.device
    k = distances.shape[0]
    if k <= max_group_size and distances.max() <= max_distance:
        # this would be common when splitting.
        return [torch.arange(k, device=device)]

    # make symmetric
    distances = torch.minimum(distances, distances.T)

    # make finite
    isfinite = distances.isfinite()
    if not isfinite.all():
        big = distances[isfinite].amax() + max_distance + 16.0
        distances.masked_fill_(isfinite.logical_not_(), big)

    # only need upper tri, and move to numpy from now on
    distances_np = distances.numpy(force=True)
    distances_triu = distances[*np.triu_indices_from(distances_np, k=1)]

    max_group_size = min(k, max_group_size)
    Z = linkage(distances_triu.numpy(force=True), method=link)
    # get tree out to max distance
    groups = maximal_leaf_groups(
        Z,
        distances=distances_np,
        max_distance=max_distance,
        max_group_size=max_group_size,
    )
    groups = [torch.atleast_1d(torch.tensor(g)) for g in groups]
    if pnoid and link == "complete":
        for g in groups:
            assert distances[g][:, g].amax() <= max_distance
    return groups


def brute_merge(
    mm: BaseMixtureModel,
    train_data: DenseSpikeData,
    eval_data: DenseSpikeData | None,
    pair_mask: Tensor | None,
    responsibilities: Tensor | None,
    cur_scores: Scores,
    cur_unit_ids: Tensor,
    skip_full: bool,
    skip_single: bool,
    max_fit_at_once: int = 8,
) -> GroupMergeResult:
    if pair_mask is not None and pair_mask.sum() == pair_mask.shape[0]:
        return None

    # score train data with mm to get the full model log liks and the
    # responsibilities that will be used for fitting below
    train_full_scores = mm.score(train_data)
    if pnoid:
        cov = train_data.lut_coverage(mm.unit_ids, mm.get_lut())
        assert torch.equal(cov, train_full_scores.candidates >= 0)
        assert cov.any(dim=1).all()
    if responsibilities is None:
        responsibilities = F.softmax(train_full_scores.log_liks, dim=1)[:, : mm.n_units]

    # include the full partition to keep code simple
    partitions, subset_to_id, id_to_subset = allowed_partitions(
        mm.unit_ids, pair_mask, skip_full=skip_full, skip_single=skip_single
    )
    n_subsets = len(subset_to_id)
    if not n_subsets:
        return None

    # fit subset models with fixed responsibilities
    subset_resps = responsibilities.new_empty((responsibilities.shape[0], n_subsets))
    for s in range(n_subsets):
        subset_resps[:, s] = responsibilities[:, id_to_subset[s]].sum(dim=1)

    subset_models_lst = []
    keep_mask = None
    any_spikes_discarded = False
    for s0 in range(0, n_subsets, max_fit_at_once):
        s1 = min(n_subsets, s0 + max_fit_at_once)
        s0m, s0valid, _, s0discard, s0mask, _ = (
            TruncatedMixtureModel.initialize_from_dense_data_with_fixed_responsibilities(
                data=train_data,
                responsibilities=subset_resps[:, s0:s1],
                signal_rank=mm.signal_rank,
                erp=mm.erp,
                min_count=mm.min_channel_count,  # this isn't used from here, shimming
                min_channel_count=mm.min_channel_count,
                noise=mm.noise,
                max_group_size=mm.max_group_size,
                max_distance=mm.max_distance,
                neighb_cov=mm.neighb_cov,
                min_iter=mm.criterion_em_iters,
                max_iter=mm.em_iters,
                prior_pseudocount=mm.prior_pseudocount,
                cl_alpha=mm.cl_alpha,
                total_log_proportion=mm.non_noise_log_proportion(),
                elbo_atol=mm.elbo_atol,
            )
        )
        subset_models_lst.append(s0m)
        assert s0valid.all()
        any_spikes_discarded = any_spikes_discarded or s0discard
        if s0discard and keep_mask is None:
            keep_mask = s0mask
        elif s0discard:
            assert keep_mask is not None
            keep_mask.logical_or_(s0mask)

    subset_models = stack_tmms(subset_models_lst)
    if any_spikes_discarded:
        assert keep_mask is not None
        (keep_spikes,) = keep_mask.nonzero(as_tuple=True)
        responsibilities = responsibilities[keep_spikes]
        subset_resps = subset_resps[keep_spikes]
        train_data = train_data.slice(keep_spikes)
        if eval_data is None:
            assert cur_scores.log_liks.shape == train_full_scores.log_liks.shape
            cur_scores = cur_scores.slice(keep_spikes)
        train_full_scores = train_full_scores.slice(keep_spikes)
        del keep_spikes
    del keep_mask

    # score the eval or train set with the all-subsets model
    train_subset_scores = subset_models.score(train_data, skip_noise=True)
    assert train_subset_scores.log_liks.shape[1] == subset_models.n_units
    if eval_data is not None and isinstance(mm, TMMView):
        # merge case. cur_scores is mm's eval scores.
        eval_data, kept_spikes = eval_data.slice_by_coverage(
            subset_models.unit_ids, subset_models.get_lut()
        )
        if torch.is_tensor(kept_spikes) and not kept_spikes.numel():
            return None

        cur_scores = cur_scores.slice(kept_spikes)
        crit_full_scores = mm.score(eval_data, skip_noise=True)
        crit_subset_scores = subset_models.score(eval_data, skip_noise=True)
    elif eval_data is not None and isinstance(mm, TruncatedMixtureModel):
        # split case. recompute scores.
        cov0 = eval_data.lut_coverage(mm.unit_ids, mm.lut)
        cov1 = eval_data.lut_coverage(subset_models.unit_ids, subset_models.get_lut())
        cov = cov0.any(dim=1).logical_and_(cov1.any(dim=1))
        if not cov.all():
            (kept_spikes,) = cov.nonzero(as_tuple=True)
            if not kept_spikes.numel():
                return None
            eval_data = eval_data.slice(kept_spikes)
            cur_scores = cur_scores.slice(kept_spikes)

        crit_full_scores = mm.score(eval_data, skip_noise=True)
        crit_subset_scores = subset_models.score(eval_data, skip_noise=True)
    else:
        assert eval_data is None
        crit_subset_scores = train_subset_scores
        crit_full_scores = train_full_scores

    # get scores for unaffected units
    assert cur_scores.log_liks.shape[1] == cur_scores.candidates.shape[1] + 1
    assert crit_full_scores.log_liks.shape[1] in (mm.n_units, mm.n_units + 1)
    assert crit_subset_scores.log_liks.shape[1] == subset_models.n_units
    assert cur_scores.log_liks.shape[0] == crit_full_scores.log_liks.shape[0]
    cur_mask = torch.isin(cur_scores.candidates, cur_unit_ids)
    if pnoid:
        assert cur_mask.any(dim=1).all()
    rest_logliks = cur_scores.log_liks.clone()
    rest_logliks[:, :-1].masked_fill_(cur_mask, -torch.inf)

    # get current model criterion
    if cur_scores.responsibilities is None:
        cur_resp = F.softmax(cur_scores.log_liks, dim=1)
    else:
        cur_resp = cur_scores.responsibilities
    assert cur_resp.shape == cur_scores.log_liks.shape
    cur_crit = ecl(cur_resp, cur_scores.log_liks, cl_alpha=mm.cl_alpha)
    del cur_resp

    # now, find the best subset. combine subset scores with remainder scores.
    # also need to adjust the log proportions here.
    k0 = rest_logliks.shape[1]
    kfull = mm.n_units
    part_logliks = F.pad(rest_logliks, (0, kfull), value=-torch.inf)
    best_part = partitions[0]
    best_score = torch.tensor(-torch.inf)
    for part in partitions:
        k1 = k0 + len(part.single_ixs)
        k2 = k1 + len(part.subset_ids)
        part_logliks[:, k0:k1] = crit_full_scores.log_liks[:, part.single_ixs]
        part_logliks[:, k1:k2] = crit_subset_scores.log_liks[:, part.subset_ids]

        part_resps = F.softmax(part_logliks[:, :k2], dim=1)
        part_score = ecl(part_resps, part_logliks[:, :k2], cl_alpha=mm.cl_alpha)
        del part_resps

        if part_score > best_score:
            best_score = part_score
            best_part = part

    if pnoid:
        assert math.isfinite(cur_crit)
        assert math.isfinite(best_score)

    # spike assignments
    train_assignments = get_part_assignments(
        best_part, train_full_scores.log_liks, train_subset_scores.log_liks
    )

    # parameters
    single_means, single_bases = mm.get_params_at(best_part.single_ixs)
    sub_means, sub_bases = subset_models.get_params_at(best_part.subset_ids)
    means = torch.concatenate((single_means, sub_means), dim=0)
    if mm.signal_rank:
        assert single_bases is not None
        assert sub_bases is not None
        bases = torch.concatenate((single_bases, sub_bases), dim=0)
    else:
        assert single_bases is sub_bases is None
        bases = None

    # fraction
    prop = responsibilities.mean(0)
    single_prop = prop[best_part.single_ixs]
    sub_props = [subset_resps[:, sid].mean(0)[None] for sid in best_part.subset_ids]
    sub_props = torch.concatenate([single_prop] + sub_props, dim=0)
    assert sub_props.shape == (best_part.n_groups,)
    sub_props /= sub_props.sum()

    return SuccessfulGroupMergeResult(
        grouping=best_part,
        improvement=(best_score - cur_crit).cpu().item(),
        train_assignments=train_assignments,
        train_indices=train_data.indices,
        means=means,
        bases=bases,
        sub_proportions=sub_props,
    )


def allowed_partitions(
    unit_ids: Tensor,
    pair_mask: Tensor | None,
    skip_full: bool = False,
    skip_single: bool = False,
):
    n_units = unit_ids.numel()
    # this will be like 0,0,1,2 to indicate (uids[0],uids[1]),uids[2],uids[3]
    group_ids = torch.zeros_like(unit_ids)

    group_partitions = []
    subset_to_id = {}
    id_to_subset = {}
    subset_id = 0
    for n_groups in range(1 + skip_single, n_units + 1 - skip_full):
        for partition in multiset_partitions(n_units, m=n_groups):
            # partition is like [0, 1, 2], [3, 4]
            subset_ids = []
            single_ixs = []
            for j, p in enumerate(partition):
                tp = tuple(p)
                p = torch.tensor(p)
                if len(tp) > 1 and tp not in subset_to_id:
                    subset_ids.append(subset_id)
                    subset_to_id[tp] = subset_id
                    id_to_subset[subset_id] = p
                    subset_id += 1
                elif len(tp) > 1:
                    subset_ids.append(subset_to_id[tp])
                elif len(tp) == 1:
                    single_ixs.append(tp[0])
                else:
                    assert False
                # first, check that this works for the mask
                # because this is nested, we break here. then if we hit the else,
                # none of these tripped, so the partition as a whole was good.
                if pair_mask is not None and not pair_mask[p[:, None], p[None]].all():
                    break
                group_ids[p] = j
            else:
                group_partition = GroupPartition(
                    unit_ids=unit_ids,
                    group_ids=group_ids.clone(),
                    n_groups=n_groups,
                    single_ixs=single_ixs,
                    subset_ids=subset_ids,
                )
                group_partitions.append(group_partition)

    return group_partitions, subset_to_id, id_to_subset


def get_part_assignments(
    part: GroupPartition, full_scores: Tensor, subset_scores: Tensor
):
    single_scores = full_scores[:, part.single_ixs]
    sub_scores = [subset_scores[:, sid, None] for sid in part.subset_ids]
    sub_scores = torch.concatenate([single_scores] + sub_scores, dim=1)
    assignments = sub_scores.argmax(dim=1, keepdim=True)
    assignments.masked_fill_(
        torch.isneginf(sub_scores.take_along_dim(assignments, 1)), -1
    )
    assignments = assignments.view(assignments.shape[0])
    return assignments


def try_kmeans(
    data: DenseSpikeData,
    k: int,
    erp: NeighborhoodInterpolator,
    gen: torch.Generator,
    min_count: int,
    feature_rank: int,
    n_iter: int = 100,
    with_proportions: bool = True,
    drop_prop: float = 0.025,
    kmeanspp_initial="random",
    n_kmeans_tries: int = 10,
    n_kmeanspp_tries: int = 10,
) -> Tensor | None:
    # interpolate whitened data
    channels = data.covered_channels(min_count)
    erp_x = data.whitenedx.view(data.x.shape[0], feature_rank, -1)
    x = erp.interp_to_chans(erp_x, data.neighborhood_ids, channels)
    x = x.view(len(x), -1)

    # kmeans
    kres = kmeans(
        x,
        n_components=k,
        random_state=gen,
        n_iter=n_iter,
        with_proportions=with_proportions,
        drop_prop=drop_prop,
        kmeanspp_initial=kmeanspp_initial,
        n_kmeans_tries=n_kmeans_tries,
        n_kmeanspp_tries=n_kmeanspp_tries,
    )
    resps = kres["responsibilities"]
    if resps is None:
        return
    assert resps.shape[1] <= k
    (big_enough,) = (resps.sum(dim=0) >= min_count).nonzero(as_tuple=True)
    if big_enough.numel() <= 1:
        return
    resps = resps[:, big_enough]

    return resps


def labels_from_scores(scores: Scores) -> np.ndarray:
    """Pick either top candidate or noise."""
    labels = scores.candidates[:, 0].clone()
    noise_better = scores.log_liks[:, -1] > scores.log_liks[:, 0]
    labels.masked_fill_(noise_better, -1)
    return labels.numpy(force=True)


# -- method helpers


def _pick_search_size(
    *,
    n_units: int,
    max_n_candidates=None,
    max_n_search=None,
    max_n_explore=None,
    refinement_cfg=None,
):
    """Get sensible candidate, search, explore set sizes."""
    if refinement_cfg is None:
        assert max_n_candidates is not None
        assert max_n_search is not None
        assert max_n_explore is not None
    else:
        assert max_n_candidates is None
        assert max_n_search is None
        assert max_n_explore is None
        max_n_candidates = refinement_cfg.n_candidates
        max_n_search = refinement_cfg.n_search or refinement_cfg.n_candidates
        max_n_explore = refinement_cfg.n_explore or max(max_n_candidates, max_n_search)

    C = min(n_units, max_n_candidates)
    S = min(n_units, max_n_search)
    E = min(n_units, max_n_explore)

    return C, S, E, max_n_candidates, max_n_search, max_n_explore


def _neighborhood_indices(
    *, neighborhoods: SpikeNeighborhoods, zero_radius: float | None, prgeom: Tensor
):
    nc_obs = neighborhoods.b.channel_counts
    obs_ix = neighborhoods.b.neighborhoods
    truncate = zero_radius is not None and zero_radius < float("inf")
    nneighb = neighborhoods.n_neighborhoods
    assert obs_ix.shape[0] == nneighb
    nc = neighborhoods.n_channels
    dev = prgeom.device

    miss_near = []
    miss_full_masks = torch.zeros((nneighb, nc), device=dev)
    for j in range(nneighb):
        j_miss_full = neighborhoods.missing_channels(j)
        miss_full_masks[j, j_miss_full] = 1.0
        if truncate:
            assert zero_radius is not None
            jobs = neighborhoods.neighborhood_channels(j)
            d = torch.cdist(prgeom[j_miss_full], prgeom[jobs]).amin(dim=1)
            miss_near.append(j_miss_full[d < zero_radius])
        else:
            miss_near.append(j_miss_full)

    max_nc_miss_near = max(jj.numel() for jj in miss_near)
    miss_near_ix = obs_ix.new_full((nneighb, max_nc_miss_near), nc)
    for j, mi in enumerate(miss_near):
        miss_near_ix[j, : mi.numel()] = mi

    return nc_obs, obs_ix, miss_near_ix, miss_full_masks


def _noise_factors(*, noise, obs_ix, miss_near_ix, cache_prefix):
    nneighb = obs_ix.shape[0]
    nc_obs = obs_ix.shape[1]
    nc_miss_near = miss_near_ix.shape[1]
    rank = noise.rank
    nc = noise.n_channels
    dev = obs_ix.device

    logdet = torch.zeros((nneighb,), device=dev)
    Cooinv = torch.zeros((nneighb, rank, nc_obs, rank, nc_obs), device=dev)
    CooinvCom = torch.zeros((nneighb, rank, nc_obs, rank, nc_miss_near), device=dev)
    Linv = torch.zeros_like(Cooinv)

    for j, (joix, jmnix) in enumerate(zip(obs_ix, miss_near_ix)):
        (joixvix,) = (joix < nc).nonzero(as_tuple=True)
        joixv = joix[joixvix]
        (jmnixvix,) = (jmnix < nc).nonzero(as_tuple=True)
        jmnixv = jmnix[jmnixvix]
        ncoi = joixv.numel()
        ncmi = jmnixv.numel()

        jCoo = noise.marginal_covariance(
            channels=joixv, cache_prefix=cache_prefix, cache_key=j
        )
        jCom = noise.offdiag_covariance(
            channels_left=joixv, channels_right=jmnixv, device=dev
        )
        jCom = jCom.to_dense().to(device=dev)

        jL = jCoo.cholesky(upper=False)  # C = LL'
        jLinv = jL.inverse().to_dense()
        jCooinv = jLinv.T @ jLinv  # Cinv = Linv' Linv
        jlogdet = 2.0 * jL.to_dense().diagonal(dim1=-2, dim2=-1).log().sum()
        jCooinvCom = jCooinv @ jCom

        logdet[j] = jlogdet
        # fancy inds to front! love that.
        jCooinv = jCooinv.view(rank, ncoi, rank, ncoi)
        Cooinv[j, :, joixvix[:, None], :, joixvix[None]] = jCooinv.permute(1, 3, 0, 2)
        jCooinvCom = jCooinvCom.view(rank, ncoi, rank, ncmi)
        CooinvCom[j, :, joixvix[:, None], :, jmnixvix[None]] = jCooinvCom.permute(
            1, 3, 0, 2
        )
        jLinv = jLinv.view(rank, ncoi, rank, ncoi)
        Linv[j, :, joixvix[:, None], :, joixvix[None]] = jLinv.permute(1, 3, 0, 2)

    obsdim = rank * nc_obs
    missdim = rank * nc_miss_near
    Cooinv = Cooinv.view(nneighb, obsdim, obsdim)
    CooinvCom = CooinvCom.view(nneighb, obsdim, missdim)
    Linv = Linv.view(nneighb, obsdim, obsdim)
    return logdet, Cooinv, CooinvCom, Linv


def _whiten_impute_and_noise_score(
    *,
    x: Tensor,
    neighborhoods: SpikeNeighborhoods,
    neighb_cov: NeighborhoodCovariance,
    batch_size=1024,
):
    wx = torch.empty_like(x)
    assert neighborhoods.n_neighborhoods == neighb_cov.obs_ix.shape[0]

    noise_loglik = wx.new_zeros((len(x),))
    CmoCooinvx = wx.new_empty(
        (len(x), neighb_cov.max_nc_miss_near, neighb_cov.feat_rank)
    )

    for ni in range(neighborhoods.n_neighborhoods):
        inni = neighborhoods.neighborhood_members(ni)

        right_factor = neighb_cov.Linv[ni].T
        neighb_CooinvCom = neighb_cov.CooinvCom[ni]

        nll_const = neighb_cov.logdet[ni] + LOG_2PI * neighb_cov.nobs[ni]

        for bs in range(0, inni.numel(), batch_size):
            binni = inni[bs : bs + batch_size]
            xb = x[binni]
            wxb = xb @ right_factor

            wx[binni] = wxb

            CmoCooinvx_batch = xb @ neighb_CooinvCom
            CmoCooinvx_batch = CmoCooinvx_batch.view(
                binni.numel(), neighb_cov.feat_rank, neighb_cov.max_nc_miss_near
            )
            CmoCooinvx[binni] = CmoCooinvx_batch.mT

            nll = wxb.square_().sum(dim=1)
            nll += nll_const
            nll *= -0.5
            assert nll.isfinite().all()
            noise_loglik[binni] = nll

    xt = x.view(x.shape[0], neighb_cov.feat_rank, neighb_cov.max_nc_obs)
    xt = xt.transpose(1, 2).contiguous()

    return xt, wx, CmoCooinvx, noise_loglik


def _whiten_and_noise_score_batch(
    *, x: Tensor, neighb_ids: Tensor, neighb_cov: NeighborhoodCovariance
):
    batch_whitener = neighb_cov.Linv[neighb_ids]
    wx = torch.bmm(batch_whitener, x[:, :, None])[:, :, 0]
    nll = torch.linalg.vector_norm(wx, dim=1).square_()
    nll.add_(neighb_cov.logdet[neighb_ids]).add_(LOG_2PI * neighb_cov.nobs[neighb_ids])
    nll.mul_(-0.5)
    return wx, nll


def _initialize_single(
    x: Tensor,
    chans: Tensor,
    noise: EmbeddedNoise,
    rank: int,
    prior_pseudocount: float,
    weight: Tensor | None = None,
    eps: float = 1e-5,
    mean: Tensor | None = None,
):
    """
    Replaces ppcalib initialize_mean and the branch where SVD was done in ppcalib.

    Also initializes the basis. In the unweighted case, the algorithm is:
     - Estimate imputed centered data (properly this should be done by imputing
       under the noise covariance model's conditional distribution for missing
       data and subtracting the missing-data mean; here we can try also using
       interpolation)
     - Whiten
     - Low-rank SVD to get SVs `s` and feature basis `Vh`
     - Convert `s` to std dev: `sqrt(s**2/(n-1))`
     - Basis is `unwhiten(Vh*(stddev-1))` -- the 1 has to do with extra identity.

    When there are weights, probably the best thing to do is eigh the weighted cov.
    It seems equivalent to do SVD of data multiplied by sqrt of weights row-wise.
    There is some inflation that needs to be corrected relative to eigh of the
    weighted cov, see formula below. It seems pretty accurate when evaluated
    numerically with random data...
    """
    n, d, c = x.shape
    x = x.view(n, d * c)

    if weight is None:
        nw = None
        if mean is None:
            mean = x.mean(dim=0)
        else:
            mean = mean.view(d * c)
        wsum = x.new_tensor(float(n))
    else:
        wsum = weight.sum()
        nw = weight / (wsum + prior_pseudocount)
        if mean is None:
            mean = (nw[:, None] * x).sum(0)
        else:
            mean = mean.view(d * c)
    assert mean is not None

    if not rank:
        mean = mean.view(d, c)
        return mean, None

    q = min(n, d * c, 10 + rank)

    # we want x(C^-0.5)=xU^-1 -- need to use upper factor.
    noise_cov = noise.marginal_covariance(chans)
    U = torch.linalg.cholesky(noise_cov, upper=True).to_dense()
    x = x - mean
    x = torch.linalg.solve_triangular(U, x, upper=True, left=False, out=x)

    # weighting the svd -- needs sqrt!
    if weight is not None:
        if prior_pseudocount:
            weight *= wsum / (wsum + prior_pseudocount)
        x *= weight.sqrt()[:, None]

    # NB they return V not Vh here, so rank dim comes last
    _, s, V = torch.svd_lowrank(x, q=q, niter=7)
    s = s[:rank]
    V = V[:, :rank]

    # convert to eigvals
    if weight is None:
        s = s.square_().div_(max(1.0, n - 1.0))
    else:
        s = s.square_().div_(wsum.sub_(1.0).clamp_(min=1.0))

    # extra 1 was picked up from identity in whitened space, remove and
    # compute the low rank factor that remains, which is our basis
    s = s.sub_(1.0).clamp_(min=eps)
    W = V.mul_(s.sqrt_().mul_(sign(V.diagonal(dim1=-2, dim2=-1))))
    W = U.T @ W

    mean = mean.view(d, c)
    # TODO build this the other way
    assert W.shape[1] == rank
    W = W.T.reshape(rank, d, c)

    return mean, W


# -- candidate set helpers


def coincidence_matrix(
    x: Tensor, y: Tensor, nx: int, ny: int, dtype=torch.long
) -> Tensor:
    """Get a confusion matrix of sorts: C[p,q] is |{i:x[i]=p,y[i]=q}|.

    x and y should be int/long dtype.

    x can have -1, y can't. x can be 1d or 2d, y matches first dim.
    Similar to but faster than stack(x,y).unique(0), especially if you
    need this matrix.
    """
    assert x.shape[:1] == y.shape

    # get rid of -1s in x (labels) and get corresponding y (neighbs)
    xpos = x >= 0
    if x.ndim == 2:
        pos_ii, pos_jj = xpos.nonzero(as_tuple=True)
        xx = x[pos_ii, pos_jj]
    elif x.ndim == 1:
        (pos_ii,) = xpos.nonzero(as_tuple=True)
        xx = x[pos_ii]
    else:
        assert False
    yy = y[pos_ii]

    # build coincidence matrix
    co = torch.zeros((nx, ny), dtype=dtype, device=x.device)
    iflat = yy.add_(xx, alpha=ny)
    one = torch.ones((1,), dtype=dtype, device=co.device).broadcast_to(iflat.shape)
    co.view(-1).scatter_add_(dim=0, index=iflat, src=one)

    return co


def candidate_adjacencies(
    labels: Tensor | None,
    neighborhoods: SpikeNeighborhoods,
    neighborhood_ids: Tensor,
    neighb_supset: Tensor,
    neighb_adj: Tensor,
    explore_steps: int,
    n_units: int,
    un_adj_lut: NeighborhoodLUT | None,
    expand_from_lut: NeighborhoodLUT | None,
    device: torch.device,
):
    if un_adj_lut is None:
        assert labels is not None
        un_adj_lut = lut_from_candidates_and_neighborhoods(
            candidates=labels,
            neighborhoods=neighborhoods,
            neighborhood_ids=neighborhood_ids,
            n_units=n_units,
            neighb_supset=neighb_supset,
        )
    if expand_from_lut is not None:
        un_adj_lut = combine_luts(expand_from_lut, un_adj_lut)

    # lut -> adjacency matrix
    un_adj = un_adj_lut.lut < un_adj_lut.unit_ids.shape[0]
    assert un_adj.any()
    un_adj = un_adj.float()

    # constraint for explore candidates is looser
    explore_adj = un_adj
    for _ in range(explore_steps):
        explore_adj = explore_adj @ neighb_adj

    return un_adj_lut, un_adj, explore_adj


def lut_from_candidates_and_neighborhoods(
    *,
    candidates: Tensor,
    neighborhoods: SpikeNeighborhoods,
    neighb_supset: Tensor,
    n_units: int,
    neighborhood_ids: Tensor,
) -> NeighborhoodLUT:
    if neighborhood_ids is None:
        neighborhood_ids = neighborhoods.b.neighborhood_ids
    assert (candidates.shape[0],) == neighborhood_ids.shape
    co = coincidence_matrix(
        x=candidates,
        y=neighborhood_ids,
        nx=n_units,
        ny=neighborhoods.n_neighborhoods,
    )
    co = co.float() @ neighb_supset
    uu, nn = co.nonzero(as_tuple=True)
    n_lut = uu.shape[0]
    lut = co.new_full(co.shape, n_lut, dtype=torch.long)
    lut[uu, nn] = torch.arange(n_lut, device=co.device)
    return NeighborhoodLUT(unit_ids=uu, neighb_ids=nn, lut=lut)


def combine_luts(*luts: NeighborhoodLUT) -> NeighborhoodLUT:
    assert len(luts) > 0
    if len(luts) == 1:
        return luts[0]
    adj = luts[0].lut < luts[0].unit_ids.shape[0]
    for l in luts[1:]:
        adj.logical_or_(l.lut < l.unit_ids.shape[0])
    unit_ids, neighb_ids = adj.nonzero(as_tuple=True)
    lut = torch.full_like(luts[0].lut, unit_ids.shape[0])
    lut[unit_ids, neighb_ids] = torch.arange(unit_ids.shape[0], device=lut.device)
    return NeighborhoodLUT(unit_ids=unit_ids, neighb_ids=neighb_ids, lut=lut)


def candidate_search_sets(
    distances: Tensor, un_adj_lut: NeighborhoodLUT, un_adj: Tensor, n_search: int
):
    n_lut = un_adj_lut.unit_ids.shape[0]

    # get lut-indexed version with extra lut index for topk below
    s = distances[un_adj_lut.unit_ids]

    # don't want to match with myself
    inf = s.new_full((1, 1), torch.inf).broadcast_to((n_lut, 1))
    s.scatter_(dim=1, index=un_adj_lut.unit_ids[:, None], src=inf)

    # flip scale so larger is better
    s.reciprocal_()

    # multiply by neighborhood-unit adjacency to set non olap to 0
    s.mul_(un_adj.T[un_adj_lut.neighb_ids, : distances.shape[0]])

    # take topk, and fill invalids (s=0) with -1
    tops, topunits = torch.topk(s, k=n_search, dim=1)
    topunits.masked_fill_(tops == 0, -1)
    # pad with row of -1s for the invalid lut ixs (===n_lut)
    topunits = F.pad(topunits, (0, 0, 0, 1), value=-1)

    return topunits


def max_units_per_neighb(lut: NeighborhoodLUT):
    coverage = lut.lut < lut.unit_ids.shape[0]
    return int(coverage.sum(dim=0).amax())


def full_proposal_by_neighb(lut: NeighborhoodLUT, max_proposed: int):
    n_neighbs = lut.lut.shape[1]
    n_lut = lut.unit_ids.shape[0]
    proposals = torch.full((n_neighbs, max_proposed), -1)
    for neighb_id in range(n_neighbs):
        row = lut.lut[:, neighb_id]
        (row_valid,) = (row < n_lut).nonzero(as_tuple=True)
        n_row = row_valid.numel()
        assert n_row <= max_proposed
        proposals[neighb_id, :n_row] = lut.unit_ids[row[row_valid]]
    return proposals


def _fill_blank_labels(
    labels: Tensor,
    un_adj: Tensor,
    explore_adj: Tensor,
    neighborhood_ids: Tensor,
    neighb_adj: Tensor,
    gen: torch.Generator,
    batch_size: int = 1024,
):
    (blank,) = (labels < 0).nonzero(as_tuple=True)
    Nblank = blank.numel()
    if not Nblank:
        return True

    # need full coverage of neighborhoods here. would prefer only to branch out with
    # explore as needed, so make those probs tiny.
    keep_same_adj = un_adj.sum(0).min().cpu().item() > 0
    if keep_same_adj:
        adj = un_adj
    else:
        adj = un_adj + explore_adj * torch.finfo(explore_adj.dtype).tiny
        uncovered = adj.sum(0).min().cpu().item() == 0
        if uncovered:
            uncovered_adj = adj.clone()
        else:
            uncovered_adj = adj

        n_steps = 0
        while adj.sum(0).min().cpu().item() == 0:
            explore_adj = explore_adj @ neighb_adj
            adj = un_adj + explore_adj * torch.finfo(explore_adj.dtype).tiny
            n_steps += 1
            if n_steps > 5:
                break

        if uncovered:
            still_uncovered = adj.sum(0).min().cpu().item() == 0
            # raise if still_uncovered, warn or log otherwise
            _log_warn_or_raise_coverage(
                uncovered_adj, neighborhood_ids, n_steps, still_uncovered
            )
    for i0 in range(0, Nblank, batch_size):
        i1 = min(Nblank, i0 + batch_size)
        ii = blank[i0:i1]
        nn = neighborhood_ids[ii]
        p = adj[:, nn].mT
        draws = torch.multinomial(p, 1, replacement=True, generator=gen)
        labels[ii] = draws.view(ii.shape[0])

    return keep_same_adj


def _bootstrap_top(
    candidates: Tensor,
    neighborhood_ids: Tensor,
    n_candidates: int,
    un_adj: Tensor,
    un_adj_lut: NeighborhoodLUT,
    gen: torch.Generator,
    batch_size: int = 1024,
):
    # use a un_adj_lut index-dependent probability here to avoid re-using
    # the current label
    p = un_adj.T[un_adj_lut.neighb_ids]
    zero = p.new_zeros((1, 1)).broadcast_to((un_adj_lut.unit_ids.shape[0], 1))
    p.scatter_(dim=1, index=un_adj_lut.unit_ids[:, None], src=zero)

    # pad with an extra ncand-1 "unit"s of epsilons to allow no-replacement
    # sampling even when p is blank for my lut. they will get filled with -1s
    p = F.pad(p, (0, n_candidates - 1), value=torch.finfo(p.dtype).tiny)

    # we've got the choice to either use the same random assignment for
    # everyone with the same lut ix, or to pick a new random assignment for
    # each spike. since this is only done once per dataset, it seems good
    # to do the latter, to avoid some kind of bad initialization. but
    # that means it must be done in batches, because there are too many
    # spikes to store a dense spikes x units matrix of longs.
    N = candidates.shape[0]
    n_units = un_adj.shape[0]
    for i0 in range(0, N, batch_size):
        i1 = min(N, i0 + batch_size)
        uu = candidates[i0:i1, 0]
        nn = neighborhood_ids[i0:i1]
        ll = un_adj_lut.lut[uu, nn]
        torch.multinomial(
            p[ll],
            n_candidates - 1,
            replacement=False,
            generator=gen,
            out=candidates[i0:i1, 1:n_candidates],
        )
        candidates[i0:i1].masked_fill_(candidates[i0:i1] >= n_units, -1)


def _get_explore_sampling_data(
    un_adj_lut: NeighborhoodLUT,
    explore_adj: Tensor,
    search_sets: Tensor,
    n_explore: int,
):
    # for each lut ix (unit-neighb pair), figure out which units overlap
    # the neighborhood, less the main unit
    p = explore_adj.T[un_adj_lut.neighb_ids]

    # zero out top unit probs
    zero = p.new_zeros((1, 1)).broadcast_to((un_adj_lut.unit_ids.shape[0], 1))
    p.scatter_(dim=1, index=un_adj_lut.unit_ids[:, None], src=zero)

    # since we're already including the main unit's search neighbors in the
    # candidate set elsewhere, zero out their probs too
    # search sets are indexed by lut bin, same as p, so rows correspond
    ii, jj = (search_sets >= 0).nonzero(as_tuple=True)
    p[ii, search_sets[ii, jj]] = 0.0

    # count them in each bin and get the max (max_explore)
    explore_counts = (p > 0).sum(dim=1)
    max_explore: int = max(n_explore, explore_counts.max().cpu().item())  # type: ignore

    # make p (n_spikes, max_explore) which is ones up to each bin's count,
    # then epsilons. and track the corresponding units. that's a topk.
    p, inds = torch.topk(p, k=max_explore)
    pzero = p == 0
    inds.masked_fill_(pzero, -1)
    p.masked_fill_(pzero, torch.finfo(p.dtype).tiny)

    return p, inds


def _sample_explore_candidates(
    p: Tensor,
    inds: Tensor,
    candidates: Tensor,
    lut_ixs: Tensor,
    n_explore: int,
    gen: torch.Generator,
    batch_size=1024,
):
    n = len(p)
    explore0 = candidates.shape[1] - n_explore
    for i0 in range(0, n, batch_size):
        i1 = min(n, i0 + batch_size)
        batch_lut_ixs = lut_ixs[i0:i1]
        # sample multinomial without replacement for each spike
        choices = torch.multinomial(p[batch_lut_ixs], n_explore, generator=gen)
        # replace those values with their entries in inds
        candidates[i0:i1, explore0:] = inds[batch_lut_ixs[:, None], choices]
    # there will be duplicates, but that's okay, because we're about to
    # call _dedup_candidates


def _dedup_candidates(candidates):
    # there's a sorting-based algorithm which scales better with shape[1],
    # but it seems slower for the actual values used here (on gpu)
    N, C = candidates.shape
    dup_mask = candidates.new_zeros((N, C - 1), dtype=torch.bool)
    eq_mask = candidates.new_zeros((N, C - 1), dtype=torch.bool)
    numel = N * C
    for j in range(candidates.shape[1] - 1):
        eq_mask = eq_mask.resize_(numel - (j + 1) * N).view(N, C - j - 1)
        torch.eq(candidates[:, j + 1 :], candidates[:, j : j + 1], out=eq_mask)
        dup_mask[:, j:].logical_or_(eq_mask)
    candidates[:, 1:].masked_fill_(dup_mask, -1)


def _count_candidates(candidates, batch_candidate_counts, batch_size):
    counts = candidates.new_zeros(batch_candidate_counts.shape)
    for b, i0 in enumerate(range(0, candidates.shape[0], batch_size)):
        counts[b] = (candidates[i0 : i0 + batch_size] >= 0).sum()
    batch_candidate_counts.copy_(counts.cpu())


def concatenate_scores(scoress: list[Scores]) -> Scores:
    assert len(scoress) > 0
    if len(scoress) == 1:
        return scoress[0]
    log_liks = torch.concatenate([s.log_liks for s in scoress], dim=0)
    candidates = torch.concatenate([s.candidates for s in scoress], dim=0)
    if scoress[0].responsibilities is None:
        responsibilities = None
    else:
        responsibilities = torch.concatenate(
            [s.responsibilities for s in scoress],  # type: ignore
            dim=0,
        )
    return Scores(
        log_liks=log_liks, responsibilities=responsibilities, candidates=candidates
    )


def stack_tmms(tmms: list[TruncatedMixtureModel]) -> BaseMixtureModel:
    if len(tmms) == 1:
        return tmms[0]
    else:
        return TMMStack(tmms)


# -- precomputing expensive things that depend on parameters and neighborhoods


LOG_2PI = math.log(math.tau)


def _update_lut_mean_batch(
    lut_params: LUTParams,
    i0: int,
    i1: int,
    means: Tensor,
    lut: NeighborhoodLUT,
    neighb_cov: NeighborhoodCovariance,
):
    """Update the "rank 0" stuff

    That means updating 4 things:
     - The observed means, muo
     - Their whitening Linvmuo
     - The noise-conditional-missing-near means CmoCooinvmuo
     - NEGATIVE TWICE the constant term in the likelihood constplogdet, which is
         obs_dim * log(2pi) + logdet(cov)
       -2x because _calc_loglik multiplies by -0.5 anyway.
       NB that logdet(cov) = logdet(Coo) + logdet(ppca term). Here we just put in
       the dim 2pi constant and the logdet(Coo). The next fn deals with the second
       term if needed.
    """
    n = i1 - i0
    uu = lut.unit_ids[i0:i1]
    nn = lut.neighb_ids[i0:i1]

    # grab these units' means with one channel of zero padding
    mpad = means.new_zeros((n, neighb_cov.feat_rank, neighb_cov.n_channels + 1))
    torch.index_select(
        means.view(means.shape[0], neighb_cov.feat_rank, -1),
        dim=0,
        index=uu,
        out=mpad[:, :, :-1],
    )

    # extract observed part (muo)
    obs_ix = neighb_cov.obs_ix[nn][:, None, :]
    mu_obs_out = lut_params.muo.view(
        lut_params.n_lut, neighb_cov.feat_rank, neighb_cov.max_nc_obs
    )
    muo = torch.take_along_dim(mpad, obs_ix, dim=2, out=mu_obs_out[i0:i1])
    muo = muo.view(n, neighb_cov.feat_rank * neighb_cov.max_nc_obs)

    # Linvmuo, CmoCooinvmuo
    torch.bmm(
        neighb_cov.Linv[nn], muo[:, :, None], out=lut_params.Linvmuo[i0:i1, :, None]
    )
    torch.bmm(
        muo[:, None], neighb_cov.CooinvCom[nn], out=lut_params.CmoCooinvmuo[i0:i1, None]
    )

    # constplogdet. add in the signal-rank-0-only terms.
    lut_params.constplogdet[i0:i1] = neighb_cov.nobs[nn].mul_(LOG_2PI)
    lut_params.constplogdet[i0:i1] += neighb_cov.logdet[nn]
    if pnoid:
        assert lut_params.constplogdet[i0:i1].isfinite().all()


def _update_lut_ppca_batch(
    lut_params: LUTParams,
    i0: int,
    i1: int,
    bases: Tensor,
    lut: NeighborhoodLUT,
    neighb_cov: NeighborhoodCovariance,
):
    """Update low-rank factors

    "T" is the posterior cov of latent variables. It's also the "inverse capacitance"
    matrix in the language that people use when talking about Woodbury. We store this
    thing (padded a bit for use elsewhere) along with its products with some other
    stuff. We also need to store what I'm calling "Woodbury root", which is the
    product of the noise covariance whitener and the product of the signal basis with
    the Cholesky factor of T (but transposed).
    """
    M = bases.shape[1]
    n = i1 - i0
    uu = lut.unit_ids[i0:i1]
    nn = lut.neighb_ids[i0:i1]

    # to start, we need the observed part of W. grab as for muo above.
    bpad = F.pad(
        bases[uu].view(uu.shape[0], M, neighb_cov.feat_rank, neighb_cov.n_channels),
        (0, 1),
        value=0.0,
    )
    Wo = bpad.take_along_dim(indices=neighb_cov.obs_ix[nn][:, None, None, :], dim=3)
    Wo = Wo.view(n, M, neighb_cov.feat_rank * neighb_cov.max_nc_obs)

    # next, the capaciatance is I + Wo Cooinv Wo'. always pos def.
    WoCooinvsqrt = Wo.bmm(neighb_cov.Linv[nn].mT)
    cap = WoCooinvsqrt.bmm(WoCooinvsqrt.mT)
    cap.diagonal(dim1=-2, dim2=-1).add_(1.0)
    if pnoid:
        assert cap.isfinite().all()
        assert torch.all(cap.diagonal(dim1=-2, dim2=-1) > 0)

    # capacitance cholesky, inverse cholesky, inverse
    # cap = LL'
    L, info = torch.linalg.cholesky_ex(cap, upper=False, check_errors=pnoid)
    if pnoid:
        assert (info == 0).all()
    Linv = torch.zeros_like(L)  # initialize to identity
    Linv.diagonal(dim1=-2, dim2=-1).add_(1.0)
    # solve LX=I, so that Linv is upper and inv(cap) = Linv' Linv.
    Linv = torch.linalg.solve_triangular(L, Linv, upper=False, out=Linv)
    T = Linv.mT.bmm(Linv)

    # Tpad, logdet. Tpad was initialized with zeros.
    assert lut_params.Tpad is not None
    lut_params.Tpad[i0:i1, :, :-1] = T
    cap_logdet = L.diagonal(dim1=-2, dim2=-1).log().sum(dim=1).mul_(2.0)
    lut_params.constplogdet[i0:i1] += cap_logdet
    if pnoid:
        assert lut_params.constplogdet[i0:i1].isfinite().all()

    # precomputed products with T
    assert lut_params.TWoCooinvsqrt is not None
    assert lut_params.TWoCooinvmuo is not None
    TWoCooinvsqrt = torch.bmm(T, WoCooinvsqrt, out=lut_params.TWoCooinvsqrt[i0:i1])
    Linvmuo = lut_params.Linvmuo[i0:i1, :, None]
    torch.bmm(TWoCooinvsqrt, Linvmuo, out=lut_params.TWoCooinvmuo[i0:i1, :, None])

    # Woodbury root
    assert lut_params.wburyroot is not None
    torch.bmm(WoCooinvsqrt.mT, Linv.mT, out=lut_params.wburyroot[i0:i1])


# -- em math impls


def _score_batch(
    candidates: Tensor,
    neighborhood_ids: Tensor,
    n_candidates: int,
    whitenedx: Tensor,
    noise_logliks: Tensor,
    log_proportions: Tensor,
    noise_log_prop: Tensor,
    lut_params: LUTParams,
    lut: NeighborhoodLUT,
    spike_ixs: Tensor | None = None,
    candidate_ixs: Tensor | None = None,
    unit_ixs: Tensor | None = None,
    neighb_ixs: Tensor | None = None,
    lut_ixs: Tensor | None = None,
    static_size: int | None = None,
    fixed_responsibilities: Tensor | None = None,
    skip_responsibility: bool = False,
    skip_noise: bool = False,
):
    n, Ctot = candidates.shape
    assert whitenedx.shape[0] == n
    assert noise_logliks.shape == (n,)
    not_fixed = fixed_responsibilities is None
    do_resp = not_fixed and not skip_responsibility
    if not not_fixed:
        assert skip_noise

    if spike_ixs is None:
        spike_ixs, candidate_ixs, unit_ixs, neighb_ixs = _sparsify_candidates(
            candidates, neighborhood_ids, static_size=static_size
        )
        lut_ixs = lut.lut[unit_ixs, neighb_ixs]
    else:
        assert candidate_ixs is not None
        assert neighb_ixs is not None
        assert lut_ixs is not None
    if pnoid:
        assert (lut_ixs < lut.unit_ids.shape[0]).all()

    lls = whitenedx.new_full((n, Ctot + int(not skip_noise)), fill_value=-torch.inf)
    if not skip_noise:
        lls[:, -1] = noise_logliks
        lls[:, -1].add_(noise_log_prop)

    if lut_params.signal_rank:
        lls[spike_ixs, candidate_ixs] = _calc_loglik_ppca(
            whitenedx=whitenedx,
            log_proportions=log_proportions,
            spike_ixs=spike_ixs,
            lut_ixs=lut_ixs,
            unit_ixs=unit_ixs,
            constplogdet=lut_params.constplogdet,
            Linvmuo=lut_params.Linvmuo,
            wburyroot=lut_params.wburyroot,
        )
    else:
        lls[spike_ixs, candidate_ixs] = _calc_loglik_rank0(
            whitenedx=whitenedx,
            log_proportions=log_proportions,
            spike_ixs=spike_ixs,
            lut_ixs=lut_ixs,
            unit_ixs=unit_ixs,
            constplogdet=lut_params.constplogdet,
            Linvmuo=lut_params.Linvmuo,
        )

    if do_resp:
        toplls, topinds = torch.topk(lls[:, :-1], n_candidates, dim=1)
        candidates = candidates.take_along_dim(indices=topinds, dim=1)
        toplls = torch.concatenate((toplls, lls[:, -1:]), dim=1)
    else:
        toplls = lls

    if do_resp:
        responsibilities = torch.softmax(toplls, dim=1)
    else:
        responsibilities = fixed_responsibilities

    return Scores(
        log_liks=toplls, responsibilities=responsibilities, candidates=candidates
    )


def _sparsify_candidates(
    candidates: Tensor, neighborhood_ids: Tensor, static_size: int | None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    cpos = candidates >= 0
    if pnoid:
        assert cpos.any(dim=1).all()
    if pnoid and static_size is not None:
        assert cpos.sum() == static_size
    if static_size is not None:
        spike_ixs, candidate_ixs = cpos.nonzero_static(size=static_size).T
    else:
        spike_ixs, candidate_ixs = cpos.nonzero(as_tuple=True)
    neighb_ixs = neighborhood_ids[spike_ixs]
    unit_ixs = candidates[spike_ixs, candidate_ixs]
    return spike_ixs, candidate_ixs, unit_ixs, neighb_ixs


# Faster inv quad term in log likelihoods
# We want to compute
#     (x - nu)' [Co + Wo Wo']^-1 (x - nu)
#         = (x-nu)' [Co^-1 - Co^-1 Wo(I_m+Wo'Co^-1Wo)^-1Wo'Co^-1] (x-nu)
# Let's say we already have computed...
#     %  name in code: whitenednu
#     z  = Co^{-1/2}nu
#     %  name in code: whitenedx
#     x' = Co^{-1/2} x
#     %  name in code: wburyroot
#     A  = (I_m+Wo'Co^-1Wo)^{-1/2} Wo'Co^{-1/2}
# Break into terms. First,
#     (A+)  (x-nu)'Co^-1(x-nu) = |x' - z|^2
# It doesn't seem helpful to break that down any further.
# Next up, while we have x' - z computed, notice that
#     (B-) (x-nu)' Co^-1 Wo(I_m+Wo'Co^-1Wo)^-1Wo'Co^-1 (x-nu)
#             = | A (x'-z') |^2.


@torch.jit.script
def _calc_loglik_rank0(
    *,
    whitenedx: Tensor,
    log_proportions: Tensor,
    spike_ixs: Tensor,
    lut_ixs: Tensor,
    unit_ixs: Tensor,
    constplogdet: Tensor,
    Linvmuo: Tensor,
):
    wdxz = whitenedx[spike_ixs]
    wdxz -= Linvmuo[lut_ixs]
    ll = wdxz.square_().sum(dim=1)
    ll += constplogdet[lut_ixs]
    ll *= -0.5
    ll += log_proportions[unit_ixs]
    return ll


@torch.jit.script
def _calc_loglik_ppca(
    *,
    whitenedx: Tensor,
    log_proportions: Tensor,
    spike_ixs: Tensor,
    lut_ixs: Tensor,
    unit_ixs: Tensor,
    constplogdet: Tensor,
    Linvmuo: Tensor,
    wburyroot: Tensor,
):
    wdxz = whitenedx[spike_ixs]
    wdxz -= Linvmuo[lut_ixs]

    term_b = wdxz[:, None].bmm(wburyroot[lut_ixs])[:, 0]
    term_a = wdxz.square_().sum(dim=1)
    term_b = term_b.square_().sum(dim=1)

    ll = term_a.sub_(term_b)

    ll += constplogdet[lut_ixs]
    ll *= -0.5
    ll += log_proportions[unit_ixs]
    return ll


def _stat_pass_batch(
    *,
    xt: Tensor,
    whitenedx: Tensor,
    CmoCooinvx: Tensor,
    responsibilities: Tensor,
    log_liks: Tensor,
    spike_ixs: Tensor,
    candidate_ixs: Tensor,
    unit_ixs: Tensor,
    neighb_ixs: Tensor,
    lut_ixs: Tensor,
    n_candidates: int,
    n_units: int,
    neighb_cov: NeighborhoodCovariance,
    lut_params: LUTParams,
    eps: Tensor,
) -> SufficientStatistics:
    n = responsibilities.shape[0]
    if lut_params.signal_rank:
        assert lut_params.TWoCooinvsqrt is not None
        assert lut_params.TWoCooinvmuo is not None
        assert lut_params.Tpad is not None
        noise_N, N, Nlut, Ulut, R, elb = _stat_pass_batch_ppca(
            xt=xt,
            whitenedx=whitenedx,
            CmoCooinvx=CmoCooinvx,
            responsibilities=responsibilities,
            log_liks=log_liks,
            spike_ixs=spike_ixs,
            candidate_ixs=candidate_ixs,
            unit_ixs=unit_ixs,
            neighb_ixs=neighb_ixs,
            lut_ixs=lut_ixs,
            n_candidates=n_candidates,
            n_units=n_units,
            neighb_cov_obs_ix=neighb_cov.obs_ix,
            neighb_cov_miss_near_ix=neighb_cov.miss_near_ix,
            feat_rank=neighb_cov.feat_rank,
            n_channels=neighb_cov.n_channels,
            max_nc_obs=neighb_cov.max_nc_obs,
            max_nc_miss_near=neighb_cov.max_nc_miss_near,
            lut_params_TWoCooinvsqrt=lut_params.TWoCooinvsqrt,
            lut_params_TWoCooinvmuo=lut_params.TWoCooinvmuo,
            lut_params_Tpad=lut_params.Tpad,
            hat_dim=lut_params.signal_rank + 1,
            n_lut=lut_params.n_lut,
            eps=eps,
        )
        stats = SufficientStatistics(
            count=n, noise_N=noise_N, N=N, Nlut=Nlut, Ulut=Ulut, R=R, elbo=elb
        )
    else:
        noise_N, N, Nlut, R, elb = _stat_pass_batch_rank0(
            xt=xt,
            CmoCooinvx=CmoCooinvx,
            responsibilities=responsibilities,
            log_liks=log_liks,
            spike_ixs=spike_ixs,
            candidate_ixs=candidate_ixs,
            unit_ixs=unit_ixs,
            neighb_ixs=neighb_ixs,
            lut_ixs=lut_ixs,
            neighb_cov_obs_ix=neighb_cov.obs_ix,
            neighb_cov_miss_near_ix=neighb_cov.miss_near_ix,
            feat_rank=neighb_cov.feat_rank,
            n_channels=neighb_cov.n_channels,
            n_candidates=n_candidates,
            max_nc_obs=neighb_cov.max_nc_obs,
            max_nc_miss_near=neighb_cov.max_nc_miss_near,
            n_units=n_units,
            n_lut=lut_params.n_lut,
            eps=eps,
        )
        stats = SufficientStatistics(
            count=n, noise_N=noise_N, N=N, Nlut=Nlut, Ulut=None, R=R, elbo=elb
        )
    return stats


@torch.jit.script
def _count_batch(
    *,
    responsibilities: Tensor,
    n_candidates: int,
    spike_ixs: Tensor,
    candidate_ixs: Tensor,
    unit_ixs: Tensor,
    lut_ixs: Tensor,
    n_units: int,
    n_lut: int,
    eps: Tensor,
) -> tuple[Tensor | None, Tensor, Tensor, Tensor, Tensor]:
    have_noise = responsibilities.shape[1] == n_candidates + 1
    if have_noise:
        noise_N = responsibilities[:, -1].sum()
    else:
        noise_N = None

    Qflat = responsibilities[spike_ixs, candidate_ixs]
    N = Qflat.new_zeros(n_units)
    Nlut = Qflat.new_zeros(n_lut)

    N.scatter_add_(dim=0, index=unit_ixs, src=Qflat)
    Nlut.scatter_add_(dim=0, index=lut_ixs, src=Qflat)

    Qnflat = N[unit_ixs].clamp_(min=eps)
    Qnflat = torch.div(Qflat, Qnflat, out=Qnflat)

    Qnflatlut = Nlut[lut_ixs].clamp_(min=eps)
    Qnflatlut = torch.div(Qflat, Qnflatlut, out=Qnflatlut)

    return noise_N, N, Nlut, Qnflat, Qnflatlut


@torch.jit.script
def _stat_pass_batch_ppca(
    *,
    xt: Tensor,
    whitenedx: Tensor,
    CmoCooinvx: Tensor,
    responsibilities: Tensor,
    log_liks: Tensor,
    spike_ixs: Tensor,
    candidate_ixs: Tensor,
    unit_ixs: Tensor,
    neighb_ixs: Tensor,
    lut_ixs: Tensor,
    n_candidates: int,
    n_units: int,
    neighb_cov_obs_ix: Tensor,
    neighb_cov_miss_near_ix: Tensor,
    feat_rank: int,
    n_channels: int,
    max_nc_obs: int,
    max_nc_miss_near: int,
    lut_params_TWoCooinvsqrt: Tensor,
    lut_params_TWoCooinvmuo: Tensor,
    lut_params_Tpad: Tensor,
    hat_dim: int,
    n_lut: int,
    eps: Tensor,
) -> tuple[Tensor | None, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    neighb_ixs, lut_ixs, candidate_ixs, unit_ixs, spike_ixs are all the
    same shape, and are sparse indices of the valid candidates for this batch.
    I.e., if candidates was (n, C), then `spike_ixs, candidate_ixs` are the
    tuple result of nonzero; unit_ixs is `candidates[spike_ixs, candidate_ixs]`,
    lut_ixs are the result of combining those with neighb_ixs and going
    to the LUT.

    Arguments
    ---------
    responsibilities: (n, C or C + 1)
        Last index is the noise dimension, if present. If it's present,
        candidates must appear too. If candidates don't appear, it's not present.
    candidates: None or (n, C)
    """
    nsp = spike_ixs.shape[0]

    # launch load / init kernels at top
    R = xt.new_zeros((n_units * (n_channels + 1), hat_dim, feat_rank))
    Ulut = torch.zeros_like(lut_params_Tpad)
    whitenedxsp = whitenedx[spike_ixs]
    obs_ix = neighb_cov_obs_ix[neighb_ixs]
    miss_near_ix = neighb_cov_miss_near_ix[neighb_ixs]
    TWoCooinvsqrt = lut_params_TWoCooinvsqrt[lut_ixs]
    TWoCooinvmuo = lut_params_TWoCooinvmuo[lut_ixs]

    noise_N, N, Nlut, Qnflat, Qnflatlut = _count_batch(
        responsibilities=responsibilities,
        n_candidates=n_candidates,
        spike_ixs=spike_ixs,
        candidate_ixs=candidate_ixs,
        unit_ixs=unit_ixs,
        lut_ixs=lut_ixs,
        n_units=n_units,
        n_lut=n_lut,
        eps=eps,
    )

    # construct U-related stuff
    ubar = TWoCooinvmuo[:, :, None].baddbmm(
        TWoCooinvsqrt, whitenedxsp[:, :, None], beta=-1.0
    )
    ubar = ubar[:, :, 0]
    hatubar = F.pad(ubar, (0, 1), value=1.0)
    hatU = lut_params_Tpad[lut_ixs]

    # not saving the bottom row, Tpad should just be 0padded on inner dim
    hatU.addcmul_(ubar[:, :, None], hatubar[:, None, :])

    # U sufficient stat (LUT binned)
    hatU *= Qnflatlut[:, None, None]
    ix = lut_ixs[:, None, None].broadcast_to(hatU.shape)
    Ulut.scatter_add_(dim=0, index=ix, src=hatU)

    # construct R-related stuff
    # NB we're saving flops here by skipping the second term and coming
    # back to it later -- it doesn't depend on data so can be had for cheap
    # we also save work by multiplying hatubar by the weights. then Ro, Rm
    # below are multiplied by weights, and it propagates through to R.
    hatubar.mul_(Qnflat[:, None])
    xc = xt[spike_ixs]
    # nsp, chan, hat dim, feat_rank
    Ro = hatubar[:, None, :, None] * xc[:, :, None, :]

    CmoCooinvxc = CmoCooinvx[spike_ixs]
    # nsp, chan, hat dim, feat_rank
    Rm = hatubar[:, None, :, None] * CmoCooinvxc[:, :, None, :]

    # the code below drops Ro, Rm into the right unit-channel indices of R
    # this used to be different, with a (nsp,hat,feat,chans) intermediate, which is slower.

    ncuix = (n_channels + 1) * unit_ixs[:, None]
    ixo = ncuix + obs_ix
    Ro = Ro.view(nsp * max_nc_obs, hat_dim, feat_rank)
    ixo = ixo.view(-1, 1, 1).broadcast_to(Ro.shape)
    R.scatter_add_(dim=0, index=ixo, src=Ro)

    ixm = ncuix + miss_near_ix
    Rm = Rm.view(nsp * max_nc_miss_near, hat_dim, feat_rank)
    ixm = ixm.view(-1, 1, 1).broadcast_to(Rm.shape)
    R.scatter_add_(dim=0, index=ixm, src=Rm)

    R = R.view(n_units, n_channels + 1, hat_dim, feat_rank)
    R = R[:, :n_channels]

    # objective
    elb = mean_elbo_dim1(responsibilities, log_liks)

    return noise_N, N, Nlut, Ulut, R, elb


@torch.jit.script
def _stat_pass_batch_rank0(
    *,
    xt: Tensor,
    CmoCooinvx: Tensor,
    responsibilities: Tensor,
    log_liks: Tensor,
    spike_ixs: Tensor,
    candidate_ixs: Tensor,
    unit_ixs: Tensor,
    neighb_ixs: Tensor,
    lut_ixs: Tensor,
    neighb_cov_obs_ix: Tensor,
    neighb_cov_miss_near_ix: Tensor,
    feat_rank: int,
    n_channels: int,
    n_candidates: int,
    max_nc_obs: int,
    max_nc_miss_near: int,
    n_units: int,
    n_lut: int,
    eps: Tensor,
) -> tuple[Tensor | None, Tensor, Tensor, Tensor, Tensor]:
    """
    neighb_ixs, lut_ixs, candidate_ixs, unit_ixs, spike_ixs are all the
    same shape, and are sparse indices of the valid candidates for this batch.
    I.e., if candidates was (n, C), then `spike_ixs, candidate_ixs` are the
    tuple result of nonzero; unit_ixs is `candidates[spike_ixs, candidate_ixs]`,
    lut_ixs are the result of combining those with neighb_ixs and going
    to the LUT.

    Arguments
    ---------
    responsibilities: (n, C or C + 1)
        Last index is the noise dimension, if present. If it's present,
        candidates must appear too. If candidates don't appear, it's not present.
    candidates: None or (n, C)
    """
    nsp = spike_ixs.shape[0]

    # launch load / init kernels at top
    xc = xt[spike_ixs]
    CmoCooinvxc = CmoCooinvx[spike_ixs]
    obs_ix = neighb_cov_obs_ix[neighb_ixs]
    miss_near_ix = neighb_cov_miss_near_ix[neighb_ixs]
    R = xt.new_zeros((n_units * (n_channels + 1), 1, feat_rank))

    noise_N, N, Nlut, Qnflat, Qnflatlut = _count_batch(
        responsibilities=responsibilities,
        n_candidates=n_candidates,
        spike_ixs=spike_ixs,
        candidate_ixs=candidate_ixs,
        unit_ixs=unit_ixs,
        lut_ixs=lut_ixs,
        n_units=n_units,
        n_lut=n_lut,
        eps=eps,
    )

    # construct R-related stuff
    # NB we're saving flops here by skipping the second term and coming
    # back to it later -- it doesn't depend on data so can be had for cheap
    # nsp, chan, hat dim, feat_rank
    Ro = xc[:, :, None, :].mul_(Qnflat[:, None, None, None])
    Rm = CmoCooinvxc[:, :, None, :].mul_(Qnflat[:, None, None, None])

    # gather (ahem) Ro, Rm onto full channel set
    ncuix = (n_channels + 1) * unit_ixs[:, None]
    ixo = ncuix + obs_ix
    Ro = Ro.view(nsp * max_nc_obs, 1, feat_rank)
    ixo = ixo.view(-1, 1, 1).broadcast_to(Ro.shape)
    R.scatter_add_(dim=0, index=ixo, src=Ro)

    ixm = ncuix + miss_near_ix
    Rm = Rm.view(nsp * max_nc_miss_near, 1, feat_rank)
    ixm = ixm.view(-1, 1, 1).broadcast_to(Rm.shape)
    R.scatter_add_(dim=0, index=ixm, src=Rm)

    R = R.view(n_units, n_channels + 1, 1, feat_rank)
    R = R[:, :n_channels]

    # objective
    elb = mean_elbo_dim1(responsibilities, log_liks)

    return noise_N, N, Nlut, R, elb


def _finalize_e_stats(
    *,
    means: Tensor,
    bases: Tensor | None,
    stats: SufficientStatistics,
    lut: NeighborhoodLUT,
    lut_params: LUTParams,
    neighb_cov: NeighborhoodCovariance,
    prior_pseudocount: float = 0.0,
    batch_size=64,
):
    """Finish the calculation of R in place."""

    # first, transpose R... (this is a bit of baggage from formerly having
    # features always as feature-rank major, channels minor)
    n_units = means.shape[0]
    hat_dim = lut_params.signal_rank + 1
    frank = neighb_cov.feat_rank
    nc = neighb_cov.n_channels
    assert stats.R.shape == (n_units, nc, hat_dim, frank)
    stats.R = stats.R.permute(0, 2, 3, 1).reshape(n_units, hat_dim, frank * nc)

    assert (lut_params.signal_rank == 0) == (bases is None)
    assert means.ndim == 2

    if prior_pseudocount:
        term = means @ neighb_cov.full_Linv.T
        term = term.square_().sum(dim=1).mean()
        term *= -0.5 * prior_pseudocount
        stats.elbo += term

    if prior_pseudocount and bases is not None:
        K, r = bases.shape[:2]
        term = bases.view(K * r, -1) @ neighb_cov.full_Linv.T
        term = term.square_().sum(dim=1).view(K, r).mean(dim=0).sum()
        term *= -0.5 * prior_pseudocount
        stats.elbo += term

    # storage
    ncm = neighb_cov.max_nc_miss_near
    What_batch = means.new_zeros((batch_size, hat_dim, frank, nc + 1))
    Uhat_batch = means.new_ones((batch_size, hat_dim, hat_dim))

    # reweighting
    denom = stats.N[lut.unit_ids].clamp_(min=torch.finfo(stats.N.dtype).tiny)
    Nlut_N = torch.div(stats.Nlut, denom, out=denom)
    if pnoid:
        assert torch.isfinite(Nlut_N).all()

    # loop because w_wcc is big
    for i0 in range(0, lut_params.n_lut, batch_size):
        # batch indices
        i1 = min(lut_params.n_lut, i0 + batch_size)
        uu = lut.unit_ids[i0:i1]
        nn = lut.neighb_ids[i0:i1]
        What = What_batch[: i1 - i0]
        Uhat = Uhat_batch[: i1 - i0]

        # fill in the blanks
        What[:, -1:, :, :-1] = means[uu, None].view(i1 - i0, 1, frank, nc)
        if lut_params.signal_rank:
            assert stats.Ulut is not None
            assert bases is not None
            Uhat[:, :-1, :] = stats.Ulut[i0:i1, :, :]
            Uhat[:, -1:, :-1] = stats.Ulut[i0:i1, :, -1:].mT
            What[:, :-1, :, :-1] = bases[uu].view(i1 - i0, hat_dim - 1, frank, nc)

        obs_ix = neighb_cov.obs_ix[nn][:, None, None, :]
        Whatobs = What.take_along_dim(dim=3, indices=obs_ix)
        Whatobs = Whatobs.view(i1 - i0, hat_dim, frank * neighb_cov.max_nc_obs)
        WoCooinvCom = Whatobs.bmm(neighb_cov.CooinvCom[nn])
        WoCooinvCom = WoCooinvCom.view(i1 - i0, hat_dim, frank, ncm)

        w_wcc = What  # first make Wmiss in place
        ix = neighb_cov.miss_near_ix[nn, None, None].broadcast_to(WoCooinvCom.shape)
        w_wcc.scatter_add_(dim=3, index=ix, src=WoCooinvCom._neg_view())
        w_wcc = w_wcc[..., :-1]
        w_wcc *= neighb_cov.miss_full_mask[nn][:, None, None, :]
        w_wcc = w_wcc.reshape(i1 - i0, hat_dim, frank * nc)

        # apply reweighting
        if lut_params.signal_rank:
            Uhat_w_w_cc = Uhat.bmm(w_wcc)
        else:
            Uhat_w_w_cc = w_wcc
        Uhat_w_w_cc *= Nlut_N[i0:i1, None, None]

        # add to corresponding positions in R
        ix = uu[:, None, None].broadcast_to(Uhat_w_w_cc.shape)
        stats.R.scatter_add_(dim=0, index=ix, src=Uhat_w_w_cc)


def _get_u_from_ulut(lut: NeighborhoodLUT, stats: SufficientStatistics):
    assert stats.Ulut is not None
    assert stats.Nlut is not None
    n_units = stats.N.numel()
    hat_dim = stats.Ulut.shape[2]

    Nz = (stats.N == 0).to(dtype=stats.N.dtype)

    Nlut_N = stats.Nlut / (stats.N + Nz)[lut.unit_ids]
    assert Nlut_N.isfinite().all()
    stats.Ulut *= Nlut_N[:, None, None]

    U = stats.Ulut.new_zeros((n_units, hat_dim, hat_dim))
    ix = lut.unit_ids[:, None, None].broadcast_to(stats.Ulut.shape)
    U[:, :-1, :].scatter_add_(dim=0, index=ix, src=stats.Ulut)
    U[:, -1:, :-1] = U[:, :-1, -1:].mT
    U[:, -1, -1].fill_(1.0)

    # fill blanks if present (N=0 implies U=0)
    U.diagonal(dim1=-2, dim2=-1).add_(Nz[:, None])

    return U


def _log_warn_or_raise_coverage(adj, neighborhood_ids, n_steps, needs_raise):
    (uncovered_neighbs,) = (adj.sum(0) == 0).cpu().nonzero(as_tuple=True)
    unique_neighbs, ucounts = neighborhood_ids.unique(return_counts=True)
    unique_neighbs = unique_neighbs.cpu()
    ucounts = ucounts.cpu()
    assert unique_neighbs.max() + 1 <= adj.shape[1]
    counts = torch.zeros(adj.shape[1], dtype=ucounts.dtype)
    counts[unique_neighbs] = ucounts
    uncovered_counts = counts[uncovered_neighbs].tolist()
    n_uncovered = sum(uncovered_counts)
    pct_uncovered = 100.0 * (n_uncovered / neighborhood_ids.shape[0])
    uncovered_neighbs = uncovered_neighbs.tolist()
    message = (
        f"Neighborhood coverage was not complete, with {len(uncovered_neighbs)} uncovered "
        f"neighborhoods and {n_uncovered} uncovered spikes ({pct_uncovered:.3f}% of set). "
        f"Neighborhoods and counts were: {uncovered_neighbs}, {uncovered_counts}. It took "
        f"{n_steps} expansion steps to fill out the coverage."
    )
    if needs_raise:
        raise ValueError(message)
    elif pct_uncovered < 0.05:
        logger.dartsortverbose(message)
    else:
        warnings.warn(message, stacklevel=2)
