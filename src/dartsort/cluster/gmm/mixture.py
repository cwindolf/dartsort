""" """

import math
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Literal, Optional, Self

import numpy as np
import torch
import torch.nn.functional as F
from sympy.utilities.iterables import multiset_partitions
from torch import Tensor
from tqdm.auto import trange, tqdm

from ...util.data_util import DARTsortSorting, subset_sorting_by_spike_count
from ...util.internal_config import (
    ComputationConfig,
    DARTsortInternalConfig,
    RefinementConfig,
)
from ...util.job_util import ensure_computation_config
from ...util.logging_util import get_logger, DARTSORTDEBUG, DARTSORTVERBOSE
from ...util.main_util import ds_save_intermediate_labels
from ...util.noise_util import EmbeddedNoise
from ...util.py_util import databag
from ...util.spiketorch import cosine_distance, elbo, entropy, sign, spawn_torch_rg
from ...util.torch_util import BModule
from ..cluster_util import linkage, maximal_leaf_groups
from ..kmeans import kmeans
from .stable_features import (
    NeighborhoodInterpolator,
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
    computation_cfg = ensure_computation_config(computation_cfg)
    device = computation_cfg.actual_device()
    rg = np.random.default_rng(seed)
    saving = save_cfg is not None and save_cfg.save_intermediate_labels
    if saving:
        assert save_step_labels_format is not None
    prog_level = 1 + logger.isEnabledFor(DARTSORTVERBOSE)

    sorting = subset_sorting_by_spike_count(
        sorting, min_spikes=refinement_cfg.min_count
    )
    sorting = sorting.flatten()

    global pnoid
    pnoid = logger.isEnabledFor(DARTSORTVERBOSE)
    if pnoid:
        logger.dartsortverbose("Extra TMM asserts are on.")

    # TODO: won't want to do this full data stuff except at the very very end
    neighb_cov, erp, train_data, val_data, full_data, noise = get_truncated_datasets(
        sorting, motion_est, refinement_cfg, device, rg
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

    # start with one round of em. below flow is like split-em-merge-em-repeat.
    tmm.em(train_data)

    for outer_it in range(refinement_cfg.n_total_iters):
        # what will we do this iteration?
        do_split = bool(outer_it) or not refinement_cfg.skip_first_split
        break_after_split = refinement_cfg.one_split_only

        if do_split:
            if val_data is not None:
                eval_scores = tmm.soft_assign(
                    val_data, needs_bootstrap=True, show_progress=prog_level
                )
            else:
                eval_scores = tmm.soft_assign(
                    train_data,
                    needs_bootstrap=False,
                    max_iter=1,
                    show_progress=prog_level,
                )
            split_res = tmm.split(
                train_data, val_data, scores=eval_scores, show_progress=prog_level > 0
            )
            logger.info(f"Split created {split_res.n_new_units} new units.")
            tmm.em(train_data, show_progress=prog_level)

        if saving:
            assert save_step_labels_format is not None
            full_scores = tmm.soft_assign(full_data, needs_bootstrap=True)
            sorting = replace(sorting, labels=labels_from_scores(full_scores))
            stepname = f"tmm{outer_it}asplit"
            ds_save_intermediate_labels(
                save_step_labels_format.format(stepname=stepname),
                sorting,
                save_step_labels_dir,
                save_cfg,
            )

        if break_after_split:
            break

        if val_data is not None:
            eval_scores = tmm.soft_assign(val_data, needs_bootstrap=True)
        else:
            eval_scores = tmm.soft_assign(train_data, needs_bootstrap=False, max_iter=1)
        merge_map = tmm.merge(
            train_data, val_data, scores=eval_scores, show_progress=prog_level > 0
        )
        logger.info(f"Merge {merge_map.mapping.shape[0]} -> {merge_map.nuniq()} units.")
        tmm.em(train_data, show_progress=prog_level)
        if saving:
            assert save_step_labels_format is not None
            full_scores = tmm.soft_assign(full_data, needs_bootstrap=True)
            sorting = replace(sorting, labels=labels_from_scores(full_scores))
            stepname = f"tmm{outer_it}asplit"
            ds_save_intermediate_labels(
                save_step_labels_format.format(stepname=stepname),
                sorting,
                save_step_labels_dir,
                save_cfg,
            )

    # final assignments
    # TODO output these soft probs somehow
    full_scores = tmm.soft_assign(full_data, needs_bootstrap=True)
    sorting = replace(sorting, labels=labels_from_scores(full_scores))
    return sorting


# -- shared objects holding precomputed neighborhood-related data


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
            neighborhoods, noise.zero_radius, prgeom.to(device=dev)
        )
        logdet, Cooinv, CooinvCom, Linv = _noise_factors(
            noise, obs_ix, miss_near_ix, cache_prefix=neighborhoods.name
        )
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
    TWoCooinv: Tensor | None
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
            TWoCooinv=ro,
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
        self.Linvmuo.resize_(n_lut_new, *self.Linvmuo.shape[1:])
        self.CmoCooinvmuo.resize_(n_lut_new, *self.CmoCooinvmuo.shape[1:])
        self.constplogdet.resize_(n_lut_new, *self.constplogdet.shape[1:])
        if not self.signal_rank:
            return
        self.TWoCooinv.resize_(n_lut_new, *self.TWoCooinv.shape[1:])  # type: ignore
        self.TWoCooinvmuo.resize_(n_lut_new, *self.TWoCooinvmuo.shape[1:])  # type: ignore
        self.Tpad.resize_(n_lut_new, *self.Tpad.shape[1:])  # type: ignore
        self.wburyroot.resize_(n_lut_new, *self.wburyroot.shape[1:])  # type: ignore


# -- messenger classes


@databag
class Scores:
    log_liks: Tensor
    responsibilities: Tensor | None
    candidates: Tensor

    def slice(self, indices: Tensor) -> "Scores":
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
        feature_dim: int,
        signal_rank: int,
        device: torch.device,
        count_dtype=torch.double,
    ) -> Self:
        hat_dim = signal_rank + 1
        if signal_rank:
            Ulut = torch.zeros((n_lut, signal_rank, hat_dim), device=device)
        else:
            Ulut = None
        return cls(
            count=0,
            noise_N=torch.zeros((), device=device, dtype=count_dtype),
            N=torch.zeros((n_units,), device=device, dtype=count_dtype),
            Nlut=torch.zeros((n_lut,), device=device, dtype=count_dtype),
            R=torch.zeros((n_units, hat_dim, feature_dim), device=device),
            Ulut=Ulut,
            elbo=torch.zeros((), device=device, dtype=count_dtype),
        )

    def combine(self, other: "SufficientStatistics"):
        """Welford running means."""
        eps = torch.finfo(self.N.dtype).tiny
        self.count += other.count
        w_count = other.count / max(1, self.count)
        self.N += other.N
        w_N = other.N.div_(self.N.clamp(min=eps))
        self.Nlut += other.Nlut

        self.elbo += (other.elbo - self.elbo) * w_count
        self.R += other.R.sub_(self.R).mul_(w_N[:, None, None])
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
class SpikeDataBatch:
    """Yielded by data object .batches()"""

    batch: Tensor | slice
    neighborhood_ids: Tensor
    x: Tensor
    candidates: Tensor
    whitenedx: Tensor
    noise_logliks: Tensor
    CmoCooinvx: Tensor | None


@databag
class DenseSpikeData:
    """Used in split/merge, like a batch but no candidates + extra logic useful for fitting."""

    indices: Tensor
    neighborhoods: SpikeNeighborhoods
    neighb_supset: Tensor
    neighborhood_ids: Tensor
    x: Tensor
    whitenedx: Tensor
    CmoCooinvx: Tensor
    noise_logliks: Tensor

    def to_batch(self, unit_ids: Tensor, lut: NeighborhoodLUT):
        """Convert dense data to a batch, which means picking candidates.

        The candidates are all of the unit ids, so that each col of the candidates
        array is just unit_ids[col] repeated. The thing is that not all units overlap
        all neighborhoods, so some of these will be -1s per the lut. But we can still
        think of each col as corresponding to a unit.
        """
        unit_ids = torch.as_tensor(unit_ids, device=self.neighborhood_ids.device)
        lut_ixs = lut.lut[unit_ids[None, :], self.neighborhood_ids[:, None]]
        covered = lut_ixs < lut.unit_ids.shape[0]
        candidates = torch.where(covered, unit_ids[None, :], -1)
        return SpikeDataBatch(
            batch=self.indices,
            neighborhood_ids=self.neighborhood_ids,
            x=self.x,
            whitenedx=self.whitenedx,
            noise_logliks=self.noise_logliks,
            CmoCooinvx=self.CmoCooinvx,
            candidates=candidates,
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
        assert weights.shape[0] == self.x.shape[0]
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
        candidates: Tensor,
        neighborhoods: SpikeNeighborhoods,
        device: torch.device,
        seed: int | np.random.Generator = 0,
        batch_size: int = 128,
        explore_neighb_steps: int = 1,
        neighb_overlap: float = 0.75,
    ):
        self.N = N
        self.n_candidates = n_candidates
        self.n_search = n_search
        self.n_explore = n_explore
        self.batch_size = batch_size
        self.n_search_total = n_candidates * n_search
        self.n_total = n_candidates + self.n_search_total + self.n_explore
        self.search_slice = slice(n_candidates, n_candidates + self.n_search_total)
        self.explore_slice = slice(n_candidates + self.n_search_total, self.n_total)

        self.candidates = candidates

        # adjacencies...
        self.neighborhoods = neighborhoods
        self.neighborhood_ids = neighborhoods.b.neighborhood_ids
        self.explore_neighb_steps = explore_neighb_steps
        self.neighb_adj = neighborhoods.adjacency(neighb_overlap)
        self.neighb_supset = neighborhoods.partial_order().float()
        nneighb = self.neighborhoods.n_neighborhoods
        assert self.neighb_supset.shape == (nneighb, nneighb)

        self.device = device
        self.rg = spawn_torch_rg(seed, device=device)

        assert self.candidates.shape[0] == N
        assert self.candidates.shape[1] >= n_candidates

    def batches(
        self, show_progress: bool = False, desc: str = "Batches"
    ) -> Iterable[SpikeDataBatch]:
        if show_progress:
            batch_starts = trange(0, self.N, self.batch_size, desc=desc)
        else:
            batch_starts = range(0, self.N, self.batch_size)
        for i0 in batch_starts:
            batch = slice(i0, min(self.N, i0 + self.batch_size))
            yield self.batch(batch)

    def batch(self, spike_indices: Tensor | slice) -> SpikeDataBatch:
        raise NotImplementedError

    def update_adjacency(self, n_units: int, un_adj_lut: NeighborhoodLUT | None = None):
        """Update my adjacency either using my labels or just a fixed LUT."""
        self.un_adj_lut, self.un_adj, self.explore_adj = candidate_adjacencies(
            labels=self.candidates[:, 0],
            neighb_supset=self.neighb_supset,
            neighb_adj=self.neighb_adj,
            n_units=n_units,
            explore_steps=self.explore_neighb_steps,
            neighborhoods=self.neighborhoods,
            un_adj_lut=un_adj_lut,
        )
        assert self.un_adj.max().item() == 1.0

    def erase_candidates(self):
        self.candidates.fill_(-1)

    def bootstrap_candidates(
        self, distances: Tensor, un_adj_lut: NeighborhoodLUT | None = None
    ) -> NeighborhoodLUT:
        self.update_adjacency(n_units=distances.shape[0], un_adj_lut=un_adj_lut)

        # fill in missing labels randomly, obeying un_adj
        _fill_blank_labels(
            labels=self.candidates[:, 0],
            un_adj=self.un_adj,
            neighborhood_ids=self.neighborhood_ids,
            gen=self.rg,
        )
        assert (self.candidates[:, 0] >= 0).all()
        assert torch.equal(
            self.un_adj,
            (self.un_adj_lut.lut < self.un_adj_lut.unit_ids.shape[0]).float(),
        )

        # fill in candidates[:, 1:n_candidates] at random obeying un_adj
        # choosing not to use distances here, since they get used in search sets
        _bootstrap_top(
            candidates=self.candidates,
            neighborhood_ids=self.neighborhood_ids,
            n_candidates=self.n_candidates,
            un_adj=self.un_adj,
            un_adj_lut=self.un_adj_lut,
            gen=self.rg,
        )

        # call update to fill in search + explore and build lut
        return self.update(new_top_candidates=None, distances=distances)

    def update(self, new_top_candidates: Tensor | None, distances: Tensor):
        """Subclasses do what they need to do to update their candidates."""
        raise NotImplementedError


class OnlineSpikeData(BatchedSpikeData):
    """Like a TruncatedSpikeData, but only stores x, and transfers to device on the fly."""

    def __init__(
        self,
        *,
        n_candidates: int,
        n_search: int,
        n_explore: int,
        x: Tensor,
        neighborhoods: SpikeNeighborhoods,
        device: torch.device,
        neighb_cov: NeighborhoodCovariance,
        seed: int | np.random.Generator = 0,
        batch_size: int = 512,
        explore_neighb_steps: int = 1,
        neighb_overlap: float = 0.75,
    ):
        super().__init__(
            N=x.shape[0],
            n_candidates=n_candidates,
            n_search=n_search,
            n_explore=n_explore,
            batch_size=batch_size,
            neighborhoods=neighborhoods,
            explore_neighb_steps=explore_neighb_steps,
            neighb_overlap=neighb_overlap,
            seed=seed,
            device=device,
            candidates=torch.full((x.shape[0], n_candidates), -1, device=device),
        )
        self.neighb_cov = neighb_cov
        self.x = x

    def update(
        self, new_top_candidates: Tensor | None, distances: Tensor
    ) -> NeighborhoodLUT:
        # fill in top spots. don't update the adjacency, because it can't change
        if new_top_candidates is not None:
            self.candidates[:] = new_top_candidates

        # store everything needed to generate batches of candidates
        # this means storing the search sets and the explore probabilities
        self.search_sets = candidate_search_sets(
            distances, self.un_adj_lut, self.un_adj, self.n_search
        )
        if self.n_explore:
            self.p, self.inds = _get_explore_sampling_data(
                un_adj_lut=self.un_adj_lut,
                explore_adj=self.explore_adj,
                search_sets=self.search_sets,
            )
        return self.un_adj_lut

    def batch(self, spike_indices: Tensor | slice) -> SpikeDataBatch:
        n_pad = self.n_total - self.n_candidates
        top = self.candidates[spike_indices]
        n = top.shape[0]
        neighb_ids = self.neighborhood_ids[spike_indices]
        candidates = F.pad(top, (0, n_pad), value=-1)
        top_lut_ixs = self.un_adj_lut.lut[top, neighb_ids[:, None]]

        torch.take_along_dim(
            self.search_sets[:, None, :],
            dim=0,
            indices=top_lut_ixs[:, :, None],
            out=candidates[:, self.search_slice].view(
                n, self.n_candidates, self.n_search
            ),
        )
        if self.n_explore:
            _sample_explore_candidates(
                p=self.p,
                inds=self.inds,
                candidates=candidates,
                lut_ixs=top_lut_ixs[:, 0],
                n_explore=self.n_explore,
                gen=self.rg,
            )
        _dedup_candidates(candidates)

        x = self.x[spike_indices].to(self.device)
        wx, noise_loglik = _whiten_and_noise_score_batch(x, neighb_ids, self.neighb_cov)

        return SpikeDataBatch(
            batch=spike_indices,
            x=x,
            candidates=candidates,
            neighborhood_ids=neighb_ids,
            whitenedx=wx,
            noise_logliks=noise_loglik,
            CmoCooinvx=None,
        )


class TruncatedSpikeData(BatchedSpikeData):
    def __init__(
        self,
        *,
        n_candidates: int,
        n_search: int,
        n_explore: int,
        dense_slice_size_per_unit: int,
        x: Tensor,
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
        n_total = n_candidates * (n_search + 1) + n_explore
        device = noise_logliks.device
        super().__init__(
            N=x.shape[0],
            n_candidates=n_candidates,
            n_search=n_search,
            n_explore=n_explore,
            batch_size=batch_size,
            neighborhoods=neighborhoods,
            explore_neighb_steps=explore_neighb_steps,
            neighb_overlap=neighb_overlap,
            seed=seed,
            device=device,
            candidates=torch.full((x.shape[0], n_total), -1, device=device),
        )

        assert noise_logliks.shape == (self.N,)
        assert x.device == neighborhoods.neighborhood_ids.device == device
        assert whitenedx.shape == x.shape
        assert neighborhoods.neighborhood_ids.shape == (self.N,)

        self.x = x
        self.whitenedx = whitenedx
        self.CmoCooinvx = CmoCooinvx
        self.noise_logliks = noise_logliks
        self.dense_slice_size_per_unit = dense_slice_size_per_unit

    @classmethod
    def initialize_from_labels(
        cls,
        n_candidates: int,
        n_search: int,
        n_explore: int,
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
        whitenedx, CmoCooinvx, noise_logliks = _whiten_impute_and_noise_score(
            x, neighborhoods, neighb_cov
        )
        self = cls(
            n_candidates=n_candidates,
            n_search=n_search,
            n_explore=n_explore,
            dense_slice_size_per_unit=dense_slice_size_per_unit,
            x=x,
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
        self, new_top_candidates: Tensor | None, distances: Tensor
    ) -> NeighborhoodLUT:
        # fill in top spots
        if new_top_candidates is not None:
            self.candidates[:, : self.n_candidates] = new_top_candidates
            self.update_adjacency(distances.shape[0])

        # -1s in candidates will get n_lut here, thanks to padding of un_adj_lut
        top_lut_ixs = self.un_adj_lut.lut[
            self.candidates[:, : self.n_candidates],
            self.neighborhood_ids[:, None],
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
            p, inds = _get_explore_sampling_data(
                un_adj_lut=self.un_adj_lut,
                explore_adj=self.explore_adj,
                search_sets=search_sets,
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

        # update lut for caller
        lut = lut_from_candidates_and_neighborhoods(
            self.candidates,
            self.neighborhoods,
            neighb_supset=self.neighb_supset,
            n_units=distances.shape[0],
        )
        return lut

    def batch(self, spike_indices: Tensor | slice) -> SpikeDataBatch:
        candidates = self.candidates[spike_indices]
        return SpikeDataBatch(
            batch=spike_indices,
            neighborhood_ids=self.neighborhood_ids[spike_indices],
            x=self.x[spike_indices],
            candidates=candidates,
            CmoCooinvx=self.CmoCooinvx[spike_indices],
            whitenedx=self.whitenedx[spike_indices],
            noise_logliks=self.noise_logliks[spike_indices],
        )

    def dense_slice(self, spike_indices: Tensor) -> DenseSpikeData:
        return DenseSpikeData(
            indices=spike_indices,
            neighborhoods=self.neighborhoods,
            neighborhood_ids=self.neighborhood_ids[spike_indices],
            neighb_supset=self.neighb_supset,
            x=self.x[spike_indices],
            whitenedx=self.whitenedx[spike_indices],
            CmoCooinvx=self.CmoCooinvx[spike_indices],
            noise_logliks=self.noise_logliks[spike_indices],
        )

    def dense_slice_by_unit(self, unit_ids: Tensor | int | None, min_count: int = 0):
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
            ixs = ixs[torch.randperm(nixs)[: self.dense_slice_size_per_unit]]
            ixs = torch.msort(ixs)

        return self.dense_slice(ixs)

    def remap(self, remapping: UnitRemapping, distances: Tensor) -> NeighborhoodLUT:
        """Re-map my top candidate labels and re-do LUTs, search, explore."""
        n_units_orig = remapping.mapping.shape[0]
        assert distances.shape[0] <= n_units_orig
        assert remapping.mapping.max() + 1 == distances.shape[0]

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
        return self.update(new_top_candidates=None, distances=distances)

    def update_from_split(
        self, split_mask: Tensor, split_labels: Tensor, distances: Tensor
    ) -> NeighborhoodLUT:
        """Erase candidates for spikes in the split, replace labels, re-bootstrap."""
        (split_ix,) = split_mask.nonzero(as_tuple=True)
        self.candidates[split_ix] = -1
        self.candidates[split_ix, 0] = split_labels[split_ix]
        # have to do a full bootstrap, bc it's hard to figure out what to do with
        # spikes whose candidates contain the units that were split. this way, the
        # lut invariants are maintained, and at least the top labels are the same.
        return self.bootstrap_candidates(distances)


class BaseMixtureModel(BModule):
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
        min_channel_count: int = 1,
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

    # -- subclasses implement

    def unit_slice(self, unit_ids: Tensor) -> "TMMView":
        raise NotImplementedError

    def get_params_at(self, indices: Tensor) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError

    @property
    def centroids(self) -> Tensor:
        """Interface used by unit_distance_matrix."""
        raise NotImplementedError

    def non_noise_log_proportion(self):
        raise NotImplementedError

    def score(self, data: DenseSpikeData) -> Scores:
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
        distance_kind: Literal["cosine"] = "cosine",
        lut_puff: float = 1.5,
        seed: int | np.random.Generator = 0,
        min_channel_count: int = 1,
        elbo_atol: float = 1e-4,
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
        )
        self.min_count = min_count
        self.split_k = split_k
        self.lut_puff = lut_puff
        self.elbo_atol = elbo_atol
        self.register_buffer("log_proportions", log_proportions)
        self.register_buffer_or_none("noise_log_prop", noise_log_prop)
        self.register_buffer("means", means)
        self.register_buffer_or_none("bases", bases)
        if bases is not None:
            assert bases.shape[0] == self.n_units
            assert bases.shape[2] == means.shape[1]
            self.signal_rank = bases.shape[1]
        else:
            self.signal_rank = 0

        self.rg = spawn_torch_rg(seed, device=means.device)

        # needs to be initialized before doing anything serious
        # see bootstrapping stage in main function below
        self.lut = NeighborhoodLUT(
            unit_ids=torch.arange(0),
            neighb_ids=torch.arange(0),
            lut=torch.arange(0)[None],
        )
        self.lut_params = None

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
        distance_kind: Literal["cosine"] = "cosine",
        min_channel_count: int = 1,
        elbo_atol: float = 1e-4,
    ) -> Self:
        """Constructor for the full mixture model, called by from_config()"""
        log_props, noise_log_prop, means, bases = initialize_parameters_by_unit(
            data=data,
            signal_rank=signal_rank,
            noise=noise,
            erp=erp,
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
            seed=seed,
            noise=noise,
            em_iters=em_iters,
            criterion_em_iters=criterion_em_iters,
            prior_pseudocount=prior_pseudocount,
            elbo_atol=elbo_atol,
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
        # assert not refinement_cfg.hard_noise  # TODO: remove all unused
        # assert not refinement_cfg.laplace_ard
        # assert not refinement_cfg.prior_pseudocount
        # assert not refinement_cfg.prior_scales_mean
        # assert not refinement_cfg.noise_fp_correction
        # assert refinement_cfg.distance_metric != "cosinesqrt"
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
        )

    @classmethod
    def initialize_from_dense_data_with_fixed_responsibilities(
        cls,
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
        n_iter: int,
        total_log_proportion: float,
        prior_pseudocount: float,
        elbo_atol: float = 1e-4,
    ) -> tuple[Self, Tensor]:
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

        # not sure whether to do anything with this spikes x unit array
        # using it to constrain candidates would mean that lut would only include
        # spikes whose neighbs are completely covered by chosen chans
        # and, input resp==0 is already used to restrict candidates according
        # to pre-existing units' neighb coverage, since they get -inf liks...
        del chan_coverage

        fdim = noise.rank * noise.n_channels
        means = data.x.new_zeros((K, fdim))
        if signal_rank:
            bases = data.x.new_zeros((K, signal_rank, fdim))
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
        log_proportions = responsibilities.mean(0).log_() + total_log_proportion

        self = cls(
            max_group_size=max_group_size,
            max_distance=max_distance,
            min_count=min_count,
            min_channel_count=min_channel_count,
            split_k=0,
            log_proportions=log_proportions,
            means=means,
            bases=bases,
            noise_log_prop=torch.tensor(-torch.inf),
            neighb_cov=neighb_cov,
            noise=noise,
            erp=erp,
            em_iters=n_iter,
            criterion_em_iters=n_iter,
            elbo_atol=elbo_atol,
            prior_pseudocount=prior_pseudocount,
        )

        # get lut, which will never change
        candidates = torch.where(responsibilities > 0, self.unit_ids[None, :], -1)
        lut = lut_from_candidates_and_neighborhoods(
            candidates,
            neighborhoods=data.neighborhoods,
            neighb_supset=data.neighb_supset,
            n_units=self.n_units,
        )
        self.update_lut(lut)

        # run some em steps
        self.fixed_weight_em(data=data, responsibilities=responsibilities)
        return self, valid

    def em(self, data: TruncatedSpikeData, show_progress: int = 1):
        if show_progress:
            iters = trange(self.em_iters, desc="EM")
        else:
            iters = range(self.em_iters)
        elbos = []
        for j in iters:
            eres = self.e_step(data, show_progress=show_progress > 1)
            elb = eres.stats.elbo.cpu().item()
            elbos.append(elb)
            if show_progress:
                iters.set_description(f"EM(elbo={elb:.3f})")  # type: ignore
            self.m_step(eres.stats)
            self.update_lut(data.update(eres.candidates, self.unit_distance_matrix()))
            if j > 1 and elbos[-1] - elbos[-2] < self.elbo_atol:
                break

        if logger.isEnabledFor(DARTSORTDEBUG):
            elbstr = ", ".join(f"{x:0.3f}" for x in elbos)
            begend = elbos[-1] - elbos[0]
            smalldif = np.diff(elbos).min()
            bigdiff = np.diff(elbos).max()
            logger.info(
                f"EM elbos={elbstr}, with end-start={begend:0.3f} and biggest/smallest "
                f"diffs {bigdiff:0.3f} and {smalldif:0.3f}."
            )
        assert elbos[-1] > elbos[0] - 1e-3

    def e_step(
        self, data: TruncatedSpikeData, show_progress: bool = False
    ) -> TruncatedEStepResult:
        assert self.lut_params is not None
        candidates = torch.empty_like(data.candidates[:, : data.n_candidates])
        stats = SufficientStatistics.zeros(
            n_units=self.n_units,
            n_lut=self.lut_params.n_lut,
            feature_dim=self.neighb_cov.feat_rank * self.neighb_cov.n_channels,
            signal_rank=self.signal_rank,
            device=self.b.means.device,
        )
        for batch in data.batches(show_progress=show_progress, desc="E"):
            batch_scores = self.score_batch(batch, data.n_candidates)
            stats.combine(self.estep_stats_batch(batch, batch_scores))
            candidates[batch.batch] = batch_scores.candidates
        _finalize_e_stats(
            self.b.means,
            self.b.bases,
            stats,
            self.lut,
            self.lut_params,
            self.neighb_cov,
            self.prior_pseudocount,
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
        else:
            assert stats.noise_N is None

        # rank 0 case: mean only
        if self.signal_rank == 0:
            assert stats.R.shape[1] == 1
            self.b.means.copy_(stats.R[:, 0])
            assert self.b.means.isfinite().all()
            return

        # solve mean and basis together
        U = _get_u_from_ulut(self.lut, stats)
        if self.prior_pseudocount:
            denom = stats.N.clamp_(min=torch.finfo(stats.N.dtype).tiny)
            tikh = self.prior_pseudocount / denom
            U.diagonal(dim1=-2, dim2=-1).add_(tikh[:, None])
        soln = torch.linalg.solve(U, stats.R)
        self.b.means.copy_(soln[:, -1])
        self.b.bases.copy_(soln[:, :-1])

        assert soln.isfinite().all()
        if not skip_proportions:
            assert lp.isfinite().all()  # type: ignore

    def fixed_weight_em(self, data: DenseSpikeData, responsibilities: Tensor):
        assert self.lut_params is not None
        batch = data.to_batch(self.unit_ids, self.lut)
        scores = Scores(
            log_liks=responsibilities,  # unused, just to fill the field.
            responsibilities=responsibilities,
            candidates=batch.candidates,
        )
        for _ in range(self.criterion_em_iters):
            stats = self.estep_stats_batch(batch, scores)
            _finalize_e_stats(
                self.b.means,
                self.b.bases,
                stats,
                self.lut,
                self.lut_params,
                self.neighb_cov,
            )
            self.m_step(stats, skip_proportions=True)
            self.update_lut(self.lut)

    def score(self, data: DenseSpikeData) -> Scores:
        batch = data.to_batch(self.unit_ids, self.lut)
        return self.score_batch(
            batch=batch,
            n_candidates=batch.candidates.shape[1],
            skip_responsibility=True,
        )

    def score_batch(
        self,
        batch: SpikeDataBatch,
        n_candidates: int,
        fixed_responsibilities: Tensor | None = None,
        skip_responsibility: bool = False,
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
        )

    def estep_stats_batch(self, batch: SpikeDataBatch, scores: Scores):
        assert self.lut_params is not None
        assert batch.CmoCooinvx is not None
        assert scores.responsibilities is not None
        spike_ixs, candidate_ixs, unit_ixs, neighb_ixs = _sparsify_candidates(
            scores.candidates, batch.neighborhood_ids
        )
        lut_ixs = self.lut.lut[unit_ixs, neighb_ixs]
        return _stat_pass_batch(
            x=batch.x,
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
        )

    def soft_assign(
        self,
        data: BatchedSpikeData,
        needs_bootstrap: bool,
        max_iter: int = 10,
        show_progress: int = 1,
    ) -> Scores:
        """Run E steps until candidates converge, holding my LUT fixed."""
        # start by telling the data how to search
        distances = self.unit_distance_matrix()
        if needs_bootstrap:
            data.erase_candidates()
            data.bootstrap_candidates(distances=distances, un_adj_lut=self.lut)

        # initialize storage for output
        scores = Scores(
            log_liks=distances.new_empty(data.N, data.n_candidates + 1),
            responsibilities=distances.new_empty(data.N, data.n_candidates + 1),
            candidates=torch.empty_like(data.candidates[:, : data.n_candidates]),
        )
        assert scores.responsibilities is not None
        assert max_iter >= 1
        if show_progress:
            iters = trange(max_iter, desc="SoftAssign")
        else:
            iters = range(max_iter)
        for it in iters:
            for batch in data.batches(show_progress=show_progress > 1):
                batch_scores = self.score_batch(batch, data.n_candidates)
                assert batch_scores.responsibilities is not None
                bix = batch.batch
                scores.candidates[bix] = batch_scores.candidates.cpu()
                scores.log_liks[bix] = batch_scores.log_liks.cpu()
                scores.responsibilities[bix] = batch_scores.responsibilities.cpu()

            is_final = it == max_iter - 1
            if not is_final and torch.equal(
                scores.candidates[:, : data.n_candidates],
                data.candidates[:, : data.n_candidates],
            ):
                logger.info(f"Soft assign completely converged at iteration {it}.")
                break
            if not is_final:
                data.update(scores.candidates, distances)

        return scores

    def update_lut(self, lut: NeighborhoodLUT):
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
        split_data = train_data.dense_slice_by_unit(unit_id, min_count=min_count_split)
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
            return None
        assert kmeans_responsibliities.shape[1] >= 2

        # initialize dense model with fixed resps
        split_model, _ = (
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
                n_iter=self.criterion_em_iters,
                elbo_atol=self.elbo_atol,
                prior_pseudocount=self.prior_pseudocount,
                total_log_proportion=self.b.log_proportions[unit_id].item(),
            )
        )
        if split_model.n_units <= 1:
            return None

        # see if that dog hunts
        if eval_data is None:
            split_eval_data = None
            assert scores.log_liks.shape[0] == train_data.N
            cur_scores_batch = scores.slice(split_data.indices)
        else:
            split_eval_data = eval_data.dense_slice_by_unit(unit_id)
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
            skip_single=True,
        )
        if merge_res is not None and merge_res.grouping.n_groups <= 1:
            return None
        if merge_res is not None and merge_res.improvement <= 0.0:
            return None

        if merge_res is None:
            # units are too separated to merge. just assign by scoring.
            train_scores = split_model.score(split_data)
            train_assignments = train_scores.candidates.take_along_dim(
                train_scores.log_liks.argmax(1, keepdim=True), dim=1
            )[:, 0]
            n_groups = split_model.n_units
            means = split_model.b.means
            bases = split_model.b.bases
            sub_proportions = F.softmax(split_model.b.log_proportions, dim=0)
        else:
            train_assignments = merge_res.train_assignments
            n_groups = merge_res.grouping.n_groups
            means = merge_res.means
            bases = merge_res.bases
            sub_proportions = merge_res.sub_proportions

        return SuccessfulUnitSplitResult(
            unit_id=int(unit_id),
            n_split=n_groups,
            train_indices=split_data.indices,
            train_assignments=train_assignments,
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
        any_split = False
        train_mask = torch.zeros_like(train_data.candidates[:, 0], dtype=torch.bool)
        train_labels = torch.full_like(train_data.candidates[:, 0], -1)
        if eval_data is not None:
            eval_mask = torch.zeros_like(eval_data.candidates[:, 0], dtype=torch.bool)
            eval_labels = torch.full_like(eval_data.candidates[:, 0], -1)
        else:
            eval_mask = eval_labels = None

        n_new_units = 0
        new_means = []
        new_log_props = []
        new_bases = []
        cur_max_label = self.n_units - 1
        if show_progress:
            unit_ids = tqdm(self.unit_ids, desc="Split")
        else:
            unit_ids = self.unit_ids
        for unit_id in unit_ids:
            res = self.split_unit(unit_id, train_data, eval_data, scores)
            if res is None:
                continue
            any_split = True
            train_mask[res.train_indices] = True
            for_current = res.train_assignments == 0
            train_labels[res.train_indices[for_current]] = unit_id
            for_other = for_current.logical_not_()
            other_labels = res.train_assignments[for_other].add_(cur_max_label)
            train_labels[res.train_indices[for_other]] = other_labels

            n_new_units += res.n_split - 1
            cur_max_label += res.n_split - 1

            # divvy up my log proportion
            split_log_props = (
                res.sub_proportions.log() + self.b.log_proportions[unit_id]
            )

            # assign split result first params to unit_id's spot
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

        # resize params to allow space for the new guys
        Korig = self.n_units
        Knew = self.n_units + n_new_units
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

        # call data methods for processing splits
        distances = self.unit_distance_matrix()
        lut = train_data.update_from_split(train_mask, train_labels, distances)
        self.update_lut(lut)

        return SplitResult(
            any_split=any_split,
            n_new_units=n_new_units,
            train_split_spike_mask=train_mask,
            train_split_spike_labels=train_labels,
            eval_split_spike_mask=eval_mask,
            eval_split_spike_labels=eval_labels,
        )

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
        if show_progress:
            groups = tqdm(groups, desc="Merge")

        for group in groups:
            if group.numel() == 1:
                continue

            view = self.unit_slice(group)
            group_train_data = train_data.dense_slice_by_unit(group)
            assert group_train_data is not None
            if eval_data is None:
                group_eval_data = None
                assert scores.log_liks.shape[0] == train_data.N
                group_scores = scores.slice(group_train_data.indices)
            else:
                group_eval_data = eval_data.dense_slice_by_unit(group)
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

            # no-merge cases
            if group_res is None:
                continue
            if group_res.improvement <= 0:
                continue
            if group_res.grouping.n_groups == group.numel():
                continue
            any_merged = True

            groups = group_res.grouping.group_ids.unique()
            assert groups.numel() == group_res.grouping.n_groups

            # new_props = props[group].sum() * sub_props
            group_log_prop = self.b.log_proportions[group].logsumexp(dim=0)
            new_log_props = group_res.sub_proportions.log() + group_log_prop

            for gix, g in enumerate(groups):
                (in_group,) = (group_res.grouping.group_ids == g).nonzero(as_tuple=True)
                if in_group.numel() == 1:
                    continue
                ids_in_group = group[in_group]
                first = ids_in_group[0]
                rest = ids_in_group[1:]

                result_map.mapping[ids_in_group] = ids_in_group[0]

                # first is the group, rest are discarded. first retains id for now.
                self.b.log_proportions[rest] = -torch.inf

                # first gets the parameter update
                self.b.means[first] = group_res.means[gix]
                self.b.log_proportions[first] = new_log_props[gix]
                if self.signal_rank:
                    assert group_res.bases is not None
                    self.b.bases[first] = group_res.bases[gix]

        assert any_merged != result_map.is_identity()
        if not any_merged:
            return result_map

        # fix up my parameters, update the datasets, and update the lut
        flat_map = self.cleanup(result_map)
        distances = self.unit_distance_matrix()
        lut = train_data.remap(remapping=flat_map, distances=distances)
        self.update_lut(lut)

        return flat_map

    def get_params_at(self, indices: Tensor):
        m = self.b.means[indices]
        if self.signal_rank:
            b = self.b.bases[indices]
        else:
            b = None
        return m, b

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
        # since we read later indices than are being written, in place is fine
        for old_id, new_id in zip(uniq_first_inds, new_ids):
            assert old_id >= new_id
            new_remapping.mapping[remapping.mapping == old_id] = new_id
            if old_id == new_id:
                continue

            self.b.means[new_id] = self.b.means[old_id]
            self.b.log_proportions[new_id] = self.b.log_proportions[old_id]
            if self.signal_rank:
                self.b.bases[new_id] = self.b.bases[old_id]
            self.b.log_proportions[old_id] = -torch.inf

        assert self.b.log_proportions[new_n_units:].isneginf().all()

        self.n_units = new_n_units
        self.unit_ids = new_ids
        self.b.means.resize_(new_n_units, *self.b.means.shape[1:])
        self.b.log_proportions.resize_(new_n_units, *self.b.log_proportions.shape[1:])
        if self.signal_rank:
            self.b.bases.resize_(new_n_units, *self.b.bases.shape[1:])

        return new_remapping


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
        )
        self.tmm = tmm

    def unit_slice(self, unit_ids: Tensor) -> "TMMView":
        return TMMView(self.tmm, self.unit_ids[unit_ids])

    def get_params_at(self, indices: Tensor):
        m = self.tmm.b.means[self.unit_ids[indices]]
        if self.signal_rank:
            b = self.tmm.b.bases[self.unit_ids[indices]]
        else:
            b = None
        return m, b

    @property
    def centroids(self):
        return self.tmm.b.means[self.unit_ids]

    def non_noise_log_proportion(self):
        return self.tmm.b.log_proportions[self.unit_ids].logsumexp(dim=0)

    def score(self, data: DenseSpikeData) -> Scores:
        batch = data.to_batch(self.unit_ids, self.tmm.lut)
        return self.tmm.score_batch(
            batch=batch,
            n_candidates=batch.candidates.shape[1],
            skip_responsibility=True,
        )


# -- helpers


def get_truncated_datasets(sorting, motion_est, refinement_cfg, device, rg):
    assert sorting.labels is not None
    labels = torch.tensor(sorting.labels, device=device)

    # TODO configure neighb overlap?
    # we assume that the core neighborhoods are exactly the same as extract ones
    assert refinement_cfg.core_radius == "extract"
    # TODO clean up stable data constructor, just keep what's needed
    # don't store full core features...? just extract feats by split.
    vp = refinement_cfg.val_proportion
    stable_data = StableSpikeDataset.from_sorting(
        sorting,
        motion_est=motion_est,
        _core_feature_splits=(),  # turn off feat cache
        core_radius="extract",
        max_n_spikes=refinement_cfg.max_n_spikes,
        split_proportions=(1.0 - vp, vp),
        interpolation_method=refinement_cfg.interpolation_method,
        kernel_name=refinement_cfg.kernel_name,
        sigma=refinement_cfg.interpolation_sigma,
        rq_alpha=refinement_cfg.rq_alpha,
        kriging_poly_degree=refinement_cfg.kriging_poly_degree,
        random_seed=rg,
        device=device,
    )
    noise = EmbeddedNoise.estimate_from_hdf5(
        sorting.parent_h5_path,
        motion_est=motion_est,
        zero_radius=refinement_cfg.cov_radius,
        cov_kind=refinement_cfg.cov_kind,
        glasso_alpha=refinement_cfg.glasso_alpha,
        interpolation_method=refinement_cfg.interpolation_method,
        kernel_name=refinement_cfg.kernel_name,
        sigma=refinement_cfg.interpolation_sigma,
        rq_alpha=refinement_cfg.rq_alpha,
        kriging_poly_degree=refinement_cfg.kriging_poly_degree,
        device=device,
        rgeom=stable_data.prgeom[:-1],  # type: ignore
    )
    neighb_cov = NeighborhoodCovariance.from_noise_and_neighborhoods(
        prgeom=stable_data.prgeom,  # type: ignore
        noise=noise,
        neighborhoods=stable_data._train_extract_neighborhoods,
    )
    n_candidates, n_search, n_explore = _pick_search_size(sorting, refinement_cfg)
    assert "train" in stable_data.split_indices
    xtrain = stable_data._train_extract_features.to(device)
    xtrain = xtrain.view(len(xtrain), -1).nan_to_num_()
    train_data = TruncatedSpikeData.initialize_from_labels(
        n_candidates=n_candidates,
        n_search=n_search,
        n_explore=n_explore,
        dense_slice_size_per_unit=refinement_cfg.n_spikes_fit,
        labels=labels[stable_data.split_indices["train"]],
        x=xtrain,
        neighborhoods=stable_data._train_extract_neighborhoods.to(device),
        neighb_cov=neighb_cov,
        seed=rg,
    )
    val_n_candidates = max(refinement_cfg.merge_group_size, n_candidates)
    if refinement_cfg.criterion.startswith("heldout"):
        assert "val" in stable_data.split_indices
        assert stable_data.core_features is not None
        split = stable_data.split_indices["val"]
        xval = stable_data.core_features[split].to(device)
        xval = xval.view(len(xval), -1).nan_to_num_()
        val_data = TruncatedSpikeData.initialize_from_labels(
            n_candidates=val_n_candidates,
            n_search=n_search,
            n_explore=n_explore,
            dense_slice_size_per_unit=refinement_cfg.n_spikes_fit,
            labels=labels[split],
            explore_neighb_steps=0,
            x=xval,
            neighborhoods=stable_data._core_neighborhoods["key_val"].to(device),  # type: ignore
            neighb_cov=neighb_cov,
            seed=rg,
            batch_size=refinement_cfg.eval_batch_size,
        )
        assert torch.equal(
            val_data.neighborhoods.neighborhoods,  # type: ignore
            train_data.neighborhoods.neighborhoods,  # type: ignore
        )
    else:
        val_data = None

    # TODO skip this if not needed
    assert stable_data.core_features is not None
    assert "key_full" in stable_data._core_neighborhoods
    xfull = stable_data.core_features.view(len(stable_data.core_features), -1)
    xfull = xfull.nan_to_num_()
    full_data = OnlineSpikeData(
        n_candidates=val_n_candidates,
        n_search=n_search,
        n_explore=n_explore,
        explore_neighb_steps=0,
        x=xfull,
        neighborhoods=stable_data._core_neighborhoods["key_full"],  # type: ignore
        device=device,
        batch_size=refinement_cfg.eval_batch_size,
        neighb_cov=neighb_cov,
        seed=rg,
    )
    assert torch.equal(
        full_data.neighborhoods.neighborhoods,  # type: ignore
        train_data.neighborhoods.neighborhoods,  # type: ignore
    )

    erp = NeighborhoodInterpolator(
        prgeom=stable_data.prgeom,  # type: ignore
        neighborhoods=stable_data._train_extract_neighborhoods,
        method=refinement_cfg.interpolation_method,
        kernel_name=refinement_cfg.kernel_name,
        sigma=refinement_cfg.interpolation_sigma,
        rq_alpha=refinement_cfg.rq_alpha,
        kriging_poly_degree=refinement_cfg.kriging_poly_degree,
    )

    return neighb_cov, erp, train_data, val_data, full_data, noise


def initialize_parameters_by_unit(
    data: TruncatedSpikeData,
    signal_rank: int,
    noise: EmbeddedNoise,
    erp: NeighborhoodInterpolator,
    prior_pseudocount: float,
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

    log_proportions = torch.zeros((puff_K + 1,), device=counts.device)
    log_proportions = log_proportions.resize_(K + 1)
    log_proportions[:K] = counts.float()
    log_proportions[-1] = log_proportions[:K].mean()
    log_proportions = F.log_softmax(log_proportions.log_(), dim=0)
    noise_log_prop = log_proportions[-1]
    log_proportions = log_proportions[:K]

    dev = data.noise_logliks.device
    means = torch.zeros((puff_K, feat_rank, nc), device=dev)
    means = means.resize_(K, *means.shape[1:])
    if signal_rank:
        bases = torch.zeros((puff_K, signal_rank, feat_rank, nc), device=dev)
        bases = bases.resize_(K, *bases.shape[1:])
    else:
        bases = None
    for k in range(K):
        kdata = data.dense_slice_by_unit(k)
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
    assert torch.equal(order, torch.arange(len(order), device=unit_ids.device))
    ix_in_unit_ids = torch.searchsorted(unit_ids, uu)
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
) -> Iterable[Tensor]:
    device = distances.device
    k = distances.shape[0]
    if k < max_group_size and distances.max() < max_distance:
        # this would be common when splitting.
        return (torch.arange(k, device=device),)

    # sometimes numerically negative
    distances.diagonal(dim1=-2, dim2=-1).fill_(0.0)

    # make symmetric
    distances = torch.minimum(distances, distances.T)

    # only need upper tri
    distances = distances[*torch.triu_indices(*distances.shape, offset=1)]

    # make finite
    isfinite = distances.isfinite()
    if not isfinite.all():
        big = distances[isfinite].amax() + max_distance + 16.0
        distances.masked_fill_(isfinite.logical_not_(), big)

    max_group_size = min(k, max_group_size)
    Z = linkage(distances.numpy(force=True), method=link)
    # get tree out to max distance
    groups = maximal_leaf_groups(Z, max_distance=max_distance)
    groups = [torch.tensor(g) for g in groups]
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
) -> GroupMergeResult:
    if pair_mask is not None and pair_mask.sum() == pair_mask.shape[0]:
        return None

    # score train data with mm to get the full model log liks and the
    # responsibilities that will be used for fitting below
    full_scores = mm.score(train_data)
    if responsibilities is None:
        responsibilities = F.softmax(full_scores.log_liks, dim=1)

    # include the full partition to keep code simple
    partitions, subset_to_id, id_to_subset = allowed_partitions(
        mm.unit_ids, pair_mask, skip_full=skip_full, skip_single=skip_single
    )
    n_big_subsets = len(subset_to_id)
    if not n_big_subsets:
        return None

    # fit subset models with fixed responsibilities
    subset_resps = responsibilities.new_empty(
        (responsibilities.shape[0], n_big_subsets)
    )
    for s in range(n_big_subsets):
        subset_resps[:, s] = responsibilities[:, id_to_subset[s]].sum(dim=1)
    subset_models, valid_subsets = (
        TruncatedMixtureModel.initialize_from_dense_data_with_fixed_responsibilities(
            data=train_data,
            responsibilities=subset_resps,
            signal_rank=mm.signal_rank,
            erp=mm.erp,
            min_count=mm.min_channel_count,  # this isn't used from here, shimming
            min_channel_count=mm.min_channel_count,
            noise=mm.noise,
            max_group_size=mm.max_group_size,
            max_distance=mm.max_distance,
            neighb_cov=mm.neighb_cov,
            n_iter=mm.criterion_em_iters,
            prior_pseudocount=mm.prior_pseudocount,
            total_log_proportion=mm.non_noise_log_proportion(),
        )
    )
    assert valid_subsets.all()

    # score the eval or train set with the all-subsets model
    train_subset_scores = subset_models.score(train_data)
    if eval_data is not None:
        crit_subset_scores = subset_models.score(eval_data)
        crit_full_scores = mm.score(eval_data)
    else:
        crit_subset_scores = train_subset_scores
        crit_full_scores = full_scores

    # get scores for unaffected units
    cur_logliks = cur_scores.log_liks
    assert cur_logliks.shape[1] == cur_scores.candidates.shape[1] + 1
    assert cur_logliks.shape[0] == crit_full_scores.log_liks.shape[0]
    cur_mask = torch.isin(cur_scores.candidates, cur_unit_ids)
    rest_logliks = cur_logliks.clone()
    rest_logliks[:, :-1].masked_fill_(cur_mask, -torch.inf)

    # get current model criterion
    # TODO did those weights do anything?
    # cur_crit = ecl(cur_scores.responsibilities, cur_scores.log_liks)
    cur_crit = cur_scores.log_liks.logsumexp(dim=1).mean()
    # cur_crit = elbo(cur_scores.responsibilities, cur_scores.log_liks)

    # now, find the best subset. combine subset scores with remainder scores.
    # also need to adjust the log proportions here.
    k0 = cur_logliks.shape[1]
    # crit_full_logliks = get_log_liks_matrix_from_scores(mm.unit_ids, crit_full_scores)
    # crit_subset_logliks = get_log_liks_matrix_from_scores(
    #     subset_models.unit_ids, crit_subset_scores
    # )
    crit_full_logliks = crit_full_scores.log_liks[:, :-1]
    crit_subset_logliks = crit_subset_scores.log_liks[:, :-1]
    kfull = crit_full_logliks.shape[1]
    assert kfull == mm.n_units
    assert crit_subset_logliks.shape[1] == subset_models.n_units
    part_logliks_holder = cur_logliks.new_empty((rest_logliks.shape[0], k0 + kfull))
    part_logliks_holder[:, :k0] = rest_logliks
    best_part = partitions[0]
    best_score = torch.tensor(-torch.inf)
    for part in partitions:
        ksingle = len(part.single_ixs)
        k1 = k0 + ksingle
        ksubset = len(part.subset_ids)
        k2 = k1 + ksubset
        part_logliks_holder[:, k0:k1] = crit_full_logliks[:, part.single_ixs]
        part_logliks_holder[:, k1:k2] = crit_subset_logliks[:, part.subset_ids]

        part_logliks = part_logliks_holder[:, :k2]
        # part_resps = F.softmax(part_logliks, dim=1)

        # part_score = ecl(part_resps, part_logliks)
        part_score = part_logliks.logsumexp(dim=1).mean()
        # part_score = elbo(part_resps, part_logliks)
        if part_score >= best_score:
            best_score = part_score
            best_part = part

    # spike assignments
    train_assignments = get_part_assignments(
        best_part, full_scores.log_liks, train_subset_scores.log_liks
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


def _pick_search_size(sorting, refinement_cfg):
    """Get sensible candidate, search, explore set sizes."""
    K = sorting.n_units
    C = min(K, refinement_cfg.n_candidates)
    S = min(C, refinement_cfg.n_search or C)
    E = S
    remainder = K - C
    S = max(0, min(remainder // C, S))
    remainder -= C * S
    E = max(0, min(remainder, E))
    return C, S, E


def _neighborhood_indices(
    neighborhoods: SpikeNeighborhoods, zero_radius: float | None, prgeom: Tensor
):
    nc_obs = neighborhoods.b.channel_counts.long()
    obs_ix = neighborhoods.b.neighborhoods
    truncate = zero_radius is not None and zero_radius < float("inf")
    nneighb = neighborhoods.n_neighborhoods
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


def _noise_factors(noise, obs_ix, miss_near_ix, cache_prefix):
    nneighbs = len(obs_ix)
    nc_obs = obs_ix.shape[1]
    nc_miss_near = miss_near_ix.shape[1]
    rank = noise.rank
    nc = noise.n_channels
    dev = obs_ix.device

    logdet = torch.zeros((nneighbs,), device=dev)
    Cooinv = torch.zeros((nneighbs, rank, nc_obs, rank, nc_obs), device=dev)
    CooinvCom = torch.zeros((nneighbs, rank, nc_obs, rank, nc_miss_near), device=dev)
    Linv = Cooinv.clone()

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
        jCom = jCom.to_dense()

        jL = jCoo.cholesky()
        jLinv = jL.inverse().to_dense()
        jCooinv = jLinv.T @ jLinv
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
    Cooinv = Cooinv.view(nneighbs, obsdim, obsdim)
    CooinvCom = CooinvCom.view(nneighbs, obsdim, missdim)
    Linv = Linv.view(nneighbs, obsdim, obsdim)
    return logdet, Cooinv, CooinvCom, Linv


def _whiten_impute_and_noise_score(
    x: Tensor,
    neighborhoods: SpikeNeighborhoods,
    neighb_cov: NeighborhoodCovariance,
    batch_size=1024,
):
    wx = torch.empty_like(x)

    noise_loglik = wx.new_zeros((len(x),))
    missdim = neighb_cov.feat_rank * neighb_cov.max_nc_miss_near
    CmoCooinvx = wx.new_empty((len(x), missdim))

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
            CmoCooinvx[binni] = xb @ neighb_CooinvCom

            nll = wxb.square_().sum(dim=1)
            nll += nll_const
            nll *= -0.5
            assert nll.isfinite().all()
            noise_loglik[binni] = nll

    return wx, CmoCooinvx, noise_loglik


def _whiten_and_noise_score_batch(
    x: Tensor, neighb_ids: Tensor, neighb_cov: NeighborhoodCovariance
):
    batch_whitener = neighb_cov.Linv[neighb_ids]
    wx = torch.bmm(batch_whitener, x[:, :, None])[:, :, 0]
    nll = torch.linalg.vector_norm(wx, dim=1).square_()
    nll.add_(neighb_cov.logdet[neighb_ids]).add_(LOG_2PI * neighb_cov.nobs[neighb_ids])
    nll.mul_(-0.5)
    return wx, nll


def _initialize_single(
    x, chans, noise, rank, prior_pseudocount, weight=None, eps=1e-5, in_place=True
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
        mean = x.mean(dim=0)
        wsum = torch.ones(())
    else:
        wsum = weight.sum()
        nw = weight / (wsum + prior_pseudocount)
        mean = (nw[:, None] * x).sum(0)

    if not rank:
        mean = mean.view(d, c)
        return mean, None

    q = min(n, d * c, 10 + rank)

    # we want x(C^-0.5)=xU^-1 -- need to use upper factor.
    noise_cov = noise.marginal_covariance(chans)
    U = torch.linalg.cholesky(noise_cov, upper=True).to_dense()
    if in_place:
        x -= mean
    else:
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
    labels: Tensor,
    neighborhoods: SpikeNeighborhoods,
    neighb_supset: Tensor,
    neighb_adj: Tensor,
    explore_steps: int,
    n_units: int,
    un_adj_lut: NeighborhoodLUT | None,
):
    if un_adj_lut is None:
        un_adj_lut = lut_from_candidates_and_neighborhoods(
            candidates=labels,
            neighborhoods=neighborhoods,
            n_units=n_units,
            neighb_supset=neighb_supset,
        )

    # lut -> adjacency matrix
    un_adj = torch.zeros(un_adj_lut.lut.shape, device=labels.device)
    un_adj = torch.lt(un_adj_lut.lut, un_adj_lut.unit_ids.shape[0], out=un_adj)

    # constraint for explore candidates is looser
    explore_adj = un_adj
    for _ in range(explore_steps):
        explore_adj = un_adj @ neighb_adj

    return un_adj_lut, un_adj, explore_adj


def lut_from_candidates_and_neighborhoods(
    candidates: Tensor,
    neighborhoods: SpikeNeighborhoods,
    neighb_supset: Tensor,
    n_units: int,
    add_extra_unit=False,
) -> NeighborhoodLUT:
    co = coincidence_matrix(
        x=candidates,
        y=neighborhoods.b.neighborhood_ids,
        nx=n_units + add_extra_unit,
        ny=neighborhoods.n_neighborhoods,
    )
    co = co.float() @ neighb_supset
    uu, nn = co.nonzero(as_tuple=True)
    n_lut = uu.shape[0]
    lut = co.new_full(co.shape, n_lut, dtype=torch.long)
    lut[uu, nn] = torch.arange(n_lut, device=co.device)
    return NeighborhoodLUT(unit_ids=uu, neighb_ids=nn, lut=lut)


def candidate_search_sets(
    distances: Tensor, un_adj_lut: NeighborhoodLUT, un_adj: Tensor, n_search: int
):
    n_lut = un_adj_lut.unit_ids.shape[0]

    # get lut-indexed version
    s = distances[un_adj_lut.unit_ids]

    # don't want to match with myself
    inf = s.new_full((1, 1), torch.inf).broadcast_to((n_lut, 1))
    s.scatter_(dim=1, index=un_adj_lut.unit_ids[:, None], src=inf)

    # flip scale so larger is better
    s.reciprocal_()

    # multiply by neighborhood-unit adjacency to set non olap to 0
    s.mul_(un_adj.T[un_adj_lut.neighb_ids, : distances.shape[0]])

    # take topk, and fill invalids (s=0) with -1
    # pad with row of -1s for the invalid lut ixs (===n_lut)
    tops = s.new_empty((n_lut, n_search))
    topunits = s.new_full((n_lut + 1, n_search), -1, dtype=torch.long)
    torch.topk(s, k=n_search, out=(tops, topunits[:-1]))
    topunits[:-1].masked_fill_(tops == 0, -1)

    return topunits


def _fill_blank_labels(
    labels: Tensor,
    un_adj: Tensor,
    neighborhood_ids: Tensor,
    gen: torch.Generator,
    batch_size: int = 1024,
):
    (blank,) = (labels < 0).nonzero(as_tuple=True)
    Nblank = blank.numel()
    if not Nblank:
        return

    # let's check the coverage here. we rely on this assumption
    # TODO: if this fails, can use explore_adj to backfill with eps?
    # or even take more steps as necessary?
    assert un_adj.sum(0).min().cpu().item() > 0
    for i0 in range(0, Nblank, batch_size):
        i1 = min(Nblank, i0 + batch_size)
        ii = blank[i0:i1]
        nn = neighborhood_ids[ii]
        p = un_adj[:, nn].mT
        draws = torch.multinomial(p, 1, replacement=True, generator=gen)
        labels[ii] = draws.view(ii.shape[0])


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
    un_adj_lut: NeighborhoodLUT, explore_adj: Tensor, search_sets: Tensor
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
    p[ii[:, None], search_sets[jj]] = 0.0

    # count them in each bin and get the max (max_explore)
    explore_counts = (p > 0).sum(dim=1)
    max_explore: int = explore_counts.max().cpu().item()  # type: ignore

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
):
    # sample multinomial without replacement for each spike
    choices = torch.multinomial(p[lut_ixs], n_explore, generator=gen)

    # replace those values with their entries in inds
    candidates[:, -n_explore:] = inds[lut_ixs[:, None], choices]
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
    muo = muo.view(n, lut_params.muo.shape[1])

    # Linvmuo, CmoCooinvmuo
    lut_params.Linvmuo[i0:i1, None] = muo[:, None].bmm(neighb_cov.Linv[nn].mT)
    lut_params.CmoCooinvmuo[i0:i1, None] = muo[:, None].bmm(neighb_cov.CooinvCom[nn])

    # constplogdet
    lut_params.constplogdet[i0:i1] = neighb_cov.nobs[nn].float().mul_(LOG_2PI)
    lut_params.constplogdet[i0:i1] += neighb_cov.logdet[nn]


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
    bpad = bases.new_zeros((n, M, neighb_cov.feat_rank, neighb_cov.n_channels + 1))
    torch.index_select(
        bases.view(bases.shape[0], M, neighb_cov.feat_rank, -1),
        dim=0,
        index=uu,
        out=bpad[:, :, :, :-1],
    )
    obs_ix = neighb_cov.obs_ix[nn][:, None, None, :]
    Wo = bpad.take_along_dim(indices=obs_ix, dim=3)
    Wo = Wo.view(n, M, neighb_cov.feat_rank * neighb_cov.max_nc_obs)

    # next, the capaciatance is I + Wo Cooinv Wo'. always pos def.
    Cooinv = neighb_cov.Cooinv[nn]
    WoCooinv = Wo.bmm(Cooinv)
    cap = WoCooinv.bmm(Wo.mT)
    cap.diagonal(dim1=-2, dim2=-1).add_(1.0)

    # capacitance cholesky, inverse cholesky, inverse
    L, _ = torch.linalg.cholesky_ex(cap)  # cap = LL'
    Linv = torch.zeros_like(L)  # initialize to identity
    Linv.diagonal(dim1=-2, dim2=-1).add_(1.0)
    # solves LX=I => X=Linv. Then T := inv(cap) = inv(LL') = Linv'Linv.
    Linv = torch.linalg.solve_triangular(L, Linv, upper=False, out=Linv)
    T = Linv.mT.bmm(Linv)

    # Tpad, logdet
    lut_params.Tpad[i0:i1, :, :-1] = T  # type: ignore[reportOptionalSubscript]
    cap_logdet = L.diagonal(dim1=-2, dim2=-1).log().sum(dim=1).mul_(2.0)
    lut_params.constplogdet[i0:i1] += cap_logdet

    # precomputed products with T
    TWoCooinv = torch.bmm(T, WoCooinv, out=lut_params.TWoCooinv[i0:i1])  # type: ignore[reportOptionalSubscript]
    muo = lut_params.muo[i0:i1, :, None]
    torch.bmm(TWoCooinv, muo, out=lut_params.TWoCooinvmuo[i0:i1, :, None])  # type: ignore[reportOptionalSubscript]

    # Woodbury root
    Cooinvsqrt = torch.index_select(neighb_cov.Linv, dim=0, index=nn, out=Cooinv)
    rt = torch.bmm(Linv, Wo, out=WoCooinv)
    torch.bmm(Cooinvsqrt, rt.mT, out=lut_params.wburyroot[i0:i1])  # type: ignore[reportOptionalSubscript]


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
    fixed_responsibilities: Tensor | None = None,
    skip_responsibility: bool = False,
):
    n, Ctot = candidates.shape
    assert whitenedx.shape[0] == n
    assert noise_logliks.shape == (n,)
    not_fixed = fixed_responsibilities is None
    do_resp = not_fixed and not skip_responsibility

    if spike_ixs is None:
        spike_ixs, candidate_ixs, unit_ixs, neighb_ixs = _sparsify_candidates(
            candidates, neighborhood_ids
        )
        lut_ixs = lut.lut[unit_ixs, neighb_ixs]
        if pnoid:
            assert (lut_ixs < lut.unit_ids.shape[0]).all()
    else:
        assert candidate_ixs is not None
        assert neighb_ixs is not None
        assert lut_ixs is not None

    lls = whitenedx.new_full((n, Ctot + not_fixed), fill_value=-torch.inf)
    if not_fixed:
        lls[:, -1] = noise_logliks
        lls[:, -1].add_(noise_log_prop)
    lls[spike_ixs, candidate_ixs] = _calc_loglik(
        whitenedx, log_proportions, spike_ixs, lut_ixs, unit_ixs, lut_params
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
    candidates, neighborhood_ids
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    cpos = candidates >= 0
    if pnoid:
        assert cpos.any(dim=1).all()
    spike_ixs, candidate_ixs = cpos.nonzero(as_tuple=True)
    neighb_ixs = neighborhood_ids[spike_ixs]
    unit_ixs = candidates[spike_ixs, candidate_ixs]
    return spike_ixs, candidate_ixs, unit_ixs, neighb_ixs


def _calc_loglik(whitenedx, log_proportions, spike_ixs, lut_ixs, unit_ixs, lut_params):
    wburyroot = lut_params.wburyroot[lut_ixs] if lut_params.signal_rank else None
    ll = woodbury_inv_quad(
        whitenedx[spike_ixs], lut_params.Linvmuo[lut_ixs], wburyroot, overwrite_wnu=True
    )
    ll += lut_params.constplogdet[lut_ixs]
    ll *= -0.5
    ll += log_proportions[unit_ixs]
    return ll


def _stat_pass_batch(
    x: Tensor,
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
) -> SufficientStatistics:
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
    n = responsibilities.shape[0]
    assert len(x) == n
    assert responsibilities.shape[1] in (n_candidates, n_candidates + 1)
    dense_case = responsibilities.shape[1] == n_candidates

    noise_N, N, Nlut, Qnflat, Qnflatlut = _count_batch(
        responsibilities=responsibilities,
        n_candidates=n_candidates,
        spike_ixs=spike_ixs,
        candidate_ixs=candidate_ixs,
        unit_ixs=unit_ixs,
        lut_ixs=lut_ixs,
        n_units=n_units,
        n_lut=lut_params.n_lut,
    )

    # construct U-related stuff
    xsp = x[spike_ixs]
    if lut_params.signal_rank:
        TWoCooinv = lut_params.TWoCooinv[lut_ixs]  # type: ignore
        TWoCooinvmuo = lut_params.TWoCooinvmuo[lut_ixs]  # type: ignore
        ubar = TWoCooinvmuo[:, :, None].baddbmm_(TWoCooinv, xsp[:, :, None], beta=-1.0)
        ubar = ubar[:, :, 0]
        del TWoCooinvmuo
        hatubar = torch.concatenate((ubar, torch.ones_like(ubar[:, :1])), dim=1)
        hatU = lut_params.Tpad[lut_ixs]  # type: ignore

        # not saving the bottom row, Tpad should just be 0padded on inner dim
        hatU.addcmul_(ubar[:, :, None], hatubar[:, None, :])

        # U sufficient stat (LUT binned)
        Ulut = torch.zeros_like(lut_params.Tpad)  # type: ignore
        hatU *= Qnflatlut[:, None, None]
        ix = lut_ixs[:, None, None].broadcast_to(hatU.shape)
        Ulut.scatter_add_(dim=0, index=ix, src=hatU)
    else:
        Ulut = None

    # construct R-related stuff
    # NB we're saving flops here by skipping the second term and coming
    # back to it later -- it doesn't depend on data so can be had for cheap
    # xc = lut_params.muo[lut_ixs]
    # xc = torch.subtract(xsp, xc, out=xc)
    xc = xsp
    # CmoCooinvxc = lut_params.CmoCooinvmuo[lut_ixs]
    # CmoCooinvxc = torch.subtract(CmoCooinvx[spike_ixs], CmoCooinvxc, out=CmoCooinvxc)
    # TODO what's up with this, and what's up with the ubar CC v0 term. is that already in
    # the second term and i only have it by mistake? and i'm not computing it rn right?
    # maybe it's included, right? and i dont center here, because that's done by lut later
    # inside the what stuff?
    # after including this it doesnt nan out anymore... but still delbo -0.01 happens
    CmoCooinvxc = CmoCooinvx[spike_ixs]
    if lut_params.signal_rank:
        Ro = hatubar[:, :, None] * xc[:, None, :]  # type: ignore[reportPossiblyUnboundVariable]
        Rm = hatubar[:, :, None] * CmoCooinvxc[:, None, :]  # type: ignore[reportPossiblyUnboundVariable]
    else:
        Ro = xc[:, None, :]
        Rm = CmoCooinvxc[:, None, :]

    # gather Ro, Rm onto full channel set
    Rf = x.new_zeros((*Ro.shape[:2], neighb_cov.feat_rank, neighb_cov.n_channels + 1))
    Ro = Ro.view(*Ro.shape[:2], neighb_cov.feat_rank, -1)
    Rm = Rm.view(*Rm.shape[:2], neighb_cov.feat_rank, -1)
    ix = neighb_cov.obs_ix[neighb_ixs][:, None, None, :]
    Rf.scatter_(dim=3, index=ix.broadcast_to(Ro.shape), src=Ro)
    ix = neighb_cov.miss_near_ix[neighb_ixs][:, None, None, :]
    Rf.scatter_(dim=3, index=ix.broadcast_to(Rm.shape), src=Rm)

    # sufficient stats for R
    full_dim = neighb_cov.feat_rank * neighb_cov.n_channels
    R = x.new_zeros(
        (
            n_units,
            lut_params.signal_rank + 1,
            neighb_cov.feat_rank,
            neighb_cov.n_channels,
        )
    )
    Rf = Rf[:, :, :, :-1].mul_(Qnflat[:, None, None, None])
    ix = unit_ixs[:, None, None, None].broadcast_to(Rf.shape)
    R.scatter_add_(dim=0, index=ix, src=Rf)
    R = R.view(*R.shape[:2], full_dim)

    # objective
    if dense_case:
        elb = torch.zeros(())
    else:
        elb = elbo(responsibilities, log_liks)

    return SufficientStatistics(
        count=n, noise_N=noise_N, N=N, Nlut=Nlut, Ulut=Ulut, R=R, elbo=elb
    )


def _count_batch(
    responsibilities,
    n_candidates,
    spike_ixs,
    candidate_ixs,
    unit_ixs,
    lut_ixs,
    n_units,
    n_lut,
) -> tuple[Tensor | None, Tensor, Tensor, Tensor, Tensor]:
    have_noise = responsibilities.shape[1] == n_candidates + 1
    if have_noise:
        noise_N = responsibilities[:, -1].sum()
    else:
        noise_N = None

    Qflat = responsibilities[spike_ixs, candidate_ixs]

    N = Qflat.new_zeros(n_units)
    N.scatter_add_(dim=0, index=unit_ixs, src=Qflat)
    Nlut = Qflat.new_zeros(n_lut)
    Nlut.scatter_add_(dim=0, index=lut_ixs, src=Qflat)

    eps = torch.finfo(N.dtype).tiny

    Qnflat = N[unit_ixs].clamp_(min=eps)
    Qnflat = torch.divide(Qflat, Qnflat, out=Qnflat)

    Qnflatlut = Nlut[lut_ixs].clamp_(min=eps)
    Qnflatlut = torch.divide(Qflat, Qnflatlut, out=Qnflatlut)

    return noise_N, N, Nlut, Qnflat, Qnflatlut


def _finalize_e_stats(
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
    hat_dim = lut_params.signal_rank + 1
    frank = neighb_cov.feat_rank
    nc = neighb_cov.n_channels
    ncm = neighb_cov.max_nc_miss_near
    What_batch = means.new_zeros((batch_size, hat_dim, frank, nc + 1))
    Uhat_batch = means.new_ones((batch_size, hat_dim, hat_dim))

    # reweighting
    denom = stats.N[lut.unit_ids].clamp_(min=torch.finfo(stats.N.dtype).tiny)
    Nlut_N = torch.divide(stats.Nlut, denom, out=denom)
    assert torch.isfinite(Nlut_N).all()

    # loop because w_wcc is big
    for i0 in range(0, lut_params.n_lut, batch_size):
        # batch indices
        i1 = min(lut_params.n_lut, i0 + batch_size)
        uu = lut.unit_ids[i0:i1]
        nn = lut.neighb_ids[i0:i1]
        What = What_batch[: i1 - i0]
        Uhat = Uhat_batch[: i1 - i0]

        # fill in the blanks if nec
        What[:, -1:, :, :-1] = means[uu, None].view(i1 - i0, 1, frank, nc)
        if lut_params.signal_rank:
            assert stats.Ulut is not None
            assert bases is not None
            Uhat[:, :-1, :] = stats.Ulut[i0:i1, :, :]
            Uhat[:, -1:, :-1] = stats.Ulut[i0:i1, :, -1:].mT
            What[:, :-1, :, :-1] = bases[uu].view(i1 - i0, hat_dim - 1, frank, nc)

        What = What.view(i1 - i0, hat_dim, frank, nc + 1)
        obs_ix = neighb_cov.obs_ix[nn][:, None, None, :]
        Whatobs = What.take_along_dim(dim=3, indices=obs_ix)
        Whatobs = Whatobs.view(i1 - i0, hat_dim, -1)
        WoTCmoCooinv = Whatobs.bmm(neighb_cov.CooinvCom[nn])
        WoTCmoCooinv = WoTCmoCooinv.view(i1 - i0, hat_dim, frank, ncm)

        w_wcc = What  # first make Wmiss in place
        ix = neighb_cov.miss_near_ix[nn, None, None].broadcast_to(WoTCmoCooinv.shape)
        w_wcc.scatter_add_(dim=3, index=ix, src=WoTCmoCooinv._neg_view())
        w_wcc = w_wcc[..., :-1]
        w_wcc *= neighb_cov.miss_full_mask[nn][:, None, None, :]
        w_wcc = w_wcc.reshape(i1 - i0, hat_dim, -1)

        # apply reweighting
        Uhat_w_w_cc = Uhat.bmm(w_wcc)
        Uhat_w_w_cc *= Nlut_N[i0:i1, None, None]

        # add to corresponding positions in R
        stats.R.scatter_add_(dim=0, index=uu[:, None, None], src=Uhat_w_w_cc)


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


def woodbury_inv_quad(whitenedx, whitenednu, wburyroot=None, overwrite_wnu=False):
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
    out = whitenednu if overwrite_wnu else None
    wdxz = torch.subtract(whitenedx, whitenednu, out=out)
    if wburyroot is None:
        return wdxz.square_().sum(dim=1)

    # term_b = torch.einsum("nj,njp->np", wdxz, wburyroot)
    term_b = wdxz[:, None].bmm(wburyroot)[:, 0]
    term_a = wdxz.square_().sum(dim=1)
    term_b = term_b.square_().sum(dim=1)
    return term_a.sub_(term_b)


def ecl(resps: Tensor, log_liks: Tensor, cl_alpha=1.0):
    h = entropy(resps, dim=1, reduce_mean=True)
    crit = log_liks.logsumexp(dim=1).mean() - cl_alpha * h
    return crit
