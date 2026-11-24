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
"""

import gc
import math
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    Self,
    cast,
    get_args,
)

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import connected_components
from scipy.special import gammainc, gammaincc, gammaln
from scipy.stats import chi2
from sympy.utilities.iterables import multiset_partitions, subsets
from torch import Tensor

from ..util.data_util import DARTsortSorting, pos_int_unique_and_counts
from ..util.internal_config import (
    ClusteringFeaturesConfig,
    ComponentDistanceMetric,
    ComputationConfig,
    DARTsortInternalConfig,
    KmeansppConfig,
    RefinementConfig,
    SplitKmeansppMetric,
)
from ..util.interpolation_util import (
    NeighborhoodFiller,
    NeighborhoodImputer,
    NeighborhoodInterpolator,
    SpikeNeighborhoods,
    pad_geom,
)
from ..util.job_util import ensure_computation_config
from ..util.logging_util import (
    DARTSORTDEBUG,
    DARTSORTVERBOSE,
    get_logger,
    progbar,
    progrange,
)
from ..util.main_util import ds_save_intermediate_labels, ds_save_intermediate_sorting
from ..util.motion import MotionInfo
from ..util.noise_util import KroneckerNoise
from ..util.py_util import databag, panic
from ..util.spiketorch import (
    _nonzero_static,
    cosine_cdist_from_dots,
    cosine_distance,
    csr_members_ordered,
    ecl,
    elbo,
    entropy,
    normeuc_distance,
    scaled_normeuc_cdist_from_dots,
    scaled_normeuc_distance,
    sign,
    sort_to_csr,
    spawn_torch_rg,
    stratified_subsample,
)
from ..util.torch_util import BModule, cleanup_and_log_gpu_usage, torch_compiler
from .cluster_util import decrumb, linkage, maximal_leaf_groups, violation_statistics
from .clustering_features import StableWaveformFeatures, _read_ix_maybe_mmap
from .kmeans import _one_gumbel_nolog, batched_kmeans

if TYPE_CHECKING:
    from ..transform.temporal_pca import BaseTemporalPCA

logger = get_logger(__name__)
pnoid = logger.isEnabledFor(DARTSORTVERBOSE)
prop_check_atol = 1e-3
prop_check_rtol = 1e-5
tiny = torch.finfo(torch.float32).tiny


# -- main


def tmm_demix(
    *,
    sorting: DARTsortSorting,
    motion: MotionInfo,
    tpca: "BaseTemporalPCA | None" = None,
    refinement_cfg: RefinementConfig,
    stable_features: StableWaveformFeatures | None = None,
    clustering_features_cfg: ClusteringFeaturesConfig | None = None,
    computation_cfg: ComputationConfig | None,
    save_step_labels_format: str | None = None,
    save_step_labels_dir: Path | None = None,
    save_cfg: DARTsortInternalConfig | None = None,
    seed: int | np.random.Generator = 0,
    skip_final_assign_and_return_mix_data: bool = False,
):
    """GMM-based spike clustering using truncated expectation maximization

    Infers #units using a cross-validation criterion evaluated over proposed
    splits and merges.
    """
    prog_level = 1 + logger.isEnabledFor(DARTSORTVERBOSE)

    saving = save_step_labels_dir is not None and (
        save_cfg is None or save_cfg.save_intermediate_labels
    )
    _init_save_kw = dict(output_dir=save_step_labels_dir, cfg=save_cfg)

    mix_data = instantiate_and_bootstrap_tmm(
        sorting=sorting,
        motion=motion,
        stable_features=stable_features,
        tpca=tpca,
        clustering_features_cfg=clustering_features_cfg,
        refinement_cfg=refinement_cfg,
        seed=seed,
        computation_cfg=computation_cfg,
        save_kwargs=_init_save_kw if saving else None,
    )
    tmm, train_data, val_data, full_data, train_ixs, _ = mix_data

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
    tmm.em(train_data, min_iters=tmm.p.main_min_iters)
    stepname = "tmm00em"
    if saving:
        save_tmm_labels(tmm=tmm, stepname=stepname, **save_kw)  # type: ignore

    logger.dartsortdebug(
        "TMM will run %s (%sx).",
        refinement_cfg.mixture_steps,
        refinement_cfg.n_total_iters,
    )
    outer_it = -1
    allow_blanks = False
    for outer_it in range(refinement_cfg.n_total_iters):
        for inner_it, step_type in enumerate(refinement_cfg.mixture_steps):
            last_outer = outer_it == refinement_cfg.n_total_iters - 1
            last_inner = inner_it == len(refinement_cfg.mixture_steps) - 1
            is_final = last_outer and last_inner
            _will_em = True
            if step_type.endswith("split"):
                run_split(
                    tmm,
                    train_data=train_data,
                    val_data=val_data,
                    prog_level=prog_level,
                    single=step_type.startswith("single"),
                )
            elif step_type == "merge":
                run_merge(
                    tmm,
                    train_data,
                    val_data,
                    prog_level,
                    full_data=full_data,
                    original_sorting=sorting,
                )
            elif step_type == "demolish":
                assert val_data is not None
                _will_em = not is_final or refinement_cfg.em_after_demolish
                tmm.demolish(
                    train_data=train_data,
                    val_data=val_data,
                    show_progress=bool(prog_level),
                    allow_uncovered=not _will_em,
                )
                allow_blanks = allow_blanks or _will_em
            else:
                panic()
            if _will_em:
                tmm.em(
                    train_data,
                    show_progress=prog_level,
                    allow_blanks=allow_blanks,
                    min_iters=tmm.p.main_min_iters,
                )

            stepname = f"tmm{outer_it}{inner_it}{step_type}"
            if saving and (not is_final or skip_final_assign_and_return_mix_data):
                save_tmm_labels(tmm=tmm, stepname=stepname, **save_kw)  # type: ignore

    if skip_final_assign_and_return_mix_data:
        return mix_data

    # final assignments
    assert full_data is not None
    sorting = relabel_and_add_scores(sorting, tmm, full_data)
    # downstream, we might care if a spike was used for training
    is_train = np.zeros(len(sorting), dtype=bool)
    is_train[train_ixs] = True
    sorting.add_ephemeral_feature("gmm_train", is_train)
    # log proportion can be useful for downstream analysis
    sorting.add_ephemeral_feature(
        "unit_log_proportions", tmm.b.log_proportions.numpy(force=True)
    )
    del tmm, train_data, val_data, full_data

    if saving:
        # final sorting gets dumped entirely
        assert save_step_labels_format is not None
        ds_save_intermediate_sorting(
            step_name=save_step_labels_format.format(stepname=stepname),
            step_sorting=sorting,
            output_dir=save_step_labels_dir,
            cfg=save_cfg,
        )

    return sorting


# -- shared objects holding precomputed neighborhood-related data


RobustnessStrategy = Literal["none", "huber"]


class MixtureModelAndDatasets(NamedTuple):
    tmm: "TruncatedMixtureModel"
    train_data: "TruncatedSpikeData"
    val_data: "TruncatedSpikeData | None"
    full_data: "StreamingSpikeData | None"
    train_ixs: Tensor | slice
    val_ixs: Tensor | None


class NeighborhoodCovariance(BModule):
    def __init__(
        self,
        feat_rank: int,
        max_nc_obs: int,
        max_nc_miss_near: int,
        n_channels: int,
        prgeom: Tensor,
        neighb_adj: Tensor,
        nobs: Tensor,
        obs_ix: Tensor,
        miss_near_ix: Tensor,
        miss_full_mask: Tensor,
        logdet: Tensor,
        temporal_whitener: Tensor,
        spatial_whitener: Tensor,
        spatial_regression: Tensor,
        full_spatial_whitener: Tensor,
        huber_k: Tensor | None,
        huber_2logzratio: Tensor | None,
    ):
        """Holds precomputed terms which depend only on the neighborhood

        Parameters
        ----------
        feat_rank : int
            tpca rank
        max_nc_obs: int
            most observed channels in any neighborhood
        max_nc_miss_near: int
            most missing channels (inside the cov zero radius)
        n_channels: int
            total n channels
        prgeom: Tensor
            padded registered geometry
        neighb_adj: Tensor
            adjacency of neighborhoods
        nobs: Tensor
            number of observed features (rank*chans) by neighborhood
        obs_ix: Tensor
            observed channels (just the neighborhoods array)
        miss_near_ix: Tensor
            channels within zero rad of observed
        miss_full_mask: Tensor
            indicator of missingness (not restricted by zero rad)
            miss_ix_full not used
        logdet: Tensor
            log det of observed covariance
        temporal_whitener: Tensor
        spatial_whitener: Tensor
        spatial_regression: Tensor
        full_spatial_whitener: Tensor
        huber_k: Tensor | None
            noise unit's Huber radius by neighborhood
        huber_2logzratio: Tensor | None
            2log(Z_huber / Z_gaussian) by neighborhood (incl in lik_const)
        """
        super().__init__()
        self.feat_rank = feat_rank
        self.max_nc_obs = max_nc_obs
        self.max_nc_miss_near = max_nc_miss_near
        self.n_channels = n_channels
        self.register_buffer("prgeom", prgeom, persistent=False)
        self.register_buffer("neighb_adj", neighb_adj, persistent=False)
        self.register_buffer("nobs", nobs, persistent=False)
        self.register_buffer("obs_ix", obs_ix, persistent=False)
        self.register_buffer("miss_near_ix", miss_near_ix, persistent=False)
        self.register_buffer("miss_full_mask", miss_full_mask, persistent=False)
        self.register_buffer("logdet", logdet, persistent=False)
        self.register_buffer("temporal_whitener", temporal_whitener, persistent=False)
        self.register_buffer("spatial_whitener", spatial_whitener, persistent=False)
        self.register_buffer("spatial_regression", spatial_regression, persistent=False)
        self.register_buffer(
            "full_spatial_whitener", full_spatial_whitener, persistent=False
        )
        lik_const = logdet + LOG_2PI * nobs
        assert (huber_k is None) == (huber_2logzratio is None)
        if huber_2logzratio is not None:
            lik_const += huber_2logzratio
        self.register_buffer("lik_const", lik_const, persistent=False)
        self.register_buffer_or_none("huber_k", huber_k, persistent=False)

    def __post_init__(self):
        """Shape validation / documentation."""
        nneighb = self.b.nobs.shape[0]
        nco = self.max_nc_obs
        assert self.nobs.shape == (nneighb,)
        assert self.obs_ix.shape == (nneighb, nco)
        assert self.miss_near_ix.shape == (nneighb, self.max_nc_miss_near)
        assert self.miss_full_mask.shape == (nneighb, self.n_channels)
        assert self.logdet.shape == (nneighb,)
        assert self.temporal_whitener.shape == (self.feat_rank, self.feat_rank)
        assert self.spatial_whitener.shape == (nneighb, nco, nco)
        assert self.spatial_regression.shape == (nneighb, nco, self.max_nc_miss_near)
        assert self.full_spatial_whitener.shape == (self.n_channels, self.n_channels)

    def pad_for_noise_score_(self, new_n_neighbs: int):
        noise_score_keys = ["spatial_whitener", "logdet", "nobs", "lik_const"]
        if self.huber_k is not None:
            noise_score_keys.append("huber_k")
        for k in noise_score_keys:
            v = getattr(self, k)
            ndim = v.ndim
            if v.shape[0] >= new_n_neighbs:
                assert torch.all(v[new_n_neighbs - 1 :] == 0)
                continue
            v = F.pad(v, [0, 0] * (ndim - 1) + [0, 1])
            self.register_buffer(k, v, persistent=False)

    @classmethod
    def from_noise_and_neighborhoods(
        cls,
        prgeom: Tensor,
        noise: KroneckerNoise,
        neighborhoods: SpikeNeighborhoods,
        neighb_overlap: float,
        huber_quantile: float | None,
    ) -> Self:
        dev = noise.device
        neighborhoods = neighborhoods.to(device=dev)
        prgeom = prgeom.to(device=dev)
        nc_obs, obs_ix, miss_near_ix, miss_full_mask = _neighborhood_indices(
            neighborhoods=neighborhoods,
            zero_radius=noise.zero_radius,
            prgeom=prgeom,
        )
        logdet, spatial_whitener, spatial_regression = _noise_factors(
            noise=noise, obs_ix=obs_ix, miss_near_ix=miss_near_ix
        )
        assert obs_ix.shape[0] == neighborhoods.n_neighborhoods
        assert obs_ix.shape[0] == miss_near_ix.shape[0] == miss_full_mask.shape[0]
        assert obs_ix.shape[0] == logdet.shape[0]
        assert obs_ix.shape[0] == spatial_whitener.shape[0]
        assert obs_ix.shape[0] == spatial_regression.shape[0]
        nobs = noise.rank * nc_obs
        if huber_quantile is None:
            huber_k = huber_2logzratio = None
        else:
            huber_k, huber_2logzratio = huber_neighborhood_constants(
                nobs, huber_quantile
            )
        return cls(
            feat_rank=noise.rank,
            max_nc_obs=obs_ix.shape[1],
            max_nc_miss_near=miss_near_ix.shape[1],
            n_channels=neighborhoods.n_channels,
            prgeom=prgeom,
            neighb_adj=neighborhoods.adjacency(neighb_overlap),
            nobs=nobs,
            obs_ix=obs_ix,
            miss_near_ix=miss_near_ix,
            miss_full_mask=miss_full_mask,
            logdet=logdet,
            temporal_whitener=noise.b.temporal_whitener,
            spatial_whitener=spatial_whitener,
            spatial_regression=spatial_regression,
            full_spatial_whitener=noise.b.spatial_whitener,
            huber_k=huber_k,
            huber_2logzratio=huber_2logzratio,
        )


class NeighborhoodLUT(BModule):
    """Which labels coincided with which spike neighborhoods? Need this all over the place."""

    def __init__(self, unit_ids: Tensor, neighb_ids: Tensor, lut: Tensor):
        super().__init__()
        self.register_buffer("unit_ids", unit_ids)
        self.register_buffer("neighb_ids", neighb_ids)
        # lut[unit_ids, neighb_ids] = arange(n_lut)
        # everything else is n_lut (not -1)
        self.register_buffer("lut", lut)

    @classmethod
    def from_units_and_neighbs(
        cls, *, unit_ids: Tensor, neighb_ids: Tensor, n_units: int, n_neighbs: int
    ) -> Self:
        n_lut = unit_ids.shape[0]
        lut = unit_ids.new_full((n_units, n_neighbs), n_lut, dtype=torch.long)
        lut[unit_ids, neighb_ids] = torch.arange(n_lut, device=lut.device)
        return cls(unit_ids=unit_ids, neighb_ids=neighb_ids, lut=lut)

    @classmethod
    def from_mask(cls, mask: Tensor) -> Self:
        uu, nn = mask.nonzero(as_tuple=True)
        return cls.from_units_and_neighbs(
            unit_ids=uu, neighb_ids=nn, n_units=mask.shape[0], n_neighbs=mask.shape[1]
        )

    @classmethod
    def from_candidates_and_neighborhoods(
        cls,
        *,
        candidates: Tensor,
        neighborhoods: SpikeNeighborhoods,
        neighb_supset: Tensor,
        n_units: int,
        neighborhood_ids: Tensor | None,
    ) -> Self:
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
        return cls.from_mask(co)

    def eq(self, other: object) -> bool:
        if not isinstance(other, NeighborhoodLUT):
            return False
        if not torch.equal(self.b.unit_ids, other.b.unit_ids):
            return False
        if not torch.equal(self.b.neighb_ids, other.b.neighb_ids):
            return False
        return torch.equal(self.b.lut, other.b.lut)

    def full_proposal_candidates(self, pad_shape_to: int | None = None) -> Tensor:
        n_proposed = max_units_per_neighb(self)
        if pad_shape_to is not None:
            n_proposed = max(n_proposed, pad_shape_to)
        proposals = full_proposal_by_neighb(self, n_proposed)
        return proposals


def lut_blank_units(lut: NeighborhoodLUT) -> Tensor:
    return (lut.b.lut == lut.b.unit_ids.shape[0]).all(1).nonzero(as_tuple=True)[0]


class LUTParams(BModule):
    """Holds precomputed terms which depend on neighborhood and GMM parameters."""

    def __init__(
        self,
        signal_rank: int,
        n_lut: int,
        # unit- and neighborhood-dependent terms
        muo: Tensor,
        Linvmuo: Tensor,
        CmoCooinvmuo: Tensor,
        # low-rank model observed-data log determinant plus other constant lik terms
        constplogdet: Tensor,
        # signal_rank > 0 only
        TWoCooinvsqrt: Tensor | None,
        TWoCooinvmuo: Tensor | None,
        # T, zero-padded on inner dim with an extra column
        Tpad: Tensor | None,
        # main factor for using inverse lemma to compute likelihoods
        wburyroot: Tensor | None,
        huber_k: Tensor | None,
        huber_2logzratio: Tensor | None,
    ):
        super().__init__()
        self.signal_rank = signal_rank
        self.n_lut = n_lut
        self.register_buffer("muo", muo, persistent=False)
        self.register_buffer("Linvmuo", Linvmuo, persistent=False)
        self.register_buffer("CmoCooinvmuo", CmoCooinvmuo, persistent=False)
        self.register_buffer("constplogdet", constplogdet, persistent=False)
        self.register_buffer_or_none("TWoCooinvsqrt", TWoCooinvsqrt, persistent=False)
        self.register_buffer_or_none("TWoCooinvmuo", TWoCooinvmuo, persistent=False)
        self.register_buffer_or_none("Tpad", Tpad, persistent=False)
        self.register_buffer_or_none("wburyroot", wburyroot, persistent=False)
        assert (huber_k is None) == (huber_2logzratio is None)
        self.register_buffer_or_none("huber_k", huber_k, persistent=False)
        self.register_buffer_or_none(
            "huber_2logzratio", huber_2logzratio, persistent=False
        )

    @classmethod
    def from_lut(
        cls,
        means: Tensor,
        bases: Tensor | None,
        neighb_cov: NeighborhoodCovariance,
        lut: NeighborhoodLUT,
        huber_quantile: float | None,
        puff=1.0,
    ):
        signal_rank = 0 if bases is None else bases.shape[1]
        if huber_quantile is None:
            huber_k = huber_2logzratio = None
        else:
            huber_k, huber_2logzratio = huber_neighborhood_constants(
                neighb_cov.b.nobs, huber_quantile
            )
        self = cls._new_empty(
            lut.b.unit_ids.shape[0],
            signal_rank,
            dim_obs=neighb_cov.max_nc_obs * neighb_cov.feat_rank,
            dim_miss_near=neighb_cov.max_nc_miss_near * neighb_cov.feat_rank,
            device=means.device,
            huber_k=huber_k,
            huber_2logzratio=huber_2logzratio,
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
        batch_size=1024,
    ):
        self._resize_(lut.b.unit_ids.shape[0])
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
        huber_k: Tensor | None,
        huber_2logzratio: Tensor | None,
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
            huber_k=huber_k,
            huber_2logzratio=huber_2logzratio,
        )
        if n0 != n_lut:
            self._resize_(n_lut)
        return self

    def _resize_(self, n_lut_new: int, *, zero_all: bool = True):
        if n_lut_new == self.n_lut:
            return

        self.n_lut = n0 = n_lut_new

        self.b.muo.resize_(n0, *self.b.muo.shape[1:])
        self.b.Linvmuo.resize_(n0, *self.b.Linvmuo.shape[1:])
        self.b.CmoCooinvmuo.resize_(n0, *self.b.CmoCooinvmuo.shape[1:])
        self.b.constplogdet.resize_(n0, *self.b.constplogdet.shape[1:])

        self.b.muo.zero_()
        self.b.Linvmuo.zero_()
        self.b.CmoCooinvmuo.zero_()
        self.b.constplogdet.zero_()

        if not self.signal_rank:
            return
        assert self.TWoCooinvsqrt is not None
        assert self.TWoCooinvmuo is not None
        assert self.Tpad is not None
        assert self.wburyroot is not None

        self.b.TWoCooinvsqrt.resize_(n0, *self.b.TWoCooinvsqrt.shape[1:])
        self.b.TWoCooinvmuo.resize_(n0, *self.b.TWoCooinvmuo.shape[1:])
        self.b.Tpad.resize_(n0, *self.b.Tpad.shape[1:])
        self.b.wburyroot.resize_(n0, *self.b.wburyroot.shape[1:])

        self.b.TWoCooinvsqrt.zero_()
        self.b.TWoCooinvmuo.zero_()
        self.b.Tpad.zero_()
        self.b.wburyroot.zero_()

    def check(self):
        muo_f = self.b.muo.isfinite().all()
        Linvmuo_f = self.b.Linvmuo.isfinite().all()
        CmoCooinvmuo_f = self.b.CmoCooinvmuo.isfinite().all()
        constplogdet_f = self.b.constplogdet.isfinite().all()
        rank0_good = all([muo_f, Linvmuo_f, CmoCooinvmuo_f, constplogdet_f])
        if not self.signal_rank:
            if not rank0_good:
                raise BadNumbersError(
                    f"LUTParams blew up: {muo_f=} {Linvmuo_f=} "
                    f"{CmoCooinvmuo_f=} {constplogdet_f=}"
                )
            return

        assert self.b.TWoCooinvsqrt is not None
        assert self.b.TWoCooinvmuo is not None
        assert self.b.Tpad is not None
        assert self.b.wburyroot is not None
        TWoCooinvsqrt_f = self.b.TWoCooinvsqrt.isfinite().all()
        TWoCooinvmuo_f = self.b.TWoCooinvmuo.isfinite().all()
        Tpad_f = self.b.Tpad.isfinite().all()
        wburyroot_f = self.b.wburyroot.isfinite().all()
        basis_good = all([TWoCooinvsqrt_f, TWoCooinvmuo_f, Tpad_f, wburyroot_f])
        if not (rank0_good and basis_good):
            raise BadNumbersError(
                f"LUTParams blew up: {rank0_good=} {basis_good=}.\n"
                f"{muo_f=} {Linvmuo_f=} {CmoCooinvmuo_f=} {constplogdet_f=}.\n"
                f"{TWoCooinvsqrt_f=} {TWoCooinvmuo_f=} {Tpad_f=} {wburyroot_f=}."
            )


# -- messenger classes


@databag
class Scores:
    log_liks: Tensor
    responsibilities: Tensor | None
    candidates: Tensor
    duties: Tensor | None

    def slice(self, indices: Tensor | slice) -> "Scores":
        if self.responsibilities is None:
            r = None
        else:
            r = self.responsibilities[indices]
        if self.duties is None:
            d = None
        else:
            d = self.duties[indices]
        return Scores(
            log_liks=self.log_liks[indices],
            responsibilities=r,
            candidates=self.candidates[indices],
            duties=d,
        )


@databag
class SufficientStatistics:
    """Represents a batch's statistics or the whole dataset's statistics in E step."""

    noise_N: Tensor | None
    resp_N: Tensor
    N: Tensor
    Nlut: Tensor
    R: Tensor
    Ulut: Tensor | None

    def diagnostic(self) -> str:
        noise_N = self.noise_N.item() if self.noise_N is not None else None
        resp_Ns = f"{self.resp_N.amin().item()}, {self.resp_N.amax().item()}"
        Ns = f"{self.N.amin().item()}, {self.N.amax().item()}"
        Nluts = f"{self.Nlut.amin().item()}, {self.Nlut.amax().item()}"
        Rna = self.R.isnan().any(dim=(1, 2)).sum().item()
        Rinf = self.R.isinf().any(dim=(1, 2)).sum().item()
        msg = (
            f"SufficientStatistics:\n"
            f" - noise count: {noise_N}\n"
            f" - unit responsibility count min/max: {resp_Ns}\n"
            f" - unit weight count min/max: {Ns}\n"
            f" - LUT weight count min/max: {Nluts}\n"
            f" - R na/inf count: {Rna}, {Rinf}"
        )
        return msg

    @classmethod
    def zeros(
        cls,
        *,
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
            noise_N=noise_N,
            resp_N=torch.zeros((n_units,), device=device, dtype=count_dtype),
            N=torch.zeros((n_units,), device=device, dtype=count_dtype),
            Nlut=torch.zeros((n_lut,), device=device, dtype=count_dtype),
            R=torch.zeros((n_units, n_channels, hat_dim, feature_rank), device=device),
            Ulut=Ulut,
        )

    def reset_(
        self, n_units: int, n_channels: int, signal_rank: int, feature_rank: int
    ):
        if self.noise_N is not None:
            self.noise_N.zero_()
        self.resp_N.zero_()
        self.N.zero_()
        self.Nlut.zero_()
        self.R.zero_()
        hat_dim = signal_rank + 1
        assert self.R.numel() == n_units * n_channels * hat_dim * feature_rank
        self.R.resize_(n_units, n_channels, hat_dim, feature_rank)
        if self.Ulut is not None:
            self.Ulut.zero_()

    def combine(self, other: "SufficientStatistics", eps: Tensor):
        """Welford running means."""
        self.resp_N += other.resp_N
        self.N += other.N
        w_N = other.N.div_(self.N.clamp(min=eps))
        self.Nlut += other.Nlut

        self.R += other.R.sub_(self.R).mul_(w_N[:, None, None, None])

        if self.noise_N is not None:
            assert other.noise_N is not None
            self.noise_N += other.noise_N

        if self.Ulut is not None:
            assert other.Ulut is not None
            w_Nlut = other.Nlut.div_(self.Nlut.clamp(min=eps))
            self.Ulut += other.Ulut.sub_(self.Ulut).mul_(w_Nlut[:, None, None])


@databag
class TMMParams:
    max_group_size: int
    split_friend_distance: float
    split_max_distance: float
    merge_max_distance: float
    split_k: int
    single_split_k: int
    distance_kind: ComponentDistanceMetric
    em_iters: int
    min_em_iters: int
    prior_pseudocount: float
    initial_basis_shrinkage: float
    criterion_em_iters: int
    cl_alpha: float
    cl_split_only: bool
    demolish_cl_alpha: float
    min_count: int
    min_channel_count: int
    split_min_count: int
    full_proposal_every: int
    main_min_iters: int
    elbo_atol: float
    robust_strategy: RobustnessStrategy
    huber_quantile: float
    criterion_clip_level: float | None
    demolition_min_resp_ratio: float
    demolish_during_selection: bool
    kmeans_tries: int
    kmeans_beta: float
    kmeanspp_tries: int
    whiten_split: bool
    split_kmeanspp_metric: SplitKmeansppMetric
    fix_responsibilities: bool
    scale_dist_args: tuple[float, float, float]
    whiten_dist: bool
    merge_violation_threshold: float | None
    merge_violation_ms: float
    merge_jitter_ms: float
    merge_censor_ms: float
    merge_min_violation_evidence: float

    @classmethod
    def from_refinement_cfg(cls, refinement_cfg: RefinementConfig):
        return cls(
            max_group_size=refinement_cfg.merge_group_size,
            split_max_distance=refinement_cfg.split_distance_threshold,
            split_friend_distance=refinement_cfg.split_friend_distance,
            merge_max_distance=refinement_cfg.merge_distance_threshold,
            distance_kind=refinement_cfg.distance_metric,
            min_count=refinement_cfg.min_count,
            split_min_count=refinement_cfg.split_min_count,
            split_k=refinement_cfg.kmeansk,
            single_split_k=refinement_cfg.single_split_k,
            kmeans_tries=refinement_cfg.kmeans_tries,
            kmeans_beta=refinement_cfg.kmeans_beta,
            kmeanspp_tries=refinement_cfg.kmeanspp_cfg.tries,
            min_channel_count=refinement_cfg.channels_count_min,
            em_iters=refinement_cfg.n_em_iters,
            criterion_em_iters=refinement_cfg.criterion_em_iters,
            min_em_iters=refinement_cfg.criterion_em_iters,
            elbo_atol=refinement_cfg.em_converged_atol,
            prior_pseudocount=refinement_cfg.prior_pseudocount,
            initial_basis_shrinkage=refinement_cfg.initial_basis_shrinkage,
            full_proposal_every=refinement_cfg.full_proposal_every,
            main_min_iters=refinement_cfg.main_min_iters,
            cl_alpha=refinement_cfg.cl_alpha,
            cl_split_only=refinement_cfg.cl_split_only,
            demolish_cl_alpha=refinement_cfg.demolish_cl_alpha,
            robust_strategy=refinement_cfg.robust_strategy,
            huber_quantile=refinement_cfg.huber_quantile,
            criterion_clip_level=refinement_cfg.criterion_clip_level,
            demolition_min_resp_ratio=refinement_cfg.demolition_min_resp_ratio,
            demolish_during_selection=refinement_cfg.demolish_during_selection,
            whiten_split=refinement_cfg.whiten_split,
            split_kmeanspp_metric=refinement_cfg.split_kmeanspp_metric,
            fix_responsibilities=refinement_cfg.fix_responsibilities_in_split,
            scale_dist_args=refinement_cfg.scale_dist_args,
            whiten_dist=refinement_cfg.whiten_dist,
            merge_violation_threshold=refinement_cfg.glom_violation_threshold,
            merge_violation_ms=refinement_cfg.glom_violation_ms,
            merge_jitter_ms=refinement_cfg.glom_jitter_ms,
            merge_censor_ms=refinement_cfg.censor_ms,
            merge_min_violation_evidence=refinement_cfg.glom_min_violation_evidence,
        )


class BadNumbersError(ValueError):
    pass


@databag
class TruncatedEStepResult:
    candidates: Tensor
    stats: SufficientStatistics
    elbo: Tensor


@databag
class SuccessfulSplitCaseResult:
    """Result of splitting an individual unit/group (or none, see next def)."""

    unit_ids: Tensor
    n_split: int
    train_indices: Tensor
    train_assignments: Tensor
    means: Tensor
    sub_proportions: Tensor
    bases: Tensor | None


SplitCaseResult = SuccessfulSplitCaseResult | None


@databag
class SplitCaseDebugInfo:
    bailed: bool = False
    bail_reason: str | None = None
    split_model: "TruncatedMixtureModel | None" = None
    kmeans_x: Tensor | None = None
    kmeans_chans: Tensor | None = None
    kmeans_responsibilities: Tensor | None = None
    merge_res: "SuccessfulGroupMergeResult | None" = None
    split_data: "DenseSpikeData | None" = None


@databag
class ModelSelectionContext:
    train_data: "TruncatedSpikeData"
    eval_data: "TruncatedSpikeData | None"
    eval_scores: Scores
    train_labels: Tensor
    eval_labels: Tensor

    @classmethod
    def from_scores(
        cls,
        train_data: "TruncatedSpikeData",
        eval_data: "TruncatedSpikeData | None",
        train_scores: Scores,
        eval_scores: Scores,
    ) -> Self:
        return cls(
            train_data=train_data,
            eval_data=eval_data,
            eval_scores=eval_scores,
            train_labels=labels_from_scores_(train_scores),
            eval_labels=labels_from_scores_(eval_scores),
        )

    @classmethod
    def from_model(
        cls,
        tmm: "TruncatedMixtureModel",
        train_data: "TruncatedSpikeData",
        eval_data: "TruncatedSpikeData | None",
        show_progress: int = 0,
    ) -> Self:
        train_scores = tmm.soft_assign(
            data=train_data,
            full_proposal_view=True,
            needs_bootstrap=False,
            show_progress=show_progress,
        )
        if eval_data is None:
            eval_scores = train_scores
        else:
            eval_scores = tmm.soft_assign(
                data=eval_data,
                full_proposal_view=True,
                needs_bootstrap=False,
                show_progress=show_progress,
            )
        return cls.from_scores(train_data, eval_data, train_scores, eval_scores)


@databag
class SplitCaseProposal:
    group: Tensor
    train_indices: Tensor
    kmeans_responsibilities: Tensor
    kmeans_x: Tensor | None
    kmeans_chans: Tensor | None


@databag
class MergeProposal:
    group: Tensor
    block: int
    pair_mask: Tensor | None
    responsibilities: Tensor | None
    cur_scores: Scores
    use_cl_alpha: bool


@databag
class SplitResult:
    """Final result of .split()"""

    n_new_units: int
    n_shrink: int
    train_unit_mask: Tensor
    train_split_spike_mask: Tensor
    train_split_spike_labels: Tensor


@databag
class GroupPartition:
    """Represents a candidate grouping of some unit_ids into subsets."""

    unit_ids: Tensor
    group_ids: Tensor
    n_groups: int
    n_groups_before_demolish: int
    single_ixs: list[int]
    subset_ids: list[int]
    unit_ids_combined: list[int]
    single_subset_order: Tensor
    single_demolish_ixs: list[int]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(n_groups={self.n_groups}"
            f", unit_ids={self.unit_ids.tolist()}"
            f", group_ids={self.group_ids.tolist()}"
            f", single_ixs={self.single_ixs}"
            f", subset_ids={self.subset_ids}"
            f", unit_ids_combined={self.unit_ids_combined}"
            f", single_subset_order={self.single_subset_order.tolist()}"
            f", single_demolish_ixs={self.single_demolish_ixs}"
            ")"
        )


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
GroupMergeResult = SuccessfulGroupMergeResult | None


@databag
class GroupDemolition:
    """Harbinger of impending unit destruction."""

    unit_ids: Tensor
    improvement: float
    demolished: Tensor | None


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

    def padded_map(self):
        return F.pad(self.mapping, (0, 1), value=-1)

    @classmethod
    def discard_mapping(
        cls,
        n_units: int,
        invalidated_ids: list[int] | Tensor,
        device: torch.device | str = "cpu",
    ) -> Self:
        device = torch.device(device)
        invalidated_ids = torch.asarray(invalidated_ids, dtype=torch.long)
        mapping = torch.zeros(n_units, dtype=torch.long, device=invalidated_ids.device)
        mapping[invalidated_ids] = -1
        (nz,) = (mapping == 0).nonzero(as_tuple=True)
        mapping[nz] = torch.arange(nz.numel()).to(mapping)
        mapping = mapping.to(device=device)
        return cls(mapping=mapping)


@databag
class EMResult:
    elbos: list[float]


@databag
class SpikeDataBatch:
    """Yielded by data object .batches()

    # x, CmoCooinvx are not needed for soft_assign/score, so they can be None.
    """

    batch: Tensor | slice | None
    neighborhood_ids: Tensor
    xt: Tensor | None
    candidates: Tensor
    whitenedx: Tensor
    noise_logliks: Tensor
    CmoCooinvx: Tensor | None
    candidate_count: int | None
    sample_weights: Tensor | None


@databag
class DenseSpikeData:
    indices: Tensor
    neighborhoods: SpikeNeighborhoods
    neighb_supset: Tensor
    neighborhood_ids: Tensor
    xt: Tensor | None
    whitenedx: Tensor
    CmoCooinvx: Tensor | None
    noise_logliks: Tensor
    sample_weights: Tensor | None
    batch_size: int

    def __post_init__(self):
        assert self.score_only == (self.CmoCooinvx is None)

    @property
    def N(self) -> int:  # noqa: N802
        return self.whitenedx.shape[0]

    @property
    def score_only(self) -> bool:
        return self.xt is None

    def x(self) -> Tensor | None:
        if self.xt is None:
            return None
        return self.xt.transpose(1, 2).reshape(self.N, -1)

    def slice(self, indices: Tensor | slice):
        if self.sample_weights is None:
            sample_weights = None
        else:
            sample_weights = self.sample_weights[indices]
        if self.score_only:
            xt = CmoCooinvx = None
        else:
            assert self.xt is not None
            assert self.CmoCooinvx is not None
            xt = self.xt[indices]
            CmoCooinvx = self.CmoCooinvx[indices]
        return DenseSpikeData(
            indices=self.indices[indices],
            neighborhoods=self.neighborhoods,
            neighb_supset=self.neighb_supset,
            neighborhood_ids=self.neighborhood_ids[indices],
            xt=xt,
            whitenedx=self.whitenedx[indices],
            CmoCooinvx=CmoCooinvx,
            noise_logliks=self.noise_logliks[indices],
            sample_weights=sample_weights,
            batch_size=self.batch_size,
        )

    def covered_channels(self, min_count=0):
        counts = self.neighborhoods.b.indicators[:, self.neighborhood_ids].sum(1)
        if min_count:
            (channels,) = (counts >= min_count).nonzero(as_tuple=True)
        else:
            (channels,) = counts.nonzero(as_tuple=True)
        return channels

    def masked_observed_mean(
        self, x: Tensor, weights: Tensor | None = None, pseudocount: float = 0.0
    ) -> Tensor:
        if weights is None:
            weights = x.new_ones((self.N, 1))
        assert weights.ndim == 2
        assert weights.shape[0] == self.N
        assert x.ndim == 3
        assert x.shape[0] == self.N
        n, rank, nco = x.shape
        nc = self.neighborhoods.n_channels
        channels = self.neighborhoods.b.neighborhoods[self.neighborhood_ids].view(-1)
        assert channels.numel() == n * nco

        counts = weights.T @ self.neighborhoods.b.indicators[:, self.neighborhood_ids].T
        counts += pseudocount
        counts = counts.where(counts > 0, 1.0)

        x = x.permute(1, 0, 2)
        means = x.new_zeros((weights.shape[1], rank, nc + 1))
        for mean, weight in zip(means, weights.T, strict=True):
            wx = (x * weight[:, None]).reshape(rank, n * nco)
            mean.index_add_(1, channels, wx)
        means = means[:, :, :nc]
        means /= counts[:, None, :]
        return means

    def weighted_covered_channels(
        self, weights: Tensor, within: Tensor | None = None, min_count: int = 0
    ) -> tuple[Tensor, Tensor]:
        assert weights.shape[0] == self.N
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

        # covered spikes are which spikes live inside each subset
        spike_chan_counts = spike_chan_inds.sum(dim=0)
        chan_coverage = spike_chan_inds.T @ unit_masks.float().T
        # spikes x units
        full_coverage = chan_coverage >= spike_chan_counts[:, None]

        return unit_masks, full_coverage


@databag
class ComponentBlockData:
    """Data for fitting a block of many units at once
     - row_indptr stores block endpoints in spike index space (diff of block_ids)
     - block_unit_ids are the units in each block, could be a group (in split)
    """
    data: DenseSpikeData
    row_indptr: np.ndarray
    block_ids: Tensor
    block_unit_ids: Tensor

    def __post_init__(self):
        assert self.row_indptr.shape == (self.n_blocks + 1,)
        assert self.row_indptr[0] == 0
        assert self.row_indptr[-1] == self.data.N
        assert self.block_ids.shape == (self.data.N,)
        assert self.block_unit_ids.ndim == 2

    @property
    def N(self) -> int:  # noqa: N802
        return self.data.N

    @property
    def n_blocks(self) -> int:
        return self.block_unit_ids.shape[0]

    @property
    def width(self) -> int:
        return self.block_unit_ids.shape[1]

    @classmethod
    def single(cls, data: DenseSpikeData, unit_ids: Tensor) -> Self:
        return cls.from_block_sizes(data, [data.N], torch.as_tensor(unit_ids)[None])

    @classmethod
    def from_block_sizes(
        cls, data: DenseSpikeData, block_sizes: Sequence[int], block_unit_ids: Tensor
    ) -> Self:
        device = data.neighborhood_ids.device
        block_ids = torch.repeat_interleave(
            torch.arange(len(block_sizes), device=device),
            torch.as_tensor(block_sizes, device=device),
            output_size=data.N,
        )
        return cls(
            data=data,
            row_indptr=np.concatenate([[0], np.cumsum(block_sizes)]),
            block_ids=block_ids,
            block_unit_ids=block_unit_ids.to(device),
        )

    def block_slice(self, block: int) -> slice:
        return slice(int(self.row_indptr[block]), int(self.row_indptr[block + 1]))

    def block_size(self, block: int) -> int:
        return int(self.row_indptr[block + 1] - self.row_indptr[block])

    def select_blocks(self, blocks: Sequence[int]) -> tuple[Self, Tensor]:
        device = self.block_ids.device
        indices = torch.concatenate(
            [
                torch.arange(self.row_indptr[b], self.row_indptr[b + 1], device=device)
                for b in blocks
            ]
        )
        selected = self.from_block_sizes(
            self.data.slice(indices),
            [self.block_size(b) for b in blocks],
            self.block_unit_ids[torch.tensor(blocks, device=device)],
        )
        return selected, indices

    def keep_blocks(self, keep: np.ndarray) -> tuple[Self, Tensor]:
        assert keep.shape == (self.n_blocks,)
        block_sizes = np.where(keep, np.diff(self.row_indptr), 0)
        indices = np.concatenate(
            [
                np.arange(self.row_indptr[b], self.row_indptr[b + 1])
                for b in np.flatnonzero(keep)
            ]
        )
        indices = torch.from_numpy(indices).to(self.block_ids.device)
        kept = self.__class__(
            data=self.data.slice(indices),
            row_indptr=np.concatenate([[0], np.cumsum(block_sizes)]),
            block_ids=self.block_ids[indices],
            block_unit_ids=self.block_unit_ids,
        )
        return kept, indices

    def with_block_unit_ids(self, block_unit_ids: Tensor) -> Self:
        return self.__class__(
            data=self.data,
            row_indptr=self.row_indptr,
            block_ids=self.block_ids,
            block_unit_ids=block_unit_ids.to(self.block_ids.device),
        )

    def unit_candidates(self, lut: NeighborhoodLUT) -> Tensor:
        units = self.block_unit_ids[self.block_ids]
        lut_ixs = lut.b.lut[units.clamp(min=0), self.data.neighborhood_ids[:, None]]
        covered = (units >= 0).logical_and_(lut_ixs < lut.b.unit_ids.shape[0])
        return units.where(covered, -1)

    def to_batches(self, lut: NeighborhoodLUT) -> list[SpikeDataBatch]:
        """Convert dense data to a batch, which means picking candidates.

        The candidates are all of the unit ids, so that each col of the candidates
        array is just unit_ids[col] repeated. The thing is that not all units overlap
        all neighborhoods, so some of these will be -1s per the lut. But we can still
        think of each col as corresponding to a unit.

        This is done by block, so candidates will match the block structure.
        """
        candidates = self.unit_candidates(lut)
        candidate_counts = (candidates >= 0).sum(dim=1).cpu()

        d = self.data
        batches = []
        for i0 in range(0, d.N, d.batch_size):
            sl = slice(i0, min(d.N, i0 + d.batch_size))
            b = SpikeDataBatch(
                batch=sl,
                neighborhood_ids=d.neighborhood_ids[sl],
                xt=None if d.xt is None else d.xt[sl],
                whitenedx=d.whitenedx[sl],
                noise_logliks=d.noise_logliks[sl],
                CmoCooinvx=None if d.CmoCooinvx is None else d.CmoCooinvx[sl],
                candidates=candidates[sl],
                sample_weights=None
                if d.sample_weights is None
                else d.sample_weights[sl],
                candidate_count=int(candidate_counts[sl].sum()),
            )
            batches.append(b)
        return batches

    def slice(self, indices: Tensor) -> Self:
        if pnoid:
            assert (indices.diff() > 0).all()
        block_ids = self.block_ids[indices]
        block_sizes = torch.bincount(block_ids, minlength=self.n_blocks).cpu().numpy()
        return self.__class__(
            data=self.data.slice(indices),
            row_indptr=np.concatenate([[0], np.cumsum(block_sizes)]),
            block_ids=block_ids,
            block_unit_ids=self.block_unit_ids,
        )


def contiguous_block_unit_ids(widths: Tensor) -> Tensor:
    widths = torch.as_tensor(widths)
    starts = widths.cumsum(dim=0) - widths
    slot = torch.arange(int(widths.max()), device=widths.device)
    block_unit_ids = starts[:, None] + slot[None, :]
    return block_unit_ids.where(slot[None, :] < widths[:, None], -1)


@databag
class ComponentBlockFitResult:
    model: "TruncatedMixtureModel"
    data: ComponentBlockData
    responsibilities: Tensor
    valid: Tensor
    kept_mask: Tensor
    block_elbos: np.ndarray


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
        sample_weights: Tensor | None,
        candidates: Tensor | None = None,
        neighb_adj: Tensor | None = None,
        neighb_supset: Tensor | None = None,
        seed: int | np.random.Generator | None = 0,
        batch_size: int = 128,
        explore_neighb_steps: int = 1,
        coverage_steps: int = 2,
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

        # per-sample approximate posterior data
        self.candidates = candidates
        if sample_weights is not None:
            assert sample_weights.shape == (N,)
        self.sample_weights = sample_weights

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
        self.coverage_steps = coverage_steps
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

    def _fill_missing(self, *, n_units: int, allow_uncovered: bool = False):
        # fill in missing labels randomly, obeying un_adj
        if self.candidates is None:
            return
        assert self.rg is not None
        same_adj, n_new = _fill_blank_labels(
            labels=self.candidates[:, 0],
            un_adj=self.un_adj,
            explore_adj=self.explore_adj,
            neighb_adj=self.neighb_adj,
            neighborhood_ids=self.neighborhood_ids,
            gen=self.rg,
            allow_uncovered=allow_uncovered,
        )
        assert same_adj is not None
        if allow_uncovered:
            assert n_new == 0
        else:
            n_units = n_units + n_new
        if not same_adj:
            logger.dartsortverbose(
                "_fill_blank_labels used explore adjacency in bootstrap_candidates."
            )
            self.update_adjacency(n_units=n_units)
        if pnoid:
            assert (self.candidates[:, 0] >= 0).all()
        if pnoid:
            assert torch.equal(
                self.un_adj,
                (self.un_adj_lut.b.lut < self.un_adj_lut.b.unit_ids.shape[0]).float(),
            )

    def batches(
        self, *, show_progress: bool = False, desc: str = "Batches"
    ) -> Iterable[SpikeDataBatch]:
        if show_progress:
            batch_starts = progrange(0, self.N, self.batch_size, desc=desc)
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
        *,
        n_units: int | None,
        un_adj_lut: NeighborhoodLUT | None = None,
        expand_lut: bool = False,
    ):
        """Update my adjacency either using my labels or just a fixed LUT."""
        if n_units is None:
            assert self.candidates is not None
            n_units = int(self.candidates.amax()) + 1
            assert n_units > 0
        self._update_sizes_from_n_units(n_units)

        if expand_lut:
            assert self.un_adj_lut is not None
            assert un_adj_lut is None
            expand_from_lut = self.un_adj_lut
        else:
            expand_from_lut = None

        new_lut = un_adj_lut is not None
        if un_adj_lut is not None:
            assert un_adj_lut.b.lut.shape[0] == n_units

        if self.candidates is None:
            assert un_adj_lut is not None
            labels = None
        else:
            labels = self.candidates[:, 0]

        count_needs_to_match_my_labels = labels is not None and not new_lut
        if pnoid and count_needs_to_match_my_labels:
            assert labels is not None
            assert labels.amax() < n_units

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
        )
        if pnoid:
            assert self.un_adj.max().item() == 1.0

    def ensure_coverage(
        self,
        *,
        allow_new_units: bool,
        allow_uncovered: bool = False,
        max_steps: int | None = None,
        next_label: int | None = None,
    ):
        self.update_adjacency(n_units=None)
        assert self.candidates is not None
        assert self.rg is not None
        res, n_new = _fill_blank_labels(
            labels=self.candidates[:, 0],
            un_adj=self.un_adj,
            explore_adj=self.explore_adj,
            neighb_adj=self.neighb_adj,
            neighborhood_ids=self.neighborhood_ids,
            gen=self.rg,
            ensure_coverage_only=True,
            allow_new_units=allow_new_units,
            allow_uncovered=allow_uncovered,
            next_label=next_label,
            max_steps=max_steps if max_steps is not None else self.coverage_steps,
        )
        assert res is None
        return n_new

    def erase_candidates(self):
        if self.candidates is not None:
            self.candidates.fill_(-1)

    def bootstrap_candidates(
        self,
        distances: Tensor,
        *,
        un_adj_lut: NeighborhoodLUT | None = None,
        allow_uncovered: bool = False,
    ) -> NeighborhoodLUT:
        self.update_adjacency(n_units=distances.shape[0], un_adj_lut=un_adj_lut)
        self._fill_missing(n_units=distances.shape[0], allow_uncovered=allow_uncovered)

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
        *,
        expand_lut: bool = False,
        skip_explore: bool = False,
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
        x: Tensor | np.ndarray,
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
            sample_weights=None,
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
        *,
        expand_lut: bool = False,
        skip_explore: bool = False,
    ) -> tuple[bool, NeighborhoodLUT]:
        # I only do candidates on the fly, so this does nothing.
        # it is an error to assign candidates which would lead to the LUT
        # changing, but that error cannot occur, because I would never
        # propose such candidates!
        assert distances.shape[0] == self.un_adj_lut.b.lut.shape[0]
        return False, self.un_adj_lut

    def update_adjacency(
        self,
        n_units: int | None,
        un_adj_lut: NeighborhoodLUT | None = None,
        *,
        expand_lut: bool = False,
    ):
        super().update_adjacency(
            n_units=n_units, un_adj_lut=un_adj_lut, expand_lut=expand_lut
        )
        # my invariant, see __init__
        assert torch.equal(self.un_adj, self.explore_adj)
        self.max_n_total = max_units_per_neighb(self.un_adj_lut)
        assert n_units is not None
        self._update_sizes_from_n_units(n_units)

        # which units can be proposed for each neighborhood? all of the ones in the LUT
        assert self.max_n_total is not None
        assert self.n_total == self.max_n_total
        proposals = full_proposal_by_neighb(self.un_adj_lut, self.n_total)
        self.proposals = proposals.to(self.device, dtype=torch.long)

    def _update_sizes_from_n_units(self, n_units: int):
        if self.max_n_total is None or self.max_n_explore is None:
            self.max_n_total = n_units
        else:
            self.max_n_total = min(
                self.max_n_candidates * (self.max_n_search + 1) + self.max_n_explore,
                n_units,
            )
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
        x = _read_ix_maybe_mmap(self.x, spike_indices).to(self.device)
        wx, noise_loglik = _whiten_and_noise_score_batch(
            x=x, neighb_ids=neighb_ids, neighb_cov=self.neighb_cov
        )
        # my x is not channels-major as the TruncatedSpikeData's is, and that's
        # what's assumed in the statistics pass.
        del x
        assert self.sample_weights is None
        return SpikeDataBatch(
            batch=spike_indices,
            xt=None,
            candidates=candidates,
            neighborhood_ids=neighb_ids,
            whitenedx=wx,
            noise_logliks=noise_loglik,
            sample_weights=None,
            CmoCooinvx=None,
            candidate_count=None,
        )

    def full_proposal_view(self, un_adj_lut: NeighborhoodLUT):
        self.update_adjacency(n_units=un_adj_lut.b.lut.shape[0], un_adj_lut=un_adj_lut)
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
        xt: Tensor | None,
        whitenedx: Tensor,
        CmoCooinvx: Tensor | None,
        noise_logliks: Tensor,
        neighborhoods: SpikeNeighborhoods,
        sample_weights: Tensor | None,
        search_adj: Literal["top", "explore"] = "top",
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
        device = cast(torch.device, neighborhoods.neighborhood_ids.device)
        super().__init__(
            N=whitenedx.shape[0],
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
            sample_weights=sample_weights,
        )
        assert self.max_n_explore is not None
        assert self.max_n_total is not None
        self._candidates_full = torch.full(
            (self.N, self.max_n_total), -1, device=device
        )
        self.candidates = self._candidates_full[:, : self.n_total]
        assert self.candidates.shape[1] >= n_candidates

        assert noise_logliks.shape == (self.N,)
        assert neighborhoods.neighborhood_ids.shape == (self.N,)
        self.storage_device = whitenedx.device
        assert self.storage_device in (device, torch.device("cpu"))
        for f in (xt, whitenedx, CmoCooinvx, noise_logliks):
            assert f is None or f.device == self.storage_device
            assert (
                self.storage_device == device
                or f is None
                or not f.numel()
                or f.is_pinned()
            )

        self.score_only = xt is None
        assert self.score_only == (CmoCooinvx is None)
        if xt is not None:
            assert whitenedx.shape == (self.N, xt.shape[1] * xt.shape[2])

        self.xt = xt
        self.whitenedx = whitenedx
        self.CmoCooinvx = CmoCooinvx
        self.noise_logliks = noise_logliks
        self.dense_slice_size_per_unit = dense_slice_size_per_unit
        self.search_adj = search_adj

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
        *,
        n_candidates: int,
        n_search: int,
        n_explore: int,
        max_n_candidates: int,
        max_n_search: int,
        max_n_explore: int,
        dense_slice_size_per_unit: int,
        labels: Tensor,
        x: Tensor,
        sample_weights: Tensor | None,
        neighborhoods: SpikeNeighborhoods,
        neighb_cov: NeighborhoodCovariance,
        search_adj: Literal["top", "explore"] = "top",
        seed: int | np.random.Generator = 0,
        explore_neighb_steps: int = 1,
        neighb_overlap: float = 0.75,
        batch_size: int = 128,
        allow_gaps=False,
        score_only: bool = False,
        xt: Tensor | None = None,
        whitenedx: Tensor | None = None,
        CmoCooinvx: Tensor | None = None,
        noise_logliks: Tensor | None = None,
    ) -> Self:
        assert len(x) == len(neighborhoods.b.neighborhood_ids)
        if score_only:
            assert xt is None
            assert whitenedx is None
            assert CmoCooinvx is None
            assert noise_logliks is None
            whitenedx, noise_logliks = _whiten_and_noise_score(
                x=x, neighborhoods=neighborhoods, neighb_cov=neighb_cov
            )
        elif xt is None:
            xt, whitenedx, CmoCooinvx, noise_logliks = _whiten_impute_and_noise_score(
                x=x, neighborhoods=neighborhoods, neighb_cov=neighb_cov
            )
        else:
            assert whitenedx is not None
            assert CmoCooinvx is not None
            assert noise_logliks is not None
        self = cls(
            n_candidates=n_candidates,
            n_search=n_search,
            n_explore=n_explore,
            max_n_candidates=max_n_candidates,
            max_n_search=max_n_search,
            max_n_explore=max_n_explore,
            dense_slice_size_per_unit=dense_slice_size_per_unit,
            xt=xt,
            whitenedx=whitenedx,
            CmoCooinvx=CmoCooinvx,
            noise_logliks=noise_logliks,
            neighborhoods=neighborhoods,
            sample_weights=sample_weights,
            seed=seed,
            batch_size=batch_size,
            explore_neighb_steps=explore_neighb_steps,
            neighb_overlap=neighb_overlap,
            search_adj=search_adj,
        )
        assert self.candidates is not None
        self.candidates[:, 0] = labels
        if pnoid and not allow_gaps:
            luniq = labels.unique().cpu()
            luniq = luniq[luniq >= 0]
            assert luniq.numel() > 0, str(luniq.tolist())
            assert luniq.numel() == luniq.amax() + 1, str(luniq.tolist())
        return self

    def update(
        self,
        new_top_candidates: Tensor | None,
        distances: Tensor,
        *,
        expand_lut: bool = False,
        skip_explore: bool = False,
    ) -> tuple[bool, NeighborhoodLUT]:
        self._update_sizes_from_n_units(distances.shape[0])
        # fill in top spots
        assert self.candidates is not None
        if new_top_candidates is not None:
            self.candidates[:, : self.n_candidates] = new_top_candidates
            self.update_adjacency(n_units=distances.shape[0], expand_lut=expand_lut)

        lut_padded = F.pad(
            self.un_adj_lut.b.lut,
            (0, 0, 0, 1),
            value=self.un_adj_lut.b.unit_ids.shape[0],
        ).long()
        top_lut_ixs = lut_padded[
            self.candidates[:, : self.n_candidates], self.neighborhood_ids[:, None]
        ]

        # fill in search neighbors with topk of distances, constrained by adj
        if self.search_adj == "top":
            search_adj = self.un_adj
        elif self.search_adj == "explore":
            search_adj = self.explore_adj
        else:
            panic(self.search_adj)
        search_sets = candidate_search_sets(
            distances, self.un_adj_lut, search_adj, self.n_search
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
        if self.n_explore and skip_explore:
            self.candidates[:, self.explore_slice] = -1
        elif self.n_explore:
            assert self.rg is not None
            p, inds = _get_explore_sampling_data(
                un_adj_lut=self.un_adj_lut,
                explore_adj=self.explore_adj,
                search_sets=search_sets,
                n_explore=self.n_explore,
            )
            assert (
                p.shape[0] == inds.shape[0] == self.un_adj_lut.b.unit_ids.shape[0] + 1
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
        lut = NeighborhoodLUT.from_candidates_and_neighborhoods(
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
        assert self.candidates is not None
        self.candidates.fill_(-1)
        self.batch_candidate_counts.zero_()

    def batch(
        self, spike_indices: Tensor | slice, batch_index: int | None = None
    ) -> SpikeDataBatch:
        assert self.candidates is not None
        candidates = self.candidates[spike_indices]
        if self.sample_weights is None:
            sample_weights = None
        else:
            sample_weights = self.sample_weights[spike_indices]
        xt, whitenedx, CmoCooinvx, noise_logliks = self.features(spike_indices)
        return SpikeDataBatch(
            batch=spike_indices,
            neighborhood_ids=self.neighborhood_ids[spike_indices],
            xt=xt,
            candidates=candidates,
            CmoCooinvx=CmoCooinvx,
            whitenedx=whitenedx,
            noise_logliks=noise_logliks,
            sample_weights=sample_weights,
            candidate_count=int(self.batch_candidate_counts[batch_index]),
        )

    def features(
        self, spike_indices: Tensor | slice
    ) -> tuple[Tensor | None, Tensor, Tensor | None, Tensor]:
        """xt, whitenedx, CmoCooinvx, noise_logliks -> device"""
        if self.storage_device == self.device:
            return (
                None if self.xt is None else self.xt[spike_indices],
                self.whitenedx[spike_indices],
                None if self.CmoCooinvx is None else self.CmoCooinvx[spike_indices],
                self.noise_logliks[spike_indices],
            )
        if torch.is_tensor(spike_indices):
            spike_indices = spike_indices.cpu()
        dev = self.device
        if self.xt is None:
            xt = None
        else:
            xt = self.xt[spike_indices].to(dev, non_blocking=True)
        if self.CmoCooinvx is None:
            CmoCooinvx = None
        else:
            CmoCooinvx = self.CmoCooinvx[spike_indices].to(dev, non_blocking=True)
        return (
            xt,
            self.whitenedx[spike_indices].to(dev, non_blocking=True),
            CmoCooinvx,
            self.noise_logliks[spike_indices].to(dev, non_blocking=True),
        )

    def dense_slice(self, spike_indices: Tensor) -> DenseSpikeData:
        if self.sample_weights is None:
            sample_weights = None
        else:
            sample_weights = self.sample_weights[spike_indices]
        xt, whitenedx, CmoCooinvx, noise_logliks = self.features(spike_indices)
        return DenseSpikeData(
            indices=spike_indices,
            neighborhoods=self.neighborhoods,
            neighborhood_ids=self.neighborhood_ids[spike_indices],
            neighb_supset=self.neighb_supset,
            xt=xt,
            whitenedx=whitenedx,
            CmoCooinvx=CmoCooinvx,
            noise_logliks=noise_logliks,
            sample_weights=sample_weights,
            batch_size=self.batch_size,
        )

    def dense_slice_by_unit(
        self,
        unit_ids: Tensor | int | None,
        gen: torch.Generator,
        labels: Tensor | None,
        min_count: int = 0,
        min_per_neighborhood: int = 0,
        min_per_unit: int = 0,
    ) -> DenseSpikeData | None:
        assert self.candidates is not None
        assert unit_ids is not None
        unit_ids = torch.as_tensor(unit_ids, device=self.candidates.device)
        n_ids = unit_ids.numel()
        if labels is None:
            labels = self.candidates[:, 0]
        else:
            assert labels.shape == self.candidates.shape[:1], (
                f"{labels.shape=} {self.candidates.shape=}"
            )
            labels = labels.to(device=unit_ids.device)
        if unit_ids.ndim == 0:
            mask = labels == unit_ids
        else:
            mask = torch.isin(labels, unit_ids)

        (ixs,) = mask.nonzero(as_tuple=True)
        if min_count and ixs.numel() < min_count:
            return None
        n_target = self.dense_slice_size_per_unit * n_ids
        nixs = ixs.numel()
        assert not (min_per_neighborhood and min_per_unit)
        if nixs > n_target and min_per_neighborhood:
            ixs = stratified_subsample(
                ixs,
                self.neighborhood_ids[ixs],
                n_groups=self.neighborhoods.n_neighborhoods,
                n_target=n_target,
                min_per_group=min_per_neighborhood,
                gen=gen,
            )
        elif nixs > n_target and min_per_unit:
            ixs = stratified_subsample(
                ixs,
                labels[ixs],
                n_groups=int(unit_ids.max()) + 1,
                n_target=n_target,
                min_per_group=min_per_unit,
                gen=gen,
            )
        elif nixs > n_target:
            perm = torch.randperm(nixs, generator=gen, device=ixs.device)
            ixs = ixs[perm[:n_target]]
            ixs = torch.msort(ixs)

        return self.dense_slice(ixs)

    def remap(
        self,
        remapping: UnitRemapping,
        distances: Tensor | None,
        *,
        allow_uncovered: bool = False,
    ) -> tuple[NeighborhoodLUT | None, int]:
        """Re-map my top candidate labels and re-do LUTs, search, explore."""
        n_units_orig = remapping.mapping.shape[0]
        ids_new = remapping.mapping.unique()
        ids_new = ids_new[ids_new >= 0]
        n_units_new = ids_new.shape[0]
        _n_units_new = ids_new.amax() + 1
        assert _n_units_new == n_units_new
        assert self.candidates is not None
        assert n_units_new <= n_units_orig

        if distances is not None:
            assert distances.shape[0] == distances.shape[1]
            assert distances.shape[0] == n_units_new
            assert distances.shape[0] <= n_units_orig
            assert remapping.mapping.amax() + 1 == distances.shape[0]
            assert _n_units_new == distances.shape[0]

        self._update_sizes_from_n_units(n_units_new)

        # extra -1 for the -1 candidates to index into
        mapping = remapping.padded_map().to(device=self.candidates.device)
        assert mapping.shape == (n_units_orig + 1,)
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
        self.candidates[:, self.n_candidates :] = -1
        if pnoid and distances is not None:
            _max_cand = self.candidates.amax()
            assert _max_cand < distances.shape[0], f"{_max_cand} {distances.shape}"
        self.update_adjacency(n_units=n_units_new)
        n_new = self.ensure_coverage(
            allow_new_units=not allow_uncovered,
            allow_uncovered=allow_uncovered,
            max_steps=0 if allow_uncovered else None,
            next_label=n_units_new,
        )
        if allow_uncovered:
            assert n_new == 0
        elif distances is not None:
            d = distances.new_full(
                (n_units_new + n_new, n_units_new + n_new), torch.inf
            )
            d[:n_units_new, :n_units_new] = distances
            d[n_units_new:, n_units_new:].fill_diagonal_(0.0)
            distances = d
        self.update_adjacency(n_units=n_units_new + n_new)
        if pnoid and distances is not None:
            _max_cand = self.candidates.amax()
            assert _max_cand < distances.shape[0], f"{_max_cand} {distances.shape}"
        if distances is not None:
            _, lut = self.update(new_top_candidates=None, distances=distances)
        else:
            lut = self.un_adj_lut
        return lut, n_new

    def update_from_split(
        self,
        unit_mask: Tensor,
        split_mask: Tensor,
        split_labels: Tensor,
        distances: Tensor,
    ) -> tuple[NeighborhoodLUT, int]:
        """Erase candidates for spikes in the split, replace labels, re-bootstrap.

        Neighborhoods left uncovered by the split get new units, and their count is returned.
        """
        assert self.candidates is not None
        self.candidates[:, 0].masked_fill_(unit_mask, -1)
        (split_ix,) = split_mask.nonzero(as_tuple=True)
        self.candidates[split_ix, 0] = split_labels[split_ix]

        n_units = distances.shape[0]
        n_new = self.ensure_coverage(allow_new_units=True, next_label=n_units)
        if n_new:
            d = distances.new_full((n_units + n_new, n_units + n_new), torch.inf)
            d[:n_units, :n_units] = distances
            d[n_units:, n_units:].fill_diagonal_(0.0)
            distances = d

        # have to do a full bootstrap, bc it's hard to figure out what to do with
        # spikes whose candidates contain the units that were split. this way, the
        # lut invariants are maintained, and at least the top labels are the same.
        return self.bootstrap_candidates(distances), n_new

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
            sample_weights=data.sample_weights,
        )
        self.proposals = proposals.to(self.device)
        self.data = data

    @classmethod
    def from_truncated_spike_data(
        cls, data: TruncatedSpikeData, un_adj_lut: NeighborhoodLUT
    ) -> Self:
        proposals = un_adj_lut.full_proposal_candidates(pad_shape_to=data.n_candidates)
        n_proposed = proposals.shape[1]
        n_explore = n_proposed - data.n_candidates
        self = cls(n_explore=n_explore, data=data, proposals=proposals)
        return self

    def update(
        self,
        new_top_candidates: Tensor | None,
        distances: Tensor,
        *,
        expand_lut: bool = False,
        skip_explore: bool = False,
    ) -> tuple[bool, NeighborhoodLUT]:
        raise ValueError("View doesn't update.")

    def update_adjacency(
        self,
        *,
        n_units: int | None,
        un_adj_lut: NeighborhoodLUT | None = None,
        expand_lut: bool = False,
    ):
        raise ValueError("View doesn't update.")

    def batch(
        self, spike_indices: Tensor | slice, batch_index: int | None = None
    ) -> SpikeDataBatch:
        neighb_ids = self.neighborhood_ids[spike_indices]
        candidates = self.proposals[neighb_ids]
        if self.sample_weights is None:
            sample_weights = None
        else:
            sample_weights = self.sample_weights[spike_indices]
        xt, whitenedx, CmoCooinvx, noise_logliks = self.data.features(spike_indices)
        return SpikeDataBatch(
            batch=spike_indices,
            neighborhood_ids=self.neighborhood_ids[spike_indices],
            xt=xt,
            candidates=candidates,
            CmoCooinvx=CmoCooinvx,
            whitenedx=whitenedx,
            noise_logliks=noise_logliks,
            sample_weights=sample_weights,
            candidate_count=None,
        )


class BaseMixtureModel(BModule):
    noise_log_prop: Tensor

    def __init__(
        self,
        *,
        unit_ids: Tensor,
        signal_rank: int,
        neighb_cov: NeighborhoodCovariance,
        noise: KroneckerNoise | None,
        erp: NeighborhoodFiller | None,
        p: TMMParams,
    ):
        super().__init__()
        self.p = p
        self.unit_ids = unit_ids
        self.n_units = unit_ids.shape[0]
        self.signal_rank = signal_rank
        self.erp: NeighborhoodFiller | None = erp
        self.neighb_cov = neighb_cov
        self.noise: KroneckerNoise | None = noise

    # -- subclasses implement

    @property
    def centroids(self) -> Tensor:
        """Interface used by unit_distance_matrix."""
        raise NotImplementedError

    def non_noise_log_proportion(self):
        raise NotImplementedError

    def score(
        self,
        data: DenseSpikeData,
        *,
        skip_noise: bool = False,
        allow_blanks: bool = False,
    ) -> Scores:
        raise NotImplementedError

    # -- shared logic

    def unit_distance_matrix(
        self, distance_kind: ComponentDistanceMetric | None = None
    ) -> Tensor:
        """Pairwise distance matrix

        These should return something with zero on the diagonal. Some centroids
        can have 0 norm, and they should have inf distance to other units.
        """
        if distance_kind is None:
            distance_kind = self.p.distance_kind

        if self.p.whiten_dist:
            assert self.noise is not None
            x = self.centroids
            x = x.view(len(x), self.noise.rank, self.noise.n_channels)
            x = self.noise.whiten(x).reshape(len(x), -1)
        else:
            x = self.centroids

        if distance_kind == "cosine":
            d = cosine_distance(x)
        elif distance_kind == "normeuc":
            d = normeuc_distance(x)
        elif distance_kind == "scaled_normeuc" or distance_kind == "euc":
            # no component euc yet: euc falls back to scaled_normeuc
            d = scaled_normeuc_distance(x, *self.p.scale_dist_args)
        else:
            panic(distance_kind)

        if pnoid:
            assert not d.isnan().any()

        return d


class TruncatedMixtureModel(BaseMixtureModel):
    LP_MIN = -1000.0
    MAX_BATCH_SPIKES = 2**18

    def __init__(
        self,
        *,
        log_proportions: Tensor,
        means: Tensor,
        bases: Tensor | None,
        noise_log_prop: Tensor | None,
        neighb_cov: NeighborhoodCovariance,
        noise: KroneckerNoise,
        erp: NeighborhoodFiller | None,
        lut_puff: float = 1.5,
        known_lut: NeighborhoodLUT | None = None,
        seed: int | np.random.Generator | torch.Generator = 0,
        p: TMMParams,
    ):
        super().__init__(
            unit_ids=torch.arange(means.shape[0], device=log_proportions.device),
            signal_rank=bases.shape[2] if bases is not None else 0,
            erp=erp,
            neighb_cov=neighb_cov,
            noise=noise,
            p=p,
        )
        self.lut_puff = lut_puff
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
        self.eps = torch.tensor(tiny, device=means.device)

        # needs to be initialized before doing anything serious
        # see bootstrapping stage of initialize_and_bootstrap_tmm
        self.lut_params: LUTParams | None = None
        if known_lut is None:
            self.lut = NeighborhoodLUT(
                unit_ids=torch.arange(0),
                neighb_ids=torch.arange(0),
                lut=torch.arange(0)[None],
            )
        else:
            self.lut = known_lut

        if bases is not None:
            assert bases.shape[0] == self.n_units
            assert bases.shape[2] == means.shape[1]
            self.signal_rank = bases.shape[1]
        else:
            self.signal_rank = 0
        self.rg = spawn_torch_rg(seed, device=means.device)

    @classmethod
    def from_state_dict(
        cls,
        motion: MotionInfo,
        state_dict,
        feature_rank: int,
        neighborhoods: SpikeNeighborhoods,
        refinement_cfg: RefinementConfig,
    ):
        noise = KroneckerNoise(
            temporal_cov=state_dict["noise.temporal_cov"],
            spatial_cov=state_dict["noise.spatial_cov"],
            zero_radius=refinement_cfg.cov_radius,
        )
        assert noise.rank == feature_rank
        assert noise.n_channels == motion.rgeom.shape[0]
        neighb_cov = NeighborhoodCovariance.from_noise_and_neighborhoods(
            prgeom=pad_geom(motion.rgeom),
            noise=noise,
            neighborhoods=neighborhoods,
            neighb_overlap=refinement_cfg.neighb_overlap,
            huber_quantile=noise_huber_quantile(refinement_cfg),
        )
        lut_dict = {
            k.removeprefix("lut."): v
            for k, v in state_dict.items()
            if k.startswith("lut.")
        }
        lut = NeighborhoodLUT(**lut_dict)
        return cls(
            log_proportions=state_dict["log_proportions"],
            means=state_dict["means"],
            bases=state_dict["bases"],
            noise_log_prop=state_dict["noise_log_prop"],
            neighb_cov=neighb_cov,
            noise=noise,
            erp=None,
            lut_puff=1.0,
            known_lut=lut,
            p=TMMParams.from_refinement_cfg(refinement_cfg),
        )

    def change_rank(
        self, new_signal_rank: int, train_data: TruncatedSpikeData, train_scores: Scores
    ):
        assert self.get_optional_buffer("bases") is None
        assert self.signal_rank == 0
        assert self.erp is not None
        assert self.noise is not None
        if new_signal_rank == 0:
            return
        _, _, _means, bases = initialize_parameters_by_unit(
            scores=train_scores,
            data=train_data,
            signal_rank=new_signal_rank,
            noise=self.noise,
            erp=self.erp,
            gen=self.rg,
            min_channel_count=self.p.min_channel_count,
            prior_pseudocount=self.p.prior_pseudocount,
            rank0_model=self,
        )
        assert _means is None
        assert bases is not None
        assert bases.shape[0] == self.n_units
        assert bases.shape[1] == new_signal_rank
        assert bases.shape[2] == self.b.means.shape[1]
        self.del_none_buffer("bases")
        self.register_buffer("bases", bases)
        self.signal_rank = new_signal_rank
        self.lut_params = None
        gc.collect()
        self.update_lut(self.lut)

    @classmethod
    def initialize_from_data_with_labels(
        cls,
        *,
        neighb_cov: NeighborhoodCovariance,
        signal_rank: int,
        data: TruncatedSpikeData,
        noise: KroneckerNoise,
        erp: NeighborhoodFiller,
        seed: int | np.random.Generator,
        p: TMMParams,
    ) -> Self:
        """Constructor for the full mixture model, called by from_config()"""
        rg = spawn_torch_rg(seed, device=neighb_cov.b.obs_ix.device)
        log_props, noise_log_prop, means, bases = initialize_parameters_by_unit(
            data=data,
            signal_rank=signal_rank,
            noise=noise,
            erp=erp,
            gen=rg,
            min_channel_count=p.min_channel_count,
            initial_basis_shrinkage=p.initial_basis_shrinkage,
            prior_pseudocount=p.prior_pseudocount,
            puff=p.split_k,
        )
        assert means is not None
        if pnoid:
            assert log_props.isfinite().all()
            assert noise_log_prop.isfinite().all()
            assert means.isfinite().all()
        return cls(
            neighb_cov=neighb_cov,
            erp=erp,
            log_proportions=log_props,
            means=means,
            bases=bases,
            noise_log_prop=noise_log_prop,
            seed=rg,
            noise=noise,
            p=p,
        )

    @classmethod
    def from_config(
        cls,
        *,
        noise: KroneckerNoise,
        erp: NeighborhoodFiller,
        neighb_cov: NeighborhoodCovariance,
        train_data: TruncatedSpikeData,
        seed: int | np.random.Generator,
        refinement_cfg: RefinementConfig,
    ) -> Self:
        assert refinement_cfg.distance_metric in get_args(ComponentDistanceMetric)

        # TODO: remove all unused refinement_cfg parameters
        p = TMMParams.from_refinement_cfg(refinement_cfg)
        if refinement_cfg.initialize_at_rank_0:
            signal_rank = 0
        else:
            signal_rank = refinement_cfg.signal_rank

        return cls.initialize_from_data_with_labels(
            noise=noise,
            erp=erp,
            neighb_cov=neighb_cov,
            seed=seed,
            data=train_data,
            signal_rank=signal_rank,
            p=p,
        )

    @classmethod
    def initialize_from_dense_data_and_responsibilities(
        cls,
        *,
        signal_rank: int,
        erp: NeighborhoodFiller,
        data: ComponentBlockData,
        responsibilities: Tensor,
        noise: KroneckerNoise,
        neighb_cov: NeighborhoodCovariance,
        total_log_proportions: Tensor,
        noise_log_prop: Tensor | float = -torch.inf,
        p: TMMParams,
        min_channel_count: int | None = None,
        fix_responsibilities: bool | None = None,
    ) -> ComponentBlockFitResult:
        """Fit units with fixed label posterior

        Used to construct hypothetical models:
         - Fitting split units based on k-means weights
         - Fitting combinations of split or main model units to evaluate
           the brute force merge criterion

        Units whose initialization fails are dropped, so initial data's units
        will differ from return which refers to fitted units.

        total_log_proportions is per-block.
        """
        if min_channel_count is None:
            min_channel_count = p.min_channel_count
        n_blocks, width = data.block_unit_ids.shape
        assert responsibilities.shape == (data.N, width)
        assert total_log_proportions.shape == (n_blocks,)
        proposed = data.block_unit_ids >= 0
        valid = torch.zeros_like(proposed)

        means = []
        bases = []
        log_proportions = []
        kept_mask = []
        for b in range(n_blocks):
            in_block = data.block_slice(b)
            (proposed_ixs,) = proposed[b].nonzero(as_tuple=True)
            block_data = data.data.slice(in_block)
            block_resp = responsibilities[in_block][:, proposed_ixs]
            channel_mask, block_means, block_bases, chan_coverage = (
                initialize_params_from_dense_data(
                    data=block_data,
                    rank=signal_rank,
                    erp=erp,
                    min_channel_count=min_channel_count,
                    noise=noise,
                    weights=block_resp,
                    prior_pseudocount=p.prior_pseudocount,
                )
            )
            assert chan_coverage is not None
            block_valid = channel_mask.any(dim=1)
            valid[b, proposed_ixs[block_valid]] = True
            means.append(block_means[block_valid])
            if signal_rank:
                assert block_bases is not None
                bases.append(block_bases[block_valid])
            block_resp = block_resp[:, block_valid]

            # deliberately not normalizing the proportions here
            # these were historically always fixed from the get
            # normalizing would be bad when we have overlapping subsets here.
            block_lp = spike_mean(block_resp, block_data.sample_weights).log_()
            log_proportions.append(block_lp + total_log_proportions[b])

            # subset data to coverage
            block_keep = (block_resp > 0).logical_and_(chan_coverage[:, block_valid])
            kept_mask.append(block_keep.any(dim=1))
        log_proportions = torch.concatenate(log_proportions)
        kept_mask = torch.concatenate(kept_mask)
        means = torch.concatenate(means)
        K = means.shape[0]
        assert log_proportions.shape == (K,)
        means = means.view(K, -1)
        if signal_rank:
            bases = torch.concatenate(bases).view(K, signal_rank, -1)
        else:
            bases = None

        self = cls(
            log_proportions=log_proportions,
            means=means,
            bases=bases,
            noise_log_prop=torch.as_tensor(noise_log_prop),
            neighb_cov=neighb_cov,
            noise=noise,
            erp=erp,
            p=p,
        )

        valid_counts = valid.sum(dim=1)
        block_unit_ids = contiguous_block_unit_ids(valid_counts)
        valid_ixs = torch.argsort(
            valid.logical_not().to(torch.int8), dim=1, stable=True
        )
        valid_ixs = valid_ixs[:, : block_unit_ids.shape[1]]
        valid_ixs = valid_ixs.where(block_unit_ids >= 0, width)
        responsibilities = F.pad(responsibilities, (0, 1))
        responsibilities = responsibilities.gather(1, valid_ixs[data.block_ids])
        data = data.with_block_unit_ids(block_unit_ids)
        if not kept_mask.all():
            (kept_ixs,) = kept_mask.nonzero(as_tuple=True)
            responsibilities = responsibilities[kept_ixs]
            data = data.slice(kept_ixs)

        # get lut, which will never change
        candidates = torch.where(
            responsibilities > 0, block_unit_ids[data.block_ids], -1
        )
        if pnoid:
            assert (responsibilities > 0).any(dim=1).all()
            assert (candidates >= 0).any(dim=1).all()
        lut = NeighborhoodLUT.from_candidates_and_neighborhoods(
            candidates=candidates,
            neighborhoods=data.data.neighborhoods,
            neighborhood_ids=data.data.neighborhood_ids,
            neighb_supset=data.data.neighb_supset,
            n_units=self.n_units,
        )
        self.update_lut(lut)

        # run some em steps
        responsibilities, block_elbos = self.dense_em(
            data=data,
            responsibilities=responsibilities,
            fix_responsibilities=fix_responsibilities,
        )
        return ComponentBlockFitResult(
            model=self,
            data=data,
            responsibilities=responsibilities,
            valid=valid,
            kept_mask=kept_mask,
            block_elbos=block_elbos,
        )

    def em(
        self,
        data: TruncatedSpikeData,
        *,
        show_progress: int = 1,
        allow_blanks: bool = False,
        min_iters: int | None = None,
    ):
        assert self.lut_params is not None
        if show_progress:
            iters = progrange(self.p.em_iters, desc="EM")
        else:
            iters = range(self.p.em_iters)
        if min_iters is None:
            min_iters = self.p.min_em_iters
        elbos = []
        for j in iters:
            if pnoid:
                self.lut_params.check()
            if j and self.p.full_proposal_every and not j % self.p.full_proposal_every:
                step_data = data.full_proposal_view(self.lut)
            else:
                step_data = data
            eres = self.e_step(
                step_data, show_progress=show_progress > 2, allow_blanks=allow_blanks
            )
            elb = eres.elbo.cpu().item()
            if not math.isfinite(elb):
                self._bad_numbers_error(elb, eres.stats)
            elbos.append(elb)
            if show_progress:
                iters.set_description(f"EM(elbo={elb:.3f})")  # type: ignore
            self.m_step(eres.stats)
            lut_changed, lut = data.update(eres.candidates, self.unit_distance_matrix())
            del lut_changed  # doesn't matter if it did or not, my parameters changed
            self.update_lut(lut)
            if j > min_iters and elbos[-1] - elbos[-2] < self.p.elbo_atol:
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
                f"{self.p.em_iters} iterations."
            )
        assert not self.p.em_iters or elbos[-1] > elbos[0] - 1e-3
        return EMResult(elbos=elbos)

    def e_step(
        self,
        data: TruncatedSpikeData | FullProposalDataView,
        *,
        show_progress: bool = False,
        allow_blanks: bool = False,
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
        elbo_sum = torch.zeros((), device=self.b.means.device, dtype=torch.double)
        elbo_weight = torch.zeros_like(elbo_sum)
        for batch in data.batches(show_progress=show_progress, desc="E"):
            batch_scores = self.score_batch(
                batch, n_candidates=data.n_candidates, allow_blanks=allow_blanks
            )
            batch_stats = self.estep_stats_batch(
                batch, batch_scores, allow_blanks=allow_blanks
            )
            stats.combine(batch_stats, eps=self.eps)
            _add_spike_elbos_(elbo_sum, batch_scores, batch.sample_weights)
            if batch.sample_weights is None:
                elbo_weight += batch.whitenedx.shape[0]
            else:
                elbo_weight += batch.sample_weights.sum()
            candidates[batch.batch] = batch_scores.candidates
        if self.p.prior_pseudocount:
            elbo_sum += _unit_log_priors(
                means=self.b.means,
                bases=self.b.bases,
                neighb_cov=self.neighb_cov,
                prior_pseudocount=self.p.prior_pseudocount,
            ).sum()
        _finalize_e_stats(
            means=self.b.means,
            bases=self.b.bases,
            stats=stats,
            lut=self.lut,
            lut_params=self.lut_params,
            neighb_cov=self.neighb_cov,
        )
        return TruncatedEStepResult(
            candidates=candidates, stats=stats, elbo=elbo_sum / elbo_weight
        )

    def m_step(
        self,
        stats: SufficientStatistics,
        *,
        skip_proportions: bool = False,
        active_units: Tensor | None = None,
    ):
        if not skip_proportions:
            assert stats.noise_N is not None
            lp = torch.concatenate([stats.resp_N, stats.noise_N[None]])
            lp = F.log_softmax(lp.log_(), dim=0).clamp_(min=self.LP_MIN)
            self.b.log_proportions.copy_(lp[:-1])
            self.b.noise_log_prop.fill_(lp[-1])
            if pnoid:
                assert self.b.log_proportions.isfinite().all()
                assert self.b.noise_log_prop.isfinite().all()
        else:
            assert stats.noise_N is None
            lp = None

        R = stats.R
        N = stats.N
        if active_units is not None:
            R = R[active_units]
            N = N[active_units]

        # rank 0 case: mean only
        if self.signal_rank == 0:
            assert R.shape[1] == 1
            means = R[:, 0]
            if self.p.prior_pseudocount:
                shrink = N / (N + self.p.prior_pseudocount)
                means = means * shrink[:, None].to(R)
            if active_units is None:
                self.b.means.copy_(means)
            else:
                self.b.means[active_units] = means
            if pnoid:
                assert self.b.means.isfinite().all()
            return

        # solve mean and basis together
        U = _get_u_from_ulut(self.lut, stats)
        if active_units is not None:
            U = U[active_units]
        if self.p.prior_pseudocount:
            tikh = self.p.prior_pseudocount / N.clamp(min=self.eps)
            U.diagonal(dim1=-2, dim2=-1).add_(tikh[:, None])
        Uchol, _ = torch.linalg.cholesky_ex(U)
        soln = torch.cholesky_solve(R, Uchol)
        if active_units is None:
            self.b.means.copy_(soln[:, -1])
            self.b.bases.copy_(soln[:, :-1])
        else:
            self.b.means[active_units] = soln[:, -1]
            self.b.bases[active_units] = soln[:, :-1]

        if pnoid:
            assert soln.isfinite().all()
        if not skip_proportions:
            assert lp is not None
            assert lp.isfinite().all()

    def _dense_batches(
        self, data: ComponentBlockData
    ) -> list[tuple[SpikeDataBatch, tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]]:
        # make use of the fixed sparsity structure, avoiding implicit nonzero()s below
        batches = []
        for batch in data.to_batches(self.lut):
            spike_ixs, candidate_ixs, unit_ixs, neighb_ixs = _sparsify_candidates(
                batch.candidates,
                batch.neighborhood_ids,
                static_size=batch.candidate_count,
            )
            lut_ixs = self.lut.b.lut[unit_ixs, neighb_ixs]
            batches.append(
                (batch, (spike_ixs, candidate_ixs, unit_ixs, neighb_ixs, lut_ixs))
            )
        return batches

    def dense_em(
        self,
        data: ComponentBlockData,
        responsibilities: Tensor,
        *,
        fix_responsibilities: bool | None = None,
        debug: bool = False,
    ) -> tuple[Tensor, np.ndarray]:
        assert self.lut_params is not None
        if fix_responsibilities is None:
            fix_responsibilities = self.p.fix_responsibilities
        if not fix_responsibilities:
            responsibilities = responsibilities.clone()
        assert responsibilities.shape == (data.N, data.width)
        device = data.block_ids.device
        block_elbos = []
        j = -1
        block_elbo_sums = torch.zeros(data.n_blocks, device=device, dtype=torch.double)
        spike_weights = data.data.sample_weights
        if spike_weights is None:
            spike_weights = block_elbo_sums.new_ones(data.N)
        block_weights = torch.zeros_like(block_elbo_sums)
        block_weights.index_add_(0, data.block_ids, spike_weights.double())
        block_weights = block_weights.where(block_weights > 0, 1.0)
        valid = (data.block_unit_ids >= 0).nonzero(as_tuple=True)
        unit_block_ids = torch.full_like(self.unit_ids, -1).to(device)
        unit_block_ids[data.block_unit_ids[valid]] = valid[0]
        assert (unit_block_ids >= 0).all()
        unit_block_ids_host = unit_block_ids.numpy(force=True)

        # track active set, and stop converged guys early
        active_blocks = np.ones(data.n_blocks, dtype=bool)
        active_units = None
        active_ixs = None
        active_data = data
        active_resp = responsibilities
        batches = self._dense_batches(active_data)

        stats = SufficientStatistics.zeros(
            n_units=self.n_units,
            n_lut=self.lut_params.n_lut,
            n_channels=self.neighb_cov.n_channels,
            feature_rank=self.neighb_cov.feat_rank,
            signal_rank=self.signal_rank,
            device=self.b.means.device,
            skip_noise=True,
        )
        if self.p.robust_strategy == "huber":
            mstep_weights_need_scores = True
        elif self.p.robust_strategy == "none":
            mstep_weights_need_scores = False
        else:
            panic(self.p.robust_strategy)
        for j in range(self.p.em_iters):
            check_elbo = j >= self.p.criterion_em_iters
            if check_elbo:
                block_elbo_sums.zero_()
            for batch, spixs in batches:
                spike_ixs, candidate_ixs, unit_ixs, neighb_ixs, lut_ixs = spixs
                bresp = active_resp[batch.batch]
                if not fix_responsibilities or check_elbo or mstep_weights_need_scores:
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
                    if not fix_responsibilities:
                        # e step. candidate order is fixed, so no topk needed.
                        bresp = batch_scores.log_liks.softmax(dim=1)
                        active_resp[batch.batch] = bresp
                        batch_scores.responsibilities = bresp
                else:
                    # too soon to test convergence, no need for log liks
                    batch_scores = Scores(
                        log_liks=bresp,  # unused, just to fill the field.
                        responsibilities=bresp,
                        candidates=batch.candidates,
                        duties=None,
                    )
                if check_elbo:
                    _add_spike_elbos_(
                        block_elbo_sums,
                        batch_scores,
                        batch.sample_weights,
                        block_ids=active_data.block_ids[batch.batch],
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
            if check_elbo and self.p.prior_pseudocount:
                unit_log_priors = _unit_log_priors(
                    means=self.b.means,
                    bases=self.b.bases,
                    neighb_cov=self.neighb_cov,
                    prior_pseudocount=self.p.prior_pseudocount,
                )
                block_elbo_sums.index_add_(0, unit_block_ids, unit_log_priors)
            _finalize_e_stats(
                means=self.b.means,
                bases=self.b.bases,
                stats=stats,
                lut=self.lut,
                lut_params=self.lut_params,
                neighb_cov=self.neighb_cov,
            )
            self.m_step(stats, skip_proportions=True, active_units=active_units)
            self.update_lut(self.lut)

            if check_elbo:
                block_elbo = block_elbo_sums.div(block_weights).cpu().numpy()
                if block_elbos:
                    block_elbo = np.where(active_blocks, block_elbo, block_elbos[-1])
                assert np.isfinite(block_elbo).all()
                block_elbos.append(block_elbo)
            if len(block_elbos) >= 2:
                converged = block_elbos[-1] - block_elbos[-2] < self.p.elbo_atol
                if converged[active_blocks].any():
                    active_blocks = np.logical_and(active_blocks, ~converged)
                    if not active_blocks.any():
                        break
                    active_units = np.flatnonzero(active_blocks[unit_block_ids_host])
                    active_units = torch.from_numpy(active_units).to(device)
                    n_active_spikes = np.diff(active_data.row_indptr)[active_blocks].sum()
                    if n_active_spikes <= 0.75 * active_data.N:
                        if not fix_responsibilities and active_ixs is not None:
                            responsibilities[active_ixs] = active_resp
                        active_data, kept = active_data.keep_blocks(active_blocks)
                        active_ixs = kept if active_ixs is None else active_ixs[kept]
                        active_resp = responsibilities[active_ixs]
                        batches = self._dense_batches(active_data)
            stats.reset_(
                n_units=self.n_units,
                n_channels=self.neighb_cov.n_channels,
                feature_rank=self.neighb_cov.feat_rank,
                signal_rank=self.signal_rank,
            )
        if not fix_responsibilities and active_ixs is not None:
            responsibilities[active_ixs] = active_resp

        if block_elbos:
            elbo_history = np.stack(block_elbos, axis=0)
        else:
            elbo_history = np.zeros((0, data.n_blocks))
        if debug and block_elbos:
            begend = elbo_history[-1] - elbo_history[0]
            delbos = np.diff(elbo_history, axis=0)
            smalldif = delbos.min(initial=0.0)
            bigdiff = delbos.max(initial=0.0)
            logger.dartsortdebug(
                f"Fixed fit elbo end-start in [{begend.min():0.4f}, {begend.max():0.4f}] "
                f"over {data.n_blocks} blocks and {j + 1} iterations, "
                f"biggest and smallest diffs {bigdiff:0.4f} and {smalldif:0.4f}."
            )

        # Since responsibilities are fixed, it's possible (I think) for the
        # support of the likelihoods to drift away slightly from the weights',
        # and this can lead to slightly negative elbo changes. giving a little
        # more leeway than in em() here for when to panic about it.
        # in fact, I'm going to comment this assertion out, but leaving the
        # comment here.
        # assert (len(elbos) < 2) or (elbos[-1] > elbos[0] - 5e-2)

        return responsibilities, elbo_history

    def score(
        self,
        data: DenseSpikeData,
        *,
        skip_noise: bool = False,
        allow_blanks: bool = False,
        skip_responsibility: bool = True,
    ) -> Scores:
        return self.score_blocks(
            ComponentBlockData.single(data, self.unit_ids),
            skip_noise=skip_noise,
            allow_blanks=allow_blanks,
            skip_responsibility=skip_responsibility,
        )

    def score_blocks(
        self,
        data: ComponentBlockData,
        *,
        skip_noise: bool = False,
        allow_blanks: bool = False,
        skip_responsibility: bool = True,
    ) -> Scores:
        scores = []
        for batch in data.to_batches(self.lut):
            batch_scores = self.score_batch(
                batch=batch,
                n_candidates=batch.candidates.shape[1],
                skip_responsibility=skip_responsibility,
                skip_noise=skip_noise,
                allow_blanks=allow_blanks,
            )
            scores.append(batch_scores)
        return concatenate_scores(scores)

    def score_features(
        self,
        features: Tensor,
        candidates: Tensor,
        neighborhood_ids: Tensor,
        n_candidates: int,
        candidate_count: int | None,
    ) -> tuple[Tensor, Scores]:
        x = features.view(len(features), -1)
        wx, noise_logliks = _whiten_and_noise_score_batch(
            x=x, neighb_ids=neighborhood_ids, neighb_cov=self.neighb_cov
        )
        batch = SpikeDataBatch(
            batch=None,
            neighborhood_ids=neighborhood_ids,
            xt=None,
            candidates=candidates,
            whitenedx=wx,
            noise_logliks=noise_logliks,
            sample_weights=None,
            CmoCooinvx=None,
            candidate_count=candidate_count,
        )
        scores = self.score_batch(
            batch=batch, n_candidates=n_candidates, allow_blanks=True
        )
        labels = labels_from_scores_(scores)
        return labels, scores

    def score_batch(
        self,
        batch: SpikeDataBatch,
        *,
        n_candidates: int,
        fixed_responsibilities: Tensor | None = None,
        skip_responsibility: bool = False,
        skip_noise: bool = False,
        spike_ixs: Tensor | None = None,
        candidate_ixs: Tensor | None = None,
        unit_ixs: Tensor | None = None,
        neighb_ixs: Tensor | None = None,
        lut_ixs: Tensor | None = None,
        allow_blanks: bool = False,
        force_rank0: bool = False,
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
            robust_strategy=self.p.robust_strategy,
            spike_ixs=spike_ixs,
            candidate_ixs=candidate_ixs,
            unit_ixs=unit_ixs,
            neighb_ixs=neighb_ixs,
            lut_ixs=lut_ixs,
            static_size=batch.candidate_count,
            allow_blanks=allow_blanks,
            skip_noise=skip_noise,
            force_rank0=force_rank0,
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
        allow_blanks: bool = False,
    ):
        assert self.lut_params is not None
        assert batch.CmoCooinvx is not None
        assert batch.xt is not None
        assert scores.responsibilities is not None
        if spike_ixs is None:
            spike_ixs, candidate_ixs, unit_ixs, neighb_ixs = _sparsify_candidates(
                scores.candidates,
                batch.neighborhood_ids,
                static_size=static_size,
                allow_blanks=allow_blanks,
            )
            lut_ixs = self.lut.b.lut[unit_ixs, neighb_ixs]
        else:
            assert candidate_ixs is not None
            assert unit_ixs is not None
            assert neighb_ixs is not None
            assert lut_ixs is not None
        if self.p.robust_strategy == "huber":
            assert scores.duties is not None
            mstep_weights = scores.duties
        elif self.p.robust_strategy == "none":
            mstep_weights = None
        else:
            panic(self.p.robust_strategy)
        return _stat_pass_batch(
            xt=batch.xt,
            whitenedx=batch.whitenedx,
            CmoCooinvx=batch.CmoCooinvx,
            responsibilities=scores.responsibilities,
            mstep_weights=mstep_weights,
            sample_weights=batch.sample_weights,
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
        force_rank0: bool = False,
    ) -> Scores:
        """Run E steps until candidates converge, holding my LUT fixed.

        TODO: full_proposal_view=True is cheap, converges in one iteration, and
        supports all use cases. It would be best to remove needs_boostrap, and
        maybe even ONLY support full_proposal_view=True. I think it's all I ever
        use in this code.
        """
        # start by telling the data how to search
        distances = self.unit_distance_matrix()
        if full_proposal_view:
            data.update_adjacency(n_units=self.n_units, un_adj_lut=self.lut)
            target_data = data.full_proposal_view(self.lut)
        elif needs_bootstrap:
            # TODO cut this?
            data.erase_candidates()
            lut = data.bootstrap_candidates(distances=distances, un_adj_lut=self.lut)
            assert lut.eq(self.lut)
            # self.update_lut(lut, no_parameter_changes=True)
            target_data = data
        else:
            target_data = data

        # initialize storage for output on data's device
        if data.candidates is None:
            # streaming setting, assumption is that this is too big for device
            dev = torch.device("cpu")
        else:
            dev = data.candidates.device
        if self.p.robust_strategy == "none":
            duties = None
        elif self.p.robust_strategy == "huber":
            duties = torch.full((data.N, data.n_candidates), torch.nan, device=dev)
        else:
            panic(self.p.robust_strategy)
        scores = Scores(
            log_liks=torch.empty((data.N, data.n_candidates + 1), device=dev),
            responsibilities=torch.empty((data.N, data.n_candidates + 1), device=dev),
            candidates=torch.full((data.N, data.n_candidates), -1, device=dev),
            duties=duties,
        )
        assert scores.responsibilities is not None
        assert max_iter >= 1
        if show_progress > 1 and not data.proposal_is_complete:
            iters = progrange(max_iter, desc="SoftAssign")
            show_batch_progress = show_progress > 2 or data.proposal_is_complete
        else:
            iters = range(max_iter)
            show_batch_progress = False
        for it in iters:
            for batch in target_data.batches(show_progress=show_batch_progress):
                batch_scores = self.score_batch(
                    batch,
                    n_candidates=data.n_candidates,
                    allow_blanks=True,
                    force_rank0=force_rank0,
                )
                assert batch_scores.responsibilities is not None
                bix = batch.batch
                scores.candidates[bix] = batch_scores.candidates.to(dev)
                scores.log_liks[bix] = batch_scores.log_liks.to(dev)
                scores.responsibilities[bix] = batch_scores.responsibilities.to(dev)
                if self.p.robust_strategy != "none":
                    assert batch_scores.duties is not None
                    assert scores.duties is not None
                    scores.duties[bix] = batch_scores.duties.to(dev)

            lut_changed, lut = data.update(
                scores.candidates, distances, expand_lut=True, skip_explore=True
            )
            if lut_changed and target_data.proposal_is_complete:
                # it can't actually change.
                assert lut.eq(self.lut)
            else:
                self.update_lut(lut, no_parameter_changes=True)
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

    def update_lut(
        self,
        lut: NeighborhoodLUT,
        *,
        no_parameter_changes: bool = False,
        puff: float | None = None,
    ):
        if no_parameter_changes and self.lut.eq(lut):
            return
        self.lut = lut
        if self.lut_params is None:
            puff = puff if puff is not None else self.lut_puff
            self.lut_params = LUTParams.from_lut(
                self.b.means,
                self.b.bases,
                self.neighb_cov,
                lut,
                huber_quantile=huber_quantile_if_huber(
                    self.p.robust_strategy, self.p.huber_quantile
                ),
                puff=puff,
            )
        else:
            self.lut_params.update(self.b.means, self.b.bases, self.neighb_cov, lut)

    def propose_split(
        self,
        ctx: ModelSelectionContext,
        group: Tensor,
        *,
        debug: bool = False,
        keep_original_components: bool = False,
    ) -> tuple[SplitCaseProposal | None, SplitCaseDebugInfo | None]:
        assert self.noise is not None
        assert self.erp is not None

        single = group.numel() == 1
        if single:
            k = self.p.single_split_k
        else:
            k = self.p.split_k

        # get dense train set slice in group
        split_data = ctx.train_data.dense_slice_by_unit(
            group,
            gen=self.rg,
            min_count=self.p.min_count,
            labels=ctx.train_labels,
            min_per_neighborhood=self.p.min_channel_count,
        )
        if split_data is None and debug:
            return None, SplitCaseDebugInfo(bailed=True, bail_reason="count")
        elif split_data is None:
            return None, None

        if keep_original_components:
            ixs = split_data.indices
            split_data_labels = ctx.train_labels.to(ixs.device)[ixs]
        else:
            split_data_labels = None

        # kmeans on interp whitened feats
        kmeans_responsibilities, kmeans_x, kmeans_chans = try_kmeans(
            data=split_data,
            k=k,
            erp=self.erp,
            gen=self.rg,
            feature_rank=self.noise.rank,
            min_count=self.p.split_min_count,
            min_channel_count=self.p.min_channel_count,
            n_kmeans_tries=self.p.kmeans_tries,
            kmeans_beta=self.p.kmeans_beta,
            n_kmeanspp_tries=self.p.kmeanspp_tries,
            bail_at=int(single),
            debug=debug,
            whiten=self.p.whiten_split,
            force_labels=split_data_labels,
            kmeanspp_metric=self.p.split_kmeanspp_metric,
            scale_dist_args=self.p.scale_dist_args,
        )
        if kmeans_responsibilities is None and debug:
            group_str = ",".join(map(str, group.tolist()))
            logger.dartsortverbose(f"Split {group_str}: kmeans bailed.")
            return None, SplitCaseDebugInfo(
                bailed=True,
                bail_reason="kmeans",
                kmeans_x=kmeans_x,
                split_data=split_data,
                kmeans_chans=kmeans_chans,
            )
        elif kmeans_responsibilities is None:
            return None, None
        assert kmeans_responsibilities.shape[1] >= 1
        if debug:
            assert kmeans_x is not None
        proposal = SplitCaseProposal(
            group=group,
            train_indices=split_data.indices,
            kmeans_responsibilities=kmeans_responsibilities,
            kmeans_x=kmeans_x,
            kmeans_chans=kmeans_chans,
        )
        return proposal, None

    def check_splits(
        self,
        ctx: ModelSelectionContext,
        proposals: list[SplitCaseProposal],
        *,
        debug: bool = False,
        fix_responsibilities: bool | None = None,
    ) -> list[tuple[SplitCaseResult, SplitCaseDebugInfo | None]]:
        assert self.noise is not None
        assert self.erp is not None
        if not proposals:
            return []
        if debug:
            assert len(proposals) == 1
        results: list[SplitCaseResult] = [None] * len(proposals)
        debug_infos: list[SplitCaseDebugInfo | None] = [None] * len(proposals)

        # initialize dense model with fixed resps
        n_kmeans_units = [p.kmeans_responsibilities.shape[1] for p in proposals]
        width = max(n_kmeans_units)
        split_fit = (
            TruncatedMixtureModel.initialize_from_dense_data_and_responsibilities(
                data=ComponentBlockData.from_block_sizes(
                    ctx.train_data.dense_slice(
                        torch.concatenate([p.train_indices for p in proposals])
                    ),
                    [p.train_indices.numel() for p in proposals],
                    contiguous_block_unit_ids(torch.tensor(n_kmeans_units)),
                ),
                responsibilities=torch.concatenate(
                    [
                        F.pad(p.kmeans_responsibilities, (0, width - n))
                        for p, n in zip(proposals, n_kmeans_units, strict=True)
                    ]
                ),
                signal_rank=self.signal_rank,
                erp=self.erp,
                noise=self.noise,
                neighb_cov=self.neighb_cov,
                total_log_proportions=torch.stack(
                    [
                        self.b.log_proportions[p.group].logsumexp(dim=0)
                        for p in proposals
                    ]
                ),
                noise_log_prop=self.b.noise_log_prop,
                p=self.p,
                fix_responsibilities=fix_responsibilities,
            )
        )
        split_model = split_fit.model
        n_split_units = (split_fit.data.block_unit_ids >= 0).sum(dim=1).tolist()
        if debug:
            kmeans_x = proposals[0].kmeans_x
            assert kmeans_x is not None
            debug_infos[0] = SplitCaseDebugInfo(
                split_model=split_model,
                kmeans_x=kmeans_x[split_fit.kept_mask],
                kmeans_chans=proposals[0].kmeans_chans,
                kmeans_responsibilities=split_fit.responsibilities,
                split_data=split_fit.data.data,
            )

        # if we start and end with 1, bail. if not single, then who knows what happened,
        # let's accept it as long as the improvement was good (below).
        survivors = []
        for j, proposal in enumerate(proposals):
            if proposal.group.numel() > 1 or n_split_units[j] > 1:
                survivors.append(j)
            elif debug:
                dinfo_j = debug_infos[j]
                assert dinfo_j is not None
                group_str = ",".join(map(str, proposal.group.tolist()))
                logger.dartsortverbose(
                    f"Split {group_str}: only {n_split_units[j]} of {n_kmeans_units[j]} sub-units."
                )
                debug_infos[j] = replace(dinfo_j, bailed=True, bail_reason="fit")
        if not survivors:
            return list(zip(results, debug_infos, strict=True))
        full_train, survivor_ixs = split_fit.data.select_blocks(survivors)
        split_responsibilities = split_fit.responsibilities[survivor_ixs]
        full_train_scores = split_model.score_blocks(full_train)
        distances = split_model.unit_distance_matrix()

        # see if that dog hunts
        merge_proposals = []
        eval_indices = []
        for b, j in enumerate(survivors):
            in_block = full_train.block_slice(b)
            if ctx.eval_data is None:
                assert ctx.eval_scores.log_liks.shape[0] == ctx.train_data.N
                cur_scores = ctx.eval_scores.slice(full_train.data.indices[in_block])
            else:
                split_eval_data = ctx.eval_data.dense_slice_by_unit(
                    proposals[j].group, gen=self.rg, labels=ctx.eval_labels
                )
                assert split_eval_data is not None
                assert ctx.eval_scores.log_liks.shape[0] == ctx.eval_data.N
                eval_indices.append(split_eval_data.indices)
                cur_scores = ctx.eval_scores.slice(split_eval_data.indices)
            units = full_train.block_unit_ids[b, : n_split_units[j]]
            merge_proposals.append(
                MergeProposal(
                    group=proposals[j].group,
                    block=b,
                    pair_mask=distances[units][:, units] <= self.p.split_max_distance,
                    responsibilities=split_responsibilities[
                        in_block, : n_split_units[j]
                    ],
                    cur_scores=cur_scores,
                    use_cl_alpha=True,
                )
            )
        if ctx.eval_data is None:
            full_eval = None
        else:
            full_eval = ComponentBlockData.from_block_sizes(
                ctx.eval_data.dense_slice(torch.concatenate(eval_indices)),
                [ixs.numel() for ixs in eval_indices],
                full_train.block_unit_ids,
            )
        merge_results = brute_merge(
            full_model=split_model,
            full_train=full_train,
            full_eval=full_eval,
            full_train_scores=full_train_scores,
            proposals=merge_proposals,
            debug=debug,
        )

        for b, j in enumerate(survivors):
            group = proposals[j].group
            merge_res = merge_results[b]
            single = group.numel() == 1
            if debug or logger.isEnabledFor(DARTSORTVERBOSE):
                group_str = ",".join(map(str, group.tolist()))
            else:
                group_str = ""
            if debug:
                debug_infos[j] = replace(debug_infos[j], merge_res=merge_res)  # ty: ignore[invalid-argument-type]

            # same as above policy in "single" case
            if single and merge_res is not None and merge_res.grouping.n_groups <= 1:
                if debug:
                    logger.dartsortdebug(
                        f"Split {group_str}: merged all {n_split_units[j]} split sub-units."
                    )
                    debug_infos[j] = replace(
                        debug_infos[j],  # ty: ignore[invalid-argument-type]
                        bailed=True,
                        bail_reason="merge <= 1 groups",
                    )
                continue

            # reject bad proposals
            if merge_res is not None and merge_res.improvement <= 0.0:
                if debug:
                    logger.dartsortdebug(
                        f"Split {group_str}: mini-merge improvement {merge_res.improvement:0.3f} too small."
                    )
                    debug_infos[j] = replace(
                        debug_infos[j],  # ty: ignore[invalid-argument-type]
                        bailed=True,
                        bail_reason=f"improvement {merge_res.improvement:0.3f}",
                    )
                continue

            # handle edge case when merge_res is none. this could arise when there are no
            # allowed partitions, so handle that first.
            pair_mask = merge_proposals[b].pair_mask
            assert pair_mask is not None
            no_allowed_partitions = not pair_mask.clone().fill_diagonal_(False).any()
            if no_allowed_partitions:
                assert merge_res is None
                # units are too separated to merge. just assign by scoring.
                in_block = full_train.block_slice(b)
                units = full_train.block_unit_ids[b, : n_split_units[j]]
                train_labels = full_train_scores.log_liks[in_block, : units.numel()]
                train_labels = train_labels.argmax(dim=1)
                train_indices = full_train.data.indices[in_block]
                n_split = units.numel()
                means = split_model.b.means[units]
                bases = split_model.b.bases[units] if self.signal_rank else None
                sub_proportions = split_model.b.log_proportions[units].softmax(dim=0)
            elif merge_res is None:
                if debug:
                    logger.dartsortdebug(
                        f"Split {group_str}: no split, merge failed with allowed partitions."
                    )
                    debug_infos[j] = replace(
                        debug_infos[j],  # ty: ignore[invalid-argument-type]
                        bailed=True,
                        bail_reason="merge_fail",  # ty: ignore[invalid-argument-type]
                    )
                continue
            else:
                train_labels = merge_res.train_assignments
                train_indices = merge_res.train_indices

                # handle -1s in group ids: requires flattening the train_assignments
                gids = merge_res.grouping.group_ids
                gkeep = (gids >= 0).logical_and_(
                    (train_labels == gids.unsqueeze(1)).any(1)
                )
                (gkeep,) = gkeep.nonzero(as_tuple=True)
                unflat_group_ids = gids[gkeep].unique()
                flatspace = gids.new_full(
                    (merge_res.grouping.n_groups_before_demolish,), -1
                )
                flat_group_ids = torch.arange(
                    unflat_group_ids.numel(), device=gids.device
                )
                flatspace[unflat_group_ids] = flat_group_ids
                if pnoid:
                    assert torch.isin(train_labels, unflat_group_ids).all()
                train_labels = flatspace[train_labels]
                if pnoid:
                    assert (train_labels >= 0).all()

                n_split = unflat_group_ids.numel()
                means = merge_res.means[unflat_group_ids]
                if merge_res.bases is not None:
                    bases = merge_res.bases[unflat_group_ids]
                else:
                    bases = None
                sub_proportions = merge_res.sub_proportions[unflat_group_ids]
                sub_proportions /= sub_proportions.sum()
                if pnoid:
                    assert (sub_proportions > 0).all()
                    _psum = sub_proportions.sum()
                    _assert_propclose(_psum, torch.ones_like(_psum))
                    assert train_labels.max() == n_split - 1

            if logger.isEnabledFor(DARTSORTVERBOSE):
                _l, _c = train_labels.unique(return_counts=True)
                imp = None if merge_res is None else merge_res.improvement
                if imp and imp > 0 and n_split != 1:
                    logger.dartsortverbose(
                        f"Split {group_str}: {n_split} parts with improvement {imp}, "
                        f"assigned to {_l.tolist()} with counts {_c.tolist()}."
                    )
            if pnoid:
                _lp = sub_proportions.sum()
                assert sub_proportions.shape == (n_split,)
                _assert_propclose(_lp, torch.ones_like(_lp))

            results[j] = SuccessfulSplitCaseResult(
                unit_ids=group,
                n_split=n_split,
                train_indices=train_indices,
                train_assignments=train_labels,
                means=means,
                bases=bases,
                sub_proportions=sub_proportions,
            )
        return list(zip(results, debug_infos, strict=True))

    def propose_and_check_splits(
        self, ctx: ModelSelectionContext, groups: Iterable[Tensor]
    ) -> Iterator[SplitCaseResult]:
        proposals = []
        n_spikes = 0
        for group in groups:
            proposal, _ = self.propose_split(ctx, group)
            if proposal is None:
                continue
            proposals.append(proposal)
            n_spikes += proposal.train_indices.numel()
            if n_spikes >= self.MAX_BATCH_SPIKES:
                for split_res, _ in self.check_splits(ctx, proposals):
                    yield split_res
                proposals = []
                n_spikes = 0
        for split_res, _ in self.check_splits(ctx, proposals):
            yield split_res

    def split(
        self,
        ctx: ModelSelectionContext,
        *,
        show_progress: bool = True,
        friend_distance: float | None = None,
        _stop_after: int | None = None,
        _dry_run: bool = False,
    ) -> SplitResult:
        if friend_distance is None:
            friend_distance = self.p.split_friend_distance
        split_groups = self.group_units_by_distance(
            distance=friend_distance,
            max_group_size=max(1, self.p.split_k - 1),
        )
        if _stop_after:
            split_groups = [
                sg for _, sg in zip(range(_stop_after), split_groups, strict=False)
            ]
        if show_progress:
            split_groups = progbar(split_groups, desc="Split", smoothing=0.0)

        split_results = self.propose_and_check_splits(ctx, split_groups)
        split_result = self._apply_splits(
            split_results,
            train_data=ctx.train_data,
            _early_stop=bool(_stop_after),
            _dry_run=_dry_run,
        )

        # split generates a lot of weird small buffers
        gc.collect()
        torch.cuda.empty_cache()

        return split_result

    def group_units_by_distance(
        self, distance: float, max_group_size: int
    ) -> Iterable[Tensor]:
        if not distance:
            return self.unit_ids[:, None]
        distances = self.unit_distance_matrix()
        # restrict my merge such that units which don't overlap at all aren't
        # considered. to trim search, but also because bad solutions can
        # occur in that case.
        un_adj = (self.lut.b.lut < self.lut.b.unit_ids.shape[0]).float()
        uu_not_adj = (un_adj @ self.neighb_cov.b.neighb_adj @ un_adj.T) == 0
        distances.masked_fill_(uu_not_adj, torch.inf)
        distances.diagonal().zero_()
        return tree_groups(
            distances, max_group_size=max_group_size, max_distance=distance
        )

    def propose_and_check_merges(
        self,
        ctx: ModelSelectionContext,
        groups: Iterable[tuple[Tensor, Tensor | None]],
        *,
        debug: bool = False,
    ) -> Iterator[tuple[Tensor, GroupMergeResult]]:
        proposals = []
        train_indices = []
        eval_indices = []
        n_spikes = 0
        for group, pair_mask in groups:
            assert group.numel() <= self.p.max_group_size
            group_train_data = ctx.train_data.dense_slice_by_unit(
                group,
                gen=self.rg,
                labels=ctx.train_labels,
                min_per_unit=self.p.min_count,
            )
            assert group_train_data is not None
            train_indices.append(group_train_data.indices)
            if ctx.eval_data is None:
                assert ctx.eval_scores.log_liks.shape[0] == ctx.train_data.N
                cur_scores = ctx.eval_scores.slice(group_train_data.indices)
            else:
                group_eval_data = ctx.eval_data.dense_slice_by_unit(
                    group, gen=self.rg, labels=ctx.eval_labels
                )
                assert group_eval_data is not None
                assert ctx.eval_scores.log_liks.shape[0] == ctx.eval_data.N
                eval_indices.append(group_eval_data.indices)
                cur_scores = ctx.eval_scores.slice(group_eval_data.indices)
            proposals.append(
                MergeProposal(
                    group=group,
                    block=len(proposals),
                    pair_mask=pair_mask,
                    responsibilities=None,
                    cur_scores=cur_scores,
                    use_cl_alpha=not self.p.cl_split_only,
                )
            )
            n_spikes += group_train_data.N
            if n_spikes >= self.MAX_BATCH_SPIKES:
                yield from self.check_merges(
                    ctx, proposals, train_indices, eval_indices, debug=debug
                )
                proposals = []
                train_indices = []
                eval_indices = []
                n_spikes = 0
        yield from self.check_merges(
            ctx, proposals, train_indices, eval_indices, debug=debug
        )

    def check_merges(
        self,
        ctx: ModelSelectionContext,
        proposals: list[MergeProposal],
        train_indices: list[Tensor],
        eval_indices: list[Tensor],
        *,
        debug: bool = False,
    ) -> list[tuple[Tensor, GroupMergeResult]]:
        if not proposals:
            return []
        block_unit_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(p.group, device=ctx.train_data.device) for p in proposals],
            batch_first=True,
            padding_value=-1,
        )
        full_train = ComponentBlockData.from_block_sizes(
            ctx.train_data.dense_slice(torch.concatenate(train_indices)),
            [ixs.numel() for ixs in train_indices],
            block_unit_ids,
        )
        if ctx.eval_data is None:
            full_eval = None
        else:
            full_eval = ComponentBlockData.from_block_sizes(
                ctx.eval_data.dense_slice(torch.concatenate(eval_indices)),
                [ixs.numel() for ixs in eval_indices],
                block_unit_ids,
            )
        group_results = brute_merge(
            full_model=self,
            full_train=full_train,
            full_eval=full_eval,
            full_train_scores=self.score_blocks(full_train),
            proposals=proposals,
            debug=debug,
        )
        return [(p.group, res) for p, res in zip(proposals, group_results, strict=True)]

    def merge(
        self,
        ctx: ModelSelectionContext,
        *,
        pair_mask: Tensor | None = None,
        show_progress: bool = True,
    ) -> UnitRemapping:
        """Break into subgroups and merge those. This will also modify data objects."""
        result_map = UnitRemapping.identity(self.n_units)
        any_merged = False

        # this is a good opportunity to discard dead units
        result_map.mapping[lut_blank_units(self.lut)] = -1

        groups = self.group_units_by_distance(
            distance=self.p.merge_max_distance, max_group_size=self.p.max_group_size
        )
        groups = [g for g in groups if g.numel() > 1]

        if show_progress:
            groups = progbar(groups, desc="Merge", smoothing=0.0)
        if pair_mask is None:
            groups_and_masks = ((group, None) for group in groups)
        else:
            groups_and_masks = (
                (group, pair_mask[group[:, None], group[None, :]]) for group in groups
            )

        for group, group_res in self.propose_and_check_merges(ctx, groups_and_masks):
            # no-merge cases
            if group_res is None:
                continue
            if group_res.improvement <= 0:
                continue
            if group_res.grouping.n_groups == group.numel():
                continue
            if logger.isEnabledFor(DARTSORTVERBOSE):
                logger.dartsortverbose(
                    "Group %s best partition %s had improvement %s.",
                    group.tolist(),
                    group_res.grouping.group_ids.tolist(),
                    group_res.improvement,
                )
            any_merged = True

            group_ids = group_res.grouping.group_ids.cpu()
            groups_full = group_ids.unique()
            groups_kept = groups_full[groups_full >= 0]
            assert groups_kept.numel() == group_res.grouping.n_groups

            # log(new_props) = props[group].sum() * sub_props)
            group_log_prop = self.b.log_proportions[group].logsumexp(dim=0)
            new_log_props = group_res.sub_proportions.log().add_(group_log_prop)
            new_log_props = new_log_props.clamp_(min=self.LP_MIN)
            if pnoid and not new_log_props[groups_kept].isfinite().all():
                raise AssertionError(
                    f"Non-finite log proportions in merge of group {group.tolist()} "
                    f"with partition {group_ids.tolist()}: {new_log_props=}, "
                    f"{group_res.sub_proportions=}, {group_log_prop=}."
                )

            # demolish case. throw these away.
            ids_discard = group[group_ids == -1]
            result_map.mapping[ids_discard] = -1
            self.b.log_proportions[ids_discard] = -torch.inf
            self.b.means[ids_discard] = torch.nan
            if self.signal_rank:
                self.b.bases[ids_discard] = torch.nan

            for g in groups_kept:
                ids_in_group = group[group_ids == g]

                # else, move component to first id in group and throw rest away
                first = ids_in_group[0]
                rest = ids_in_group[1:]
                if pnoid:
                    assert first == ids_in_group.min()

                # first retains id (for now, cleanup below remaps to flat)
                result_map.mapping[ids_in_group] = first

                # first gets the parameter update
                self.b.log_proportions[first] = new_log_props[g]
                self.b.means[first] = group_res.means[g]
                if self.signal_rank:
                    assert group_res.bases is not None
                    self.b.bases[first] = group_res.bases[g]

                # rest are poisoned
                self.b.log_proportions[rest] = -torch.inf
                self.b.means[rest] = torch.nan
                if self.signal_rank:
                    self.b.bases[rest] = torch.nan
            if pnoid:
                _assert_propclose(
                    self.b.log_proportions[group].logsumexp(dim=0), group_log_prop
                )

        # possible early exit
        noop = result_map.is_identity()
        if any_merged:
            assert not noop
        if noop:
            return result_map

        # fix up my parameters, update the datasets, and update the lut
        flat_map = self.cleanup(result_map)
        distances = self.unit_distance_matrix()
        if pnoid:
            assert not distances.isnan().any()
        lut, n_new = ctx.train_data.remap(remapping=flat_map, distances=distances)
        assert lut is not None
        assert not n_new
        self.update_lut(lut)
        assert self.lut_params is not None
        self._check_logprop()
        self.lut_params.check()

        # merge generates a lot of weird small buffers
        gc.collect()
        torch.cuda.empty_cache()

        return flat_map

    def demolish(
        self,
        *,
        train_data: TruncatedSpikeData,
        val_data: TruncatedSpikeData,
        show_progress: bool = True,
        allow_uncovered: bool = False,
        _stop_after: int | None = None,
    ) -> UnitRemapping:
        """Destroy units with bad generalization."""
        # score train and validation sets, determine unit responsibilities in both
        train_scores = self.soft_assign(
            data=train_data, needs_bootstrap=False, full_proposal_view=True
        )
        val_scores = self.soft_assign(
            data=val_data, needs_bootstrap=False, full_proposal_view=True
        )
        mean_train_resp = mean_responsibilities(
            scores=train_scores, n_units=self.n_units, weights=train_data.sample_weights
        )
        mean_val_resp = mean_responsibilities(
            scores=val_scores, n_units=self.n_units, weights=val_data.sample_weights
        )
        val_row_crit = proportion_adjusted_row_criterion(
            log_liks=val_scores.log_liks,
            candidates=val_scores.candidates,
            orig_log_props=self.b.log_proportions,
            new_log_props=demolition_log_proportions(
                mean_train_resp, self.b.noise_log_prop, self.n_units
            ),
            cl_alpha=self.p.demolish_cl_alpha,
        )

        # checking this invariant at the top
        assert self.b.log_proportions.isfinite().all()

        # units in groups may care about demolition of their neighbors, so work in groups
        groups = self.group_units_by_distance(
            distance=self.p.merge_max_distance, max_group_size=self.p.max_group_size
        )
        if show_progress:
            groups = progbar(groups, desc="Demolish", smoothing=0.0)
        demolished = torch.zeros_like(self.unit_ids, dtype=torch.bool)
        for j, group in enumerate(groups):
            group_res = evaluate_group_demolitions(
                mm=self,
                group=group,
                mean_train_resp=mean_train_resp,
                mean_eval_resp=mean_val_resp,
                train_scores=train_scores,
                eval_scores=val_scores,
                eval_row_crit=val_row_crit,
                train_data=train_data,
                eval_data=val_data,
            )
            if _stop_after and j > _stop_after:
                # for profiling
                return cast(UnitRemapping, None)
            if group_res.improvement <= 0:
                continue
            assert group_res.demolished is not None
            group_demolished = group_res.demolished.cpu()
            if not group_demolished.any():
                continue
            demolish_ids = group[group_demolished]
            if logger.isEnabledFor(DARTSORTVERBOSE):
                logger.dartsortverbose(
                    "Demolish %s (in group %s) with improvement=%s.",
                    demolish_ids.tolist(),
                    group.tolist(),
                    group_res.improvement,
                )
            demolished[demolish_ids] = True

        # apply demolition to mixture model and train data
        # update log proportions and LUT
        res = self.destroy_units(
            unit_ids=self.unit_ids[demolished],
            train_data=train_data,
            train_scores=train_scores,
            allow_uncovered=allow_uncovered,
        )
        assert self.lut_params is not None
        self.lut_params.check()
        return res

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

    def destroy_units(
        self,
        unit_ids: Tensor,
        *,
        train_data: TruncatedSpikeData,
        train_scores: Scores,
        full_proposal_view: bool = True,
        allow_uncovered: bool = False,
    ) -> UnitRemapping:
        """Remove components from the mixture and reallocate their log proportions

        Log proportions are reallocated by removing unit_ids' responsiblities
        for train spikes, re-normalizing, then comparing the means of those
        responsibilities to the original means.
        """
        # determine mean responsibilities before and after removing units
        orig_prop = mean_responsibilities(
            train_scores, n_units=self.n_units, weights=train_data.sample_weights
        )
        destroy_scores = remove_units_from_scores(train_scores, unit_ids)
        new_prop = mean_responsibilities(
            destroy_scores, n_units=self.n_units, weights=train_data.sample_weights
        )
        assert orig_prop.shape == new_prop.shape == (self.n_units + 1,)

        # this op won't change noise log prop, so remove that factor
        orig_prop = orig_prop[: self.n_units]
        orig_prop /= orig_prop.sum()
        new_prop = new_prop[: self.n_units]
        new_prop /= new_prop.sum()

        # check some conditions hold
        assert torch.all(new_prop[unit_ids] == 0)
        rest_mask = torch.isin(self.unit_ids, unit_ids.to(self.unit_ids)).logical_not_()
        rest_ids = self.unit_ids[rest_mask]
        new_rest_prop = new_prop[rest_ids]
        old_rest_prop = orig_prop[rest_ids]
        assert torch.all(new_rest_prop >= old_rest_prop - 1e-7)
        assert new_rest_prop.isfinite().all()

        # new proportions will have -inf for unit_ids, normalized for rest
        new_lp = torch.full_like(new_prop, -torch.inf)
        new_lp[rest_ids] = new_rest_prop.log().clamp(min=TruncatedMixtureModel.LP_MIN)
        new_lp = new_lp.log_softmax(dim=0)

        # correctly normalize
        non_noise_lp = torch.log1p(-self.b.noise_log_prop.exp())
        new_lp += non_noise_lp

        # apply update
        self.b.log_proportions[:] = new_lp.to(self.b.log_proportions)
        self.b.log_proportions.clamp_(min=self.LP_MIN)
        self._check_logprop()

        # now, having done this, re-score train data and do a candidate update
        # this gives train_data the opportunity to push new units into the top
        # slot instead of having -1s. if those were -1s, some neighborhoods could
        # have no top candidates, leading to a crash in update() called in remap()
        _ = self.soft_assign(
            data=train_data,
            full_proposal_view=full_proposal_view,
            needs_bootstrap=False,
        )

        # discard the units and notify train data
        remap = UnitRemapping.discard_mapping(self.n_units, unit_ids)
        self.cleanup(remap)
        lut, n_new = train_data.remap(
            remapping=remap,
            distances=None if allow_uncovered else self.unit_distance_matrix(),
            allow_uncovered=allow_uncovered,
        )
        assert lut is not None
        if allow_uncovered:
            assert not n_new
        if n_new:
            self.add_new_blank_units(n_new)
        self.update_lut(lut, no_parameter_changes=not n_new)

        return remap

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
            remapping.mapping.cpu(),
            return_index=True,
            return_inverse=False,
            return_counts=False,
        )
        uniq_remapped_ids = torch.from_numpy(uniq_remapped_ids)
        uniq_first_inds = torch.from_numpy(uniq_first_inds)
        assert uniq_first_inds.shape == uniq_remapped_ids.shape
        uniq_first_inds = uniq_first_inds[uniq_remapped_ids >= 0]
        uniq_remapped_ids = uniq_remapped_ids[uniq_remapped_ids >= 0]

        # figure out flat mapping this is filled out in the for loop
        # (could probably use the unique inverse if you had your thinking cap on)
        new_n_units = uniq_remapped_ids.numel()
        new_ids = torch.arange(new_n_units)
        new_remapping = UnitRemapping(mapping=torch.full_like(remapping.mapping, -1))

        # check invariant below holds at the top, else can't guarantee it below
        assert self.b.log_proportions[uniq_first_inds].isfinite().all()

        # poison discarded units
        (discard,) = (remapping.mapping.cpu() == -1).nonzero(as_tuple=True)
        self.b.log_proportions[discard] = -torch.inf
        self.b.means[discard] = torch.nan
        if self.signal_rank:
            self.b.bases[discard] = torch.nan
        assert self.b.log_proportions[uniq_first_inds].isfinite().all()

        # rearrange parameters and delete parameters for destroyed units
        if (uniq_first_inds >= new_ids).all():
            # since we read later indices than are being written, in place is fine
            _assert_sameids(uniq_first_inds.sort().values, uniq_first_inds)
            _assert_sameids(new_ids, torch.arange(new_n_units))
            old_id = new_id = -1
            for old_id, new_id in zip(uniq_first_inds, new_ids, strict=True):
                assert old_id >= new_id
                new_remapping.mapping[remapping.mapping == old_id] = new_id
                if old_id == new_id:
                    continue

                self.b.means[new_id] = self.b.means[old_id]
                self.b.log_proportions[new_id] = self.b.log_proportions[old_id]
                if self.signal_rank:
                    self.b.bases[new_id] = self.b.bases[old_id]

                self.b.means[old_id] = torch.nan
                self.b.log_proportions[old_id] = -torch.inf
                if self.signal_rank:
                    self.b.bases[old_id] = torch.nan
            assert new_id == new_n_units - 1 <= old_id
        else:
            # this case is really only hit in testing. i'm asserting that this is
            # a permutation, because that's all i've got implemented
            # new_ids is sorted, uniq_first_inds not
            _assert_sameids(uniq_first_inds.sort().values, new_ids)
            _assert_sameids(new_ids, uniq_remapped_ids)
            new_remapping.mapping.copy_(remapping.mapping)
            self.b.means[new_ids] = self.b.means[uniq_first_inds]
            self.b.log_proportions[new_ids] = self.b.log_proportions[uniq_first_inds]
            if self.signal_rank:
                self.b.bases[new_ids] = self.b.bases[uniq_first_inds]
            self.b.log_proportions[new_n_units:] = -torch.inf

        # double check to fix up log props. should exp-sum to 1-noiseprop
        logsum = self.b.log_proportions.logsumexp(dim=0)
        targsum = torch.log(1.0 - self.b.noise_log_prop.exp())
        _assert_propclose(logsum, targsum)
        self.b.log_proportions.add_(targsum - logsum)
        assert self.b.log_proportions[:new_n_units].isfinite().all()
        assert self.b.log_proportions[new_n_units:].isneginf().all()
        logger.dartsortdebug("Cleanup %s -> %s.", self.n_units, new_n_units)

        self.n_units = new_n_units
        self.unit_ids = new_ids
        self.b.means.resize_(new_n_units, *self.b.means.shape[1:])
        self.b.log_proportions.resize_(new_n_units, *self.b.log_proportions.shape[1:])
        if self.signal_rank:
            self.b.bases.resize_(new_n_units, *self.b.bases.shape[1:])

        # final checks (historically easy for me to get this remapping stuff wrong)
        self._check_logprop()
        assert self.b.means[:, 0].isfinite().all()
        if self.signal_rank:
            assert self.b.bases[:, 0, 0].isfinite().all()

        return new_remapping

    def add_new_blank_units(self, n_new: int):
        if not n_new:
            return
        new_n_units = self.n_units + n_new
        self.b.means.resize_(new_n_units, *self.b.means.shape[1:])
        self.b.log_proportions.resize_(new_n_units, *self.b.log_proportions.shape[1:])
        self.b.means[self.n_units :].zero_()
        self.b.log_proportions[self.n_units :].fill_(self.LP_MIN)
        if self.signal_rank:
            self.b.bases.resize_(new_n_units, *self.b.bases.shape[1:])
            self.b.bases[self.n_units :].zero_()
        self.n_units = new_n_units
        self.unit_ids = torch.arange(new_n_units)

    def _apply_splits(
        self,
        unit_splits: Iterable[SplitCaseResult],
        train_data: TruncatedSpikeData,
        *,
        _early_stop: bool = False,
        _dry_run: bool = False,
    ):
        assert train_data.candidates is not None
        cur_train_labels = train_data.candidates[:, 0]
        train_labels = torch.full_like(cur_train_labels, -1)
        train_candidate_mask = torch.zeros_like(train_labels, dtype=torch.bool)
        train_labels_mask = torch.zeros_like(train_labels, dtype=torch.bool)

        n_new_units = 0
        n_shrink = 0
        new_means = []
        new_log_props = []
        new_bases = []
        cur_max_label = self.n_units - 1
        invalidated_ids = []

        new_logprop = torch.asarray(
            self.b.log_proportions, copy=True, dtype=torch.double
        )

        for res in unit_splits:
            if res is None or _dry_run:
                continue

            unit_ids = res.unit_ids.tolist()
            n_group = len(unit_ids)

            # which candidates need updating? which labels change directly from this split?
            cur_cands_changed = torch.isin(
                cur_train_labels, res.unit_ids.to(cur_train_labels)
            )
            train_candidate_mask.logical_or_(cur_cands_changed)
            train_labels_mask[res.train_indices] = True

            # assign labels within the group; split ids 0 through groupsize-1 get orig ids
            for j, unit_id in enumerate(res.unit_ids):
                for_j = res.train_assignments == j
                train_labels[res.train_indices[for_j]] = unit_id

            # handle case where unit count decreases
            new_group_count = min(res.n_split, n_group)
            cur_out_ids = res.unit_ids[:new_group_count]
            nogrowth = res.n_split <= n_group
            if nogrowth:
                # note: train_labels already has -1s for these guys. still, need to invalidate
                # params, return log proportion, etc. let's mark these as invalid and throw them
                # away with a rotate later.
                group_invalidated_ids = unit_ids[new_group_count:]
                invalidated_ids.extend(group_invalidated_ids)
                n_shrink += n_group - res.n_split
            else:
                # making new ids. start them at 0, offset by K.
                (for_other,) = (res.train_assignments >= n_group).nonzero(as_tuple=True)
                assert for_other.numel() > 0
                other_labels = res.train_assignments[for_other]
                other_labels -= other_labels.amin()
                other_labels.add_(cur_max_label + 1)
                train_labels[res.train_indices[for_other]] = other_labels
                n_new_units += res.n_split - n_group
                cur_max_label += res.n_split - n_group
                group_invalidated_ids = []

            # divvy up my log proportion
            assert res.sub_proportions.shape == res.means.shape[:1] == (res.n_split,)
            group_lp = self.b.log_proportions[res.unit_ids].double().logsumexp(dim=0)
            split_log_props = res.sub_proportions.double().log() + group_lp
            split_log_props = split_log_props.clamp_(min=self.LP_MIN)

            # -- assign new parameters
            # first split result units go into the group ids, later get new units
            # start with proportions
            new_logprop[cur_out_ids] = split_log_props[:new_group_count]
            new_logprop[group_invalidated_ids] = -torch.inf

            # rest of parameters
            # assign split result first params to unit_id's spot
            self.b.means[cur_out_ids] = res.means[:new_group_count]
            self.b.means[group_invalidated_ids] = torch.nan
            if self.signal_rank:
                assert res.bases is not None
                self.b.bases[cur_out_ids] = res.bases[:new_group_count]
                self.b.bases[group_invalidated_ids] = torch.nan

            # append rest to new_* lists
            if nogrowth:
                continue
            new_means.append(res.means[n_group:])
            new_log_props.append(split_log_props[n_group:])
            if self.signal_rank:
                assert res.bases is not None
                new_bases.append(res.bases[n_group:])
            else:
                new_bases.append(None)

        if _early_stop or _dry_run:
            # this is for profiling / debugging
            return cast(SplitResult, None)

        assert cur_max_label + 1 == self.n_units + n_new_units
        assert train_labels.max() <= cur_max_label
        assert n_new_units == sum(m.shape[0] for m in new_means)
        assert n_new_units == sum(lp.numel() for lp in new_log_props)
        if self.signal_rank:
            assert all(b is not None for b in new_bases)
            assert n_new_units == sum(b.shape[0] for b in new_bases)  # type: ignore  # ty: ignore[x]

        # resize params to allow space for the new guys
        Korig = self.n_units
        self.n_units = Knew = Korig + n_new_units
        logger.dartsortdebug(
            f"Split created {n_new_units} new units ({Korig} -> {Knew}), while "
            f"removing {n_shrink} units."
        )
        self.b.means.resize_(Knew, *self.b.means.shape[1:])
        self.b.log_proportions.resize_(Knew, *self.b.log_proportions.shape[1:])
        if self.signal_rank:
            self.b.bases.resize_(Knew, *self.b.bases.shape[1:])
        self.unit_ids = torch.arange(Knew)

        # assign new unit parameters
        k0 = Korig
        self.b.log_proportions[:Korig] = new_logprop
        for nm, nlp, nb in zip(new_means, new_log_props, new_bases, strict=True):
            newk = nlp.numel()
            self.b.means[k0 : k0 + newk] = nm
            self.b.log_proportions[k0 : k0 + newk] = nlp.to(self.b.log_proportions)
            if nb is not None:
                self.b.bases[k0 : k0 + newk] = nb
            k0 += newk
        assert k0 == Knew
        self._check_logprop(allow_neginf=True)

        # discard invalidated units, so that many labels need to change
        discard = UnitRemapping.discard_mapping(
            Knew, invalidated_ids, train_labels.device
        )
        self.cleanup(discard)
        # tell train data about this
        lut, n_new = train_data.remap(remapping=discard, distances=None)
        del lut  # ignoring this LUT, since we'll shortly get another
        self.add_new_blank_units(n_new)
        # and we will shortly give these labels to train data
        F.threshold(train_labels, -1, Knew, inplace=True)
        train_labels = discard.padded_map()[train_labels]
        self._check_logprop()
        if pnoid:
            assert self.b.means.isfinite().all()
            assert self.b.log_proportions.isfinite().all()
            if self.signal_rank:
                assert self.b.bases.isfinite().all()

        # call data methods for processing splits
        distances = self.unit_distance_matrix()
        if pnoid:
            assert discard.mapping.max() < distances.shape[0]
            assert train_labels.max() < distances.shape[0]
        lut, n_new = train_data.update_from_split(
            train_candidate_mask, train_labels_mask, train_labels, distances
        )
        self.add_new_blank_units(n_new)
        self.update_lut(lut)
        assert self.lut_params is not None
        self.lut_params.check()

        return SplitResult(
            n_new_units=n_new_units,
            n_shrink=n_shrink,
            train_unit_mask=train_candidate_mask,
            train_split_spike_mask=train_labels_mask,
            train_split_spike_labels=train_labels,
        )

    def _check_logprop(self, *, allow_neginf: bool = False):
        if not allow_neginf:
            assert self.b.log_proportions.isfinite().all()
        assert self.noise_log_prop.isfinite()
        _lp = torch.logaddexp(self.noise_log_prop, self.non_noise_log_proportion())
        _assert_propclose(_lp, torch.zeros_like(_lp))

    def _bad_numbers_error(
        self, elb: float | None, stats: SufficientStatistics | None = None
    ):
        """Build an exception to help with numerical problems."""
        msg = "TMM blew up."
        if elb is not None:
            msg += f" (elbo: {elb})"
        msg += "\n"

        # about me
        my_msg = ""
        mean_f = self.b.means.isfinite().all(dim=1)
        if not mean_f.all():
            my_msg += f" - {mean_f.logical_not().sum()} bad means.\n"
        prop_f = self.b.log_proportions.isfinite()
        if not prop_f.all():
            my_msg += f" - {prop_f.logical_not().sum()} bad props.\n"
        if not self.b.noise_log_prop.isfinite():
            my_msg += f" - noise prop {self.b.noise_log_prop.item()}.\n"
        if self.signal_rank:
            basis_f = self.b.bases.isfinite().all(dim=(1, 2))
            if not basis_f.all():
                my_msg += f" - {basis_f.logical_not().sum()} bad bases.\n"
        if not my_msg:
            my_msg = " - Model parameters are fine.\n"
        msg += my_msg

        # about stats
        if stats is not None:
            msg += stats.diagnostic()

        raise BadNumbersError(msg)


# -- helpers


def get_truncated_datasets(
    *,
    sorting: DARTsortSorting,
    motion: MotionInfo,
    tpca: "BaseTemporalPCA | None" = None,
    clustering_features_cfg: ClusteringFeaturesConfig | None,
    refinement_cfg: RefinementConfig,
    device: torch.device,
    rg: int | np.random.Generator,
    noise: KroneckerNoise | None = None,
    stable_features: StableWaveformFeatures | None = None,
    save_kwargs: dict | None = None,
    computation_cfg: ComputationConfig | None = None,
):
    rg = np.random.default_rng(rg)

    # we assume that the core neighborhoods are exactly the same as extract ones
    if stable_features is None:
        assert clustering_features_cfg is not None
        stable_features = StableWaveformFeatures.from_config(
            sorting=sorting,
            motion=motion,
            clustering_features_cfg=clustering_features_cfg,
            computation_cfg=computation_cfg,
        )
    full_features = stable_features.features
    full_neighbs = stable_features.neighborhoods.to(device=device)
    del stable_features
    prgeom = pad_geom(motion.rgeom, device=device)
    assert full_features.shape[2] == full_neighbs.b.neighborhoods.shape[1]
    assert full_features.shape[0] == full_neighbs.b.neighborhood_ids.shape[0]
    feature_rank = full_features.shape[1]
    if clustering_features_cfg is not None:
        assert 0 < feature_rank <= clustering_features_cfg.feature_rank
    full_features = full_features.reshape(len(full_features), -1)

    train_ixs, val_ixs, train_weights, val_weights = train_val_split(
        sorting=sorting,
        neighborhood_ids=full_neighbs.b.neighborhood_ids,
        refinement_cfg=refinement_cfg,
        rg=rg,
    )
    train_neighbs = full_neighbs.slice(train_ixs)
    val_neighbs = full_neighbs.slice(val_ixs)

    if noise is None:
        assert sorting.parent_h5_path is not None
        noise = KroneckerNoise.estimate_from_hdf5(
            sorting.parent_h5_path,
            motion=motion,
            tpca=tpca,
            rank=feature_rank,
            zero_radius=refinement_cfg.cov_radius,
            cov_kind=refinement_cfg.cov_kind,
            interp_params=refinement_cfg.noise_interp_params,
            device=device,
        )

    gc.collect()
    torch.cuda.empty_cache()

    assert isinstance(noise, KroneckerNoise)
    noise.to(device=device)
    assert noise.rank == feature_rank
    neighb_cov = NeighborhoodCovariance.from_noise_and_neighborhoods(
        prgeom=prgeom,
        noise=noise,
        neighborhoods=full_neighbs,
        neighb_overlap=refinement_cfg.neighb_overlap,
        huber_quantile=noise_huber_quantile(refinement_cfg),
    )

    train_x = _read_ix_maybe_mmap(full_features, train_ixs)
    if refinement_cfg.spike_storage == "device" or not sorting.n_units:
        train_x = train_x.to(device=device)
    elif refinement_cfg.spike_storage == "cpu":
        assert train_x.device.type == "cpu"
    else:
        panic(refinement_cfg.spike_storage)

    if not sorting.n_units:
        logger.dartsortdebug("Initializing GMM with kmeans++")
        kmeanspp_res = truncated_kmeanspp(
            x=full_features,
            train_x=train_x,
            neighborhoods=full_neighbs,
            neighb_cov=neighb_cov,
            train_ixs=train_ixs,
            train_neighbs=train_neighbs,
            batch_size=refinement_cfg.eval_batch_size,
            seed=rg,
            metric=refinement_cfg.distance_metric,
            scale_dist_args=refinement_cfg.scale_dist_args,
            min_count=refinement_cfg.min_count,
            cfg=refinement_cfg.kmeanspp_cfg,
        )
        sorting = sorting.ephemeral_replace(
            labels=kmeanspp_res.full_labels.numpy(force=True)
        )
        if save_kwargs is not None:
            ds_save_intermediate_labels(
                step_sorting=sorting, **save_kwargs, step_name="tmm000akmeanspp"
            )
        _t_xt = kmeanspp_res.xt
        _t_whitenedx = kmeanspp_res.whitenedx
        _t_CmoCooinvx = kmeanspp_res.CmoCooinvx
        _t_noise_logliks = kmeanspp_res.noise_logliks
        del kmeanspp_res
        if refinement_cfg.spike_storage == "cpu" and device.type != "cpu":
            _t_xt = _pinned_copy(_t_xt)
            _t_whitenedx = _pinned_copy(_t_whitenedx)
            _t_CmoCooinvx = _pinned_copy(_t_CmoCooinvx)
            _t_noise_logliks = _pinned_copy(_t_noise_logliks)
        cleanup_and_log_gpu_usage(computation_cfg=None, message="kmeanspp", dev=device)
    else:
        _t_xt = _t_whitenedx = _t_CmoCooinvx = _t_noise_logliks = None

    sorting = decrumb(sorting, min_size=refinement_cfg.min_count, flatten=True)
    if not sorting.n_units:
        raise ValueError("No units to cluster.")
    assert sorting.labels is not None
    labels = torch.tensor(sorting.labels, device=device)

    n_candidates, n_search, n_explore, max_candidates, max_search, max_explore = (
        _pick_search_size(n_units=sorting.n_units, refinement_cfg=refinement_cfg)
    )

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
    train_data = TruncatedSpikeData.initialize_from_labels(
        n_candidates=n_candidates,
        n_search=n_search,
        n_explore=n_explore,
        max_n_candidates=max_candidates,
        max_n_search=max_search,
        max_n_explore=max_explore,
        search_adj=refinement_cfg.search_adj,
        dense_slice_size_per_unit=refinement_cfg.n_spikes_fit,
        labels=labels[train_ixs].to(device=device),
        x=train_x,
        sample_weights=None
        if train_weights is None
        else train_weights.to(device=device),
        neighborhoods=train_neighbs,
        neighb_cov=neighb_cov,
        neighb_overlap=refinement_cfg.neighb_overlap,
        explore_neighb_steps=refinement_cfg.explore_neighb_steps,
        seed=rg,
        batch_size=refinement_cfg.train_batch_size,
        xt=_t_xt,
        whitenedx=_t_whitenedx,
        CmoCooinvx=_t_CmoCooinvx,
        noise_logliks=_t_noise_logliks,
    )
    del train_x, _t_xt, _t_whitenedx, _t_CmoCooinvx, _t_noise_logliks
    on_device = train_data.storage_device == train_data.device
    assert on_device == (
        refinement_cfg.spike_storage == "device" or device.type == "cpu"
    )
    assert torch.equal(full_neighbs.b.neighborhoods, train_neighbs.b.neighborhoods)
    val_n_candidates = min(
        sorting.n_units, max(refinement_cfg.merge_group_size, n_candidates)
    )
    max_val_n_candidates = max(val_n_candidates, max_candidates)
    if refinement_cfg.hold_out_criterion:
        val_x = _read_ix_maybe_mmap(full_features, val_ixs)
        if refinement_cfg.spike_storage == "device":
            val_x = val_x.to(device=device)
        elif refinement_cfg.spike_storage == "cpu":
            assert val_x.device.type == "cpu"
        else:
            panic(refinement_cfg.spike_storage)
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
            x=val_x,
            neighborhoods=val_neighbs,
            sample_weights=None
            if val_weights is None
            else val_weights.to(device=device),
            neighb_cov=neighb_cov,
            seed=rg,
            batch_size=refinement_cfg.eval_batch_size,
            allow_gaps=True,
            score_only=True,
        )
        del val_x
        on_device = val_data.storage_device == val_data.device
        assert on_device == (
            refinement_cfg.spike_storage == "device" or device.type == "cpu"
        )
        assert torch.equal(
            val_data.neighborhoods.b.neighborhoods,
            train_data.neighborhoods.b.neighborhoods,
        )
    else:
        val_data = None

    if refinement_cfg.impute_kind == "interp":
        assert clustering_features_cfg is not None
        erp = NeighborhoodInterpolator(
            prgeom=prgeom,
            neighborhoods=full_neighbs,
            params=clustering_features_cfg.interp_params,
        )
    elif refinement_cfg.impute_kind == "impute":
        erp = NeighborhoodImputer(noise=noise, neighborhoods=full_neighbs)
    else:
        panic()

    gc.collect()
    torch.cuda.empty_cache()

    return neighb_cov, erp, train_data, val_data, full_data, noise, train_ixs, val_ixs


def train_val_split(
    *,
    sorting: DARTsortSorting,
    neighborhood_ids: Tensor,
    refinement_cfg: RefinementConfig,
    rg: np.random.Generator,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    # how many waveforms will be used for fitting? fit = train \cup val.
    n_fit = min(len(sorting), refinement_cfg.sampling_cfg.more_waveforms_fit)
    if refinement_cfg.hold_out_criterion:
        train_prop = 1.0 - refinement_cfg.val_proportion
    else:
        train_prop = 1.0
    fit_prop_of_total = n_fit / len(sorting)

    # data splits: -1 full, 0 train, 1 val. stratify by unit, or by neighborhood if no labels (for kmeanspp)
    labeled = sorting.labels is not None and sorting.labels.max() >= 0
    if labeled:
        groups = sorting.labels
    else:
        groups = neighborhood_ids.numpy(force=True)
    sampling_cfg = refinement_cfg.sampling_cfg
    bucket_sampling = (
        labeled and n_fit < len(sorting) and sampling_cfg.bucket_sampling_power > 0
    )
    if bucket_sampling:
        _, counts, _ = pos_int_unique_and_counts(groups)
        fit_prop = bucket_sampling_rates(
            counts,
            n_fit=max(1, round(fit_prop_of_total * counts.sum())),
            power=sampling_cfg.bucket_sampling_power,
            max_reweighting=sampling_cfg.bucket_sampling_max_reweighting,
        )
    else:
        fit_prop = fit_prop_of_total
    split_mask = _ensure_split_covers_all(
        groups, rg=rg, fit_prop=fit_prop, train_prop=train_prop
    )

    train_indices = torch.asarray(np.flatnonzero(split_mask == 0))
    val_indices = torch.asarray(np.flatnonzero(split_mask == 1))
    if bucket_sampling:
        train_weights, val_weights = _bucket_sample_weights(
            groups, split_mask, train_indices.numpy(), val_indices.numpy()
        )
    else:
        train_weights = val_weights = None

    return train_indices, val_indices, train_weights, val_weights


def _bucket_sample_weights(
    groups: np.ndarray,
    split_mask: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> tuple[Tensor, Tensor]:
    grouped = groups >= 0
    ids, counts = np.unique(groups[grouped], return_counts=True)
    fit = split_mask >= 0
    assert grouped[fit].all()
    fit_counts = np.bincount(np.searchsorted(ids, groups[fit]), minlength=ids.size)
    assert (fit_counts > 0).all()
    group_weights = (fit_counts.sum() / counts.sum()) / (fit_counts / counts)
    train_weights = group_weights[np.searchsorted(ids, groups[train_indices])]
    val_weights = group_weights[np.searchsorted(ids, groups[val_indices])]
    return (
        torch.asarray(train_weights, dtype=torch.float),
        torch.asarray(val_weights, dtype=torch.float),
    )


def bucket_sampling_rates(
    counts: np.ndarray, *, n_fit: int, power: float, max_reweighting: float
) -> np.ndarray:
    assert counts.ndim == 1 and (counts > 0).all()
    assert power >= 0 and max_reweighting >= 1
    total = counts.sum()
    assert 0 < n_fit <= total
    uniform = n_fit / total
    if power == 0:
        return np.full(counts.shape, uniform)
    lo_rate = uniform / max_reweighting
    hi_rate = min(1.0, uniform * max_reweighting)
    shape = counts.astype(np.float64) ** -power

    def rates_at(log_scale: float) -> np.ndarray:
        return np.clip(np.exp(log_scale) * shape, lo_rate, hi_rate)

    lo = np.log(lo_rate) - np.log(shape.max())
    hi = np.log(hi_rate) - np.log(shape.min())
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if (counts * rates_at(mid)).sum() > n_fit:
            hi = mid
        else:
            lo = mid
    return rates_at(lo)


def _ensure_split_covers_all(
    groups: np.ndarray,
    *,
    rg: np.random.Generator,
    fit_prop: float | np.ndarray,
    train_prop: float,
) -> np.ndarray:
    """stratified sampling for train set, -1 full, 0 train, 1 val"""
    _, counts, n_ungrouped = pos_int_unique_and_counts(groups)
    if isinstance(fit_prop, np.ndarray):
        assert fit_prop.shape == counts.shape
    fit_counts = np.minimum(counts, np.ceil(counts * fit_prop).astype(np.int64))
    assert fit_counts.dtype.kind == "i"
    train_counts = np.maximum(train_prop * fit_counts, 1).astype(np.int64)

    # reorder for csr style indptr
    order = np.argsort(groups, kind="stable")[n_ungrouped:]
    indptr = np.concatenate([[0], np.cumsum(counts)])

    split_mask = np.full(groups.shape, -1, dtype=np.int8)
    for j, (train_count, fit_count) in enumerate(
        zip(train_counts, fit_counts, strict=True)
    ):
        in_group = order[indptr[j] : indptr[j + 1]]
        if in_group.size > fit_count:
            fit_ixs = rg.choice(in_group, size=fit_count, replace=False)
            fit_ixs.sort()
        elif in_group.size == fit_count:
            fit_ixs = in_group
        else:
            panic()
        if fit_ixs.size > train_count:
            train_ixs = rg.choice(fit_ixs, size=train_count, replace=False)
            train_ixs.sort()
        elif fit_ixs.size == train_count:
            train_ixs = fit_ixs
        else:
            panic()

        split_mask[fit_ixs] = 1
        split_mask[train_ixs] = 0

    return split_mask


def instantiate_and_bootstrap_tmm(
    *,
    sorting: DARTsortSorting,
    motion: MotionInfo,
    tpca: "BaseTemporalPCA | None" = None,
    clustering_features_cfg: ClusteringFeaturesConfig | None,
    stable_features: StableWaveformFeatures | None,
    refinement_cfg: RefinementConfig,
    seed: np.random.Generator | int = 0,
    computation_cfg: ComputationConfig | None = None,
    save_kwargs: dict | None = None,
    noise: KroneckerNoise | None = None,
) -> MixtureModelAndDatasets:
    global pnoid
    pnoid = pnoid or logger.isEnabledFor(DARTSORTVERBOSE)
    if pnoid:
        logger.info("Extra TMM asserts are on.")

    rg = np.random.default_rng(seed)
    computation_cfg = ensure_computation_config(computation_cfg)
    device = computation_cfg.actual_device()

    neighb_cov, erp, train_data, val_data, full_data, noise, train_ixs, val_ixs = (
        get_truncated_datasets(
            sorting=sorting,
            motion=motion,
            tpca=tpca,
            stable_features=stable_features,
            clustering_features_cfg=clustering_features_cfg,
            refinement_cfg=refinement_cfg,
            device=device,
            rg=rg,
            save_kwargs=save_kwargs,
            computation_cfg=computation_cfg,
            noise=noise,
        )
    )

    # every spike's feature subset needs to be covered by that of a mixture component,
    # or in other words, at least one spike from each neighborhood id needs to be within
    # the exploration set. that's a hard assumption: otherwise we get all-neginf columns
    # of the likelihood and things break. i'm not sure if the right way to fix that is
    # to handle that case, or to enforce full coverage here, but i'm doing the latter.
    train_data.ensure_coverage(allow_new_units=True)

    # bootstrapping phase: tmm needs lut, lut needs distances, distances need tmm...
    tmm = TruncatedMixtureModel.from_config(
        noise=noise,
        erp=erp,
        neighb_cov=neighb_cov,
        train_data=train_data,
        refinement_cfg=refinement_cfg,
        seed=rg,
    )
    logger.dartsortdebug(
        "Initialize TMM with %s candidates, rank %s, CLa %s.",
        train_data.n_candidates,
        refinement_cfg.signal_rank,
        refinement_cfg.cl_alpha,
    )
    D = tmm.unit_distance_matrix()
    lut = train_data.bootstrap_candidates(D)
    tmm.update_lut(lut)

    if refinement_cfg.initialize_at_rank_0 and refinement_cfg.signal_rank > 0:
        tmm.em(train_data)
        if save_kwargs is not None:
            assert full_data is not None
            full_scores = tmm.soft_assign(
                data=full_data, full_proposal_view=True, needs_bootstrap=False
            )
            ds_save_intermediate_labels(
                step_sorting=sorting.ephemeral_replace(
                    labels=labels_from_scores(full_scores, remove_noise=True)
                ),
                **save_kwargs,
                step_name="tmm000brank0em",
            )
            del full_scores
        scores = tmm.soft_assign(
            data=train_data,
            full_proposal_view=True,
            needs_bootstrap=False,
        )
        tmm.change_rank(
            new_signal_rank=refinement_cfg.signal_rank,
            train_data=train_data,
            train_scores=scores,
        )
        logger.dartsortdebug(
            f"Update signal rank from 0->{tmm.signal_rank} after first EM round."
        )

    assert not isinstance(val_ixs, slice)

    return MixtureModelAndDatasets(
        tmm, train_data, val_data, full_data, train_ixs, val_ixs
    )


@databag
class TruncatedKmeansppResult:
    full_labels: Tensor
    centroid_ixs: Tensor
    phase_centroid_counts: tuple[int, int, int]

    # data smuggling
    xt: Tensor
    whitenedx: Tensor
    CmoCooinvx: Tensor
    noise_logliks: Tensor


@databag
class NeighborhoodCenters:
    values: Tensor
    shared: Tensor
    normsq: Tensor
    n_feats: Tensor


class TruncatedKmeanspp:
    """Neighborhood-sparsity-aware kmeans++

    Sparse flavor of kmeans++ which has some strategies for picking k

    3 stages per try:
        - 1: make sure neighbs are covered
        - 2: do usual kmeans++ until a simple coverage criterion is hit
        - 3: check stopping criteria at each iteration

    sampling="d2" draws wppt distsq^(alpha/2) [5]

    Stopping criteria come mainly from [2, 4]. Some refs below.

    [1] Arthur and Vassilvitskii. k-means++: the advantages of careful seeding.
        SODA 2007. https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
    [2] Kulis and Jordan. Revisiting k-means: new algorithms via Bayesian
        nonparametrics. ICML 2012. https://arxiv.org/abs/1111.0352
    [3] Gonzalez. Clustering to minimize the maximum intercluster distance. TCS
        38:293-306, 1985. https://doi.org/10.1016/0304-3975(85)90224-5
    [4] Bahmani et al. Scalable k-means++. VLDB 2012.
        https://arxiv.org/abs/1203.6402
    [5] Bamas et al. Analyzing D^alpha seeding for k-means. ICML 2024.
        https://arxiv.org/abs/2310.13474
    """

    def __init__(
        self,
        *,
        X: Tensor,
        Xneighbixs: Tensor,
        neighborhoods: Tensor,
        neighb_rel_inds: Tensor,
        visible: Tensor,
        feat_rank: int,
        max_k: int,
        metric: ComponentDistanceMetric,
        scale_dist_args: tuple[float, float, float],
        min_count: float,
        cfg: KmeansppConfig,
    ):
        self.X = X
        self.Xneighbixs = Xneighbixs
        self.neighborhoods = neighborhoods
        self.neighb_rel_inds = neighb_rel_inds
        self.visible = visible
        self.feat_rank = feat_rank
        self.max_k = max_k
        self.metric = metric
        self.scale_dist_args = scale_dist_args
        self.min_count = min_count
        self.cfg = cfg
        self.n_neighb = visible.shape[0]
        self.order, self.indptr, self.counts = sort_to_csr(Xneighbixs, self.n_neighb)

    @classmethod
    def fit_predict(
        cls,
        *,
        x: Tensor | np.ndarray,
        train_x: Tensor,
        neighborhoods: SpikeNeighborhoods,
        neighb_cov: NeighborhoodCovariance,
        train_ixs: Tensor | slice,
        train_neighbs: SpikeNeighborhoods,
        batch_size: int,
        seed: np.random.Generator | int,
        metric: ComponentDistanceMetric,
        scale_dist_args: tuple[float, float, float],
        min_count: float,
        cfg: KmeansppConfig,
    ) -> TruncatedKmeansppResult:
        if metric not in ("euc", "cosine", "normeuc", "scaled_normeuc"):
            panic(metric)
        base_rg = np.random.default_rng(seed)
        max_k = min(
            train_x.shape[0], cfg.max_components_per_channel * neighb_cov.n_channels
        )
        n_channels = neighb_cov.n_channels
        neighb_chans = neighborhoods.b.neighborhoods
        max_nc_obs = neighb_chans.shape[1]
        assert torch.equal(neighb_chans, train_neighbs.b.neighborhoods)
        if isinstance(train_ixs, slice):
            pass
        elif torch.is_tensor(train_ixs):
            assert train_x.shape[0] == train_ixs.shape[0]
        else:
            panic(train_ixs)

        xt, whitenedx, CmoCooinvx, noise_logliks = _whiten_impute_and_noise_score(
            x=train_x, neighborhoods=train_neighbs, neighb_cov=neighb_cov
        )

        neighb_rel_inds = _neighb_relative_index(neighb_chans, n_channels)
        obs = (neighb_rel_inds < max_nc_obs).to(whitenedx)
        if cfg.neighb_overlap is None:
            neighb_adj = neighb_cov.b.neighb_adj
        else:
            neighb_adj = neighborhoods.adjacency(cfg.neighb_overlap).to(
                whitenedx.device
            )
        visible = torch.logical_and(neighb_adj > 0, obs @ obs.T > 0)
        del obs

        if cfg.min_fraction is not None:
            min_count = cfg.min_fraction * train_x.shape[0]
        kmeanspp = cls(
            X=whitenedx,
            Xneighbixs=train_neighbs.b.neighborhood_ids,
            neighborhoods=neighb_chans,
            neighb_rel_inds=neighb_rel_inds,
            visible=visible,
            feat_rank=neighb_cov.feat_rank,
            max_k=max_k,
            metric=metric,
            scale_dist_args=scale_dist_args,
            min_count=min_count,
            cfg=cfg,
        )
        centroid_ixs, phase_centroid_counts = kmeanspp.best_of_tries(base_rg)
        full_labels = kmeanspp.label(
            x=x,
            neighborhoods=neighborhoods,
            neighb_cov=neighb_cov,
            centroid_ixs=centroid_ixs,
            batch_size=batch_size,
        )

        return TruncatedKmeansppResult(
            full_labels=full_labels,
            centroid_ixs=centroid_ixs,
            phase_centroid_counts=phase_centroid_counts,
            xt=xt,
            whitenedx=whitenedx,
            CmoCooinvx=CmoCooinvx,
            noise_logliks=noise_logliks,
        )

    def label(
        self,
        *,
        x: Tensor | np.ndarray,
        neighborhoods: SpikeNeighborhoods,
        neighb_cov: NeighborhoodCovariance,
        centroid_ixs: Tensor,
        batch_size: int,
        chunk_size: int = 2**20,
    ) -> Tensor:
        """In batches, label each x by its best centroid"""
        max_nc_obs = self.neighborhoods.shape[1]
        device = self.X.device
        n_neighbs = neighborhoods.n_neighborhoods
        neighb_ids = neighborhoods.b.neighborhood_ids
        cent_visible = self.visible[:, self.Xneighbixs[centroid_ixs]]

        neighb_centers: dict[int, tuple[Tensor, NeighborhoodCenters]] = {}
        for ni in range(n_neighbs):
            (cand,) = cent_visible[ni].nonzero(as_tuple=True)
            if not cand.numel():
                # kmeans++ ran out of components before covering this neighborhood
                continue
            neighb_centers[ni] = (
                cand,
                self.centers_in_neighborhood(ni, centroid_ixs[cand]),
            )

        full_labels = torch.full(x.shape[:1], -1, dtype=torch.long, device=device)
        for i0 in range(0, x.shape[0], chunk_size):
            i1 = min(x.shape[0], i0 + chunk_size)
            x_chunk = _read_ix_maybe_mmap(x, slice(i0, i1))
            chunk_neighb_ids = neighb_ids[i0:i1]
            order, indptr, _ = sort_to_csr(chunk_neighb_ids, n_neighbs)
            indptr_list = indptr.tolist()
            for ni, (cand, centers) in neighb_centers.items():
                members = order[indptr_list[ni] : indptr_list[ni + 1]]
                for bs in range(0, members.numel(), batch_size):
                    bix = members[bs : bs + batch_size]
                    xb = x_chunk[bix.to(x_chunk.device)].to(device)
                    wx, _ = _whiten_and_noise_score_batch(
                        x=xb,
                        neighb_ids=chunk_neighb_ids[bix],
                        neighb_cov=neighb_cov,
                    )
                    xsq = wx.view(bix.shape[0], self.feat_rank, max_nc_obs)
                    xsq = xsq.square().sum(dim=1)
                    d = self.center_dists(wx, xsq, centers)
                    full_labels[i0 + bix] = cand[d.argmin(dim=1)]
            del x_chunk

        n_unlabeled = (full_labels < 0).sum().item()
        if n_unlabeled:
            logger.warning(f"kmeans++ left {n_unlabeled} spikes unlabeled.")
        return full_labels

    def centers_in_neighborhood(
        self, ni: int, center_ixs: Tensor
    ) -> NeighborhoodCenters:
        max_nc_obs = self.neighborhoods.shape[1]
        n_centers = center_ixs.shape[0]
        rel_inds = self.neighb_rel_inds[self.Xneighbixs[center_ixs]]
        rel_inds = rel_inds[:, self.neighborhoods[ni]]
        shared = rel_inds < max_nc_obs
        cent = self.X[center_ixs].view(n_centers, self.feat_rank, max_nc_obs)
        cent = F.pad(cent, (0, 1))
        values = torch.take_along_dim(cent, rel_inds[:, None, :], dim=2)
        values = values.reshape(n_centers, self.feat_rank * max_nc_obs)
        return NeighborhoodCenters(
            values=values,
            shared=shared.to(values),
            normsq=values.square().sum(dim=1),
            n_feats=shared.sum(dim=1).mul(self.feat_rank).to(values),
        )

    def center_dists(
        self, wx: Tensor, xsq: Tensor, centers: NeighborhoodCenters
    ) -> Tensor:
        """(spikes, centers) distances of whitened spikes with per-channel norms xsq"""
        scale_std, scale_min, scale_max = self.scale_dist_args
        # foil, but only on overlap!
        if self.metric == "euc":
            d = xsq @ centers.shared.T
            d = d.addmm_(wx, centers.values.T, alpha=-2.0)
            return d.add_(centers.normsq).relu_().div_(centers.n_feats)
        elif self.metric == "cosine":
            return cosine_cdist_from_dots(
                wx @ centers.values.T, xsq @ centers.shared.T, centers.normsq
            )
        elif self.metric == "normeuc":
            return scaled_normeuc_cdist_from_dots(
                wx @ centers.values.T,
                xsq @ centers.shared.T,
                centers.normsq,
                scale_var=0.0,
            )
        elif self.metric == "scaled_normeuc":
            return scaled_normeuc_cdist_from_dots(
                wx @ centers.values.T,
                xsq @ centers.shared.T,
                centers.normsq,
                scale_var=scale_std**2,
                scale_min=scale_min,
                scale_max=scale_max,
            )
        else:
            panic(self.metric)

    def best_of_tries(
        self, base_rg: np.random.Generator
    ) -> tuple[Tensor, tuple[int, int, int]]:
        cfg = self.cfg
        n = self.X.shape[0]
        centroid_ixs: Tensor | None = None
        phase_centroid_counts: tuple[int, int, int] | None = None
        best_score = torch.inf
        best_phi = torch.inf
        centroid_cost = self.min_count * cfg.stop_dist**2 / n
        if logger.isEnabledFor(DARTSORTDEBUG):
            pbar = progrange(cfg.tries, desc="kmeans++ tries")
        else:
            pbar = None
        for _ in range(cfg.tries) if pbar is None else pbar:
            gen = spawn_torch_rg(base_rg, device=self.X.device)
            try_ixs, phi, try_phase_centroid_counts = self.one_try(gen)
            if cfg.selection == "phi":
                score = phi
            elif cfg.selection == "marginal":
                score = phi + centroid_cost * try_ixs.shape[0]
            else:
                panic(cfg.selection)
            logger.dartsortdebug(
                f"kmeans++ try: k={try_ixs.shape[0]} by phase {try_phase_centroid_counts} "
                f"phi={phi:.5f} score={score:.5f}"
            )
            if centroid_ixs is None or score < best_score:
                if pbar is not None:
                    pbar.set_description(
                        f"best: k={try_ixs.shape[0]} phi={phi:.4f} {cfg.selection}={score:.4f}"
                    )
                centroid_ixs = try_ixs
                phase_centroid_counts = try_phase_centroid_counts
                best_score = score
                best_phi = phi
        assert centroid_ixs is not None and phase_centroid_counts is not None
        logger.dartsortdebug(
            f"kmeans++: {centroid_ixs.shape[0]} centroids (phases: {phase_centroid_counts}) "
            f"at dist={math.sqrt(best_phi):0.3f}, best of {cfg.tries} by {cfg.selection} "
            f"(centroid cost {centroid_cost:0.3g} over {n} train spikes)."
        )
        return centroid_ixs, phase_centroid_counts

    def one_try(
        self, gen: torch.Generator
    ) -> tuple[Tensor, float, tuple[int, int, int]]:
        cfg = self.cfg
        n = self.X.shape[0]
        max_nc_obs = self.neighborhoods.shape[1]
        assert self.X.shape == (n, self.feat_rank * max_nc_obs)
        assert cfg.alpha > 0

        # Initialize
        train_centroid_ixs = self.X.new_full((self.max_k,), n, dtype=torch.long)
        distsq = self.X.new_full((n,), torch.inf)

        buf = torch.empty_like(distsq)
        wbuf = torch.empty_like(distsq)
        stop_distsq = cfg.stop_dist**2
        k = 0

        # -- phase 1: cover neighborhoods so dists are finite
        covered = self.counts == 0
        while k < self.max_k and not covered.all():
            (uncov,) = covered.logical_not().nonzero(as_tuple=True)
            nj = uncov[
                torch.multinomial(self.counts[uncov].to(self.X), 1, generator=gen)[0]
            ]
            j = torch.rand((), device=self.X.device, generator=gen)
            j = j.mul_(self.counts[nj]).long()
            train_centroid_ixs[k] = self.order[self.indptr[nj] + j]
            ix, d, _ = self.propose(train_centroid_ixs[k])
            if d is not None:
                _truncated_kmeanspp_commit_(distsq, ix, d)
            covered.logical_or_(self.visible[self.Xneighbixs[train_centroid_ixs[k]]])
            k += 1
        k_after_cover = k

        # -- phase 2: regular kmeanspp until well covered neighborhoods
        while k < self.max_k:
            mass = _truncated_kmeanspp_uncovered_mass(
                distsq, self.Xneighbixs, self.n_neighb, stop_distsq, buf
            )
            (unsat,) = (mass >= self.min_count).nonzero(as_tuple=True)
            if not unsat.numel():
                break
            w = _truncated_kmeanspp_weights(distsq, cfg.alpha, out=wbuf)
            train_centroid_ixs[k] = _one_gumbel_nolog(w, gen, buf)
            ix, d, _ = self.propose(train_centroid_ixs[k])
            if d is not None:
                _truncated_kmeanspp_commit_(distsq, ix, d)
            k += 1

        k_after_coverage = k

        # -- phase 3: with stopping and other criteria
        floor = self.min_count * stop_distsq
        centroid_budget = floor / n
        fails = 0
        phase3_enabled = cfg.stopping != "patience" or cfg.patience > 0
        while phase3_enabled and k < self.max_k:
            best_gain, best = -1.0, None
            w = _truncated_kmeanspp_weights(distsq, cfg.alpha, out=wbuf)
            # TODO remove greedy
            # TODO maybe remove alpha
            # TODO parallel proposals for patience: many proposals at once, take the first that passes
            # TODO proposals are entirely independent for neighborhoods that don't overlap, could segment the adjacency into a checkerboard...
            for _ in range(cfg.greedy_proposals):
                cix = _one_gumbel_nolog(w, gen, buf)
                ix, d, _ = self.propose(cix)
                if d is None:
                    continue
                gain = distsq[ix].sub_(d).clamp_(min=0.0).sum().item()
                if gain > best_gain:
                    best_gain, best = gain, (cix, ix, d)
            if cfg.stopping in ("patience", "patientdpmeanspp"):
                if best is None or best_gain < floor:
                    fails += 1
                    if fails >= cfg.patience:
                        break
                    continue
                fails = 0
            if cfg.stopping in ("dpmeanspp", "patientdpmeanspp"):
                if best is None:
                    break
                if distsq.mean().item() <= centroid_budget * k * (math.log2(k) + 2):
                    break
            if best is None:
                panic(cfg.stopping)
            train_centroid_ixs[k], ix, d = best
            _truncated_kmeanspp_commit_(distsq, ix, d)
            k += 1

        phi = distsq.mean().item()
        if not math.isfinite(phi):
            logger.warning(
                f"kmeans++ hit max_k={self.max_k}, leaving "
                f"{distsq.isinf().sum().item()} uncovered"
            )

        phase_centroid_counts = (
            k_after_cover,
            k_after_coverage - k_after_cover,
            k - k_after_coverage,
        )
        return train_centroid_ixs[:k], phi, phase_centroid_counts

    def propose(
        self, centroid_ix: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        return _truncated_kmeanspp_propose(
            X=self.X,
            centroid_ix=centroid_ix,
            Xneighbixs=self.Xneighbixs,
            neighborhoods=self.neighborhoods,
            neighb_rel_inds=self.neighb_rel_inds,
            visible=self.visible,
            order=self.order,
            indptr=self.indptr,
            counts=self.counts,
            feat_rank=self.feat_rank,
            metric=self.metric,
            scale_dist_args=self.scale_dist_args,
        )


truncated_kmeanspp = TruncatedKmeanspp.fit_predict


def _truncated_kmeanspp_weights(distsq: Tensor, alpha: float, out: Tensor) -> Tensor:
    if alpha == 2.0:
        return distsq
    return torch.pow(distsq, alpha / 2.0, out=out)


def _truncated_kmeanspp_uncovered_mass(
    distsq: Tensor,
    neighborhood_ids: Tensor,
    n_neighborhoods: int,
    stop_distsq: float,
    buf: Tensor,
) -> Tensor:
    """Sort of distance-limited soft count of uncovered spikes in each neighborhood"""
    w = torch.reciprocal(distsq, out=buf).mul_(-stop_distsq).add_(1.0).clamp_(min=0.0)
    mass = distsq.new_zeros(n_neighborhoods)
    return mass.index_add_(0, neighborhood_ids, w)


def _truncated_kmeanspp_propose(
    *,
    X: Tensor,
    centroid_ix: Tensor,
    Xneighbixs: Tensor,
    neighborhoods: Tensor,
    neighb_rel_inds: Tensor,
    visible: Tensor,
    order: Tensor,
    indptr: Tensor,
    counts: Tensor,
    feat_rank: int,
    metric: ComponentDistanceMetric,
    scale_dist_args: tuple[float, float, float],
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    cent_neighb = Xneighbixs[centroid_ix]
    (vis_neighbs,) = visible[cent_neighb].nonzero(as_tuple=True)
    ix = csr_members_ordered(order, indptr, counts, vis_neighbs)
    if not ix.numel():
        return ix, None, None

    max_nc_obs = neighborhoods.shape[1]
    rel_inds = neighb_rel_inds[cent_neighb][neighborhoods[Xneighbixs[ix]]]
    shared = rel_inds < max_nc_obs
    n_shared = shared.sum(dim=1)
    cent = F.pad(X[centroid_ix].view(feat_rank, max_nc_obs), (0, 1))
    cent = cent[:, rel_inds].permute(1, 0, 2)
    n_feats = n_shared.mul(feat_rank).to(X)
    xv = X[ix].view(ix.numel(), feat_rank, max_nc_obs)

    if metric == "euc":
        d = xv - cent
        d = d.square_().mul_(shared.unsqueeze(1)).sum(dim=(1, 2))
        d = d.div_(n_feats.clamp(min=1))
    elif metric == "cosine":
        d = cosine_cdist_from_dots(*_shared_dots_normsqs(xv, cent, shared))
        d = d[:, 0].square_()
    elif metric == "normeuc":
        d = scaled_normeuc_cdist_from_dots(
            *_shared_dots_normsqs(xv, cent, shared), scale_var=0.0
        )
        d = d[:, 0].square_()
    elif metric == "scaled_normeuc":
        scale_std, scale_min, scale_max = scale_dist_args
        d = scaled_normeuc_cdist_from_dots(
            *_shared_dots_normsqs(xv, cent, shared),
            scale_var=scale_std**2,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        d = d[:, 0].square_()
    else:
        panic(metric)
    d.masked_fill_(n_shared == 0, torch.inf)
    return ix, d, n_feats


def _shared_dots_normsqs(
    xv: Tensor, cent: Tensor, shared: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    # (n, 1) dots, (n,) and (n, 1) norms of spikes and centroid on each pair's shared chans
    mask = shared.unsqueeze(1).to(xv)
    xs = xv * mask
    cs = cent * mask
    dots = (xs * cs).sum(dim=(1, 2))[:, None]
    return dots, xs.square().sum(dim=(1, 2)), cs.square().sum(dim=(1, 2))[:, None]


def _truncated_kmeanspp_commit_(distsq: Tensor, ix: Tensor, d: Tensor) -> None:
    distsq[ix] = torch.minimum(distsq[ix], d, out=d)


def run_split(
    tmm: TruncatedMixtureModel,
    *,
    train_data: TruncatedSpikeData,
    val_data: TruncatedSpikeData | None,
    prog_level: int,
    single: bool = False,
    _stop_after: int | None = None,
    _dry_run: bool = False,
) -> SplitResult | None:
    split_res = tmm.split(
        ModelSelectionContext.from_model(
            tmm, train_data, val_data, show_progress=prog_level - 1
        ),
        show_progress=prog_level > 0,
        friend_distance=0.0 if single else None,
        _stop_after=_stop_after,
        _dry_run=_dry_run,
    )
    if _stop_after or _dry_run:
        return
    logger.info(
        f"Split created {split_res.n_new_units} new units and combined some to remove "
        f"{split_res.n_shrink} units, for count change "
        f"{split_res.n_new_units - split_res.n_shrink:+}."
    )
    return split_res


def run_merge(
    tmm: TruncatedMixtureModel,
    train_data: TruncatedSpikeData,
    val_data: TruncatedSpikeData | None,
    prog_level: int,
    full_data: BatchedSpikeData | None = None,
    original_sorting: DARTsortSorting | None = None,
) -> UnitRemapping:
    pair_mask = violation_pair_mask(
        tmm=tmm,
        full_data=full_data,
        original_sorting=original_sorting,
        prog_level=prog_level,
    )
    merge_map = tmm.merge(
        ModelSelectionContext.from_model(
            tmm, train_data, val_data, show_progress=prog_level - 1
        ),
        pair_mask=pair_mask,
        show_progress=prog_level > 0,
    )
    logger.info(f"Merge {merge_map.mapping.shape[0]} -> {merge_map.nuniq()} units.")
    return merge_map


def violation_pair_mask(
    *,
    tmm: TruncatedMixtureModel,
    full_data: BatchedSpikeData | None,
    original_sorting: DARTsortSorting | None,
    prog_level: int,
) -> Tensor | None:
    """Which pairs of units are allowed to merge by the refractory violation check

    Units are assigned over the whole dataset, rather than just the training
    subset, so that the violation counts get as much evidence as possible.
    Pairs whose jittered chance level is too small to say anything are filled
    with a ratio of 0, which allows their merge.
    """
    if tmm.p.merge_violation_threshold is None:
        return None
    if full_data is None or original_sorting is None:
        return None

    full_scores = tmm.soft_assign(
        data=full_data,
        full_proposal_view=True,
        needs_bootstrap=False,
        show_progress=prog_level - 1,
    )
    sorting = original_sorting.ephemeral_replace(labels=labels_from_scores(full_scores))
    violation = violation_statistics(
        sorting,
        censor_ms=tmm.p.merge_censor_ms,
        viol_ms=tmm.p.merge_violation_ms,
        jitter_ms=tmm.p.merge_jitter_ms,
    )
    assert violation.jitter_counts is not None
    ratio = violation.jitter_viol_ratio(
        tmm.p.merge_min_violation_evidence, fill_value=0.0
    )

    # violation_statistics sizes its matrices by the largest label present, so
    # dead units at the end of the id range are missing. they can merge.
    allowed = np.ones((tmm.n_units, tmm.n_units), dtype=bool)
    nu = ratio.shape[0]
    assert nu <= tmm.n_units
    allowed[:nu, :nu] = ratio < tmm.p.merge_violation_threshold

    # singleton parts are checked against the diagonal in allowed_partitions
    np.fill_diagonal(allowed, True)

    n_blocked = np.triu(np.logical_not(allowed), k=1).sum()
    logger.dartsortdebug(
        "Violation check blocks %d of %d unit pairs from merging.",
        n_blocked,
        (tmm.n_units * (tmm.n_units - 1)) // 2,
    )

    return torch.from_numpy(allowed)


def save_tmm_labels(
    *,
    tmm: TruncatedMixtureModel,
    stepname: str,
    save_step_labels_format: str | None,
    full_data: BatchedSpikeData,
    original_sorting: DARTsortSorting,
    save_step_labels_dir: Path | None,
    save_cfg: DARTsortInternalConfig | None,
    remove_noise: bool = True,
    full_proposal: bool = True,
):
    assert save_step_labels_format is not None
    full_scores = tmm.soft_assign(
        data=full_data,
        full_proposal_view=full_proposal,
        needs_bootstrap=not full_proposal,
    )
    sorting = original_sorting.ephemeral_replace(
        labels=labels_from_scores(full_scores, remove_noise=remove_noise)
    )
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
    noise: KroneckerNoise,
    erp: NeighborhoodFiller,
    prior_pseudocount: float,
    gen: torch.Generator,
    initial_basis_shrinkage: float = 1.0,
    scores: Scores | None = None,
    min_channel_count: int = 1,
    rank0_model: TruncatedMixtureModel | None = None,
    K: int | None = None,
    puff=1.0,
):
    if scores is None:
        assert data.candidates is not None
        labels = data.candidates[:, 0]
    else:
        labels = labels_from_scores_(scores)
    units, counts = labels.unique(return_counts=True)
    K_ = int(units.amax().item()) + 1
    if K is None:
        K = K_
    else:
        assert K >= K_
    counts = counts[units >= 0]
    units = units[units >= 0]
    feat_rank = noise.rank
    nc = data.neighborhoods.n_channels

    puff_K = max(K, int(puff * K))

    # do log_softmax in separate tensor (tmp) so that noise_log_prop is not
    # a view of an element of log_proportions buffer
    if rank0_model is None:
        log_proportions = torch.zeros((puff_K + 1,), device=counts.device)
        log_proportions = log_proportions.resize_(K)
        tmp = log_proportions.new_empty((K + 1,))
        tmp[units] = counts.float()
        tmp[-1] = tmp[:K].mean()
        tmp = F.log_softmax(tmp.log_(), dim=0)
        tmp = tmp.clamp_(min=TruncatedMixtureModel.LP_MIN)
        noise_log_prop = tmp[-1].clone()
        log_proportions.copy_(tmp[:K])
    else:
        log_proportions = rank0_model.b.log_proportions
        noise_log_prop = rank0_model.b.noise_log_prop

    dev = data.device
    if rank0_model is None:
        means = torch.zeros((puff_K, feat_rank, nc), device=dev)
        means = means.resize_(K, *means.shape[1:])
    else:
        means = None
    if signal_rank:
        bases = torch.zeros((puff_K, signal_rank, feat_rank, nc), device=dev)
        bases = bases.resize_(K, *bases.shape[1:])
    else:
        bases = None

    for k in units:
        kdata = data.dense_slice_by_unit(k, gen=gen, labels=None)
        assert kdata is not None
        if scores is not None:
            assert scores.responsibilities is not None
            if pnoid:
                assert (scores.candidates[kdata.indices, 0] == k).all()
            kweight = scores.responsibilities[kdata.indices, 0]
        else:
            kweight = None
        _, unit_means, unit_bases, _ = initialize_params_from_dense_data(
            kdata,
            erp=erp,
            rank=signal_rank,
            noise=noise,
            min_channel_count=min_channel_count,
            initial_basis_shrinkage=initial_basis_shrinkage,
            prior_pseudocount=prior_pseudocount,
            mean=rank0_model.b.means[k] if rank0_model is not None else None,
            single_weight=kweight,
        )
        if means is not None:
            means[k] = unit_means[0]
        if bases is not None:
            assert unit_bases is not None
            bases[k] = unit_bases[0]

    # flatten
    if means is not None:
        means = means.view(K, -1)
    if bases is not None:
        bases = bases.view(K, signal_rank, -1)

    return log_proportions, noise_log_prop, means, bases


def _desparsifiers(unit_ids, candidates):
    """Dense candidate array -> indices of >-1s and their positions in unit_ids"""
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
    """Basically scores.log_liks, but grabs and reorders to match unit_ids on dim 1."""
    ii, jj, ix_in_unit_ids = _desparsifiers(unit_ids, scores.candidates)
    n = scores.log_liks.shape[0]
    log_liks = scores.log_liks.new_full((n, unit_ids.shape[0]), -torch.inf)
    log_liks[ii, ix_in_unit_ids] = scores.log_liks[ii, jj]
    return log_liks


def get_responsibilities_matrix_from_scores(unit_ids: Tensor, scores: Scores) -> Tensor:
    """Basically scores.responsibilities, but grabs and reorders to match unit_ids on dim 1."""
    assert scores.responsibilities is not None
    ii, jj, ix_in_unit_ids = _desparsifiers(unit_ids, scores.candidates)
    n = scores.responsibilities.shape[0]
    responsibilities = scores.responsibilities.new_zeros((n, unit_ids.shape[0]))
    responsibilities[ii, ix_in_unit_ids] = scores.responsibilities[ii, jj]
    return responsibilities


def initialize_params_from_dense_data(
    data: DenseSpikeData,
    rank: int,
    erp: NeighborhoodFiller,
    noise: KroneckerNoise,
    prior_pseudocount: float,
    initial_basis_shrinkage: float = 1.0,
    mean: Tensor | None = None,
    min_channel_count: int = 1,
    weights: Tensor | None = None,
    single_weight: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    """Weighted mean and basis, possibly by multiple weight vectors at once.

    The chosen channel neighborhood is the union of my spikes' neighborhoods. If there
    are weights, the neighborhood will be refined for each weight vector.

    TODO: choosing to be bold for now and always interpolate densely to the channels.
    That's different from what was done before (per-channel missing weighting.) Can
    change this by supporting erp=None later here.
    """
    covered_chans = data.covered_channels(min_channel_count)
    nc = data.neighborhoods.n_channels

    x_erp = data.x()
    assert x_erp is not None
    x_erp = x_erp.view(data.N, noise.rank, -1)
    if weights is None:
        unit_weights = None if single_weight is None else single_weight[:, None]
        unit_masks = covered_chans.new_ones((1, covered_chans.numel()), dtype=torch.bool)
        full_coverage = None
        if mean is not None:
            means = mean.view(1, noise.rank, -1)
        elif single_weight is None:
            means = data.masked_observed_mean(x_erp)
        else:
            means = data.masked_observed_mean(
                x_erp, weights=unit_weights, pseudocount=prior_pseudocount
            )
    else:
        # weighted case. re-choose channel neighborhoods.
        assert mean is None
        assert single_weight is None
        unit_weights = weights
        unit_masks, full_coverage = data.weighted_covered_channels(
            weights=weights, within=covered_chans, min_count=min_channel_count
        )
        means = data.masked_observed_mean(
            x_erp, weights=weights, pseudocount=prior_pseudocount
        )
    K = means.shape[0]
    channel_mask = unit_masks.new_zeros((K, nc))
    channel_mask[:, covered_chans] = unit_masks
    masked_means = means * channel_mask[:, None, :].to(means)
    if not rank:
        return channel_mask, masked_means, None, full_coverage

    # imputed, centered data on covered chans
    source_channels = data.neighborhoods.b.neighborhoods[data.neighborhood_ids]
    source_means = F.pad(means, (0, 1))[:, :, source_channels]
    centered = x_erp[None] - source_means.permute(0, 2, 1, 3)
    centered = erp.interp_to_chans(
        centered.reshape(K * data.N, noise.rank, -1),
        data.neighborhood_ids.repeat(K),
        target_channels=covered_chans,
        mean=means.new_zeros((noise.rank, nc)),
    )
    centered = centered.view(K, data.N, noise.rank, covered_chans.numel())
    covered_bases = _initialize_bases(
        centered,
        unit_masks,
        covered_chans,
        noise,
        rank=rank,
        weights=unit_weights,
        prior_pseudocount=prior_pseudocount,
        initial_basis_shrinkage=initial_basis_shrinkage,
    )
    bases = covered_bases.new_zeros((K, rank, noise.rank, nc))
    bases[:, :, :, covered_chans] = covered_bases
    return channel_mask, masked_means, bases, full_coverage


def tree_groups(
    distances: Tensor, max_group_size: int, max_distance: float, link="complete"
) -> list[Tensor]:
    """Extract groups up to size max_group_size and (@link-linked) distances out to size max_distance."""
    k = distances.shape[0]
    if k <= max_group_size and distances.max() <= max_distance:
        # this would be common when splitting.
        return [torch.arange(k)]

    if pnoid:
        assert not distances.isnan().any()

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
    order = torch.argsort(torch.tensor([g[0] for g in groups])).tolist()
    groups = [groups[j] for j in order]
    return groups


def brute_merge(
    *,
    full_model: "TruncatedMixtureModel",
    full_train: ComponentBlockData,
    full_eval: ComponentBlockData | None,
    full_train_scores: Scores,
    proposals: list[MergeProposal],
    debug: bool = False,
) -> list[GroupMergeResult]:
    """Brute-force model selection

    Considering all partitions of cur_unit_ids into subgroups (obeying pair_mask),
    fit candidate models and check their heldout validation criterion using eval_data.
    Return the best one.

    This handles a block of groups / proposals, so many groups at once. The return is
    a list with the result for each proposal.
    """
    assert full_model.erp is not None
    assert full_model.noise is not None
    p = full_model.p
    results: list[GroupMergeResult] = [None] * len(proposals)

    # what possible subgroups are there?
    live = []
    n_units = []
    train_full_scores = []
    responsibilities = []
    partitions = []
    subset_resps = []
    for k, prop in enumerate(proposals):
        if (
            prop.pair_mask is not None
            and prop.pair_mask.sum() == prop.pair_mask.shape[0]
        ):
            continue
        if full_eval is not None and not full_eval.block_size(prop.block):
            logger.dartsortdebug(
                "Empty eval data (cur ids: %s)", prop.group.cpu().tolist()
            )
            continue

        # score train data with mm to get the full model log liks and the
        # responsibilities that will be used for fitting below
        n = int((full_train.block_unit_ids[prop.block] >= 0).sum())
        scores = _block_scores(
            full_train_scores, full_train.block_slice(prop.block), n, noise=True
        )
        if pnoid:
            block_units = full_train.block_unit_ids[prop.block, :n]
            assert torch.logical_or(
                scores.candidates == block_units.to(scores.candidates),
                scores.candidates == -1,
            ).all()
            assert (scores.candidates >= 0).any(dim=1).all()
        resp = prop.responsibilities
        if resp is None:
            resp = scores.log_liks.softmax(dim=1)[:, :n]

        parts, subset_to_id, id_to_subset = allowed_partitions(
            torch.arange(n, device=full_model.unit_ids.device), pair_mask=prop.pair_mask
        )
        assert len(subset_to_id) == len(id_to_subset)
        if debug:
            logger.dartsortdebug("brute_merge: %s partitions.", len(parts))
        if not id_to_subset:
            continue
        live.append(k)
        n_units.append(n)
        train_full_scores.append(scores)
        responsibilities.append(resp)
        partitions.append(parts)
        subset_resps.append(
            torch.stack(
                [resp[:, id_to_subset[s]].sum(dim=1) for s in range(len(id_to_subset))],
                dim=1,
            )
        )
    if not live:
        return results
    cur_scores = [proposals[k].cur_scores for k in live]
    live_blocks = [proposals[k].block for k in live]

    # fit subgroup models with fixed responsibilities
    subset_input, _ = full_train.select_blocks(live_blocks)
    n_subsets = [r.shape[1] for r in subset_resps]
    width = max(n_subsets)
    subset_fit = TruncatedMixtureModel.initialize_from_dense_data_and_responsibilities(
        data=subset_input.with_block_unit_ids(
            contiguous_block_unit_ids(torch.tensor(n_subsets))
        ),
        responsibilities=torch.concatenate(
            [F.pad(r, (0, width - r.shape[1])) for r in subset_resps]
        ),
        signal_rank=full_model.signal_rank,
        erp=full_model.erp,
        noise=full_model.noise,
        neighb_cov=full_model.neighb_cov,
        total_log_proportions=torch.stack(
            [
                full_model.b.log_proportions[
                    full_train.block_unit_ids[b, :n]
                ].logsumexp(dim=0)
                for b, n in zip(live_blocks, n_units, strict=True)
            ]
        ),
        p=p,
    )
    assert int(subset_fit.valid.sum()) == sum(n_subsets)
    subset_models = subset_fit.model
    subset_train = subset_fit.data
    if not subset_fit.kept_mask.all():
        for g in range(len(live)):
            kept_mask = subset_fit.kept_mask[subset_input.block_slice(g)]
            (kept,) = kept_mask.nonzero(as_tuple=True)
            responsibilities[g] = responsibilities[g][kept]
            subset_resps[g] = subset_resps[g][kept]
            train_full_scores[g] = train_full_scores[g].slice(kept)
            if full_eval is None:
                assert cur_scores[g].log_liks.shape[0] == kept_mask.shape[0]
                cur_scores[g] = cur_scores[g].slice(kept)

    # score the eval or train set with the all-subsets model
    train_subset_scores = subset_models.score_blocks(subset_train, skip_noise=True)
    if full_eval is None:
        crit_data = subset_train
        crit_full_scores = train_full_scores
        crit_subset_scores = [
            _block_scores(
                train_subset_scores,
                subset_train.block_slice(g),
                n_subsets[g],
                noise=False,
            )
            for g in range(len(live))
        ]
    else:
        crit_data, _ = full_eval.select_blocks(live_blocks)
        subset_eval = crit_data.with_block_unit_ids(subset_train.block_unit_ids)
        covered = (crit_data.unit_candidates(full_model.lut) >= 0).any(dim=1)
        covered.logical_and_(
            (subset_eval.unit_candidates(subset_models.lut) >= 0).any(dim=1)
        )
        for g in range(len(live)):
            (kept,) = covered[crit_data.block_slice(g)].nonzero(as_tuple=True)
            cur_scores[g] = cur_scores[g].slice(kept)
        if not covered.all():
            (kept,) = covered.nonzero(as_tuple=True)
            crit_data = crit_data.slice(kept)
            subset_eval = subset_eval.slice(kept)
        crit_full_all = full_model.score_blocks(crit_data, skip_noise=True)
        crit_subset_all = subset_models.score_blocks(subset_eval, skip_noise=True)
        crit_full_scores = [
            _block_scores(crit_full_all, crit_data.block_slice(g), n, noise=False)
            for g, n in enumerate(n_units)
        ]
        crit_subset_scores = [
            _block_scores(crit_subset_all, crit_data.block_slice(g), n, noise=False)
            for g, n in enumerate(n_subsets)
        ]

    for g, k in enumerate(live):
        if not crit_data.block_size(g):
            continue
        prop = proposals[k]

        # if demolition is happening, see where
        if p.demolish_during_selection:
            _cand = torch.arange(responsibilities[g].shape[1])[None]
            _cand = _cand.to(responsibilities[g].device)
            mean_train_resp = mean_responsibilities(
                responsibilities=responsibilities[g],
                candidates=_cand.broadcast_to(responsibilities[g].shape),
                n_units=responsibilities[g].shape[1],
            )
            mean_eval_resp = mean_responsibilities(
                scores=crit_full_scores[g],
                candidates=_cand.broadcast_to(crit_full_scores[g].log_liks.shape),
                n_units=responsibilities[g].shape[1],
            )
            can_demolish_mask = (
                mean_train_resp / mean_eval_resp
            ) > p.demolition_min_resp_ratio
        else:
            can_demolish_mask = None

        # identify best partition and extract its parameters and assignments
        in_crit_block = crit_data.block_slice(g)
        crit_weights = crit_data.data.sample_weights
        if crit_weights is not None:
            crit_weights = crit_weights[in_crit_block]
        assert cur_scores[g].log_liks.shape[1] == cur_scores[g].candidates.shape[1] + 1
        assert crit_full_scores[g].log_liks.shape[1] in (n_units[g], n_units[g] + 1)
        assert crit_subset_scores[g].log_liks.shape[1] == n_subsets[g]
        assert cur_scores[g].log_liks.shape[0] == crit_full_scores[g].log_liks.shape[0]
        best_part, best_imp = _select_partition(
            cur_unit_ids=prop.group.to(cur_scores[g].candidates.device),
            partitions=partitions[g],
            cur_scores=cur_scores[g],
            crit_full_scores=crit_full_scores[g],
            crit_subset_scores=crit_subset_scores[g],
            train_full_responsibilities=responsibilities[g],
            train_full_log_liks=train_full_scores[g].log_liks,
            cl_alpha=p.cl_alpha if prop.use_cl_alpha else 0.0,
            can_demolish_mask=can_demolish_mask,
            debug=debug,
            clip_level=p.criterion_clip_level,
            eval_weights=crit_weights,
        )

        in_block = subset_train.block_slice(g)
        full_units = full_train.block_unit_ids[prop.block, : n_units[g]]
        subset_units = subset_train.block_unit_ids[g, : n_subsets[g]]
        train_weights = subset_train.data.sample_weights
        if train_weights is not None:
            train_weights = train_weights[in_block]
        train_assignments, means, bases, sub_props = _extract_partition(
            part=best_part,
            responsibilities=responsibilities[g],
            subset_resps=subset_resps[g],
            train_full_scores=train_full_scores[g],
            train_subset_scores=_block_scores(
                train_subset_scores, in_block, n_subsets[g], noise=False
            ),
            full_means=full_model.b.means[full_units],
            full_bases=full_model.b.bases[full_units]
            if full_model.signal_rank
            else None,
            subset_means=subset_models.b.means[subset_units],
            subset_bases=(
                subset_models.b.bases[subset_units] if full_model.signal_rank else None
            ),
            train_weights=train_weights,
        )
        results[k] = SuccessfulGroupMergeResult(
            grouping=best_part,
            improvement=best_imp.item(),
            train_assignments=train_assignments,
            train_indices=subset_train.data.indices[in_block],
            means=means,
            bases=bases,
            sub_proportions=sub_props,
        )

    return results


def _block_scores(
    scores: Scores, in_block: slice, n_units: int, *, noise: bool
) -> Scores:
    log_liks = scores.log_liks[in_block]
    if noise:
        log_liks = torch.concatenate([log_liks[:, :n_units], log_liks[:, -1:]], dim=1)
    else:
        log_liks = log_liks[:, :n_units]
    duties = None if scores.duties is None else scores.duties[in_block, :n_units]
    assert scores.responsibilities is None
    return Scores(
        log_liks=log_liks,
        responsibilities=None,
        candidates=scores.candidates[in_block, :n_units],
        duties=duties,
    )


def allowed_partitions(
    unit_ids: Tensor,
    *,
    pair_mask: Tensor | None,
    skip_full: bool = False,
    skip_single: bool = False,
) -> tuple[list[GroupPartition], dict[tuple[int], int], dict[int, list[int]]]:
    n_units = unit_ids.numel()
    # this will be like 0,0,1,2 to indicate (uids[0],uids[1]),uids[2],uids[3]
    group_ids = torch.zeros_like(unit_ids)

    group_partitions = []
    subset_to_id: dict[tuple[int], int] = {}
    id_to_subset: dict[int, list[int]] = {}
    subset_id = 0
    for n_groups in range(1 + skip_single, n_units + 1 - skip_full):
        for partition in multiset_partitions(n_units, m=n_groups):
            # partition is like [0, 1, 2], [3, 4]
            subset_ids = []
            single_ixs = []
            uids_combined = []
            single_group_ids = []
            subset_group_ids = []
            for j, p in enumerate(partition):
                tp = tuple(p)
                p = torch.tensor(p)
                if len(tp) > 1 and tp not in subset_to_id:
                    subset_ids.append(subset_id)
                    subset_group_ids.append(j)
                    subset_to_id[tp] = subset_id
                    id_to_subset[subset_id] = p.tolist()
                    subset_id += 1
                    uids_combined.extend(unit_ids[p].tolist())
                elif len(tp) > 1:
                    subset_ids.append(subset_to_id[tp])
                    subset_group_ids.append(j)
                    uids_combined.extend(unit_ids[p].tolist())
                elif len(tp) == 1:
                    single_ixs.append(tp[0])
                    single_group_ids.append(j)
                else:
                    panic()
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
                    n_groups_before_demolish=n_groups,
                    single_ixs=single_ixs,
                    subset_ids=subset_ids,
                    unit_ids_combined=uids_combined,
                    # map single, subset ixs to group ids
                    single_subset_order=torch.argsort(
                        torch.asarray(single_group_ids + subset_group_ids)
                    ),
                    single_demolish_ixs=[],
                )
                group_partitions.append(group_partition)

    return group_partitions, subset_to_id, id_to_subset


def all_demolished_partitions(
    partitions: list[GroupPartition], can_demolish_mask: Tensor | None
) -> list[GroupPartition]:
    if can_demolish_mask is None or not can_demolish_mask.any():
        return partitions
    allp = []
    for part in partitions:
        if pnoid:
            assert (
                can_demolish_mask.shape == part.unit_ids.shape == part.group_ids.shape
            )
        if part.subset_ids:
            allp.append(part)
            continue

        single_ixs = torch.tensor(part.single_ixs, dtype=torch.long)
        npart = single_ixs.numel()
        (part_demo_ix,) = can_demolish_mask[single_ixs].nonzero(as_tuple=True)

        for demo_ixs in subsets(part_demo_ix.tolist()):
            demo_ixs = list(demo_ixs)
            if len(demo_ixs) == npart:
                continue
            demo_group_ids = part.group_ids.clone()
            demo_group_ids[single_ixs[demo_ixs]] = -1
            demo_part = replace(
                part,
                n_groups=part.n_groups - len(demo_ixs),
                group_ids=demo_group_ids,
                single_demolish_ixs=demo_ixs,
            )
            allp.append(demo_part)
    return allp


def _select_partition(
    *,
    cur_unit_ids: Tensor,
    partitions: list[GroupPartition],
    cur_scores: Scores,
    crit_full_scores: Scores,
    crit_subset_scores: Scores,
    cl_alpha: float,
    can_demolish_mask: Tensor | None,
    train_full_responsibilities: Tensor,
    train_full_log_liks: Tensor,
    debug: bool,
    clip_level: float | None = None,
    eval_weights: Tensor | None = None,
) -> tuple[GroupPartition, Tensor]:
    if eval_weights is not None:
        assert can_demolish_mask is None, "demolish during selection is unweighted"
        assert eval_weights.shape == (cur_scores.log_liks.shape[0],)
    per_spike = clip_level is not None or eval_weights is not None
    cur_mask = torch.isin(cur_scores.candidates, cur_unit_ids)
    cur_log_liks = cur_scores.log_liks
    if pnoid:
        assert cur_mask.any(dim=1).all()
    rest_logliks = cur_log_liks.clone()
    rest_logliks[:, :-1].masked_fill_(cur_mask, -torch.inf)

    # get current model criterion
    if cur_scores.responsibilities is None:
        cur_resp = cur_log_liks.softmax(dim=1)
    else:
        cur_resp = cur_scores.responsibilities
    assert cur_resp.shape == cur_scores.log_liks.shape
    del cur_scores

    cur_crit = ecl(cur_resp, cur_log_liks, cl_alpha=cl_alpha)
    if debug:
        h = entropy(cur_resp).item()
        logger.dartsortdebug(
            f"brute_merge cur score {cur_crit.item():.4f} entropy {h:.4f}"
        )
    if per_spike:
        cur_crit_per_spike = ecl(
            cur_resp, cur_log_liks, cl_alpha=cl_alpha, reduce_mean=False
        )
    else:
        cur_crit_per_spike = None
    del cur_resp

    # if destruction is allowed, consider partitions with single-unit destruction
    sel_partitions = all_demolished_partitions(partitions, can_demolish_mask)
    improvements = rest_logliks.new_full((len(sel_partitions),), -torch.inf)
    for j, part in enumerate(sel_partitions):
        single_ixs = torch.tensor(part.single_ixs, dtype=torch.long)
        subset_ids = torch.tensor(part.subset_ids, dtype=torch.long)
        single_demolish_ixs = torch.tensor(part.single_demolish_ixs, dtype=torch.long)
        part_score, part_h = _determine_part_score(
            rest_logliks=rest_logliks,
            crit_full_log_liks=crit_full_scores.log_liks,
            crit_subset_log_liks=crit_subset_scores.log_liks,
            train_full_responsibilities=train_full_responsibilities,
            train_full_log_liks=train_full_log_liks,
            single_ixs=single_ixs,
            subset_ids=subset_ids,
            single_demolish_ixs=single_demolish_ixs,
            cl_alpha=cl_alpha,
            return_entropy=debug,
        )
        if per_spike:
            assert cur_crit_per_spike is not None
            part_per_spike, _ = _determine_part_score(
                rest_logliks=rest_logliks,
                crit_full_log_liks=crit_full_scores.log_liks,
                crit_subset_log_liks=crit_subset_scores.log_liks,
                train_full_responsibilities=train_full_responsibilities,
                train_full_log_liks=train_full_log_liks,
                single_ixs=single_ixs,
                subset_ids=subset_ids,
                single_demolish_ixs=single_demolish_ixs,
                cl_alpha=cl_alpha,
                return_entropy=False,
                reduce_mean=False,
            )
            part_differences = part_per_spike - cur_crit_per_spike
        else:
            part_differences = None
        if clip_level is None and eval_weights is None:
            improvements[j] = part_score - cur_crit
        else:
            assert part_differences is not None
            if clip_level is None:
                selected_differences = part_differences
            else:
                selected_differences = part_differences.clamp(-clip_level, clip_level)
            if eval_weights is None:
                improvements[j] = selected_differences.mean()
            else:
                improvements[j] = (
                    selected_differences * eval_weights
                ).sum() / eval_weights.sum()
        if debug:
            assert part_h is not None
            h = part_h.item()
            part_imp = (part_score - cur_crit).item()
            logger.dartsortdebug(
                f"brute_merge {part} score {part_score.item():.4f} imp {part_imp:.4f} entropy {h:.4f}"
            )
            assert math.isfinite(h)
            assert math.isfinite(part_score.item())

    best_imp, best_ix = improvements.max(dim=0)
    best_part = sel_partitions[int(best_ix.item())]

    if pnoid:
        assert math.isfinite(cur_crit)
        assert math.isfinite(best_imp)

    return best_part, best_imp


def _determine_part_score(
    *,
    rest_logliks: Tensor,
    crit_full_log_liks: Tensor,
    crit_subset_log_liks: Tensor,
    train_full_responsibilities: Tensor,
    train_full_log_liks: Tensor,
    single_ixs: Tensor,
    subset_ids: Tensor,
    single_demolish_ixs: Tensor,
    cl_alpha: float,
    return_entropy: bool,
    reduce_mean: bool = True,
):
    single_ll = crit_full_log_liks[:, single_ixs]
    subset_ll = crit_subset_log_liks[:, subset_ids]
    part_logliks = torch.concatenate([rest_logliks, single_ll, subset_ll], dim=1)
    if single_demolish_ixs.numel():
        # determine and apply LP adjustment
        assert not subset_ids.numel()
        train_resp = train_full_responsibilities[:, single_ixs]
        mean_resp_adj = demolished_proportions(
            responsibilities=train_resp,
            log_liks=train_full_log_liks[:, single_ixs],
            demolish_ixs=single_demolish_ixs,
        )
        lp_adj = mean_resp_adj.log() - train_resp.mean(0).log()
        lp_adj.nan_to_num_(nan=0.0)
        lp_adj[single_demolish_ixs] = -torch.inf
        part_logliks[:, -lp_adj.numel() :] += lp_adj
    part_resps = part_logliks.softmax(dim=1)
    if return_entropy:
        h = entropy(part_resps)
    else:
        h = None
    return ecl(part_resps, part_logliks, cl_alpha=cl_alpha, reduce_mean=reduce_mean), h


def demolished_proportions(
    *, responsibilities: Tensor, log_liks: Tensor, demolish_ixs: Tensor
) -> Tensor:
    n, k = responsibilities.shape
    assert log_liks.shape == (n, k)
    survivors = torch.ones(k, dtype=torch.bool, device=responsibilities.device)
    survivors[demolish_ixs] = False

    freed = responsibilities[:, demolish_ixs].sum(dim=1)
    survivor_posterior = log_liks[:, survivors].softmax(dim=1).nan_to_num_()
    new_resp = torch.zeros_like(responsibilities)
    new_resp[:, survivors] = responsibilities[:, survivors].addcmul_(
        freed[:, None], survivor_posterior
    )

    props = responsibilities.mean(0)
    new_props = new_resp.mean(0)
    return new_props.mul_(props.sum() / new_props.sum())


def spike_mean(x: Tensor, weights: Tensor | None) -> Tensor:
    if weights is None:
        return x.mean(0)
    assert weights.shape == x.shape[:1]
    w = weights.view(-1, *([1] * (x.ndim - 1)))
    return (x * w).sum(0) / weights.sum()


def _extract_partition(
    *,
    part: GroupPartition,
    responsibilities: Tensor,
    subset_resps: Tensor,
    train_full_scores: Scores,
    train_subset_scores: Scores,
    full_means: Tensor,
    full_bases: Tensor | None,
    subset_means: Tensor,
    subset_bases: Tensor | None,
    train_weights: Tensor | None,
):
    # spike assignments
    invalid, train_assignments = get_part_assignments(
        part, train_full_scores.log_liks, train_subset_scores.log_liks
    )
    reorder = part.single_subset_order.to(device=train_assignments.device)
    train_assignments = torch.argsort(reorder)[train_assignments]
    train_assignments.masked_fill_(invalid, -1)

    # parameters
    means = torch.concatenate(
        (full_means[part.single_ixs], subset_means[part.subset_ids]), dim=0
    )
    means = means[reorder]
    if full_bases is not None:
        assert subset_bases is not None
        bases = torch.concatenate(
            (full_bases[part.single_ixs], subset_bases[part.subset_ids]), dim=0
        )
        bases = bases[reorder]
    else:
        assert subset_bases is None
        bases = None

    # fraction
    if part.single_demolish_ixs:
        assert not part.subset_ids
        assert train_weights is None
        single_prop = demolished_proportions(
            responsibilities=responsibilities[:, part.single_ixs],
            log_liks=train_full_scores.log_liks[:, part.single_ixs],
            demolish_ixs=torch.tensor(part.single_demolish_ixs),
        )
    else:
        single_prop = spike_mean(responsibilities, train_weights)[part.single_ixs]
    sub_props = [
        spike_mean(subset_resps[:, sid], train_weights)[None] for sid in part.subset_ids
    ]
    sub_props = torch.concatenate([single_prop, *sub_props], dim=0)
    sub_props = sub_props[reorder]
    assert sub_props.shape == (len(part.single_ixs) + len(part.subset_ids),)
    assert sub_props.shape == (part.n_groups + len(part.single_demolish_ixs),)
    sub_props /= sub_props.sum()
    if pnoid:
        assert sub_props.isfinite().all()

    return train_assignments, means, bases, sub_props


def get_part_assignments(
    part: GroupPartition, full_scores: Tensor, subset_scores: Tensor
):
    single_scores = full_scores[:, part.single_ixs]
    if part.single_demolish_ixs:
        single_scores[:, part.single_demolish_ixs] = -torch.inf
    sub_scores = [subset_scores[:, sid, None] for sid in part.subset_ids]
    sub_scores = torch.concatenate([single_scores, *sub_scores], dim=1)
    assignments = sub_scores.argmax(dim=1, keepdim=True)
    neginf = torch.isneginf(sub_scores.take_along_dim(assignments, 1))
    assignments.masked_fill_(neginf, -1)
    neginf = neginf.view(assignments.shape[0])
    assignments = assignments.view(assignments.shape[0])
    return neginf, assignments


def try_kmeans(
    *,
    data: DenseSpikeData,
    k: int,
    erp: NeighborhoodFiller,
    gen: torch.Generator,
    min_count: int,
    min_channel_count: int,
    feature_rank: int,
    n_iter: int = 100,
    n_kmeans_tries: int = 25,
    kmeans_beta: float = 1.0,
    n_kmeanspp_tries: int = 25,
    debug: bool = False,
    whiten: bool = False,
    bail_at: int = 1,
    force_labels: Tensor | None = None,
    kmeanspp_metric: SplitKmeansppMetric = "euc",
    scale_dist_args: tuple[float, float, float] | None = None,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    # interpolate whitened data
    channels = data.covered_channels(min_channel_count)
    if whiten:
        erp_x = data.whitenedx
    else:
        erp_x = data.x()
        assert erp_x is not None
    erp_x = erp_x.view(data.N, feature_rank, -1)
    mean = data.masked_observed_mean(erp_x)[0]
    x = erp.interp_to_chans(erp_x, data.neighborhood_ids, channels, mean=mean)
    x = x.view(len(x), -1)
    x_ret = x if debug else None

    if force_labels is not None:
        # debugging vis path.
        _, labels = force_labels.to(x.device).unique(return_inverse=True)
        e = F.one_hot(labels).to(x)
        log_props = e.sum(0).div(e.sum()).log_()
        centroids = (e / e.sum(0)).T @ x
        dists = torch.cdist(x, centroids).square_()
        resps = log_props.sub(dists, alpha=0.5 * kmeans_beta).softmax(dim=1)
    else:
        kres = batched_kmeans(
            x,
            k,
            seed=gen,
            n_iter=n_iter,
            kmeanspp_seeds_per_try=n_kmeanspp_tries,
            n_tries=n_kmeans_tries,
            beta=kmeans_beta,
            kmeanspp_metric=kmeanspp_metric,
            scale_dist_args=scale_dist_args,
        )
        resps = kres.responsibilities
    if resps is None:
        return None, x_ret, channels
    assert resps.shape[1] <= k or force_labels is not None
    big_enough_mask = resps.sum(dim=0) >= min_count
    n_big = big_enough_mask.sum().cpu().item()
    if n_big <= bail_at:
        return None, x_ret, channels
    elif n_big < resps.shape[1]:
        resps = _combine_similar_resps(resps, big_enough_mask, n_big)
    return resps, x_ret, channels


def evaluate_group_demolitions(
    *,
    mm: TruncatedMixtureModel,
    group: Tensor,
    train_data: TruncatedSpikeData,
    eval_data: TruncatedSpikeData,
    mean_train_resp: Tensor | None,
    mean_eval_resp: Tensor | None,
    train_scores: Scores,
    eval_scores: Scores,
    eval_row_crit: Tensor | None = None,
) -> GroupDemolition:
    group_ = group.to(train_scores.candidates)
    train_weights = train_data.sample_weights
    eval_weights = eval_data.sample_weights

    # select which units in the group are candidates for demolition
    if mean_train_resp is None:
        mean_train_resp = mean_responsibilities(
            scores=train_scores, n_units=mm.n_units, weights=train_weights
        )
    if mean_eval_resp is None:
        mean_eval_resp = mean_responsibilities(
            scores=eval_scores, n_units=mm.n_units, weights=eval_weights
        )
    ratio = mean_train_resp[group] / mean_eval_resp[group]
    can_demolish = ratio > mm.p.demolition_min_resp_ratio

    if not can_demolish.any():
        return GroupDemolition(unit_ids=group, improvement=0.0, demolished=None)

    # criterion:
    # loop over possible demolitions, evaluate their improvements, track best
    alpha = mm.p.demolish_cl_alpha
    if eval_row_crit is None:
        eval_row_crit = proportion_adjusted_row_criterion(
            log_liks=eval_scores.log_liks,
            candidates=eval_scores.candidates,
            orig_log_props=mm.b.log_proportions,
            new_log_props=demolition_log_proportions(
                mean_train_resp, mm.b.noise_log_prop, mm.n_units
            ),
            cl_alpha=alpha,
        )
    best_demo = GroupDemolition(
        unit_ids=group, improvement=0.0, demolished=torch.zeros_like(can_demolish)
    )
    best_imp = 0.0

    for demo_mask in submasks(can_demolish, skip_full=True):
        imp = _evaluate_single_demolition(
            orig_log_props=mm.b.log_proportions,
            noise_log_prop=mm.b.noise_log_prop,
            cl_alpha=alpha,
            group=group_,
            demolish_mask=demo_mask,
            train_scores=train_scores,
            eval_scores=eval_scores,
            eval_row_crit=eval_row_crit,
            train_weights=train_weights,
            eval_weights=eval_weights,
        )
        if imp > best_imp:
            best_demo = GroupDemolition(
                unit_ids=group, improvement=imp, demolished=demo_mask
            )
            best_imp = imp

    return best_demo


def _evaluate_single_demolition(
    orig_log_props: Tensor,
    noise_log_prop: Tensor,
    cl_alpha: float,
    group: Tensor,
    demolish_mask: Tensor,
    train_scores: Scores,
    eval_scores: Scores,
    eval_row_crit: Tensor,
    train_weights: Tensor | None = None,
    eval_weights: Tensor | None = None,
) -> float:
    n_units = orig_log_props.numel()
    assert demolish_mask.shape == group.shape

    # determine mean responsibility after demolition on train set
    chopping_block = group[demolish_mask]
    if not chopping_block.numel():
        return 0.0
    adj_train_resp = mean_responsibilities_after_removal(
        scores=train_scores,
        remove_units=chopping_block,
        n_units=n_units,
        weights=train_weights,
    )
    assert adj_train_resp.shape == (n_units + 1,)

    # determine what adjustment of proportions would result
    new_log_props = demolition_log_proportions(adj_train_resp, noise_log_prop, n_units)

    # evaluate effect on heldout set
    return proportion_adjust_criterion_change(
        scores=eval_scores,
        orig_log_props=orig_log_props,
        new_log_props=new_log_props,
        cl_alpha=cl_alpha,
        orig_row_criterion=eval_row_crit,
        weights=eval_weights,
    )


def demolition_log_proportions(
    mean_resp: Tensor, noise_log_prop: Tensor, n_units: int
) -> Tensor:
    unit_resp = mean_resp[:n_units]
    non_noise_lp = torch.log1p(-noise_log_prop.exp())
    return (unit_resp / unit_resp.sum()).log() + non_noise_lp


def submasks(mask: Tensor, *, skip_empty=True, skip_full=False):
    mask_ = mask.cpu()
    (on,) = mask_.nonzero(as_tuple=True)
    on = on.tolist()
    for subset in subsets(on):
        if skip_empty and not len(subset):
            continue
        if skip_full and len(subset) == len(mask):
            continue
        m = torch.zeros_like(mask)
        m[list(subset)] = True
        yield m


def labels_from_scores_(scores: Scores, *, remove_noise: bool = True) -> Tensor:
    """Pick either top candidate or noise."""
    if pnoid:
        nc = scores.candidates.shape[1]
        assert (scores.log_liks[:, 0, None] >= scores.log_liks[:, :nc]).all()
    labels = scores.candidates[:, 0].clone()
    if remove_noise:
        noise_better = scores.log_liks[:, -1] > scores.log_liks[:, 0]
        labels.masked_fill_(noise_better, -1)
    return labels


def labels_from_scores(scores: Scores, *, remove_noise: bool = True) -> np.ndarray:
    return labels_from_scores_(scores, remove_noise=remove_noise).numpy(force=True)


def relabel_and_add_scores(
    sorting: DARTsortSorting,
    tmm: TruncatedMixtureModel,
    full_data: BatchedSpikeData,
    *,
    full_proposal_view=True,
    needs_bootstrap=False,
) -> DARTsortSorting:
    full_scores = tmm.soft_assign(
        data=full_data,
        full_proposal_view=full_proposal_view,
        needs_bootstrap=needs_bootstrap,
    )

    labels = labels_from_scores(full_scores)
    neighb_ids = full_data.neighborhood_ids.numpy(force=True).copy()
    candidates = full_scores.candidates.numpy(force=True).astype(np.int32)
    log_liks = full_scores.log_liks.numpy(force=True)
    assert full_scores.responsibilities is not None
    resp = full_scores.responsibilities.numpy(force=True)

    sorting = sorting.ephemeral_replace(
        labels=labels,
        neighborhood_ids=neighb_ids,
        gmm_candidates=candidates,
        gmm_log_liks=log_liks,
        gmm_responsibilities=resp,
    )
    return sorting


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


def _neighb_relative_index(
    neighborhoods: Tensor, n_channels: int | None = None
) -> Tensor:
    """rel_inds[j, c] = index of c in neighborhoods[j] else max_nc_obs"""
    nneighb, max_nc_obs = neighborhoods.shape
    if n_channels is None:
        # this allows supporting non-channel_index neighborhooods
        n_channels = nneighb
    rel_inds = neighborhoods.new_full((nneighb, n_channels + 1), max_nc_obs)
    positions = torch.arange(max_nc_obs, device=neighborhoods.device)
    rel_inds.scatter_(1, neighborhoods, positions.broadcast_to(neighborhoods.shape))
    # a spike's padding is shared with nobody
    rel_inds[:, n_channels] = max_nc_obs
    return rel_inds


def _noise_factors(*, noise: KroneckerNoise, obs_ix: Tensor, miss_near_ix: Tensor):
    nneighb = obs_ix.shape[0]
    nc_obs = obs_ix.shape[1]
    nc_miss_near = miss_near_ix.shape[1]
    rank = noise.rank
    nc = noise.n_channels
    dev = obs_ix.device

    logdet = torch.zeros((nneighb,), device=dev)
    spatial_whitener = torch.zeros((nneighb, nc_obs, nc_obs), device=dev)
    spatial_regression = torch.zeros((nneighb, nc_obs, nc_miss_near), device=dev)
    spatial_infos = torch.zeros((nneighb,), dtype=torch.int, device=dev)
    temporal_logdet = -2.0 * noise.b.temporal_whitener.diagonal().log().sum()

    for j, (joix, jmnix) in enumerate(zip(obs_ix, miss_near_ix, strict=True)):
        (joixvix,) = (joix < nc).nonzero(as_tuple=True)
        joixv = joix[joixvix]
        (jmnixvix,) = (jmnix < nc).nonzero(as_tuple=True)
        jmnixv = jmnix[jmnixvix]
        ncoi = joixv.numel()
        ncmi = jmnixv.numel()
        assert ncoi > 0

        # -- observed-only stuff
        # calculate observed-only stuff
        jspatial_cov = noise.b.spatial_cov[joixv][:, joixv]
        jspatial_L, spatial_infos[j] = torch.linalg.cholesky_ex(jspatial_cov)
        jspatial_Linv = torch.linalg.solve_triangular(
            jspatial_L, torch.eye(ncoi).to(jspatial_L), upper=False
        )
        jspatial_logdet = 2.0 * jspatial_L.diagonal().log().sum()
        jlogdet = ncoi * temporal_logdet + rank * jspatial_logdet

        # set observed-only stuff
        logdet[j] = jlogdet
        spatial_whitener[j, joixvix[:, None], joixvix[None]] = jspatial_Linv

        # -- observed-missing stuff
        if ncmi == 0:
            # i am a lonely island
            # importantly, that means my Com is 0. so everything is 0 below.
            continue
        jspatial_Com = noise.b.spatial_cov[joixv][:, jmnixv]
        jspatial_CooinvCom = jspatial_Linv.T @ (jspatial_Linv @ jspatial_Com)
        spatial_regression[j, joixvix[:, None], jmnixvix[None]] = jspatial_CooinvCom

    nbail = int(spatial_infos.count_nonzero())
    if nbail:
        raise ValueError(
            f"Noise covariance not positive definite in {nbail} neighborhoods."
        )
    return logdet, spatial_whitener, spatial_regression


def _huber_twicerho_and_weight(sqdist: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    r = sqdist.clamp_(min=0.0).sqrt_()
    c = torch.minimum(r, k)
    weight = k.div(r).clamp_(max=1.0)
    twicerho = r.mul_(2.0).sub_(c).mul_(c)
    return twicerho, weight


def _whiten_and_noise_score(
    *,
    x: Tensor,
    neighborhoods: SpikeNeighborhoods,
    neighb_cov: NeighborhoodCovariance,
    batch_size=1024,
) -> tuple[Tensor, Tensor]:
    dev = cast(torch.device, neighborhoods.neighborhood_ids.device)
    on_device = x.device == dev
    if on_device:
        wx = torch.empty_like(x)
        noise_loglik = wx.new_zeros((len(x),))
    else:
        assert x.device.type == "cpu"
        wx = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
        noise_loglik = torch.zeros((len(x),), dtype=x.dtype, pin_memory=True)
    assert neighborhoods.n_neighborhoods == neighb_cov.b.obs_ix.shape[0]
    rank = neighb_cov.feat_rank
    nco = neighb_cov.max_nc_obs
    temporal_whitener = neighb_cov.b.temporal_whitener

    for ni in range(neighborhoods.n_neighborhoods):
        inni = neighborhoods.neighborhood_members(ni)
        if not on_device:
            inni = inni.cpu()

        spatial_right_factor = neighb_cov.b.spatial_whitener[ni].T
        nll_const = neighb_cov.b.lik_const[ni]
        huber_k = None if neighb_cov.huber_k is None else neighb_cov.b.huber_k[ni]

        for bs in range(0, inni.numel(), batch_size):
            binni = inni[bs : bs + batch_size]
            xb = x[binni].to(device=dev).view(binni.numel(), rank, nco)
            wxb = temporal_whitener @ xb @ spatial_right_factor
            wxb = wxb.view(binni.numel(), rank * nco)

            wx[binni] = wxb.to(wx.device)

            nll = wxb.square_().sum(dim=1)
            if huber_k is not None:
                nll, _ = _huber_twicerho_and_weight(nll, huber_k)
            nll += nll_const
            nll *= -0.5
            assert nll.isfinite().all()
            noise_loglik[binni] = nll.to(noise_loglik.device)

    return wx, noise_loglik


def _whiten_impute_and_noise_score(
    *,
    x: Tensor,
    neighborhoods: SpikeNeighborhoods,
    neighb_cov: NeighborhoodCovariance,
    batch_size=1024,
):
    dev = cast(torch.device, neighborhoods.neighborhood_ids.device)
    on_device = x.device == dev
    cmo_shape = (len(x), neighb_cov.max_nc_miss_near, neighb_cov.feat_rank)
    if on_device:
        wx = torch.empty_like(x)
        noise_loglik = wx.new_zeros((len(x),))
        CmoCooinvx = wx.new_empty(cmo_shape)
    else:
        assert x.device.type == "cpu"
        wx = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
        noise_loglik = torch.zeros((len(x),), dtype=x.dtype, pin_memory=True)
        CmoCooinvx = torch.empty(cmo_shape, dtype=x.dtype, pin_memory=True)
    assert neighborhoods.n_neighborhoods == neighb_cov.b.obs_ix.shape[0]
    rank = neighb_cov.feat_rank
    nco = neighb_cov.max_nc_obs
    temporal_whitener = neighb_cov.b.temporal_whitener

    for ni in range(neighborhoods.n_neighborhoods):
        inni = neighborhoods.neighborhood_members(ni)
        if not on_device:
            inni = inni.cpu()

        spatial_right_factor = neighb_cov.b.spatial_whitener[ni].T
        spatial_regression = neighb_cov.b.spatial_regression[ni]

        nll_const = neighb_cov.b.lik_const[ni]
        huber_k = None if neighb_cov.huber_k is None else neighb_cov.b.huber_k[ni]

        for bs in range(0, inni.numel(), batch_size):
            binni = inni[bs : bs + batch_size]
            xb = x[binni].to(dev).view(binni.numel(), rank, nco)
            wxb = temporal_whitener @ xb @ spatial_right_factor
            wxb = wxb.view(binni.numel(), rank * nco)

            wx[binni] = wxb.to(wx.device)

            CmoCooinvx_batch = xb @ spatial_regression
            CmoCooinvx[binni] = CmoCooinvx_batch.mT.to(CmoCooinvx.device)

            nll = wxb.square_().sum(dim=1)
            if huber_k is not None:
                nll, _ = _huber_twicerho_and_weight(nll, huber_k)
            nll += nll_const
            nll *= -0.5
            assert nll.isfinite().all()
            noise_loglik[binni] = nll.to(noise_loglik.device)

    xt = x.view(x.shape[0], neighb_cov.feat_rank, neighb_cov.max_nc_obs)
    if on_device:
        xt = xt.transpose(1, 2).contiguous()
    else:
        xt = _pinned_copy(xt.transpose(1, 2))

    return xt, wx, CmoCooinvx, noise_loglik


def _pinned_copy(x: Tensor) -> Tensor:
    pinned = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
    pinned.copy_(x)
    return pinned


def _whiten_and_noise_score_batch(
    *, x: Tensor, neighb_ids: Tensor, neighb_cov: NeighborhoodCovariance
):
    return _whiten_and_noise_score_batch_k(
        x,
        neighb_ids,
        neighb_cov.b.temporal_whitener,
        neighb_cov.b.spatial_whitener,
        neighb_cov.b.lik_const,
        neighb_cov.huber_k,
    )


@torch_compiler(fullgraph=False)
def _whiten_and_noise_score_batch_k(
    x: Tensor,
    neighb_ids: Tensor,
    temporal_whitener: Tensor,
    spatial_whitener: Tensor,
    lik_const: Tensor,
    huber_k: Tensor | None,
):
    n = x.shape[0]
    x_rc = x.reshape(n, temporal_whitener.shape[0], spatial_whitener.shape[1])
    wx = (temporal_whitener @ x_rc).bmm(spatial_whitener[neighb_ids].mT).view(n, -1)
    nll = torch.linalg.vector_norm(wx, dim=1).square_()
    if huber_k is not None:
        nll, _ = _huber_twicerho_and_weight(nll, huber_k[neighb_ids])
    const = lik_const[neighb_ids]
    nll += const
    nll *= -0.5
    return wx, nll


def _initialize_bases(
    centered: Tensor,
    channel_mask: Tensor,
    chans: Tensor,
    noise: KroneckerNoise,
    *,
    rank: int,
    initial_basis_shrinkage: float = 1.0,
    prior_pseudocount: float = 0.0,
    weights: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Initialize a component's basis, or a group of components' bases

    Initializes the basis. In the unweighted case, the algorithm is:
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
    K, n, d, c = centered.shape
    assert channel_mask.shape == (K, c)
    assert rank > 0

    # we want x(C^-0.5)=xU^-1 -- need to use upper factor.
    # noise is Kron, so time and space are separate and the spatial part varies
    # by unit according to its channel neighborhood (eye padded)
    temporal_factor = torch.linalg.cholesky(noise.b.temporal_cov)
    unit_pairs = channel_mask[:, :, None].logical_and(channel_mask[:, None, :])
    spatial_cov = noise.b.spatial_cov[chans][:, chans]
    eye = torch.eye(c, device=spatial_cov.device, dtype=spatial_cov.dtype)
    spatial_factor = torch.linalg.cholesky(torch.where(unit_pairs, spatial_cov, eye))
    x = centered * channel_mask[:, None, None, :].to(centered)
    x = torch.linalg.solve_triangular(temporal_factor, x, upper=False)
    x = torch.linalg.solve_triangular(
        spatial_factor[:, None].mT, x, upper=True, left=False
    )
    x = x.reshape(K, n, d * c)

    # weighting the covariance
    if weights is None:
        wsum = x.new_full((K,), float(n))
        gram = x.mT @ x
    else:
        assert weights.shape == (n, K)
        wsum = weights.sum(dim=0)
        weights = weights.T
        if prior_pseudocount:
            weights = weights * (wsum / (wsum + prior_pseudocount))[:, None]
        gram = (x * weights[:, :, None]).mT @ x

    # top eigenpairs, in decreasing order, converted to covariance eigvals
    s, V = torch.linalg.eigh(gram)
    s = s[:, -rank:].flip(-1)
    V = V[:, :, -rank:].flip(-1)
    s = s / (wsum - 1.0).clamp(min=1.0)[:, None]

    # svd flipper
    flip = sign(V.gather(1, V.abs().argmax(dim=1, keepdim=True)))

    # extra 1.0 was picked up from identity in whitened space, remove and
    # compute the low rank factor that remains, which is our basis (scaled)
    s = s.sub_(1.0).clamp_(min=eps)
    ev = s.sqrt_() / initial_basis_shrinkage
    W = V * (ev[:, None, :] * flip)
    W = W.view(K, d, c, rank).permute(0, 3, 1, 2)
    W = temporal_factor @ W @ spatial_factor[:, None].mT
    return W * channel_mask[:, None, None, :].to(W)


# -- candidate set helpers


def coincidence_matrix(
    x: Tensor, y: Tensor, nx: int, ny: int, dtype=torch.long, batch_size: int = 2**18
) -> Tensor:
    """Get a confusion matrix of sorts: C[p,q] is |{i:x[i]=p,y[i]=q}|.

    x and y should be int/long dtype.

    x can have -1, y can't. x can be 1d or 2d, y matches first dim.
    Similar to but faster than stack(x,y).unique(0), especially if you
    need this matrix.

    The -1s in x are routed into a dump row which is dropped on the way out, so
    that no index has to be materialized for the valid entries: nonzero() on an
    x-shaped mask costs several times x itself, and x here is the full candidate
    set.
    """
    assert x.shape[:1] == y.shape
    if x.ndim == 1:
        x = x[:, None]
    assert x.ndim == 2

    if pnoid:
        assert x.max() < nx, f"{x.max()} {nx}"
        assert y.max() < ny, f"{y.max()} {ny}"
        assert y.min() >= 0

    co = torch.zeros(((nx + 1) * ny,), dtype=dtype, device=x.device)
    one = torch.ones((1,), dtype=dtype, device=co.device)
    for i0 in range(0, x.shape[0], batch_size):
        i1 = min(x.shape[0], i0 + batch_size)
        iflat = torch.where(x[i0:i1] >= 0, x[i0:i1], nx)
        iflat = iflat.mul_(ny).add_(y[i0:i1, None]).view(-1)
        co.scatter_add_(dim=0, index=iflat, src=one.broadcast_to(iflat.shape))

    return co.view(nx + 1, ny)[:nx]


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
):
    if un_adj_lut is None:
        assert labels is not None
        un_adj_lut = NeighborhoodLUT.from_candidates_and_neighborhoods(
            candidates=labels,
            neighborhoods=neighborhoods,
            neighborhood_ids=neighborhood_ids,
            n_units=n_units,
            neighb_supset=neighb_supset,
        )
    if expand_from_lut is not None:
        un_adj_lut = combine_luts(expand_from_lut, un_adj_lut)

    # lut -> adjacency matrix
    un_adj = un_adj_lut.b.lut < un_adj_lut.b.unit_ids.shape[0]
    assert un_adj.any()
    un_adj = un_adj.float()

    # constraint for explore candidates is looser
    explore_adj = un_adj
    for _ in range(explore_steps):
        explore_adj = explore_adj @ neighb_adj

    return un_adj_lut, un_adj, explore_adj


def combine_luts(*luts: NeighborhoodLUT) -> NeighborhoodLUT:
    assert len(luts) > 0
    if len(luts) == 1:
        return luts[0]
    adj = luts[0].b.lut < luts[0].b.unit_ids.shape[0]
    for ll in luts[1:]:
        adj.logical_or_(ll.lut < ll.unit_ids.shape[0])  # type: ignore  # ty: ignore[x]
    unit_ids, neighb_ids = adj.nonzero(as_tuple=True)
    lut = torch.full_like(luts[0].b.lut, unit_ids.shape[0])
    lut[unit_ids, neighb_ids] = torch.arange(unit_ids.shape[0], device=lut.device)
    return NeighborhoodLUT(unit_ids=unit_ids, neighb_ids=neighb_ids, lut=lut)


def candidate_search_sets(
    distances: Tensor,
    un_adj_lut: NeighborhoodLUT,
    un_adj: Tensor,
    n_search: int,
    eps: float = 1e-8,
):
    n_lut = un_adj_lut.b.unit_ids.shape[0]

    # get lut-indexed version with extra lut index for topk below
    s = distances[un_adj_lut.b.unit_ids]

    # don't want to match with myself
    inf = s.new_full((1, 1), torch.inf).broadcast_to((n_lut, 1))
    s.scatter_(dim=1, index=un_adj_lut.b.unit_ids[:, None], src=inf)

    # avoid nans if there are exact 0 distances
    s.clamp_(min=eps)

    # flip scale so larger is better
    s.reciprocal_()

    # multiply by neighborhood-unit adjacency to set non olap to 0
    s *= un_adj.T[un_adj_lut.b.neighb_ids, : distances.shape[0]]

    # take topk, and fill invalids (s=0) with -1
    tops, topunits = torch.topk(s, k=n_search, dim=1)
    topunits.masked_fill_(tops == 0, -1)
    # pad with row of -1s for the invalid lut ixs (===n_lut)
    topunits = F.pad(topunits, (0, 0, 0, 1), value=-1).long()

    return topunits


def max_units_per_neighb(lut: NeighborhoodLUT):
    coverage = lut.b.lut < lut.b.unit_ids.shape[0]
    return int(coverage.sum(dim=0).amax())


def full_proposal_by_neighb(lut: NeighborhoodLUT, max_proposed: int):
    """Returns n_neighbs x max possible proposed units array, basically transposing the LUT's nonzeros."""
    n_neighbs = lut.b.lut.shape[1]
    n_lut = lut.b.unit_ids.shape[0]
    proposals = torch.full((n_neighbs, max_proposed), -1, dtype=torch.long)
    for neighb_id in range(n_neighbs):
        row = lut.b.lut[:, neighb_id]
        (row_valid,) = (row < n_lut).nonzero(as_tuple=True)
        n_row = row_valid.numel()
        assert n_row <= max_proposed
        proposals[neighb_id, :n_row] = lut.b.unit_ids[row[row_valid]]
    return proposals


def _fill_blank_labels(
    labels: Tensor,
    un_adj: Tensor,
    *,
    explore_adj: Tensor,
    neighborhood_ids: Tensor,
    neighb_adj: Tensor,
    gen: torch.Generator,
    batch_size: int = 1024,
    ensure_coverage_only: bool = False,
    allow_uncovered: bool = False,
    allow_new_units: bool = False,
    next_label: int | None = None,
    max_steps=2,
):
    assert neighborhood_ids.shape == labels.shape
    assert un_adj.shape == explore_adj.shape
    assert un_adj.shape[1] == neighb_adj.shape[0]
    assert un_adj.shape[1] == neighb_adj.shape[1]

    (blank,) = (labels < 0).nonzero(as_tuple=True)
    Nblank = blank.numel()
    if not Nblank and ensure_coverage_only:
        return None, 0
    elif not Nblank:
        return True, 0

    # need full coverage of neighborhoods here. would prefer only to branch out with
    # explore as needed, so make those probs tiny.
    keep_same_adj = un_adj.sum(0).amin().cpu().item() > 0
    n_new_units = 0
    if next_label is None:
        next_label = un_adj.shape[0]
    if keep_same_adj:
        adj = un_adj
    else:
        adj = un_adj + explore_adj * tiny
        uncovered = adj.sum(0).amin().cpu().item() == 0
        if uncovered:
            uncovered_adj = adj.clone()
        else:
            uncovered_adj = adj

        n_steps = -1
        for n_steps in range(min(int(uncovered), max_steps)):  # noqa: B007
            explore_adj = explore_adj @ neighb_adj
            adj = un_adj + explore_adj * tiny
            if adj.sum(0).min().cpu().item() > 0:
                break

        still_uncovered = adj.sum(0).amin().cpu().item() == 0
        if still_uncovered and ensure_coverage_only and allow_new_units:
            # make new units by getting connected components of the uncovered
            # neighborhood graph
            (uncovered_neighbs,) = (adj.sum(0) == 0).nonzero(as_tuple=True)
            assert uncovered_neighbs.numel() > 0
            uncovered_graph = neighb_adj[uncovered_neighbs][:, uncovered_neighbs]
            uncovered_graph = uncovered_graph.numpy(force=True)
            n_new_units, neighb_component_ids = connected_components(uncovered_graph)
            neighb_component_ids = torch.asarray(neighb_component_ids).to(labels)
            if pnoid:
                assert (neighb_component_ids >= 0).all()
            neighb_component_ids = neighb_component_ids - neighb_component_ids.amin()
            n_spikes = 0
            if pnoid:
                Kold = labels.max().item() + 1 + int(Nblank > 0)
                _lshp = labels.unique().shape
                assert _lshp[0] <= Kold, f"{Kold=} {_lshp=}"
            for j, nid in enumerate(uncovered_neighbs):
                (in_nid,) = (neighborhood_ids == nid).nonzero(as_tuple=True)
                n_spikes += in_nid.numel()
                if pnoid:
                    assert (labels[in_nid] < 0).all()
                labels[in_nid] = next_label + neighb_component_ids[j]
            if pnoid:
                Knew = labels.max() + 1 + int((labels < 0).any())
                assert labels.unique().shape[0] <= Knew
            logger.dartsortverbose(
                f"_fill_blank_labels made {n_new_units} new units for {n_spikes} spikes."
            )
            return None, n_new_units

        if uncovered:
            # raise if still_uncovered, warn or log otherwise
            _log_warn_or_raise_coverage(
                uncovered_adj,
                neighborhood_ids,
                n_steps,
                adj,
                no_raise=allow_uncovered,
            )

    if ensure_coverage_only:
        return None, n_new_units

    if pnoid:
        assert adj.amin() >= 0

    for i0 in range(0, Nblank, batch_size):
        i1 = min(Nblank, i0 + batch_size)
        ii = blank[i0:i1]
        nn = neighborhood_ids[ii]
        p = adj.T[nn]
        p_is_0 = p.sum(1) <= 0.0
        p[:, 0].clamp_min_(p_is_0.to(p))
        draws = torch.multinomial(p, 1, replacement=True, generator=gen)
        draws = draws.view(ii.shape[0])
        draws.masked_fill_(p_is_0, -1)
        labels[ii] = draws

    return keep_same_adj, n_new_units


def _bootstrap_top(
    candidates: Tensor,
    neighborhood_ids: Tensor,
    n_candidates: int,
    un_adj: Tensor,
    un_adj_lut: NeighborhoodLUT,
    gen: torch.Generator,
    batch_size: int = 1024,
):
    if n_candidates == 1:
        return

    # use a un_adj_lut index-dependent probability here to avoid re-using
    # the current label
    p = un_adj.T[un_adj_lut.b.neighb_ids]
    zero = p.new_zeros((1, 1)).broadcast_to((un_adj_lut.b.unit_ids.shape[0], 1))
    p.scatter_(dim=1, index=un_adj_lut.b.unit_ids[:, None], src=zero)

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
        ll = un_adj_lut.b.lut[uu, nn]
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
    p = explore_adj.T[un_adj_lut.b.neighb_ids]

    # zero out top unit probs
    zero = p.new_zeros((1, 1)).broadcast_to((un_adj_lut.b.unit_ids.shape[0], 1))
    p.scatter_(dim=1, index=un_adj_lut.b.unit_ids[:, None], src=zero)

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

    # pad p with an extra LUT index (row), inds with a -1
    # the tiny mops up cases with no overlaps, and that case is
    # sometimes allowed by the logic which gets us here
    p = F.pad(p, (0, 0, 0, 1), value=torch.finfo(p.dtype).tiny)
    inds = F.pad(inds, (0, 0, 0, 1), value=-1).long()

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
    batch_iter = range(0, candidates.shape[0], batch_size)
    assert counts.shape[0] == len(batch_iter)
    for b, i0 in enumerate(batch_iter):
        counts[b] = (candidates[i0 : i0 + batch_size] >= 0).count_nonzero()
    batch_candidate_counts.copy_(counts.cpu())


@torch_compiler(fullgraph=False)
def _combine_similar_resps(resps: Tensor, keep_mask: Tensor, n_keep: int) -> Tensor:
    n_discard = resps.shape[1] - n_keep
    assert n_discard + n_keep == resps.shape[1]
    discard_mask = torch.logical_not(keep_mask)
    discard_ix = _nonzero_static(discard_mask, size=n_discard)[:, 0]
    keep_ix = _nonzero_static(keep_mask, size=n_keep)[:, 0]
    assert keep_ix.numel() + discard_ix.numel() == resps.shape[1]
    kept_resp = resps[:, keep_ix]
    discard_resp = resps[:, discard_ix]
    sim = discard_resp.T @ kept_resp
    match = sim.argmax(1)
    kept_resp[:, match] += discard_resp
    return kept_resp


def concatenate_scores(scoress: list[Scores], dim=0) -> Scores:
    assert len(scoress) > 0
    if len(scoress) == 1:
        return scoress[0]
    log_liks = torch.concatenate([s.log_liks for s in scoress], dim=dim)
    candidates = torch.concatenate([s.candidates for s in scoress], dim=dim)
    if scoress[0].responsibilities is None:
        responsibilities = None
    else:
        responsibilities = torch.concatenate(
            [cast(Tensor, s.responsibilities) for s in scoress], dim=dim
        )
    if scoress[0].duties is None:
        assert all(s.duties is None for s in scoress)
        duties = None
    else:
        duties = torch.concatenate([cast(Tensor, s.duties) for s in scoress], dim=dim)
    return Scores(
        log_liks=log_liks,
        responsibilities=responsibilities,
        candidates=candidates,
        duties=duties,
    )


def remove_units_from_scores(
    scores: Scores, unit_ids: Tensor, *, sort=True, batch_size: int = 2**16
) -> Scores:
    """Return copy of scores where log_liks for candidates in unit_ids are -inf (and resps are 0)."""
    nu = unit_ids.shape[0]
    if not nu:
        return ensure_sorted_scores(scores) if sort else scores
    new_cand = torch.empty_like(scores.candidates)
    new_log_liks = torch.empty_like(scores.log_liks)
    new_resp = torch.empty_like(new_log_liks)
    n, ncand = new_cand.shape
    unit_ids = unit_ids.sort().values.to(new_cand)
    for i0 in range(0, n, batch_size):
        i1 = min(n, i0 + batch_size)

        bcand = new_cand[i0:i1]
        bcand.copy_(scores.candidates[i0:i1])
        binds = torch.searchsorted(unit_ids, bcand)
        binds = binds.clamp_(0, nu - 1)
        bye = bcand == unit_ids[binds]
        bcand.masked_fill_(bye, -1)

        new_log_liks[i0:i1].copy_(scores.log_liks[i0:i1])
        new_log_liks[i0:i1, :ncand].masked_fill_(bye, -torch.inf)

        bresp = new_log_liks[i0:i1].softmax(dim=1)
        torch.nan_to_num(bresp, out=new_resp[i0:i1])

    scores = Scores(
        candidates=new_cand,
        log_liks=new_log_liks,
        responsibilities=new_resp,
        duties=scores.duties,
    )
    if sort:
        return ensure_sorted_scores(scores)
    else:
        return scores


def proportion_adjust_scores(
    *,
    scores: Scores,
    orig_log_props: Tensor,
    new_log_props: Tensor,
    with_responsibilities: bool,
    sort=True,
    batch_size: int = 2**16,
) -> Scores:
    """What would the scores be if the log props were these, not those?"""
    diff = new_log_props - orig_log_props
    diff.nan_to_num_()
    diff = F.pad(diff, (0, 1))

    new_log_liks = torch.empty_like(scores.log_liks)
    if with_responsibilities:
        new_resp = torch.empty_like(new_log_liks)
    else:
        new_resp = None
    n, ncand = scores.candidates.shape

    for i0 in range(0, n, batch_size):
        i1 = min(n, i0 + batch_size)

        bdiff = diff[scores.candidates[i0:i1]]
        torch.add(
            scores.log_liks[i0:i1, :ncand], bdiff, out=new_log_liks[i0:i1, :ncand]
        )
        new_log_liks[i0:i1, ncand] = scores.log_liks[i0:i1, ncand]

        if new_resp is None:
            continue

        bresp = torch.softmax(new_log_liks[i0:i1], dim=1)
        torch.nan_to_num(bresp, out=new_resp[i0:i1])

    scores = Scores(
        log_liks=new_log_liks,
        candidates=scores.candidates,
        responsibilities=new_resp,
        duties=scores.duties,
    )
    if sort:
        return ensure_sorted_scores(scores)
    else:
        return scores


def ensure_sorted_scores(scores: Scores) -> Scores:
    """Re-sort score properties so that log liks decrease."""
    C = scores.candidates.shape[1]
    order = scores.log_liks[:, :C].argsort(dim=1, descending=True, stable=True)

    candidates = scores.candidates.take_along_dim(indices=order, dim=1)
    log_liks = torch.empty_like(scores.log_liks)
    torch.take_along_dim(
        scores.log_liks[:, :C], indices=order, dim=1, out=log_liks[:, :C]
    )
    if scores.log_liks.shape[1] > C:
        assert scores.log_liks.shape[1] == C + 1
        log_liks[:, -1] = scores.log_liks[:, -1]
    if scores.responsibilities is not None:
        responsibilities = F.softmax(log_liks, dim=1).nan_to_num_()
    else:
        responsibilities = None
    if scores.duties is not None:
        duties = scores.duties.take_along_dim(indices=order, dim=1)
    else:
        duties = None

    return Scores(
        candidates=candidates,
        log_liks=log_liks,
        responsibilities=responsibilities,
        duties=duties,
    )


def drop_units_and_update_scores(
    train_scores: Scores,
    scores: Scores | None,
    n_units: int,
    remove_ids: Tensor,
    with_responsibilities: bool = True,
):
    # determine mean responsibilities before and after removing units
    orig_prop = mean_responsibilities(train_scores, n_units=n_units)
    train_scores = remove_units_from_scores(train_scores, remove_ids)
    new_prop = mean_responsibilities(train_scores, n_units=n_units)
    assert orig_prop.shape == new_prop.shape == (n_units + 1,)

    # pick log props
    model_prop = orig_prop[:n_units].sum().item()
    new_prop *= model_prop / new_prop.sum()
    orig_log_prop = orig_prop.log()
    new_log_prop = new_prop.log()

    # update the train scores
    train_scores = proportion_adjust_scores(
        scores=train_scores,
        orig_log_props=orig_log_prop,
        new_log_props=new_log_prop,
        with_responsibilities=with_responsibilities,
    )

    # update other scores
    if scores is not None:
        scores = remove_units_from_scores(scores, remove_ids, sort=False)
        scores = proportion_adjust_scores(
            scores=scores,
            orig_log_props=orig_log_prop,
            new_log_props=new_log_prop,
            with_responsibilities=with_responsibilities,
        )

    return train_scores, scores


def mean_responsibilities(
    scores: Scores | None = None,
    n_units: int | None = None,
    responsibilities: torch.Tensor | None = None,
    candidates: torch.Tensor | None = None,
    batch_size: int = 2**16,
    weights: Tensor | None = None,
) -> torch.Tensor:
    """Average per-spike responsibility by unit for scores a Scores object."""
    if responsibilities is not None:
        resp = responsibilities
    elif scores is not None and scores.responsibilities is None:
        resp = torch.softmax(scores.log_liks, dim=1)
    elif scores is not None:
        resp = scores.responsibilities
    else:
        panic()

    if candidates is not None:
        cand = candidates
    elif scores is not None:
        cand = scores.candidates
    else:
        panic()
    assert resp is not None
    assert cand is not None
    ncand = cand.shape[1]
    assert resp.shape[1] in (ncand, ncand + 1)
    includes_noise = resp.shape[1] == ncand + 1
    assert resp.shape[0] == cand.shape[0]
    n = resp.shape[0]

    if n_units is None:
        n_units = int(cand.amax()) + 1
    assert n_units > 0

    # welford running mean responsiblity by batches
    knoise = n_units + includes_noise
    resp_mean = resp.new_zeros(knoise + 1, dtype=torch.double)
    rsum_batch = resp_mean.clone()
    if weights is None:
        for i0 in range(0, n, batch_size):
            i1 = min(n, i0 + batch_size)
            nbatch = i1 - i0
            rsum_batch.zero_()

            cix = F.threshold(cand[i0:i1], threshold=-1, value=knoise)
            rsum_batch.scatter_add_(
                dim=0,
                index=cix.view(-1),
                src=resp[i0:i1, :ncand].double().reshape(-1),
            )
            if includes_noise:
                rsum_batch[n_units] += resp[i0:i1, -1].double().sum()

            rmean_batch = rsum_batch.mul_(1.0 / nbatch)
            resp_mean += rmean_batch.sub_(resp_mean).mul_(nbatch / i1)
    else:
        assert weights.shape == (n,)
        for i0 in range(0, n, batch_size):
            i1 = min(n, i0 + batch_size)
            bweights = weights[i0:i1].double()
            cix = F.threshold(cand[i0:i1], threshold=-1, value=knoise)
            resp_mean.scatter_add_(
                dim=0,
                index=cix.view(-1),
                src=resp[i0:i1, :ncand].double().mul_(bweights[:, None]).reshape(-1),
            )
            if includes_noise:
                resp_mean[n_units] += resp[i0:i1, -1].double().mul_(bweights).sum()
        resp_mean /= weights.double().sum()
    resp_mean = resp_mean[:knoise]

    # check that things didn't explode
    assert resp_mean.isfinite().all(), "Responsibility mean not finite"
    assert torch.all(resp_mean >= -prop_check_atol), "Responsiblity mean negative"
    resp_mean.relu_()
    assert torch.all(resp_mean <= 1.0 + prop_check_atol), (
        f"Responsiblity mean > 1, largest: {resp_mean.amax()}"
    )
    if includes_noise:
        _assert_propclose(resp_mean.sum(), resp_mean.new_ones(()))
    else:
        assert torch.lt(
            resp_mean.sum(), resp_mean.new_full((), 1.0 + prop_check_atol)
        ), f"Responsiblity sum > 1, to wit {resp_mean.sum()}"

    return resp_mean


def mean_responsibilities_after_removal(
    scores: Scores,
    remove_units: Tensor,
    n_units: int,
    batch_size: int = 2**18,
    weights: Tensor | None = None,
) -> Tensor:
    n, ncand = scores.candidates.shape
    assert scores.log_liks.shape == (n, ncand + 1)
    assert weights is None or weights.shape == (n,)
    remove_units = remove_units.sort().values
    remove_units = remove_units.to(scores.candidates)
    nremove = remove_units.shape[0]

    # welford running mean responsiblity by batches, as in mean_responsibilities
    knoise = n_units + 1
    resp_mean = scores.log_liks.new_zeros(knoise + 1, dtype=torch.double)
    rsum_batch = resp_mean.clone()
    for i0 in range(0, n, batch_size):
        i1 = min(n, i0 + batch_size)
        nbatch = i1 - i0
        rsum_batch.zero_()

        bcand = scores.candidates[i0:i1]
        binds = torch.searchsorted(remove_units, bcand)
        binds = binds.clamp_(0, nremove - 1)
        bye = bcand == remove_units[binds]

        bll = scores.log_liks[i0:i1].clone()
        bll[:, :ncand].masked_fill_(bye, -torch.inf)
        bresp = bll.softmax(dim=1).nan_to_num_()

        cix = F.threshold(bcand, threshold=-1, value=knoise)
        if weights is None:
            rsum_batch.scatter_add_(
                dim=0,
                index=cix.reshape(-1),
                src=bresp[:, :ncand].double().reshape(-1),
            )
            rsum_batch[n_units] += bresp[:, -1].double().sum()

            rmean_batch = rsum_batch.mul_(1.0 / nbatch)
            resp_mean += rmean_batch.sub_(resp_mean).mul_(nbatch / i1)
        else:
            bresp = bresp.double().mul_(weights[i0:i1].double()[:, None])
            resp_mean.scatter_add_(
                dim=0, index=cix.reshape(-1), src=bresp[:, :ncand].reshape(-1)
            )
            resp_mean[n_units] += bresp[:, -1].sum()

    if weights is not None:
        resp_mean /= weights.double().sum()
    resp_mean = resp_mean[:knoise]
    _assert_propclose(resp_mean.sum(), resp_mean.new_ones(()))
    return resp_mean


def proportion_adjust_criterion_change(
    *,
    scores: Scores,
    orig_log_props: Tensor,
    new_log_props: Tensor,
    cl_alpha: float,
    orig_row_criterion: Tensor,
    batch_size: int = 2**18,
    weights: Tensor | None = None,
) -> float:
    """How much would ecl change if the log props were these, not those?"""
    cand, log_liks = scores.candidates, scores.log_liks
    n = cand.shape[0]
    assert orig_row_criterion.shape == (n,)
    assert weights is None or weights.shape == (n,)

    mean = orig_row_criterion.new_zeros((), dtype=torch.double)
    for i0 in range(0, n, batch_size):
        i1 = min(n, i0 + batch_size)
        nbatch = i1 - i0

        borig = orig_row_criterion[i0:i1]
        badj = proportion_adjusted_row_criterion(
            log_liks=log_liks[i0:i1],
            candidates=cand[i0:i1],
            orig_log_props=orig_log_props,
            new_log_props=new_log_props,
            cl_alpha=cl_alpha,
        )

        if weights is None:
            bmean = badj.sub_(borig).mean()
            mean += bmean.sub_(mean).mul_(nbatch / i1)
        else:
            mean += badj.sub_(borig).mul_(weights[i0:i1].double()).sum()

    if weights is not None:
        mean /= weights.double().sum()
    return mean.item()


def proportion_adjusted_row_criterion(
    *,
    log_liks: Tensor,
    candidates: Tensor,
    orig_log_props: Tensor,
    new_log_props: Tensor,
    cl_alpha: float,
) -> Tensor:
    diff = (new_log_props.double() - orig_log_props.double()).nan_to_num_()
    diff = F.pad(diff, (0, 1))
    ncand = candidates.shape[1]
    log_liks = log_liks.to(dtype=torch.double, copy=True)
    log_liks[:, :ncand] += diff[candidates]
    return ecl(resps=None, log_liks=log_liks, cl_alpha=cl_alpha, reduce_mean=False)


# -- precomputing expensive things that depend on parameters and neighborhoods


LOG_2PI = math.log(math.tau)


def huber_quantile_if_huber(
    robust_strategy: RobustnessStrategy, huber_quantile: float
) -> float | None:
    if robust_strategy == "huber":
        return huber_quantile
    elif robust_strategy == "none":
        return None
    else:
        panic(robust_strategy)


def noise_huber_quantile(refinement_cfg: RefinementConfig) -> float | None:
    if not refinement_cfg.huber_noise:
        return None
    return huber_quantile_if_huber(
        refinement_cfg.robust_strategy, refinement_cfg.huber_quantile
    )


def huber_neighborhood_constants(
    nobs: Tensor, quantile: float
) -> tuple[Tensor, Tensor]:
    """Huber radius k and 2log(Z_huber / Z_gaussian) for each observed dim d

    k^2 = chi2(d).ppf(quantile), and
        Z_huber / Z_gaussian = P(d/2, k^2/2)
            + 2^(1-d/2) Gamma(d) Q(d, k^2) exp(k^2/2) / (Gamma(d/2) k^d),
    with P and Q the regularized lower and upper incomplete gamma functions.
    """
    assert 0.0 < quantile < 1.0
    d = nobs.numpy(force=True).astype(np.float64)
    assert np.all(d > 0)
    ksq = chi2.ppf(quantile, d)
    log_tail = (1.0 - 0.5 * d) * math.log(2.0)
    log_tail += gammaln(d) - gammaln(0.5 * d)
    log_tail += 0.5 * ksq - 0.5 * d * np.log(ksq)
    log_tail += np.log(gammaincc(d, ksq))
    log_core = np.log(gammainc(0.5 * d, 0.5 * ksq))
    log_ratio = np.logaddexp(log_core, log_tail)
    assert np.all(np.isfinite(log_ratio))
    k = torch.asarray(np.sqrt(ksq), dtype=torch.float, device=nobs.device)
    twologzratio = torch.asarray(2.0 * log_ratio, dtype=torch.float, device=nobs.device)
    return k, twologzratio


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
       the dim 2pi constant and the logdet(Coo), plus 2log(Z_huber / Z_gaussian)
       when using Huber tails. The next fn deals with the second term if needed.
    """
    n = i1 - i0
    uu = lut.b.unit_ids[i0:i1]
    nn = lut.b.neighb_ids[i0:i1]

    # grab these units' means with one channel of zero padding
    mpad = means.new_zeros((n, neighb_cov.feat_rank, neighb_cov.n_channels + 1))
    torch.index_select(
        means.view(means.shape[0], neighb_cov.feat_rank, -1),
        dim=0,
        index=uu,
        out=mpad[:, :, :-1],
    )

    # extract observed part (muo)
    obs_ix = neighb_cov.b.obs_ix[nn][:, None, :]
    mu_obs_out = lut_params.b.muo.view(
        lut_params.n_lut, neighb_cov.feat_rank, neighb_cov.max_nc_obs
    )
    muo = torch.take_along_dim(mpad, obs_ix, dim=2, out=mu_obs_out[i0:i1])

    # Linvmuo, CmoCooinvmuo
    Linvmuo = neighb_cov.b.temporal_whitener @ muo
    Linvmuo = Linvmuo.bmm(neighb_cov.b.spatial_whitener[nn].mT)
    lut_params.b.Linvmuo[i0:i1] = Linvmuo.view(n, -1)
    CmoCooinvmuo = muo.bmm(neighb_cov.b.spatial_regression[nn])
    lut_params.b.CmoCooinvmuo[i0:i1] = CmoCooinvmuo.view(n, -1)

    # constplogdet. add in the signal-rank-0-only terms.
    lut_params.constplogdet[i0:i1] = neighb_cov.nobs[nn].mul_(LOG_2PI)  # type: ignore  # ty: ignore[x]
    lut_params.constplogdet[i0:i1] += neighb_cov.b.logdet[nn]  # ty: ignore[not-subscriptable]
    if lut_params.huber_2logzratio is not None:
        lut_params.constplogdet[i0:i1] += lut_params.b.huber_2logzratio[nn]  # ty: ignore[not-subscriptable]
    if pnoid:
        assert lut_params.constplogdet[i0:i1].isfinite().all()  # type: ignore  # ty: ignore[x]


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
    uu = lut.b.unit_ids[i0:i1]
    nn = lut.b.neighb_ids[i0:i1]

    # to start, we need the observed part of W. grab as for muo above.
    bpad = F.pad(
        bases[uu].view(uu.shape[0], M, neighb_cov.feat_rank, neighb_cov.n_channels),
        (0, 1),
        value=0.0,
    )
    Wo = bpad.take_along_dim(indices=neighb_cov.b.obs_ix[nn][:, None, None, :], dim=3)

    # next, the capaciatance is I + Wo Cooinv Wo'. always pos def.
    WoCooinvsqrt = neighb_cov.b.temporal_whitener @ Wo
    WoCooinvsqrt = WoCooinvsqrt @ neighb_cov.b.spatial_whitener[nn, None].mT
    WoCooinvsqrt = WoCooinvsqrt.view(n, M, neighb_cov.feat_rank * neighb_cov.max_nc_obs)
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
    lut_params.Tpad[i0:i1, :, :-1] = T  # type: ignore  # ty: ignore[x]
    cap_logdet = L.diagonal(dim1=-2, dim2=-1).log().sum(dim=1).mul_(2.0)
    lut_params.constplogdet[i0:i1] += cap_logdet  # ty: ignore[not-subscriptable]
    if pnoid:
        assert lut_params.b.constplogdet[i0:i1].isfinite().all()

    # precomputed products with T
    assert lut_params.TWoCooinvsqrt is not None
    assert lut_params.TWoCooinvmuo is not None
    TWoCooinvsqrt = torch.bmm(T, WoCooinvsqrt, out=lut_params.TWoCooinvsqrt[i0:i1])  # type: ignore  # ty: ignore[x]
    Linvmuo = lut_params.b.Linvmuo[i0:i1, :, None]
    torch.bmm(TWoCooinvsqrt, Linvmuo, out=lut_params.TWoCooinvmuo[i0:i1, :, None])  # type: ignore  # ty: ignore[x]

    # Woodbury root
    assert lut_params.wburyroot is not None
    torch.bmm(WoCooinvsqrt.mT, Linv.mT, out=lut_params.wburyroot[i0:i1])  # type: ignore  # ty: ignore[x]


# -- em math impls


def _score_batch(
    *,
    candidates: Tensor,
    neighborhood_ids: Tensor,
    n_candidates: int,
    whitenedx: Tensor,
    noise_logliks: Tensor,
    log_proportions: Tensor,
    noise_log_prop: Tensor,
    lut_params: LUTParams,
    lut: NeighborhoodLUT,
    robust_strategy: RobustnessStrategy,
    spike_ixs: Tensor | None = None,
    candidate_ixs: Tensor | None = None,
    unit_ixs: Tensor | None = None,
    neighb_ixs: Tensor | None = None,
    lut_ixs: Tensor | None = None,
    static_size: int | None = None,
    fixed_responsibilities: Tensor | None = None,
    skip_responsibility: bool = False,
    skip_noise: bool = False,
    allow_blanks: bool = False,
    force_rank0: bool = False,
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
            candidates,
            neighborhood_ids,
            static_size=static_size,
            allow_blanks=allow_blanks,
        )
        lut_ixs = lut.b.lut[unit_ixs, neighb_ixs]
    else:
        assert candidate_ixs is not None
        assert neighb_ixs is not None
        assert lut_ixs is not None
    if pnoid:
        assert (lut_ixs < lut.b.unit_ids.shape[0]).all()

    lls = whitenedx.new_full((n, Ctot + int(not skip_noise)), fill_value=-torch.inf)
    if not skip_noise:
        lls[:, -1] = noise_logliks
        lls[:, -1] += noise_log_prop

    if lut_params.signal_rank and not force_rank0:
        sparse_lls, sparse_weights = _calc_loglik_ppca(
            whitenedx=whitenedx,
            log_proportions=log_proportions,
            spike_ixs=spike_ixs,
            lut_ixs=lut_ixs,
            unit_ixs=unit_ixs,
            neighb_ixs=neighb_ixs,
            constplogdet=lut_params.constplogdet,
            Linvmuo=lut_params.Linvmuo,
            wburyroot=lut_params.wburyroot,
            huber_k=lut_params.huber_k,
        )
    else:
        sparse_lls, sparse_weights = _calc_loglik_rank0(
            whitenedx=whitenedx,
            log_proportions=log_proportions,
            spike_ixs=spike_ixs,
            lut_ixs=lut_ixs,
            unit_ixs=unit_ixs,
            neighb_ixs=neighb_ixs,
            constplogdet=lut_params.constplogdet,
            Linvmuo=lut_params.Linvmuo,
            huber_k=lut_params.huber_k,
        )
    lls[spike_ixs, candidate_ixs] = sparse_lls
    del sparse_lls
    if sparse_weights is None:
        huber_weights = None
    else:
        huber_weights = whitenedx.new_zeros((n, Ctot))
        huber_weights[spike_ixs, candidate_ixs] = sparse_weights
    del sparse_weights

    if do_resp:
        n_top = min(n_candidates, Ctot)
        toplls, topinds = torch.topk(lls[:, :-1], n_top, dim=1)
        candidates = candidates.take_along_dim(indices=topinds, dim=1)
        if huber_weights is not None:
            huber_weights = huber_weights.take_along_dim(indices=topinds, dim=1)
        if n_top < n_candidates:
            n_pad = n_candidates - n_top
            toplls = F.pad(toplls, (0, n_pad), value=-torch.inf)
            candidates = F.pad(candidates, (0, n_pad), value=-1)
            if huber_weights is not None:
                huber_weights = F.pad(huber_weights, (0, n_pad), value=0.0)
        toplls = torch.concatenate((toplls, lls[:, -1:]), dim=1)
    else:
        toplls = lls
    del lls

    if robust_strategy == "huber":
        assert huber_weights is not None
        assert huber_weights.shape == candidates.shape
        candidate_duties = huber_weights
    elif robust_strategy == "none":
        assert huber_weights is None
        candidate_duties = None
    else:
        panic(robust_strategy)

    if do_resp:
        responsibilities = torch.softmax(toplls, dim=1)
    else:
        responsibilities = fixed_responsibilities

    return Scores(
        log_liks=toplls,
        responsibilities=responsibilities,
        candidates=candidates,
        duties=candidate_duties,
    )


def _sparsify_candidates(
    candidates: Tensor,
    neighborhood_ids: Tensor,
    *,
    static_size: int | None,
    allow_blanks: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    cpos = candidates >= 0
    if pnoid and not allow_blanks:
        assert cpos.any(dim=1).all()
    if pnoid and static_size is not None:
        assert cpos.sum() == static_size
    if static_size is not None:
        spike_ixs, candidate_ixs = _nonzero_static(cpos, size=static_size).T
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


@torch_compiler(fullgraph=False)
def _calc_loglik_rank0(
    *,
    whitenedx: Tensor,
    log_proportions: Tensor,
    spike_ixs: Tensor,
    lut_ixs: Tensor,
    unit_ixs: Tensor,
    neighb_ixs: Tensor,
    constplogdet: Tensor,
    Linvmuo: Tensor,
    huber_k: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    wdxz = whitenedx[spike_ixs]
    wdxz -= Linvmuo[lut_ixs]
    ll = wdxz.square_().sum(dim=1)
    if huber_k is None:
        weight = None
    else:
        ll, weight = _huber_twicerho_and_weight(ll, huber_k[neighb_ixs])
    ll += constplogdet[lut_ixs]
    ll *= -0.5
    ll += log_proportions[unit_ixs]
    return ll, weight


@torch_compiler(fullgraph=False)
def _calc_loglik_ppca(
    *,
    whitenedx: Tensor,
    log_proportions: Tensor,
    spike_ixs: Tensor,
    lut_ixs: Tensor,
    unit_ixs: Tensor,
    neighb_ixs: Tensor,
    constplogdet: Tensor,
    Linvmuo: Tensor,
    wburyroot: Tensor,
    huber_k: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    wdxz = whitenedx[spike_ixs]
    wdxz -= Linvmuo[lut_ixs]

    term_b = wdxz[:, None].bmm(wburyroot[lut_ixs])[:, 0]
    term_a = wdxz.square_().sum(dim=1)
    term_b = term_b.square_().sum(dim=1)

    ll = term_a.sub_(term_b)
    ll = ll.nan_to_num_(nan=torch.inf)
    if huber_k is None:
        weight = None
    else:
        ll, weight = _huber_twicerho_and_weight(ll, huber_k[neighb_ixs])

    ll += constplogdet[lut_ixs]
    ll *= -0.5
    ll += log_proportions[unit_ixs]
    return ll, weight


def _stat_pass_batch(
    *,
    xt: Tensor,
    whitenedx: Tensor,
    CmoCooinvx: Tensor,
    responsibilities: Tensor,
    mstep_weights: Tensor | None,
    sample_weights: Tensor | None,
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
    if mstep_weights is not None:
        assert mstep_weights.shape == (n, n_candidates)
    if sample_weights is not None:
        assert sample_weights.shape == (n,)
        responsibilities = responsibilities * sample_weights[:, None]
    if lut_params.signal_rank:
        assert lut_params.TWoCooinvsqrt is not None
        assert lut_params.TWoCooinvmuo is not None
        assert lut_params.Tpad is not None
        noise_N, resp_N, N, Nlut, Ulut, R = _stat_pass_batch_ppca(
            xt=xt,
            whitenedx=whitenedx,
            CmoCooinvx=CmoCooinvx,
            responsibilities=responsibilities,
            mstep_weights=mstep_weights,
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
        return SufficientStatistics(
            noise_N=noise_N, resp_N=resp_N, N=N, Nlut=Nlut, Ulut=Ulut, R=R
        )
    else:
        noise_N, resp_N, N, Nlut, R = _stat_pass_batch_rank0(
            xt=xt,
            CmoCooinvx=CmoCooinvx,
            responsibilities=responsibilities,
            mstep_weights=mstep_weights,
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
        return SufficientStatistics(
            noise_N=noise_N, resp_N=resp_N, N=N, Nlut=Nlut, Ulut=None, R=R
        )


@torch_compiler(fullgraph=False)
def _count_batch(
    *,
    responsibilities: Tensor,
    weightsflat: Tensor | None,
    n_candidates: int,
    spike_ixs: Tensor,
    candidate_ixs: Tensor,
    unit_ixs: Tensor,
    lut_ixs: Tensor,
    n_units: int,
    n_lut: int,
    eps: Tensor,
) -> tuple[Tensor | None, Tensor, Tensor, Tensor, Tensor, Tensor]:
    have_noise = responsibilities.shape[1] == n_candidates + 1
    if have_noise:
        noise_N = responsibilities[:, -1].sum()
    else:
        noise_N = None

    Qflat = responsibilities[spike_ixs, candidate_ixs]
    resp_N = Qflat.new_zeros(n_units)
    resp_N.scatter_add_(dim=0, index=unit_ixs, src=Qflat)
    if weightsflat is None:
        QWflat = Qflat
    else:
        QWflat = Qflat * weightsflat

    N = Qflat.new_zeros(n_units)
    Nlut = Qflat.new_zeros(n_lut)
    N.scatter_add_(dim=0, index=unit_ixs, src=QWflat)
    Nlut.scatter_add_(dim=0, index=lut_ixs, src=QWflat)

    Qnflat = N[unit_ixs].clamp_(min=eps)
    Qnflat = torch.div(QWflat, Qnflat, out=Qnflat)

    Qnflatlut = Nlut[lut_ixs].clamp_(min=eps)
    Qnflatlut = torch.div(Qflat, Qnflatlut, out=Qnflatlut)

    return noise_N, resp_N, N, Nlut, Qnflat, Qnflatlut


@torch_compiler(fullgraph=False)
def _stat_pass_batch_ppca(
    *,
    xt: Tensor,
    whitenedx: Tensor,
    CmoCooinvx: Tensor,
    responsibilities: Tensor,
    mstep_weights: Tensor | None,
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

    Parameters
    ----------
    responsibilities : (n, C or C + 1)
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
    if mstep_weights is None:
        weightsflat = None
    else:
        weightsflat = mstep_weights[spike_ixs, candidate_ixs]

    noise_N, resp_N, N, Nlut, Qnflat, Qnflatlut = _count_batch(
        responsibilities=responsibilities,
        weightsflat=weightsflat,
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
    # weights multiply E[u]E[hatu]' but not the posterior covariance T
    if weightsflat is None:
        hatU.addcmul_(ubar[:, :, None], hatubar[:, None, :])
    else:
        wubar = ubar * weightsflat[:, None]
        hatU.addcmul_(wubar[:, :, None], hatubar[:, None, :])

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

    return noise_N, resp_N, N, Nlut, Ulut, R


@torch_compiler(fullgraph=False)
def _stat_pass_batch_rank0(
    *,
    xt: Tensor,
    CmoCooinvx: Tensor,
    responsibilities: Tensor,
    mstep_weights: Tensor | None,
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

    Parameters
    ----------
    responsibilities : (n, C or C + 1)
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
    if mstep_weights is None:
        weightsflat = None
    else:
        weightsflat = mstep_weights[spike_ixs, candidate_ixs]

    noise_N, resp_N, N, Nlut, Qnflat, _Qnflatlut = _count_batch(
        responsibilities=responsibilities,
        weightsflat=weightsflat,
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

    return noise_N, resp_N, N, Nlut, R


def _add_spike_elbos_(
    elbo_sums: Tensor,
    scores: Scores,
    sample_weights: Tensor | None,
    block_ids: Tensor | None = None,
):
    assert scores.responsibilities is not None
    spike_elbos = elbo(scores.responsibilities, scores.log_liks, reduce_mean=False)
    spike_elbos = spike_elbos.double()
    if sample_weights is not None:
        spike_elbos = spike_elbos * sample_weights
    if block_ids is None:
        elbo_sums += spike_elbos.sum()
    else:
        elbo_sums.index_add_(0, block_ids, spike_elbos)


def _unit_log_priors(
    *,
    means: Tensor,
    bases: Tensor | None,
    neighb_cov: NeighborhoodCovariance,
    prior_pseudocount: float,
) -> Tensor:
    """Ridge prior."""
    n_units = means.shape[0]
    frank = neighb_cov.feat_rank
    nc = neighb_cov.n_channels
    temporal_whitener = neighb_cov.b.temporal_whitener
    spatial_right_factor = neighb_cov.b.full_spatial_whitener.T
    term = temporal_whitener @ means.view(n_units, frank, nc) @ spatial_right_factor
    term = term.square_().sum(dim=(1, 2))
    if bases is not None:
        bterm = bases.reshape(n_units, -1, frank, nc)
        bterm = temporal_whitener @ bterm @ spatial_right_factor
        term = term + bterm.square_().sum(dim=(1, 2, 3))
    return term.double().mul_(-0.5 * prior_pseudocount)


def _finalize_e_stats(
    *,
    means: Tensor,
    bases: Tensor | None,
    stats: SufficientStatistics,
    lut: NeighborhoodLUT,
    lut_params: LUTParams,
    neighb_cov: NeighborhoodCovariance,
    batch_size=1024,
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

    # storage
    ncm = neighb_cov.max_nc_miss_near
    What_batch = means.new_zeros((batch_size, hat_dim, frank, nc + 1))
    Uhat_batch = means.new_ones((batch_size, hat_dim, hat_dim))

    # reweighting
    denom = stats.N[lut.b.unit_ids].clamp_(min=tiny)
    Nlut_N = torch.div(stats.Nlut, denom, out=denom)
    if pnoid:
        assert torch.isfinite(Nlut_N).all()

    # loop because w_wcc is big
    for i0 in range(0, lut_params.n_lut, batch_size):
        # batch indices
        i1 = min(lut_params.n_lut, i0 + batch_size)
        uu = lut.b.unit_ids[i0:i1]
        nn = lut.b.neighb_ids[i0:i1]
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

        obs_ix = neighb_cov.b.obs_ix[nn][:, None, None, :]
        Whatobs = What.take_along_dim(dim=3, indices=obs_ix)
        WoCooinvCom = Whatobs @ neighb_cov.b.spatial_regression[nn, None]
        assert WoCooinvCom.shape == (i1 - i0, hat_dim, frank, ncm)

        w_wcc = What  # first make Wmiss in place
        ix = neighb_cov.b.miss_near_ix[nn, None, None].broadcast_to(WoCooinvCom.shape)
        w_wcc.scatter_add_(dim=3, index=ix, src=WoCooinvCom._neg_view())
        w_wcc = w_wcc[..., :-1]
        w_wcc *= neighb_cov.b.miss_full_mask[nn][:, None, None, :]
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

    Nlut_N = stats.Nlut / (stats.N + Nz)[lut.b.unit_ids]
    assert Nlut_N.isfinite().all()
    stats.Ulut *= Nlut_N[:, None, None]

    U = stats.Ulut.new_zeros((n_units, hat_dim, hat_dim))
    ix = lut.b.unit_ids[:, None, None].broadcast_to(stats.Ulut.shape)
    U[:, :-1, :].scatter_add_(dim=0, index=ix, src=stats.Ulut)
    U[:, -1:, :-1] = U[:, :-1, -1:].mT
    U[:, -1, -1] = 1.0

    # fill blanks if present (N=0 implies U=0)
    U.diagonal(dim1=-2, dim2=-1).add_(Nz[:, None])

    return U


def _log_warn_or_raise_coverage(
    orig_adj, neighborhood_ids, n_steps, final_adj, no_raise
):
    n_orig_uncov, orig_uncov_neighbs, pct_orig_uncov, orig_uncov_counts = (
        _check_coverage_stats(orig_adj, neighborhood_ids)
    )
    n_new_uncov, _new_uncov_neighbs, pct_new_uncov, _new_uncov_counts = (
        _check_coverage_stats(final_adj, neighborhood_ids)
    )
    message = (
        f"Neighborhood coverage was not complete, with {len(orig_uncov_neighbs)} uncovered "
        f"neighborhoods and {n_orig_uncov} uncovered spikes ({pct_orig_uncov:.3f}% of set). "
        f"Neighborhoods and counts were: {orig_uncov_neighbs}, {orig_uncov_counts}. It took "
        f"{n_steps} expansion steps to fill out the coverage with final uncovered fraction "
        f"{pct_new_uncov:.3f}% ({n_new_uncov} spikes)."
    )
    needs_raise = n_new_uncov > 0
    if needs_raise and not no_raise:
        raise ValueError(message)
    elif pct_orig_uncov < 0.05:
        logger.dartsortverbose(message)
    else:
        warnings.warn(message, stacklevel=2)


def _check_coverage_stats(adj, neighborhood_ids):
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
    return n_uncovered, uncovered_neighbs, pct_uncovered, uncovered_counts


def _assert_propclose(aprop, bprop):
    torch.testing.assert_close(aprop, bprop, atol=prop_check_atol, rtol=prop_check_rtol)


def _assert_sameids(aids, bids):
    if not torch.equal(aids, bids):
        if aids.numel() < 10:
            raise AssertionError(f"{aids=} did not equal {bids=}")
        else:
            torch.testing.assert_close(aids, bids)


if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals(
        [
            TruncatedMixtureModel,
            SpikeNeighborhoods,
            NeighborhoodCovariance,
            LUTParams,
            NeighborhoodLUT,
        ]
    )
