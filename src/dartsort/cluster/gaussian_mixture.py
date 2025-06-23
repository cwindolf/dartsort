from logging import getLogger
import threading
from dataclasses import replace
from typing import Literal, Optional, Any
import warnings
import traceback
import time

import numba
import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from linear_operator import operators
from scipy.cluster.hierarchy import linkage
from scipy.sparse import coo_array, csc_array
from scipy.special import logsumexp
from scipy.spatial import KDTree
from tqdm.auto import tqdm, trange
from sympy.utilities.iterables import multiset_partitions

from ..util import more_operators, noise_util, spiketorch
from ..util.sparse_util import (
    csc_insert,
    get_csc_storage,
    coo_to_torch,
    coo_to_scipy,
    csc_sparse_mask_rows,
    coo_sparse_mask_rows,
    csc_sparse_getrow,
    sparse_topk,
    sparse_reassign,
    integers_without_inner_replacement,
    allocate_topk,
    topk_sparse_insert,
    topk_sparse_tocsc,
)
from .cluster_util import (
    agglomerate,
    combine_distances,
    leafsets,
    is_largest_set_smaller_than,
)
from .kmeans import kmeans
from .modes import smoothed_dipscore_at
from .ppcalib import ppca_em
from .stable_features import (
    SpikeFeatures,
    SpikeNeighborhoods,
    StableSpikeDataset,
    occupied_chans,
)
from ._truncated_em_helpers import _elbo_prior_correction
from . import truncated_mixture
from ..util.logging_util import DARTsortLogger, DARTSORTDEBUG, DARTSORTVERBOSE

logger: DARTsortLogger = getLogger(__name__)


# -- main class


class SpikeMixtureModel(torch.nn.Module):
    """Business logic class

    Handles labels, splits, grabbing SpikeFeaturs batches from the
    SpikeStableDataset, computing distances and bimodality scores.

    The actual numerical computations (log likelihoods, M step
    formulas) are deferred to the GaussianUnit class (and
    subclasses?) below.
    """

    def __init__(
        self,
        data: StableSpikeDataset,
        noise: noise_util.EmbeddedNoise,
        n_spikes_fit: int = 4096,
        mean_kind="full",
        cov_kind="ppca",
        use_proportions: bool = True,
        proportions_sample_size: int = 2**16,
        likelihood_batch_size: int = 2**15,
        channels_strategy: Literal["all", "snr", "count", "count_core"] = "count",
        channels_count_min: float = 25.0,
        channels_snr_amp: float = 1.0,
        with_noise_unit: bool = True,
        prior_pseudocount: float = 25.0,
        prior_scales_mean=True,
        ppca_rank: int = 0,
        ppca_initial_em_iter: int = 3,
        ppca_inner_em_iter: int = 3,
        ppca_atol: float = 0.05,
        ppca_warm_start: bool = True,
        n_threads: int = 4,
        min_count: int = 50,
        n_em_iters: int = 25,
        kmeans_k: int = 4,
        kmeans_n_iter: int = 100,
        kmeans_drop_prop: float = 0.025,
        kmeans_with_proportions: bool = False,
        kmeans_kmeanspp_initial: str = "random",
        split_em_iter: int = 0,
        split_whiten: bool = True,
        ppca_in_split: bool = True,
        truncated_noise: bool = False,
        distance_metric: Literal[
            "noise_metric", "kl", "reverse_kl", "symkl"
        ] = "noise_metric",
        distance_normalization_kind: Literal["none", "noise", "channels"] = "noise",
        criterion_normalization_kind: Literal["none", "noise", "channels"] = "none",
        merge_linkage: str = "single",
        merge_distance_threshold: float = 3.0,
        merge_bimodality_threshold: float = 0.1,
        criterion_threshold: float | None = 0.0,
        criterion: Literal[
            "heldout_loglik",
            "heldout_elbo",
            "loglik",
            "elbo",
        ] = "heldout_elbo",
        merge_decision_algorithm="brute",
        split_decision_algorithm="tree",
        split_bimodality_threshold: float = 0.1,
        merge_bimodality_cut: float = 0.0,
        merge_bimodality_overlap: float = 0.80,
        merge_bimodality_weighted: bool = True,
        merge_bimodality_score_kind: str = "tv",
        merge_bimodality_masked: bool = False,
        well_connected_distance: float = 0.1,
        merge_sym_function: np.ufunc = np.minimum,
        em_converged_prop: float = 0.001,
        em_converged_churn: float = 0.01,
        em_converged_atol: float = 1e-5,
        em_converged_logpx_tol: float = 1e-5,
        min_overlap: float = 0.0,
        hard_noise=False,
        min_log_prop=-50.0,
        random_seed: int = 0,
        lls_keep_k: int | None = 3,
        laplace_ard=False,
        prior_corrected_criterion=False,
    ):
        super().__init__()

        # key data structures for loading and modeling spikes
        self.data = data
        self.noise = noise

        # parameters
        self.n_spikes_fit = n_spikes_fit
        self.likelihood_batch_size = likelihood_batch_size
        self.n_threads = n_threads if n_threads != 0 else 1
        self.min_count = min_count
        self.channels_count_min = channels_count_min
        self.n_em_iters = n_em_iters
        self.kmeans_k = kmeans_k
        self.kmeans_n_iter = kmeans_n_iter
        self.kmeans_with_proportions = kmeans_with_proportions
        self.kmeans_kmeanspp_initial = kmeans_kmeanspp_initial
        self.kmeans_drop_prop = kmeans_drop_prop
        self.distance_metric = distance_metric
        self.distance_normalization_kind = distance_normalization_kind
        self.criterion_normalization_kind = criterion_normalization_kind
        self.merge_distance_threshold = merge_distance_threshold
        self.criterion = criterion
        self.criterion_threshold = criterion_threshold
        self.merge_bimodality_threshold = merge_bimodality_threshold
        self.split_bimodality_threshold = split_bimodality_threshold
        self.merge_bimodality_cut = merge_bimodality_cut
        self.merge_bimodality_overlap = merge_bimodality_overlap
        self.merge_bimodality_score_kind = merge_bimodality_score_kind
        self.merge_bimodality_weighted = merge_bimodality_weighted
        self.merge_bimodality_masked = merge_bimodality_masked
        self.merge_sym_function = merge_sym_function
        self.merge_linkage = merge_linkage
        self.em_converged_prop = em_converged_prop
        self.em_converged_atol = em_converged_atol
        self.em_converged_churn = em_converged_churn
        self.em_converged_logpx_tol = em_converged_logpx_tol
        self.split_em_iter = split_em_iter
        self.split_whiten = split_whiten
        self.use_proportions = use_proportions
        self.hard_noise = hard_noise
        self.proportions_sample_size = proportions_sample_size
        self.merge_decision_algorithm = merge_decision_algorithm
        self.split_decision_algorithm = split_decision_algorithm
        self.min_overlap = min_overlap
        self.min_log_prop = min_log_prop
        self.prior_pseudocount = prior_pseudocount
        self.truncated_noise = truncated_noise
        self.well_connected_distance = well_connected_distance
        self.lls_keep_k = lls_keep_k
        self.laplace_ard = laplace_ard
        self.prior_corrected_criterion = prior_corrected_criterion
        self.prior_scales_mean = prior_scales_mean

        # store labels on cpu since we're always nonzeroing / writing np data
        assert self.data.original_sorting.labels is not None
        labels = self.data.original_sorting.labels
        self.labels = torch.asarray(labels, dtype=torch.long, copy=True)

        # this is populated by self.m_step()
        self._units = torch.nn.ModuleDict()
        self.log_proportions = None

        # store arguments to the unit constructor in a dict
        self.ppca_rank = ppca_rank
        self.channels_strategy = channels_strategy
        self.unit_args = dict(
            noise=noise,
            mean_kind=mean_kind,
            cov_kind=cov_kind,
            channels_strategy=channels_strategy,
            channels_count_min=channels_count_min,
            channels_snr_amp=channels_snr_amp,
            prior_pseudocount=prior_pseudocount,
            prior_scales_mean=prior_scales_mean,
            ppca_rank=ppca_rank,
            ppca_initial_em_iter=ppca_initial_em_iter,
            ppca_inner_em_iter=ppca_inner_em_iter,
            ppca_atol=ppca_atol,
            ppca_warm_start=ppca_warm_start,
            laplace_ard=laplace_ard,
        )
        extra_split_args = dict(channels_strategy="active")
        if not ppca_in_split:
            extra_split_args.update(cov_kind="zero", ppca_rank=0)
        self.split_unit_args = self.unit_args | extra_split_args

        # clustering with noise unit to hopefully grab false positives
        self.with_noise_unit = with_noise_unit
        if self.with_noise_unit:
            noise_args = dict(
                mean_kind="zero", cov_kind="zero", channels_strategy="all"
            )
            noise_args = self.unit_args | noise_args
            self.noise_unit = GaussianUnit(
                rank=self.data.rank, n_channels=data.n_channels, **noise_args
            )
            logger.dartsortdebug("Fitting noise unit...")
            self.noise_unit.fit(None, None)
            # these only need computing once, but not in init so that
            # there is time for the user to .cuda() me before then
            self._noise_log_likelihoods = None
        self._stack = None

        # multithreading stuff. thread-local rgs, control access to labels, etc.
        self._rg = np.random.default_rng(random_seed)
        self.labels_lock = threading.Lock()
        self.lock = threading.Lock()
        self.storage = threading.local()
        self.next_round_annotations = {}
        self.tmm = None

        self.to(self.data.device)

    @property
    def cov_kind(self):
        return self.unit_args["cov_kind"]

    @cov_kind.setter
    def cov_kind(self, value):
        self.unit_args["cov_kind"] = value
        for unit in self._units.values():
            unit.cov_kind = value

    # -- unit management

    # There is a dict style api for getting units. But, there's
    # a difference between a unit ID and a label ID. A label ID
    # is a positive number present in self.labels. A unit ID is
    # a key of self._units. These may disagree: for instance,
    # after reassignment in the E step, not all unit IDs may
    # be assigned spikes, so label_ids is a subset of unit_ids.

    def __getitem__(self, ix):
        ix = self.normalize_key(ix)
        if ix not in self:
            raise KeyError(
                f"Mixture has no unit with ID {ix}. "
                f"{ix in self=} {ix in self.unit_ids()=} "
                f"\n{self.unit_ids()=}"
            )
        return self._units[ix]

    def get(self, ix, default=None):
        ix = self.normalize_key(ix)
        if ix not in self:
            return default
        return self._units[ix]

    def __setitem__(self, ix, value):
        ix = self.normalize_key(ix)
        self._stack = None
        self._units[ix] = value

    def __delitem__(self, ix):
        ix = self.normalize_key(ix)
        self._stack = None
        del self._units[ix]

    def __contains__(self, ix):
        ix = self.normalize_key(ix)
        return ix in self._units

    def update(self, other):
        if isinstance(other, dict):
            other = other.items()
        for k, v in other:
            self[k] = v

    def empty(self):
        return not self._units

    def clear_units(self, new_ids=None):
        if new_ids is None:
            self._stack = None
            self._units.clear()
        else:
            for k in new_ids:
                if k in self:
                    del self[k]

    def __len__(self):
        return len(self._units)

    def unit_ids(self):
        uids = sorted(int(k) for k in self._units.keys())
        return np.array(list(uids))

    def ids_and_units(self):
        uids = self.unit_ids()
        units = [self[u] for u in uids]
        return uids, units

    def n_units(self):
        uids = self.unit_ids()
        nu_u = 0
        if len(uids):
            nu_u = max(uids) + 1
        lids, _ = self.label_ids()
        nu_l = 0
        if lids.numel():
            nu_l = lids.max().item() + 1
        return max(nu_u, nu_l)

    def label_ids(self, split="train"):
        labels = self.labels
        if split is not None:
            labels = self.labels[self.data.split_indices[split]]
        uids, counts = torch.unique(labels, return_counts=True)
        kept = uids >= 0
        counts = counts[kept]
        uids = uids[kept]
        return uids, counts

    def ids(self):
        return torch.arange(self.n_units())

    def n_labels(self):
        label_ids, _ = self.label_ids()
        nu = label_ids.max() + 1
        return nu

    def missing_ids(self):
        lids, _ = self.label_ids()
        mids = [lid for lid in lids if lid not in self]
        return torch.tensor(mids)

    # -- headliners

    def to_sorting(self):
        labels = self.labels.numpy(force=False).copy()
        return replace(self.data.original_sorting, labels=labels)

    def tvi(
        self,
        n_iter=None,
        show_progress=True,
        liks_from_tmm="if_train",
        final_e_step=True,
        final_split="kept",
        n_threads=None,
        batch_size=1024,
        algorithm="em",
        scheduler=None,
        sgd_lr=0.1,
        initialization="topk",
        atol=None,
    ):
        if n_threads is None:
            n_threads = self.n_threads
        if atol is None:
            atol = self.em_converged_atol

        n_iter = self.n_em_iters if n_iter is None else n_iter
        assert n_iter > 0
        step_progress = show_progress and bool(max(0, int(show_progress) - 1))

        # initialize me
        missing_ids = self.missing_ids()
        if len(missing_ids):
            self.m_step(show_progress=step_progress, fit_ids=missing_ids)
        self.cleanup()

        # update from my stack
        ids, means, covs, logdets, alpha = self.stack_units(
            mean_only=False, with_alpha=True
        )
        ids_, dkl = self.distances(kind="kl", normalization_kind="none")
        assert torch.equal(torch.asarray(ids), torch.asarray(ids_))

        # try reassigning without noise unit...
        lls = self.log_likelihoods(with_noise_unit=True, show_progress=True)
        self.update_proportions(lls)
        lls = lls[:, self.data.split_indices["train"].numpy()]
        assert self.with_noise_unit

        assert self.log_proportions is not None
        keep_mask = self.log_proportions[ids].isfinite().cpu()
        (keep_ids,) = keep_mask.nonzero(as_tuple=True)
        ids = ids[keep_ids]
        assert torch.equal(torch.asarray(ids), torch.asarray(keep_ids))
        keep_mask_nonoise = torch.concatenate(
            (keep_mask, torch.zeros((1,), dtype=torch.bool))
        )
        lls_keep = csc_sparse_mask_rows(lls, keep_mask_nonoise.numpy(), in_place=True)
        del lls  # overwritten
        n_units = len(ids)
        n_spikes = lls_keep.shape[1]

        if self.tmm is None:
            self.tmm = truncated_mixture.SpikeTruncatedMixtureModel(
                data=self.data,
                noise=self.noise,
                M=self.ppca_rank * (self.cov_kind == "ppca"),
                n_threads=n_threads,
                batch_size=batch_size,
                alpha0=self.prior_pseudocount,
                laplace_ard=self.laplace_ard,
                prior_scales_mean=self.prior_scales_mean,
            )
        self.tmm.set_sizes(n_units)

        if initialization == "topk":
            nz_lines, nz_init = sparse_topk(
                lls_keep,
                log_proportions=self.log_proportions[ids].numpy(force=True),
                k=self.tmm.n_candidates,
            )
            init = np.empty((n_spikes, self.tmm.n_candidates), dtype=np.int64)
            z_lines = np.setdiff1d(np.arange(n_spikes), nz_lines)
            z_init = integers_without_inner_replacement(
                self.rg, high=n_units, size=(len(z_lines), init.shape[1])
            )
            init[z_lines] = z_init
            init[nz_lines] = nz_init
            labels = init[:, 0]
        elif initialization == "nearest":
            nz_lines, init_, *_ = loglik_reassign(lls_keep)
            init = self.rg.integers(len(ids), size=n_spikes)
            init[nz_lines] = init_
            labels = init
        else:
            assert False

        unmatched = labels < 0
        if unmatched.any():
            g = self.data.prgeom[:-1]
            coms = np.array([self[j].com(g).numpy(force=True).item() for j in ids])
            uix = self.data.split_indices["train"][unmatched]
            ux = g[self.data.original_sorting.channels[uix]].numpy(force=True)
            coms = KDTree(coms)
            _, closest = coms.query(ux, workers=-1)
            assert (closest < coms.n).all()
            labels[unmatched] = closest

        self.tmm.set_parameters(
            labels=torch.from_numpy(init),
            means=means[ids],
            bases=covs[ids].permute(0, 3, 1, 2) if covs is not None else None,
            alpha=alpha[ids] if alpha is not None else None,
            log_proportions=self.log_proportions[ids],
            noise_log_prop=self.log_proportions[-1],
            kl_divergences=dkl[keep_ids[:, None], keep_ids[None, :]],
        )

        if show_progress:
            its = trange(n_iter, desc=f"t{algorithm}", **tqdm_kw)
        else:
            its = range(n_iter)

        if algorithm == "adam":
            opt = torch.optim.Adam(tmm.parameters(), lr=sgd_lr, maximize=True)
        elif algorithm == "sgd":
            opt = torch.optim.SGD(tmm.parameters(), lr=sgd_lr, maximize=True)
        else:
            assert algorithm == "em"

        if algorithm != "em" and scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter - 1)
        else:
            assert algorithm == "em" or scheduler is None

        records = []
        train_records = []
        tic = time.perf_counter()
        labels = None
        prev_elbo = -np.inf
        done = False
        tmm_labels = None
        for j in its:
            is_final = done or j == n_iter - 1
            if is_final or algorithm == "em":
                res = self.tmm.step(
                    show_progress=step_progress,
                    hard_label=is_final,
                    with_probs=is_final and liks_from_tmm,
                    tic=tic,
                )
                tmm_labels = res.pop("labels", None)
                records.append(res)
                done = np.isclose(prev_elbo, res["obs_elbo"], atol=atol, rtol=0)
                if done:
                    logger.dartsortdebug(
                        f"Done at iteration {j} with {prev_elbo=} cur_elbo={res['obs_elbo']}"
                    )
                prev_elbo = res["obs_elbo"]
            elif algorithm in ("sgd", "adam"):
                res = self.tmm.sgd_epoch(opt, show_progress=step_progress, tic=tic)
                if scheduler is not None:
                    sched.step()
                train_records.extend(res["train_records"])
                records.append(
                    dict(
                        obs_elbo=res["obs_elbo"],
                        noise_lp=res["noise_lp"],
                        wall=res["wall"],
                    )
                )
            else:
                assert False
            msg = f"t{algorithm}[oelbo/n={res['obs_elbo']:0.2f}]"
            if show_progress:
                its.set_description(msg)  # pyright: ignore
            if is_final and j < n_iter - 1:
                break

        assert tmm_labels is not None
        if logger.isEnabledFor(DARTSORTDEBUG):
            es = np.array([r["obs_elbo"] for r in records])
            logger.dartsortdebug(
                f"Any ELBO decrease? {(np.diff(es) < 0).any()}. "
                f"The most negative change was {np.diff(es).min()}."
            )
            logger.dartsortdebug(f"ELBOs: {es.tolist()}")
            logger.dartsortdebug(f"After TVI, {np.unique(tmm_labels).shape=}")

        # reupdate my GaussianUnits
        self.clear_units()
        channels, counts = self.tmm.channel_occupancy(
            tmm_labels,
            min_count=self.channels_count_min,
            # min_prop=0.05,
            # count_per_unit=self.n_spikes_fit,
            # neighborhoods=self.data.neighborhoods(neighborhood="core")[1],
            rg=self.rg,
        )
        for j in range(len(self.tmm.means)):
            basis = None
            if self.tmm.bases is not None:
                basis = self.tmm.bases[j, ..., :-1].permute(1, 2, 0)
            alpha = None
            if self.tmm.M and self.tmm.laplace_ard:
                alpha = self.tmm.alpha[j]
            self[j] = GaussianUnit.from_parameters(
                rank=self.data.rank,
                n_channels=self.data.n_channels,
                mean=self.tmm.means[j, :, :-1],
                basis=basis,
                channels=channels[j],
                channel_counts=counts[j],
                alpha=alpha,
                **self.unit_args,
            )
        assert self.log_proportions is not None
        lps = torch.concatenate(
            (self.tmm.log_proportions, self.tmm.noise_log_prop.view(-1))
        )
        self.log_proportions = lps.log_softmax(dim=0)

        result = dict(records=records, train_records=train_records)
        if not final_e_step:
            return result

        # final e step for caller
        if (
            final_split == "train" and liks_from_tmm == "if_train"
        ) or liks_from_tmm is True:
            assert final_split == "train"
            topk_arr = (
                self.tmm.candidates.candidates[:, : self.tmm.n_candidates].numpy(
                    force=True
                ),
                res["probs"].numpy(force=True),
            )
            log_liks = topk_sparse_tocsc(
                topk_arr,
                n_rows=len(self.tmm.log_proportions),  # does not include noise
                extra_row=self.tmm.processor.noise_logliks.numpy(force=True),
                column_support=self.data.split_indices[final_split],
                n_columns_full=self.data.n_spikes,
            )
            unit_churn, reas_count, spike_logliks, log_liks = self.reassign(log_liks)
            # do what happens in e_step
        else:
            unit_churn, reas_count, log_liks, spike_logliks = self.e_step(
                show_progress=show_progress, split=final_split
            )
            log_liks, _ = self.cleanup(log_liks, relabel_split=final_split)
        result["log_liks"] = log_liks
        return result

    def em(
        self,
        n_iter=None,
        force_refit=False,
        show_progress=True,
        final_e_step=True,
        final_split="kept",
    ):
        n_iter = self.n_em_iters if n_iter is None else n_iter
        step_progress = False
        if show_progress:
            its = trange(n_iter, desc="EM", **tqdm_kw)
            step_progress = max(0, int(show_progress) - 1)
        else:
            its = range(n_iter)
        train_ix = self.data.split_indices["train"]

        # if we have no units, we can't E step.
        if force_refit:
            missing_ids = None
        else:
            missing_ids = self.missing_ids()
        if missing_ids is None or len(missing_ids):
            self.m_step(show_progress=step_progress, fit_ids=missing_ids)
            self.cleanup(min_count=1)

        convergence_props = {}
        log_liks = None
        self.train_meanlogpxs = []
        for it in its:
            is_final = it == n_iter - 1

            # for convergence testing...
            log_liks, convergence_props = self.cleanup(
                log_liks, min_count=1, clean_props=convergence_props
            )
            assert convergence_props is not None  # for typing.

            recompute_mask = None
            if "adif" in convergence_props:
                recompute_mask = convergence_props["adif"] > 0

            unit_churn, reas_count, log_liks, spike_logliks = self.e_step(
                show_progress=step_progress,
                recompute_mask=recompute_mask,
                prev_log_liks=log_liks,
            )
            convergence_props["unit_churn"] = unit_churn
            log_liks, convergence_props = self.cleanup(
                log_liks, clean_props=convergence_props
            )
            assert convergence_props is not None
            meanlogpx = spike_logliks.mean()
            self.train_meanlogpxs.append(meanlogpx.item())

            # M step: fit units based on responsibilities
            to_fit = convergence_props["unit_churn"] >= self.em_converged_churn
            mres = self.m_step(
                log_liks, show_progress=step_progress, to_fit=to_fit, compare=True
            )
            convergence_props["adif"] = mres["adif"]

            # extra info for description
            max_adif = mres["max_adif"]
            reas_prop = reas_count / self.data.n_spikes_train
            if show_progress:
                opct = (self.labels[train_ix] < 0).sum() / self.data.n_spikes_train
                opct = f"{100 * opct:.1f}"
                nu = len(to_fit)
                cpct = convergence_props["unit_churn"].max().item()
                cpct = f"{100 * cpct:.1f}"
                rpct = f"{100 * reas_prop:.1f}"
                adif = f"{max_adif:.2f}"
                msg = (
                    f"EM[K={nu},Ka={to_fit.sum()};{opct}%fp,"
                    f"{rpct}%reas,{cpct}%mc,dmu={adif};logpx/n={meanlogpx:.1f}]"
                )
                its.set_description(msg)

            if reas_prop < self.em_converged_prop:
                logger.info(f"Labels converged with {reas_prop=}.")
                if not is_final:
                    break
            if max_adif is not None and max_adif < self.em_converged_atol:
                logger.info(f"Parameters converged with {max_adif=}.")
                if not is_final:
                    break
            if len(self.train_meanlogpxs) > 2:
                logp_improvement = self.train_meanlogpxs[-1] - self.train_meanlogpxs[-2]
                if logp_improvement < self.em_converged_logpx_tol:
                    logger.info(
                        f"Log likelihood converged with {logp_improvement=} "
                        f"and {self.train_meanlogpxs=}."
                    )
                    if not is_final:
                        break

        if not final_e_step:
            return

        # final e step for caller
        unit_churn, reas_count, log_liks, spike_logliks = self.e_step(
            show_progress=step_progress, split=final_split
        )
        log_liks, _ = self.cleanup(log_liks, relabel_split=final_split)
        return log_liks

    def e_step(
        self,
        show_progress=False,
        prev_log_liks=None,
        recompute_mask=None,
        split="train",
    ):
        # E step: get responsibilities and update hard assignments
        log_liks = self.log_likelihoods(
            show_progress=show_progress,
            previous_logliks=prev_log_liks,
            recompute_mask=recompute_mask,
            split=split,
        )
        # replace log_liks by csc
        unit_churn, reas_count, spike_logliks, log_liks = self.reassign(log_liks)
        return unit_churn, reas_count, log_liks, spike_logliks

    def m_step(
        self,
        likelihoods=None,
        show_progress=False,
        to_fit=None,
        fit_ids=None,
        compare=False,
        split="train",
    ) -> dict:
        """Beware that this flattens the labels."""
        warm_start = not self.empty()
        unit_ids, spike_counts = self.label_ids(split=split)
        if to_fit is not None:
            fit_ids = unit_ids[to_fit[unit_ids]]
        total = fit_ids is None
        if total:
            fit_ids = unit_ids
        if warm_start:
            _, prev_means, *_ = self.stack_units(mean_only=True)

        if self.use_proportions and likelihoods is not None:
            self.update_proportions(likelihoods)

        fit_full_indices, fit_split_indices = quick_indices(
            self.rg,
            unit_ids.numpy(),
            self.labels.numpy(),
            split_indices=None if split is None else self.data.split_indices[split],
            max_sizes=spike_counts.clamp(max=self.n_spikes_fit).numpy(force=True),
        )

        logger.dartsortdebug(f"M step for {len(fit_ids)} units.")

        pool = Parallel(self.n_threads, backend="threading", return_as="generator")
        results = pool(
            delayed(self.fit_unit)(
                j,
                likelihoods=likelihoods,
                warm_start=warm_start,
                indices=fit_full_indices[j.item()],
                split_indices=torch.from_numpy(fit_split_indices[j.item()]),
                **self.next_round_annotations.get(j, {}),
            )
            for j in fit_ids
        )
        if show_progress:
            results = tqdm(
                results, desc="M step", unit="unit", total=len(fit_ids), **tqdm_kw
            )

        self.clear_scheduled_annotations()
        self.update(zip(fit_ids, results))

        max_adif = adif = None
        self._stack = None
        if warm_start and compare:
            ids, new_means, *_ = self.stack_units(mean_only=True)
            dmu = (prev_means - new_means).abs_().view(len(new_means), -1)
            adif_ = torch.max(dmu, dim=1).values
            max_adif = adif_.max()
            adif = torch.zeros(self.n_units())
            adif[ids] = adif_.to(adif)

        return dict(max_adif=max_adif, adif=adif)

    def log_likelihoods(
        self,
        unit_ids=None,
        with_noise_unit=True,
        use_storage=True,
        show_progress=False,
        previous_logliks=None,
        recompute_mask=None,
        split="train",
    ):
        """Noise unit last so that rows correspond to unit ids without 1 offset"""
        if unit_ids is None:
            unit_ids = self.unit_ids()

        topk_sparse = bool(self.lls_keep_k)

        # get the core neighborhood structure corresponding to this split
        split_indices, spike_neighborhoods = self.data.neighborhoods(split=split)
        _, full_core_neighborhoods = self.data.neighborhoods(split="full")

        # how many units does each core neighborhood overlap with?
        n_cores: int = spike_neighborhoods.n_neighborhoods
        with_noise_unit = self.with_noise_unit and with_noise_unit
        # everyone overlaps the noise unit
        core_overlaps = torch.full(
            (n_cores,), int(with_noise_unit), dtype=torch.int32, device=self.data.device
        )

        # for each unit, determine which spikes will be computed
        unit_neighb_info = []
        nnz = 0
        for j in unit_ids:
            unit = self[j]
            if recompute_mask is None or recompute_mask[j]:
                covered_neighbs, neighbs, ns_unit = (
                    spike_neighborhoods.subset_neighborhoods(
                        unit.channels,
                        add_to_overlaps=None if topk_sparse else core_overlaps,
                        batch_size=self.likelihood_batch_size,
                    )
                )
                if not topk_sparse:
                    unit.annotations["covered_neighbs"] = covered_neighbs
                unit_neighb_info.append((j, neighbs, ns_unit))
            else:
                assert previous_logliks is not None
                assert hasattr(previous_logliks, "row_nnz")
                ns_unit = previous_logliks.row_nnz[j]
                unit_neighb_info.append((j, ns_unit))
                if not topk_sparse:
                    covered_neighbs = unit.annotations["covered_neighbs"]
            if not topk_sparse:
                core_overlaps[covered_neighbs] += 1
            nnz += ns_unit

        if split_indices is None or split_indices == slice(None):
            n_spikes = self.data.n_spikes
            prefix = "Log likelihoods"
        else:
            n_spikes = split_indices.numel()
            prefix = f"{split.capitalize()} log likelihoods"
        logger.dartsortdebug(
            f"{prefix}: {nnz} total for {len(unit_ids)} units. That's "
            f"{nnz / len(unit_ids):0.1f}/unit or {nnz / n_spikes:0.1f}/spike."
        )

        # how many units does each spike overlap with? needed to write csc
        # embed the split indices in the global space
        spike_overlaps_ = core_overlaps[spike_neighborhoods.neighborhood_ids]
        spike_overlaps = np.zeros(self.data.n_spikes, dtype=np.int32)
        spike_overlaps[split_indices] = spike_overlaps_.numpy(force=True)

        # add in space for the noise unit
        if with_noise_unit:
            nnz = nnz + n_spikes

        @delayed
        def _ll_job(args):
            if len(args) == 2:
                j, ns = args
                ix, ll = csc_sparse_getrow(previous_logliks, j, ns)
            else:
                assert len(args) == 3
                j, neighbs, ns = args
                ix, ll = self.unit_log_likelihoods(
                    unit_id=j,
                    neighborhood_info=neighbs,
                    split=split,
                    ns=ns,
                    return_sorted=False,
                )
                if ix is not None:
                    if self.log_proportions is not None:
                        ll.add_(self.log_proportions[j].to(ll))
                    ix = ix.numpy(force=True)
                    ll = ll.numpy(force=True)
            split_ix = None
            if topk_sparse and split != "full" and ix is not None:
                # find split indices
                split_ix = np.searchsorted(split_indices.numpy(force=True), ix)
            return j, ix, ll, split_ix

        if not topk_sparse:
            # get the big nnz-length csc buffers. these can be huge so we cache them.
            csc_indices, csc_data = get_csc_storage(nnz, self.storage, use_storage)
            # csc compressed indptr. spikes are columns.
            indptr = np.concatenate(([0], np.cumsum(spike_overlaps, dtype=np.int64)))
            del spike_overlaps
            # each spike starts at writing at its indptr. as we gather more units for each
            # spike, we increment the spike's "write head". idea is to directly make csc
            write_offsets = indptr[:-1].copy()
            row_nnz = np.zeros(max(unit_ids) + 1, dtype=np.int64)
        else:
            ncols = (
                self.data.n_spikes
                if split_indices == slice(None)
                else len(split_indices)
            )
            topk_arr = allocate_topk(ncols, min(max(unit_ids) + 1, self.lls_keep_k))

        pool = Parallel(self.n_threads, backend="threading", return_as="generator")
        results = pool(_ll_job(ninfo) for ninfo in unit_neighb_info)
        if show_progress:
            results = tqdm(
                results,
                total=len(unit_neighb_info),
                desc="Likelihoods",
                unit="unit",
                **tqdm_kw,
            )
        j = -1
        for j, inds, liks, split_inds in results:
            if inds is None:
                continue
            if topk_sparse:
                ins_inds = split_inds if split_inds is not None else inds
                topk_sparse_insert(j, ins_inds, liks, topk_arr)
            else:
                row_nnz[j] = len(liks)
                csc_insert(j, write_offsets, inds, csc_indices, csc_data, liks)

        if topk_sparse:
            nlls = None
            if with_noise_unit:
                nlls = self.noise_log_likelihoods(indices=split_indices)
                if self.log_proportions is not None:
                    nlls = nlls + self.log_proportions[-1].numpy(force=True)
            log_liks = topk_sparse_tocsc(
                topk_arr,
                j + 1,
                extra_row=nlls,
                column_support=split_indices,
                n_columns_full=self.data.n_spikes,
            )
            assert log_liks.shape[0] == j + 1 + with_noise_unit
            rows, counts = np.unique(log_liks.indices, return_counts=True)
            row_nnz = np.zeros(log_liks.shape[0], dtype=np.int64)
            row_nnz[rows] = counts
            log_liks.row_nnz = row_nnz[: log_liks.shape[0] - with_noise_unit]
        else:
            if with_noise_unit:
                liks = self.noise_log_likelihoods(indices=split_indices)
                data_ixs = write_offsets[split_indices]
                # assert np.array_equal(data_ixs, ccol_indices[1:] - 1)  # just fyi
                csc_indices[data_ixs] = j + 1
                if self.log_proportions is not None:
                    liks = liks + self.log_proportions[-1].numpy(force=True)
                csc_data[data_ixs] = liks

            nrows = j + 1 + with_noise_unit
            shape = (nrows, self.data.n_spikes)
            log_liks = csc_array((csc_data, csc_indices, indptr), shape=shape)
            log_liks.has_canonical_format = True
            log_liks.row_nnz = row_nnz

        return log_liks

    def update_proportions(self, log_liks):
        if not self.use_proportions:
            return

        spike_ixs = self.data.split_indices["train"].numpy()
        spike_ixs, _ = shrinkfit(spike_ixs, self.proportions_sample_size, self.rg)
        log_liks = log_liks[:, spike_ixs]
        log_liks = log_liks.tocoo()
        log_liks = coo_to_torch(log_liks, torch.float, copy_data=True)

        # softmax over units, logged
        log_resps = torch.sparse.log_softmax(log_liks, dim=0)
        log_resps = coo_to_scipy(log_resps).tocsr()

        # now, we want the mean of the softmaxes over the spike dim (dim=1)
        # since we have log softmaxes, that means need to exp, then take mean, then log
        log_props = logmeanexp(log_resps)
        log_props = torch.asarray(log_props, dtype=torch.float, device=self.data.device)
        log_props = log_props.log_softmax(dim=0).clamp_(min=self.min_log_prop)
        self.log_proportions = log_props

    def reassign(self, log_liks):
        spike_ix, assignments, spike_logliks, log_liks_csc = loglik_reassign(
            log_liks, has_noise_unit=self.with_noise_unit, hard_noise=self.hard_noise
        )
        assignments = torch.from_numpy(assignments).to(self.labels)

        # track reassignments
        original = self.labels[spike_ix]
        same = torch.zeros_like(assignments)
        torch.eq(original, assignments, out=same)

        # total number of reassigned spikes
        reassign_count = len(same) - same.sum()

        # helpers for intersection over union
        (kept,) = (assignments >= 0).nonzero(as_tuple=True)
        (kept0,) = (original >= 0).nonzero(as_tuple=True)

        # intersection
        n_units = max(log_liks.shape[0] - self.with_noise_unit, original.max() + 1)
        intersection = torch.zeros(n_units, dtype=torch.long)
        spiketorch.add_at_(intersection, assignments[kept], same[kept])

        # union by include/exclude
        union = torch.zeros_like(intersection)
        _1 = union.new_ones((1,))
        union -= intersection
        spiketorch.add_at_(union, assignments[kept], _1.broadcast_to(kept.shape))
        spiketorch.add_at_(union, original[kept0], _1.broadcast_to(kept0.shape))

        # define "churn" as 1-iou
        iou = intersection / union
        unit_churn = 1.0 - iou

        # update labels
        self.labels.fill_(-1)
        self.labels[spike_ix] = assignments

        logger.dartsortdebug(
            f"Reassign with {100 * unit_churn.mean().item():0.2f}% mean churn."
        )

        return unit_churn, reassign_count, spike_logliks, log_liks_csc

    def cleanup(
        self,
        log_liks=None,
        min_count=None,
        clean_props=None,
        split="train",
        relabel_split="full",
        check_mean=True,
    ) -> tuple[Optional[csc_array], Optional[dict]]:
        """Remove too-small units, make label space contiguous, tidy all properties"""
        if min_count is None:
            min_count = self.min_count

        split_indices = self.data.split_indices[split]
        label_ids, counts = torch.unique(self.labels[split_indices], return_counts=True)
        counts = counts[label_ids >= 0]
        label_ids = label_ids[label_ids >= 0]
        big_enough = counts >= min_count

        n_units = 0
        if label_ids.numel():
            n_units = max(label_ids.max().item() + 1, len(self._units))
        keep = torch.zeros(n_units, dtype=torch.bool)
        countsf = torch.zeros(n_units, dtype=torch.long)
        keep[label_ids] = big_enough
        countsf[label_ids] = counts
        blank = None
        if check_mean and label_ids.numel():
            ids, means, _, _ = self.stack_units(
                ids=label_ids,
                units=[self.get(i) for i in label_ids],
                mean_only=True,
            )
            infs = torch.logical_not(means.view(len(ids), -1).isfinite().all(dim=1))
            infs = ids[infs.to(ids.device)]
            keep[infs] = False
        self._stack = None

        kept_count = "n/a" if not keep.any() else countsf[keep].min()
        logger.dartsortdebug(
            f"Retain {keep.sum()} of {n_units} units, smallest kept count was {kept_count}. "
            f"{big_enough.sum()} met yield."
        )
        if logger.isEnabledFor(DARTSORTVERBOSE):
            logger.dartsortverbose(
                f"Cleanup: {counts=} {big_enough=} {blank=} {keep=} "
                f"{label_ids=} {keep[label_ids]=} {keep.sum()=}."
            )

        if keep.all():
            return log_liks, clean_props

        if not keep.any():
            raise ValueError(
                f"GMM threw away all units. {counts=} {big_enough=} {blank=}"
            )

        if clean_props:
            clean_props = {k: v[keep] for k, v in clean_props.items()}

        keep_noise = keep.clone()
        if self.with_noise_unit:
            keep_noise = torch.concatenate((keep, torch.ones_like(keep[:1])))
        keep = keep.numpy(force=True)

        kept_ids = label_ids[keep[label_ids]]
        new_ids = torch.arange(kept_ids.numel())
        old2new = dict(zip(kept_ids, new_ids))
        self._relabel(kept_ids, split=relabel_split)

        if self.log_proportions is not None:
            self.log_proportions = self.log_proportions[keep_noise].log_softmax(0)
            self.log_proportions.clamp_(min=self.min_log_prop)

        if not self.empty():
            keep_units = {ni: self[oi] for oi, ni in zip(kept_ids, new_ids)}
            self.clear_units()
            self.update(keep_units)

        if self.next_round_annotations:
            next_round_annotations = {}
            for j, nra in self.next_round_annotations.items():
                if j in old2new:
                    next_round_annotations[old2new[j]] = nra
            self.next_round_annotations = next_round_annotations

        if log_liks is None:
            return log_liks, clean_props

        keep_ll = keep_noise.numpy(force=True)
        assert keep_ll.size == log_liks.shape[0]

        if isinstance(log_liks, coo_array):
            log_liks = coo_sparse_mask_rows(log_liks, keep_ll)
        elif isinstance(log_liks, csc_array):
            row_nnz = log_liks.row_nnz[keep]
            log_liks = csc_sparse_mask_rows(log_liks, keep_ll, in_place=True)
            log_liks.row_nnz = row_nnz
        else:
            assert False

        return log_liks, clean_props

    def merge(self, log_liks=None, show_progress=True):
        if self.n_units() <= 1:
            return
        new_labels, new_ids = self.merge_units(
            likelihoods=log_liks, show_progress=show_progress
        )
        if new_labels is None:
            return
        self.labels.copy_(torch.asarray(new_labels))

        unique_new_ids = np.unique(new_ids)
        kept_units = {}
        for new_id in unique_new_ids:
            merge_parents = np.flatnonzero(new_ids == new_id)
            self.schedule_annotations(new_id, merge_parents=merge_parents)

            if merge_parents.size == 1:
                orig_id = merge_parents.item()
                orig_id = self.normalize_key(orig_id)
                kept_units[new_id] = self[orig_id]
        self.clear_units()
        self.update(kept_units)

        if self.log_proportions is not None:
            log_props = self.log_proportions.numpy(force=True)

            # sum the proportions within each merged ID
            assert np.array_equal(unique_new_ids, np.arange(unique_new_ids.size))
            new_log_props = np.empty(
                unique_new_ids.size + self.with_noise_unit, dtype=log_props.dtype
            )
            if self.with_noise_unit:
                new_log_props[-1] = log_props[-1]
            for j in unique_new_ids:
                new_log_props[j] = logsumexp(log_props[:-1][new_ids == j])
            self.log_proportions = torch.asarray(
                new_log_props, device=self.log_proportions.device
            )

        self.log_liks = None

    def split(self, show_progress=True):
        pool = Parallel(
            self.n_threads, backend="threading", return_as="generator_unordered"
        )
        unit_ids = self.unit_ids()
        results = pool(delayed(self.kmeans_split_unit)(j) for j in unit_ids)
        if show_progress:
            results = tqdm(
                results, total=len(unit_ids), desc="Split", unit="unit", **tqdm_kw
            )

        new_count = 0
        clear_ids = []
        for res in results:
            new_count += 1
            if "new_ids" in res:
                new_count += len(res["new_ids"]) - 1
                for nid in res["new_ids"]:
                    self.schedule_annotations(nid, split_parent=res["parent_id"])
            clear_ids.extend(res["clear_ids"])

        logger.dartsortdebug(f"{new_count} units after split.")

        # split invalidates labels outside train set
        self.clear_units(clear_ids)

    def distances(
        self, kind=None, normalization_kind=None, units=None, show_progress=True
    ):
        # default to my settings but allow user to experiment
        if kind is None:
            kind = self.distance_metric
        if normalization_kind is None:
            normalization_kind = self.distance_normalization_kind

        if units is None:
            ids, units = self.ids_and_units()
            nu = max(ids) + 1
        else:
            nu = len(units)
            ids = range(nu)

        # stack unit data into one place
        mean_only = kind in ("noise_metric", "cosine")
        ids, means, covs, logdets = self.stack_units(
            nu=len(ids), ids=ids, units=units, mean_only=mean_only
        )
        n = len(ids)

        if kind == "cosine":
            means = means.view(n, -1)
            dot = means @ means.T
            norm = means.square_().sum(1).sqrt_()
            norm[norm == 0] = 1
            dot /= norm[:, None]
            dot /= norm[None, :]
            return ids, dot

        if kind in ("kl", "reverse_kl", "symkl"):
            W = None
            if covs is not None:
                W = covs.reshape(n, -1, self.ppca_rank)
            dists = spiketorch.woodbury_kl_divergence(
                C=self.noise.marginal_covariance(device=means.device),
                mu=means.reshape(n, -1),
                W=W,
            )
        elif kind == "noise_metric":
            dists = spiketorch.woodbury_kl_divergence(
                C=self.noise.marginal_covariance(device=means.device),
                mu=means.reshape(n, -1),
            )
        else:
            assert False

        if kind == "reverse_kl":
            dists = dists.T
        if kind == "symkl":
            dists *= 0.5
            dists = dists + dists.T

        dists = dists.numpy(force=True)

        if normalization_kind == "noise":
            denom = self.noise_unit.divergence(
                means, other_covs=covs, other_logdets=logdets, kind=kind
            )
            denom = denom.sqrt_().numpy(force=True).clip(min=1e-6)
            dists[:, ids] /= denom[None, :]
            dists[ids, :] /= denom[:, None]
        elif normalization_kind == "channels":
            dists /= self.data.n_channels
        else:
            assert normalization_kind in (None, "none")

        return ids, dists

    def bimodalities(
        self,
        log_liks,
        compute_mask=None,
        cut=None,
        weighted=True,
        min_overlap=None,
        masked=None,
        dt_s=2.0,
        max_spikes=2048,
        show_progress=True,
    ):
        if cut is None:
            cut = self.merge_bimodality_cut
        if cut == "auto":
            cut = None
        if min_overlap is None:
            min_overlap = self.merge_bimodality_overlap
        if masked is None:
            masked = self.merge_bimodality_masked
        nu = self.n_units()
        in_units = [
            torch.nonzero(self.labels == j, as_tuple=True)[0] for j in range(nu)
        ]
        scores = np.full((nu, nu), np.inf, dtype=np.float32)
        np.fill_diagonal(scores, 0.0)

        @delayed
        def bimod_job(i, j):
            scores[i, j] = scores[j, i] = self.unit_pair_bimodality(
                id_a=i,
                id_b=j,
                log_liks=log_liks,
                cut=cut,
                weighted=weighted,
                min_overlap=min_overlap,
                in_units=in_units,
                masked=masked,
                max_spikes=max_spikes,
                dt_s=dt_s,
            )

        if compute_mask is None:
            compute_mask = np.ones((nu, nu), dtype=bool)
        compute_mask = np.logical_or(compute_mask, compute_mask.T)
        compute_mask[np.tril_indices(nu)] = False
        ii, jj = np.nonzero(compute_mask)

        pool = Parallel(
            self.n_threads, backend="threading", return_as="generator_unordered"
        )
        results = pool(bimod_job(i, j) for i, j in zip(ii, jj))
        if show_progress:
            results = tqdm(
                results, desc="Bimodality", total=ii.size, unit="pair", **tqdm_kw
            )
        for _ in results:
            pass

        return scores

    # -- helpers

    def random_indices(
        self,
        unit_id=None,
        unit_ids=None,
        max_size=None,
        split_name="train",
    ):
        # localize labels to the split
        labels = self.labels
        if split_name is not None:
            labels = self.labels[self.data.split_indices[split_name]]

        # split_indices_full are inds relative to the split
        if unit_id is not None:
            in_u = labels == unit_id
        elif unit_ids is not None:
            in_u = torch.isin(labels, unit_ids)
        else:
            assert False
        (split_indices_full,) = in_u.nonzero(as_tuple=True)

        # convert to the full index space if nec
        if split_name is not None and split_name != "full":
            split_ixs = self.data.split_indices[split_name]
            assert isinstance(split_ixs, torch.Tensor)
            indices_full = split_ixs[split_indices_full]
        else:
            indices_full = split_indices_full

        split_indices = split_indices_full
        indices, choices = shrinkfit(indices_full, max_size, self.rg)
        split_indices = split_indices[choices]

        return indices_full, indices, split_indices

    def random_spike_data(
        self,
        unit_id=None,
        unit_ids=None,
        indices=None,
        split_indices=None,
        max_size=None,
        neighborhood="extract",
        split_name="train",
        with_reconstructions=False,
        return_full_indices=False,
        with_neighborhood_ids=False,
        allow_buffer=False,
    ):
        if indices is None and split_indices is None:
            indices_full, indices, split_indices = self.random_indices(
                unit_id=unit_id,
                unit_ids=unit_ids,
                max_size=max_size,
                split_name=split_name,
            )
        elif indices is None and split_indices is not None:
            indices = self.data.split_indices[split_name][split_indices]
        assert indices is not None

        feature_buffer = None
        if allow_buffer and split_name == "train" and neighborhood == "extract":
            feature_buffer = self.train_extract_buffer()

        sp = self.data.spike_data(
            indices=indices,
            split_indices=split_indices,
            neighborhood=neighborhood,
            with_reconstructions=with_reconstructions,
            with_neighborhood_ids=with_neighborhood_ids,
            split=split_name,
            feature_buffer=feature_buffer,
        )

        if return_full_indices:
            return indices_full, sp
        return sp

    def fit_unit(
        self,
        unit_id=None,
        unit_ids=None,
        indices=None,
        split_indices=None,
        likelihoods=None,
        weights=None,
        features=None,
        warm_start=False,
        **unit_args,
    ):
        if features is None:
            features = self.random_spike_data(
                unit_id=unit_id,
                unit_ids=unit_ids,
                indices=indices,
                split_indices=split_indices,
                max_size=self.n_spikes_fit,
                with_neighborhood_ids=True,
                allow_buffer=True,
            )
        if weights is None and likelihoods is not None:
            weights = self.get_fit_weights(
                unit_id=unit_id,
                unit_ids=unit_ids,
                indices=features.indices,
                likelihoods=likelihoods,
            )
            assert weights is not None
            assert weights.shape == (len(features),)
            (valid,) = weights.nonzero(as_tuple=True)
            valid = valid.cpu()
            weights = weights[valid]
            features = features[valid]
        unit_args = self.unit_args | unit_args

        _, train_extract_neighborhoods = self.data.neighborhoods(neighborhood="extract")
        core_neighborhoods = core_ids = None
        if self.channels_strategy.endswith("core"):
            assert features.split_indices is not None
            _, core_neighborhoods = self.data.neighborhoods()
            core_ids = core_neighborhoods.neighborhood_ids[features.split_indices]

        if unit_id is not None and warm_start and unit_id in self:
            unit = self[unit_id]
            unit.fit(
                features,
                weights,
                neighborhoods=train_extract_neighborhoods,
                core_neighborhood_ids=core_ids,
                core_neighborhoods=core_neighborhoods,
            )
        else:
            unit = GaussianUnit.from_features(
                features,
                weights,
                neighborhoods=train_extract_neighborhoods,
                core_neighborhood_ids=core_ids,
                core_neighborhoods=core_neighborhoods,
                **unit_args,
            )
        return unit

    def unit_log_likelihoods(
        self,
        unit_id=None,
        unit=None,
        spike_indices=None,
        spike_split_indices=None,
        spikes=None,
        neighborhood_info=None,
        split="train",
        ns=None,
        show_progress=False,
        ignore_channels=False,
        desc_prefix="",
        return_sorted=True,
    ):
        """Log likelihoods of core spikes for a single unit

        If spike_indices are provided, then the returned spike_indices are exactly
        those.

        Returns
        -------
        spike_indices
        log_likelihoods
        """
        if unit is None:
            unit = self[unit_id]

        if ignore_channels:
            core_channels = torch.arange(self.data.n_channels)
        else:
            core_channels = unit.channels

        if spikes is not None:
            # implies inds_already
            spike_indices = spikes.indices
            if spikes.split_indices is not None:
                spike_split_indices = spikes.split_indices

        split_indices, spike_neighborhoods = self.data.neighborhoods(split=split)
        if spike_split_indices is not None and spike_indices is None:
            spike_indices = split_indices[spike_split_indices]

        inds_already = spike_indices is not None
        if neighborhood_info is None or ns is None:
            if inds_already:
                # in this case, the indices returned in the structure are
                # relative indices inside spike_indices
                neighborhood_info, ns = spike_neighborhoods.spike_neighborhoods(
                    core_channels,
                    spike_indices=spike_split_indices,
                    neighborhood_ids=getattr(spikes, "neighborhood_ids", None),
                )
            else:
                cn, neighborhood_info, ns = spike_neighborhoods.subset_neighborhoods(
                    core_channels, batch_size=self.likelihood_batch_size
                )
                unit.annotations["covered_neighbs"] = cn
        if not ns:
            if inds_already:
                return None
            return None, None

        if inds_already:
            log_likelihoods = torch.full(
                (len(spike_indices),), -torch.inf, device=self.data.device
            )
        else:
            spike_indices = torch.empty(ns, dtype=torch.long)
            offset = 0
            log_likelihoods = torch.empty(ns)

        jobs = neighborhood_info
        if show_progress:
            jobs = tqdm(
                jobs,
                desc=f"{desc_prefix}logliks",
                total=len(neighborhood_info),
                **tqdm_kw,
            )

        for neighb_id, neighb_chans, neighb_member_ixs, batch_start in jobs:
            chans_valid = spike_neighborhoods.valid_mask(neighb_id)
            neighb_chans = neighb_chans[chans_valid]

            ixs = sixs = neighb_member_ixs

            if spikes is not None:
                sp = spikes[neighb_member_ixs]
                features = sp.features
                features = features[..., chans_valid]
            elif inds_already:
                # TODO what are the split indices here...
                sixs = None
                if spike_split_indices is not None:
                    sixs = spike_split_indices[neighb_member_ixs]
                sp = self.data.spike_data(
                    spike_indices[neighb_member_ixs],
                    split_indices=sixs,
                    with_channels=False,
                    neighborhood="core",
                )
                features = sp.features
                features = features[..., chans_valid]
            elif spike_neighborhoods.has_feature_cache():
                features = spike_neighborhoods.neighborhood_features(
                    neighb_id,
                    batch_start=batch_start,
                    batch_size=self.likelihood_batch_size,
                )
                features = features.to(self.data.device)
            else:
                # full split case
                assert split_indices == slice(None)

                sp = self.data.spike_data(ixs, with_channels=False, neighborhood="core")
                features = sp.features
                features = features[..., chans_valid]

            lls = unit.log_likelihood(features, neighb_chans, neighborhood_id=neighb_id)

            if inds_already:
                log_likelihoods[neighb_member_ixs.to(log_likelihoods.device)] = lls
            else:
                nbatch = len(lls)
                if split_indices is not None and split_indices != slice(None):
                    neighb_member_ixs = split_indices[neighb_member_ixs]
                spike_indices[offset : offset + nbatch] = neighb_member_ixs
                log_likelihoods[offset : offset + nbatch] = lls
                offset += nbatch

        if return_sorted and not inds_already:
            spike_indices, order = spike_indices.sort()
            log_likelihoods = log_likelihoods[order]

        if inds_already:
            return log_likelihoods

        return spike_indices, log_likelihoods

    def dense_log_likelihoods(
        self,
        spikes,
        unit_ids=None,
        units=None,
        log_proportions=None,
        ignore_channels=True,
    ):
        full = False
        if unit_ids is not None:
            units = [self[u] for u in unit_ids]
            log_proportions = self.log_proportions[unit_ids]

        if units is None:
            ids = self.unit_ids()
            assert np.array_equal(ids, np.arange(len(ids)))
            units = self._units.values()
            log_proportions = self.log_proportions
            nu = len(units) + 1
            full = True
        else:
            nu = len(units)

        liks = spikes.features.new_full((nu, len(spikes)), -torch.inf)
        for j, (u, lp) in enumerate(zip(units, log_proportions)):
            ull = self.unit_log_likelihoods(
                unit=u, spikes=spikes, ignore_channels=ignore_channels
            )
            if ull is not None:
                liks[j] = ull.add_(lp)

        if full:
            nll = self.noise_log_likelihoods(indices=spikes.indices)
            liks[-1] = torch.asarray(nll).to(liks)
            liks[-1] += self.log_proportions[-1]

        return liks

    def noise_log_likelihoods(self, indices=None, show_progress=False):
        if self._noise_log_likelihoods is None:
            _noise_six, _noise_log_likelihoods = self.unit_log_likelihoods(
                unit=self.noise_unit,
                show_progress=show_progress,
                desc_prefix="Noise ",
                split="full",
            )
            del _noise_six  # noise overlaps with all, ignore.
            self._noise_log_likelihoods = _noise_log_likelihoods.numpy(force=True)
        self.core_noise_truncation_factors = None
        self.train_extract_noise_truncation_factors = None
        if self.truncated_noise:
            self.core_noise_truncation_factors = self.noise_mahal_chi_survival(
                split="full", neighborhood="core"
            )
            self.train_extract_noise_truncation_factors = self.noise_mahal_chi_survival(
                split="train", neighborhood="extract"
            )
            nb_core_full = self.data.neighborhoods(split="full", neighborhood="core")[1]
            self._noise_log_likelihoods += self.core_noise_truncation_factors[
                nb_core_full.neighborhood_ids
            ].numpy(force=True)
        if indices is not None:
            return self._noise_log_likelihoods[indices]
        return self._noise_log_likelihoods

    def kmeans_split_unit(
        self,
        unit_id,
        debug=False,
        merge_kind=None,
        criterion=None,
        decision_algorithm=None,
        ignore_channels=True,
        kmeans_n_iter=None,
        min_overlap=None,
        distance_metric=None,
        distance_normalization_kind=None,
    ):
        if criterion is None:
            criterion = self.criterion
        if kmeans_n_iter is None:
            kmeans_n_iter = self.kmeans_n_iter
        logger.dartsortverbose(
            f"Split {unit_id}: {criterion=} {decision_algorithm=} {ignore_channels=} "
            f"{min_overlap=} {distance_metric=} {kmeans_n_iter=}."
        )

        # get spike data and use interpolation to fill it out to the
        # unit's channel set
        result = dict(parent_id=unit_id, new_ids=[unit_id], clear_ids=[])
        unit = self[unit_id]
        if not unit.channels.numel():
            return result

        indices_full, sp = self.random_spike_data(
            unit_id,
            return_full_indices=True,
            with_neighborhood_ids=True,
            max_size=self.n_spikes_fit,
        )
        if not indices_full.numel() > self.min_count:
            return result

        Xo = X = self.data.interp_to_chans(sp, unit.channels)
        if self.split_whiten:
            X = self.noise.whiten(X, channels=unit.channels)

        if debug:
            result.update(indices_full=indices_full, sp=sp, X=Xo, Xw=X)
        else:
            del Xo

        # run kmeans with kmeans++ initialization
        k = min(self.kmeans_k, len(X) // (self.min_count / 2))
        kmeans_res = kmeans(
            X.view(len(X), -1),
            n_iter=kmeans_n_iter,
            n_components=self.kmeans_k,
            random_state=self.rg,
            kmeanspp_initial=self.kmeans_kmeanspp_initial,
            with_proportions=self.kmeans_with_proportions,
            drop_prop=self.kmeans_drop_prop,
            drop_sum=self.channels_count_min,
        )
        split_labels = kmeans_res["labels"]
        responsibilities = kmeans_res["responsibilities"]
        assert split_labels is not None
        del kmeans_res
        if debug:
            result["split_labels"] = split_labels
            result["responsibilities"] = responsibilities

        split_labels = split_labels.cpu()
        split_ids, split_labels = split_labels.unique(return_inverse=True)
        assert split_ids.min() >= 0
        if split_labels.unique().numel() <= 1:
            return result
        assert responsibilities is not None
        responsibilities = responsibilities[:, split_ids]

        # avoid oversplitting by doing a mini merge here
        split_labels = self.split_decision(
            unit_id,
            hyp_fit_spikes=sp,
            hyp_fit_resps=responsibilities.T,
            criterion=criterion,
            debug_info=result if debug else None,
            decision_algorithm=decision_algorithm,
            ignore_channels=ignore_channels,
            min_overlap=min_overlap,
            distance_metric=distance_metric,
            distance_normalization_kind=distance_normalization_kind,
        )
        if split_labels is None:
            logger.dartsortverbose(f"Split {unit_id} bailed.")
            return result
        # flatten the label space
        kept = np.flatnonzero(split_labels >= 0)
        if not kept.size:
            logger.dartsortverbose(f"Split {unit_id} threw away all spikes.")
            return result
        split_ids, flat_labels, split_counts = np.unique(
            split_labels[kept], return_inverse=True, return_counts=True
        )
        logger.dartsortverbose(
            f"Split {unit_id} into {split_ids.size} / {split_counts.tolist()}, "
            f"with {split_labels.numel() - kept.size} -1s; original had "
            f"{indices_full.numel()} train spikes in full, split ran on {len(sp)}."
        )
        n_new_units = split_ids.size - 1
        del split_ids  # just making this clear, because those ids changed anyway
        split_labels[kept] = torch.from_numpy(flat_labels)

        if debug:
            result["merge_labels"] = split_labels
            return result

        # need to add my val inds into the indices_full
        all_indices_full, *_ = self.random_indices(unit_id=unit_id, split_name=None)

        split_labels = torch.asarray(split_labels, device=self.labels.device)
        if n_new_units < 1:
            # quick case
            with self.labels_lock:
                self.labels[all_indices_full] = -1
                self.labels[sp.indices[kept]] = unit_id
            if kept.size < self.min_count:
                result["clear_ids"] = [unit_id]
            return result

        # else, tack new units onto the end
        # we need to lock up the labels array access here because labels.max()
        # depends on what other splitting units are doing!
        with self.labels_lock:
            # new indices start here
            next_label = self.labels.max() + 1

            # new indices are already >= 1, so subtract 1
            split_labels[split_labels >= 1] += next_label - 1
            split_labels[split_labels == 0] = unit_id
            logger.dartsortverbose(
                f"Split {unit_id}: my new labels are {split_labels.unique()}."
            )

            # unit 0 takes the place of the current unit
            self.labels[all_indices_full] = -1
            self.labels[sp.indices] = split_labels

            if self.log_proportions is None:
                return

            # each sub-unit's prop is its fraction of assigns * orig unit prop
            new_log_props = torch.asarray(np.log(split_counts))
            new_log_props = new_log_props.to(self.log_proportions)
            new_log_props = new_log_props.log_softmax(0) + self.log_proportions[unit_id]
            assert new_log_props.numel() == n_new_units + 1

            cur_len_with_noise = self.log_proportions.numel()
            noise_log_prop = self.log_proportions[-1]
            self.log_proportions = torch.cat(
                (self.log_proportions, self.log_proportions.new_empty(n_new_units)),
                dim=0,
            )
            self.log_proportions[unit_id] = new_log_props[0]
            self.log_proportions[cur_len_with_noise - 1 : -1] = new_log_props[1:]
            self.log_proportions[-1] = noise_log_prop

        new_ids = torch.unique(split_labels)
        new_ids = new_ids[new_ids >= 0]
        result["new_ids"] = result["clear_ids"] = new_ids
        return result

    def unit_pair_bimodality(
        self,
        id_a,
        id_b,
        log_liks,
        loglik_ix_a=None,
        loglik_ix_b=None,
        cut=None,
        weighted=True,
        min_overlap=None,
        in_units=None,
        masked=True,
        max_spikes=2048,
        dt_s=2.0,
        score_kind=None,
        debug=False,
    ):
        if score_kind is None:
            score_kind = self.merge_bimodality_score_kind
        if cut is None:
            cut = self.merge_bimodality_cut
        if cut == "auto":
            cut = None
        if min_overlap is None:
            min_overlap = self.merge_bimodality_overlap
        if in_units is not None:
            ina = in_units[id_a]
            inb = in_units[id_b]
        else:
            (ina,) = torch.nonzero(self.labels == id_a, as_tuple=True)
            (inb,) = torch.nonzero(self.labels == id_b, as_tuple=True)

        if min(ina.numel(), inb.numel()) < 10:
            if debug:
                return dict(score=np.inf)
            return np.inf

        if masked:
            times_a = self.data.times_seconds[ina]
            times_b = self.data.times_seconds[inb]
            ina = ina[(getdt(times_b, times_a) <= dt_s).cpu()]
            inb = inb[(getdt(times_a, times_b) <= dt_s).cpu()]

        ina, _ = shrinkfit(ina, max_spikes, self.rg)
        inb, _ = shrinkfit(inb, max_spikes, self.rg)

        in_pair = torch.concatenate((ina, inb))
        is_b = torch.zeros(in_pair.shape, dtype=bool)
        is_b[ina.numel() :] = 1
        in_pair, order = in_pair.sort()
        is_b = is_b[order]

        lixa = id_a if loglik_ix_a is None else loglik_ix_a
        lixb = id_b if loglik_ix_b is None else loglik_ix_b
        # a - b. if >0, a>b.
        log_lik_diff = get_diff_sparse(
            log_liks, lixa, lixb, in_pair, return_extra=debug
        )

        debug_info = None
        if debug:
            log_lik_diff, extra = log_lik_diff
            debug_info = {}
            debug_info["log_lik_diff"] = log_lik_diff
            # adds keys: xi, xj, keep_inds
            debug_info.update(extra)
            debug_info["in_pair_kept"] = in_pair[extra["keep_inds"]]
            # qda adds keys: domain, alternative_density, cut, score, score_kind,
            # uni_density, sample_weights, samples

        score = qda(
            is_b.numpy(force=True),
            diff=log_lik_diff,
            cut=cut,
            weighted=weighted,
            min_overlap=min_overlap,
            score_kind=score_kind,
            debug_info=debug_info,
        )
        if debug:
            return debug_info
        return score

    def tree_merge(
        self,
        distances,
        current_log_liks,
        unit_ids=None,
        max_distance=1e6,
        threshold=None,
        criterion="elbo",
        sym_function=np.minimum,
        max_group_size=8,
        min_overlap=None,
        show_progress=False,
        decision_algorithm=None,
        brute_size=4,
        cosines=None,
        reevaluate_cur_liks=True,
    ):
        r"""Tree merge

        The final decision is a group of leaf and non-leaf nodes whose descendants
        partition the leaves. Each leaf is assigned to its highest-value major
        ancestor, where an ancestor is major if it has higher value than all
        of its descendants.
        """
        if threshold is None:
            threshold = self.criterion_threshold
        if decision_algorithm is None:
            decision_algorithm = self.merge_decision_algorithm
        if min_overlap is None:
            min_overlap = self.min_overlap

        if distances.shape[0] == 1:
            return None, None, None, None, None

        # heuristic unit groupings to investigate
        np.fill_diagonal(distances, 0.0)  # sometimes numerically negative...
        distances = sym_function(distances, distances.T)
        distances = distances[np.triu_indices(len(distances), k=1)]
        finite = np.isfinite(distances)
        if not finite.any():
            return None, None, None, None, None
        if not finite.all():
            big = max(0, distances[finite].max()) + max_distance + 1
            distances[np.logical_not(finite)] = big
        Z = linkage(distances)
        n_units = len(Z) + 1
        brute_size = min(n_units, brute_size)
        n_branches = n_units - 1

        if unit_ids is None:
            unit_ids = np.arange(n_units)
        else:
            unit_ids = np.asarray(unit_ids)
            assert len(unit_ids) == n_units
        # figure out the leaf nodes in each cluster in the hierarchy
        # up to max_distance
        leaf_descendants = leafsets(Z)

        # walk up from the leaves
        its = enumerate(Z)
        if show_progress:
            step = decision_algorithm
            if step == "brute":
                step = "brute step 1/2"
            its = tqdm(its, desc=f"Merge: {step}", total=n_branches, **tqdm_kw)

        # and build this set of data:
        # improvements: for a branch, how much does the model improve by
        # merging the corresponding subtree?
        improvements = np.full(n_branches, -np.inf)
        # overlaps: for a branch, the proportion of spikes which the merge
        # unit overlapped with
        overlaps = np.full(n_branches, -np.inf)
        # leaf scores: the CURRENT BEST improvement for an ancestor branch
        # of each leaf node. initialized to zero because the leaves don't
        # change the model.
        leaf_scores = np.zeros(n_units, dtype=bool)
        # group IDs: the CURRENT BEST group label for each leaf node.
        # if we encounter a new best score for all leaves in a subtree,
        # we update the leaf scores and the group id is set to the
        group_ids = np.arange(n_units)
        # was the brute force algorithm used in this branch?
        brute_indicator = is_largest_set_smaller_than(
            Z,
            leaf_descendants,
            max_size=brute_size if decision_algorithm == "brute" else -1,
        )

        brute_jobs = []
        shared_args = (
            current_log_liks,
            cosines,
            min_overlap,
            criterion,
            reevaluate_cur_liks,
        )

        for i, (pa, pb, dist, nab) in its:
            if not np.isfinite(dist) or dist > max_distance:
                continue

            # check if should merge
            leaves = leaf_descendants[n_units + i]
            if len(leaves) > max_group_size:
                continue
            cluster_ids = unit_ids[leaves]

            if not brute_indicator[i] and (
                decision_algorithm == "tree" or len(leaves) > brute_size
            ):
                # groups larger than brute_size are handled by tree case
                level_cosines = None
                if cosines is not None:
                    level_cosines = cosines[cluster_ids][:, cluster_ids]
                crit = self.validation_criterion(
                    current_log_liks,
                    current_unit_ids=cluster_ids,
                    reevaluate_cur_liks=reevaluate_cur_liks,
                    cosines=level_cosines,
                    in_bag=not criterion.startswith("heldout_"),
                )
                overlaps[i] = crit["overlap"]
                if overlaps[i] >= min_overlap:
                    improvements[i] = crit["improvements"][
                        criterion.removeprefix("heldout_")
                    ]
                if improvements[i] >= leaf_scores[leaves].max():
                    leaf_scores[leaves] = improvements[i]
                    group_ids[leaves] = n_units + i

            elif brute_indicator[i]:
                # the brute force thing is slow, so let's do it in parallel.
                brute_jobs.append(
                    delayed(self._brute_merge_job)(i, leaves, cluster_ids, *shared_args)
                )

        if decision_algorithm == "brute":
            n_jobs = min(self.n_threads, len(brute_jobs))
            pool = Parallel(
                n_jobs, backend="threading", return_as="generator_unordered"
            )
            results = pool(brute_jobs)
            if show_progress:
                desc = "Merge: brute step 2/2"
                results = tqdm(
                    results, desc=desc, unit="branch", total=len(brute_jobs), **tqdm_kw
                )
            for (
                i,
                leaves,
                cluster_ids,
                brute_group_ids,
                brute_improvement,
                brute_overlap,
            ) in results:
                improvements[i] = brute_improvement
                if brute_improvement > 0:
                    result_group_ids = []
                    for bgid in brute_group_ids:
                        if bgid == 0:
                            result_group_ids.append(n_units + i)
                        else:
                            result_group_ids.append(cluster_ids[bgid])
                    group_ids[leaves] = result_group_ids
                    overlaps[i] = brute_overlap
                    leaf_scores[leaves] = brute_improvement

        return Z, group_ids, improvements, overlaps, brute_indicator

    def _brute_merge_job(
        self,
        i,
        leaves,
        cluster_ids,
        current_log_liks,
        cosines,
        min_overlap,
        criterion,
        reevaluate_cur_liks,
    ):
        brute_group_ids, brute_improvement, brute_overlap = self.brute_merge(
            current_log_liks,
            cluster_ids,
            cosines=cosines,
            min_overlap=min_overlap,
            criterion=criterion,
            reevaluate_cur_liks=reevaluate_cur_liks,
        )
        return i, leaves, cluster_ids, brute_group_ids, brute_improvement, brute_overlap

    def brute_merge(
        self,
        log_likelihoods,
        current_unit_ids,
        min_overlap=0.8,
        criterion=None,
        cosines=None,
        min_cosine=0.5,
        reevaluate_cur_liks=True,
    ):
        if criterion is None:
            criterion = self.criterion
        units = [self[cuid] for cuid in current_unit_ids]
        current_unit_ids = torch.tensor(current_unit_ids)
        n_cur = len(units)

        best_improvement = 0.0
        best_group_ids = np.arange(n_cur)
        best_overlap = 1.0

        # units_memo = {(ci.item(),): self[ci] for ci in current_unit_ids}
        units_memo = {}

        for group_ids, part, ids_part in all_partitions(current_unit_ids):
            # responsibilities and memoized units at this level
            level_ids_part = [p for p in ids_part if len(p) > 1]
            level_current_ids = sum(level_ids_part, start=())
            if not len(level_current_ids):
                continue

            min_conn = np.inf
            for ip in level_ids_part:
                if ip not in units_memo:
                    logger.dartsortverbose(f"brute_merge: Fit %s.", ip)
                    try:
                        units_memo[ip] = self.fit_unit(
                            unit_ids=torch.tensor(ip), likelihoods=log_likelihoods
                        )
                    except ValueError as e:
                        raise ValueError(f"Couldn't fit {ip=} in merge.") from e
                if len(ip) > 1 and cosines is not None:
                    ip = np.array(ip)
                    conn = cos_connectivity(cosines[ip][:, ip])
                    min_conn = min(conn, min_conn)
            # for i in level_current_ids:
            #     if (i,) not in units_memo:
            #         logger.dartsortverbose(f"brute_merge: Fit %s.", i)
            #         try:
            #             units_memo[(i,)] = self.fit_unit(
            #                 unit_ids=torch.tensor((i,)), likelihoods=log_likelihoods
            #             )
            #         except ValueError as e:
            #             raise ValueError(f"Couldn't fit {i=} in merge.") from e
            if min_conn < min_cosine:
                logger.dartsortverbose(
                    f"brute_merge: Bail on %s (%s) because conn was %s.",
                    ids_part,
                    level_ids_part,
                    min_conn,
                )
                continue

            level_units = [units_memo[ip] for ip in level_ids_part]
            level_lp = [
                self.log_proportions[list(ip)].logsumexp(0) for ip in level_ids_part
            ]
            level_lp = torch.tensor(level_lp).to(self.log_proportions)

            crit = self.validation_criterion(
                log_likelihoods,
                current_unit_ids=level_current_ids,
                hyp_units=level_units,
                hyp_log_props=level_lp,
                reevaluate_cur_liks=reevaluate_cur_liks,
                # cur_units=[units_memo[(i,)] for i in level_current_ids],
                in_bag=not criterion.startswith("heldout_"),
            )
            overlap = crit["overlap"]
            if overlap < min_overlap:
                continue
            improvement = crit["improvements"][criterion.removeprefix("heldout_")]
            logger.dartsortverbose(
                "brute_merge: ids_part=%s (%s) group_ids=%s imp=%s level_lp=%s cur_lp=%s",
                ids_part,
                level_ids_part,
                group_ids,
                improvement,
                level_lp,
                self.log_proportions[current_unit_ids].tolist(),
            )

            # store it as best if it was
            if improvement > best_improvement:
                best_group_ids = group_ids
                best_improvement = improvement
                best_overlap = overlap

        return best_group_ids, best_improvement, best_overlap

    def split_decision(
        self,
        unit_id,
        hyp_fit_spikes,
        hyp_fit_resps,
        criterion=None,
        sym_function=np.minimum,
        max_distance=None,
        debug_info=None,
        decision_algorithm=None,
        ignore_channels=True,
        fit_same_channels=False,
        min_overlap=None,
        distance_metric=None,
        min_cosine=0.5,
        distance_normalization_kind=None,
    ):
        debug = debug_info is not None
        current_unit_ids = [unit_id]
        if criterion is None:
            criterion = self.criterion
        if max_distance is None:
            max_distance = self.merge_distance_threshold
        if decision_algorithm is None:
            decision_algorithm = self.split_decision_algorithm
        if min_overlap is None:
            min_overlap = self.min_overlap
        if distance_metric is None:
            distance_metric = self.distance_metric
        if distance_normalization_kind is None:
            distance_normalization_kind = self.distance_normalization_kind

        # -- evaluate full model
        fit_channels = self[unit_id].channels.clone() if fit_same_channels else None
        n_fit = hyp_fit_resps.shape[1]
        if n_fit < self.min_count:
            return None
        assert len(hyp_fit_spikes) == n_fit
        full_info = self.validation_criterion(
            self.log_liks,
            current_unit_ids=current_unit_ids,
            hyp_fit_resps=hyp_fit_resps,
            hyp_fit_spikes=hyp_fit_spikes,
            hyp_fit_channels=fit_channels,
            ignore_channels=ignore_channels,
            in_bag=not criterion.startswith("heldout_"),
        )
        best_improvement = full_info["improvements"][criterion.removeprefix("heldout_")]
        units = full_info["hyp_units"]
        n_units = len(units)
        best_group_ids = np.arange(n_units)
        full_labels = hyp_fit_resps.argmax(0)
        if debug:
            debug_info["reas_labels"] = full_labels
            debug_info["units"] = units
            debug_info["full_improvement"] = best_improvement
        if n_units <= 1:
            if debug:
                debug_info["bail"] = f" since {n_units=} {hyp_fit_resps.shape=}"
            return None

        cur_fit_liks = self.get_log_likelihoods(hyp_fit_spikes.indices, self.log_liks)
        cur_resp = torch.sparse.softmax(cur_fit_liks, dim=0)
        cur_resp = cur_resp.index_select(
            dim=0,
            index=torch.asarray(
                current_unit_ids, dtype=torch.long, device=cur_resp.device
            ),
        )
        cur_resp = cur_resp.sum(dim=0).to_dense()

        _, cosines = self.distances(
            units=units,
            show_progress=False,
            kind="cosine",
        )

        full_conn = cos_connectivity(cosines)

        if decision_algorithm == "tree" or debug:
            # -- make linkage and leafsets
            _, distances = self.distances(
                units=units,
                show_progress=False,
                kind=distance_metric,
                normalization_kind=distance_normalization_kind,
            )
            np.fill_diagonal(distances, 0.0)
            assert distances.shape == (n_units, n_units)  # pyright: ignore
            if debug:
                debug_info["distances"] = distances
            distances = sym_function(distances, distances.T)
            distances = distances[np.triu_indices(n_units, k=1)]
            if not (distances > -1e-4).all():
                warnings.warn(f"Minimum distance in split was {distances.min():0.3f}?")
            distances = distances.clip(min=0.0)
            finite = np.isfinite(distances)
            if not finite.any():
                return None
            if not finite.all():
                big = max(0, distances[finite].max()) + max_distance + 1
                distances[np.logical_not(finite)] = big
        if decision_algorithm == "tree":
            Z = linkage(distances)
            assert n_units == len(Z) + 1
            n_branches = n_units - 1
            leaf_descendants = leafsets(Z)

            # -- evaluate submodels
            # group IDs: the CURRENT group label for each leaf node.
            group_ids = np.arange(n_units)
            # best group IDs: the current BEST group label for each leaf node.
            best_group_ids = group_ids.copy()
            if debug:
                debug_info["Z"] = Z
                debug_info["improvements"] = np.full(n_branches, -np.inf)
                debug_info["overlaps"] = np.full(n_branches, -np.inf)

            for i, (pa, pb, dist, nab) in enumerate(Z):
                if not np.isfinite(dist) or dist > max_distance:
                    continue

                # the partition at this level is...
                leaves = leaf_descendants[n_units + i]
                group_ids[leaves] = n_units + i

                # combine the resps to match the partition
                cur_ids = np.unique(group_ids)
                level_resps = hyp_fit_resps.new_zeros((len(cur_ids), n_fit))
                min_conn = np.inf
                for j, cid in enumerate(cur_ids):
                    in_cid = group_ids == cid
                    in_cid = np.flatnonzero(in_cid)
                    level_resps[j] = hyp_fit_resps[in_cid].sum(0)

                    if len(in_cid) > 1:
                        conn = cos_connectivity(cosines[in_cid][:, in_cid])
                        min_conn = min(conn, min_conn)
                if min_conn < min_cosine:
                    continue

                # evaluate the corresponding model
                crit = self.validation_criterion(
                    self.log_liks,
                    current_unit_ids=current_unit_ids,
                    hyp_fit_spikes=hyp_fit_spikes,
                    hyp_fit_resps=level_resps,
                    hyp_fit_channels=fit_channels,
                    cur_resp=cur_resp,
                    ignore_channels=ignore_channels,
                    in_bag=not criterion.startswith("heldout_"),
                )
                improvement = crit["improvements"][criterion.removeprefix("heldout_")]
                olap = crit["overlap"]

                # store it as best if it was
                if olap >= min_overlap and improvement > best_improvement:
                    best_group_ids = group_ids.copy()
                    best_improvement = improvement
                if debug:
                    debug_info["improvements"][i] = improvement
                    debug_info["overlaps"][i] = crit["overlap"]
                    if "level_units" not in debug_info:
                        debug_info["level_units"] = {}
                    debug_info["level_units"][i] = crit["hyp_units"]
        elif decision_algorithm == "brute":
            merged_unit_memo = {}
            for jj, subunit in enumerate(units):
                merged_unit_memo[(jj,)] = subunit
            for group_ids, part, ids_part in all_partitions(np.arange(len(units))):
                # responsibilities and memoized units at this level
                level_resps = hyp_fit_resps.new_empty((len(part), n_fit))
                level_units = [None] * len(part)
                olap = 1.0
                min_conn = np.inf
                for j, p in enumerate(part):
                    lresps = hyp_fit_resps[p]
                    level_resps[j] = lresps.sum(0)
                    if tuple(ids_part[j]) in merged_unit_memo:
                        level_units[j] = merged_unit_memo[tuple(ids_part[j])]
                    if len(ids_part[j]) > 1:
                        ipj = np.array(ids_part[j])
                        conn = cos_connectivity(cosines[ipj][:, ipj])
                        min_conn = min(conn, min_conn)
                if min_conn < min_cosine:
                    continue

                crit = self.validation_criterion(
                    self.log_liks,
                    current_unit_ids=current_unit_ids,
                    hyp_fit_spikes=hyp_fit_spikes,
                    hyp_fit_resps=level_resps,
                    hyp_fit_channels=fit_channels,
                    cur_resp=cur_resp,
                    ignore_channels=ignore_channels,
                    in_bag=not criterion.startswith("heldout_"),
                )
                improvement = crit["improvements"][criterion.removeprefix("heldout_")]
                logger.dartsortverbose(f"Split {unit_id}: {ids_part=} {improvement=}.")

                # memoize
                for j, hu in enumerate(crit["hyp_units"]):
                    if tuple(ids_part[j]) not in merged_unit_memo:
                        merged_unit_memo[tuple(ids_part[j])] = hu

                # store it as best if it was
                if improvement > best_improvement:
                    best_group_ids = group_ids
                    best_improvement = improvement
                    if debug:
                        debug_info["level_units"] = {0: [u for u in crit["hyp_units"]]}
                        debug_info["improvements"] = [improvement]
                        debug_info["ids_part"] = ids_part
                        debug_info["overlap"] = olap
        else:
            assert False

        # -- organize labels...
        best_improvement = best_improvement
        assert np.isfinite(best_improvement)
        # if best_improvement < 0:
        #     return None
        best_group_ids = torch.asarray(best_group_ids)
        labels = best_group_ids[full_labels.cpu()]
        _, labels = labels.unique(return_inverse=True)
        return labels

    def validation_criterion(
        self,
        current_log_liks,
        current_unit_ids,
        hyp_units=None,
        hyp_fit_spikes=None,
        hyp_fit_resps=None,
        hyp_log_props=None,
        cur_resp=None,
        hyp_fit_channels=None,
        label_fit_spikes=False,
        fit_max_factor=2,
        ignore_channels=True,
        reevaluate_cur_liks=True,
        refit_cur_units=False,
        cur_units=None,
        cosines=None,
        min_cosine=0.5,
        in_bag=False,
    ) -> dict[
        Literal["improvements", "overlap", "hyp_units", "eval_labels", "fit_labels"],
        Any,
    ]:
        """Validation criteria to choose between current or hypothetical model

        This code handles two cases: splitting and merging.

        In both cases, a validation statistic is computed for the current
        model and for a hypothetical model: V_C and V_H.

        In the split case, current_unit_ids contains a single ID, and
        hypothetical resps indicate how to divide up its responsibilities
        when fitting split subunits. Then the split should be accepted if
        V_H>V_C.

        In the merge case, current_unit_ids contains several IDs, and
        hypothetical resps are implicitly 1s (set to None), so that the
        merged unit to be fit gathers responsibilities from all the
        current_unit_ids. Then the merge should be accepted if V_H>V_C.

        Arguments
        ---------
        current_log_liks : csr_array
            Sparse array of log likelihoods for the current set of units
        current_unit_ids : list[int]
            IDs of current units on the chopping block
        hyp_fit_spikes : SpikeFeatures
            Spikes to use to fit hypothetical units. This is even needed
            in the split case! That's because during the split we consider
            merging subsets of the split units. So splits are kinda merges
            themselves...
            If None, these are chosen at random, and hyp_fit_resps better
            be None too.
        hyp_fit_resps : Tensor (n_spikes_fit, n_hyp_units)
            n_fit_units units will be fit using these weights to the
            spikes above. Weights will be determined by dividing the
            sum of current unit responsibilities among the hyp units
            using these responsibilities.
            If None, these are assumed to be ones, and the resulting
            responsibility is the sum of current units' resps.
        """
        verbose = logger.isEnabledFor(DARTSORTVERBOSE)

        # -- validate args
        current_unit_ids = np.array(current_unit_ids)
        corder = np.argsort(current_unit_ids)
        if cur_units is not None:
            cur_units = [cur_units[o] for o in corder]
        current_unit_ids = current_unit_ids[corder]
        n_cur = len(current_unit_ids)
        irrix = np.setdiff1d(np.arange(current_log_liks.shape[0]), current_unit_ids)
        current_unit_ids = torch.asarray(current_unit_ids, dtype=torch.long)
        irrix = torch.tensor(list(irrix))

        # -- check cosine connectivity
        if cosines is not None:
            assert cosines.shape == (n_cur, n_cur)
            conn = cos_connectivity(cosines)
            if conn < min_cosine:
                logger.dartsortverbose(f"vc: Too small {conn=}")
                return {
                    "improvements": dict(elbo=-np.inf, loglik=-np.inf),
                    "overlap": conn,
                    "hyp_units": None,
                    "eval_labels": None,
                    "fit_labels": None,
                }

        # -- fit hypothetical units
        # pick spikes to fit (if necessary)
        if hyp_fit_spikes is None:
            fit_max_count = min(fit_max_factor, n_cur) * self.n_spikes_fit
            hyp_fit_spikes = self.random_spike_data(
                unit_ids=current_unit_ids,
                with_neighborhood_ids=True,
                max_size=fit_max_count,
            )

        # get total current responsibility for fit spikes
        fit_any = hyp_units is None or any(u is None for u in hyp_units)
        if fit_any and cur_resp is None:
            cur_fit_liks = self.get_log_likelihoods(
                hyp_fit_spikes.indices, current_log_liks
            )
            cur_resps = torch.sparse.softmax(cur_fit_liks, dim=0)
            cur_resps = cur_resps.index_select(
                dim=0, index=current_unit_ids.to(cur_resps.device)
            ).to_dense()
            cur_resp = cur_resps.sum(dim=0)
        if fit_any:
            # fit weights for hypothetical units
            hyp_fit_weights = cur_resp.unsqueeze(0)
            if hyp_fit_resps is not None:
                hyp_fit_weights = hyp_fit_weights * hyp_fit_resps
            n_hyp, n_fit = hyp_fit_weights.shape
            assert n_fit == len(hyp_fit_spikes)
            if hyp_units is not None:
                given_hyp_units = hyp_units
            else:
                given_hyp_units = [None] * n_hyp
            hyp_units = [None] * n_hyp
            for j, w in enumerate(hyp_fit_weights):
                if given_hyp_units[j] is not None:
                    hyp_units[j] = given_hyp_units[j]
                    continue
                hyp_units[j] = self.fit_unit(
                    features=hyp_fit_spikes, weights=w, channels=hyp_fit_channels
                )

        assert hyp_units is not None
        n_hyp = len(hyp_units)
        logger.dartsortverbose(f"vc: {current_unit_ids=} {n_cur=} {n_hyp=}")

        # what are the log props?
        cur_log_prop = self.log_proportions[current_unit_ids].logsumexp(
            dim=0, keepdim=True
        )
        if hyp_fit_resps is not None:
            hyp_rel_log_props = hyp_fit_resps.mean(1).log_()
            hyp_rel_log_props = torch.log_softmax(hyp_rel_log_props, dim=0)
            hyp_log_props = cur_log_prop + hyp_rel_log_props
        elif hyp_log_props is None:
            hyp_log_props = cur_log_prop.broadcast_to(n_hyp)

        # -- in the split step, we want hyp labels for the fit spikes
        fit_labels = fit_liks = None
        if label_fit_spikes:
            fit_liks = self.dense_log_likelihoods(
                hyp_fit_spikes,
                units=hyp_units,
                log_proportions=hyp_log_props,
                ignore_channels=ignore_channels,
            )
            vals, fit_labels = fit_liks.max(dim=0)
            fit_labels = torch.where(vals.isfinite(), fit_labels, -1)

        # -- grab eval spikes from within current units
        eval_split_name = "train" if in_bag else "val"
        split_indices = []
        eval_cur_labels = []
        for uid in current_unit_ids:
            ixs_full, ixs, split_ixs = self.random_indices(
                uid, split_name=eval_split_name
            )
            eval_cur_labels.append(ixs.new_full(ixs.shape, uid))
            # coeft = self.log_proportions[uid].exp().broadcast_to(ixs.shape)
            split_indices.append(split_ixs)
        split_indices = torch.concatenate(split_indices)
        assert len(split_indices), f"No labels on split {eval_split_name}"
        split_indices, order = split_indices.sort()
        eval_cur_labels = torch.concatenate(eval_cur_labels)[order]
        spikes = self.random_spike_data(
            split_indices=split_indices,
            split_name=eval_split_name,
            neighborhood="core",
            with_neighborhood_ids=True,
        )

        # -- evaluate eval log likelihoods
        # never ignore current non-irrelevant units
        cur_liks_full = self.get_log_likelihoods(
            spikes.indices, current_log_liks, dense=True
        )
        irr_liks = cur_liks_full[irrix]
        if reevaluate_cur_liks:
            if refit_cur_units:
                cur_units = []
                for w in cur_resps:
                    cur_units.append(
                        self.fit_unit(
                            features=hyp_fit_spikes,
                            weights=w,
                            channels=hyp_fit_channels,
                        )
                    )
                cur_liks = self.dense_log_likelihoods(
                    spikes,
                    units=cur_units,
                    log_proportions=self.log_proportions[current_unit_ids],
                    ignore_channels=ignore_channels,
                )
            elif cur_units is not None:
                cur_liks = self.dense_log_likelihoods(
                    spikes,
                    units=cur_units,
                    log_proportions=self.log_proportions[current_unit_ids],
                    ignore_channels=ignore_channels,
                )
            else:
                cur_liks = self.dense_log_likelihoods(
                    spikes, unit_ids=current_unit_ids, ignore_channels=ignore_channels
                )
            cur_liks_full = torch.concatenate((irr_liks, cur_liks), dim=0)
        else:
            cur_liks = cur_liks_full[current_unit_ids]

        cur_logliks = cur_liks_full.logsumexp(dim=0)

        # hypothetical units
        hyp_liks = self.dense_log_likelihoods(
            spikes=spikes,
            units=hyp_units,
            log_proportions=hyp_log_props,
            ignore_channels=ignore_channels,
        )
        hyp_liks_full = torch.concatenate((irr_liks, hyp_liks), dim=0)
        hyp_logliks = hyp_liks_full.logsumexp(dim=0)

        # we can only work on this subset...
        # valid = torch.logical_and(cur_logliks.isfinite(), hyp_logliks.isfinite())
        valid = torch.logical_and(
            cur_liks.isfinite().all(dim=0), hyp_logliks.isfinite()
        )
        (vix,) = valid.cpu().nonzero(as_tuple=True)
        if vix.numel() == len(spikes):
            vix = slice(None)
        cur_loglik = cur_logliks[vix]  # .mean()
        hyp_loglik = hyp_logliks[vix]  # .mean()

        # -- evaluate eval elbos
        Qcur = cur_liks_full[:, vix].softmax(dim=0)
        Qhyp = hyp_liks_full[:, vix].softmax(dim=0)
        cur_elbo = spiketorch.elbo(
            Qcur, cur_liks_full[:, vix], dim=0, reduce_mean=False
        )
        hyp_elbo = spiketorch.elbo(
            Qhyp, hyp_liks_full[:, vix], dim=0, reduce_mean=False
        )

        # -- compute final class weighted metrics
        splitting = n_cur == 1
        if not splitting:
            eval_labels = eval_cur_labels
        else:
            eval_labels = hyp_liks.argmax(0)

        # reweight by proportion
        # prop = cur_log_prop.exp() * len(self.log_proportions)
        nu = len(self.log_proportions)
        l = eval_cur_labels[vix]
        ls, ix, ct = l.unique(return_inverse=True, return_counts=True)

        w = self.log_proportions[l].exp() / ct[ix].to(self.log_proportions)
        cur_loglik = (w * cur_loglik).sum() * nu
        hyp_loglik = (w * hyp_loglik).sum() * nu
        cur_elbo = (w * cur_elbo).sum() * nu
        hyp_elbo = (w * hyp_elbo).sum() * nu

        # always hyp-cur
        cur_loglik = cur_loglik.cpu().item()
        hyp_loglik = hyp_loglik.cpu().item()
        cur_elbo = cur_elbo.cpu().item()
        hyp_elbo = hyp_elbo.cpu().item()

        if self.prior_corrected_criterion and self.prior_pseudocount:
            # in an ideal world, i would run tem here to get the new units
            # and N would be the train set size...
            _, cm, cw, _, ca = self.stack_units(
                units=cur_units, mean_only=False, with_alpha=True
            )
            if cw is not None:
                cw = cw.permute(0, 3, 1, 2).reshape(len(cw), self.ppca_rank, -1)
            cec = _elbo_prior_correction(
                self.prior_pseudocount,
                self.data.n_spikes_train,
                cm.reshape(len(cm), -1),
                cw,
                self.noise.full_inverse(),
                alpha=ca,
            )
            cec = cec.cpu().item()
            _, hm, hw, _, ha = self.stack_units(
                units=hyp_units, mean_only=False, with_alpha=True
            )
            if hw is not None:
                hw = hw.permute(0, 3, 1, 2).reshape(len(hw), self.ppca_rank, -1)
            hec = _elbo_prior_correction(
                self.prior_pseudocount,
                self.data.n_spikes_train,
                hm.reshape(len(hm), -1),
                hw,
                self.noise.full_inverse(),
                alpha=ha,
            )
            hec = hec.cpu().item()
            # rescale from main likelihood units to nu/n scale
            cur_elbo += nu * cec
            cur_loglik += nu * cec
            hyp_elbo += nu * hec
            hyp_loglik += nu * hec

        improvements = dict(loglik=hyp_loglik - cur_loglik, elbo=hyp_elbo - cur_elbo)
        hyp_criteria = dict(loglik=hyp_loglik, elbo=hyp_elbo)
        cur_criteria = dict(loglik=cur_loglik, elbo=cur_elbo)

        # -- compute overlap for the caller
        # TODO return early?
        # caller may not want to shrink the model if one of the classes
        # was poorly covered when computing the metrics
        ids, ixs, counts = eval_labels.unique(return_inverse=True, return_counts=True)
        props = ixs.new_zeros(ids.shape, dtype=torch.float)
        spiketorch.add_at_(props, ixs[vix], 1.0)
        props /= counts
        overlap = props.min()

        merged_criteria = cur_criteria if splitting else hyp_criteria
        full_criteria = hyp_criteria if splitting else cur_criteria
        res = dict(
            improvements=improvements,
            merged_criteria=merged_criteria,
            full_criteria=full_criteria,
            overlap=overlap,
            fit_labels=fit_labels,
            fit_liks=fit_liks,
            eval_labels=eval_labels,
            hyp_units=hyp_units,
        )
        return res

    def get_log_likelihoods(
        self,
        indices,
        likelihoods,
        proportions_already=True,
        unit_ids=None,
        dense=False,
    ):
        # torch's index_select is painfully slow
        # weights = torch.index_select(likelihoods, 1, features.indices)
        # here we have weights as a csc_array
        if torch.is_tensor(indices):
            indices = indices.numpy(force=True)
        liks = likelihoods[:, indices]
        if unit_ids is not None:
            liks = liks[unit_ids]

        liks = coo_to_torch(liks.tocoo(), torch.float, copy_data=True)
        liks = liks.to(self.data.device)
        if not proportions_already and self.log_proportions is not None:
            log_props = self.log_proportions
            if unit_ids is not None:
                log_props = log_props[unit_ids]
            log_props = log_props[liks.indices()[0]]
            liks.values().add_(log_props)

        if dense:
            inds = liks.indices()
            data = liks.values()
            liks = data.new_full(liks.shape, -torch.inf)
            liks[tuple(inds)] = data

        return liks

    def get_fit_weights(self, unit_id, indices, unit_ids=None, likelihoods=None):
        """Responsibilities for subset of spikes."""
        if likelihoods is None:
            return None

        weights = self.get_log_likelihoods(indices, likelihoods)
        weights = torch.sparse.softmax(weights, dim=0)
        if unit_ids is not None:
            assert unit_id is None
            weights = sum(weights[uid].to_dense() for uid in unit_ids)
        else:
            weights = weights[unit_id].to_dense()
        return weights

    def noise_mahal_chi_survival(self, split="full", neighborhood="core"):
        split_indices, spike_neighborhoods = self.data.neighborhoods(split=split)
        cn, neighborhood_info, ns = spike_neighborhoods.subset_neighborhoods(
            torch.arange(self.data.n_channels), batch_size=self.likelihood_batch_size
        )
        mahals = torch.full((len(neighborhood_info),), torch.inf)
        dfs = torch.zeros((len(neighborhood_info),), dtype=torch.long)

        for (
            neighb_id,
            neighb_chans,
            neighb_member_ixs,
            batch_start,
        ) in neighborhood_info:
            chans_valid = spike_neighborhoods.valid_mask(neighb_id)
            neighb_chans = neighb_chans[chans_valid]

            if spike_neighborhoods.has_feature_cache():
                features = spike_neighborhoods.neighborhood_features(
                    neighb_id,
                    batch_start=batch_start,
                    batch_size=self.likelihood_batch_size,
                )
                features = features.to(self.data.device)
            else:
                # full split case
                assert split_indices == slice(None)

                sp = self.data.spike_data(
                    neighb_member_ixs, with_channels=False, neighborhood="core"
                )
                features = sp.features
                features = features[..., chans_valid]

            my_mahals = self.noise_unit.log_likelihood(
                features, neighb_chans, neighborhood_id=neighb_id, inv_quad_only=True
            )
            my_min_mahal = my_mahals.min().cpu()
            mahals[neighb_id] = min(mahals[neighb_id], my_min_mahal)
            dfs[neighb_id] = np.prod(features.shape[1:])

        # what is p(mahal<=min obs)?
        chi2 = torch.distributions.chi2.Chi2(df=dfs)
        chi2_cdfs = chi2.cdf(mahals)

        # how would we renormalize our noise ll to the volume with the
        # ball cut out? that's log(1/(1-cdf)).
        log_factor = -torch.log1p(-chi2_cdfs)

        return log_factor

    # -- gizmos

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["lock"]
        del state["labels_lock"]
        del state["storage"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()
        self.labels_lock = threading.Lock()
        self.storage = threading.local()

    @staticmethod
    def normalize_key(ix):
        if torch.is_tensor(ix):
            ix = ix.numpy(force=True).item()
        elif isinstance(ix, np.ndarray):
            ix = ix.item()
        ix = int(ix)
        return str(ix)

    @property
    def rg(self):
        # thread-local rgs since they aren't safe
        if not hasattr(self.storage, "rg"):
            with self.lock:
                self.storage.rg = self._rg.spawn(1)[0]
        return self.storage.rg

    def train_extract_buffer(self):
        if not hasattr(self.storage, "_train_extract_buffer"):
            shape = self.n_spikes_fit, *self.data._train_extract_features.shape[1:]
            dtype = self.data._train_extract_features.dtype
            pin = self.data.device.type == "cuda"
            self.storage._train_extract_buffer = torch.empty(
                shape, dtype=dtype, pin_memory=pin
            )
        return self.storage._train_extract_buffer

    def _relabel(self, old_labels, new_labels=None, flat=False, split=None):
        """Re-label units

        !! This could invalidate self._units and props.

        Suggested to only call .cleanup(), this is its low-level helper.

        Arguments
        ---------
        old_labels : (n_units,)
        new_labels : optional (n_units,)
        flat : bool
            Just do self.labels[i] = new_labels[self.labels[i]]
            In other words, old_labels is arange(n_units).
            Why would that be useful? Merge step.
            But why would not flat ever be useful? Throwing away
            units -- in that case, we seach for the position of each spike's
            label in old_labels and index new_labels with that, so that
            cleanup can call relabel with unit_ids[big_enough].
        """
        split_indices = slice(None)
        if split is not None:
            split_indices = self.data.split_indices[split]

        if new_labels is None:
            new_labels = torch.arange(len(old_labels))

        original = self.labels[split_indices]

        if flat:
            kept = original >= 0
            label_indices = original[kept]
        else:
            label_indices = torch.searchsorted(old_labels, original, right=True) - 1
            kept = old_labels[label_indices] == original
            label_indices = label_indices[kept]

        unkept = torch.logical_not(kept)
        if split_indices != slice(None):
            unkept = split_indices[unkept]
            kept = split_indices[kept]

        if new_labels is not None:
            label_indices = new_labels.to(self.labels)[label_indices]

        self.labels[kept] = label_indices
        self.labels[unkept] = -1
        self._stack = None

    def stack_units(
        self,
        nu=None,
        units=None,
        ids=None,
        mean_only=True,
        use_cache=False,
        with_alpha=False,
    ):
        if units is not None and not mean_only:
            mean_only = units[0].cov_kind == "zero"
        if ids is not None:
            assert units is not None
        elif units is not None:
            ids = np.arange(len(units))
        else:
            ids, units = self.ids_and_units()
        if nu is None:
            nu = len(ids)

        if use_cache and self._stack is not None:
            assert not with_alpha
            if mean_only or self._stack[-1] is not None:
                return self._stack

        rank, nc = self.data.rank, self.data.n_channels

        means = torch.full((nu, rank, nc), torch.nan, device=self.data.device)
        alpha = None
        covs = logdets = None
        if not mean_only:
            if self.cov_kind == "ppca" and self.ppca_rank:
                covs = means.new_full((nu, rank, nc, self.ppca_rank), torch.nan)
                if with_alpha and self.laplace_ard:
                    alpha = means.new_full((nu, self.ppca_rank), torch.nan)
            logdets = means.new_full((nu,), torch.nan)

        for j, unit in enumerate(units):
            if not hasattr(unit, "mean"):
                continue
            means[j] = unit.mean
            if covs is not None:
                covs[j] = unit.W
            if logdets is not None:
                logdets[j] = unit.logdet()
            if with_alpha and unit.alpha is not None:
                alpha[j] = unit.alpha
                assert alpha[j].isfinite().all()

        if use_cache:
            self._stack = ids, means, covs, logdets
        else:
            self._stack = None

        if with_alpha:
            return ids, means, covs, logdets, alpha
        return ids, means, covs, logdets

    def schedule_annotations(self, unit_id, **annotations):
        if unit_id not in self.next_round_annotations:
            self.next_round_annotations[unit_id] = {}
        self.next_round_annotations[unit_id].update(annotations)

    def clear_scheduled_annotations(self):
        self.next_round_annotations.clear()

    # merge utils

    def merge_units(
        self,
        units=None,
        override_unit_id=None,
        hyp_fit_resps=None,
        likelihoods=None,
        log_proportions=None,
        spike_data=None,
        labels=None,
        show_progress=False,
        merge_kind=None,
        merge_criterion=None,
        debug_info=None,
    ):
        """Unit merging logic

        Returns
        -------
        new_labels : int array
            Same shape as `labels`, if supplied, or self.labels otherwise.
        new_ids : int array
            Has length `len(units)`: maps each unit to its new ID. If `units`
            was not supplied, then that would mean all of my units.
        """
        # merge full label set by default
        if labels is None:
            labels = self.labels

        # merge behavior is either a hierarchical merge or this tree-based
        # idea, depending on the value of a parameter
        if merge_criterion is None:
            merge_criterion = self.criterion
        if merge_kind is None:
            if merge_criterion == "bimodality":
                merge_kind = "hierarchical"
            else:
                merge_kind = "tree"

        # distances are needed by both methods
        unit_ids, distances = self.distances(units=units, show_progress=show_progress)
        if debug_info is not None:
            debug_info["distances"] = distances
        if distances.shape[0] == 1:
            return None, None
        pdist = distances[np.triu_indices(len(distances), k=1)]
        if not (pdist <= self.merge_distance_threshold).any():
            return None, None

        if merge_kind == "hierarchical":
            return self.hierarchical_bimodality_merge(
                distances,
                labels,
                likelihoods,
                show_progress=show_progress,
                debug_info=debug_info,
            )
        elif merge_kind == "tree":
            _ids, cosines = self.distances(units=units, kind="cosine")
            assert torch.equal(torch.asarray(unit_ids), torch.asarray(_ids))
            Z, group_ids, improvements, overlaps, brute_indicator = self.tree_merge(
                distances,
                current_log_liks=likelihoods,
                unit_ids=unit_ids,
                max_distance=self.merge_distance_threshold,
                criterion=merge_criterion,
                sym_function=self.merge_sym_function,
                cosines=cosines,
                show_progress=show_progress,
            )
            if debug_info is not None:
                debug_info["Z"] = Z
                debug_info["improvements"] = improvements
                debug_info["overlaps"] = overlaps
            group_ids = torch.asarray(group_ids)
            _, new_ids = group_ids.unique(return_inverse=True)
            new_labels = torch.asarray(labels).clone()
            (kept,) = (new_labels >= 0).nonzero(as_tuple=True)
            new_labels[kept] = new_ids[new_labels[kept]]
        else:
            assert False

        return new_labels, new_ids

    def hierarchical_bimodality_merge(
        self, distances, labels, likelihoods, show_progress=True, debug_info=None
    ):
        do_bimodality = self.merge_bimodality_threshold is not None
        if do_bimodality:
            if isinstance(likelihoods, csc_array):
                compute_mask = distances <= self.merge_distance_threshold
                bimodalities = self.bimodalities(
                    likelihoods,
                    compute_mask=compute_mask,
                    show_progress=show_progress,
                    weighted=self.merge_bimodality_weighted,
                )
            else:
                assert torch.is_tensor(likelihoods)
                assert likelihoods.layout == torch.strided
                bimodalities = bimodalities_dense(
                    likelihoods.numpy(force=True),
                    labels,
                    cut=self.merge_bimodality_cut,
                    min_overlap=self.merge_bimodality_overlap,
                    score_kind=self.merge_bimodality_score_kind,
                )

        distances = (distances, bimodalities)
        thresholds = (
            self.merge_distance_threshold,
            self.merge_bimodality_threshold,
        )
        distances = combine_distances(
            distances,
            thresholds,
            sym_function=self.merge_sym_function,
        )
        new_labels, new_ids = agglomerate(
            labels,
            distances,
            linkage_method=self.merge_linkage,
        )
        if debug_info is not None:
            debug_info["bimodalities"] = bimodalities

        return new_labels, new_ids


# -- modeling class

# our model per class k and spike n is that
#  x_n | l_n=k, mu_k, C_k, G ~ N(mu_k, J_n (C_k + G) J_n^T)
# where:
#  - x_n is the feature being clustered, living on chans N_n
#  - l_n is its label
#  - C_k is the unit (signal) covariance
#  - G is the noise covariance
#  - J = [e_{N_{n,1}}, ..., e_{N_{n,|N_n|}}] is the channel
#    neighborhood extractor matrix

# the prior on the mean and covariance is based on the noise model.
# that model is used in Normal-Wishart calculations and applied with
# a pseudocount (the N-W pseudocount parameters combined):
#  mu_k, Sigma_k ~ NW(m, k0, G, k0)
# where
#  - m is the noise mean (0?)
#  - k0 is the pseudocount
#  - G is the noise cov

# we can have different kinds of unit covariances C_k as well as
# different noise covariances G. in each case, we need to compute the
# inverse (or at least the square root and log determinant) of the sum
# of the signal and noise covariances for subsets of channels. in some
# cases that is very easy (eg both diagonal), in some cases it is
# Woodbury (signal = low rank, noise info cached). we also need
# appropriate M step formulas.

# approach to handling the likelihoods: use linear_operator by G. Pleiss
# et al. The noise object has a marginal_covariance which returns the
# best representation available. These might need to be cached somehow.
# Then the GM class gets the linear operator it needs on the channel subsets
# (which also need to be cached) of relevant spikes. and then we use
# linear_operator.inv_quad_logdet.


class GaussianUnit(torch.nn.Module):
    # store reusable buffers to avoid lots of large allocations
    # this is used during .fit() for a waveform buffer
    storage = threading.local()

    def __init__(
        self,
        rank: int,
        n_channels: int,
        noise: noise_util.EmbeddedNoise,
        mean_kind="full",
        cov_kind="zero",
        prior_type="niw",
        channels_strategy="count",
        channels_count_min=50.0,
        channels_snr_amp=1.0,
        prior_pseudocount=0,
        ppca_initial_em_iter=1,
        ppca_inner_em_iter=1,
        ppca_atol=0.05,
        ppca_rank=0,
        ppca_warm_start: bool = True,
        laplace_ard=False,
        prior_scales_mean=False,
        **annotations,
    ):
        super().__init__()
        self.rank = rank
        self.n_channels = n_channels
        self.noise = noise
        self.prior_pseudocount = prior_pseudocount
        self.mean_kind = mean_kind
        self.prior_type = prior_type
        self.channels_strategy = channels_strategy
        self.channels_count_min = channels_count_min
        self.channels_snr_amp = channels_snr_amp
        self.cov_kind = cov_kind
        self.ppca_rank = ppca_rank
        self.ppca_inner_em_iter = ppca_inner_em_iter
        self.ppca_initial_em_iter = ppca_initial_em_iter
        self.ppca_atol = ppca_atol
        self.ppca_warm_start = ppca_warm_start
        self.laplace_ard = laplace_ard
        self.prior_scales_mean = prior_scales_mean

        self.annotations = annotations

    @classmethod
    def from_parameters(
        cls,
        mean,
        basis=None,
        channels=None,
        channel_counts=None,
        channels_amp=0.25,
        alpha=None,
        **unit_args,
    ):
        M = 0 if basis is None else basis.shape[-1]
        self = cls(**unit_args)
        self.register_buffer("mean", mean)
        if basis is not None:
            self.register_buffer("W", basis)
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

        snr = mean.square().sum(dim=0).sqrt()
        if channels is not None:
            assert channel_counts is not None
            channels = torch.asarray(channels)
            channel_counts = torch.asarray(channel_counts)
            snr = snr * channel_counts.to(snr.device).sqrt()
            channels = channels[channel_counts[channels] >= self.channels_count_min]
        else:
            channels = snr > channels_amp
            (channels,) = channels.nonzero(as_tuple=True)
        self.register_buffer("channels", channels.to(mean.device))
        self.register_buffer("snr", snr.to(mean.device))
        return self

    @classmethod
    def from_features(
        cls,
        features,
        weights,
        noise,
        neighborhoods=None,
        mean_kind="full",
        cov_kind="zero",
        prior_type="niw",
        channels_strategy="count",
        ppca_rank=0,
        channels_count_min=50.0,
        channels_snr_amp=1.0,
        prior_pseudocount=10,
        ppca_inner_em_iter=1,
        ppca_atol=0.05,
        core_neighborhoods=None,
        core_neighborhood_ids=None,
        ppca_warm_start=True,
        channels=None,
        laplace_ard=False,
        prior_scales_mean=False,
        **annotations,
    ):
        self = cls(
            rank=features.features.shape[1],
            n_channels=noise.n_channels,
            noise=noise,
            mean_kind=mean_kind,
            cov_kind=cov_kind,
            prior_type=prior_type,
            prior_pseudocount=prior_pseudocount,
            channels_strategy=channels_strategy,
            channels_count_min=channels_count_min,
            channels_snr_amp=channels_snr_amp,
            ppca_rank=ppca_rank,
            ppca_inner_em_iter=ppca_inner_em_iter,
            ppca_atol=ppca_atol,
            ppca_warm_start=ppca_warm_start,
            laplace_ard=laplace_ard,
            prior_scales_mean=prior_scales_mean,
            **annotations,
        )
        self.fit(
            features,
            weights,
            neighborhoods=neighborhoods,
            core_neighborhoods=core_neighborhoods,
            core_neighborhood_ids=core_neighborhood_ids,
            achans=channels,
        )
        self = self.to(features.features.device)
        return self

    def n_params(self, channels=None, on_channels=True):
        p = channels.new_zeros(len(channels))
        ncv = torch.isin(channels, self.channels).sum(1)

        if self.mean_kind == "full":
            p += self.rank * ncv
        elif self.mean_kind == "zero":
            pass
        else:
            assert False

        if self.cov_kind == "zero":
            pass
        elif self.cov_kind == "ppca":
            p += self.ppca_rank * self.rank * ncv
        else:
            assert False

        return p

    def fit(
        self,
        features: Optional[SpikeFeatures],
        weights: Optional[torch.Tensor] = None,
        neighborhoods: Optional["SpikeNeighborhoods"] = None,
        show_progress: bool = False,
        core_neighborhood_ids: Optional[torch.Tensor] = None,
        core_neighborhoods: Optional["SpikeNeighborhoods"] = None,
        achans=None,
    ):
        if features is None or len(features) < self.channels_count_min:
            if features is None:
                logger.dartsortverbose(f"Tried to fit a unit with {features=}.")
            else:
                logger.dartsortverbose(
                    f"Unit had too few spikes ({len(features)}) for min "
                    f"count {self.channels_count_min}."
                )
            self.pick_channels(None, None)
            return
        new_zeros = features.features.new_zeros

        if weights is not None:
            (kept,) = (weights > 0).cpu().nonzero(as_tuple=True)
            features = features[kept]
            weights = weights[kept]
            assert weights.isfinite().all()

        if achans is not None:
            achans_full = achans
            achans, achan_counts = occupied_chans(
                features, self.n_channels, neighborhoods=neighborhoods
            )
            vachans = spiketorch.isin_sorted(achans, achans_full)
            achans = achans[vachans]
            achan_counts = achan_counts[vachans]
            needs_direct = True
        elif self.channels_strategy.endswith("fuzzcore"):
            achans_full, achan_counts = occupied_chans(
                features, self.n_channels, neighborhoods=neighborhoods, weights=weights
            )
            achans, _ = occupied_chans(
                features,
                neighborhood_ids=core_neighborhood_ids,
                n_channels=self.n_channels,
                neighborhoods=core_neighborhoods,
                fuzz=1,
            )
            in_full = spiketorch.isin_sorted(achans, achans_full)
            achans = achans[in_full]
            achan_counts = achan_counts[spiketorch.isin_sorted(achans_full, achans)]
            full_big_enough = achan_counts > self.channels_count_min
            achan_counts = achan_counts[full_big_enough]
            achans = achans[full_big_enough]
            needs_direct = True
        elif self.channels_strategy.endswith("core"):
            achans, achan_counts = occupied_chans(
                features,
                neighborhood_ids=core_neighborhood_ids,
                n_channels=self.n_channels,
                neighborhoods=core_neighborhoods,
            )
            needs_direct = True
        else:
            achans, achan_counts = occupied_chans(
                features, self.n_channels, neighborhoods=neighborhoods, weights=weights
            )
            vachans = achan_counts >= self.channels_count_min
            achans = achans[vachans]
            achan_counts = achan_counts[vachans]
            needs_direct = False

        je_suis = bool(achans.numel())
        if not je_suis:
            logger.dartsortverbose("Unit had no active channels.")
        do_pca = self.cov_kind == "ppca" and self.ppca_rank

        can_warm_start = False
        if je_suis and hasattr(self, "channels"):
            can_warm_start = (
                spiketorch.isin_sorted(achans.cpu(), self.channels.cpu()).all().item()
            )

        active_mean = active_W = active_alpha = None
        n_iter = self.ppca_initial_em_iter
        if can_warm_start and hasattr(self, "mean") and self.ppca_warm_start:
            active_mean = self.mean[:, achans]
            assert active_mean.isfinite().all()
            n_iter = self.ppca_inner_em_iter
        if can_warm_start and hasattr(self, "W") and self.ppca_warm_start:
            active_W = self.W[:, achans]
            assert active_W.isfinite().all()
        if can_warm_start and hasattr(self, "alpha") and self.ppca_warm_start:
            active_alpha = self.alpha
            if active_alpha is not None:
                assert active_alpha.isfinite().all()

        if je_suis:
            try:
                res = ppca_em(
                    sp=features,
                    noise=self.noise,
                    neighborhoods=neighborhoods,
                    active_channels=achans,
                    active_mean=active_mean,
                    active_W=active_W,
                    weights=weights,
                    cache_prefix="extract",
                    M=self.ppca_rank if self.cov_kind == "ppca" else 0,
                    n_iter=n_iter,
                    em_converged_atol=self.ppca_atol,
                    prior_pseudocount=self.prior_pseudocount,
                    show_progress=show_progress,
                    cache_local_direct=needs_direct,
                    laplace_ard=self.laplace_ard,
                    prior_scales_mean=self.prior_scales_mean,
                    alpha=active_alpha,
                )
            except ValueError as e:
                warnings.warn(f"ppca_em {e} {active_mean=} {active_alpha=} {active_W=} {weights=}")
                self.pick_channels(None, None)
                return

        if hasattr(self, "mean"):
            mean_full = self.mean
            mean_full.fill_(0.0)
        else:
            mean_full = new_zeros((self.noise.rank, self.noise.n_channels))

        if hasattr(self, "W"):
            W_full = self.W
            W_full.fill_(0.0)
        elif do_pca:
            W_full = new_zeros((self.noise.rank, self.noise.n_channels, self.ppca_rank))

        if je_suis:
            mu = res["mu"]
            if not mu.isfinite().all():
                raise ValueError(
                    f"Fit exploded, with {mu.shape=} {mu.isnan().any()=} "
                    f"{mu.isinf().any()=} {mu.isfinite().sum()/mu.numel()=} "
                    f"{achan_counts=}."
                )
            mean_full[:, achans] = mu
            W = res.get("W")
            if do_pca:
                assert W is not None
                assert W.isfinite().all()
                W_full[:, achans] = W

        self.register_buffer("mean", mean_full)
        if do_pca:
            self.register_buffer("W", W_full)
        nobs = res["nobs"] if je_suis else None
        self.pick_channels(achans, nobs)

        if je_suis and do_pca and self.laplace_ard:
            self.register_buffer("alpha", res["alpha"])
        else:
            self.alpha = None

    def pick_channels(self, active_chans, nobs=None):
        if self.channels_strategy.startswith("all"):
            self.register_buffer("channels", torch.arange(self.n_channels))
            return

        if nobs is None or not active_chans.numel():
            self.snr = torch.zeros(self.n_channels)
            self.register_buffer("channels", torch.arange(0))
            return

        amp = torch.linalg.vector_norm(self.mean[:, active_chans], dim=0)
        snr = amp * nobs.sqrt()
        full_snr = self.mean.new_zeros(self.mean.shape[1])
        full_snr[active_chans] = snr
        self.snr = full_snr.cpu()

        if self.channels_strategy == "active":
            self.register_buffer("channels", active_chans)
            return

        if self.channels_strategy.startswith("snr"):
            snr_min = np.sqrt(self.channels_count_min) * self.channels_snr_amp
            strong = snr >= snr_min
            self.register_buffer("channels", active_chans[strong.cpu()])
            return
        if self.channels_strategy.startswith("count"):
            strong = nobs >= self.channels_count_min
            self.register_buffer("channels", active_chans[strong.cpu()])
            return

        assert False

    def com(self, geom):
        w = self.snr / self.snr.sum()
        return (w.unsqueeze(1) * geom).sum(0)

    def marginal_covariance(
        self, channels=None, cache_key=None, device=None, signal_only=False
    ):
        channels_ = channels
        if channels is None:
            channels_ = torch.arange(self.n_channels)
        if signal_only:
            sz = channels_.numel() * self.noise.rank
            ncov = operators.ZeroLinearOperator(
                sz, sz, dtype=self.noise.global_std.dtype, device=device
            )
        else:
            ncov = self.noise.marginal_covariance(
                channels, cache_key=cache_key, device=device
            )
        zero_signal = (
            self.cov_kind == "zero"
            or (self.cov_kind == "ppca" and not self.ppca_rank)
            or not hasattr(self, "W")
        )
        if zero_signal:
            return ncov
        if self.cov_kind == "ppca" and self.ppca_rank:
            root = self.W[:, channels_].reshape(-1, self.ppca_rank)
            root = operators.LowRankRootLinearOperator(root)
            if signal_only:
                return root
            # this calls .add_low_rank, and it's genuinely bugged.
            # the log liks that come out look wrong. can't say why.
            # return ncov + root
            return more_operators.LowRankRootSumLinearOperator(root, ncov)
        assert False

    def logdet(self, channels=None):
        return self.marginal_covariance(channels).logdet()

    def log_likelihood(
        self, features, channels, neighborhood_id=None, inv_quad_only=False
    ) -> torch.Tensor:
        """Log likelihood for spike features living on the same channels."""
        if not len(features):
            return features.new_zeros((0,))
        mean = self.noise.mean_full[:, channels]
        if self.mean_kind == "full" and hasattr(self, "mean"):
            mean = mean + self.mean[:, channels]
        features = features - mean

        cov = self.marginal_covariance(
            channels, cache_key=neighborhood_id, device=features.device
        )
        y = features.view(len(features), -1)
        ll = spiketorch.ll_via_inv_quad(cov, y, inv_quad_only=inv_quad_only)
        return ll

    def divergence(
        self,
        other_means,
        other_covs=None,
        other_logdets=None,
        kind="noise_metric",
    ):
        """Compute my distance to other units

        To make use of batch dimensions, this asks for other units' means and
        dense covariance matrices and also possibly log covariance determinants.
        """
        if kind == "noise_metric":
            return self.noise_metric_divergence(other_means)
        if kind in ("kl", "symkl"):
            kl1 = self.kl_divergence(other_means, other_covs, other_logdets)
            if kind == "kl":
                return kl1
        if kind in ("reverse_kl", "symkl"):
            kl2 = self.reverse_kl_divergence(other_means, other_covs, other_logdets)
            if kind == "reverse_kl":
                return kl2
        if kind == "symkl":
            return 0.5 * (kl1 + kl2)
        raise ValueError(f"Unknown divergence {kind=}.")

    def noise_metric_divergence(self, other_means):
        dmu = other_means
        if self.mean_kind != "zero":
            dmu = dmu - self.mean
        dmu = dmu.view(len(other_means), -1)
        noise_cov = self.noise.marginal_covariance(device=dmu.device)
        return noise_cov.inv_quad(dmu.T, reduce_inv_quad=False)

    def kl_divergence(
        self,
        other_means=None,
        other_covs=None,
        other_logdets=None,
        other=None,
        return_extra=False,
    ):
        """DKL(self || others)
        = 0.5 * {
            tr(So^-1 Ss)
            + (mus - muo)^T So^-1 (mus - muo)
            - k
            + log(|So| / |Ss|)
          }
        """
        is_ppca = False
        if other_covs is not None:
            is_ppca = other_covs.shape[-2] < other_covs.shape[-1]

        if other is not None:
            is_ppca = other.cov_kind == "ppca" and other.ppca_rank > 0
            other_means = other.mean.unsqueeze(0)
            other_covs = None
            if is_ppca:
                other_covs = other.W.unsqueeze(0)
            other_logdets = torch.atleast_1d(other.logdet())

        n = other_means.shape[0]
        dmu = other_means
        if self.mean_kind != "zero":
            dmu = dmu - self.mean
        dmu = dmu.view(n, -1)
        k = dmu.shape[1]

        # get all the other covariance operators
        ncov = self.noise.marginal_covariance()
        ncov_batch = ncov._expand_batch((n,))
        if is_ppca:
            oW = other_covs.reshape(n, k, self.ppca_rank)
            root = operators.LowRankRootLinearOperator(oW)
            other_covs = more_operators.LowRankRootSumLinearOperator(root, ncov_batch)
        else:
            other_covs = ncov_batch

        # get trace term
        tr = float(k)
        if is_ppca:
            my_dense_cov = self.marginal_covariance().to_dense()
            solve = other_covs.solve(my_dense_cov)
            assert solve.shape == (n, k, k)
            tr = solve.diagonal(dim1=1, dim2=2).sum(dim=1)

        # get inv quad term
        # inv_quad = other_covs.inv_quad(dmu.unsqueeze(-1), reduce_inv_quad=False)
        # inv_quad = inv_quad[:, 0]
        inv_quad = dmu.new_empty(len(dmu))
        for bs in range(0, len(dmu), 32):
            dmub = dmu[bs : bs + 32]
            cb = other_covs[bs : bs + 32]
            iq = cb.solve(dmub.unsqueeze(2))[:, :, 0]
            inv_quad[bs : bs + 32] = (iq * dmub).sum(1)
        assert inv_quad.shape == dmu.shape[:1]

        # get logdet term
        ld = 0.0
        if is_ppca:
            ld = other_logdets - self.logdet()
        kl = 0.5 * (inv_quad + ((tr - k) + ld))
        if return_extra:
            return dict(kl=kl, inv_quad=inv_quad, ld=ld, tr=tr, k=k)
        return kl

    def reverse_kl_divergence(
        self, other_means, other_covs, other_logdets, batch_size=32
    ):
        """DKL(others || self)
        = 0.5 * {
            tr(Ss^-1 So)
            + (mus - muo)^T Ss^-1 (mus - muo)
            - k
            + log(|Ss| / |So|)
          }
        """
        n = other_means.shape[0]
        dmu = other_means
        if self.mean_kind != "zero":
            dmu = dmu - self.mean
        dmu = dmu.view(n, -1)

        # compute the inverse quad and self log det terms
        my_cov = self.marginal_covariance()
        k = my_cov.shape[0]
        assert dmu.shape[1] == k
        inv_quad, self_logdet = my_cov.inv_quad_logdet(
            dmu.T, logdet=True, reduce_inv_quad=False
        )

        # other covs
        tr = k
        ld = 0.0
        if self.cov_kind == "ppca" and self.ppca_rank:
            oW = other_covs.reshape(n, k, self.ppca_rank)
            tr = other_covs.new_empty((n,))
            for bs in range(0, n, batch_size):
                be = min(n, bs + batch_size)
                res = my_cov.solve(oW[bs:be])
                res = res @ oW[bs:be].mT
                tr[bs:be] = res.diagonal(dim1=-2, dim2=-1).sum(dim=1)
            ncov = self.noise.full_dense_cov()
            tr += torch.trace(my_cov.solve(ncov))
            ld = self_logdet - other_logdets
        return 0.5 * (inv_quad + (tr - k + ld))


# -- utilities


def all_partitions(ids):
    ids = np.asarray(ids).ravel()
    group_ids = np.zeros(len(ids), dtype=np.int64)
    for m in range(1, len(ids) + 1):
        for partition in multiset_partitions(len(ids), m=m):
            ids_partition = []
            for j, p in enumerate(partition):
                group_ids[p] = j
                ids_partition.append(tuple(ids[p].tolist()))
            yield group_ids.copy(), partition, ids_partition


log2pi = torch.log(torch.tensor(2.0 * torch.pi))
tqdm_kw = dict(smoothing=0, mininterval=0.2)


def get_average_parameter_counts(
    full_units,
    merged_unit,
    spikes_core,
    core_neighborhoods,
    weights=None,
    use_proportions=True,
    reduce=True,
):
    # parameter counting... since we use marginal likelihoods, I'm restricting
    # the parameter counts to just the marginal set considered for each spike.
    # then, aic and bic formulas are changed slightly below to match.
    nids = spikes_core.neighborhood_ids
    unique_nids, inverse = torch.unique(nids, return_inverse=True)
    unique_chans = core_neighborhoods.neighborhoods[unique_nids]
    unique_k_merged = merged_unit.n_params(unique_chans)
    unique_k_full = [u.n_params(unique_chans) for u in full_units]
    unique_k_full = torch.stack(unique_k_full, dim=1).sum(1)
    k_merged = unique_k_merged[inverse].to(torch.float)
    k_full = unique_k_full[inverse].to(torch.float)

    if reduce:
        if weights is None:
            k_merged = k_merged.mean()
            k_full = k_full.mean()
        else:
            k_merged = torch.linalg.vecdot(weights, k_merged)
            k_full = torch.linalg.vecdot(weights, k_full)

    # for aic: k is avg
    if use_proportions:
        k_full += len(full_units) - 1

    return k_full, k_merged


def class_sum(classes, inverse_inds, x, weights=None):
    wsum = x.new_zeros(len(classes))
    x = x * weights if weights is not None else x
    spiketorch.add_at_(wsum, inverse_inds.to(x.device), x)
    return wsum


def to_full_probe(features, weights, n_channels, storage):
    """
    Arguments
    ---------
    features : SpikeFeatures
    weights : tensor
    n_channels : int
        Total channel count
    storage : optional bunch / threading.local

    Returns
    -------
    features_full : tensor
        Features on the full channel count
    weights_full : tensor
        Same, accounting for missing observations
    count_data : tensor
        (n_channels,) sum of weights
    weights_normalized : tensor
        weights divided by their sum for each feature
    """
    n, r, c = features.features.shape
    features_full = get_nans(
        features.features, storage, "features_full", (n, r, n_channels + 1)
    )
    targ_inds = features.channels.unsqueeze(1).broadcast_to(features.features.shape)
    targ_inds = targ_inds.to(features_full.device)
    features_full.scatter_(2, targ_inds, features.features)
    features_full = features_full[:, :, :-1]
    weights_full = features_full[:, :1, :].isfinite().to(features_full)
    if weights is not None:
        weights_full = weights_full.mul_(weights[:, None, None])
    features_full = features_full.nan_to_num_()
    count_data = weights_full.sum(0)
    weights_normalized = weights_full / count_data
    weights_normalized = weights_normalized.nan_to_num_()
    return features_full, weights_full, count_data, weights_normalized


def get_nans(target, storage, name, shape):
    if storage is None:
        return target.new_full(shape, torch.nan)

    buffer = getattr(storage, name, None)
    if buffer is None:
        buffer = target.new_full(shape, torch.nan)
        setattr(storage, name, buffer)
    else:
        if any(bs < ts for bs, ts in zip(buffer.shape, shape)):
            buffer = target.new_full(shape, torch.nan)
            setattr(storage, name, buffer)
        region = tuple(slice(0, ts) for ts in shape)
        buffer = buffer[region]
        buffer.fill_(torch.nan)
    if buffer.device != target.device:
        buffer = buffer.to(target.device)
        setattr(storage, name, buffer)

    return buffer


def loglik_reassign(
    log_liks,
    has_noise_unit=False,
    proportions=None,
    log_proportions=None,
    hard_noise=False,
):
    nz_lines, log_liks_csc, assignments, spike_logliks = sparse_reassign(
        log_liks,
        proportions=proportions,
        log_proportions=log_proportions,
        hard_noise=hard_noise,
    )
    n_units = log_liks.shape[0] - has_noise_unit
    if has_noise_unit:
        assignments[assignments >= n_units] = -1
    return nz_lines, assignments, spike_logliks, log_liks_csc


def logmeanexp(x_csr):
    """Log of mean of exp in x_csr's rows (mean over columns)

    Sparse zeros are treated as negative infinities.
    """
    log_mean_exp = np.zeros(x_csr.shape[0], dtype=x_csr.dtype)
    log_N = np.log(x_csr.shape[1]).astype(x_csr.dtype)
    for j in range(x_csr.shape[0]):
        row = x_csr[[j]]
        # missing vals in the row are -inf, exps are 0s, so ignore in sum
        # dividing by N is subtracting log N
        log_mean_exp[j] = logsumexp(row.data) - log_N
    return log_mean_exp


def bimodalities_dense(
    log_liks,
    labels,
    cut=0.0,
    weighted=True,
    min_overlap=0.95,
    score_kind="tv",
):
    """Bimodality scores from dense data

    Given dense arrays of log likelihoods (with -infs) and labels, return a matrix
    of bimodality scores.
    """
    if cut == "auto":
        cut = None
    n_units = len(log_liks)
    bimodalities = np.zeros((n_units, n_units), dtype=np.float32)
    for i in range(n_units):
        for j in range(i + 1, n_units):
            ij = np.array([i, j])
            in_pair = np.flatnonzero(np.isin(labels, ij))
            if not in_pair.size:
                bimodalities[j, i] = bimodalities[i, j] = np.inf
                continue
            pair_log_liks = log_liks[ij][:, in_pair]
            bimodalities[j, i] = bimodalities[i, j] = qda(
                labels[in_pair] == j,
                pair_log_liks[0],
                pair_log_liks[1],
                cut=cut,
                weighted=weighted,
                min_overlap=min_overlap,
                score_kind=score_kind,
            )
    return bimodalities


def qda(
    in_b,
    log_liks_a=None,
    log_liks_b=None,
    diff=None,
    cut=None,
    weighted=True,
    min_overlap=0.80,
    min_count=10,
    score_kind="tv",
    debug_info=None,
):
    # "in b not a"-ness
    if diff is None:
        diff = log_liks_b - log_liks_a
    keep = np.isfinite(diff)
    keep_prop = keep.mean()
    if debug_info is not None:
        debug_info["keep_prop"] = keep_prop
    if keep_prop < min_overlap or keep.sum() < min_count:
        return np.inf
    in_b = in_b[keep]
    diff = diff[keep]
    if in_b.all() or not in_b.any():
        return np.inf

    if weighted:
        b_prop = in_b.mean()
        a_prop = 1.0 - b_prop
        diff, keep_keep, inv = np.unique(diff, return_index=True, return_inverse=True)
        keep = keep[keep_keep]
        sample_weights = np.zeros(diff.shape)
        np.add.at(sample_weights, inv, np.where(in_b, a_prop / 0.5, b_prop / 0.5))
        assert np.all(sample_weights > 0)
    else:
        diff, keep_keep, inv = np.unique(diff, return_index=True, return_inverse=True)
        sample_weights = np.zeros(diff.shape)
        np.add.at(sample_weights, inv, 1.0)
        assert np.all(sample_weights > 0)

    return smoothed_dipscore_at(
        cut,
        diff.astype(np.float64),
        sample_weights=sample_weights,
        dipscore_only=True,
        score_kind=score_kind,
        debug_info=debug_info,
    )


def getdt(times_i, times_j):
    ni = times_i.numel()
    iij = torch.searchsorted(times_i, times_j)
    dji = torch.minimum(
        times_j - times_i[iij.clip(0, ni - 1)],
        times_i[(1 + iij).clip(0, ni - 1)] - times_j,
    )
    return dji


def shrinkfit(x, max_size, rg):
    """Randomly subsample to fit x in max_size."""
    n = len(x)
    if max_size is None or n <= max_size:
        return x, slice(None)

    choices = rg.choice(n, size=max_size, replace=False)
    choices.sort()
    if torch.is_tensor(x):
        choices = torch.from_numpy(choices)
        return x[choices.to(x.device)], choices
    else:
        return x[choices], choices


def get_diff_sparse(sparse_arr, i, j, cols, return_extra=False):
    xs = sparse_arr[:, cols]
    xi = xs[[i]].tocoo()
    xj = xs[[j]].tocoo()
    indsi = xi.coords[1]
    indsj = xj.coords[1]
    xi = xi.data
    xj = xj.data

    ikeep = np.isin(indsi, indsj)
    jkeep = np.isin(indsj, indsi)

    diff = np.full(cols.shape, np.nan)
    xj = xj[jkeep]
    xi = xi[ikeep]
    diff[indsi[ikeep]] = xi - xj

    if return_extra:
        return diff, dict(xi=xi, xj=xj, keep_inds=indsi[ikeep])

    return diff


def noise_whiten(
    sp, noise, neighborhoods, mean_full=None, with_whitened_means=True, in_place=False
):
    """
    sp : SpikeFeatures
        Needs neighborhood_ids.
    noise : EmbeddedNoise
    neighborhoods : SpikeNeighborhoods
    mean_full : optional tensor
    """
    assert sp.neighborhood_ids is not None

    # centering
    z = sp.features if in_place else sp.features.clone()
    if mean_full is not None:
        mean_full = mean_full.view(noise.rank, noise.n_channels)
        mean_full = F.pad(mean_full, (0, 1, 0, 0))
        z -= mean_full[:, sp.channels].permute(1, 0, 2)
    z = z.nan_to_num_()

    # whitening
    nbids, nbinv = torch.unique(sp.neighborhood_ids, return_inverse=True)
    if with_whitened_means:
        nu = z.new_zeros((nbids.numel(), *z.shape[1:]))
    for j, nbid in enumerate(nbids):
        nbchans = neighborhoods.neighborhoods[nbid]
        nbvalid = neighborhoods.valid_mask(nbid)
        nbchans = nbchans[nbvalid]
        innb = sp.neighborhood_ids == nbid
        nbcov = noise.marginal_covariance(
            channels=nbchans, cache_key=nbid, device=z.device
        )
        nbz = z[innb][:, :, nbvalid]
        nbz = nbcov.sqrt_inv_matmul(nbz.view(innb.sum(), -1).T).T
        mask = torch.logical_and(innb[:, None], nbvalid[None, :])
        mask = mask.unsqueeze(1).broadcast_to(z.shape)
        z[mask] = nbz.reshape(-1)
        if with_whitened_means:
            wm = nbcov.sqrt_inv_matmul(mean_full[:, nbchans].reshape(-1))
            nu[j, :, nbvalid] = wm.reshape(-1, nbchans.numel())
    nu = nu[nbinv]

    spw = replace(sp, features=z) if in_place else sp
    if with_whitened_means:
        return spw, nu
    return spw


def cos_connectivity(cosines):
    if cosines.shape == (2, 2):
        return cosines[1, 0]
    conn = 1.0 - cosines
    if torch.is_tensor(conn):
        conn = conn.numpy(force=True)
    conn = conn[np.triu_indices(len(conn), k=1)]
    conn = linkage(conn, method="single")
    assert len(conn) == len(cosines) - 1
    conn = conn[-1, 2]
    conn = 1.0 - conn
    return conn


def quick_indices(rg, unit_ids, labels, split_indices=None, max_sizes=4096):
    """It's slow to do lots of nonzero(labels==j).

    This goes all in one by looking for the first yea many spikes in a random order.
    """
    labels_in_split = labels if split_indices is None else labels[split_indices]
    random_order = rg.permutation(len(labels_in_split))
    reordered_labels = labels_in_split[random_order]

    counts_so_far = np.zeros(unit_ids.max() + 1, dtype=np.int32)
    counts_so_far[unit_ids] = 0
    n_active = unit_ids.size
    reordered_indices = np.full(
        (unit_ids.max() + 1, np.max(max_sizes)), labels.size + 1, dtype=labels.dtype
    )

    full_max_sizes = np.zeros(unit_ids.max() + 1, dtype=np.int32)
    full_max_sizes[unit_ids] = max_sizes

    _quick_indices(
        np.int32(n_active),
        counts_so_far,
        reordered_labels,
        reordered_indices,
        full_max_sizes,
    )

    orig_indices = {}
    in_split_indices = {}
    for u in unit_ids:
        myixs = reordered_indices[u]
        myixs = myixs[myixs < labels.size + 1]
        mysplitixs = random_order[myixs]
        mysplitixs.sort()
        myorigixs = mysplitixs
        if split_indices is not None:
            myorigixs = split_indices[myorigixs]
        orig_indices[u] = myorigixs
        in_split_indices[u] = mysplitixs

    return orig_indices, in_split_indices


sig = [
    "void(i4, i4[::1], i8[::1], i8[:, ::1], i4[::1])",
    "void(i4, i4[::1], i4[::1], i4[:, ::1], i4[::1])",  # windows runner
]


@numba.njit(sig, error_model="numpy", nogil=True)
def _quick_indices(n_active, counts_so_far, reordered_labels, indices, max_sizes):
    for i in range(len(reordered_labels)):
        label = reordered_labels[i]

        my_count = counts_so_far[label]
        max_size = max_sizes[label]

        if my_count < max_size:
            indices[label, my_count] = i

        my_count += 1
        counts_so_far[label] = my_count

        if my_count == max_size:
            n_active -= 1
            if n_active == 0:
                break


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    import sys

    log = file if hasattr(file, "write") else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback
