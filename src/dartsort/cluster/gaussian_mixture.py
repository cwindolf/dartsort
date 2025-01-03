import logging
import threading
from dataclasses import replace
from typing import Literal, Optional

import numba
import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from linear_operator import operators
from scipy.cluster.hierarchy import linkage
from scipy.sparse import coo_array, csc_array
from scipy.special import logsumexp
from tqdm.auto import tqdm, trange

from ..util import more_operators, noise_util, spiketorch
from .cluster_util import agglomerate, combine_distances, leafsets
from .kmeans import kmeans
from .modes import smoothed_dipscore_at
from .ppcalib import ppca_em
from .stable_features import SpikeFeatures, StableSpikeDataset, occupied_chans

logger = logging.getLogger(__name__)

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
        likelihood_batch_size: int = 2**16,
        channels_strategy: Literal["all", "snr", "count"] = "count",
        channels_count_min: float = 25.0,
        channels_snr_amp: float = 1.0,
        with_noise_unit: bool = True,
        prior_pseudocount: float = 0.0,
        ppca_rank: int = 0,
        ppca_inner_em_iter: int = 25,
        ppca_atol: float = 0.05,
        n_threads: int = 4,
        min_count: int = 50,
        n_em_iters: int = 25,
        kmeans_k: int = 5,
        kmeans_n_iter: int = 100,
        kmeans_drop_prop: float = 0.025,
        kmeans_with_proportions: bool = False,
        kmeans_kmeanspp_initial: str = "random",
        split_em_iter: int = 0,
        split_whiten: bool = True,
        ppca_in_split: bool = False,
        distance_metric: Literal["noise_metric", "kl", "reverse_kl", "symkl"] = "symkl",
        distance_normalization_kind: Literal["none", "noise", "channels"] = "noise",
        criterion_normalization_kind: Literal["none", "noise", "channels"] = "none",
        merge_linkage: str = "single",
        merge_distance_threshold: float = 1.0,
        merge_bimodality_threshold: float = 0.1,
        merge_criterion_threshold: float | None = 1.0,
        merge_criterion: Literal[
            "heldout_loglik", "heldout_ccl", "loglik", "ccl", "aic", "bic", "icl"
        ] = "heldout_ccl",
        split_bimodality_threshold: float = 0.1,
        merge_bimodality_cut: float = 0.0,
        merge_bimodality_overlap: float = 0.80,
        merge_bimodality_weighted: bool = True,
        merge_bimodality_score_kind: str = "tv",
        merge_bimodality_masked: bool = False,
        merge_sym_function: np.ufunc = np.minimum,
        em_converged_prop: float = 0.02,
        em_converged_churn: float = 0.01,
        em_converged_atol: float = 1e-2,
        random_seed: int = 0,
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
        self.merge_criterion = merge_criterion
        self.merge_criterion_threshold = merge_criterion_threshold
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
        self.split_em_iter = split_em_iter
        self.split_whiten = split_whiten
        self.use_proportions = use_proportions
        self.proportions_sample_size = proportions_sample_size

        # store labels on cpu since we're always nonzeroing / writing np data
        assert self.data.original_sorting.labels is not None
        self.labels = self.data.original_sorting.labels[data.kept_indices]
        self.labels = torch.from_numpy(self.labels)

        # this is populated by self.m_step()
        self._units = torch.nn.ModuleDict()
        self.log_proportions = None

        # store arguments to the unit constructor in a dict
        self.ppca_rank = ppca_rank
        self.unit_args = dict(
            noise=noise,
            mean_kind=mean_kind,
            cov_kind=cov_kind,
            channels_strategy=channels_strategy,
            channels_count_min=channels_count_min,
            channels_snr_amp=channels_snr_amp,
            prior_pseudocount=prior_pseudocount,
            ppca_rank=ppca_rank,
            ppca_inner_em_iter=ppca_inner_em_iter,
            ppca_atol=ppca_atol,
        )
        if ppca_in_split:
            self.split_unit_args = self.unit_args
        else:
            self.split_unit_args = self.unit_args | dict(cov_kind="zero", ppca_rank=0)

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

    def keys(self):
        return self.unit_ids()

    def values(self):
        return self.ids_and_units()[1]

    def items(self):
        for k, v in zip(self.ids_and_units()):
            yield k, v

    def ids_and_units(self):
        uids = self.unit_ids()
        units = [self[u] for u in uids]
        return uids, units

    def n_units(self):
        uids = self.unit_ids()
        nu_u = 0
        if len(uids):
            nu_u = max(uids) + 1
        lids = self.label_ids()
        nu_l = 0
        if lids.numel():
            nu_l = lids.max() + 1
        return max(nu_u, nu_l)

    def label_ids(self, with_counts=False, split=None):
        labels = self.labels
        if split is not None:
            labels = self.labels[self.data.split_indices[split]]
        uids = torch.unique(labels, return_counts=with_counts)
        if with_counts:
            uids, counts = uids
            counts = counts[uids >= 0]
            uids = uids[uids >= 0]
            return uids, counts
        return uids[uids >= 0]

    def ids(self):
        return torch.arange(self.n_units())

    def n_labels(self):
        label_ids = self.label_ids()
        nu = label_ids.max() + 1
        return nu

    def missing_ids(self):
        mids = [lid for lid in self.label_ids() if lid not in self]
        return torch.tensor(mids)

    # -- headliners

    def to_sorting(self):
        labels = np.full_like(self.data.original_sorting.labels, -1)
        labels[self.data.kept_indices] = self.labels.numpy(force=True)
        return replace(self.data.original_sorting, labels=labels)

    def em(self, n_iter=None, show_progress=True, final_e_step=True):
        n_iter = self.n_em_iters if n_iter is None else n_iter
        if show_progress:
            its = trange(n_iter, desc="EM", **tqdm_kw)
            step_progress = max(0, int(show_progress) - 1)
        else:
            its = range(n_iter)

        # if we have no units, we can't E step.
        missing_ids = self.missing_ids()
        if len(missing_ids):
            self.m_step(show_progress=step_progress, fit_ids=missing_ids)
            self.cleanup(min_count=1)

        convergence_props = {}
        log_liks = None
        for _ in its:
            # for convergence testing...
            log_liks, convergence_props = self.cleanup(
                log_liks, min_count=1, clean_props=convergence_props
            )

            recompute_mask = None
            if "adif" in convergence_props:
                recompute_mask = convergence_props["adif"] > 0
                # recompute_mask = torch.ones(self.n_units(), dtype=bool)

            unit_churn, reas_count, log_liks, spike_logliks = self.e_step(
                show_progress=step_progress,
                recompute_mask=recompute_mask,
                prev_log_liks=log_liks,
            )
            convergence_props["unit_churn"] = unit_churn
            log_liks, convergence_props = self.cleanup(
                log_liks, clean_props=convergence_props
            )
            meanlogpx = spike_logliks.mean()

            # M step: fit units based on responsibilities
            to_fit = convergence_props["unit_churn"] >= self.em_converged_churn
            mres = self.m_step(
                log_liks, show_progress=step_progress, to_fit=to_fit, compare=True
            )
            convergence_props["adif"] = mres["adif"]

            # extra info for description
            max_adif = mres["max_adif"]
            if show_progress:
                opct = (self.labels < 0).sum() / self.data.n_spikes
                opct = f"{100 * opct:.1f}"
                nu = self.n_units().item()
                reas_prop = reas_count / self.data.n_spikes
                rpct = f"{100 * reas_prop:.1f}"
                adif = f"{max_adif:.2f}"
                msg = (
                    f"EM[K={nu},Ka={to_fit.sum()};{opct}%fp,"
                    f"{rpct}%reas,dmu={adif};logpx/n={meanlogpx:.1f}]"
                )
                its.set_description(msg)

            if reas_prop < self.em_converged_prop:
                logger.info(f"Labels converged with {reas_prop=}")
                break
            if max_adif is not None and max_adif < self.em_converged_atol:
                logger.info(f"Parameters converged with {max_adif=}")
                break

        if not final_e_step:
            return

        # final e step for caller
        unit_churn, reas_count, log_liks, spike_logliks = self.e_step(
            show_progress=step_progress
        )
        log_liks, _ = self.cleanup(log_liks)
        return log_liks

    def e_step(self, show_progress=False, prev_log_liks=None, recompute_mask=None):
        # E step: get responsibilities and update hard assignments
        log_liks = self.log_likelihoods(
            show_progress=show_progress,
            previous_logliks=prev_log_liks,
            recompute_mask=recompute_mask,
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
    ):
        """Beware that this flattens the labels."""
        warm_start = not self.empty()
        unit_ids, spike_counts = self.label_ids(with_counts=True, split=split)
        if to_fit is not None:
            fit_ids = unit_ids[to_fit[unit_ids]]
        total = fit_ids is None
        if total:
            fit_ids = unit_ids
        if warm_start:
            _, prev_means, *_ = self.stack_units(mean_only=True)

        if self.use_proportions and likelihoods is not None:
            self.update_proportions(likelihoods)
        if self.log_proportions is not None:
            assert len(self.log_proportions) > unit_ids.max() + self.with_noise_unit

        fit_indices = quick_indices(
            self.rg,
            unit_ids.numpy(),
            self.labels.numpy(),
            split_indices=None if split is None else self.data.split_indices[split],
            max_sizes=spike_counts.clamp(max=self.n_spikes_fit).numpy(force=True),
        )

        pool = Parallel(self.n_threads, backend="threading", return_as="generator")
        results = pool(
            delayed(self.fit_unit)(
                j,
                likelihoods=likelihoods,
                warm_start=warm_start,
                indices=fit_indices[j.item()],
                **self.next_round_annotations.get(j, {}),
            )
            for j in fit_ids
        )
        if show_progress:
            results = tqdm(
                results, desc="M step", unit="unit", total=len(fit_ids), **tqdm_kw
            )

        self.clear_scheduled_annotations()
        if total:
            self.clear_units()
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
    ):
        """Noise unit last so that rows correspond to unit ids without 1 offset"""
        if unit_ids is None:
            unit_ids = self.unit_ids()

        # how many units does each core neighborhood overlap with?
        n_cores = self.data.core_neighborhoods.n_neighborhoods
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
                    self.data.core_neighborhoods.subset_neighborhoods(
                        unit.channels,
                        add_to_overlaps=core_overlaps,
                        batch_size=self.likelihood_batch_size,
                    )
                )
                unit.annotations["covered_neighbs"] = covered_neighbs
                unit_neighb_info.append((j, neighbs, ns_unit))
            else:
                assert previous_logliks is not None
                row = previous_logliks[[j]].tocoo(copy=True)
                six = row.coords[1]
                ns_unit = row.nnz
                if "covered_neighbs" in unit.annotations:
                    covered_neighbs = unit.annotations["covered_neighbs"]
                else:
                    covered_neighbs = self.data.core_neighborhoods.neighborhood_ids[six]
                    covered_neighbs = torch.unique(covered_neighbs)
                unit_neighb_info.append((j, six, row.data, ns_unit))
            core_overlaps[covered_neighbs] += 1
            nnz += ns_unit

        # how many units does each spike overlap with? needed to write csc
        spike_overlaps = core_overlaps[self.data.core_neighborhoods.neighborhood_ids]
        spike_overlaps = spike_overlaps.numpy(force=True)

        # add in space for the noise unit
        if with_noise_unit:
            nnz = nnz + self.data.n_spikes

        @delayed
        def job(ninfo):
            if len(ninfo) == 4:
                j, coo, data, ns = ninfo
                return j, coo, data
            else:
                assert len(ninfo) == 3
                j, neighbs, ns = ninfo
                ix, ll = self.unit_log_likelihoods(
                    unit_id=j, neighbs=neighbs, ns=ns, return_sorted=False
                )
                return j, ix, ll

        # get the big nnz-length csc buffers. these can be huge so we cache them.
        csc_indices, csc_data = get_csc_storage(nnz, self.storage, use_storage)
        # csc compressed indptr. spikes are columns.
        indptr = np.concatenate(([0], np.cumsum(spike_overlaps, dtype=int)))
        del spike_overlaps
        # each spike starts at writing at its indptr. as we gather more units for each
        # spike, we increment the spike's "write head". idea is to directly make csc
        write_offsets = indptr[:-1].copy()
        pool = Parallel(self.n_threads, backend="threading", return_as="generator")
        results = pool(job(ninfo) for ninfo in unit_neighb_info)
        if show_progress:
            results = tqdm(
                results,
                total=len(unit_neighb_info),
                desc="Likelihoods",
                unit="unit",
                **tqdm_kw,
            )
        for j, inds, liks in results:
            if inds is None:
                continue
            if torch.is_tensor(inds):
                inds = inds.numpy(force=True)
                liks = liks.numpy(force=True)
            csc_insert(j, write_offsets, inds, csc_indices, csc_data, liks)

        if with_noise_unit:
            inds, liks = self.noise_log_likelihoods()
            data_ixs = write_offsets[inds]
            # assert np.array_equal(data_ixs, ccol_indices[1:] - 1)  # just fyi
            csc_indices[data_ixs] = j + 1
            csc_data[data_ixs] = liks.numpy(force=True)

        shape = (self.n_units() + with_noise_unit, self.data.n_spikes)
        log_liks = csc_array((csc_data, csc_indices, indptr), shape=shape)
        log_liks.has_canonical_format = True

        return log_liks

    def update_proportions(self, log_liks):
        if not self.use_proportions:
            return

        # have to jump through some hoops because torch sparse tensors
        # don't implement .mean() yet??
        if log_liks.shape[1] > self.proportions_sample_size:
            sample = self.rg.choice(
                log_liks.shape[1], size=self.proportions_sample_size, replace=False
            )
            sample.sort()
            log_liks = log_liks[:, sample]
        log_liks = log_liks.tocoo()
        log_liks = coo_to_torch(log_liks, torch.float, copy_data=True)

        # log proportions are added to the likelihoods
        if self.log_proportions is not None:
            log_props_vec = self.log_proportions.cpu()[log_liks.indices()[0]]
            log_liks.values().add_(log_props_vec)

        # softmax over units, logged
        log_resps = torch.sparse.log_softmax(log_liks, dim=0)
        log_resps = coo_to_scipy(log_resps).tocsr()

        # now, we want the mean of the softmaxes over the spike dim (dim=1)
        # since we have log softmaxes, that means need to exp, then take mean, then log
        log_props = logmeanexp(log_resps)
        self.log_proportions = torch.asarray(
            log_props, dtype=torch.float, device=self.data.device
        )

    def reassign(self, log_liks):
        n_units = log_liks.shape[0] - self.with_noise_unit
        assignments, spike_logliks, log_liks_csc = loglik_reassign(
            log_liks,
            has_noise_unit=self.with_noise_unit,
            log_proportions=self.log_proportions,
        )
        assignments = torch.from_numpy(assignments).to(self.labels)

        # track reassignments
        same = torch.zeros_like(assignments)
        torch.eq(self.labels, assignments, out=same)

        # total number of reassigned spikes
        reassign_count = len(same) - same.sum()

        # and per unit IoU
        intersection = torch.zeros(n_units, dtype=int)
        (kept,) = (assignments >= 0).nonzero(as_tuple=True)
        (orig_kept,) = (self.labels >= 0).nonzero(as_tuple=True)
        spiketorch.add_at_(intersection, assignments[kept], same[kept])
        union = torch.zeros_like(intersection)
        union -= intersection
        _1 = union.new_ones((1,))
        spiketorch.add_at_(union, assignments[kept], _1.broadcast_to(kept.shape))
        spiketorch.add_at_(union, self.labels[orig_kept], _1.broadcast_to(orig_kept.shape))
        iou = intersection / union
        unit_churn = 1.0 - iou

        # update labels
        self.labels.copy_(assignments)

        return unit_churn, reassign_count, spike_logliks, log_liks_csc

    def cleanup(self, log_liks=None, min_count=None, clean_props=None):
        """Remove too-small units, make label space contiguous, tidy all properties"""
        if min_count is None:
            min_count = self.min_count

        label_ids, counts = torch.unique(self.labels, return_counts=True)
        counts = counts[label_ids >= 0]
        label_ids = label_ids[label_ids >= 0]
        big_enough = counts >= min_count

        keep = torch.zeros(self.n_units(), dtype=bool)
        keep[label_ids] = big_enough
        self._stack = None

        if keep.all():
            return log_liks, clean_props

        if clean_props:
            clean_props = {k: v[keep] for k, v in clean_props.items()}

        keep_noise = keep.clone()
        if self.with_noise_unit:
            keep_noise = torch.concatenate((keep, torch.ones_like(keep[:1])))
        keep = keep.numpy(force=True)

        kept_ids = label_ids[big_enough]
        new_ids = torch.arange(kept_ids.numel())
        old2new = dict(zip(kept_ids, new_ids))
        self._relabel(kept_ids)

        if self.log_proportions is not None:
            lps = self.log_proportions.numpy(force=True)
            lps = lps[keep_noise.numpy(force=True)]
            # logsumexp to 0 (sumexp to 1) again
            lps -= logsumexp(lps)
            self.log_proportions = self.log_proportions.new_tensor(lps)

        if not self.empty():
            keep_units = {}
            for oi, ni in zip(kept_ids, new_ids):
                u = self[oi]
                keep_units[ni] = u
            self.clear_units()
            self.update(keep_units)

        if self.next_round_annotations:
            next_round_annotations = {}
            for j, nra in self.next_round_annotations.items():
                if keep[j]:
                    next_round_annotations[old2new[j]] = nra
            self.next_round_annotations = next_round_annotations

        if log_liks is None:
            return log_liks, clean_props

        keep_ll = keep_noise.numpy(force=True)
        assert keep_ll.size == log_liks.shape[0]

        if isinstance(log_liks, coo_array):
            log_liks = coo_sparse_mask_rows(log_liks, keep_ll)
        elif isinstance(log_liks, csc_array):
            keep_ll = np.flatnonzero(keep_ll)
            assert keep_ll.max() <= log_liks.shape[0] - self.with_noise_unit
            log_liks = log_liks[keep_ll]
        else:
            assert False

        return log_liks, clean_props

    def merge(self, log_liks=None, show_progress=True):
        new_labels, new_ids = self.merge_units(
            likelihoods=log_liks, show_progress=show_progress
        )
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

        clear_ids = []
        for res in results:
            if "new_ids" in res:
                for nid in res["new_ids"]:
                    self.schedule_annotations(nid, split_parent=res["parent_id"])
            clear_ids.extend(res["clear_ids"])
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
        mean_only = kind == "noise_metric"
        ids, means, covs, logdets = self.stack_units(
            nu=len(ids), ids=ids, units=units, mean_only=mean_only
        )

        # output will land here
        dists = np.full((nu, nu), np.inf, dtype=np.float32)
        np.fill_diagonal(dists, 0.0)

        # reverse KL is faster since there is only one cov to solve with
        transposed = False
        averaged = False
        kind_ = kind
        if kind == "kl":
            kind_ = "reverse_kl"
            transposed = True
        if kind == "symkl":
            kind_ = "reverse_kl"
            averaged = True

        # worker fn for parallelization
        @delayed
        def dist_job(j, unit):
            d = unit.divergence(
                means, other_covs=covs, other_logdets=logdets, kind=kind_
            )
            d = d.numpy(force=True).astype(dists.dtype)
            if transposed:
                dists[ids, j] = d
            else:
                dists[j, ids] = d

        pool = Parallel(
            self.n_threads, backend="threading", return_as="generator_unordered"
        )
        results = pool(dist_job(j, u) for j, u in zip(ids, units))
        if show_progress:
            results = tqdm(
                results, desc="Distances", total=len(ids), unit="unit", **tqdm_kw
            )
        for _ in results:
            pass

        if averaged:
            dists *= 0.5
            dists += dists.T

        # normalize by dividing by the divergence under the noise unit
        if normalization_kind == "noise":
            denom = self.noise_unit.divergence(
                means, other_covs=covs, other_logdets=logdets, kind=kind
            )
            denom = denom.sqrt_().numpy(force=True)
            denom[denom == 0] = 1.0
            dists[:, ids] /= denom[None, :]
            dists[ids, :] /= denom[:, None]
        elif normalization_kind == "channels":
            dists /= self.data.n_channels

        return dists

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
        indices_full=None,
        max_size=None,
        return_full_indices=False,
        split_name="train",
    ):
        if indices_full is None:
            if unit_id is not None:
                in_u = self.labels == unit_id
            elif unit_ids is not None:
                in_u = torch.isin(self.labels, unit_ids)
            else:
                assert False
            (indices_full,) = in_u.nonzero(as_tuple=True)

        if split_name is not None:
            indices_full = self.data.restrict_to_split(
                indices_full, split_name=split_name
            )

        n_full = indices_full.numel()
        if max_size and n_full > max_size:
            indices = self.rg.choice(n_full, size=max_size, replace=False)
            indices.sort()
            indices = torch.asarray(indices, device=indices_full.device)
            indices = indices_full[indices]
        else:
            indices = indices_full

        if return_full_indices:
            return indices_full, indices
        return indices

    def random_spike_data(
        self,
        unit_id=None,
        unit_ids=None,
        indices=None,
        indices_full=None,
        max_size=None,
        neighborhood="extract",
        split_name="train",
        with_reconstructions=False,
        return_full_indices=False,
        with_neighborhood_ids=False,
    ):
        if indices is None:
            indices_full, indices = self.random_indices(
                unit_id=unit_id,
                unit_ids=unit_ids,
                max_size=max_size,
                indices_full=indices_full,
                split_name=split_name,
                return_full_indices=True,
            )

        sp = self.data.spike_data(
            indices,
            neighborhood=neighborhood,
            with_reconstructions=with_reconstructions,
            with_neighborhood_ids=with_neighborhood_ids,
        )

        if return_full_indices:
            return indices_full, sp
        return sp

    def fit_unit(
        self,
        unit_id=None,
        indices=None,
        likelihoods=None,
        weights=None,
        features=None,
        warm_start=False,
        verbose=False,
        **unit_args,
    ):
        if features is None:
            features = self.random_spike_data(
                unit_id=unit_id, indices=indices, max_size=self.n_spikes_fit, with_neighborhood_ids=True
            )
        if verbose:
            logger.info(f"Fit {unit_id=} {features=}")
        if weights is None and likelihoods is not None:
            weights = self.get_fit_weights(unit_id, features.indices, likelihoods)
            (valid,) = torch.nonzero(weights, as_tuple=True)
            valid = valid.cpu()
            weights = weights[valid]
            features = features[valid]
        if verbose and weights is not None:
            logger.info(f"{weights.sum()=} {weights.min()=} {weights.max()=}")
        unit_args = self.unit_args | unit_args
        if warm_start and unit_id in self:
            unit = self[unit_id]
            unit.fit(features, weights, neighborhoods=self.data.extract_neighborhoods)
        else:
            unit = GaussianUnit.from_features(
                features,
                weights,
                neighborhoods=self.data.extract_neighborhoods,
                **unit_args,
            )
        return unit

    def unit_log_likelihoods(
        self,
        unit_id=None,
        unit=None,
        spike_indices=None,
        spikes=None,
        neighbs=None,
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
            spike_indices = spikes.indices
        inds_already = spike_indices is not None
        if neighbs is None or ns is None:
            if inds_already:
                # in this case, the indices returned in the structure are
                # relative indices inside spike_indices
                neighbs, ns = self.data.core_neighborhoods.spike_neighborhoods(
                    core_channels, spike_indices
                )
            else:
                cn, neighbs, ns = self.data.core_neighborhoods.subset_neighborhoods(
                    core_channels, batch_size=self.likelihood_batch_size
                )
                unit.annotations["covered_neighbs"] = cn
        if not ns:
            return None, None

        if not inds_already:
            spike_indices = torch.empty(ns, dtype=int)
            offset = 0
            log_likelihoods = torch.empty(ns)
        else:
            log_likelihoods = torch.full(
                (len(spike_indices),), -torch.inf, device=self.data.device
            )

        jobs = neighbs
        if show_progress:
            jobs = tqdm(
                jobs, desc=f"{desc_prefix}logliks", total=len(neighbs), **tqdm_kw
            )

        for neighb_id, neighb_chans, neighb_member_ixs, batch_start in jobs:
            chans_valid = self.data.core_neighborhoods.valid_mask(neighb_id)
            neighb_chans = neighb_chans[chans_valid]

            if spikes is not None:
                sp = spikes[neighb_member_ixs]
            elif inds_already:
                sp = self.data.spike_data(
                    spike_indices[neighb_member_ixs],
                    with_channels=False,
                    neighborhood="core",
                )
                features = sp.features
                features = features[..., chans_valid]
            else:
                # sp = self.data.spike_data(
                #     neighb_member_ixs,
                #     with_channels=False,
                #     neighborhood="core",
                # )
                features = self.data.core_neighborhoods.neighborhood_features(
                    neighb_id,
                    batch_start=batch_start,
                    batch_size=self.likelihood_batch_size,
                )
                features = features.to(self.data.device)

            lls = unit.log_likelihood(features, neighb_chans, neighborhood_id=neighb_id)

            if inds_already:
                log_likelihoods[neighb_member_ixs.to(log_likelihoods.device)] = lls
            else:
                nbatch = len(lls)
                spike_indices[offset : offset + nbatch] = neighb_member_ixs
                log_likelihoods[offset : offset + nbatch] = lls
                offset += nbatch

        if return_sorted and not inds_already:
            spike_indices, order = spike_indices.sort()
            log_likelihoods = log_likelihoods[order]

        return spike_indices, log_likelihoods

    def noise_log_likelihoods(self, indices=None, show_progress=False):
        if self._noise_log_likelihoods is None:
            self._noise_six, self._noise_log_likelihoods = self.unit_log_likelihoods(
                unit=self.noise_unit, show_progress=show_progress, desc_prefix="Noise "
            )
        if indices is not None:
            return self._noise_log_likelihoods[indices]
        return self._noise_six, self._noise_log_likelihoods

    def kmeans_split_unit(self, unit_id, debug=False):
        # get spike data and use interpolation to fill it out to the
        # unit's channel set
        result = dict(parent_id=unit_id, new_ids=[unit_id], clear_ids=[])
        unit = self[unit_id]
        if not unit.channels.numel():
            return result

        indices_full, sp = self.random_spike_data(
            unit_id, return_full_indices=True, with_neighborhood_ids=True
        )
        if not indices_full.numel() > self.min_count:
            return result

        Xo = X = self.data.interp_to_chans(sp, unit.channels)
        if self.split_whiten:
            X = self.noise.whiten(X, channels=unit.channels)

        if debug:
            result.update(dict(indices_full=indices_full, sp=sp, X=Xo, Xw=X))
        else:
            del Xo

        # run kmeans with kmeans++ initialization
        split_labels, responsibilities = kmeans(
            X.view(len(X), -1),
            n_iter=self.kmeans_n_iter,
            n_components=self.kmeans_k,
            random_state=self.rg,
            kmeanspp_initial=self.kmeans_kmeanspp_initial,
            with_proportions=self.kmeans_with_proportions,
            drop_prop=self.kmeans_drop_prop,
        )
        if debug:
            result["split_labels"] = split_labels
            result["responsibilities"] = responsibilities
        split_labels = split_labels.cpu()
        split_ids, split_labels = split_labels.unique(return_inverse=True)
        assert split_ids.min() >= 0
        if split_labels.unique().numel() <= 1:
            return result
        responsibilities = responsibilities[:, split_ids]

        # avoid oversplitting by doing a mini merge here
        split_labels = self.mini_merge(
            sp,
            split_labels,
            unit_id,
            weights=responsibilities.T,
            debug=debug,
            debug_info=result,
        )
        split_ids, split_counts = np.unique(split_labels, return_counts=True)
        valid = split_ids >= 0
        if not valid.any():
            return result
        split_ids = split_ids[valid]
        if not np.array_equal(split_ids, np.arange(len(split_ids))):
            raise ValueError(f"Bad {split_ids=}")

        if debug:
            result["merge_labels"] = split_labels
            return result

        split_labels = torch.asarray(split_labels, device=self.labels.device)
        n_new_units = split_ids.size - 1
        if n_new_units < 1:
            # quick case
            with self.labels_lock:
                self.labels[indices_full] = -1
                self.labels[sp.indices[split_labels >= 0]] = unit_id
            return result

        # else, tack new units onto the end
        # we need to lock up the labels array access here because labels.max()
        # depends on what other splitting units are doing!
        with self.labels_lock:
            # new indices start here
            next_label = self.labels.max() + 1

            # new indices are already >= 1, so subtract 1
            split_labels[split_labels >= 1] += next_label - 1
            # unit 0 takes the place of the current unit
            split_labels[split_labels == 0] = unit_id
            self.labels[indices_full] = -1
            self.labels[sp.indices] = split_labels

            if self.log_proportions is None:
                return

            # each sub-unit's prop is its fraction of assigns * orig unit prop
            split_counts = split_counts[valid]
            new_log_props = np.log(split_counts) - np.log(split_counts.sum())
            new_log_props = torch.from_numpy(new_log_props).to(self.log_proportions)
            new_log_props += self.log_proportions[unit_id]
            assert new_log_props.numel() == n_new_units + 1

            cur_len_with_noise = self.log_proportions.numel()
            noise_log_prop = self.log_proportions[-1]
            # self.log_proportions.resize_(cur_len_with_noise + n_new_units)
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

    def mini_merge(
        self,
        spike_data,
        labels,
        unit_id,
        weights=None,
        debug=False,
        debug_info=None,
        n_em_iter=None,
    ):
        """Given labels for a small bag of data, fit and merge."""
        if n_em_iter is None:
            n_em_iter = self.split_em_iter

        # E/M sub-units
        units = lls = None
        for _ in range(max(1, n_em_iter)):
            unique_labels, label_counts = labels.unique(return_counts=True)
            valid = unique_labels >= 0
            unique_labels = unique_labels[valid]
            label_counts = label_counts[valid]
            big_enough = label_counts >= self.channels_count_min
            if not big_enough.any():
                labels.fill_(-1)
                return labels

            kept = torch.isin(labels, unique_labels[big_enough])
            kept_labels = labels[kept]
            labels.fill_(-1)
            labels[kept] = torch.searchsorted(unique_labels[big_enough], kept_labels)
            weights = weights[big_enough]

            units = []
            for j, label in enumerate(unique_labels[big_enough]):
                (in_label,) = torch.nonzero(labels == label, as_tuple=True)
                w = None if weights is None else weights[j][in_label]
                features = spike_data[in_label.to(spike_data.indices.device)]
                unit = GaussianUnit.from_features(
                    features,
                    weights=w,
                    neighborhoods=self.data.extract_neighborhoods,
                    **self.split_unit_args,
                )
                if unit.channels.numel():
                    units.append(unit)

            if len(units) <= 1:
                return labels

            # determine their bimodalities while at once mini-reassigning
            lls = spike_data.features.new_full(
                (len(units), len(spike_data)), -torch.inf
            )
            for j, unit in enumerate(units):
                inds_, lls_ = self.unit_log_likelihoods(
                    unit=unit, spike_indices=spike_data.indices, ignore_channels=True
                )
                if lls_ is not None:
                    lls[j] = lls_
            best_liks, labels = lls.max(dim=0)
            labels[torch.isinf(best_liks)] = -1
            labels = labels.cpu()
            weights = F.softmax(lls, dim=0)

        labels = labels.numpy(force=True)
        ids, counts = np.unique(labels, return_counts=True)
        valid = np.logical_and(ids >= 0, counts >= self.min_count)
        ids = ids[valid]
        units = [units[ii] for ii in ids]
        keepers = np.flatnonzero(np.isin(labels, ids))
        labels[keepers] = np.searchsorted(ids, labels[keepers])
        if ids.size <= 1:
            return labels

        new_labels, new_ids = self.merge_units(
            units=units,
            labels=labels,
            override_unit_id=unit_id,
            likelihoods=lls[ids],
            spike_data=spike_data,
            debug_info=debug_info,
        )
        if debug:
            debug_info["reas_labels"] = labels
            debug_info["units"] = units
            debug_info["lls"] = lls

        return new_labels

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

        ina = shrinkfit(ina, max_spikes, self.rg)
        inb = shrinkfit(inb, max_spikes, self.rg)

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
        unit_ids=None,
        units=None,
        labels=None,
        override_unit_id=None,
        spikes_extract=None,
        max_distance=1.0,
        threshold=None,
        criterion="heldout_ccl",
        likelihoods=None,
        weights=None,
        spikes_per_subunit=4096,
        sym_function=np.maximum,
        normalization_kind=None,
        show_progress=False,
    ):
        # heuristic unit groupings to investigate
        distances = sym_function(distances, distances.T)
        distances = distances[np.triu_indices(len(distances), k=1)]
        Z = linkage(distances)
        n_units = len(Z) + 1

        # figure out the leaf nodes in each cluster in the hierarchy
        # up to max_distance
        clusters = leafsets(Z)

        if threshold is None:
            threshold = self.merge_criterion_threshold

        # walk backwards through the tree
        already_merged_leaves = set()
        improvements = np.full(n_units - 1, -np.inf)
        group_ids = np.arange(n_units)
        its = reversed(list(enumerate(Z)))
        in_bag = not criterion.startswith("heldout_")
        criterion = criterion.removeprefix("heldout_")
        if show_progress:
            its = tqdm(its, desc="Tree", total=n_units - 1, **tqdm_kw)
        for i, (pa, pb, dist, nab) in its:
            if not np.isfinite(dist) or dist > max_distance:
                continue

            # did we already merge a cluster containing this one?
            pleaves = clusters.get(pa, [pa]) + clusters.get(pb, [pb])
            contained = [l in already_merged_leaves for l in pleaves]
            if any(contained):
                assert all(contained)
                improvements[i] = np.inf
                continue

            # check if should merge
            leaves = clusters[n_units + i]
            cluster_ids = leaves if unit_ids is None else unit_ids[leaves]
            level_spe = None
            if spikes_extract is not None:
                in_level = np.flatnonzero(np.isin(labels, cluster_ids))
                level_spe = spikes_extract[in_level]

            level_likelihoods = likelihoods
            level_units = units
            if units is not None:
                level_units = [units[l] for l in leaves]
                level_likelihoods = likelihoods[leaves][:, in_level]

            crit = self.merge_criteria(
                unit_ids=cluster_ids,
                units=level_units,
                spikes_extract=level_spe,
                likelihoods=level_likelihoods,
                in_bag=in_bag,
                spikes_per_subunit=spikes_per_subunit,
                override_unit_id=override_unit_id,
            )
            if crit is not None:
                improvements[i] = crit[criterion]

            # how to actually merge?
            if improvements[i] > -threshold:
                group_ids[leaves] = n_units + i
                already_merged_leaves.update(leaves)

        return Z, group_ids, improvements

    def merge_criteria(
        self,
        unit_ids,
        units=None,
        likelihoods=None,
        override_unit_id=None,
        spikes_extract=None,
        in_bag=False,
        spikes_per_subunit=2048,
        min_overlap=0.8,
        class_balancing="worst",
        debug=False,
    ):
        """See if a single unit explains a group

        This code handles two cases. In the split step, we're computing things for a
        group of hypothetical units which don't correspond to labels in self.labels.
        In that case, likelihoods is dense and has len(likelihoods)==len(unit_ids).
        Otherwise, likelihoods is the usual csc array and unit_ids indexes its rows.

        And, it computes several criteria.
        """
        # check our two cases...
        if isinstance(likelihoods, csc_array):
            # "merge step" case. using units in self._units
            assert units is None
            assert override_unit_id is None
        else:
            # "split step" case. using pre-fit hypothetical units.
            assert torch.is_tensor(likelihoods)
            assert units is not None
            assert unit_ids is not None
            assert override_unit_id is not None
            assert spikes_extract is not None
            assert len(units) == len(unit_ids) == len(likelihoods)
            assert likelihoods.shape[1] == len(spikes_extract)

        unit_ids = torch.asarray(unit_ids)
        dim_units = 0  # naming this for clarity in sums below

        # pick spikes for merged unit fit
        if spikes_extract is None:
            in_subunits = [
                self.random_indices(u, max_size=spikes_per_subunit) for u in unit_ids
            ]
            in_any = torch.cat(in_subunits)
            in_any, in_order = torch.sort(in_any)
            spikes_extract = self.data.spike_data(in_any, with_neighborhood_ids=True)
        fit_ix = spikes_extract.indices

        # fit merged unit
        fit_weights = None
        if isinstance(likelihoods, csc_array):
            fit_weights = self.get_log_likelihoods(fit_ix, likelihoods)
            fit_weights = torch.sparse.softmax(fit_weights, dim=dim_units)
            fit_weights = fit_weights.to_dense()[unit_ids].sum(dim=dim_units)
            assert fit_weights.shape == (len(spikes_extract),)
        merged_unit = self.fit_unit(features=spikes_extract, weights=fit_weights)

        # pick spikes for likelihood computation
        if in_bag:
            spikes_core = self.data.spike_data(
                fit_ix, neighborhood="core", with_neighborhood_ids=True
            )
        else:
            spikes_core = self.random_spike_data(
                unit_id=override_unit_id,
                unit_ids=unit_ids,
                split_name="val",
                neighborhood="core",
                with_neighborhood_ids=True,
            )

        # get original units' log likelihoods
        # two cases: pre-computed dense ones, or we need to grab from sparse
        # we also grab the sum of responsibilities in the first case. this
        # helps us compute the local change in log lik due to this merge
        lik_weights = None
        if isinstance(likelihoods, csc_array):
            lik_ix = spikes_core.indices

            # below we compute likelihoods, but the proportion vector should be re-normalized.
            # need to multiply by 1/sum(suprops), aka subtract logsumexp(logsuprops)
            prop_correction = torch.logsumexp(self.log_proportions[unit_ids], dim=0)

            if in_bag:
                full_logliks = self.get_log_likelihoods(
                    lik_ix, likelihoods, unit_ids=unit_ids, dense=True
                )
                full_logliks -= prop_correction
                lik_weights = fit_weights
            else:
                full_logliks_sp = self.get_log_likelihoods(lik_ix, likelihoods)
                lik_weights = torch.sparse.softmax(full_logliks_sp, dim=dim_units)
                data = full_logliks_sp.values()
                full_logliks = data.new_full(full_logliks_sp.shape, -torch.inf)
                full_logliks[*full_logliks_sp.indices()] = data
                full_logliks = full_logliks[unit_ids] - prop_correction
                lik_weights = lik_weights.to_dense()[unit_ids].sum(dim=dim_units)
        else:
            full_logliks = spikes_core.features.new_full(
                (len(units), len(spikes_core)), -torch.inf
            )
            for j, unit in enumerate(units):
                _, ull = self.unit_log_likelihoods(unit=unit, spikes=spikes_core)
                if ull is not None:
                    full_logliks[j] = ull
            props = F.softmax(full_logliks, dim=dim_units).mean(1)
            full_logliks += props.log()[:, None]
        assert full_logliks.shape == (len(unit_ids), len(spikes_core))
        labels = full_logliks.argmax(0)
        labids, labixs, labcts = labels.unique(return_inverse=True, return_counts=True)
        if len(labids) < len(full_logliks):
            return None
        if labcts.min() < self.min_count:
            return None

        # compute entropy correction for class likelihood
        log_resps = F.log_softmax(full_logliks, dim=dim_units)
        log_resps.nan_to_num_(neginf=0.0, nan=torch.nan)
        ec = -(log_resps * log_resps.exp()).sum(dim=dim_units)

        # full model's log likelihood for each spike is logsumexp over units
        if debug:
            subunit_logliks = full_logliks
        keep0 = full_logliks.isfinite().all(dim=dim_units)
        full_logliks = torch.logsumexp(full_logliks, dim=dim_units)

        # merged model's likelihood
        _, merged_logliks = self.unit_log_likelihoods(
            unit=merged_unit, spikes=spikes_core
        )

        # it's possible for spikes to be ignored. that's fine, but within reason.
        if merged_logliks is None:
            return None
        keep = torch.logical_and(keep0, merged_logliks.isfinite())
        keep_mask = keep.cpu()
        labprops = torch.zeros_like(labcts)
        spiketorch.add_at_(labprops, labixs, keep.to(labcts))
        labprops = labprops / labcts
        if labprops.min() < min_overlap:
            if debug:
                return dict(
                    info=dict(
                        unit_logliks=merged_logliks[keep],
                        subunit_logliks=subunit_logliks[:, keep],
                        keep_mask=keep_mask,
                        merged_unit=merged_unit,
                    )
                )
            return None

        (keep,) = keep_mask.nonzero(as_tuple=True)
        n = keep.numel()

        labixs = labixs[keep]
        merged_logliks = merged_logliks[keep]
        full_logliks = full_logliks[keep]
        ec = ec[keep]
        if lik_weights is not None:
            lik_weights = lik_weights[keep]

        # get averages in each class
        if lik_weights is not None:
            class_w = class_sum(labids, labixs, lik_weights)
        else:
            class_w = labcts.to(torch.float)
        class_ec = class_sum(labids, labixs, ec, lik_weights) / class_w
        class_fll = class_sum(labids, labixs, full_logliks, lik_weights) / class_w
        class_mll = class_sum(labids, labixs, merged_logliks, lik_weights) / class_w

        if class_balancing == "worst":
            worst_ix = torch.argmin(class_mll - class_fll)
            ec = class_ec[worst_ix]
            fll = class_fll[worst_ix]
            mll = class_mll[worst_ix]
        elif class_balancing == "balanced":
            ec = class_ec.mean()
            fll = class_fll.mean()
            mll = class_mll.mean()
        else:
            assert False

        full_ll = fll.cpu().item()
        merged_ll = mll.cpu().item()
        ec = ec.cpu().item()

        full_criteria = dict(loglik=full_ll, ccl=full_ll - ec)
        merged_criteria = dict(loglik=merged_ll, ccl=merged_ll)
        if in_bag:
            # in-bag metrics include information criteria
            if units is None:
                units = [self[u] for u in unit_ids]
            k_full, k_merged = get_average_parameter_counts(
                units,
                merged_unit,
                spikes_core[keep],
                self.data.core_neighborhoods,
                use_proportions=self.use_proportions,
                reduce=False,
            )
            class_w = class_w.cpu()
            if lik_weights is not None:
                lik_weights = lik_weights.cpu()
            labixs = labixs.cpu()
            k_full = class_sum(labids, labixs, k_full, lik_weights) / class_w
            k_merged = class_sum(labids, labixs, k_merged, lik_weights) / class_w
            if class_balancing == "worst":
                k_full = k_full[worst_ix]
                k_merged = k_merged[worst_ix]
                n = class_w[worst_ix].cpu().item()
            elif class_balancing == "balanced":
                k_full = k_full.mean()
                k_merged = k_merged.mean()
                n = class_w.mean().cpu().item()

            # skip factors of 2
            full_criteria["aic"] = k_full / n - full_ll
            merged_criteria["aic"] = k_merged / n - merged_ll
            full_criteria["bic"] = 0.5 * (k_full * np.log(n) / n) - full_ll
            merged_criteria["bic"] = 0.5 * (k_merged * np.log(n) / n) - merged_ll
            full_criteria["icl"] = full_criteria["bic"] + ec
            merged_criteria["icl"] = merged_criteria["bic"]

        improvements = {}
        for k in full_criteria.keys():
            # *ic*: smaller=better, so diff=merged-full<0 means merge is good
            sign = -1 if k in ("aic", "bic", "icl") else 1
            improvements[k] = sign * (merged_criteria[k] - full_criteria[k])

        if debug:
            results = dict(
                improvements=improvements,
                full_criteria=full_criteria,
                merged_criteria=merged_criteria,
                info=dict(
                    unit_logliks=merged_logliks,
                    subunit_logliks=subunit_logliks[:, keep],
                    keep_mask=keep_mask,
                    merged_unit=merged_unit,
                ),
            )
            return results
        return improvements

    def get_log_likelihoods(
        self,
        indices,
        likelihoods,
        with_proportions=True,
        unit_ids=None,
        dense=False,
    ):
        # torch's index_select is painfully slow
        # weights = torch.index_select(likelihoods, 1, features.indices)
        # here we have weights as a csc_array
        liks = likelihoods[:, indices]
        if unit_ids is not None:
            liks = liks[unit_ids]

        liks = coo_to_torch(liks.tocoo(), torch.float, copy_data=True)
        liks = liks.to(self.data.device)
        if with_proportions and self.log_proportions is not None:
            log_props = self.log_proportions
            if unit_ids is not None:
                log_props = log_props[unit_ids]
            log_props = log_props[liks.indices()[0]]
            liks.values().add_(log_props)

        if dense:
            inds = liks.indices()
            data = liks.values()
            liks = data.new_full(liks.shape, -torch.inf)
            liks[*inds] = data

        return liks

    def get_fit_weights(self, unit_id, indices, likelihoods=None):
        """Responsibilities for subset of spikes."""
        if likelihoods is None:
            return None

        weights = self.get_log_likelihoods(indices, likelihoods)
        weights = torch.sparse.softmax(weights, dim=0)
        weights = weights[unit_id].to_dense()
        return weights

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

    def _relabel(self, old_labels, new_labels=None, flat=False):
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
        if new_labels is None:
            new_labels = torch.arange(len(old_labels))

        if flat:
            kept = self.labels >= 0
            label_indices = self.labels[kept]
        else:
            kept = torch.isin(self.labels, old_labels)
            label_indices = torch.searchsorted(old_labels, self.labels[kept])

        self.labels[kept] = new_labels.to(self.labels)[label_indices]
        self.labels[torch.logical_not(kept)] = -1
        self._stack = None

    def stack_units(
        self, nu=None, units=None, ids=None, mean_only=True, use_cache=False
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
            if mean_only or self._stack[-1] is not None:
                return self._stack

        rank, nc = self.data.rank, self.data.n_channels

        means = torch.full((nu, rank, nc), torch.nan, device=self.data.device)
        covs = logdets = None
        if not mean_only:
            if self.cov_kind == "ppca" and self.ppca_rank:
                covs = means.new_full((nu, rank, nc, self.ppca_rank), torch.nan)
            logdets = means.new_full((nu,), torch.nan)

        for j, unit in enumerate(units):
            means[j] = unit.mean
            if covs is not None:
                covs[j] = unit.W
            if logdets is not None:
                logdets[j] = unit.logdet()

        if use_cache:
            self._stack = ids, means, covs, logdets
        else:
            self._stack = None

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
        likelihoods=None,
        spike_data=None,
        labels=None,
        show_progress=False,
        merge_kind=None,
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
        if merge_kind is None:
            merge_kind = "hierarchical"
            if self.merge_criterion_threshold is not None:
                merge_kind = "tree"

        # distances are needed by both methods
        distances = self.distances(units=units, show_progress=show_progress)
        if debug_info is not None:
            debug_info["distances"] = distances

        if merge_kind == "hierarchical":
            return self.hierarchical_bimodality_merge(
                distances,
                labels,
                likelihoods,
                show_progress=show_progress,
                debug_info=debug_info,
            )
        elif merge_kind == "tree":
            Z, group_ids, improvements = self.tree_merge(
                distances,
                labels=labels,
                units=units,
                override_unit_id=override_unit_id,
                spikes_extract=spike_data,
                max_distance=self.merge_distance_threshold,
                criterion=self.merge_criterion,
                likelihoods=likelihoods,
                sym_function=self.merge_sym_function,
                show_progress=show_progress,
            )
            if debug_info is not None:
                debug_info["Z"] = Z
                debug_info["improvements"] = improvements

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
        prior_pseudocount=10,
        ppca_inner_em_iter=1,
        ppca_atol=0.05,
        ppca_rank=0,
        scale_mean: float = 0.1,
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
        self.scale_mean = scale_mean
        self.scale_alpha = float(prior_pseudocount)
        self.scale_beta = float(prior_pseudocount) / scale_mean
        self.ppca_rank = ppca_rank
        self.ppca_inner_em_iter = ppca_inner_em_iter
        self.ppca_atol = ppca_atol
        self.annotations = annotations

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
        scale_mean: float = 0.1,
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
            scale_mean=scale_mean,
            ppca_rank=ppca_rank,
            ppca_inner_em_iter=ppca_inner_em_iter,
            ppca_atol=ppca_atol,
            **annotations,
        )
        self.fit(features, weights, neighborhoods=neighborhoods)
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
        features: SpikeFeatures,
        weights: Optional[torch.Tensor] = None,
        neighborhoods: Optional["SpikeNeighborhoods"] = None,
        show_progress: bool = False,
    ):
        if features is None or len(features) < self.channels_count_min:
            self.pick_channels(None, None)
            return
        new_zeros = features.features.new_zeros

        if weights is not None:
            (kept,) = (weights > 0).cpu().nonzero(as_tuple=True)
            features = features[kept]
            weights = weights[kept]

        achans = occupied_chans(features, self.n_channels, neighborhoods=neighborhoods)
        # achans = achans.cpu()
        je_suis = achans.numel()
        do_pca = self.cov_kind == "ppca" and self.ppca_rank

        active_mean = active_W = None
        if hasattr(self, "mean"):
            active_mean = self.mean[:, achans]
        if hasattr(self, "W"):
            active_W = self.W[:, achans]

        if je_suis:
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
                n_iter=self.ppca_inner_em_iter,
                em_converged_atol=self.ppca_atol,
                mean_prior_pseudocount=self.prior_pseudocount,
                show_progress=show_progress,
                W_initialization="zeros",
            )

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
            mean_full[:, achans] = res["mu"]
            if res.get("W", None) is not None:
                W_full[:, achans] = res["W"]
        self.register_buffer("mean", mean_full)
        if do_pca:
            self.register_buffer("W", W_full)
        nobs = res["nobs"] if je_suis else None
        self.pick_channels(achans, nobs)

    def pick_channels(self, active_chans, nobs=None):
        if self.channels_strategy == "all":
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

        if self.channels_strategy == "snr":
            snr_min = np.sqrt(self.channels_count_min) * self.channels_snr_amp
            strong = snr >= snr_min
            self.register_buffer("channels", active_chans[strong.cpu()])
            return
        if self.channels_strategy == "count":
            strong = nobs >= self.channels_count_min
            self.register_buffer("channels", active_chans[strong.cpu()])
            return

        assert False

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
            self.cov_kind == "zero" or self.cov_kind == "ppca" and not self.ppca_rank
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

    def log_likelihood(self, features, channels, neighborhood_id=None) -> torch.Tensor:
        """Log likelihood for spike features living on the same channels."""
        mean = self.noise.mean_full[:, channels]
        if self.mean_kind == "full":
            mean = mean + self.mean[:, channels]
        features = features - mean

        cov = self.marginal_covariance(
            channels, cache_key=neighborhood_id, device=features.device
        )
        y = features.view(len(features), -1)
        ll = spiketorch.ll_via_inv_quad(cov, y)
        return ll

    def divergence(
        self, other_means, other_covs=None, other_logdets=None, kind="noise_metric"
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
        is_ppca = self.cov_kind == "ppca" and self.ppca_rank

        if other is not None:
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
        inv_quad = other_covs.inv_quad(dmu.unsqueeze(-1), reduce_inv_quad=False)
        assert inv_quad.shape == (n, 1)
        inv_quad = inv_quad[:, 0]

        # get logdet term
        ld = 0.0
        if is_ppca:
            ld = other_logdets - self.logdet()
        kl = 0.5 * (inv_quad + ((tr - k) + ld))
        if return_extra:
            return dict(kl=kl, inv_quad=inv_quad, ld=ld, tr=tr, k=k)
        return kl

    def reverse_kl_divergence(self, other_means, other_covs, other_logdets):
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
            solve = my_cov.solve(oW)
            ncov = self.noise.full_dense_cov()
            solve = solve @ oW.mT
            tr = solve.diagonal(dim1=-2, dim2=-1).sum(dim=1)
            tr += torch.trace(my_cov.solve(ncov))
            ld = self_logdet - other_logdets
        return 0.5 * (inv_quad + ((tr - k) + ld))


# -- utilities


log2pi = torch.log(torch.tensor(2.0 * torch.pi))
tqdm_kw = dict(smoothing=0, mininterval=1.0 / 24.0)


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
    spiketorch.add_at_(wsum, inverse_inds, x)
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


def get_coo_storage(ns_total, storage, use_storage):
    if not use_storage:
        # coo_uix = np.empty(ns_total, dtype=int)
        coo_six = np.empty(ns_total, dtype=int)
        coo_data = np.empty(ns_total, dtype=np.float32)
        return coo_six, coo_data

    if hasattr(storage, "coo_data"):
        if storage.coo_data.size < ns_total:
            # del storage.coo_uix
            del storage.coo_six
            del storage.coo_data
        # storage.coo_uix = np.empty(ns_total, dtype=int)
        storage.coo_six = np.empty(ns_total, dtype=int)
        storage.coo_data = np.empty(ns_total, dtype=np.float32)
    else:
        # storage.coo_uix = np.empty(ns_total, dtype=int)
        storage.coo_six = np.empty(ns_total, dtype=int)
        storage.coo_data = np.empty(ns_total, dtype=np.float32)

    # return storage.coo_uix, storage.coo_six, storage.coo_data
    return storage.coo_six, storage.coo_data


def get_csc_storage(ns_total, storage, use_storage):
    if not use_storage:
        csc_row_indices = np.empty(ns_total, dtype=int)
        csc_data = np.empty(ns_total, dtype=np.float32)
        return csc_row_indices, csc_data

    if hasattr(storage, "csc_data"):
        if storage.csc_data.size < ns_total:
            del storage.csc_row_indices
            del storage.csc_data
        storage.csc_row_indices = np.empty(ns_total, dtype=int)
        storage.csc_data = np.empty(ns_total, dtype=np.float32)
    else:
        storage.csc_row_indices = np.empty(ns_total, dtype=int)
        storage.csc_data = np.empty(ns_total, dtype=np.float32)

    return storage.csc_row_indices, storage.csc_data


def coo_to_torch(coo_array, dtype, transpose=False, is_coalesced=True, copy_data=False):
    coo = (
        torch.from_numpy(coo_array.coords[int(transpose)]),
        torch.from_numpy(coo_array.coords[1 - int(transpose)]),
    )
    s0, s1 = coo_array.shape
    if transpose:
        s0, s1 = s1, s0
    res = torch.sparse_coo_tensor(
        torch.row_stack(coo),
        torch.asarray(coo_array.data, dtype=torch.float, copy=copy_data),
        size=(s0, s1),
        is_coalesced=is_coalesced,
    )
    return res


def coo_to_scipy(coo_tensor):
    data = coo_tensor.values().numpy(force=True)
    coords = coo_tensor.indices().numpy(force=True)
    return coo_array((data, coords), shape=coo_tensor.shape)


def marginal_loglik(
    indices, log_proportions, log_likelihoods, unit_ids=None, reduce="mean"
):
    if unit_ids is not None:
        # renormalize log props
        log_proportions = log_proportions[unit_ids]
        log_proportions = log_proportions - logsumexp(log_proportions)

    log_likelihoods = log_likelihoods[:, indices]
    if unit_ids is not None:
        log_likelihoods = log_likelihoods[unit_ids]

    # .indices == row inds for a csc_array
    # since log_proportions and log_likelihoods both were sliced by the same
    # unit_ids, the row indices match.
    props = log_proportions[log_likelihoods.indices]
    log_liks = props + log_likelihoods.data

    if reduce == "mean":
        ll = log_liks.mean()
    elif reduce == "sum":
        ll = log_liks.sum()
    else:
        assert False

    return ll


def loglik_reassign(
    log_liks, has_noise_unit=False, proportions=None, log_proportions=None
):
    log_liks_csc, assignments, spike_logliks = sparse_reassign(
        log_liks,
        return_csc=True,
        proportions=proportions,
        log_proportions=log_proportions,
    )
    n_units = log_liks.shape[0] - has_noise_unit
    if has_noise_unit:
        assignments[assignments >= n_units] = -1
    return assignments, spike_logliks, log_liks_csc


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


def sparse_reassign(
    liks, match_threshold=None, return_csc=False, proportions=None, log_proportions=None
):
    """Reassign spikes to units with largest likelihood

    liks is (n_units, n_spikes). This computes the argmax for each column,
    treating sparse 0s as -infs rather than as 0s.

    Turns out that scipy's sparse argmin/max have a slow python inner loop,
    this uses a numba replacement, but I'd like to upstream a cython version.
    """
    if not liks.nnz:
        return np.full(liks.shape[1], -1), np.full(liks.shape[1], -np.inf)

    # csc is needed here for this to be fast
    liks = liks.tocsc()
    nz_lines = np.flatnonzero(np.diff(liks.indptr))

    # see scipy csc argmin/argmax for reference here. this is just numba-ing
    # a special case of that code which has a python hot loop.
    assignments = np.full(liks.shape[1], -1)
    # these will be filled with logsumexps
    likelihoods = np.full(liks.shape[1], -np.inf, dtype=np.float32)

    # get log proportions, either given logs or otherwise...
    if log_proportions is None:
        if proportions is None:
            log_proportions = np.full(
                liks.shape[0], -np.log(liks.shape[0]), dtype=np.float32
            )
        elif torch.is_tensor(proportions):
            log_proportions = proportions.log().numpy(force=True)
        else:
            log_proportions = np.log(proportions)
    else:
        if torch.is_tensor(log_proportions):
            log_proportions = log_proportions.numpy(force=True)
    log_proportions = log_proportions.astype(np.float32)

    # this loop ignores sparse zeros. so, no sweat for negative inputs.
    hot_argmax_loop(
        assignments,
        likelihoods,
        nz_lines,
        liks.indptr,
        liks.data,
        liks.indices,
        log_proportions,
    )

    if return_csc:
        return liks, assignments, likelihoods
    return assignments, likelihoods


# csc can have int32 or 64 coos on dif platforms? is this an intp? :P
sigs = [
    "void(i8[::1], f4[::1], i8[::1], i4[::1], f4[::1], i4[::1], f4[::1])",
    "void(i8[::1], f4[::1], i8[::1], i8[::1], f4[::1], i8[::1], f4[::1])",
]


@numba.njit(
    sigs,
    error_model="numpy",
    nogil=True,
    parallel=True,
)
def hot_argmax_loop(
    assignments, scores, nz_lines, indptr, data, indices, log_proportions
):
    # for i in nz_lines:
    for j in numba.prange(nz_lines.shape[0]):
        i = nz_lines[j]
        p = indptr[i]
        q = indptr[i + 1]
        ix = indices[p:q]
        dx = data[p:q] + log_proportions[ix]
        best = dx.argmax()
        assignments[i] = ix[best]
        mx = dx.max()
        scores[i] = mx + np.log(np.exp(dx - mx).sum())


sig = "void(i8, i8[::1], i8[::1], i8[::1], f4[::1], f4[::1])"


@numba.njit(sig, error_model="numpy", nogil=True, parallel=True)
def csc_insert(row, write_offsets, inds, csc_indices, csc_data, liks):
    """Insert elements into a CSC sparse array

    To use this, you need to know the indptr in advance. Then this function
    can help you to insert a row into the array. You have to insert all nz
    elements for that row at once in a single call to this function, and
    rows must be written in order.

    (However, the columns within each row can be unordered.)

    write_offsets should be initialized at the indptr -- so, they are
    each column's "write head", indicating how many rows have been written
    for that column so far.

    Then, this updates the row indices (csc_indices) array with the correct
    row for each column (inds), and adds the data in the right place. The
    "write heads" are incremented, so that when this fn is called for the next
    row things are in the right place.

    This would be equivalent to:
        data_ixs = write_offsets[inds]
        csc_indices[data_ixs] = row
        csc_data[data_ixs] = liks
        write_offsets[inds] += 1
    """
    for j in numba.prange(inds.shape[0]):
        ind = inds[j]
        data_ix = write_offsets[ind]
        csc_indices[data_ix] = row
        csc_data[data_ix] = liks[j]
        write_offsets[ind] += 1


def coo_sparse_mask_rows(coo, keep_mask):
    """Row indexing with a boolean mask."""
    if keep_mask.all():
        return coo

    kept_label_indices = np.flatnonzero(keep_mask)
    ii, jj = coo.coords
    ixs = np.searchsorted(kept_label_indices, ii)
    ixs.clip(0, kept_label_indices.size - 1, out=ixs)
    valid = np.flatnonzero(kept_label_indices[ixs] == ii)
    coo = coo_array(
        (coo.data[valid], (ixs[valid], jj[valid])),
        shape=(kept_label_indices.size, coo.shape[1]),
    )
    return coo


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
    n = x.numel()
    if n <= max_size:
        return x

    choices = rg.choice(n, size=max_size, replace=False)
    choices.sort()
    choices = torch.from_numpy(choices)
    return x[choices.to(x.device)]


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
        nbvalid = nbchans < noise.n_channels
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


def template_scale_map(
    xw,
    nu,
    weights,
    alpha=1.0,
    beta=1.0,
    allow_destroy=False,
    xtol=1e-2,
    n_iter=2000,
    lr=1e-1,
):
    n = len(xw)
    ztnu = xw if allow_destroy else xw.clone()
    nuw = nu if allow_destroy else nu.clone()
    del xw, nu

    nuw.mul_(weights.unsqueeze(1))
    ztnusq = ztnu.mul_(nuw).sum(dim=(1, 2)).square_()
    nutnu = nuw.square_().sum(dim=(1, 2))
    del nuw
    alpha_Non2_1 = alpha + n / 2 - 1.0
    # init = torch.sum(ztnusq / (1.0 + nutnu))
    init = torch.tensor(1.0)

    with torch.enable_grad():
        log_lambd = torch.log(init)
        log_lambd.requires_grad_(True)

        opt = torch.optim.Adam([log_lambd], lr)
        for j in range(n_iter):
            opt.zero_grad()
            lambd = log_lambd.exp()
            nutnu_p_lambd = lambd + nutnu
            logp = (
                alpha_Non2_1 * log_lambd
                - beta * lambd
                - 0.5 * nutnu_p_lambd.log().sum()
                + 0.5 * (ztnusq / nutnu_p_lambd).sum()
            )
            loss = -logp
            loss.backward()
            opt.step()

            if j <= 8:
                continue

            new_lambd = log_lambd.exp()
            if torch.abs(new_lambd - lambd) < xtol:
                break

    final = log_lambd.clone().detach().exp_()
    return 1.0 / final.sqrt()


def quick_indices(
    rg, unit_ids, labels, split_indices=None, max_sizes=4096
):
    """It's slow to do lots of nonzero(labels==j).

    This goes all in one by looking for the first yea many spikes in a random order.
    """
    labels_in_split = labels if split_indices is None else labels[split_indices]
    random_order = rg.permutation(len(labels_in_split))
    reordered_labels = labels_in_split[random_order]

    counts_so_far = np.zeros(unit_ids.max() + 1, dtype=np.int32)
    counts_so_far[unit_ids] = 0
    n_active = unit_ids.size
    reordered_indices = np.full((unit_ids.max() + 1, np.max(max_sizes)), labels.size + 1)

    full_max_sizes = np.zeros(unit_ids.max() + 1, dtype=np.int32)
    full_max_sizes[unit_ids] = max_sizes

    _quick_indices(np.int32(n_active), counts_so_far, reordered_labels, reordered_indices, full_max_sizes)

    to_original_index = random_order
    if split_indices is not None:
        to_original_index = split_indices[to_original_index]

    indices = {}
    for u in unit_ids:
        myixs = reordered_indices[u]
        myixs = myixs[myixs < labels.size + 1]
        myixs = to_original_index[myixs]
        myixs.sort()
        indices[u] = myixs

    return indices


sig = "void(i4, i4[::1], i8[::1], i8[:, ::1], i4[::1])"


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
