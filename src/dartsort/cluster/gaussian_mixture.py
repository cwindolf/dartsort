import threading
from dataclasses import replace

import numba
import numpy as np
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from linear_operator import operators
from scipy.sparse import coo_array, csc_array
from scipy.special import logsumexp
from tqdm.auto import tqdm, trange

from ..util import more_operators, noise_util
from .cluster_util import agglomerate
from .kmeans import kmeans
from .modes import smoothed_dipscore_at
from .stable_features import (
    SpikeFeatures,
    StableSpikeDataset,
    occupied_chans,
    zero_pad_to_chans,
)

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
        cov_kind="zero",
        use_proportions: bool = True,
        proportions_sample_size: int = 2**16,
        prior_type="niw",
        channels_strategy="snr",
        channels_strategy_snr_min=5.0,
        scale_mean: float = 0.1,
        prior_pseudocount: float = 10.0,
        random_seed: int = 0,
        n_threads: int = 4,
        min_count: int = 50,
        n_em_iters: int = 25,
        kmeans_k: int = 5,
        kmeans_n_iter: int = 100,
        kmeans_drop_prop: float = 0.025,
        kmeans_with_proportions: bool = False,
        kmeans_kmeanspp_initial: str = "mean",
        split_em_iter: int = 1,
        split_whiten: bool = True,
        distance_metric: str = "noise_metric",
        distance_noise_normalized: bool = True,
        merge_linkage: str = "complete",
        merge_distance_threshold: float = 1.0,
        merge_bimodality_threshold: float = 0.1,
        split_bimodality_threshold: float = 0.1,
        merge_bimodality_cut: float = 0.0,
        merge_bimodality_overlap: float = 0.80,
        merge_bimodality_weighted: bool = True,
        merge_bimodality_score_kind: str = "tv",
        merge_bimodality_masked: bool = False,
        merge_sym_function: callable = np.minimum,
        em_converged_prop: float = 0.02,
        em_converged_atol: float = 1e-2,
    ):
        super().__init__()

        # key data structures for loading and modeling spikes
        self.data = data
        self.noise = noise

        # parameters
        self.n_spikes_fit = n_spikes_fit
        self.n_threads = n_threads
        self.min_count = min_count
        self.n_em_iters = n_em_iters
        self.kmeans_k = kmeans_k
        self.kmeans_n_iter = kmeans_n_iter
        self.kmeans_with_proportions = kmeans_with_proportions
        self.kmeans_kmeanspp_initial = kmeans_kmeanspp_initial
        self.kmeans_drop_prop = kmeans_drop_prop
        self.distance_metric = distance_metric
        self.distance_noise_normalized = distance_noise_normalized
        self.merge_distance_threshold = merge_distance_threshold
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
        self.split_em_iter = split_em_iter
        self.split_whiten = split_whiten
        self.use_proportions = use_proportions
        self.proportions_sample_size = proportions_sample_size

        # store labels on cpu since we're always nonzeroing / writing np data
        self.labels = self.data.original_sorting.labels[data.kept_indices]
        self.labels = torch.from_numpy(self.labels)

        # this is populated by self.m_step()
        self.units = torch.nn.ModuleList()
        self.log_proportions = None

        # store arguments to the unit constructor in a dict
        self.unit_args = dict(
            noise=noise,
            mean_kind=mean_kind,
            cov_kind=cov_kind,
            prior_type=prior_type,
            channels_strategy=channels_strategy,
            channels_strategy_snr_min=channels_strategy_snr_min,
            prior_pseudocount=prior_pseudocount,
            scale_mean=scale_mean,
        )

        # clustering with noise unit to hopefully grab false positives
        noise_args = dict(mean_kind="zero", cov_kind="zero", channels_strategy="all")
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

    # -- headliners

    def to_sorting(self):
        labels = np.full_like(self.data.original_sorting.labels, -1)
        labels[self.data.kept_indices] = self.labels.numpy(force=True)
        return replace(self.data.original_sorting, labels=labels)

    def unit_ids(self):
        uids = torch.unique(self.labels)
        return uids[uids >= 0]

    def em(self, n_iter=None, show_progress=True, final_e_step=True):
        n_iter = self.n_em_iters if n_iter is None else n_iter
        if show_progress:
            its = trange(n_iter, desc="EM", **tqdm_kw)
            step_progress = max(0, int(show_progress) - 1)
        else:
            its = range(n_iter)

        # if we have no units, we can't E step.
        if not self.units:
            self.m_step(show_progress=step_progress)

        for _ in its:
            # for convergence testing...
            self.cleanup(min_count=1)
            means, *_ = self.stack_units(mean_only=True)

            reas_count, log_liks, spike_logliks = self.e_step(
                show_progress=step_progress
            )
            meanlogpx = spike_logliks.mean()

            # M step: fit units based on responsibilities
            max_adif = self.m_step(
                log_liks, show_progress=step_progress, prev_means=means
            )

            # extra info for description
            if show_progress:
                opct = (self.labels < 0).sum() / self.data.n_spikes
                opct = f"{100 * opct:.1f}"
                nu = len(self.units)
                reas_prop = reas_count / self.data.n_spikes
                rpct = f"{100 * reas_prop:.1f}"
                adif = f"{max_adif:.2f}"
                msg = f"EM[{nu=},out={opct}%,reas={rpct}%,dmu={adif},meanlogpx={meanlogpx:g}]"
                its.set_description(msg)

            if reas_prop < self.em_converged_prop:
                break
            if max_adif < self.em_converged_atol:
                break

        if not final_e_step:
            return

        # final e step for caller
        reas_count, log_liks, spike_logliks = self.e_step(clean_units=True)
        return log_liks

    def e_step(self, show_progress=False):
        # E step: get responsibilities and update hard assignments
        log_liks = self.log_likelihoods(show_progress=show_progress)
        # replace log_liks by csc
        reas_count, spike_logliks, log_liks = self.reassign(log_liks)

        # garbage collection -- get rid of low count labels
        log_liks = self.cleanup(log_liks)

        return reas_count, log_liks, spike_logliks

    def m_step(self, likelihoods=None, show_progress=False, prev_means=None):
        """Beware that this flattens the labels."""
        needs_append = not self.units
        unit_ids = self.unit_ids()
        if not needs_append:
            assert unit_ids.max() < len(self.units)

        if self.use_proportions and likelihoods is not None:
            self.update_proportions(likelihoods)
        if self.log_proportions is not None:
            assert len(self.log_proportions) > unit_ids.max() + 1

        pool = Parallel(self.n_threads, backend="threading", return_as="generator")
        results = pool(
            delayed(self.fit_unit)(
                j, likelihoods=likelihoods, **self.next_round_annotations.get(j, {})
            )
            for j in unit_ids
        )
        if show_progress:
            results = tqdm(
                results, desc="M step", unit="unit", total=len(unit_ids), **tqdm_kw
            )
        if needs_append:
            for j, unit in enumerate(zip(unit_ids, results)):
                assert unit.annotations["unit_id"] == j
                self.units.append(unit)
        if self.log_proportions is not None:
            # this is the index of the noise unit. it's got to be larger than
            # the largest unit index
            maxix = self.log_proportions.numel() - 1
            assert (unit_ids < maxix).all()
            ixs = torch.cat((unit_ids, torch.tensor([maxix])))
            self.log_proportions = self.log_proportions[ixs]
        if prev_means is not None:
            nu = len(unit_ids)
            prev_means = prev_means[unit_ids]
            new_means, *_ = self.stack_units(mean_only=True)
            dmu = (prev_means - new_means).abs_().view(nu, -1)
            adif = torch.max(dmu)
            return adif
        self._stack = None

    def log_likelihoods(
        self, unit_ids=None, with_noise_unit=True, use_storage=True, show_progress=False
    ):
        """Noise unit last so that rows correspond to unit ids without 1 offset"""
        if unit_ids is None:
            unit_ids = range(len(self.units))

        # how many units does each core neighborhood overlap with?
        n_cores = self.data.core_neighborhoods.n_neighborhoods
        if with_noise_unit:
            # everyone overlaps the noise unit
            core_overlaps = torch.ones(n_cores, dtype=int, device=self.data.device)
        else:
            core_overlaps = torch.zeros(n_cores, dtype=int, device=self.data.device)

        # for each unit, determine which spikes will be computed
        neighb_info = []
        nnz = 0
        for j in unit_ids:
            unit = self.units[j]
            neighbs, ns_unit = self.data.core_neighborhoods.subset_neighborhoods(
                unit.channels, add_to_overlaps=core_overlaps
            )
            neighb_info.append((j, neighbs, ns_unit))
            nnz += ns_unit

        # how many units does each spike overlap with? needed to write csc
        spike_overlaps = core_overlaps[
            self.data.core_neighborhoods.neighborhood_ids
        ].numpy(force=True)

        # add in space for the noise unit
        if with_noise_unit:
            nnz = nnz + self.data.n_spikes

        # get the big nnz-length csc buffers. these can be huge so we cache them.
        csc_indices, csc_data = get_csc_storage(nnz, self.storage, use_storage)
        # csc compressed indptr. spikes are columns.
        indptr = np.concatenate(([0], np.cumsum(spike_overlaps)))
        del spike_overlaps
        # each spike starts at writing at its indptr. as we gather more units for each
        # spike, we increment the spike's "write head". idea is to directly make csc
        write_offsets = indptr[:-1].copy()
        pool = Parallel(self.n_threads, backend="threading", return_as="generator")
        results = pool(
            delayed(self.unit_log_likelihoods)(unit_id=j, neighbs=neighbs, ns=ns)
            for j, neighbs, ns in neighb_info
        )
        if show_progress:
            results = tqdm(
                results,
                total=len(neighb_info),
                desc="Likelihoods",
                unit="unit",
                **tqdm_kw,
            )
        for j, (inds, liks) in enumerate(results):
            if inds is None:
                continue
            inds = inds.numpy(force=True)
            liks = liks.numpy(force=True)
            csc_insert(j, write_offsets, inds, csc_indices, csc_data, liks)
            # inds = inds.numpy(force=True)
            # data_ixs = write_offsets[inds]
            # csc_indices[data_ixs] = j
            # csc_data[data_ixs] = liks.numpy(force=True)
            # write_offsets[inds] += 1

        if with_noise_unit:
            inds, liks = self.noise_log_likelihoods()
            data_ixs = write_offsets[inds]
            # assert np.array_equal(data_ixs, ccol_indices[1:] - 1)  # just fyi
            csc_indices[data_ixs] = j + 1
            csc_data[data_ixs] = liks.numpy(force=True)

        shape = (len(unit_ids) + with_noise_unit, self.data.n_spikes)
        log_liks = csc_array((csc_data, csc_indices, indptr), shape=shape)
        log_liks.has_canonical_format = True

        return log_liks

    def update_proportions(self, log_liks):
        if not self.use_proportions:
            return

        # have to jump through some hoops because torch sparse tensors
        # don't implement .mean() yet??
        sample = self.rg.choice(
            log_liks.shape[1], size=self.proportions_sample_size, replace=False
        )
        sample.sort()
        log_liks = log_liks[:, sample].tocoo()
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
        has_noise_unit = log_liks.shape[1] > len(self.units)
        assignments, spike_logliks, log_liks_csc = loglik_reassign(
            log_liks,
            has_noise_unit=has_noise_unit,
            log_proportions=self.log_proportions,
        )
        assignments = torch.from_numpy(assignments).to(self.labels)
        reassign_count = (self.labels != assignments).sum()
        self.labels.copy_(assignments)
        return reassign_count, spike_logliks, log_liks_csc

    def cleanup(self, log_liks=None, clean_units=True, min_count=None):
        """Remove too-small units

        Also handles bookkeeping to throw those units out of the sparse
        log_liks array, and to throw away small units in self.units.
        clean_units should be true unless your next move is an m step.
        For instance if it's an E step, it's like the cleanup never happened.
        """
        unit_ids, counts = torch.unique(self.labels, return_counts=True)
        counts = counts[unit_ids >= 0]
        unit_ids = unit_ids[unit_ids >= 0]
        if min_count is None:
            min_count = self.min_count
        big_enough = counts >= min_count
        keep = torch.zeros(len(self.units), dtype=bool)
        keep[unit_ids] = big_enough
        if keep.all():
            return log_liks

        keep_noise = torch.concatenate((keep, torch.ones_like(keep[:1])))
        keep = keep.numpy(force=True)
        if log_liks is not None:
            has_noise_unit = log_liks.shape[1] > len(self.units)

        self.relabel_units(unit_ids[big_enough])
        if clean_units and self.log_proportions is not None:
            lps = self.log_proportions.numpy(force=True)
            lps = lps[keep_noise.numpy(force=True)]
            # logsumexp to 0 (sumexp to 1) again
            lps -= logsumexp(lps)
            self.log_proportions = self.log_proportions.new_tensor(lps)
        if clean_units and len(self.units):
            keep_units = [u for j, u in enumerate(self.units) if keep[j]]
            del self.units[:]
            self.units.extend(keep_units)
        if log_liks is None:
            return

        keep_ll = keep_noise.numpy(force=True) if has_noise_unit else keep
        assert keep_ll.size == log_liks.shape[0]

        if isinstance(log_liks, coo_array):
            log_liks = coo_sparse_mask_rows(log_liks, keep_ll)
        elif isinstance(log_liks, csc_array):
            keep_ll = np.flatnonzero(keep_ll)
            log_liks = log_liks[keep_ll]
        else:
            assert False

        return log_liks

    def merge(self, log_liks=None, show_progress=True):
        distances = self.distances(show_progress=show_progress)
        bimodalities = None
        if self.merge_bimodality_threshold is not None:
            compute_mask = distances <= self.merge_distance_threshold
            bimodalities = self.bimodalities(
                log_liks,
                compute_mask=compute_mask,
                show_progress=show_progress,
                weighted=self.merge_bimodality_weighted,
            )
        distances = combine_distances(
            distances,
            self.merge_distance_threshold,
            bimodalities,
            self.merge_bimodality_threshold,
            sym_function=self.merge_sym_function,
        )
        new_labels, new_ids = agglomerate(
            self.labels,
            distances,
            linkage_method=self.merge_linkage,
        )
        self.labels.copy_(torch.from_numpy(new_labels))
        del self.units[:]
        if self.log_proportions is not None:
            log_props = self.log_proportions.numpy(force=True)

            # sum the proportions within each merged ID
            unique_new_ids = np.unique(new_ids)
            assert np.array_equal(unique_new_ids, np.arange(unique_new_ids.size))
            new_log_props = np.empty(unique_new_ids.size + 1, dtype=log_props.dtype)
            new_log_props[-1] = log_props[-1]  # noise unit
            for j in unique_new_ids:
                new_log_props[j] = logsumexp(log_props[:-1][new_ids == j])
            self.log_proportions = torch.asarray(
                new_log_props, device=self.log_proportions.device
            )
        self._stack = None

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
        for _ in results:
            pass
        del self.units[:]
        self._stack = None

    def distances(
        self, kind=None, noise_normalized=None, units=None, show_progress=True
    ):
        # default to my settings but allow user to experiment
        if kind is None:
            kind = self.distance_metric
        if noise_normalized is None:
            noise_normalized = self.distance_noise_normalized

        if units is None:
            units = self.units
        nu = len(units)

        # stack unit data into one place
        mean_only = kind == "noise_metric"
        means, covs, logdets = self.stack_units(units, mean_only=mean_only)

        # compute denominator of noised normalized distances
        if noise_normalized:
            denom = self.noise_unit.divergence(
                means, other_covs=covs, other_logdets=logdets, kind=kind
            )
            denom = np.sqrt(denom.numpy(force=True))

        dists = np.zeros((nu, nu), dtype=np.float32)

        @delayed
        def dist_job(j, unit):
            d = unit.divergence(
                means, other_covs=covs, other_logdets=logdets, kind=kind
            )
            d = d.numpy(force=True).astype(dists.dtype)
            if noise_normalized:
                d /= denom * denom[j]
            dists[j] = d

        pool = Parallel(
            self.n_threads, backend="threading", return_as="generator_unordered"
        )
        results = pool(dist_job(j, u) for j, u in enumerate(units))
        if show_progress:
            results = tqdm(results, desc="Distances", total=nu, unit="unit", **tqdm_kw)
        for _ in results:
            pass

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
        nu = len(self.units)
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
    ):
        if indices_full is None:
            if unit_id is not None:
                in_u = self.labels == unit_id
            elif unit_ids is not None:
                in_u = torch.isin(self.labels, unit_ids)
            else:
                assert False
            (indices_full,) = torch.nonzero(in_u, as_tuple=True)

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
        verbose=False,
        **unit_args,
    ):
        if features is None:
            features = self.random_spike_data(
                unit_id, indices, max_size=self.n_spikes_fit, with_neighborhood_ids=True
            )
        if verbose:
            print(f"{unit_id=} {features=}")
        if weights is None and likelihoods is not None:
            weights = self.get_fit_weights(unit_id, features.indices, likelihoods)
        if verbose and weights is not None:
            print(f"{weights.sum()=} {weights.min()=} {weights.max()=}")
        unit_args = self.unit_args | unit_args
        if len(self.units) > unit_id:
            unit = self.units[unit_id]
            assert unit.annotations.get(unit_id, unit_id) == unit_id
            unit.fit(features, weights, neighborhoods=self.data.extract_neighborhoods)
        else:
            unit = GaussianUnit.from_features(
                features,
                weights,
                neighborhoods=self.data.extract_neighborhoods,
                unit_id=unit_id,
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
            unit = self.units[unit_id]
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
                neighbs, ns = self.data.core_neighborhoods.subset_neighborhoods(
                    core_channels
                )
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

        jobs = neighbs.items()
        if show_progress:
            jobs = tqdm(
                jobs, desc=f"{desc_prefix}logliks", total=len(neighbs), **tqdm_kw
            )

        for neighb_id, (neighb_chans, neighb_member_ixs) in jobs:
            if spikes is not None:
                sp = spikes[neighb_member_ixs]
            elif inds_already:
                sp = self.data.spike_data(
                    spike_indices[neighb_member_ixs],
                    with_channels=False,
                    neighborhood="core",
                )
            else:
                sp = self.data.spike_data(
                    neighb_member_ixs, with_channels=False, neighborhood="core"
                )
            features = sp.features
            chans_valid = neighb_chans < self.data.n_channels
            features = features[..., chans_valid]
            neighb_chans = neighb_chans[chans_valid]
            lls = unit.log_likelihood(features, neighb_chans, neighborhood_id=neighb_id)

            if inds_already:
                log_likelihoods[neighb_member_ixs.to(log_likelihoods.device)] = lls
            else:
                spike_indices[offset : offset + len(sp)] = neighb_member_ixs
                log_likelihoods[offset : offset + len(sp)] = lls
                offset += len(sp)

        if not inds_already:
            spike_indices, order = spike_indices.sort()
            log_likelihoods = log_likelihoods[order]

        return spike_indices, log_likelihoods

    def noise_log_likelihoods(self, show_progress=False):
        if self._noise_log_likelihoods is None:
            self._noise_six, self._noise_log_likelihoods = self.unit_log_likelihoods(
                unit=self.noise_unit, show_progress=show_progress, desc_prefix="Noise "
            )
        return self._noise_six, self._noise_log_likelihoods

    def kmeans_split_unit(self, unit_id, debug=False):
        # get spike data and use interpolation to fill it out to the
        # unit's channel set
        unit = self.units[unit_id]
        if not unit.channels.numel():
            return {} if debug else None

        indices_full, sp = self.random_spike_data(
            unit_id, return_full_indices=True, with_neighborhood_ids=True
        )
        if not indices_full.numel() > self.min_count:
            return {} if debug else None

        Xo = X = self.data.interp_to_chans(sp, unit.channels)
        if self.split_whiten:
            X = self.noise.whiten(X, channels=unit.channels)

        if debug:
            debug_info = dict(indices_full=indices_full, sp=sp, X=Xo, Xw=X)
        else:
            del Xo
            debug_info = None

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
            debug_info["split_labels"] = split_labels
            debug_info["responsibilities"] = responsibilities
        if split_labels.unique().numel() <= 1:
            return debug_info

        # avoid oversplitting by doing a mini merge here
        split_labels = self.mini_merge(
            sp,
            split_labels,
            weights=responsibilities,
            debug=debug,
            debug_info=debug_info,
        )
        split_ids, split_counts = np.unique(split_labels, return_counts=True)
        valid = split_ids >= 0
        if not valid.any():
            return debug_info
        split_ids = split_ids[valid]
        assert np.array_equal(split_ids, np.arange(len(split_ids)))

        if debug:
            debug_info["merge_labels"] = split_labels
            return debug_info

        split_labels = torch.asarray(split_labels, device=self.labels.device)
        n_new_units = split_ids.size - 1
        if n_new_units < 1:
            # quick case
            with self.labels_lock:
                self.labels[indices_full] = -1
                self.labels[sp.indices[split_labels >= 0]] = unit_id
            return

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

    def mini_merge(
        self,
        spike_data,
        labels,
        weights=None,
        debug=False,
        debug_info=None,
        n_em_iter=None,
    ):
        """Given labels for a small bag of data, fit and merge."""
        if n_em_iter is None:
            n_em_iter = self.split_em_iter

        # E/M sub-units
        for _ in range(n_em_iter):
            units = []
            for label in labels.unique():
                (in_label,) = torch.nonzero(labels == label, as_tuple=True)
                w = None if weights is None else weights[in_label, label]
                features = spike_data[in_label.to(spike_data.indices.device)]
                unit = GaussianUnit.from_features(
                    features,
                    weights=w,
                    neighborhoods=self.data.extract_neighborhoods,
                    **self.unit_args,
                )
                units.append(unit)

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
            weights = F.softmax(lls.T, dim=1)

        labels = labels.numpy(force=True)
        ids = np.unique(labels)
        valid = ids >= 0
        ids = ids[valid]
        if ids.size <= 1:
            return labels

        bimodalities = bimodalities_dense(
            lls.numpy(force=True),
            labels,
            ids=np.arange(len(units)),
            cut=self.merge_bimodality_cut,
            min_overlap=self.merge_bimodality_overlap,
            score_kind=self.merge_bimodality_score_kind,
        )

        # determine their distances
        distances = self.distances(units=units, show_progress=False)

        # return merged labels
        distances = combine_distances(
            distances,
            self.merge_distance_threshold,
            bimodalities,
            self.split_bimodality_threshold,
            sym_function=self.merge_sym_function,
        )
        new_labels, new_ids = agglomerate(
            labels,
            distances,
            linkage_method="complete",
        )
        if debug:
            debug_info["reas_labels"] = labels
            debug_info["units"] = units
            debug_info["lls"] = lls
            debug_info["bimodalities"] = bimodalities
            debug_info["distances"] = distances

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

    def unit_group_criterion(
        self,
        unit_ids,
        likelihoods=None,
        spikes_per_subunit=2048,
        fit_type="refit_all",
        debug=False,
    ):
        """See if a single unit explains a group as far as AIC/BIC/MDL go."""
        assert fit_type in ("avg_preexisting", "refit_all")
        unit_ids = torch.tensor(unit_ids)

        # pick spikes for likelihood computation
        in_subunits = [
            self.random_indices(u, max_size=spikes_per_subunit) for u in unit_ids
        ]
        in_any = torch.cat(in_subunits)
        in_any, in_order = torch.sort(in_any)
        spikes_extract = self.data.spike_data(in_any)
        spikes_core = self.data.spike_data(in_any, neighborhood="core")
        n = in_any.numel()

        if fit_type == "refit_all":
            units = []
            subunit_logliks = spikes_core.features.new_full((len(unit_ids), len(in_any)), -torch.inf)
            full_loglik = 0.0
            for i, k in enumerate(unit_ids):
                u = self.fit_unit(unit_id=k, indices=in_any, likelihoods=likelihoods, features=spikes_extract)
                units.append(u)
                _, subunit_logliks[i] = self.unit_log_likelihoods(unit=u, spikes=spikes_core)
            subunit_log_props = F.softmax(subunit_logliks, dim=0).mean(1).log_()
            # loglik per spik
            full_loglik = torch.logsumexp(subunit_logliks.T + subunit_log_props, dim=1).mean()
            unit = self.fit_unit(indices=in_any, features=spikes_extract)
            likelihoods = None
        elif fit_type == "avg_preexisting":
            unit = self.units[unit_ids[0]].avg_with(*[self.units[u] for u in unit_ids[1:]])
            if debug:
                subunit_logliks = likelihoods[:, in_any][unit_ids]
            full_loglik = marginal_loglik(
                indices=in_any.numpy(force=True),
                log_proportions=self.log_proportions,
                log_likelihoods=likelihoods,
                unit_ids=unit_ids,
            )
        else:
            assert False

        # extract likelihoods... no proportions!
        _, unit_logliks = self.unit_log_likelihoods(unit=unit, spikes=spikes_core)
        unit_loglik = unit_logliks.mean()

        # parameter counting... since we use marginal likelihoods, I'm restricting
        # the parameter counts to just the marginal set considered for each spike.
        # then, aic and bic formulas are changed slightly below to match.
        nids = self.data.core_neighborhoods.neighborhood_ids[in_any]
        unique_nids, inverse = torch.unique(nids, return_inverse=True)
        unique_chans = self.data.core_neighborhoods.neighborhoods[unique_nids]
        unique_k_merged = unit.n_params(unique_chans)
        unique_k_full = [self.units[u].n_params(unique_chans) for u in unit_ids]
        unique_k_full = torch.stack(unique_k_full, dim=1)
        k_merged = unique_k_merged[inverse]
        k_full = unique_k_full[inverse]

        # for aic: k is avg
        k_merged_avg = k_merged.sum() / n
        k_full_avg = k_full.sum() / n
        if self.use_proportions:
            k_full_avg += len(unit_ids) - 1

        # compute some criteria
        # actually computing AIC/BIC per example (divide by N)
        # logliks here are already mean log liks.
        aic_full = (2 * k_full_avg) / n - 2 * full_loglik
        aic_merged = (2 * k_merged_avg) / n - 2 * unit_loglik
        bic_full = (k_full_avg * np.log(n)) / n - 2 * full_loglik
        bic_merged = (k_merged_avg * np.log(n)) / n - 2 * unit_loglik
        res = dict(
            aic_full=aic_full,
            aic_merged=aic_merged,
            bic_full=bic_full,
            bic_merged=bic_merged,
        )
        if debug:
            debug_info = dict(
                full_loglik=full_loglik,
                unit_loglik=unit_loglik,
                unit_logliks=unit_logliks,
                subunit_logliks=subunit_logliks,
                indices=in_any,
                unit=unit,
            )
            res.update(debug_info)
        return res

    def get_log_likelihoods(
        self,
        indices,
        likelihoods,
        with_proportions=True,
        unit_ids=None,
        to_dense=False,
    ):
        # torch's index_select is painfully slow
        # weights = torch.index_select(likelihoods, 1, features.indices)
        # here we have weights as a csc_array
        liks = likelihoods[:, indices]
        liks = coo_to_torch(liks.tocoo(), torch.float, copy_data=True)
        liks = liks.to(self.data.device)
        if with_proportions and self.log_proportions is not None:
            log_props_vec = self.log_proportions[liks.indices()[0]]
            liks.values().add_(log_props_vec)
        if unit_ids is None:
            return liks
        return liks[unit_ids]

    def get_fit_weights(self, unit_id, indices, likelihoods=None):
        """Normalized responsibilities for subset of spikes."""
        if likelihoods is None:
            return None

        weights = self.get_log_likelihoods(indices, likelihoods)
        weights = torch.sparse.softmax(weights, dim=0)
        weights = weights[unit_id].to_dense()
        return weights

    # -- gizmos

    @property
    def rg(self):
        # thread-local rgs since they aren't safe
        if not hasattr(self.storage, "rg"):
            with self.lock:
                self.storage.rg = self._rg.spawn(1)[0]
        return self.storage.rg

    def relabel_units(self, old_labels, new_labels=None, flat=False):
        """Re-label units

        !! This could invalidate self.units and props. It's left to the caller
        to know when to del self.units[:] or to manually update that
        list after calling this.

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

    def stack_units(self, units=None, mean_only=True, use_cache=False):
        if units is None:
            units = self.units
        if use_cache and self._stack is not None:
            if mean_only or self._stack[1] is not None:
                return self._stack

        nu, rank, nc = len(units), self.data.rank, self.data.n_channels

        means = torch.zeros((nu, rank, nc), device=self.data.device)
        covs = logdets = None
        if not mean_only:
            covs = means.new_zeros((nu, rank * nc, rank * nc))
            logdets = means.new_zeros((nu,))

        for j, unit in enumerate(units):
            means[j] = unit.mean
            if covs is not None:
                covs[j] = unit.dense_cov()
                logdets[j] = unit.logdet

        if use_cache:
            self._stack = means, covs, logdets

        return means, covs, logdets


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
        channels_strategy="snr",
        channels_strategy_snr_min=50.0,
        prior_pseudocount=10,
        scale_mean: float = 0.1,
        mean=None,
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
        self.channels_strategy_snr_min = channels_strategy_snr_min
        self.cov_kind = cov_kind
        self.scale_mean = scale_mean
        self.scale_alpha = float(prior_pseudocount)
        self.scale_beta = float(prior_pseudocount) / scale_mean
        self.annotations = annotations
        if mean is not None:
            self.register_buffer("mean", mean)

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
        channels_strategy="snr",
        channels_strategy_snr_min=50.0,
        prior_pseudocount=10,
        scale_mean: float = 0.1,
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
            channels_strategy_snr_min=channels_strategy_snr_min,
            scale_mean=scale_mean,
        )
        self.fit(features, weights, neighborhoods=neighborhoods)
        self = self.to(features.features.device)
        return self

    def avg_with(self, *others):
        new = self.__class__(
            mean_kind=self.mean_kind,
            cov_kind=self.cov_kind,
            rank=self.rank,
            n_channels=self.n_channels,
            noise=self.noise,
        )
        new.register_buffer("mean", (self.mean + sum(o.mean for o in others)) / (1 + len(others)))
        new.register_buffer("channels", torch.cat([self.channels, *[o.channels for o in others]]).unique())
        assert self.cov_kind == "zero"
        return new

    def n_params(self, channels=None, on_channels=True):
        p = channels.new_zeros(len(channels))
        if self.mean_kind == "full":
            p += self.rank * torch.isin(channels, self.channels).sum(1)
        # my cov
        assert self.cov_kind == "zero"
        return p

    def fit(self, features: SpikeFeatures, weights: torch.Tensor, neighborhoods=None):
        if features is None:
            self.pick_channels(None, None)
            return
        n = len(features)
        r = self.noise.rank

        achans = occupied_chans(features, self.n_channels, neighborhoods=neighborhoods)
        target_padded = features.features.new_zeros(n, r, achans.numel() + 1)
        if weights is None:
            weights = features.features.new_ones(n)
        afeats, aweights = zero_pad_to_chans(
            features,
            achans,
            self.n_channels,
            weights=weights,
            target_padded=target_padded,
        )
        assert torch.isfinite(afeats).all()
        assert torch.isfinite(aweights).all()
        aweights_sum = aweights.sum(0)

        # assigns self.mean
        self.fit_mean(achans, afeats, aweights, aweights_sum)

        # assigns self.cov, self.logdet
        self.fit_cov(
            features,
            achans,
            aweights,
            aweights_sum,
            neighborhoods=neighborhoods,
            target_padded=target_padded,
        )
        del features  # overwritten

        self.pick_channels(achans, aweights_sum)

    def fit_mean(self, achans, afeats, aweights, aweights_sum) -> SpikeFeatures:
        if self.mean_kind == "zero":
            return

        assert self.mean_kind == "full"
        aweights_norm = aweights / aweights_sum
        am = torch.linalg.vecdot(aweights_norm.unsqueeze(1), afeats, dim=0)

        if self.prior_type == "niw":
            assert self.noise.mean_kind == "zero"
            count_full = self.prior_pseudocount + aweights_sum
            # w0 = self.prior_pseudocount / count_full
            w1 = aweights_sum / count_full
            am = am * w1  # + self.noise.mean_full * w0
        elif self.prior_type == "none":
            pass
        else:
            assert False

        mean_full = am.new_zeros((self.noise.rank, self.noise.n_channels))
        mean_full[:, achans] = am
        self.register_buffer("mean", mean_full)

    def fit_cov(
        self,
        features,
        achans,
        aweights,
        aweights_sum,
        neighborhoods=None,
        target_padded=None,
    ):
        if self.cov_kind == "zero":
            self.logdet = self.noise.logdet
            return

        if self.cov_kind == "scaled_template":
            # todo: is there some issue with centering and weights
            # zeros get filled in, don't want to subtract mean and leave nonzero
            spw, nu = noise_whiten(
                features,
                self.noise,
                neighborhoods,
                mean_full=self.mean,
                with_whitened_means=True,
                in_place=True,
            )
            del features  # overwritten
            wfeats, _ = zero_pad_to_chans(
                spw, achans, self.noise.n_channels, target_padded=target_padded
            )
            spw = replace(spw, features=nu)
            wnu, _ = zero_pad_to_chans(spw, achans, self.noise.n_channels)
            del spw  # overwritten
            self.template_std = template_scale_map(
                wfeats,
                wnu,
                aweights,
                alpha=self.scale_alpha,
                beta=self.scale_beta,
                allow_destroy=True,
            )
            return

        assert False

    def pick_channels(self, active_chans, aweights_sum):
        if self.channels_strategy == "all":
            self.register_buffer("channels", torch.arange(self.n_channels))
            return

        if not active_chans.numel():
            return

        if self.channels_strategy == "snr":
            amp = torch.linalg.vector_norm(self.mean[:, active_chans], dim=0)
            snr = amp * aweights_sum.sqrt().view(-1)
            full_snr = self.mean.new_zeros(self.mean.shape[1])
            full_snr[active_chans] = snr
            self.register_buffer("snr", full_snr)
            strong = snr >= self.channels_strategy_snr_min
            self.register_buffer("channels", active_chans[strong])
            return

        assert False

    def marginal_covariance(self, channels, cache_key=None, device=None):
        ncov = self.noise.marginal_covariance(
            channels, cache_key=cache_key, device=device
        )
        if self.cov_kind == "zero":
            return ncov
        if self.cov_kind == "scaled_template":
            root = self.template_std * self.mean[:, channels].reshape(-1, 1)
            root = operators.LowRankRootLinearOperator(root)
            cov = more_operators.LowRankRootSumLinearOperator(
                root,
                ncov,
            )
            return cov

        assert False

    def dense_cov(self):
        return self.cov().to_dense()

    def log_likelihood(self, features, channels, neighborhood_id=None) -> torch.Tensor:
        """Log likelihood for spike features living on the same channels."""
        mean = self.noise.mean_full[:, channels]
        if self.mean_kind == "full":
            mean = mean + self.mean[:, channels]
        features = features - mean

        cov = self.marginal_covariance(
            channels, cache_key=neighborhood_id, device=features.device
        )

        inv_quad, logdet = cov.inv_quad_logdet(
            features.view(len(features), -1).T,
            logdet=True,
            reduce_inv_quad=False,
        )
        ll = -0.5 * (inv_quad + logdet + log2pi * mean.numel())
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
        if kind == "kl":
            return self.kl_divergence(other_means, other_covs, other_logdets)
        raise ValueError(f"Unknown divergence {kind=}.")

    def noise_metric_divergence(self, other_means):
        dmu = other_means
        if self.mean_kind != "zero":
            dmu = dmu - self.mean
        dmu = dmu.view(len(other_means), -1)
        noise_cov = self.noise.marginal_covariance(device=dmu.device)
        return noise_cov.inv_quad(dmu.T, reduce_inv_quad=False)

    def kl_divergence(self, other_means, other_covs, other_logdets):
        """DKL(others || self)"""
        n = other_means.shape[0]
        dmu = other_means
        if self.mean_kind != "zero":
            dmu = dmu - self.mean

        # compute the inverse quad and self log det terms
        my_cov = self.cov()
        inv_quad, self_logdet = my_cov.inv_quad_logdet(
            dmu.view(n, -1).T, logdet=True, reduce_inv_quad=True
        )
        tr = torch.trace(my_cov.solve(other_covs))
        k = my_cov.shape[0]
        ld = self_logdet - other_logdets
        return 0.5 * (tr + inv_quad - k + ld)


# -- utilities


log2pi = torch.log(torch.tensor(2.0 * torch.pi))
tqdm_kw = dict(smoothing=0, mininterval=1.0 / 24.0)


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


def marginal_loglik(indices, log_proportions, log_likelihoods, unit_ids=None, reduce="mean"):
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
    """
    data_ixs = write_offsets[inds]
    csc_indices[data_ixs] = j
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


def combine_distances(
    dists_a,
    thresh_a,
    dists_b=None,
    thresh_b=None,
    agg_function=np.maximum,
    sym_function=np.minimum,
):
    """Combine two distance matrices and symmetrize them

    They have different reference thresholds, but the result of this function
    has threshold 1.
    """
    if thresh_b is None:
        dists = dists_a / thresh_a
    else:
        dists = agg_function(dists_a / thresh_a, dists_b / thresh_b)
    return sym_function(dists, dists.T)


def bimodalities_dense(
    log_liks,
    labels,
    ids=None,
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
        cut = None  # should just use "auto".
    if ids is None:
        ids = np.unique(labels)
    bimodalities = np.zeros((ids.size, ids.size), dtype=np.float32)
    for i in range(ids.size):
        for j in range(i + 1, ids.size):
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
