import threading
from dataclasses import replace

import numba
import numpy as np
import torch
from dartsort.util import data_util, noise_util
from joblib import Parallel, delayed
from scipy.sparse import coo_array
from tqdm.auto import tqdm, trange

from .cluster_util import agglomerate
from .kmeans import kmeans
from .modes import smoothed_dipscore_at
from .stable_features import SpikeFeatures, StableSpikeDataset

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
        sorting: data_util.DARTsortSorting,
        data: StableSpikeDataset,
        noise: noise_util.EmbeddedNoise,
        n_spikes_fit: int = 4096,
        mean_kind="full",
        cov_kind="zero",
        prior_type="niw",
        channels_strategy="snr",
        channels_strategy_snr_min=50.0,
        prior_pseudocount=10,
        random_seed: int = 0,
        n_threads: int = 4,
        min_count: int = 50,
        n_em_iters: int = 25,
        kmeans_k: int = 5,
        kmeans_n_iter: int = 100,
        kmeans_drop_prop: float = 0.025,
        kmeans_with_proportions: bool = False,
        kmeans_kmeanspp_initial: str = "mean",
        distance_metric: str = "noise_metric",
        distance_noise_normalized: bool = True,
        merge_distance_threshold: float = 1.0,
        merge_bimodality_threshold: float = 0.1,
        merge_sym_function: callable = np.minimum,
    ):
        self.original_sorting = sorting
        self.data = data
        self.noise = noise
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
        self.merge_sym_function = merge_sym_function

        # store on cpu
        self.labels = sorting.labels[data.kept_indices]
        # self.register_buffer("labels", torch.from_numpy(labels))

        self.unit_args = dict(
            rank=data.rank,
            n_channels=data.n_channels,
            noise=noise,
            mean_kind=mean_kind,
            cov_kind=cov_kind,
            prior_type=prior_type,
            channels_strategy=channels_strategy,
            channels_strategy_snr_min=channels_strategy_snr_min,
            prior_pseudocount=prior_pseudocount,
        )
        noise_args = self.unit_args | dict(mean_kind="zero", cov_kind="zero")
        self.noise_unit = GaussianUnit(**noise_args)
        self.units = torch.nn.ModuleList()
        self._rg = np.random.default_rng(random_seed)

        self._noise_log_likelihoods = None

        # multithreading stuff
        self.labels_lock = threading.Lock()
        self.lock = threading.Lock()
        self.storage = threading.local()

    # -- headliners

    def to_sorting(self):
        labels = np.full_like(self.original_sorting.labels, -1)
        labels[self.data.kept_indices] = self.labels.cpu()
        return replace(self.original_sorting, labels=labels)

    def unit_ids(self):
        uids = torch.unique(self.labels)
        return uids[uids >= 0]

    def em(self, n_iter=None, show_progress=False):
        n_iter = self.n_em_iters if n_iter is None else n_iter
        its = trange(n_iter, desc="EM") if show_progress else range(n_iter)
        for j in its:
            log_liks = self.log_likelihoods()
            self.reassign(log_liks)
            log_liks = self.cleanup(log_liks)
            self.m_step(log_liks)

    def m_step(self, likelihoods=None):
        del self.units[:]  # no .clear() on ModuleList?
        if likelihoods is not None:
            likelihoods = coo_to_torch(likelihoods)
        pool = Parallel(self.n_threads, backend="threading", return_as="generator")
        for unit in pool(
            delayed(self.fit_unit)(j, likelihoods=likelihoods) for j in self.unit_ids()
        ):
            self.units.append(unit)

    def log_likelihoods(self, with_noise_unit=True, use_storage=True):
        """Noise unit last so that rows correspond to unit ids without 1 offset"""
        # determine how much storage space we need by figuring out how many spikes
        # are overlapping with each unit
        neighb_info = []
        ns_total = 0
        for j, unit in enumerate(self.units):
            neighbs, ns_unit = self.data.core_neighborhoods.subset_neighborhoods(
                unit.channels
            )
            neighb_info.append(j, neighbs, ns_unit)
            ns_total += ns_unit

        # add in space for the noise unit
        if with_noise_unit:
            ns_total = ns_total + self.data.n_spikes

        coo_uix, coo_six, coo_data = get_coo_storage(
            ns_total, self.storage, use_storage
        )
        offset = 0
        pool = Parallel(self.n_threads, backend="threading", return_as="generator")
        for unit_id, inds, liks in pool(
            delayed(self.unit_log_likelihoods)(unit_id=j, neighbs=neighbs, ns=ns)
            for j, neighbs, ns in neighb_info
        ):
            nnew = inds.numel()
            coo_uix[offset : offset + nnew] = unit_id
            coo_six[offset : offset + nnew] = inds.numpy(force=True)
            coo_data[offset : offset + nnew] = liks.numpy(force=True)
            offset += nnew

        if with_noise_unit:
            noise_six, noise_ll = self.noise_log_likelihoods()
            coo_uix[offset:] = unit_id + 1
            coo_six[offset:] = noise_six
            coo_data[offset:] = noise_ll

        # log liks as sparse matrix. sparse zeros are not 0 but -inf!!
        log_liks = coo_array(
            (coo_data, (coo_uix, coo_six)),
            shape=(len(self.units) + with_noise_unit, self.data.n_spikes),
        )
        return log_liks

    def reassign(self, log_liks):
        has_noise_unit = log_liks.shape[1] > len(self.units)
        assignments = loglik_reassign(log_liks, has_noise_unit=has_noise_unit)
        self.labels.copy_(assignments)

    def cleanup(self, log_liks=None):
        """Remove too-small units

        Also handles bookkeeping to throw those units out of the sparse
        log_liks array.
        """
        unit_ids, counts = torch.unique(self.labels, return_counts=True)
        counts = counts[unit_ids >= 0]
        unit_ids = unit_ids[unit_ids >= 0]
        big_enough = counts > self.min_count
        if big_enough.all():
            return log_liks

        self.relabel_units(unit_ids[big_enough])
        if log_liks is None:
            return

        has_noise_unit = log_liks.shape[1] > len(self.units)
        if has_noise_unit:
            big_enough = torch.concatenate(
                (big_enough, torch.ones_like(big_enough[:1]))
            )
        log_liks = coo_sparse_mask_rows(log_liks, big_enough.numpy(force=True))
        return log_liks

    def merge(self, log_liks=None):
        pass

    def split(self):
        pass

    def distances(self, kind=None, noise_normalized=None, units=None):
        # default to my settings but allow user to experiment
        if kind is None:
            kind = self.distance_metric
        if noise_normalized is None:
            noise_normalized = self.distance_noise_normalized

        if units is None:
            units = self.units

        # stack unit data into one place
        nu, rank, nc = len(units), self.data.rank, self.data.n_channels
        means = self.noise_unit.mean.new_zeros((nu, rank, nc))
        covs = logdets = None
        if kind in ("kl_divergence",):
            covs = means.new_zeros((nu, rank * nc, rank * nc))
            logdets = means.new_zeros((nu,))
        for j, unit in enumerate(units):
            means[j] = unit.mean
            if covs is not None:
                covs[j] = unit.dense_cov()
                logdets[j] = unit.logdet

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

        pool = Parallel(self.n_threads, backend="threading", return_as="generator")
        for res in pool(dist_job(j, u) for j, u in enumerate(units)):
            pass

        return dists

    # -- helpers

    def random_spike_data(
        self,
        unit_id=None,
        indices=None,
        max_size=None,
        neighborhood="extract",
        with_reconstructions=False,
        return_full_indices=False,
    ):
        if indices is None:
            (indices_full,) = torch.nonzero(self.labels == unit_id)
            n_in_unit = indices_full.numel()
            if max_size and n_in_unit > max_size:
                indices = self.rg.choice(n_in_unit, size=max_size, replace=False)
                indices.sort()
                indices = torch.asarray(indices, device=indices_full.device)
                indices = indices_full[indices]
            else:
                indices = indices_full

        sp = self.data.spike_data(
            indices,
            neighborhood=neighborhood,
            with_reconstructions=with_reconstructions,
        )

        if return_full_indices:
            return indices_full, sp
        return sp

    def fit_unit(
        self, unit_id=None, indices=None, likelihoods=None, weights=None, **unit_args
    ):
        features = self.random_spike_data(unit_id, indices, max_size=self.n_spikes_fit)
        if weights is None and likelihoods is not None:
            weights = torch.index_select(likelihoods, 1, indices)
            weights = weights.to(self.data.device).coalesce()
            weights = torch.sparse.softmax(weights, dim=0)
            weights = weights[unit_id].to_dense()
        unit_args = self.unit_args | unit_args
        unit = GaussianUnit.from_features(features, weights, **unit_args)
        return unit

    def unit_log_likelihoods(
        self,
        unit_id=None,
        unit=None,
        spike_indices=None,
        neighbs=None,
        ns=None,
        show_progress=False,
        desc_prefix="",
    ):
        """
        Returns
        -------
        spike_indices
        log_likelihoods
        """
        if unit is None:
            unit = self.units[unit_id]
        inds_already = spike_indices is not None
        if neighbs is None or ns is None:
            if inds_already:
                neighbs, ns = self.data.core_neighborhoods.spike_neighborhoods(
                    unit.channels, spike_indices
                )
            else:
                neighbs, ns = self.data.core_neighborhoods.subset_neighborhoods(
                    unit.channels
                )
        if not ns:
            return None, None

        if not inds_already:
            spike_indices = torch.empty(ns, dtype=int)
            offset = 0
            log_likelihoods = torch.empty(ns)
        else:
            log_likelihoods = torch.full(ns, -torch.inf)

        jobs = neighbs.items()
        if show_progress:
            jobs = tqdm(
                jobs, desc=f"{desc_prefix}logliks", total=len(neighbs), **tqdm_kw
            )

        for neighb_id, (neighb_chans, neighb_member_ixs) in jobs:
            if inds_already:
                sp = self.data.spike_data(
                    spike_indices[neighb_member_ixs], with_channels=False
                )
            else:
                sp = self.data.spike_data(neighb_member_ixs, with_channels=False)

            features = sp.features
            chans_valid = neighb_chans < self.data.n_channels
            features = features[..., chans_valid]
            neighb_chans = neighb_chans[chans_valid]
            lls = unit.log_likelihoods(features, neighb_chans)

            if inds_already:
                log_likelihoods[neighb_member_ixs] = lls
            else:
                spike_indices[offset : offset + sp.n_spikes] = neighb_member_ixs
                log_likelihoods[offset : offset + sp.n_spikes] = lls
                offset += sp.n_spikes

        if not inds_already:
            spike_indices, order = spike_indices.sort()
            log_likelihoods = log_likelihoods[order]

        return unit_id, spike_indices, log_likelihoods

    def noise_log_likelihoods(self):
        if self._noise_log_likelihoods is None:
            self._noise_six, self._noise_log_likelihoods = self.unit_log_likelihoods(
                unit=self.noise_unit, show_progress=True, desc_prefix="Noise "
            )
        return self._noise_six, self._noise_log_likelihoods

    def kmeans_split_unit(self, unit_id):
        # get spike data and use interpolation to fill it out to the
        # unit's channel set
        unit = self.units[unit_id]
        indices_full, sp = self.random_spike_data(unit_id, return_full_indices=True)
        X = self.data.interp_to_chans(sp, unit)

        # run kmeans with kmeans++ initialization
        split_labels, responsibilities = kmeans(
            X,
            n_iter=self.kmeans_n_iter,
            n_components=self.kmeans_k,
            random_state=self.rg,
            kmeanspp_initial=self.kmeans_kmeanspp_initial,
            with_proportions=self.kmeans_with_proportions,
            drop_prop=self.kmeans_drop_prop,
        )
        if split_labels.unique().numel() <= 1:
            return 0, []
        weights = responsibilities / responsibilities.sum(0)

        # avoid oversplitting by doing a mini merge here
        split_labels, split_units = self.mini_merge(sp, split_labels, weights)

        # tack these new units onto the end
        with self.labels_lock:
            next_label = len(self.units)
            for u in split_units[1:]:
                self.units.append(u)
            split_labels = torch.asarray(split_labels, device=self.labels.device)
            split_labels[split_labels >= 1] += next_label - 1

            self.units[unit_id] = split_units[0]
            self.labels[indices_full] = -1
            self.labels[sp.indices] = split_labels

    def mini_merge(self, spike_data, labels, weights=None):
        """Given labels for a small bag of data, fit and merge."""
        # fit sub-units
        split_ids = labels.unique()
        units = []
        for label in split_ids:
            (in_label,) = torch.nonzero(labels == labels, as_tuple=True)
            weights = None if weights is None else weights[in_label]
            features = spike_data[in_label]
            unit = GaussianUnit.from_features(features, weights, **self.unit_args)
            units.append(unit)

        # determine their distances
        distances = self.distances(units=units)

        # determine their bimodalities while at once mini-reassigning
        lls = spike_data.features.new_full((len(units), len(spike_data)), -torch.inf)
        for j, unit in enumerate(units):
            lls[j] = self.unit_log_likelihoods(
                unit=unit, spike_indices=spike_data.indices
            )
        best_liks, labels = lls.max(dim=0)
        labels[torch.isinf(best_liks)] = -1
        labels = labels.numpy(force=True)
        kept = np.flatnonzero(labels >= 0)
        ids, labels[kept], counts = np.unique(labels[kept])
        ids = ids[counts > 0]
        units = [u for j, u in enumerate(units) if counts[j]]
        bimodalities = bimodalities_dense(
            lls.numpy(force=True)[ids], labels, ids=np.arange(ids.size)
        )

        # return merged labels
        distances = combine_distances(
            distances,
            self.merge_distance_threshold,
            bimodalities,
            self.merge_bimodality_threshold,
            sym_function=self.merge_sym_function,
        )
        new_labels = agglomerate(
            labels,
            distances,
            linkage_method="complete",
            threshold=1.0,
        )
        return new_labels, units

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

        !! This could invalidate self.units. It's left to the caller
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

    @classmethod
    def from_features(
        cls,
        features,
        weights,
        noise,
        mean_kind="full",
        cov_kind="zero",
        prior_type="niw",
        channels_strategy="snr",
        channels_strategy_snr_min=50.0,
        prior_pseudocount=10,
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
        )
        self.fit(features, weights)
        return self

    def fit(self, features: SpikeFeatures, weights):
        features_full, weights_full, count_data, weights_normalized = to_full_probe(
            features, weights, self.n_channels, self.storage
        )
        # assigns self.mean
        self.fit_mean(features_full, weights_normalized, count_data)
        emp_mean, features_full = self.fit_mean(
            features_full, weights_normalized, count_data
        )
        # assigns self.cov, self.logdet
        self.fit_cov(emp_mean, features_full, weights_full, weights_normalized)
        self.pick_channels(count_data)

    def fit_mean(self, features_full, weights_normalized, count_data) -> SpikeFeatures:
        """Fit mean and return centered features on the full probe."""
        if self.mean_kind == "zero":
            return torch.zeros_like(features_full[0]), features_full

        assert self.mean_kind == "full"
        emp_mean = torch.vecdot(weights_normalized, features_full, dim=0)

        if self.prior_type == "niw":
            count_full = self.prior_pseudocount + count_data
            w0 = self.prior_pseudocount / count_full
            w1 = count_data / count_full
            m = emp_mean * w1 + self.noise.mean_full * w0
        elif self.prior_type == "none":
            pass
        else:
            assert False

        self.register_buffer("mean", m)
        features_full.sub_(m)

        return emp_mean, features_full

    def fit_cov(self, *args):
        if self.cov_kind == "zero":
            self.logdet = self.noise.logdet
            return

        assert False

    def dense_cov(self):
        if self.cov_kind == "zero":
            cov = self.noise.marginal_covariance()
            return cov.to_dense()
        assert False

    def pick_channels(self, count_data):
        if self.channels_strategy == "all":
            self.register_buffer("channels", torch.arange(self.n_channels))
        elif self.channels_strategy == "snr":
            snr = torch.linalg.norm(self.mean) * count_data.sqrt()
            (channels,) = torch.nonzero(snr >= self.channels_strategy_snr_min)
            self.register_buffer("channels", channels)
        else:
            assert False

    def log_likelihood(self, features, channels) -> torch.Tensor:
        """Log likelihood for spike features living on the same channels."""
        mean = self.noise.mean_full[:, channels]
        if self.mean_kind == "full":
            mean = mean + self.mean[:, channels]
        features = features - mean

        cov = self.noise.marginal_covariance(channels)
        # self.cov_kind nonzero to be implemented

        inv_quad, logdet = cov.inv_quad_logdet(
            features.view(len(features), -1),
            logdet=True,
            reduce_inv_quad=True,
        )
        ll = -0.5 * (inv_quad + (logdet + log2pi).sum(1))
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
        dmu = other_means - self.mean
        noise_cov = self.noise.marginal_covariance()
        return noise_cov.inv_quad(dmu, reduce_inv_quad=True)

    def kl_divergence(self, other_means, other_covs, other_logdets):
        """DKL(other || self)"""
        n = other_means.shape[0]
        dmu = other_means - self.mean

        # compute the inverse quad and self log det terms
        inv_quad, self_logdet = self.cov.inv_quad_logdet(
            dmu.view(n, -1), logdet=True, reduce_inv_quad=True
        )


# -- utilities


log2pi = torch.log(2.0 * torch.pi)
tqdm_kw = dict(smoothing=0, mininterval=1.0 / 24.0)


def to_full_probe(features, weights, n_channels, storage):
    features_full = get_zeros(features.features, storage, "features_full")
    targ_inds = features.channels[:, None].broadcast_to(features.features.shape)
    features_full.scatter_(2, targ_inds, features.features)
    weights_full = features_full[:, 0, :].isfinite().to(features_full)
    if weights is not None:
        weights_full = weights_full.mul_(weights[:, None])
    features_full = features_full.nan_to_num_()
    count_data = weights_full.sum(0, keepdim=True)
    weights_normalized = weights_full / count_data
    return features_full, weights_full, count_data, weights_normalized


def get_zeros(target, storage, name, shape):
    if storage is None:
        return target.new_zeros(shape)

    buffer = getattr(storage, name, None)
    if buffer is None:
        buffer = target.new_zeros(shape)
        setattr(storage, name, buffer)
    else:
        if any(bs < ts for bs, ts in zip(buffer.shape, shape)):
            buffer = target.new_zeros(shape)
            setattr(storage, name, buffer)
        region = tuple(slice(0, ts) for ts in shape)
        buffer = buffer[region]
        buffer.fill_(0.0)

    return buffer


def get_coo_storage(ns_total, storage, use_storage):
    if not use_storage:
        coo_uix = np.empty(ns_total, dtype=int)
        coo_six = np.empty(ns_total, dtype=int)
        coo_data = np.empty(ns_total, dtype=np.float32)
        return coo_uix, coo_six, coo_data

    if hasattr(storage, "coo_data"):
        if storage.coo_data.size < ns_total:
            del storage.coo_uix
            del storage.coo_six
            del storage.coo_data
    else:
        storage.coo_uix = np.empty(ns_total, dtype=int)
        storage.coo_six = np.empty(ns_total, dtype=int)
        storage.coo_data = np.empty(ns_total, dtype=np.float32)

    return storage.coo_uix, storage.coo_six, storage.coo_data


def coo_to_torch(coo_array, dtype):
    coo = (
        torch.from_numpy(coo_array.coords[0]),
        torch.from_numpy(coo_array.coords[1]),
    )
    res = torch.sparse_coo_tensor(
        torch.row_stack(coo),
        torch.asarray(coo_array.data, dtype=torch.float),
        size=coo_array.shape,
    )
    return res


def loglik_reassign(log_liks, has_noise_unit=False):
    assignments = sparse_reassign(log_liks)
    n_units = log_liks.shape[1] - has_noise_unit
    if has_noise_unit:
        assignments[assignments >= n_units] = -1
    return assignments


def sparse_reassign(liks, match_threshold=None, batch_size=512, return_csc=False):
    """Reassign spikes to units with largest likelihood

    liks is (n_units, n_spikes). This computes the argmax for each column,
    treating sparse 0s as -infs rather than as 0s.

    Turns out that scipy's sparse argmin/max have a slow python inner loop,
    this uses a numba replacement, but I'd like to upstream a cython version.
    """
    if not liks.nnz:
        return np.full(liks.shape[1], -1)

    # csc is needed here for this to be fast
    # trick is to make the largest negative item >1, so that
    # sparse 0s can't be the max
    scores = liks.tocsc()
    nz_lines = np.flatnonzero(np.diff(scores.indptr))
    offset = scores.data.min() - 1
    scores.data -= offset

    # see scipy csc argmin/argmax for reference here. this is just numba-ing
    # a special case of that code which has a python hot loop.
    assignments = np.full(scores.shape[1], -1)
    hot_argmax_loop(assignments, nz_lines, scores.indptr, scores.data, scores.indices)

    if return_csc:
        scores.data += offset
        return scores, assignments
    return assignments


@numba.njit(
    numba.void(
        numba.int64[:], numba.int64[:], numba.int32[:], numba.float32[:], numba.int32[:]
    ),
    error_model="numpy",
    nogil=True,
)
def hot_argmax_loop(assignments, nz_lines, indptr, data, indices):
    for i in nz_lines:
        p = indptr[i]
        q = indptr[i + 1]
        ix = indices[p:q]
        dx = data[p:q]
        assignments[i] = ix[dx.argmax()]


def coo_sparse_mask_rows(coo, keep_mask):
    """Row indexing with a boolean mask."""
    if keep_mask.all():
        return coo

    kept_label_indices = np.flatnonzero(keep_mask.numpy(force=True))
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
    dists_b,
    thresh_b,
    agg_function=np.maximum,
    sym_function=np.minimum,
):
    """Combine two distance matrices and symmetrize them

    They have different reference thresholds, but the result of this function
    has threshold 1.
    """
    dists = agg_function(dists_a / thresh_a, dists_b / thresh_b)
    return sym_function(dists, dists.T)


def bimodalities_dense(
    log_liks, labels, ids=None, cut=None, weighted=True, min_overlap=0.95
):
    """Bimodality scores from dense data

    Given dense arrays of log likelihoods (with -infs) and labels, return a matrix
    of bimodality scores.
    """
    if ids is None:
        ids = np.unique(labels)
    bimodalities = np.zeros((ids.size, ids.size), dtype=np.float32)
    for i in range(ids.size):
        for j in range(1, ids.size):
            ij = np.array([i, j])
            in_pair = np.flatnonzero(np.isin(labels, ij))
            pair_log_liks = log_liks[ij][:, in_pair]
            bimodalities[j, i] = bimodalities[i, j] = qda(
                pair_log_liks[0],
                pair_log_liks[1],
                labels[in_pair] == j,
                cut=cut,
                weighted=weighted,
                min_overlap=min_overlap,
            )
    return bimodalities


def qda(
    log_liks_a,
    log_liks_b,
    in_b,
    cut=None,
    weighted=True,
    min_overlap=0.95,
    score_kind="tv",
):
    # "in b not a"-ness
    diff = log_liks_b - log_liks_a
    keep = np.isfinite(diff)
    if not keep.mean() >= min_overlap:
        return np.inf
    in_b = in_b[keep]
    diff = diff[keep]

    if weighted:
        b_prop = in_b.mean()
        a_prop = 1.0 - b_prop
        sample_weights = np.where(in_b, a_prop / 0.5, b_prop / 0.5)
    else:
        sample_weights = np.ones_like(diff)

    return smoothed_dipscore_at(
        cut,
        diff,
        sample_weights=sample_weights,
        dipscore_only=True,
        score_kind=score_kind,
    )
