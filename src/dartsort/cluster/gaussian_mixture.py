import threading
from dataclasses import replace

import linear_operator
import numpy as np
import torch
from dartsort.util import data_util, noise_util
from joblib import Parallel, delayed

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
        min_firing_rate: float = 0.01,
    ):
        self.original_sorting = sorting
        self.data = data
        self.noise = noise
        self.n_spikes_fit = n_spikes_fit
        self.n_threads = n_threads
        self.min_firing_rate = min_firing_rate

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
        self.rg = np.random.default_rng(random_seed)

    def to_sorting(self):
        labels = np.full_like(self.original_sorting.labels, -1)
        labels[self.data.kept_indices] = self.labels.cpu()
        return replace(self.original_sorting, labels=labels)

    def unit_ids(self):
        uids = torch.unique(self.labels)
        return uids[uids >= 0]

    def cleanup(self):
        pass

    def m_step(self, likelihoods=None):
        del self.units[:]  # no .clear() on ModuleList?
        pool = Parallel(self.n_threads, backend="threading", return_as="generator")
        for unit in pool(
            delayed(self.fit_unit)(j, likelihoods=likelihoods) for j in self.unit_ids()
        ):
            self.units.append(unit)

    def reassign(self):
        """Noise unit last so that rows correspond to unit ids without 1 offset"""

    def merge(self):
        pass

    def split(self):
        pass

    def random_spike_data(
        self,
        unit_id=None,
        indices=None,
        max_size=None,
        neighborhood="extract",
        with_reconstructions=False,
    ):
        if indices is None:
            (indices,) = torch.nonzero(self.labels == unit_id)
            n_in_unit = indices.numel()
            if max_size and n_in_unit > max_size:
                indices = self.rg.choice(n_in_unit, size=max_size, replace=False)

        return self.data.spike_data(
            indices,
            neighborhood=neighborhood,
            with_reconstructions=with_reconstructions,
        )

    def fit_unit(self, unit_id=None, indices=None, likelihoods=None, weights=None):
        features = self.random_spike_data(unit_id, indices, max_size=self.n_spikes_fit)
        if weights is None and likelihoods is not None:
            weights = torch.index_select(likelihoods, 1, indices)
            weights = weights.to(self.data.device).coalesce()
            weights = torch.sparse.softmax(weights, dim=0)
            weights = weights[unit_id].to_dense()
        unit = GaussianUnit.from_features(features, weights, **self.unit_args)
        return unit

    def kmeans_split_unit(self):
        pass


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
        self.fit_mean(features_full, weights_normalized, count_data)
        emp_mean, features_full = self.fit_mean(
            features_full, weights_normalized, count_data
        )
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
            return

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


# -- utilities


log2pi = torch.log(2 * torch.pi)


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
