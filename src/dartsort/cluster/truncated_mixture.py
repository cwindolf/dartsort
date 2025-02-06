import numpy as np
import torch
import threading
from linear_operator import operators
from linear_operator.utils.cholesky import psd_safe_cholesky
import torch.nn.functional as F

from ..util.noise_util import EmbeddedNoise
from .stable_features import SpikeFeatures, SpikeNeighborhoods, StableSpikeDataset
from ._truncated_em_helpers import neighb_lut, TEBatchData, TEBatchResult, _te_batch
from ..util import spiketorch


class SpikeTruncatedMixtureModel(torch.nn.Module):
    def __init__(
        self,
        data: StableSpikeDataset,
        noise: EmbeddedNoise,
        rank: int = 0,
        n_candidates: int = 3,
        n_search: int = 1,
    ):
        """
        Noise unit is label 0. Will have to figure out hard noise.
        Or should noise not really be its own unit? Just stored separately?
        """
        self.data = data
        self.noise = noise
        self.rank = rank
        train_indices, self.train_neighborhoods = self.data.neighborhoods("extract")
        self.n_spikes = train_indices.numel()
        self.processor = TruncatedExpectationProcessor(noise, self.train_neighborhoods)
        self.candidates = CandidateSet(self.n_spikes, n_candidates, n_search)

    def initialize_parameters(self, train_labels):
        # run ppcas, or svds?
        # compute noise log liks? or does TEP want to do that? or, perhaps we do
        # that by setting noise to the explore unit in the first iteration, and
        # then storing?
        pass

    def em_step(self):
        unit_neighborhood_counts = self.propose_new_candidates()
        self.processor.update(self.means, self.bases, unit_neighborhood_counts)
        # get statistics buffers. maybe they are already intialized
        # parallel gather process() into the buffers. use a helper job function.
        # can process() actually add its results into thread local buffers and combine those
        # for us? maybe the parallel for is handled by that class and this one stays simple

    def kl_divergences(self):
        """Pairwise KL divergences."""
        pass


class TruncatedExpectationProcessor(torch.nn.Module):
    def __init__(
        self,
        noise: EmbeddedNoise,
        neighborhoods: SpikeNeighborhoods,
        features: torch.Tensor,
    ):
        # initialize fixed noise-related arrays
        self.initialize_fixed(noise, neighborhoods)
        self.neighborhood_ids = neighborhoods.neighborhood_ids
        self.features = features

        # place to store thread-local work and output arrays
        self._locals = threading.local()

    def initialize_fixed(self, noise, neighborhoods):
        """Neighborhood-dependent precomputed matrices.

        These are:
         - Coo_logdet
            Log determinant of marginal noise covariances
         - Coo_inv
            Inverse of '''
         - Cooinv_Com
            Above, product with observed-missing cov block
         - obs_ix
            Observed channel indices
         - miss_ix
            Missing channel indices
         - nobs
            Number of observed channels
        """
        self.rank = R = noise.rank
        self.nc = nc = noise.n_channels

        # Get Coos
        # Have to do everything as a list. That's what the
        # noise object supports, but also, we don't want to
        # pad with 0s since that would make logdets 0, Chols
        # confusing etc.
        # We will use the neighborhood valid ixs to pad later.
        # Since we cache everything we need, update can avoid
        # having to do lists stuff.
        Coo = [
            noise.marginal_covariance(
                channels=neighborhoods.neighborhood_channels(ni),
                cache_prefix=neighborhoods.name,
                cache_key=ni,
            )
            for ni in range(neighborhoods.n_neighborhoods)
        ]
        Com = [
            noise.marginal_covariance(
                channels_left=neighborhoods.neighborhood_channels(ni),
                channels=neighborhoods.missing_channels(ni),
            ).to_dense()
            for ni in range(neighborhoods.n_neighborhoods)
        ]

        # Get choleskys. linear_operator will jitter to help
        # with numerics for us if we do everything via chol.
        Coo = [operators.CholLinearOperator(C.cholesky()) for C in Coo]
        Coo_logdet = torch.tensor([C.logdet() for C in Coo])
        Coo_inv = [C.inverse().to_dense() for C in Coo]
        Cooinv_Com = [Co @ Cm for Co, Cm in zip(Coo, Com)]

        # figure out dimensions
        nc_obs = neighborhoods.channel_counts
        self.nc_obs = nco = nc_obs.max().to(int)
        self.nc_miss = ncm = nc - nc_obs.min().to(int)
        nobs = R * nc_obs

        # understand channel subsets
        # these arrays are used to scatter into D-shaped dims.
        # actually... maybe into D+1-shaped dims, so that we can use
        # D as the invalid indicator
        self.register_buffer("obs_ix", neighborhoods.neighborhoods)
        self.register_buffer(
            "miss_ix",
            torch.full((neighborhoods.n_neighborhoods, ncm), fill_value=nc),
        )
        for ni in range(neighborhoods.n_neighborhoods):
            neighb_missing = neighborhoods.missing_channels(ni)
            self.miss_ix[: neighb_missing.numel()] = neighb_missing

        # allocate buffers
        self.register_buffer("Coo_logdet", Coo_logdet)
        self.register_buffer("nobs", nobs)
        self.register_buffer(
            "Coo_inv",
            torch.zeros(neighborhoods.n_neighborhoods, R * nco, R * nco),
        )
        self.register_buffer(
            "Cooinv_Com",
            torch.zeros(neighborhoods.n_neighborhoods, R * nco, R * ncm),
        )

        # loop to fill relevant channels of zero padded buffers
        # for observed channels, the valid subset for each neighborhood
        # (neighborhoods.valid_mask) is exactly where we need to put
        # the matrices so that they hit the right parts of the features
        # for missing channels, we're free to do it in any consistent
        # way. things just need to align betweein miss_ix and Coinv_Com.
        for ni in range(neighborhoods.n_neighborhoods):
            oi = neighborhoods.valid_mask(ni)
            (mi,) = (self.miss_ix[ni] < nc).nonzero(as_tuple=True)
            self.Coo_inv[ni].view(R, nco, R, nco)[:, oi, :, oi] = Coo_inv[ni]
            self.Coinv_Com[ni].view(R, nco, R, ncm)[:, oi, :, mi] = Cooinv_Com[ni]

    def update(
        self,
        log_proportions,
        means,
        noise_log_prop,
        bases=None,
        unit_neighborhood_counts=None,
    ):
        Nu, r, Nc = means.shape
        assert Nc in (self.nc, self.nc + 1)
        already_padded = Nc == self.Nc + 1
        assert log_proportions.shape == (Nu,)
        self.M = 0
        if bases is not None:
            Nu_, d, self.M = bases.shape
            assert Nu_ == Nu
            assert d == self.rank * (self.Nc + already_padded)

        if unit_neighborhood_counts is not None:
            # update lookup table and re-initialize storage
            res = neighb_lut(unit_neighborhood_counts)
            self.lut, self.lut_units, self.lut_neighbs = res
        nlut = len(self.lut)

        # observed parts of W...
        mu = means
        W = bases
        if not already_padded:
            mu = F.pad(mu, (0, 1))
            if self.M:
                assert W is not None
                W = F.pad(W, (0, 1))

        obs_ix = self.obs_ix[self.lut_neighbs]
        miss_ix = self.miss_ix[self.lut_neighbs]

        # proportions stuff
        self.register_buffer("noise_log_prop", noise_log_prop)
        self.register_buffer("log_proportions", log_proportions)

        # mean stuff
        nu = mu[self.lut_units[:, None], :, obs_ix]
        assert nu.shape == (nlut, r, self.nc_obs)
        tnu = mu[self.lut_units[:, None], :, miss_ix]
        self.register_buffer("nu", nu)
        self.register_buffer("tnu", tnu)
        assert nu.shape == (nlut, r, self.nc_miss)
        Cooinv = self.Coo_inv[self.lut_neighbs]
        self.register_buffer("Cooinv_nu", torch.bmm(Cooinv, nu.view(nlut, -1, 1)))

        # basis stuff
        if not self.M:
            return
        assert W is not None

        # load basis
        Wobs = W.view(Nu, self.M, r, Nc)[self.lut_units[:, None], :, :, obs_ix]
        assert Wobs.shape == (nlut, self.M, r, self.nc_obs)
        Wobs = Wobs.view(nlut, self.M, -1)
        Wmiss = W.view(Nu, self.M, r, Nc)[self.lut_units[:, None], :, :, miss_ix]
        assert Wmiss.shape == (nlut, self.M, r, self.nc_miss)
        Wmiss = Wmiss.view(nlut, self.M, -1)

        self.register_buffer("Cobsinv_WobsT", torch.bmm(Cooinv, Wobs.mT))

        cap = torch.bmm(Wobs, self.Cobsinv_WobsT)
        cap.diagonal(dim1=-2, dim2=-1).add_(1.0)
        assert cap.shape == (Nu, self.M, self.M)
        cap = operators.CholLinearOperator(psd_safe_cholesky(cap))

        # matrix determinant lemma
        noise_logdets = self.Coo_logdet[self.lut_neighbs]
        cap_logdets = cap.logdet()
        self.register_buffer("obs_logdets", noise_logdets + cap_logdets)

        # this is used in the log lik Woodbury, and it's also the posterior
        # covariance of the ppca embedding (usually I call that T).
        self.register_buffer("inv_cap", cap.inverse().to_dense())

        # gizmo matrix used in a certain part of the "imputation"
        self.register_buffer(
            "W_WCC", Wmiss - Wobs.mT.bmm(self.Cooinv_Com[self.lut_neighbs])
        )

    def te_step(self):
        pass

    def process_batch(
        self,
        batch_indices,
        with_grads=False,
        with_stats=False,
        with_kl=False,
        with_elbo=False,
        with_obs_elbo=False,
    ) -> TEBatchResult:
        bd = self.load_batch(batch_indices=batch_indices)
        return self.te_batch(
            bd=bd,
            with_grads=with_grads,
            with_stats=with_stats,
            with_kl=with_kl,
            with_elbo=with_elbo,
            with_obs_elbo=with_obs_elbo,
        )

    # long method... putting it in another file to keep the flow
    # clear in this one. this method is the one that does all the math.
    te_batch = _te_batch

    def load_batch(
        self,
        batch_indices,
        with_grads=False,
        with_stats=False,
        with_kl=False,
        with_elbo=False,
        with_obs_elbo=False,
    ) -> TEBatchData:
        candidates = self.candidates[batch_indices]
        x = self.features[batch_indices]
        neighborhood_ids = self.neighborhood_ids[batch_indices]

        # todo: index_select into buffers?
        lut_ixs = self.unit_neighb_lut[candidates, neighborhood_ids[:, None]]

        # neighborhood-dependent
        Coo_logdet = self.Coo_logdet[neighborhood_ids]
        Coo_inv = self.Coo_inv[neighborhood_ids]
        Coinv_Com = self.Coinv_Com[neighborhood_ids]
        obs_ix = self.obs_ix[neighborhood_ids]
        miss_ix = self.miss_ix[neighborhood_ids]
        nobs = self.nobs[neighborhood_ids]

        # unit-dependent
        log_proportions = self.log_proportions[candidates]

        # neighborhood + unit-dependent
        nu = self.nu[lut_ixs]
        tnu = self.tnu[lut_ixs]
        Cooinv_nu = self.Cooinv_nu[lut_ixs]
        if self.M:
            obs_logdets = self.obs_logdets[lut_ixs]
            Wobs = self.Wobs[lut_ixs]
            Cobsinv_WobsT = self.Cobsinv_WobsT[lut_ixs]
            W_WCC = self.W_WCC[lut_ixs]
            inv_cap = self.inv_cap[lut_ixs]
        else:
            Wobs = Cobsinv_WobsT = W_WCC = inv_cap = None
            obs_logdets = Coo_logdet[:, None].broadcast_to(candidates.shape)

        # per-spike
        noise_lls = None
        if self.noise_logliks is not None:
            noise_lls = self.noise_logliks[batch_indices]

        return TEBatchData(
            n=len(x),
            #
            x=x,
            candidates=candidates,
            #
            Coo_logdet=Coo_logdet,
            Coo_inv=Coo_inv,
            Coinv_Com=Coinv_Com,
            obs_ix=obs_ix,
            miss_ix=miss_ix,
            nobs=nobs,
            #
            log_proportions=log_proportions,
            #
            Wobs=Wobs,
            nu=nu,
            tnu=tnu,
            Cooinv_WobsT=Cobsinv_WobsT,
            Cooinv_nu=Cooinv_nu,
            obs_logdets=obs_logdets,
            W_WCC=W_WCC,
            inv_cap=inv_cap,
            #
            noise_lls=noise_lls,
        )


class CandidateSet:
    def __init__(
        self,
        neighborhood_ids,
        n_candidates=3,
        n_search=2,
        n_explore=1,
        random_seed=0,
        device=None,
    ):
        """
        Invariant 1: self.candidates[:, :self.n_candidates] are the best units for
        each spike.

        Arguments
        ---------
        n_candidates : int
            Size of K_n
        n_search : int
            Number of nearby units according to KL to consider for each unit
            The total size of self.candidates[n] is
                n_candidates + n_candidates*n_search + n_explore
        """
        self.neighborhood_ids = neighborhood_ids
        self.n_spikes = neighborhood_ids.numel()
        self.n_candidates = n_candidates
        self.n_search = n_search
        self.n_explore = n_explore
        n_total = self.n_candidates * (self.n_search + 1) + self.n_explore

        can_pin = device is not None and device.type == "cuda"
        self.candidates = torch.empty(
            (self.n_spikes, n_total), dtype=torch.long, pin_memory=can_pin
        )
        self.rg = np.random.default_rng(random_seed)

    def initialize_candidates(self, labels, closest_neighbors):
        """Imposes invariant 1, or at least tries to start off well."""
        torch.index_select(
            closest_neighbors,
            dim=0,
            index=labels,
            out=self.candidates[:, : self.n_candidates],
        )

    def propose_candidates(self, unit_search_neighbors, neighborhood_explore_units):
        """Assumes invariant 1 and does not mess it up.

        Arguments
        ---------
        unit_search_neighbors: LongTensor (n_units, n_search)
        neighborhood_explore_units: LongTensor (n_neighborhoods, unbounded)
            Filled with n_units where irrelevant.
        """
        C = self.n_candidates
        n_search = C * self.n_search
        n_neighbs = len(neighborhood_explore_units)
        n_units = len(unit_search_neighbors)

        top = self.candidates[:, :C]

        # write the search units in place
        torch.index_select(
            unit_search_neighbors,
            dim=0,
            index=top,
            out=self.candidates[:, C : C + n_search].view(
                self.n_spikes, C, self.n_search
            ),
        )

        # sample the explore units and then write. how many units per neighborhood?
        n_units_ = torch.tensor(n_units)[None].broadcast_to(1, n_neighbs)
        n_explore = torch.searchsorted(neighborhood_explore_units, n_units_)
        n_explore = n_explore[self.neighborhood_ids]
        targs = self.rg.integers(
            n_explore.numpy()[:, None], size=(self.n_spikes, self.n_explore)
        )
        targs = torch.from_numpy(targs)
        # torch take_along_dim has an out= while np's does not, so use it.
        torch.take_along_dim(
            neighborhood_explore_units,
            targs,
            dim=1,
            out=self.candidates[:, -self.n_explore :],
        )

        # lastly... which neighborhoods are present in which units?
        unit_neighborhood_counts = np.zeros((n_units, n_neighbs), dtype=int)
        np.add.at(
            unit_neighborhood_counts,
            (self.candidates, self.neighborhood_ids[:, None]),
            1,
        )
        return unit_neighborhood_counts
