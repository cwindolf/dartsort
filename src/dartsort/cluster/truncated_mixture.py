import numpy as np
import torch
import threading

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

    def update(self, new_means, new_bases, unit_neighborhood_counts=None):
        if unit_neighborhood_counts is not None:
            # update lookup table and re-initialize storage
            self.unit_neighb_lut = neighb_lut(unit_neighborhood_counts)

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
        Wobs = self.Wobs[lut_ixs]
        Cooinv_nu = self.Cooinv_nu[lut_ixs]
        obs_logdets = self.obs_logdets[lut_ixs]
        Cobsinv_WobsT = self.Cobsinv_WobsT[lut_ixs]
        T = self.T[lut_ixs]
        W_WCC = self.W_WCC[lut_ixs]
        inv_cap = self.inv_cap

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
            T=T,
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
