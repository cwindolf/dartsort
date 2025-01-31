from typing import Union, Optional
import dataclasses
import numpy as np
import torch
import threading

from ..util.noise_util import EmbeddedNoise
from .stable_features import SpikeFeatures, SpikeNeighborhoods, StableSpikeDataset
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
        unit_candidate_neighborhood_ids = self.propose_new_candidates()
        self.processor.update(self.means, self.bases, unit_candidate_neighborhood_ids)
        # get statistics buffers. maybe they are already intialized
        # parallel gather process() into the buffers. use a helper job function.
        # can process() actually add its results into thread local buffers and combine those
        # for us? maybe the parallel for is handled by that class and this one stays simple

    def kl_divergences(self):
        pass


@dataclasses.dataclass(slots=True, kw_only=True, frozen=True)
class TEBatchResult:
    indices: torch.Tensor
    candidates: torch.Tensor

    batch_elbo: Optional[float] = None
    batch_obs_elbo: Optional[float] = None

    R: Optional[torch.Tensor] = None
    U: Optional[torch.Tensor] = None
    m: Optional[torch.Tensor] = None

    ddW: Optional[torch.Tensor] = None
    ddm: Optional[torch.Tensor] = None


class TruncatedExpectationProcessor(torch.nn.Module):
    def __init__(self, noise: EmbeddedNoise, neighborhoods: SpikeNeighborhoods):
        # store number of neighborhoods

        # initialize fixed noise-related arrays
        self.CmoCooinv = ...

        # place to store thread-local work and output arrays
        self._locals = threading.local()

    def update(self, new_means, new_bases, unit_candidate_neighborhood_ids=None):
        if unit_candidate_neighborhood_ids:
            # update lookup table and re-initialize storage
            ...
        else:
            assert 1  # that i have storage initialized
        pass

    def process_batch(
        self,
        sp: SpikeFeatures,
        candidates: torch.Tensor,
        with_grads=False,
        with_stats=False,
        with_kl=False,
        with_elbo=False,
        with_obs_elbo=False,
    ) -> TEBatchResult:
        assert sp.split_indices is not None
        do = with_grads or with_stats or with_kl or with_elbo or with_obs_elbo
        if not do:  # you okay?
            return TEBatchResult(indices=sp.split_indices, candidates=candidates)

        assert False


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
        unit_neighborhood_counts = np.zeros((n_units, n_neighbs), dtype=np.int32)
        np.add.at(
            unit_neighborhood_counts,
            (self.candidates, self.neighborhood_ids[:, None]),
            1,
        )
        max_overlaps = unit_neighborhood_counts.sum(1).max()
        unit_candidate_neighborhood_ids = np.full((n_units, max_overlaps), n_neighbs)
        for j in range(n_units):
            mine = np.flatnonzero(unit_neighborhood_counts[j])
            unit_candidate_neighborhood_ids[j, : len(mine)] = mine

        return unit_candidate_neighborhood_ids
