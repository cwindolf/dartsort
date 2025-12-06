import time
from itertools import repeat

import numpy as np
import torch
import torch.nn.functional as F
from linear_operator import operators
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import nn
from tqdm.auto import tqdm, trange

from ...util import spiketorch
from ...util.logging_util import DARTSORTVERBOSE, get_logger
from ...util.multiprocessing_util import get_pool
from ...util.noise_util import EmbeddedNoise
from ...util.sparse_util import (
    erase_dups,
    fisher_yates_replace,
    integers_without_inner_replacement,
)
from ._truncated_em_helpers import (
    TEBatchResult,
    TEStepResult,
    _elbo_prior_correction,
    _finalize_missing_full_m,
    _finalize_missing_full_R,
    _grad_basis,
    _grad_counts,
    _grad_mean,
    _processor_update_mean_batch,
    _processor_update_pca_batch,
    _te_batch_e,
    _te_batch_m_counts,
    _te_batch_m_ppca,
    _te_batch_m_rank0,
    missing_indices,
    neighb_lut,
    observed_and_missing_marginals,
    units_overlapping_neighborhoods,
)
from .stable_features import SpikeNeighborhoods, StableSpikeDataset

logger = get_logger(__name__)


class SpikeTruncatedMixtureModel(nn.Module):
    def __init__(
        self,
        data: StableSpikeDataset,
        noise: EmbeddedNoise,
        M: int = 0,
        n_candidates: int = 5,
        n_search: int | None = 3,
        n_explore: int | None = None,
        n_units: int | None = None,
        random_seed=0,
        n_threads: int = 0,
        batch_size=2**7,
        update_batch_size=2**7,
        metric="kl",
        exact_kl=True,
        search_type="topk",
        random_search_max_distance=0.5,
        fixed_noise_proportion=None,
        sgd_batch_size=None,
        Cinv_in_grad=True,
        alpha0=25.0,
        laplace_ard=False,
        alpha_max=1e6,
        alpha_min=1e-6,
        prior_scales_mean=True,
        neighborhood_adjacency_overlap=0.75,
        search_neighborhood_steps=0,
        explore_neighborhood_steps=1,
        noise_log_priors=None,
        min_log_prop=-50.0,
    ):
        super().__init__()

        self.data = data
        self.noise = noise
        self.device = self.data.device
        train_indices, self.train_neighborhoods = self.data.neighborhoods("extract")
        self.train_neighborhoods = self.train_neighborhoods.cpu()

        if laplace_ard:
            assert alpha0 and alpha0 > 0
        self.has_prior = alpha0 and (prior_scales_mean or M)
        self.alpha0 = alpha0
        self.laplace_ard = laplace_ard
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.prior_scales_mean = prior_scales_mean
        self.fixed_noise_proportion = fixed_noise_proportion
        self.exact_kl = exact_kl
        self.sgd_batch_size = sgd_batch_size
        self.min_log_prop = min_log_prop
        self.metric = metric
        self.search_type = search_type
        if search_type == "random":
            assert metric == "cosine"

        self.n_spikes = train_indices.numel()
        self.M = M
        self.data_dim = self.data.rank * self.data.n_channels

        self.candidates = CandidateSet(
            neighborhoods=self.train_neighborhoods,
            random_seed=random_seed,
            device=self.data.device,
            neighborhood_adjacency_overlap=neighborhood_adjacency_overlap,
            search_neighborhood_steps=search_neighborhood_steps,
            explore_neighborhood_steps=explore_neighborhood_steps,
            search_type=search_type,
            random_search_max_distance=random_search_max_distance,
        )

        self.initial_n_candidates = n_candidates
        self.initial_n_search = n_search
        self.initial_n_explore = n_explore
        self.set_sizes(n_units)
        self.processor = TruncatedExpectationProcessor(
            noise=noise,
            noise_log_priors=noise_log_priors,
            neighborhoods=self.train_neighborhoods,
            features=self.data._train_extract_features,
            random_seed=random_seed,
            n_threads=n_threads,
            batch_size=batch_size,
            pgeom=data.prgeom,
            Cinv_in_grad=Cinv_in_grad,
            update_batch_size=update_batch_size,
            device=self.device,
        )
        self.to(device=self.device)
        self._parameters_initialized = False

    def set_sizes(self, n_units: int | None = None):
        n_candidates = self.initial_n_candidates
        n_search = self.initial_n_search
        n_explore = self.initial_n_explore
        if n_units is not None:
            n_candidates = min(n_units, n_candidates)
        if n_search is None:
            n_search = n_candidates
        if n_explore is None:
            if self.search_type == "topk":
                n_explore = n_search
            elif self.search_type == "random":
                n_explore = 0
        if n_units is not None:
            self.n_units = n_units
            remainder = n_units - n_candidates
            n_search = max(0, min(remainder // n_candidates, n_search))
            remainder -= n_candidates * n_search
            n_explore = max(0, min(remainder, n_explore))

        self.n_candidates = n_candidates
        self.n_search = n_search
        self.n_explore = n_explore

        self.candidates.update_sizes(
            n_units, self.n_candidates, self.n_search, self.n_explore
        )

    def clear_parameters(self):
        self._parameters_initialized = False
        nu = self.n_units
        rank_ncp1 = self.noise.rank, self.noise.n_channels + 1

        self.means = nn.Parameter(
            torch.full((nu, *rank_ncp1), torch.nan), requires_grad=False
        )

        if self.M:
            self.bases = nn.Parameter(
                torch.full((nu, self.M, *rank_ncp1), torch.nan), requires_grad=False
            )
        else:
            self.bases = None
        if self.M and self.laplace_ard and self.has_prior:
            self.alpha = nn.Parameter(
                torch.full((nu, self.M), self.alpha0, dtype=torch.float64),
                requires_grad=False,
            )
        else:
            self.alpha = self.alpha0

        self.log_proportions = nn.Parameter(
            torch.full((nu,), 1.0 / nu).log_(), requires_grad=False
        )
        self.register_buffer("noise_log_prop", torch.log(torch.tensor(1.0 / nu)))
        self.register_buffer("_N", torch.zeros(nu + 1))

    def set_parameters(
        self,
        labels,
        means,
        log_proportions,
        noise_log_prop,
        bases=None,
        alpha=None,
        divergences=None,
    ):
        """Parameters are stored padded with an extra channel."""
        self.set_sizes(means.shape[0])
        means = torch.asarray(means, device=self.device)
        log_proportions = torch.asarray(log_proportions, device=self.device)
        if bases is not None:
            bases = torch.asarray(bases, device=self.device)
        if alpha is not None:
            alpha = torch.asarray(alpha, device=self.device)
        if divergences is not None:
            divergences = torch.asarray(divergences, device=self.device)
        assert means.shape == (self.n_units, self.noise.rank, self.noise.n_channels)
        assert log_proportions.shape == (self.n_units,)
        assert means.isfinite().all()
        assert log_proportions.isfinite().all()
        assert noise_log_prop.isfinite()

        self._parameters_initialized = True

        self.means = nn.Parameter(F.pad(means, (0, 1)), requires_grad=False)
        assert self.means.isfinite().all()

        self.register_buffer("_N", torch.zeros(self.n_units + 1))
        if self.fixed_noise_proportion:
            noise_log_prop = torch.log(torch.tensor(self.fixed_noise_proportion))
            inv_noise_log_prop = torch.log(
                torch.tensor(1.0 - self.fixed_noise_proportion)
            )
            self.register_buffer("noise_log_prop", noise_log_prop)
            self.log_proportions = nn.Parameter(
                torch.log_softmax(log_proportions, dim=0) + inv_noise_log_prop,
                requires_grad=False,
            )
        else:
            self.log_proportions = nn.Parameter(
                log_proportions.clone(), requires_grad=False
            )
            self.noise_log_prop = nn.Parameter(
                noise_log_prop.clone(), requires_grad=False
            )
            assert self.log_proportions.isfinite().all()

        if self.M:
            assert bases is not None
            assert bases.shape == (
                self.n_units,
                self.M,
                self.data.rank,
                self.data.n_channels,
            )
            self.bases = nn.Parameter(F.pad(bases, (0, 1)), requires_grad=False)
            assert self.bases.isfinite().all()
        else:
            self.bases = None

        if self.M and self.laplace_ard and self.has_prior:
            assert bases is not None
            if alpha is None:
                alpha = bases.new_full(
                    (self.n_units, self.M), self.alpha0, dtype=torch.float64
                )
            else:
                alpha = torch.asarray(alpha, dtype=torch.float64, device=bases.device)
                alpha = torch.where(alpha.isnan(), self.alpha0, alpha)
                assert alpha.shape == (self.n_units, self.M)
            self.alpha = nn.Parameter(alpha, requires_grad=False)
            assert self.alpha.isfinite().all()
        else:
            self.alpha = self.alpha0

        self.initialize_candidates(labels, divergences)
        self.to(means.device)

    def initialize_candidates(self, labels, divergences=None):
        self.register_buffer(
            "divergences", self.means.new_empty((self.n_units, self.n_units))
        )
        if divergences is None:
            self.update_divergences()
        else:
            self.divergences[:] = divergences.to(self.divergences)
        self.candidates.initialize_candidates(
            labels, self.divergences, fill_blanks=self._parameters_initialized
        )

    def prepare_step(self):
        if self._parameters_initialized:
            candidates, unit_neighborhood_counts = self.candidates.propose_candidates(
                self.divergences
            )
            assert (
                candidates.untyped_storage()
                == self.candidates.candidates.untyped_storage()
            )
            self.processor.update(
                log_proportions=self.log_proportions,
                noise_log_prop=self.noise_log_prop,
                means=self.means,
                bases=self.bases,
                unit_neighborhood_counts=unit_neighborhood_counts,
            )
            candidates = candidates.to(self.device)
            candidates_needs_update = (
                candidates.untyped_storage()
                != self.candidates.candidates.untyped_storage()
            )
        elif self._mean_initialized:
            # M>0 model takes two initialization passes
            candidates = self.candidates.candidates[:, :1]
            # no E step is run, so no reassignment should be done
            candidates_needs_update = False
            self.candidates.reinit_neighborhoods(
                candidates[:, 0],
                self.candidates.neighborhood_ids,
                constrain_searches=False,
            )
            self.processor.update(
                log_proportions=self.log_proportions,
                noise_log_prop=self.noise_log_prop,
                means=self.means,
                bases=None,
                unit_neighborhood_counts=self.candidates.unit_neighborhood_counts,
            )
        else:
            candidates = self.candidates.candidates[:, :1]
            # no E step is run, so no reassignment should be done
            candidates = candidates.to(self.device)
            candidates_needs_update = False
        return candidates, candidates_needs_update

    def step(self, show_progress=False, hard_label=False, with_probs=False, tic=None):
        candidates, candidates_needs_update = self.prepare_step()
        result = self.processor.truncated_e_step(
            candidates=candidates,
            n_candidates=self.n_candidates if self._parameters_initialized else 1,
            show_progress=show_progress,
            with_kl=self.metric == "kl" and not self.exact_kl,
            with_hard_labels=hard_label,
            with_probs=with_probs,
            initializing=not self._parameters_initialized,
        )
        assert result.obs_elbo is not None
        assert result.N is not None
        assert result.m is not None

        if candidates_needs_update:
            _c = candidates[:, : self.n_candidates].to(self.candidates.candidates)
            self.candidates.candidates[:, : self.n_candidates] = _c
            if logger.isEnabledFor(DARTSORTVERBOSE):
                assert (_c[:, 0] >= 0).all()

        if self.has_prior and self.prior_scales_mean:
            mean_scale = result.N / (result.N + self.alpha0)
            self.means[..., :-1] = result.m * mean_scale[:, None, None]
        else:
            self.means[..., :-1] = result.m
        if logger.isEnabledFor(DARTSORTVERBOSE):
            assert self.means[..., :-1].isfinite().all()

        W = None
        if self.bases is not None:
            assert result.R is not None
            assert result.U is not None

            # just to avoid numerical issues when a unit dies
            blank = (result.U.diagonal(dim1=-2, dim2=-1) == 0).all(dim=1)
            result.U.diagonal(dim1=-2, dim2=-1).add_(blank[:, None].to(result.U))

            if self.has_prior:
                N_denom = result.N.clamp(min=1e-5)
                tikh = self.alpha / N_denom.unsqueeze(1)
                result.U.diagonal(dim1=-2, dim2=-1).add_(
                    torch.asarray(tikh, dtype=result.U.dtype, device=result.U.device)
                )

            W = torch.linalg.solve(result.U, result.R.view(*result.U.shape[:-1], -1))
            assert W.shape == (self.n_units, self.M, self.data_dim)
            self.bases[..., :-1] = W.view(self.bases[..., :-1].shape)
            if logger.isEnabledFor(DARTSORTVERBOSE):
                assert self.bases[..., :-1].isfinite().all()

        nc = None
        if self.has_prior and self.laplace_ard and self.M:
            assert W is not None
            assert isinstance(self.alpha, nn.Parameter)
            assert result.R is not None

            nc = (result.R[:, :, 0] != 0).sum(dim=2).clamp_(min=1)
            denom = W.to(torch.float64, copy=True).square_().sum(dim=2).div_(nc)
            assert denom.shape == self.alpha.shape
            amin = self.alpha_min * result.N.clamp(min=1.0).unsqueeze(1)
            amax = self.alpha_max * result.N.clamp(min=1.0).unsqueeze(1)
            self.alpha[:] = (1.0 / denom).clamp_(amin, amax)
            if logger.isEnabledFor(DARTSORTVERBOSE):
                assert self.alpha.isfinite().all()

        assert result.N.min() >= 0
        if self.fixed_noise_proportion:
            noise_log_prop = torch.log(torch.tensor(self.fixed_noise_proportion))
            inv_noise_log_prop = torch.log(
                torch.tensor(1.0 - self.fixed_noise_proportion)
            )
            self.noise_log_prop.fill_(noise_log_prop)
            self.log_proportions[:] = (
                torch.log_softmax(result.N.log(), dim=0) + inv_noise_log_prop
            )
        else:
            assert result.noise_N is not None
            assert result.N is not None
            self._N[0] = result.noise_N
            self._N[1:] = result.N
            lp = torch.log_softmax(self._N.log(), dim=0).clamp_(min=self.min_log_prop)
            self.noise_log_prop.fill_(lp[0])
            self.log_proportions[:] = lp[1:]

        if self.metric == "kl" and not self.exact_kl:
            assert result.kl is not None
            self.distances[:] = result.kl
        else:
            self.update_divergences()

        obs_elbo = result.obs_elbo
        if self.has_prior:
            epc = _elbo_prior_correction(
                alpha0=self.alpha0,
                total_count=result.count,
                nc=nc,
                mu=self.means[..., :-1].reshape(self.n_units, -1),
                W=W,
                Cinv=self.noise.full_inverse(),
                alpha=self.alpha if self.laplace_ard and self.M else None,
                mean_prior=self.prior_scales_mean,
            )
            obs_elbo += epc

        result = dict(
            obs_elbo=obs_elbo.numpy(force=True).item(),
            noise_lp=self.noise_log_prop.numpy(force=True).copy(),
            labels=result.hard_labels,
            probs=result.probs,
        )
        self._parameters_initialized = True
        if tic is not None:
            result["wall"] = time.perf_counter() - tic
        return result

    def sgd_epoch(
        self,
        opt,
        show_progress=False,
        tic=None,
    ):
        # things that don't change...
        search_neighbors = self.candidates.search_sets(self.distances)

        # metrics
        count = 0
        records = []
        dev = self.means.device
        obs_elbo = torch.tensor(0.0, device=dev, dtype=torch.double)

        for batch in self.processor.batches(shuffle=True, show_progress=show_progress):
            opt.zero_grad()
            batch_result = self.sgd_batch(batch, search_neighbors)
            self.set_grads(
                ddlogpi=batch_result.ddlogpi,
                ddlognoisep=batch_result.ddlognoisep,
                ddm=batch_result.ddm,
                ddW=batch_result.ddW,
            )
            opt.step()
            self.project_noise_prop()

            # update candidate set for batch spikes
            self.candidates.candidates[batch, : self.candidates.n_candidates] = (
                batch_result.candidates
            )

            # metrics
            boelbo = batch_result.obs_elbo
            if not boelbo.isfinite():
                raise ValueError(f"batch elbo diverged! {boelbo=}")
            bn = len(batch_result.candidates)
            record = dict(obs_elbo=boelbo.numpy(force=True).item())
            if tic is not None:
                record["wall"] = time.perf_counter() - tic
            records.append(record)
            count += bn
            obs_elbo += (boelbo - obs_elbo) * (bn / count)
            if not obs_elbo.isfinite():
                raise ValueError(
                    f"running elbo diverged! {boelbo=} {count=} {bn=} {obs_elbo=}"
                )

        # per-epoch updates
        self.update_divergences()

        result = dict(
            obs_elbo=obs_elbo.numpy(force=True).item(),
            noise_lp=self.noise_log_prop.numpy(force=True).copy(),
            train_records=records,
        )
        if tic is not None:
            result["wall"] = time.perf_counter() - tic
        return result

    def sgd_batch(self, batch_indices, search_neighbors):
        # mini-update of search sets and local parameters
        batch_candidates, unit_neighborhood_counts = self.candidates.propose_candidates(
            self.distances, indices=batch_indices, constrain_searches=False
        )
        self.processor.update(
            log_proportions=self.log_proportions.clone().detach(),
            noise_log_prop=self.noise_log_prop.clone().detach(),
            means=self.means.clone().detach(),
            bases=self.bases.clone().detach() if self.bases is not None else None,
            unit_neighborhood_counts=unit_neighborhood_counts,
        )

        # get the batch elbo and gradients
        result = self.processor.process_batch(
            batch_indices=batch_indices,
            candidates=batch_candidates,
            with_obs_elbo=True,
            with_grads=True,
        )
        return result

    def set_grads(self, ddlogpi, ddlognoisep, ddm, ddW):
        self.log_proportions.grad = ddlogpi
        if not self.fixed_noise_proportion:
            self.noise_log_prop.grad = ddlognoisep
        ddm = ddm.view(*self.means.shape[:2], -1)
        self.means.grad = F.pad(ddm, (0, 1))
        if self.M:
            assert self.bases is not None
            self.bases.grad = F.pad(ddW, (0, 1))

    def project_noise_prop(self):
        if self.fixed_noise_proportion:
            noise_log_prop = torch.log(torch.tensor(self.fixed_noise_proportion))
            inv_noise_log_prop = torch.log(
                torch.tensor(1.0 - self.fixed_noise_proportion)
            )
            self.noise_log_prop.fill_(noise_log_prop)
            self.log_proportions[:] = (
                torch.log_softmax(self.log_proportions, dim=0) + inv_noise_log_prop
            )
        else:
            self._N[0] = self.noise_log_prop
            self._N[1:] = self.log_proportions
            lp = torch.log_softmax(self._N, dim=0)
            self.noise_log_prop.fill_(lp[0])
            self.log_proportions[:] = lp[1:]

    def update_divergences(self):
        if not self._parameters_initialized:
            self.divergences.fill_(torch.inf)
            self.divergences.diagonal(dim1=-2, dim2=-1).fill_(0.0)
            return

        if self.metric == "kl":
            W = self.bases
            if W is not None:
                W = W[..., :-1].reshape(len(W), self.M, -1).mT
            spiketorch.woodbury_kl_divergence(
                C=self.noise._full_cov,
                mu=self.means[..., :-1].reshape(len(self.means), -1),
                W=W,
                out=self.divergences,
            )
        elif self.metric == "cosine":
            self.divergences[:] = spiketorch.cosine_distance(self.means[..., :-1])
            self.divergences[:].sqrt_()
        else:
            assert False

    def channel_occupancy(
        self,
        labels,
        min_count=1,
        min_prop=0,
        count_per_unit=None,
        neighborhoods=None,
        rg=0,
    ):
        if neighborhoods is None:
            neighborhoods = self.train_neighborhoods
        shp = self.n_units, neighborhoods.n_neighborhoods
        unit_neighborhood_counts = np.zeros(shp, dtype=np.int64)

        labels = labels.numpy(force=True).copy()
        if count_per_unit:
            units = np.unique(labels)
            rg = np.random.default_rng(rg)
            for u in units[units >= 0]:
                inu = np.flatnonzero(labels == u)
                if inu.size > count_per_unit:
                    labels[inu] = -1
                    labels[rg.choice(inu, size=count_per_unit, replace=False)] = u

        valid = np.flatnonzero(labels >= 0)
        assert neighborhoods.neighborhood_ids.shape == labels.shape
        vneighbs = neighborhoods.neighborhood_ids[valid].cpu()
        np.add.at(unit_neighborhood_counts, (labels[valid], vneighbs), 1)
        # nu x nneighb
        neighb_occupancy = unit_neighborhood_counts.astype(float)
        # nneighb x nchans
        neighb_to_chans = neighborhoods.indicators.T.numpy(force=True)
        counts = neighb_occupancy @ neighb_to_chans
        props = [row / max(1, row.max()) for row in counts]
        channels = [
            np.flatnonzero(np.logical_and(row >= min_count, prop >= min_prop))
            for row, prop in zip(counts, props)
        ]
        return channels, counts


class TruncatedExpectationProcessor(torch.nn.Module):
    def __init__(
        self,
        noise: EmbeddedNoise,
        neighborhoods: SpikeNeighborhoods,
        features: torch.Tensor,
        noise_log_priors=None,
        batch_size: int = 2**8,
        update_batch_size: int = 2**8,
        n_threads: int = 0,
        random_seed: int = 0,
        precompute_invx=True,
        Cinv_in_grad=True,
        pgeom=None,
        device=None,
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # initialize fixed noise-related arrays
        self.noise = noise
        self.n_spikes = features.shape[0]
        self.nc_obs = features.shape[2]
        self.M = 0
        self.rank = noise.rank
        self.n_channels = noise.n_channels
        self.update_batch_size = update_batch_size

        assert self.nc_obs == neighborhoods.neighborhoods.shape[1]
        assert (self.n_spikes,) == neighborhoods.neighborhood_ids.shape
        # can_pin = device.type == "cuda"
        can_pin = False
        self.features = torch.empty(
            features.shape, dtype=features.dtype, pin_memory=can_pin
        )
        self.features.copy_(features)
        self.features.nan_to_num_()
        self.neighborhoods = neighborhoods
        self.neighborhood_ids = neighborhoods.neighborhood_ids
        self.n_neighborhoods = neighborhoods.n_neighborhoods
        self.Cinv_in_grad = Cinv_in_grad
        self.batch_size = batch_size
        if noise_log_priors is None:
            self.noise_log_priors = None
        else:
            noise_log_priors = torch.asarray(noise_log_priors, dtype=features.dtype)
            self.register_buffer("noise_log_priors", noise_log_priors)

        # M is updated by self.update() when a basis is assigned here.
        # all buffers are initialized here or when update() is called
        # update calls initialize_changing(), which initializes things that
        # depend on the number of units or the LUT size
        self.register_buffer("nobs", self.rank * neighborhoods.channel_counts.long())
        self.register_buffer("obs_ix", neighborhoods.neighborhoods.to(device))
        miss_ix, miss_ix_full, miss_full_masks = missing_indices(
            neighborhoods, zero_radius=noise.zero_radius, pgeom=pgeom, device=device
        )
        self.nc_miss = miss_ix.shape[1]
        self.nc_miss_full = miss_ix_full.shape[1]
        self.register_buffer("miss_ix", miss_ix.to(device))
        self.register_buffer("miss_ix_full", miss_ix_full.to(device))
        self.register_buffer("miss_full_masks", miss_full_masks.to(device))
        self.initialize_fixed(noise, neighborhoods, pgeom=pgeom, device=device)
        self._changing_initialized = False

        if precompute_invx:
            self.precompute_invx()

        # thread pool
        self._rg = np.random.default_rng(random_seed)
        n_jobs, Executor, context = get_pool(n_jobs=n_threads, cls="ThreadPoolExecutor")
        self.pool = Executor(max_workers=n_jobs, mp_context=context)
        self.to(device)

    @property
    def device(self) -> torch.device:
        assert hasattr(self, "Coo_logdet")
        return self.Coo_logdet.device

    def truncated_e_step(
        self,
        candidates,
        n_candidates,
        with_kl=False,
        show_progress=False,
        with_hard_labels=False,
        with_probs=False,
        initializing=False,
    ) -> TEStepResult:
        """E step of truncated VEM

        Will update candidates[:, :self.n_candidates] in place.
        """
        # sufficient statistics
        dev = self.device
        nc = self.n_channels
        mean_shp = (self.n_units, self.rank, self.n_channels)

        obs_elbo = torch.tensor(0.0, device=dev, dtype=torch.double)

        N = torch.zeros((self.n_units,), device=dev, dtype=torch.double)
        Nlut = torch.zeros((self.nlut,), device=dev, dtype=torch.double)
        noise_N = torch.tensor(0.0, device=dev, dtype=torch.double)

        m = torch.zeros(mean_shp, device=dev)
        R = U = Ulut = None
        if self.M:
            r_shp = (self.n_units, self.M, self.rank, self.n_channels)
            R = torch.zeros(r_shp, device=dev)
            u_shp = (self.n_units, self.M, self.M)
            U = torch.zeros(u_shp, device=dev)
            Ulut_shp = (self.nlut, self.M, self.M)
            Ulut = torch.zeros(Ulut_shp, device=dev)

        dkl = None
        if with_kl:
            nu2 = (self.n_units, self.n_units)
            dkl = torch.zeros(nu2, device=dev)
            ncc = dkl.clone()

        # will be updated...
        top_candidates = candidates[:, :n_candidates]
        hard_labels = None
        if with_hard_labels:
            hard_labels = torch.empty_like(top_candidates[:, 0])
        probs = None
        if with_probs:
            probs = torch.empty(candidates[:, :n_candidates].shape)

        # do we need to initialize the noise log likelihoods?
        first_run = not hasattr(self, "noise_logliks")
        noise_logliks = None
        if first_run:
            noise_logliks = torch.empty((self.n_spikes,), device=dev)

        # run loop
        jobs = (
            (
                candidates,
                n_candidates,
                batchix,
                with_kl,
                with_hard_labels,
                with_probs,
                initializing,
            )
            for batchix in self.batches()
        )
        count = 0
        results = self.pool.map(self._te_step_job, jobs)
        if show_progress:
            results = tqdm(
                results,
                total=int(np.ceil(self.n_spikes / self.batch_size)),
                desc="Batches",
            )
        for result in results:
            assert result is not None  # why, pyright??

            noise_N += result.noise_N
            N += result.N
            Nlut += result.Nlut

            N_denom = torch.where(N == 0, 1.0, N)

            # weight for the welford running averages below
            n1_n01 = result.N.div(N_denom)[:, None, None]

            # welford for the mean
            m += result.m[:, :, :nc].sub_(m).mul_(n1_n01)

            # running avg for elbo/n
            count += len(result.candidates)
            obs_elbo += (result.obs_elbo - obs_elbo) * (len(result.candidates) / count)

            if with_kl:
                ncc += result.ncc
                dkl += (result.dkl - dkl).div_(ncc.clamp(min=1.0))

            if self.M:
                assert R is not None
                assert result.R is not None
                assert Ulut is not None
                assert result.Ulut is not None
                R += result.R[..., :nc].sub_(R).mul_(n1_n01[..., None])

                Nlut_denom = torch.where(Nlut == 0, 1.0, Nlut)
                n1_n01_lut = result.Nlut.div_(Nlut_denom)
                Ulut += result.Ulut.sub_(Ulut).mul_(n1_n01_lut[:, None, None])

            # could do these in the threads? unless shuffled.
            top_candidates[result.indices] = result.candidates
            if noise_logliks is not None:
                noise_logliks[result.indices] = result.noise_lls

            if with_hard_labels:
                assert hard_labels is not None
                assert result.hard_labels is not None
                hard_labels[result.indices] = result.hard_labels

            if with_probs:
                assert result.indices is not None
                probs[result.indices] = result.probs

        # some things are more efficiently computed in LUT bins and
        # then reweighted and used later. well, here they are.
        N_denom_lut = N_denom[self.lut_units]
        Nlut_N = Nlut.div_(N_denom_lut)
        del Nlut

        self._finalize_missing_full_m(Nlut_N, m)
        if self.M:
            self._finalize_missing_full_R(Nlut_N, R, Ulut)

            assert U is not None
            assert Ulut is not None
            ix = self.lut_units[:, None, None].broadcast_to(Ulut.shape)
            U.scatter_add_(dim=0, index=ix, src=Ulut.mul_(Nlut_N[:, None, None]))

        if first_run:
            self.register_buffer("noise_logliks", noise_logliks)

        if with_kl:
            assert dkl is not None  # pyright
            dkl[ncc < 1] = torch.inf

        return TEStepResult(
            obs_elbo=obs_elbo,
            noise_N=noise_N,
            N=N,
            R=R,
            U=U,
            m=m,
            kl=dkl,
            hard_labels=hard_labels,
            count=count,
            probs=probs,
        )

    def _te_step_job(self, args):
        (
            candidates,
            n_candidates,
            batch_indices,
            with_kl,
            with_hard_labels,
            with_probs,
            initializing,
        ) = args
        return self.process_batch(
            candidates=candidates,
            n_candidates=n_candidates,
            batch_indices=batch_indices,
            with_stats=True,
            with_obs_elbo=True,
            with_kl=with_kl,
            with_hard_labels=with_hard_labels,
            with_probs=with_probs,
            initializing=initializing,
        )

    def batches(self, shuffle=False, show_progress=False, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if shuffle:
            shuf = torch.from_numpy(self._rg.permutation(self.n_spikes))
            for sl in self.batches(show_progress=show_progress, batch_size=batch_size):
                ix = shuf[sl]
                ix.sort()
                yield ix
        else:
            if show_progress:
                starts = trange(
                    0,
                    self.n_spikes,
                    batch_size,
                    desc="Batches",
                    smoothing=0,
                    mininterval=0.2,
                )
            else:
                starts = range(0, self.n_spikes, batch_size)

            for batch_start in starts:
                batch_end = min(self.n_spikes, batch_start + batch_size)
                yield slice(batch_start, batch_end)

    def process_batch(
        self,
        batch_indices,
        candidates,
        n_candidates,
        initializing=False,
        with_grads=False,
        with_stats=False,
        with_kl=False,
        with_elbo=False,
        with_obs_elbo=False,
        with_hard_labels=False,
        with_probs=False,
        with_invquad=False,
        with_edata=False,
        with_origcandidates=False,
    ) -> TEBatchResult:
        assert not with_elbo  # not implemented yet
        if torch.is_tensor(batch_indices):
            n = batch_indices.numel()
        elif isinstance(batch_indices, slice):
            n = batch_indices.stop - batch_indices.start
        else:
            assert False
        if candidates.shape[0] > n:
            candidates = candidates[batch_indices]
        else:
            assert candidates.shape[0] == n
        candidates = candidates.to(self.device)
        origcandidates = candidates.clone() if with_origcandidates else None

        # "E step" within the E step.
        if not initializing:
            neighb_ids, edata = self.load_batch_e(
                candidates=candidates, batch_indices=batch_indices
            )
            eres = _te_batch_e(  # pyright: ignore [reportCallIssue]
                n_units=self.n_units,
                n_candidates=n_candidates,
                noise_log_prop=self.noise_log_prop,
                candidates=candidates,
                with_kl=with_kl,
                with_probs=with_probs,
                with_invquad=with_invquad,
                **edata,
            )
            candidates = eres["candidates"]
            Q = eres["Q"]
        else:
            Q = torch.zeros((n, n_candidates + 1), device=self.device)
            Q[:, :-1][candidates >= 0] = 1.0
        assert candidates is not None
        assert candidates.shape == (n, n_candidates)
        assert Q is not None
        assert Q.shape == (n, n_candidates + 1)

        oelbo = None
        if with_obs_elbo:
            oelbo = spiketorch.elbo(Q, eres["log_liks"], dim=1)

        hard_labels = None
        if with_hard_labels:
            hard_labels = candidates[:, 0].clone()
            is_noise = (Q[:, -1:] > Q[:, :-1]).all(dim=1)
            hard_labels[is_noise] = -1

        noise_N = N = None
        mres = {}
        if with_grads or with_stats:
            mdata = self.load_batch_m(batch_indices, candidates, neighb_ids)
            noise_N, N, Nlut, cpos = _te_batch_m_counts(
                self.n_units, self.nlut, candidates, mdata["lut_ixs"], Q
            )
            if initializing:
                noise_N = N.mean()
            ekw = dict(
                rank=self.rank,
                n_units=self.n_units,
                nlut=self.nlut,
                nc=self.n_channels,
                nc_obs=self.nc_obs,
                nc_miss=self.nc_miss,
                candidates=candidates,
                candidates_pos=cpos,
                Q=Q,
                N=N,
            )
            if self.M:
                mres = _te_batch_m_ppca(**ekw, Nlut=Nlut, **mdata)
            else:
                mres = _te_batch_m_rank0(**ekw, **mdata)

        ddlogpi = ddlognoisep = ddm = ddW = None
        if with_grads:
            Ntot, ddlogpi, ddlognoisep = _grad_counts(
                noise_N, N, self.log_proportions, self.noise_log_prop
            )
            active = slice(None)
            Cinv = None
            if self.Cinv_in_grad:
                # (active,) = N.nonzero(as_tuple=True)
                Cinv = self.noise.full_inverse(device=self.device)
            ddm = _grad_mean(
                Ntot, N, mres["m"][..., :-1], self.means, Cinv=Cinv, active=active
            )
            if self.M:
                ddW = _grad_basis(
                    Ntot,
                    N,
                    mres["R"][..., :-1],
                    self.bases,
                    mres["U"],
                    Cinv=Cinv,
                    active=active,
                )

        return TEBatchResult(
            indices=batch_indices,
            candidates=candidates,
            obs_elbo=oelbo,
            noise_N=noise_N,
            N=N[: self.n_units],
            Nlut=Nlut[: self.nlut],
            m=mres.get("m"),
            R=mres.get("R"),
            Ulut=mres.get("Ulut"),
            ddlogpi=ddlogpi,
            ddlognoisep=ddlognoisep,
            ddm=ddm,
            ddW=ddW,
            ncc=eres["ncc"],
            dkl=eres["dkl"],
            noise_lls=eres["noise_lls"],
            hard_labels=hard_labels,
            probs=eres["probs"],
            invquad=eres["invquad"],
            edata=edata if with_edata else None,
            origcandidates=origcandidates,
        )

    def initialize_fixed(
        self, noise, neighborhoods, pgeom=None, device: str | torch.device = "cpu"
    ):
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
        Coo, Com = observed_and_missing_marginals(noise, neighborhoods, self.miss_ix)

        # precomputed needed cov factors
        Coo_chol = [C.cholesky() for C in Coo]
        Coo_invsqrt = [L.inverse().to_dense() for L in Coo_chol]
        assert all(c.isfinite().all() for c in Coo_invsqrt)
        Coo_inv = [Li.T @ Li for Li in Coo_invsqrt]
        Coo_logdet = [2 * L.to_dense().diagonal().log().sum() for L in Coo_chol]
        Coo_logdet = torch.tensor(Coo_logdet, device=device)
        Cooinv_Com = [Coi @ Cm for Coi, Cm in zip(Coo_inv, Com)]

        R = self.rank
        nco = self.nc_obs
        nc = self.n_channels
        ncm = self.nc_miss

        self.register_buffer("Coo_logdet", Coo_logdet)
        z = {"Coo_inv": nco, "Cooinv_Com": self.nc_miss, "Coo_invsqrt": nco}
        for k, dim in z.items():
            shp = neighborhoods.n_neighborhoods, R * nco, self.rank * dim
            buf = torch.zeros(shp, device=device)
            self.register_buffer(k, buf)

        # loop to fill relevant channels of zero padded buffers
        # for observed channels, the valid subset for each neighborhood
        # (neighborhoods.valid_mask) is exactly where we need to put
        # the matrices so that they hit the right parts of the features
        # for missing channels, we're free to do it in any consistent
        # way. things just need to align betweein miss_ix and Cooinv_Com.
        for ni in range(neighborhoods.n_neighborhoods):
            oi = neighborhoods.valid_mask(ni)
            (mi,) = (self.miss_ix[ni] < nc).nonzero(as_tuple=True)
            ncoi = oi.numel()
            ncmi = mi.numel()

            # adv ix brings the indexed axes to the front
            self.Coo_inv[ni].view(R, nco, R, nco)[:, oi[:, None], :, oi[None, :]] = (
                Coo_inv[ni].view(R, ncoi, R, ncoi).permute(1, 3, 0, 2).to(device)
            )
            self.Cooinv_Com[ni].view(R, nco, R, ncm)[:, oi[:, None], :, mi[None, :]] = (
                Cooinv_Com[ni].view(R, ncoi, R, ncmi).permute(1, 3, 0, 2).to(device)
            )
            self.Coo_invsqrt[ni].view(R, nco, R, nco)[
                :, oi[:, None], :, oi[None, :]
            ] = (Coo_invsqrt[ni].view(R, ncoi, R, ncoi).permute(1, 3, 0, 2).to(device))

    def initialize_changing(self, log_proportions, means, noise_log_prop, bases):
        """Initialize or resize all parameter-dependent arrays

        Some of these are "LUT" indexed. Those are padded with an extra entry
        to be zeroed out, which is just to handle -1 candidates which map to
        nlut via the LUT.
        """
        #  check and set shapes
        self.n_units, r, Nc = means.shape
        assert Nc == self.n_channels + 1
        assert r == self.rank
        assert log_proportions.shape == (self.n_units,)
        if bases is not None:
            Nu_, self.M, r_, nc_ = bases.shape
            assert Nu_ == self.n_units
            assert r == r_
            assert nc_ == Nc
        else:
            self.M = 0

        self.means = means
        self.bases = bases

        # for -1 invalid unit index, make sure likelihood is always -inf.
        log_proportions = F.pad(log_proportions, (0, 1), value=-torch.inf)

        nlut = len(self.nobs_ix)
        nup1 = self.n_units + 1
        nlutp1 = nlut + 1
        lut_units_pad = torch.cat([self.lut_units, 0 * self.lut_units[:1]])
        nobs_ix_pad = torch.cat([self.nobs_ix, 0 * self.nobs_ix[:1]])
        nu = means[lut_units_pad[:, None], :, nobs_ix_pad].permute(0, 2, 1)

        if self._changing_initialized:
            self.log_proportions.resize_(nup1, *self.log_proportions.shape[1:])
            self.log_proportions.copy_(log_proportions)
            self.noise_log_prop.copy_(noise_log_prop)

            self.nu.resize_(nlutp1, *self.nu.shape[1:])
            self.whitenednu.resize_(nlutp1, *self.whitenednu.shape[1:])
            self.Cmo_Cooinv_nu.resize_(nlutp1, *self.Cmo_Cooinv_nu.shape[1:])

            self.nu.copy_(nu)

            if self.M:
                self.Wobs.resize_(nlutp1, *self.Wobs.shape[1:])
                self.Cmo_Cooinv_WobsT.resize_(nlutp1, *self.Cmo_Cooinv_WobsT.shape[1:])
                self.inv_cap.resize_(nlutp1, *self.inv_cap.shape[1:])
                self.obs_logdets.resize_(nlutp1)
                self.wburyroot.resize_(nlutp1, *self.wburyroot.shape[1:])
            self.zero_lut_final()

            return

        # initialize
        M_ix = torch.arange(self.M, device=nu.device)[None, :, None, None]
        self.register_buffer("M_ix", M_ix)
        r_ix = torch.arange(r, device=nu.device)[None, None, :, None]
        self.register_buffer("r_ix", r_ix)
        self.register_buffer("noise_log_prop", noise_log_prop)
        self.register_buffer("log_proportions", log_proportions)
        self.register_buffer("nu", nu)
        nlut_big = int(1.25 * nlutp1)
        bufs = {
            "whitenednu": (self.rank * self.nc_obs,),
            "Cmo_Cooinv_nu": (self.rank * self.nc_miss,),
        }
        if self.M:
            bufs["Wobs"] = (self.M, self.rank * self.nc_obs)
            bufs["inv_cap"] = (self.M, self.M)
            bufs["obs_logdets"] = ()
            bufs["wburyroot"] = (self.rank * self.nc_obs, self.M)
            bufs["inv_cap_Wobs_Cooinv"] = (self.M, self.rank * self.nc_obs)
            bufs["Cmo_Cooinv_WobsT"] = (self.rank * self.nc_miss, self.M)
        for k, v in bufs.items():
            self.register_buffer(k, nu.new_empty((nlut_big, *v)))
            getattr(self, k).resize_(nlutp1, *v)
        self.zero_lut_final()

    def zero_lut_final(self):
        if self.M:
            lut_buf_names = [
                "nu",
                "whitenednu",
                "Cmo_Cooinv_nu",
                "Wobs",
                "inv_cap",
                "obs_logdets",
                "wburyroot",
                "inv_cap_Wobs_Cooinv",
                "Cmo_Cooinv_WobsT",
            ]
        else:
            lut_buf_names = ["nu", "whitenednu", "Cmo_Cooinv_nu"]
        for k in lut_buf_names:
            getattr(self, k)[-1] = 0.0

    def precompute_invx(self):
        # precomputed Cooinv_x and whitenedx
        n = len(self.features)
        X = self.features.view(n, -1)
        # can_pin = self.device.type == "cuda"
        can_pin = False
        self.whitenedx = torch.empty(X.shape, dtype=X.dtype, pin_memory=can_pin)
        shp = (n, self.rank * self.nc_miss)
        self.Cmo_Cooinv_x = torch.empty(shp, dtype=X.dtype, pin_memory=can_pin)
        for ni in trange(self.n_neighborhoods, desc="invx"):
            mask = self.neighborhoods.neighborhood_members(ni)
            # !note the transpose! x @=y is x = x@y, not y@x
            Xmask = X[mask]
            self.whitenedx[mask] = Xmask @ self.Coo_invsqrt[ni].T.to(X)
            self.Cmo_Cooinv_x[mask] = Xmask @ self.Cooinv_Com[ni].to(X)

    _update_mean_batch = _processor_update_mean_batch
    _update_pca_batch = _processor_update_pca_batch
    _finalize_missing_full_m = _finalize_missing_full_m
    _finalize_missing_full_R = _finalize_missing_full_R

    def update(
        self,
        log_proportions,
        means,
        noise_log_prop,
        bases=None,
        unit_neighborhood_counts=None,
    ):
        if unit_neighborhood_counts is not None:
            # update lookup table and re-initialize storage
            res = neighb_lut(unit_neighborhood_counts)
            lut, lut_units, lut_neighbs = res
            self.lut_units = torch.asarray(lut_units, device=self.device)
            self.lut_neighbs = torch.asarray(lut_neighbs, device=self.device)
            self.lut = torch.asarray(lut, device=self.device)

        self.nobs_ix = self.obs_ix[self.lut_neighbs]
        self.nmiss_ix_full = self.miss_ix_full[self.lut_neighbs]
        self.initialize_changing(log_proportions, means, noise_log_prop, bases)

        self.nlut = len(self.lut_units)
        bs = self.update_batch_size
        batches = [slice(i0, min(i0 + bs, self.nlut)) for i0 in range(0, self.nlut, bs)]

        # update mean parameters
        for _ in self.pool.map(self._update_mean_batch, batches):
            pass
        if self.M:
            assert bases is not None
            for _ in self.pool.map(self._update_pca_batch, zip(batches, repeat(bases))):
                pass

    def load_batch_e(
        self, batch_indices, candidates
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        """Load data for the E step within the E step."""
        n, C = candidates.shape
        vcand_ii, vcand_jj = (candidates >= 0).nonzero(as_tuple=True)
        nv = vcand_ii.numel()
        vcand_uu = candidates[vcand_ii, vcand_jj]
        if isinstance(batch_indices, slice):
            assert batch_indices.step in (1, None)
            vcand_ss = vcand_ii + batch_indices.start
        else:
            vcand_ss = batch_indices.to(vcand_ii)[vcand_ii]
        neighborhood_ids = self.neighborhood_ids[batch_indices]
        vcand_nn = neighborhood_ids.to(vcand_ii)[vcand_ii]
        lut_ixs = self.lut[vcand_uu, vcand_nn]
        lut_ixs[lut_ixs == np.prod(self.lut.shape)] = 0

        # some things are only needed in the first pass when computing noise lls
        Coo_logdet = None
        noise_log_priors = None
        if hasattr(self, "noise_logliks"):
            noise_lls = self.noise_logliks[batch_indices]
        else:
            noise_lls = None
            if self.noise_log_priors is not None:
                noise_log_priors = self.noise_log_priors[batch_indices]
        if not hasattr(self, "noise_logliks") or not self.M:
            Coo_logdet = self.Coo_logdet[neighborhood_ids]

        # and some only if ppca...
        if self.M:
            obs_logdets = self.obs_logdets[lut_ixs]
            wburyroot = self.wburyroot[lut_ixs]
        else:
            assert Coo_logdet is not None
            obs_logdets = Coo_logdet[vcand_ii]
            wburyroot = None

        return neighborhood_ids, dict(
            vcand_ii=vcand_ii,
            vcand_jj=vcand_jj,
            whitenedx=self.whitenedx[batch_indices].to(self.device, non_blocking=True),
            whitenednu=self.whitenednu[lut_ixs].view(nv, -1),
            nobs=self.nobs[neighborhood_ids],
            vnobs=self.nobs[vcand_nn],
            obs_logdets=obs_logdets,
            Coo_logdet=Coo_logdet,
            log_proportions=self.log_proportions[vcand_uu],
            noise_lls=noise_lls,
            wburyroot=wburyroot,
            noise_log_priors=noise_log_priors,
        )

    def load_batch_m(self, batch_indices, candidates, neighborhood_ids):
        n, C = candidates.shape
        lut_ixs = self.lut[candidates, neighborhood_ids[:, None]]
        lut_ixs[lut_ixs == np.prod(self.lut.shape)] = self.nlut

        x = self.features[batch_indices].view(n, -1)
        x = x.to(self.device, non_blocking=True)
        Cmo_Cooinv_x = self.Cmo_Cooinv_x[batch_indices]
        Cmo_Cooinv_x = Cmo_Cooinv_x.to(self.device, non_blocking=True)

        # common args
        data = dict(
            obs_ix=self.obs_ix[neighborhood_ids],
            miss_ix=self.miss_ix[neighborhood_ids],
            lut_ixs=lut_ixs,
            x=x,
            Cmo_Cooinv_x=Cmo_Cooinv_x,
            Cmo_Cooinv_nu=self.Cmo_Cooinv_nu[lut_ixs],
        )
        if not self.M:
            return data

        # rank>0 args
        data.update(
            inv_cap=self.inv_cap[lut_ixs],
            inv_cap_Wobs_Cooinv=self.inv_cap_Wobs_Cooinv[lut_ixs],
            Wobs=self.Wobs[lut_ixs],
            Cmo_Cooinv_WobsT=self.Cmo_Cooinv_WobsT[lut_ixs],
            nu=self.nu[lut_ixs].reshape(n, C, -1),
        )
        return data


class CandidateSet:
    def __init__(
        self,
        neighborhoods,
        search_neighborhood_steps=0,
        explore_neighborhood_steps=1,
        neighborhood_adjacency_overlap=0.75,
        random_seed=0,
        search_type="topk",
        random_search_distance_upper_bound=2.0,
        random_search_max_distance=0.5,
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
        self.neighborhood_ids = neighborhoods.neighborhood_ids.cpu()
        self.n_neighborhoods = neighborhoods.n_neighborhoods
        # neighborhoods x channels
        self.neighborhood_indicators = neighborhoods.indicators.T.numpy(force=True)
        # self.neighborhood_subset[i, j] == 1 iff neighb j subset neighb i
        self.neighborhood_subset = (
            self.neighborhood_indicators[None, :]
            <= self.neighborhood_indicators[:, None]
        ).all(2)
        self.neighborhood_subset = self.neighborhood_subset.astype(np.float32)
        self.n_spikes = self.neighborhood_ids.numel()
        self.device = device
        self.rg = np.random.default_rng(random_seed)
        self.gen = spiketorch.spawn_torch_rg(self.rg, device)
        self.search_neighborhood_steps = search_neighborhood_steps
        self.explore_neighborhood_steps = explore_neighborhood_steps
        self.search_type = search_type
        self.random_search_distance_upper_bound = random_search_distance_upper_bound
        self.random_search_max_distance = random_search_max_distance
        self.neighb_adjacency = None
        self.neighborhood_adjacency_overlap = neighborhood_adjacency_overlap
        if search_neighborhood_steps or explore_neighborhood_steps:
            self.neighb_adjacency = neighborhoods.adjacency(
                neighborhood_adjacency_overlap
            )
            self.neighb_adjacency = self.neighb_adjacency.numpy(force=True)

        # initialized in update_sizes()
        self._candidates = None
        self.un_adj = None
        self.unit_neighborhood_counts = None
        self._initialized = False

    def update_sizes(self, n_units=None, n_candidates=3, n_search=2, n_explore=1):
        self.n_candidates = n_candidates
        self.n_search = n_search
        self.n_explore = n_explore
        self.n_units = n_units

        # candidates buffer
        n_total = self.n_candidates + self.n_candidates * self.n_search + self.n_explore
        if self._candidates is None:
            can_pin = self.device is not None and self.device.type == "cuda"
            self._candidates = torch.empty(
                (self.n_spikes, n_total), dtype=torch.long, pin_memory=can_pin
            )
        elif n_total > self._candidates.shape[1]:
            self._candidates.resize_((self.n_spikes, n_total))
        self._candidates.fill_(-1)
        self.candidates = self._candidates[:, :n_total]
        # has the candidates buffer been set to something other than -1s?
        self._initialized = False

        # unit neighborhood counts array
        if n_units is None:
            return
        self.unit_neighborhood_counts = np.zeros(
            (n_units + 1, self.n_neighborhoods), dtype=np.int32
        )

    def initialize_candidates(self, labels, distances, fill_blanks=True):
        """Imposes invariant 1, or at least tries to start off well."""
        assert labels.shape[0] == self.n_spikes
        cinit = labels.view(len(labels), -1)
        (cmask,) = (cinit[:, 0] >= 0).nonzero(as_tuple=True)
        un_adj = self.reinit_neighborhoods(
            cinit[cmask, 0], self.neighborhood_ids[cmask]
        )

        if labels.ndim == 1 and not fill_blanks:
            self.candidates[:, 0] = labels
        elif labels.ndim == 1:
            closest_neighbors = self.search_sets(
                distances, n_search=self.n_candidates, un_adj=un_adj
            )
            assert closest_neighbors.shape[1] <= self.n_candidates
            closest_neighbors = closest_neighbors.to(self.candidates)
            self.candidates[:, 0] = labels
            assert un_adj is not None
            selection_ix = torch.from_numpy(un_adj[-1])[labels, self.neighborhood_ids]
            torch.index_select(
                closest_neighbors[:, : self.n_candidates - 1],
                dim=0,
                index=selection_ix.to(labels),
                out=self.candidates[:, 1 : self.n_candidates],
            )
        else:
            assert fill_blanks
            assert labels.shape == (self.n_spikes, self.n_candidates)
            labels, ixs = labels.sort(dim=1, descending=True)
            dup = labels.diff(dim=1) == 0
            labels[:, 1:][dup] = -1
            labels, ixs2 = labels.sort(dim=1, descending=True)
            ixs = ixs.take_along_dim(ixs2, dim=1)
            fisher_yates_replace(self.rg, self.n_units, labels.numpy())
            labels = labels.take_along_dim(ixs.argsort(dim=1), dim=1)
            self.candidates[:, : self.n_candidates] = labels
            if logger.isEnabledFor(DARTSORTVERBOSE):
                assert torch.all(self.candidates[:, : self.n_candidates] >= 0)
                assert torch.all(
                    self.candidates[:, : self.n_candidates]
                    .sort(dim=1)
                    .values.diff(dim=1)
                    > 0
                )
        self._initialized = True

    def search_sets(
        self,
        distances,
        n_search=None,
        constrain_searches=True,
        un_adj=None,
        allow_overwrite=True,
    ):
        """Search space for evolutionary candidate updates.

        The choice of dim in topk sets the direction of KL.
        We want, for each unit p, to propose others with small
        D(p||q)=E_p[log(p/q)]. Since self.divergences[i,j]=d(i||j),
        that means we use dim=1 in the topk.

        On the other hand, it's not as though reverse KL is entirely
        unmotivated, and it seems to have slightly faster convergence?
        """
        assert distances.shape == (self.n_units, self.n_units)
        if n_search is None:
            n_search = self.n_search

        if self.search_type == "topk" and not constrain_searches:
            if not allow_overwrite:
                distances = distances.clone()
            distances.diagonal().fill_(torch.inf)
            max_nneighbs = distances.isfinite().sum(0).max()

            k = min(n_search, distances.shape[0] - 1, max_nneighbs)
            _, topkinds = torch.topk(distances.T, k=k, dim=1, largest=False)
            assert topkinds.shape == (self.n_units, k)
            topkinds[distances.take_along_dim(topkinds, dim=1).isinf()] = -1
            return topkinds
        elif self.search_type == "topk":
            assert constrain_searches
            assert un_adj is not None
            adj_uu, adj_nn, unit_neighb_adj, un_adj_lut = un_adj
            # here, each unit is be assigned a different set of search units
            # for each neighborhood ID (so each spike gets different search
            # units depending on its neighborhood and current candidates.)
            # - For neighborhood IDs not adjacent to any of the unit's
            #   neighborhoods: the set is empty.
            # - Otherwise: closest J units which are also adjacent to the
            #   neighborhood.
            # adj_uu, adj_nn are a sparsity helper structure that allow us
            # to skip representing the empty sets above.
            # so, search matrix will be indexed on the first dim by a lut
            # ix which represents a unit/neighb pair.
            # TODO is this slow
            if not allow_overwrite:
                distances = distances.clone()
            distances.diagonal().fill_(torch.inf)
            max_nneighbs = distances.isfinite().sum(0).max()
            k = min(n_search, distances.shape[0] - 1, max_nneighbs)
            topinds = torch.full((len(adj_uu), k), -1)
            with np.errstate(divide="ignore"):
                unit_neighb_adj_inf = torch.from_numpy(1.0 / unit_neighb_adj.T).to(
                    distances
                )
            for j, (uu, nn) in enumerate(zip(adj_uu, adj_nn)):
                dists_uunn = distances.T[uu] + unit_neighb_adj_inf[nn]
                topd_uunn, topi_uunn = torch.topk(dists_uunn, k=k, dim=0, largest=False)
                topi_uunn[topd_uunn.isinf()] = -1
                topinds[j] = topi_uunn
            return topinds
        elif self.search_type == "random":
            probs = self.random_search_distance_upper_bound - distances
            eps = 2.0**-30
            probs[distances > self.random_search_max_distance] = eps
            probs.diagonal().fill_(0.0)
            max_nneighbs = (probs > 2 * eps).sum(1).max()
            k = min(n_search, distances.shape[0] - 1, max_nneighbs)
            if k > 0:
                inds = torch.multinomial(
                    probs, num_samples=k, replacement=False, generator=self.gen
                )
                assert inds.shape == (self.n_units, k)
                inds = inds.sort().values
                far = (
                    distances.take_along_dim(inds, dim=1)
                    > self.random_search_max_distance
                )
                inds[far] = -1
                arange_nu = torch.arange(self.n_units, device=inds.device)
                inds[inds == arange_nu[:, None]] = -1
            else:
                inds = torch.zeros((self.n_units, 0), dtype=torch.long)
            return inds
        else:
            assert False

    def search_candidates(self, top, unit_search_neighbors, neighb_ids, un_adj=None):
        if un_adj is None:
            assert unit_search_neighbors.ndim == 2
            return unit_search_neighbors[top]

        assert unit_search_neighbors.ndim == 2
        assert un_adj is not None
        adj_uu, adj_nn, unit_neighb_adj, un_adj_lut = un_adj
        neighb_ids = neighb_ids.cpu()
        adj_lut_ixs = un_adj_lut[top, neighb_ids[:, None].broadcast_to(top.shape)]
        assert adj_lut_ixs.shape == top.shape
        cands = unit_search_neighbors[adj_lut_ixs]
        cands[top < 0] = -1
        assert cands.ndim == 3
        cands = cands.view(top.shape[0], top.shape[1] * unit_search_neighbors.shape[1])
        return cands

    def reinit_neighborhoods(self, labels, neighb_ids, constrain_searches=True):
        # count to determine which units are relevant to each neighborhood
        # this is how "explore" candidates are suggested. the policy is strict:
        # just the top unit counts for each spike. this is to keep the lut smallish,
        # tho it is still huge, hopefully without making the search too bad...
        assert self.unit_neighborhood_counts is not None
        assert labels.shape == neighb_ids.shape
        self.unit_neighborhood_counts.fill(0)
        neighb_ids = neighb_ids.cpu()
        np.add.at(self.unit_neighborhood_counts, (labels, neighb_ids), 1)

        if not constrain_searches:
            return None

        # if `full`, then this is all done in place.
        # determine adjacency of units and neighborhoods
        unit_neighb_ind = (self.unit_neighborhood_counts[:-1] > 0).astype("float32")
        unit_neighb_adj = unit_neighb_ind

        # include neighborhoods which are completely covered already
        unit_neighb_adj = unit_neighb_adj @ self.neighborhood_subset

        # include neighborhoods which are adjacent by steps
        for _ in range(self.search_neighborhood_steps):
            assert self.neighb_adjacency is not None
            unit_neighb_adj = unit_neighb_adj @ self.neighb_adjacency

        # this is really close to neighb_lut(), but it allows for adjacent
        # neighborhoods. ok, probably most of the time we're not doing that,
        # so could only compute this once...
        adj_uu, adj_nn = np.nonzero(unit_neighb_adj)
        un_adj_lut = np.zeros(unit_neighb_adj.shape, dtype=np.int64)
        un_adj_lut[adj_uu, adj_nn] = np.arange(len(adj_uu))

        return adj_uu, adj_nn, unit_neighb_adj, un_adj_lut

    def update_explore_candidates(self, explore_target, neighb_ids):
        assert explore_target.shape[1] == self.n_explore
        if not self.n_explore:
            return

        explore_adj = self.unit_neighborhood_counts
        for _ in range(self.explore_neighborhood_steps):
            explore_adj = explore_adj.astype("float32") @ self.neighb_adjacency
        neighborhood_explore_units = units_overlapping_neighborhoods(explore_adj)
        neighborhood_explore_units = torch.from_numpy(neighborhood_explore_units)

        # sample the explore units and then write. how many units per neighborhood?
        # explore units are chosen per neighborhood. we start by figuring out how many
        # such units there are in each neighborhood.
        n_units_ = (
            torch.tensor(self.n_units)[None]
            .broadcast_to(self.n_neighborhoods, 1)
            .contiguous()
        )
        n_explore = torch.searchsorted(neighborhood_explore_units, n_units_).view(
            self.n_neighborhoods
        )
        n_explore = n_explore[neighb_ids]
        targs = integers_without_inner_replacement(
            self.rg, n_explore.numpy(), size=explore_target.shape
        )
        targs = torch.from_numpy(targs)
        explore = neighborhood_explore_units[neighb_ids[:, None], targs]
        explore[targs < 0] = -1
        explore_target = explore

    def ensure_adjacent(self, top, neighb_ids, un_adj):
        if un_adj is None:
            return
        adj_uu, adj_nn, unit_neighb_adj, un_adj_lut = un_adj
        neighb_ids = neighb_ids.cpu()
        unit_neighb_not_adj = torch.from_numpy(unit_neighb_adj == 0).to(top.device)
        invalid = unit_neighb_not_adj[top, neighb_ids[:, None].broadcast_to(top.shape)]
        top[invalid] = -1

    def ensure_no_blanks(self, top, neighb_ids, un_adj):
        (blanks,) = (top[:, 0] < 0).nonzero(as_tuple=True)
        if not blanks.numel():
            return

        # -- pick units to fill top at random according to overlaps with neighb_ids
        # construct array of probabilities with tiny prob on non-overlapping units
        adj_uu, adj_nn, unit_neighb_adj, un_adj_lut = un_adj
        neighb_ids = neighb_ids.cpu()
        probs = unit_neighb_adj[:, neighb_ids[blanks]]
        assert probs.shape == (blanks.numel(), unit_neighb_adj.shape[0])
        eps = 2.0**-30
        probs.clamp_min(min=eps)

        # pick without replacement along dim 1. then non-overlaps chosen only if there
        # were not enough overlaps (whp)
        inds = torch.multinomial(
            probs, num_samples=top.shape[1], replacement=False, generator=self.gen
        )
        assert inds.shape == (blanks.numel(), top.shape[1])
        # find the non-overlapping guys and overwrite with -1s.
        inds[probs.take_along_dim(inds, dim=1) < 0.5] = -1

        # finish
        top[blanks] = inds

    def propose_candidates(self, distances, indices=None, constrain_searches=True):
        """Assumes invariant 1 and does not mess it up.

        Arguments
        ---------
        unit_search_neighbors: LongTensor (n_units, n_search)
        """
        assert self._initialized

        full = indices is None
        if full:
            indices = slice(None)

        # some data structures needed in advance
        neighb_ids = self.neighborhood_ids[indices]
        # this is stored as a property mainly for tests. it isn't modified elsewhere.
        self.un_adj = self.reinit_neighborhoods(
            self.candidates[indices, 0],
            neighb_ids,
            constrain_searches=constrain_searches,
        )
        unit_search_neighbors = self.search_sets(
            distances, constrain_searches=constrain_searches, un_adj=self.un_adj
        )
        unit_search_neighbors = unit_search_neighbors.to(self.candidates)
        assert unit_search_neighbors.shape[1] <= self.n_search
        n_search = unit_search_neighbors.shape[1]

        # determine some shape parameters and nicknames
        C = self.n_candidates
        n_search_total = C * n_search
        search_slice = slice(C, C + n_search_total)
        total = C + n_search_total + self.n_explore
        explore_slice = slice(C + n_search_total, total)
        candidates = self.candidates[indices, :total]
        top = candidates[:, :C]

        # make sure all candidates are in adjacent neighborhoods else vanquish them
        if constrain_searches:
            self.ensure_no_blanks(top, neighb_ids, self.un_adj)
            self.ensure_adjacent(top, neighb_ids, self.un_adj)

        # update search sets
        if n_search:
            candidates[:, search_slice] = self.search_candidates(
                top, unit_search_neighbors, neighb_ids, un_adj=self.un_adj
            )

        # which to explore? done in-place in first arg.
        self.update_explore_candidates(candidates[:, explore_slice], neighb_ids)

        # replace duplicates with -1. TODO: replace quadratic algorithm with -1.
        erase_dups(candidates.numpy())
        if logger.isEnabledFor(DARTSORTVERBOSE):
            c_sorted = self.candidates[:, : self.n_candidates].sort(dim=1).values
            diff = c_sorted.diff(dim=1)
            valid = torch.logical_or(c_sorted[:, 1:] == -1, diff > 0)
            assert valid.all()
            if candidates[:, self.n_candidates :].numel():
                assert candidates[:, self.n_candidates :].min() >= -1

        # update counts for the rest of units
        if candidates.shape[1] > 1:
            np.add.at(
                self.unit_neighborhood_counts,
                (candidates[:, 1:total], neighb_ids[:, None]),
                1,
            )

        # this was padded to allow for -1s in candidates
        unit_neighborhood_counts = self.unit_neighborhood_counts[:-1]

        return candidates, unit_neighborhood_counts
