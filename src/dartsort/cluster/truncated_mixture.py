import numpy as np
import torch
from logging import getLogger
from torch import nn
import time
import joblib
from linear_operator import operators
from linear_operator.utils.cholesky import psd_safe_cholesky
import torch.nn.functional as F
from tqdm.auto import trange


from ..util.noise_util import EmbeddedNoise
from ..util.sparse_util import (
    integers_without_inner_replacement,
    erase_dups,
    fisher_yates_replace,
)
from .stable_features import SpikeNeighborhoods, StableSpikeDataset
from ._truncated_em_helpers import (
    neighb_lut,
    TEBatchResult,
    TEStepResult,
    _te_batch_e,
    _te_batch_m_counts,
    _te_batch_m_rank0,
    _te_batch_m_ppca,
    units_overlapping_neighborhoods,
    _grad_counts,
    _grad_mean,
    _grad_basis,
    _elbo_prior_correction,
)
from ..util import spiketorch
from ..util.logging_util import DARTSORTDEBUG, DARTSORTVERBOSE

logger = getLogger(__name__)


class SpikeTruncatedMixtureModel(nn.Module):
    def __init__(
        self,
        data: StableSpikeDataset,
        noise: EmbeddedNoise,
        M: int = 0,
        n_candidates: int = 3,
        n_search: int | None = 5,
        n_explore: int | None = None,
        n_units: int | None = None,
        covariance_radius: float | None = None,
        noise_trunc_factors: torch.Tensor | None = None,
        random_seed=0,
        n_threads: int = 0,
        batch_size=2**11,
        exact_kl=True,
        fixed_noise_proportion=None,
        sgd_batch_size=None,
        Cinv_in_grad=True,
        alpha0=5.0,
        laplace_ard=False,
        alpha_max=1e6,
        alpha_min=1e-6,
        prior_scales_mean=False,
        min_log_prop=-50.0,
    ):
        super().__init__()

        self.data = data
        self.noise = noise
        train_indices, self.train_neighborhoods = self.data.neighborhoods("extract")

        if laplace_ard:
            assert alpha0 and alpha0 > 0
        self.has_prior = bool(alpha0)
        self.alpha0 = alpha0
        self.laplace_ard = laplace_ard
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.prior_scales_mean = prior_scales_mean
        self.covariance_radius = covariance_radius
        self.fixed_noise_proportion = fixed_noise_proportion
        self.exact_kl = exact_kl
        self.sgd_batch_size = sgd_batch_size
        self.min_log_prop = min_log_prop

        self.n_spikes = train_indices.numel()
        self.M = M
        self.data_dim = self.data.rank * self.data.n_channels

        self.candidates = CandidateSet(
            neighborhoods=self.train_neighborhoods,
            random_seed=random_seed,
            device=self.data.device,
        )

        self.initial_n_candidates = n_candidates
        self.initial_n_search = n_search
        self.initial_n_explore = n_explore
        self.set_sizes(n_units)
        self.processor = TruncatedExpectationProcessor(
            noise=noise,
            neighborhoods=self.train_neighborhoods,
            features=self.data._train_extract_features,
            random_seed=random_seed,
            n_threads=n_threads,
            batch_size=batch_size,
            covariance_radius=covariance_radius,
            pgeom=data.prgeom,
            Cinv_in_grad=Cinv_in_grad,
            noise_trunc_factors=noise_trunc_factors,
        )
        self.to(device=self.data.device)

    def set_sizes(self, n_units: int | None = None):
        n_candidates = self.initial_n_candidates
        n_search = self.initial_n_search
        n_explore = self.initial_n_explore
        if n_units is not None:
            n_candidates = min(n_units, n_candidates)
        if n_search is None:
            n_search = n_candidates
        if n_explore is None:
            n_explore = n_search
        if n_units is not None:
            remainder = n_units - n_candidates
            n_search = max(0, min(remainder // n_candidates, n_search))
            remainder -= n_candidates * n_search
            n_explore = max(0, min(remainder, n_explore))

        self.n_candidates = n_candidates
        self.n_search = n_search
        self.n_explore = n_explore

        self.candidates.update_sizes(self.n_candidates, self.n_search, self.n_explore)

    def set_parameters(
        self,
        labels,
        means,
        log_proportions,
        noise_log_prop,
        kl_divergences,
        bases=None,
        alpha=None,
    ):
        """Parameters are stored padded with an extra channel."""
        self.n_units = means.shape[0]
        self.set_sizes(self.n_units)
        assert means.shape == (self.n_units, self.noise.rank, self.noise.n_channels)
        assert log_proportions.shape == (self.n_units,)
        assert kl_divergences.shape == (self.n_units, self.n_units)
        assert means.isfinite().all()
        assert log_proportions.isfinite().all()
        assert noise_log_prop.isfinite()

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

        self.register_buffer(
            "kl_divergences",
            torch.asarray(kl_divergences, dtype=self.means.dtype, copy=True),
        )

        self.candidates.initialize_candidates(
            labels, self.search_sets(n_search=self.candidates.n_candidates)
        )

        M = 0 if bases is None else bases.shape[1]
        assert M == self.M
        if M:
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

        self.to(means.device)

    def step(self, show_progress=False, hard_label=False, with_probs=False, tic=None):
        search_neighbors = self.search_sets()
        candidates, unit_neighborhood_counts = self.candidates.propose_candidates(
            search_neighbors
        )
        assert (
            candidates.untyped_storage() == self.candidates.candidates.untyped_storage()
        )
        self.processor.update(
            log_proportions=self.log_proportions,
            noise_log_prop=self.noise_log_prop,
            means=self.means,
            bases=self.bases,
            unit_neighborhood_counts=unit_neighborhood_counts,
        )
        result = self.processor.truncated_e_step(
            candidates=candidates,
            n_candidates=self.n_candidates,
            show_progress=show_progress,
            with_kl=not self.exact_kl,
            with_hard_labels=hard_label,
            with_probs=with_probs,
        )
        assert result.obs_elbo is not None
        assert result.N is not None
        assert result.m is not None

        if self.has_prior and self.prior_scales_mean:
            mean_scale = result.N / (result.N + self.alpha0)
            self.means[..., :-1] = result.m * mean_scale[:, None, None]
        else:
            self.means[..., :-1] = result.m
        if logger.isEnabledFor(DARTSORTDEBUG):
            assert self.means[..., :-1].isfinite().all()

        W = None
        if self.bases is not None:
            assert result.R is not None
            assert result.U is not None

            blank = torch.all(result.U.diagonal(dim1=-2, dim2=-1) == 0, dim=1)
            assert torch.equal(blank, result.N == 0)
            if blank.any():
                # just to avoid numerical issues when a unit dies
                result.U[blank] += torch.eye(self.M, device=result.U.device)

            if self.has_prior:
                N_denom = result.N.clamp(min=1.0)
                tikh = self.alpha / N_denom.unsqueeze(1)
                result.U.diagonal(dim1=-2, dim2=-1).add_(tikh.to(result.U))

            # Uc = psd_safe_cholesky(result.U)
            # W = torch.cholesky_solve(result.R.view(*result.U.shape[:-1], -1), Uc)
            W = torch.linalg.solve(result.U, result.R.view(*result.U.shape[:-1], -1))
            assert W.shape == (self.n_units, self.M, self.data_dim)
            self.bases[..., :-1] = W.view(self.bases[..., :-1].shape)
            if logger.isEnabledFor(DARTSORTDEBUG):
                assert self.bases[..., :-1].isfinite().all()

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
            if logger.isEnabledFor(DARTSORTDEBUG):
                assert self.alpha.isfinite().all()

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
            lp = torch.log_softmax(self._N.log(), dim=0)
            self.noise_log_prop.fill_(lp[0])
            self.log_proportions[:] = lp[1:].clamp_(min=self.min_log_prop)

        if self.exact_kl:
            self.update_dkl()
        else:
            assert result.kl is not None
            self.kl_divergences[:] = result.kl

        obs_elbo = result.obs_elbo
        if self.has_prior:
            # update elbo
            obs_elbo += _elbo_prior_correction(
                self.alpha0,
                result.count,
                self.means[..., :-1].reshape(self.n_units, -1),
                W,
                self.noise.full_inverse(),
                alpha=self.alpha if self.laplace_ard and self.M else None,
                mean_prior=self.prior_scales_mean,
            )

        result = dict(
            obs_elbo=obs_elbo.numpy(force=True).item(),
            noise_lp=self.noise_log_prop.numpy(force=True).copy(),
            labels=result.hard_labels,
            probs=result.probs,
        )
        if tic is not None:
            result["wall"] = time.perf_counter() - tic
        return result

    def sgd_epoch(
        self,
        opt,
        show_progress=False,
        tic=None,
    ):
        """"""

        # things that don't change...
        search_neighbors = self.search_sets()

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
        self.update_dkl()

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
            search_neighbors, indices=batch_indices
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

    def update_dkl(self):
        W = self.bases
        if W is not None:
            W = W[..., :-1].reshape(len(W), self.M, -1).mT
        spiketorch.woodbury_kl_divergence(
            C=self.noise._full_cov,
            mu=self.means[..., :-1].reshape(len(self.means), -1),
            W=W,
            out=self.kl_divergences,
        )

    def search_sets(self, n_search=None):
        """Search space for evolutionary candidate updates.

        The choice of dim in topk sets the direction of KL.
        We want, for each unit p, to propose others with small
        D(p||q)=E_p[log(p/q)]. Since self.kl_divergences[i,j]=d(i||j),
        that means we use dim=1 in the topk.
        """
        if n_search is None:
            n_search = self.candidates.n_search
        self.kl_divergences.diagonal().fill_(torch.inf)
        k = min(n_search, self.kl_divergences.shape[0] - 1)
        _, topkinds = torch.topk(self.kl_divergences, k=k, dim=1, largest=False)
        return topkinds

    def channel_occupancy(
        self,
        labels,
        min_count=0,
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
        vneighbs = self.candidates.neighborhood_ids[valid]
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
        batch_size: int = 2**14,
        n_threads: int = 0,
        random_seed: int = 0,
        precompute_invx=True,
        covariance_radius=None,
        Cinv_in_grad=True,
        noise_trunc_factors=None,
        pgeom=None,
    ):
        super().__init__()

        # initialize fixed noise-related arrays
        self.noise = noise
        self.n_spikes = features.shape[0]
        self.nc_obs = features.shape[2]
        assert self.nc_obs == neighborhoods.neighborhoods.shape[1]
        if features.isnan().any():
            # TODO: something about this...?
            self.features = features.nan_to_num()
        else:
            self.features = features.clone()
        if noise_trunc_factors is not None:
            self.register_buffer("noise_trunc_factors", noise_trunc_factors)
        else:
            self.noise_trunc_factors = None
        self.initialize_fixed(
            noise, neighborhoods, pgeom=pgeom, covariance_radius=covariance_radius
        )
        self.neighborhoods = neighborhoods
        self.neighborhood_ids = neighborhoods.neighborhood_ids
        self.n_neighborhoods = neighborhoods.n_neighborhoods
        self.Cinv_in_grad = Cinv_in_grad
        if precompute_invx:
            self.precompute_invx()
        self.batch_size = batch_size

        # M is updated by self.update() when a basis is assigned here.
        self.M = 0

        # thread pool
        self._rg = np.random.default_rng(random_seed)
        self.pool = joblib.Parallel(
            n_jobs=n_threads or 1, backend="threading", return_as="generator_unordered"
        )

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
    ) -> TEStepResult:
        """E step of truncated VEM

        Will update candidates[:, :self.n_candidates] in place.
        """
        # sufficient statistics
        dev = self.device
        mean_shp = (self.n_units, self.rank, self.nc)
        m = torch.zeros(mean_shp, device=dev, dtype=torch.double)
        N = torch.zeros((self.n_units,), device=dev, dtype=torch.double)
        dkl = None
        if with_kl:
            nu2 = (self.n_units, self.n_units)
            dkl = torch.zeros(nu2, device=dev, dtype=torch.double)
            ncc = dkl.clone()
        noise_N = torch.tensor(0.0, device=dev, dtype=torch.double)
        obs_elbo = torch.tensor(0.0, device=dev, dtype=torch.double)
        R = U = None
        if self.M:
            r_shp = (self.n_units, self.M, self.rank, self.nc)
            R = torch.zeros(r_shp, device=dev, dtype=torch.double)
            u_shp = (self.n_units, self.M, self.M)
            U = torch.zeros(u_shp, device=dev, dtype=torch.double)

        # will be updated...
        top_candidates = candidates[:, : n_candidates]
        hard_labels = None
        if with_hard_labels:
            hard_labels = torch.empty_like(top_candidates[:, 0])
        probs = None
        if with_probs:
            probs = torch.empty(candidates[:, : n_candidates].shape)

        # do we need to initialize the noise log likelihoods?
        first_run = not hasattr(self, "noise_logliks")
        noise_logliks = None
        if first_run:
            noise_logliks = torch.empty((self.n_spikes,), device=dev)

        # run loop
        jobs = (
            self._te_step_job(
                candidates, n_candidates, batchix, with_kl, with_hard_labels, with_probs
            )
            for batchix in self.batches(show_progress=show_progress)
        )
        count = 0
        for result in self.pool(jobs):
            assert result is not None  # why, pyright??

            noise_N += result.noise_N
            N += result.N

            # weight for the welford running averages below
            n1_n01 = result.N.div(N.clamp(min=1.0))[:, None, None]

            # welford for the mean
            m += result.m[:, :, :-1].sub_(m).mul_(n1_n01)

            # running avg for elbo/n
            count += len(result.candidates)
            if not result.obs_elbo.isfinite():
                raise ValueError(
                    f"batch elbo diverged! {result.obs_elbo=} {len(result.candidates)=}"
                )
            obs_elbo += (result.obs_elbo - obs_elbo) * (len(result.candidates) / count)
            if not obs_elbo.isfinite():
                raise ValueError(
                    f"running elbo diverged! {obs_elbo=} {result.obs_elbo=}"
                    f"{count=} {len(result.candidates)=}"
                )

            if with_kl:
                ncc += result.ncc
                dkl += (result.dkl - dkl).div_(ncc.clamp(min=1.0))

            if self.M:
                assert R is not None
                assert U is not None
                R += result.R[..., :-1].sub_(R).mul_(n1_n01[..., None])
                U += result.U.sub_(U).mul_(n1_n01)

            # could do these in the threads? unless shuffled.
            top_candidates[result.indices] = result.candidates
            if noise_logliks is not None:
                noise_logliks[result.indices] = result.noise_lls

            if with_hard_labels:
                assert result.hard_labels is not None
                hard_labels[result.indices] = result.hard_labels

            if with_probs:
                probs[result.indices] = result.probs

        if first_run:
            self.register_buffer("noise_logliks", noise_logliks)

        if with_kl:
            assert dkl is not None  # pyright
            dkl[ncc < 1] = torch.inf

        return TEStepResult(
            elbo=None,
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

    @joblib.delayed
    def _te_step_job(
        self, candidates, n_candidates, batch_indices, with_kl, with_hard_labels, with_probs
    ):
        return self.process_batch(
            candidates=candidates,
            n_candidates=n_candidates,
            batch_indices=batch_indices,
            with_stats=True,
            with_obs_elbo=True,
            with_kl=with_kl,
            with_hard_labels=with_hard_labels,
            with_probs=with_probs,
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
        with_grads=False,
        with_stats=False,
        with_kl=False,
        with_elbo=False,
        with_obs_elbo=False,
        with_hard_labels=False,
        with_probs=False,
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

        # "E step" within the E step.
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
            **edata,
        )
        candidates = eres["candidates"]
        assert candidates is not None
        assert candidates.shape == (n, n_candidates)
        Q = eres["Q"]
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
            noise_N, N = _te_batch_m_counts(self.n_units, candidates, Q)
            shapekw = dict(
                rank=self.rank,
                n_units=self.n_units,
                nc=self.nc,
                nc_obs=self.nc_obs,
                nc_miss=self.nc_miss,
            )
            ekw = dict(candidates=candidates, Q=Q, N=N)
            if self.M:
                mres = _te_batch_m_ppca(**shapekw, **ekw, **mdata)
            else:
                mres = _te_batch_m_rank0(**shapekw, **ekw, **mdata)

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
            N=N,
            m=mres.get("m"),
            R=mres.get("R"),
            U=mres.get("U"),
            ddlogpi=ddlogpi,
            ddlognoisep=ddlognoisep,
            ddm=ddm,
            ddW=ddW,
            ncc=eres["ncc"],
            dkl=eres["dkl"],
            noise_lls=eres["noise_lls"],
            hard_labels=hard_labels,
            probs=eres["probs"],
        )

    def initialize_fixed(
        self, noise, neighborhoods, pgeom=None, covariance_radius=None
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
        self.rank = R = noise.rank
        self.nc = nc = noise.n_channels

        # determine missing channels
        missing_chans = []
        truncate = covariance_radius and np.isfinite(covariance_radius)
        for ni in range(neighborhoods.n_neighborhoods):
            mix = neighborhoods.missing_channels(ni)
            if truncate:
                assert pgeom is not None
                oix = neighborhoods.neighborhood_channels(ni)
                d = torch.cdist(pgeom[:nc], pgeom[oix]).min(dim=1).values
                assert d[oix].max() < covariance_radius
                mix = mix[d[mix] <= covariance_radius]
            missing_chans.append(mix)

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
        if self.noise_trunc_factors is not None:
            assert len(noise_trunc_factors) == len(Coo)
        device = Coo[0].device
        Com = []
        for ni in range(neighborhoods.n_neighborhoods):
            Comi = noise.offdiag_covariance(
                channels_left=neighborhoods.neighborhood_channels(ni),
                channels_right=missing_chans[ni],
            ).to_dense()
            if truncate:
                assert pgeom is not None
                oix = neighborhoods.neighborhood_channels(ni)
                d = torch.cdist(pgeom[oix], pgeom[missing_chans[ni]])
                mask = (d > 0).to(torch.float)[None, :, None, :]
                mask = mask.broadcast_to((R, d.shape[0], R, d.shape[1]))
                Comi.view(mask.shape).mul_(mask)
            Com.append(Comi)

        # Get choleskys. linear_operator will jitter to help
        # with numerics for us if we do everything via chol.
        Coo_chol = [C.cholesky() for C in Coo]
        Coo_invsqrt = [L.inverse().to_dense() for L in Coo_chol]
        assert all(c.isfinite().all() for c in Coo_invsqrt)
        Coo_inv = [Li.T @ Li for Li in Coo_invsqrt]
        Coo_logdet = [2 * L.to_dense().diagonal().log().sum() for L in Coo_chol]
        Coo_logdet = torch.tensor(Coo_logdet, device=device)
        Cooinv_Com = [Coi @ Cm for Coi, Cm in zip(Coo_inv, Com)]

        # figure out dimensions
        nc_obs = neighborhoods.channel_counts
        self.nc_miss = ncm = max(map(len, missing_chans))
        nco = self.nc_obs

        # understand channel subsets
        # these arrays are used to scatter into D-shaped dims.
        # actually... maybe into D+1-shaped dims, so that we can use
        # D as the invalid indicator
        self.register_buffer("obs_ix", neighborhoods.neighborhoods.to(device))
        miss_ix = torch.full(
            (neighborhoods.n_neighborhoods, ncm), fill_value=nc, device=device
        )
        self.register_buffer("miss_ix", miss_ix)
        for ni in range(neighborhoods.n_neighborhoods):
            self.miss_ix[ni, : missing_chans[ni].numel()] = missing_chans[ni]

        # allocate buffers
        self.register_buffer("Coo_logdet", Coo_logdet)
        self.register_buffer("nobs", R * nc_obs)
        z = [("Coo_inv", nco), ("Cooinv_Com", ncm), ("Coo_invsqrt", nco)]
        for k, d2 in z:
            shp = neighborhoods.n_neighborhoods, R * nco, R * d2
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
            # remember, adv ix brings the indexed axes to the front
            self.Coo_inv[ni].view(R, nco, R, nco)[:, oi[:, None], :, oi[None, :]] = (
                Coo_inv[ni].view(R, ncoi, R, ncoi).permute(1, 3, 0, 2).to(device)
            )
            self.Cooinv_Com[ni].view(R, nco, R, ncm)[:, oi[:, None], :, mi[None, :]] = (
                Cooinv_Com[ni].view(R, ncoi, R, ncmi).permute(1, 3, 0, 2).to(device)
            )
            self.Coo_invsqrt[ni].view(R, nco, R, nco)[
                :, oi[:, None], :, oi[None, :]
            ] = Coo_invsqrt[ni].view(R, ncoi, R, ncoi).permute(1, 3, 0, 2).to(device)

    def precompute_invx(self):
        # precomputed Cooinv_x and whitenedx
        n = len(self.features)
        X = self.features.view(n, -1)
        self.whitenedx = X.clone()
        self.Cmo_Cooinv_x = self.whitenedx.new_empty(n, self.rank * self.nc_miss)
        for ni in trange(self.n_neighborhoods, desc="invx"):
            mask = self.neighborhoods.neighborhood_members(ni)
            # !note the transpose! x @=y is x = x@y, not y@x
            self.whitenedx[mask] @= self.Coo_invsqrt[ni].T.to(X)
            self.Cmo_Cooinv_x[mask] = X[mask] @ self.Cooinv_Com[ni].to(X)

    def update(
        self,
        log_proportions,
        means,
        noise_log_prop,
        bases=None,
        unit_neighborhood_counts=None,
        batch_size=1024,
    ):
        Nu, r, Nc = means.shape
        assert Nc in (self.nc, self.nc + 1)
        assert r == self.rank
        already_padded = Nc == self.nc + 1
        assert log_proportions.shape == (Nu,)
        self.M = 0
        self.n_units = Nu
        if bases is not None:
            Nu_, self.M, r_, nc_ = bases.shape
            assert Nu_ == Nu
            assert r == r_
            assert nc_ == Nc

        if unit_neighborhood_counts is not None:
            # update lookup table and re-initialize storage
            res = neighb_lut(unit_neighborhood_counts)
            lut, self.lut_units, self.lut_neighbs = res
            self.lut = torch.asarray(lut, device=self.device)
        nlut = len(self.lut_units)

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
        self.register_buffer("means", mu[..., :-1].reshape(Nu, -1))
        if self.M:
            self.register_buffer("bases", W[..., :-1].reshape(Nu, self.M, -1))

        # mean obs/hid parts
        nu = mu[self.lut_units[:, None], :, obs_ix].permute(0, 2, 1)
        tnu = mu[self.lut_units[:, None], :, miss_ix].permute(0, 2, 1)

        # whitened means
        self.register_buffer("nu", nu)
        self.register_buffer("tnu", tnu)
        shp = (nlut, self.rank * self.nc_obs, 1)
        Cmo_Cooinv_nu = nu.new_empty((nlut, self.rank * self.nc_miss))
        whitenednu = nu.new_empty(shp)
        for bs in range(0, nlut, batch_size):
            be = min(nlut, bs + batch_size)
            nbatch = self.lut_neighbs[bs:be]
            Cooinvsqrtbatch = self.Coo_invsqrt[nbatch]
            Cooinv_Combatch = self.Cooinv_Com[nbatch]
            nubatch = nu[bs:be].reshape(be - bs, -1, 1)
            torch.bmm(Cooinvsqrtbatch, nubatch, out=whitenednu[bs:be])
            Cmo_Cooinv_nu[bs:be] = Cooinv_Combatch.mT.bmm(nubatch).squeeze()
        self.register_buffer("whitenednu", whitenednu[..., 0])
        self.register_buffer("Cmo_Cooinv_nu", Cmo_Cooinv_nu)

        # basis stuff
        if not self.M:
            return
        assert W is not None

        # load basis
        Wobs = W[self.lut_units[:, None], :, :, obs_ix].permute(0, 2, 3, 1)
        assert Wobs.shape == (nlut, self.M, r, self.nc_obs)
        Wobs = Wobs.reshape(nlut, self.M, -1)
        Wmiss = W[self.lut_units[:, None], :, :, miss_ix].permute(0, 2, 3, 1)
        assert Wmiss.shape == (nlut, self.M, r, self.nc_miss)
        Wmiss = Wmiss.reshape(nlut, self.M, -1)
        self.register_buffer("Wobs", Wobs)

        Cooinv_WobsT = torch.empty(
            (nlut, self.rank * self.nc_obs, self.M), device=Wobs.device
        )
        Cmo_Cooinv_WobsT = torch.empty(
            (nlut, self.rank * self.nc_miss, self.M), device=Wobs.device
        )
        for bs in range(0, nlut, batch_size):
            be = min(nlut, bs + batch_size)
            Cooinvbatch = self.Coo_inv[self.lut_neighbs[bs:be]]
            torch.bmm(Cooinvbatch, Wobs[bs:be].mT, out=Cooinv_WobsT[bs:be])
            Cmo_Cooinv_batch = self.Cooinv_Com[self.lut_neighbs[bs:be]].mT
            torch.bmm(Cmo_Cooinv_batch, Wobs[bs:be].mT, out=Cmo_Cooinv_WobsT[bs:be])
        self.register_buffer("Cmo_Cooinv_WobsT", Cmo_Cooinv_WobsT)

        cap = torch.bmm(Wobs, Cooinv_WobsT)
        cap.diagonal(dim1=-2, dim2=-1).add_(1.0)
        assert cap.shape == (nlut, self.M, self.M)
        # LL' = cap
        cap_chol = operators.DenseLinearOperator(cap).cholesky()
        # cap^{-1} = L^-T L^-1, this is L-1.
        cap_invsqrt = cap_chol.inverse().to_dense()
        cap_logdets = cap_chol.diagonal(dim1=-2, dim2=-1).log().sum(dim=1).mul_(2.0)
        cap_inv = cap_invsqrt.mT.bmm(cap_invsqrt)
        self.register_buffer("inv_cap", cap_inv)

        # matrix determinant lemma
        noise_logdets = self.Coo_logdet[self.lut_neighbs]
        self.register_buffer("obs_logdets", noise_logdets + cap_logdets)

        # this is used in the log lik Woodbury, and it's also the posterior
        # covariance of the ppca embedding (usually I call that T).
        wburyroot = torch.empty_like(Cooinv_WobsT)
        inv_cap_Wobs_Cooinv = torch.empty_like(wburyroot.mT)
        W_WCC = Wmiss
        inv_cap_W_WCC = torch.empty_like(W_WCC)
        del Wmiss
        for bs in range(0, nlut, batch_size):
            be = min(nlut, bs + batch_size)
            Cooinv = self.Coo_inv[self.lut_neighbs[bs:be]]
            Cooinvsqrt = self.Coo_invsqrt[self.lut_neighbs[bs:be]]
            coeft = Wobs[bs:be].mT.bmm(cap_invsqrt[bs:be].mT)
            torch.bmm(Cooinvsqrt, coeft, out=wburyroot[bs:be])

            inv_cap_Wobs_Cooinv[bs:be] = cap_inv[bs:be].bmm(Wobs[bs:be]).bmm(Cooinv)

            W_WCC[bs:be].baddbmm_(
                Wobs[bs:be], self.Cooinv_Com[self.lut_neighbs[bs:be]], alpha=-1
            )
            inv_cap_W_WCC[bs:be] = cap_inv[bs:be].bmm(W_WCC[bs:be])
        self.register_buffer("wburyroot", wburyroot)
        self.register_buffer("inv_cap_Wobs_Cooinv", inv_cap_Wobs_Cooinv)
        self.register_buffer("W_WCC", W_WCC)
        self.register_buffer("inv_cap_W_WCC", inv_cap_W_WCC)

        # gizmo matrix used in a certain part of the "imputation"
        for bs in range(0, nlut, batch_size):
            be = min(len(W_WCC), bs + batch_size)

    def load_batch_e(
        self, batch_indices, candidates
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        """Load data for the E step within the E step."""
        n, C = candidates.shape
        neighborhood_ids = self.neighborhood_ids[batch_indices]
        lut_ixs = self.lut[candidates, neighborhood_ids[:, None]]
        lut_ixs[lut_ixs == np.prod(self.lut.shape)] = 0

        # some things are only needed in the first pass when computing noise lls
        Coo_logdet = None
        if hasattr(self, "noise_logliks"):
            noise_lls = self.noise_logliks[batch_indices]
        else:
            noise_lls = None
        noise_trunc_factors = None
        if self.noise_trunc_factors is not None and noise_lls is None:
            noise_trunc_factors = self.noise_trunc_factors[neighborhood_ids]
        if not hasattr(self, "noise_logliks") or not self.M:
            Coo_logdet = self.Coo_logdet[neighborhood_ids]

        # and some only if ppca...
        if self.M:
            obs_logdets = self.obs_logdets[lut_ixs]
            wburyroot = self.wburyroot[lut_ixs]
        else:
            assert Coo_logdet is not None
            obs_logdets = Coo_logdet[:, None].broadcast_to(candidates.shape)
            wburyroot = None

        return neighborhood_ids, dict(
            whitenedx=self.whitenedx[batch_indices].to(self.device),
            whitenednu=self.whitenednu[lut_ixs].view(n, C, -1),
            nobs=self.nobs[neighborhood_ids],
            obs_logdets=obs_logdets,
            Coo_logdet=Coo_logdet,
            log_proportions=self.log_proportions[candidates],
            noise_lls=noise_lls,
            noise_trunc_factors=noise_trunc_factors,
            wburyroot=wburyroot,
        )

    def load_batch_m(self, batch_indices, candidates, neighborhood_ids):
        n, C = candidates.shape
        lut_ixs = self.lut[candidates, neighborhood_ids[:, None]]
        lut_ixs[lut_ixs == np.prod(self.lut.shape)] = 0

        # common args
        data = dict(
            obs_ix=self.obs_ix[neighborhood_ids],
            miss_ix=self.miss_ix[neighborhood_ids],
            x=self.features[batch_indices].view(n, -1).to(self.device),
            nu=self.nu[lut_ixs].reshape(n, C, -1),
            tnu=self.tnu[lut_ixs].reshape(n, C, -1),
            Cmo_Cooinv_x=self.Cmo_Cooinv_x[batch_indices].to(self.device),
            Cmo_Cooinv_nu=self.Cmo_Cooinv_nu[lut_ixs],
        )
        if not self.M:
            return data

        # rank>0 args
        data.update(
            inv_cap=self.inv_cap[lut_ixs],
            inv_cap_Wobs_Cooinv=self.inv_cap_Wobs_Cooinv[lut_ixs],
            W_WCC=self.W_WCC[lut_ixs],
            Wobs=self.Wobs[lut_ixs],
            Cmo_Cooinv_WobsT=self.Cmo_Cooinv_WobsT[lut_ixs],
            inv_cap_W_WCC=self.inv_cap_W_WCC[lut_ixs],
        )
        return data


class CandidateSet:
    def __init__(self, neighborhoods, random_seed=0, device=None):
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
        self.neighborhood_ids = neighborhoods.neighborhood_ids
        self.n_neighborhoods = neighborhoods.n_neighborhoods
        self.n_spikes = self.neighborhood_ids.numel()
        self.device = device
        self.rg = np.random.default_rng(random_seed)
        self._candidates = None
        self._initialized = False

    def update_sizes(self, n_candidates=3, n_search=2, n_explore=1):
        self.n_candidates = n_candidates
        self.n_search = n_search
        self.n_explore = n_explore
        n_total = self.n_candidates + self.n_candidates * self.n_search + self.n_explore
        if self._candidates is None:
            can_pin = self.device is not None and self.device.type == "cuda"
            self._candidates = torch.empty(
                (self.n_spikes, n_total), dtype=torch.long, pin_memory=can_pin
            )
        elif n_total > self._candidates.shape[1]:
            self._candidates.resize_((self.n_spikes, n_total))
        self.candidates = self._candidates[:, :n_total]
        self._initialized = False

    def initialize_candidates(self, labels, closest_neighbors):
        """Imposes invariant 1, or at least tries to start off well."""
        assert labels.shape[0] == self.n_spikes
        assert closest_neighbors.shape[1] <= self.n_candidates

        if labels.ndim == 1:
            torch.index_select(
                closest_neighbors,
                dim=0,
                index=labels,
                out=self.candidates[:, : self.n_candidates],
            )
        else:
            assert labels.shape == (self.n_spikes, self.n_candidates)
            self.candidates[:, : self.n_candidates] = labels
            invalid = labels < 0
            if invalid.any():
                (inv_i,) = invalid.any(dim=1).nonzero(as_tuple=True)
                self.candidates[:, : self.n_candidates][invalid] = -1
                self.candidates[inv_i, : self.n_candidates] = (
                    self.candidates[inv_i, : self.n_candidates]
                    .sort(descending=True)
                    .values
                )
                fisher_yates_replace(
                    self.rg,
                    len(closest_neighbors),
                    self.candidates[:, : self.n_candidates].numpy(),
                )
            if logger.isEnabledFor(DARTSORTVERBOSE):
                logger.dartsortverbose(
                    f"Candiate init had {invalid.sum()=} {invalid.shape=}"
                )
                logger.dartsortverbose(
                    f"Still I assure you that {torch.all(self.candidates[:, :self.n_candidates]>=0)=} and "
                    f"{torch.all(self.candidates[:, :self.n_candidates].sort(dim=1).values.diff(dim=1)>0)=}"
                )
        self._initialized = True

    def propose_candidates(self, unit_search_neighbors, indices=None):
        """Assumes invariant 1 and does not mess it up.

        Arguments
        ---------
        unit_search_neighbors: LongTensor (n_units, n_search)
        """
        assert unit_search_neighbors.shape[1] <= self.n_search
        assert self._initialized

        n_search = unit_search_neighbors.shape[1]

        full = indices is None
        if full:
            indices = slice(None)
        C = self.n_candidates
        n_search_total = C * n_search
        search_slice = slice(C, C + n_search_total)
        total = C + n_search_total + self.n_explore
        explore_slice = slice(C + n_search_total, total)
        n_neighbs = self.n_neighborhoods
        n_units = len(unit_search_neighbors)

        # if `full`, then this is all done in place.
        candidates = self.candidates[indices, :total]
        neighb_ids = self.neighborhood_ids[indices]
        del indices
        n_spikes = len(candidates)

        top = candidates[:, :C]

        # write the search units in place, if not batching
        if n_search:
            target = candidates[:, search_slice].view(n_spikes, C, n_search)
            assert (
                target.untyped_storage().data_ptr()
                == candidates.untyped_storage().data_ptr()
            )
            target[:] = unit_search_neighbors[top]

        # count to determine which units are relevant to each neighborhood
        # this is how "explore" candidates are suggested. the policy is strict:
        # just the top unit counts for each spike. this is to keep the lut smallish,
        # tho it is still huge, hopefully without making the search too bad...
        unit_neighborhood_counts = np.zeros((n_units, n_neighbs), dtype=np.int64)
        np.add.at(unit_neighborhood_counts, (top[:, 0], neighb_ids), 1)

        # which to explore?
        neighborhood_explore_units = units_overlapping_neighborhoods(
            unit_neighborhood_counts
        )
        neighborhood_explore_units = torch.from_numpy(neighborhood_explore_units)

        # sample the explore units and then write. how many units per neighborhood?
        # explore units are chosen per neighborhood. we start by figuring out how many
        # such units there are in each neighborhood.
        n_units_ = torch.tensor(n_units)[None].broadcast_to(n_neighbs, 1).contiguous()
        if self.n_explore:
            n_explore = torch.searchsorted(neighborhood_explore_units, n_units_).view(
                n_neighbs
            )
            n_explore = n_explore[neighb_ids]
            targs = integers_without_inner_replacement(
                self.rg, n_explore.numpy(), size=(n_spikes, self.n_explore)
            )
            if logger.isEnabledFor(DARTSORTVERBOSE):
                logger.dartsortverbose(
                    f"{n_explore.min()=} {n_explore.max()=} {targs.min()=} {targs.max()=}"
                )
            targs = torch.from_numpy(targs)
            explore = neighborhood_explore_units[neighb_ids[:, None], targs]
            explore[targs < 0] = -1
            candidates[:, explore_slice] = explore

        # update counts for the rest of units
        if candidates.shape[1] > 1:
            np.add.at(
                unit_neighborhood_counts, (candidates[:, 1:], neighb_ids[:, None]), 1
            )

        # replace duplicates with -1. TODO: replace quadratic algorithm with -1.
        erase_dups(candidates.numpy())
        if logger.isEnabledFor(DARTSORTVERBOSE):
            logger.dartsortverbose(
                f"I have maintained that {torch.all(self.candidates[:, :self.n_candidates]>=0)=} and"
                f"{torch.all(self.candidates[:, :self.n_candidates].sort(dim=1).values.diff(dim=1)>0)=}"
            )
        assert (candidates[:, : self.n_candidates] >= 0).all()

        return candidates, unit_neighborhood_counts
