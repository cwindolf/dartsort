from typing import Optional
import numpy as np
import torch
import threading
import joblib
from linear_operator import operators
from linear_operator.utils.cholesky import psd_safe_cholesky
import torch.nn.functional as F
from tqdm.auto import trange


from ..util.noise_util import EmbeddedNoise
from .stable_features import SpikeNeighborhoods, StableSpikeDataset
from ._truncated_em_helpers import (
    neighb_lut,
    TEBatchData,
    TEBatchResult,
    TEStepResult,
    _te_batch,
    units_overlapping_neighborhoods,
    woodbury_kl_divergence,
)


class SpikeTruncatedMixtureModel(torch.nn.Module):
    def __init__(
        self,
        data: StableSpikeDataset,
        noise: EmbeddedNoise,
        M: int = 0,
        n_candidates: int = 3,
        n_search: int = 5,
        n_explore: int = None,
        covariance_radius: Optional[float] = 500.0,
        random_seed=0,
        n_threads: int = 0,
        batch_size=2048,
        exact_kl=True,
    ):
        super().__init__()
        if n_search is None:
            n_search = n_candidates
        if n_explore is None:
            n_explore = n_search
        self.data = data
        self.noise = noise
        self.M = M
        self.covariance_radius = covariance_radius
        self.exact_kl = exact_kl
        train_indices, self.train_neighborhoods = self.data.neighborhoods("extract")
        self.n_spikes = train_indices.numel()
        self.processor = TruncatedExpectationProcessor(
            noise=noise,
            neighborhoods=self.train_neighborhoods,
            features=self.data._train_extract_features,
            n_candidates=n_candidates,
            random_seed=random_seed,
            n_threads=n_threads,
            batch_size=batch_size,
            covariance_radius=covariance_radius,
            pgeom=data.prgeom,
        )
        self.candidates = CandidateSet(
            neighborhoods=self.train_neighborhoods,
            n_candidates=n_candidates,
            n_search=n_search,
            n_explore=n_explore,
            random_seed=random_seed,
            device=self.data.device,
        )
        self.to(self.data.device)

    def set_parameters(
        self, labels, means, log_proportions, noise_log_prop, kl_divergences, bases=None
    ):
        """Parameters are stored padded with an extra channel."""
        # TODO: updating
        nu = means.shape[0]
        assert means.shape == (nu, self.noise.rank, self.noise.n_channels)
        self.register_buffer("means", F.pad(means, (0, 1)))

        assert log_proportions.shape == (nu,)
        self.register_buffer("log_proportions", log_proportions.clone())
        self.register_buffer("noise_log_prop", noise_log_prop.clone())
        self.register_buffer("_N", torch.zeros(nu + 1))

        assert kl_divergences.shape == (nu, nu)
        self.register_buffer(
            "kl_divergences",
            torch.asarray(kl_divergences, dtype=self.means.dtype, copy=True),
        )

        self.candidates.initialize_candidates(
            labels, self.search_sets(n_search=self.candidates.n_candidates)
        )

        self.M = 0 if bases is None else bases.shape[1]
        if self.M:
            assert bases is not None
            assert bases.shape == (nu, self.M, self.data.rank, self.data.n_channels)
            self.register_buffer("bases", F.pad(bases, (0, 1)))
        else:
            self.bases = None
        self.to(means.device)

    def initialize_parameters(self, train_labels):
        # run ppcas, or svds?
        # compute noise log liks? or does TEP want to do that? or, perhaps we do
        # that by setting noise to the explore unit in the first iteration, and
        # then storing?
        raise NotImplementedError

    def step(self, show_progress=False):
        print("step a", flush=True)
        search_neighbors = self.search_sets()
        print("step b", flush=True)
        unit_neighborhood_counts = self.candidates.propose_candidates(search_neighbors)
        print("step c", flush=True)
        self.processor.update(
            log_proportions=self.log_proportions,
            noise_log_prop=self.noise_log_prop,
            means=self.means,
            bases=self.bases,
            unit_neighborhood_counts=unit_neighborhood_counts,
        )
        print("step d", flush=True)
        result = self.processor.truncated_e_step(
            self.candidates.candidates,
            show_progress=show_progress,
            with_kl=not self.exact_kl,
        )
        print(f"{result.kl=}")
        print("step f", flush=True)
        self.means[..., :-1] = result.m
        if self.bases is not None:
            assert result.R is not None
            assert result.U is not None
            Uc = psd_safe_cholesky(result.U)
            W = torch.cholesky_solve(result.R.view(*result.U.shape[:-1], -1), Uc)
            self.bases[..., :-1] = W.view(self.bases[..., :-1].shape)

        self._N[0] = result.noise_N
        self._N[1:] = result.N
        lp = self._N.log() - self._N.sum().log()
        self.noise_log_prop.fill_(lp[0])
        self.log_proportions[:] = lp[1:]
        if self.exact_kl:
            print("hi exact")
            self.update_dkl()
        else:
            self.kl_divergences[:] = result.kl

        return dict(obs_elbo=result.obs_elbo)

    def update_dkl(self):
        W = self.bases
        if W is not None:
            W = W[..., :-1].reshape(len(W), self.M, -1).mT
        woodbury_kl_divergence(
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
        _, topkinds = torch.topk(self.kl_divergences, k=n_search, dim=1, largest=False)
        return topkinds


class TruncatedExpectationProcessor(torch.nn.Module):
    def __init__(
        self,
        noise: EmbeddedNoise,
        neighborhoods: SpikeNeighborhoods,
        features: torch.Tensor,
        n_candidates: int,
        batch_size: int = 2**10,
        n_threads: int = 0,
        random_seed: int = 0,
        precompute_invx=True,
        covariance_radius=None,
        pgeom=None,
    ):
        super().__init__()

        # initialize fixed noise-related arrays
        self.initialize_fixed(
            noise, neighborhoods, pgeom=pgeom, covariance_radius=covariance_radius
        )
        self.neighborhood_ids = neighborhoods.neighborhood_ids
        self.n_neighborhoods = neighborhoods.n_neighborhoods
        if features.isnan().any():
            print("Yeah. nans. Guess not possible to do in place?")
            self.features = features.nan_to_num()
        else:
            self.features = features.clone()
        if precompute_invx:
            self.precompute_invx()
        self.batch_size = batch_size
        self.n_candidates = n_candidates

        # M is updated by self.update() when a basis is assigned here.
        self.M = 0

        # place to store thread-local work and output arrays
        # TODO
        self._locals = threading.local()

        # thread pool
        self._rg = np.random.default_rng(random_seed)
        self.pool = joblib.Parallel(
            n_jobs=n_threads or 1, backend="threading", return_as="generator_unordered"
        )

    @property
    def device(self):
        assert hasattr(self, "Coo_logdet")
        return self.Coo_logdet.device

    def truncated_e_step(
        self, candidates, with_kl=True, show_progress=False
    ) -> TEStepResult:
        """E step of truncated VEM

        Will update candidates[:, :self.n_candidates] in place.
        """
        # sufficient statistics
        dev = self.device
        m = torch.zeros(
            (self.n_units, self.rank, self.nc), device=dev, dtype=torch.double
        )
        N = torch.zeros((self.n_units,), device=dev, dtype=torch.double)
        dkl = None
        if with_kl:
            dkl = torch.zeros(
                (self.n_units, self.n_units), device=dev, dtype=torch.double
            )
        ncc = torch.zeros((self.n_units, self.n_units), device=dev, dtype=torch.double)
        noise_N = torch.tensor(0.0, device=dev, dtype=torch.double)
        obs_elbo = torch.tensor(0.0, device=dev, dtype=torch.double)
        R = U = None
        if self.M:
            R = torch.zeros((self.n_units, self.M, self.rank, self.nc), device=dev)
            U = torch.zeros((self.n_units, self.M, self.M), device=dev)

        # will be updated...
        top_candidates = candidates[:, : self.n_candidates]

        # do we need to initialize the noise log likelihoods?
        first_run = not hasattr(self, "noise_logliks")
        noise_logliks = None
        if first_run:
            noise_logliks = torch.empty((len(self.features),), device=dev)

        # run loop
        jobs = (
            self._te_step_job(candidates, batchix, with_kl)
            for batchix in self.batches(show_progress=show_progress)
        )
        count = 0
        for result in self.pool(jobs):
            assert result is not None  # why, pyright??

            # weighted welford
            N_N0 = N.div((N + result.N).clamp(min=1.0))[:, None, None]
            N += result.N
            N_N1 = result.N.div(N.clamp(min=1.0))[:, None, None]
            m += (result.m[:, :, :-1] - m).mul_(N_N1)
            # m *= N_N0
            # m += result.m[:, :, :-1] * N_N1
            noise_N += result.noise_N

            # running avg for elbo/n
            count += len(result.candidates)
            obs_elbo += (result.obs_elbo - obs_elbo) * (len(result.candidates) / count)

            if with_kl:
                ncc += result.ncc
                dkl += (result.dkl - dkl).div_(ncc.clamp(min=1.0))

            if self.M:
                assert R is not None
                assert U is not None
                R += (result.R[..., :-1] - R).mul_(N_N1[..., None])
                U += (result.U - U).mul_(N_N1)
                # R += result.R
                # U += result.U

            # could do these in the threads? unless shuffled.
            top_candidates[result.indices] = result.candidates
            if first_run:
                assert noise_logliks is not None  # pyright...
                noise_logliks[result.indices] = result.noise_lls

        if first_run:
            self.register_buffer("noise_logliks", noise_logliks)

        if with_kl:
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
        )

    @joblib.delayed
    def _te_step_job(self, candidates, batch_indices, with_kl):
        return self.process_batch(
            candidates=candidates,
            batch_indices=batch_indices,
            with_stats=True,
            with_obs_elbo=True,
            with_kl=with_kl,
        )

    def batches(self, shuffle=False, show_progress=False):
        n = len(self.features)
        if shuffle:
            shuf = self._rg.permutation(n)
            for sl in self.batches(show_progress=show_progress):
                ix = shuf[sl]
                ix.sort()
                yield ix
        else:
            if show_progress:
                starts = trange(
                    0, n, self.batch_size, desc="Batches", smoothing=0, mininterval=0.2
                )
            else:
                starts = range(0, n, self.batch_size)

            for batch_start in starts:
                batch_end = min(n, batch_start + self.batch_size)
                yield slice(batch_start, batch_end)

    def process_batch(
        self,
        batch_indices,
        candidates,
        with_grads=False,
        with_stats=False,
        with_kl=False,
        with_elbo=False,
        with_obs_elbo=False,
    ) -> TEBatchResult:
        bd = self.load_batch(candidates=candidates, batch_indices=batch_indices)
        return self.te_batch(
            bd=bd,
            with_grads=with_grads,
            with_stats=with_stats,
            with_kl=with_kl,
            with_elbo=with_elbo,
            with_obs_elbo=with_obs_elbo,
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
        Coo = [operators.CholLinearOperator(C.cholesky()) for C in Coo]
        Coo_logdet = torch.tensor([C.logdet() for C in Coo], device=device)
        Coo_inv = [C.inverse().to_dense() for C in Coo]
        Cooinv_Com = [Co @ Cm for Co, Cm in zip(Coo_inv, Com)]

        # figure out dimensions
        nc_obs = neighborhoods.channel_counts
        self.nc_obs = nco = nc_obs.max().to(int)
        self.nc_miss = ncm = max(map(len, missing_chans))

        # understand channel subsets
        # these arrays are used to scatter into D-shaped dims.
        # actually... maybe into D+1-shaped dims, so that we can use
        # D as the invalid indicator
        self.register_buffer("obs_ix", neighborhoods.neighborhoods.to(device))
        self.register_buffer(
            "miss_ix",
            torch.full(
                (neighborhoods.n_neighborhoods, ncm), fill_value=nc, device=device
            ),
        )
        for ni in range(neighborhoods.n_neighborhoods):
            self.miss_ix[ni, : missing_chans[ni].numel()] = missing_chans[ni]

        # allocate buffers
        self.register_buffer("Coo_logdet", Coo_logdet)
        self.register_buffer("nobs", R * nc_obs)
        self.register_buffer(
            "Coo_inv",
            torch.zeros(neighborhoods.n_neighborhoods, R * nco, R * nco, device=device),
        )
        self.register_buffer(
            "Cooinv_Com",
            torch.zeros(neighborhoods.n_neighborhoods, R * nco, R * ncm, device=device),
        )

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

    def precompute_invx(self):
        # precomputed Cooinv_x
        self.Cooinv_x = self.features.clone().view(len(self.features), -1)
        for ni in trange(self.n_neighborhoods, desc="invx"):
            (mask,) = (self.neighborhood_ids == ni).nonzero(as_tuple=True)
            self.Cooinv_x[mask] @= self.Coo_inv[ni].to(self.Cooinv_x)

    def update(
        self,
        log_proportions,
        means,
        noise_log_prop,
        bases=None,
        unit_neighborhood_counts=None,
        batch_size=1024,
    ):
        print(f"update {means.shape=}", flush=True)
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
            print(f"new {lut.shape=} {self.lut_units.shape=} {self.lut_neighbs.shape=}")
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

        # mean stuff
        nu = mu[self.lut_units[:, None], :, obs_ix].permute(0, 2, 1)
        tnu = mu[self.lut_units[:, None], :, miss_ix].permute(0, 2, 1)
        self.register_buffer("nu", nu)
        self.register_buffer("tnu", tnu)
        Cooinv_nu = torch.empty(
            (len(self.lut_neighbs), self.Coo_inv.shape[1], 1), device=nu.device
        )
        for bs in range(0, len(self.lut_neighbs), batch_size):
            be = min(len(self.lut_neighbs), bs + batch_size)
            Cooinv = self.Coo_inv[self.lut_neighbs[bs:be]]
            torch.bmm(Cooinv, nu[bs:be].reshape(be - bs, -1, 1), out=Cooinv_nu[bs:be])
        self.register_buffer("Cooinv_nu", Cooinv_nu[..., 0])
        # self.register_buffer(
        #     "Cooinv_nu", torch.bmm(Cooinv, nu.reshape(nlut, -1, 1))[..., 0]
        # )

        # basis stuff
        if not self.M:
            print("update bail 0", flush=True)
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
        Cobsinv_WobsT = torch.empty(
            (len(self.lut_neighbs), self.Coo_inv.shape[1], self.M), device=Wobs.device
        )
        for bs in range(0, len(self.lut_neighbs), batch_size):
            be = min(len(self.lut_neighbs), bs + batch_size)
            Cooinv = self.Coo_inv[self.lut_neighbs[bs:be]]
            torch.bmm(Cooinv, Wobs[bs:be].mT, out=Cobsinv_WobsT[bs:be])
        self.register_buffer("Cobsinv_WobsT", Cobsinv_WobsT)
        # self.register_buffer("Cobsinv_WobsT", torch.bmm(Cooinv, Wobs.mT))

        cap = torch.bmm(Wobs, self.Cobsinv_WobsT)
        cap.diagonal(dim1=-2, dim2=-1).add_(1.0)
        assert cap.shape == (nlut, self.M, self.M)
        cap = operators.CholLinearOperator(psd_safe_cholesky(cap))

        # matrix determinant lemma
        noise_logdets = self.Coo_logdet[self.lut_neighbs]
        cap_logdets = cap.logdet()
        self.register_buffer("obs_logdets", noise_logdets + cap_logdets)

        # this is used in the log lik Woodbury, and it's also the posterior
        # covariance of the ppca embedding (usually I call that T).
        self.register_buffer("inv_cap", cap.inverse().to_dense())

        # gizmo matrix used in a certain part of the "imputation"
        print(f"{self.Cooinv_Com.shape=} {Wobs.shape=}")
        W_WCC = Wmiss
        del Wmiss
        for bs in range(0, len(W_WCC), batch_size):
            be = min(len(W_WCC), bs + batch_size)
            W_WCC[bs:be].baddbmm_(
                Wobs[bs:be], self.Cooinv_Com[self.lut_neighbs[bs:be]], alpha=-1
            )
        self.register_buffer("W_WCC", W_WCC)

    # long method... putting it in another file to keep the flow
    # clear in this one. this method is the one that does all the math.
    te_batch = _te_batch

    def load_batch(
        self,
        batch_indices,
        candidates,
        with_grads=False,
        with_stats=False,
        with_kl=False,
        with_elbo=False,
        with_obs_elbo=False,
    ) -> TEBatchData:
        candidates = candidates[batch_indices].to(self.device)
        n, c = candidates.shape
        x = self.features[batch_indices].view(n, -1).to(self.device)
        Cooinv_x = None
        if hasattr(self, "Cooinv_x"):
            Cooinv_x = self.Cooinv_x[batch_indices].to(self.device)
        neighborhood_ids = self.neighborhood_ids[batch_indices]

        # todo: index_select into buffers?
        lut_ixs = self.lut[candidates, neighborhood_ids[:, None]]

        # neighborhood-dependent
        Coo_logdet = self.Coo_logdet[neighborhood_ids]
        Coo_inv = self.Coo_inv[neighborhood_ids]
        Cooinv_Com = self.Cooinv_Com[neighborhood_ids]
        obs_ix = self.obs_ix[neighborhood_ids]
        miss_ix = self.miss_ix[neighborhood_ids]
        nobs = self.nobs[neighborhood_ids]

        # unit-dependent
        log_proportions = self.log_proportions[candidates]

        # neighborhood + unit-dependent
        nu = self.nu[lut_ixs].reshape(n, c, -1)
        tnu = self.tnu[lut_ixs].reshape(n, c, -1)
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
        if hasattr(self, "noise_logliks"):
            noise_lls = self.noise_logliks[batch_indices]

        return TEBatchData(
            indices=batch_indices,
            n=len(x),
            #
            x=x,
            candidates=candidates,
            #
            Coo_logdet=Coo_logdet,
            Coo_inv=Coo_inv,
            Cooinv_Com=Cooinv_Com,
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
            #
            Cooinv_x=Cooinv_x,
        )


class CandidateSet:
    def __init__(
        self,
        neighborhoods,
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
        self.neighborhood_ids = neighborhoods.neighborhood_ids
        self.n_neighborhoods = neighborhoods.n_neighborhoods
        self.n_spikes = self.neighborhood_ids.numel()
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

    def propose_candidates(self, unit_search_neighbors):
        """Assumes invariant 1 and does not mess it up.

        Arguments
        ---------
        unit_search_neighbors: LongTensor (n_units, n_search)
        """
        C = self.n_candidates
        n_search = C * self.n_search
        n_neighbs = self.n_neighborhoods
        n_units = len(unit_search_neighbors)

        top = self.candidates[:, :C]

        # write the search units in place
        target = self.candidates[:, C : C + n_search].view(
            self.n_spikes, C, self.n_search
        )
        assert (
            target.untyped_storage().data_ptr()
            == self.candidates.untyped_storage().data_ptr()
        )
        target[:] = unit_search_neighbors[top]

        # count to determine which units are relevant to each neighborhood
        # this is how "explore" candidates are suggested. the policy is strict:
        # just the top unit counts for each spike. this is to keep the lut smallish,
        # tho it is still huge, hopefully without making the search too bad...
        unit_neighborhood_counts = np.zeros((n_units, n_neighbs), dtype=int)
        np.add.at(
            unit_neighborhood_counts, (self.candidates[:, 0], self.neighborhood_ids), 1
        )

        # which to explore?
        neighborhood_explore_units = units_overlapping_neighborhoods(
            unit_neighborhood_counts
        )
        neighborhood_explore_units = torch.from_numpy(neighborhood_explore_units)

        # sample the explore units and then write. how many units per neighborhood?
        # explore units are chosen per neighborhood. we start by figuring out how many
        # such units there are in each neighborhood.
        n_units_ = torch.tensor(n_units)[None].broadcast_to(n_neighbs, 1).contiguous()
        n_explore = torch.searchsorted(neighborhood_explore_units, n_units_).view(
            n_neighbs
        )
        n_explore = n_explore[self.neighborhood_ids]
        targs = self.rg.integers(
            n_explore.numpy()[:, None], size=(self.n_spikes, self.n_explore)
        )
        targs = torch.from_numpy(targs)
        self.candidates[:, -self.n_explore :] = neighborhood_explore_units[
            self.neighborhood_ids[:, None], targs
        ]

        # update counts for the rest of units
        np.add.at(
            unit_neighborhood_counts,
            (self.candidates[:, 1:], self.neighborhood_ids[:, None]),
            1,
        )
        return unit_neighborhood_counts
