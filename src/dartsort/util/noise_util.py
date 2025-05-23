import warnings
from logging import getLogger

import h5py
import linear_operator
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from linear_operator import operators
from scipy.fftpack import next_fast_len
from scipy.spatial.distance import pdist, squareform
from sklearn.covariance import GraphicalLassoCV, graphical_lasso
from tqdm.auto import trange

from ..util import more_operators, spiketorch
from ..util.logging_util import DARTSORTDEBUG, DARTSORTVERBOSE

logger = getLogger(__name__)


class FullNoise(torch.nn.Module):
    """Do not use this, it's just for comparison to the others."""

    def __init__(self, std, vt, nt, nc):
        super().__init__()
        self.std = std
        self.vt = vt
        self.nt = nt
        self.nc = nc

    def full_covariance(self):
        return (self.vt.T * self.std.square()) @ self.vt

    def simulate(self, size=1, t=None, generator=None):
        device = self.spatial_std.device
        assert t == self.nt
        samples = torch.randn(size, t * self.nc, generator=generator, device=device)
        samples = torch.einsum(
            "ni,i,ij->nj",
            samples,
            self.std,
            self.vt,
        )
        return samples.view(size, self.nt, self.nc)

    @classmethod
    def estimate(cls, snippets):
        snippets = torch.asarray(snippets)
        snippets = snippets.to(torch.promote_types(snippets.dtype, torch.float))

        n, t, c = snippets.shape

        x = snippets.view(n, t * c)
        # u_full, sing, vt_full = torch.linalg.svd(x, full_matrices=False)
        # std = sing / torch.sqrt(torch.tensor(n - 1).to(snippets))

        cov = torch.cov(x.T)
        eigvals, v = torch.linalg.eigh(cov)
        std = eigvals.sqrt()
        vt = v.T.contiguous()

        return cls(std, vt)


class FactorizedNoise(torch.nn.Module):
    """Spatial/temporal factorized noise. See .estimate()."""

    def __init__(self, spatial_std, vt_spatial, temporal_std, vt_temporal):
        super().__init__()
        self.spatial_std = spatial_std
        self.vt_spatial = vt_spatial
        self.temporal_std = temporal_std
        self.vt_temporal = vt_temporal

    def full_covariance(self):
        spatial_cov = self.spatial_std[:, None] * self.vt_spatial
        spatial_cov = spatial_cov.T @ spatial_cov
        temporal_cov = self.temporal_std[:, None] * self.vt_temporal
        temporal_cov = temporal_cov.T @ temporal_cov
        return torch.kron(temporal_cov, spatial_cov)

    def simulate(self, size=1, t=None, generator=None):
        c = self.spatial_std.numel()
        t_ = self.temporal_std.numel()
        device = self.spatial_std.device
        if t is None:
            t = t_
        assert t == t_
        samples = torch.randn(size, t, c, generator=generator, device=device)
        return torch.einsum(
            "ntc,t,tu,c,cd->nud",
            samples,
            self.temporal_std,
            self.vt_temporal,
            self.spatial_std,
            self.vt_spatial,
        )

    @classmethod
    def estimate(cls, snippets):
        """Estimate factorized spatiotemporal noise covariance

        If snippets has shape (n, t, c), the full t*c x t*c covariance is
        too big. Let's say it factorizes as
            spatial_cov[c, c'] * temporal_cov[t, t'].
        This is not identifiable, but it's okay if we assume that the global
        scale falls into the spatial part (say). So, we first estimate the
        spatial cov, then apply spatial whitening to the snippets, and finally
        estimate the temporal cov.

        This is an SVD reimplementation of a Yass fn which used covariance matrices.

        Arguments
        ---------
        snippets : torch.Tensor
            (n, t, c) array of noise snippets. If you don't want spikes in there,
            this function won't help you with that.
        """
        snippets = torch.asarray(snippets)
        snippets = snippets.to(torch.promote_types(snippets.dtype, torch.float))

        n, t, c = snippets.shape
        sqrt_nt_minus_1 = torch.tensor(n * t - 1, dtype=snippets.dtype).sqrt()
        sqrt_nc_minus_1 = torch.tensor(n * c - 1, dtype=snippets.dtype).sqrt()
        assert n * t > c**2
        assert n * c > t**2

        # estimate spatial covariance
        x_spatial = snippets.view(n * t, c)
        u_spatial, spatial_sing, vt_spatial = torch.linalg.svd(
            x_spatial, full_matrices=False
        )
        spatial_std = spatial_sing / sqrt_nt_minus_1

        # extract whitened temporal snips
        x_temporal = u_spatial.view(n, t, c).permute(0, 2, 1).reshape(n * c, t)
        x_temporal.mul_(sqrt_nt_minus_1)
        _, temporal_sing, vt_temporal = torch.linalg.svd(
            x_temporal, full_matrices=False
        )
        del _
        temporal_std = temporal_sing / sqrt_nc_minus_1

        return cls(spatial_std, vt_spatial, temporal_std, vt_temporal)


class WhiteNoise(torch.nn.Module):
    """White noise to mimic the StationaryFactorizedNoise for use in sims."""

    def __init__(self, n_channels, scale=1.0):
        super().__init__()
        self.n_channels = n_channels
        self.scale = scale

    def simulate(self, size=1, t=None, generator=None, chunk_t=None):
        assert t is not None
        x = torch.randn(size, t, self.n_channels, generator=generator)
        if self.scale != 1.0:
            x *= self.scale
        return x


class StationaryFactorizedNoise(torch.nn.Module):
    def __init__(self, spatial_std, vt_spatial, kernel_fft, block_size, t):
        super().__init__()
        self.spatial_std = spatial_std
        self.vt_spatial = vt_spatial
        self.kernel_fft = kernel_fft
        self.block_size = block_size
        self.t = t

    def spatial_cov(self):
        rt = self.spatial_std * self.vt_spatial.T
        return rt @ rt.T

    def whiten(self, snippet):
        wsnip = snippet @ (self.vt_spatial / self.spatial_std[:, None])
        wsnip = spiketorch.single_inv_oaconv1d(
            wsnip.T,
            s2=self.t,
            f2=1 / self.kernel_fft,
            block_size=self.block_size,
            norm="ortho",
        )
        # zca
        wsnip = wsnip.T @ self.vt_spatial.T
        return wsnip

    def simulate(self, size=1, t=None, generator=None, chunk_t=None):
        """Simulate stationary factorized noise

        White noise is the same in FFT space or not. So we start there,
        filter, inverse FFT, then spatially dewhiten. The filtering + inverse
        FFT are basically overlap-add convolution, and thanks to Chris Langfield
        for help with overlap-add.
        """
        if t is None:
            t = self.t
        assert t >= self.t
        c = self.spatial_std.numel()
        device = self.spatial_std.device

        if chunk_t:
            out = torch.zeros((t + self.t, c))
            taper = torch.linspace(0.0, 1.0, steps=self.t)
            for bs in trange(0, t, chunk_t):
                chunk = self.simulate(t=chunk_t + self.t, generator=generator)[0]
                if bs > 0:
                    chunk[: self.t] *= taper[:, None]
                chunk[-self.t :] *= (1 - taper)[:, None]
                out[bs : bs + chunk_t + self.t] += chunk.cpu()
            return out[None, : -self.t]

        # need extra room at the edges to do valid convolution
        t_padded = t + self.t - 1
        noise = torch.randn(size * c, t_padded, generator=generator, device=device)
        noise = spiketorch.single_inv_oaconv1d(
            noise,
            s2=self.t,
            f2=self.kernel_fft,
            block_size=self.block_size,
            norm="ortho",
        )
        noise = noise.view(size, c, t)
        spatial_part = self.spatial_std[:, None] * self.vt_spatial
        return torch.einsum("nct,cd->ntd", noise, spatial_part)

    @classmethod
    def estimate(cls, snippets):
        """Estimate factorized temporally stationary noise

        When simulating long samples, the above approach can't scale. Here let
            spatial_cov[c, c'] * temporal_cov[t - t'].

        Now, remember that stationary kernels diagonalize in Fourier basis. FFT
        is the eigh of circulant things. So we use an FFT based version of the
        SVD algorithm above.

        Arguments
        ---------
        snippets : torch.Tensor
            (n, t, c) array of noise snippets. If you don't want spikes in there,
            this function won't help you with that.
        """
        snippets = torch.asarray(snippets)
        snippets = snippets.to(torch.promote_types(snippets.dtype, torch.float))

        n, t, c = snippets.shape
        sqrt_nt_minus_1 = torch.tensor(n * t - 1, dtype=snippets.dtype).sqrt()
        assert n * t > c**2
        assert n * c > t

        # estimate spatial covariance
        x_spatial = snippets.view(n * t, c)
        u_spatial, spatial_sing, vt_spatial = torch.linalg.svd(
            x_spatial, full_matrices=False
        )
        spatial_std = spatial_sing / sqrt_nt_minus_1

        # extract whitened temporal snips
        x_temporal = u_spatial.view(n, t, c).permute(0, 2, 1).reshape(n * c, t)
        x_temporal.mul_(sqrt_nt_minus_1)

        # estimate stationary temporal covariance via FFT
        block_size = next_fast_len(t)
        xt_fft = torch.fft.rfft(x_temporal, n=block_size, norm="ortho")
        kernel_fft = (xt_fft * xt_fft.conj()).mean(0).sqrt_()

        return cls(spatial_std, vt_spatial, kernel_fft, block_size, t)

    def unit_false_positives(
        self,
        low_rank_templates,
        min_threshold=5.0,
        radius=10,
        generator=None,
        size=100,
        t=4096,
    ):
        """Run template detection in noise residual to check for false positives

        Given a collection of (SVD factorized) templates, compute their template
        matching objective in simulated recording snippets drawn according to
        this noise object's distribution. Then, detect peaks (with a refractory
        condition) to estimate the false positive count.

        Arguments
        ---------
        low_rank_templates : LowRankTemplates
        min_threshold : float
            The bare minimum deconv objective
        radius : int
            Refractory period
        generator : torch.Generator
            For reproducibility.
        size : int
            Number of recording snipppets
        t : int
            Length of each recording snippet

        Returns
        -------
        total_samples : int
            Total number of valid temporal samples
        fp_dataframe : pd.DataFrame
            With columns `units` and `scores`
        """
        from ..detect import detect_and_deduplicate

        singular = torch.asarray(
            low_rank_templates.singular_values,
            device=self.spatial_std.device,
            dtype=self.spatial_std.dtype,
        )
        spatial = torch.asarray(
            low_rank_templates.spatial_components,
            device=self.spatial_std.device,
            dtype=self.spatial_std.dtype,
        )
        spatial_singular = singular.unsqueeze(-1) * spatial
        temporal = torch.asarray(
            low_rank_templates.temporal_components,
            device=self.spatial_std.device,
            dtype=self.spatial_std.dtype,
        )
        normsq = singular.square().sum(1)
        negnormsq = normsq.neg().unsqueeze(1)
        nu, nt = temporal.shape[:2]
        obj = None  # var for reusing buffers
        units = []
        scores = []
        for j in trange(size, desc="False positives"):
            # note: simulating with padding so that the valid conv has length t.
            sample = self.simulate(t=t + nt - 1, generator=generator)[0].T
            assert sample.isfinite().all()
            obj = spiketorch.convolve_lowrank(
                sample,
                spatial_singular,
                temporal,
                out=obj,
                padding=0,
            )
            assert obj.isfinite().all()
            assert obj.shape == (nu, t)
            obj = torch.add(negnormsq, obj, alpha=2.0, out=obj)
            assert obj.isfinite().all()

            # find peaks...
            peak_times, peak_units, peak_energies = detect_and_deduplicate(
                obj.T,
                min_threshold,
                peak_sign="pos",
                dedup_temporal_radius=radius,
                return_energies=True,
            )
            assert peak_energies.isfinite().all()
            units.append(peak_units.numpy(force=True))
            scores.append(peak_energies.numpy(force=True))

        total_samples = size * t
        data = dict(units=np.concatenate(units), scores=np.concatenate(scores))
        fp_dataframe = pd.DataFrame(data)

        return dict(
            total_samples=total_samples,
            fp_dataframe=fp_dataframe,
            normsq=normsq.numpy(force=True),
        )


class EmbeddedNoise(torch.nn.Module):
    """Handles computations related to noise in TPCA space.

    Can have a couple of kinds of mean. mean_kind == ...
      - "zero": noise was already centered, my mean is 0
      - "by_rank": same mean on all channels
      - "full": value per rank, chan

    Default is zero, because we usually do not do "centering" in our TPCA
    (i.e., it is just linear not affine). Since the channels are highpassed,
    they have mean 0, and still do in TPCA space.

    And cov_kind = ...
      - "scalar": one global variance
      - "diagonal_by_rank": same variance across chans, varies by rank
      - "diagonal": value per rank, chan
      - "factorized": kronecker prod of dense rank and chan factors
      - "factorized_by_rank": same, but chan factor varies by rank
      - "factorized_rank_diag" : factorized, but rank factor is diagonal
      - "factorized_by_rank_rank_diag" : factorized_by_rank, but rank factor is diagonal
         this one is block diagonal and therefore nicer than factorized_by_rank.
      - "full": not generally practical.
    """

    def __init__(
        self,
        rank,
        n_channels,
        mean_kind="zero",
        cov_kind="factorized_by_rank_rank_diag",
        mean=None,
        global_std=None,
        rank_std=None,
        full_std=None,
        rank_vt=None,
        channel_std=None,
        channel_vt=None,
        full_cov=None,
        zero_radius=None,
    ):
        super().__init__()
        self.rank = rank
        self.n_channels = n_channels
        self.mean_kind = mean_kind
        self.cov_kind = cov_kind
        self.D = rank * n_channels
        self.zero_radius = zero_radius

        self.register_buffer("chans_arange", torch.arange(n_channels))
        if mean is not None:
            self.register_buffer("mean", mean)
        self.register_buffer("global_std", global_std)
        if rank_std is not None:
            self.register_buffer("rank_std", rank_std)
        if channel_std is not None:
            self.register_buffer("channel_std", channel_std)
        if full_std is not None:
            self.register_buffer("full_std", full_std)

        if rank_vt is not None:
            self.register_buffer("rank_vt", rank_vt)
        if channel_vt is not None:
            self.register_buffer("channel_vt", channel_vt)

        if full_cov is not None:
            self.register_buffer("full_cov", full_cov)

        # precompute stuff
        self._full_cov = None
        self._full_covinvcov = None
        self._full_inverse = None
        self._logdet = None
        self.register_buffer("mean_full", self.mean_rc().clone().detach())
        self.cache = {}

    @property
    def logdet(self):
        if self._logdet is None:
            self.marginal_covariance()
        return self._logdet

    @property
    def device(self):
        return self.global_std.device

    def mean_rc(self):
        """Return noise mean as a rank x channels tensor"""
        shape = self.rank, self.n_channels
        if self.mean_kind == "zero":
            return torch.zeros(shape)
        elif self.mean_kind == "by_rank":
            return self.mean[:, None].broadcast_to(shape).contiguous()
        elif self.mean_kind == "full":
            return self.mean

    def marginal_mean(self):
        """Return noise mean as a rank x channels tensor"""
        shape = self.rank, self.n_channels
        if self.mean_kind == "zero":
            return torch.zeros(shape)
        if self.mean_kind == "by_rank":
            return self.mean[:, None].broadcast_to(shape).contiguous()
        if self.mean_kind == "full":
            return self.mean
        assert False

    def whitener(self, channels=slice(None)):
        cov = self.marginal_covariance(channels=channels)
        chol = cov.cholesky()
        chans_eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        whitener = torch.linalg.solve_triangular(chol, chans_eye, upper=False)
        return whitener.reshape(cov.shape)

    def whiten(self, data, channels=slice(None)):
        assert self.mean_kind == "zero"
        cov = self.marginal_covariance(channels=channels)
        assert data.ndim == 3
        data = data.reshape(len(data), -1)
        chol = cov.cholesky().to_dense()
        res = torch.linalg.solve_triangular(chol, data.T, upper=False)
        res = res.T.unsqueeze(2)

        # contour integral quad with lanczos is too fancy here...
        # res = linear_operator.sqrt_inv_matmul(cov, data.unsqueeze(2))
        assert res.ndim == 3 and res.shape == (*data.shape, 1)
        return res

    def full_dense_cov(self, device=None):
        if self._full_cov is None:
            self.marginal_covariance(device=device)
        if device is not None:
            self._full_cov_dense = self._full_cov_dense.to(device)
        return self._full_cov_dense

    def full_covinvcov(self, device=None):
        if self._full_cov is None:
            self.marginal_covariance(device=device)
        if self._full_covinvcov is None:
            self._full_covinvcov = self._full_cov.solve(self.full_dense_cov())
        if device is not None:
            self._full_covinvcov = self._full_covinvcov.to(device)
        return self._full_covinvcov

    def full_inverse(self, device=None):
        if self._full_cov is None:
            self.marginal_covariance(device=device)
        if self._full_inverse is None:
            self._full_inverse = self._full_cov.inverse().to_dense()
        if device is not None:
            self._full_inverse = self._full_inverse.to(device)
        return self._full_inverse

    def marginal_covariance(
        self, channels=slice(None), cache_prefix=None, cache_key=None, device=None
    ):
        if device is not None:
            device = torch.device(device)
        if cache_key is not None and torch.is_tensor(cache_key):
            cache_key = cache_key.item()
        if cache_prefix is not None:
            cache_key = (cache_prefix, cache_key)
        if cache_key is not None and cache_key in self.cache:
            res = self.cache[cache_key]
            if device is not None and device != res.device:
                res = res.to(device)
                self.cache[cache_key] = res
            # TODO: why is this necessary? I thought that calling solve would cache this.
            if not hasattr(res, "_memoize_cache"):
                res.cholesky()
            return res
        if channels is None or channels == slice(None):
            if self._full_cov is None:
                fcov = self._marginal_covariance()
                self._full_cov_dense = fcov.to_dense()
                if not isinstance(fcov, operators.ConstantDiagLinearOperator):
                    fcov = operators.CholLinearOperator(fcov.cholesky())
                self._full_cov = fcov
                self._logdet = self._full_cov.logdet()
            if device is not None and device != self._full_cov.device:
                self._full_cov = self._full_cov.to(device)
                self._logdet = self._logdet.to(device)
            return self._full_cov
        cov = self._marginal_covariance(channels)
        if device is not None and device != cov.device:
            cov = cov.to(device)
        if cache_key is not None:
            self.cache[cache_key] = cov
        return cov

    def offdiag_covariance(
        self, channels_left=slice(None), channels_right=slice(None), device=None
    ):
        odc = self._marginal_covariance(
            channels=channels_right, channels_left=channels_left
        )
        if device is not None:
            odc = odc.to(device)
        return odc

    def _marginal_covariance(self, channels=slice(None), channels_left=None):
        channels = self.chans_arange[channels]
        have_left = channels_left is not None
        if channels_left is None:
            channels_left = channels
        nc = channels.numel()
        ncl = channels_left.numel()

        if have_left:
            if self.cov_kind in ("scalar", "diagonal_by_rank", "diagonal"):
                # if torch.isin(channels_left.to(channels), channels).any():
                #     raise ValueError(
                #         "Don't know how to get non-marginal covariance block "
                #         f"for cov_kind={self.cov_kind} when the block overlaps "
                #         "with the diagonal."
                #     )
                sz = self.rank * ncl, self.rank * nc
                return operators.ZeroLinearOperator(*sz, dtype=self.global_std.dtype)

        if self.cov_kind == "scalar":
            eye = operators.IdentityLinearOperator(self.rank * nc, device=self.device)
            return self.global_std.square() * eye

        if self.cov_kind == "diagonal_by_rank":
            rank_diag = operators.DiagLinearOperator(self.rank_std**2)
            chans_eye = operators.IdentityLinearOperator(nc, device=self.device)
            return torch.kron(rank_diag, chans_eye)

        if self.cov_kind == "diagonal":
            chans_std = self.full_std[:, channels]
            return operators.DiagLinearOperator(chans_std**2)

        if self.cov_kind == "full":
            marg_cov = self.full_cov[:, channels_left][..., channels]
            r = marg_cov.shape[0]
            marg_cov = marg_cov.reshape(r * ncl, r * nc)
            return linear_operator.to_linear_operator(marg_cov)

        if self.cov_kind in "factorized":
            rank_root = self.rank_vt.T * self.rank_std
            rank_cov = rank_root @ rank_root.T
            chan_root = self.channel_vt.T[channels] * self.channel_std
            chan_root_right = chan_root.T
            if have_left:
                chan_root = self.channel_vt.T[channels_left] * self.channel_std
            chan_cov = chan_root @ chan_root_right
            return operators.KroneckerProductLinearOperator(rank_cov, chan_cov)

        if self.cov_kind == "factorized_rank_diag":
            rank_cov = operators.DiagLinearOperator(self.rank_std.square())
            chan_root = self.channel_vt.T[channels] * self.channel_std
            chan_root_right = chan_root.T
            if have_left:
                chan_root = self.channel_vt.T[channels_left] * self.channel_std
            chan_cov = chan_root @ chan_root_right
            return torch.kron(rank_cov, chan_cov)

        if self.cov_kind == "factorized_by_rank_rank_diag":
            rank_std = self.rank_std.view(self.rank, 1, 1)
            chans_std = self.channel_std[:, :, None]  # no slice here!
            rc_std = rank_std * chans_std
            chans_vt = self.channel_vt[:, :, channels]
            chan_rootl = chan_root = rc_std * chans_vt
            if have_left:
                chans_vtl = self.channel_vt[:, :, channels_left]
                chan_rootl = rc_std * chans_vtl
            blocks = torch.bmm(chan_rootl.mT, chan_root)
            blocks = linear_operator.to_linear_operator(blocks)
            if have_left:
                return more_operators.NonSquareBlockLinearOperator(blocks)
            return operators.BlockDiagLinearOperator(blocks)

        assert False

    @classmethod
    def estimate(
        cls,
        snippets,
        mean_kind="zero",
        cov_kind="scalar",
        shrinkage=5e-3,
        glasso_alpha: int | float | None = None,
        eps=1e-4,
        zero_radius: float | None = None,
        rgeom=None,
    ):
        """Factory method to estimate noise model from TPCA snippets

        Arguments
        ---------
        snippets : torch.Tensor
            (n, rank, c) array of tpca-embedded noise snippets
            missing values are okay, indicate by NaN please
        """
        n, rank, n_channels = snippets.shape
        init_kw = dict(
            rank=rank, n_channels=n_channels, mean_kind=mean_kind, cov_kind=cov_kind
        )
        x = torch.asarray(snippets)
        x = x.to(torch.promote_types(x.dtype, torch.float))
        if zero_radius is not None or shrinkage:
            assert cov_kind.startswith("factorized")
            assert rgeom is not None

        # estimate mean and center data
        if mean_kind == "zero":
            mean = None
        elif mean_kind == "by_rank":
            mean = torch.nanmean(x, dim=(0, 2))
            assert mean.isfinite().all()
            x = x - mean.unsqueeze(1)
        elif mean_kind == "full":
            mean = torch.nanmean(x, dim=0)
            mean = torch.where(
                mean.isnan().all(1).unsqueeze(1),
                torch.nanmean(mean, dim=1).unsqueeze(1),
                mean,
            )
            x = x - mean
        else:
            assert False

        # estimate covs
        # still n, rank, n_channels
        dxsq = x.square()
        full_var = torch.nanmean(dxsq, dim=0)
        rank_var = torch.nanmean(full_var, dim=1)
        rank_std = rank_var.sqrt_()
        assert rank_var.isfinite().all()
        global_var = torch.nanmean(rank_var)
        global_std = global_var.sqrt()

        if cov_kind == "scalar":
            return cls(mean=mean, global_std=global_std, **init_kw)

        if cov_kind == "by_rank":
            return cls(mean=mean, global_std=global_std, rank_std=rank_std, **init_kw)

        if cov_kind == "diagonal":
            full_var = torch.where(
                full_var.isnan().all(1).unsqueeze(1),
                rank_var.unsqueeze(1),
                full_var,
            )
            full_std = full_var.sqrt()
            return cls(mean=mean, global_std=global_std, full_std=full_std, **init_kw)

        if cov_kind == "full":
            x = x.view(n, rank * n_channels)
            present = torch.isfinite(x).any(dim=0)
            cov = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
            vcov = spiketorch.nancov(x[:, present], force_posdef=True, eps=eps)
            cov[present[:, None] & present[None, :]] = vcov.view(-1)
            cov = cov.reshape(rank, n_channels, rank, n_channels)
            return cls(mean=mean, global_std=global_std, full_cov=cov, **init_kw)

        assert cov_kind.startswith("factorized")
        # handle rank part first, then the spatial part

        # start by getting the rank part of the cov, if necessary, and whitening
        # the ranks to leave behind x_spatial
        if "rank_diag" in cov_kind:
            # rank part is diagonal
            rank_vt = None
            x_spatial = x.div_(rank_std.unsqueeze(1))
            del x
        else:
            # full cov estimation for rank part via svd
            # we have NaNs, but we can get rid of them because channels are either all
            # NaN or not. below, for the spatial part, no such luck and we have to
            # evaluate the covariance in a masked way
            x_rank = x.permute(0, 2, 1).reshape(n * n_channels, rank)
            valid = x_rank.isfinite().all(dim=1)
            x_rankv = x_rank[valid]
            del x
            u_rankv, rank_sing, rank_vt = torch.linalg.svd(x_rankv, full_matrices=False)
            correction = torch.tensor(len(x_rankv) - 1.0).sqrt()
            rank_std = rank_sing / correction

            # whitened spatial part -- reuse storage
            x_spatial = x_rank
            del x_rank
            x_spatial[valid] = u_rankv
            x_spatial = x_spatial.reshape(n, n_channels, rank).permute(0, 2, 1)
            x_spatial.mul_(correction)

        spatial_mask = None
        if zero_radius:
            assert rgeom is not None
            assert rgeom.shape[0] == n_channels
            spatial_mask = squareform(pdist(rgeom)) < zero_radius

        # spatial part could be "by rank" or same for all ranks
        # either way, there are nans afoot
        if "by_rank" in cov_kind:
            channel_std = torch.ones_like(x_spatial[0])
            channel_vt = torch.zeros((rank, n_channels, n_channels)).to(channel_std)
            for q in range(rank):
                xq = x_spatial[:, q]
                validq = xq.isfinite().any(0)
                covq = spiketorch.nancov(xq[:, validq])
                if shrinkage:
                    covq = F.softshrink(covq, shrinkage)
                fullcovq = torch.eye(xq.shape[1], dtype=covq.dtype, device=covq.device)
                fullcovq[validq[:, None] & validq[None, :]] = covq.view(-1)
                if spatial_mask is not None:
                    fullcovq *= torch.asarray(spatial_mask).to(fullcovq)
                if glasso_alpha:
                    res = graphical_lasso(
                        fullcovq.numpy(force=True),
                        alpha=glasso_alpha,
                        mode="cd",
                        max_iter=100,
                        verbose=logger.isEnabledFor(DARTSORTDEBUG),
                    )
                    fullcovq = torch.from_numpy(res[0]).to(fullcovq)
                qeig, qv = torch.linalg.eigh(fullcovq)
                channel_std[q] = qeig.sqrt()
                channel_vt[q] = qv.T
        else:
            x_spatial = x_spatial.reshape(n * rank, n_channels)
            valid = x_spatial.isfinite().any(0)

            if "noise" in cov_kind:
                xx = x_spatial.numpy(force=True)
                invalid = np.isnan(xx)
                xx[invalid] = (
                    np.random.default_rng(0).normal(size=invalid.sum()).astype(xx.dtype)
                )
                x_spatial = torch.from_numpy(xx).to(x_spatial)
                valid = slice(None)
                init_kw["cov_kind"] = cov_kind.removesuffix("noise")

            cov = spiketorch.nancov(x_spatial[:, valid].double(), force_posdef=True)
            cov.diagonal().add_(eps)
            if shrinkage:
                cov = F.softshrink(cov, shrinkage)

            if glasso_alpha and isinstance(glasso_alpha, int):
                logger.dartsortdebug(
                    f"Run glasso cv on {x_spatial.shape=} {cov.abs().max()=}"
                )
                # todo: clean up, propagate, ...
                glasso = GraphicalLassoCV(
                    alphas=glasso_alpha,
                    n_jobs=1,
                    verbose=logger.isEnabledFor(DARTSORTDEBUG),
                    assume_centered=True,
                    eps=eps,
                )
                xx = x_spatial[:, valid].double().numpy(force=True)
                invalid = np.isnan(xx)
                xx[invalid] = np.random.default_rng(0).normal(size=invalid.sum())
                with warnings.catch_warnings(action="ignore"):
                    glasso.fit(xx)
                logger.dartsortdebug(f"Best alpha was {glasso.alpha_=}")
                glasso_alpha = glasso.alpha_

            if glasso_alpha and isinstance(glasso_alpha, float):
                logger.dartsortdebug(f"Run glasso on {cov.shape=}")
                res = graphical_lasso(
                    cov.numpy(force=True),
                    alpha=glasso_alpha,
                    mode="cd",
                    max_iter=100,
                    verbose=logger.isEnabledFor(DARTSORTDEBUG),
                )
                cov = torch.from_numpy(res[0]).to(cov)

            cov_spatial = cov = cov.to(x_spatial)
            if valid != slice(None):
                cov_spatial = torch.eye(
                    x_spatial.shape[1], dtype=x_spatial.dtype, device=x_spatial.device
                )
                cov_spatial[valid[:, None] & valid[None, :]] = cov.view(-1)
            if spatial_mask is not None:
                cov_spatial *= torch.asarray(spatial_mask).to(cov_spatial)
            channel_eig, channel_v = torch.linalg.eigh(cov_spatial)
            channel_std = channel_eig.sqrt()
            channel_vt = channel_v.T.contiguous()

        return cls(
            mean=mean,
            global_std=global_std,
            rank_std=rank_std,
            rank_vt=rank_vt,
            channel_std=channel_std,
            channel_vt=channel_vt,
            zero_radius=zero_radius,
            **init_kw,
        )

    @classmethod
    def estimate_from_hdf5(
        cls,
        hdf5_path,
        mean_kind="zero",
        cov_kind="factorizednoise",
        motion_est=None,
        interpolation_method="normalized",
        kernel_name="rbf",
        sigma=20.0,
        rq_alpha=1.0,
        kriging_poly_degree=-1,
        device=None,
        shrinkage=5e-3,
        glasso_alpha: int | float | None = None,
        zero_radius: float | None = None,
    ):
        from dartsort.util.drift_util import registered_geometry

        logger.dartsortdebug(
            f"Estimate embedded noise with {mean_kind=} {cov_kind=} {motion_est is None=} {glasso_alpha=}"
        )

        with h5py.File(hdf5_path, "r", locking=False) as h5:
            geom = h5["geom"][:]
        rgeom = geom
        if motion_est is not None:
            rgeom = registered_geometry(geom, motion_est=motion_est)
        snippets = interpolate_residual_snippets(
            motion_est,
            hdf5_path,
            geom,
            rgeom,
            method=interpolation_method,
            kernel_name=kernel_name,
            sigma=sigma,
            rq_alpha=rq_alpha,
            kriging_poly_degree=kriging_poly_degree,
            device=device,
        )
        return cls.estimate(
            snippets,
            shrinkage=shrinkage,
            mean_kind=mean_kind,
            cov_kind=cov_kind,
            glasso_alpha=glasso_alpha,
            zero_radius=zero_radius,
            rgeom=rgeom,
        )


def interpolate_residual_snippets(
    motion_est,
    hdf5_path,
    geom,
    registered_geom,
    residual_times_s_dataset_name="residual_times_seconds",
    residual_dataset_name="residual",
    method="normalized",
    kernel_name="rbf",
    sigma=20.0,
    rq_alpha=1.0,
    kriging_poly_degree=-1,
    workers=None,
    device=None,
):
    """PCA-embed and interpolate residual snippets to the registered probe"""
    from dartsort.util import data_util, drift_util, interpolation_util

    assert geom.shape[1] == 2, "Haven't implemented 3d probes here."

    if device is None:
        device = "cuda" if torch.cuda.is_available() else None

    tpca = data_util.get_tpca(hdf5_path)
    if device is not None:
        tpca = tpca.to(tpca.components.dtype)
        tpca = tpca.to(device)

    with h5py.File(hdf5_path, "r", locking=False) as h5:
        channel_index = h5["channel_index"][:]
        snippets = h5[residual_dataset_name][:]
        times_s = h5[residual_times_s_dataset_name][:]
    channel_index = torch.from_numpy(channel_index).to(tpca.components.device)
    snippets = torch.from_numpy(snippets).to(tpca.components)
    times_s = torch.from_numpy(times_s).to(tpca.components)

    # tpca project
    if tpca.temporal_slice is not None:
        snippets = snippets[:, tpca.temporal_slice]
    n, t, c = snippets.shape
    snippets = snippets.permute(0, 2, 1).reshape(n * c, t)
    snippets = tpca._transform_in_probe(snippets)
    snippets = snippets.reshape(n, c, -1).permute(0, 2, 1)

    # -- interpolate
    # fill in the registered probe with residual snippets using interpolation,
    # with possible missing values
    source_geom = torch.asarray(geom).to(snippets)
    source_pos = source_geom[None].broadcast_to(n, *geom.shape).contiguous()

    # - precompute
    precomputed_data = interpolation_util.interp_precompute(
        interpolation_util.pad_geom(source_geom),
        channel_index=channel_index,
        method=method,
        kernel_name=kernel_name,
        sigma=sigma,
        rq_alpha=rq_alpha,
        kriging_poly_degree=kriging_poly_degree,
        source_geom_is_padded=True,
    )
    if precomputed_data is not None:
        nc, nc_pc, nc_pc_ = precomputed_data.shape
        assert nc_pc == nc_pc_
        assert nc == c
        extra_dim = nc_pc - channel_index.shape[1]
        assert extra_dim >= 0
        # embed into full probe...
        pc_full = precomputed_data.new_zeros((c, c + extra_dim, c + extra_dim))
        for j in range(c):
            chans = channel_index[j]
            (valid,) = (chans < c).nonzero(as_tuple=True)
            cvalid = chans[valid]
            pc_full[j, cvalid[:, None], cvalid[None, :]] = precomputed_data[
                j, valid[:, None], valid[None, :]
            ]
            if extra_dim > 0:
                pc_full[j, -extra_dim:, -extra_dim:] = precomputed_data[
                    j, -extra_dim:, -extra_dim:
                ]
                pc_full[j, -extra_dim:, cvalid[None, :]] = precomputed_data[
                    j, -extra_dim:, valid[None, :]
                ]
                pc_full[j, cvalid[:, None], -extra_dim:] = precomputed_data[
                    j, valid[:, None], -extra_dim:
                ]
        precomputed_data = pc_full[None]

    if motion_est is None:
        # no drift case, no missing values, but still interpolate to avoid
        # statistical differences between drifty/no drift versions of the sorter
        target_geom = torch.asarray(registered_geom).to(snippets)
        assert torch.equal(source_geom, target_geom)
        target_pos = target_geom[None].broadcast_to(n, *geom.shape).contiguous()

        snippets = interpolation_util.kernel_interpolate(
            snippets,
            source_pos,
            target_pos,
            method=method,
            kernel_name=kernel_name,
            sigma=sigma,
            rq_alpha=rq_alpha,
            kriging_poly_degree=kriging_poly_degree,
            precomputed_data=precomputed_data,
            allow_destroy=True,
        )
        return snippets

    # goal
    # - determine what registered channels the shifting source positions
    #   would land on
    # - interpolate residual snippets from the shifted source position
    #   to the target positions
    # except that's slow, because kriging interpolator would require
    # inverting a matrix for each shifted source geom. so instead,
    # let's shift the targets inversely and use the same source geom,
    # even if that has a bit of a different meaning.

    # determine each channel's drift over time
    source_depths = source_pos[:, :, 1].reshape(-1)
    source_t = times_s[:, None].broadcast_to(source_pos[:, :, 1].shape).reshape(-1)
    source_shifts = motion_est.disp_at_s(source_t.cpu(), source_depths.cpu())
    source_shifts = source_shifts.reshape(source_pos[:, :, 1].shape).astype("float32")
    source_shifts_xy = np.stack([np.zeros_like(source_shifts), source_shifts], axis=-1)
    source_pos_shifted = source_pos.numpy(force=True) + source_shifts_xy

    # query the target geom for the closest source pos, within reason
    kdtree = drift_util.KDTree(registered_geom)
    match_distance = drift_util.pdist(geom).min()
    _, targ_inds = kdtree.query(
        source_pos_shifted.reshape(-1, geom.shape[1]),
        distance_upper_bound=match_distance,
        workers=workers or -1,
    )
    targ_inds = torch.from_numpy(targ_inds).reshape(source_pos.shape[:2])
    assert (targ_inds < kdtree.n).all()
    target_pos = registered_geom[targ_inds]
    target_pos_shifted = target_pos - source_shifts_xy
    target_pos_shifted = torch.asarray(target_pos_shifted).to(snippets)

    # allocate output storage with an extra channel of NaN needed later
    snippets = interpolation_util.kernel_interpolate(
        snippets,
        source_pos,
        target_pos_shifted,
        method=method,
        kernel_name=kernel_name,
        sigma=sigma,
        rq_alpha=rq_alpha,
        kriging_poly_degree=kriging_poly_degree,
        precomputed_data=precomputed_data,
        allow_destroy=True,
    )

    # now, let's embed these into the full registered probe
    snippets_full = snippets.new_full((n, tpca.rank, len(registered_geom)), torch.nan)
    targ_inds = targ_inds[:, None].broadcast_to(snippets.shape)
    snippets_full.scatter_(2, targ_inds.to(snippets.device), snippets)

    return snippets_full


def fp_control_threshold(
    fp_dataframe,
    fp_num_frames,
    tp_unit_ids,
    tp_counts,
    clustering_num_frames,
    template_normsqs,
    clustering_subsampling_rate=1.0,
    max_fp_per_input_spike=1.0,
    resolution=1.0,
    min_threshold_factor=0.5,
    min_threshold=5.0,
):
    """Global threshold for template matching to control the false positive rate

    For each unit, as a function of the threshold, we can estimate the number
    of false positives per frame using `fp_dataframe` and `fp_num_frames`. We
    can also estimate the true positive rate using `labels` and
    `clustering_num_frames`.

    Then, for each unit, we can find the largest threshold such that the false
    positive rate divided by the true positive rate is <max_fp_per_input_spike.
    This controls the FPR after deconv per unit to be
    <max_fp_per_input_spike/2*max_fp_per_input_spike.
    We can then take the max over units to get a global threshold.

    Arguments
    ---------
    fp_dataframe: pd.DataFrame
    fp_num_frames : int
        Above two as returned by .unit_false_positives()
    tp_unit_ids, tp_counts : np.array
        Unique of clustering labels
    clustering_num_frames : int
        Number of valid recording frames used during clustering
    clustering_subsampling_rate : float
        If the full set of detected spikes from the whole `clustering_num_frames`
        frame collection was subsampled, then we should calculate rates per frame
        by dividing by clustering_subsampling_rate * clustering_num_frames
    max_fp_per_input_spike : float
        The control parameter described above.
    resolution : float
        The rates as a function of threshold will be estimated as a function of
        (squared) deconv threshold at this resolution.

    Returns
    -------
    threshold : float
        Note: this is in squared objective units.
    """
    tp_counts = tp_counts[tp_unit_ids >= 0]
    tp_unit_ids = tp_unit_ids[tp_unit_ids >= 0]
    fp_units = np.unique(fp_dataframe.units)
    assert np.isin(fp_units, tp_unit_ids).all()
    has_fp = np.isin(tp_unit_ids, fp_units)
    logger.dartsortdebug(f"{has_fp.mean() * 100:.1f}% of units had FPs")

    # initialize the threshold with a min factor
    threshold = max(min_threshold, min_threshold_factor * template_normsqs.min())
    logger.dartsortdebug(f"fp control: min possible {threshold=}")
    if not len(fp_dataframe):
        return threshold

    # resolution-spaced grid which will be used for searches below
    start = resolution * np.ceil(threshold / resolution)
    end = resolution * np.ceil(fp_dataframe.scores.max() / resolution)
    domain = np.arange(start, end + resolution, resolution)
    domaini = 0

    # account for different time ranges
    fp_per_tp = fp_num_frames / (clustering_subsampling_rate * clustering_num_frames)

    for uid, tp in zip(tp_unit_ids[has_fp], tp_counts[has_fp]):
        in_uid = fp_dataframe.units == uid
        if not in_uid.any():
            continue

        # does the unit even have any fps at this threshold?
        fp_scores = fp_dataframe[in_uid].scores.values
        if threshold > fp_scores.max():
            continue

        # largest number of false positives we can allow in this unit
        max_fp = max_fp_per_input_spike * tp * fp_per_tp

        # check if we are already under the limit
        if fp_scores.size <= max_fp:
            continue

        # pick the smallest threshold such that the estimated fp is < max_fp
        fp_scores.sort()
        fp_ix = 0
        for i, x in enumerate(domain[domaini:]):
            rel_ix = np.searchsorted(fp_scores[fp_ix:], x, side="left")
            if rel_ix == 0:
                continue
            fp_ix += rel_ix
            n_fp = fp_scores.size - fp_ix
            if n_fp <= max_fp:
                assert x >= threshold
                domaini += i
                threshold = x
                logger.dartsortdebug(f"fp control: new {threshold=}")
                break
        else:
            # a break should always be hit thanks to the continues above
            assert False

    return threshold
