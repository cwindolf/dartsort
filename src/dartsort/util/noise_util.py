import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

import h5py
import linear_operator
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from linear_operator import operators
from scipy.fftpack import next_fast_len
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from sklearn.covariance import GraphicalLassoCV, graphical_lasso
from sklearn.decomposition import PCA
from torch import Tensor

if TYPE_CHECKING:
    from ..transform.temporal_pca import BaseTemporalPCA
from ..util import more_operators, spiketorch
from ..util.data_util import DARTsortSorting
from ..util.internal_config import (
    ComputationConfig,
    InterpolationParams,
    WhiteningConfig,
    tps_interp_clampna_extrap_params,
)
from ..util.job_util import ensure_computation_config
from ..util.logging_util import DARTSORTDEBUG, get_logger, progbar, progrange
from ..util.motion import MotionInfo
from ..util.spiketorch import spawn_torch_rg
from ..util.torch_util import BModule
from ..util.waveform_util import make_channel_index

logger = get_logger(__name__)


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
        device = cast(torch.device, self.spatial_std.device)
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

        return cls(std, vt, nt=t, nc=c)


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

        Parameters
        ----------
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

    margin = 0

    def __init__(self, n_channels, scale=1.0):
        super().__init__()
        self.n_channels = n_channels
        self.scale = scale

    def unwhiten(self, snippet):
        return snippet

    def simulate(self, size=1, t=None, generator=None, chunk_t=None):
        assert t is not None
        x = torch.randn(size, t, self.n_channels, generator=generator)
        if self.scale != 1.0:
            x *= self.scale
        return x


class StationaryFactorizedNoise(torch.nn.Module):
    def __init__(self, spatial_std, vt_spatial, kernel_fft, block_size, t):
        super().__init__()
        self.spatial_std = torch.asarray(spatial_std)
        self.vt_spatial = torch.asarray(vt_spatial)
        self.kernel_fft = torch.asarray(kernel_fft)
        self.block_size = block_size
        self.t = t
        self.margin = (self.t - 1) // 2

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

    def unwhiten(self, snippet):
        snippet = torch.asarray(
            snippet, device=self.spatial_std.device, dtype=self.spatial_std.dtype
        )
        flat = snippet.ndim == 2
        if flat:
            snippet = snippet[None]
        size, t_padded, c = snippet.shape
        t = t_padded - self.t + 1
        out = snippet.new_empty(size, c, t)
        for j in range(size):
            out[j] = spiketorch.single_inv_oaconv1d(
                snippet[j].T,
                s2=self.t,
                f2=self.kernel_fft,
                block_size=self.block_size,
                norm="ortho",
            )
        spatial_part = self.spatial_std[:, None] * self.vt_spatial
        out = torch.einsum("nct,cd->ntd", out, spatial_part)
        if flat:
            out = out[0]
        return out

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
            for bs in progrange(0, t, chunk_t):
                chunk = self.simulate(t=chunk_t + self.t, generator=generator)[0]
                if bs > 0:
                    chunk[: self.t] *= taper[:, None]
                chunk[-self.t :] *= (1 - taper)[:, None]
                out[bs : bs + chunk_t + self.t] += chunk.cpu()
            return out[None, : -self.t]

        # need extra room at the edges to do valid convolution
        t_padded = t + self.t - 1
        noise = torch.randn(size, t_padded, c, generator=generator, device=device)
        return self.unwhiten(noise)

    @classmethod
    def estimate(cls, snippets):
        """Estimate factorized temporally stationary noise

        When simulating long samples, the above approach can't scale. Here let
            spatial_cov[c, c'] * temporal_cov[t - t'].

        Now, remember that stationary kernels diagonalize in Fourier basis. FFT
        is the eigh of circulant things. So we use an FFT based version of the
        SVD algorithm above.

        Parameters
        ----------
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

        Parameters
        ----------
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
        for j in progrange(size, desc="False positives"):
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
                relative_peak_radius=1,
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


class EmbeddedNoise(BModule):
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
        rank: int,
        n_channels: int,
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
        self.rank: int = rank
        self.n_channels: int = n_channels
        self.mean_kind = mean_kind
        self.cov_kind = cov_kind
        self.D = rank * n_channels
        self.zero_radius = zero_radius

        self.register_buffer("chans_arange", torch.arange(n_channels))
        self.register_buffer("global_std", global_std)

        device = None
        if global_std is not None:
            device = global_std.device

        if mean is not None:
            self.register_buffer("mean", mean)
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
            self.register_buffer("channel_vt_zpad", F.pad(channel_vt, (0, 1)))
        if full_cov is not None:
            self.register_buffer("full_cov", full_cov)
            if device is None:
                device = full_cov.device

        # precompute stuff
        self._full_cov = None
        self._full_covinvcov = None
        self._full_inverse = None
        self._full_whitener = None
        self._logdet = None
        self.register_buffer("mean_full", self.mean_rc().clone().detach())
        self.cache = {}
        self.to(device)

    @property
    def logdet(self) -> Tensor:
        if self._logdet is None:
            self.marginal_covariance()
        return cast(Tensor, self._logdet)

    @property
    def device(self) -> torch.device:
        return self.b.chans_arange.device

    def mean_rc(self) -> Tensor:
        """Return noise mean as a rank x channels tensor"""
        shape = self.rank, self.n_channels
        if self.mean_kind == "zero":
            return torch.zeros(shape)
        elif self.mean_kind == "by_rank":
            return cast(Tensor, self.mean)[:, None].broadcast_to(shape).contiguous()
        elif self.mean_kind == "full":
            return cast(Tensor, self.mean)
        else:
            assert False

    def marginal_mean(self):
        """Return noise mean as a rank x channels tensor"""
        shape = self.rank, self.n_channels
        if self.mean_kind == "zero":
            return torch.zeros(shape)
        if self.mean_kind == "by_rank":
            return cast(Tensor, self.mean)[:, None].broadcast_to(shape).contiguous()
        if self.mean_kind == "full":
            return self.mean
        assert False

    def whitener(self, channels=slice(None)):
        cov = self.marginal_covariance(channels=channels)
        chol = cov.cholesky().to_dense()
        eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        whitener = torch.linalg.solve_triangular(chol, eye, upper=False)
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

    def whiten_full(self, data):
        # self._full_whitener = None
        if self._full_whitener is None:
            self._full_whitener = self.whitener()
        data = data.reshape(len(data), -1)
        return data @ self._full_whitener

    def full_dense_cov(self, device=None):
        if self._full_cov is None:
            self.marginal_covariance(device=device)
        if device is not None:
            self._full_cov_dense = self._full_cov_dense.to(device)
        return self._full_cov_dense

    def full_covinvcov(self, device=None):
        if self._full_cov is None:
            self.marginal_covariance(device=device)
        assert self._full_cov is not None
        if self._full_covinvcov is None:
            self._full_covinvcov = self._full_cov.solve(self.full_dense_cov())
        if device is not None:
            self._full_covinvcov = self._full_covinvcov.to(device)
        return self._full_covinvcov

    def full_inverse(self, device=None):
        if self._full_cov is None:
            self.marginal_covariance(device=device)
        assert self._full_cov is not None
        if self._full_inverse is None:
            self._full_inverse = self._full_cov.inverse().to_dense()
        if device is not None:
            self._full_inverse = self._full_inverse.to(device)
        return self._full_inverse

    def cov_batch_mul(self, x: Tensor, channels_src: Tensor, channels_targ: Tensor):
        assert x.ndim == 3
        n = x.shape[0]
        assert x.shape[2] == channels_src.shape[1]
        assert channels_src.ndim == 2
        assert channels_src.shape[0] == n
        if channels_targ.ndim == 2:
            assert channels_targ.shape[0] == n
        else:
            assert channels_targ.ndim == 1

        if self.cov_kind == "factorized":
            rank_root = self.b.rank_vt.T * self.b.rank_std
            rank_cov = rank_root @ rank_root.T
            rank_cov = rank_cov[None].broadcast_to(n, *rank_cov.shape)
            x = rank_cov.bmm(x)

            chan_root_left = self.b.channel_vt_zpad.T[channels_src] * self.b.channel_std
            chan_root_right = (
                self.b.channel_vt_zpad.T[channels_targ] * self.b.channel_std
            )
            if chan_root_right.ndim == 2:
                targ_shp = (chan_root_left.shape[0], *chan_root_right.shape)
                chan_root_right = chan_root_right[None].broadcast_to(targ_shp)
            chan_cov = chan_root_left.bmm(chan_root_right.mT)
            return x.bmm(chan_cov)
        elif self.cov_kind == "scalar":
            if channels_targ.ndim == 1:
                channels_targ = channels_targ[None, :].broadcast_to(
                    (n, channels_targ.numel())
                )
            channels_src = torch.where(
                channels_src == self.n_channels, -2, channels_src
            )
            eyes = (channels_src[:, :, None] == channels_targ[:, None, :]).float()
            chan_cov = eyes * self.b.global_std**2
            return x.bmm(chan_cov)
        elif self.cov_kind == "full":
            # this branch is for debugging. it is slow.
            if channels_targ.ndim == 1:
                channels_targ = channels_targ[None, :].broadcast_to(
                    (n, channels_targ.numel())
                )
            cov_zpad = F.pad(self.b.full_cov, (0, 1, 0, 0, 0, 1))[None]
            cov = cov_zpad.take_along_dim(
                indices=channels_src[:, None, :, None, None], dim=2
            )
            assert cov.shape == (
                n,
                self.rank,
                channels_src.shape[1],
                self.rank,
                self.n_channels + 1,
            )
            cov = cov.take_along_dim(
                indices=channels_targ[:, None, None, None, :], dim=4
            )
            assert cov.shape == (
                n,
                self.rank,
                channels_src.shape[1],
                self.rank,
                channels_targ.shape[1],
            )
            cov = cov.view(
                n,
                self.rank * channels_src.shape[1],
                self.rank * channels_targ.shape[1],
            )
            x = x.view(n, 1, -1).bmm(cov)
            return x.view(n, self.rank, channels_targ.shape[1])
        else:
            raise NotImplementedError(
                f"Need to implement cov_batch_mul for {self.cov_kind=}."
            )

    def cov_batch(self, channels_left: Tensor, channels_right: Tensor):
        if self.cov_kind != "factorized":
            raise NotImplementedError(
                f"Need to implement cov_batch for {self.cov_kind=}."
            )
        rank_root = self.b.rank_vt.T * self.b.rank_std
        rank_cov = rank_root @ rank_root.T
        chan_root_left = self.b.channel_vt_zpad.T[channels_left] * self.b.channel_std
        chan_root_right = self.b.channel_vt_zpad.T[channels_right] * self.b.channel_std
        if chan_root_right.ndim == 2:
            chan_root_right = chan_root_right[None].broadcast_to(
                chan_root_left.shape[0], *chan_root_right.shape
            )
        chan_cov = chan_root_left.bmm(chan_root_right.mT)
        rank_cov = rank_cov[None].broadcast_to(
            channels_left.shape[0], self.rank, self.rank
        )
        return operators.KroneckerProductLinearOperator(rank_cov, chan_cov)

    def marginal_covariance(
        self,
        channels: Tensor | slice = slice(None),
        cache_prefix=None,
        cache_key=None,
        device=None,
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
                self._logdet = cast(Tensor, self._logdet).to(device)
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

    def _marginal_covariance(
        self, channels: Tensor | slice = slice(None), channels_left=None
    ):
        channels = self.b.chans_arange[channels]
        have_left = channels_left is not None
        if channels_left is None:
            channels_left = channels
        nc = channels.numel()
        ncl = int(channels_left.numel())

        if have_left:
            if self.cov_kind in ("scalar", "diagonal_by_rank", "diagonal"):
                # if torch.isin(channels_left.to(channels), channels).any():
                #     raise ValueError(
                #         "Don't know how to get non-marginal covariance block "
                #         f"for cov_kind={self.cov_kind} when the block overlaps "
                #         "with the diagonal."
                #     )
                return operators.ZeroLinearOperator(
                    self.rank * ncl,  # type: ignore
                    self.rank * nc,  # type: ignore
                    dtype=self.b.global_std.dtype,
                )

        if self.cov_kind == "scalar":
            eye = operators.IdentityLinearOperator(self.rank * nc, device=self.device)
            return self.b.global_std.square() * eye

        if self.cov_kind == "diagonal_by_rank":
            rank_diag = operators.DiagLinearOperator(self.b.rank_std**2)
            chans_eye = operators.IdentityLinearOperator(nc, device=self.device)
            return torch.kron(rank_diag, chans_eye)  # type: ignore

        if self.cov_kind == "diagonal":
            chans_std = self.b.full_std[:, channels]
            return operators.DiagLinearOperator(chans_std**2)

        if self.cov_kind == "full":
            marg_cov = self.b.full_cov[:, channels_left][..., channels]
            r = marg_cov.shape[0]
            marg_cov = marg_cov.reshape(r * ncl, r * nc)
            return linear_operator.to_linear_operator(marg_cov)

        if self.cov_kind in "factorized":
            rank_root = self.b.rank_vt.T * self.b.rank_std
            rank_cov = rank_root @ rank_root.T
            chan_root = self.b.channel_vt.T[channels] * self.b.channel_std
            chan_root_right = chan_root.T
            if have_left:
                chan_root = self.b.channel_vt.T[channels_left] * self.b.channel_std
            chan_cov = chan_root @ chan_root_right
            return operators.KroneckerProductLinearOperator(rank_cov, chan_cov)

        if self.cov_kind == "factorized_rank_diag":
            rank_cov = operators.DiagLinearOperator(self.b.rank_std.square())
            chan_root = self.b.channel_vt.T[channels] * self.b.channel_std
            chan_root_right = chan_root.T
            if have_left:
                chan_root = self.b.channel_vt.T[channels_left] * self.b.channel_std
            chan_cov = chan_root @ chan_root_right
            return torch.kron(rank_cov, chan_cov)  # type: ignore

        if self.cov_kind == "factorized_by_rank_rank_diag":
            rank_std = self.b.rank_std.view(self.rank, 1, 1)
            chans_std = self.b.channel_std[:, :, None]  # no slice here!
            rc_std = rank_std * chans_std
            chans_vt = self.b.channel_vt[:, :, channels]
            chan_rootl = chan_root = rc_std * chans_vt
            if have_left:
                chans_vtl = self.b.channel_vt[:, :, channels_left]
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
        shrinkage=0.0,
        glasso_alpha: int | float | None = None,
        eps=1e-4,
        zero_radius: float | None = None,
        svd_batch_size=256,
        rgeom=None,
    ):
        """Factory method to estimate noise model from TPCA snippets

        Parameters
        ----------
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
            return cls(mean=mean, global_std=global_std, **init_kw)  # type: ignore

        if cov_kind == "by_rank":
            return cls(mean=mean, global_std=global_std, rank_std=rank_std, **init_kw)  # type: ignore

        if cov_kind == "diagonal":
            full_var = torch.where(
                full_var.isnan().all(1).unsqueeze(1),
                rank_var.unsqueeze(1),
                full_var,
            )
            full_std = full_var.sqrt()
            return cls(mean=mean, global_std=global_std, full_std=full_std, **init_kw)  # type: ignore

        if cov_kind == "full":
            x = x.view(n, rank * n_channels)
            present = torch.isfinite(x).any(dim=0)
            cov = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
            vcov = spiketorch.nancov(x[:, present], force_posdef=True, eps=eps)
            assert torch.is_tensor(vcov)
            cov[present[:, None] & present[None, :]] = vcov.view(-1)
            cov = cov.reshape(rank, n_channels, rank, n_channels)
            return cls(mean=mean, global_std=global_std, full_cov=cov, **init_kw)  # type: ignore

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

            # doing this on CPU since the SVD can use a lot of memory.
            x_rank = x.permute(0, 2, 1)
            x_rank = x_rank.cpu().reshape(n * n_channels, rank)
            (valid,) = x_rank.isfinite().all(dim=1).nonzero(as_tuple=True)
            x_rankv = x_rank[valid]
            orig_device = x.device
            del x
            assert not x_rankv.requires_grad
            u_rankv, rank_sing, rank_vt = torch.linalg.svd(x_rankv, full_matrices=False)

            correction = torch.tensor(len(x_rankv) - 1.0).sqrt()
            rank_std = rank_sing / correction
            rank_std = rank_std.to(orig_device)
            correction = correction.to(orig_device)

            # whitened spatial part -- reuse storage
            x_spatial = x_rank
            del x_rank
            x_spatial[valid] = u_rankv
            x_spatial = x_spatial.reshape(n, n_channels, rank).permute(0, 2, 1)
            x_spatial = x_spatial.to(orig_device)
            x_spatial.mul_(correction)

        spatial_mask = None
        if zero_radius:
            assert rgeom is not None
            assert rgeom.shape[0] == n_channels
            rg_np = rgeom.numpy(force=True) if torch.is_tensor(rgeom) else rgeom
            spatial_mask = squareform(pdist(rg_np)) < zero_radius

        # spatial part could be "by rank" or same for all ranks
        # either way, there are nans afoot
        if "by_rank" in cov_kind:
            channel_std = torch.ones_like(x_spatial[0])
            channel_vt = torch.zeros((rank, n_channels, n_channels)).to(channel_std)
            for q in range(rank):
                xq = x_spatial[:, q]
                validq = xq.isfinite().any(0)
                covq = spiketorch.nancov(xq[:, validq])
                assert torch.is_tensor(covq)
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
                cov = torch.cov(x_spatial.T.double())
                cov = spiketorch.enforce_posdef(cov, eps=eps)
            else:
                cov = spiketorch.nancov(
                    x_spatial[:, valid].double(), force_posdef=True, eps=eps
                )
            assert torch.is_tensor(cov)
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
            assert cov_spatial.isfinite().all()
            if valid != slice(None):
                assert torch.is_tensor(valid)
                cov_spatial = torch.eye(
                    x_spatial.shape[1], dtype=x_spatial.dtype, device=x_spatial.device
                )
                cov_spatial[valid[:, None] & valid[None, :]] = cov.view(-1)
            assert cov_spatial.isfinite().all()
            if spatial_mask is not None:
                cov_spatial *= torch.asarray(spatial_mask).to(cov_spatial)
            assert cov_spatial.isfinite().all()
            channel_eig, channel_v = torch.linalg.eigh(cov_spatial.double())
            if spatial_mask is None:
                assert channel_eig.isfinite().all()
            else:
                # the spatial mask is, i guess, not so plausible as a cov...
                channel_eig[channel_eig < eps] = eps
            channel_std = channel_eig.sqrt().float()
            channel_vt = channel_v.T.contiguous().float()
            assert channel_std.isfinite().all()
            assert channel_vt.isfinite().all()

        return cls(
            mean=mean,
            global_std=global_std,
            rank_std=rank_std,
            rank_vt=rank_vt,
            channel_std=channel_std,
            channel_vt=channel_vt,
            zero_radius=zero_radius,
            **init_kw,  # type: ignore
        )

    @classmethod
    def estimate_from_hdf5(
        cls,
        hdf5_path,
        mean_kind="zero",
        cov_kind="factorizednoise",
        motion: MotionInfo | None = None,
        tpca: "PCA | BaseTemporalPCA | None" = None,
        interp_params: InterpolationParams = tps_interp_clampna_extrap_params,
        device=None,
        rank: int | None = None,
        shrinkage=0.0,
        glasso_alpha: int | float | None = None,
        zero_radius: float | None = None,
        rgeom=None,
    ):
        with h5py.File(hdf5_path, "r", locking=False) as h5:
            geom = cast(h5py.Dataset, h5["geom"])[:]
        if motion is None:
            motion = MotionInfo.from_motion_est(geom=geom)
        snippets = interpolate_residual_snippets(
            motion=motion,
            hdf5_path=hdf5_path,
            interp_params=interp_params,
            tpca=tpca,
            device=device,
            rank=rank,
            do_tpca=True,
        )
        assert snippets.shape[0] > 1
        logger.dartsortdebug(
            f"Estimate embedded noise with {mean_kind=} {cov_kind=} {glasso_alpha=} "
            f"from {snippets.shape=}."
        )
        if rank is not None:
            assert snippets.shape[1] == rank
        return cls.estimate(
            snippets,
            shrinkage=shrinkage,
            mean_kind=mean_kind,
            cov_kind=cov_kind,
            glasso_alpha=glasso_alpha,
            zero_radius=zero_radius,
            rgeom=rgeom,
        )

    def detection_prior_log_prob(self, templates_pca_projected, threshold=10.0):
        """
        Computes:
            z(T) = log[p(noise det | T)]
                 = log[P(|N|^2 - |N - T|^2 > threshold^2)]
                 = log[log P(2N.T > threshold^2 + |T|^2)]

        Then, later, in mixture modeling one can compute
            p(l = noise | x, T) = log pi_noise + log N(x | noise) - z(T)

        It's a little bit counterintuitive: this boosts the noise probability more
        for units with higher signal templates, since their FP probability is lower.

        Note that since N ~ N(0, C), N.T ~ N(0, tr TCT'). Or, whatever version of
        that makes the dimensions work out. Thus we need to compute
            log normal_sf(0.5 * (thresh^2 + |T|^2) ; mean=0, scale=sqrt(tr TCT))
        """
        C = self.full_dense_cov()
        templates_pca_projected = torch.asarray(
            templates_pca_projected, device=C.device
        )
        T = templates_pca_projected.reshape(len(templates_pca_projected), -1)
        assert C.shape == (T.shape[1], T.shape[1])
        tr = torch.einsum("nc,cd,nd->n", T, C, T)
        scale = tr.sqrt_()
        crit = T.square().sum(dim=1).add_(threshold**2).mul_(0.5)
        return norm.logsf(crit.numpy(force=True), scale=scale.numpy(force=True))

    def channelwise_detection_prior_log_prob(
        self, threshold=4.0, n_samples=8192, seed=0
    ):
        """Compute the marginal probability that the norm is larger than threshold on each channel.

        TPCA is isometry, so no need to know the basis when working with the norm. Further, we can
        sample independent normals using the eigs.
        """
        probs = np.zeros(self.n_channels)
        rg = np.random.default_rng(seed)
        samples = rg.normal(size=(n_samples, self.rank))
        tmp = samples.copy()
        for channel in range(self.n_channels):
            chan = torch.atleast_1d(torch.tensor(channel))
            marginal_cov = self.marginal_covariance(channels=chan).to_dense()
            marginal_cov = marginal_cov.numpy(force=True).astype(np.float64)
            eigs = np.linalg.eigvalsh(marginal_cov)
            stds = np.sqrt(eigs)
            norms = np.linalg.norm(np.multiply(samples, stds, out=tmp), axis=1)
            probs[channel] = np.mean(norms > threshold)
        return np.log(probs)


def generate_interpolated_residual_snippets(
    *,
    motion: MotionInfo,
    hdf5_path: str | Path,
    residual_times_s_dataset_name="residual_times_seconds",
    residual_dataset_name="residual",
    interp_params: InterpolationParams = tps_interp_clampna_extrap_params,
    do_tpca: bool,
    tpca: "PCA | BaseTemporalPCA | None" = None,
    device: torch.device | str | None = None,
    rank: int | None = None,
    batch_size=64,
    show_progress=True,
):
    """PCA-embed and interpolate residual snippets to the registered probe"""
    from ..transform import BaseTemporalPCA
    from . import data_util, interpolation_util

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    with h5py.File(hdf5_path, "r", locking=False) as h5:
        channel_index = cast(h5py.Dataset, h5["channel_index"])[:]
        nr = h5["n_residuals"][()]
        snippets = cast(h5py.Dataset, h5[residual_dataset_name])[:nr]
        times_s_np = cast(h5py.Dataset, h5[residual_times_s_dataset_name])[:nr]
    channel_index = torch.from_numpy(channel_index)
    snippets = torch.from_numpy(snippets)
    assert snippets.shape[0] == times_s_np.shape[0] == nr

    # allow out-of-order residual sampling
    order = np.argsort(times_s_np)
    if not np.array_equal(order, np.arange(len(order))):
        times_s_np = times_s_np[order]
        snippets = snippets[order]
    assert snippets.shape[0] == times_s_np.shape[0]

    # tpca project
    if do_tpca:
        if tpca is None:
            tpca = cast(BaseTemporalPCA, data_util.get_tpca(hdf5_path))
        elif not isinstance(tpca, BaseTemporalPCA):
            tpca = BaseTemporalPCA.from_sklearn(
                channel_index=channel_index,
                pca=tpca,
                trim_rank_to=rank,
                spike_length_samples=snippets.shape[1],
            )

        assert tpca is not None
        assert isinstance(tpca, BaseTemporalPCA)
        if device is not None:
            tpca = tpca.to(tpca.b.components.dtype)
        tpca = tpca.to(device=device)
        if tpca.temporal_slice is not None:
            assert isinstance(tpca.temporal_slice, slice)
            snippets = snippets[:, tpca.temporal_slice]
        dim = tpca.rank
        if rank is not None:
            dim = min(dim, rank)
    else:
        dim = snippets.shape[1]
        tpca = None

    erp = interpolation_util.ToFullProbeInterpolator(
        motion=motion, params=interp_params, device=device
    )
    erp = erp.to(device=device)

    inds = []
    if motion is not None:
        if motion.time_bins_s.size <= 1:
            tt = np.concatenate([motion.time_bins_s, times_s_np])
            dbin = 3 * np.ptp(tt)
        else:
            dt = np.diff(motion.time_bins_s)
            assert (dt > 0).all()
            dbin = dt.mean()
        assert motion.time_bins_s[0] - dbin < times_s_np.min()
        assert motion.time_bins_s[-1] + dbin > times_s_np.max()
        i0 = i1 = 0
        for j, tbc in enumerate(motion.time_bins_s):
            # math done this way to avoid float issues; asserts check it.
            if j:
                left = 0.5 * (motion.time_bins_s[j - 1] + tbc)
            else:
                left = tbc - dbin
            if j < motion.time_bins_s.shape[0] - 1:
                right = 0.5 * (motion.time_bins_s[j + 1] + tbc)
            else:
                right = tbc + dbin
            i0 = i0 + np.searchsorted(times_s_np[i0:], left)
            i1 = i0 + np.searchsorted(times_s_np[i0:], right)
            if len(inds):
                assert i0 == inds[-1][1]
            for i00 in range(i0, i1, batch_size):
                i11 = min(i1, i00 + batch_size)
                inds.append((i00, i11, tbc))
        assert i1 == len(times_s_np)
    else:
        for i0 in range(0, len(snippets), batch_size):
            inds.append((i0, min(i0 + batch_size, len(snippets)), 0.0))

    for i0, i1, tbc in (
        progbar(inds, desc="Interpolate resid") if show_progress else inds
    ):
        batch = snippets[i0:i1].to(device=device, non_blocking=True)
        if tpca is not None:
            batch = tpca.force_embed(batch)[:, :dim]
        yield erp.interp_at_time(t_s=tbc, waveforms=batch)


def interpolate_residual_snippets(
    *,
    motion: MotionInfo,
    hdf5_path: str | Path,
    residual_times_s_dataset_name="residual_times_seconds",
    residual_dataset_name="residual",
    interp_params: InterpolationParams = tps_interp_clampna_extrap_params,
    do_tpca: bool,
    tpca: "PCA | BaseTemporalPCA | None" = None,
    device: torch.device | str | None = None,
    rank: int | None = None,
    batch_size=64,
    show_progress=True,
):
    with h5py.File(hdf5_path, "r", locking=False) as h5:
        nr = h5["n_residuals"][()]
    snips_out = None
    i0 = 0
    for snip in generate_interpolated_residual_snippets(
        motion=motion,
        hdf5_path=hdf5_path,
        residual_times_s_dataset_name=residual_times_s_dataset_name,
        residual_dataset_name=residual_dataset_name,
        interp_params=interp_params,
        do_tpca=do_tpca,
        tpca=tpca,
        device=device,
        rank=rank,
        batch_size=batch_size,
        show_progress=show_progress,
    ):
        if snips_out is None:
            snips_out = snip.new_empty((nr, *snip.shape[1:]))
        snips_out[i0 : i0 + snip.shape[0]] = snip.to(snips_out)
        i0 += snip.shape[0]
    return snips_out


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
    min_threshold_factor=0.1,
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

    Parameters
    ----------
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


def fp_control_threshold_from_h5(
    hdf5_path,
    low_rank_templates,
    unit_ids=None,
    spike_counts=None,
    rg: int | np.random.Generator = 0,
    refractory_radius_frames=10,
    num_frames=0,
    max_fp_per_input_spike=1.0,
):
    from ..util.data_util import get_residual_snips

    resids = get_residual_snips(hdf5_path)
    noise = StationaryFactorizedNoise.estimate(resids)

    # be reproducible
    rg = np.random.default_rng(rg)
    generator = spiketorch.spawn_torch_rg(rg)

    # restrict my low rank templates to usual geom...
    # TODO: this is not a very logical way to handle drift here...

    # simulate fps
    fp_res = noise.unit_false_positives(
        low_rank_templates,
        radius=refractory_radius_frames,
        generator=generator,
    )
    return fp_control_threshold(
        fp_res["fp_dataframe"],
        fp_res["total_samples"],
        tp_unit_ids=unit_ids,
        tp_counts=spike_counts,
        clustering_num_frames=num_frames,
        template_normsqs=fp_res["normsq"],
        max_fp_per_input_spike=max_fp_per_input_spike,
    )


def residual_covariance(
    sorting: DARTsortSorting,
    do_interpolation: bool,
    motion: MotionInfo | None,
    interp_params: InterpolationParams = tps_interp_clampna_extrap_params,
    device: torch.device | None = None,
    residual_times_s_dataset_name="residual_times_seconds",
    residual_dataset_name="residual",
    seed: int = 0,
    batch_size=256,
) -> Tensor:
    assert sorting.parent_h5_path is not None

    if do_interpolation:
        assert motion is not None
        snipgen = generate_interpolated_residual_snippets(
            motion=motion,
            hdf5_path=sorting.parent_h5_path,
            interp_params=interp_params,
            device=device,
            do_tpca=False,
            residual_times_s_dataset_name=residual_times_s_dataset_name,
            residual_dataset_name=residual_dataset_name,
            batch_size=batch_size,
        )
    else:
        snipgen = sorting._yield_dataset(residual_dataset_name, batch_size=batch_size)

    rg = spawn_torch_rg(seed=seed, device=device)
    cov = None
    N = 0
    for snip in snipgen:
        snip = torch.asarray(snip).to(device=device, non_blocking=True)
        nc = snip.shape[2]
        snip = snip.view(-1, nc)
        if cov is None:
            cov = snip.new_zeros((nc, nc))

        isna = snip.isnan()
        rvs = torch.randn(
            size=snip.shape, generator=rg, device=device, dtype=snip.dtype
        )
        snip.masked_scatter_(isna, rvs)

        scov = torch.cov(snip.T)
        n = snip.shape[0]
        N += n
        w = n / N
        cov += scov.sub_(cov).mul_(w)
    assert cov is not None

    return cov


def fullzca_whitener(
    cov: np.ndarray, channel_index: np.ndarray | None = None, eps=1e-6
) -> np.ndarray:
    del channel_index
    vals, vecs = eigh(cov, driver="ev")
    wv = 1.0 / (np.sqrt(vals + eps))
    return (vecs * wv) @ vecs.T


def localzca_whitener(
    cov: np.ndarray, channel_index: np.ndarray, eps=1e-6
) -> np.ndarray:
    """"""
    w = np.zeros_like(cov)
    for j, chans in enumerate(channel_index):
        chans = chans[chans < len(channel_index)]
        (ixj,) = np.flatnonzero(chans == j)
        cj = cov[chans][:, chans]
        wj = fullzca_whitener(cj, eps=eps)
        w[j, chans] = wj[ixj]
    return w


def sparsechol_whitener(
    cov: np.ndarray, channel_index: np.ndarray, eps=1e-6
) -> np.ndarray:
    """(Transpose of) whitener of Schäfer et al., https://arxiv.org/pdf/2004.14455.

    This one's actually bad, don't use it. (It is not zero-phase and does bad stuff
    to spatial dimension.)
    """
    w = np.zeros_like(cov)
    for j, chans in enumerate(channel_index):
        chans = chans[chans <= j]
        (ixj,) = np.flatnonzero(chans == j)
        cj = cov[chans][:, chans]
        vals, vecs = eigh(cj, driver="ev")
        pj = (vecs / (vals + eps)) @ vecs.T
        # pj = np.linalg.inv(cj)
        w[j, chans] = pj[:, ixj] / np.sqrt(pj[ixj, ixj])
    return w


# these should be left-side whiteners, meaning precision = W'W
whitening_estimators = {
    "fullzca": fullzca_whitener,
    "localzca": localzca_whitener,
    "sparsechol": sparsechol_whitener,
}


class SpatialWhitener(BModule):
    def __init__(self, whitener: Tensor, covariance: Tensor):
        super().__init__()
        self.register_buffer("whitener", whitener)
        self.register_buffer("covariance", covariance)

    @classmethod
    def blank(cls, n_channels: int, device: torch.device):
        w = torch.zeros((n_channels, n_channels), device=device)
        return cls(w, torch.zeros_like(w))

    @classmethod
    def from_numpy(cls, whitener: np.ndarray, covariance: np.ndarray):
        logger.dartsortverbose("Load whitener from numpy.")
        return cls(
            whitener=torch.asarray(whitener), covariance=torch.asarray(covariance)
        )

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        return self.b.whitener.numpy(force=True), self.b.covariance.numpy(force=True)

    @classmethod
    def from_config(
        cls,
        *,
        sorting: DARTsortSorting,
        motion: MotionInfo | None,
        whiten_cfg: WhiteningConfig,
        computation_cfg: ComputationConfig | None = None,
    ) -> Self:
        logger.dartsortdebug(
            "Estimating %s-%s whitener.", whiten_cfg.strategy, whiten_cfg.estimator
        )
        device = ensure_computation_config(computation_cfg).actual_device()
        cov = residual_covariance(
            sorting=sorting,
            do_interpolation=whiten_cfg.strategy == "postwhiten",
            motion=motion,
            interp_params=whiten_cfg.interp_params,
            device=device,
        )
        geom = getattr(sorting, "geom", None)
        assert geom is not None
        neighbs = make_channel_index(geom, radius=whiten_cfg.radius, to_torch=False)
        cov_np = cov.numpy(force=True).astype(np.float64)
        whitener = whitening_estimators[whiten_cfg.estimator](
            cov_np, channel_index=neighbs
        )
        whitener = torch.asarray(whitener).to(cov)
        return cls(whitener=whitener, covariance=cov)

    def whiten_traces_spatial_major(
        self, x: Tensor, out: Tensor | None = None
    ) -> Tensor:
        return torch.mm(self.b.whitener, x.T, out=out)

    def whiten(self, x: Tensor, out: Tensor | None = None) -> Tensor:
        *shp, c = x.shape
        x = x.reshape(-1, c)
        x = torch.mm(x, self.b.whitener.T, out=out)
        x = x.reshape(*shp, c)
        return x

    def transpose_whiten(self, x: Tensor, out: Tensor | None = None) -> Tensor:
        *shp, c = x.shape
        x = x.reshape(-1, c)
        x = torch.mm(x, self.b.whitener, out=out)
        x = x.reshape(*shp, c)
        return x

    def prec_mul(self, x: Tensor) -> Tensor:
        *shp, c = x.shape
        x = x.reshape(-1, c)
        x = x @ (self.b.whitener.T @ self.b.whitener)
        x = x.reshape(*shp, c)
        return x

    def local_whiteners(self, channel_index: Tensor, eps=1e-6):
        channel_index = torch.asarray(channel_index)
        nc, cloc = channel_index.shape
        assert nc == self.b.covariance.shape[0]
        w = self.b.covariance.new_zeros((nc, cloc, cloc))
        for j, chans in enumerate(channel_index):
            mask = (chans < nc).nonzero()[:, 0]
            chans = chans[mask]
            cj = self.b.covariance[chans][:, chans]
            wj = fullzca_whitener(cj.numpy(force=True).astype(np.float64), eps=eps)
            w[j, mask[:, None], mask[None, :]] = torch.asarray(wj).to(w)
        return w
