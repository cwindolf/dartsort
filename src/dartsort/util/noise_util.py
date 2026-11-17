from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.fftpack import next_fast_len
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from torch import Tensor

if TYPE_CHECKING:
    from ..transform.temporal_pca import BaseTemporalPCA
from ..util import spiketorch
from ..util.data_util import DARTsortSorting
from ..util.internal_config import (
    ComputationConfig,
    InterpolationParams,
    NoiseCovKind,
    WhiteningConfig,
    tps_interp_clampna_extrap_params,
)
from ..util.job_util import ensure_computation_config
from ..util.logging_util import get_logger, progbar, progrange
from ..util.motion import MotionInfo
from ..util.py_util import panic
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
        for _ in progrange(size, desc="False positives"):
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
            _peak_times, peak_units, peak_energies = detect_and_deduplicate(
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


class KroneckerNoise(BModule):
    def __init__(
        self,
        temporal_cov: Tensor,
        spatial_cov: Tensor,
        zero_radius: float | None = None,
    ):
        super().__init__()
        self.rank: int = temporal_cov.shape[0]
        self.n_channels: int = spatial_cov.shape[0]
        assert temporal_cov.shape == (self.rank, self.rank)
        assert spatial_cov.shape == (self.n_channels, self.n_channels)
        self.zero_radius = zero_radius
        self.register_buffer("temporal_cov", temporal_cov)
        self.register_buffer("spatial_cov", spatial_cov)

        temporal_chol, temporal_info = torch.linalg.cholesky_ex(temporal_cov)
        spatial_chol, spatial_info = torch.linalg.cholesky_ex(spatial_cov)
        if temporal_info or spatial_info:
            raise ValueError("Noise covariance not positive definite.")
        temporal_whitener = torch.linalg.solve_triangular(
            temporal_chol, torch.eye(self.rank).to(temporal_chol), upper=False
        )
        spatial_whitener = torch.linalg.solve_triangular(
            spatial_chol, torch.eye(self.n_channels).to(spatial_chol), upper=False
        )
        self.register_buffer("temporal_whitener", temporal_whitener, persistent=False)
        self.register_buffer("spatial_whitener", spatial_whitener, persistent=False)

    @property
    def device(self) -> torch.device:
        return self.b.temporal_cov.device

    @classmethod
    def identity(
        cls,
        rank: int,
        n_channels: int,
        std: float = 1.0,
        zero_radius: float | None = None,
        device: torch.device | None = None,
    ) -> Self:
        temporal_cov = torch.eye(rank, device=device).mul_(std**2)
        spatial_cov = torch.eye(n_channels, device=device)
        return cls(temporal_cov, spatial_cov, zero_radius=zero_radius)

    def whiten(self, x: Tensor) -> Tensor:
        assert x.shape[1:] == (self.rank, self.n_channels)
        return self.b.temporal_whitener @ x @ self.b.spatial_whitener.T

    @classmethod
    def estimate(
        cls,
        snippets: Tensor | np.ndarray,
        *,
        cov_kind: NoiseCovKind,
        zero_radius: float | None = None,
        eps: float = 1e-4,
        batch_size: int = 2**16,
    ) -> Self:
        n, rank, n_channels = snippets.shape
        x = torch.asarray(snippets)
        x = x.to(torch.promote_types(x.dtype, torch.float))

        if cov_kind == "identity":
            full_var = torch.nanmean(x.square(), dim=0)
            rank_var = torch.nanmean(full_var, dim=1)
            global_std = torch.nanmean(rank_var).sqrt().item()
            return cls.identity(
                rank,
                n_channels,
                std=global_std,
                zero_radius=zero_radius,
                device=x.device,
            )
        elif cov_kind != "kronecker":
            panic(cov_kind)

        # full cov estimation for rank part via svd
        # we have NaNs, but we can get rid of them because channels are either all
        # NaN or not.

        # doing this on CPU since the SVD can use a lot of memory.
        orig_device = x.device
        x_rank = x.permute(0, 2, 1).cpu().reshape(n * n_channels, rank)
        del x
        (valid,) = x_rank.isfinite().all(dim=1).nonzero(as_tuple=True)
        u_rankv, rank_sing, rank_vt = torch.linalg.svd(
            x_rank[valid], full_matrices=False
        )
        correction = torch.tensor(valid.numel() - 1.0).sqrt()
        rank_std = (rank_sing / correction).to(orig_device)
        rank_root = rank_vt.to(orig_device).T * rank_std
        temporal_cov = rank_root @ rank_root.T

        # whitened spatial part -- reuse storage
        x_spatial = x_rank
        del x_rank
        x_spatial[valid] = u_rankv
        x_spatial = x_spatial.reshape(n, n_channels, rank).permute(0, 2, 1)
        x_spatial = x_spatial.to(orig_device)
        x_spatial.mul_(correction.to(orig_device))
        x_spatial = x_spatial.reshape(n * rank, n_channels)

        cov = spatial_covariance(x_spatial.split(batch_size))
        min_eig = torch.linalg.eigvalsh(cov)[0]
        if min_eig < eps:
            cov.diagonal().add_(eps - min_eig)
        spatial_cov = cov.to(x_spatial)

        return cls(temporal_cov, spatial_cov, zero_radius=zero_radius)

    @classmethod
    def estimate_from_hdf5(
        cls,
        hdf5_path: str | Path,
        *,
        cov_kind: NoiseCovKind,
        motion: MotionInfo | None = None,
        tpca: "PCA | BaseTemporalPCA | None" = None,
        interp_params: InterpolationParams = tps_interp_clampna_extrap_params,
        device: torch.device | None = None,
        rank: int | None = None,
        zero_radius: float | None = None,
    ) -> Self:
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
        if rank is not None:
            assert snippets.shape[1] == rank
        logger.dartsortdebug(
            f"Estimate embedded noise with {cov_kind=} from {snippets.shape=}."
        )
        return cls.estimate(snippets, cov_kind=cov_kind, zero_radius=zero_radius)


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


def spatial_covariance(batches: Iterable[Tensor]) -> Tensor:
    cov = pair_counts = None
    for xb in batches:
        assert xb.ndim == 2
        xb = xb.to(torch.promote_types(xb.dtype, torch.float))
        if cov is None:
            cov = xb.new_zeros((xb.shape[1], xb.shape[1]), dtype=torch.double)
            pair_counts = torch.zeros_like(cov)
        assert pair_counts is not None

        observed = xb.isfinite().to(xb)
        xb = xb.nan_to_num()
        cov += (xb.T @ xb).double()
        pair_counts += (observed.T @ observed).double()
    assert cov is not None and pair_counts is not None
    return cov / pair_counts.clamp_(min=1.0)


def residual_covariance(
    sorting: DARTsortSorting,
    do_interpolation: bool,
    motion: MotionInfo | None,
    interp_params: InterpolationParams = tps_interp_clampna_extrap_params,
    device: torch.device | None = None,
    residual_times_s_dataset_name="residual_times_seconds",
    residual_dataset_name="residual",
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

    batches = (
        torch.asarray(snip).to(device=device, non_blocking=True).view(-1, snip.shape[2])
        for snip in snipgen
    )
    return spatial_covariance(batches).float()


def residual_welch_whitener(
    sorting: DARTsortSorting,
    device: torch.device | None = None,
    residual_dataset_name="residual",
    batch_size=1024,
    temporal_length: int = 11,
    spatial_whitener: torch.Tensor | None = None,
):
    """Estimate a 0-phase whitening convolution kernel with Welch's method"""
    assert sorting.parent_h5_path is not None
    snipgen = sorting._yield_dataset(residual_dataset_name, batch_size=batch_size)

    if spatial_whitener is not None:
        W = torch.asarray(spatial_whitener, device=device)
    else:
        W = None

    # Welch's method to estimate residual PSD
    snip_psds = []
    block_len = -1
    for snip in snipgen:
        if block_len < 0:
            block_len = next_fast_len(snip.shape[1])

        # handle early stopping in residual extractor, ignore nans
        # if there are nans, they'll be the same on dims 1,2.
        ina = np.flatnonzero(np.isnan(snip[:, 0, 0]))
        if ina.size:
            snip = snip[: ina[0]]
        if not snip.size:
            continue

        # to torch, whiten
        snip = torch.asarray(snip).to(device=device, non_blocking=True)
        if W is not None:
            snip = torch.einsum("ntc,dc->ntd", snip, W)

        # Welch
        snip = snip.permute(0, 2, 1).reshape(-1, snip.shape[1])
        periodogram = torch.fft.rfft(snip, n=block_len, norm="ortho")
        dens = (periodogram * periodogram.conj()).mean(dim=0)
        snip_psds.append(dens)

    snip_psds = torch.stack(snip_psds, dim=0)
    spectral_density = snip_psds.mean(0).sqrt_()

    # estimate 0-phase FIR whitener
    wkernel = torch.fft.irfft(1.0 / spectral_density, n=block_len)
    wkernel = torch.fft.fftshift(wkernel)

    # trim to requested length
    assert temporal_length <= wkernel.shape[0]
    i0 = wkernel.shape[0] // 2 - temporal_length // 2
    i1 = i0 + temporal_length
    wkernel = wkernel[i0:i1].clone()
    assert wkernel.isfinite().all()

    return wkernel


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


class Whitener(BModule):
    def __init__(
        self, whitener: Tensor, covariance: Tensor, temporal_kernel: Tensor | None
    ):
        super().__init__()
        self.register_buffer("whitener", whitener)
        self.register_buffer("covariance", covariance)
        self.temporal = temporal_kernel is not None
        self.register_buffer_or_none("temporal_kernel", temporal_kernel)
        if temporal_kernel is not None:
            self.temporal_length = temporal_kernel.shape[0]
            tk_twice = self._convolve(temporal_kernel)
        else:
            self.temporal_length = 0
            tk_twice = None
        self.register_buffer_or_none("temporal_kernel_twice", tk_twice)

    @classmethod
    def blank(cls, n_channels: int, device: torch.device, temporal_length: int | None):
        w = torch.zeros((n_channels, n_channels), device=device)
        if temporal_length:
            k = torch.zeros((temporal_length,), device=device)
        else:
            k = None
        return cls(w, torch.zeros_like(w), k)

    @classmethod
    def from_numpy(
        cls,
        whitener: np.ndarray,
        covariance: np.ndarray,
        temporal_kernel: np.ndarray | None,
    ):
        if temporal_kernel is None:
            tk = None
            tmsg = ""
        else:
            tmsg = f" with temporal length {temporal_kernel.shape[0]}"
            tk = torch.asarray(temporal_kernel)
        logger.dartsortverbose("Load whitener%s from numpy.", tmsg)
        return cls(
            whitener=torch.asarray(whitener),
            covariance=torch.asarray(covariance),
            temporal_kernel=tk,
        )

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        tk = self.b.temporal_kernel
        if tk is not None:
            tk = tk.numpy(force=True)
        return (
            self.b.whitener.numpy(force=True),
            self.b.covariance.numpy(force=True),
            tk,
        )

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
            "Estimating %s-%s whitener%s.",
            whiten_cfg.strategy,
            whiten_cfg.estimator,
            ""
            if not whiten_cfg.temporal_length
            else f"temporal length: {whiten_cfg.temporal_length}",
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

        if whiten_cfg.temporal_length:
            assert whiten_cfg.strategy != "postwhiten"
            temporal_kernel = residual_welch_whitener(
                sorting=sorting,
                device=device,
                temporal_length=whiten_cfg.temporal_length,
                spatial_whitener=whitener,
            )
            assert temporal_kernel.shape == (whiten_cfg.temporal_length,)
        else:
            temporal_kernel = None

        return cls(whitener=whitener, covariance=cov, temporal_kernel=temporal_kernel)

    def _convolve(self, x: Tensor, twice=False, padding="same"):
        if not self.temporal:
            return x
        *shp, t = x.shape
        x = x.reshape(-1, 1, t)
        if twice:
            k = self.b.temporal_kernel_twice
        else:
            k = self.b.temporal_kernel
        if padding == "full":
            padding = k.shape[0] - 1
        k = k.to(device=x.device)
        res = F.conv1d(
            input=x,
            weight=k[None, None],
            padding=padding,
            groups=x.shape[1],
        )
        if padding == "same":
            assert res.shape[-1] == t
            ot = t
        else:
            assert isinstance(padding, int)
            ot = t + padding
        res = res.reshape(*shp, ot)
        return res

    def whiten_traces_spatial_major(
        self, x: Tensor, out: Tensor | None = None, padding="same"
    ) -> Tensor:
        assert x.ndim == 2
        out = torch.mm(self.b.whitener, x.T, out=out)
        out = self._convolve(out, padding=padding)
        return out

    def whiten(
        self, x: Tensor, out: Tensor | None = None, spatial_only: bool = False
    ) -> Tensor:
        *shp, c = x.shape
        x = x.reshape(-1, c)
        x = torch.mm(x, self.b.whitener.T, out=out)
        x = x.reshape(*shp, c)
        if self.temporal and not spatial_only:
            x = self._convolve(x.mT).mT
        return x

    def transpose_whiten(
        self, x: Tensor, out: Tensor | None = None, spatial_only: bool = False
    ) -> Tensor:
        *shp, c = x.shape
        x = x.reshape(-1, c)
        x = torch.mm(x, self.b.whitener, out=out)
        x = x.reshape(*shp, c)
        if self.temporal and not spatial_only:
            x = self._convolve(x.mT).mT
        return x

    def prec_mul(self, x: Tensor) -> Tensor:
        *shp, c = x.shape
        x = x.reshape(-1, c)
        x = x @ (self.b.whitener.T @ self.b.whitener)
        x = x.reshape(*shp, c)
        if self.temporal:
            x = self._convolve(x.mT, twice=True).mT
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
