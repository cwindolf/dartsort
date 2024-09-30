import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dartsort.util import spiketorch
from dartsort.detect import detect_and_deduplicate
from scipy.fftpack import next_fast_len
from tqdm.auto import trange


class FactorizedNoise(torch.nn.Module):
    """Spatial/temporal factorized noise. See .estimate().
    """

    def __init__(self, spatial_std, vt_spatial, temporal_std, vt_temporal):
        super().__init__()
        self.spatial_std = spatial_std
        self.vt_spatial = vt_spatial
        self.temporal_std = temporal_std
        self.vt_temporal = vt_temporal

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
        sqrt_nt_minus_1 = torch.sqrt(torch.tensor(n * t - 1, dtype=snippets.dtype))
        sqrt_nc_minus_1 = torch.sqrt(torch.tensor(n * c - 1, dtype=snippets.dtype))
        assert n * t > c ** 2
        assert n * c > t ** 2

        # estimate spatial covariance
        x_spatial = snippets.view(n * t, c)
        u_spatial, spatial_sing, vt_spatial = torch.linalg.svd(x_spatial, full_matrices=False)
        spatial_std = spatial_sing / sqrt_nt_minus_1

        # extract whitened temporal snips
        x_temporal = u_spatial.view(n, t, c).permute(0, 2, 1).reshape(n * c, t)
        x_temporal.mul_(sqrt_nt_minus_1)
        _, temporal_sing, vt_temporal = torch.linalg.svd(x_temporal, full_matrices=False)
        del _
        temporal_std = temporal_sing / sqrt_nc_minus_1

        return cls(spatial_std, vt_spatial, temporal_std, vt_temporal)


class StationaryFactorizedNoise(torch.nn.Module):
    def __init__(self, spatial_std, vt_spatial, kernel_fft, block_size, t):
        super().__init__()
        self.spatial_std = spatial_std
        self.vt_spatial = vt_spatial
        self.kernel_fft = kernel_fft
        self.block_size = block_size
        self.t = t

    def simulate(self, size=1, t=None, generator=None):
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

        # need extra room at the edges to do valid convolution
        t_padded = t + self.t - 1
        noise = torch.randn(size * c, t_padded, generator=generator, device=device)
        noise = spiketorch.single_inv_oaconv1d(
            noise, s2=self.t, f2=self.kernel_fft, block_size=self.block_size, norm="ortho"
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
        sqrt_nt_minus_1 = torch.sqrt(torch.tensor(n * t - 1, dtype=snippets.dtype))
        assert n * t > c ** 2
        assert n * c > t

        # estimate spatial covariance
        x_spatial = snippets.view(n * t, c)
        u_spatial, spatial_sing, vt_spatial = torch.linalg.svd(x_spatial, full_matrices=False)
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
        unit_batch_size=32,
    ):
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
        negnormsq = spatial_singular.square().sum((1, 2)).neg_().unsqueeze(1)
        nu, nt = temporal.shape[:2]
        obj = None  # var for reusing buffers
        units = []
        scores = []
        for j in trange(size, desc="False positives"):
            sample = self.simulate(t=t + nt - 1, generator=generator)[0].T
            obj = spiketorch.convolve_lowrank(
                sample,
                spatial_singular,
                temporal,
                out=obj,
            )
            assert obj.shape == (nu, t)
            obj = torch.add(negnormsq, obj, alpha=2.0, out=obj)

            # find peaks...
            peak_times, peak_units, peak_energies = detect_and_deduplicate(
                obj.T,
                min_threshold,
                peak_sign="pos",
                dedup_temporal_radius=radius,
                return_energies=True,
            )
            units.append(peak_units.numpy(force=True))
            scores.append(peak_energies.numpy(force=True))

        total_samples = size * t
        df = pd.DataFrame(
            dict(
                units=np.concatenate(units),
                scores=np.concatenate(scores),
            )
        )

        return total_samples, df
