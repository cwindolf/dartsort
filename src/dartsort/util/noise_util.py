import h5py
import numpy as np
import pandas as pd
import torch
import linear_operator
from linear_operator import operators
from scipy.fftpack import next_fast_len
from tqdm.auto import trange

from ..util import drift_util, spiketorch


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
        unit_batch_size=32,
    ):
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

    Default here for now is factorized_by_rank_rank_diag. This is because empirically,
    rank covs were observed to be diagonal-ish (not extremely close, but not super
    far). And, spatial covs look to be of spatially-decaying kernel type, but the band
    width depends on the rank (since ranks ~ frequency bands). So let's model those
    differences by default -- it doesn't cost much.
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
    ):
        super().__init__()
        self.rank = rank
        self.n_channels = n_channels
        self.mean_kind = mean_kind
        self.cov_kind = cov_kind

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

        # precompute stuff
        self._full_cov = None
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

    def whiten(self, data, channels=slice(None)):
        assert self.mean_kind == "zero"
        cov = self.marginal_covariance(channels=channels)
        assert data.ndim == 3
        data = data.reshape(len(data), -1)
        res = linear_operator.sqrt_inv_matmul(cov, data.unsqueeze(2))
        assert res.ndim == 3 and res.shape == (*data.shape, 1)
        return res

    def marginal_covariance(self, channels=slice(None), cache_key=None, device=None):
        if device is not None:
            device = torch.device(device)
        if cache_key is not None and cache_key in self.cache:
            res = self.cache[cache_key]
            if device is not None and device != res.device:
                res = res.to(device)
                self.cache[cache_key] = res
            return res
        if channels == slice(None):
            if self._full_cov is None:
                self._full_cov = self._marginal_covariance()
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

    def _marginal_covariance(self, channels=slice(None)):
        channels = self.chans_arange[channels]
        nc = channels.numel()

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

        if self.cov_kind == "factorized":
            rank_root = self.rank_vt.T * self.rank_std
            rank_root = operators.RootLinearOperator(rank_root)
            chan_root = self.channel_vt.T * self.channel_std
            chan_root = operators.RootLinearOperator(chan_root)
            return torch.kron(rank_root, chan_root)

        if self.cov_kind == "factorized_rank_diag":
            rank_cov = operators.DiagLinearOperator(self.rank_std.square())
            chan_root = self.channel_vt.T[channels] * self.channel_std
            chan_cov = chan_root @ chan_root.T
            # chan_cov = operators.RootLinearOperator(chan_root)
            return torch.kron(rank_cov, chan_cov)

        if self.cov_kind == "factorized_by_rank_rank_diag":
            chans_vt = self.channel_vt[:, :, channels]
            r, C, c = chans_vt.shape
            blocks = chans_vt.new_empty((r, c, c))
            chans_std = self.channel_std[:, :]  # no slice here!
            for q, sq in enumerate(self.rank_std):
                blocks[q] = sq * chans_vt[q].T @ chans_vt[q]
            blocks = linear_operator.to_linear_operator(blocks)
            return operators.BlockDiagLinearOperator(blocks)

        assert False

    @classmethod
    def estimate(cls, snippets, mean_kind="zero", cov_kind="scalar"):
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

        # spatial part could be "by rank" or same for all ranks
        # either way, there are nans afoot
        if "by_rank" in cov_kind:
            channel_std = torch.ones_like(x_spatial[0])
            channel_vt = torch.zeros((rank, n_channels, n_channels)).to(channel_std)
            for q in range(rank):
                xq = x_spatial[:, q]
                validq = xq.isfinite().any(0)
                covq = spiketorch.nancov(xq[:, validq])
                fullcovq = torch.eye(xq.shape[1], dtype=covq.dtype, device=covq.device)
                fullcovq[validq[:, None] & validq[None, :]] = covq.view(-1)
                qeig, qv = torch.linalg.eigh(fullcovq)
                channel_std[q] = qeig.sqrt()
                channel_vt[q] = qv.T
        else:
            x_spatial = x_spatial.reshape(n * rank, n_channels)
            cov_spatial = spiketorch.nancov(x_spatial)
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
            **init_kw,
        )

    @classmethod
    def estimate_from_hdf5(
        cls,
        hdf5_path,
        mean_kind="zero",
        cov_kind="factorized_by_rank_rank_diag",
        motion_est=None,
        interpolation_method="kriging",
        sigma=20.0,
        device=None,
    ):
        from dartsort.util.drift_util import registered_geometry
        with h5py.File(hdf5_path, "r", locking=False) as h5:
            geom = h5["geom"][:]
        rgeom = registered_geometry(geom, motion_est=motion_est)
        snippets = interpolate_residual_snippets(
            motion_est,
            hdf5_path,
            geom,
            rgeom,
            sigma=sigma,
            interpolation_method=interpolation_method,
            device=device
        )
        return cls.estimate(snippets, mean_kind=mean_kind, cov_kind=cov_kind)


def interpolate_residual_snippets(
    motion_est,
    hdf5_path,
    geom,
    registered_geom,
    sigma=10.0,
    residual_times_s_dataset_name="residual_times_seconds",
    residual_dataset_name="residual",
    channels_mode="round",
    interpolation_method="normalized",
    workers=None,
    device=None,
):
    """PCA-embed and interpolate residual snippets to the registered probe"""
    from dartsort.util import interpolation_util, data_util

    if device is None:
        device = "cuda" if torch.cuda.is_available() else None

    tpca = data_util.get_tpca(hdf5_path)
    if device is not None:
        tpca = tpca.to(device)

    with h5py.File(hdf5_path, "r", locking=False) as h5:
        snippets = h5[residual_dataset_name][:]
        times_s = h5[residual_times_s_dataset_name][:]
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
    # source positions
    source_geom = torch.asarray(geom).to(snippets)
    psc, psd = source_geom.shape
    source_pos = source_geom[None].broadcast_to(n, psc, psd).contiguous()
    source_depths = source_pos[:, :, 1].reshape(-1)
    source_t = times_s[:, None].broadcast_to(source_pos[:, :, 1].shape).reshape(-1)
    source_shifts = 0
    if motion_est is not None:
        source_reg_depths = motion_est.correct_s(source_t.cpu(), source_depths.cpu())
        source_reg_depths = torch.asarray(source_reg_depths).to(snippets)
        source_shifts = source_reg_depths - source_depths
        source_shifts = source_shifts.reshape(source_pos[:, :, 1].shape)

    # target positions
    # we'll query the target geom for the closest source pos, within reason
    kdtree = drift_util.KDTree(registered_geom)
    prgeom = interpolation_util.pad_geom(registered_geom)
    match_distance = drift_util.pdist(geom).min() / 2
    _, targ_inds = kdtree.query(
        source_pos.reshape(-1, psd).numpy(force=True),
        distance_upper_bound=match_distance,
        workers=workers or -1,
    )
    targ_inds = torch.from_numpy(targ_inds).reshape(source_pos.shape[:2])
    target_pos = prgeom[targ_inds].to(snippets)
    if motion_est is not None:
        target_pos[:, :, 1] = target_pos[:, :, 1] - source_shifts

    # if kriging, we need a pseudoinverse
    skis = None
    if interpolation_method.startswith("kriging"):
        skis = interpolation_util.get_source_kernel_pinvs(source_geom, None, sigma=sigma)
        skis = skis.broadcast_to((len(snippets), *skis.shape[1:]))

    # allocate output storage with an extra channel of NaN needed later
    snippets = interpolation_util.kernel_interpolate(
        snippets,
        source_pos,
        target_pos,
        source_kernel_invs=skis,
        sigma=sigma,
        allow_destroy=True,
        interpolation_method=interpolation_method,
    )

    # now, let's embed these into the full registered probe
    snippets_full = snippets.new_full((n, tpca.rank, len(registered_geom)), torch.nan)
    targ_inds = targ_inds[:, None].broadcast_to(snippets.shape)
    snippets_full.scatter_(2, targ_inds.to(snippets.device), snippets)

    return snippets_full


def get_discovery_control(
    units,
    tp_scores,
    tp_labels,
    fp_scores,
    fp_labels,
    fp_num_frames,
    rec_total_frames,
):
    """Pick thresholds for deconvolution

    TP scores are the deconv thresholds at which "true positive" (well,
    at least from our best guess) spikes would be matched during deconv.
    FP scores are the deconv thresholds at which false positive (according
    to our noise estimate) events would be detected. Note that some units
    will have no false positives -- they're just that good.

    Then:
      - (unit_tp_scores < thresh).mean() is an estimate of the false negative
        rate for that unit. If the clustering was 'unbiased', it's a good estimate.

        If clustering was conservative (biased to high scoring spikes), it
        underestimates the FNR. If we pick largest thresh s.t. this
        underest < maxfnr, then we picked a threshold that was too big.

      - (unit_fp_scores > thresh).sum() * (rec_total_frames / fp_num_frames) * n_spikes_unit
        estimates the number of false positive spikes per real spike for that unit.

    fnr control options. max fnr = alpha.
      - per-unit alpha: pick thresh = alpha qtile per unit. (max thresh s.t. fnr est < alpha)
      - worst-unit alpha: min of above.
          (why should we choose to miss any spikes in big units? it won't even cost FPs.)
      - fudged per unit: fudge_factor * per-unit alpha
          (account for conservative clustering bias. but how to pick fudge_factor?)
      - worst-unit fudged.

    fp control options: max fp per input spike = K, say 2.5 or 5?
      - per-unit: thresh = min thresh s.t. fp/spk est < K
      - worst-unit: max of above

    I think a reasonable goal is to do the best we can to control FNR while not
    blowing up the spike count. In other words, the threshold we choose is
        max(count_control_threshold, fnr_control_threshold).

    This can be done globally or per unit. If global, we do...
        max(count_control_thresholds.max(), fnr_control_thresholds.min()).
    """
    pass
