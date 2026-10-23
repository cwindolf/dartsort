from pathlib import Path
from threading import local
from typing import Literal, cast

import h5py
import numpy as np
import torch

from ..util.drift_util import bending_laplacian
from ..util.internal_config import ComputationConfig, InterpolationParams
from ..util.interpolation_util import FromFullProbeInterpolator
from ..util.job_util import ensure_computation_config
from ..util.logging_util import progbar
from ..util.motion import MotionInfo
from ..util.multiprocessing_util import current_worker_rank, pool_from_cfg
from ..util.py_util import databag
from .transform_base import BaseWaveformFeaturizer

_my_name = "reduction_data"
_worker_bufnames = ("count", "mean", "batch_xbar", "meansq", "batch_xsqbar")


class TemplateWaveformReducer(BaseWaveformFeaturizer):
    """Running mean or save-then-median template waveforms

    Algorithm switches between on the fly Welford with no backing storage
    or nanmedian over an h5 dataset. Final templates are gathered from my
    `.reduction_results()` method.

    Some reduction logic could break out to a base class if there's ever
    another reduction to do...
    """

    # i am a multi in the sense that i save 0 or 1 datasets
    is_multi = True

    def __init__(
        self,
        geom=None,
        channel_index=None,
        name_prefix=None,
        *,
        with_raw_std_dev: bool,
        n_units: int,
        feature_dim: int,
        output_channels: int,
        reduction: Literal["mean", "median"],
        dtype=torch.float32,
    ):
        self.online = reduction != "median"
        if name_prefix is None:
            pfx = ""
        else:
            pfx = name_prefix + "_"
        names = [] if self.online else [f"{pfx}{_my_name}"]
        super().__init__(geom=geom, channel_index=channel_index, name=names)
        self.name_prefix = name_prefix
        self.reduction = reduction
        self.n_units = n_units
        self.feature_dim = feature_dim
        self.output_channels = output_channels
        self.shape = [] if self.online else [(self.feature_dim, output_channels)]
        self.dtype = [] if self.online else [dtype]
        self.with_raw_std_dev = with_raw_std_dev
        self._initialize((self.feature_dim, output_channels))

    def transform(self, waveforms: torch.Tensor, **spike_data: torch.Tensor):
        assert waveforms.shape[1] == self.feature_dim
        assert waveforms.shape[2] == self.output_channels
        labels = spike_data["labels"].long()

        # if median, early out
        if not self.online:
            return {self.name[0]: waveforms}
        if not waveforms.numel():
            return {}

        # else, Welford running mean algorithm below, in per rank bufs
        rank = current_worker_rank()
        assert rank < self.b.count.shape[0]
        count = self.b.count[rank]
        mean = self.b.mean[rank]

        # zero the scratch storage
        batch_xbar = self.b.batch_xbar[rank].zero_()
        if self.with_raw_std_dev:
            meansq = self.b.meansq[rank]
            batch_xsqbar = self.b.batch_xsqbar[rank].zero_()
        else:
            meansq = batch_xsqbar = None

        # sum weight per unit
        # TODO this isn't hit by any code. but it could be if weighting was a thing.
        if "weights" in spike_data:
            weights = spike_data["weights"]
        else:
            weights = count.new_ones((1,)).broadcast_to(labels.shape)
        batch_w = count.new_zeros(self.n_units)
        batch_w.scatter_add_(index=labels, dim=0, src=weights)

        # per-spike weight for averaging inside the batch
        spike_w = weights / batch_w[labels]
        x = waveforms.nan_to_num()
        wx = x * spike_w[:, None, None]

        # channelwise weights for this batch
        nz = waveforms[0, 0].isfinite().to(batch_w)
        batch_w = batch_w[:, None] * nz

        # Welford weights
        count += batch_w
        batch_w /= count
        batch_w.nan_to_num_()

        # handle means
        labels_ix = labels[:, None, None].broadcast_to(wx.shape)
        batch_xbar.scatter_add_(dim=0, index=labels_ix, src=wx)
        mean += batch_xbar.sub_(mean).mul_(batch_w[:, None])
        if self.with_raw_std_dev:
            wxx = x.mul_(wx)
            assert meansq is not None and batch_xsqbar is not None
            batch_xsqbar.scatter_add_(dim=0, index=labels_ix, src=wxx)
            meansq += batch_xsqbar.sub_(meansq).mul_(batch_w[:, None])

        return {}

    def reduction_results(
        self,
        hdf5_path: Path,
        labels: torch.Tensor,
        show_progress: bool = False,
        computation_cfg: ComputationConfig | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Gather mean or median results."""
        # if reduction was mean, merge per-worker means
        if self.online:
            count, mean, meansq = _welford_merge(
                self.b.count, self.b.mean, self.get_optional_buffer("meansq")
            )
            if meansq is not None:
                std = meansq.sub_(mean.square()).abs_().sqrt_()
                std = std.numpy(force=True)
            else:
                std = None
            return count.numpy(force=True), mean.numpy(force=True), std

        # else, nanmedian in a loop
        computation_cfg = ensure_computation_config(computation_cfg)
        dev = computation_cfg.actual_device()

        n_jobs, Executor, context, *_ = pool_from_cfg(
            computation_cfg, check_local=True, small=True, cpu=dev.type == "cpu"
        )
        try:
            with Executor(
                max_workers=n_jobs,
                mp_context=context,
                initializer=_reduction_init,
                initargs=(hdf5_path, self.name[0], labels, dev, self.with_raw_std_dev),
            ) as pool:
                results = pool.map(_reduction_job, range(self.n_units))
                if show_progress:
                    results = progbar(
                        results, total=self.n_units, desc=f"Medians:{dev.type}:{n_jobs}"
                    )

                count = np.full(
                    (self.n_units, self.output_channels), dtype=np.int32, fill_value=-1
                )
                mean_shape = (self.n_units, self.feature_dim, self.output_channels)
                mean = np.full(mean_shape, dtype=np.float32, fill_value=np.nan)
                if self.with_raw_std_dev:
                    std = mean.copy()
                else:
                    std = None

                for r in results:
                    if r is None:
                        continue
                    count[r.j] = r.count
                    mean[r.j] = r.mean
                    if std is not None:
                        assert r.std is not None
                        std[r.j] = r.std

            assert (count >= 0).all()
            assert count.max() > 0
            mean = np.nan_to_num(mean, copy=False)
            return count, mean, std
        finally:
            _reduction_stuff.ctx = None

    def _initialize(self, wf_shape: tuple[int, int]):
        if not self.online:
            return
        self.register_buffer("count", torch.zeros((1, self.n_units, wf_shape[1])))
        self.register_buffer("mean", torch.zeros((1, self.n_units, *wf_shape)))
        self.register_buffer("batch_xbar", torch.zeros((1, self.n_units, *wf_shape)))
        if self.with_raw_std_dev:
            self.register_buffer("meansq", self.b.mean.clone())
            self.register_buffer("batch_xsqbar", self.b.mean.clone())
        else:
            self.register_buffer_or_none("meansq", None)
            self.register_buffer_or_none("batch_xsqbar", None)

    def register_workers(self, n_workers: int):
        if not self.online:
            return
        for name in _worker_bufnames:
            buf = self.get_optional_buffer(name)
            if buf is not None:
                buf.resize_((n_workers, *buf.shape[1:])).zero_()


class SuperresTemplateWaveformReducer(BaseWaveformFeaturizer):
    is_multi = True

    def __init__(
        self,
        geom,
        channel_index,
        *,
        srgeom: np.ndarray,
        n_units: int,
        feature_dim: int,
        interp_params: InterpolationParams,
        fit_radius_um: float,
        smoothness: float,
        shrinkage: float,
        motion: MotionInfo | None = None,
    ):
        super().__init__(geom=geom, channel_index=channel_index, name=[])
        self.shape = []
        self.dtype = []
        self.n_units = n_units
        self.feature_dim = feature_dim
        self.interp_params = interp_params
        self.fit_radius_um = fit_radius_um
        self.smoothness = smoothness
        self.shrinkage = shrinkage
        self.motion = motion
        self.srgeom = srgeom
        self.register_buffer("count", torch.zeros((1, n_units, 1)))
        self.register_buffer(
            "mean", torch.zeros((1, n_units, feature_dim, len(srgeom)))
        )
        self.srgeom_erp: FromFullProbeInterpolator | None = None
        self.rgeom_erp: FromFullProbeInterpolator | None = None
        self._chunk_counts: list[dict[float, torch.Tensor]] = [{}]
        self._kernel_cache: list[tuple[float, torch.Tensor] | None] = [None]

    def needs_precompute(self):
        return self.srgeom_erp is None

    def attach_motion(self, motion: MotionInfo):
        self.motion = motion

    def precompute(self):
        assert self.motion is not None
        device = self.b.geom.device
        self.srgeom_erp = FromFullProbeInterpolator(
            motion=self.motion,
            params=self.interp_params,
            device=device,
            registered_geom=self.srgeom,
        )
        self.rgeom_erp = FromFullProbeInterpolator(
            motion=self.motion, params=self.interp_params, device=device
        )

    def register_workers(self, n_workers: int):
        for name in ("count", "mean"):
            buf = getattr(self.b, name)
            buf.resize_((n_workers, *buf.shape[1:])).zero_()
        self._chunk_counts = [{} for _ in range(n_workers)]
        self._kernel_cache = [None] * n_workers

    def transform(self, waveforms: torch.Tensor, **spike_data: torch.Tensor):
        if not waveforms.numel():
            return {}
        assert waveforms.shape[1:] == (self.feature_dim, self.b.geom.shape[0])
        assert waveforms.isfinite().all()
        rank = current_worker_rank()
        assert rank < self.b.count.shape[0]
        t_s = float(spike_data["chunk_center_s"])
        kernel = self._kernel_at_time(rank, t_s)

        # this batch's mean of erp(t)^T x per unit
        units, inverse, unit_counts = torch.unique(
            spike_data["labels"].long(), return_inverse=True, return_counts=True
        )
        unit_counts = unit_counts.to(waveforms.dtype)
        batch_mean = waveforms.new_zeros((units.numel(), *waveforms.shape[1:]))
        batch_mean.index_add_(0, inverse, waveforms)
        batch_mean = batch_mean.div_(unit_counts[:, None, None]) @ kernel

        # Welford update
        total = self.b.count[rank, units] + unit_counts[:, None]
        self.b.count[rank, units] = total
        weight = (unit_counts[:, None] / total)[:, :, None]
        mean = self.b.mean[rank, units]
        self.b.mean[rank, units] = mean.add_((batch_mean - mean).mul_(weight))

        chunk_counts = self._chunk_counts[rank].get(t_s)
        if chunk_counts is None:
            chunk_counts = waveforms.new_zeros(self.n_units)
            self._chunk_counts[rank][t_s] = chunk_counts
        chunk_counts[units] += unit_counts

        return {}

    def _kernel_at_time(self, rank: int, t_s: float) -> torch.Tensor:
        cached = self._kernel_cache[rank]
        if cached is not None and cached[0] == t_s:
            return cached[1]
        assert self.srgeom_erp is not None
        kernel = self.srgeom_erp.interpolation_kernel_at_time(t_s)
        self._kernel_cache[rank] = (t_s, kernel)
        return kernel

    def superres_results(
        self,
        prior: torch.Tensor,
        main_channels: torch.Tensor,
        device: torch.device,
        unit_batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve for each unit's local template (fit_radius_um) on srgeom

        Minimizes sum_i ||x_i - A(t_i) F||^2 + n shrinkage ||F - F0||^2 + smoothness ||L F||^2
        """
        assert self.srgeom_erp is not None and self.rgeom_erp is not None
        self.to(device)
        count, mean, _ = _welford_merge(self.b.count, self.b.mean, None)
        chunk_counts = _merge_chunk_counts(self._chunk_counts, device)

        # prior on srgeom, and fit nodes
        prior = prior.to(device=device, dtype=torch.float)
        assert prior.isfinite().all()
        srgeom = self.srgeom_erp.b.rgeom
        prior = self.rgeom_erp.interp_to(srgeom, prior)
        centers = self.rgeom_erp.b.rgeom[main_channels.to(device)]
        is_fit = torch.cdist(centers, srgeom) <= self.fit_radius_um
        n_fit = is_fit.sum(1)
        # each unit's fit nodes first, then padding with other nodes
        fit_nodes = torch.argsort(
            is_fit.logical_not().to(torch.int8), dim=1, stable=True
        )
        fit_nodes = fit_nodes[:, : int(n_fit.max())]
        valid = torch.arange(fit_nodes.shape[1], device=device) < n_fit[:, None]
        prior_out = prior * is_fit.logical_not()[:, None, :]

        # sum over spikes of erp^T erp and erp_fit^T erp F0_out, by chunk
        n_nodes = len(self.srgeom)
        gram = prior.new_zeros((self.n_units, fit_nodes.shape[1], fit_nodes.shape[1]))
        cross = prior.new_zeros((self.n_units, fit_nodes.shape[1], self.feature_dim))
        node_counts = prior.new_zeros((self.n_units, n_nodes))
        for t_s in sorted(chunk_counts):
            (units,) = chunk_counts[t_s].nonzero(as_tuple=True)
            n = chunk_counts[t_s][units][:, None, None]
            kernel = self.srgeom_erp.interpolation_kernel_at_time(t_s)
            kernel_fit = kernel[:, fit_nodes[units]].permute(1, 0, 2)
            kernel_fit.mul_(valid[units][:, None, :])
            gram[units] += n * torch.bmm(kernel_fit.mT, kernel_fit)
            model_out = prior_out[units] @ kernel.T
            cross[units] += n * torch.bmm(kernel_fit.mT, model_out.mT)
            node_counts[units] += n[:, :, 0] * kernel.ne(0).any(dim=0)
            del kernel, kernel_fit, model_out

        # solve in batches of units
        laplacian = bending_laplacian(self.srgeom)
        laplacian = torch.asarray(laplacian, device=device)
        features = prior.clone()
        for i0 in range(0, self.n_units, unit_batch_size):
            ix = slice(i0, min(i0 + unit_batch_size, self.n_units))
            nodes = fit_nodes[ix]
            ok = valid[ix].double()
            node_ix = nodes[:, None, :].expand(-1, self.feature_dim, -1)

            lap_fit = laplacian[:, nodes].permute(1, 0, 2).mul_(ok[:, None, :])
            r0 = laplacian @ prior_out[ix].double().mT
            P = torch.bmm(lap_fit.mT, lap_fit)
            q = torch.bmm(lap_fit.mT, r0)
            del lap_fit, r0

            prior_fit = prior[ix].gather(2, node_ix).mT.double()
            data_fit = mean[ix].gather(2, node_ix).mT.double() * count[ix, :, None]
            shrinkage = self.shrinkage * count[ix, :, None].double().clamp(min=1.0)
            system = (
                gram[ix].double()
                + shrinkage * torch.diag_embed(ok)
                + self.smoothness * P
                + torch.diag_embed(1.0 - ok)
            )
            rhs = data_fit - cross[ix].double() + shrinkage * prior_fit
            rhs = rhs.sub_(self.smoothness * q).mul_(ok[:, :, None])
            fit = torch.linalg.solve(system, rhs)
            fit = torch.where(ok[:, :, None] > 0, fit, prior_fit)
            features[ix] = features[ix].scatter(2, node_ix, fit.mT.float())
            del P, q, system, rhs, fit, prior_fit, data_fit

        return node_counts.numpy(force=True), features.numpy(force=True)


def _merge_chunk_counts(
    worker_chunk_counts: list[dict[float, torch.Tensor]], device: torch.device
) -> dict[float, torch.Tensor]:
    merged = {}
    for chunk_counts in worker_chunk_counts:
        for t_s, counts in chunk_counts.items():
            counts = counts.to(device)
            if t_s in merged:
                merged[t_s] += counts
            else:
                merged[t_s] = counts.clone()
    return merged


def _welford_merge(
    count: torch.Tensor, mean: torch.Tensor, meansq: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    total = count[0]
    merged_mean = mean[0]
    merged_meansq = None if meansq is None else meansq[0]
    for w in range(1, count.shape[0]):
        total += count[w]
        weight = (count[w] / total).nan_to_num_()[:, None]
        merged_mean += (mean[w] - merged_mean).mul_(weight)
        if merged_meansq is not None:
            assert meansq is not None
            merged_meansq += (meansq[w] - merged_meansq).mul_(weight)
    return total, merged_mean, merged_meansq


_reduction_stuff = local()
_reduction_stuff.ctx = None


@databag
class _ReductionStuff:
    h5: h5py.File
    dataset: np.ndarray
    labels: torch.Tensor
    dev: torch.device
    do_std: bool


@databag
class _ReductionResult:
    j: int
    count: np.ndarray
    mean: np.ndarray
    std: np.ndarray | None


def _reduction_init(
    hdf5_path: Path,
    dataset_name: str,
    labels: torch.Tensor,
    dev: torch.device,
    do_std: bool,
):

    labels = torch.asarray(labels, dtype=torch.int32, device=dev)

    # load labels of spikes that entered reduction
    with h5py.File(hdf5_path, "r", locking=False, libver="latest") as h5:
        labels = labels[h5["indices"][:]]
        assert h5.get(dataset_name, getclass=True) == h5py.Dataset
        assert h5[dataset_name].chunks is None
        assert h5[dataset_name].compression is None
        dtype = h5[dataset_name].dtype
        shape = h5[dataset_name].shape
        offset = h5[dataset_name].id.get_offset()
        assert shape[:1] == labels.shape

    # make a memmap of the waveforms dataset
    dataset = np.memmap(hdf5_path, dtype=dtype, mode="r", offset=offset, shape=shape)

    _reduction_stuff.ctx = _ReductionStuff(
        h5=h5,
        dataset=dataset,
        labels=labels,
        dev=dev,
        do_std=do_std,
    )


def _reduction_job(j: int) -> _ReductionResult | None:
    p = cast(_ReductionStuff, _reduction_stuff.ctx)
    (inu,) = (p.labels == j).nonzero(as_tuple=True)
    inu = inu.cpu()
    if not inu.numel():
        return None

    x = torch.asarray(p.dataset[inu.numpy()], device=p.dev)
    count = x[:, 0].isfinite().sum(0)
    mean = torch.nanmedian(x, dim=0).values
    if p.do_std:
        xbar = x.nanmean(dim=0)
        xsqbar = x.square_().nanmean(dim=0)
        std = xsqbar.sub_(xbar.square_()).abs_().sqrt_()
        std = std.numpy(force=True)
    else:
        std = None

    return _ReductionResult(
        j=j, count=count.numpy(force=True), mean=mean.numpy(force=True), std=std
    )
