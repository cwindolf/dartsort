from pathlib import Path
from threading import local
from typing import Literal, cast

import h5py
import torch
import numpy as np
from tqdm.auto import tqdm

from ..util.internal_config import ComputationConfig
from ..util.job_util import ensure_computation_config
from ..util.multiprocessing_util import pool_from_cfg
from ..util.py_util import databag
from .transform_base import BaseWaveformFeaturizer

_my_name = "reduction_data"


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
        if not self.online:
            names = [f"{pfx}{_my_name}"]
        else:
            names = []
        super().__init__(geom=geom, channel_index=channel_index, name=names)
        self.name_prefix = name_prefix
        self.reduction = reduction
        self.n_units = n_units
        self.feature_dim = feature_dim
        self.output_channels = output_channels
        self.shape = [(self.feature_dim, output_channels)]
        self.dtype = [dtype]
        self.with_raw_std_dev = with_raw_std_dev
        self._initialize((self.feature_dim, output_channels))

    def transform(self, waveforms: torch.Tensor, **fixed_properties: torch.Tensor):
        assert waveforms.shape[1] == self.feature_dim
        assert waveforms.shape[2] == self.output_channels
        labels = fixed_properties["labels"]

        # if median, early out
        if not self.online:
            return {self.name[0]: waveforms}
        if not waveforms.numel():
            return {}

        # else, Welford running mean algorithm below

        # zero the scratch storage
        batch_xbar = self.b.batch_xbar.zero_()
        if self.with_raw_std_dev:
            batch_xsqbar = self.b.batch_xsqbar.zero_()
        else:
            batch_xsqbar = None

        # sum weight per unit
        if "weights" in fixed_properties:
            weights = fixed_properties["weights"]
        else:
            weights = self.b.count.new_ones((1,)).broadcast_to(labels.shape)
        batch_w = self.b.count.new_zeros(self.n_units)
        batch_w.scatter_add_(index=labels, dim=0, src=weights)

        # per-spike weight for averaging inside the batch
        spike_w = weights / batch_w[labels]
        x = waveforms.nan_to_num()
        wx = x * spike_w[:, None, None]

        # channelwise weights for this batch
        nz = waveforms[0, 0].isfinite().to(batch_w)
        batch_w = batch_w[:, None] * nz

        # Welford weights
        self.count += batch_w
        batch_w /= self.count
        batch_w.nan_to_num_()

        # handle means
        labels_ix = labels[:, None, None].broadcast_to(wx.shape)
        batch_xbar.scatter_add_(dim=0, index=labels_ix, src=wx)
        self.mean += batch_xbar.sub_(self.b.mean).mul_(batch_w[:, None])
        if self.with_raw_std_dev:
            wxx = x.mul_(wx)
            assert batch_xsqbar is not None
            batch_xsqbar.scatter_add_(dim=0, index=labels_ix, src=wxx)
            self.meansq += batch_xsqbar.sub_(self.b.meansq).mul_(batch_w[:, None])

        return {}

    def reduction_results(
        self,
        hdf5_path: Path,
        labels: torch.Tensor,
        show_progress: bool = False,
        computation_cfg: ComputationConfig | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Gather mean or median results."""
        # if reduction was mean, done already
        if self.online:
            if self.with_raw_std_dev:
                std = self.b.meansq.sub_(self.b.mean.square()).abs_().sqrt_()
                std = std.numpy(force=True)
            else:
                std = None
            return self.b.count.numpy(force=True), self.b.mean.numpy(force=True), std

        # else, nanmedian in a loop
        computation_cfg = ensure_computation_config(computation_cfg)
        dev = computation_cfg.actual_device()

        n_jobs, Executor, context, *_ = pool_from_cfg(
            computation_cfg, check_local=True, small=True
        )
        with Executor(
            max_workers=n_jobs,
            mp_context=context,
            initializer=_reduction_init,
            initargs=(hdf5_path, self.name[0], labels, dev, self.with_raw_std_dev),
        ) as pool:
            results = pool.map(_reduction_job, range(self.n_units))
            if show_progress:
                results = tqdm(
                    results, total=self.n_units, desc=f"Medians:{dev.type}:{n_jobs}"
                )

            count = np.full(
                (self.n_units, self.output_channels), dtype=np.int32, fill_value=-1
            )
            wf_shape = (self.feature_dim, self.output_channels)
            mean = np.full(
                (self.n_units, *wf_shape), dtype=np.float32, fill_value=np.nan
            )
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

        global _reduction_stuff
        del _reduction_stuff.ctx
        _reduction_stuff.ctx = None
        assert (count >= 0).all()
        assert count.max() > 0
        mean = np.nan_to_num(mean, copy=False)

        return count, mean, std

    def _initialize(self, wf_shape: tuple[int, int]):
        if not self.online:
            return
        self.register_buffer("count", torch.zeros((self.n_units, wf_shape[1])))
        self.register_buffer("mean", torch.zeros((self.n_units, *wf_shape)))
        self.register_buffer("batch_xbar", torch.zeros((self.n_units, *wf_shape)))
        if self.with_raw_std_dev:
            self.register_buffer("meansq", self.b.mean.clone())
            self.register_buffer("batch_xsqbar", self.b.mean.clone())
        else:
            self.register_buffer_or_none("meansq", None)
            self.register_buffer_or_none("batch_xsqbar", None)


_reduction_stuff = local()
_reduction_stuff.ctx = None


@databag
class _ReductionStuff:
    h5: h5py.File
    dataset: h5py.Dataset
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
    global _reduction_stuff
    h5 = h5py.File(hdf5_path, "r", locking=False, swmr=True, libver="latest")
    _reduction_stuff.ctx = _ReductionStuff(
        h5=h5,
        dataset=cast(h5py.Dataset, h5[dataset_name]),
        labels=torch.asarray(labels, dtype=torch.int32, copy=True, device=dev),
        dev=dev,
        do_std=do_std,
    )


def _reduction_job(j: int) -> _ReductionResult | None:
    global _reduction_stuff
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
