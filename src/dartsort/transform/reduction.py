from pathlib import Path
from typing import Literal, cast

import h5py
import torch
from tqdm.auto import trange

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

    def transform(
        self,
        waveforms: torch.Tensor,
        *,
        labels: torch.Tensor,
        **fixed_properties,
    ):
        assert waveforms.shape[1] == self.feature_dim
        assert waveforms.shape[2] == self.output_channels

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
        self, hdf5_path: Path, labels: torch.Tensor, show_progress: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Gather mean or median results."""
        # if reduction was mean, done already
        if self.online:
            if self.with_raw_std_dev:
                std = self.b.meansq.sub_(self.b.mean.square()).abs_().sqrt_()
            else:
                std = None
            return self.b.count, self.b.mean, std

        # else, nanmedian in a loop
        count = torch.zeros((self.n_units, self.output_channels), dtype=torch.int32)
        wf_shape = (self.feature_dim, self.output_channels)
        mean = torch.zeros((self.n_units, *wf_shape))
        if self.with_raw_std_dev:
            std = mean.clone()
        else:
            std = None
        with h5py.File(hdf5_path, "r", locking=False) as h5:
            dataset = cast(h5py.Dataset, h5[self.name[0]])
            assert dataset.shape[1:] == wf_shape
            if show_progress:
                it = trange(self.n_units, desc="Medians")
            else:
                it = range(self.n_units)
            for j in it:
                (inu,) = (labels == j).cpu().nonzero(as_tuple=True)
                if not inu.numel():
                    continue
                x = torch.asarray(dataset[inu.numpy()])
                count[j] = x[:, 0].isfinite().sum(0)
                mean[j] = torch.nanmedian(x, dim=0).values
                if std is not None:
                    xbar = x.nanmean(dim=0)
                    xsqbar = x.square_().nanmean(dim=0)
                    std[j] = xsqbar.sub_(xbar.square_()).abs_().sqrt_()
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
