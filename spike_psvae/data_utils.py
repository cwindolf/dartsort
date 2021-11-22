import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .localization import localize_waveforms_batched


class ContiguousRandomBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        # TODO this is copied code, not written to be reproducible yet.
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.N = len(data_source)
        self.batch_size = batch_size
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.start_inds = batch_size * torch.arange(self.N // batch_size)

    def __iter__(self):
        yield from (
            range(self.start_inds[si], self.start_inds[si] + self.batch_size)
            for si in torch.randperm(
                self.N // self.batch_size, generator=self.generator
            )
        )


class SpikeHDF5Dataset(Dataset):
    def __init__(self, h5_path, x, ys, y_min=None):
        self.h5 = h5py.File(h5_path, "r")
        self.x = self.h5[x]
        self.ys = torch.tensor(
            np.stack(
                [self.h5[y][:].astype(np.float32) for y in ys],
                axis=1,
            )
        )
        self.len = len(self.ys)

        self.y_min = y_min
        if y_min is not None and "y" in ys:
            self.good_inds = np.flatnonzero(self.h5["y"][:] > y_min)
            self.len = len(self.good_inds)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.y_min is not None:
            idx = self.good_inds[idx]

        return torch.tensor(self.x[idx], dtype=torch.float), self.ys[idx]


class LocalizingHDF5Dataset(Dataset):
    def __init__(
        self,
        waveforms,
        geom,
        ys,
        y_min=None,
        channel_radius=10,
        geomkind="updown",
    ):
        self.x = torch.as_tensor(waveforms, dtype=torch.float)
        xs, ys, z_rels, z_abss, alphas = localize_waveforms_batched(
            waveforms,
            geom,
            maxchans=None,
            channel_radius=channel_radius,
            n_workers=1,
            jac=False,
            geomkind=geomkind,
            batch_size=512,
        )
        data = dict(x=xs, y=ys, z_rel=z_rels, z_abs=z_abss, alpha=alphas)
        self.ys = torch.tensor(
            np.stack(
                [data[y][:].astype(np.float32) for y in ys],
                axis=1,
            )
        )
        self.len = len(self.ys)

        self.y_min = y_min
        if y_min is not None and "y" in ys:
            self.good_inds = np.flatnonzero(ys > y_min)
            self.len = len(self.good_inds)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.y_min is not None:
            idx = self.good_inds[idx]

        return self.x[idx], self.ys[idx]
