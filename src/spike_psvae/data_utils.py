import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .localization import localize_waveforms_batched
from .waveform_utils import get_local_waveforms


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


class SpikeDataset(Dataset):
    def __init__(
        self, waveforms, ys, good_inds=None, minimum=None, maximum=None
    ):
        self.waveforms = waveforms
        self.ys = ys
        self.good_inds = good_inds
        self.len = len(ys) if good_inds is None else len(good_inds)
        self.minimum = minimum
        self.half_dminmax = None
        if maximum is not None:
            self.half_dminmax = (maximum - minimum) / 2.0

    def __len__(self):
        return self.len

    def normalize(self, input):
        input = input - self.minimum
        input /= self.half_dminmax
        input -= 1.0
        return input

    def unnormalize(self, input):
        input = input + 1.0
        input *= self.half_dminmax
        input += self.minimum
        return input

    def __getitem__(self, idx):
        if self.good_inds is not None:
            idx = self.good_inds[idx]

        bx = torch.as_tensor(self.waveforms[idx], dtype=torch.float)
        by = self.ys[idx]

        if self.minimum is not None:
            bx = self.normalize(bx)

        return bx, by


class SpikeHDF5Dataset(SpikeDataset):
    def __init__(self, h5_path, x, supkeys, y_min=None, standardize=True):
        self.h5 = h5py.File(h5_path, "r")
        x = self.h5[x]
        sups = torch.tensor(
            np.stack(
                [self.h5[key][:].astype(np.float32) for key in supkeys],
                axis=1,
            )
        )

        self.y_min = y_min
        good_inds = None
        if y_min is not None and "y" in supkeys:
            good_inds = np.flatnonzero(self.h5["y"][:] > y_min)

        mins = maxs = None
        if standardize:
            assert "minimum" in self.h5
            mins = self.h5["minimum"]
            maxs = self.h5["maximum"]

        super().__init__(
            x, sups, good_inds=good_inds, minimum=mins, maximum=maxs
        )


class LocalizingHDF5Dataset(SpikeDataset):
    def __init__(
        self,
        waveforms,
        geom,
        supkeys,
        y_min=None,
        channel_radius=10,
        repeat_to_min_length=500_000,
        standardize=True,
        geomkind="updown",
    ):
        local_wfs, maxchans = get_local_waveforms(
            waveforms, channel_radius, geom, maxchans=None, geomkind=geomkind
        )
        local_wfs = torch.as_tensor(local_wfs, dtype=torch.float32)
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
        sups = torch.tensor(
            np.stack(
                [data[key][:].astype(np.float32) for key in supkeys],
                axis=1,
            )
        )
        self.real_len = len(sups)
        good_inds = None
        if y_min is not None and "y" in supkeys:
            good_inds = np.flatnonzero(ys > y_min)
            self.real_len = len(good_inds)
        len_ = max(
            self.real_len, (repeat_to_min_length // len(sups) + 1) * len(sups)
        )

        mins = maxs = None
        if standardize:
            mins = local_wfs.min(dim=0).values
            maxs = local_wfs.max(dim=0).values
            print("mins shape", mins.shape)

        super().__init__(
            local_wfs, sups, good_inds=good_inds, minimum=mins, maximum=maxs
        )
        self.len = len_

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        idx = idx % self.real_len
        return super().__getitem__(idx)
