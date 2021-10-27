import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SpikeHDF5Dataset(Dataset):

    def __init__(self, h5_path, x, ys):
        self.h5 = h5py.File(h5_path, "r")
        self.x = self.h5[x]
        self.ys = torch.tensor(
            np.stack(
                [self.h5[y][:].astype(np.float32) for y in ys],
                axis=1,
            )
        )

        self.len = len(self.ys[0])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float),
            self.ys[idx]
        )
