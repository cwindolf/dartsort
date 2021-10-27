import h5py
import numpy as np
from torch.utils.data import Dataset


class SpikeHDF5Dataset(Dataset):

    def __init__(self, h5_path, x, ys):
        self.h5 = h5py.File(h5_path, "r")
        self.x = self.h5[x]
        self.ys = np.stack(
            [self.h5[y][:] for y in ys],
            axis=1,
        )
        self.len = len(self.ys[0])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.x[idx], self.ys[idx]
