import argparse
import h5py
import numpy as np
from tqdm.auto import trange


BATCH_SIZE = 1024


ap = argparse.ArgumentParser()

ap.add_argument("input_h5")
ap.add_argument("dataset")
ap.add_argument("--min_key", default="minimum", required=False)
ap.add_argument("--max_key", default="maximum", required=False)

args = ap.parse_args()


with h5py.File(args.input_h5, "r") as h5:
    assert args.min_key not in h5
    assert args.max_key not in h5

    dset = h5[args.dataset]
    N, T, C = dset.shape

    minproj = np.full((T, C), np.inf, dtype=dset.dtype)
    maxproj = np.full((T, C), -np.inf, dtype=dset.dtype)

    for n in trange(0, N, BATCH_SIZE):
        minproj = np.minimum(
            dset[n : n + BATCH_SIZE].min(axis=0),
            minproj,
        )
        maxproj = np.maximum(
            dset[n : n + BATCH_SIZE].max(axis=0),
            maxproj,
        )

with h5py.File(args.input_h5, "r+") as h5:
    h5.create_dataset(args.min_key, data=minproj)
    h5.create_dataset(args.max_key, data=maxproj)
