"""Make AE features

Operates on the outputs of scripts/subtract.py and
scripts/localize.py.
"""
import argparse
import numpy as np
import h5py
import time
from spike_psvae import featurize


ap = argparse.ArgumentParser()

ap.add_argument("subtracted_h5")
ap.add_argument("locs_npz")
ap.add_argument("output_npz")
ap.add_argument("--rank", type=int, default=10)
ap.add_argument("--n_jobs", type=int, default=1)

args = ap.parse_args()


# -- util


class timer:
    def __init__(self, name="timer"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.t = time.time() - self.start
        print(self.name, "took", self.t, "s")


# -- featurize


with timer("featurize"):
    with h5py.File(args.subtracted_h5, "r") as h5:
        with np.load(args.locs_npz) as locs_f:
            waveforms = h5["cleaned_waveforms"]
            maxchans = h5["cleaned_max_channels"][:]
            firstchans = h5["cleaned_first_channels"][:]
            geom = h5["geom"][:]
            x, y, z_rel, z_abs, alpha = locs_f["locs"].T

            features, errors = featurize.relocated_ae_batched(
                waveforms,
                firstchans,
                maxchans,
                geom,
                x,
                y,
                z_rel,
                alpha,
                rank=args.rank,
                n_jobs=args.n_jobs,
            )

with timer("save"):
    np.savez(args.output_npz, features=features, errors=errors)
