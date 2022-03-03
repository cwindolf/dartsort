"""Make AE features

Operates on the outputs of scripts/subtract.py and
scripts/localize.py.
"""
import argparse
import numpy as np
import h5py
import time
from spike_psvae.linear_ae import LinearRelocAE


ap = argparse.ArgumentParser()

ap.add_argument("subtracted_h5")
ap.add_argument("locs_npz")
ap.add_argument("output_npz")
ap.add_argument("--rank", type=int, default=10)
# ap.add_argument("--n_jobs", type=int, default=1)

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
            firstchans = h5["first_channels"][:]
            maxchans = h5["spike_index"][:, 1]
            geom = h5["geom"][:]
            x, y, z_rel, z_abs, alpha = locs_f["locs"].T

            ae = LinearRelocAE(args.rank, geom)
            ae.fit(waveforms, x, y, z_abs, alpha, firstchans, maxchans)
            features, errors = ae.transform(
                waveforms, x, y, z_abs, alpha, firstchans, maxchans, return_error=True
            )

with timer("save"):
    np.savez(args.output_npz, features=features, errors=errors)
