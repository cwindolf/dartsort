"""
Clean, localize, register.

Operates on the HDF5 output of scripts/subtract.py
"""
import argparse
import h5py
import time
import numpy as np
from spike_psvae import localization, ibme

ap = argparse.ArgumentParser()

ap.add_argument("subtracted_h5")
ap.add_argument("out_npz")
ap.add_argument("--n_jobs", type=int, default=1)
ap.add_argument("--n_channels", type=int, default=20)
ap.add_argument("--min_ptp", type=float, default=None)

args = ap.parse_args()


class timer:
    def __init__(self, name="timer"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.t = time.time() - self.start
        print(self.name, "took", self.t, "s")


# -- clean wfs if nec

with h5py.File(args.subtracted_h5, "r") as f:
    if "cleaned_waveforms" not in f:
        raise ValueError(
            "Input H5 should contain cleaned waveforms from subtraction"
        )


# -- localize

with timer("localization"):
    maxptp, x, y, z_rel, z_abs, alpha = localization.localize_h5(
        args.subtracted_h5,
        wfs_key="cleaned_waveforms",
        n_workers=args.n_jobs,
        min_ptp=args.min_ptp,
    )
    with h5py.File(args.subtracted_h5, "r") as f:
        times = (f["spike_index"][:, 0] - f["start_sample"][()]) / 30000


# -- register

with timer("registration"):
    z_reg, dispmap = ibme.register_nonrigid(
        maxptp,
        z_abs,
        times,
        robust_sigma=1,
        rigid_disp=200,
        disp=100,
        denoise_sigma=0.1,
        destripe=False,
        n_windows=[5, 10],
        n_iter=1,
        widthmul=0.5,
    )
    z_reg -= (z_reg - z_abs).mean()
    dispmap -= dispmap.mean()

with timer("save"):
    np.savez(
        args.out_npz,
        locs=np.c_[x, y, z_rel, z_abs, alpha],
        t=times,
        maxptp=maxptp,
        z_reg=z_reg,
        dispmap=dispmap,
    )
