"""
Clean, localize, register.

Operates on the HDF5 output of scripts/subtract.py
"""
import argparse
import h5py
import time
import numpy as np
from spike_psvae import subtract, localization
from npx import reg

ap = argparse.ArgumentParser()

ap.add_argument("subtracted_h5")
ap.add_argument("out_npz")
ap.add_argument("--n_jobs", type=int, default=1)
ap.add_argument("--n_channels", type=int, default=20)
ap.add_argument("--overwrite", action="store_true")

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

with timer("cleaning"):
    with h5py.File(args.subtracted_h5, "r+") as f:
        if args.overwrite:
            for k in (
                "cleaned_waveforms",
                "cleaned_max_channels",
                "cleaned_first_channels",
            ):
                if k in f:
                    del f[k]
        doclean = True
        if "cleaned_waveforms" in f:
            print("Cleaned waveforms already in the file, skipping.")
            doclean = False

    if doclean:
        subtract.clean_waveforms(args.subtracted_h5, n_workers=args.n_jobs, num_channels=args.n_channels)


# -- localize

with timer("localization"):
    with h5py.File(args.subtracted_h5, "r") as f:
        crelmcs = f["cleaned_max_channels"][:] - f["cleaned_first_channels"][:]
        N = len(f["spike_index"])
        maxptp = []
        for bs in range(0, N, 1024):
            be = min(bs + 1024, N)
            maxptp.append(
                f["cleaned_waveforms"][bs:be][
                    np.arange(be - bs), :, crelmcs[bs:be]
                ].ptp(1)
            )
        maxptp = np.concatenate(maxptp).astype(float)
        times = (f["spike_index"][:, 0] - f["start_sample"][()]) / 30000
        x, y, z_rel, z_abs, alpha = localization.localize_waveforms_batched(
            f["cleaned_waveforms"],
            f["geom"][:],
            f["cleaned_first_channels"][:],
            f["cleaned_max_channels"][:],
            n_workers=args.n_jobs,
        )


# -- register

with timer("registration"):
    z_rigid_reg, p_rigid = reg.register_rigid(
        maxptp,
        z_abs,
        times,
        robust_sigma=0,
        disp=400,
        denoise_sigma=0.1,
        destripe=False,
    )
    z_reg, dispmap = reg.register_nonrigid(
        maxptp,
        z_rigid_reg,
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
    dispmap = dispmap - p_rigid[None, :]
    dispmap -= dispmap.mean()

with timer("save"):
    np.savez(
        args.out_npz,
        locs=np.c_[x, y, z_rel, z_abs, alpha],
        t=times,
        maxptp=maxptp,
        z_rigid_reg=z_rigid_reg,
        z_reg=z_reg,
        dispmap=dispmap,
    )
