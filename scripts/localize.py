"""
Clean, localize, register.

Operates on the HDF5 output of scripts/subtract.py
"""
import argparse
import h5py
import time
import numpy as np
from tqdm.auto import trange
from spike_psvae import subtract, localization

ap = argparse.ArgumentParser()

ap.add_argument("subtracted_h5")
ap.add_argument("out_npz")
ap.add_argument("--n_jobs", type=int, default=1)
ap.add_argument("--n_channels", type=int, default=20)

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

with h5py.File(args.subtracted_h5, "r+") as f:
    if "cleaned_waveforms" not in f:
        raise ValueError("Input H5 should contain cleaned waveforms from subtraction")

    if "cleaned_first_channels" in f:
        cfirstchans = f["cleaned_first_channels"][:]
        cmaxchans = f["cleaned_max_channels"][:]
    else:
        cfirstchans = f["first_channels"][:]
        cmaxchans = f["spike_index"][:, 1]
    crelmcs = cfirstchans - cmaxchans


# -- localize

with timer("localization"):
    with h5py.File(args.subtracted_h5, "r", libver="latest") as f:
        N = len(f["spike_index"])
        maxptp = []
        for bs in trange(0, N, 4096, desc="Grabbing max PTP"):
            be = min(bs + 4096, N)
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
            cfirstchans,
            cmaxchans,
            n_workers=args.n_jobs,
            n_channels=args.n_channels,
        )


# -- register

try:
    from npx import reg
    with timer("registration"):
        z_reg, dispmap = reg.register_nonrigid(
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
except ImportError:
    print("Sorry I need to install the registration in this repo will do that soon")
