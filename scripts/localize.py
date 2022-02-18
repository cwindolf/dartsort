"""
Clean, localize, register.
"""
import argparse
import h5py
import time
from spike_psvae import subtract, localization
from npx import reg

ap = argparse.ArgumentParser()

ap.add_argument("subtracted_h5")
ap.add_argument("out_npz")
ap.add_argument("--n_jobs", dtype=int, default=1)

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
        if "cleaned_waveforms" in f:
            print("Cleaned waveforms already in the file, skipping.")
        else:
            subtract.clean_waveforms(f)


# -- localize

with timer("localization"):
    with h5py.File(args.subtracted_h5, "r") as f:
        maxptp = f["cleaned_waveforms"][:].ptp(1).ptp(1)
        times = (f["spike_index"][:, 0] - f["start_sample"][()]) / 30000
        x, y, z_rel, z_abs, alpha = localization.localize_waveforms_batched(
            f["cleaned_waveforms"],
            f["geom"][:],
            f["first_channels"][:],
            f["max_channels"][:],
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
        n_windows=[5, 30, 60],
        n_iter=1,
        widthmul=0.25,
    )

with timer("save"):
    np.savez(
        args.out_npz,
        locs=np.c_[x, y, z_rel, z_abs, alpha]
        times = 
    )
