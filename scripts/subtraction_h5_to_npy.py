import argparse
import h5py
import numpy as np
from pathlib import Path

from spike_psvae import waveform_utils


ap = argparse.ArgumentParser()

ap.add_argument("sub_h5")
ap.add_argument("out_folder")
ap.add_argument("--wfs", action="store_true")

args = ap.parse_args()

out_folder = Path(args.out_folder)
out_folder.mkdir(exist_ok=True)


sub = h5py.File(args.sub_h5, "r")

firstchans = sub["first_channels"][:]
if args.wfs:
    rwfs, firstchans, _ = waveform_utils.relativize_waveforms_np1(
        sub["cleaned_waveforms"][:],
        firstchans,
        sub["geom"][:],
        sub["spike_index"][:, 1],
        feat_chans=20,
    )
    np.save(out_folder / "wfs.npy", rwfs)
    np.save(out_folder / "ptps.npy", rwfs.ptp(1))

locs = sub["localizations"][:]
np.save(
    out_folder / "localization_results.npy",
    np.c_[
        locs[:, 0],
        locs[:, 1],
        locs[:, 2],
        locs[:, 3],
        sub["maxptps"][:],
        firstchans,
        sub["spike_index"][:, 1],
    ],
)
np.save(out_folder / "z_reg.npy", sub["z_reg"][:])
np.save(out_folder / "np1_channel_map.npy", sub["geom"][:])
np.save(out_folder / "spike_index.npy", sub["spike_index"][:])

for f in out_folder.glob("*.npy"):
    print(f, np.load(out_folder / f).shape)