import argparse
import h5py
import numpy as np
from pathlib import Path

from spike_psvae import waveform_utils


ap = argparse.ArgumentParser()

ap.add_argument("sub_h5")
ap.add_argument("locs_npz")
ap.add_argument("feats_npz")
ap.add_argument("out_folder")
ap.add_argument("--nowfs", action="store_true")

args = ap.parse_args()

out_folder = Path(args.out_folder)
out_folder.mkdir(exist_ok=True)


sub = h5py.File(args.sub_h5, "r")
locs = np.load(args.locs_npz)
feats = np.load(args.feats_npz)
ls = locs["locs"]

firstchans = sub["first_channels"][:]
if not args.nowfs:
    rwfs, firstchans, _ = waveform_utils.relativize_waveforms_np1(
        sub["cleaned_waveforms"][:], firstchans, sub["geom"][:], sub["spike_index"][:, 1], feat_chans=20
    )
    np.save(out_folder / "wfs.npy", rwfs)
    np.save(out_folder / "ptps.npy", rwfs.ptp(1))

np.save(out_folder / "ae_features.npy", feats["features"])
np.save(
    out_folder / "localization_results.npy",
    np.c_[ls[:, 0], ls[:, 3], ls[:, 1], ls[:, 4], locs["maxptp"], firstchans, sub["spike_index"][:, 1]],
)
np.save(out_folder / "z_reg.npy", locs["z_reg"])
np.save(out_folder / "np1_channel_map.npy", sub["geom"][:])
np.save(out_folder / "reconstruction_errors.npy", feats["errors"])
np.save(out_folder / "spike_index.npy", sub["spike_index"][:])

