import argparse
import h5py
import numpy as np


ap = argparse.ArgumentParser()

ap.add_argument("sub_h5")
ap.add_argument("locs_npz")
ap.add_argument("feats_npz")
ap.add_argument("out_h5")

args = ap.parse_args()


with h5py.File(args.out_h5, "w") as h5:
    with h5py.File(args.sub_h5, "r") as sub:
        h5.create_dataset("geom", data=sub["geom"][:])
    with np.load(args.locs_npz) as f:
        h5.create_dataset("x", data=f["locs"][:, 0])
        h5.create_dataset("y", data=f["locs"][:, 1])
        h5.create_dataset("z_reg", data=f["z_reg"])
        h5.create_dataset("alpha", data=f["locs"][:, 4])
        h5.create_dataset("times", data=f["t"])
        h5.create_dataset("maxptp", data=f["maxptp"])        
    with np.load(args.feats_npz) as f:
        h5.create_dataset("loadings_orig", data=f["features"])
