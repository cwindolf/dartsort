import argparse
import h5py
import numpy as np

from isosplit import isosplit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("feats_h5")
    ap.add_argument("--key", default="isosplit_labels", required=False)
    ap.add_argument("--npcs", default=3, type=int, required=False)
    args = ap.parse_args()

    with h5py.File(args.feats_h5, "r") as f:
        assert args.key not in f

        x = f["x"][:]
        y = f["y"][:]
        z = f["z_reg"][:]
        alpha = f["alpha"][:]
        pcs = f["loadings_reloc"][:, :args.npcs]

    feats = np.c_[x, y, z, alpha, pcs]

    print("Running clustering...")
    labels = isosplit(feats)

    with h5py.File(args.feats_h5, "r+") as f:
        f.create_dataset(args.key, data=labels)
