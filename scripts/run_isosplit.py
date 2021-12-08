import argparse
import h5py
import numpy as np

from isosplit import isosplit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("feats_h5")
    ap.add_argument("--key", default="isosplit_labels", required=False)
    ap.add_argument("--key_orig", default="isosplit_labels_orig", required=False)
    ap.add_argument("--npcs", default=3, type=int, required=False)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--ymin", type=float, default=1e-10)
    args = ap.parse_args()

    with h5py.File(args.feats_h5, "r") as f:
        assert args.overwrite or not (args.key in f or args.key_orig in f)
        y = f["y"][:]
        N = len(y)
        good = range(N)
        if args.ymin > 0:
            good = np.flatnonzero(y)
        y = y[good]

        x = f["x"][:][good]
        z = f["z_reg"][:][good]
        alpha = f["alpha"][:][good]
        pcs_orig = f["loadings_orig"][:, :args.npcs][good]
        pcs_reloc = f["loadings_reloc"][:, :args.npcs][good]

    feats_orig = np.c_[x, y, z, alpha, pcs_orig].T
    feats_reloc = np.c_[x, y, z, alpha, pcs_reloc].T

    print("Running clustering on original ...")
    labels_orig = isosplit(feats_orig, K_init=1000)
    print("Got", labels_orig.max() + 1, "clusters")
    labels_orig_full = np.full(-1, size=N, dtype=np.int32)
    labels_orig_full[good] = labels_orig

    print("Running clustering on relocated ...")
    labels_reloc = isosplit(feats_reloc, K_init=1000)
    print("Got", labels_reloc.max() + 1, "clusters")
    labels_reloc_full = np.full(-1, size=N, dtype=np.int32)
    labels_reloc_full[good] = labels_reloc

    with h5py.File(args.feats_h5, "r+") as f:
        if args.key in f and args.overwrite:
            del f[args.key]
        if args.key_orig in f and args.overwrite:
            del f[args.key_orig]
        f.create_dataset(args.key, data=labels_reloc_full)
        f.create_dataset(args.key_orig, data=labels_orig_full)
