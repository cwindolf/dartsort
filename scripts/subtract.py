import os
import argparse
import numpy as np
from spike_psvae import subtract


# -- args


ap = argparse.ArgumentParser()

ap.add_argument("standardized_bin")
ap.add_argument("out_folder")
ap.add_argument(
    "--thresholds",
    default=[12, 10, 8, 6, 5, 4],
    type=lambda x: list(map(int, x.split(","))),
)
ap.add_argument("--geom", default=None, type="str")
ap.add_argument("--tpca_rank", type=int, default=8)
ap.add_argument("--n_sec_chunk", type=int, default=1)
ap.add_argument("--t_start", type=int, default=0)
ap.add_argument("--t_end", type=int, default=None)
ap.add_argument("--n_jobs", type=int, default=1)
ap.add_argument("--nogpu", action="store_true")
ap.add_argument("--noclean", action="store_true")
ap.add_argument("--n_sec_pca", type=int, default=20)

args = ap.parse_args()


# -- load geom if we are not reading from .meta


geom = None
if args.geom is None:
    print(
        "Will try to load geometry from a .meta file in "
        "the same directory as the binary."
    )
elif args.geom in ["np1", "np2"]:
    from ibllib.ephys import neuropixel
    print(
        f"Using the typical {args.geom} geometry. "
        "Will try to load from .meta if --geom is not set"
    )
    ch = neuropixel.dense_layout(version=int(args.geom[-1]))
    geom = np.c_[ch["x"], ch["y"]]
elif os.path.isfile(args.geom) and args.geom.endswith(".npy"):
    print("Will load geometry array from", args.geom)
    geom = np.load(args.geom)
    assert geom.ndim == 2 and geom.shape[1] == 2
else:
    raise ValueError("Not sure what to do with --geom", args.geom)


# -- run subtraction


if args.nogpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

subtract.subtraction(
    args.standardized_bin,
    args.out_folder,
    geom=geom,
    thresholds=args.thresholds,
    n_sec_chunk=args.n_sec_chunk,
    tpca_rank=args.tpca_rank,
    n_jobs=args.n_jobs,
    t_start=args.t_start,
    t_end=args.t_end,
    do_clean=not args.noclean,
    n_sec_pca=args.n_sec_pca,
)
