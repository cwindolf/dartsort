import os
import argparse
import numpy as np
from spike_psvae import subtract


# -- args


ap = argparse.ArgumentParser()

ap.add_argument("standardized_bin")
ap.add_argument("output_h5")
ap.add_argument("--geom", default=None, choices=["np1", "np2"])
ap.add_argument("--tpca_rank", type=int, default=7)
ap.add_argument("--clean", action="store_true")
ap.add_argument("--n_sec_chunk", type=int, default=1)
ap.add_argument("--t_start", type=int, default=0)
ap.add_argument("--t_end", type=int, default=None)
ap.add_argument("--n_jobs", type=int, default=1)
ap.add_argument("--nogpu", action="store_true")

args = ap.parse_args()


# -- load geom if we are not reading from .meta


geom = None
if args.geom is not None:
    from ibllib.ephys import neuropixel
    print(
        f"Using the typical {args.geom} geometry. "
        "Will try to load from .meta if --geom is not set"
    )
    ch = neuropixel.dense_layout(version=int(args.geom[-1]))
    geom = np.c_[ch["x"], ch["y"]]


# -- run subtraction


if args.nogpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

subtract.subtraction(
    args.standardized_bin,
    args.output_h5,
    geom=geom,
    n_sec_chunk=args.n_sec_chunk,
    tpca_rank=args.tpca_rank,
    n_jobs=args.n_jobs,
    t_start=args.t_start,
    t_end=args.t_end,
    do_clean=args.clean,
)
