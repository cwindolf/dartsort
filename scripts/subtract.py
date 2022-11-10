"""Detection, subtraction, denoising and localization script

This script is a CLI for the function `subtraction` in subtract.py,
which runs the subtraction and localization pass.

This will also run registration and add the registered z coordinates
and displacement map to the output HDF5 file from subtraction.

See the documentation of `subtract.subtraction` for lots of detail.
"""
import os
import argparse
import h5py
import numpy as np
from spike_psvae import subtract, ibme

if __name__ == "__main__":
    # -- args
    ap = argparse.ArgumentParser(__doc__)

    g = ap.add_argument_group("Data input/output")
    g.add_argument("standardized_bin")
    g.add_argument("out_folder")
    g.add_argument("--overwrite", action="store_true")

    g = ap.add_argument_group("Pipeline configuration")
    g.add_argument("--geom", default=None, type=str)
    g.add_argument("--noclean", action="store_true")
    g.add_argument("--noresidual", action="store_true")
    g.add_argument("--notpca", action="store_true")

    g = ap.add_argument_group("Subtraction configuration")
    g.add_argument(
        "--thresholds",
        default=[12, 10, 8, 6, 5, 4],
        type=lambda x: list(map(int, x.split(","))),
    )
    g.add_argument("--nndetect", action="store_true")
    g.add_argument("--dndetect", action="store_true")
    g.add_argument(
        "--neighborhood_kind",
        default="firstchan",
        choices=["firstchan", "box", "circle"],
    )
    g.add_argument(
        "--enforce_decrease_kind",
        default="columns",
        choices=["columns", "radial", "none"],
    )
    g.add_argument("--extract_box_radius", default=200, type=int)

    g = ap.add_argument_group(
        "Time range: use the whole dataset, or a subset?"
    )
    g.add_argument("--t_start", type=int, default=0)
    g.add_argument("--t_end", type=int, default=None)

    g = ap.add_argument_group("Temporal PCA")
    g.add_argument("--tpca_rank", type=int, default=8)
    g.add_argument("--n_sec_pca", type=int, default=20)

    g = ap.add_argument_group("Registration")
    g.add_argument("--noregister", action="store_true")
    g.add_argument(
        "--n_windows",
        default=10,
        type=lambda x: list(map(int, x.split(","))),
    )

    g = ap.add_argument_group("Chunking and parallelism")
    g.add_argument("--n_sec_chunk", type=int, default=1)
    g.add_argument("--n_jobs", type=int, default=1)
    g.add_argument("--n_loc_workers", type=int, default=4)
    g.add_argument("--nogpu", action="store_true")

    g = ap.add_argument_group("Localization")
    g.add_argument(
        "--localization_kind",
        default="logbarrier",
        choices=["logbarrier", "original", "none"],
    )
    g.add_argument("--localize_radius", default=100, type=int)

    args = ap.parse_args()

    # -- load geom if we are not reading from .meta

    geom = None
    if args.geom is None:
        pass
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

    sub_h5 = subtract.subtraction(
        args.standardized_bin,
        args.out_folder,
        neighborhood_kind=args.neighborhood_kind,
        extract_box_radius=args.extract_box_radius,
        enforce_decrease_kind=args.enforce_decrease_kind,
        geom=geom,
        thresholds=args.thresholds,
        nn_detect=args.nndetect,
        denoise_detect=args.dndetect,
        n_sec_chunk=args.n_sec_chunk,
        tpca_rank=args.tpca_rank,
        n_jobs=args.n_jobs,
        t_start=args.t_start,
        t_end=args.t_end,
#         do_clean=not args.noclean,
        n_sec_pca=args.n_sec_pca if not args.notpca else None,
        localization_kind=args.localization_kind,
        save_residual=not args.noresidual,
        loc_workers=args.n_loc_workers,
        localize_radius=args.localize_radius,
        overwrite=args.overwrite,
    )

    # -- registration

    if (args.localization_kind != "none") and (not args.noregister):
        with h5py.File(sub_h5, "r+") as h5:
            do_reg = True
            if "z_reg" in h5 and not args.overwrite:
                print(
                    "Resumed run already had registered z. Set --overwrite if you "
                    "want to re-run registration."
                )
                do_reg = False
            elif "z_reg" in h5 and args.overwrite:
                del h5["z_reg"]
                del h5["dispmap"]

            if do_reg:
                samples = h5["spike_index"][:, 0] - h5["start_sample"][()]
                z_abs = h5["localizations"][:, 2]
                maxptps = h5["maxptps"]

                z_reg, dispmap = ibme.register_nonrigid(
                    maxptps,
                    z_abs,
                    samples / 30000,
                    # robust_sigma=1.0,
                    corr_threshold=0.6,
                    disp=200 * args.n_windows,
                    denoise_sigma=0.1,
                    rigid_init=False,
                    n_windows=args.n_windows,
                    widthmul=1.0,
                )
                z_reg -= (z_reg - z_abs).mean()
                dispmap -= dispmap.mean()
                h5.create_dataset("z_reg", data=z_reg)
                h5.create_dataset("dispmap", data=dispmap)
