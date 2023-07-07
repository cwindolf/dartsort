import argparse
import shutil
from pathlib import Path

import numpy as np
# from brainbox.io import spikeglx
import spikeglx
from neurodsp import utils, voltage
from tqdm.auto import trange

ap = argparse.ArgumentParser()

ap.add_argument("input_binary")
ap.add_argument("--output-binary", type=str, default=None)
ap.add_argument("--no-bad-channels", action="store_true")
ap.add_argument("--output-dtype", default="float32")
ap.add_argument("--robust", default="float32")
ap.add_argument("--n-jobs", type=int, default=1)

args = ap.parse_args()


binary = Path(args.input_binary)
folder = binary.parent
output_dtype = np.dtype(args.output_dtype).type

standardized_file = folder / f"destriped_{binary.name}"
standardized_file = standardized_file.with_suffix(".bin")
if args.output_binary is not None:
    standardized_file = Path(args.output_binary)
    assert standardized_file.parent.exists()

# run destriping
sr = spikeglx.Reader(binary)
h = sr.geometry
if not standardized_file.exists():
    batch_size_secs = 1
    batch_intervals_secs = 50
    # scans the file at constant interval, with a demi batch starting offset
    nbatches = int(
        np.floor((sr.rl - batch_size_secs) / batch_intervals_secs - 0.5)
    )
    wrots = np.zeros((nbatches, sr.nc - sr.nsync, sr.nc - sr.nsync))
    for ibatch in trange(nbatches, desc="destripe batches"):
        ifirst = int(
            (ibatch + 0.5) * batch_intervals_secs * sr.fs
            + batch_intervals_secs
        )
        ilast = ifirst + int(batch_size_secs * sr.fs)
        sample = voltage.destripe(
            sr[ifirst:ilast, : -sr.nsync].T, fs=sr.fs, neuropixel_version=1
        )
        np.fill_diagonal(
            wrots[ibatch, :, :],
            1 / utils.rms(sample) * sr.sample2volts[: -sr.nsync],
        )

    wrot = np.median(wrots, axis=0)
    voltage.decompress_destripe_cbin(
        sr.file_bin,
        h=h,
        wrot=wrot,
        output_file=standardized_file,
        dtype=output_dtype,
        nc_out=sr.nc - sr.nsync,
        reject_channels=not args.no_bad_channels,
        nprocesses=args.n_jobs,
    )

    # also copy the companion meta-data file
    shutil.copy(
        sr.file_meta_data,
        standardized_file.parent / f"{standardized_file.stem}.meta",
    )
