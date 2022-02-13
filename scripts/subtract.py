import argparse
from spike_psvae import subtract


# -- args


ap = argparse.ArgumentParser()

ap.add_argument("standardized_bin")
ap.add_argument("output_h5")

args = ap.parse_args()


# -- run subtraction

subtract.subtraction(
    args.standardized_bin,
    args.output_h5,
)
