import argparse
import h5py
import numpy as np
from npx import reg, lib, cuts
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RectBivariateSpline
from spike_psvae.vis_utils import MidpointNormalize

fs = 30_000

plt.rc("figure", dpi=300)

ap = argparse.ArgumentParser()

ap.add_argument("input_h5")

args = ap.parse_args()


with h5py.File(args.input_h5, "r") as input_h5:
    maxptp = input_h5["maxptp"][:]

    geom = input_h5["geom"][:]
    loadings_orig = input_h5["loadings_orig"][:]
    loadings_reloc = input_h5["loadings_reloc"][:]
    pcs_orig = input_h5["pcs_orig"][:]
    pcs_reloc = input_h5["pcs_reloc"][:]
    spike_index = input_h5["spike_index"][:]

    x = input_h5["x"][:]
    y = input_h5["y"][:]
    z_rel = input_h5["z_rel"][:]
    z_abs = input_h5["z_abs"][:]
    alpha = input_h5["alpha"][:]
print("data is loaded", flush=True)
times = spike_index[:, 0] / 30000


big = np.flatnonzero(maxptp >= 6)
maxptp = maxptp[big]
loadings_orig = loadings_orig[big]
loadings_reloc = loadings_reloc[big]
spike_index = spike_index[big]
x = x[big]
y = y[big]
z_rel = z_rel[big]
z_abs = z_abs[big]
alpha = alpha[big]
times = times[big]


R_p1_orig, _, _ = lib.faster(loadings_orig[:, 0] + 0, z_abs, times)
R_p2_orig, _, _ = lib.faster(loadings_orig[:, 1] + 0, z_abs, times)
R_p3_orig, _, _ = lib.faster(loadings_orig[:, 2] + 0, z_abs, times)

R_p1_reloc, _, _ = lib.faster(loadings_reloc[:, 0] + 0, z_abs, times)
R_p2_reloc, _, _ = lib.faster(loadings_reloc[:, 1] + 0, z_abs, times)
R_p3_reloc, _, _ = lib.faster(loadings_reloc[:, 2] + 0, z_abs, times)

aspect = 0.5 * R_p1_orig.shape[1] / R_p1_orig.shape[0]

fig, ((aa, ab), (ac, ad), (ae, af)) = plt.subplots(
    3, 2, figsize=(8, 8), sharey=True, sharex=True, dpi=300,
)
aa.imshow(
    R_p1_orig,
    cmap=plt.cm.seismic,
    norm=MidpointNormalize(),
    aspect=aspect,
    interpolation="none",
)
ac.imshow(
    R_p2_orig,
    cmap=plt.cm.seismic,
    norm=MidpointNormalize(),
    aspect=aspect,
    interpolation="nearest",
)
ae.imshow(
    R_p3_orig,
    cmap=plt.cm.seismic,
    norm=MidpointNormalize(),
    aspect=aspect,
    interpolation="nearest",
)
ab.imshow(
    R_p1_reloc,
    cmap=plt.cm.seismic,
    norm=MidpointNormalize(),
    aspect=aspect,
    interpolation="nearest",
)
ad.imshow(
    R_p2_reloc,
    cmap=plt.cm.seismic,
    norm=MidpointNormalize(),
    aspect=aspect,
    interpolation="nearest",
)
af.imshow(
    R_p3_reloc,
    cmap=plt.cm.seismic,
    norm=MidpointNormalize(),
    aspect=aspect,
    interpolation="nearest",
)

aa.set_title("PC1 without")
ac.set_title("PC2 without")
ae.set_title("PC3 without")

ab.set_title("PC1 with")
ad.set_title("PC2 with")
af.set_title("PC3 with")

fig.suptitle("PCA loadings with/without relocation")

plt.tight_layout()

plt.savefig("/Users/charlie/Desktop/pcraster.png", dpi=300)
