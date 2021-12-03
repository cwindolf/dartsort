import argparse
import h5py
import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.cm as cm


fs = 30_000


ap = argparse.ArgumentParser()

ap.add_argument("input_h5")
ap.add_argument("out")

args = ap.parse_args()


with h5py.File(args.input_h5, "r") as input_h5:
    geom = input_h5["geom"][:]
    loadings_orig = input_h5["loadings_orig"][:]
    loadings_reloc = input_h5["loadings_reloc"][:]
    maxptp = input_h5["maxptp"][:]
    pcs_orig = input_h5["pcs_orig"][:]
    pcs_reloc = input_h5["pcs_reloc"][:]
    spike_index = input_h5["spike_index"][:]

    x = input_h5["x"][:]
    y = input_h5["y"][:]
    z_rel = input_h5["z_rel"][:]
    z_abs = input_h5["z_abs"][:]
    alpha = input_h5["alpha"][:]

times = (spike_index[:, 0] // fs).astype(int)
assert np.all(times[:-1] <= times[1:])

ptpmin = maxptp.min()
ptpmax = maxptp.max()
colors = cm.viridis((maxptp - ptpmin) / (ptpmax - ptpmin))

pcomin = loadings_orig.min(axis=0)
pcrmin = loadings_reloc.min(axis=0)
pcomax = loadings_orig.max(axis=0)
pcrmax = loadings_reloc.max(axis=0)

fig, ((aa, ab), (ac, ad)) = plt.subplots(2, 2, figsize=(6, 6))
scatter_a = aa.scatter([pcomin[0]], [pcomin[1]], c=[colors[0]], s=1)
scatter_b = ab.scatter([pcrmin[0]], [pcrmin[1]], c=[colors[0]], s=1)
scatter_c = ac.scatter([pcomin[0]], [pcomin[1]], c=[colors[0]], s=1)
scatter_d = ad.scatter([pcrmin[0]], [pcrmin[1]], c=[colors[0]], s=1)

aa.set_title("Before relocation")
aa.set_xlabel("pc1")
aa.set_ylabel("pc2")
ac.set_xlabel("pc3")
ac.set_ylabel("pc4")
ab.set_title("After relocation")
ab.set_xlabel("pc1")
ab.set_ylabel("pc2")
ad.set_xlabel("pc3")
ad.set_ylabel("pc4")

aa.set_xlim([pcomin[0] - 3, pcomax[0] + 3])
aa.set_ylim([pcomin[1] - 3, pcomax[1] + 3])
ab.set_xlim([pcrmin[0] - 3, pcrmax[0] + 3])
ab.set_ylim([pcrmin[1] - 3, pcrmax[1] + 3])
ac.set_xlim([pcomin[2] - 3, pcomax[2] + 3])
ac.set_ylim([pcomin[3] - 3, pcomax[3] + 3])
ad.set_xlim([pcrmin[2] - 3, pcrmax[2] + 3])
ad.set_ylim([pcrmin[3] - 3, pcrmax[3] + 3])

plt.tight_layout(pad=1.0)


def run(t):
    where = np.flatnonzero(times == t)

    scatter_a.set_offsets(loadings_orig[where, :2])
    scatter_a.set_color(colors[where])

    scatter_b.set_offsets(loadings_reloc[where, :2])
    scatter_b.set_color(colors[where])

    scatter_c.set_offsets(loadings_orig[where, 2:4])
    scatter_c.set_color(colors[where])

    scatter_d.set_offsets(loadings_reloc[where, 2:4])
    scatter_d.set_color(colors[where])

    return scatter_a, scatter_b, scatter_c, scatter_d


ani = anim.FuncAnimation(
    fig, run, tqdm(np.arange(times.max())), interval=1, blit=True
)
writer = anim.FFMpegWriter(fps=10, bitrate=1000)
ani.save(args.out, writer=writer)
