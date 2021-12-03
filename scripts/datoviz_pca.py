import argparse
import h5py
import numpy as np
from npx import reg, lib, cuts
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RectBivariateSpline

from datoviz import canvas, run, colormap


fs = 30_000


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

# try:
#     z_abs = np.load("z_abs_reg_nr.npy")
# except:
# z_abs = reg.register_nonrigid(
#     maxptp,
#     z_abs,
#     spike_index[:, 0] / fs,
#     robust_sigma=1,
#     batch_size=1,
#     step_size=1,
#     disp=100,
#     n_windows=20,
# )
# np.save("z_abs_reg_nr20.npy", z_abs)
# p = np.load("/Users/charlie/Downloads/cortexlab_adc_disp_estimate.npy")
times = spike_index[:, 0] / 30000
print(z_abs.min(), z_abs.max())
R, dd, tt = lib.faster(maxptp, z_abs, times)


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


print(times.min(), times.max())
# z_abs = reg.register_nonrigid(
#     maxptp,
#     z_abs,
#     spike_index[:, 0] / fs,
#     robust_sigma=1,
#     batch_size=1,
#     step_size=1,
#     disp=100,
#     n_windows=20,
#     n_iter=3,
# )
# p = np.save("z_abs_reg_nr20.npy", z_abs)
# p = np.load()
# z_abs = z_abs - interp1d(np.arange(len(p)), p, kind="nearest", fill_value="extrapolate")(times)
# tshift = np.load("data/np2_pcbme_20.npy")
# print(tshift.shape, dd.shape, tt.shape)
# lerp = RectBivariateSpline(
#     np.arange(tshift.shape[0]),
#     np.arange(tshift.shape[1]),
#     tshift,
#     kx=3,
#     ky=3,
# )
# # z_abs -= z_abs.min()
# z_abs = z_abs - lerp(z_abs, times, grid=False)

# z_abs -= z_abs.min()
# R, dd, tt = lib.faster(maxptp, z_abs.copy(), times)
# cuts.plot(R)
# plt.show()

# -- process times
times_s = (spike_index[:, 0] // (10 * fs)).astype(int)
assert np.all(times_s[:-1] <= times_s[1:])
at_0 = np.flatnonzero(times_s == 0)

# -- process colors
ptpmin = maxptp.min()
ptpmax = maxptp.max()
colors = colormap(maxptp, vmin=ptpmin, vmax=ptpmax, cmap="viridis")
print(colors.shape)
geom_colors = np.c_[
    np.ones_like(geom[:, 0]),
    np.zeros_like(geom[:, 0]),
    np.zeros_like(geom[:, 0]),
    np.ones_like(geom[:, 0]),
]

# -- make 3d data for figs
# x, z, pc1
pos_orig = loadings_orig[:, :3]
# x, z, pc1
pos_reloc = loadings_reloc[:, :3]
# pc1, z, pc2
# pos_reloc_c = np.c_[loadings_reloc[:, 0], z_abs, loadings_reloc[:, 1]]
# pos_geom = np.c_[geom[:, 0], geom[:, 1], np.zeros_like(geom[:, 0])]
# print(pos_reloc_c.shape)

# -- set up vis
c = canvas(show_fps=True)
s = c.scene(rows=1, cols=2)

panel_orig_a = s.panel(row=0, col=0, controller="arcball")
# panel_orig_b = s.panel(row=1, col=0, controller="arcball")
panel_reloc_a = s.panel(row=0, col=1, controller="arcball")
# panel_reloc_b = s.panel(row=1, col=1, controller="arcball")
# panel_orig_a.link_to(panel_orig_b)
panel_orig_a.link_to(panel_reloc_a)
# panel_orig_a.link_to(panel_reloc_b)

# the visuals at time 0
vis_orig_a = panel_orig_a.visual("point")
vis_orig_a.data("pos", pos_orig[at_0])
vis_orig_a.data("color", colors[at_0])

# vis_orig_b = panel_orig_b.visual("point")
# vis_orig_b.data("pos", pos_orig_b[at_0])
# vis_orig_b.data("color", colors[at_0])

vis_reloc_a = panel_reloc_a.visual("point")
vis_reloc_a.data("pos", pos_reloc[at_0])
vis_reloc_a.data("color", colors[at_0])

# vis_reloc_b = panel_reloc_b.visual("point")
# vis_reloc_b.data("pos", pos_reloc_b[at_0])
# vis_reloc_b.data("color", colors[at_0])


# geom visuals
# for panel in [panel_orig_a, panel_reloc_a]:  #, panel_orig_b, panel_reloc_b]:
#     gvis = panel.visual("point")
#     print(pos_geom.shape)
#     gvis.data("pos", pos_geom)
#     gvis.data("color", geom_colors)

# title_a = panel_a.visual("text")
# ta = "PCA before"
# title_a.data("glyph", np.array([ord(c) - 32 for c in ta], dtype=np.uint16))
# title_a.data("length", np.array([len(ta)]))
# pa = (
#  loadings_orig[:1000, :3].max(axis=0) - loadings_orig[:1000, :3].min(axis=0)
# ) / 2 + loadings_orig[:1000, :3].min(axis=0)
# title_a.data(
#     "pos", np.array([pa[0], loadings_orig[:1000, :3].max(axis=0)[1], 0])
# )

# slider
gui = c.gui("hi")
slider_t0 = gui.control(
    "slider_int", "t (10s)", vmin=0, vmax=times_s.max(), value=0
)


def change_t0(t0):
    t0, t1 = np.searchsorted(times_s, [t0, t0 + 1])
    vis_orig_a.data("pos", pos_orig[t0:t1])
    vis_orig_a.data("color", colors[t0:t1])
    # vis_orig_b.data("pos", pos_orig_b[t0:t1])
    # vis_orig_b.data("color", colors[t0:t1])
    vis_reloc_a.data("pos", pos_reloc[t0:t1])
    vis_reloc_a.data("color", colors[t0:t1])
    # vis_reloc_b.data("pos", pos_reloc_b[t0:t1])
    # vis_reloc_b.data("color", colors[t0:t1])


slider_t0.connect(change_t0)
change_t0(0)
print("hi")

run()
