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
ap.add_argument("--dispmap", default=None, required=False)
ap.add_argument("--zreg", default=None, required=False)
ap.add_argument("--locs", default=None, required=False)

args = ap.parse_args()


with h5py.File(args.input_h5, "r") as input_h5:
    maxptp = input_h5["maxptp"][:]

    geom = input_h5["geom"][:]
    loadings_orig = input_h5["loadings_orig"][:]
    stds_orig = np.std(loadings_orig, axis=0)
    print("stds", stds_orig.shape)
    loadings_orig /= stds_orig / 15
    loadings_reloc = input_h5["loadings_reloc"][:]
    stds_reloc = np.std(loadings_reloc, axis=0)
    print("stds", stds_reloc.shape)
    loadings_reloc /= stds_reloc / 15
    pcs_orig = input_h5["pcs_orig"][:]
    pcs_reloc = input_h5["pcs_reloc"][:]
    spike_index = input_h5["spike_index"][:]

    x = input_h5["x"][:]
    y = input_h5["y"][:]
    # z_rel = input_h5["z_rel"][:]
    z_abs = input_h5["z_abs"][:] if "z_reg" in input_h5 else input_h5["z_abs"][:]
    alpha = input_h5["alpha"][:]
print("data is loaded", flush=True)
times = spike_index[:, 0] / 30000

z_abs -= z_abs.min()
R, dd, tt = lib.faster(maxptp, z_abs.copy(), times)
cuts.plot(R)
plt.show()

if args.zreg:
    z_abs = np.load(args.zreg)

if args.dispmap:
    dispmap = np.load(args.dispmap)
    lerp = RectBivariateSpline(
        np.arange(dispmap.shape[0]),
        np.arange(dispmap.shape[1]),
        dispmap,
        kx=3,
        ky=3,
    )
    z_abs = z_abs - lerp(z_abs, times, grid=False)
    z_abs -= z_abs.min()

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
# print(z_abs.min(), z_abs.max())
# R, dd, tt = lib.faster(maxptp, z_abs, times)


# big = np.flatnonzero(maxptp >= 6)
# maxptp = maxptp[big]
# loadings_orig = loadings_orig[big]
# loadings_reloc = loadings_reloc[big]
# spike_index = spike_index[big]
# x = x[big]
# y = y[big]
# z_rel = z_rel[big]
# z_abs = z_abs[big]
# alpha = alpha[big]
# times = times[big]


# print(times.min(), times.max())
# z_abs -= z_abs.min()
# z_abs = reg.register_nonrigid(
#     maxptp,
#     z_abs,
#     spike_index[:, 0] / fs,
#     robust_sigma=1,
#     batch_size=1,
#     step_size=1,
#     disp=100,
#     n_windows=30,
#     n_iter=1,
#     widthmul=0.25,
# )
# np.save("zlast1.npy", z_abs)
# z_abs = np.load("zlast0.npy")
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
# z_abs -= z_abs.min()
# z_abs = z_abs - lerp(z_abs, times, grid=False)

z_abs -= z_abs.min()
R, dd, tt = lib.faster(maxptp, z_abs.copy(), times)
cuts.plot(R)
plt.show()



# -- process times
times_s = (spike_index[:, 0] // (20 * fs)).astype(int)
plt.hist(times_s, bins=np.arange(times_s.max() + 1)); plt.show()
assert np.all(times_s[:-1] <= times_s[1:])
at_0 = np.flatnonzero(times_s == 0)

big = range(len(maxptp))
if not np.all(maxptp >= 6):
    big = np.flatnonzero(maxptp >= 6)


a0, a1 = np.searchsorted(times_s[big], [30, 30 + 1])
b0, b1 = np.searchsorted(times_s[big], [29, 29 + 1])
plt.hist(z_abs[big][a0:a1], bins=256)
plt.hist(z_abs[big][b0:b1], bins=256)
plt.show()


# -- process colors
ptpmin = maxptp.min()
ptpmax = maxptp.max()
colors = colormap(maxptp, vmin=ptpmin, vmax=ptpmax, cmap="viridis")[big]
print(colors.shape)
geom_colors = np.c_[
    np.ones_like(geom[:, 0]),
    np.zeros_like(geom[:, 0]),
    np.zeros_like(geom[:, 0]),
    np.ones_like(geom[:, 0]),
]

# -- make 3d data for figs
# x, z, pc1
# pos_orig_a = np.c_[x, z_abs, loadings_orig[:, 0]]
pos_orig_a = np.c_[loadings_orig[:, 0], z_abs, x][big]

# x, z, pc2
# pos_orig_b = np.c_[x, z_abs, loadings_orig[:, 1]]
pos_orig_b = np.c_[loadings_orig[:, 1], z_abs, x][big]

# pc1, z, pc2
# pos_orig_c = np.c_[loadings_orig[:, 0], z_abs, loadings_orig[:, 1]]
# x, z, pc1
# pos_reloc_a = np.c_[x, z_abs, loadings_reloc[:, 0]]
pos_reloc_a = np.c_[loadings_reloc[:, 0], z_abs, x][big]

# x, z, pc2
# pos_reloc_b = np.c_[x, z_abs, loadings_reloc[:, 1]]
pos_reloc_b = np.c_[loadings_reloc[:, 1], z_abs, x][big]

# pc1, z, pc2
# pos_reloc_c = np.c_[loadings_reloc[:, 0], z_abs, loadings_reloc[:, 1]]
pos_geom = np.c_[geom[:, 0], geom[:, 1], np.zeros_like(geom[:, 0])]

# -- set up vis
c = canvas(show_fps=True)
s = c.scene(rows=2, cols=2)

panel_orig_a = s.panel(row=0, col=0, controller="arcball")
panel_orig_b = s.panel(row=1, col=0, controller="arcball")
panel_reloc_a = s.panel(row=0, col=1, controller="arcball")
panel_reloc_b = s.panel(row=1, col=1, controller="arcball")
panel_orig_a.link_to(panel_orig_b)
panel_orig_a.link_to(panel_reloc_a)
panel_orig_a.link_to(panel_reloc_b)

# the visuals at time 0
vis_orig_a = panel_orig_a.visual("point")
vis_orig_a.data("pos", pos_orig_a[at_0])
vis_orig_a.data("ms", np.array([2.5] * len(at_0)))
vis_orig_a.data("color", colors[at_0])
# vis_orig_a.data("ec", np.ones_like(colors[at_0]))

vis_orig_b = panel_orig_b.visual("point")
vis_orig_b.data("pos", pos_orig_b[at_0])
vis_orig_b.data("ms", np.array([2.5] * len(at_0)))
vis_orig_b.data("color", colors[at_0])
# vis_orig_b.data("ec", np.ones_like(colors[at_0]))

vis_reloc_a = panel_reloc_a.visual("point")
vis_reloc_a.data("pos", pos_reloc_a[at_0])
vis_reloc_a.data("ms", np.array([2.5] * len(at_0)))
vis_reloc_a.data("color", colors[at_0])
# vis_reloc_a.data("ec", np.ones_like(colors[at_0]))

vis_reloc_b = panel_reloc_b.visual("point")
vis_reloc_b.data("pos", pos_reloc_b[at_0])
vis_reloc_b.data("ms", np.array([2.5] * len(at_0)))
vis_reloc_b.data("color", colors[at_0])
# vis_reloc_b.data("ec", np.ones_like(colors[at_0]))


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
    "slider_int", "t (20s)", vmin=0, vmax=times_s.max(), value=0
)

maxz = np.ceil(z_abs.max())
zz = np.zeros(3).reshape(1, -1)
zzz = zz + np.array([[0, maxz, 0]])
xx = np.array([[0, 0, x.min()]])
xxx = np.array([[0, 0, x.max()]])

minlo0 = loadings_orig[:, 0].min()
r_minlo0 = np.array([[minlo0, 0, 0]])
minlo1 = loadings_orig[:, 1].min()
r_minlo1 = np.array([[minlo1, 0, 0]])
minro0 = loadings_reloc[:, 0].min()
r_minro0 = np.array([[minro0, 0, 0]])
minro1 = loadings_reloc[:, 1].min()
r_minro1 = np.array([[minro1, 0, 0]])
maxlo0 = loadings_orig[:, 0].max()
r_maxlo0 = np.array([[maxlo0, 0, 0]])
maxlo1 = loadings_orig[:, 1].max()
r_maxlo1 = np.array([[maxlo1, 0, 0]])
maxro0 = loadings_reloc[:, 0].max()
r_maxro0 = np.array([[maxro0, 0, 0]])
maxro1 = loadings_reloc[:, 1].max()
r_maxro1 = np.array([[maxro1, 0, 0]])

p_o0 = np.r_[zz, zzz, xx, xxx, r_minlo0, r_maxlo0]
p_o1 = np.r_[zz, zzz, xx, xxx, r_minlo1, r_maxlo1]
p_r0 = np.r_[zz, zzz, xx, xxx, r_minlo0, r_maxro0]
p_r1 = np.r_[zz, zzz, xx, xxx, r_minlo1, r_maxro1]

cc = np.zeros((6, 4))


def change_t0(t0):

    s0, s1 = np.searchsorted(times_s[big], [t0, t0 + 1])
    print(t0, s0, s1)
    vis_orig_a.data("pos", np.r_[p_o0, pos_orig_a[s0:s1]])
    vis_orig_a.data("color", np.r_[cc, colors[s0:s1]])
    vis_orig_b.data("pos", np.r_[p_o1, pos_orig_b[s0:s1]])
    vis_orig_b.data("color", np.r_[cc, colors[s0:s1]])
    vis_reloc_a.data("pos", np.r_[p_r0, pos_reloc_a[s0:s1]])
    vis_reloc_a.data("color", np.r_[cc, colors[s0:s1]])
    vis_reloc_b.data("pos", np.r_[p_r1, pos_reloc_b[s0:s1]])
    vis_reloc_b.data("color", np.r_[cc, colors[s0:s1]])
    # if t0 == 30:
    #     plt.scatter(pos_orig_a[s0:s1][:, 1], pos_orig_a[s0:s1][:, 0], s=1)
    #     _s0, _s1 = np.searchsorted(times_s[big], [t0 - 1, t0])
    #     plt.scatter(pos_orig_a[_s0:_s1][:, 1], pos_orig_a[_s0:_s1][:, 0], s=1)
    #     plt.show()


slider_t0.connect(change_t0)
change_t0(0)
print("hi")

run()
