"""
[1]: https://github.com/datoviz/datoviz/blob/6957dfc460f46c8f517e657e4fa0243d48f4804a/bindings/cython/tests/test_default.py#L171  # noqa
"""
import argparse
import h5py
import numpy as np
from scipy.stats import zscore

import datoviz


class MarkerVis:
    def __init__(
        self,
        panel,
        times,
        xs,
        ys,
        zs,
        colors,
        title,
        dt=10.0,
        sz=5,
        pad=5,
        dark=False,
    ):
        self.xypad = np.array(
            [
                [xs.min() - pad, ys.min() - 10 * pad, zs.min() - 10],
                [xs.max() + pad, ys.max() + 10 * pad, zs.min() - 10],
            ]
        )
        self.cpad = np.zeros((2, 4))
        self.pos = np.c_[xs, ys, zs]
        self.colors = colors
        self.times = times
        self.dt = dt
        self.t = 0

        lo, hi = np.searchsorted(times, [0, dt])
        self.vis = panel.visual("point")
        self.vis.data("pos", np.r_[self.xypad, self.pos[lo:hi]])
        self.vis.data("color", np.r_[self.cpad, colors[lo:hi]])
        self.vis.data("ms", np.array([sz] * (2 + hi - lo)))

        text = panel.visual("text")
        # If you don't like it, imagine how i feel. See link [1] in __doc__
        glyph = np.array([ord(i) - 32 for i in title], dtype=np.uint16)
        text.data("glyph", glyph)
        text.data("length", np.array([len(title)], dtype=np.uint32))
        textcolor = (
            np.array([[255, 255, 255, 255]], dtype=np.uint8)
            if dark
            else np.array([[0, 0, 0, 255]], dtype=np.uint8)
        )
        text.data("color", textcolor)
        text.data(
            "pos",
            np.array(
                [[(xs.min() + xs.max()) / 2, ys.max() - 5 * pad, zs.max() + 1]]
            ),
        )

    def change_t(self, t=None):
        if t is None:
            t = self.t
        lo, hi = np.searchsorted(times, [t, t + self.dt])
        print(lo, hi, hi - lo)
        self.vis.data("pos", np.r_[self.xypad, self.pos[lo:hi]])
        self.vis.data("color", np.r_[self.cpad, self.colors[lo:hi]])


if __name__ == "__main__":
    # args
    ap = argparse.ArgumentParser()
    ap.add_argument("input_h5")
    ap.add_argument("which", choices=["orig", "yza", "xyza"])
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--spikelabels", action="store_true")
    ap.add_argument("--controller", default="axes")
    ap.add_argument("--nopca", action="store_true")
    ap.add_argument("--threshold", type=float, default=6.0)

    args = ap.parse_args()

    # load big spikes
    with h5py.File(args.input_h5, "r") as f:
        maxptp = f["maxptp"][:]
        big = np.flatnonzero(maxptp >= args.threshold)
        if "good_mask" in f:
            big = np.intersect1d(big, np.flatnonzero(f["good_mask"][:]))
        maxptp = maxptp[big]
        show_pca = not args.nopca and "loadings_orig" in f

        z = f["z_reg"][:][big] if "z_reg" in f else f["z_abs"][:][big]
        times = (
            f["times"][:][big]
            if "times" in f
            else f["spike_index"][:, 0][big] / 30000
        )

        if show_pca:
            loadings = f[f"loadings_{args.which}"][:][big]
            loadings /= np.std(loadings, axis=0) / 16
            data = dict(
                x=f["x"][:][big],
                y=f["y"][:][big],
                alpha=f["alpha"][:][big],
                pc1=loadings[:, 0],
                pc2=loadings[:, 1],
                pc3=loadings[:, 2],
            )
        else:
            data = dict(
                x=f["x"][:][big],
                y=f["y"][:][big],
                alpha=f["alpha"][:][big],
            )

        if args.spikelabels:
            print("spike labels")
            labels = f["spike_train"][:, 1][big]
        elif args.labels:
            labels = f[f"labels_{args.which}"][:][big]

    # set up vis
    canvas = datoviz.canvas(show_fps=False)
    scene = canvas.scene(rows=1, cols=len(data))

    # remove outliers for vis
    mask = np.ones(len(z), dtype=bool)
    for k, v in data.items():
        if k == "alpha":
            continue
        mask &= np.abs(zscore(v)) <= 5
    mask = np.flatnonzero(mask)
    for k, v in data.items():
        data[k] = v[mask]
    z = z[mask]
    maxptp = maxptp[mask]
    times = times[mask]
    if args.labels or args.spikelabels:
        labels = labels[mask]

    # sort if nec
    if not (times[:-1] <= times[1:]).all():
        print("Sorting")
        order = np.argsort(times)
        times = times[order]
        for k in data.keys():
            data[k] = data[k][order]
        labels = labels[order]
        maxptp = maxptp[order]
        z = z[order]

    # process colors
    if args.labels or args.spikelabels:
        labels = labels.astype(float)
        print("Unique labels", np.unique(labels).shape)
        labels /= labels.max()
        colors = datoviz.colormap(
            labels, vmin=0.0, vmax=1.0, cmap="glasbey_hv"
        )
    else:
        ptpmin = maxptp.min()
        ptpmax = maxptp.max()
        colors = datoviz.colormap(
            maxptp, vmin=ptpmin, vmax=ptpmax, cmap="viridis"
        )
    colors[:, 3] = 127

    # run all the vis
    prevpanel = None
    viss = []
    for c, (k, v) in enumerate(data.items()):
        panel = scene.panel(0, c, controller=args.controller)

        viss.append(
            MarkerVis(
                panel,
                times,
                v,
                z,
                maxptp,
                colors,
                k,
                dark=args.controller == "panzoom",
            )
        )

        if prevpanel is not None:
            prevpanel.link_to(panel)
        prevpanel = panel

    # GUI and callbacks
    gui = canvas.gui(f"hi -- {args.which}")
    slider_t0 = gui.control(
        "slider_int",
        "t (10s)",
        vmin=0,
        vmax=int(np.ceil(times.max())) // 10,
        value=0,
    )

    def change_t(t):
        for vis in viss:
            vis.change_t(t)

    slider_t0.connect(change_t)
    change_t(0)

    # alright...
    datoviz.run()
