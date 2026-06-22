"""Toolkit for extensible single unit summary plots

The goal is to make it easy to add and remove plots without the plotting
code turning into a web of if statements, for loops, and bizarre subplot
and subfigure mazes.

Relies on the DARTsortAnalysis object of utils/analysis.py to do most of
the data work so that this file can focus on plotting (sort of MVC).
"""

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from KDEpy import FFTKDE
from matplotlib.legend_handler import HandlerTuple
from tqdm.auto import tqdm

from ..evaluate.analysis import DARTsortAnalysis, WaveformsBag
from ..util.job_util import get_global_computation_config
from ..util.multiprocessing_util import CloudpicklePoolExecutor, cloudpickle, get_pool
from . import layout
from .analysis_plots import bimod_stats, centered_bins, isi_hist, plot_correlogram
from .colors import glasbey1024
from .waveforms import geomplot, geomplot_templates

# -- main class. see fn make_unit_summary below to make lots of UnitPlots.


class UnitPlot(layout.BasePlot):
    can_sharey = True

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id: int):
        raise NotImplementedError


class UnitMultiPlot(layout.BaseMultiPlot):
    def plots(self, sorting_analysis: DARTsortAnalysis, unit_id: int):
        # return [UnitPlot()]
        raise NotImplementedError


class UnitTextInfo(UnitPlot):
    kind = "text"
    height = 0.5

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id: int):
        axis = panel.subplots()
        axis.axis("off")
        msg = f"unit {unit_id}\n"

        h5_path = sorting_analysis.sorting.parent_h5_path
        if h5_path:
            msg += f"from: {h5_path.name}\n"

        nspikes = cast(np.ndarray, sorting_analysis.sorting.labels == unit_id).sum()
        msg += f"count: {nspikes}\n"

        assert sorting_analysis.template_data is not None
        temps = sorting_analysis.template_data.unit_templates(unit_id)
        if not temps.size:
            msg += "no template\n(too few spikes)"
        elif temps.shape[0] == 1:
            ptp = np.ptp(temps, 1).max(1)[0]
            msg += f"ptp: {ptp:0.2f} su\n"
            snr = ptp * np.sqrt(nspikes)
            msg += f"snr: {snr:.1f}"
        else:
            assert False

        axis.text(0, 0, msg, fontsize=6.5)


# -- small summary plots


class ACG(UnitPlot):
    kind = "histogram"
    height = 0.75

    def __init__(self, max_lag=50, bin=1, unit="samples"):
        super().__init__()
        self.max_lag = max_lag
        self.bin = bin
        self.unit = unit
        if unit == "ms":
            self.width = 2

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id: int):
        axis = panel.subplots()
        which = sorting_analysis.in_unit(unit_id)
        t = sorting_analysis.sorting.times_samples[which]
        samples_per_ms = sorting_analysis.sorting.sampling_frequency / 1000
        if self.unit == "samples":
            max_lag_samples = self.max_lag
        elif self.unit == "ms":
            max_lag_samples = int(np.ceil(self.max_lag * samples_per_ms))
        else:
            assert False
        plot_correlogram(
            axis,
            t,
            max_lag=max_lag_samples,
            bin=self.bin,
            samples_per_ms=samples_per_ms,
            to_ms=self.unit == "ms",
        )
        axis.grid(which="both")
        axis.set_ylabel("acg")


class ISIHistogram(UnitPlot):
    kind = "histogram"
    height = 0.75

    def __init__(self, bin_ms=0.1, max_ms=5):
        super().__init__()
        self.bin_ms = bin_ms
        self.max_ms = max_ms

    def draw(
        self,
        panel,
        sorting_analysis: DARTsortAnalysis,
        unit_id,
        axis=None,
        color="k",
        label=None,
    ):
        if axis is None:
            axis = panel.subplots()
        which = sorting_analysis.in_unit(unit_id)
        assert sorting_analysis.times_seconds is not None
        times_s = sorting_analysis.times_seconds[which]
        isi_hist(
            times_s,
            axis,
            bin_ms=self.bin_ms,
            max_ms=self.max_ms,
            color=color,
            label=label,
        )
        axis.grid(which="both")


class XZScatter(UnitPlot):
    kind = "scatter"

    def __init__(
        self,
        registered=True,
        amplitude_color_cutoff=15,
        probe_margin_um=100,
        colorbar=False,
    ):
        super().__init__()
        self.registered = registered
        self.amplitude_color_cutoff = amplitude_color_cutoff
        self.probe_margin_um = probe_margin_um
        self.colorbar = colorbar

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id, axis=None):
        if axis is None:
            axis = panel.subplots()

        in_unit = sorting_analysis.in_unit(unit_id, at_most=50_000)
        assert sorting_analysis.x is not None
        assert sorting_analysis.amplitudes is not None
        x = sorting_analysis.x[in_unit]
        if self.registered:
            assert sorting_analysis.registered_z is not None
            z = sorting_analysis.registered_z[in_unit]
        else:
            assert sorting_analysis.z is not None
            z = sorting_analysis.z[in_unit]
        geomx, geomz = sorting_analysis.geom.T
        pad = self.probe_margin_um
        valid = x == np.clip(x, geomx.min() - pad, geomx.max() + pad)
        valid &= z == np.clip(z, geomz.min() - pad, geomz.max() + pad)
        amps = sorting_analysis.amplitudes[in_unit][valid]
        c = dict(c=np.minimum(amps, self.amplitude_color_cutoff))
        s = axis.scatter(
            x[valid],
            z[valid],
            lw=0,
            s=3,
            **c,
            rasterized=True,
        )
        axis.set_xlabel("x (um)")
        reg_str = "reg " * self.registered
        axis.set_ylabel(reg_str + "z (um)")
        if self.colorbar:
            plt.colorbar(s, ax=axis, shrink=0.5, label="amp (su)")


class PCAScatter(UnitPlot):
    kind = "scatter"

    def __init__(
        self,
        amplitude_color_cutoff=15,
        pca_radius_um=75.0,
        colorbar=False,
    ):
        super().__init__()
        self.amplitude_color_cutoff = amplitude_color_cutoff
        self.colorbar = colorbar
        self.pca_radius_um = pca_radius_um

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id, axis=None):
        if axis is None:
            axis = panel.subplots()

        which, loadings = sorting_analysis.unit_pca_features(unit_id=unit_id)
        if which is None:
            return
        assert loadings is not None
        assert sorting_analysis.amplitudes is not None
        amps = sorting_analysis.amplitudes[which]
        c = dict(c=np.minimum(amps, self.amplitude_color_cutoff))
        s = axis.scatter(
            *loadings.T,
            lw=0,
            s=3,
            rasterized=True,
            **c,
        )
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        if self.colorbar:
            plt.colorbar(s, ax=axis, shrink=0.5, label="amp (su)")

        return axis


# -- wide scatter plots


class TimeFeatScatter(UnitPlot):
    kind = "ctimefeat"
    width = 2
    height = 0.75

    def __init__(
        self,
        feat_name,
        color_by_template_if_possible=False,
        color_by_amplitude=True,
        amplitude_color_cutoff=15,
        alpha=1.0,
        label=None,
        cbar=True,
    ):
        super().__init__()
        self.feat_name = feat_name
        self.amplitude_color_cutoff = amplitude_color_cutoff
        self.color_by_amplitude = color_by_amplitude
        self.color_by_template_if_possible = color_by_template_if_possible
        self.alpha = alpha
        self.label = label or feat_name
        self.cbar = cbar

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id: int):
        axis = panel.subplots()
        assert sorting_analysis.times_seconds is not None
        assert sorting_analysis.amplitudes is not None

        in_unit = sorting_analysis.in_unit(unit_id, at_most=50_000)
        t = sorting_analysis.times_seconds[in_unit]
        feat = sorting_analysis.named_feature(self.feat_name, which=in_unit)
        c = None
        cbar = self.cbar
        did_by_template = False
        if c is None and self.color_by_template_if_possible:
            temp_ix = getattr(sorting_analysis.sorting, "template_inds", None)
            if temp_ix is not None:
                temp_ix = temp_ix[in_unit]
                c = glasbey1024[temp_ix % len(glasbey1024)]
                did_by_template = True
                cbar = False
        if c is None and self.color_by_amplitude:
            amps = sorting_analysis.amplitudes[in_unit]
            c = np.minimum(amps, self.amplitude_color_cutoff)
        s = axis.scatter(t, feat, c=c, lw=0, s=3, alpha=self.alpha, rasterized=True)
        axis.set_xlabel("time (s)")
        axis.set_ylabel(self.label)
        axis.grid()
        axis.set_axisbelow(True)
        if cbar and self.color_by_amplitude:
            plt.colorbar(s, ax=axis, shrink=0.5, pad=0.01, label="amp (su)")
        if did_by_template:
            axis.text(
                0.97,
                0.97,
                "color: template",
                ha="right",
                va="top",
                transform=axis.transAxes,
                fontsize="small",
            )


class TimeZScatter(TimeFeatScatter):
    def __init__(self, **kwargs):
        super().__init__(feat_name="z", label="z (um)", **kwargs)


class TimeRegZScatter(TimeFeatScatter):
    def __init__(self, **kwargs):
        super().__init__(feat_name="registered_z", label="reg. z (um)", **kwargs)


class TimeAmpScatter(TimeFeatScatter):
    def __init__(self, color_by_template_if_possible=True, **kwargs):
        super().__init__(
            feat_name="amplitudes",
            label="amp (su)",
            color_by_template_if_possible=color_by_template_if_possible,
            **kwargs,
        )


class AmplitudeHistogramByDiscreteVariable(UnitPlot):
    kind = "camphist"
    width = 2
    height = 0.75

    def __init__(self, var="channels"):
        self.var = var

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id: int):
        axis = panel.subplots()
        assert sorting_analysis.amplitudes is not None
        z = getattr(sorting_analysis.sorting, self.var, None)
        if z is None:
            axis.axis("off")
            return
        in_unit = sorting_analysis.in_unit(unit_id, at_most=50_000)
        a = sorting_analysis.amplitudes[in_unit]
        z = z[in_unit]
        bins = np.arange(np.floor(a.min()), np.ceil(a.max()) + 0.1)
        uqz = np.unique(z)
        for uz in uqz:
            axis.hist(
                a[z == uz],
                color=glasbey1024[uz % len(glasbey1024)],
                histtype="step",
                lw=1,
                label=uz,
                bins=bins,
            )
        if uqz.size < 4:
            axis.legend(title=self.var, fancybox=False, loc="upper right")
        else:
            msg = f"{self.var}, {uqz.size} uniques"
            if uqz.size < 8:
                msg += ":\n"
                msg += ",".join(list(map(str, uqz.tolist())))
            axis.text(
                0.97,
                0.97,
                msg,
                fontsize="small",
                ha="right",
                va="top",
                transform=axis.transAxes,
            )
        axis.grid()
        axis.set_axisbelow(True)
        axis.semilogy()
        axis.set_xlabel("amplitude (s.u.)")


# -- waveform plots


class WaveformPlot(UnitPlot):
    kind = "waveform"
    width = 3
    height = 2
    can_sharey = False

    # for my subclasses
    wfs_kind = ""

    def __init__(
        self,
        count=100,
        color="k",
        color_by="z",
        alpha=0.1,
        show_template=True,
        template_color="orange",
        max_abs_template_scale=1.5,
        legend=True,
        title=None,
    ):
        super().__init__()
        self.count = count
        self.color = color
        self.alpha = alpha
        self.show_template = show_template
        self.template_color = template_color
        self.legend = legend
        self.max_abs_template_scale = max_abs_template_scale
        self.title = title
        self.color_by = color_by

    def get_waveforms(
        self, sorting_analysis: DARTsortAnalysis, unit_id: int
    ) -> WaveformsBag | None:
        raise NotImplementedError

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id, axis=None):
        if axis is None:
            axis = panel.subplots()

        waves = self.get_waveforms(sorting_analysis, unit_id)
        tslice = None if waves is None else waves.temporal_slice
        trough_offset_samples = sorting_analysis.trough_offset_samples
        spike_length_samples = sorting_analysis.spike_length_samples
        if tslice is not None and tslice.start is not None:
            trough_offset_samples = (
                sorting_analysis.trough_offset_samples - tslice.start
            )
            spike_length_samples = sorting_analysis.spike_length_samples - tslice.start
        if tslice is not None and tslice.stop is not None:
            spike_length_samples = tslice.stop - tslice.start

        if not self.color_by or waves is None:
            ckw = dict(color=self.color)
        elif self.color_by == "z":
            assert sorting_analysis.z is not None
            cc = sorting_analysis.z[waves.which]
            if cc.std() > 1e-12:
                cc = (cc - cc.min()) / np.ptp(cc)
            else:
                cc = np.full_like(cc, 0.5)
            ckw = dict(colors=plt.get_cmap("berlin")(cc))
        else:
            assert False

        max_abs_amp = None
        show_template = self.show_template
        template_color = self.template_color
        assert sorting_analysis.coarse_template_data is not None
        if show_template:
            templates = sorting_analysis.coarse_template_data.unit_templates(unit_id)
            show_template = bool(templates.size)
        else:
            templates = None

        if show_template:
            assert templates is not None
            templates = trim_waveforms(
                templates,
                old_offset=sorting_analysis.coarse_template_data.trough_offset_samples,
                new_offset=trough_offset_samples,
                new_length=spike_length_samples,
            )
            max_abs_amp = self.max_abs_template_scale * np.nanmax(np.abs(templates))

        if templates is not None and waves is not None and templates.shape[0] == 1:
            assert waves.channel_index.shape[0] == templates.shape[2]
            x = waves.waveforms.reshape(len(waves.waveforms), -1)
            y = np.pad(templates[0], [(0, 0), (0, 1)], constant_values=np.nan)
            y = np.broadcast_to(y, (x.shape[0], *y.shape))
            if waves.channels is not None:
                c = waves.channels[:, None, :]
            else:
                c = waves.channel_index[waves.main_channel][None, None, :]
            c = np.broadcast_to(c, waves.waveforms.shape)
            y = np.take_along_axis(y, axis=2, indices=c)
            y = y.reshape(len(waves.waveforms), -1)
            assert np.array_equal(np.isnan(x), np.isnan(y))
            x = np.nan_to_num(x)
            y = np.nan_to_num(y)
            xy = np.einsum("nj,nj->n", x, y)
            yn = np.linalg.norm(y, axis=1)
            ynsq = yn**2
            s = xy / ynsq
            m = np.sqrt((xy**2) / ynsq)
            div = np.linalg.norm(x - y * s[:, None]) / yn
            rmsg = f" sc: {s.mean().item():0.3f} mq: {m.mean().item():0.3f} div: {div.mean().item():0.3f}"
        else:
            rmsg = ""

        handles = {}
        if waves is not None:
            if np.isfinite(waves.waveforms[:, 0, :]).any():
                max_abs_amp = self.max_abs_template_scale * np.nanpercentile(
                    np.abs(waves.waveforms), 99
                )
            ls = geomplot(
                waves.waveforms,
                channels=waves.channels,
                max_channels=np.full(len(waves.waveforms), waves.main_channel),
                channel_index=waves.channel_index,
                geom=waves.geom,
                ax=axis,
                show_zero=False,
                subar=True,
                msbar=False,
                zlim="tight",
                alpha=self.alpha,
                max_abs_amp=max_abs_amp,
                trough_offset=trough_offset_samples,
                lw=1,
                **ckw,
            )
            handles["waveforms"] = ls

        if show_template:
            assert templates is not None
            if waves is not None and waves.channels is not None:
                channels = np.unique(waves.channels)
            else:
                mc = sorting_analysis.unit_max_channel(unit_id)
                channels = sorting_analysis.vis_channel_index[mc]
            channels = channels[channels < len(sorting_analysis.vis_channel_index)]
            cc = np.broadcast_to(channels, (len(templates), *channels.shape))
            ls = geomplot(
                templates[:, :, channels],
                geom=sorting_analysis.registered_geom,
                channels=cc,
                ax=axis,
                show_zero=False,
                zlim="tight",
                color=template_color,
                alpha=1,
                max_abs_amp=max_abs_amp,
                lw=1,
            )
            handles["mean"] = ls

        shift_str = "shifted " * sorting_analysis.shifting
        if self.title is None:
            axis.set_title(shift_str + self.wfs_kind + rmsg, fontsize="small")
        else:
            axis.set_title(self.title + rmsg, fontsize="small")
        axis.set_xticks([])
        axis.set_yticks([])

        if self.legend:
            axis.legend(
                handles.values(),
                handles.keys(),
                handler_map={tuple: HandlerTuple(ndivide=None)},
                fancybox=False,
                loc="upper left",
            )


class RawWaveformPlot(WaveformPlot):
    wfs_kind = "raw wfs"

    def get_waveforms(
        self, sorting_analysis: DARTsortAnalysis, unit_id: int
    ) -> WaveformsBag | None:
        return sorting_analysis.unit_raw_waveforms(unit_id, max_count=self.count)


class TPCAWaveformPlot(WaveformPlot):
    wfs_kind = "c-c tpca wfs"

    def get_waveforms(
        self, sorting_analysis: DARTsortAnalysis, unit_id: int
    ) -> WaveformsBag | None:
        return sorting_analysis.unit_tpca_waveforms(unit_id, max_count=self.count)


# -- merge-focused plots


class NearbyCoarseTemplatesPlot(UnitPlot):
    title = "nearby coarse templates"
    kind = "neighbors"
    width = 3
    height = 2
    can_sharey = False

    def __init__(self, n_neighbors=5, legend=True):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.legend = legend

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id, axis=None):
        if axis is None:
            axis = panel.subplots()
        if np.asarray(unit_id).size > 1:
            unit_id = unit_id[0]
        (
            neighbor_ixs,
            neighbor_ids,
            neighbor_dists,
            neighbor_coarse_templates,
        ) = sorting_analysis.nearby_coarse_templates(
            unit_id, n_neighbors=self.n_neighbors
        )
        if not np.equal(neighbor_ids, unit_id).any():
            axis.axis("off")
            return
        assert neighbor_ids[0] == unit_id
        geomplot_templates(
            axis,
            neighbor_ids,
            neighbor_coarse_templates,
            sorting_analysis.vis_channel_index,
            sorting_analysis.registered_geom,
            main_channel=sorting_analysis.unit_max_channel(unit_id),
            title="",
        )


class CoarseTemplateDistancePlot(UnitPlot):
    title = "coarse template distance"
    kind = "neighbors"
    width = 3
    height = 2

    def __init__(
        self,
        channel_show_radius_um=50,
        n_neighbors=5,
        dist_vmax=1.0,
        show_values=True,
    ):
        super().__init__()
        self.channel_show_radius_um = channel_show_radius_um
        self.n_neighbors = n_neighbors
        self.dist_vmax = dist_vmax
        self.show_values = show_values

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id, axis=None):
        if np.asarray(unit_id).size > 1:
            unit_id = unit_id[0]
        if axis is None:
            axis = panel.subplots()
        (
            neighbor_ixs,
            neighbor_ids,
            neighbor_dists,
            neighbor_coarse_templates,
        ) = sorting_analysis.nearby_coarse_templates(
            unit_id, n_neighbors=self.n_neighbors
        )
        colors = np.array(glasbey1024)[neighbor_ids % len(glasbey1024)]
        if not np.equal(neighbor_ids, unit_id).any():
            axis.axis("off")
            return
        assert neighbor_ids[0] == unit_id

        im = axis.imshow(
            neighbor_dists,
            vmin=0,
            vmax=self.dist_vmax,
            cmap="RdGy",
            origin="lower",
            interpolation="none",
            aspect="auto",
        )
        if self.show_values:
            if sorting_analysis.merge_lags is not None:
                lags = sorting_analysis.merge_lags[neighbor_ixs][:, neighbor_ixs]
            else:
                lags = None
            for (i, j), d in np.ndenumerate(neighbor_dists):
                txt = f"{d:.2f}".lstrip("0")
                if lags is not None:
                    txt += f":{lags[i, j].item():+d}"

                axis.text(i, j, txt, ha="center", va="center")
        plt.colorbar(im, ax=axis, shrink=0.3)
        if sorting_analysis.merge_r2 is not None:
            r2 = sorting_analysis.merge_r2[neighbor_ixs]
            if np.isclose(r2.min(), 1.0):
                xt = neighbor_ids
            else:
                xt = [
                    f"{nid.item()}\n" + f"{rr:.2f}".lstrip("0")
                    for nid, rr in zip(neighbor_ids, r2)
                ]
        else:
            xt = neighbor_ids
        axis.set_xticks(range(len(neighbor_ids)), xt)
        axis.set_yticks(range(len(neighbor_ids)), neighbor_ids)
        for i, (tx, ty) in enumerate(
            zip(axis.xaxis.get_ticklabels(), axis.yaxis.get_ticklabels())
        ):
            tx.set_color(colors[i])
            ty.set_color(colors[i])
        axis.set_title(self.title, fontsize="small")


class NeighborQDAMatrices(UnitPlot):
    kind = "tall"
    width = 2
    height = 4

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id):
        neighbor_ixs, neighbor_ids, _, _ = sorting_analysis.nearby_coarse_templates(
            unit_id
        )
        colors = np.array(glasbey1024)[neighbor_ids % len(glasbey1024)]
        if not np.equal(neighbor_ids, unit_id).any():
            panel.subplots().axis("off")
            return
        assert neighbor_ids[0] == unit_id

        axes = panel.subplots(nrows=2)

        qda = sorting_analysis.qda
        assert qda is not None

        score = qda.score[neighbor_ixs][:, neighbor_ixs]
        iou = qda.iou[neighbor_ixs][:, neighbor_ixs]
        min_ratio = qda.min_ratio[neighbor_ixs][:, neighbor_ixs]
        coverage = qda.coverage[neighbor_ixs][:, neighbor_ixs]

        for ax, (sc, ol), title in zip(
            axes, [(score, iou), (min_ratio, coverage)], ["qda/iou", "ratio/coverage"]
        ):
            ax.imshow(
                sc,
                vmin=0,
                cmap="plasma",
                origin="lower",
                interpolation="none",
                aspect="auto",
            )
            vm = sc.max()
            for (i, j), d in np.ndenumerate(sc):
                ostr = f"{ol[i, j]:.2f}".lstrip("0")
                if ostr == ".00":
                    ostr = "0"
                else:
                    ostr = ostr.rstrip("0")
                dstr = f"{d:.2f}".lstrip("0")
                if dstr == ".00":
                    dstr = "0"
                else:
                    dstr = dstr.rstrip("0")
                txt = f"{dstr}\n({ostr})"
                ax.text(
                    i, j, txt, ha="center", va="center", c="k" if d > vm / 2 else "w"
                )
            ax.set_xticks(range(len(neighbor_ids)), neighbor_ids)
            ax.set_yticks(range(len(neighbor_ids)), neighbor_ids)
            for i, (tx, ty) in enumerate(
                zip(ax.xaxis.get_ticklabels(), ax.yaxis.get_ticklabels())
            ):
                tx.set_color(colors[i])
                ty.set_color(colors[i])
            ax.set_title(title, fontsize="small")


class NeighborCCGPlot(UnitPlot):
    kind = "bneighborccg"

    def __init__(self, n_neighbors=5, max_lag=50, bin=1, unit="samples"):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.max_lag = max_lag
        self.height = 2
        self.width = 2
        self.unit = unit
        self.bin = bin

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id: int):
        (
            neighbor_ixs,
            neighbor_ids,
            neighbor_dists,
            neighbor_coarse_templates,
        ) = sorting_analysis.nearby_coarse_templates(
            unit_id, n_neighbors=self.n_neighbors + 1
        )
        # assert neighbor_ids[0] == unit_id
        neighbor_ids = neighbor_ids[1:]
        if not neighbor_ids.size:
            return
        colors = np.array(glasbey1024)[neighbor_ids % len(glasbey1024)]

        my_st = sorting_analysis.sorting.times_samples[
            sorting_analysis.in_unit(unit_id)
        ]
        neighb_sts = [
            sorting_analysis.sorting.times_samples[sorting_analysis.in_unit(nid)]
            for nid in neighbor_ids
        ]

        axes = panel.subplots(
            ncols=1,
            nrows=len(neighb_sts),
            sharey="row",
            sharex=True,
            squeeze=False,
        )
        axes = axes.T

        samples_per_ms = sorting_analysis.sorting.sampling_frequency / 1000
        if self.unit == "samples":
            max_lag_samples = self.max_lag
        elif self.unit == "ms":
            max_lag_samples = int(np.ceil(self.max_lag * samples_per_ms))
        else:
            assert False

        for j in range(len(neighb_sts)):
            plot_correlogram(
                axes[0, j],
                my_st,
                neighb_sts[j],
                max_lag=max_lag_samples,
                bin=self.bin,
                samples_per_ms=samples_per_ms,
                to_ms=self.unit == "ms",
                fc=colors[j],
            )
            axes[0, j].grid(which="both")
            axes[0, j].text(
                0.97,
                0.97,
                f"vs. unit {neighbor_ids[j]}",
                ha="right",
                va="top",
                transform=axes[0, j].transAxes,
                fontsize="small",
            )


class NeighborQDAPlot(UnitPlot):
    kind = "neighbors"

    def __init__(self, count=5, log=False, ncols=1, kind="kde", kde_bin=1.0):
        super().__init__()
        self.count = count
        self.log = log
        self.ncols = ncols
        self.width = 1.5 * ncols
        self.height = 1.5 * count
        self.kind = kind
        self.kde_bin = kde_bin

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id: int):
        neighbor_ixs, neighbor_ids, _, _ = sorting_analysis.nearby_coarse_templates(
            unit_id
        )
        if not np.equal(neighbor_ids, unit_id).any():
            panel.subplots().axis("off")
            return
        assert neighbor_ids[0] == unit_id
        neighbor_ids = neighbor_ids[1:]
        neighbor_ixs = neighbor_ixs[1:]
        if not neighbor_ids.size:
            panel.subplots().axis("off")
            return
        colors = np.array(glasbey1024)[neighbor_ids % len(glasbey1024)]

        candidates = sorting_analysis.sorting.gmm_candidates
        log_liks = sorting_analysis.sorting.gmm_log_liks

        axes = panel.subplots(
            squeeze=False,
            nrows=int(np.ceil(len(neighbor_ixs) / self.ncols)),
            ncols=self.ncols,
            gridspec_kw=dict(hspace=0.0, wspace=0.0),
        )
        in_unit_id = sorting_analysis.in_unit(unit_id)
        for ax, nix, nid, color in zip(axes.flat, neighbor_ixs, neighbor_ids, colors):
            in_nid = sorting_analysis.in_unit(nid)
            na = in_unit_id.size
            nb = in_nid.size
            if na + nb < 20:
                ax.set_title(f"{nid}: {na=} {nb=}", fontsize="small")
                continue

            in_pair = np.concatenate([in_unit_id, in_nid])
            cand = candidates[in_pair]
            ll = log_liks[in_pair]

            my_mask = cand == unit_id
            nid_mask = cand == nid

            overlap = np.logical_and(nid_mask.any(1), my_mask.any(1))
            count = overlap.sum()
            iou = count / in_pair.shape[0]
            cov = min(overlap[:na].sum() / na, overlap[na:].sum() / nb)
            iout = f"{nid}: iou=" + f"{iou:.2f}".lstrip("0")
            covt = f"{nid}: cov=" + f"{cov:.2f}".lstrip("0")
            if count < 16:
                ax.set_title(
                    f"{nid}: {iout}\n{covt}\nolap count {count}", fontsize="small"
                )
                continue

            _, my_ix = np.nonzero(my_mask[overlap])
            my_ll = np.take_along_axis(ll[overlap], my_ix[:, None], axis=1)[:, 0]
            _, their_ix = np.nonzero(nid_mask[overlap])
            their_ll = np.take_along_axis(ll[overlap], their_ix[:, None], axis=1)[:, 0]
            dll = my_ll - their_ll

            ax.axvline(0, color="k", lw=0.8)
            ax.grid()
            ax.set_axisbelow(True)
            hstat = kstat = ""

            if self.kind == "hist":
                ax.hist(
                    dll,
                    bins=32,
                    color=color,
                    histtype="step",
                    log=self.log,
                    density=True,
                )
            elif self.kind == "kde":
                bines, bincs = centered_bins(dll)
                if not bincs.size:
                    continue
                hist, _ = np.histogram(dll, bins=bines)
                hist = hist.astype(np.float64) / hist.sum()
                ax.stairs(hist, edges=bines, color="k", zorder=10)
                hsa, hsb = bimod_stats(hist)
                hstat = f"a:{fstr(hsa)}, b:{fstr(hsb)}"
                try:
                    kdest = FFTKDE(bw="ISJ").fit(dll)
                except ValueError as e:
                    ax.set_title(f"{iout}\n{covt}\n{hstat}", fontsize="small")
                    ax.text(
                        0.5,
                        0.5,
                        f"KDE fail, n={len(dll)}\n{str(e)[:10]}.",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                    continue
                kde = cast(np.ndarray, kdest.evaluate(bincs))
                kde /= kde.sum()
                ksa, ksb = bimod_stats(kde)
                kstat = f"a:{fstr(ksa)}, b:{fstr(ksb)}"
                if not np.isfinite(kde).all():
                    ax.text(
                        0.5,
                        0.5,
                        f"KDE nan\nn={len(dll)}",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                else:
                    ax.step(y=kde, x=bincs, color=color, zorder=11)
                    if self.log:
                        ax.semilogy()

            ax.set_title(f"{iout}\n{covt}\n{hstat}\n{kstat}", fontsize="small")

        for ax in axes.flat[len(neighbor_ids) :]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("density")
        for ax in axes[-1]:
            ax.set_xlabel("my ll - their ll")


# -- main routines


def default_plots(sorting_analysis=None):
    p = [
        UnitTextInfo(),
        ACG(),
        ACG(max_lag=50.0, bin=0.5, unit="ms"),
        ISIHistogram(),
        ISIHistogram(bin_ms=0.25, max_ms=50.0),
        XZScatter(),
        AmplitudeHistogramByDiscreteVariable(),
        TimeAmpScatter(),
        RawWaveformPlot(),
        NearbyCoarseTemplatesPlot(),
        CoarseTemplateDistancePlot(),
        NeighborCCGPlot(),
        NeighborCCGPlot(max_lag=50.0, bin=0.5, unit="ms"),
    ]
    if sorting_analysis is not None and sorting_analysis.has_localizations():
        p.extend([TimeZScatter(), TimeRegZScatter()])
    if sorting_analysis is not None and sorting_analysis.has_pca():
        p.extend([PCAScatter(), TPCAWaveformPlot()])
    if sorting_analysis is not None and sorting_analysis.qda is not None:
        p.extend([NeighborQDAMatrices(), NeighborQDAPlot()])
    if sorting_analysis is not None and hasattr(
        sorting_analysis.sorting, "template_inds"
    ):
        p.extend([AmplitudeHistogramByDiscreteVariable("template_inds")])
    return p


def no_pca_unit_plots(sorting_analysis=None):
    del sorting_analysis
    return (
        UnitTextInfo(),
        ACG(),
        ISIHistogram(),
        XZScatter(),
        TimeZScatter(),
        TimeRegZScatter(),
        TimeAmpScatter(),
        RawWaveformPlot(),
        NearbyCoarseTemplatesPlot(),
        CoarseTemplateDistancePlot(),
        NeighborCCGPlot(),
    )


@np.errstate(over="raise")
def make_unit_summary(
    sorting_analysis: DARTsortAnalysis,
    unit_id,
    amplitude_color_cutoff=15.0,
    pca_radius_um=75.0,
    plots=None,
    max_height=4,
    figsize=(18, 8.5),
    figure=None,
    gizmo_name="sorting_analysis",
    **other_global_params,
):
    if plots is None:
        plots = default_plots(sorting_analysis)
    # notify plots of global params
    for p in plots:
        p.notify_global_params(
            amplitude_color_cutoff=amplitude_color_cutoff,
            pca_radius_um=pca_radius_um,
            **other_global_params,
        )

    figure = layout.flow_layout(
        plots,
        max_height=max_height,
        figsize=figsize,
        figure=figure,
        unit_id=unit_id,
        **{gizmo_name: sorting_analysis},
    )

    return figure


def make_all_summaries(
    sorting_analysis: DARTsortAnalysis,
    save_folder,
    plots=None,
    amplitude_color_cutoff=15.0,
    pca_radius_um=75.0,
    max_height=4,
    figsize=(18, 8.5),
    dpi=200,
    image_ext="png",
    n_jobs=None,
    show_progress=True,
    namebyamp=False,
    overwrite=False,
    unit_ids=None,
    gizmo_name="sorting_analysis",
    n_units=None,
    seed=0,
    taskname="summaries",
    **other_global_params,
):
    if plots is None:
        plots = default_plots(sorting_analysis)
    save_folder = Path(save_folder)
    if unit_ids is None:
        unit_ids = sorting_analysis.sorting.unit_ids
    if n_units is not None and n_units < len(unit_ids):
        rg = np.random.default_rng(seed)
        unit_ids = rg.choice(unit_ids, size=n_units, replace=False)
        unit_ids.sort()
    if not overwrite and all_summaries_done(
        unit_ids,
        save_folder,
        sorting_analysis=sorting_analysis,
        namebyamp=namebyamp,
        ext=image_ext,
    ):
        return

    save_folder.mkdir(exist_ok=True, parents=True)

    global_params = dict(
        amplitude_color_cutoff=amplitude_color_cutoff,
        pca_radius_um=pca_radius_um,
        **other_global_params,
    )
    initargs = (
        sorting_analysis,
        plots,
        max_height,
        figsize,
        dpi,
        save_folder,
        image_ext,
        overwrite,
        global_params,
        gizmo_name,
        namebyamp,
    )
    if n_jobs is None:
        n_jobs = get_global_computation_config().n_jobs_cpu
    n_jobs, Executor, context = get_pool(n_jobs, cls=CloudpicklePoolExecutor)
    if n_jobs:
        initargs = (cloudpickle.dumps(initargs),)  # type: ignore
    with Executor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=_summary_init,
        initargs=initargs,
    ) as pool:
        results = pool.map(_summary_job, unit_ids)
        if show_progress:
            results = tqdm(
                results,
                desc=f"Unit {taskname}",
                smoothing=0,
                total=len(unit_ids),
            )
        for _ in results:
            pass


# -- utilities


def trim_waveforms(waveforms, old_offset=42, new_offset=42, new_length=121):
    if waveforms.shape[1] == new_length and old_offset == new_offset:
        return waveforms

    start = old_offset - new_offset
    end = start + new_length
    return waveforms[:, start:end]


def pngname(unit_id, sorting_analysis=None, namebyamp=False, ext="png"):
    if not namebyamp:
        return f"unit{unit_id:04d}.{ext}"
    if sorting_analysis is None:
        raise ValueError("Need a sorting_analysis if namebyamp.")
    amp = float(sorting_analysis.unit_amplitudes(unit_id).item())
    amp = f"{amp:07.2f}"
    return f"amp{amp}_unit{unit_id:04d}.{ext}"


def all_summaries_done(
    unit_ids, save_folder, sorting_analysis=None, namebyamp=False, ext="png"
):
    if not save_folder.exists():
        return False
    return all(
        (
            save_folder
            / pngname(
                unit_id, sorting_analysis=sorting_analysis, namebyamp=namebyamp, ext=ext
            )
        ).exists()
        for unit_id in unit_ids
    )


# -- parallelism helpers


class SummaryJobContext:
    def __init__(
        self,
        sorting_analysis: DARTsortAnalysis,
        plots,
        max_height,
        figsize,
        dpi,
        save_folder,
        image_ext,
        overwrite,
        global_params,
        gizmo_name,
        namebyamp,
    ):
        self.sorting_analysis = sorting_analysis
        self.plots = plots
        self.max_height = max_height
        self.figsize = figsize
        self.dpi = dpi
        self.save_folder = save_folder
        self.image_ext = image_ext
        self.overwrite = overwrite
        self.global_params = global_params
        self.gizmo_name = gizmo_name
        self.namebyamp = namebyamp


_summary_job_context = None


def _summary_init(*args):
    global _summary_job_context
    if len(args) == 1:
        args = cloudpickle.loads(args[0])  # type: ignore
    _summary_job_context = SummaryJobContext(*args)


def _summary_job(unit_id: int):
    # handle resuming/overwriting
    assert _summary_job_context is not None
    ext = _summary_job_context.image_ext
    tmp_out = _summary_job_context.save_folder / f"tmp_unit{unit_id:04d}.{ext}"
    imfn = pngname(
        unit_id,
        sorting_analysis=_summary_job_context.sorting_analysis,
        namebyamp=_summary_job_context.namebyamp,
        ext=_summary_job_context.image_ext,
    )
    final_out = _summary_job_context.save_folder / imfn
    if tmp_out.exists():
        tmp_out.unlink()
    if not _summary_job_context.overwrite and final_out.exists():
        return
    if _summary_job_context.overwrite and final_out.exists():
        final_out.unlink()

    fig = plt.figure(
        figsize=_summary_job_context.figsize,
        layout="constrained",
        # dpi=_summary_job_context.dpi,
    )
    try:
        make_unit_summary(
            _summary_job_context.sorting_analysis,
            unit_id,
            plots=_summary_job_context.plots,
            max_height=_summary_job_context.max_height,
            figsize=_summary_job_context.figsize,
            figure=fig,
            gizmo_name=_summary_job_context.gizmo_name,
            **_summary_job_context.global_params,
        )
    except Exception as e:
        raise ValueError(f"Error making plots for {unit_id=}.") from e

    # the save is done sort of atomically to help with the resuming and avoid
    # half-baked image files
    fig.savefig(tmp_out, dpi=_summary_job_context.dpi)
    tmp_out.rename(final_out)
    plt.close(fig)


def fstr(x):
    return f"{x:0.2f}".lstrip("0")
