"""Toolkit for extensible single unit summary plots

The goal is to make it easy to add and remove plots without the plotting
code turning into a web of if statements, for loops, and bizarre subplot
and subfigure mazes.

Relies on the DARTsortAnalysis object of utils/analysis.py to do most of
the data work so that this file can focus on plotting (sort of MVC).
"""

from dataclasses import replace
from pathlib import Path

from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple

from ..cluster import split
from ..config import raw_template_config
from ..util.analysis import DARTsortAnalysis
from ..util.multiprocessing_util import CloudpicklePoolExecutor, get_pool
from . import layout
from .colors import glasbey1024
from .waveforms import geomplot, geomplot_max_chan_overtime

# -- main class. see fn make_unit_summary below to make lots of UnitPlots.

class UnitPlot(layout.BasePlot):
    can_sharey = True

    def draw(self, panel, sorting_analysis, unit_id):
        raise NotImplementedError


class UnitMultiPlot(layout.BaseMultiPlot):
    def plots(self, sorting_analysis, unit_id):
        # return [UnitPlot()]
        raise NotImplementedError


class UnitTextInfo(UnitPlot):
    kind = "text"
    height = 0.5
    width = 1

    def draw(self, panel, sorting_analysis, unit_id, sorting_previous_step):
        axis = panel.subplots()
        axis.axis("off")
        msg = f"unit {unit_id}\n"

        msg += f"feature source: {sorting_analysis.hdf5_path.name}\n"

        nspikes = sorting_analysis.spike_counts[
            sorting_analysis.unit_ids == unit_id
        ].sum()
        msg += f"n spikes: {nspikes}\n"

        temps = sorting_analysis.template_data.unit_templates(unit_id)
        if temps.size:
            ptp = temps.ptp(1).max(1).mean()
            msg += f"mean superres maxptp: {ptp:0.1f}su\n"
            in_unit = sorting_analysis.template_data.unit_mask(unit_id)
            counts = sorting_analysis.template_data.spike_counts[in_unit]
            snrs = temps.ptp(1).max(1) * np.sqrt(counts)
            msg += "template snrs:\n  " + ", ".join(f"{s:0.1f}" for s in snrs)
        else:
            msg += "no template (too few spikes)"

        axis.text(0, 0, msg, fontsize=6.5)


# -- small summary plots


class ACG(UnitPlot):
    kind = "histogram"
    height = 0.75
    width = 1

    def __init__(self, max_lag=50):
        super().__init__()
        self.max_lag = max_lag

    def draw(self, panel, sorting_analysis, unit_id, sorting_previous_step):
        axis = panel.subplots()
        times_samples = sorting_analysis.times_samples(
            which=sorting_analysis.in_unit(unit_id)
        )
        lags, acg = correlogram(times_samples, max_lag=self.max_lag)
        bar(axis, lags, acg, fill=True, color="k")
        axis.set_xlabel("lag (samples)", fontsize=5, labelpad=2)
        axis.set_ylabel("acg", fontsize=5, labelpad=2)
        axis.xaxis.set_tick_params(labelsize=5)
        axis.yaxis.set_tick_params(labelsize=5)


class ISIHistogram(UnitPlot):
    kind = "histogram"
    height = 0.75
    width = 1

    def __init__(self, bin_ms=0.1, max_ms=5):
        super().__init__()
        self.bin_ms = bin_ms
        self.max_ms = max_ms

    def draw(self, panel, sorting_analysis, unit_id, sorting_previous_step):
        axis = panel.subplots()
        times_s = sorting_analysis.times_seconds(
            which=sorting_analysis.in_unit(unit_id)
        )
        dt_ms = np.diff(times_s) * 1000
        bin_edges = np.arange(
            0,
            self.max_ms + self.bin_ms,
            self.bin_ms,
        )
        # counts, _ = np.histogram(dt_ms, bin_edges)
        # bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        # axis.bar(bin_centers, counts)
        plt.hist(dt_ms, bin_edges, color="k")
        axis.set_xlabel("isi (ms)", fontsize=5, labelpad=2)
        axis.set_ylabel(f"count (out of {dt_ms.size} total isis)", fontsize=5, labelpad=2)
        axis.xaxis.set_tick_params(labelsize=5)
        axis.yaxis.set_tick_params(labelsize=5)


class XZScatter(UnitPlot):
    kind = "scatter"
    width = 1

    def __init__(
        self,
        relocate_amplitudes=False,
        registered=True,
        amplitude_color_cutoff=15,
        probe_margin_um=100,
        colorbar=False,
    ):
        super().__init__()
        self.relocate_amplitudes = relocate_amplitudes
        self.registered = registered
        self.amplitude_color_cutoff = amplitude_color_cutoff
        self.probe_margin_um = probe_margin_um
        self.colorbar = colorbar

    def draw(self, panel, sorting_analysis, unit_id, axis=None):
        if axis is None:
            axis = panel.subplots()

        unit_id = np.atleast_1d(unit_id)
        multi_unit = unit_id.size > 1
        for uid in unit_id:
            in_unit = sorting_analysis.in_unit(uid)
            x = sorting_analysis.x(which=in_unit)
            z = sorting_analysis.z(which=in_unit, registered=self.registered)
            geomx, geomz = sorting_analysis.geom.T
            pad = self.probe_margin_um
            valid = x == np.clip(x, geomx.min() - pad, geomx.max() + pad)
            valid &= z == np.clip(z, geomz.min() - pad, geomz.max() + pad)
            if multi_unit:
                c = dict(color=glasbey1024[uid % len(glasbey1024)])
            else:
                amps = sorting_analysis.amplitudes(
                    which=in_unit[valid], relocated=self.relocate_amplitudes
                )
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
        reloc_str = "reloc " * self.relocate_amplitudes
        if not multi_unit and self.colorbar:
            plt.colorbar(s, ax=axis, shrink=0.5, label=reloc_str + "amp (su)")


class PCAScatter(UnitPlot):
    kind = "scatter"
    width = 1

    def __init__(
        self,
        relocate_amplitudes=False,
        relocated=True,
        amplitude_color_cutoff=15,
        pca_radius_um=75.0,
        colorbar=False,
    ):
        super().__init__()
        self.relocated = relocated
        self.relocate_amplitudes = relocate_amplitudes
        self.amplitude_color_cutoff = amplitude_color_cutoff
        self.colorbar = colorbar
        self.pca_radius_um = pca_radius_um

    def draw(self, panel, sorting_analysis, unit_id, axis=None):
        if axis is None:
            axis = panel.subplots()
        unit_id = np.atleast_1d(unit_id)
        multi_unit = unit_id.size > 1
        which, loadings = sorting_analysis.unit_pca_features(
            unit_id=unit_id,
            relocated=self.relocated,
            pca_radius_um=self.pca_radius_um,
        )
        if which is not None:
            for uid in unit_id:
                if multi_unit:
                    c = dict(color=glasbey1024[uid % len(glasbey1024)])
                    thisu = np.flatnonzero(
                        sorting_analysis.sorting.labels[which] == uid
                    )
                else:
                    amps = sorting_analysis.amplitudes(
                        which=which, relocated=self.relocate_amplitudes
                    )
                    c = dict(c=np.minimum(amps, self.amplitude_color_cutoff))
                    thisu = slice(None)
                s = axis.scatter(
                    *loadings[thisu].T,
                    lw=0,
                    s=3,
                    **c,
                    rasterized=True,
                )

        reloc_str = "reloc " * self.relocated
        axis.set_xlabel(reloc_str + "PC1")
        axis.set_ylabel(reloc_str + "PC2")
        if not multi_unit:
            reloc_amp_str = "reloc " * self.relocate_amplitudes
            if which is not None and self.colorbar:
                plt.colorbar(s, ax=axis, shrink=0.5, label=reloc_amp_str + "amp (su)")

        return axis


# -- wide scatter plots


class TimeZScatter(UnitPlot):
    kind = "widescatter"
    width = 1.25
    height = 1.25


    def __init__(
        self,
        relocate_amplitudes=False,
        registered=True,
        amplitude_color_cutoff=15,
        probe_margin_um=100,
    ):
        super().__init__()
        self.relocate_amplitudes = relocate_amplitudes
        self.registered = registered
        self.amplitude_color_cutoff = amplitude_color_cutoff
        self.probe_margin_um = probe_margin_um

    def draw(self, panel, sorting_analysis, unit_id, sorting_previous_step):
        unit_id = np.atleast_1d(unit_id)
        multi_unit = unit_id.size > 1
        axis = panel.subplots()

        for uid in unit_id:
            in_unit = sorting_analysis.in_unit(uid)
            t = sorting_analysis.times_seconds(which=in_unit)
            z = sorting_analysis.z(which=in_unit, registered=self.registered)
            geomx, geomz = sorting_analysis.geom.T
            pad = self.probe_margin_um
            valid = z == np.clip(z, geomz.min() - pad, geomz.max() + pad)
            if multi_unit:
                c = dict(color=glasbey1024[uid % len(glasbey1024)])
            else:
                amps = sorting_analysis.amplitudes(
                    which=in_unit[valid], relocated=self.relocate_amplitudes
                )
                c = dict(c=np.minimum(amps, self.amplitude_color_cutoff))

            s = axis.scatter(
                t[valid],
                z[valid],
                lw=0,
                s=3,
                **c,
                rasterized=True,
            )
        axis.set_xlabel("time (s)")
        reg_str = "reg " * self.registered
        axis.set_ylabel(reg_str + "z (um)")
        reloc_str = "reloc " * self.relocate_amplitudes
        if not multi_unit:
            plt.colorbar(s, ax=axis, shrink=0.5, label=reloc_str + "amp (su)")


class TFeatScatter(UnitPlot):
    kind = "widescatter"
    width = 1.25
    height = 1.25
    

    def __init__(
        self,
        feat_name,
        color_by_amplitude=True,
        relocate_amplitudes=False,
        amplitude_color_cutoff=15,
        alpha=0.1,
    ):
        super().__init__()
        self.relocate_amplitudes = relocate_amplitudes
        self.feat_name = feat_name
        self.amplitude_color_cutoff = amplitude_color_cutoff
        self.color_by_amplitude = color_by_amplitude
        self.alpha = alpha

    def draw(self, panel, sorting_analysis, unit_id, sorting_previous_step):
        axis = panel.subplots()
        in_unit = sorting_analysis.in_unit(unit_id)
        t = sorting_analysis.times_seconds(which=in_unit)
        feat = sorting_analysis.named_feature(self.feat_name, which=in_unit)
        c = None
        if self.color_by_amplitude:
            amps = sorting_analysis.amplitudes(
                which=in_unit, relocated=self.relocate_amplitudes
            )
            c = np.minimum(amps, self.amplitude_color_cutoff)
        s = axis.scatter(t, feat, c=c, lw=0, s=3, alpha=self.alpha, rasterized=True)
        axis.set_xlabel("time (s)")
        axis.set_ylabel(self.feat_name)
        if self.color_by_amplitude:
            reloc_str = "reloc " * self.relocate_amplitudes
            plt.colorbar(s, ax=axis, shrink=0.5, label=reloc_str + "amp (su)")


class TimeAmpScatter(UnitPlot):
    kind = "widescatter"
    width = 1.25
    height = 1.25


    def __init__(
        self, relocate_amplitudes=False, amplitude_color_cutoff=15, alpha=0.05
    ):
        super().__init__()
        self.relocate_amplitudes = relocate_amplitudes
        self.amplitude_color_cutoff = amplitude_color_cutoff
        self.alpha = alpha

    def draw(self, panel, sorting_analysis, unit_id, axis=None):
        if axis is None:
            axis = panel.subplots()
        unit_id = np.atleast_1d(unit_id)
        multi_unit = unit_id.size > 1

        for uid in unit_id:
            in_unit = sorting_analysis.in_unit(uid)
            t = sorting_analysis.times_seconds(which=in_unit)
            amps = sorting_analysis.amplitudes(
                which=in_unit, relocated=self.relocate_amplitudes
            )
            c = dict(
                color=glasbey1024[uid % len(glasbey1024)] if multi_unit else "k",
                alpha=1 if multi_unit else self.alpha,
            )
            axis.scatter(t, amps, lw=0, s=3, rasterized=True, **c)
        axis.set_xlabel("time (s)")
        reloc_str = "reloc " * self.relocate_amplitudes
        axis.set_ylabel(reloc_str + "amp (su)")


# -- waveform plots


class WaveformPlot(UnitPlot):
    kind = "waveform"
    width = 3
    height = 2
    can_sharey = False

    def __init__(
        self,
        trough_offset_samples=42,
        spike_length_samples=121,
        count=100,
        channel_show_radius_um=50,
        max_n_chan=20,
        relocated=False,
        color="k",
        alpha=0.1,
        show_superres_templates=True,
        superres_template_cmap=plt.cm.jet,
        show_template=True,
        template_color="orange",
        max_abs_template_scale=1.5,
        legend=True,
        template_index=None,
        title=None,
        overtime=False,
    ):
        super().__init__()
        self.count = count
        self.channel_show_radius_um = channel_show_radius_um
        self.max_n_chan = max_n_chan
        self.relocated = relocated
        self.color = color
        self.trough_offset_samples = trough_offset_samples
        self.spike_length_samples = spike_length_samples
        self.alpha = alpha
        self.show_template = show_template
        self.template_color = template_color
        self.show_superres_templates = show_superres_templates
        self.superres_template_cmap = superres_template_cmap
        self.legend = legend
        self.max_abs_template_scale = max_abs_template_scale
        self.template_index = template_index
        self.title = title
        self.overtime = overtime
        if overtime:
            self.show_template=False
            self.show_superres_templates = False

    def get_waveforms(self, sorting_analysis, unit_id):
        raise NotImplementedError

    def draw(self, panel, sorting_analysis, unit_id, axis=None):
        if axis is None:
            axis = panel.subplots()
        which, waveforms, max_chan, geom, ci = self.get_waveforms(
            sorting_analysis, unit_id
        )

        max_abs_amp = None
        show_template = self.show_template
        
        template_color = self.template_color
        if self.template_index is None and show_template:
            templates = sorting_analysis.coarse_template_data.unit_templates(unit_id)
            show_template = bool(templates.size)
        if self.template_index is not None and show_template:
            templates = sorting_analysis.template_data.templates[self.template_index]
            templates = templates[None]
            show_template = bool(templates.size)
            sup_temp_ids = sorting_analysis.unit_template_indices(unit_id)
            template_color = self.superres_template_cmap(
                np.linspace(0, 1, num=sup_temp_ids.size)
            )
            template_color = template_color[sup_temp_ids == self.template_index]
        if show_template:
            templates = trim_waveforms(
                templates,
                old_offset=sorting_analysis.coarse_template_data.trough_offset_samples,
                new_offset=self.trough_offset_samples,
                new_length=self.spike_length_samples,
            )
            max_abs_amp = self.max_abs_template_scale * np.nanmax(np.abs(templates))

        show_superres_templates = (
            self.show_superres_templates and self.template_index is None
        )
        if show_superres_templates:
            suptemplates = sorting_analysis.template_data.unit_templates(unit_id)
            show_superres_templates = bool(suptemplates.size)
        if show_superres_templates:
            suptemplates = trim_waveforms(
                suptemplates,
                old_offset=sorting_analysis.template_data.trough_offset_samples,
                new_offset=self.trough_offset_samples,
                new_length=self.spike_length_samples,
            )
            show_superres_templates = suptemplates.shape[0] > 1
            max_abs_amp = self.max_abs_template_scale * np.nanmax(np.abs(suptemplates))

        handles = {}
        if waveforms is not None and not self.overtime:
            if np.isfinite(waveforms[:, 0, :]).any():
                max_abs_amp = self.max_abs_template_scale * np.nanpercentile(
                    np.abs(waveforms), 99
                )
            ls = geomplot(
                waveforms,
                colors=colors,
                # max_channels=np.full(len(waveforms), max_chan),
                # channel_index=ci,
                geom=geom[showchans],
                ax=axis,
                show_zero=False,
                subar=True,
                msbar=False,
                zlim="tight",
                color=self.color,
                alpha=self.alpha,
                max_abs_amp=max_abs_amp,
                lw=1,
            )
            handles["waveforms"] = ls[0]
        elif waveforms is not None:
            wfs_stacked = np.vstack(waveforms)
            if np.isfinite(wfs_stacked[:, 0, :]).any():
                max_abs_amp = self.max_abs_template_scale *  np.nanpercentile(np.abs(wfs_stacked), 99)

            colors = None
            if sorting_previous_step is not None:
                sorting_analysis, unit_id, sorting_previous_step
                labels_unit_previous = sorting_previous_step.labels[sorting_analysis.sorting.labels==unit_id]
                labels_unit_previous, labels_unit_ordered = np.unique(labels_unit_previous, return_inverse=True)
                if len(labels_unit_previous)>1:
                    colors = self.superres_template_cmap(
                                np.linspace(0, 1, num=labels_unit_ordered.max()+1)[labels_unit_ordered]
                            )

            ls = geomplot_max_chan_overtime(
                waveforms,
                wfs_stacked,
                colors=colors,
                geom=geom,
                ax=axis,
                show_zero=False,
                subar=True,
                msbar=False,
                zlim="tight",
                color=self.color,
                alpha=self.alpha,
                max_abs_amp=max_abs_amp,
                lw=1,
            )
            handles["waveforms"] = ls[0]

        if show_superres_templates:
            showchans = ci[max_chan]
            showchans = showchans[showchans < len(geom)]
            colors = self.superres_template_cmap(
                np.linspace(0, 1, num=suptemplates.shape[0])
            )
            suphandles = []
            for i in range(suptemplates.shape[0]):
                ls = geomplot(
                    suptemplates[i][:, showchans],
                    geom=geom[showchans],
                    ax=axis,
                    show_zero=False,
                    zlim="tight",
                    color=colors[i],
                    alpha=1,
                    max_abs_amp=max_abs_amp,
                    lw=1,
                )
                suphandles.append(ls[0])
            handles["superres templates"] = tuple(suphandles)

        if show_template:
            showchans = ci[max_chan]
            showchans = showchans[showchans < len(geom)]
            ls = geomplot(
                templates[:, :, showchans],
                geom=geom[showchans],
                ax=axis,
                show_zero=False,
                zlim="tight",
                color=template_color,
                alpha=1,
                max_abs_amp=max_abs_amp,
                lw=1,
            )
            handles["mean"] = ls[0]

        reloc_str = "reloc. " * (self.relocated * (not self.overtime))
        shift_str = "shifted " * (sorting_analysis.shifting * (not self.overtime))
        if self.title is None:
            axis.set_title(reloc_str + shift_str + self.wfs_kind)
        else:
            axis.set_title(self.title)
        axis.set_xticks([])
        axis.set_yticks([])

        if self.legend:
            axis.legend(
                handles.values(),
                handles.keys(),
                handler_map={tuple: HandlerTuple(ndivide=None)},
                fancybox=False,
                loc="upper left",
                fontsize=5
            )


class RawWaveformPlot(WaveformPlot):
    wfs_kind = "raw wfs"

    def get_waveforms(self, sorting_analysis, unit_id):
        return sorting_analysis.unit_raw_waveforms(
            unit_id,
            template_index=self.template_index,
            max_n_chan=self.max_n_chan,
            max_count=self.count,
            channel_show_radius_um=self.channel_show_radius_um,
            trough_offset_samples=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
            relocated=self.relocated,
        )


class TPCAWaveformPlot(WaveformPlot):
    wfs_kind = "coll.-cl. tpca wfs"

    def get_waveforms(self, sorting_analysis, unit_id):
        return sorting_analysis.unit_tpca_waveforms(
            unit_id,
            template_index=self.template_index,
            max_n_chan=self.max_n_chan,
            max_count=self.count,
            channel_show_radius_um=self.channel_show_radius_um,
            relocated=self.relocated,
        )

class MaxChanOverTime(WaveformPlot):
    wfs_kind = "Max channel over time"

    def get_waveforms(self, sorting_analysis, unit_id):
        return sorting_analysis.max_chan_waveforms_over_time(
            unit_id,
            template_index=self.template_index,
            max_count=self.count,
            trough_offset_samples=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
        )

# -- merge-focused plots


class NearbyCoarseTemplatesPlot(UnitPlot):
    title = "nearby coarse templates"
    kind = "neighbors"
    width = 3
    height = 2
    can_sharey = False

    def __init__(self, channel_show_radius_um=50, n_neighbors=5, legend=True, max_n_chan=384):
        super().__init__()
        self.channel_show_radius_um = channel_show_radius_um
        self.n_neighbors = n_neighbors
        self.legend = legend
        self.max_n_chan = max_n_chan

    def draw(self, panel, sorting_analysis, unit_id, axis=None):
        if axis is None:
            axis = panel.subplots()
        if np.asarray(unit_id).size > 1:
            unit_id = unit_id[0]
        (
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
        chan = neighbor_coarse_templates[0].ptp(0).argmax()
        
        ci = sorting_analysis.show_channel_index(self.channel_show_radius_um, max_n_chan=self.max_n_chan)
        channels = ci[chan]
        channels = channels[channels<len(sorting_analysis.show_geom)]
        neighbor_coarse_templates = np.pad(
            neighbor_coarse_templates,
            [(0, 0), (0, 0), (0, 1)],
            constant_values=np.nan,
        )
        
        neighbor_coarse_templates = neighbor_coarse_templates[:, :, channels]
        maxamp = np.nanmax(np.abs(neighbor_coarse_templates))

        labels = []
        handles = []
        for uid, color, template in reversed(
            list(zip(neighbor_ids, colors, neighbor_coarse_templates))
        ):
            lines = geomplot(
                template[None],
                # max_channels=[chan],
                # channel_index=ci,
                geom=sorting_analysis.show_geom[channels],
                ax=axis,
                show_zero=False,
                max_abs_amp=maxamp,
                subar=True,
                bar_color="k",
                bar_background="w",
                zlim="tight",
                color=color,
            )
            labels.append(str(uid))
            handles.append(lines[0])
        axis.legend(handles=handles, labels=labels, fancybox=False, loc="lower center", fontsize=5)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(self.title)


class CoarseTemplateDistancePlot(UnitPlot):
    title = "coarse template distance"
    kind = "neighbors"
    width = 3
    height = 1.25

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


    def draw(self, panel, sorting_analysis, unit_id, axis=None):
        if np.asarray(unit_id).size > 1:
            unit_id = unit_id[0]
        if axis is None:
            axis = panel.subplots()
        (
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
            cmap=plt.cm.RdGy,
            origin="lower",
            interpolation="none",
        )
        if self.show_values:
            for (j, i), label in np.ndenumerate(neighbor_dists):
                axis.text(i, j, f"{label:.2f}", ha="center", va="center", fontsize=5)
        # plt.colorbar(im, ax=axis, shrink=0.3)
        axis.set_xticks(range(len(neighbor_ids)), neighbor_ids, fontsize=5)
        axis.set_yticks(range(len(neighbor_ids)), neighbor_ids, fontsize=5)
        for i, (tx, ty) in enumerate(
            zip(axis.xaxis.get_ticklabels(), axis.yaxis.get_ticklabels())
        ):
            tx.set_color(colors[i])
            ty.set_color(colors[i])
        axis.set_title(self.title, fontsize=8)        
        axis.xaxis.set_tick_params(labelsize=5)
        axis.yaxis.set_tick_params(labelsize=5)



class NeighborCCGPlot(UnitPlot):
    kind = "neighbors"
    width = 3
    height = 0.75

    def __init__(self, n_neighbors=3, max_lag=50):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.max_lag = max_lag

    def draw(self, panel, sorting_analysis, unit_id, sorting_previous_step):
        (
            neighbor_ids,
            neighbor_dists,
            neighbor_coarse_templates,
        ) = sorting_analysis.nearby_coarse_templates(
            unit_id, n_neighbors=self.n_neighbors + 1
        )
        # assert neighbor_ids[0] == unit_id
        neighbor_ids = neighbor_ids[1:]
        colors = np.array(glasbey1024)[neighbor_ids % len(glasbey1024)]

        my_st = sorting_analysis.times_samples(which=sorting_analysis.in_unit(unit_id))
        neighb_sts = [
            sorting_analysis.times_samples(which=sorting_analysis.in_unit(nid))
            for nid in neighbor_ids
        ]

        axes = panel.subplots(nrows=2, sharey="row", sharex=True, squeeze=False, ncols=len(neighb_sts))
        for j in range(len(neighb_sts)):
            clags, ccg = correlogram(my_st, neighb_sts[j], max_lag=self.max_lag)
            merged_st = np.concatenate((my_st, neighb_sts[j]))
            merged_st.sort()
            alags, acg = correlogram(merged_st, max_lag=self.max_lag)

            bar(axes[0, j], clags, ccg, fill=True, fc=colors[j])  # , ec="k", lw=1)
            bar(axes[1, j], alags, acg, fill=True, fc=colors[j])  # , ec="k", lw=1)
            axes[0, j].set_title(f"unit {neighbor_ids[j]}")
        axes[0, 0].set_ylabel("ccg")
        axes[1, 0].set_ylabel("merged acg")
        axes[1, len(neighb_sts) // 2].set_xlabel("lag (samples)")


# -- evaluation plots


class SplitStrategyPlot(UnitPlot):
    kind = "spliteval"
    width = 5.5
    height = 4

    def __init__(
        self,
        split_name,
        split_strategy,
        peeling_hdf5_filename,
        recording,
        channel_show_radius_um=50.0,
        amplitude_color_cutoff=15.0,
        pca_radius_um=75.0,
        split_strategy_kwargs=None,
        motion_est=None,
    ):
        super().__init__()
        if split_strategy_kwargs is None:
            split_strategy_kwargs = {}
        if motion_est is not None:
            split_strategy_kwargs["motion_est"] = motion_est
        split_strategy_kwargs["peeling_hdf5_filename"] = peeling_hdf5_filename

        self.split_name = split_name
        print(split_name)
        self.split_strategy = split_strategy
        self.split_strategy_kwargs = split_strategy_kwargs
        self.motion_est = motion_est
        self.recording = recording
        self.channel_show_radius_um = channel_show_radius_um
        self.amplitude_color_cutoff = amplitude_color_cutoff
        self.pca_radius_um = pca_radius_um

        self.no_split_plots = [
            XZScatter(),
            TimeAmpScatter(),
            TimeAmpScatter(relocate_amplitudes=True),
            PCAScatter(),
            TimeZScatter(),
        ]

        self.plots = [
            XZScatter(),
            TimeAmpScatter(),
            TimeAmpScatter(relocate_amplitudes=True),
            PCAScatter(),
            TimeZScatter(),
            # may need to update n neighbors for these
            NearbyCoarseTemplatesPlot(),
            CoarseTemplateDistancePlot(),
        ]

    def draw(self, panel, sorting_analysis, unit_id, sorting_previous_step):
        """
        - x vs reg z
        - time vs amp
        - time vs reloc amp
        - reloc PC scatter
        - time vs z
        - time vs z reg?
        - templates
        - template dist
        """
        # run the split
        print("split", self.split_name, self.split_strategy_kwargs)
        split_strategy = split.split_strategies_by_class_name[self.split_strategy](
            **self.split_strategy_kwargs
        )
        in_unit = sorting_analysis.in_unit(unit_id)
        split_result = split_strategy.split_cluster(in_unit)
        print("draw", self.split_name)

        # re-make the sorting analysis
        split_labels = np.full_like(sorting_analysis.sorting.labels, -1)
        if split_result.is_split:
            split_labels[in_unit] = split_result.new_labels
            unit_ids, counts = np.unique(split_result.new_labels, return_counts=True)
            counts = counts[unit_ids >= 0]
            unit_ids = unit_ids[unit_ids >= 0]
            order = np.argsort(counts)[::-1]
            counts = counts[order]
            unit_ids = unit_ids[order]
            counts = list(map(str, counts))
            print(f"{unit_ids=}")
            print(f"{counts=}")
        else:
            split_labels[in_unit] = 0
            unit_ids = 0
            counts = [str(in_unit.size)]
        split_sorting = replace(sorting_analysis.sorting, labels=split_labels)
        split_sorting_analysis = DARTsortAnalysis.from_sorting(
            self.recording,
            split_sorting,
            motion_est=self.motion_est,
            name=f"{self.split_name} {unit_id}",
            template_config=raw_template_config,
            allow_template_reload=False,
            n_jobs_templates=0,
        )

        # make a new flow layout here
        make_unit_summary(
            split_sorting_analysis,
            unit_id=unit_ids,
            channel_show_radius_um=self.channel_show_radius_um,
            amplitude_color_cutoff=self.amplitude_color_cutoff,
            pca_radius_um=self.pca_radius_um,
            plots=self.plots if split_result.is_split else self.no_split_plots,
            max_height=self.height,
            figsize=(self.width, self.height),
            figure=panel,
            hspace=0.0,
        )
        desc = f"not split. {in_unit.size} total spikes."
        if split_result.is_split:
            cs = ", ".join(counts)
            desc = f"split into {unit_ids.size} units with counts:\n{cs}"
        panel.suptitle(f"{self.split_name}, unit {unit_id}. {desc}")


# -- multi plots
# these have multiple plots per unit, and we don't know in advance how many
# for instance, making separate plots of spikes belonging to each superres template


class SuperresWaveformMultiPlot(UnitMultiPlot):
    def __init__(
        self,
        kind="raw",
        trough_offset_samples=42,
        spike_length_samples=121,
        count=250,
        channel_show_radius_um=50,
        relocated=False,
        color="k",
        alpha=0.1,
        show_superres_templates=True,
        superres_template_cmap=plt.cm.jet,
        show_template=True,
        template_color="orange",
        max_abs_template_scale=1.35,
        legend=True,
    ):
        super().__init__()
        self.kind = kind
        self.count = count
        self.channel_show_radius_um = channel_show_radius_um
        self.relocated = relocated
        self.color = color
        self.trough_offset_samples = trough_offset_samples
        self.spike_length_samples = spike_length_samples
        self.alpha = alpha
        self.show_template = show_template
        self.template_color = template_color
        self.show_superres_templates = show_superres_templates
        self.superres_template_cmap = superres_template_cmap
        self.legend = legend
        self.max_abs_template_scale = max_abs_template_scale

    def plots(self, sorting_analysis, unit_id):
        if self.kind == "raw":
            plot_cls = RawWaveformPlot
        elif self.kind == "tpca":
            plot_cls = TPCAWaveformPlot
        else:
            assert False

        return [
            plot_cls(
                count=self.count,
                channel_show_radius_um=self.channel_show_radius_um,
                relocated=self.relocated,
                color=self.color,
                trough_offset_samples=self.trough_offset_samples,
                spike_length_samples=self.spike_length_samples,
                alpha=self.alpha,
                show_template=self.show_template,
                template_color=self.template_color,
                show_superres_templates=self.show_superres_templates,
                superres_template_cmap=self.superres_template_cmap,
                legend=self.legend,
                max_abs_template_scale=self.max_abs_template_scale,
                template_index=template_index,
                title=f"{sorting_analysis.template_data.spike_counts[template_index]} spikes assigned",
            )
            for template_index in sorting_analysis.unit_template_indices(unit_id)
        ]


# -- main routines

default_plots = (
    UnitTextInfo(),
    ACG(),
    ISIHistogram(),
    XZScatter(),
    PCAScatter(),
    TimeZScatter(),
    TimeZScatter(registered=False),
    TimeAmpScatter(),
    TimeAmpScatter(relocate_amplitudes=True),
    RawWaveformPlot(),
    TPCAWaveformPlot(relocated=True),
    NearbyCoarseTemplatesPlot(max_n_chan=20),
    CoarseTemplateDistancePlot(),
    NeighborCCGPlot(),
    MaxChanOverTime(overtime=True),
)


template_assignment_plots = (
    UnitTextInfo(),
    RawWaveformPlot(),
    SuperresWaveformMultiPlot(),
)


def make_unit_summary(
    sorting_analysis,
    unit_id,
    sorting_previous_step=None,
    channel_show_radius_um=50.0,
    amplitude_color_cutoff=15.0,
    pca_radius_um=75.0,
    plots=default_plots,
    max_height=4,
    figsize=(11, 8.5),
    hspace=0.1,
    figure=None,
    **other_global_params,
):
    # notify plots of global params
    for p in plots:
        p.notify_global_params(
            channel_show_radius_um=channel_show_radius_um,
            amplitude_color_cutoff=amplitude_color_cutoff,
            pca_radius_um=pca_radius_um,
            **other_global_params,
        )

    figure = layout.flow_layout(
        plots,
        max_height=max_height,
        figsize=figsize,
        hspace=hspace,
        figure=figure,
        sorting_analysis=sorting_analysis,
        unit_id=unit_id,
        sorting_previous_step=sorting_previous_step,
    )

    return figure


def make_all_summaries(
    sorting_analysis,
    save_folder,
    plots=default_plots,
    sorting_previous_step=None,
    channel_show_radius_um=50.0,
    amplitude_color_cutoff=15.0,
    pca_radius_um=75.0,
    max_height=4,
    figsize=(16, 8.5),
    hspace=0.1,
    dpi=200,
    image_ext="png",
    n_jobs=0,
    show_progress=True,
    overwrite=False,
    unit_ids=None,
    **other_global_params,
):
    save_folder = Path(save_folder)
    if unit_ids is None:
        unit_ids = sorting_analysis.unit_ids
    if not overwrite and all_summaries_done(
        unit_ids, save_folder, ext=image_ext
    ):
        return

    save_folder.mkdir(exist_ok=True)

    global_params = dict(
        channel_show_radius_um=channel_show_radius_um,
        amplitude_color_cutoff=amplitude_color_cutoff,
        pca_radius_um=pca_radius_um,
        **other_global_params,
    )

    n_jobs, Executor, context = get_pool(n_jobs, cls=CloudpicklePoolExecutor)
    from cloudpickle import dumps

    initargs = (
        sorting_analysis,
        plots,
        max_height,
        figsize,
        hspace,
        dpi,
        save_folder,
        image_ext,
        overwrite,
        global_params,
    )
    with Executor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=_summary_init,
        initargs=(dumps(initargs),),
    ) as pool:
        results = pool.map(_summary_job, unit_ids)
        if show_progress:
            results = tqdm(
                results,
                desc="Unit summaries",
                smoothing=0,
                total=len(unit_ids),
            )
        for res in results:
            pass


# -- utilities


def correlogram(times_a, times_b=None, max_lag=50):
    lags = np.arange(-max_lag, max_lag + 1)
    ccg = np.zeros(len(lags), dtype=int)

    times_a = np.sort(times_a)
    auto = times_b is None
    if auto:
        times_b = times_a
    else:
        times_b = np.sort(times_b)

    for i, lag in enumerate(lags):
        lagged_b = times_b + lag
        insertion_inds = np.searchsorted(times_a, lagged_b)
        found = insertion_inds < len(times_a)
        ccg[i] = np.sum(times_a[insertion_inds[found]] == lagged_b[found])

    if auto:
        ccg[lags == 0] = 0

    return lags, ccg


def trim_waveforms(waveforms, old_offset=42, new_offset=42, new_length=121):
    if waveforms.shape[1] == new_length and old_offset == new_offset:
        return waveforms

    start = old_offset - new_offset
    end = start + new_length
    return waveforms[:, start:end]


def all_summaries_done(unit_ids, save_folder, ext="png"):
    return save_folder.exists() and all(
        (save_folder / f"unit{unit_id:04d}.{ext}").exists() for unit_id in unit_ids
    )


# -- parallelism helpers


class SummaryJobContext:
    def __init__(
        self,
        sorting_analysis,
        sorting_previous_step,
        plots,
        max_height,
        figsize,
        hspace,
        dpi,
        save_folder,
        image_ext,
        overwrite,
        global_params,
    ):
        self.sorting_analysis = sorting_analysis
        self.plots = plots
        self.max_height = max_height
        self.figsize = figsize
        self.hspace = hspace
        self.dpi = dpi
        self.save_folder = save_folder
        self.image_ext = image_ext
        self.overwrite = overwrite
        self.global_params = global_params


_summary_job_context = None


def _summary_init(args):
    global _summary_job_context
    from cloudpickle import loads

    args = loads(args)
    _summary_job_context = SummaryJobContext(*args)


def _summary_job(unit_id):
    # handle resuming/overwriting
    ext = _summary_job_context.image_ext
    tmp_out = _summary_job_context.save_folder / f"tmp_unit{unit_id:04d}.{ext}"
    final_out = _summary_job_context.save_folder / f"unit{unit_id:04d}.{ext}"
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
    make_unit_summary(
        _summary_job_context.sorting_analysis,
        unit_id,
        hspace=_summary_job_context.hspace,
        plots=_summary_job_context.plots,
        max_height=_summary_job_context.max_height,
        figsize=_summary_job_context.figsize,
        figure=fig,
        **_summary_job_context.global_params,
    )

    # the save is done sort of atomically to help with the resuming and avoid
    # half-baked image files
    fig.savefig(tmp_out, dpi=_summary_job_context.dpi)
    tmp_out.rename(final_out)
    plt.close(fig)


def bar(ax, x, y, **kwargs):
    dx = np.diff(x).min()
    x0 = np.concatenate((x - dx, x[-1:] + dx))
    ax.stairs(y, x0, **kwargs)
