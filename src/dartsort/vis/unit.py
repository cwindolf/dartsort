"""Toolkit for extensible single unit summary plots

The goal is to make it easy to add and remove plots without the plotting
code turning into a web of if statements, for loops, and bizarre subplot
and subfigure mazes.

Relies on the DARTsortAnalysis object of utils/analysis.py to do most of
the data work so that this file can focus on plotting (sort of MVC).
"""
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.figure import Figure
import numpy as np
from tqdm.auto import tqdm

from ..util.multiprocessing_util import get_pool
from .waveforms import geomplot

# -- main class. see fn make_unit_summary below to make lots of UnitPlots.


class UnitPlot:
    kind: str
    width = 1
    height = 1

    def draw(self, axis, sorting_analysis, unit_id):
        raise NotImplementedError


class TextInfo(UnitPlot):
    kind = "text"
    height = 0.5

    def draw(self, axis, sorting_analysis, unit_id):
        axis.axis("off")
        msg = f"unit {unit_id}\n"

        msg += f"feature source: {sorting_analysis.hdf5_path.name}\n"

        nspikes = sorting_analysis.spike_counts[
            sorting_analysis.unit_ids == unit_id
        ].sum()
        msg += f"n spikes: {nspikes}\n"
        axis.text(0, 0, msg, fontsize=6.5)

        temps = sorting_analysis.template_data.unit_templates(unit_id)
        if temps.size:
            ptp = temps.ptp(1).max(1).mean()
            msg += f"mean superres maxptp: {ptp:0.1f}su\n"
        else:
            msg += "no template (too few spikes)"


# -- small summary plots


class ACG(UnitPlot):
    kind = "histogram"
    height = 0.75

    def __init__(self, max_lag=50):
        self.max_lag = max_lag

    def draw(self, axis, sorting_analysis, unit_id):
        times_samples = sorting_analysis.times_samples(
            which=sorting_analysis.in_unit(unit_id)
        )
        lags, acg = correlogram(times_samples, max_lag=self.max_lag)
        axis.bar(lags, acg)
        axis.set_xlabel("lag (samples)")
        axis.set_ylabel("acg")


class ISIHistogram(UnitPlot):
    kind = "histogram"
    height = 0.75

    def __init__(self, bin_ms=0.1, max_ms=5):
        self.bin_ms = bin_ms
        self.max_ms = max_ms

    def draw(self, axis, sorting_analysis, unit_id):
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
        plt.hist(dt_ms, bin_edges)
        axis.set_xlabel("isi (ms)")
        axis.set_ylabel(f"count (out of {dt_ms.size} total isis)")


class XZScatter(UnitPlot):
    kind = "scatter"

    def __init__(
        self,
        relocate_amplitudes=False,
        registered=True,
        max_amplitude=15,
        probe_margin_um=100,
    ):
        self.relocate_amplitudes = relocate_amplitudes
        self.registered = registered
        self.max_amplitude = max_amplitude
        self.probe_margin_um = probe_margin_um

    def draw(self, axis, sorting_analysis, unit_id):
        in_unit = sorting_analysis.in_unit(unit_id)
        x = sorting_analysis.x(which=in_unit)
        z = sorting_analysis.z(which=in_unit, registered=self.registered)
        geomx, geomz = sorting_analysis.geom.T
        pad = self.probe_margin_um
        valid = x == np.clip(x, geomx.min() - pad, geomx.max() + pad)
        valid &= z == np.clip(z, geomz.min() - pad, geomz.max() + pad)
        amps = sorting_analysis.amplitudes(
            which=in_unit[valid], relocated=self.relocate_amplitudes
        )
        s = axis.scatter(
            x[valid], z[valid], c=np.minimum(amps, self.max_amplitude), lw=0, s=3
        )
        axis.set_xlabel("x (um)")
        reg_str = "registered " * self.registered
        axis.set_ylabel(reg_str + "z (um)")
        reloc_str = "relocated " * self.relocate_amplitudes
        plt.colorbar(s, ax=axis, shrink=0.5, label=reloc_str + "amplitude (su)")


class PCAScatter(UnitPlot):
    kind = "scatter"

    def __init__(self, relocate_amplitudes=False, relocated=True, max_amplitude=15):
        self.relocated = relocated
        self.relocate_amplitudes = relocate_amplitudes
        self.max_amplitude = max_amplitude

    def draw(self, axis, sorting_analysis, unit_id):
        in_unit = sorting_analysis.in_unit(unit_id)
        loadings = sorting_analysis.unit_pca_features(
            unit_id=unit_id, relocated=self.relocated
        )
        amps = sorting_analysis.amplitudes(
            which=in_unit, relocated=self.relocate_amplitudes
        )
        s = axis.scatter(*loadings.T, c=np.minimum(amps, self.max_amplitude), lw=0, s=3)
        reloc_str = "relocated " * self.relocated
        axis.set_xlabel(reloc_str + "per-unit PC1 (um)")
        axis.set_ylabel(reloc_str + "per-unit PC2 (um)")
        reloc_amp_str = "relocated " * self.relocate_amplitudes
        plt.colorbar(s, ax=axis, shrink=0.5, label=reloc_amp_str + "amplitude (su)")


# -- wide scatter plots


class TZScatter(UnitPlot):
    kind = "widescatter"
    width = 2

    def __init__(
        self,
        relocate_amplitudes=False,
        registered=True,
        max_amplitude=15,
        probe_margin_um=100,
    ):
        self.relocate_amplitudes = relocate_amplitudes
        self.registered = registered
        self.max_amplitude = max_amplitude
        self.probe_margin_um = probe_margin_um

    def draw(self, axis, sorting_analysis, unit_id):
        in_unit = sorting_analysis.in_unit(unit_id)
        t = sorting_analysis.times_seconds(which=in_unit)
        z = sorting_analysis.z(which=in_unit, registered=self.registered)
        geomx, geomz = sorting_analysis.geom.T
        pad = self.probe_margin_um
        valid = z == np.clip(z, geomz.min() - pad, geomz.max() + pad)
        amps = sorting_analysis.amplitudes(
            which=in_unit[valid], relocated=self.relocate_amplitudes
        )
        s = axis.scatter(
            t[valid], z[valid], c=np.minimum(amps, self.max_amplitude), lw=0, s=3
        )
        axis.set_xlabel("time (seconds)")
        reg_str = "registered " * self.registered
        axis.set_ylabel(reg_str + "z (um)")
        reloc_str = "relocated " * self.relocate_amplitudes
        plt.colorbar(s, ax=axis, shrink=0.5, label=reloc_str + "amplitude (su)")


class TFeatScatter(UnitPlot):
    kind = "widescatter"
    width = 2

    def __init__(
        self,
        feat_name,
        color_by_amplitude=True,
        relocate_amplitudes=False,
        max_amplitude=15,
    ):
        self.relocate_amplitudes = relocate_amplitudes
        self.feat_name = feat_name
        self.max_amplitude = max_amplitude
        self.color_by_amplitude = color_by_amplitude

    def draw(self, axis, sorting_analysis, unit_id):
        in_unit = sorting_analysis.in_unit(unit_id)
        t = sorting_analysis.times_seconds(which=in_unit)
        feat = sorting_analysis.named_feature(self.feat_name, which=in_unit)
        c = None
        if self.color_by_amplitude:
            amps = sorting_analysis.amplitudes(
                which=in_unit, relocated=self.relocate_amplitudes
            )
            c = np.minimum(amps, self.max_amplitude)
        s = axis.scatter(t, feat, c=c, lw=0, s=3)
        axis.set_xlabel("time (seconds)")
        axis.set_ylabel(self.feat_name)
        if self.color_by_amplitude:
            reloc_str = "relocated " * self.relocate_amplitudes
            plt.colorbar(s, ax=axis, shrink=0.5, label=reloc_str + "amplitude (su)")


class TAmpScatter(UnitPlot):
    kind = "widescatter"
    width = 2

    def __init__(self, relocate_amplitudes=False, max_amplitude=15):
        self.relocate_amplitudes = relocate_amplitudes
        self.max_amplitude = max_amplitude

    def draw(self, axis, sorting_analysis, unit_id):
        in_unit = sorting_analysis.in_unit(unit_id)
        t = sorting_analysis.times_seconds(which=in_unit)
        amps = sorting_analysis.amplitudes(
            which=in_unit, relocated=self.relocate_amplitudes
        )
        axis.scatter(t, amps, c="k", lw=0, s=3)
        axis.set_xlabel("time (seconds)")
        reloc_str = "relocated " * self.relocate_amplitudes
        axis.set_ylabel(reloc_str + "amplitude (su)")


# -- waveform plots


class WaveformPlot(UnitPlot):
    kind = "waveform"
    width = 2
    height = 2
    title = "waveforms"

    def __init__(
        self,
        trough_offset_samples=42,
        spike_length_samples=121,
        count=250,
        show_radius_um=50,
        relocated=False,
        color="k",
        alpha=0.1,
        show_superres_templates=True,
        superres_template_cmap=plt.cm.winter,
        show_template=True,
        template_color="orange",
        max_abs_template_scale=1.35,
        legend=True,
    ):
        self.count = count
        self.show_radius_um = show_radius_um
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

    def get_waveforms(self, sorting_analysis, unit_id):
        raise NotImplementedError

    def draw(self, axis, sorting_analysis, unit_id):
        waveforms, max_chan, geom, ci = self.get_waveforms(sorting_analysis, unit_id)

        max_abs_amp = None
        show_template = self.show_template
        if show_template:
            templates = sorting_analysis.coarse_template_data.unit_templates(unit_id)
            show_template = bool(templates.size)
        if show_template:
            templates = trim_waveforms(
                templates,
                old_offset=sorting_analysis.coarse_template_data.trough_offset_samples,
                new_offset=self.trough_offset_samples,
                new_length=self.spike_length_samples,
            )
            max_abs_amp = self.max_abs_template_scale * np.abs(templates).max()
        show_superres_templates = self.show_superres_templates
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
            max_abs_amp = self.max_abs_template_scale * np.abs(suptemplates).max()

        ls = geomplot(
            waveforms,
            max_channels=np.full(len(waveforms), max_chan),
            channel_index=ci,
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
        handles = [ls[0]]
        labels = ["waveforms"]

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
            handles.append(tuple(suphandles))
            labels.append("superres templates")

        if show_template:
            showchans = ci[max_chan]
            showchans = showchans[showchans < len(geom)]
            ls = geomplot(
                templates[:, :, showchans],
                geom=geom[showchans],
                ax=axis,
                show_zero=False,
                zlim="tight",
                color=self.template_color,
                alpha=1,
                max_abs_amp=max_abs_amp,
                lw=1,
            )
            handles.append(ls[0])
            labels.append("mean of superres templates")

        reloc_str = "relocated " * self.relocated
        shift_str = "shifted " * sorting_analysis.shifting
        axis.set_title(reloc_str + shift_str + self.title)
        reg_str = "registered " * sorting_analysis.shifting
        axis.set_ylabel(reg_str + "depth (um)")
        axis.set_xticks([])

        if self.legend:
            axis.legend(
                handles,
                labels,
                handler_map={tuple: HandlerTuple(ndivide=None)},
                fancybox=False,
                loc="upper left",
            )


class RawWaveformPlot(WaveformPlot):
    title = "raw waveforms"

    def get_waveforms(self, sorting_analysis, unit_id):
        return sorting_analysis.unit_raw_waveforms(
            unit_id,
            max_count=self.count,
            show_radius_um=self.show_radius_um,
            trough_offset_samples=self.trough_offset_samples,
            spike_length_samples=self.spike_length_samples,
            relocated=self.relocated,
        )


class TPCAWaveformPlot(WaveformPlot):
    title = "collision-cleaned tpca waveforms"

    def get_waveforms(self, sorting_analysis, unit_id):
        return sorting_analysis.unit_tpca_waveforms(
            unit_id,
            max_count=self.count,
            show_radius_um=self.show_radius_um,
            relocated=self.relocated,
        )


# -- main routines

default_plots = (
    TextInfo(),
    ACG(),
    ISIHistogram(),
    XZScatter(),
    PCAScatter(),
    TZScatter(),
    TZScatter(registered=False),
    TAmpScatter(),
    TAmpScatter(relocate_amplitudes=True),
    RawWaveformPlot(),
    TPCAWaveformPlot(relocated=True),
)


def make_unit_summary(
    sorting_analysis,
    unit_id,
    plots=default_plots,
    max_height=4,
    figsize=(11, 8.5),
    figure=None,
):
    # -- lay out the figure
    columns = summary_layout(plots, max_height=max_height)

    # -- draw the figure
    width_ratios = [column[0].width for column in columns]
    if figure is None:
        figure = plt.figure(figsize=figsize, layout="constrained")
    subfigures = figure.subfigures(
        nrows=1, ncols=len(columns), hspace=0.1, width_ratios=width_ratios
    )
    all_panels = subfigures.tolist()
    for column, subfig in zip(columns, subfigures):
        n_cards = len(column)
        height_ratios = [card.height for card in column]
        remaining_height = max_height - sum(height_ratios)
        if remaining_height > 0:
            height_ratios.append(remaining_height)

        cardfigs = subfig.subfigures(
            nrows=n_cards + (remaining_height > 0), ncols=1, height_ratios=height_ratios
        )
        cardfigs = np.atleast_1d(cardfigs)
        all_panels.extend(cardfigs)

        for cardfig, card in zip(cardfigs, column):
            axes = cardfig.subplots(nrows=len(card.plots), ncols=1)
            axes = np.atleast_1d(axes)
            for plot, axis in zip(card.plots, axes):
                plot.draw(axis, sorting_analysis, unit_id)

    # clean up the panels, or else things get clipped
    for panel in all_panels:
        panel.set_facecolor([0, 0, 0, 0])
        panel.patch.set_facecolor([0, 0, 0, 0])

    return figure


def make_all_summaries(
    sorting_analysis,
    save_folder,
    plots=default_plots,
    max_height=4,
    figsize=(11, 8.5),
    dpi=200,
    image_ext="png",
    n_jobs=0,
    show_progress=True,
    overwrite=False,
):
    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True)

    n_jobs, Executor, context = get_pool(n_jobs)
    with Executor(
        max_workers=n_jobs,
        mp_context=context,
        initializer=_summary_init,
        initargs=(
            sorting_analysis,
            plots,
            max_height,
            figsize,
            dpi,
            save_folder,
            image_ext,
            overwrite,
        ),
    ) as pool:
        jobs = sorting_analysis.unit_ids
        results = pool.map(_summary_job, jobs)
        if show_progress:
            results = tqdm(
                results,
                desc="Unit summaries",
                smoothing=0,
                total=len(jobs),
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


# -- plotting helpers


Card = namedtuple("Card", ["kind", "width", "height", "plots"])


def summary_layout(plots, max_height=4):
    plots_by_kind = {}
    for plot in plots:
        if plot.kind not in plots_by_kind:
            plots_by_kind[plot.kind] = []
        plots_by_kind[plot.kind].append(plot)

    # break plots into groups ("cards") by kind
    cards = []
    for kind, plots in plots_by_kind.items():
        width = max(p.width for p in plots)
        card_plots = []
        for plot in plots:
            if sum(p.height for p in card_plots) + plot.height <= max_height:
                card_plots.append(plot)
            else:
                cards.append(
                    Card(
                        plots[0].kind,
                        width,
                        sum(p.height for p in card_plots),
                        card_plots,
                    )
                )
                card_plots = []
        if card_plots:
            cards.append(
                Card(
                    plots[0].kind, width, sum(p.height for p in card_plots), card_plots
                )
            )
    cards = sorted(cards, key=lambda card: card.width)

    # flow the same-width cards over columns
    columns = [[]]
    cur_width = cards[0].width
    for card in cards:
        if card.width != cur_width:
            columns.append([card])
            cur_width = card.width
            continue

        if sum(c.height for c in columns[-1]) + card.height <= max_height:
            columns[-1].append(card)
        else:
            columns.append([card])

    return columns


# -- parallelism helpers


class SummaryJobContext:
    def __init__(
        self,
        sorting_analysis,
        plots,
        max_height,
        figsize,
        dpi,
        save_folder,
        image_ext,
        overwrite,
    ):
        self.sorting_analysis = sorting_analysis
        self.plots = plots
        self.max_height = max_height
        self.figsize = figsize
        self.dpi = dpi
        self.save_folder = save_folder
        self.image_ext = image_ext
        self.overwrite = overwrite


_summary_job_context = None


def _summary_init(*args):
    global _summary_job_context
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
        plots=_summary_job_context.plots,
        max_height=_summary_job_context.max_height,
        figsize=_summary_job_context.figsize,
        figure=fig,
    )

    # the save is done sort of atomically to help with the resuming and avoid
    # half-baked image files
    fig.savefig(tmp_out, dpi=_summary_job_context.dpi)
    tmp_out.rename(final_out)
    plt.close(fig)