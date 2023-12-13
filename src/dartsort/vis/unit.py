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
import numpy as np
from tqdm.auto import tqdm

from ..multiprocessing_util import get_pool
from .waveforms import geomplot

# -- main class. see fn make_unit_summary below to make lots of UnitPlots.


class UnitPlot:
    kind: str
    width = 1
    height = 1

    def draw(self, axis, sorting_analysis, unit_id):
        raise NotImplementedError


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
        axis.bar(lags[:-1], acg)
        axis.set_xlabel("lag (samples)")
        axis.set_ylabel("acg")


class ISIHistogram(UnitPlot):
    kind = "histogram"
    height = 0.75

    def __init__(self, bin_ms=0.1):
        self.bin_ms = bin_ms

    def draw(self, axis, sorting_analysis, unit_id):
        times_ms = (
            sorting_analysis.times_seconds(which=sorting_analysis.in_unit(unit_id))
            / 1000
        )
        bin_edges = np.arange(
            np.floor(times_ms.min()),
            np.floor(times_ms.max()),
            self.bin_ms,
        )
        axis.hist(times_ms, bins=bin_edges)
        axis.set_xlabel("isi (ms)")
        axis.set_ylabel("count")


class XZScatter(UnitPlot):
    kind = "scatter"

    def __init__(self, relocate_amplitudes=False, registered=True, max_amplitude=15):
        self.relocate_amplitudes = relocate_amplitudes
        self.registered = registered
        self.max_amplitude = max_amplitude

    def draw(self, axis, sorting_analysis, unit_id):
        in_unit = sorting_analysis.in_unit(unit_id)
        x = sorting_analysis.x(which=in_unit)
        z = sorting_analysis.z(which=in_unit, registered=self.registered)
        amps = sorting_analysis.amplitudes(
            which=in_unit, relocated=self.relocate_amplitudes
        )
        s = axis.scatter(x, z, c=np.minimum(amps, self.max_amplitude), lw=0, s=3)
        self.set_xlabel("x (um)")
        reg_str = "registered " * self.registered
        self.set_ylabel(reg_str + "z (um)")
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
        loadings = sorting_analysis.pca_features(
            which=in_unit, relocated=self.relocated
        )
        amps = sorting_analysis.amplitudes(
            which=in_unit, relocated=self.relocate_amplitudes
        )
        s = axis.scatter(*loadings.T, c=np.minimum(amps, self.max_amplitude), lw=0, s=3)
        reloc_str = "relocated " * self.relocated
        self.set_xlabel(reloc_str + "per-unit PC1 (um)")
        self.set_ylabel(reloc_str + "per-unit PC2 (um)")
        reloc_amp_str = "relocated " * self.relocate_amplitudes
        plt.colorbar(s, ax=axis, shrink=0.5, label=reloc_amp_str + "amplitude (su)")


# -- wide scatter plots


class TZScatter(UnitPlot):
    kind = "widescatter"
    width = 2

    def __init__(self, relocate_amplitudes=False, registered=True, max_amplitude=15):
        self.relocate_amplitudes = relocate_amplitudes
        self.registered = registered
        self.max_amplitude = max_amplitude

    def draw(self, axis, sorting_analysis, unit_id):
        in_unit = sorting_analysis.in_unit(unit_id)
        t = sorting_analysis.times_seconds(which=in_unit)
        z = sorting_analysis.z(which=in_unit, registered=self.registered)
        amps = sorting_analysis.amplitudes(
            which=in_unit, relocated=self.relocate_amplitudes
        )
        s = axis.scatter(t, z, c=np.minimum(amps, self.max_amplitude), lw=0, s=3)
        self.set_xlabel("time (seconds)")
        reg_str = "registered " * self.registered
        self.set_ylabel(reg_str + "z (um)")
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
        z = sorting_analysis.named_feature(self.feat_name, which=in_unit)
        c = None
        if self.color_by_amplitude:
            amps = sorting_analysis.amplitudes(
                which=in_unit, relocated=self.relocate_amplitudes
            )
            c = np.minimum(amps, self.max_amplitude)
        s = axis.scatter(t, z, c=c, lw=0, s=3)
        self.set_xlabel("time (seconds)")
        self.set_ylabel(self.feat_name)
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
        self.set_xlabel("time (seconds)")
        reloc_str = "relocated " * self.relocate_amplitudes
        self.set_ylabel(reloc_str + "amplitude (su)")


# -- waveform plots


class WaveformPlot(UnitPlot):
    kind = "waveform"
    width = 2
    height = 2

    def __init__(
        self,
        trough_offset_samples=42,
        spike_length_samples=121,
        count=250,
        show_radius_um=75,
        relocated=False,
        color="k",
    ):
        self.count = count
        self.show_radius_um = show_radius_um
        self.relocated = relocated
        self.color = color
        self.trough_offset_samples = trough_offset_samples
        self.spike_length_samples = spike_length_samples

    def get_waveforms(self, sorting_analysis, unit_id):
        raise NotImplementedError

    def draw(self, axis, sorting_analysis, unit_id):
        waveforms, max_chan, geom, ci = self.get_waveforms(sorting_analysis, unit_id)
        geomplot(
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
        )


class RawWaveformPlot(WaveformPlot):
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
    def get_waveforms(self, sorting_analysis, unit_id):
        return sorting_analysis.unit_tpca_waveforms(
            unit_id,
            max_count=self.count,
            show_radius_um=self.show_radius_um,
            relocated=self.relocated,
        )


# -- main routines

default_plots = (
    ACG(),
    ISIHistogram(),
    XZScatter(),
    PCAScatter(),
    TZScatter(),
    TZScatter(registered=False),
    TAmpScatter(),
    TAmpScatter(relocate_amplitudes=True),
    RawWaveformPlot(),
    TPCAWaveformPlot(),
)


def make_unit_summary(
    sorting_analysis,
    unit_id,
    plots=default_plots,
    max_height=4,
    figsize=(11, 8.5),
):
    # -- lay out the figure
    columns = summary_layout(plots, max_height=max_height)

    # -- draw the figure
    width_ratios = [column[0].width for column in columns]
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
            height_ratios.append([remaining_height])

        cardfigs = subfig.subfigures(
            nrows=n_cards + (remaining_height > 0), ncols=1, height_ratios=height_ratios
        )
        all_panels.extend(cardfigs)

        for cardfig, card in zip(cardfigs, column):
            axes = cardfig.subplots(nrows=len(card.plots), ncols=1)
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
        ),
    ) as pool:
        jobs = sorting_analysis.unit_ids
        if show_progress:
            jobs = tqdm(
                jobs,
                desc="Unit summaries",
                smoothing=0,
            )
        for res in pool.map(_summary_job, jobs):
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
        insertion_inds = np.searchsorted(times_a, times_b + lag)
        ccg[i] = np.sum(times_a[insertion_inds] == times_b)

    if auto:
        ccg[lags == 0] = 0

    return lags, ccg


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
                    Card(plots[0].kind, width, sum(p.height for p in card_plots))
                )
                card_plots = []
        if card_plots:
            cards.append(Card(plots[0].kind, width, sum(p.height for p in card_plots)))
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
        self, sorting_analysis, plots, max_height, figsize, dpi, save_folder, image_ext
    ):
        self.sorting_analysis = sorting_analysis
        self.plots = plots
        self.max_height = max_height
        self.figsize = figsize
        self.dpi = dpi
        self.save_folder = save_folder
        self.image_ext = image_ext


_summary_job_context = None


def _summary_init(*args):
    global _summary_job_context
    _summary_job_context = SummaryJobContext(*args)


def _summary_job(unit_id):
    fig = make_unit_summary(
        _summary_job_context.orting_analysis,
        unit_id,
        plots=_summary_job_context.plots,
        max_height=_summary_job_context.max_height,
        figsize=_summary_job_context.figsize,
    )
    ext = _summary_job_context.image_ext
    fig.savefig(
        _summary_job_context.save_folder / f"unit{unit_id:04d}.{ext}",
        dpi=_summary_job_context.dpi,
    )
