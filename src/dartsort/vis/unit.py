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
from ..util.internal_config import raw_template_cfg
from ..util.job_util import get_global_computation_config
from ..evaluate.analysis import DARTsortAnalysis, WaveformsBag
from ..util.multiprocessing_util import CloudpicklePoolExecutor, get_pool, cloudpickle
from . import layout
from .analysis_plots import isi_hist, correlogram, plot_correlogram, bar
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
            msg += f"feature source: {h5_path.name}\n"

        nspikes = (sorting_analysis.sorting.labels == unit_id).sum()
        msg += f"n spikes: {nspikes}\n"

        assert sorting_analysis.template_data is not None
        temps = sorting_analysis.template_data.unit_templates(unit_id)
        if not temps.size:
            msg += "no template (too few spikes)"
        elif temps.shape[0] == 1:
            ptp = np.ptp(temps, 1).max(1)[0]
            msg += f"maxptp: {ptp:0.2f} su\n"
            snr = ptp * np.sqrt(nspikes)
            msg += f"template snr: {snr:.1f}"
        else:
            ptp = np.ptp(temps, 1).max(1).mean()
            msg += f"mean superres maxptp: {ptp:0.1f}su\n"
            in_unit = sorting_analysis.template_data.unit_mask(unit_id)
            counts = sorting_analysis.template_data.spike_counts[in_unit]
            snrs = np.ptp(temps, 1).max(1) * np.sqrt(counts)
            msg += "template snrs:\n  " + ", ".join(f"{s:0.1f}" for s in snrs)

        axis.text(0, 0, msg, fontsize=6.5)


# -- small summary plots


class ACG(UnitPlot):
    kind = "histogram"
    height = 0.75

    def __init__(self, max_lag=50):
        super().__init__()
        self.max_lag = max_lag

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id: int):
        axis = panel.subplots()
        which = sorting_analysis.in_unit(unit_id)
        times_samples = sorting_analysis.sorting.times_samples[which]
        plot_correlogram(axis, times_samples, max_lag=self.max_lag)
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
        times_s = sorting_analysis.times_seconds[which]
        isi_hist(
            times_s,
            axis,
            bin_ms=self.bin_ms,
            max_ms=self.max_ms,
            color=color,
            label=label,
        )


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

        in_unit = sorting_analysis.in_unit(unit_id)
        x = sorting_analysis.x[in_unit]
        if self.registered:
            z = sorting_analysis.registered_z[in_unit]
        else:
            z = sorting_analysis.z[in_unit]
        geomx, geomz = sorting_analysis.geom.T
        pad = self.probe_margin_um
        valid = x == np.clip(x, geomx.min() - pad, geomx.max() + pad)
        valid &= z == np.clip(z, geomz.min() - pad, geomz.max() + pad)
        amps = sorting_analysis.amplitudes[in_unit]
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
    kind = "widescatter"
    width = 2

    def __init__(
        self,
        feat_name,
        color_by_amplitude=True,
        amplitude_color_cutoff=15,
        alpha=1.0,
        label=None,
    ):
        super().__init__()
        self.feat_name = feat_name
        self.amplitude_color_cutoff = amplitude_color_cutoff
        self.color_by_amplitude = color_by_amplitude
        self.alpha = alpha
        self.label = label or feat_name

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id: int):
        axis = panel.subplots()
        in_unit = sorting_analysis.in_unit(unit_id)
        t = sorting_analysis.times_seconds[in_unit]
        feat = sorting_analysis.named_feature(self.feat_name, which=in_unit)
        c = None
        if self.color_by_amplitude:
            amps = sorting_analysis.amplitudes[in_unit]
            c = np.minimum(amps, self.amplitude_color_cutoff)
        s = axis.scatter(t, feat, c=c, lw=0, s=3, alpha=self.alpha, rasterized=True)
        axis.set_xlabel("time (s)")
        axis.set_ylabel(self.label)
        if self.color_by_amplitude:
            plt.colorbar(s, ax=axis, shrink=0.5, label="amp (su)")


class TimeZScatter(TimeFeatScatter):
    def __init__(self, **kwargs):
        super().__init__(feat_name="z", label="z (um)", **kwargs)


class TimeRegZScatter(TimeFeatScatter):
    def __init__(self, **kwargs):
        super().__init__(feat_name="registered_z", label="reg. z (um)", **kwargs)


class TimeAmpScatter(TimeFeatScatter):
    def __init__(self, **kwargs):
        super().__init__(feat_name="amplitudes", label="amp (su)", **kwargs)


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

        handles = {}
        if waves is not None:
            if np.isfinite(waves.waveforms[:, 0, :]).any():
                max_abs_amp = self.max_abs_template_scale * np.nanpercentile(
                    np.abs(waves.waveforms), 99
                )
            ls = geomplot(
                waves.waveforms,
                max_channels=np.full(len(waves.waveforms), waves.main_channel),
                channel_index=waves.channel_index,
                geom=waves.geom,
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
            handles["waveforms"] = ls

        if show_template:
            assert templates is not None
            if waves is None:
                showchans = sorting_analysis.vis_channel_index[
                    sorting_analysis.unit_max_channel(unit_id)
                ]
                geom = sorting_analysis.registered_geom
            else:
                showchans = waves.channel_index[waves.main_channel]
                geom = waves.geom
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
            handles["mean"] = ls

        shift_str = "shifted " * sorting_analysis.shifting
        if self.title is None:
            axis.set_title(shift_str + self.wfs_kind)
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
            )


class RawWaveformPlot(WaveformPlot):
    wfs_kind = "raw wfs"

    def get_waveforms(
        self, sorting_analysis: DARTsortAnalysis, unit_id: int
    ) -> WaveformsBag | None:
        return sorting_analysis.unit_raw_waveforms(unit_id, max_count=self.count)


class TPCAWaveformPlot(WaveformPlot):
    wfs_kind = "coll.-cl. tpca wfs"

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
            title="",
        )


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

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id, axis=None):
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
            cmap="RdGy",
            origin="lower",
            interpolation="none",
        )
        if self.show_values:
            for (j, i), label in np.ndenumerate(neighbor_dists):
                axis.text(i, j, f"{label:.2f}", ha="center", va="center")
        plt.colorbar(im, ax=axis, shrink=0.3)
        axis.set_xticks(range(len(neighbor_ids)), neighbor_ids)
        axis.set_yticks(range(len(neighbor_ids)), neighbor_ids)
        for i, (tx, ty) in enumerate(
            zip(axis.xaxis.get_ticklabels(), axis.yaxis.get_ticklabels())
        ):
            tx.set_color(colors[i])
            ty.set_color(colors[i])
        axis.set_title(self.title)


class NeighborCCGPlot(UnitPlot):
    kind = "neighbors"
    width = 3
    height = 0.75

    def __init__(self, n_neighbors=3, max_lag=50):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.max_lag = max_lag

    def draw(self, panel, sorting_analysis: DARTsortAnalysis, unit_id: int):
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

        my_st = sorting_analysis.sorting.times_samples[
            sorting_analysis.in_unit(unit_id)
        ]
        neighb_sts = [
            sorting_analysis.sorting.times_samples[sorting_analysis.in_unit(nid)]
            for nid in neighbor_ids
        ]

        axes = panel.subplots(
            nrows=2, sharey="row", sharex=True, squeeze=False, ncols=len(neighb_sts)
        )
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


# -- main routines

default_plots = (
    UnitTextInfo(),
    ACG(),
    ISIHistogram(),
    XZScatter(),
    PCAScatter(),
    TimeZScatter(),
    TimeRegZScatter(),
    TimeAmpScatter(),
    RawWaveformPlot(),
    TPCAWaveformPlot(),
    NearbyCoarseTemplatesPlot(),
    CoarseTemplateDistancePlot(),
    NeighborCCGPlot(),
)


template_assignment_plots = (
    UnitTextInfo(),
    RawWaveformPlot(),
)


def make_unit_summary(
    sorting_analysis: DARTsortAnalysis,
    unit_id,
    amplitude_color_cutoff=15.0,
    pca_radius_um=75.0,
    plots=default_plots,
    max_height=4,
    figsize=(16, 8.5),
    figure=None,
    gizmo_name="sorting_analysis",
    **other_global_params,
):
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
        **{gizmo_name: sorting_analysis},  # type: ignore
    )

    return figure


def make_all_summaries(
    sorting_analysis: DARTsortAnalysis,
    save_folder,
    plots=default_plots,
    amplitude_color_cutoff=15.0,
    pca_radius_um=75.0,
    max_height=4,
    figsize=(16, 8.5),
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
    if n_jobs:
        initargs = (cloudpickle.dumps(initargs),)
    n_jobs, Executor, context = get_pool(n_jobs, cls=CloudpicklePoolExecutor)  # type: ignore
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
        for res in results:
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
        raise ValueError(f"Need a sorting_analysis if namebyamp.")
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
        args = cloudpickle.loads(args[0])
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
