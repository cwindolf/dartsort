import numpy as np

from ..evaluate.analysis import DARTsortAnalysis
from . import analysis_plots
from .colors import glasbey1024
from .layout import BasePlot, flow_layout


class OverviewPlot(BasePlot):
    def draw(
        self,
        panel,
        sorting_analysis,
    ):
        raise NotImplementedError


class SortingTextInfo(OverviewPlot):
    kind = "text"
    height = 0.5

    def draw(self, panel, sorting_analysis: DARTsortAnalysis):
        axis = panel.subplots()
        axis.axis("off")
        msg = ""

        qdf = sorting_analysis.summary_df()

        if sorting_analysis.name:
            msg += f"Sorting: {sorting_analysis.name}\n"

        h5_path = sorting_analysis.sorting.parent_h5_path
        if h5_path is not None:
            msg += f"feature source: {h5_path.name}\n"

        nspikes = sorting_analysis.sorting.times_samples.size
        if sorting_analysis.sorting.labels is None:
            triaged = 0
        else:
            triaged = np.sum(sorting_analysis.sorting.labels < 0)
        msg += f"{nspikes} total spikes\n"
        if triaged:
            tpct = 100 * triaged / nspikes
            msg += f"{triaged} ({tpct:0.1f}%) noise\n"
        else:
            msg += "no noise spikes\n"

        if 'isi_violations_ratio' in qdf.columns:
            vi = qdf.isi_violations_ratio
            nclean = (vi < 0.1).sum()
            nviol = (vi >= 0.1).sum()
            cleanyield = qdf.num_spikes[vi < 0.1].sum()
            cleanpct = f"{100 * cleanyield / nspikes:0.1f}%"

            msg += f"{nclean} <10%viol ({nviol} viol) units\n"
            msg += f"{cleanyield} spikes in <10%viol ({cleanpct})\n"

        n_units = sorting_analysis.sorting.n_units
        msg += f"{n_units} units"

        axis.text(0, 0, msg, fontsize=6.5)


class CoarseTemplateMaxChannelsPlot(OverviewPlot):
    kind = "scatter"
    height = 3.5

    def draw(self, panel, sorting_analysis: DARTsortAnalysis):
        axis = panel.subplots()
        analysis_plots.scatter_max_channel_waveforms(
            axis,
            sorting_analysis.coarse_template_data,
            geom=sorting_analysis.recording.get_channel_locations(),
            waveform_height=0.05,
            waveform_width=0.75,
            show_geom=True,
        )


class FiringRateHistogram(OverviewPlot):
    kind = "histogram"
    width = 2

    def __init__(self, n_bins=128, log=True):
        self.n_bins = n_bins
        self.log = log

    def draw(self, panel, sorting_analysis: DARTsortAnalysis):
        assert sorting_analysis.coarse_template_data is not None
        assert sorting_analysis.sorting.labels is not None
        qdf = sorting_analysis.summary_df()
        show = dict(fr=qdf.ds_firing_rates)
        if 'firing_rate' in qdf.columns:
            show['sifr'] = qdf.firing_rate
        axis = panel.subplots()
        axis.hist(
            show.values(),
            label=show.keys(),
            bins=self.n_bins,
            log=self.log,
            histtype="step",
            color="k" if len(show) <= 1 else ["k", "gray"],
        )
        axis.set_ylim(bottom=0)
        if len(show) > 1:
            axis.legend(fancybox=False)
        axis.grid()
        axis.set_xlabel("firing rate")


class ISIViolCDF(OverviewPlot):
    kind = "histogram"
    width = 2

    def __init__(self, n_bins=128, log=True):
        self.n_bins = n_bins
        self.log = log

    def draw(self, panel, sorting_analysis: DARTsortAnalysis):
        assert sorting_analysis.coarse_template_data is not None
        assert sorting_analysis.sorting.labels is not None
        qdf = sorting_analysis.summary_df()
        if 'isi_violation_ratio' not in qdf.columns:
            ax = panel.subplots()
            ax.axis('off')
            return
        vi = qdf.isi_violation_ratio
        axis = panel.subplots()
        axis.hist(
            vi.values,
            bins=self.n_bins,
            log=self.log,
            histtype="step",
            color="k",
            cumulative=True,
        )
        axis.set_ylim(bottom=0)
        axis.grid(which='both')
        axis.set_xlabel("per-unit spike count")


class MergeDistanceMatrix(OverviewPlot):
    kind = "heatmap"
    width = 2
    height = 2

    def __init__(
        self, dendrogram_linkage=None, show_unit_labels=False, dendrogram_threshold=0.25
    ):
        self.dendrogram_linkage = dendrogram_linkage
        self.show_unit_labels = show_unit_labels
        self.dendrogram_threshold = dendrogram_threshold

    def draw(self, panel, sorting_analysis: DARTsortAnalysis):
        assert sorting_analysis.coarse_template_data is not None
        analysis_plots.distance_matrix_dendro(
            panel,
            sorting_analysis.merge_distances,
            unit_ids=sorting_analysis.coarse_template_data.unit_ids,
            dendrogram_linkage=self.dendrogram_linkage,
            dendrogram_threshold=self.dendrogram_threshold,
            show_unit_labels=self.show_unit_labels,
        )


class RasterPlot(OverviewPlot):
    kind = "scatter"
    width = 2

    def __init__(self, colors=glasbey1024, **scatter_kwargs):
        self.colors = colors
        self.scatter_kwargs = dict(s=3, lw=0) | scatter_kwargs

    def draw(self, panel, sorting_analysis: DARTsortAnalysis):
        assert sorting_analysis.sorting.labels is not None
        axis = panel.subplots()
        axis.scatter(
            sorting_analysis.times_seconds,
            sorting_analysis.sorting.labels,
            c=self.colors[sorting_analysis.sorting.labels % len(self.colors)],
            **self.scatter_kwargs,
        )
        axis.set_xlabel("time (s)")
        axis.set_ylabel("unit")


sorting_plots = (
    SortingTextInfo(),
    CoarseTemplateMaxChannelsPlot(),
    FiringRateHistogram(),
    ISIViolCDF(),
    MergeDistanceMatrix(),
    # RasterPlot(),
)


def make_sorting_summary(
    sorting_analysis,
    plots=sorting_plots,
    max_height=4,
    figsize=(11, 8.5),
    figure=None,
):
    figure = flow_layout(
        plots,
        max_height=max_height,
        figsize=figsize,
        figure=figure,
        sorting_analysis=sorting_analysis,
    )

    return figure


# def sorting_animation(
#     sorting_analysis,
# )
