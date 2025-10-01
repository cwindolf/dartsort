import numpy as np

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

    def draw(self, panel, sorting_analysis):
        axis = panel.subplots()
        axis.axis("off")
        msg = ""

        if sorting_analysis.name:
            msg += f"Sorting: {sorting_analysis.name}\n"

        if sorting_analysis.hdf5_path is not None:
            msg += f"feature source: {sorting_analysis.hdf5_path.name}\n"

        nspikes = sorting_analysis.sorting.labels.size
        triaged = np.sum(sorting_analysis.sorting.labels < 0)
        msg += f"{nspikes} total spikes\n"
        if triaged:
            tpct = 100 * triaged / nspikes
            msg += f"{triaged} ({tpct:0.1f}%) of these are triaged"
        else:
            msg += f"no triaged spikes"

        n_units = sorting_analysis.sorting.n_units
        msg += f"{n_units} units"

        axis.text(0, 0, msg, fontsize=6.5)


class CoarseTemplateMaxChannelsPlot(OverviewPlot):
    kind = "scatter"
    height = 3.5

    def draw(self, panel, sorting_analysis):
        axis = panel.subplots()
        analysis_plots.scatter_max_channel_waveforms(
            axis,
            sorting_analysis.coarse_template_data,
            geom=sorting_analysis.recording.get_channel_locations(),
            waveform_height=0.05,
            waveform_width=0.75,
            show_geom=True,
        )


class SpikeCountHistogram(OverviewPlot):
    kind = "histogram"
    width = 2

    def __init__(self, n_bins=128, log=True):
        self.n_bins = n_bins
        self.log = log

    def draw(self, panel, sorting_analysis):
        axis = panel.subplots()
        axis.hist(
            sorting_analysis.coarse_template_data.spike_counts,
            bins=self.n_bins,
            log=self.log,
            histtype="stepfilled",
            color="k",
        )
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

    def draw(self, panel, sorting_analysis):
        analysis_plots.distance_matrix_dendro(
            panel,
            sorting_analysis.merge_dist,
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

    def draw(self, panel, sorting_analysis):
        axis = panel.subplots()
        axis.scatter(
            sorting_analysis.sorting.times_seconds,
            sorting_analysis.sorting.labels,
            c=self.colors[sorting_analysis.sorting.labels % len(self.colors)],
            **self.scatter_kwargs,
        )
        axis.set_xlabel("time (s)")
        axis.set_ylabel("unit")


sorting_plots = (
    SortingTextInfo(),
    CoarseTemplateMaxChannelsPlot(),
    SpikeCountHistogram(),
    MergeDistanceMatrix(),
    RasterPlot(),
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
