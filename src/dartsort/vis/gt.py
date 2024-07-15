import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import unit
from .colors import glasbey1024
from .layout import BasePlot, flow_layout
from .unit import UnitPlot, make_all_summaries

# -- single-unit


class GTUnitMixin:
    def draw(self, panel, comparison, gt_unit_id):
        return super().draw(panel, comparison.gt_analysis, gt_unit_id)


class PredictedUnitMixin:
    def draw(self, panel, comparison, tested_unit_id):
        return super().draw(panel, comparison.predicted_analysis, tested_unit_id)


class BestMatchUnitMixin:
    def draw(self, panel, comparison, gt_unit_id):
        tested_unit_id = comparison.get_match(gt_unit_id)
        return super().draw(panel, comparison.predicted_analysis, tested_unit_id)


class UnitComparisonPlot(BasePlot):
    kind = "unit_comparison"

    def draw(self, panel, comparison, gt_unit_id):
        tested_unit_id = comparison.get_match(gt_unit_id)
        return self._draw(panel, comparison, gt_unit_id, tested_unit_id)

    def _draw(self, panel, comparison, gt_unit_id, tested_unit_id):
        raise NotImplementedError


class BestMatchIsiComparison(UnitComparisonPlot):
    kind = "histogram"
    width = 1
    height = 1

    def __init__(self, max_lag=50):
        self.unit_isi = unit.ISIHistogram(max_lag=max_lag)

    def _draw(self, panel, comparison, gt_unit_id, tested_unit_id):
        axis = panel.subplots()
        self.unit_isi.draw(panel, comparison.gt_analysis, gt_unit_id, axis=axis)
        self.unit_isi.draw(panel, comparison.gt_analysis, tested_unit_id, color=glasbey1024[tested_unit_id], axis=axis)
        axis.set_ylabel("count")


class GTUnitTextInfo(UnitComparisonPlot):
    kind = "_info"
    width = 1
    height = 1
    pass

    def _draw(self, panel, comparison, gt_unit_id, tested_unit_id):
        axis = panel.subplots()
        axis.axis("off")
        msg = f"GT unit: {gt_unit_id}\n"
        msg = f"Hungarian matched unit: {tested_unit_id}\n"

        gt_nspikes = comparison.gt_sorting_analysis.spike_counts[
            comparison.gt_sorting_analysis.unit_ids == gt_unit_id
        ].sum()
        tested_nspikes = comparison.tested_sorting_analysis.spike_counts[
            comparison.tested_sorting_analysis.unit_ids == tested_unit_id
        ].sum()
        msg += f"{gt_nspikes} spikes in GT unit\n"
        msg += f"{tested_nspikes} spikes in matched unit\n"

        gt_temp = comparison.gt_sorting_analysis.coarse_template_data.unit_templates(gt_unit_id)
        tested_temp = comparison.tested_sorting_analysis.coarse_template_data.unit_templates(tested_unit_id)
        gt_ptp = gt_temp.ptp(1).max(1).squeeze()
        assert gt_ptp.size == 1
        tested_ptp = tested_temp.ptp(1).max(1).squeeze()
        assert tested_ptp.size == 1
        msg += f"GT PTP: {gt_ptp:0.1f}; matched PTP: {tested_ptp:0.1f}\n"
        msg += f"{tested_nspikes} spikes in matched unit\n"

        axis.text(0, 0, msg, fontsize=6.5)


class BestMatchVennPlot(UnitComparisonPlot):
    def __init__(self, gt_color="r", matched_color="gold", tested_color="b"):
        self.gt_color = gt_color
        self.matched_color = matched_color
        self.tested_color = tested_color

    def _draw(self, panel, comparison, gt_unit_id, tested_unit_id):
        import matplotlib_venn
        pass


class BestMatchWaveformsPlot(UnitComparisonPlot):
    def __init__(self, gt_color="r", matched_color="gold", tested_color="b"):
        self.gt_color = gt_color
        self.matched_color = matched_color
        self.tested_color = tested_color

    def _draw(self, panel, comparison, gt_unit_id, tested_unit_id):
        pass


class TemplateDistanceHistogram(UnitComparisonPlot):
    pass


class NearbyTemplatesMultiChan(UnitComparisonPlot):
    pass


class NearbyTemplatesSingleChan(UnitComparisonPlot):
    pass


class NearbyTemplatesConfusionMatrix(UnitComparisonPlot):
    pass


# -- comparisons

class ComparisonPlot(BasePlot):
    kind = "comparison"
    width = 2
    height = 2

    def draw(self, panel, comparison):
        raise NotImplementedError


class AgreementMatrix(ComparisonPlot):
    kind = "wide"
    width = 3
    height = 1

    def draw(self, panel, comparison):
        agreement = comparison.comparison.get_ordered_agreement_scores()
        ax = panel.subplots()
        ax.imshow(agreement, vmin=0, vmax=1)
        # plt.colorbar(im, ax=ax, shrink=0.3)
        ax.set_title("all units' agreements")
        ax.set_ylabel(f"{comparison.gt_name} unit")
        ax.set_xlabel(f"{comparison.tested_name} unit")


class TrimmedAgreementMatrix(ComparisonPlot):
    kind = "matrix"
    width = 3
    height = 3

    def draw(self, panel, comparison):
        agreement = comparison.comparison.get_ordered_agreement_scores()
        ax = panel.subplots()
        im = ax.imshow(agreement.values[:, :agreement.shape[0]], vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.3)
        ax.set_title("Hungarian matches' agreements")
        ax.set_ylabel(f"{comparison.gt_name} unit")
        ax.set_xlabel(f"{comparison.tested_name} unit")


class MetricRegPlot(ComparisonPlot):
    kind = "gtmetric"

    def __init__(self, x="gt_ptp_amplitude", y="accuracy", color="b"):
        self.x = x
        self.y = y
        self.color = color

    def draw(self, panel, comparison):
        ax = panel.subplots()
        df = comparison.unit_info_dataframe()
        sns.regplot(
            data=df,
            x=self.x,
            y=self.y,
            logistic=True,
            color=self.color,
            ax=ax,
        )
        met = df[self.y].mean()
        ax.set_title(f"mean {self.y}: {met:.3f}", fontsize="small")


class MetricHistogram(ComparisonPlot):
    kind = "wide"
    width = 3
    height = 2

    def __init__(self, xs=("recall", "precision", "accuracy"), colors="rgb"):
        self.xs = list(xs)
        self.colors = colors

    def draw(self, panel, comparison):
        ax = panel.subplots()
        df = comparison.unit_info_dataframe()
        df = df[self.xs].melt(value_vars=self.xs, var_name='metric')
        sns.histplot(
            data=df,
            x='value',
            hue='metric',
            palette=list(self.colors),
            element='step',
            ax=ax,
            bins=np.linspace(0, 1, 21),
        )
        sns.move_legend(ax, 'upper left')


class TemplateDistanceMatrix(ComparisonPlot):
    kind = "wide"
    width = 3
    height = 1

    def __init__(self, cmap=plt.cm.magma):
        self.cmap = cmap

    def draw(self, panel, comparison):
        agreement = comparison.comparison.get_ordered_agreement_scores()
        row_order = agreement.index
        col_order = np.array(agreement.columns)
        dist = comparison.template_distances[row_order, :][:, col_order]

        ax = panel.subplots()
        ax.imshow(dist, vmin=0, cmap=self.cmap)
        # plt.colorbar(im, ax=ax, shrink=0.3)
        ax.set_title("all units' agreements")
        ax.set_ylabel(f"{comparison.gt_name} unit")
        ax.set_xlabel(f"{comparison.tested_name} unit")


class TrimmedTemplateDistanceMatrix(ComparisonPlot):
    kind = "matrix"
    width = 3
    height = 3

    def __init__(self, cmap=plt.cm.magma):
        self.cmap = cmap

    def draw(self, panel, comparison):
        agreement = comparison.comparison.get_ordered_agreement_scores()
        row_order = agreement.index
        col_order = np.array(agreement.columns)[:agreement.shape[0]]
        dist = comparison.template_distances[row_order, :][:, col_order]

        ax = panel.subplots()
        im = ax.imshow(dist, vmin=0, cmap=self.cmap)
        plt.colorbar(im, ax=ax, shrink=0.3)
        ax.set_title("Hungarian matches' agreements")
        ax.set_ylabel(f"{comparison.gt_name} unit")
        ax.set_xlabel(f"{comparison.tested_name} unit")


gt_overview_plots = (
    MetricHistogram(),
    TrimmedAgreementMatrix(),
    AgreementMatrix(),
    TrimmedTemplateDistanceMatrix(),
    TemplateDistanceMatrix(),
    MetricRegPlot(x="gt_ptp_amplitude", y="accuracy"),
    MetricRegPlot(x="gt_ptp_amplitude", y="recall", color="r"),
    MetricRegPlot(x="gt_ptp_amplitude", y="precision", color="g"),
    MetricRegPlot(x="gt_firing_rate", y="accuracy"),
    MetricRegPlot(x="gt_firing_rate", y="recall", color="r"),
    MetricRegPlot(x="gt_firing_rate", y="precision", color="g"),
    MetricRegPlot(x="gt_match_temp_dist", y="precision", color="g"),
)


def make_gt_overview_summary(
    comparison,
    plots=gt_overview_plots,
    max_height=6,
    figsize=(11, 8.5),
    figure=None,
    suptitle=None,
):
    figure = flow_layout(
        plots,
        max_height=max_height,
        figsize=figsize,
        figure=figure,
        comparison=comparison,
    )
    if suptitle is not None:
        figure.suptitle(suptitle)

    return figure
