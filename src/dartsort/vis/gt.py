import matplotlib.pyplot as plt
import seaborn as sns

from .layout import BasePlot, flow_layout
from .unit import UnitPlot, make_all_summaries


class GTUnitMixin:
    def draw(self, panel, comparison, gt_unit_id):
        return super().draw(panel, comparison.gt_analysis, gt_unit_id)


class PredictedUnitMixin:
    def draw(self, panel, comparison, predicted_unit_id):
        return super().draw(panel, comparison.predicted_analysis, predicted_unit_id)


class BestMatchUnitMixin:
    def draw(self, panel, comparison, gt_unit_id):
        predicted_unit_id = comparison.get_match(gt_unit_id)
        return super().draw(panel, comparison.predicted_analysis, predicted_unit_id)


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
        im = ax.imshow(agreement, vmin=0, vmax=1)
        # plt.colorbar(im, ax=ax, shrink=0.3)
        ax.set_title("all units' agreements")
        ax.set_ylabel("gt unit")
        ax.set_xlabel("sorter unit")


class TrimmedAgreementMatrix(ComparisonPlot):
    kind = "matrix"
    width = 3
    height = 3

    def draw(self, panel, comparison):
        agreement = comparison.comparison.get_ordered_agreement_scores()
        ax = panel.subplots()
        im = ax.imshow(agreement.values[:, :agreement.shape[0]], vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.3)
        ax.set_title("best matches' agreements")
        ax.set_ylabel("gt unit")
        ax.set_xlabel("sorter unit")


class MetricRegPlot(ComparisonPlot):
    kind = "gtmetric"

    def __init__(self, x="gt_ptp_amplitude", y="accuracy", color="b"):
        self.x = x
        self.y = y
        self.color = color

    def draw(self, panel, comparison):
        ax = panel.subplots()
        sns.regplot(
            data=comparison.unit_info_dataframe(),
            x=self.x,
            y=self.y,
            logistic=True,
            color=self.color,
            ax=ax,
        )


class MetricHistogram(ComparisonPlot):
    kind = "wide"
    width = 3
    height = 2

    def __init__(self, xs=["recall", "precision", "accuracy"], colors="rgb"):
        self.xs = xs
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
            element='poly',
            ax=ax,
        )
        sns.move_legend(ax, 'upper left')


gt_overview_plots = (
    TrimmedAgreementMatrix(),
    MetricHistogram(),
    AgreementMatrix(),
    MetricRegPlot(x="gt_ptp_amplitude", y="accuracy"),
    MetricRegPlot(x="gt_ptp_amplitude", y="recall", color="r"),
    MetricRegPlot(x="gt_ptp_amplitude", y="precision", color="g"),
    MetricRegPlot(x="gt_firing_rate", y="accuracy"),
    MetricRegPlot(x="gt_firing_rate", y="recall", color="r"),
    MetricRegPlot(x="gt_firing_rate", y="precision", color="g"),
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