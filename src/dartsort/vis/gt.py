import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import FuncNorm
import warnings

from .layout import BasePlot, flow_layout


table_cmap = ["managua", "cividis"]
for table_cmap in table_cmap:
    if table_cmap in plt.colormaps:
        break
else:
    assert False


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
        ax.set_title("all agreements")
        ax.set_ylabel(f"{comparison.gt_name} unit")
        ax.set_xlabel(f"{comparison.tested_name} unit")


class TrimmedAgreementMatrix(ComparisonPlot):
    kind = "matrix"
    width = 3
    height = 2

    def __init__(self, trim_kind="auto", ordered=True, cmap=table_cmap):
        self.trim_kind = trim_kind
        self.ordered = ordered
        self.cmap = cmap

    def draw(self, panel, comparison):
        if self.ordered:
            agreement = comparison.comparison.get_ordered_agreement_scores()
        else:
            agreement = comparison.comparison.agreement_scores
        if self.trim_kind == "auto":
            pass
        else:
            assert self.ordered
            agreement.values[:, : agreement.shape[0]]
        ax = panel.subplots()
        im = ax.imshow(agreement.T, vmin=0, vmax=1, cmap=self.cmap)
        plt.colorbar(im, ax=ax, shrink=0.3)
        ax.set_title("Hung. match agreements")
        ax.set_xlabel(f"{comparison.gt_name} unit" + ("(ord)" * self.ordered))
        ax.set_ylabel(f"{comparison.tested_name} unit" + ("(ord)" * self.ordered))


class TrimmedTemplateDistanceMatrix(ComparisonPlot):
    kind = "matrix"
    width = 3
    height = 2

    def __init__(self, trim_kind="auto", ordered=True, cmap=plt.cm.magma):
        self.trim_kind = trim_kind
        self.ordered = ordered
        self.cmap = cmap

    def draw(self, panel, comparison):
        agreement = comparison.comparison.get_ordered_agreement_scores()
        row_order = agreement.index
        col_order = agreement.columns
        dist = comparison.template_distances[row_order, :][:, col_order]

        ax = panel.subplots()
        log1p_norm = FuncNorm((np.log1p, np.expm1), vmin=0)
        im = ax.imshow(dist.T, norm=log1p_norm, cmap=self.cmap)
        plt.colorbar(im, ax=ax, shrink=0.3)
        ax.set_title("Hung. match temp dists")
        ax.set_xlabel(f"{comparison.gt_name} unit")
        ax.set_ylabel(f"{comparison.tested_name} unit")


class MetricRegPlot(ComparisonPlot):
    kind = "gtmetric"
    width = 2
    height = 2

    def __init__(
        self,
        x="gt_ptp_amplitude",
        y="accuracy",
        color="b",
        log_x=False,
        logistic=True,
        lowess=False,
        log_y=False,
        quant_cmap="viridis",
    ):
        self.x = x
        self.y = y
        self.color = color
        self.quant_cmap = quant_cmap
        self.log_x = log_x
        self.logistic = logistic
        self.lowess = lowess
        self.log_y = log_y

    def draw(self, panel, comparison):
        ax = panel.subplots()
        df = comparison.unit_info_dataframe(force_distances=self.x == "min_temp_dist")

        qcolor = self.color in df.columns

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            finite_y = np.isfinite(df[self.y].values)
            finite_x = np.isfinite(df[self.x].values)
            df_show = df[np.logical_and(finite_y, finite_x)]

            if qcolor:
                ax.scatter(
                    df_show[self.x],
                    df_show[self.y],
                    c=df_show[self.color],
                    cmap=plt.get_cmap(self.quant_cmap),
                    lw=0,
                )

            sns.regplot(
                data=df_show,
                x=self.x,
                y=self.y,
                logistic=self.logistic,
                lowess=self.lowess,
                ax=ax,
                scatter=not qcolor,
                line_kws=dict(color="k"),
                **({} if qcolor else dict(color=self.color)),
            )
        if self.log_x and self.log_y:
            ax.loglog()
        elif self.log_x:
            ax.semilogx()
        elif self.log_y:
            ax.semilogy()
        met = df[self.y].mean()
        n_inf_y = np.logical_not(finite_y).sum()
        n_inf_x = np.logical_not(finite_x).sum()
        title = f"mean {self.y}: {met:.3f}"
        if n_inf_y or n_inf_x:
            title = f"{title}, yinf: {n_inf_y}, xinf: {n_inf_x}"
        ax.set_title(title, fontsize="small")
        ax.grid(which="both")


class MetricDistribution(ComparisonPlot):
    kind = "wide"

    def __init__(
        self,
        xs=("recall", "accuracy", "precision"),
        colors=("r", "b", "g"),
        flavor="hist",
        width=3,
        height=2,
    ):
        self.xs = list(xs)
        self.colors = colors
        self.flavor = flavor
        self.width = width
        self.height = height

    def draw(self, panel, comparison):
        ax = panel.subplots()
        df = comparison.unit_info_dataframe()
        keep = [x in df for x in self.xs]
        xs = [x for i, x in enumerate(self.xs) if keep[i]]
        colors = [c for i, c in enumerate(self.colors) if keep[i]]
        df = df[xs].melt(value_vars=xs, var_name="metric")
        if self.flavor == "hist":
            sns.histplot(
                data=df,
                x="value",
                hue="metric",
                palette=list(colors),
                element="step",
                ax=ax,
                bins=np.linspace(0, 1, 21),
            )
            sns.move_legend(ax, "upper left", frameon=False)
        elif self.flavor == "box":
            sns.boxplot(
                data=df,
                x="metric",
                y="value",
                hue="metric",
                palette=list(colors),
                ax=ax,
                legend=False,
                showmeans=True,
            )
            ax.tick_params(axis="x", rotation=90)
            ax.set_ylim([-0.05, 1.05])
            ax.set(xlabel=None, ylabel=None)
            ax.grid(which="both", axis="y")


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
        log1p_norm = FuncNorm((np.log1p, np.expm1), vmin=0)
        ax.imshow(dist, cmap=self.cmap, norm=log1p_norm)
        # plt.colorbar(im, ax=ax, shrink=0.3)
        ax.set_title("all temp dists")
        ax.set_ylabel(f"{comparison.gt_name} unit")
        ax.set_xlabel(f"{comparison.tested_name} unit")


class TemplateDistancesHistogram(ComparisonPlot):
    kind = "wide"
    width = 3
    height = 2

    def __init__(self, axis=0):
        self.axis = axis

    def draw(self, panel, comparison):
        ax = panel.subplots()
        d = np.nan_to_num(comparison.template_distances, nan=np.inf)
        vm = min(d.min(0).max(), d.min(1).max())
        min_gt_dist_for_tested_units = d.min(axis=self.axis)
        finite = np.isfinite(min_gt_dist_for_tested_units)
        x = min_gt_dist_for_tested_units[finite]
        bins = np.logspace(np.log10(d.min()), np.log10(vm), 96)
        ax.hist(x, bins=bins, color="orange", log=True)
        ax.semilogx()
        ax.grid(which="both")
        ax.set_ylabel("count")
        ninf = np.logical_not(finite).sum()
        if self.axis == 0:
            ax.set_xlabel("dist to GT (min over GT of tested-GT dists)")
            ax.set_title(
                f"tested template distances to GT library ({ninf} infs)",
                fontsize="small",
            )
        elif self.axis == 1:
            ax.set_xlabel("dist to tested (min over tested of tested-GT dists)")
            ax.set_title(
                f"GT template distances to tested library ({ninf} infs)",
                fontsize="small",
            )
        else:
            assert False


box = MetricDistribution(flavor="box", width=2, height=3.5)
box.kind = "gtmetric"
full_gt_overview_plots = (
    MetricRegPlot(x="gt_ptp_amplitude", y="accuracy"),
    MetricRegPlot(x="gt_ptp_amplitude", y="recall", color="r"),
    MetricRegPlot(x="gt_ptp_amplitude", y="precision", color="g"),
    MetricRegPlot(x="gt_ptp_amplitude", y="gt_dt_rms", color="palevioletred"),
    MetricRegPlot(x="gt_firing_rate", y="accuracy"),
    MetricRegPlot(x="gt_firing_rate", y="recall", color="r"),
    MetricRegPlot(x="gt_firing_rate", y="precision", color="g"),
    MetricRegPlot(x="min_temp_dist", y="precision", color="g"),
    MetricRegPlot(x="min_temp_dist", y="recall", color="gt_ptp_amplitude", log_x=True),
    MetricRegPlot(
        x="min_temp_dist",
        y="gt_dt_rms",
        color="gt_ptp_amplitude",
        log_x=True,
        logistic=False,
        lowess=True,
    ),
    MetricRegPlot(
        x="min_temp_dist", y="unsorted_recall", color="gt_ptp_amplitude", log_x=True
    ),
    MetricRegPlot(
        x="gt_ptp_amplitude",
        y="min_temp_dist",
        color="orange",
        logistic=False,
        lowess=True,
        log_y=True,
    ),
    MetricRegPlot(
        x="gt_firing_rate",
        y="min_temp_dist",
        color="orange",
        logistic=False,
        lowess=True,
        log_y=True,
    ),
    MetricRegPlot(
        x="gt_dt_rms",
        y="gt_ptp_amplitude",
        color="orange",
        logistic=False,
        lowess=True,
        log_y=True,
    ),
    MetricRegPlot(x="gt_ptp_amplitude", y="unsorted_recall", color="purple"),
    box,
    MetricDistribution(),
    TrimmedAgreementMatrix(),
    TrimmedTemplateDistanceMatrix(),
    TemplateDistancesHistogram(0),
    TemplateDistancesHistogram(1),
)

default_gt_overview_plots = (
    MetricRegPlot(x="gt_ptp_amplitude", y="accuracy"),
    MetricRegPlot(x="gt_ptp_amplitude", y="recall", color="r"),
    MetricRegPlot(x="gt_ptp_amplitude", y="precision", color="g"),
    MetricRegPlot(x="gt_firing_rate", y="accuracy"),
    MetricRegPlot(x="gt_firing_rate", y="recall", color="r"),
    MetricRegPlot(x="gt_firing_rate", y="precision", color="g"),
    MetricRegPlot(x="gt_ptp_amplitude", y="unsorted_recall", color="purple"),
    box,
    MetricDistribution(),
    TrimmedAgreementMatrix(),
    TrimmedAgreementMatrix(ordered=False),
)

# multi comparisons stuff
# box and whisker between sorters


def make_gt_overview_summary(
    comparison,
    plots=default_gt_overview_plots,
    max_height=10,
    figsize=(18, 14),
    figure=None,
    suptitle=True,
    same_width_flow=True,
):
    if not comparison.has_templates:
        plots_ = []
        for plot in plots:
            if isinstance(plot, MetricRegPlot):
                if "temp" in plot.x:
                    continue
                if "temp" in plot.y:
                    continue
            if isinstance(plot, TemplateDistanceMatrix):
                continue
            if isinstance(plot, TrimmedTemplateDistanceMatrix):
                continue
            plots_.append(plot)
        plots = plots_
    figure = flow_layout(
        plots,
        max_height=max_height,
        figsize=figsize,
        figure=figure,
        comparison=comparison,
        same_width_flow=same_width_flow,
    )
    if suptitle is True:
        figure.suptitle(f"{comparison.gt_name} vs. {comparison.tested_name}")
    elif suptitle:
        figure.suptitle(suptitle)

    return figure
