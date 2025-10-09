from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

from ..evaluate.comparison import DARTsortGTVersus
from .colors import glasbey1024
from .layout import BasePlot, flow_layout


_cm = {
    "gt_ptp_amplitude": "viridis",
    "gt_firing_rate": "cividis",
    "min_temp_dist": [k for k in ("managua", "magma") if k in plt.colormaps][0],
}
_c = {
    "accuracy": "green",
    "precision": "b",
    "recall": "r",
    "unsorted_recall": "blueviolet",
    "min_temp_dist": "darkorange",
    "n_units": "teal",
}
_o = {
    "accuracy": 1,
    "precision": 1,
    "recall": 1,
    "unsorted_recall": 1,
    "min_temp_dist": -1,
}

_legkw = dict(
    borderpad=0.2,
    labelspacing=0.2,
    handlelength=1.0,
    columnspacing=1.0,
)
_box_kw = dict(showmeans=True, meanprops=dict(markeredgecolor="k", marker="+", markersize=15))
_scatter_kw = dict(linewidths=0, s=5)
_reg_common = dict(ci=None, scatter_kws=_scatter_kw)
_logistic = dict(logistic=True, **_reg_common)
_none = dict(fit_reg=False, **_reg_common)
_lowess = dict(lowess=True, **_reg_common)
_linear = dict(**_reg_common)
_regkind = {
    "accuracy": _logistic,
    "precision": _logistic,
    "recall": _logistic,
    "unsorted_recall": _logistic,
    "min_temp_dist": _lowess,
    "n_units": _none,
}
default_metrics = tuple([k for k in _c.keys() if k != "n_units"])


class VersusPlot(BasePlot):
    kind = "vs"
    width = 1
    height = 1

    # some plots only work for vs of two sorters
    # the main fn below will filter plots depending on if the vs has only two
    two_only = False

    def draw(self, panel, vs: DARTsortGTVersus):
        raise NotImplementedError


class MetricColumn(VersusPlot):
    kind = "column"

    def __init__(
        self,
        x: str = "gt_ptp_amplitude",
        metrics: Sequence[str] = default_metrics,
        diff=False,
        hmul=1.0,
        logx=True,
        box=False,
        box_x_cuts=None,
    ):
        self.x = x
        self.nmetrics = len(metrics)
        self.metrics = metrics
        self.height = hmul * self.nmetrics
        self.two_only = self.diff = diff
        self.logx = logx
        self.box = box
        self.box_x_cuts = box_x_cuts

    def draw(self, panel, vs):
        axes = panel.subplots(nrows=self.nmetrics, sharex=True)
        df = vs.unit_versus_dataframe()
        if self.diff:
            lines = [Line2D([0, 1], [0, 0], color='k')]
        else:
            spal = sns.color_palette(glasbey1024[: vs.n_vs])
            lines = [Line2D([0, 1], [0, 0], color=c) for c in spal]

        if self.diff:
            df_a = df[df[vs.sorter_var] == vs.a_name]
            df_b = df[df[vs.sorter_var] == vs.b_name]
            df_a = df_a.sort_values(by="gt_unit_id")
            df_b = df_b.sort_values(by="gt_unit_id")
            assert np.array_equal(df_b.gt_unit_id, df_b.gt_unit_id)
            df = df_a.copy()
            for met in self.metrics:
                print(' - ')
                print(f"{vs.a_name=} {df_a[met].mean()=} {df[met].mean()=}")
                print(f"{df_b[met].mean()=}")
                df[met] -= df_b[met]
                print(f"{df[met].mean()=}")

        x = self.x
        if self.box and self.box_x_cuts:
            bins = [0] + self.box_x_cuts + [int(np.ceil(df[x].max().item()))]
            df = df.copy(deep=False)
            edge_strs = [f"{int(b)}" for b in bins]
            bin_strs = np.array([f"{a}-{b}" for a, b in zip(bins, bins[1:])])
            binix = np.searchsorted(bins, df[x].values, side="right") - 1
            assert binix.min() >= 0
            df["binix"] = binix
            df[f"{x} bin"] = bin_strs[binix]
            df = df.sort_values(by="binix")
            x = f"{x} bin"
            if self.diff:
                for met in self.metrics:
                    for binst in bin_strs:
                        print(' -  - ')
                        print(f"{met=} {df[met].mean()=} {binst=} {df[df[x] == binst][met].mean()=}")
        elif self.box:
            x = None

        for ax, met in zip(axes.flat, self.metrics):
            logy = met == "min_temp_dist" and not self.diff

            if self.diff:
                ax.axhline(0, lw=0.8, color="k")
                ckw = dict(color=_c[met])
            else:
                ckw = dict(hue=vs.sorter_var, hue_order=vs.other_names, palette=spal)

            if not self.box:
                regkw = _lowess if self.diff else _regkind[met]
                if self.diff:
                    regkw = regkw | dict(line_kws=dict(color="k"))
                    sns.regplot(df, ax=ax, x=x, y=met, **regkw, **ckw)
                else:
                    # first, scatter in random order
                    sdf = df.sample(frac=1)
                    sns.scatterplot(
                        sdf,
                        ax=ax,
                        x=x,
                        y=met,
                        legend=False,
                        **ckw,
                        **_scatter_kw,
                    )

                    # next, regressions.
                    for sorter, color in zip(vs.other_names, glasbey1024):
                        sdf = df[df[vs.sorter_var] == sorter]
                        sns.regplot(
                            sdf,
                            ax=ax,
                            x=x,
                            y=met,
                            color=color,
                            scatter=False,
                            **regkw,
                        )
            else:
                sns.boxplot(
                    df, x=x, y=met, legend=False, log_scale=logy, ax=ax, **_box_kw, **ckw
                )

            if logy and self.logx and not self.box:
                ax.loglog()
            elif logy and not self.box:
                ax.semilogy()
            elif self.logx and not self.box:
                ax.semilogx()
            ax.set_ylabel(met, color=_c[met], fontsize="small")
            ax.grid(which="both")
            # sns.despine(ax=ax, bottom=self.diff)
            if self.diff:
                mean = df[met].mean()
                lines = [Line2D([0, 1], [0, 0], color=_c[met])]
                labels = [f"{vs.a_name}-{vs.b_name} ({mean:.2f})"]
                ax.legend(
                    handles=lines,
                    labels=labels,
                    fancybox=False,
                    loc=("upper" if _o[met] < 0 else "lower") + " right",
                    **_legkw,
                )
            else:
                means = [df[df[vs.sorter_var] == s][met].mean() for s in vs.other_names]
                labels = [f"{n} ({m:.2f})" for n, m in zip(vs.other_names, means)]
                ax.legend(
                    handles=lines,
                    labels=labels,
                    fancybox=False,
                    loc=("upper" if _o[met] < 0 else "lower") + " right",
                    **_legkw,
                )
        ax.set_xlabel(self.x)

        if self.diff and self.box and self.box_x_cuts is None:
            axes.flat[0].set_title("perf diff boxplots")
        elif self.diff and self.box:
            axes.flat[0].set_title("amp binned perf diff boxplots")
        elif self.box and self.box_x_cuts is None:
            axes.flat[0].set_title("perf boxplots")
        elif self.box:
            axes.flat[0].set_title("amp binned perf boxplots")
        elif self.diff:
            axes.flat[0].set_title(f"perf diff scatter vs. {self.x}")
        else:
            axes.flat[0].set_title(f"perf scatter vs. {self.x}")


class OrderedPerformance(VersusPlot):
    """Inspired by spikeinterface.benchmark.benchmark_plot_tools.plot_performances_ordered"""

    def __init__(self, metrics=default_metrics, hmul=1.0):
        self.nmetrics = len(metrics)
        self.metrics = metrics
        self.height = hmul * self.nmetrics

    def draw(self, panel, vs):
        axes = panel.subplots(nrows=self.nmetrics, sharex=True)
        df = vs.unit_versus_dataframe()
        x = np.arange(vs.n_gt_units)

        for ax, met in zip(axes.flat, self.metrics):
            for sorter, color in zip(vs.other_names, glasbey1024):
                y = df[df[vs.sorter_var] == sorter][met]
                y = y[np.argsort(-_o[met] * y)]
                ax.step(x, y, color=color, lw=1, label=sorter)
            ax.grid(which="both")
            ax.set_ylabel(met, color=_c[met])
            if met == "min_temp_dist":
                ax.semilogy()
        ax.set_xlabel("ordered GT units")
        axes.flat[-1].legend(loc="upper left", fancybox=False, **_legkw)
        axes.flat[0].set_title("sorted performance")


def get_versus_plots(vs) -> list[VersusPlot]:
    plots = [
        MetricColumn(),
        MetricColumn(diff=True),
        MetricColumn(box=True),
        MetricColumn(box=True, box_x_cuts=[5, 10, 15, 20]),
        MetricColumn(box=True, box_x_cuts=[5, 10, 15, 20], diff=True),
        OrderedPerformance(),
    ]
    if not vs.is_two:
        plots = [p for p in plots if not p.two_only]
    return plots


def make_versus_summary(
    vs: DARTsortGTVersus,
    plots: Sequence[VersusPlot] | None=None,
    max_height=8,
    figsize=(10, 10),
    figure=None,
    suptitle=True,
):
    if plots is None:
        plots = get_versus_plots(vs)

    figure = flow_layout(
        plots,
        max_height=max_height,
        figsize=figsize,
        figure=figure,
        vs=vs,
    )
    if suptitle is True:
        vs_str = " vs. ".join(vs.other_names)
        figure.suptitle(f"{vs.gt_name} compare: {vs_str}")
    elif suptitle:
        figure.suptitle(suptitle)

    return figure
