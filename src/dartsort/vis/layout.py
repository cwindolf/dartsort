from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np


class BasePlot:
    kind: str
    width = 1
    height = 1

    def draw(self, *args, **kwargs):
        raise NotImplementedError

    def notify_global_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)


class BaseMultiPlot:
    def plots(self):
        # return [BasePlot()]
        raise NotImplementedError

    def notify_global_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)


Card = namedtuple("Card", ["kind", "width", "height", "plots"])


def flow_layout(
    plots,
    max_height=4,
    figsize=(8.5, 11),
    figure=None,
    hspace=0.1,
    **plot_kwargs
):
    columns = flow_layout_columns(plots, max_height=max_height, **plot_kwargs)

    # -- draw the figure
    width_ratios = [column[0].width for column in columns]
    if figure is None:
        figure = plt.figure(figsize=figsize) #, layout="constrained") ? needed
    subfigures = figure.subfigures(
        nrows=1,
        ncols=len(columns),
        hspace=hspace,
        width_ratios=width_ratios,
        squeeze=False,
    )
    all_panels = subfigures[0].tolist()
    for column, subfig in zip(columns, subfigures[0]):
        n_cards = len(column)
        height_ratios = [card.height for card in column]
        remaining_height = max_height - sum(height_ratios)
        if remaining_height > 0:
            height_ratios.append(remaining_height)

        cardfigs = subfig.subfigures(
            nrows=n_cards + (remaining_height > 0),
            ncols=1,
            height_ratios=height_ratios,
            hspace=0.1,
        )
        cardfigs = np.atleast_1d(cardfigs)
        all_panels.extend(cardfigs)

        for cardfig, card in zip(cardfigs, column):
            panels = cardfig.subfigures(
                nrows=len(card.plots),
                ncols=1,
                height_ratios=[p.height for p in card.plots],
                hspace=hspace,
            )
            panels = np.atleast_1d(panels)
            for plot, panel in zip(card.plots, panels):
                plot.draw(panel, **plot_kwargs)
            all_panels.extend(panels)

    # clean up the panels, or else things get clipped
    for panel in all_panels:
        panel.set_facecolor([0, 0, 0, 0])
        panel.patch.set_facecolor([0, 0, 0, 0])

    return figure


def flow_layout_columns(plots, max_height=4, **plot_kwargs):
    all_plots = []
    for plot in plots:
        # if isinstance(plot, BasePlot):
        all_plots.append(plot)
        # elif isinstance(plot, BaseMultiPlot):
        #     all_plots.extend(plot.plots(**plot_kwargs))
        # else:
        #     assert False
    plots = all_plots

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
                card_plots = [plot]
        if card_plots:
            cards.append(
                Card(
                    plots[0].kind,
                    width,
                    sum(p.height for p in card_plots),
                    card_plots,
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

        if (
            sum(c.height for c in columns[-1]) + card.height
            <= max_height + 1e-8
        ):
            columns[-1].append(card)
        else:
            columns.append([card])

    return columns
