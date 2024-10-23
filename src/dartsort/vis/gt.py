import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import FuncNorm
import warnings

from . import unit
from .colors import glasbey1024
from .layout import BasePlot, flow_layout
from .unit import UnitPlot, make_all_summaries, make_unit_summary
from .waveforms import geomplot


# -- single-unit


class GTUnitMixin:
    def draw(self, panel, comparison, unit_id):
        return super().draw(panel, comparison.gt_analysis, unit_id)


class PredictedUnitMixin:
    def draw(self, panel, comparison, tested_unit_id):
        return super().draw(panel, comparison.predicted_analysis, tested_unit_id)


class MatchUnitMixin:
    def draw(self, panel, comparison, unit_id):
        tested_unit_id = comparison.get_match(unit_id)
        return super().draw(panel, comparison.predicted_analysis, tested_unit_id)


class UnitComparisonPlot(BasePlot):
    kind = "unit_comparison"

    def draw(self, panel, comparison, unit_id):
        tested_unit_id = comparison.get_match(unit_id)
        return self._draw(panel, comparison, unit_id, tested_unit_id)

    def _draw(self, panel, comparison, unit_id, tested_unit_id):
        raise NotImplementedError


class MatchIsiComparison(UnitComparisonPlot):
    kind = "histogram"
    width = 2
    height = 2

    def __init__(self, bin_ms=0.1, max_ms=5):
        self.unit_isi = unit.ISIHistogram(bin_ms=bin_ms, max_ms=max_ms)

    def _draw(self, panel, comparison, unit_id, tested_unit_id):
        axis = panel.subplots()
        self.unit_isi.draw(panel, comparison.gt_analysis, unit_id, axis=axis, label=comparison.gt_name)
        self.unit_isi.draw(panel, comparison.tested_analysis, tested_unit_id, color=glasbey1024[tested_unit_id], axis=axis, label=comparison.tested_name)
        axis.set_ylabel("count")
        axis.legend(loc="upper left")


class GTUnitTextInfo(UnitComparisonPlot):
    kind = "_info"
    width = 2
    height = 2

    def _draw(self, panel, comparison, unit_id, tested_unit_id):
        axis = panel.subplots()
        axis.axis("off")
        msg = f"GT unit: {unit_id}\n"
        msg += f"Hungarian matched unit: {tested_unit_id}\n"

        gt_nspikes = comparison.gt_analysis.spike_counts[
            comparison.gt_analysis.unit_ids == unit_id
        ].sum()
        tested_nspikes = comparison.tested_analysis.spike_counts[
            comparison.tested_analysis.unit_ids == tested_unit_id
        ].sum()
        msg += f"{gt_nspikes} spikes in GT unit\n"
        msg += f"{tested_nspikes} spikes in matched unit\n"

        gt_temp = comparison.gt_analysis.coarse_template_data.unit_templates(unit_id)
        tested_temp = comparison.tested_analysis.coarse_template_data.unit_templates(tested_unit_id)
        gt_ptp = np.ptp(gt_temp, 1).max(1).squeeze()
        assert gt_ptp.size == 1
        tested_ptp = np.ptp(tested_temp, 1).max(1).squeeze()
        assert tested_ptp.size == 1
        msg += f"GT PTP: {gt_ptp:0.1f}; matched PTP: {tested_ptp:0.1f}\n"
        msg += f"{tested_nspikes} spikes in matched unit\n"

        inds = comparison.get_spikes_by_category(unit_id)
        tp = inds['matched_gt_indices'].size
        fn = inds['only_gt_indices'].size
        fp = inds['only_tested_indices'].size
        acc = tp / (tp + fn + fp)
        rec = tp / (tp + fn)
        prec = tp / (tp + fp)
        fdr = fp / (tp + fp)
        acc = f"{100*acc:0.1f}".rstrip("0").rstrip(".")
        rec = f"{100*rec:0.1f}".rstrip("0").rstrip(".")
        prec = f"{100*prec:0.1f}".rstrip("0").rstrip(".")
        fdr = f"{100*fdr:0.1f}".rstrip("0").rstrip(".")
        msg += f"acc (tpr)={acc}%\nrecall={rec}%\nprec={prec}%\nfdr={fdr}%"

        axis.text(0, 0, msg, fontsize=6.5)


class MatchVennPlot(UnitComparisonPlot):
    width = 3
    height = 2

    def __init__(self, gt_color="r", matched_color="gold", tested_color="b"):
        self.gt_color = gt_color
        self.matched_color = matched_color
        self.tested_color = tested_color

    def _draw(self, panel, comparison, unit_id, tested_unit_id):
        import matplotlib_venn
        inds = comparison.get_spikes_by_category(unit_id)
        tp = inds['matched_gt_indices'].size
        fn = inds['only_gt_indices'].size
        fp = inds['only_tested_indices'].size
        sizes = {'11': tp, '10': fn, '01': fp}

        ax = panel.subplots()
        v = matplotlib_venn.venn2(
            sizes,
            set_colors=(self.gt_color, self.tested_color),
            set_labels=(f"{comparison.gt_name}#{unit_id}", f"{comparison.tested_name}#{inds['tested_unit']}"),
            ax=ax,
        )
        v.get_patch_by_id('11').set_color(self.matched_color)


class MatchWaveformsPlot(UnitComparisonPlot):
    kind = "waveforms"

    def __init__(self, gt_color="r", alpha=0.25, matched_color="gold", tested_color="b", channel_show_radius_um=25, count=50, width=3, height=3, single=False):
        self.colors = dict(tp=matched_color, fp=tested_color, fn=gt_color)
        self.channel_show_radius_um = channel_show_radius_um
        self.count = count
        self.width = width
        self.height = height
        self.single = single
        self.alpha = alpha

    def _draw(self, panel, comparison, unit_id, tested_unit_id):
        rad = self.channel_show_radius_um
        if self.single:
            rad = 0
        w = comparison.get_raw_waveforms_by_category(
            unit_id,
            tested_unit=tested_unit_id,
            max_samples_per_category=self.count,
            random_seed=0,
            channel_show_radius_um=rad,
        )

        waveforms = []
        colors = []
        for kind, color in self.colors.items():
            if not w[kind].size:
                continue

            waveforms.append(w[kind])
            colors.append(np.broadcast_to([color], w[kind].shape[:1]))
        waveforms = np.concatenate(waveforms)
        colors = np.concatenate(colors)
        max_channels = np.broadcast_to([w['max_chan']], colors.shape)

        ax = panel.subplots()
        geomplot(
            waveforms,
            max_channels=max_channels,
            channel_index=w['channel_index'],
            geom=w['geom'],
            ax=ax,
            show_zero=False,
            max_abs_amp=None,
            annotate_z=True,
            subar=True,
            colors=colors,
            alpha=self.alpha,
        )
        ax.axis("off")

class TemplateDistanceHistogram(UnitComparisonPlot):
    """All tested units' distances to a GT unit's template"""

    kind = "histogram"

    def draw(self, panel, comparison, unit_id):
        ax = panel.subplots()
        ax.hist(comparison.template_distances[unit_id], histtype='step')
        ax.set_xlabel("tested unit's template distance to GT template")


class NearbyTemplates(UnitComparisonPlot):
    kind = "waveforms"

    def __init__(self, channel_show_radius_um=25, n_neighbors=5, width=3, height=3, single=False):
        self.width = width
        self.height = height
        self.channel_show_radius_um = channel_show_radius_um
        self.single = single
        self.n_neighbors = n_neighbors

    def draw(self, panel, comparison, unit_id):
        neighb_ids, neighb_dists, neighb_coarse_templates = comparison.nearby_tested_templates(
            unit_id, n_neighbors=self.n_neighbors
        )
        max_chan = comparison.gt_analysis.unit_max_channel(unit_id)
        geom = comparison.gt_analysis.show_geom
        rad = self.channel_show_radius_um
        if self.single:
            rad = 0
        channel_index = comparison.gt_analysis.show_channel_index(rad)
        channels = channel_index[max_chan]
        channels = channels[channels < len(geom)]

        gt_template = comparison.gt_analysis.coarse_template_data.unit_templates(unit_id)

        templates_vis = np.concatenate((neighb_coarse_templates, gt_template))[..., channels]
        colors_vis = np.concatenate((glasbey1024[neighb_ids % len(glasbey1024)], np.zeros_like(glasbey1024[:1])))
        channels_vis = np.broadcast_to(channels[None], (len(colors_vis), channels.size))

        ax = panel.subplots()
        geomplot(
            templates_vis,
            channels=channels_vis,
            geom=geom,
            ax=ax,
            show_zero=False,
            subar=not self.single,
            colors=colors_vis,
        )
        ax.axis("off")
        ax.set_title(f"nearby {comparison.tested_name} templates")


class NearbyTemplatesDistanceMatrix(UnitComparisonPlot):
    kind = "amatrix"
    width = 3
    height = 3

    def __init__(self, n_neighbors=5, cmap=plt.cm.magma_r):
        self.n_neighbors = n_neighbors
        self.cmap = cmap

    def draw(self, panel, comparison, unit_id):
        gt_neighb_ids, gt_neighb_dists, gt_neighb_coarse_templates = comparison.gt_analysis.nearby_coarse_templates(
            unit_id, n_neighbors=self.n_neighbors
        )
        tested_neighb_ids, tested_neighb_dists, tested_neighb_coarse_templates = comparison.nearby_tested_templates(
            unit_id, n_neighbors=self.n_neighbors
        )
        dists = comparison.template_distances[gt_neighb_ids][:, tested_neighb_ids]
        ax = panel.subplots()
        log1p_norm = FuncNorm((np.log1p, np.expm1), vmin=0)
        im = ax.imshow(dists, norm=log1p_norm, cmap=self.cmap)
        plt.colorbar(im, ax=ax, shrink=0.3)
        ax.set_ylabel(f"{comparison.gt_name} unit")
        ax.set_xlabel(f"{comparison.tested_name} unit")
        ax.set_yticks(range(len(dists)), gt_neighb_ids)
        ax.set_xticks(range(len(dists.T)), tested_neighb_ids)
        for t, ix in zip(ax.get_xticklabels(), tested_neighb_ids):
            t.set_color(glasbey1024[int(ix) % len(glasbey1024)])
        ax.set_title(f"nearby gt/tested temp dist")


class NearbyTemplatesConfusionMatrix(UnitComparisonPlot):
    kind = "amatrix"
    width = 3
    height = 3

    def __init__(self, n_neighbors=5, cmap=plt.cm.bone):
        self.n_neighbors = n_neighbors
        self.cmap = cmap

    def draw(self, panel, comparison, unit_id):
        gt_neighb_ids, gt_neighb_dists, gt_neighb_coarse_templates = comparison.gt_analysis.nearby_coarse_templates(
            unit_id, n_neighbors=self.n_neighbors
        )
        tested_neighb_ids, tested_neighb_dists, tested_neighb_coarse_templates = comparison.nearby_tested_templates(
            unit_id, n_neighbors=self.n_neighbors
        )
        conf = comparison.comparison.get_confusion_matrix()
        conf_row_labels = np.array([int(c) if c != 'FP' else 1_000_000 for c in conf.index])
        conf_rows = np.searchsorted(conf_row_labels, gt_neighb_ids)
        assert np.array_equal(conf.index[conf_rows], gt_neighb_ids)
        conf_col_labels = np.array([int(c) if c != 'FN' else 1_000_000 for c in conf.columns])
        col_order = np.argsort(conf_col_labels)
        conf_cols = np.searchsorted(conf_col_labels, tested_neighb_ids, sorter=col_order)
        conf_cols = col_order[conf_cols]
        assert np.array_equal(conf_col_labels[conf_cols], tested_neighb_ids)
        conf = conf.values[conf_rows][:, conf_cols]
        ax = panel.subplots()
        im = ax.imshow(conf, cmap=self.cmap)
        plt.colorbar(im, ax=ax, shrink=0.3)
        ax.set_ylabel(f"{comparison.gt_name} unit")
        ax.set_xlabel(f"{comparison.tested_name} unit")
        ax.set_yticks(range(len(conf)), gt_neighb_ids)
        ax.set_xticks(range(len(conf.T)), tested_neighb_ids)
        for t, ix in zip(ax.get_xticklabels(), tested_neighb_ids):
            t.set_color(glasbey1024[int(ix) % len(glasbey1024)])
        ax.set_title(f"nearby gt/tested confusion")


default_unit_comparison_plots = (
    GTUnitTextInfo(),
    MatchIsiComparison(),
    MatchVennPlot(),
    NearbyTemplatesDistanceMatrix(),
    NearbyTemplatesConfusionMatrix(),
    MatchWaveformsPlot(),
    MatchWaveformsPlot(single=True, width=2, height=1),
    # MatchWaveformsPlot(average=True, ),
    # MatchWaveformsPlot(average=True, single=True, width=2, height=1),
    NearbyTemplates(),
    NearbyTemplates(single=True, width=2, height=1),
)


def make_unit_comparison(
    comparison,
    unit_id,
    plots=default_unit_comparison_plots,
    max_height=6,
    figsize=(11, 8.5),
    hspace=0.1,
    figure=None,
    **other_global_params,
):
    return make_unit_summary(
        comparison,
        unit_id,
        plots=plots,
        max_height=max_height,
        figsize=figsize,
        hspace=hspace,
        figure=figure,
        gizmo_name="comparison",
        **other_global_params,
    )


def make_all_unit_comparisons(
    comparison,
    save_folder,
    plots=default_unit_comparison_plots,
    max_height=6,
    figsize=(11, 8.5),
    hspace=0.1,
    dpi=200,
    image_ext="png",
    n_jobs=0,
    show_progress=True,
    overwrite=False,
    unit_ids=None,
    **other_global_params,
):
    return make_all_summaries(
        comparison, save_folder, plots=plots,
        max_height=max_height,
        figsize=figsize,
        hspace=hspace,
        dpi=dpi,
        image_ext=image_ext,
        n_jobs=n_jobs,
        show_progress=show_progress,
        overwrite=overwrite,
        unit_ids=unit_ids,
        gizmo_name="comparison",
        **other_global_params,
    )


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
        ax.set_title("all agreements")
        ax.set_ylabel(f"{comparison.gt_name} unit")
        ax.set_xlabel(f"{comparison.tested_name} unit")


class TrimmedAgreementMatrix(ComparisonPlot):
    kind = "matrix"
    width = 3
    height = 2

    def draw(self, panel, comparison):
        agreement = comparison.comparison.get_ordered_agreement_scores()
        ax = panel.subplots()
        im = ax.imshow(agreement.values[:, :agreement.shape[0]], vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.3)
        ax.set_title("Hung. match agreements")
        ax.set_ylabel(f"{comparison.gt_name} unit")
        ax.set_xlabel(f"{comparison.tested_name} unit")


class MetricRegPlot(ComparisonPlot):
    kind = "gtmetric"
    width = 2
    height = 2

    def __init__(self, x="gt_ptp_amplitude", y="accuracy", color="b", log_x=False):
        self.x = x
        self.y = y
        self.color = color
        self.log_x = log_x

    def draw(self, panel, comparison):
        ax = panel.subplots()
        df = comparison.unit_info_dataframe()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            df_show = df[np.isfinite(df[self.y].values)]
            df_show = df_show[np.isfinite(df_show[self.x].values)]
            sns.regplot(
                data=df_show,
                x=self.x,
                y=self.y,
                logistic=True,
                # logx=self.log_x,
                color=self.color,
                ax=ax,
            )
        if self.log_x:
            ax.semilogx()
        met = df[self.y].mean()
        ax.set_title(f"mean {self.y}: {met:.3f}", fontsize="small")


class MetricDistribution(ComparisonPlot):
    kind = "wide"

    def __init__(self, xs=("recall", "accuracy", "temp_dist", "precision"), colors=("r", "b", "orange", "g"), flavor="hist", width=3, height=2):
        self.xs = list(xs)
        self.colors = colors
        self.flavor = flavor
        self.width = width
        self.height = height

    def draw(self, panel, comparison):
        ax = panel.subplots()
        df = comparison.unit_info_dataframe()
        df = df[self.xs].melt(value_vars=self.xs, var_name='metric')
        if self.flavor == "hist":
            sns.histplot(
                data=df,
                x='value',
                hue='metric',
                palette=list(self.colors),
                element='step',
                ax=ax,
                bins=np.linspace(0, 1, 21),
            )
            sns.move_legend(ax, 'upper left', frameon=False)
        elif self.flavor == "box":
            sns.boxplot(
                data=df,
                x='metric',
                y='value',
                hue='metric',
                palette=list(self.colors),
                ax=ax,
                legend=False,
            )
            ax.tick_params(axis="x", rotation=90)
            ax.set_ylim([-0.05, 1.05])
            ax.set(xlabel=None, ylabel=None)


class TemplateDistanceMatrix(ComparisonPlot):
    kind = "wide"
    width = 3
    height = 1

    def __init__(self, cmap=plt.cm.magma_r):
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


class TrimmedTemplateDistanceMatrix(ComparisonPlot):
    kind = "matrix"
    width = 3
    height = 2

    def __init__(self, cmap=plt.cm.magma_r):
        self.cmap = cmap

    def draw(self, panel, comparison):
        agreement = comparison.comparison.get_ordered_agreement_scores()
        row_order = agreement.index
        dist = comparison.template_distances[row_order, :]

        ax = panel.subplots()
        log1p_norm = FuncNorm((np.log1p, np.expm1), vmin=0)
        im = ax.imshow(dist, norm=log1p_norm, cmap=self.cmap)
        plt.colorbar(im, ax=ax, shrink=0.3)
        ax.set_title("Hung. match temp dists")
        ax.set_ylabel(f"{comparison.gt_name} unit")
        ax.set_xlabel(f"{comparison.tested_name} unit")

box = MetricDistribution(flavor="box", width=2, height=3.5)
box.kind = "gtmetric"
gt_overview_plots = (
    MetricRegPlot(x="gt_ptp_amplitude", y="accuracy", log_x=True),
    MetricRegPlot(x="gt_ptp_amplitude", y="recall", color="r", log_x=True),
    MetricRegPlot(x="gt_ptp_amplitude", y="precision", color="g", log_x=True),
    MetricRegPlot(x="gt_firing_rate", y="accuracy"),
    MetricRegPlot(x="gt_firing_rate", y="recall", color="r"),
    MetricRegPlot(x="gt_firing_rate", y="precision", color="g"),
    MetricRegPlot(x="temp_dist", y="precision", color="g", log_x=True),
    MetricRegPlot(x="gt_ptp_amplitude", y="temp_dist", color="orange", log_x=True),
    MetricRegPlot(x="gt_firing_rate", y="temp_dist", color="orange"),
    MetricRegPlot(x="gt_ptp_amplitude", y="unsorted_recall", color="purple", log_x=True),
    box,
    MetricDistribution(),
    TrimmedAgreementMatrix(),
    TrimmedTemplateDistanceMatrix(),
)

gt_overview_plots_no_temp_dist = (
    MetricRegPlot(x="gt_ptp_amplitude", y="accuracy", log_x=True),
    MetricRegPlot(x="gt_ptp_amplitude", y="recall", color="r", log_x=True),
    MetricRegPlot(x="gt_ptp_amplitude", y="precision", color="g", log_x=True),
    MetricRegPlot(x="gt_firing_rate", y="accuracy"),
    MetricRegPlot(x="gt_firing_rate", y="recall", color="r"),
    MetricRegPlot(x="gt_firing_rate", y="precision", color="g"),
    MetricRegPlot(x="gt_ptp_amplitude", y="unsorted_recall", color="purple", log_x=True),
    box,
    MetricDistribution(xs=("recall", "accuracy", "temp_dist", "precision")),
    TrimmedAgreementMatrix(),
)

# multi comparisons stuff
# box and whisker between sorters



def make_gt_overview_summary(
    comparison,
    plots=gt_overview_plots,
    max_height=6,
    figsize=(11, 8.5),
    figure=None,
    suptitle=True,
    same_width_flow=True,
):
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
