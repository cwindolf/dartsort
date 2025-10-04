import matplotlib.pyplot as plt
import numpy as np

try:
    from matplotlib_venn import venn2
except ImportError:

    def venn2(*args, **kwargs):
        raise ImportError("`matplotlib_venn` is needed for venn plots.")


from .colors import glasbey1024
from .layout import BasePlot
from .unit import make_all_summaries, make_unit_summary
from .waveforms import geomplot


# -- single-unit


class UnitComparisonPlot(BasePlot):
    kind = "unit_comparison"

    def draw(self, panel, comparison, unit_id):
        tested_unit_id = comparison.get_match(unit_id)
        return self._draw(panel, comparison, unit_id, tested_unit_id)

    def _draw(self, panel, comparison, unit_id, tested_unit_id):
        raise NotImplementedError


class GTUnitTextInfo(UnitComparisonPlot):
    kind = "_info"
    width = 2
    height = 4

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
        tested_temp = comparison.tested_analysis.coarse_template_data.unit_templates(
            tested_unit_id
        )
        gt_ptp = np.ptp(gt_temp, 1).max(1).squeeze()
        assert gt_ptp.size == 1
        tested_ptp = np.ptp(tested_temp, 1).max(1).squeeze()
        assert tested_ptp.size == 1
        msg += f"GT PTP: {gt_ptp:0.1f}; matched PTP: {tested_ptp:0.1f}\n"
        msg += f"{tested_nspikes} spikes in matched unit\n"

        inds = comparison.get_spikes_by_category(unit_id)
        tp = inds["matched_gt_indices"].size
        fn = inds["only_gt_indices"].size
        fp = inds["only_tested_indices"].size
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
        inds = comparison.get_spikes_by_category(unit_id)
        tp = inds["matched_gt_indices"].size
        fn = inds["only_gt_indices"].size
        fp = inds["only_tested_indices"].size
        sizes = {"11": tp, "10": fn, "01": fp}

        ax = panel.subplots()
        v = venn2(
            sizes,
            set_colors=(self.gt_color, self.tested_color),
            set_labels=(
                f"{comparison.gt_name}#{unit_id}",
                f"{comparison.tested_name}#{inds['tested_unit']}",
            ),
            ax=ax,
        )
        v.get_patch_by_id("11").set_color(self.matched_color)


class UnsortedVennPlot(UnitComparisonPlot):
    width = 3
    height = 2

    def __init__(self, unsorted_matched_color="purple", unsorted_missed_color="darkorange"):
        self.unsorted_matched_color = unsorted_matched_color
        self.unsorted_missed_color = unsorted_missed_color

    def _draw(self, panel, comparison, unit_id, tested_unit_id):
        inds = comparison.get_spikes_by_category(unit_id)
        tp = inds["unsorted_tp_indices"].size
        fn = inds["unsorted_fn_indices"].size
        fp = 0  # can't assess here
        sizes = {"11": tp, "10": fn, "01": fp}

        ax = panel.subplots()
        v = venn2(
            subsets=sizes,
            set_colors=(self.unsorted_missed_color, 'k'),
            set_labels=(
                f"{comparison.gt_name}#{unit_id}",
                f"{comparison.tested_name}#unsorted",
            ),
            ax=ax,
        )
        v.get_patch_by_id("11").set_color(self.unsorted_matched_color)


class MatchRawWaveformsPlot(UnitComparisonPlot):
    def __init__(
        self,
        alpha=0.25,
        gt_color="r",
        matched_color="gold",
        tested_color="b",
        unsorted_matched_color="purple",
        unsorted_missed_color="darkorange",
        channel_show_radius_um=25,
        count=50,
        width=3,
        height=3,
        single_channel=False,
        randomize=True,
        show_sorted_matches=True,
        show_unsorted_matches=False,
    ):
        self.colors = {}
        if show_sorted_matches:
            self.colors["tp"] = matched_color
            self.colors["fp"] = tested_color
            self.colors["fn"] = gt_color
        if show_unsorted_matches:
            self.colors["unsorted_tp"] = unsorted_matched_color
            self.colors["unsorted_fn"] = unsorted_missed_color

        self.radius = 0 if single_channel else channel_show_radius_um
        self.count = count
        self.alpha = alpha
        self.randomize = randomize

        self.kind = "traces" if single_channel else "waveforms"
        self.width = width
        self.height = height

    def _draw(self, panel, comparison, unit_id, tested_unit_id):
        w = comparison.get_raw_waveforms_by_category(
            unit_id,
            tested_unit=tested_unit_id,
            max_samples_per_category=self.count,
            random_seed=0,
            channel_show_radius_um=self.radius,
        )

        waveforms = []
        colors = []
        for kind, color in self.colors.items():
            if w[kind] is None or not w[kind].size:
                continue
            waveforms.append(w[kind])
            colors.append(np.broadcast_to([color], w[kind].shape[:1]))

        waveforms = np.concatenate(waveforms)
        colors = np.concatenate(colors)
        max_channels = np.broadcast_to([w["max_chan"]], colors.shape)

        ax = panel.subplots()
        geomplot(
            waveforms,
            max_channels=max_channels,
            channel_index=w["channel_index"],
            geom=w["geom"],
            ax=ax,
            show_zero=False,
            max_abs_amp=None,
            annotate_z=True,
            subar=True,
            colors=colors,
            alpha=self.alpha,
            randomize=self.randomize,
        )
        ax.axis("off")


class TemplateDistanceHistogram(UnitComparisonPlot):
    """All tested units' distances to a GT unit's template"""

    kind = "histogram"

    def _draw(self, panel, comparison, unit_id, tested_unit_id):
        ax = panel.subplots()
        d = np.nan_to_num(comparison.template_distances, nan=np.inf)
        vm = min(d.min(0).max(), d.min(1).max())
        bins = np.logspace(np.log10(d.min()), np.log10(vm), 96)
        x = d[unit_id]
        x = x[np.isfinite(x)]
        ax.hist(x, color="orange", bins=bins)
        ax.grid(which='both')
        ax.semilogx()
        ax.set_xlabel("tested template distance to GT")
        ax.set_ylabel("count")
        ax.axvline(d[unit_id][tested_unit_id], c='k', label='Hung. match dist.')
        ax.legend(frameon=False, loc='upper right')


class NearbyTemplates(UnitComparisonPlot):
    def __init__(
        self,
        channel_show_radius_um=25,
        n_neighbors=5,
        width=3,
        height=3,
        single_channel=False,
    ):
        self.width = width
        self.height = height
        self.channel_show_radius_um = channel_show_radius_um
        self.single_channel = single_channel
        self.n_neighbors = n_neighbors
        self.kind = "traces" if single_channel else "waveforms"

    def draw(self, panel, comparison, unit_id):
        (
            neighb_ids,
            neighb_dists,
            neighb_coarse_templates,
        ) = comparison.nearby_tested_templates(unit_id, n_neighbors=self.n_neighbors)
        max_chan = comparison.gt_analysis.unit_max_channel(unit_id)
        geom = comparison.gt_analysis.show_geom
        rad = self.channel_show_radius_um
        if self.single_channel:
            rad = 0
        channel_index = comparison.gt_analysis.show_channel_index(rad)
        channels = channel_index[max_chan]
        channels = channels[channels < len(geom)]

        gt_template = comparison.gt_analysis.coarse_template_data.unit_templates(
            unit_id
        )

        templates_vis = np.concatenate((neighb_coarse_templates, gt_template))[
            ..., channels
        ]
        colors_vis = np.concatenate(
            (glasbey1024[neighb_ids % len(glasbey1024)], np.zeros_like(glasbey1024[:1]))
        )
        channels_vis = np.broadcast_to(channels[None], (len(colors_vis), channels.size))

        ax = panel.subplots()
        geomplot(
            templates_vis,
            channels=channels_vis,
            geom=geom,
            ax=ax,
            show_zero=False,
            subar=not self.single_channel,
            colors=colors_vis,
        )
        ax.axis("off")
        ax.set_title(f"nearby {comparison.tested_name} templates")


class NearbyTemplatesDistanceMatrix(UnitComparisonPlot):
    kind = "amatrix"
    width = 3
    height = 3

    def __init__(self, n_neighbors=5, cmap="magma"):
        self.n_neighbors = n_neighbors
        self.cmap = plt.get_cmap(cmap)

    def draw(self, panel, comparison, unit_id):
        (
            gt_neighb_ids,
            gt_neighb_dists,
            gt_neighb_coarse_templates,
        ) = comparison.gt_analysis.nearby_coarse_templates(
            unit_id, n_neighbors=self.n_neighbors
        )
        (
            tested_neighb_ids,
            tested_neighb_dists,
            tested_neighb_coarse_templates,
        ) = comparison.nearby_tested_templates(unit_id, n_neighbors=self.n_neighbors)
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

    def __init__(self, n_neighbors=5, cmap="bone"):
        self.n_neighbors = n_neighbors
        self.cmap = plt.get_cmap(cmap)

    def draw(self, panel, comparison, unit_id):
        (
            gt_neighb_ids,
            gt_neighb_dists,
            gt_neighb_coarse_templates,
        ) = comparison.gt_analysis.nearby_coarse_templates(
            unit_id, n_neighbors=self.n_neighbors
        )
        (
            tested_neighb_ids,
            tested_neighb_dists,
            tested_neighb_coarse_templates,
        ) = comparison.nearby_tested_templates(unit_id, n_neighbors=self.n_neighbors)
        conf = comparison.comparison.get_confusion_matrix()
        conf_row_labels = np.array(
            [int(c) if c != "FP" else 1_000_000 for c in conf.index]
        )
        conf_rows = np.searchsorted(conf_row_labels, gt_neighb_ids)
        assert np.array_equal(conf.index[conf_rows], gt_neighb_ids)
        conf_col_labels = np.array(
            [int(c) if c != "FN" else 1_000_000 for c in conf.columns]
        )
        col_order = np.argsort(conf_col_labels)
        conf_cols = np.searchsorted(
            conf_col_labels, tested_neighb_ids, sorter=col_order
        )
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
    MatchVennPlot(),
    NearbyTemplatesDistanceMatrix(),
    NearbyTemplatesConfusionMatrix(),
    MatchRawWaveformsPlot(),
    MatchRawWaveformsPlot(single_channel=True, width=2, height=1),
    MatchRawWaveformsPlot(
        show_sorted_matches=False,
        show_unsorted_matches=True,
    ),
    MatchRawWaveformsPlot(
        show_sorted_matches=False,
        show_unsorted_matches=True,
        single_channel=True,
        width=2,
        height=1,
    ),
    NearbyTemplates(),
    NearbyTemplates(single_channel=True, width=2, height=1),
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
        comparison,
        save_folder,
        plots=plots,
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
