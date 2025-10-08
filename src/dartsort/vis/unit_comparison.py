import warnings
from logging import getLogger

import matplotlib.pyplot as plt
from matplotlib.colors import FuncNorm
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

try:
    from matplotlib_venn import venn2
except ImportError:

    def venn2(*args, **kwargs):
        raise ImportError("`matplotlib_venn` is needed for venn plots.")

from .analysis_plots import correlogram, stackbar
from .colors import glasbey1024
from .layout import BasePlot
from .unit import make_all_summaries, make_unit_summary
from .waveforms import geomplot


logger = getLogger(__name__)


_class_colors = {
    "tp": "darkorange",
    "fp": "b",
    "fn": "r",
    "unsorted_tp": "blueviolet",
    "unsorted_fn": "tan",
}

_nmeth_names = {
    "templates": "local",
    "siagreement": "sitop",
    "greedy": "greedytop",
}


# -- single-unit


class UnitComparisonPlot(BasePlot):
    kind = "unit_comparison"
    neighbor_method = "templates"
    n_neighbors = 5

    def draw(self, panel, comparison, unit_id):
        tested_unit_id = comparison.get_match(unit_id)
        return self._draw(panel, comparison, unit_id, tested_unit_id)

    def _draw(self, panel, comparison, unit_id, tested_unit_id):
        raise NotImplementedError

    def neighbors(self, comparison, unit_id, which="gt", method=None, n=None):
        """Gather neighboring tested or GT ids and templates for a GT unit."""
        # TODO: maybe this should be a method of the comparison? or it should
        # call one which has most of this logic? (nice to set neighbor_method
        # as instance properties of these plots)
        if method is None:
            method = self.neighbor_method
        if n is None:
            n = self.n_neighbors

        if method == "templates" and which == "gt":
            ids, dists, templates = comparison.nearby_gt_templates(unit_id, n_neighbors=n)
            del dists  # just naming it for clarity
        elif method == "templates" and which == "tested":
            ids, dists, templates = comparison.nearby_tested_templates(unit_id, n_neighbors=n)
            del dists
        elif method == "siagreement" and which == "gt":
            # gt units with top agreement to tested match
            tested_unit_id = comparison.get_match(unit_id)
            a = comparison.agreement_scores[tested_unit_id]
            ids = a.sort_values(ascending=False).index[:n].values if n else []
            if (ids == unit_id).any():
                ids = [unit_id] + list(ids[ids != unit_id])
            else:
                ids = [unit_id] + list(ids[:n - 1])
            templates = comparison.gt_analysis.coarse_template_data.unit_templates(ids)
        elif method == "siagreement" and which == "tested":
            # tested units with top agreement to gt
            a = comparison.agreement_scores.loc[unit_id]
            ids = a.sort_values(ascending=False).index[:n].values
            templates = comparison.tested_analysis.coarse_template_data.unit_templates(ids)
        elif method == "greedy" and which == "gt":
            tested_unit_id = comparison.get_match(unit_id)
            g = comparison.greedy_iou[:, tested_unit_id]
            ids = np.argsort(g)[::-1][:n - 1] if n else []
            if (ids == unit_id).any():
                ids = [unit_id] + list(ids[ids != unit_id])
            else:
                ids = [unit_id] + list(ids[:n - 1])
            templates = comparison.gt_analysis.coarse_template_data.unit_templates(ids)
        elif method == "greedy" and which == "tested":
            g = comparison.greedy_iou[unit_id]
            ids = np.argsort(g)[::-1][:n]
            templates = comparison.tested_analysis.coarse_template_data.unit_templates(ids)
        else:
            raise ValueError(f"Unknown neighbor method {method}.")

        return np.asarray(ids), templates

class GTUnitTextInfo(UnitComparisonPlot):
    kind = "_info"
    width = 3
    height = 2.5

    def _draw(self, panel, comparison, unit_id, tested_unit_id):
        axis = panel.subplots()
        axis.axis("off")
        msg = f"GT unit: {unit_id}\n"
        msg += f"Hung. match: {tested_unit_id}\n"
        best_match_id = comparison.get_best_match(unit_id)
        msg += f"Best match: {best_match_id}\n"
        if best_match_id != tested_unit_id:
            msg += " -!- Hungarian =/= best -!-\n"
        msg += "\n"

        gt_nspikes = comparison.gt_analysis.spike_counts[
            comparison.gt_analysis.unit_ids == unit_id
        ].sum()
        tested_nspikes = comparison.tested_analysis.spike_counts[
            comparison.tested_analysis.unit_ids == tested_unit_id
        ].sum()
        msg += f"{gt_nspikes} spikes in GT unit\n"
        msg += f"{tested_nspikes} spikes in matched unit\n"
        msg += "\n"
        
        gt_temp = comparison.gt_analysis.coarse_template_data.unit_templates(unit_id)
        gt_ptp = np.ptp(gt_temp, 1).max(1).squeeze()
        assert gt_ptp.size == 1
        msg += f"GT PTP: {gt_ptp:0.1f}\n"

        if tested_unit_id >= 0:
            tested_temp = comparison.tested_analysis.coarse_template_data.unit_templates(
                tested_unit_id
            )
            tested_ptp = np.ptp(tested_temp, 1).max(1).squeeze()
            msg += f"matched PTP: {tested_ptp:0.1f}\n"
        msg += "\n"

        inds = comparison.get_spikes_by_category(unit_id)
        tp = inds["matched_gt_indices"].size
        fn = inds["only_gt_indices"].size
        fp = inds["only_tested_indices"].size
        acc = tp / (tp + fn + fp)
        rec = tp / max((tp + fn), 1)
        prec = tp / max((tp + fp), 1)
        fdr = fp / max((tp + fp), 1)
        acc = f"{100*acc:0.1f}".rstrip("0").rstrip(".")
        rec = f"{100*rec:0.1f}".rstrip("0").rstrip(".")
        prec = f"{100*prec:0.1f}".rstrip("0").rstrip(".")
        fdr = f"{100*fdr:0.1f}".rstrip("0").rstrip(".")
        msg += f"acc (tpr)={acc}%\nrecall={rec}%\nprec={prec}%\nfdr={fdr}%"

        axis.text(0, 0, msg, fontsize=8)


class MatchVennPlot(UnitComparisonPlot):
    kind = "venn"
    width = 3
    height = 1.75

    def __init__(
        self,
        gt_color=None,
        matched_color=None,
        tested_color=None,
    ):
        self.gt_color = gt_color or _class_colors["fn"]
        self.matched_color = matched_color or _class_colors["tp"]
        self.tested_color = tested_color or _class_colors["fp"]

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
        if (p := v.get_patch_by_id("11")) is not None:
            p.set_color(self.matched_color)
        v.set_labels[0].set_color(self.gt_color)
        if len(v.set_labels) > 1:
            v.set_labels[1].set_color(self.tested_color)


class UnsortedVennPlot(UnitComparisonPlot):
    kind = "venn"
    width = 3
    height = 1.75

    def __init__(
        self,
        unsorted_matched_color=None,
        unsorted_missed_color=None,
    ):
        self.unsorted_matched_color = unsorted_matched_color or _class_colors["unsorted_tp"]
        self.unsorted_missed_color = unsorted_missed_color or _class_colors["unsorted_fn"]

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
                f"{comparison.tested_name}#unsort",
            ),
            ax=ax,
        )
        if (p := v.get_patch_by_id("11")) is not None:
            p.set_color(self.unsorted_matched_color)
        v.set_labels[0].set_color(self.unsorted_missed_color)
        if len(v.set_labels) > 1:
            v.set_labels[1].set_color(self.unsorted_matched_color)


class MatchRawWaveformsPlot(UnitComparisonPlot):
    def __init__(
        self,
        alpha=0.25,
        gt_color=None,
        matched_color=None,
        tested_color=None,
        unsorted_matched_color=None,
        unsorted_missed_color=None,
        channel_show_radius_um=35,
        count=50,
        single_channel=False,
        order="tprandom",
        show_sorted_matches=True,
        show_unsorted_matches=False,
        average=False,
    ):
        self.colors = {}
        self.show_sorted_matches = show_sorted_matches
        self.show_unsorted_matches = show_unsorted_matches
        if show_sorted_matches:
            self.colors["tp"] = matched_color or _class_colors["tp"]
            self.colors["fp"] = tested_color or _class_colors["fp"]
            self.colors["fn"] = gt_color or _class_colors["fn"]
        if show_unsorted_matches:
            self.colors["unsorted_tp"] = unsorted_matched_color or _class_colors["unsorted_tp"]
            self.colors["unsorted_fn"] = unsorted_missed_color or _class_colors["unsorted_fn"]

        self.radius = 0 if single_channel else channel_show_radius_um
        self.count = count
        self.alpha = alpha
        if order == "random":
            self.randomize = True
        if order == "tprandom":
            self.randomize = show_unsorted_matches
        elif order == "tpfpfn":
            # this ordering is because of insertion ordering of colors
            # dict above
            self.randomize = False
        else:
            assert False
        self.order = order

        self.kind = "traces" if single_channel else "waveforms"
        self.width = 2 if single_channel else 5
        self.height = 3 if single_channel else 5
        self.average = average

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
        maa = float("-inf")
        for kind, color in self.colors.items():
            if w[kind] is None or not w[kind].size:
                continue
            maa = max(maa, np.percentile(np.nanmax(np.abs(w[kind]), axis=(1, 2)), 90))
            if self.average:
                avg = w[kind].mean(0, keepdims=True)
                waveforms.append(avg)
            else:
                waveforms.append(w[kind])
            colors.append(np.broadcast_to([color], waveforms[-1].shape[:1]))

        if self.order == "tprandom" and not self.average and len(waveforms) > 1:
            wlast = np.concatenate(waveforms[1:])
            clast = np.concatenate(colors[1:])
            n = len(wlast)
            shuf = np.random.default_rng(0).permutation(n)
            waveforms = [waveforms[0], wlast[shuf]]
            colors = [colors[0], clast[shuf]]

        waveforms = np.concatenate(waveforms)
        colors = np.concatenate(colors)
        max_channels = np.broadcast_to([w["max_chan"]], colors.shape)

        ax = panel.subplots()
        max_abs_amp = maa
        chans = w["channel_index"][w["max_chan"]]
        chans = chans[chans < len(w["geom"])]
        geomplot(
            waveforms,
            channels=np.broadcast_to(chans[None], (len(waveforms), *chans.shape)),
            geom=w["geom"],
            ax=ax,
            show_zero=False,
            max_abs_amp=max_abs_amp,
            annotate_z=True,
            subar=True,
            colors=colors,
            alpha=1.0 if self.average else self.alpha,
            randomize=self.randomize,
        )
        ax.axis("off")
        handles = {
            k: Line2D([0, 1], [0, 0], color=v, lw=1)
            for k, v in self.colors.items()
        }
        if self.average and tested_unit_id >= 0:
            tested_template = comparison.tested_analysis.coarse_template_data.unit_templates(
                tested_unit_id
            )
            tested_template = tested_template[:, :, chans]
            testedline = geomplot(
                tested_template,
                channels=chans[None],
                geom=w["geom"],
                ax=ax,
                max_abs_amp=max_abs_amp,
                show_zero=False,
                subar=False,
                annotate_z=False,
                color=glasbey1024[tested_unit_id % len(glasbey1024)],
                linestyle=(1, (1, 1)),
            )
            tk = f"{comparison.tested_analysis.name}#{tested_unit_id}"
            handles[tk] = testedline
        if self.average:
            gt_template = comparison.gt_analysis.coarse_template_data.unit_templates(
                unit_id
            )
            gt_template = gt_template[:, :, chans]
            gtline = geomplot(
                gt_template,
                channels=chans[None],
                geom=w["geom"],
                ax=ax,
                max_abs_amp=max_abs_amp,
                show_zero=False,
                subar=False,
                annotate_z=False,
                color='k',
                linestyle=(0, (1, 1)),
            )
            handles['GT'] = gtline
                
        ax.legend(
            handles=handles.values(),
            labels=handles.keys(),
            fancybox=False,
            loc="upper center",
            borderpad=0.2,
            labelspacing=0.2,
            handlelength=1.0,
            columnspacing=1.0,
            frameon=False,
            ncols=min(3, len(handles)),
        )

        if self.show_unsorted_matches:
            k = "unsorted tp/fn"
        if self.show_sorted_matches:
            k = "sorted tp/fp/fn"
        a = ""
        if self.average:
            a = "avg "
        ax.set_title(f"{k} {a}waveforms")


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
        channel_show_radius_um=35.0,
        n_neighbors=5,
        single_channel=False,
        which="tested",
        neighbor_method="templates",
    ):
        self.channel_show_radius_um = channel_show_radius_um
        self.single_channel = single_channel
        self.n_neighbors = n_neighbors
        self.kind = "traces" if single_channel else "waveforms"
        self.width = 2 if single_channel else 4
        self.height = 2 if single_channel else 7
        self.which = which
        self.neighbor_method = neighbor_method

    def draw(self, panel, comparison, unit_id):
        neighb_ids, templates = self.neighbors(comparison, unit_id, which=self.which)
        # reverse so matched/gt unit comes last for draw order
        neighb_ids = neighb_ids[::-1]
        templates = templates[::-1]
        
        if self.which == "tested":
            sname = comparison.tested_name
            # stack on the GT template
            gt_template = comparison.gt_analysis.coarse_template_data.unit_templates(
                unit_id
            )
            templates = np.concatenate([templates, gt_template], axis=0)
            colors = [*glasbey1024[neighb_ids % len(glasbey1024)], 'k']
            labels = list(map(str, neighb_ids)) + [f"{comparison.gt_name}#{unit_id}"]
        elif self.which == "gt":
            sname = comparison.gt_name
            gt_template = templates[-1]
            colors = list(glasbey1024[neighb_ids[:-1] % len(glasbey1024)]) + ['k']
            labels = list(map(str, neighb_ids))
        else:
            assert False

        max_chan = comparison.gt_analysis.unit_max_channel(unit_id)
        geom = comparison.gt_analysis.show_geom
        rad = 0 if self.single_channel else self.channel_show_radius_um
        rad = float(rad) + 0
        channel_index = comparison.gt_analysis.show_channel_index(
            channel_show_radius_um=rad
        )
        channels = channel_index[max_chan]
        channels = channels[channels < len(geom)]
        templates = templates[:, :, channels]

        ax = panel.subplots()
        maa = np.nanmax(np.abs(gt_template))
        handles = []
        for t, c in zip(templates, colors):
            ls = '-'
            if isinstance(c, str) and c == 'k':
                ls = (0, (1, 1))
            lines = geomplot(
                t[None],
                channels=channels[None],
                geom=geom,
                ax=ax,
                show_zero=False,
                subar=not self.single_channel,
                color=c,
                zlim="tight",
                max_abs_amp=maa,
                linestyle=ls,
            )
            handles.append(lines)
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
        ns = _nmeth_names[self.neighbor_method]
        ax.set_title(f"{ns} {sname} templates")
        if not self.single_channel:
            ax.legend(
                handles=handles,
                labels=labels,
                fancybox=False,
                loc="upper center",
                borderpad=0.2,
                labelspacing=0.2,
                handlelength=1.0,
                columnspacing=1.0,
                frameon=False,
                ncols=min(3, len(templates)),
            )


class NearbyTemplatesDistanceMatrix(UnitComparisonPlot):
    kind = "minimatrix"
    width = 3
    height = 2.5

    def __init__(self, n_neighbors=5, cmap="magma", neighbor_method="templates"):
        self.n_neighbors = n_neighbors
        self.neighbor_method = neighbor_method
        self.cmap = plt.get_cmap(cmap)

    def draw(self, panel, comparison, unit_id):
        gt_neighb_ids, _ = self.neighbors(comparison, unit_id, which="gt")
        tested_neighb_ids, _ = self.neighbors(comparison, unit_id, which="tested")
        dists = comparison.template_distances[gt_neighb_ids][:, tested_neighb_ids]
        ax = panel.subplots()
        # log1p_norm = FuncNorm((np.log1p, np.expm1), vmin=0)
        sqrt_norm = FuncNorm((np.sqrt, np.square), vmin=0)
        im = ax.imshow(
            dists,
            # norm=log1p_norm,
            norm=sqrt_norm,
            cmap=self.cmap,
        )
        plt.colorbar(im, ax=ax, shrink=0.6)
        ax.set_ylabel(f"{comparison.gt_name} unit")
        ax.set_xlabel(f"{comparison.tested_name} unit")
        ax.set_yticks(range(len(dists)), gt_neighb_ids)
        ax.set_xticks(range(len(dists.T)), tested_neighb_ids, rotation="vertical")
        for t, ix in zip(ax.get_xticklabels(), tested_neighb_ids):
            t.set_color(glasbey1024[int(ix) % len(glasbey1024)])
        ns = _nmeth_names[self.neighbor_method]
        ax.set_title(f"{ns} temp dist")


class NearbyTemplatesConfusionMatrix(UnitComparisonPlot):
    kind = "minimatrix"
    width = 3
    height = 2.5

    def __init__(self, n_neighbors=5, cmap="pink", confusion_kind="siconfusion", neighbor_method="templates"):
        self.n_neighbors = n_neighbors
        self.cmap = plt.get_cmap(cmap)
        self.confusion_kind = confusion_kind
        self.neighbor_method = neighbor_method

    def draw(self, panel, comparison, unit_id):
        gt_neighb_ids, _ = self.neighbors(comparison, unit_id, which="gt")
        tested_neighb_ids, _ = self.neighbors(comparison, unit_id, which="tested")

        if self.confusion_kind == "siconfusion":
            conf = comparison.comparison.get_confusion_matrix()
        elif self.confusion_kind == "siagreement":
            conf = comparison.agreement_scores
        elif self.confusion_kind == "greedy":
            conf = comparison.greedy_confusion
            union = (conf.sum(0, keepdims=True) + conf.sum(1, keepdims=True)) - conf
            conf = conf[gt_neighb_ids][:, tested_neighb_ids]
            conf_rows = gt_neighb_ids
            conf_cols = tested_neighb_ids

        if self.confusion_kind.startswith("si"):
            conf = conf[conf.index != "FP"].sort_index()
            conf_row_labels = conf.index.values
            conf_rows = np.searchsorted(conf_row_labels, gt_neighb_ids, side="right") - 1
            assert np.array_equal(conf.index[conf_rows], gt_neighb_ids)

            conf_col_labels = np.array(
                [int(c) if c != "FN" else 1_000_000 for c in conf.columns]
            )
            col_order = np.argsort(conf_col_labels)
            conf_cols = np.searchsorted(
                conf_col_labels, tested_neighb_ids, sorter=col_order
            )
            conf_cols = col_order[conf_cols]
            # sometimes a nearest template has no spikes. sad but true.
            conf_cols_found = tested_neighb_ids == conf_cols
            conf_cols = conf_cols[conf_cols_found]
            tested_neighb_ids = conf_cols
            cfull = conf.values
            union = (cfull.sum(0, keepdims=True) + cfull.sum(1, keepdims=True)) - cfull
            conf = cfull[conf_rows][:, conf_cols]

        if self.confusion_kind in ("siconfusion", "greedy"):
            union = union[conf_rows][:, conf_cols]
            union[union == 0] = 1
            conf = conf / union
            suffix = " iou"
        else:
            suffix = ""

        conf = np.nan_to_num(conf)
        if conf.min() < -1e-3:
            warnings.warn(f"Large {conf.min()=} with {self.confusion_kind=}.")
        conf = np.abs(np.clip(conf, min=0.0))
        logger.info(f"{unit_id=} {self.confusion_kind=} {conf=}")
    
        ax = panel.subplots()
        sqrt_norm = FuncNorm((np.sqrt, np.square), vmin=0, vmax=max(conf.max(), 0.01))
        im = ax.imshow(conf, cmap=self.cmap, norm=sqrt_norm, interpolation="none")
        for (j, i), v in np.ndenumerate(conf):
            ax.text(i, j, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.4)
        ax.set_ylabel(f"{comparison.gt_name} unit")
        ax.set_xlabel(f"{comparison.tested_name} unit")
        ns = _nmeth_names[self.neighbor_method]
        ax.set_title(f"{ns} {self.confusion_kind}{suffix}", fontsize=10)
        ax.set_yticks(range(len(conf)), gt_neighb_ids)
        if len(tested_neighb_ids):
            ax.set_xticks(range(len(conf.T)), tested_neighb_ids, rotation="vertical")
            for t, ix in zip(ax.get_xticklabels(), tested_neighb_ids):
                t.set_color(glasbey1024[int(ix) % len(glasbey1024)])


class NeighborCCGBreakdown(UnitComparisonPlot):
    """Tested unit's CCG with nearby GT units, broken down by fp/fn."""
    kind = "ccg"
    width = 3

    def __init__(
        self,
        n_neighbors=5,
        neighbor_method="templates",
        categories=("fn", "fp"),
        max_lag=50,
    ):
        self.n_neighbors = n_neighbors
        self.neighbor_method = neighbor_method
        self.height = 1 + 1.5 * len(categories)
        self.categories = categories
        assert all(cat in _class_colors for cat in categories)
        self.max_lag = max_lag

    def _draw(self, panel, comparison, unit_id, tested_unit_id):
        gt_ids, _ = self.neighbors(comparison, unit_id, which="gt", n=self.n_neighbors + 1)
        gt_ids = gt_ids[1:]  # remove main.
        gta = comparison.gt_analysis
        ta = comparison.tested_analysis
        gt_sts = {u: gta.times_samples(gta.in_unit(u)) for u in gt_ids}
        cat_spikes = comparison.get_spikes_by_category(unit_id, tested_unit_id)
        colors = glasbey1024[gt_ids % len(glasbey1024)]

        axes = panel.subplots(nrows=len(self.categories), sharex=True)
        h = 1.5 * len(self.categories)
        for cat, ax in zip(self.categories, axes.flat):
            cat_st = cat_spikes[f"{cat}_times_samples"]
            ccgs = []
            for u, gt_st in gt_sts.items():
                clags, ccg = correlogram(cat_st, gt_st, max_lag=self.max_lag)
                ccgs.append(ccg)

            stackbar(ax, clags, ccgs, colors=colors, labels=gt_ids)
            sns.despine(ax=ax, left=True)
            if cat == self.categories[0]:
                ax.legend(
                    loc="lower center",
                    borderpad=0.2,
                    labelspacing=0.2,
                    handlelength=1.0,
                    columnspacing=1.0,
                    frameon=False,
                    ncols=min(3, len(gt_ids)),
                    bbox_to_anchor=(0, 1, h / (1 + h), 1 / h),
                )
            ax.grid(which='both')
            ax.axvline(0, lw=0.8, color='k', alpha=0.5)
            ax.set_ylabel(f'GTCCG v. {cat}', color=_class_colors[cat])
            if max(map(max, ccgs)) == 0:
                ax.set_yticks([])
        ax.set_xlabel('lag (samples)')
        ns = _nmeth_names[self.neighbor_method]
        cs = " / ".join(self.categories)
        panel.suptitle(f"{ns} GT CCGs for {cs}", fontsize=10)
        


def _get_default_unit_comparison_plots():
    return (
        GTUnitTextInfo(),
        MatchVennPlot(),
        UnsortedVennPlot(),
        NeighborCCGBreakdown(),
        NeighborCCGBreakdown(neighbor_method="siagreement"),
        NearbyTemplatesDistanceMatrix(),
        # NearbyTemplatesConfusionMatrix(),
        NearbyTemplatesConfusionMatrix(confusion_kind="siagreement"),
        NearbyTemplatesConfusionMatrix(confusion_kind="greedy"),
        NearbyTemplatesConfusionMatrix(neighbor_method="siagreement", confusion_kind="siagreement"),
        NearbyTemplatesConfusionMatrix(neighbor_method="siagreement", confusion_kind="greedy"),
        NearbyTemplatesConfusionMatrix(neighbor_method="greedy", confusion_kind="siagreement"),
        NearbyTemplatesConfusionMatrix(neighbor_method="greedy", confusion_kind="greedy"),
        # MatchRawWaveformsPlot(single_channel=True),
        # MatchRawWaveformsPlot(
        #     show_sorted_matches=False,
        #     show_unsorted_matches=True,
        #     single_channel=True,
        # ),
        # NearbyTemplates(single_channel=True),
        MatchRawWaveformsPlot(),
        MatchRawWaveformsPlot(
            show_sorted_matches=False,
            show_unsorted_matches=True,
        ),
        NearbyTemplates(),
        MatchRawWaveformsPlot(average=True),
        MatchRawWaveformsPlot(
            show_sorted_matches=False,
            show_unsorted_matches=True,
            average=True,
        ),
        NearbyTemplates(which="gt"),
    )
default_unit_comparison_plots = _get_default_unit_comparison_plots()

def make_unit_comparison(
    comparison,
    unit_id,
    plots=None,
    max_height=18,
    figsize=(14, 16),
    figure=None,
    channel_show_radius_um=35.0,
    **other_global_params,
):
    if plots is None:
        plots = _get_default_unit_comparison_plots()
    return make_unit_summary(
        comparison,
        unit_id,
        plots=plots,
        max_height=max_height,
        figsize=figsize,
        figure=figure,
        gizmo_name="comparison",
        channel_show_radius_um=channel_show_radius_um,
        **other_global_params,
    )


def make_all_unit_comparisons(
    comparison,
    save_folder,
    plots=None,
    max_height=18,
    figsize=(14, 16),
    dpi=300,
    image_ext="png",
    namebyamp=True,
    n_jobs=0,
    show_progress=True,
    overwrite=False,
    unit_ids=None,
    **other_global_params,
):
    if plots is None:
        plots = _get_default_unit_comparison_plots()
    return make_all_summaries(
        comparison,
        save_folder,
        plots=plots,
        max_height=max_height,
        figsize=figsize,
        dpi=dpi,
        image_ext=image_ext,
        namebyamp=namebyamp,
        n_jobs=n_jobs,
        show_progress=show_progress,
        overwrite=overwrite,
        unit_ids=unit_ids,
        gizmo_name="comparison",
        taskname="comparisons",
        **other_global_params,
    )
