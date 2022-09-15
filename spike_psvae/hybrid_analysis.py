"""Hybrid analysis helpers

These are just a couple of classes which are basically bags of
data that would be used when making summary plots, and which compute
some metrics etc in the constructor. They help make wrangling a bunch
of sorts a little easier.
"""
import re
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib import colors
import statsmodels.api as sm
import colorcet as cc
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
import seaborn as sns
from tqdm.auto import tqdm
from pathlib import Path
import pickle

from spikeinterface.extractors import NumpySorting
from spikeinterface.comparison import compare_sorter_to_ground_truth

from spike_psvae.spikeio import get_binary_length, read_waveforms

# from spike_psvae.snr_templates import get_templates
from spike_psvae import (
    localize_index,
    cluster_viz,
    cluster_viz_index,
    pyks_ccg,
    cluster_utils,
    snr_templates,
    waveform_utils,
    deconvolve,
)


class Sorting:
    """
    An object to hold onto a spike train and associated localizations
    and templates, and to compute basic statistics about the spike train,
    which will be used by the HybridSorting below.
    """

    def __init__(
        self,
        raw_bin,
        geom,
        spike_times,
        spike_labels,
        name,
        templates=None,
        spike_maxchans=None,
        spike_xzptp=None,
        unsorted=False,
        fs=30_000,
        n_close_units=3,
        template_n_spikes=250,
        cache_dir=None,
        overwrite=False,
        do_cleaned_templates=False,
    ):
        n_spikes_full = spike_labels.shape[0]
        assert spike_labels.shape == spike_times.shape == (n_spikes_full,)
        T_samples, T_sec = get_binary_length(raw_bin, len(geom), fs)
        print("Initializing sorting", name)

        self.name = name
        self.geom = geom
        self.name_lo = re.sub("[^a-z0-9]+", "_", name.lower())
        self.fs = fs
        self.n_close_units = n_close_units
        self.unsorted = unsorted
        self.raw_bin = raw_bin
        self.original_spike_train = np.c_[spike_times, spike_labels]
        self.cleaned_templates = None
        self.do_cleaned_templates = do_cleaned_templates

        # see if we can load up expensive stuff from cache
        # this will check if the sorting in the cache uses the same
        # spike train and raw bin file path, and
        cached = False
        if not overwrite and (cache_dir and templates is None):
            cached, cached_templates = self.try_to_load_from_cache(cache_dir)
            templates = cached_templates if cached else templates

        which = np.flatnonzero(spike_labels >= 0)
        which = which[np.argsort(spike_times[which])]

        self.spike_times = spike_times[which]
        self.spike_labels = spike_labels[which]
        self.unit_labels, self.unit_spike_counts = np.unique(
            self.spike_labels, return_counts=True
        )
        self.n_units = self.unit_labels.size
        full_spike_counts = np.zeros(self.unit_labels.max() + 1, dtype=int)
        full_spike_counts[self.unit_labels] = self.unit_spike_counts
        self.unit_firing_rates = self.unit_spike_counts / T_sec
        self.contiguous_labels = (
            self.unit_labels.size == self.unit_labels.max() + 1
        )

        self.templates = templates
        if templates is None and not unsorted:
            print("Computing raw templates")
            # self.cleaned_templates, _, self.templates, _ = get_templates(
            #     np.c_[self.spike_times, self.spike_labels],
            #     geom,
            #     raw_bin,
            #     return_raw_cleaned=True,
            # )
            self.templates = deconvolve.get_templates(
                raw_bin,
                self.spike_times[:, None],
                self.spike_labels,
                geom,
                n_samples=template_n_spikes,
                pbar=True,
            )

        if self.cleaned_templates is None and do_cleaned_templates:
            print("Computing cleaned templates")
            (
                cleaned_templates,
                extra
            ) = snr_templates.get_templates(
                np.c_[self.spike_times, self.spike_labels],
                self.geom,
                self.raw_bin,
                self.templates.ptp(1).argmax(1),
                tpca_rank=5,
            )
            self.cleaned_templates = cleaned_templates
            self.snrs = extra["snr_by_channel"]
            self.denoised_templates = extra["denoised_templates"]
            self.raw_templates = extra["orig_raw_templates"]
            self.snr_weights = extra["weights"]

        if not unsorted:
            assert self.templates.shape[0] >= self.unit_labels.max() + 1

        if self.templates is not None:
            self.template_ptps = self.templates.ptp(1)
            self.template_maxptps = self.template_ptps.max(1)
            self.template_maxchans = self.template_ptps.argmax(1)
            self.template_locs = localize_index.localize_ptps_index(
                self.template_ptps[full_spike_counts > 0],
                geom,
                self.template_maxchans[full_spike_counts > 0],
                np.stack([np.arange(len(geom))] * len(geom), axis=0),
                n_channels=20,
                n_workers=None,
                pbar=True,
            )
            self.template_locs = list(self.template_locs)
            for i, loc in enumerate(self.template_locs):
                loc_ = np.zeros_like(self.template_maxptps)
                loc_[full_spike_counts > 0] = loc
                self.template_locs[i] = loc_

            self.template_xzptp = np.c_[
                self.template_locs[0],
                self.template_locs[3],
                self.template_maxptps,
            ]
            self.template_feats = np.c_[
                self.template_locs[0],
                self.template_locs[3],
                30 * np.log(self.template_maxptps),
            ]
            self.close_units = self.compute_closest_units()

        if spike_maxchans is None:
            assert not unsorted
            print(
                f"Sorting {name} has no intrinsic maxchans. "
                "Using template maxchans."
            )
            self.spike_maxchans = self.template_maxchans[self.spike_labels]
        else:
            assert spike_maxchans.shape == (n_spikes_full,)
            self.spike_maxchans = spike_maxchans[which]

        self.spike_index = np.c_[self.spike_times, self.spike_maxchans]
        self.spike_train = np.c_[self.spike_times, self.spike_labels]
        self.n_spikes = len(self.spike_index)

        self.spike_xzptp = None
        if spike_xzptp is not None:
            if not spike_xzptp.shape == (n_spikes_full, 3):
                raise ValueError(
                    "Not all data had the same shape. "
                    f"{n_spikes_full=} {spike_labels.shape=} {spike_xzptp.shape=}"
                )
            self.spike_xzptp = spike_xzptp[which]
            self.spike_feats = np.c_[
                self.spike_xzptp[:, :2],
                30 * np.log(self.spike_xzptp[:, 2]),
            ]

        if not self.unsorted:
            self.contam_ratios = np.empty(self.unit_labels.shape)
            self.contam_p_values = np.empty(self.unit_labels.shape)
            for i in tqdm(self.unit_labels, desc="ccg"):
                st = self.get_unit_spike_train(i)
                (
                    self.contam_ratios[i],
                    self.contam_p_values[i],
                ) = pyks_ccg.ccg_metrics(st, st, 500, self.fs / 1000)

        if cache_dir and (overwrite or not cached):
            self.save_to_cache(cache_dir)

    def get_unit_spike_train(self, unit):
        return self.spike_times[self.spike_labels == unit]

    def get_unit_maxchans(self, unit):
        return self.spike_maxchans[self.spike_labels == unit]

    @property
    def np_sorting(self):
        return NumpySorting.from_times_labels(
            times_list=self.spike_times,
            labels_list=self.spike_labels,
            sampling_frequency=self.fs,
        )

    # -- caching logic so we don't re-compute templates all the time
    # cache invalidation is based on the spike train!

    def try_to_load_from_cache(self, cache_dir):
        my_cache = Path(cache_dir) / self.name_lo
        meta_pkl = my_cache / "meta.pkl"
        st_npy = my_cache / "st.npy"
        temps_npy = my_cache / "temps.npy"
        snr_temps_pkl = my_cache / "snr_temps.pkl"
        paths = [my_cache, meta_pkl, st_npy, temps_npy]
        if not all(p.exists() for p in paths):
            # no cache saved
            print(f"There is no cache to load for sorting {self.name}")
            return False, None

        with open(meta_pkl, "rb") as jar:
            meta = pickle.load(jar)
        cache_bin_file = meta["bin_file"]
        if cache_bin_file != self.raw_bin:
            print(
                f"Won't load sorting {self.name} from cache: different binary path"
            )
            return False, None

        cache_st = np.load(st_npy)
        if not np.array_equal(cache_st, self.original_spike_train):
            print(
                f"Won't load sorting {self.name} from cache: different spike train"
            )
            return False, None

        print(f"Loading sorting {self.name} from cache")
        temps = np.load(temps_npy, allow_pickle=True)
        if temps is None or temps.size <= 1:
            return False, None

        if (
            self.do_cleaned_templates
            and self.cleaned_templates is None
            and snr_temps_pkl.exists()
        ):
            with open(snr_temps_pkl, "rb") as jar:
                (
                    self.cleaned_templates,
                    self.snrs,
                    self.snr_weights,
                    self.denoised_templates,
                    self.raw_templates,
                ) = pickle.load(jar)

        return True, temps

    def save_to_cache(self, cache_dir):
        my_cache = Path(cache_dir) / self.name_lo
        my_cache.mkdir(parents=True, exist_ok=True)

        meta_pkl = my_cache / "meta.pkl"
        st_npy = my_cache / "st.npy"
        temps_npy = my_cache / "temps.npy"
        snr_temps_pkl = my_cache / "snr_temps.pkl"

        with open(meta_pkl, "wb") as jar:
            pickle.dump(dict(bin_file=self.raw_bin), jar)
        np.save(st_npy, self.original_spike_train)
        np.save(temps_npy, self.templates)

        if self.cleaned_templates is not None:
            with open(snr_temps_pkl, "wb") as jar:
                pickle.dump(
                    (
                        self.cleaned_templates,
                        self.snrs,
                        self.snr_weights,
                        self.denoised_templates,
                        self.raw_templates,
                    ),
                    jar,
                )

    # -- below are some plots that the sorting can make about itself

    def array_scatter(
        self,
        zlim=(-50, 3900),
        axes=None,
        do_ellipse=True,
        max_n_spikes=500_000,
        pad_zfilter=50,
        annotate=True,
    ):
        pct_shown = 100
        if self.n_spikes > max_n_spikes:
            sample = np.random.default_rng(0).choice(
                self.n_spikes, size=max_n_spikes, replace=False
            )
            sample.sort()
            pct_shown = np.round(100 * max_n_spikes / self.n_spikes)
        else:
            sample = np.arange(self.n_spikes)

        z_hidden = np.flatnonzero(
            (self.spike_xzptp[:, 1] < zlim[0] - pad_zfilter)
            | (self.spike_xzptp[:, 1] > zlim[1] + pad_zfilter)
        )
        sample = np.setdiff1d(sample, z_hidden)

        fig, axes = cluster_viz_index.array_scatter(
            self.spike_labels[sample],
            self.geom,
            self.spike_xzptp[sample, 0],
            self.spike_xzptp[sample, 1],
            self.spike_xzptp[sample, 2],
            annotate=annotate,
            zlim=zlim,
            axes=axes,
            do_ellipse=do_ellipse,
        )
        axes[0].scatter(*self.geom.T, marker="s", s=2, color="orange")
        return fig, axes, pct_shown

    def compute_closest_units(self):
        num_close_clusters = min(10, self.n_units - 1)

        assert self.contiguous_labels
        n_units = self.templates.shape[0]

        close_clusters = np.zeros((n_units, num_close_clusters), dtype=int)
        for i in range(n_units):
            close_clusters[i] = cluster_utils.get_closest_clusters_kilosort(
                i,
                dict(zip(self.unit_labels, self.template_xzptp[:, 1])),
                num_close_clusters=num_close_clusters,
            )

        close_templates = np.zeros(
            (n_units, min(self.n_units - 1, self.n_close_units)), dtype=int
        )
        for i in tqdm(range(n_units)):
            cos_dist = np.zeros(num_close_clusters)
            vis_channels = np.flatnonzero(self.templates[i].ptp(0) >= 1.0)
            for j in range(num_close_clusters):
                idx = close_clusters[i, j]
                # this is max abs norm distance (L_\infty)
                cos_dist[j] = cdist(
                    self.templates[i, :, vis_channels].ravel()[None, :],
                    self.templates[idx, :, vis_channels].ravel()[None, :],
                    "minkowski",
                    p=np.inf,
                )
            close_templates[i] = close_clusters[i][
                cos_dist.argsort()[: min(self.n_units - 1, self.n_close_units)]
            ]

        return close_templates

    def template_maxchan_vis(self, secondary_minptp=3, n_secondary=3):
        fig, (aa, ab, ac) = plt.subplots(nrows=3, figsize=(6, 12))
        count_argsort = np.argsort(self.unit_spike_counts)[::-1]

        # aa: plot templates colored by unit
        colors_uniq = cc.m_glasbey_hv(
            np.arange(len(self.unit_labels)) % len(cc.glasbey_hv)
        )
        for i in count_argsort:
            u = self.unit_labels[i]
            aa.plot(
                self.templates[u, :, self.template_maxchans[u]],
                color=colors_uniq[u],
                alpha=0.5,
            )
            vis_chans = np.setdiff1d(
                np.flatnonzero(self.templates[u].ptp(0) > secondary_minptp),
                [self.template_maxchans[u]],
            )
            if not vis_chans.any():
                continue
            vis_chans_sort = np.argsort(self.templates[u].ptp(0)[vis_chans])[::-1]
            vis_chans = vis_chans[vis_chans_sort[:n_secondary]]
            for c in vis_chans:
                ac.plot(
                    self.templates[u, :, c],
                    color=colors_uniq[u],
                    alpha=0.5,
                )

        # ab: plot templates colored by count, in descending order
        # so that we can actually see the small count templates.
        norm = colors.LogNorm(
            self.unit_spike_counts.min(),
            self.unit_spike_counts.max(),
        )
        mappable = plt.cm.ScalarMappable(
            norm=norm,
            cmap=plt.cm.inferno,
        )
        for i in count_argsort:
            u = self.unit_labels[i]
            ab.plot(
                self.templates[u, :, self.template_maxchans[u]],
                color=mappable.cmap(norm(self.unit_spike_counts[i])),
                alpha=0.5,
            )
        plt.colorbar(
            mappable,
            ax=ab,
            label="spike count",
        )
        aa.set_title("primary channels by unit")
        ab.set_title("primary channels by count")
        ac.set_title(f"top {n_secondary} secondary channels with ptp>{secondary_minptp} by unit")
        fig.suptitle(
            f"{self.name}, template maxchan traces, {len(self.unit_labels)} units.",
            fontsize=12,
            y=0.92,
        )
        return fig

    def cleaned_temp_vis(self, unit, ax=None, nchans=20):
        assert self.contiguous_labels and self.cleaned_templates is not None

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure

        temp = self.cleaned_templates[unit]
        raw_temp = self.raw_templates[unit]
        denoised_temp = self.denoised_templates[unit]
        weights = self.snr_weights[unit]

        # get on fewer chans
        ci = waveform_utils.make_contiguous_channel_index(
            self.geom.shape[0], nchans
        )
        tmc = temp.ptp(0).argmax()

        # make plot
        amp = np.abs(temp).max()
        rlines = cluster_viz_index.pgeom(
            raw_temp[:, ci[tmc]],
            tmc,
            ci,
            self.geom,
            max_abs_amp=amp,
            color="gray",
            ax=ax,
        )
        dlines = cluster_viz_index.pgeom(
            denoised_temp[:, ci[tmc]],
            tmc,
            ci,
            self.geom,
            max_abs_amp=amp,
            color="green",
            show_zero=False,
            ax=ax,
        )
        lines = cluster_viz_index.pgeom(
            temp[:, ci[tmc]],
            tmc,
            ci,
            self.geom,
            max_abs_amp=amp,
            color="orange",
            lw=1,
            show_zero=False,
            ax=ax,
        )
        wlines = cluster_viz_index.pgeom(
            weights[:, ci[tmc]],
            tmc,
            ci,
            self.geom,
            max_abs_amp=amp,
            color="purple",
            lw=1,
            show_zero=False,
            ax=ax,
        )
        ax.legend(
            (rlines[0], dlines[0], lines[0], wlines[0]),
            ("raw", "denoised", "final", "weight"),
            fancybox=False,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        return (
            fig,
            ax,
            self.snrs[unit].max(),
            raw_temp.ptp(0).max(),
            temp.ptp(0).max(),
        )

    def unit_summary_fig(self, unit, dz=50, nchans=16, n_wfs_max=100, show_chan_label=True, chan_labels=None):
        have_loc = self.spike_xzptp is not None
        height_ratios = [1, 1, 3] if have_loc else [1, 3]
        fig, axes = plt.subplot_mosaic(
            "aat\nxyz\nddd" if have_loc else "aat\nddd",
            gridspec_kw=dict(
                height_ratios=[1, 2, 5] if have_loc else [1, 5],
            ),
            figsize=(6, 2 * sum(height_ratios)),
        )

        in_unit = np.flatnonzero(self.spike_train[:, 1] == unit)
        unit_st = self.spike_train[in_unit, 0]
        cx, cz, cptp = self.template_xzptp[unit]

        # text summaries
        unit_props = dict(
            unit=unit,
            snr=self.templates[unit].ptp(1).max() * np.sqrt(self.unit_spike_counts[unit]),
            n_spikes=self.unit_spike_counts[unit],
            template_ptp=self.templates[unit].ptp(1).max(),
            max_channel=self.template_maxchans[unit],
            trough_sample=self.templates[unit, :, self.template_maxchans[unit]].argmin(),
        )
        axes["t"].text(
            0,
            1,
            "\n".join(
                (f"{k}: {v}" if np.issubdtype(v, np.integer) else f"{k}: {v:0.2f}")
                for k, v in unit_props.items()),
            transform=axes["t"].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(edgecolor="none", facecolor="none"),
        )
        axes["t"].axis("off")

        # ISI distribution
        cluster_viz.plot_isi_distribution(unit_st, ax=axes["a"], cdf=False)

        # Scatter
        if have_loc:
            self.array_scatter(
                zlim=(cz - dz, cz + dz),
                axes=[axes[k] for k in "xyz"],
                do_ellipse=True,
                annotate=True,
            )
            axes["y"].set_yticks([])
            axes["z"].set_yticks([])

        # Waveforms
        ci = waveform_utils.make_contiguous_channel_index(
            self.geom.shape[0], nchans
        )
        choices = slice(None)
        if in_unit.size > n_wfs_max:
            choices = np.random.choice(in_unit.size, n_wfs_max, replace=False)
            choices.sort()
        maxchans = self.template_maxchans[
            self.spike_train[in_unit[choices], 1]
        ]
        wfs, skipped = read_waveforms(
            self.spike_train[in_unit[choices], 0],
            self.raw_bin,
            len(self.geom),
            channel_index=ci,
            max_channels=maxchans,
        )
        max_abs_amp = np.abs(wfs).max()
        wf_lines = cluster_viz_index.pgeom(
            wfs,
            maxchans,
            ci,
            self.geom,
            ax=axes["d"],
            max_abs_amp=max_abs_amp,
            color="k",
            alpha=0.05,
            show_chan_label=False,
        )
        rt_lines = cluster_viz_index.pgeom(
            self.templates[unit][:, ci[self.template_maxchans[unit]]],
            self.template_maxchans[unit],
            ci,
            self.geom,
            ax=axes["d"],
            max_abs_amp=max_abs_amp,
            color="b",
            lw=1,
            show_chan_label=show_chan_label,
            chan_labels=chan_labels,
        )
        ch = cl = ()
        if self.cleaned_templates is not None:
            ct_lines = cluster_viz_index.pgeom(
                self.cleaned_templates[unit][
                    :, ci[self.template_maxchans[unit]]
                ],
                self.template_maxchans[unit],
                ci,
                self.geom,
                ax=axes["d"],
                max_abs_amp=max_abs_amp,
                color="orange",
                lw=1,
                show_chan_label=False,
            )
            ch = (ct_lines[0],)
            cl = ("cleaned template",)
        axes["d"].legend(
            (wf_lines[0], rt_lines[0], *ch),
            ("waveforms", "raw template", *cl),
            fancybox=False,
            loc="lower right",
        )
        axes["d"].set_xticks([])
        axes["d"].set_yticks([])

        return fig, axes, self.template_maxptps[unit]


class HybridComparison:
    """
    An object which computes some hybrid metrics and stores references
    to the ground truth and compared sortings, so that everything is
    in one place for later plotting / analysis code.
    """

    def __init__(
        self, gt_sorting, new_sorting, geom, match_score=0.1, dt_samples=5
    ):
        assert gt_sorting.contiguous_labels

        self.gt_sorting = gt_sorting
        self.new_sorting = new_sorting
        self.unsorted = new_sorting.unsorted
        self.geom = geom

        self.average_performance = (
            self.weighted_average_performance
        ) = _na_avg_performance
        if not new_sorting.unsorted:
            gt_comparison = compare_sorter_to_ground_truth(
                gt_sorting.np_sorting,
                new_sorting.np_sorting,
                gt_name=gt_sorting.name,
                tested_name=new_sorting.name,
                sampling_frequency=gt_sorting.fs,
                exhaustive_gt=False,
                match_score=match_score,
                verbose=True,
                delta_time=dt_samples / (gt_sorting.fs / 1000),
            )

            self.ordered_agreement = (
                gt_comparison.get_ordered_agreement_scores()
            )
            self.best_match_12 = gt_comparison.best_match_12.values.astype(int)
            self.gt_matched = self.best_match_12 >= 0

            # matching units and accuracies
            self.performance_by_unit = gt_comparison.get_performance().astype(
                float
            )
            # average the metrics over units
            self.average_performance = gt_comparison.get_performance(
                method="pooled_with_average"
            ).astype(float)
            # average metrics, weighting each unit by its spike count
            self.weighted_average_performance = (
                self.performance_by_unit
                * gt_sorting.unit_spike_counts[:, None]
            ).sum(0) / gt_sorting.unit_spike_counts.sum()

        # unsorted performance
        tp, fn, fp, num_gt, detected = unsorted_confusion(
            gt_sorting.spike_index,
            new_sorting.spike_index,
            n_samples=dt_samples,
        )
        # as in spikeinterface, the idea of a true negative does not make sense here
        # accuracy with tn=0 is called threat score or critical success index, apparently
        self.unsorted_accuracy = tp / (tp + fn + fp)
        # this is what I was originally calling the unsorted accuracy
        self.unsorted_recall = tp / (tp + fn)
        self.unsorted_precision = tp / (tp + fp)
        self.unsorted_false_discovery_rate = fp / (tp + fp)
        self.unsorted_miss_rate = fn / num_gt
        self.unsorted_recall_by_unit = np.array(
            [
                detected[gt_sorting.spike_labels == u].mean()
                for u in gt_sorting.unit_labels
            ]
        )

    def get_best_new_match(self, gt_unit):
        return int(self.best_match_12[gt_unit])

    def get_closest_new_unit(self, gt_unit):
        gt_loc = self.gt_sorting.template_xzptp[gt_unit]
        new_template_locs = self.new_sorting.template_xzptp
        return np.argmin(cdist(gt_loc[None], new_template_locs).squeeze())


# -- library


def unsorted_confusion(
    gt_spike_index, new_spike_index, n_samples=12, n_channels=4
):
    cmul = n_samples / n_channels
    n_gt_spikes = len(gt_spike_index)
    n_new_spikes = len(gt_spike_index)

    gt_kdt = KDTree(np.c_[gt_spike_index[:, 0], gt_spike_index[:, 1] * cmul])
    sorter_kdt = KDTree(
        np.c_[new_spike_index[:, 0], cmul * new_spike_index[:, 1]]
    )
    query = gt_kdt.query_ball_tree(sorter_kdt, n_samples + 0.1)

    # this is a boolean array of length n_gt_spikes
    detected = np.array([len(lst) > 0 for lst in query], dtype=bool)

    # from the above, we can compute a couple of metrics
    true_positives = detected.sum()
    false_negatives = n_gt_spikes - true_positives
    false_positives = n_new_spikes - true_positives

    return (
        true_positives,
        false_negatives,
        false_positives,
        n_gt_spikes,
        detected,
    )


def density_near_gt(hybrid_comparison, radius=50):
    gt_feats = hybrid_comparison.gt_sorting.template_feats
    new_spike_feats = hybrid_comparison.new_sorting.spike_feats
    gt_kdt = KDTree(gt_feats)
    new_kdt = KDTree(new_spike_feats)
    query = gt_kdt.query_ball_tree(new_kdt, r=radius)
    density = np.array([len(q) for q in query])
    return density


_na_avg_performance = pd.Series(
    index=[
        "accuracy",
        "recall",
        "precision",
        "false_discovery_rate",
        "miss_rate",
    ],
    data=[np.nan] * 5,
)


# -- plotting helpers


def plotgistic(
    df,
    x="gt_ptp",
    y=None,
    c="gt_firing_rate",
    title=None,
    cmap=plt.cm.plasma,
    legend=True,
    ax=None,
    ylim=[-0.05, 1.05],
):
    ylab = y
    xlab = x
    clab = c
    y = df[y].values
    x = df[x].values
    X = sm.add_constant(x)
    c = df[c].values

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # fit the logistic function to the data
    def resids(beta):
        return y - 1 / (1 + np.exp(-X @ beta))

    res = least_squares(resids, np.array([1.0, 1]))
    b = res.x

    # plot the logistic line
    domain = np.linspace(x.min(), x.max())
    (l,) = ax.plot(
        domain, 1 / (1 + np.exp(-sm.add_constant(domain) @ b)), color="k"
    )

    # scatter with legend
    leg = ax.scatter(x, y, marker="x", c=c, cmap=cmap, alpha=0.75)
    h, labs = leg.legend_elements(num=4)
    if legend:
        ax.legend(
            (*h, l),
            (*labs, "logistic fit"),
            title=clab,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
        )

    if title:
        n_missed = (y < 1e-8).sum()
        plt.title(title + f" -- {n_missed} missed")

    if ylim is None:
        dy = y.max() - y.min()
        ylim = [y.min() - 0.05 * dy, y.max() + 0.05 * dy]
    ax.set_ylim(ylim)
    ax.set_xlim([x.min() - 0.5, x.max() + 0.5])
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)

    return fig, ax


def make_diagnostic_plot(hybrid_comparison, gt_unit):
    new_unit = hybrid_comparison.get_best_new_match(gt_unit)
    new_str = f"{hybrid_comparison.new_sorting.name} match {new_unit}"
    if new_unit < 0:
        new_unit = hybrid_comparison.get_closest_new_unit(gt_unit)
        new_str = f"No {hybrid_comparison.new_sorting.name} match, using closest unit {new_unit}."

    gt_np_sorting = hybrid_comparison.gt_sorting.np_sorting
    new_np_sorting = hybrid_comparison.new_sorting.np_sorting
    gt_spike_train = gt_np_sorting.get_unit_spike_train(gt_unit)
    new_spike_train = new_np_sorting.get_unit_spike_train(new_unit)
    gt_maxchans = hybrid_comparison.gt_sorting.get_unit_maxchans(gt_unit)
    new_maxchans = hybrid_comparison.new_sorting.get_unit_maxchans(new_unit)
    gt_template_zs = hybrid_comparison.gt_sorting.template_locs[2]
    new_template_zs = hybrid_comparison.new_sorting.template_locs[2]

    gt_ptp = hybrid_comparison.gt_sorting.template_maxptps[gt_unit]

    fig, agreement = cluster_viz.diagnostic_plots(
        new_unit,
        gt_unit,
        new_spike_train,
        gt_spike_train,
        hybrid_comparison.new_sorting.templates,
        hybrid_comparison.gt_sorting.templates,
        new_maxchans,
        gt_maxchans,
        hybrid_comparison.geom,
        hybrid_comparison.gt_sorting.raw_bin,
        dict(enumerate(new_template_zs)),
        dict(enumerate(gt_template_zs)),
        hybrid_comparison.new_sorting.spike_index,
        hybrid_comparison.gt_sorting.spike_index,
        hybrid_comparison.new_sorting.spike_labels,
        hybrid_comparison.gt_sorting.spike_labels,
        hybrid_comparison.new_sorting.close_units[new_unit],
        hybrid_comparison.gt_sorting.close_units[gt_unit],
        scale=7,
        sorting1_name=hybrid_comparison.new_sorting.name,
        sorting2_name=hybrid_comparison.gt_sorting.name,
        num_channels=40,
        num_spikes_plot=100,
        t_range=(30, 90),
        num_rows=3,
        alpha=0.1,
        delta_frames=12,
        num_close_clusters=5,
        tpca_rank=6,
    )

    fig.suptitle(f"GT unit {gt_unit}. {new_str}")

    return fig, gt_ptp, agreement


def array_scatter_vs(scatter_comparison, vs_comparison, do_ellipse=True):
    fig, axes, pct_shown = scatter_comparison.new_sorting.array_scatter(
        do_ellipse=do_ellipse
    )
    
    if vs_comparison is None:
        return fig, axes, None, pct_shown

    scatter_match = scatter_comparison.gt_matched
    vs_match = vs_comparison.gt_matched
    match = scatter_match + 2 * vs_match
    colors = ["k", "b", "r", "purple"]

    gt_x, gt_z, gt_ptp = scatter_comparison.gt_sorting.template_xzptp.T
    log_gt_ptp = np.log(gt_ptp)

    ls = []
    for i, c in enumerate(colors):
        matchix = match == i
        gtxix = gt_x[matchix]
        gtzix = gt_z[matchix]
        gtpix = log_gt_ptp[matchix]
        axes[0].scatter(gtxix, gtzix, color=c, marker="x", s=15)
        axes[2].scatter(gtxix, gtzix, color=c, marker="x", s=15)
        l = axes[1].scatter(gtpix, gtzix, color=c, marker="x", s=15)
        ls.append(l)

    leg_artist = plt.figlegend(
        ls,
        [
            "no match",
            f"{scatter_comparison.new_sorting.name} match",
            f"{vs_comparison.new_sorting.name} match",
            "both",
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        borderaxespad=-10,
    )

    return fig, axes, leg_artist, pct_shown


def near_gt_scatter_vs(step_comparisons, vs_comparison, gt_unit, dz=100):
    nrows = len(step_comparisons)
    fig, axes = plt.subplots(
        nrows=nrows + 1,
        ncols=3,
        sharex="col",
        sharey=True,
        figsize=(6, 2 * nrows + 1),
        gridspec_kw=dict(
            hspace=0.25, wspace=0.0, height_ratios=[1] * nrows + [0.1]
        ),
    )
    gt_x, gt_z, gt_ptp = vs_comparison.gt_sorting.template_xzptp.T
    log_gt_ptp = np.log(gt_ptp)
    gt_unit_z = gt_z[gt_unit]
    gt_unit_ptp = gt_ptp[gt_unit]
    zlim = gt_unit_z - dz, gt_unit_z + dz
    colors = ["k", "b", "r", "purple"]
    vs_match = vs_comparison.gt_matched

    for i, comp in enumerate(step_comparisons):
        comp.new_sorting.array_scatter(zlim=zlim, axes=axes[i])

        match = comp.gt_matched + 2 * vs_match
        ls = []
        for j, c in enumerate(colors):
            matchix = match == j
            gtxix = gt_x[matchix]
            gtzix = gt_z[matchix]
            gtpix = log_gt_ptp[matchix]
            axes[i, 0].scatter(gtxix, gtzix, color=c, marker="x", s=15)
            axes[i, 2].scatter(gtxix, gtzix, color=c, marker="x", s=15)
            l = axes[i, 1].scatter(gtpix, gtzix, color=c, marker="x", s=15)
            ls.append(l)

        u = comp.best_match_12[gt_unit]
        matchstr = "no match"
        if u >= 0:
            matchstr = f"matching unit {u}"
        axes[i, 1].set_title(
            f"{comp.new_sorting.name}, {matchstr}", fontsize=8
        )

        if i < nrows - 1:
            for ax in axes[i]:
                ax.set_xlabel("")

    for ax in axes[-1]:
        ax.set_axis_off()

    leg_artist = plt.figlegend(
        ls,
        [
            "no match",
            f"row sorter match",
            f"{vs_comparison.new_sorting.name} match",
            "both",
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        borderaxespad=5,
    )

    return fig, axes, leg_artist, gt_unit_ptp


def plot_agreement_matrix(hybrid_comparison, cmap=plt.cm.plasma):
    axes = sns.heatmap(hybrid_comparison.ordered_agreement, cmap=cmap)
    return axes
