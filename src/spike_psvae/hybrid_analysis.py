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
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from spikeinterface.extractors import NumpySorting
from spikeinterface.comparison import compare_sorter_to_ground_truth
from spike_psvae.denoise import (
    denoise_wf_nn_tmp_single_channel,
    SingleChanDenoiser,
)

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
    deconv_resid_merge,
    relocation,
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
        spike_xyza=None,
        spike_z_reg=None,
        spike_maxptps=None,
        unsorted=False,
        fs=30_000,
        n_close_units=3,
        template_n_spikes=250,
        cache_dir=None,
        overwrite=False,
        do_cleaned_templates=False,
        cleaned_templates=None,
        trough_offset=42,
        spike_length_samples=121,
        extra=None,
    ):
        self.n_spikes_full = n_spikes_full = spike_labels.shape[0]
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
        self.cleaned_templates = cleaned_templates
        self.do_cleaned_templates = do_cleaned_templates
        self.trough_offset = trough_offset
        self.spike_length_samples = spike_length_samples

        # see if we can load up expensive stuff from cache
        # this will check if the sorting in the cache uses the same
        # spike train and raw bin file path, and
        cached = False
        if not overwrite and (cache_dir and templates is None):
            cached, cached_templates = self.try_to_load_from_cache(cache_dir)
            templates = cached_templates if cached else templates

        which = np.flatnonzero(spike_labels >= 0)
        self.was_sorted = (np.diff(spike_times[which]) >= 0).all()
        which = which[np.argsort(spike_times[which], kind="stable")]
        self.which = which

        self.spike_times = spike_times[which]
        self.spike_labels = spike_labels[which]
        self.unit_labels, self.unit_spike_counts = np.unique(
            self.spike_labels, return_counts=True
        )
        self.unit_label_to_index = dict(
            (v, k) for k, v in enumerate(self.unit_labels)
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
            self.templates = snr_templates.get_raw_templates(
                np.c_[self.spike_times, self.spike_labels],
                geom,
                raw_bin,
                max_spikes_per_unit=template_n_spikes,
                reducer=np.median,
                spike_length_samples=spike_length_samples,
                trough_offset=trough_offset,
                pbar=True,
                seed=0,
                n_jobs=-1,
            )
        assert (
            self.templates.shape[0]
            == self.spike_labels.max() + 1
            >= self.n_units
        )

        if self.cleaned_templates is None and do_cleaned_templates:
            print("Computing cleaned templates")
            (cleaned_templates, extra) = snr_templates.get_templates(
                np.c_[self.spike_times, self.spike_labels],
                self.geom,
                self.raw_bin,
                self.templates.ptp(1).argmax(1),
                tpca_rank=5,
                spike_length_samples=spike_length_samples,
                trough_offset=trough_offset,
            )
            self.cleaned_templates = cleaned_templates
            assert self.cleaned_templates.shape[0] == self.templates.shape[0]

        if extra is not None:
            self.snrs = extra["snr_by_channel"]
            self.denoised_templates = extra["denoised_templates"]
            self.raw_templates = extra["raw_templates"]
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
                # n_channels=20,
                radius=100,
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
            assert (
                self.template_locs[0].size
                == self.template_xzptp.shape[0]
                == self.templates.shape[0]
            )

        if spike_maxchans is None:
            assert not unsorted
            print(
                f"Sorting {name} has no per-spike maxchans. "
                "Will use template maxchans when spike maxchans are needed."
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
        elif not any(
            a is None for a in (spike_xyza, spike_z_reg, spike_maxptps)
        ):
            self.spike_xzptp = np.c_[
                spike_xyza[:, 0], spike_z_reg, spike_maxptps
            ][which]
        self.spike_xyza = spike_xyza[which] if spike_xyza is not None else None
        self.spike_z_reg = (
            spike_z_reg[which] if spike_z_reg is not None else None
        )

        if not self.unsorted:
            self.contam_ratios = np.empty(self.unit_labels.shape)
            self.contam_p_values = np.empty(self.unit_labels.shape)
            for i, unit in enumerate(tqdm(self.unit_labels, desc="ccg")):
                st = self.get_unit_spike_train(unit)
                (
                    self.contam_ratios[i],
                    self.contam_p_values[i],
                ) = pyks_ccg.ccg_metrics(st, st, 500, self.fs / 1000)

        if cache_dir and (overwrite or not cached):
            self.save_to_cache(cache_dir)

        self._template_residuals = None
        self._norm_template_residuals = None
        self._close_units = None

    def get_unit_spike_train(self, unit):
        return self.spike_times[self.spike_labels == unit]

    def get_unit_maxchans(self, unit):
        return self.spike_maxchans[self.spike_labels == unit]

    def resid_matrix(
        self,
        units,
        n_jobs=-1,
        pbar=True,
        lambd=0.001,
        allowed_scale=0.1,
        normalized=True,
    ):
        assert self.cleaned_templates is not None
        thresh = 0.9 * np.square(self.cleaned_templates).sum(axis=(1, 2)).min()
        dists = calc_resid_matrix(
            self.cleaned_templates,
            units,
            self.cleaned_templates,
            units,
            thresh=thresh,
            n_jobs=n_jobs,
            auto=True,
            pbar=pbar,
            lambd=lambd,
            allowed_scale=allowed_scale,
            normalized=normalized,
        )
        return thresh, dists

    @property
    def template_residuals(self):
        if self._template_residuals is None:
            thresh, dists = self.resid_matrix(
                self.unit_labels, normalized=False
            )
            self._template_residuals = dists
        return self._template_residuals

    @property
    def norm_template_residuals(self):
        if self._norm_template_residuals is None:
            thresh, dists = self.resid_matrix(self.unit_labels, normalized=True)
            self._norm_template_residuals = dists
        return self._norm_template_residuals

    @property
    def np_sorting(self):
        return NumpySorting.from_times_labels(
            times_list=self.spike_times,
            labels_list=self.spike_labels,
            sampling_frequency=self.fs,
        )

    @property
    def close_units(self):
        if self._close_units is None:
            self._close_units = self.compute_closest_units()
        return self._close_units

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
        zlim=None,
        axes=None,
        do_ellipse=True,
        max_n_spikes=500_000,
        pad_zfilter=50,
        annotate=True,
    ):
        if zlim is None:
            zlim = (self.geom.min() - 50, self.geom.max() + 50)
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
        n_units = self.unit_labels.size

        close_clusters = np.zeros((n_units, num_close_clusters), dtype=int)
        for i, unit in enumerate(self.unit_labels):
            close_clusters[i] = cluster_utils.get_closest_clusters_kilosort(
                unit,
                dict(
                    zip(
                        self.unit_labels,
                        self.template_xzptp[self.unit_labels, 1],
                    )
                ),
                num_close_clusters=num_close_clusters,
            )

        close_templates = np.zeros(
            (n_units, min(self.n_units - 1, self.n_close_units)), dtype=int
        )
        for i, unit in enumerate(tqdm(self.unit_labels)):
            cos_dist = np.zeros(num_close_clusters)
            vis_channels = np.flatnonzero(self.templates[unit].ptp(0) >= 1.0)
            for j in range(num_close_clusters):
                idx = close_clusters[i, j]
                # this is max abs norm distance (L_\infty)
                cos_dist[j] = cdist(
                    self.templates[unit, :, vis_channels].ravel()[None, :],
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
            vis_chans_sort = np.argsort(self.templates[u].ptp(0)[vis_chans])[
                ::-1
            ]
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
        ac.set_title(
            f"top {n_secondary} secondary channels with ptp>{secondary_minptp} by unit"
        )
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

    def unit_summary_fig(
        self,
        unit,
        dz=50,
        nchans=16,
        plot_channel_index=None,
        n_wfs_max=100,
        show_chan_label=True,
        show_scatter=True,
        chan_labels=None,
        relocated=False,
        stored_channel_index=None,
        stored_maxchans=None,
        stored_waveforms=None,
        stored_tpca_projs=None,
        stored_tpca=None,
        stored_order=None,
    ):
        unit_index = self.unit_label_to_index[unit]
        show_scatter = show_scatter and self.spike_xzptp is not None
        height_ratios = [1, 1, 1, 2, 5] if show_scatter else [1, 1, 1, 5]
        fig, axes = plt.subplot_mosaic(
            "aat\nbbt\ncct\nxyz\nddd" if show_scatter else "aat\nbbt\ncct\nddd",
            gridspec_kw=dict(
                height_ratios=height_ratios,
            ),
            figsize=(6, 2 * sum(height_ratios)),
        )

        in_unit = np.flatnonzero(self.spike_train[:, 1] == unit)
        unit_st = self.spike_train[in_unit, 0]
        cx, cz, cptp = self.template_xzptp[unit]

        # text summaries
        unit_props = dict(
            unit=unit,
            snr=self.templates[unit].ptp(1).max()
            * np.sqrt(self.unit_spike_counts[unit_index]),
            n_spikes=self.unit_spike_counts[unit_index],
            template_ptp=self.templates[unit].ptp(1).max(),
            max_channel=self.template_maxchans[unit],
            trough_sample=self.templates[
                unit, :, self.template_maxchans[unit]
            ].argmin(),
        )
        axes["t"].text(
            0,
            1,
            "\n".join(
                (
                    f"{k}: {v}"
                    # if np.issubdtype(v, np.integer)
                    # else f"{k}: {v:0.2f}"
                )
                for k, v in unit_props.items()
            ),
            transform=axes["t"].transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(edgecolor="none", facecolor="none"),
        )
        axes["t"].axis("off")

        # ISI distribution
        cluster_viz.plot_isi_distribution(
            unit_st, ax=axes["a"], cdf=False, bins=1000 / self.fs
        )
        cluster_viz_index.plot_ccg(
            unit_st, ms_frames=self.fs / 1000, ax=axes["b"]
        )

        # Scatter
        if show_scatter:
            self.array_scatter(
                zlim=(cz - dz, cz + dz),
                axes=[axes[k] for k in "xyz"],
                do_ellipse=True,
                annotate=True,
            )
            axes["y"].set_yticks([])
            axes["z"].set_yticks([])

        # Waveforms
        if plot_channel_index is not None:
            ci = plot_channel_index
        else:
            ci = waveform_utils.make_contiguous_channel_index(
                self.geom.shape[0], nchans
            )
        choices = np.arange(in_unit.size)
        if in_unit.size > n_wfs_max:
            choices = np.random.choice(in_unit.size, n_wfs_max, replace=False)
            choices.sort()
        maxchans = self.template_maxchans[self.spike_train[in_unit[choices], 1]]

        # load waveforms: either from h5 (stored_*) or disk
        kept = np.arange(len(in_unit[choices]))
        if stored_waveforms is None and stored_tpca_projs is None:
            if relocated:
                (
                    wfs,
                    skipped,
                ) = relocation.load_relocated_waveforms_on_channel_subset(
                    np.c_[self.spike_train[in_unit[choices], 0], maxchans],
                    self.raw_bin,
                    self.spike_xyza[in_unit[choices]],
                    self.spike_z_reg[in_unit[choices]],
                    self.geom,
                    target_channels=ci[self.template_maxchans[unit]],
                    fill_value=np.nan,
                    trough_offset=self.trough_offset,
                    spike_length_samples=self.spike_length_samples,
                )
            else:
                wfs, skipped = read_waveforms(
                    self.spike_train[in_unit[choices], 0],
                    self.raw_bin,
                    len(self.geom),
                    channel_index=ci,
                    max_channels=maxchans,
                    trough_offset=self.trough_offset,
                    spike_length_samples=self.spike_length_samples,
                )
            kept = np.setdiff1d(kept, skipped)
        else:
            assert stored_maxchans is not None
            assert stored_channel_index is not None
            if stored_order is None:
                stored_order = np.arange(self.n_spikes_full)
            load_stored = stored_order[self.which[in_unit[choices]]]
            reord = np.argsort(load_stored)
            choices = choices[reord]
            load_stored = load_stored[reord]
            mcs = stored_maxchans[load_stored]

            if stored_waveforms is not None:
                assert stored_waveforms.shape[0] == self.n_spikes_full
                wfs = stored_waveforms[load_stored]
            elif stored_tpca_projs is not None:
                assert stored_tpca_projs.shape[0] == self.n_spikes_full
                projs = stored_tpca_projs[load_stored]
                wfs = stored_tpca.inverse_transform(
                    projs, mcs, stored_channel_index
                )

            if relocated:
                wfs = relocation.get_relocated_waveforms_on_channel_subset(
                    mcs,
                    wfs,
                    self.spike_xyza[in_unit[choices]],
                    self.spike_z_reg[in_unit[choices]],
                    stored_channel_index,
                    self.geom,
                    target_channels=ci[self.template_maxchans[unit]],
                )
            else:
                wfs = waveform_utils.restrict_wfs_to_chans(
                    wfs,
                    max_channels=mcs,
                    channel_index=stored_channel_index,
                    dest_channels=ci[self.template_maxchans[unit]],
                )
            kept = np.flatnonzero(~np.isnan(wfs).all(axis=(1, 2)))

        max_abs_amp = None
        if kept.size:
            max_abs_amp = np.nanmax(np.abs(self.templates[unit]))
            wf_lines = cluster_viz_index.pgeom(
                wfs[kept],
                maxchans[kept],
                ci,
                self.geom,
                ax=axes["d"],
                max_abs_amp=max_abs_amp,
                color="k",
                alpha=0.05,
                show_chan_label=False,
                zlim="auto",
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
            alpha=1,
            show_chan_label=show_chan_label,
            chan_labels=chan_labels,
            zlim="auto",
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
                zlim="auto",
            )
            ch = (ct_lines[0],)
            cl = ("cleaned template",)
        if kept.size:
            axes["d"].legend(
                (wf_lines[0], rt_lines[0], *ch),
                ("waveforms", "raw template", *cl),
                fancybox=False,
                loc="lower right",
            )
        else:
            axes["d"].legend(
                (rt_lines[0], *ch),
                ("raw template", *cl),
                fancybox=False,
                loc="lower right",
            )
        axes["d"].set_xticks([])
        axes["d"].set_yticks([])

        # pca plot
        if kept.size:
            pca_chans = np.flatnonzero(
                cdist(self.geom[self.template_maxchans[unit]][None], self.geom)[
                    0
                ]
                < 75
            )
            pca_wfs = waveform_utils.restrict_wfs_to_chans(
                wfs,
                source_channels=ci[self.template_maxchans[unit]],
                dest_channels=pca_chans,
            )
            kept2 = ~np.isnan(pca_wfs).any(axis=(1, 2))
            if kept2.sum() > 1:
                pca_projs = PCA(2).fit_transform(
                    pca_wfs[kept2].reshape(kept2.sum(), -1)
                )
                axes["c"].scatter(*pca_projs.T, color="k", s=1)
        axes["c"].set_xlabel("unit pc1")
        axes["c"].set_ylabel("unit pc2")

        return fig, axes, self.template_maxptps[unit]

    def make_unit_summaries(
        self,
        out_folder,
        dz=50,
        nchans=16,
        units=None,
        n_wfs_max=250,
        show_chan_label=True,
        show_scatter=True,
        chan_labels=None,
        relocated=False,
        stored_channel_index=None,
        stored_maxchans=None,
        stored_waveforms=None,
        stored_tpca_projs=None,
        stored_tpca=None,
        stored_order=None,
        n_jobs=-1,
    ):
        out_folder = Path(out_folder)
        out_folder.mkdir(exist_ok=True, parents=True)

        def job(unit):
            fig, axes, ptp = self.unit_summary_fig(
                unit,
                dz=dz,
                nchans=nchans,
                n_wfs_max=n_wfs_max,
                show_chan_label=show_chan_label,
                show_scatter=show_scatter,
                chan_labels=chan_labels,
                relocated=relocated,
                stored_channel_index=stored_channel_index,
                stored_maxchans=stored_maxchans,
                stored_waveforms=stored_waveforms,
                stored_tpca_projs=stored_tpca_projs,
                stored_tpca=stored_tpca,
                stored_order=stored_order,
            )
            fig.savefig(
                out_folder / f"{self.name_lo}_unit{unit:03d}.png", dpi=300
            )
            plt.close(fig)

        with Parallel(n_jobs) as p:
            for res in p(
                delayed(job)(unit)
                for unit in tqdm(
                    np.intersect1d(
                        self.unit_labels,
                        units if units is not None else self.unit_labels,
                    ),
                    desc="Unit summaries",
                )
            ):
                pass

    def unit_resid_study(
        self,
        unit,
        n_max=5,
        max_dist=4,
        plot_chans=10,
        lambd=0.001,
        allowed_scale=0.1,
        tmin=None,
        tmax=None,
    ):
        unit_temp = self.cleaned_templates[unit]
        thresh = 0.9 * np.square(self.cleaned_templates).sum(axis=(1, 2)).min()
        resid_v_new = calc_resid_matrix(
            unit_temp[None],
            np.array([0]),
            self.cleaned_templates,
            np.arange(self.cleaned_templates.shape[0]),
            thresh=thresh,
            n_jobs=1,
            pbar=False,
            lambd=lambd,
            allowed_scale=allowed_scale,
        ).squeeze()

        # ignore large distances
        # resid_v_new[resid_v_new > max_dist] = np.inf

        resid_is_finite = np.isfinite(resid_v_new)
        if not resid_is_finite.any():
            return None, None

        resid_is_finite = np.flatnonzero(resid_is_finite)

        resid_vals = resid_v_new[resid_is_finite]
        sort = np.argsort(resid_vals)[:n_max]
        sorted_near_units = self.unit_labels[resid_is_finite[sort]]

        fig, axes = plt.subplot_mosaic(
            "a.b\n...\nccc",
            gridspec_kw=dict(
                height_ratios=[1.5, 0.1, 4], width_ratios=[1, 0.1, 1]
            ),
            figsize=(4, 8),
        )

        axes["a"].plot(resid_vals[sort])
        axes["a"].set_xticks(
            np.arange(len(sort)),
            sorted_near_units,
        )
        axes["a"].set_xlabel("all nearby units")
        axes["a"].set_ylabel("resid dist")

        sorted_near_units = sorted_near_units

        thresh, near_unit_distmat = self.resid_matrix(
            sorted_near_units,
            n_jobs=1,
            pbar=False,
            lambd=lambd,
            allowed_scale=allowed_scale,
        )
        near_dist_df = pd.DataFrame(
            near_unit_distmat,
            index=sorted_near_units,
            columns=sorted_near_units,
        )
        vals = near_unit_distmat[np.isfinite(near_unit_distmat)]
        if not vals.size:
            vals = np.array([0, 0])
        sns.heatmap(
            near_dist_df, vmin=vals.min(), vmax=vals.max(), ax=axes["b"]
        )
        axes["b"].set_xlabel("nearby units")
        axes["b"].set_title(
            f"{len(sorted_near_units)} closest pairwise resid dists",
            fontsize=8,
        )

        ls = []
        hs = []
        plotci = waveform_utils.make_contiguous_channel_index(
            self.geom.shape[0], plot_chans
        )
        pal = sns.color_palette(n_colors=len(sorted_near_units))
        pal = pal[::-1]
        gtmc = self.template_maxchans[unit]
        max_abs = np.abs(self.cleaned_templates[sorted_near_units]).max()
        for j, nearby in enumerate(sorted_near_units[::-1]):
            lines = cluster_viz_index.pgeom(
                self.cleaned_templates[nearby][tmin:tmax, plotci[gtmc]],
                gtmc,
                plotci,
                self.geom,
                ax=axes["c"],
                color=pal[j],
                max_abs_amp=max_abs,
                show_zero=not j,
                x_extension=0.9,
            )
            ls.append(lines[0])
            hs.append(str(nearby))

        # plot gt template
        lines = cluster_viz_index.pgeom(
            self.templates[unit][tmin:tmax, plotci[gtmc]],
            gtmc,
            plotci,
            self.geom,
            ax=axes["c"],
            color="k",
            linestyle="--",
            max_abs_amp=max_abs,
            show_zero=not j,
            x_extension=0.9,
        )
        ls.append(lines[0])
        hs.append(f"unit{unit}")

        axes["c"].legend(ls, hs, title="nearby units", loc="lower right")
        axes["c"].set_title("templates of nearby units around unit maxchan")

        return fig, axes

    def make_resid_studies(
        self,
        out_folder,
        n_jobs=-1,
    ):
        out_folder = Path(out_folder)
        out_folder.mkdir(exist_ok=True, parents=True)

        def job(unit):
            fig, axes = self.unit_resid_study(
                unit,
                n_max=5,
                max_dist=4,
                plot_chans=10,
                lambd=0.001,
                allowed_scale=0.1,
                tmin=None,
                tmax=None,
            )
            if fig is not None:
                fig.savefig(
                    out_folder / f"{self.name_lo}_unit{unit:03d}.png", dpi=300
                )
                plt.close(fig)

        with Parallel(n_jobs) as p:
            for res in p(
                delayed(job)(unit)
                for unit in tqdm(self.unit_labels, desc="Resid studies")
            ):
                pass


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
                self.performance_by_unit * gt_sorting.unit_spike_counts[:, None]
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
        axes[i, 1].set_title(f"{comp.new_sorting.name}, {matchstr}", fontsize=8)

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


def sym_resid_dist(temp_a, temp_b, thresh, lambd=0.001, allowed_scale=0.1):
    maxres_a, _ = deconv_resid_merge.check_additional_merge(
        temp_a, temp_b, thresh, lambd=lambd, allowed_scale=allowed_scale
    )
    maxres_b, _ = deconv_resid_merge.check_additional_merge(
        temp_b, temp_a, thresh, lambd=lambd, allowed_scale=allowed_scale
    )
    return min(maxres_a, maxres_b)


def calc_resid_matrix(
    templates_a,
    units_a,
    templates_b,
    units_b,
    normalized=True,
    thresh=8,
    n_jobs=-1,
    vis_ptp_thresh=1,
    auto=False,
    pbar=True,
    lambd=0.001,
    allowed_scale=0.1,
):
    # we will calculate resid dist for templates that overlap at all
    # according to these channel neighborhoods
    chans_a = [
        np.flatnonzero(temp.ptp(0) > vis_ptp_thresh) for temp in templates_a
    ]
    chans_b = [
        np.flatnonzero(temp.ptp(0) > vis_ptp_thresh) for temp in templates_b
    ]

    def job(i, j, ua, ub):
        return (
            i,
            j,
            sym_resid_dist(
                templates_a[ua],
                templates_b[ub],
                thresh,
                lambd=lambd,
                allowed_scale=allowed_scale,
            ),
        )

    jobs = []
    resid_matrix = np.full((units_a.size, units_b.size), np.inf)
    for i, ua in enumerate(units_a):
        for j, ub in enumerate(units_b):
            if auto and ua == ub:
                continue
            if np.intersect1d(chans_a[ua], chans_b[ub]).size:
                jobs.append(delayed(job)(i, j, ua, ub))

    if pbar:
        jobs = tqdm(jobs, desc="Resid matrix")
    for i, j, dist in Parallel(n_jobs)(jobs):
        resid_matrix[i, j] = dist

    if normalized:
        rms_a = np.array(
            [
                np.sqrt(
                    np.square(templates_a[ua]).sum()
                    / (np.abs(templates_a[ua]) > 0).sum()
                )
                for ua in units_a
            ]
        )
        rms_b = np.array(
            [
                np.sqrt(
                    np.square(templates_b[ub]).sum()
                    / (np.abs(templates_b[ub]) > 0).sum()
                )
                for ub in units_b
            ]
        )
        resid_matrix = resid_matrix / np.sqrt(rms_a[:, None] * rms_b[None, :])

    return resid_matrix


def resid_dfs(hybrid_comparison, lambd=0.001, allowed_scale=0.1):
    gts = hybrid_comparison.gt_sorting
    news = hybrid_comparison.new_sorting
    thresh = 0.9 * np.square(gts.templates).sum(axis=(1, 2)).min()
    print(f"{thresh=}")

    resid_matrix = calc_resid_matrix(
        gts.templates,
        gts.unit_labels,
        news.templates,
        news.unit_labels,
        thresh,
        lambd=lambd,
        allowed_scale=allowed_scale,
    )

    # ordered versions
    # agreement order
    resid_matchord_df = hybrid_comparison.ordered_agreement.copy()
    resid_matchord_df[:] = 0.0
    for i, gtu in enumerate(resid_matchord_df.index):
        for j, newu in enumerate(resid_matchord_df.columns):
            resid_matchord_df.values[i, j] = resid_matrix[gtu, newu]

    # z order
    gtzord = np.argsort(gts.template_xzptp[:, 1])
    newzord = np.argsort(news.template_xzptp[:, 1])
    resid_zord_df = pd.DataFrame(
        resid_matrix[gtzord, :][:, newzord],
        index=gts.unit_labels[gtzord],
        columns=news.unit_labels[newzord],
    )

    return resid_matrix, resid_matchord_df, resid_zord_df


def plot_agreement_matrix(
    hybrid_comparison, with_resid=False, cmap=plt.cm.plasma
):
    if with_resid:
        resid_matrix, resid_matchord_df, resid_zord_df = resid_dfs(
            hybrid_comparison
        )
        vals = resid_matrix[np.isfinite(resid_matrix)]

        newu_zord = hybrid_comparison.new_sorting.unit_labels[
            np.argsort(hybrid_comparison.new_sorting.template_xzptp[:, 1])
        ]
        (
            selfthresh,
            new_self_resids,
        ) = hybrid_comparison.new_sorting.resid_matrix(newu_zord)
        self_vals = new_self_resids[np.isfinite(new_self_resids)]
        newu_zord_df = pd.DataFrame(
            new_self_resids, index=newu_zord, columns=newu_zord
        )

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        oa = hybrid_comparison.ordered_agreement.copy()
        oa.values[oa.values == 0] = np.inf
        sns.heatmap(oa, vmin=0, vmax=1.0, cmap=plt.cm.gnuplot2_r, ax=axes[0, 0])
        axes[0, 0].set_title("spike train agreement")

        sns.heatmap(
            resid_matchord_df,
            vmin=vals.min(),
            vmax=vals.max(),
            cmap=cmap,
            ax=axes[1, 0],
        )
        axes[1, 0].set_title("agreement ordered deconv resid")

        sns.heatmap(
            resid_zord_df,
            vmin=vals.min(),
            vmax=vals.max(),
            cmap=cmap,
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("z ordered deconv resid")

        axes[0, 1].hist(vals, bins=32)
        axes[0, 1].set_title("deconv resid histogram")

        sns.heatmap(
            newu_zord_df,
            vmin=self_vals.min(),
            vmax=self_vals.max(),
            cmap=cmap,
            ax=axes[1, 2],
        )
        axes[1, 2].set_title(
            f"z ordered {hybrid_comparison.new_sorting.name_lo} self resid dists"
        )

        axes[0, 2].hist(self_vals, bins=32)
        axes[0, 2].set_title(
            f"{hybrid_comparison.new_sorting.name_lo} self resid dist histogram"
        )

        return fig, axes
    else:
        ax = sns.heatmap(hybrid_comparison.ordered_agreement, cmap=cmap)
        return ax.figure, ax


def gtunit_resid_study(
    hybrid_comparison,
    gt_unit,
    n_max=5,
    max_dist=4,
    plot_chans=10,
    lambd=0.001,
    allowed_scale=0.1,
    tmin=10,
    tmax=100,
):
    gt_temp = hybrid_comparison.gt_sorting.templates[gt_unit]
    thresh = (
        0.9
        * np.square(hybrid_comparison.gt_sorting.templates)
        .sum(axis=(1, 2))
        .min()
    )
    new_temps = hybrid_comparison.new_sorting.cleaned_templates
    resid_v_new = calc_resid_matrix(
        gt_temp[None],
        np.array([0]),
        new_temps,
        np.arange(new_temps.shape[0]),
        thresh=thresh,
        n_jobs=1,
        pbar=False,
        lambd=lambd,
        allowed_scale=allowed_scale,
    ).squeeze()

    # ignore large distances
    # resid_v_new[resid_v_new > max_dist] = np.inf

    resid_is_finite = np.isfinite(resid_v_new)
    if not resid_is_finite.any():
        return None, None

    resid_is_finite = np.flatnonzero(resid_is_finite)

    resid_vals = resid_v_new[resid_is_finite]
    sort = np.argsort(resid_vals)[:n_max]
    sorted_near_units = hybrid_comparison.new_sorting.unit_labels[
        resid_is_finite[sort]
    ]

    fig, axes = plt.subplot_mosaic(
        "a.b\n...\nccc",
        gridspec_kw=dict(height_ratios=[1.5, 0.1, 4], width_ratios=[1, 0.1, 1]),
        figsize=(4, 8),
    )

    axes["a"].plot(resid_vals[sort])
    axes["a"].set_xticks(
        np.arange(len(sort)),
        sorted_near_units,
    )
    axes["a"].set_xlabel("all nearby sorter units")
    axes["a"].set_ylabel("normalized resid dist")

    sorted_near_units = sorted_near_units

    thresh, near_unit_distmat = hybrid_comparison.new_sorting.resid_matrix(
        sorted_near_units,
        n_jobs=1,
        pbar=False,
        lambd=lambd,
        allowed_scale=allowed_scale,
    )
    near_dist_df = pd.DataFrame(
        near_unit_distmat, index=sorted_near_units, columns=sorted_near_units
    )
    vals = near_unit_distmat[np.isfinite(near_unit_distmat)]
    if not vals.size:
        vals = np.array([0, 0])
    sns.heatmap(near_dist_df, vmin=vals.min(), vmax=vals.max(), ax=axes["b"])
    axes["b"].set_xlabel("nearby sorter units")
    axes["b"].set_title(
        f"{len(sorted_near_units)} closest pairwise resid dists", fontsize=8
    )

    ls = []
    hs = []
    plotci = waveform_utils.make_contiguous_channel_index(
        hybrid_comparison.new_sorting.geom.shape[0], plot_chans
    )
    pal = sns.color_palette(n_colors=len(sorted_near_units))
    pal = pal[::-1]
    gtmc = hybrid_comparison.gt_sorting.template_maxchans[gt_unit]
    max_abs = np.abs(
        hybrid_comparison.new_sorting.cleaned_templates[sorted_near_units]
    ).max()
    for j, unit in enumerate(sorted_near_units[::-1]):
        lines = cluster_viz_index.pgeom(
            hybrid_comparison.new_sorting.cleaned_templates[unit][
                tmin:tmax, plotci[gtmc]
            ],
            gtmc,
            plotci,
            hybrid_comparison.new_sorting.geom,
            ax=axes["c"],
            color=pal[j],
            max_abs_amp=max_abs,
            show_zero=not j,
            x_extension=0.9,
        )
        ls.append(lines[0])
        hs.append(str(unit))

    # plot gt template
    lines = cluster_viz_index.pgeom(
        gt_temp[tmin:tmax, plotci[gtmc]],
        gtmc,
        plotci,
        hybrid_comparison.new_sorting.geom,
        ax=axes["c"],
        color="k",
        linestyle="--",
        max_abs_amp=max_abs,
        show_zero=not j,
        x_extension=0.9,
    )
    ls.append(lines[0])
    hs.append(f"GT{gt_unit}")

    axes["c"].legend(ls, hs, title="nearby units", loc="lower right")
    axes["c"].set_title("templates of nearby units around GT maxchan")

    return fig, axes


def calc_template_snrs(
    templates,
    spike_train,
    raw_binary_file,
    trough_offset=42,
    spike_length_samples=121,
    max_spikes_per_unit=10000,
    seed=0,
):
    snrs = []
    rg = np.random.default_rng(seed)
    wf_buffer = np.empty(
        (max_spikes_per_unit, *templates.shape[1:]), dtype=templates.dtype
    )
    C = templates.shape[2]
    T_samples, _ = get_binary_length(raw_binary_file, C, 1)

    for u, t in enumerate(tqdm(templates)):
        in_unit = np.flatnonzero(spike_train[:, 1] == u)
        if in_unit.size > max_spikes_per_unit:
            in_unit = rg.choice(
                in_unit, replace=False, size=max_spikes_per_unit
            )
        random_times = rg.choice(
            np.arange(trough_offset, T_samples - spike_length_samples),
            size=max_spikes_per_unit,
            replace=False,
        )

        wfs, _ = read_waveforms(
            spike_train[in_unit, 0],
            raw_binary_file,
            C,
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
            buffer=wf_buffer,
        )
        numerator = np.abs(np.einsum("ij,nij->n", t, wfs) / C).mean()

        noise, _ = read_waveforms(
            random_times,
            raw_binary_file,
            C,
            trough_offset=trough_offset,
            spike_length_samples=spike_length_samples,
            buffer=wf_buffer,
        )
        denominator = np.abs(np.einsum("ij,nij->n", t, noise) / C).mean()
        snrs.append(numerator / denominator)

    return np.array(snrs)
